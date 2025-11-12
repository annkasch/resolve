from email.policy import strict
from resolve.conditional_neural_process_family.memory_bank import MemoryBank
import torch
import torch.nn as nn
from resolve.conditional_neural_process_family.class_attention import CrossAttentionDual,  CrossAttention
from resolve.conditional_neural_process_family.feature_encoder import FeatureEncoder, MLPEncoder, MLP
from resolve.conditional_neural_process_family.latent_space import LatentPosteriorCT, LatentTokenProj, LatentPriorTheta, LatentPriorC, masked_mean, kl_gauss
from resolve.conditional_neural_process_family.transformer_encoder import ContextTransformerEncoder
import torch.nn.functional as F

class FeedForward(nn.Module):
    def __init__(self, d_model, expansion=4, dropout=0.1):
        super().__init__()
        hidden = expansion * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class BilinearFull(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.Wp = nn.Parameter(torch.zeros(D, D))
        self.Wn = nn.Parameter(torch.zeros(D, D))
        self.Wd = nn.Parameter(torch.zeros(D, D))
        nn.init.xavier_uniform_(self.Wp)
        nn.init.xavier_uniform_(self.Wn)
        nn.init.xavier_uniform_(self.Wd)
        #self.ln_t = nn.LayerNorm(D); self.ln_c = nn.LayerNorm(D)

    def forward(self, r_t, r_pos, r_neg, r_diff=None):
        #r_t  = self.ln_t(r_t); r_pos = self.ln_c(r_pos); r_neg = self.ln_c(r_neg), r_diff = self.ln_c(r_diff)
        # Δ_pos = r_t^T Wp r_pos, Δ_neg = r_t^T Wn r_neg, Δ_diff = r_t^T Wd r_diff
        delta_pos = torch.einsum('bnd,df,bnf->bn', r_t, self.Wp, r_pos).unsqueeze(-1)
        delta_neg = torch.einsum('bnd,df,bnf->bn', r_t, self.Wn, r_neg).unsqueeze(-1)
        delta = delta_pos - delta_neg
        if r_diff is not None: 
            delta_diff = torch.einsum('bnd,df,bnf->bn', r_t, self.Wd, r_diff).unsqueeze(-1)
            delta += delta_diff
        return delta
    
class DecoderHead(nn.Module):
    """Maps z_t -> logit."""
    def __init__(self, in_dim, hidden=[256, 256], out_dim=1):
        super().__init__()
        self.net = MLP([in_dim] + hidden + [out_dim])
        #self.net = MLP([in_dim] + hidden + [out_dim*2])

    def forward(self, z_t):
        hidden = self.net(z_t)
        #mu, rho = torch.split(hidden, hidden.size(-1) // 2, dim=-1)
        #sigma = F.softplus(rho) + 1e-6
        return hidden

class AttnCNP(nn.Module):
    def __init__(self,
                 d_theta, d_phi, d_y,
                 d_model=32,
                 encoder_hidden=[128,128],
                 mode="concat",
                 theta_embed_dim=None,
                 n_heads=4):
        super().__init__()

        self.d_model = d_model
        self.d_theta = d_theta
        self.d_phi   = d_phi
        self.d_y     = d_y

        # encoders
        # Target query encoder: x_t = (theta,phi)  → R_t (no y_t)
        self.qry_enc = FeatureEncoder(
            phi_dim=d_phi, y_dim=None, theta_in_dim=d_theta,
            hidden=encoder_hidden, out_dim=d_model,
            mode=mode, theta_embed_dim=theta_embed_dim, use_layernorm=True
        )

        self.mom_enc = FeatureEncoder(
            phi_dim=d_phi, y_dim=None, theta_in_dim=d_theta,
            hidden=encoder_hidden, out_dim=d_model,
            mode=mode, theta_embed_dim=theta_embed_dim, use_layernorm=True
        )
        self.mom_enc.load_state_dict(self.qry_enc.state_dict(), strict=True)   # start identical
        self.mom_enc.eval()
        for p in self.mom_enc.parameters(): p.requires_grad_(False)
        self.memory_bank = MemoryBank(d_model, self.qry_enc, self.mom_enc, tau=0.999, use_faiss=True)

        # Simpler & consistent with qry/tgt encoders:
        self.ctx_enc = FeatureEncoder(
            phi_dim=d_phi, y_dim=d_y, theta_in_dim=d_theta,
            hidden=encoder_hidden, out_dim=d_model,
            mode=mode, theta_embed_dim=theta_embed_dim, use_layernorm=True
        )


        # Cross-attention (class-conditional dual)
        self.attn = CrossAttentionDual(d_model, n_heads=n_heads, out_dim=d_model)

        # Norms
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_r  = nn.LayerNorm(d_model)

        in_dim = 2*d_model

        # Bernoulli decoder: base + Δ (additive logit)
        self.base_decoder  = DecoderHead(in_dim, out_dim=1)  # logits
        self.bilinear_head = BilinearFull(d_model)                # logits Δ

    def forward(
        self,
        query_theta, query_phi,
        context_theta, context_phi, context_y,
        *,
        mask_c=None,
        **_
    ):
        """
        Shapes
        -------
        context_theta: (B, Nc, d_theta)
        context_phi  : (B, Nc, d_phi)
        context_y    : (B, Nc, 1)
        query_theta  : (B, Nt, d_theta)
        query_phi    : (B, Nt, d_phi)
        query_y      : (B, Nt, 1)   # only used when train=True (for q(z|C,T))

        Returns
        -------
        {"logits": (B,Nt,1), "loss_kl": scalar, "aux": dict}
        """
        device = context_phi.device
        B, Nc, _ = context_phi.shape
        if mask_c is None:
            mask_c = torch.ones(B, Nc, dtype=torch.bool, device=device)

        # context encodings (x_c,y_c) → R_ctx
        R_ctx  = self.ctx_enc(theta=context_theta, phi=context_phi, y=context_y)  # (B,Nc,D)
        wS_ctx = (context_y.squeeze(-1) > 0.5).float()                            # (B,Nc) in [0,1]

        # target query (x_t) → R_t
        
        R_t = self.qry_enc(theta=query_theta, phi=query_phi)                  # (B,Nt,D)
        ids, sims = self.memory_bank.topM_batch(query_theta, query_phi, M=5)

        r_all,_,_,_ = self.attn(
            Q_src=R_t,
            K_src=self.norm_kv(R_ctx),
            wS_ctx=wS_ctx,
            mask=mask_c
        )

        rC = r_all.mean(dim=1, keepdim=True).expand(-1, R_t.shape[1], -1)     # (B,Nt,D)

        logits = self.base_decoder(torch.cat([R_t, rC], dim=-1))                                     # (B,Nt,1)

        return {"logits": logits}

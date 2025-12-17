import torch
import torch.nn as nn
from resolve.conditional_neural_process_family.class_attention import CrossAttentionDual,  CrossAttention, SimplePoolAttention
from resolve.conditional_neural_process_family.feature_encoder import FeatureEncoder, MLP


    
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

        # Simpler & consistent with qry/tgt encoders:
        self.ctx_enc = FeatureEncoder(
            phi_dim=d_phi, y_dim=d_y, theta_in_dim=d_theta,
            hidden=encoder_hidden, out_dim=d_model,
            mode=mode, theta_embed_dim=theta_embed_dim, use_layernorm=True
        )

        self.attn = SimplePoolAttention()

        # Norms
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.norm_r  = nn.LayerNorm(d_model)

        in_dim = 2*d_model
        self.base_decoder  = DecoderHead(in_dim, out_dim=1)  # logits

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

        Returns
        -------
        {"logits": (B,Nt,1)}
        """
        device = context_phi.device

        B, Nc, _ = context_phi.shape
        if mask_c is None:
            mask_c = torch.ones(B, Nc, dtype=torch.bool, device=device)

        # context encodings (x_c,y_c) → R_ctx
        R_ctx  = self.ctx_enc(theta=context_theta, phi=context_phi, y=context_y)  # (B,Nc,D)
        # target query (x_t) → R_t
        R_t = self.qry_enc(theta=query_theta, phi=query_phi)                  # (B,Nt,D)

        r_all, _ = self.attn(
            Q_src=R_t,                     # (B, Nt, D)
            K_src=self.norm_kv(R_ctx),     # (B, Nc, D)
            mask=mask_c,                   # (B, Nc)
            logit_bias_ctx=None,        # (B, Nc) or None
            beta=1.0             # scalar hyperparam, e.g. 1.0
        )

        rC = r_all.mean(dim=1, keepdim=True).expand(-1, R_t.shape[1], -1)     # (B,Nt,D)

        logits = self.base_decoder(torch.cat([R_t, rC], dim=-1))                                     # (B,Nt,1)

        return {"logits": logits}

    def save(self,state, path):
        # drop all tree.leaf_cache.* entries from the state dict
        torch.save(state, path+'_model.pth')
    
    def load(self, path):
        state = torch.load(path+'_model.pth', map_location='cpu')
        self.load_state_dict(state['model_state'])



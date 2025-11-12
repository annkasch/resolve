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
"""
# Context side
        self.ctx_enc = ContextTransformerEncoder(theta_dim=d_theta,
            phi_dim=d_phi,
            y_dim=d_y,
            embed_dim= d_model,
            depth= 4,
            num_heads= 4,
            mlp_ratio= 4.0,
            dropout= 0.0,
            proj_out_dim= None,  # if set, final linear to this dim
            use_cls_token= False       # set False: mean over feature tokens
        )
"""
class AttnLNP(nn.Module):
    def __init__(self,
                 d_theta, d_phi, d_y,
                 d_model=32,
                 encoder_hidden=[128,128],
                 mode="concat",
                 theta_embed_dim=None,
                 n_heads=4,
                 z_dim=8):
        super().__init__()

        # Context encoder (set-based). You had a Transformer; that’s fine too.
        # Keep this line if you want the transformer version instead:
        # self.ctx_enc = ContextTransformerEncoder(...)

        # Simpler & consistent with qry/tgt encoders:
        from resolve.conditional_neural_process_family.feature_encoder import FeatureEncoder
        self.ctx_enc = FeatureEncoder(
            phi_dim=d_phi, y_dim=d_y, theta_in_dim=d_theta,
            hidden=encoder_hidden, out_dim=d_model,
            mode=mode, theta_embed_dim=theta_embed_dim, use_layernorm=True
        )

        self.d_model = d_model
        self.d_theta = d_theta
        self.d_phi   = d_phi
        self.d_y     = d_y

        # Target query encoder: x_t = (theta,phi)  → R_t (no y_t)
        self.qry_enc = FeatureEncoder(
            phi_dim=d_phi, y_dim=None, theta_in_dim=d_theta,
            hidden=encoder_hidden, out_dim=d_model,
            mode=mode, theta_embed_dim=theta_embed_dim, use_layernorm=True
        )

        # Target encoder for the POSTERIOR path ONLY (uses y_t at train)
        self.tgt_enc = FeatureEncoder(
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

        # Latent
        self.z_dim           = int(z_dim) if z_dim is not None else 0
        self.use_latent      = (self.z_dim > 0)

        in_dim = 2*d_model
        if self.use_latent:
            self.post_q      = LatentPosteriorCT(d_model=self.d_model, z_dim=self.z_dim)   # q(z|C,T)
            self.prior_p     = LatentPriorC(d_model=self.d_model,    z_dim=self.z_dim)     # p(z|C)
            self.prior_theta = LatentPriorTheta(d_theta=self.d_theta, z_dim=self.z_dim)    # p(z|θ̄)
            self.latent_proj = LatentTokenProj(z_dim=self.z_dim, d_model=self.d_model, s_bias_init=-3.0)
            in_dim += self.z_dim

        # Bernoulli decoder: base + Δ (additive logit)
        self.base_decoder  = DecoderHead(in_dim, out_dim=1)  # logits
        self.bilinear_head = BilinearFull(d_model)                # logits Δ

    def forward(
        self,
        query_theta, query_phi, query_y,           # query_y needed only when train=True
        context_theta, context_phi, context_y,
        *,
        mask_c=None,
        train=True,
        # --- stability knobs (all optional) ---
        step: int = 0,                 # global step for warmups
        kl_warmup_steps: int = 5000,   # ramp β from 0→beta_max over this many steps
        beta_max: float = 0.5,         # max KL weight after warmup
        latent_warmup_steps: int = 500,# run deterministic path for first N steps
        num_z_samples_eval: int = 16,  # MC samples at eval
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

        # Encode context & targets
        # context encodings (x_c,y_c) → R_ctx
        R_ctx  = self.ctx_enc(theta=context_theta, phi=context_phi, y=context_y)  # (B,Nc,D)
        wS_ctx = (context_y.squeeze(-1) > 0.5).float()                            # (B,Nc) in [0,1]
        rbar_c = masked_mean(R_ctx, mask_c, dim=1)                                # (B,D)

        # target query (x_t) → R_t
        R_t_raw = self.qry_enc(theta=query_theta, phi=query_phi)                  # (B,Nt,D)
        R_t     = self.norm_q(R_t_raw)
        t_mask  = torch.ones(B, R_t_raw.size(1), dtype=torch.bool, device=device)

        # Deterministic warmup (no z)
        if not self.use_latent or (train and step < latent_warmup_steps):
            r_all, r_pos, r_neg, r_diff = self.attn(
                Q_src=R_t,
                K_src=self.norm_kv(R_ctx),
                wS_ctx=wS_ctx,
                mask=mask_c
            )
            r_all  = self.norm_r(r_all); r_pos = self.norm_r(r_pos)
            r_neg  = self.norm_r(r_neg); r_diff = self.norm_r(r_diff)

            rC = r_all.mean(dim=1, keepdim=True).expand(-1, R_t.shape[1], -1)     # (B,Nt,D)
            base_in = self._dec_in(R_t, rC, z=None)                                  # (B,Nt,2D)

            mu0   = self.base_decoder(R_t, R_ctx.mean(dim=1, keepdim=True).expand(-1, R_t.shape[1], -1))                                     # (B,Nt,1)
            delta = self.bilinear_head(R_t, r_pos, r_neg, r_diff)                  # (B,Nt,1)
            logits = mu0
            return {"logits": logits}


        # Latent path: q(z|C,T) (train) / p(z|C) (eval)
        # KL annealing
        beta = 0.0 if kl_warmup_steps <= 0 else beta_max * min(1.0, step / max(1, kl_warmup_steps))

        if train:
            # posterior needs target labels
            R_t_post = self.tgt_enc(theta=query_theta, phi=query_phi, y=query_y)   # (B,Nt,D)
            rbar_t   = masked_mean(R_t_post, t_mask, dim=1)                        # (B,D)
            mu_q, logvar_q = self.post_q(rbar_c, rbar_t)                           # q(z|C,T)

            mu_p, logvar_p = self.prior_p(rbar_c)                                  # p(z|C)

            # stability clamps
            logvar_q = logvar_q.clamp(-6.0, 2.0)
            logvar_p = logvar_p.clamp(-6.0, 2.0)

            loss_kl = beta * kl_gauss(mu_q, logvar_q, mu_p, logvar_p).mean()
            num_z = 1                                                               # 1 sample @train
            mu, logvar = mu_q, logvar_q
        else:
            # eval uses prior only
            mu, logvar = self.prior_p(rbar_c)                                       # (B,z_dim)
            logvar = logvar.clamp(-6.0, 2.0)
            loss_kl = R_ctx.new_zeros(())
            num_z = max(1, int(num_z_samples_eval))

        std = (0.5 * logvar).exp()

        # Monte Carlo over z
        logits_mc = []
        for _ in range(num_z):
            z = mu + std * torch.randn_like(mu)                                     # (B,z_dim)

            # (a) OPTIONAL: FiLM(z) on target features to force usage of z
            if hasattr(self, "z_to_gamma") and hasattr(self, "z_to_beta"):
                gamma = self.z_to_gamma(z).unsqueeze(1)                              # (B,1,D)
                beta_z = self.z_to_beta(z).unsqueeze(1)                              # (B,1,D)
                R_t_mod = gamma * R_t + beta_z                                       # (B,Nt,D)
            else:
                R_t_mod = R_t

            # (b) Append latent token into context K/V with a base-rate weight
            token, s_base = self.latent_proj(z)                                      # (B,D),(B,1)
            token  = token.unsqueeze(1)                                              # (B,1,D)
            s_base = s_base.squeeze(-1)                                              # (B,)

            R_ctx_aug = torch.cat([R_ctx, token], dim=1)                             # (B,Nc+1,D)
            mask_aug  = torch.cat([mask_c, torch.ones(B,1, dtype=torch.bool, device=device)], dim=1)
            wS_aug    = torch.cat([wS_ctx, s_base.unsqueeze(1)], dim=1)              # (B,Nc+1)

            # (c) Attention pooling
            r_all, r_pos, r_neg, r_diff = self.attn(
                Q_src=R_t_mod,
                K_src=self.norm_kv(R_ctx_aug),
                wS_ctx=wS_aug,
                mask=mask_aug
            )
            r_all  = self.norm_r(r_all); r_pos = self.norm_r(r_pos)
            r_neg  = self.norm_r(r_neg); r_diff = self.norm_r(r_diff)

            # (d) Decoder input includes z (ANP-style)
            rC = r_all.mean(dim=1, keepdim=True).expand(-1, R_t.shape[1], -1)        # (B,Nt,D)
            z_exp = z.unsqueeze(1).expand(-1, R_t.shape[1], -1)                      # (B,Nt,z)
            base_in = self._dec_in(R_t_mod, rC, z)                        # (B,Nt,2D+z)

            mu0   = self.base_decoder(base_in)                                       # (B,Nt,1)
            delta = self.bilinear_head(R_t_mod, r_pos, r_neg, r_diff)                # (B,Nt,1)
            logits_mc.append(mu0 + delta)                                            # keep as logits (NO sigmoid)

        # average logits across z-samples (prob-avg also OK; keep logits for BCEWithLogits)
        logits = torch.stack(logits_mc, 0).mean(0)                                   # (B,Nt,1)

        return {"logits": logits, "loss_kl": loss_kl, "aux": {"beta": beta}}
    
    def _dec_in(self, R_t_mod, rC, z=None):
        # R_t_mod, rC: (B,Nt,D)
        if self.z_dim > 0:
            if z is None:
                z_exp = torch.zeros(R_t_mod.size(0), R_t_mod.size(1), self.z_dim,
                                    device=R_t_mod.device, dtype=R_t_mod.dtype)
            else:
                z_exp = z.unsqueeze(1).expand(-1, R_t_mod.size(1), -1)  # (B,Nt,z)
            return torch.cat([R_t_mod, rC, z_exp], dim=-1)
        else:
            return torch.cat([R_t_mod, rC], dim=-1)
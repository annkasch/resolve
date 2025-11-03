import torch
import torch.nn as nn
from resolve.conditional_neural_process_family.class_attention import GlobalContextAttentionDual,  GlobalContextAttention
from resolve.conditional_neural_process_family.memory_bank import MemoryBank
from resolve.conditional_neural_process_family.target_encoder import TargetQueryEncoder, TargetEncoder
from resolve.conditional_neural_process_family.context_encoder import ContextConditionalEncoder, ThetaEncoder, MLP
from resolve.conditional_neural_process_family.latent_space import LatentPosterior, LatentPriorTheta, LatentTokenProj, masked_mean, kl_gauss

class BernoulliHead(nn.Module):
    """Maps z_t -> logit."""
    def __init__(self, in_dim, hidden=[256, 256]):
        super().__init__()
        self.net = MLP([in_dim] + hidden + [1])

    def forward(self, z_t):
        B, Nt, _ = z_t.shape
        return self.net(z_t.view(B*Nt, -1)).view(B, Nt, 1)

# ---------- full model ----------
class HCTargetAttnLNP(nn.Module):
    def __init__(self, d_theta, d_phi, d_y, d_model=128, z_dim=8, use_theta_prior=True):
        super().__init__()
        # Context side
        self.ctx_enc = ContextConditionalEncoder(
            phi_dim=d_phi, y_dim=d_y, theta_in_dim=d_theta,
            theta_embed_dim=64, hidden=[128, 128], out_dim=d_model,
            mode='film', use_layernorm=True
        )
        
        # --- Latent bits ---
        self.post_q = LatentPosterior(d_model, z_dim=z_dim)
        self.latent_proj = LatentTokenProj(z_dim=z_dim, d_model=d_model)
        self.use_theta_prior = use_theta_prior
        if use_theta_prior:
            # simple θ̄ prior: mean over valid context θ
            self.prior_p = LatentPriorTheta(d_theta, z_dim=z_dim)

        self.theta_enc_t = ThetaEncoder(d_theta, [128, 128], d_model)
        
        # Target query R^{(t)}
        self.tquery = TargetQueryEncoder(
            theta_in_dim=d_theta, phi_dim=d_phi,
            hidden=[128], out_dim=d_model, theta_encoder=self.theta_enc_t
        )

        # Attention pooling (two passes with different value weights)
        #self.attn = GlobalContextAttention(d_model, n_heads=4, out_dim=d_model)
        self.attn = GlobalContextAttentionDual(d_model, n_heads=4, out_dim=d_model)
        
        # Target encoder -> z_t
        self.tgt_enc = TargetEncoder(
            theta_in_dim=d_theta, phi_dim=d_phi, r_dim=d_model,
            hidden=[128, 128], out_dim=d_model, theta_encoder=self.theta_enc_t
        )

        # Bernoulli decoder
        self.decoder = BernoulliHead(d_model)

    def forward(self, query_theta, query_phi, context_theta, context_phi, context_y, **kwargs):
        """
        Forward pass.

        Args:
            context_theta (Tensor): (B, Nc, d_theta)
            context_phi   (Tensor): (B, Nc, d_phi)
            context_y     (Tensor): (B, Nc, d_y)
            query_theta (Tensor): (B, Nt, d_theta)
            query_phi   (Tensor): (B, Nt, d_phi)

        Keyword Args:
            mask_c (Tensor, optional): (B, Nc) boolean mask for valid context rows.
            beta
            train_mode
        """
        mask_c = kwargs.get("mask_c", None)
        beta = kwargs.get("beta", 1.0)
        train_mode = kwargs.get("train_mode", True)

        B, Nc, _ = context_phi.shape
        if mask_c is None:
            mask_c = torch.ones(B, Nc, dtype=torch.bool, device=context_phi.device)

        # 1) Context encodings
        R_ctx = self.ctx_enc(context_phi, context_y, context_theta)  # (B,Nc,D)

        # --- Latent q(z|C) from pooled context embedding ---
        r_bar = masked_mean(R_ctx, mask_c, dim=1)  # (B,D)
        mu_q, logvar_q = self.post_q(r_bar)        # (B,z_dim)

        # reparameterize
        if train_mode:
            eps = torch.randn_like(mu_q)
            z = mu_q + (0.5*logvar_q).exp() * eps
        else:
            z = mu_q  # mean at eval (or sample multiple z’s outside)

        # Optional θ-conditioned prior p(z|θ̄)
        if self.use_theta_prior:
            theta_bar = masked_mean(context_theta, mask_c, dim=1)  # (B,d_theta)
            mu_p, logvar_p = self.prior_p(theta_bar)
        else:
            mu_p = torch.zeros_like(mu_q)
            logvar_p = torch.zeros_like(logvar_q)

        kl = kl_gauss(mu_q, logvar_q, mu_p, logvar_p)  # (B,)
        #print(mu_q.norm().mean(), logvar_q.mean(),kl.mean())

        # 2) Build per-target query
        R_t = self.tquery(query_theta, query_phi)              # (B,Nt,D)

        # 3) Latent -> token and base-rate weight
        token, s_base = self.latent_proj(z)            # (B,D), (B,1)
        token = token.unsqueeze(1)                     # (B,1,D)
        s_base = s_base.squeeze(-1)                    # (B,)
        #print(s_base.mean())
        # 4) Extend context with latent token
        R_ctx_ext = torch.cat([R_ctx, token], dim=1)   # (B,Nc+1,D)
        mask_ext  = torch.cat([mask_c, torch.ones(B,1, dtype=torch.bool, device=mask_c.device)], dim=1)

        # class weights for Dual attention: add latent token’s base-rate as its y
        wS_ctx = context_y.squeeze(-1).clamp(0,1)           # (B,Nc)
        wS_ext = torch.cat([wS_ctx, s_base.unsqueeze(1)], dim=1)  # (B,Nc+1)

        # 5) Attention pooling
        r_pos, r_neg = self.attn(Q_src=R_t, K_src=R_ctx_ext, V_src=R_ctx_ext, wS=wS_ext, mask=mask_ext)

        # 6) Target encoder and decoder
        z_t = self.tgt_enc(query_theta, query_phi, r_pos, r_neg)   # (B,Nt,D)
        logit = self.decoder(z_t)                          # (B,Nt,1)

        # Return KL for ELBO
        output = {
            "logits": [logit],
            "kl_term": (beta * kl.mean()),
        }
        
        return output
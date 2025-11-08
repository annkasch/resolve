import torch
import torch.nn as nn
from resolve.conditional_neural_process_family.class_attention import CrossAttentionDual,  CrossAttention
from resolve.conditional_neural_process_family.feature_encoder import FeatureEncoder, MLPEncoder, MLP
from resolve.conditional_neural_process_family.latent_space import LatentPosterior, LatentPriorTheta, LatentTokenProj, masked_mean, kl_gauss

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
    
class BernoulliHead(nn.Module):
    """Maps z_t -> logit."""
    def __init__(self, in_dim, hidden=[256, 256]):
        super().__init__()
        self.net = MLP([in_dim] + hidden + [1])

    def forward(self, z_t):
        B, Nt, _ = z_t.shape
        return self.net(z_t.view(B*Nt, -1)).view(B, Nt, 1)

# ---------- full model ----------
class AttnLNP(nn.Module):
    def __init__(self, d_theta, d_phi, d_y, d_model=32, encoder_hidden=[128,128], mode="concat", theta_embed_dim=None, n_heads=4, z_dim=8, use_theta_prior=True):
        super().__init__()
        # Context side
        self.ctx_enc = FeatureEncoder(
            phi_dim=d_phi, y_dim=d_y, theta_in_dim=d_theta, hidden=encoder_hidden, out_dim=d_model,
            mode=mode, theta_embed_dim=theta_embed_dim, use_layernorm=True
        )
        
        # Latent bits
        z_dim = int(z_dim) if z_dim is not None else 0
        self.use_latent = True if z_dim > 0 else False
        if self.use_latent:
            self.post_q = LatentPosterior(d_model, z_dim=z_dim)
            self.latent_proj = LatentTokenProj(z_dim=z_dim, d_model=d_model)
            self.use_theta_prior = use_theta_prior
            if use_theta_prior:
                # simple θ̄ prior: mean over valid context θ
                self.prior_p = LatentPriorTheta(d_theta, z_dim=z_dim)

        # Target query R^{(t)}
        self.qry_enc = FeatureEncoder(
            phi_dim=d_phi, y_dim=None, theta_in_dim=d_theta, hidden=encoder_hidden, out_dim=d_model,
            mode=mode, theta_embed_dim=theta_embed_dim, use_layernorm=True
        )

        # Attention pooling (two passes with different value weights)
        #self.attn = CrossAttention(d_model, n_heads=4, out_dim=d_model)
        self.attn = CrossAttentionDual(d_model, n_heads=n_heads, out_dim=d_model)
        
        # Target encoder -> z_t
        self.tgt_enc = MLPEncoder(
            in_dim=d_model,
            hidden=encoder_hidden, out_dim=d_model
        )

        # Normalization & feed-forward
        self.norm_q  = nn.LayerNorm(d_model)   # for queries (targets)
        self.norm_kv = nn.LayerNorm(d_model)   # for keys/values (context)
        self.norm_ff = nn.LayerNorm(d_model)   # before FFN
        self.ffn   = FeedForward(d_model, expansion=4, dropout=0.1)

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
        """
        mask_c = kwargs.get("mask_c", None)
        beta = kwargs.get("beta", 1.0)
        train_mode = kwargs.get("train_mode", True)

        B, Nc, _ = context_phi.shape
        if mask_c is None:
            mask_c = torch.ones(B, Nc, dtype=torch.bool, device=context_phi.device)

        # Context encodings
        R_ctx = self.ctx_enc(theta=context_theta, phi=context_phi, y=context_y)  # (B,Nc,D)
        # class weights for Dual attention
        wS_ctx = (context_y.squeeze(-1) > 0.5)          # (B,Nc)

        loss_kl = 0.
        if self.use_latent:
            # Latent q(z|C) from pooled context embedding
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
            loss_kl = (beta * kl.mean())

            # Latent -> token and base-rate weight
            token, s_base = self.latent_proj(z)            # (B,D), (B,1)
            token = token.unsqueeze(1)                     # (B,1,D)
            s_base = s_base.squeeze(-1)                    # (B,)

            # Extend context with latent token
            R_ctx = torch.cat([R_ctx, token], dim=1)   # (B,Nc+1,D)
            mask_c  = torch.cat([mask_c, torch.ones(B,1, dtype=torch.bool, device=mask_c.device)], dim=1)
            # add latent token’s base-rate as its y
            wS_ctx = torch.cat([wS_ctx, s_base.unsqueeze(1)], dim=1)  # (B,Nc+1)

        # Build per-target query
        R_t = self.qry_enc(theta=query_theta, phi=query_phi)              # (B,Nt,D)

        # Attention pooling
        r_pos, r_neg, r_all, loss_attn = self.attn(Q_src=self.norm_q(R_t), K_src=self.norm_kv(R_ctx), wS=wS_ctx, mask=mask_c)

        x = R_t + r_all               # residual after attention
        # Feed-Forward + Residual
        x = x + self.ffn(self.norm_ff(x))
        logit = self.decoder(x)                          # (B,Nt,1)

        # Return KL for ELBO
        output = {
            "logits": [logit],
            "loss": loss_kl + loss_attn
        }
        
        return output
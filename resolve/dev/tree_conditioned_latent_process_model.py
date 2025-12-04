import torch
import torch.nn as nn
import torch.nn.functional as F
from resolve.conditional_neural_process_family.class_attention import SimplePoolAttention, CrossAttention, CrossAttentionWithMoE
from resolve.conditional_neural_process_family.feature_encoder import FeatureEncoder, MLP
from resolve.conditional_neural_process_family.latent_space import LatentPosteriorCT, LatentTokenProj, LatentPriorC, masked_mean, kl_gauss
from resolve.network_architectures.lightgbm import LGBMWithLeafCache
from resolve.network_architectures.transformer_encoder import TransformerEncoder


class DecoderHead(nn.Module):
    """Maps z_t -> logit."""
    def __init__(self, in_dim, hidden=[256, 256], out_dim=1):
        super().__init__()
        self.out_dim = out_dim
        self.net = MLP([in_dim] + hidden + [out_dim])

    def forward(self, z_t):
        hidden = self.net(z_t)
        out = torch.split(hidden, hidden.size(-1) // self.out_dim, dim=-1)
        if self.out_dim == 2:
            out = list(out)
            # Clamp sigma to avoid degenerate cases
            out[1] = out[1].clamp(min=-4.0, max=1.0)  # σ in [~0.018, ~2.7]
            out[1] = out[1].exp()

        return out

class TreeConditionedLNP(nn.Module):
    def __init__(self,
                 d_theta, d_phi, d_y,
                 out_dim,
                 tree_config,
                 d_model=32,
                 encoder_hidden=[128,128],
                 mode="concat",
                 theta_embed_dim=None,
                 n_heads=4,
                 z_dim: int | None = None
                 ):
        super().__init__()

        self.d_model = d_model
        self.d_theta = d_theta
        self.d_phi   = d_phi
        self.d_y     = d_y

        self.tree = LGBMWithLeafCache(config=tree_config["config"], 
                                   task=tree_config.get("task","binary"), 
                                   out_dim=d_y,
                                   num_samples=tree_config.get("num_samples", None),
                                   use_parameter_search=tree_config.get("use_parameter_search", False), 
                                   use_leaf_embeddings=tree_config.get("use_leaf_embeddings", False))
        
        d_phi_ctx = d_phi + d_y + self.tree.leaf_embed_dim
        
        # Simpler & consistent with qry/tgt encoders:
        self.ctx_enc = FeatureEncoder(
            phi_dim=d_phi_ctx, y_dim=d_y, theta_in_dim=d_theta,
            hidden=encoder_hidden, out_dim=d_model,
            mode=mode, theta_embed_dim=theta_embed_dim, use_layernorm=True
        )
        """
        self.ctx_enc = TransformerEncoder(theta_dim=d_theta,
            phi_dim=d_phi_ctx,
            y_dim=d_y,
            embed_dim= d_model,
            depth= 1,
            num_heads= n_heads,
            mlp_ratio= 4.0,
            dropout= 0.0,
            proj_out_dim= None,  # if set, final linear to this dim
            use_cls_token= False,      # set False: mean over feature tokens
            use_tokenizer= False,
        )
        """
        
        # encoders
        # Target query encoder: x_t = (theta,phi)  → R_t (no y_t)
        d_phi_tgt = d_phi + self.tree.leaf_embed_dim
        self.qry_enc = FeatureEncoder(
            phi_dim=d_phi_tgt, y_dim=None, theta_in_dim=d_theta,
            hidden=encoder_hidden, out_dim=d_model,
            mode=mode, theta_embed_dim=theta_embed_dim, use_layernorm=True
        )

        self.tgt_enc = FeatureEncoder(
            phi_dim=d_phi_tgt, y_dim=d_y, theta_in_dim=d_theta,
            hidden=encoder_hidden, out_dim=d_model,
            mode=mode, theta_embed_dim=theta_embed_dim, use_layernorm=True
        )

        #self.attn = CrossAttention(d_model, n_heads=n_heads)
        self.attn = CrossAttentionWithMoE(dim=d_model, num_heads=n_heads)

        # Latent
        self.z_dim           = int(z_dim) if z_dim is not None else 0
        self.use_latent      = (self.z_dim > 0)

        in_dim = 2*d_model
        if self.use_latent:
            self.post_q      = LatentPosteriorCT(d_model=self.d_model, z_dim=self.z_dim)   # q(z|C,T)
            self.prior_p     = LatentPriorC(d_model=self.d_model,    z_dim=self.z_dim)     # p(z|C)
            self.latent_proj = LatentTokenProj(z_dim=self.z_dim, d_model=self.d_model, s_bias_init=-3.0)
            in_dim += self.z_dim

        self.base_decoder  = DecoderHead(in_dim, out_dim=out_dim)  # logits

        self.register_buffer("temperature", torch.ones(1))

    def fit(self,X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        query_theta: torch.Tensor | None = None,
        query_phi: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        loader=None,**kwargs):
        self.tree.fit(X, y, query_theta, query_phi, target, loader)

    def forward(
        self,
        query_theta, query_phi, target_y,
        query_idx,
        context_theta, context_phi, context_y,
        context_idx,
        *,
        mask_c=None,
        train=True,
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

        Returns
        -------
        {"logits": (B,Nt,1)}
        """
        device = context_phi.device

        B, Nc, _ = context_phi.shape
        if mask_c is None:
            mask_c = torch.ones(B, Nc, dtype=torch.bool, device=device)
        self.tree._out_device = device

        with torch.no_grad():
            out = self.tree(query_theta=context_theta, query_phi=context_phi, query_idx=context_idx)
        score_ctx = out["logits"][0]            # (B,T,1)
        leaf_emb = out["leaf_embeddings"]   # (B,T,embed_dim)

        phi_cnp_ctx = torch.cat([context_phi, score_ctx.to(device), leaf_emb.to(device)], dim=-1)
        # context encodings (x_c,y_c) → R_ctx
        R_ctx  = self.ctx_enc(context_theta, phi_cnp_ctx, y=context_y)  # (B,Nc,D)
        rbar_c = masked_mean(R_ctx, mask_c, dim=1)  
        wS_ctx = (context_y.squeeze(-1) > 0.5).float()  
        # target query (x_t) → R_t
        with torch.no_grad():
            out = self.tree(query_theta=query_theta, query_phi=query_phi, query_idx=query_idx)
        score_tgt = out["logits"][0]            # (B,T,1)
        leaf_emb = out["leaf_embeddings"]   # (B,T,embed_dim)

        phi_cnp_tgt = torch.cat([query_phi, leaf_emb.to(device)], dim=-1)
        # context encodings (x_c,y_c) → R_ctx
        R_t = self.qry_enc(theta=query_theta, phi=phi_cnp_tgt)                  # (B,Nt,D)
        t_mask  = torch.ones(B, R_t.size(1), dtype=torch.bool, device=device)


        # Deterministic warmup (no z)
        if not self.use_latent or (train and step < latent_warmup_steps):
            h_t = R_t
            for _ in range(2):
                h_t = self.attn(q_tokens=h_t, kv_tokens=R_ctx, key_mask=wS_ctx)

            base_in = self._dec_in(R_t, h_t, z=None) 
            logits = self.base_decoder(base_in)  
            return {"logits": logits, "scores": score_tgt}

        # Latent path: q(z|C,T) (train) / p(z|C) (eval)
        # KL annealing    
        beta = 0.0 if kl_warmup_steps <= 0 else beta_max * min(1.0, step / max(1, kl_warmup_steps))

        if train:
            R_t_post = self.tgt_enc(theta=query_theta, phi=phi_cnp_tgt, y=target_y)   # (B,Nt,D)
            rbar_t   = masked_mean(R_t_post, t_mask, dim=1)                        # (B,D)
            mu_q, logvar_q = self.post_q(rbar_c, rbar_t)                           # q(z|C,T)

            mu_p, logvar_p = self.prior_p(rbar_c)                                  # p(z|C)

            logvar_q = logvar_q.clamp(-6.0, 2.0)
            logvar_p = logvar_p.clamp(-6.0, 2.0)

            loss_kl = beta * kl_gauss(mu_q, logvar_q, mu_p, logvar_p).mean()
            mu, logvar = mu_q, logvar_q
            num_z = 1
        else:
            mu, logvar = self.prior_p(rbar_c)                                      # (B,z_dim)
            logvar = logvar.clamp(-6.0, 2.0)
            loss_kl = R_ctx.new_zeros(())
            num_z = max(1, int(num_z_samples_eval))

        std = (0.5 * logvar).exp()

        logits_mc = []
        for _ in range(num_z):
            z = mu + std * torch.randn_like(mu)                                    # (B,z_dim)

            if hasattr(self, "z_to_gamma") and hasattr(self, "z_to_beta"):
                gamma  = self.z_to_gamma(z).unsqueeze(1)                           # (B,1,D)
                beta_z = self.z_to_beta(z).unsqueeze(1)                            # (B,1,D)
                R_t_mod = gamma * R_t + beta_z                                     # (B,Nt,D)
            else:
                R_t_mod = R_t

            token, s_base = self.latent_proj(z)                                    # (B,D),(B,1)
            token  = token.unsqueeze(1)                                            # (B,1,D)
            # s_base = s_base.squeeze(-1)   # currently unused

            R_ctx_aug = torch.cat([R_ctx, token], dim=1)                           # (B,Nc+1,D)
            mask_lat = torch.ones(B, 1, dtype=torch.bool, device=device)
            mask_aug = torch.cat([wS_ctx.bool(), mask_lat], dim=1)  # (B, Nc+1)

            h_t_mod = R_t_mod
            for _ in range(2):
                h_t_mod = self.attn(q_tokens=h_t_mod, kv_tokens=R_ctx_aug, key_mask=mask_aug)

            base_in = self._dec_in(R_t_mod, h_t_mod, z)                             # (B,Nt,2D+z_dim)
            mu0 = self.base_decoder(base_in)                                      # (B,Nt,1)
            logits_mc.append(mu0)

        logits = torch.stack(logits_mc, 0).mean(0)                                 # (B,Nt,1)

        return {"logits": logits, "kl_term": loss_kl, "aux": {"beta": beta}} 
    
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
    
    def save(self, path):
        torch.save(self.state_dict(), path+'_model.pth')
        self.tree.save(path)


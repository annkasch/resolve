import torch
import torch.nn as nn
import torch.nn.functional as F
from resolve.conditional_neural_process_family.class_attention import SimplePoolAttention, CrossAttention, CrossAttentionWithMoE
from resolve.conditional_neural_process_family.feature_encoder import FeatureEncoder, MLP
from resolve.network_architectures.lightgbm import LGBMWithLeafCache
from resolve.network_architectures.transformer_encoder import TransformerEncoder
from resolve.helpers.losses import logit_normal_bernoulli_nll


class DecoderHead(nn.Module):
    """Maps z_t -> logit."""
    def __init__(self, in_dim, hidden=[256, 256], out_dim=1):
        super().__init__()
        self.out_dim = out_dim
        self.net = MLP([in_dim] + hidden + [out_dim])

    def forward(self, z_t):
        hidden = self.net(z_t)
        return hidden

class TreeConditionedCNP(nn.Module):
    def __init__(self,
                 d_theta, d_phi, d_y,
                 out_dim,
                 tree_config,
                 d_model=32,
                 encoder_hidden=[128,128],
                 mode="concat",
                 theta_embed_dim=None,
                 n_heads=4
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
        """
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
        
        # encoders
        # Target query encoder: x_t = (theta,phi)  → R_t (no y_t)
        d_phi_tgt = d_phi + self.tree.leaf_embed_dim
        self.qry_enc = FeatureEncoder(
            phi_dim=d_phi_tgt, y_dim=None, theta_in_dim=d_theta,
            hidden=encoder_hidden, out_dim=d_model,
            mode=mode, theta_embed_dim=theta_embed_dim, use_layernorm=True
        )

        #self.attn = CrossAttention(d_model, n_heads=n_heads)
        self.attn = CrossAttentionWithMoE(dim=d_model, num_heads=n_heads)

        # Norms
        self.norm_r  = nn.LayerNorm(d_model)

        in_dim = 2*d_model
        self.base_decoder  = DecoderHead(in_dim, out_dim=out_dim)  # logits

        self.register_buffer("temperature", torch.ones(1))

    def fit(self,X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        query_theta: torch.Tensor | None = None,
        query_phi: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        loader=None,**kwargs):
        if self.tree._fitted == False:
            self.tree.fit(X, y, query_theta, query_phi, target, loader)
            nsamples = query_phi.shape[-2] if loader is None else loader.dataset.num_samples()
            self.tree.enable_leaf_cache(nsamples)

    def forward(
        self,
        query_theta, query_phi,
        query_idx,
        context_theta, context_phi, context_y,
        context_idx,
        target_y,
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
        self.tree._out_device = device

        with torch.no_grad():
            out = self.tree(query_theta=context_theta, query_phi=context_phi, query_idx=context_idx)
        score_ctx = out["logits"][0]            # (B,T,1)
        leaf_emb = out["leaf_embeddings"]   # (B,T,embed_dim)

        phi_cnp_ctx = torch.cat([context_phi, score_ctx.to(device), leaf_emb.to(device)], dim=-1)
        # context encodings (x_c,y_c) → R_ctx
        R_ctx  = self.ctx_enc(context_theta, phi_cnp_ctx, y=context_y)  # (B,Nc,D)
        wS_ctx = (context_y.squeeze(-1) > 0.5).float()  
        # target query (x_t) → R_t
        with torch.no_grad():
            out = self.tree(query_theta=query_theta, query_phi=query_phi, query_idx=query_idx)
        score_tgt = out["logits"][0]            # (B,T,1)
        leaf_emb = out["leaf_embeddings"]   # (B,T,embed_dim)

        phi_cnp_tgt = torch.cat([query_phi, leaf_emb.to(device)], dim=-1)
        # context encodings (x_c,y_c) → R_ctx
        R_t = self.qry_enc(theta=query_theta, phi=phi_cnp_tgt)                  # (B,Nt,D)

        h_t = R_t
        for _ in range(1):
            h_t = self.attn(q_tokens=h_t, kv_tokens=R_ctx, key_mask=wS_ctx)


        logit_cnp = self.base_decoder(torch.cat([R_t, h_t], dim=-1))  
        
        logit = logit_cnp[...,0]
        if self.base_decoder.out_dim == 2:
            sigma = logit_cnp[...,1]
            sigma = sigma.clamp(min=-4.0, max=-0.5)  # σ in [~0.018, ~2.7]
            sigma = sigma.exp()
            lambda_ = 0.01
            loss, p = logit_normal_bernoulli_nll([logit,sigma], target_y)
            mu, sigma = p
            out.update({"logits": [logit], "Norm": [mu,sigma], "scores": score_tgt, "loss": lambda_* loss.mean()})
        else:
            out.update({"logits": [logit],"scores": score_tgt})
        
        return out

    def save(self, path):
        torch.save(self.state_dict(), path+'_model.pth')
        self.tree.save(path)


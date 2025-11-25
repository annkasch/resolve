import torch
import torch.nn as nn
from resolve.conditional_neural_process_family.class_attention import SimplePoolAttention
from resolve.conditional_neural_process_family.feature_encoder import FeatureEncoder, MLP
from resolve.network_architectures.xgboost import XGBoostWrapper

    
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


class TreeConditionedCNP(nn.Module):
    def __init__(self,
                 d_theta, d_phi, d_y,
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

        self.tree = XGBoostWrapper(config=tree_config["config"], task=tree_config.get("task","binary"), use_parameter_search=tree_config.get("use_parameter_search", False), use_leaf_embeddings=tree_config.get("use_leaf_embeddings", False))
    
        d_phi = d_phi + d_y + self.tree.leaf_embed_dim
        # Simpler & consistent with qry/tgt encoders:
        self.ctx_enc = FeatureEncoder(
            phi_dim=d_phi, y_dim=d_y, theta_in_dim=d_theta,
            hidden=encoder_hidden, out_dim=d_model,
            mode=mode, theta_embed_dim=theta_embed_dim, use_layernorm=True
        )
        # encoders
        # Target query encoder: x_t = (theta,phi)  → R_t (no y_t)
        self.qry_enc = FeatureEncoder(
            phi_dim=d_phi, y_dim=None, theta_in_dim=d_theta,
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

    def fit(self,X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        query_theta: torch.Tensor | None = None,
        query_phi: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        loader=None,):
        self.tree.fit(X, y, query_theta, query_phi, target, loader)

    def forward(
        self,
        query_theta, query_phi,
        query_idx,
        context_theta, context_phi, context_y,
        context_idx,
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
        # target query (x_t) → R_t
        with torch.no_grad():
            out = self.tree(query_theta=query_theta, query_phi=query_phi, query_idx=query_idx)
        score_tgt = out["logits"][0]            # (B,T,1)
        leaf_emb = out["leaf_embeddings"]   # (B,T,embed_dim)

        phi_cnp_tgt = torch.cat([query_phi, score_tgt.to(device), leaf_emb.to(device)], dim=-1)
        # context encodings (x_c,y_c) → R_ctx
        R_t = self.qry_enc(theta=query_theta, phi=phi_cnp_tgt)                  # (B,Nt,D)

        r_all, _ = self.attn(
            Q_src=R_t,                     # (B, Nt, D)
            K_src=self.norm_kv(R_ctx),     # (B, Nc, D)
            mask=mask_c,                   # (B, Nc)
            logit_bias_ctx=None,        # (B, Nc) or None
            beta=1.0             # scalar hyperparam, e.g. 1.0
        )

        rC = r_all.mean(dim=1, keepdim=True).expand(-1, R_t.shape[1], -1)     # (B,Nt,D)

        logits = self.base_decoder(torch.cat([R_t, rC], dim=-1))                                     # (B,Nt,1)

        return {"logits": logits, "scores": score_tgt}

    def save(self, path):
        torch.save(self.state_dict(), path+'_model.pth')
        self.tree.save(path)


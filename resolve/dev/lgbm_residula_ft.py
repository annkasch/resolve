import torch
import torch.nn as nn
import torch.nn.functional as F
from resolve.conditional_neural_process_family.feature_encoder import MLP
from resolve.network_architectures.transformer_encoder import TransformerEncoder
from resolve.network_architectures.lightgbm import LGBMWithLeafCache


class DecoderHead(nn.Module):
    """Maps z_t -> logit."""
    def __init__(self, in_dim, hidden=[256, 256], out_dim=1):
        super().__init__()
        self.net = MLP([in_dim] + hidden + [out_dim])
        self.out_dim=out_dim

    def forward(self, z_t):
        B, Nt, _ = z_t.shape
        return self.net(z_t.view(B*Nt, -1)).view(B, Nt, self.out_dim)

# ---------- full model ----------
class LGBMResidualFT(nn.Module):
    def __init__(self, d_theta, d_phi, d_y, tree_config, d_model=64, depth=1, n_heads=4, threshold = [0.,1.], use_tokenizer=False,use_cls_token=False):
        super().__init__()
        self.tree = LGBMWithLeafCache(config=tree_config["config"], 
                                   task=tree_config.get("task","binary"), 
                                   out_dim=d_y,
                                   num_samples=tree_config.get("num_samples", None),
                                   use_parameter_search=tree_config.get("use_parameter_search", False), 
                                   use_leaf_embeddings=tree_config.get("use_leaf_embeddings", False))
        
        # Context side
        use_cls_token = use_cls_token if use_tokenizer else False
        d_phi_tgt = d_phi + self.tree.leaf_embed_dim
        self.qrt_enc = TransformerEncoder(theta_dim=d_theta,
            phi_dim=d_phi_tgt,
            y_dim=None,
            embed_dim= d_model,
            depth= depth,
            num_heads= n_heads,
            mlp_ratio= 4.0,
            dropout= 0.0,
            proj_out_dim= None,  # if set, final linear to this dim
            use_cls_token= use_cls_token,      # set False: mean over feature tokens
            use_tokenizer= use_tokenizer,
            
        )
        self.threshold = threshold
        
        
        # decoder
        self.decoder = DecoderHead(d_model)

        self.register_buffer("temperature", torch.ones(1))

    def forward(self, query_theta, query_phi, query_idx, **kwargs):
        """
        Forward pass.

        Args:
            query_theta (Tensor): (B, Nt, d_theta)
            query_phi   (Tensor): (B, Nt, d_phi)

        Keyword Args:
            mask_c (Tensor, optional): (B, Nc) boolean mask for valid context rows.
        """
        device = query_phi.device
        with torch.no_grad():
            out = self.tree(query_theta=query_theta, query_phi=query_phi, query_idx=query_idx)
        score_lgbm = out["logits"][0].to(query_phi.device)            # (B,T,1)
        leaf_emb = out["leaf_embeddings"]   # (B,T,embed_dim)

        #mask = ((score_lgbm > self.threshold[0]) & (score_lgbm < self.threshold[1])).float().to(query_phi.device)

        phi_cnp_tgt = torch.cat([query_phi, leaf_emb.to(device)], dim=-1)
        # Transformer style encoder
        R_t = self.qrt_enc(theta=query_theta, phi=phi_cnp_tgt)  # (B, Nc, D)

        # decoder
        logit_ft = self.decoder(R_t)                         # (B,Nt,1)
        #eps = 1e-9
        #logit_lgbm = torch.log((score_lgbm + eps) / (1 - score_lgbm + eps))
        #logit_final = logit_lgbm + mask * logit_ft

        #out = {"logits": [logit_final], "score": score_lgbm}
        out = {"logits": [logit_ft], "score": score_lgbm}
        return out
    
    def save(self, path):
        torch.save(self.state_dict(), path+'_model.pth')
    
    def fit(self,X: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        query_theta: torch.Tensor | None = None,
        query_phi: torch.Tensor | None = None,
        target: torch.Tensor | None = None,
        loader=None,):
        if self.tree._fitted == False:
            self.tree.fit(X, y, query_theta, query_phi, target, loader)
            nsamples = query_phi.shape[-2] if loader is None else loader.dataset.num_samples()
            self.tree.enable_leaf_cache(nsamples, next(self.parameters()).device)
    
    def save(self, path):
        torch.save(self.state_dict(), path+'_model.pth')
        self.tree.save(path)
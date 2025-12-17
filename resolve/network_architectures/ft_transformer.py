import torch
import torch.nn as nn
import torch.nn.functional as F
from resolve.conditional_neural_process_family.feature_encoder import MLP
from resolve.network_architectures.transformer_encoder import TransformerEncoder


class BernoulliHead(nn.Module):
    """Maps z_t -> logit."""
    def __init__(self, in_dim, hidden=[256, 256], out_dim=1):
        super().__init__()
        self.net = MLP([in_dim] + hidden + [out_dim])
        self.out_dim=out_dim

    def forward(self, z_t):
        B, Nt, _ = z_t.shape
        return self.net(z_t.view(B*Nt, -1)).view(B, Nt, self.out_dim)

# ---------- full model ----------
class FTTransformer(nn.Module):
    def __init__(self, d_theta, d_phi, d_y, d_model=64, depth=1, n_heads=4, use_cls_token=True):
        super().__init__()
        # Context side
        self.qrt_enc = TransformerEncoder(theta_dim=d_theta,
            phi_dim=d_phi,
            y_dim=None,
            embed_dim= d_model,
            depth= depth,
            num_heads= n_heads,
            mlp_ratio= 4.0,
            dropout= 0.0,
            proj_out_dim= None,  # if set, final linear to this dim
            use_cls_token= use_cls_token,      # set False: mean over feature tokens
            use_tokenizer= True,
        )
        
        # Bernoulli decoder
        self.decoder = BernoulliHead(d_model)

    def forward(self, query_theta, query_phi, **kwargs):
        """
        Forward pass.

        Args:
            query_theta (Tensor): (B, Nt, d_theta)
            query_phi   (Tensor): (B, Nt, d_phi)

        Keyword Args:
            mask_c (Tensor, optional): (B, Nc) boolean mask for valid context rows.
        """

        # Transformer style encoder
        R_t = self.qrt_enc(query_theta, query_phi)  # (B, Nc, D)

        # Bernoulli decoder
        logit = self.decoder(R_t)                         # (B,Nt,1)

        out = {"logits": [logit]}

        return out
    
    def save(self,state, path):
        # drop all tree.leaf_cache.* entries from the state dict
        torch.save(state, path+'_model.pth')
    
    def load(self, path):
        state = torch.load(path+'_model.pth', map_location='cpu')
        self.load_state_dict(state['model_state'])
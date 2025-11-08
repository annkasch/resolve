import torch
import torch.nn as nn
import torch.nn.functional as F
from resolve.conditional_neural_process_family.feature_encoder import MLP
from resolve.conditional_neural_process_family.transformer_encoder import ContextTransformerEncoder
from resolve.conditional_neural_process_family.transformer_decoder import TargetTransformerDecoder

class BernoulliHead(nn.Module):
    """Maps z_t -> logit."""
    def __init__(self, in_dim, hidden=[256, 256]):
        super().__init__()
        self.net = MLP([in_dim] + hidden + [1])

    def forward(self, z_t):
        B, Nt, _ = z_t.shape
        return self.net(z_t.view(B*Nt, -1)).view(B, Nt, 1)

# ---------- full model ----------
class TransformerCNP(nn.Module):
    def __init__(self, d_theta, d_phi, d_y, d_model=32, encoder_sizes=[128,128], theta_embed_dim=32, n_heads=4, mode = 'film'):
        super().__init__()
        # Context side
        self.ctx_enc = ContextTransformerEncoder(theta_dim=d_theta,
            phi_dim=d_phi,
            y_dim=d_y,
            embed_dim= d_model,
            depth= 1,
            num_heads= 4,
            mlp_ratio= 4.0,
            dropout= 0.0,
            proj_out_dim= None,  # if set, final linear to this dim
            use_cls_token= False       # set False: mean over feature tokens
        )

        self.tgt_dec = TargetTransformerDecoder(target_theta_dim=d_theta,
            target_phi_dim=d_phi,
            y_dim=d_y,
            embed_dim= d_model,
            enc_dim= d_model,
            depth= 1,
            num_heads= 4,
            mlp_ratio= 4.0,
            dropout= 0.0,
            use_cls_token= False       # set False: mean over feature tokens
        )
        
        # Bernoulli decoder
        self.decoder = BernoulliHead(d_model)


    def forward(self, query_theta, query_phi, context_theta, context_phi, context_y,
            qry_theta_cell=None, return_ctx_for_write=True, **kwargs):
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

        # Context transformer style encoder
        R_ctx = self.ctx_enc(context_theta, context_phi, context_y)  # (B, Nc, D)

        # Target transformers style decoder -> z_t
        z_t = self.tgt_dec(query_theta, query_phi, R_ctx, context_y)

        # Bernoulli decoder
        logit = self.decoder(z_t)                         # (B,Nt,1)

        output = {
            "logits": [logit],
        }
        out = {"logits": [logit]}

        if return_ctx_for_write:
            out["R_ctx_for_write"] = R_ctx.detach()

        return out
import torch
import torch.nn as nn
from resolve.conditional_neural_process_family.feature_encoder import FeatureEncoder, MLP

class NeuralDensityRatioEstimator(nn.Module):
    def __init__(self,
                 d_theta, d_phi, d_y,
                 d_model=64,
                 encoder_hidden=[128,128],
                 decoder_hidden=[128,128],
                 mode="concat",
                 theta_embed_dim=None):
        super().__init__()
        self.qry_enc = FeatureEncoder(
            phi_dim=d_phi, y_dim=None, theta_in_dim=d_theta,
            hidden=encoder_hidden, out_dim=d_model,
            mode=mode, theta_embed_dim=theta_embed_dim, use_layernorm=True
        )
        self.head = MLP([d_model] + decoder_hidden + [d_y])

    def forward(
        self,
        query_theta, query_phi,
        **_
    ):

        R_t = self.qry_enc(theta=query_theta, phi=query_phi)   
        logits = self.head(R_t)

        return {"logits": [logits]}

    def save(self, path):
        torch.save(self.state_dict(), path+'_model.pth')


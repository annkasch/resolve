import torch
import torch.nn as nn
import torch.nn.functional as F
from resolve.conditional_neural_process_family.context_encoder import MLP

# ---------- target side ----------
class TargetQueryEncoder(nn.Module):
    """Builds R^{(t)} from (θ^t, φ^t)."""
    def __init__(self, theta_in_dim, phi_dim, hidden, out_dim, theta_encoder=None):
        super().__init__()
        self.theta_enc = theta_encoder
        theta_feat_dim = theta_in_dim if theta_encoder is None else \
                         theta_encoder.mlp.net[-1].out_features
        in_dim = theta_feat_dim + phi_dim
        self.mlp = MLP([in_dim] + hidden + [out_dim])

    def forward(self, query_theta, query_phi):
        theta_f = self.theta_enc(query_theta) if self.theta_enc else query_theta
        x = torch.cat([theta_f, query_phi], dim=-1)
        leading = x.shape[:-1]
        out = self.mlp(x.view(-1, x.shape[-1]))
        return out.view(*leading, -1)            # (B,Nt,D)

class TargetEncoder(nn.Module):
    """z_t = f([θ^t, φ^t, r^+_t, r^-_t])."""
    def __init__(self, theta_in_dim, phi_dim, r_dim, hidden, out_dim, theta_encoder=None):
        super().__init__()
        self.theta_enc = theta_encoder
        theta_feat_dim = theta_in_dim if theta_encoder is None else \
                         theta_encoder.mlp.net[-1].out_features
        in_dim = theta_feat_dim + phi_dim + 2*r_dim
        self.mlp = MLP([in_dim] + hidden + [out_dim])

    def forward(self, query_theta, query_phi, r_pos, r_neg):
        if r_pos.shape[1] == 1 and query_phi.shape[1] > 1:  # safety
            r_pos = r_pos.expand(-1, query_phi.shape[1], -1)
            r_neg = r_neg.expand_as(r_pos)
        theta_f = self.theta_enc(query_theta) if self.theta_enc else query_theta
        x = torch.cat([theta_f, query_phi, r_pos, r_neg], dim=-1)
        B, Nt, _ = x.shape
        z = self.mlp(x.view(B*Nt, -1))
        return z.view(B, Nt, -1)
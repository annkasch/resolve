import torch
import torch.nn as nn
from resolve.conditional_neural_process_family.context_encoder import MLP

def masked_mean(x, mask, dim=1, keepdim=False, eps=1e-8):
    # x: (B,N,*) , mask: (B,N) bool
    w = mask.float()
    s = (x * w.unsqueeze(-1)).sum(dim=dim, keepdim=keepdim)
    n = w.sum(dim=dim, keepdim=keepdim).clamp_min(eps)
    return s / n

def kl_gauss(mu_q, logvar_q, mu_p, logvar_p):
    # elementwise KL( N(mu_q,σ_q^2) || N(mu_p,σ_p^2) ); returns (B,)
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kl = 0.5 * (
        (var_q / var_p) +
        ((mu_p - mu_q).pow(2) / var_p) -
        1.0 + (logvar_p - logvar_q)
    ).sum(-1)
    return kl  # (B,)

class LatentPosterior(nn.Module):
    """q(z|C) from pooled context embedding r_bar (dim=D)."""
    def __init__(self, d_model, z_dim=8):
        super().__init__()
        self.net = MLP([d_model, d_model, 2*z_dim])  # outputs [mu, logvar]

    def forward(self, r_bar):
        h = self.net(r_bar)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

class LatentPriorTheta(nn.Module):
    """Optional p(z|thetā) prior head; otherwise use N(0,I)."""
    def __init__(self, d_theta, z_dim=8):
        super().__init__()
        self.net = MLP([d_theta, 2*z_dim])

    def forward(self, theta_bar):
        h = self.net(theta_bar)
        mu, logvar = torch.chunk(h, 2, dim=-1)
        return mu, logvar

class LatentTokenProj(nn.Module):
    """z -> token in context space, and a base-rate weight s in [0,1]."""
    def __init__(self, z_dim, d_model):
        super().__init__()
        self.to_token = MLP([z_dim, d_model])
        self.to_base = MLP([z_dim, 1])  # sigmoid later

    def forward(self, z):
        token = self.to_token(z)             # (B,D)
        s = torch.sigmoid(self.to_base(z))   # (B,1)  ~ P(signal | z)
        return token, s
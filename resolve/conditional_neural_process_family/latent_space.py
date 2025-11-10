import torch
import torch.nn as nn
from resolve.conditional_neural_process_family.feature_encoder import MLP

def masked_mean(x, mask, dim=1, keepdim=False, eps=1e-8):
    w = mask.float()
    s = (x * w.unsqueeze(-1)).sum(dim=dim, keepdim=keepdim)
    n = w.sum(dim=dim, keepdim=keepdim).clamp_min(eps)
    return s / n

def kl_gauss(mu_q, logvar_q, mu_p, logvar_p):
    var_q = logvar_q.exp()
    var_p = logvar_p.exp()
    kl = 0.5 * ((var_q / var_p) + ((mu_p - mu_q).pow(2) / var_p) - 1.0 + (logvar_p - logvar_q)).sum(-1)
    return kl

class LatentPosteriorCT(nn.Module):
    def __init__(self, d_model, z_dim=8):
        super().__init__()
        self.net = MLP([2*d_model, d_model, 2*z_dim])
    def forward(self, rbar_c, rbar_t):
        h = self.net(torch.cat([rbar_c, rbar_t], -1))
        return torch.chunk(h, 2, -1)

class LatentPriorC(nn.Module):
    def __init__(self, d_model, z_dim=8):
        super().__init__()
        self.net = MLP([d_model, d_model, 2*z_dim])
    def forward(self, rbar_c):
        h = self.net(rbar_c)
        return torch.chunk(h, 2, -1)

class LatentPriorTheta(nn.Module):
    def __init__(self, d_theta, z_dim=8):
        super().__init__()
        self.net = MLP([d_theta, 2*z_dim])
    def forward(self, thetabar):
        h = self.net(thetabar)
        return torch.chunk(h, 2, -1)

class LatentTokenProj(nn.Module):
    def __init__(self, z_dim, d_model, s_bias_init=-3.0):
        super().__init__()
        self.to_token = MLP([z_dim, d_model])
        self.to_base  = MLP([z_dim, 1])
        last_linear = None
        for m in self.to_base.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None and last_linear.bias is not None:
            nn.init.constant_(last_linear.bias, s_bias_init)
    def forward(self, z):
        token = self.to_token(z)               # (B,D)
        s     = torch.sigmoid(self.to_base(z)) # (B,1)
        return token, s
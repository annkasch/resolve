from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- utils ----------
class MLP(nn.Module):
    def __init__(self, sizes, activation=nn.ReLU, last_activation=False):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i+1]))
            is_last = (i == len(sizes) - 2)
            if (not is_last) or last_activation:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MLPEncoder(nn.Module):
    """Encodes x into an embedding."""
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        layers = [in_dim] + (hidden or []) + [out_dim]
        self.mlp = MLP(layers)

    def forward(self, x):
        leading = x.shape[:-1]
        flat = x.reshape(-1, x.shape[-1])
        z = self.mlp(flat)
        return z.view(*leading, -1)

# encoder: concate: R^{(θ, phi)} or film: R^{(phi,y|θ)}
class FeatureEncoder(nn.Module):
    """
    Encodes (theta, phi, y) -> R^(c)
    Shapes:
      theta: (B, Nc, d_theta)
      phi:   (B, Nc, d_phi)
      y:     (B, Nc, d_y)
      return:  (B, Nc, D)
    """
    def __init__(self,
                 phi_dim: int,
                 theta_in_dim: int,
                 hidden: list[int],
                 out_dim: int,
                 y_dim: Optional[int],
                 theta_embed_dim: Optional[int],
                 mode: str = "concat",
                 use_layernorm: bool = False):
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in {"concat", "film"}, "mode must be 'concat' or 'film'"
        theta_embed_dim = int(theta_embed_dim) if theta_embed_dim is not None else 0
        # θ encoder used only to condition the context features
        self.theta_enc = MLPEncoder(theta_in_dim, [], theta_embed_dim) if theta_embed_dim > 0 else None

        if self.mode == "concat":
            in_dim = theta_in_dim + phi_dim 
            if y_dim:
                in_dim += y_dim
            # allow empty hidden
            layers = [in_dim] + (hidden or []) + [out_dim]
            self.content = MLP(layers)
            self.norm = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()

        else:  # FiLM
            # feature trunk on [phi,y]
            base_dims = [phi_dim + y_dim] + (hidden or [out_dim])
            self.feature_layers = nn.ModuleList()
            self.film_layers = nn.ModuleList()
            for i in range(len(base_dims) - 1):
                din, dout = base_dims[i], base_dims[i+1]
                self.feature_layers.append(nn.Linear(din, dout))
                self.film_layers.append(nn.Linear(theta_embed_dim, 2 * dout))
            # final projection if hidden was provided; if not, FiLM already outputs out_dim
            self.final = (
                nn.Linear(base_dims[-1], out_dim)
                if base_dims[-1] != out_dim
                else nn.Identity()
            )
            self.norm = nn.LayerNorm(out_dim) if use_layernorm else nn.Identity()

    def forward(self, theta, phi, y=None):
        B, Nc, _ = phi.shape

        if self.mode == "concat":
            x = torch.cat([theta, phi, y], dim=-1) if y is not None else torch.cat([theta, phi], dim=-1)
            out = self.content(x.view(B * Nc, -1)).view(B, Nc, -1)
            return self.norm(out)

        # FiLM path
        theta_emb = self.theta_enc(theta)                       # (B,Nc,Eθ)
        h = torch.cat([phi, y], dim=-1).view(B * Nc, -1) if y is not None else phi.view(B * Nc, -1)
        cond = theta_emb.view(B * Nc, -1)                         # (B*Nc, Eθ)
        for lin, film in zip(self.feature_layers, self.film_layers):
            h = lin(h)
            gamma, beta = film(cond).chunk(2, dim=-1)            # (B*Nc, d)
            h = F.relu(gamma * h + beta)
        h = self.final(h)
        h = h.view(B, Nc, -1)
        return self.norm(h)
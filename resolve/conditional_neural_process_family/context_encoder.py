import torch
import torch.nn as nn
import torch.nn.functional as F
from resolve.conditional_neural_process_family.class_attention import GlobalContextAttentionDual,  GlobalContextAttention
from resolve.conditional_neural_process_family.memory_bank import MemoryBank

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

class ThetaEncoder(nn.Module):
    """Encodes θ into an embedding."""
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.mlp = MLP([in_dim] + hidden + [out_dim])

    def forward(self, theta):
        leading = theta.shape[:-1]
        flat = theta.reshape(-1, theta.shape[-1])
        z = self.mlp(flat)
        return z.view(*leading, -1)

# context encoder: R^{(c,phi|θ)}
class ContextConditionalEncoder(nn.Module):
    """
    Encodes (context_theta, context_phi, context_y) -> R^(c)
    Shapes:
      context_theta: (B, Nc, d_theta)
      context_phi:   (B, Nc, d_phi)
      context_y:     (B, Nc, d_y)
      return:  (B, Nc, D)
    """
    def __init__(self,
                 phi_dim: int,
                 y_dim: int,
                 theta_in_dim: int,
                 theta_embed_dim: int,
                 hidden: list[int],
                 out_dim: int,
                 mode: str = "concat",
                 use_layernorm: bool = False):
        super().__init__()
        self.mode = mode.lower()
        assert self.mode in {"concat", "film"}, "mode must be 'concat' or 'film'"

        # θ encoder used only to condition the context features
        self.theta_enc = ThetaEncoder(theta_in_dim, [], theta_embed_dim)

        if self.mode == "concat":
            in_dim = phi_dim + y_dim + theta_embed_dim
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

    def forward(self, context_phi, context_y, context_theta):
        B, Nc, _ = context_phi.shape
        theta_emb = self.theta_enc(context_theta)                       # (B,Nc,Eθ)

        if self.mode == "concat":
            x = torch.cat([context_phi, context_y, theta_emb], dim=-1)        # (B,Nc,·)
            out = self.content(x.view(B * Nc, -1)).view(B, Nc, -1)
            return self.norm(out)

        # FiLM path
        h = torch.cat([context_phi, context_y], dim=-1).view(B * Nc, -1)      # (B*Nc, d)
        cond = theta_emb.view(B * Nc, -1)                         # (B*Nc, Eθ)
        for lin, film in zip(self.feature_layers, self.film_layers):
            h = lin(h)
            gamma, beta = film(cond).chunk(2, dim=-1)            # (B*Nc, d)
            h = F.relu(gamma * h + beta)
        h = self.final(h)
        h = h.view(B, Nc, -1)
        return self.norm(h)
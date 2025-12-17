import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent


# -------------------------
# Small helper MLP
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class AffineCoupling(nn.Module):
    """
    y = x * exp(s) + t on unmasked dims, identity on masked dims.
    s is *bounded* to keep exp(s) in a safe range.
    """
    def __init__(self, dim, hidden_dims, mask, scale_limit=2.0):
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", mask)  # (D,)
        self.scale_limit = scale_limit

        self.st_net = MLP(in_dim=dim, hidden_dims=hidden_dims, out_dim=2 * dim)

        # important: start near identity – last layer zero init
        last_linear = None
        for m in self.st_net.modules():
            if isinstance(m, nn.Linear):
                last_linear = m
        if last_linear is not None:
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def forward(self, x):
        m = self.mask
        mx = x * m

        st = self.st_net(mx)
        s, t = st.chunk(2, dim=-1)

        # *** BOUND s to avoid exp overflow ***
        s = torch.tanh(s) * self.scale_limit  # s in [-scale_limit, scale_limit]

        # only apply to unmasked dims
        inv_m = 1.0 - m
        s = s * inv_m
        t = t * inv_m

        y = x * torch.exp(s) + t
        log_det = s.sum(dim=-1)

        # DEBUG safety check
        if not torch.isfinite(y).all():
            print("Non-finite in AffineCoupling.forward")
            print("  s min/max:", s.min().item(), s.max().item())
            print("  t min/max:", t.min().item(), t.max().item())
            print("  x min/max:", x.min().item(), x.max().item())
            raise RuntimeError("NaN/inf in AffineCoupling")

        return y, log_det

    def inverse(self, y):
        m = self.mask
        my = y * m

        st = self.st_net(my)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * self.scale_limit

        inv_m = 1.0 - m
        s = s * inv_m
        t = t * inv_m

        x = (y - t) * torch.exp(-s)
        return x
'''
# -------------------------
# RealNVP affine coupling layer
# -------------------------
class AffineCoupling(nn.Module):
    """
    RealNVP-style affine coupling layer.

    y = x * m + (1 - m) * (x * exp(s) + t)
    where (s, t) = NN(m * x)

    Both forward (x -> y) and inverse (y -> x) are cheap and exact.
    """
    def __init__(self, dim, hidden_dims, mask):
        super().__init__()
        self.dim = dim
        self.register_buffer("mask", mask)  # (D,)

        # Network that predicts scale and shift for the unmasked part
        self.st_net = MLP(in_dim=dim, hidden_dims=hidden_dims, out_dim=2 * dim)

    def forward(self, x):
        """
        Forward transformation: x -> y, plus log|det J|
        x: (N, D)
        Returns:
            y: (N, D)
            log_det: (N,)
        """
        #print("x has NaNs:", torch.isnan(x).any().item())
        #print("x has infs:", torch.isinf(x).any().item())
        m = self.mask
        mx = x * m

        st = self.st_net(mx)
        s, t = st.chunk(2, dim=-1)

        # Only affect unmasked dims
        s = s * (1 - m)
        t = t * (1 - m)

        y = x * torch.exp(s) + t
        # log det is sum of s over transformed dims
        log_det = (s).sum(dim=-1)

        return y, log_det

    def inverse(self, y):
        """
        Inverse transformation: y -> x.
        Uses that the masked part is identity and y_mask = x_mask.
        """
        m = self.mask
        my = y * m

        st = self.st_net(my)
        s, t = st.chunk(2, dim=-1)

        s = s * (1 - m)
        t = t * (1 - m)

        # Solve x from y = x * exp(s) + t  on unmasked dims
        x = (y - t) * torch.exp(-s)
        return x
'''

# -------------------------
# Full Normalizing Flow
# -------------------------
class NVPFlow(nn.Module):
    """
    Classic RealNVP-style normalizing flow for density estimation.

    - Base distribution: standard Normal on R^D
    - flow_layers: sequence of affine coupling layers with alternating masks

    Methods:
      - log_prob(x): log p_X(x)
      - sample(num_samples): draws x ~ p_X
    """
    def __init__(self, dim, n_flow_layers=6, hidden_dims=[128, 128]):
        super().__init__()
        self.dim = dim

        # Create alternating binary masks
        mask_even = torch.cat([torch.ones(dim // 2),
                               torch.zeros(dim - dim // 2)])
        mask_odd = 1.0 - mask_even

        masks = []
        for i in range(n_flow_layers):
            masks.append(mask_even if i % 2 == 0 else mask_odd)

        self.flow_layers = nn.ModuleList([
            AffineCoupling(dim=dim, hidden_dims=hidden_dims, mask=masks[i])
            for i in range(n_flow_layers)
        ])

        # Standard Normal base distribution over R^D
        self.register_buffer("base_loc", torch.zeros(dim))
        self.register_buffer("base_scale", torch.ones(dim))
    
    def _base_dist(self):
        # this will be on whatever device base_loc/base_scale are on
        return Independent(Normal(self.base_loc, self.base_scale), 1)

    # ---- x -> z, log p(x) ----
    def forward(self, query_theta, query_phi, **kwargs):
        """
        Apply all coupling layers: x -> z, accumulate log_det.
        Returns:
            z: (N, D)
            log_det_total: (N,)
        """
        x = torch.cat([query_theta, query_phi], dim=-1)
        if not torch.isfinite(x).all():
            print("Input x has non-finite values!")
            print("x stats:", x.min().item(), x.max().item())
            raise RuntimeError("Non-finite input to flow.")
        log_det_total = torch.zeros(x.shape[0], device=x.device)
        z = x

        for i, layer in enumerate(self.flow_layers):
            z, log_det = layer(z)
            if not torch.isfinite(z).all():
                print(f"NaNs or infs after flow layer {i}")
                print("z stats:", z.min().item(), z.max().item())
                raise RuntimeError("Non-finite values in z")
            log_det_total = log_det_total + log_det
        
        log_prob = log_det_total + self.log_prob(z)
        loss = -log_prob.mean()

        # Sampling
        with torch.no_grad():
            samples = self.sample(x.shape[-2])   # (1000, 2)

        output = {"logits": [samples], "loss": loss, "log_prob": log_prob}
        return output

    # ---- z -> x (for sampling) ----
    def _inverse_flow(self, z):
        """
        Apply inverse coupling layers in reverse order: z -> x.
        """
        x = z
        for layer in reversed(self.flow_layers):
            x = layer.inverse(x)
        return x

    def log_prob(self, z):
        """
        Compute log p_X(x) via change of variables:
        log p_X(x) = log p_Z(z) + log |det J_f(x)|
        where z = f(x)
        """

        base_dist = self._base_dist()
        log_pz = base_dist.log_prob(z)
        return log_pz                       # (N,)

    def sample(self, num_samples, device=None):
        """
        Sample x ~ p_X by sampling from base and applying inverse flow.
        """
        if device is None:
            device = next(self.parameters()).device
        z = self._base_dist().sample((num_samples,)).to(device)  # (N, D)
        x = self._inverse_flow(z)
        return x

    def save(self,state, path):
        # drop all tree.leaf_cache.* entries from the state dict
        torch.save(state, path+'_model.pth')
    
    def load(self, path):
        state = torch.load(path+'_model.pth', map_location='cpu')
        self.load_state_dict(state['model_state'])
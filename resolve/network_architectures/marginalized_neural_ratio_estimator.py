import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(in_dim, hidden_dims, out_dim=None, dropout_p=0.0, final_activation=None):
    layers = []
    prev = in_dim
    for h in hidden_dims:
        layers += [nn.Linear(prev, h), nn.ReLU(inplace=True)]
        if dropout_p > 0:
            layers.append(nn.Dropout(dropout_p))
        prev = h
    if out_dim is not None:
        layers.append(nn.Linear(prev, out_dim))
        if final_activation is not None:
            layers.append(final_activation)
    return nn.Sequential(*layers)


class MarginalizedNeuralRatioEstimator(nn.Module):
    """
    φ-marginalizing baseline ratio model (no FiLM, no Memory, no Latents).

    Learns a pointwise log-density ratio r(theta, phi) ≈ log p(theta,phi|S) - log p(theta,phi|B),
    then marginalizes over φ via log-mean-exp to produce a θ-level score:

        z(theta) = log E_{phi ~ p(phi|theta)} [ exp( r(theta, phi) ) ]

    Inputs:
      - query_theta: shape (B, T, d_theta)
      - query_phi:
          * shape (B, T, d_phi)           -> treated as M=1 (no Monte Carlo)
          * shape (B, T, M, d_phi)        -> Monte Carlo marginalization over axis M
      - (optional) phi_log_weights: shape (B, T, M) log-weights (e.g., log p/q for IW). If absent, uniform.

    Output:
      output["logits"] = [r_theta_phi, z_theta]
        - r_theta_phi: pointwise logits, shape (B, T, M) or (B, T) if M=1
        - z_theta:     marginalized θ-level logit, shape (B, T)
    """

    def __init__(
        self,
        d_theta: int,
        d_phi: int,
        theta_hidden_dims=(128, 64),
        phi_hidden_dims=(128, 64),
        head_hidden_dims=(128, 64),
        dropout_p: float = 0.0,
    ):
        super().__init__()

        # Encoders
        self.theta_encoder = _mlp(d_theta, list(theta_hidden_dims), out_dim=theta_hidden_dims[-1], dropout_p=dropout_p)
        self.phi_encoder   = _mlp(d_phi,   list(phi_hidden_dims),   out_dim=phi_hidden_dims[-1],   dropout_p=dropout_p)

        # Ratio head: takes concatenated [R_theta, R_phi] -> scalar logit r(theta,phi)
        comb_in = theta_hidden_dims[-1] + phi_hidden_dims[-1]
        self.ratio_head = _mlp(comb_in, list(head_hidden_dims), out_dim=1, dropout_p=dropout_p)

    def _ensure_phi_has_mc_axis(self, query_phi):
        """
        Ensure query_phi has an explicit MC axis M.
        - If input is (B, T, d_phi), convert to (B, T, 1, d_phi) and return flag M=1.
        - If input is already (B, T, M, d_phi), return as is and flag M>1.
        """
        if query_phi.dim() == 3:
            # (B, T, d_phi) -> (B, T, 1, d_phi)
            return query_phi.unsqueeze(2), True
        elif query_phi.dim() == 4:
            return query_phi, False
        else:
            raise ValueError(f"query_phi must be (B,T,d_phi) or (B,T,M,d_phi), got shape {tuple(query_phi.shape)}")

    def forward(self, query_theta, query_phi, target_y, phi_log_weights=None, phi_mask=None, **kwargs):
        """
        Args:
            query_theta: (B, T, d_theta)
            query_phi:   (B, T, d_phi) or (B, T, M, d_phi)
            phi_log_weights (optional): (B, T, M) log-weights for importance weighting
            phi_mask (optional): (B, T, M) boolean mask where True=keep, False=exclude

        Returns:
            dict with:
              - "logits": [r_theta_phi, z_theta]
                  r_theta_phi: (B, T, M) or (B, T) if M==1
                  z_theta: (B, T)
              - "aux": shapes and intermediates (optional for debugging)
        """
        B, T, d_theta = query_theta.shape
        # Encode theta: (B,T,dθ) -> (B,T,hθ)
        R_theta = self.theta_encoder(query_theta)  # broadcasted over last dim by nn.Sequential

        # Ensure φ has MC axis
        query_phi_mc, squeezed = self._ensure_phi_has_mc_axis(query_phi)  # (B,T,M,dφ)
        Bp, Tp, M, d_phi = query_phi_mc.shape
        assert Bp == B and Tp == T, "Theta and Phi batch/time dims must match"

        # Encode phi: (B,T,M,dφ) -> (B,T,M,hφ)
        R_phi = self.phi_encoder(query_phi_mc)

        # Expand/align theta encoding across M: (B,T,1,hθ) -> (B,T,M,hθ)
        R_theta_exp = R_theta.unsqueeze(2).expand(B, T, M, R_theta.shape[-1])

        # Combine and get pointwise ratio logits: (B,T,M, hθ+hφ) -> (B,T,M,1) -> (B,T,M)
        R_cat = torch.cat([R_theta_exp, R_phi], dim=-1)
        r_logits = self.ratio_head(R_cat).squeeze(-1)  # (B,T,M)

        # Handle optional importance weights / masking
        if phi_log_weights is None:
            phi_log_weights = torch.zeros((B, T, M), device=r_logits.device, dtype=r_logits.dtype)
        else:
            # Validate shape
            if phi_log_weights.shape != (B, T, M):
                raise ValueError(f"phi_log_weights must be (B,T,M), got {tuple(phi_log_weights.shape)}")

        phi_mask = target_y > 0.5
        if phi_mask is not None:
            # Masked positions get -inf log-weight so they drop out of log-sum-exp
            if phi_mask.shape != (B, T, M):
                raise ValueError(f"phi_mask must be (B,T,M), got {tuple(phi_mask.shape)}")
            neg_inf = torch.finfo(r_logits.dtype).min
            phi_log_weights = phi_log_weights.masked_fill(~phi_mask, neg_inf)

        # φ-marginalization: z_theta = logsumexp(log_w + r) - logsumexp(log_w)
        # Numerically stable with torch.logsumexp
        numerator   = torch.logsumexp(phi_log_weights + r_logits, dim=2)       # (B,T)
        denom_norm  = torch.logsumexp(phi_log_weights,           dim=2)        # (B,T)
        z_theta = numerator - denom_norm                                         # (B,T)

        # If input φ had shape (B,T,dφ), squeeze r_logits back to (B,T)
        if squeezed:
            r_out = r_logits.squeeze(2)  # (B,T)
        else:
            r_out = r_logits             # (B,T,M)

        output = {
            "logits": [r_out, z_theta],
            "aux": {
                "R_theta_shape": tuple(R_theta.shape),
                "R_phi_shape": tuple(R_phi.shape),
                "r_logits_shape": tuple(r_logits.shape),
                "z_theta_shape": tuple(z_theta.shape),
                "used_mc_samples": M,
            },
        }
        return output

    @torch.no_grad()
    def score_theta(self, query_theta, query_phi, phi_log_weights=None, phi_mask=None):
        """
        Convenience wrapper to get only the marginalized θ-level logit z(theta).
        """
        out = self.forward(query_theta, query_phi, phi_log_weights=phi_log_weights, phi_mask=phi_mask)
        return out["logits"][1]
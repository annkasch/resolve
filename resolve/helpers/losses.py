import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def bce_with_logits(z, y, **kward):
    z[0]=z[0].view(-1)
    return F.binary_cross_entropy_with_logits(z[0], y.view(-1), reduction="none"), torch.sigmoid(z[0])

def log_prob(z, y, **kward):
    dist = torch.distributions.normal.Normal(loc=z[0], scale=z[1])
    #dist = torch.distributions.Independent(torch.distributions.Normal(loc=z[0], scale=z[1]), reinterpreted_batch_ndims=1  # This assumes the last dim is your m-dim output
    #    )
    log_prob = -1.*dist.log_prob(y)
    return log_prob.view(-1), z[0].view(-1)

class AsymmetricFocalWithFPPenalty(nn.Module):
    """
    Binary classification loss:
      L = AFL(logits, targets)
          + lambda_fp * E_{y=0}[ ReLU(sigmoid(z) - tau_fp)^2 ],
    where z = logits + bias, bias = logit(prior_pos) if prior_pos is given.

    Asymmetric Focal Loss (AFL):
      positives:  α_pos * (1 - p)^γ_pos * BCE(z, y=1)
      negatives:  α_neg * (    p)^γ_neg * BCE(z, y=0)

    Args:
        prior_pos: float in (0,1) or None. If None, no bias shift is applied.
        alpha_pos: weight for positives.
        alpha_neg: weight for negatives. If None, uses (1 - alpha_pos).
        gamma_pos: focusing parameter for positives.
        gamma_neg: focusing parameter for negatives.
        lambda_fp: weight of the false-positive quadratic penalty (negatives only).
        lambda_tp: weight of the true-positive quadratic reward (positives only).
        tau_fp: probability threshold for penalizing negatives (p > tau_fp).
        reduction: 'mean' | 'sum' | 'none'
        eps: small constant for numeric stability.
    """
    def __init__(
        self,
        prior_pos: float | None = None,
        alpha_pos: float = 0.2,
        alpha_neg: float | None = None,
        gamma_pos: float = 2.0,
        gamma_neg: float = 4.0,
        lambda_fp: float = 0.0,
        lambda_tp: float = 0.0,
        tau_fptp: float = 0.5,
        reduction: str = "mean",
        eps: float = 1e-6,
        base_loss_fn = bce_with_logits
    ):
        super().__init__()
        assert 0.0 <= alpha_pos <= 2.0
        if alpha_neg is None:
            alpha_neg = 1.0 - alpha_pos
        assert 0.0 <= alpha_neg <= 2.0
        assert 0.0 <= tau_fptp <= 1.0
        assert lambda_fp >= 0.0
        assert lambda_tp >= 0.0
        assert reduction in ("mean", "sum", "none")

        self.alpha_pos = float(alpha_pos)
        self.alpha_neg = float(alpha_neg)
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.lambda_fp = float(lambda_fp)
        self.lambda_tp = float(lambda_tp)
        self.tau_fptp = float(tau_fptp)
        self.reduction = reduction
        self.eps = float(eps)
        self.base_loss_fn = base_loss_fn

        # Precompute a stable bias (or disable if None)
        if prior_pos is None:
            self.bias = 0.0
        else:
            pi = min(max(float(prior_pos), eps), 1.0 - eps)
            self.bias = math.log(pi / (1.0 - pi))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (N,) or (N,1) raw scores
        targets: (N,) or (N,1) with values in {0,1}
        """
        # Shift + probabilities (p used by focal weights and FP penalty)
        z = logits
        z[0] = logits[0] + self.bias

        # Base per-sample loss (N,)
        base, p = self.base_loss_fn(z, targets)
        
        
        # Masks (allow slightly fuzzy labels; >=0.5 -> positive)
        targets = targets.view(-1).float()
        pos_mask = targets >= 0.5
        neg_mask = ~pos_mask

        # Asymmetric focal weights
        # Compute asymmetric focal weights without boolean indexing copies
        # w = alpha_pos * (1 - p)^gamma_pos on positives, else alpha_neg * p^gamma_neg
        one_minus_p = 1.0 - p
        w_pos = self.alpha_pos * torch.pow(one_minus_p, self.gamma_pos)
        w_neg = self.alpha_neg * torch.pow(p,           self.gamma_neg)
        weight = torch.where(pos_mask, w_pos, w_neg)

        # Focal term
        focal_term = weight * base

        # False-positive penalty for negatives: (ReLU(p - tau_fp))^2
        if self.lambda_fp > 0.0 and neg_mask.any():
            overshoot = torch.relu(p[neg_mask] - self.tau_fptp)
            penalty = overshoot # ** 2 #a smooth, differentiable penalty that grows quadratically with the model’s confidence in the wrong direction
            loss = focal_term.clone()
            loss[neg_mask] = loss[neg_mask] + self.lambda_fp * penalty
        else:
            loss = focal_term
        
        if self.lambda_tp > 0.0 and pos_mask.any():
            overshoot = torch.relu(p[pos_mask] - self.tau_fptp)
            reward = overshoot # ** 2 # 
            loss[pos_mask] = loss[pos_mask] + self.lambda_tp * reward

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
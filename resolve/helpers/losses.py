from gettext import npgettext
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def bce_with_logits(z, y, **kward):
    # z can be list/tuple or tensor
    z0 = z[0] if isinstance(z, (list, tuple)) else z
    z0 = z0.reshape(-1)
    y  = y.reshape(-1)
    return F.binary_cross_entropy_with_logits(z0, y, reduction="none"), torch.sigmoid(z0)

def log_prob(z, y, **kward):
    z0 = z[0] if isinstance(z, (list, tuple)) else z
    z1 = z[1] if (isinstance(z, (list, tuple)) and len(z) > 1) else None
    if z1 is None:
        raise ValueError("log_prob expects z=[mu, sigma].")
    dist = torch.distributions.Normal(loc=z0, scale=z1)
    # per-sample negative log-likelihood
    nll = -dist.log_prob(y).reshape(-1)
    return nll, z0.reshape(-1)

def skip_loss(z, y, **kward):
    Y = torch.full_like(y.reshape(-1), float("nan"))
    return Y, Y

def brier(z, y, **kward):
    z0 = z[0] if isinstance(z, (list, tuple)) else z
    z0 = z0.reshape(-1)
    y  = y.reshape(-1)
    p = torch.sigmoid(z0)
    mse = F.mse_loss(p, y, reduction="none")
    return mse, p

def recon_loss_mse(x_hat, y, x, **kward):
    # x_hat[0]: (N, M) or (B, T, M); x ground truth of same last-dim
    xh = x_hat[0]
    if x is None:
        raise ValueError("recon_loss_mse requires keyword arg x=<target features>")
    # flatten batch/time but keep feature dim
    if xh.dim() > 2:
        xh = xh.reshape(-1, xh.shape[-1])
    if x.dim() > 2:
        x = x.reshape(-1, x.shape[-1])
    mse_vec = F.mse_loss(xh, x, reduction="none").mean(dim=1)  # per-sample
    return mse_vec, mse_vec

class AsymmetricFocalWithFPPenalty(nn.Module):
    """
    Binary classification loss:
      L = AFL(logits, targets_y)
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
    """
    def __init__(
        self,
        alpha_pos: float = 0.2,
        alpha_neg: float | None = None,
        gamma_pos: float = 2.0,
        gamma_neg: float = 4.0,
        lambda_fp: float = 0.0,
        tau_fp: float = 0.5,
        lambda_tp: float = 0.0,
        tau_tp: float = 0.5,
        reduction: str = "mean",
        base_loss_fn = bce_with_logits
    ):
        super().__init__()
        assert alpha_pos >= 0.0
        if alpha_neg is None:
            alpha_neg = 1.0 - alpha_pos
        assert alpha_neg >= 0.0
        assert 0.0 <= tau_fp <= 1.0
        assert 0.0 <= tau_tp <= 1.0
        assert lambda_fp >= 0.0
        assert lambda_tp >= 0.0
        assert reduction in ("mean", "sum", "none")

        self.alpha_pos = float(alpha_pos)
        self.alpha_neg = float(alpha_neg)
        self.gamma_pos = float(gamma_pos)
        self.gamma_neg = float(gamma_neg)
        self.lambda_fp = float(lambda_fp)
        self.lambda_tp = float(lambda_tp)
        self.tau_fp    = float(tau_fp)
        self.tau_tp    = float(tau_tp)
        self.reduction = reduction
        self.base_loss_fn = base_loss_fn
        self.p = None
    
    def _ensure_container(self, logits):
        # normalize to a list-like [z, ...]
        if isinstance(logits, (list, tuple)):
            return list(logits)
        return [logits]

    def forward(self, logits: torch.Tensor, targets_y: torch.Tensor, targets_x: Optional[torch.Tensor] ) -> torch.Tensor:
        """
        logits:  (N,) or (N,1) raw scores
        targets_y: (N,) or (N,1) with values in {0,1}
        """
        # Normalize inputs and devices
        z_list = self._ensure_container(logits)
        z0 = z_list[0]
        y=targets_y
        x= targets_x

        # Base per-sample loss and probability-like output
        base_loss, p = self.base_loss_fn(z_list, y, x=x)
        # base_loss: (N,), p: (N,) probabilities
        base_loss = base_loss.reshape(-1)
        p = p.reshape(-1)
        self.p = p  # expose for metrics when needed
        
        # Masks (>= tau_tp is positive)
        pos_mask = (y >= self.tau_tp)
        neg_mask = ~pos_mask

        # Asymmetric focal weights
        one_minus_p = 1.0 - self.p
        w_pos = self.alpha_pos * torch.pow(one_minus_p, self.gamma_pos)
        w_neg = self.alpha_neg * torch.pow(self.p,           self.gamma_neg)
        weight = torch.where(pos_mask, w_pos, w_neg)

        # Focal term
        loss = weight * base_loss  # (N,)

        # False-positive penalty on negatives
        if self.lambda_fp > 0.0:
            #overshoot_fp = torch.relu(p[neg_mask] - self.tau_fp)
            #loss[neg_mask] = loss[neg_mask] + self.lambda_fp * (overshoot_fp ** 2)
            overshoot_fp = torch.relu(self.p - self.tau_fp)
            penalty_fp = overshoot_fp ** 2 * (1.0 - targets_y)
            loss = loss + self.lambda_fp * penalty_fp

        # True-positive reward on positives
        if self.lambda_tp > 0.0:
            #overshoot_tp = torch.relu(p[pos_mask] - self.tau_tp)
            #loss[pos_mask] = loss[pos_mask] - self.lambda_tp * (overshoot_tp ** 2)
            overshoot_tp = (p - self.tau_tp).relu()
            reward_tp = overshoot_tp.square() * y
            loss = loss - (self.lambda_tp * reward_tp)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
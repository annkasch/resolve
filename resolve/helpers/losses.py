from gettext import npgettext
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def logit_normal_bernoulli_nll(
    z: torch.Tensor,
    y: torch.Tensor,
    num_points: int = 5,
    eps: float = 1e-12,
    **kward
) -> torch.Tensor:
    """
    Logit-Normal Bernoulli negative log-likelihood.

    We assume a latent logit ℓ ~ N(mu, sigma^2), probability p = sigmoid(ℓ),
    and label y ~ Bernoulli(p). This function computes:

        NLL = -log ∫ Bernoulli(y | sigmoid(ℓ)) N(ℓ | mu, sigma^2) dℓ

    using Gauss–Hermite quadrature.

    Args
    ----
    mu      : (...,)   mean of the logit distribution
    sigma   : (...,)   std of the logit distribution (must be > 0)
    y       : (...,)   binary labels in {0,1}
    num_points : int   number of GH quadrature points (5 or 10 supported)
    reduction  : str   "mean", "sum", or "none"
    eps     : float    numerical stability epsilon

    Returns
    -------
    nll : scalar tensor if reduction != "none", else same shape as mu/y
    """
    if num_points not in _GH_TABLE:
        raise ValueError(f"num_points={num_points} not supported; use 5 or 10.")

    mu = z[0]
    sigma = z[1]
    # Ensure shapes are compatible
    if y.shape != mu.shape:
        y = y.expand_as(mu)

    # Clamp sigma to avoid degenerate cases
    sigma = sigma.clamp_min(1e-8)

    # Get GH nodes/weights on correct device/dtype
    gh = _GH_TABLE[num_points]
    x = gh["x"].to(mu.device, mu.dtype)   # (M,)
    w = gh["w"].to(mu.device, mu.dtype)   # (M,)

    # Expand mu, sigma, y with a quadrature dimension M at the end
    # mu, sigma, y: (...,) -> (..., 1)
    mu_e    = mu.unsqueeze(-1)
    sigma_e = sigma.unsqueeze(-1)
    y_e     = y.unsqueeze(-1)

    # Sample logits at GH nodes: ℓ_i = mu + sqrt(2)*sigma*x_i
    # Result shape: (..., M)
    L = mu_e + math.sqrt(2.0) * sigma_e * x

    # Bernoulli probability at each sample: p_i = sigmoid(ℓ_i)
    P = torch.sigmoid(L)  # (..., M)

    # Likelihood at each sample: p_i^y * (1-p_i)^(1-y)
    lik = P * y_e + (1.0 - P) * (1.0 - y_e)  # (..., M)

    # Integrate over GH weights: ∑_i w_i * lik_i
    integral = (lik * w).sum(dim=-1)  # (...,)

    # Negative log-likelihood
    nll = -torch.log(integral.clamp_min(eps))  # (...,)

    return nll, P

def bce_with_logits(z, y, **kward):
    # z can be list/tuple or tensor
    z0 = z[0] if isinstance(z, (list, tuple)) else z
    z0 = z0.reshape(-1)
    y  = y.reshape(-1)
    return F.binary_cross_entropy_with_logits(z0, y, reduction="none"), torch.sigmoid(z0)

def gaussian_nll(z, y, **kward):

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

    def forward(self, logits: torch.Tensor, targets_y: torch.Tensor, **kwarg) -> torch.Tensor:
        """
        logits:  (N,) or (N,1) raw scores
        targets_y: (N,) or (N,1) with values in {0,1}
        """
        # Normalize inputs and devices
        z_list = self._ensure_container(logits)

        y= targets_y

        # Base per-sample loss (N,)
        base_loss, self.p = self.base_loss_fn(z_list, targets_y, x=kwarg.get("targets_x",None))
        
        if self.base_loss_fn != logit_normal_bernoulli_nll:
            # Masks (allow slightly fuzzy labels; >=0.5 -> positive)
            y = targets_y.view(-1).float()
        
            # Masks (>= tau_tp is positive)
            pos_mask = (y >= self.tau_tp)
            neg_mask = ~pos_mask

            # Asymmetric focal weights
            one_minus_p = 1.0 - self.p
            w_pos = self.alpha_pos * torch.pow(one_minus_p, self.gamma_pos)
            w_neg = self.alpha_neg * torch.pow(self.p,self.gamma_neg)
            weight = torch.where(pos_mask, w_pos, w_neg)

            # Focal term
            loss = weight * base_loss  # (N,)
            
            # False-positive penalty on negatives
            if self.lambda_fp > 0.0:
                overshoot_fp = torch.relu(self.p[neg_mask] - self.tau_fp)
                loss[neg_mask] = loss[neg_mask] + self.lambda_fp * (overshoot_fp ** 2)

            # True-positive reward on positives
            if self.lambda_tp > 0.0:
                overshoot_tp = torch.relu(self.p[pos_mask]+self.tau_tp)
                loss[pos_mask] = loss[pos_mask] - self.lambda_tp * (overshoot_tp ** 2)
        else: loss = base_loss

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

# Precomputed Gauss–Hermite nodes and weights for 5 and 10 points
# These are standard values; you can extend if you want more accuracy.
_GH_TABLE = {
    5: {
        "x": torch.tensor([
            -2.0201828704560856,
            -0.9585724646138185,
             0.0000000000000000,
             0.9585724646138185,
             2.0201828704560856
        ]),
        "w": torch.tensor([
            0.0199532420590459,
            0.3936193231522412,
            0.9453087204829419,
            0.3936193231522412,
            0.0199532420590459
        ]),
    },
    10: {
        "x": torch.tensor([
            -3.4361591188377376,
            -2.5327316742327897,
            -1.7566836492998819,
            -1.0366108297895137,
            -0.3429013272237046,
             0.3429013272237046,
             1.0366108297895137,
             1.7566836492998819,
             2.5327316742327897,
             3.4361591188377376
        ]),
        "w": torch.tensor([
            0.0004825731850073,
            0.0128803115355099,
            0.0931265981708253,
            0.3368363231280011,
            0.7246295952243925,
            0.7246295952243925,
            0.3368363231280011,
            0.0931265981708253,
            0.0128803115355099,
            0.0004825731850073
        ]),
    },
    "20": {
        "x": torch.tensor([
            -5.38748089001123286,
            -4.60368244955074427,
            -3.94476404011562521,
            -3.34785456738321613,
            -2.78880605842813072,
            -2.25497400208927588,
            -1.73853771211658621,
            -1.23407621539532301,
            -0.73747372854539443,
            -0.24534070830090125,
            0.24534070830090125,
            0.73747372854539443,
            1.23407621539532301,
            1.73853771211658621,
            2.25497400208927588,
            2.78880605842813072,
            3.34785456738321613,
            3.94476404011562521,
            4.60368244955074427,
            5.38748089001123286
        ]),
        "w": torch.tensor([
            2.22939364553415215e-13,
            4.39934099227318055e-10,
            1.08606937076928170e-07,
            7.80255647853206469e-06,
            0.00022833863601635396,
            0.00324377334223786183,
            0.024810520887463670,
            0.10901720602002332,
            0.28667550536283413,
            0.46224366960061009,
            0.46224366960061009,
            0.28667550536283413,
            0.10901720602002332,
            0.024810520887463670,
            0.00324377334223786183,
            0.00022833863601635396,
            7.80255647853206469e-06,
            1.08606937076928170e-07,
            4.39934099227318055e-10,
            2.22939364553415215e-13
        ])
    }
}
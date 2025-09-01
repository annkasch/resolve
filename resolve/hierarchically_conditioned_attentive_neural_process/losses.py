import math
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class AsymmetricFocalWithFPPenalty(nn.Module):
    """
    Binary classification loss:
      L = AFL(logits, targets)  +  lambda_fp * E_{y=0}[ ReLU(sigmoid(logits)-tau_fp)^2 ]
    where AFL is an Asymmetric Focal Loss (separate alpha/gamma for pos/neg).

    Args:
        alpha_pos: weight for positive class (typical: 0.1–0.5). alpha_neg = 1 - alpha_pos if not given.
        alpha_neg: optional explicit negative weight. If None, uses (1 - alpha_pos).
        gamma_pos: focusing parameter for positives (typical: 2).
        gamma_neg: focusing parameter for negatives; set higher to punish hard negatives (typical: 3–5).
        lambda_fp: strength of FP penalty (typical: 0.1–1.0).
        tau_fp: probability above which a background prediction is penalized (typical: 0.9–0.97).
        reduction: 'mean' | 'sum' | 'none'
    """
    def __init__(
        self,
        prior_pos: float = 0.,
        alpha_pos: float = 0.2,
        alpha_neg: float | None = None,
        gamma_pos: float = 2.0,
        gamma_neg: float = 4.0,
        lambda_fp: float = 0.2,
        tau_fp: float = 0.9,
        reduction: str = "mean",
    ):
        super().__init__()
        assert 0.0 <= alpha_pos <= 1.0
        if alpha_neg is None:
            alpha_neg = 1.0 - alpha_pos
        assert 0.0 <= alpha_neg <= 1.0
        assert reduction in ("mean", "sum", "none")
        assert 0.0 <= tau_fp <= 1.0
        assert lambda_fp >= 0.0

        self.prior_pos = prior_pos  # for logit adjustment
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.lambda_fp = lambda_fp
        self.tau_fp = tau_fp
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (N,) or (N,1) raw scores
        targets: (N,) or (N,1) with values in {0,1}
        """
        # Flatten to (N,)
        logits = logits.view(-1)
        targets = targets.view(-1).float()

        

        # Bias 
        bias = math.log(self.prior_pos / (1 - self.prior_pos))
        z = logits + bias
        # Probabilities
        p = torch.sigmoid(z)
        # Base BCE with logits (per-sample)
        bce = F.binary_cross_entropy_with_logits(z, targets, reduction="none")
        pt = targets * p + (1 - targets) * (1 - p)                      # prob of the true class

        # Focal weights (asymmetric)
        # For positives: pt = p, weight = alpha_pos * (1 - p)^gamma_pos
        # For negatives: pt = 1-p, weight = alpha_neg * (p)^gamma_neg
        pos_mask = (targets == 1.0)
        neg_mask = ~pos_mask

        weight = torch.zeros_like(bce)
        if pos_mask.any():
            w_pos = self.alpha_pos * torch.pow((1.0 - p[pos_mask]).clamp(min=1e-6), self.gamma_pos)
            weight[pos_mask] = w_pos
        if neg_mask.any():
            w_neg = self.alpha_neg * torch.pow(p[neg_mask].clamp(min=1e-6), self.gamma_neg)
            weight[neg_mask] = w_neg

        focal_term = weight * bce  # per-sample

        # FP penalty: only for negatives, penalize probs above tau_fp
        if self.lambda_fp > 0 and neg_mask.any():
            p_neg = p[neg_mask]
            fp_overshoot = torch.relu(p_neg - self.tau_fp)  # (p - tau)+
            fp_penalty = (fp_overshoot ** 2)  # smooth quadratic
            # Build a per-sample vector aligned with focal_term
            fp_vec = torch.zeros_like(focal_term)
            fp_vec[neg_mask] = fp_penalty
            loss = focal_term + self.lambda_fp * fp_vec
        else:
            loss = focal_term
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss
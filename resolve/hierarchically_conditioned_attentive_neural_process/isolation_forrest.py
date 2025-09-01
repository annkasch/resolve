# pip install scikit-learn
import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

def _masked_mean(x, mask):
    # x: (B, N, D), mask: (B, N) True=valid
    if mask is None:
        return x.mean(dim=1, keepdim=True)
    w = mask.float().unsqueeze(-1)                 # (B,N,1)
    s = (x * w).sum(dim=1, keepdim=True)          # (B,1,D)
    c = w.sum(dim=1, keepdim=True).clamp_min(1.0) # (B,1,1)
    return s / c

def _masked_weighted_mean(x, weights, mask):
    # x: (B,N,D), weights: (B,N) in [0,1], mask: (B,N)
    if mask is not None:
        weights = weights * mask.float()
    w = weights.unsqueeze(-1)                      # (B,N,1)
    s = (x * w).sum(dim=1, keepdim=True)          # (B,1,D)
    c = w.sum(dim=1, keepdim=True).clamp_min(1e-6)
    return s / c



class IsolationForestWrapper:
    """
    Unsupervised anomaly scorer for targets given context.
    Optionally calibrate to probabilities if you have labeled
    target outcomes (0/1) on a validation set.
    """
    def __init__(self, n_estimators=200, max_samples="auto",
                 contamination="auto", random_state=0):
        self.iforest = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
            verbose=0
        )
        self.calibrator = None  # Platt calibration via logistic regression

    def fit(self, X, y_target_labels=None):
        self.iforest.fit(X)

        # Optional probability calibration if labels are available
        if y_target_labels is not None:
            y = np.asarray(y_target_labels).reshape(-1)
            # decision_function: positive=inlier, negative=outlier
            d = self.iforest.decision_function(X).reshape(-1, 1)
            self.calibrator = LogisticRegression(max_iter=1000)
            self.calibrator.fit(d, y)

    def build_features_raw(self,theta, phi):
        """[θ | φ] -> (B*N, dθ+dφ) numpy"""
        feats = torch.cat([theta, phi], dim=-1)                   # (B,N,dθ+dφ)
        return feats.reshape(-1, feats.shape[-1]).cpu().numpy()

    def build_bkg_only_samples(self, theta, phi, y, mask_c=None, bkg_thresh=0.5):
        """
        Extract background-only rows from the context.
        theta_c:(B,Nc,dθ) phi_c:(B,Nc,dφ) y_c:(B,Nc,1) in [0,1] (0=bkg,1=sig)
        mask_c:(B,Nc) True=valid (optional)
        Returns X_bkg: (Nbkg, dθ+dφ) numpy
        """
        B, Nc, _ = theta.shape
        if mask_c is None:
            mask_c = torch.ones(B, Nc, dtype=torch.bool, device=theta.device)

        y_flat = y.squeeze(-1)                                  # (B,Nc)
        is_bkg = (y_flat <= bkg_thresh) & mask_c                  # boolean (B,Nc)

        # Gather only background rows
        theta_b = theta[is_bkg]                                 # (Nbkg,dθ)
        phi_b   = phi[is_bkg]                                   # (Nbkg,dφ)
        X_bkg = torch.cat([theta_b, phi_b], dim=-1).cpu().numpy() # (Nbkg,dθ+dφ)
        return X_bkg

    def build_signal_only_samples(self, theta, phi, y, mask_c=None, bkg_thresh=0.5):
        """
        Extract signal-only rows from the context.
        theta_c:(B,Nc,dθ) phi_c:(B,Nc,dφ) y_c:(B,Nc,1) in [0,1] (0=bkg,1=sig)
        mask_c:(B,Nc) True=valid (optional)
        Returns X_bkg: (Nbkg, dθ+dφ) numpy
        """
        B, Nc, _ = theta.shape
        if mask_c is None:
            mask_c = torch.ones(B, Nc, dtype=torch.bool, device=theta.device)

        y_flat = y.squeeze(-1)                                  # (B,Nc)
        is_signal = (y_flat >= bkg_thresh) & mask_c                  # boolean (B,Nc)

        # Gather only background rows
        theta_s = theta[is_signal]                                 # (Nbkg,dθ)
        phi_s   = phi[is_signal]                                   # (Nbkg,dφ)
        X_sig = torch.cat([theta_s, phi_s], dim=-1).cpu().numpy() # (Nbkg,dθ+dφ)
        return X_sig

    def build_target_features(self,theta_c, phi_c, y_c, theta_t, phi_t, mask_c=None):
        """
        Inputs:
        theta_c: (B,Nc,d_theta)
        phi_c:   (B,Nc,d_phi)
        y_c:     (B,Nc,1)   soft label/weight in [0,1]
        theta_t: (B,Nt,d_theta)
        phi_t:   (B,Nt,d_phi)
        mask_c:  (B,Nc) boolean, True=valid
        Returns:
        X: (B*Nt, F) numpy array of tabular features for IsolationForest
        """
        B, Nc, d_theta = theta_c.shape
        _, _, d_phi = phi_c.shape
        _, Nt, _ = phi_t.shape
        device = theta_c.device

        if mask_c is None:
            mask_c = torch.ones(B, Nc, dtype=torch.bool, device=device)

        y_flat = y_c.squeeze(-1).clamp(0, 1)          # (B,Nc)

        # Context summaries (cheap, no pairwise Nt×Nc)
        theta_ctx_mean = _masked_mean(theta_c, mask_c)        # (B,1,d_theta)
        phi_ctx_mean   = _masked_mean(phi_c,   mask_c)        # (B,1,d_phi)
        phi_pos_mean   = _masked_weighted_mean(phi_c, y_flat, mask_c)     # (B,1,d_phi)
        phi_neg_mean   = _masked_weighted_mean(phi_c, 1.0 - y_flat, mask_c)

        y_mean = (_masked_weighted_mean(y_flat.unsqueeze(-1), torch.ones_like(y_flat), mask_c)
                ).squeeze(1)                                    # (B,1) -> (B,)

        # Broadcast summaries to targets
        theta_ctx_mean_t = theta_ctx_mean.expand(B, Nt, d_theta)
        phi_ctx_mean_t   = phi_ctx_mean.expand(B, Nt, d_phi)
        phi_pos_mean_t   = phi_pos_mean.expand(B, Nt, d_phi)
        phi_neg_mean_t   = phi_neg_mean.expand(B, Nt, d_phi)
        y_mean_t         = y_mean.view(B, 1).expand(B, Nt)       # (B,Nt)

        # Deltas to encode “how unusual is this target vs context”
        dtheta_t = theta_t - theta_ctx_mean_t                    # (B,Nt,d_theta)
        dphi_pos = phi_t - phi_pos_mean_t                        # (B,Nt,d_phi)
        dphi_neg = phi_t - phi_neg_mean_t                        # (B,Nt,d_phi)
        dphi_ctx = phi_t - phi_ctx_mean_t                        # (B,Nt,d_phi)

        # Concatenate features per target
        feats = torch.cat([
            theta_t,                 # (B,Nt,d_theta)
            phi_t,                   # (B,Nt,d_phi)
            dtheta_t,                # (B,Nt,d_theta)
            dphi_ctx, dphi_pos, dphi_neg,  # 3 * d_phi
            y_mean_t.unsqueeze(-1),  # (B,Nt,1)
        ], dim=-1)                   # -> (B,Nt,F)

        X = feats.reshape(B*Nt, -1).detach().cpu().numpy()
        return X

    @torch.no_grad()
    def predict(self, X,
                return_prob=False):
        # Lower decision_function -> more anomalous
        d = self.iforest.decision_function(X)  # shape (B*Nt,)
        scores = -d  # higher = more anomalous (like a "risk" score)

        if not return_prob:
            return scores

        if self.calibrator is not None:
            p = self.calibrator.predict_proba(d.reshape(-1, 1))[:, 1]
        else:
            # Probability-ish fallback: squash to [0,1] via logistic on z-scored decision
            z = (d - d.mean()) / (d.std() + 1e-8)
            p = 1.0 / (1.0 + np.exp(z))  # anomalies -> higher prob
        return scores, p
import torch
import torch.nn as nn

# Needs scikit-learn
from sklearn.ensemble import IsolationForest
import numpy as np

class IsolationForestWrapper(nn.Module):
    def __init__(
        self,
        n_estimators=100,
        max_samples="auto",
        contamination="auto",
        max_features=1.0,
        bootstrap=False,
        n_jobs=None,
        random_state=None,
        warm_start=False,
        invert_scores=True,
    ):
        super().__init__()
        self.iforest = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            max_features=max_features,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=random_state,
            warm_start=warm_start,
        )
        # score_samples(): larger = more normal. If invert_scores=True we flip sign
        # so larger => more anomalous, which often matches "logit" intuition.
        self.invert_scores = invert_scores
        self._fitted = False

    @staticmethod
    def _concat_inputs(query_theta: torch.Tensor, query_phi: torch.Tensor) -> torch.Tensor:
        """
        Concatenates along last dim, like your AE:
          expected shapes:
            - (B, T, D_theta), (B, T, D_phi) -> (B, T, D_theta + D_phi)
            - or (N, D_theta), (N, D_phi) -> (N, D)
        """
        if query_theta is None and query_phi is None:
            raise ValueError("Provide at least one of query_theta or query_phi.")

        if query_theta is None:
            return query_phi
        if query_phi is None:
            return query_theta
        return torch.cat([query_theta, query_phi], dim=-1)

    @staticmethod
    def _to_2d_numpy(x: torch.Tensor) -> np.ndarray:
        """
        Flattens (B, T, D) -> (B*T, D) for sklearn, or leaves (N, D) as-is.
        Moves to CPU and detaches.
        """
        if x.dim() == 3:
            B, T, D = x.shape
            x2 = x.reshape(B*T, D)
            return x2.detach().cpu().numpy(), (B, T, D)
        elif x.dim() == 2:
            N, D = x.shape
            return x.detach().cpu().numpy(), (N, D)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(x.shape)}")

    @staticmethod
    def _from_2d_numpy(scores: np.ndarray, original_shape):
        """
        Restores scores to (B, T, 1) or (N, 1) to mirror batch/time layout.
        """
        if len(original_shape) == 3:  # (B, T, D)
            B, T, _ = original_shape
            return torch.from_numpy(scores.astype(np.float32)).reshape(B, T, 1)
        else:  # (N, D)
            N, _ = original_shape
            return torch.from_numpy(scores.astype(np.float32)).reshape(N, 1)

    def fit(self, X: torch.Tensor = None, query_theta: torch.Tensor = None, query_phi: torch.Tensor = None):
        """
        Fit the Isolation Forest.
          - Either pass a pre-concatenated X of shape (N, D) or (B, T, D),
            or pass (query_theta, query_phi) and it will concatenate along last dim.
        """
        if X is None:
            X = self._concat_inputs(query_theta, query_phi)

        X_np, _ = self._to_2d_numpy(X)
        self.iforest.fit(X_np)
        self._fitted = True
        return self

    def forward(self, query_theta: torch.Tensor, query_phi: torch.Tensor, **kwargs):
        """
        Returns {"logits": [scores]} with shape (B, T, 1) or (N, 1).
        By default, higher numbers => more anomalous (invert_scores=True).
        """
        if not self._fitted:
            raise RuntimeError("IsolationForestWrapper not fitted. Call .fit(...) before .forward(...).")

        X = self._concat_inputs(query_theta, query_phi)
        X_np, original_shape = self._to_2d_numpy(X)

        # score_samples: higher = more normal. We often want "anomaly intensity".
        scores = self.iforest.score_samples(X_np)  # shape (N_total,)
        if self.invert_scores:
            scores = -scores

        scores_t = self._from_2d_numpy(scores, original_shape)

        # Match your AE API: a dict with "logits" as a list
        return {"logits": [scores_t.to(X.device)]}

    @torch.no_grad()
    def encode(self, x):
        """
        API parity. For IF there's no latent code; we return the same anomaly scores.
        Accepts x as (N,D) or (B,T,D).
        """
        X_np, original_shape = self._to_2d_numpy(x)
        if not self._fitted:
            raise RuntimeError("IsolationForestWrapper not fitted. Call .fit(...) first.")
        scores = self.iforest.score_samples(X_np)
        if self.invert_scores:
            scores = -scores
        return self._from_2d_numpy(scores, original_shape)

    @torch.no_grad()
    def reconstruct(self, x):
        """
        No reconstruction in IF. For API compatibility, we return the same as encode().
        """
        return self.encode(x)
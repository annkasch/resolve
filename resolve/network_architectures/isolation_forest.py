import torch
import torch.nn as nn
import numpy as np

# scikit-learn import kept local-friendly
from sklearn.ensemble import IsolationForest


class IsolationForestWrapper(nn.Module):
    """
    PyTorch-friendly wrapper around sklearn IsolationForest with an AE-like API.

    - forward(query_theta, query_phi, **kwargs) -> {"logits": [scores]}
    - fit(...) can take:
        * loader=DataLoader/IterableDataset yielding (theta, phi) or {"query_theta":..., "query_phi":...}
        * query_theta/query_phi tensors
        * pre-concatenated X tensor
    - Device safe: .to()/cuda()/cpu()/half()/float() are no-ops for sklearn.
      Outputs are moved to the requested device.

    Output semantics:
      If invert_scores=True (default): larger score => more anomalous.
      If invert_scores=False: follows sklearn.score_samples (larger => more normal).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: str | int | float = 512,
        contamination: str | float = "auto",
        max_features: int | float = 1.0,
        bootstrap: bool = False,
        n_jobs: int | None = None,
        random_state: int | None = None,
        warm_start: bool = False,
        invert_scores: bool = True,
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
        self.invert_scores = invert_scores
        self._fitted = False

        # device handling: sklearn stays on CPU; we only control where OUTPUT tensors go
        self._out_device = torch.device("cpu")
        # hint for trainers to skip moving underlying model
        self.cpu_only = True

    # ---------------------- utility: input shape & conversion ----------------------

    @staticmethod
    def _concat_inputs(query_theta: torch.Tensor | None,
                       query_phi: torch.Tensor | None) -> torch.Tensor:
        """Concatenate along last dim; supports 2D (N,D) or 3D (B,T,D)."""
        if query_theta is None and query_phi is None:
            raise ValueError("Provide at least one of query_theta or query_phi.")

        if query_theta is None:
            return query_phi
        if query_phi is None:
            return query_theta
        return torch.cat([query_theta, query_phi], dim=-1)

    @staticmethod
    def _to_2d_numpy(x: torch.Tensor) -> tuple[np.ndarray, tuple]:
        """
        Flatten (B,T,D)->(B*T,D) or (N,D)->(N,D), detach to CPU numpy.
        Returns (np_array, original_shape_tuple)
        """
        if x.dim() == 3:
            B, T, D = x.shape
            x2 = x.reshape(B * T, D)
            return x2.detach().cpu().numpy(), (B, T, D)
        elif x.dim() == 2:
            N, D = x.shape
            return x.detach().cpu().numpy(), (N, D)
        else:
            raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(x.shape)}")

    @staticmethod
    def _from_2d_numpy(scores: np.ndarray, original_shape: tuple) -> torch.Tensor:
        """Restore scores to (B,T,1) or (N,1)."""
        if len(original_shape) == 3:  # (B,T,D)
            B, T, _ = original_shape
            return torch.from_numpy(scores.astype(np.float32)).reshape(B, T, 1)
        else:  # (N,D)
            N, _ = original_shape
            return torch.from_numpy(scores.astype(np.float32)).reshape(N, 1)

    def fit(self,
            X: torch.Tensor | None = None,
            query_theta: torch.Tensor | None = None,
            query_phi: torch.Tensor | None = None,
            loader=None):
        """
        Fit the Isolation Forest.

        Args:
            loader: DataLoader/IterableDataset yielding either
                    - dict with keys {"query_theta", "query_phi"}  OR
                    - tuple/list (theta, phi)
            X: pre-concatenated tensor of shape (N,D) or (B,T,D)
            query_theta/query_phi: tensors to be concatenated along last dim
        """

        # Case 1: build X from loader batches
        if loader is not None:
            parts = []
            for i, batch in enumerate(loader):
                _, query, _ = batch
                theta = query.theta
                phi = query.phi
                
                X_batch = self._concat_inputs(theta, phi)
                parts.append(X_batch.cpu())

            if len(parts) == 0:
                raise ValueError("Loader yielded no batches.")
            # concatenate along batch dimension (dim=0)

            parts_flat = [p.reshape(-1, p.size(-1)) for p in parts]  # [num_points, 14]
            X = torch.cat(parts_flat, dim=0)  # total length = 1000 + 808
            del parts, parts_flat  # free

        # Case 2: tensors provided directly
        elif X is None:
            X = self._concat_inputs(query_theta, query_phi)

        # Convert to numpy (2D) and fit sklearn model on CPU
        X_np, _ = self._to_2d_numpy(X)
        self.iforest.fit(X_np)
        self._fitted = True

        return self

    def forward(self, query_theta: torch.Tensor, query_phi: torch.Tensor, **kwargs):
        """
        Returns {"logits": [scores_tensor]} with shape (B,T,1) or (N,1).
        """
        if not self._fitted:
            raise RuntimeError("IsolationForestWrapper not fitted. Call .fit(...) first.")

        X = self._concat_inputs(query_theta, query_phi)
        X_np, original_shape = self._to_2d_numpy(X)

        scores = self.iforest.score_samples(X_np)  # higher => more normal (sklearn)
        if self.invert_scores:
            scores = -scores  # higher => more anomalous

        scores_t = self._from_2d_numpy(scores, original_shape).to(self._out_device)
        return {"logits": [scores_t]}

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        API parity. Returns anomaly scores with same shape as forward output (without dict).
        """
        if not self._fitted:
            raise RuntimeError("IsolationForestWrapper not fitted. Call .fit(...) first.")
        X_np, original_shape = self._to_2d_numpy(x)
        scores = self.iforest.score_samples(X_np)
        if self.invert_scores:
            scores = -scores
        return self._from_2d_numpy(scores, original_shape).to(self._out_device)

    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """No reconstruction exists; mirrors encode() for API compatibility."""
        return self.encode(x)

    # ---------------------- device / dtype hooks (no-ops for sklearn) ----------------------

    def to(self, device=None, *args, **kwargs):
        # Keep sklearn model on CPU; remember desired output device for tensors.
        if device is not None:
            self._out_device = torch.device(device)
        return self

    def cuda(self, *args, **kwargs):
        return self.to("cuda")

    def cpu(self):
        return self.to("cpu")

    def half(self):
        # No trainable tensors here; keep as no-op to be AMP-safe.
        return self

    def float(self):
        return self
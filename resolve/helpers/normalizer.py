from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import torch


class Normalizer:
    def __init__(self, method: str = None):
        self.method = method
        self.scalers = {}

    def _make_scaler(self):
        """Factory for the chosen normalization method."""
        if self.method == "zscore":
            return StandardScaler(copy=False, with_mean=True, with_std=True)
        elif self.method == "minmax":
            return MinMaxScaler()
        else:
            return StandardScaler(copy=False, with_mean=False, with_std=False)

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

    def _to_tensor(self, x, ref: torch.Tensor):
        return torch.from_numpy(x).to(ref.device, dtype=ref.dtype)

    def _get_scaler(self, feature_grp: str):
        """Retrieve or lazily create a scaler for the given group."""
        if feature_grp not in self.scalers:
            self.scalers[feature_grp] = self._make_scaler()
        return self.scalers[feature_grp]

    def validate_fitted(self, feature_groups=("theta", "phi")):
        """Raise a clear error unless every requested scaler has been fitted."""
        missing = []
        for feature_group in feature_groups:
            scaler = self.scalers.get(feature_group)
            if scaler is None:
                missing.append(feature_group)
                continue
            try:
                check_is_fitted(scaler)
            except NotFittedError:
                missing.append(feature_group)

        if missing:
            raise ValueError(
                "Normalizer requires fitted scaler state for feature groups: "
                f"{missing}."
            )

    def fit(self, x: torch.Tensor, feature_grp: str):
        self._get_scaler(feature_grp).fit(self._to_numpy(x))
    
    def fit_transform(self, x: torch.Tensor, feature_grp: str) -> torch.Tensor:
        transformed = self._get_scaler(feature_grp).fit_transform(
            self._to_numpy(x)
        )
        return self._to_tensor(transformed, x)

    def transform(self, x: torch.Tensor, feature_grp: str) -> torch.Tensor:
        transformed = self._get_scaler(feature_grp).transform(
            self._to_numpy(x)
        )
        return self._to_tensor(transformed, x)

    def inverse_transform(
        self,
        x: torch.Tensor,
        feature_grp: str,
    ) -> torch.Tensor:
        transformed = self._get_scaler(feature_grp).inverse_transform(
            self._to_numpy(x)
        )
        return self._to_tensor(transformed, x)

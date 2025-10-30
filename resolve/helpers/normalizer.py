from sklearn.preprocessing import StandardScaler, MinMaxScaler
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

    def fit(self, x: torch.Tensor, feature_grp: str):
        self._get_scaler(feature_grp).fit(x)
    
    def fit_transform(self, x: torch.Tensor, feature_grp: str) -> torch.Tensor:
        return torch.from_numpy(self._get_scaler(feature_grp).fit_transform(x))

    def transform(self, x: torch.Tensor, feature_grp: str) -> torch.Tensor:
        return torch.from_numpy(self._get_scaler(feature_grp).transform(x))

    def inverse_transform(self, x: torch.Tensor, feature_grp: str):
        return torch.from_numpy(self._get_scaler(feature_grp).inverse_transform(x))

    def fit_transform_as_f32(self, as_f32, **feature_groups):
        """
        Fit and transform multiple feature groups (e.g., theta, phi).
        
        Example:
            theta, phi = self.fit_transform(theta=theta, phi=phi)
        """

        transformed = {}
        for name, data in feature_groups.items():
            transformed[name] = self.fit_transform(data, name)
            if as_f32:
                transformed[name] = transformed[name].float();
                transformed[name].contiguous()
        return tuple(transformed.values())
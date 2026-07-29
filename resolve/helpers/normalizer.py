from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import numpy as np
import torch


class Normalizer:
    def __init__(self, method: str = None):
        self.method = method
        self.scalers = {}
        self.feature_labels = {}

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

    @staticmethod
    def _normalize_labels(feature_labels):
        if feature_labels is None:
            return None
        labels = tuple(feature_labels)
        if not labels or any(
            not isinstance(label, str) or not label for label in labels
        ):
            raise ValueError("feature_labels must contain nonempty strings.")
        if len(set(labels)) != len(labels):
            raise ValueError("feature_labels must be unique.")
        return labels

    def _bind_feature_schema(
        self,
        x: torch.Tensor,
        feature_grp: str,
        feature_labels=None,
    ):
        labels = self._normalize_labels(feature_labels)
        if labels is None:
            return
        feature_count = 1 if x.ndim == 1 else x.shape[-1]
        if len(labels) != feature_count:
            raise ValueError(
                f"Feature group {feature_grp!r} has {feature_count} columns "
                f"but {len(labels)} labels were supplied."
            )
        existing = self.feature_labels.get(feature_grp)
        if existing is not None and existing != labels:
            raise ValueError(
                f"Feature group {feature_grp!r} was already fitted for labels "
                f"{existing}, not {labels}."
            )
        self.feature_labels[feature_grp] = labels

    def validate_schema(self, feature_grp: str, feature_labels):
        labels = self._normalize_labels(feature_labels)
        fitted_labels = self.feature_labels.get(feature_grp)
        if fitted_labels is None:
            raise ValueError(
                f"Normalizer has no fitted feature-label schema for "
                f"{feature_grp!r}; refit it through DataLoaderManager."
            )
        if fitted_labels != labels:
            raise ValueError(
                f"Normalizer feature labels for {feature_grp!r} are "
                f"{fitted_labels}, but the configured labels are {labels}."
            )

        scaler = self.scalers.get(feature_grp)
        fitted_count = getattr(scaler, "n_features_in_", None)
        if fitted_count != len(labels):
            raise ValueError(
                f"Normalizer scaler for {feature_grp!r} was fitted with "
                f"{fitted_count} features, but {len(labels)} are configured."
            )

    def fit(
        self,
        x: torch.Tensor,
        feature_grp: str,
        feature_labels=None,
    ):
        self._bind_feature_schema(x, feature_grp, feature_labels)
        self._get_scaler(feature_grp).fit(self._to_numpy(x))

    def fit_chunks(
        self,
        chunks,
        feature_grp: str,
        feature_labels=None,
    ):
        """Incrementally fit one feature group from bounded-size chunks."""
        scaler = self._get_scaler(feature_grp)
        count = 0
        mean = None
        squared_deviation = None
        data_min = None
        data_max = None
        for chunk in chunks:
            if chunk.shape[0] == 0:
                continue
            self._bind_feature_schema(
                chunk,
                feature_grp,
                feature_labels if count == 0 else None,
            )
            values = np.asarray(self._to_numpy(chunk), dtype=np.float64)
            if values.ndim == 1:
                values = values.reshape(-1, 1)
            if self.method == "zscore":
                chunk_count = values.shape[0]
                chunk_mean = values.mean(axis=0)
                centered = values - chunk_mean
                chunk_squared_deviation = np.square(centered).sum(axis=0)
                if count == 0:
                    mean = chunk_mean
                    squared_deviation = chunk_squared_deviation
                    count = chunk_count
                else:
                    delta = chunk_mean - mean
                    combined_count = count + chunk_count
                    mean = mean + delta * chunk_count / combined_count
                    squared_deviation = (
                        squared_deviation
                        + chunk_squared_deviation
                        + np.square(delta)
                        * count
                        * chunk_count
                        / combined_count
                    )
                    count = combined_count
            elif self.method == "minmax":
                chunk_min = values.min(axis=0)
                chunk_max = values.max(axis=0)
                data_min = (
                    chunk_min
                    if data_min is None
                    else np.minimum(data_min, chunk_min)
                )
                data_max = (
                    chunk_max
                    if data_max is None
                    else np.maximum(data_max, chunk_max)
                )
                count += values.shape[0]
            else:
                scaler.partial_fit(values)
                count += values.shape[0]
        if count == 0:
            raise ValueError(
                f"Cannot fit feature group {feature_grp!r} from no samples."
            )
        if self.method == "zscore":
            scaler.mean_ = mean
            scaler.var_ = squared_deviation / count
            scaler.scale_ = np.sqrt(scaler.var_)
            scaler.scale_[scaler.scale_ == 0.0] = 1.0
            scaler.n_samples_seen_ = count
            scaler.n_features_in_ = mean.shape[0]
        elif self.method == "minmax":
            scaler.data_min_ = data_min
            scaler.data_max_ = data_max
            scaler.data_range_ = data_max - data_min
            denominator = scaler.data_range_.copy()
            denominator[denominator == 0.0] = 1.0
            feature_min, feature_max = scaler.feature_range
            scaler.scale_ = (feature_max - feature_min) / denominator
            scaler.min_ = feature_min - data_min * scaler.scale_
            scaler.n_samples_seen_ = count
            scaler.n_features_in_ = data_min.shape[0]
    
    def fit_transform(
        self,
        x: torch.Tensor,
        feature_grp: str,
        feature_labels=None,
    ) -> torch.Tensor:
        self._bind_feature_schema(x, feature_grp, feature_labels)
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

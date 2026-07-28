import os
import math
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset, get_worker_info
from typing import List, Optional, Sequence, Tuple, Dict, Union
import collections
from resolve.helpers.normalizer import Normalizer
from resolve.helpers.sampler import Sampler
from resolve.helpers.splitter import Splitter

ContextSet = collections.namedtuple("ContextSet", ("theta", "phi", "y", "idx", "file_indices"))
QuerySet   = collections.namedtuple("QuerySet",   ("theta", "phi", "idx", "file_indices"))

BatchCollection = collections.namedtuple(
    "BatchCollection",
    ("context", "query", "target_y")
)

class InMemoryIterableData(IterableDataset):
    def __init__(self, files: Sequence[str], batch_size: int = 1000,
                 parameter_config: Dict = None, dataset_config: Dict = None, positive_condition: Optional[List]=None,
                 normalizer: Optional[Normalizer] = None, mode: Optional[str] = "train") -> None:
        super().__init__()
        
        self.files, self.shuffle, self.seed = list(files), dataset_config["shuffle_dataset"], dataset_config["seed"]
        self.parameter_config, self.dataset_config = (parameter_config or {}), dataset_config
        self.batch_size = batch_size
        
        self.mode = mode
        self._normalizer = self._prepare_normalizer(normalizer)
        self.sampler = Sampler(positive_condition, shuffle=self.shuffle, seed=self.seed)
        self.nepochs = 1
        self._base_indices = {}
        self._built_epochs = {}

        # load all data into memory
        theta, phi, y, fidx = self._load_data_to_mem(self.files, self.parameter_config)
        
        self.theta_to_id = self.sampler.get_unique_ids(theta)

        self.data = self._set_data(theta, phi, y, fidx)
        self.build_batches(0)

    @staticmethod
    def _canonical_normalization_method(method):
        return None if method in (None, "none") else method

    def _prepare_normalizer(
        self,
        normalizer: Optional[Normalizer],
    ) -> Optional[Normalizer]:
        configured_method = self.dataset_config.get(
            "use_feature_normalization",
            None,
        )
        canonical_method = self._canonical_normalization_method(
            configured_method
        )
        if canonical_method not in (None, "zscore", "minmax"):
            raise ValueError(
                "Unsupported feature normalization method "
                f"{configured_method!r}; expected 'none', 'zscore', or "
                "'minmax'."
            )

        if self.mode == "train":
            if normalizer is not None:
                raise ValueError(
                    "Training datasets create and fit their own normalizer; "
                    "do not inject one."
                )
            return None

        if normalizer is None:
            if canonical_method is not None:
                raise ValueError(
                    f"{self.mode.capitalize()} data using "
                    f"{configured_method!r} normalization requires a fitted "
                    "training normalizer."
                )
            return Normalizer(configured_method)

        if not isinstance(normalizer, Normalizer):
            raise TypeError("normalizer must be a Normalizer instance.")

        normalizer_method = self._canonical_normalization_method(
            normalizer.method
        )
        if normalizer_method != canonical_method:
            raise ValueError(
                f"Injected normalizer method {normalizer.method!r} does not "
                f"match configured method {configured_method!r}."
            )
        if canonical_method is not None:
            normalizer.validate_fitted()
        return normalizer
        
    def make_empty_like(self,*tensors):
                return [torch.empty_like(t) for t in tensors]
    
    def _set_data(self, theta: torch.Tensor, phi: torch.Tensor, y: torch.Tensor, fidx: torch.Tensor):
        self.context_ratio = float(
            self.dataset_config.get("context_ratio", 1.0 / 3.0)
        )
        if not 0.0 <= self.context_ratio < 1.0:
            raise ValueError("context_ratio must be in [0, 1).")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        self.context_is_subset = bool(
            self.dataset_config.get("context_is_subset", True)
        )
        self.batch_size_ctx = math.floor(
            self.batch_size * self.context_ratio
        )
        if self.context_ratio > 0.0 and self.batch_size_ctx == 0:
            raise ValueError(
                "A positive context_ratio requires a batch_size large enough "
                "for at least one context and one query sample."
            )
        self.batch_size_tgt = (
            self.batch_size
            if self.context_is_subset
            else self.batch_size - self.batch_size_ctx
        )

        pos_mask = self.sampler.get_positive_indices(y)
        positive_ratio_data = pos_mask.sum(dim=0)/y.shape[0]
        splitter = Splitter(self.shuffle, seed=self.seed)
        idx = torch.arange(phi.shape[0])
        data = {}
        
        if self.mode == "train":
            split_indices = {}
            val_size = self.dataset_config.get("val_ratio", 0.2)
            if val_size > 0.0:
                idx, split_indices["validate"] = splitter.train_test_split(
                    idx,
                    groups=theta[idx],
                    test_size=val_size,
                )

            configured_test_size = self.dataset_config.get("test_ratio", 0.2)
            if configured_test_size > 0.0:
                remaining_fraction = 1.0 - val_size
                if remaining_fraction <= 0.0:
                    raise ValueError(
                        "val_ratio must leave data available for training "
                        "and testing."
                    )
                idx, split_indices["test"] = splitter.train_test_split(
                    idx,
                    groups=theta[idx],
                    test_size=configured_test_size / remaining_fraction,
                )
            split_indices["train"] = idx

            # Fit feature transforms on training rows only, then apply them
            # consistently to all partitions.
            self._normalizer = Normalizer(
                self.dataset_config.get("use_feature_normalization", None)
            )
            self._normalizer.fit(theta.index_select(0, idx), "theta")
            self._normalizer.fit(phi.index_select(0, idx), "phi")
            theta = self._normalizer.transform(theta, "theta").float().contiguous()
            phi = self._normalizer.transform(phi, "phi").float().contiguous()

            data.update(
                {
                    "data": {
                        "theta": theta,
                        "phi": phi,
                        "y": y,
                        "file_indices": fidx,
                    }
                }
            )

            # Apply mixup to training data only.
            if self.dataset_config and self.dataset_config.get('mixup_ratio', 0.) > 0.0:
                theta[idx], phi[idx], y[idx], fidx[idx] = self.sampler.mix_by_file_chunks(
                            theta[idx], phi[idx], y[idx], fidx[idx],self.dataset_config.get('mixup_ratio'),
                            use_beta=self.dataset_config.get('use_beta', None),
                            margin=float(self.dataset_config.get('mixup_margin', 0.0)),
                            seed=self.seed,
                )

            data = {"data": data["data"]}
            for split_mode, split_idx in split_indices.items():
                self._base_indices[split_mode] = split_idx.clone()
                data[split_mode] = self._empty_mode_plan(
                    positive_ratio_data
                )
        else:
            if self._canonical_normalization_method(
                self.dataset_config.get("use_feature_normalization", None)
            ) is None:
                self._normalizer.fit(theta, "theta")
                self._normalizer.fit(phi, "phi")
            theta = self._normalizer.transform(x=theta, feature_grp="theta")
            phi = self._normalizer.transform(x=phi, feature_grp="phi")
            theta = theta.float().contiguous(); phi = phi.float().contiguous()
            self._base_indices[self.mode] = idx.clone()
            data = {
                "data": {
                    "theta": theta,
                    "phi": phi,
                    "y": y,
                    "file_indices": fidx,
                },
                self.mode: self._empty_mode_plan(positive_ratio_data),
            }
            
        return data

    def _empty_mode_plan(self, positive_ratio):
        empty = torch.empty(0, dtype=torch.long)
        return {
            "context": {
                "indices": empty,
                "batches": (),
                "batch_size": self.batch_size_ctx,
                "ratio": self.context_ratio,
            },
            "target": {
                "indices": empty,
                "batches": (),
                "batch_size": self.batch_size_tgt,
                "ratio": 1.0 - self.context_ratio,
            },
            "meta": {
                "num_epochs": 1,
                "pos_frac": positive_ratio,
                "num_batches": 0,
            },
        }

    def _indices_for_epoch(self, mode, epoch):
        indices = self._base_indices[mode]
        positive_ratio = self.dataset_config.get(
            "positive_ratio_train",
            None,
        )
        if (
            mode != "train"
            or isinstance(positive_ratio, list)
            or positive_ratio is None
        ):
            return indices, None

        selected, _, unused, meta = self.sampler.groupaware_pos_sampling(
            self.data["data"]["file_indices"].index_select(0, indices),
            self.data["data"]["y"].index_select(0, indices),
            indices,
            target_pos_frac=positive_ratio,
            max_pos_reuse_per_epoch=self.dataset_config.get(
                "max_positive_reuse",
                0,
            ),
            sticky_frac=0.25,
            seed=self.seed,
            epoch=epoch,
        )
        meta["unused"] = unused
        return selected, meta

    def _rebalance_singleton_batch(self, batches):
        batches = list(batches)
        if self.context_ratio == 0.0 or len(batches) < 2:
            return batches

        singleton_index = next(
            (
                index
                for index, batch in enumerate(batches)
                if batch.numel() == 1
            ),
            None,
        )
        if singleton_index is None:
            return batches

        donor_index = next(
            (
                index
                for index, batch in enumerate(batches)
                if batch.numel() > 2
            ),
            None,
        )
        if donor_index is None:
            raise ValueError(
                "A positive context_ratio requires at least two samples in "
                "every paired batch."
            )

        donor = batches[donor_index]
        batches[donor_index] = donor[:-1]
        batches[singleton_index] = torch.cat(
            (donor[-1:], batches[singleton_index])
        )
        return batches

    def build_batches(self, epoch, mode=None):
        mode = self.mode if mode is None else mode
        if mode not in self._base_indices:
            raise ValueError(
                f"Mode {mode!r} has no data. Available modes: "
                f"{sorted(self._base_indices)}."
            )

        plan_epoch = int(epoch) if mode == "train" else 0
        if self._built_epochs.get(mode) == plan_epoch:
            return

        indices, sampling_meta = self._indices_for_epoch(mode, plan_epoch)
        combined_batches, _, _ = self.sampler.build_batches(
            indices,
            batch_size=self.batch_size,
            epoch=plan_epoch,
        )
        combined_batches = self._rebalance_singleton_batch(combined_batches)

        context_batches = []
        target_batches = []
        for combined_batch in combined_batches:
            if self.context_ratio > 0.0:
                if combined_batch.numel() < 2:
                    raise ValueError(
                        "A positive context_ratio requires at least one "
                        "context and one query sample per batch."
                    )
                context_size = min(
                    combined_batch.numel() - 1,
                    max(
                        1,
                        math.floor(
                            combined_batch.numel() * self.context_ratio
                        ),
                    ),
                )
                context_batch = combined_batch[:context_size]
            else:
                context_batch = combined_batch.new_empty((0,))

            if self.context_is_subset:
                target_batch = combined_batch
            else:
                target_batch = combined_batch[context_batch.numel():]

            context_batches.append(context_batch)
            target_batches.append(target_batch)

        mode_data = self.data[mode]
        mode_data["context"]["batches"] = tuple(context_batches)
        mode_data["target"]["batches"] = tuple(target_batches)
        mode_data["context"]["indices"] = (
            torch.cat(context_batches)
            if context_batches
            else torch.empty(0, dtype=torch.long)
        )
        mode_data["target"]["indices"] = (
            torch.cat(target_batches)
            if target_batches
            else torch.empty(0, dtype=torch.long)
        )
        mode_data["meta"]["num_batches"] = len(target_batches)
        if sampling_meta is not None:
            mode_data["meta"].update(sampling_meta)
            mode_data["meta"]["num_batches"] = len(target_batches)
        self._built_epochs[mode] = plan_epoch

    @staticmethod
    def _decode_hdf5_labels(dataset, file_path: str, dataset_key: str) -> List[str]:
        if dataset.ndim not in (1, 2):
            raise ValueError(
                f"HDF5 dataset {dataset_key!r} in {file_path!r} must be 1D or "
                f"2D, got shape {dataset.shape}."
            )
        if "labels" not in dataset.attrs:
            raise ValueError(
                f"HDF5 dataset {dataset_key!r} in {file_path!r} has no "
                "'labels' attribute."
            )

        raw_labels = np.atleast_1d(dataset.attrs["labels"]).tolist()
        labels = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in raw_labels
        ]
        expected_labels = 1 if dataset.ndim == 1 else dataset.shape[1]
        if len(labels) != expected_labels:
            raise ValueError(
                f"HDF5 dataset {dataset_key!r} in {file_path!r} has "
                f"{expected_labels} columns but {len(labels)} labels."
            )

        duplicate_labels = sorted(
            {label for label in labels if labels.count(label) > 1}
        )
        if duplicate_labels:
            raise ValueError(
                f"HDF5 dataset {dataset_key!r} in {file_path!r} has duplicate "
                f"labels: {duplicate_labels}."
            )
        return labels

    @classmethod
    def _read_hdf5_columns(
        cls,
        hdf,
        file_path: str,
        parameter_spec: Dict,
    ) -> np.ndarray:
        dataset_key = parameter_spec["key"]
        if dataset_key not in hdf:
            raise ValueError(
                f"HDF5 file {file_path!r} is missing dataset {dataset_key!r}."
            )

        requested_labels = list(parameter_spec["selected_labels"])
        duplicate_requests = sorted(
            {
                label
                for label in requested_labels
                if requested_labels.count(label) > 1
            }
        )
        if duplicate_requests:
            raise ValueError(
                f"Configured labels for HDF5 dataset {dataset_key!r} contain "
                f"duplicates: {duplicate_requests}."
            )
        if not requested_labels:
            raise ValueError(
                f"No labels configured for HDF5 dataset {dataset_key!r}."
            )

        dataset = hdf[dataset_key]
        labels = cls._decode_hdf5_labels(dataset, file_path, dataset_key)
        missing_labels = [
            label for label in requested_labels if label not in labels
        ]
        if missing_labels:
            raise ValueError(
                f"HDF5 dataset {dataset_key!r} in {file_path!r} is missing "
                f"requested labels: {missing_labels}."
            )

        if dataset.ndim == 1:
            if len(requested_labels) != 1:
                raise ValueError(
                    f"HDF5 dataset {dataset_key!r} in {file_path!r} is 1D "
                    f"but {len(requested_labels)} labels were requested."
                )
            return dataset[:].reshape(-1, 1)

        physical_indices = [labels.index(label) for label in requested_labels]
        sorted_pairs = sorted(
            enumerate(physical_indices),
            key=lambda pair: pair[1],
        )
        sorted_indices = [physical_index for _, physical_index in sorted_pairs]
        values = dataset[:, sorted_indices]

        configured_order = [0] * len(sorted_pairs)
        for loaded_position, (configured_position, _) in enumerate(sorted_pairs):
            configured_order[configured_position] = loaded_position
        return values[:, configured_order]

    @staticmethod
    def _validate_row_counts(
        file_path: str,
        theta,
        phi,
        y,
    ) -> None:
        row_counts = {
            "theta": theta.shape[0],
            "phi": phi.shape[0],
            "target": y.shape[0],
        }
        if len(set(row_counts.values())) != 1:
            raise ValueError(
                f"Inconsistent row counts in {file_path!r}: {row_counts}."
            )

    @classmethod
    def _read_in_from_file(
        cls,
        file_path: str,
        parameter_config: Dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if file_path.endswith(('.h5', '.hdf5')):

            with h5py.File(file_path, 'r') as hdf:
                phi = cls._read_hdf5_columns(
                    hdf,
                    file_path,
                    parameter_config["phi"],
                )
                theta = cls._read_hdf5_columns(
                    hdf,
                    file_path,
                    parameter_config["theta"],
                )
                y = cls._read_hdf5_columns(
                    hdf,
                    file_path,
                    parameter_config["target"],
                )

            phi = torch.from_numpy(phi)
            theta = torch.from_numpy(theta)
            y = torch.from_numpy(y)

        elif file_path.endswith('.csv'):
            # --- CSV reading using column names (selected_labels) ---
            df = pd.read_csv(file_path)

            def select_labels(df: pd.DataFrame, labels: Union[str, List[str]]) -> pd.DataFrame:
                """Select one or multiple columns by name."""
                if isinstance(labels, str):
                    return df[[labels]]
                elif isinstance(labels, list):
                    return df[labels]
                else:
                    raise ValueError(f"Invalid label type: {type(labels)}")

            # Extract φ, θ, and y by column labels
            phi = select_labels(df, parameter_config['phi']['selected_labels'])
            theta = select_labels(df, parameter_config['theta']['selected_labels'])
            y = select_labels(df, parameter_config['target']['selected_labels'])

            # Convert to torch tensors
            phi = torch.tensor(phi.values, dtype=torch.float32)
            theta = torch.tensor(theta.values, dtype=torch.float32)
            y = torch.tensor(y.values, dtype=torch.float32)

            # Ensure y has shape (N, 1)
            if y.ndim == 1:
                y = y.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        cls._validate_row_counts(file_path, theta, phi, y)

        # ensure float32 on CPU
        phi = phi.contiguous().to(torch.float32)
        theta = theta.contiguous().to(torch.float32)
        y = y.contiguous().to(torch.float32)
        return theta, phi, y

    def _load_data_to_mem(self, files: Sequence[str], cfg: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Thetas, Phis, ys, file_inds = [], [], [], []
        for i, fp in enumerate(files):
            if not os.path.exists(fp): raise FileNotFoundError(fp)
            Thetai, Phii, yi = self._read_in_from_file(fp, cfg)
            Thetas.append(Thetai); Phis.append(Phii); ys.append(yi)
            file_inds.append(torch.full((Phii.size(0),), i, dtype=torch.long))
        Theta, Phi, y, fidx = torch.cat(Thetas, 0).contiguous(), torch.cat(Phis, 0).contiguous(), torch.cat(ys, 0).contiguous(),torch.cat(file_inds, 0).contiguous()
        return Theta, Phi, y, fidx

    def set_mode(self, mode):
        self.mode = mode
    
    def set_normalizer(self, method_or_obj):
        if isinstance(method_or_obj, Normalizer):
            self._normalizer = method_or_obj
        else:
            self._normalizer = Normalizer(method_or_obj)

    def _compute_worker_slice(self, n: int) -> Tuple[int, int]:
        info = get_worker_info()
        if info is None: return 0, n
        per = int(math.ceil(n / info.num_workers)); s = info.id * per; e = min(s + per, n); return s, e

    def __iter__(self):
        """Iterator for train/validate/test. Uses precomputed batch-index plans if present."""

        batches_tgt = self.data[self.mode]["target"].get("batches", None)
        total_batches = len(batches_tgt)
        b_start, b_end = self._compute_worker_slice(total_batches)  # reuse same helper; it just slices a range
        if b_start >= b_end:
            return iter(())
        theta = self.data["data"]["theta"]
        phi   = self.data["data"]["phi"]
        y     = self.data["data"]["y"]
        file_indices = self.data["data"]["file_indices"]

        for b in range(b_start, b_end):
            idx_tgt = self.data[self.mode]["target"]["batches"][b]
            b_phi_tgt = phi.index_select(0, idx_tgt).unsqueeze(0)
            b_theta_tgt = theta.index_select(0, idx_tgt).unsqueeze(0)
            b_y_tgt     = y.index_select(0, idx_tgt).unsqueeze(0)
            b_file_idx_tgt = file_indices.index_select(0, idx_tgt).unsqueeze(0)

            if self.context_ratio > 0.:
                idx_ctx = self.data[self.mode]["context"]["batches"][b]
                b_phi_ctx = phi.index_select(0, idx_ctx).unsqueeze(0)
                b_theta_ctx = theta.index_select(0, idx_ctx).unsqueeze(0)
                b_y_ctx     = y.index_select(0, idx_ctx).unsqueeze(0)
                b_file_idx_ctx = file_indices.index_select(0, idx_ctx).unsqueeze(0)
            else:
                b_theta_ctx, b_phi_ctx, b_y_ctx, idx_ctx, b_file_idx_ctx = torch.empty(0), torch.empty(0), torch.empty(0),torch.empty(0), torch.empty(0) 


            batch = BatchCollection(
                context=ContextSet(theta=b_theta_ctx.contiguous(), phi=b_phi_ctx.contiguous(), y=b_y_ctx.contiguous(), idx=idx_ctx, file_indices=b_file_idx_ctx),
                query=QuerySet(theta=b_theta_tgt.contiguous(), phi=b_phi_tgt.contiguous(), idx=idx_tgt, file_indices=b_file_idx_tgt),
                target_y=b_y_tgt.contiguous(),
            )
            yield batch

    def __getitem__(self, index):
        """
        Return the i-th yielded batch.
        """
        idx_tgt = self.data[self.mode]["target"]["batches"][index]
        b_phi_tgt = self.data["data"]["phi"].index_select(0, idx_tgt).unsqueeze(0)
        b_theta_tgt = self.data["data"]["theta"].index_select(0, idx_tgt).unsqueeze(0) if self.data[self.mode]["target"].get("theta",None) is not None else None
        b_y_tgt     = self.data["data"]["y"].index_select(0, idx_tgt).unsqueeze(0)     if self.data[self.mode]["target"].get("y",None)     is not None else None
        b_file_idx_tgt = self.data["data"]["file_indices"].index_select(0, idx_tgt).unsqueeze(0)

        if self.context_ratio > 0.:
            idx_ctx = self.data[self.mode]["context"]["batches"][index]
            b_phi_ctx = self.data["data"]["phi"].index_select(0, idx_ctx).unsqueeze(0)
            b_theta_ctx = self.data["data"]["theta"].index_select(0, idx_ctx).unsqueeze(0) if self.data[self.mode]["context"].get("theta",None) is not None else None
            b_y_ctx     = self.data["data"]["y"].index_select(0, idx_ctx).unsqueeze(0)     if self.data[self.mode]["context"].get("y",None)     is not None else None
            b_file_idx_ctx = self.data["data"]["file_indices"].index_select(0, idx_ctx).unsqueeze(0)
        else:
            b_theta_ctx, b_phi_ctx, b_y_ctx, idx_ctx, b_file_idx_ctx = torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0) 


        batch = BatchCollection(
            context=ContextSet(theta=b_theta_ctx.contiguous(), phi=b_phi_ctx.contiguous(), y=b_y_ctx.contiguous(), idx=idx_ctx, file_indices=b_file_idx_ctx),
            query=QuerySet(theta=b_theta_tgt.contiguous(), phi=b_phi_tgt.contiguous(), idx=idx_tgt, file_indices=b_file_idx_tgt),
            target_y=b_y_tgt.contiguous(),
        )

        yield batch
 
    def close(self):
        """Delete all tensors and arrays from memory to free up resources."""
        # Clear main data dictionary
        if hasattr(self, 'data'):
            for mode in self.data:
                for key in self.data[mode]:
                    self.data[mode][key] = None
            self.data = None
        
        # Clear unused data
        for attr in ['_theta_unused', '_phi_unused', '_y_unused', '_fidx_unused']:
            if hasattr(self, attr):
                setattr(self, attr, None)
        
        # Clear normalizer
        if hasattr(self, '_normalizer'):
            self._normalizer = None
        
        # Clear other attributes that might hold data
        self.files = None

    def __len__(self) -> int:
        return self.data[self.mode]["meta"]["num_batches"]
    
    def num_samples(self) -> int:
        return self.data["data"]["phi"].shape[-2]
    
    def get_data(self, key: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get all data tensors for a given key ('train','test', 'validate')."""
        if key not in self.data.keys():
            raise ValueError(f"Invalid key: {key}. Must be one of {list(self.data.keys())}.")
        
        if self.context_is_subset or self.context_ratio == 0.0:
            idx = self.data[key]["target"]["indices"]
        else:
            idx = torch.cat(
                (
                    self.data[key]["context"]["indices"],
                    self.data[key]["target"]["indices"],
                )
            )
        theta = self.data["data"]["theta"]
        phi = self.data["data"]["phi"]
        y = self.data["data"]["y"]

        return theta.index_select(0, idx), phi.index_select(0, idx), y.index_select(0, idx)
    
    def get_positives(self, key: str):
        theta, phi, y = self.get_data(key)
        pos_mask = self.sampler.get_positive_indices(y)
        pos_idx = pos_mask.nonzero(as_tuple=False).view(-1)
        return theta.index_select(0, pos_idx), phi.index_select(0, pos_idx), y.index_select(0, pos_idx)
    
    def get_negatives(self, key: str):
        theta, phi, y = self.get_data(key)
        pos_mask = self.sampler.get_positive_indices(y)
        neg_idx = (~pos_mask).nonzero(as_tuple=False).view(-1)
        return theta.index_select(0, neg_idx), phi.index_select(0, neg_idx), y.index_select(0, neg_idx)
    

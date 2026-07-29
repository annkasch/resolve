import math
import torch
from torch.utils.data import Dataset
from typing import Optional, Sequence, Tuple
from resolve.helpers.batch_requests import BatchRequest, BatchSpan
from resolve.helpers.batch_types import BatchCollection, ContextSet, QuerySet
from resolve.helpers.data_source import (
    DataValidationError,
    DatasetSettings,
    ValidatedDataSource,
    ValidationIssue,
)
from resolve.helpers.data_store import (
    DataStore,
    InMemoryDataStore,
    StreamingMaterializationError,
)
from resolve.helpers.normalizer import Normalizer
from resolve.helpers.sampler import MixupPlan, Sampler
from resolve.helpers.splitter import Splitter

class InMemoryIterableData(Dataset):
    _MODE_TO_CODE = {
        "train": 0,
        "validate": 1,
        "test": 2,
        "inference": 3,
    }
    _CODE_TO_MODE = {
        code: mode for mode, code in _MODE_TO_CODE.items()
    }

    def __init__(self, data_source: ValidatedDataSource, batch_size: int = 1000,
                 dataset_config: DatasetSettings = None, positive_condition: Optional[Sequence[str]]=None,
                 normalizer: Optional[Normalizer] = None, mode: Optional[str] = "train",
                 data_store: Optional[DataStore] = None) -> None:
        super().__init__()

        if not isinstance(data_source, ValidatedDataSource):
            raise TypeError(
                "data_source must be a ValidatedDataSource produced by "
                "preflight_data_loader()."
            )
        self.data_source = data_source
        self.files = [str(path) for path in data_source.paths]
        self.shuffle = dataset_config.shuffle_dataset
        self.seed = dataset_config.seed
        self.dataset_config = dataset_config
        self.batch_size = batch_size
        
        self.mode = mode
        self._normalizer = self._prepare_normalizer(normalizer)
        self.sampler = Sampler(positive_condition, shuffle=self.shuffle, seed=self.seed)
        self._base_indices = {}
        self._positive_pools = {}
        self.mixup_plan = MixupPlan.empty()
        self._built_epochs = {}
        self._batch_plan_build_count = torch.tensor(
            0,
            dtype=torch.int64,
        ).share_memory_()
        self._iteration_mode = torch.tensor(
            self._MODE_TO_CODE[self.mode],
            dtype=torch.int64,
        ).share_memory_()
        self._iteration_epoch = torch.tensor(
            0,
            dtype=torch.int64,
        ).share_memory_()

        self.store = (
            InMemoryDataStore.from_source(
                self.data_source,
                chunk_rows=self.dataset_config.stream_chunk_rows,
            )
            if data_store is None
            else data_store
        )
        if not isinstance(self.store, DataStore):
            raise TypeError("data_store must implement the DataStore contract.")
        if self.store.backend == "memory":
            theta, phi, y, fidx = self.store.materialize()
        else:
            theta = None
            phi = None
            y = self.store.metadata_targets
            fidx = self.store.metadata_file_indices
        self._validate_loaded_data(theta, phi, y, fidx)

        self.data = self._set_data(theta, phi, y, fidx)
        if self.store.backend == "memory":
            stored = self.data["data"]
            self.store.replace_tensors(
                stored["theta"],
                stored["phi"],
                stored["y"],
                stored["file_indices"],
            )
        self.build_batches(0)

    @staticmethod
    def _canonical_normalization_method(method):
        return None if method in (None, "none") else method

    def _prepare_normalizer(
        self,
        normalizer: Optional[Normalizer],
    ) -> Optional[Normalizer]:
        configured_method = self.dataset_config.use_feature_normalization
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
            normalizer.validate_schema(
                "theta",
                self.data_source.selected_labels("theta"),
            )
            normalizer.validate_schema(
                "phi",
                self.data_source.selected_labels("phi"),
            )
        return normalizer

    def _validate_loaded_data(self, theta, phi, y, fidx):
        row_counts = {
            "store": self.store.num_samples,
            "file_indices": fidx.shape[0],
        }
        if theta is not None:
            row_counts["theta"] = theta.shape[0]
        if phi is not None:
            row_counts["phi"] = phi.shape[0]
        if y is not None:
            row_counts["target"] = y.shape[0]
        issues = []
        if len(set(row_counts.values())) != 1:
            issues.append(
                ValidationIssue(
                    "loaded data",
                    f"has inconsistent row counts: {row_counts}",
                )
            )
        if not row_counts["store"]:
            issues.append(
                ValidationIssue(
                    "loaded data",
                    "contains no samples",
                )
            )
        if issues:
            raise DataValidationError(issues)

    @staticmethod
    def _split_or_raise(
        splitter,
        indices,
        groups,
        test_size,
        split_name,
    ):
        try:
            remaining, held_out = splitter.train_test_split(
                indices,
                groups=groups,
                test_size=test_size,
            )
        except ValueError as error:
            raise DataValidationError(
                (
                    ValidationIssue(
                        f"dataset split.{split_name}",
                        f"cannot be created: {error}",
                    ),
                )
            ) from error

        issues = []
        if remaining.numel() == 0:
            issues.append(
                ValidationIssue(
                    f"dataset split.{split_name}",
                    "leaves no samples for subsequent training splits",
                )
            )
        if held_out.numel() == 0:
            issues.append(
                ValidationIssue(
                    f"dataset split.{split_name}",
                    "contains no samples",
                )
            )
        if issues:
            raise DataValidationError(issues)
        return remaining, held_out

    def _validate_partition_sizes(self, split_indices):
        if self.context_ratio == 0.0:
            return
        issues = [
            ValidationIssue(
                f"dataset split.{mode}",
                "requires at least two samples when context_ratio is positive",
            )
            for mode, indices in split_indices.items()
            if indices.numel() < 2
        ]
        if issues:
            raise DataValidationError(issues)
        
    def _fit_normalizer_on_indices(self, normalizer, indices):
        chunk_rows = self.dataset_config.stream_chunk_rows

        def chunks(feature_group):
            for index_chunk in torch.split(indices, chunk_rows):
                theta, phi, _target, _file_indices = self.store.read_rows(
                    index_chunk
                )
                yield theta if feature_group == "theta" else phi

        normalizer.fit_chunks(
            chunks("theta"),
            "theta",
            self.data_source.selected_labels("theta"),
        )
        normalizer.fit_chunks(
            chunks("phi"),
            "phi",
            self.data_source.selected_labels("phi"),
        )

    def _set_data(self, theta, phi, y, fidx):
        self.context_ratio = self.dataset_config.context_ratio
        if not 0.0 <= self.context_ratio < 1.0:
            raise ValueError("context_ratio must be in [0, 1).")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        self.context_is_subset = self.dataset_config.context_is_subset
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

        if self.mode != "inference" and y is None:
            raise DataValidationError(
                (
                    ValidationIssue(
                        "loaded data.target",
                        f"is required for {self.mode} mode",
                    ),
                )
            )
        if y is not None and self.sampler.positive_fn is not None:
            pos_mask = self.sampler.get_positive_indices(y)
            positive_ratio_data = pos_mask.sum(dim=0) / y.shape[0]
        else:
            positive_ratio_data = None
        splitter = Splitter(self.shuffle, seed=self.seed)
        idx = torch.arange(self.store.num_samples)
        data = {}
        
        if self.mode == "train":
            split_indices = {}
            val_size = self.dataset_config.val_ratio
            if val_size > 0.0:
                idx, split_indices["validate"] = self._split_or_raise(
                    splitter,
                    idx,
                    (
                        theta[idx]
                        if theta is not None
                        else fidx.index_select(0, idx)
                    ),
                    val_size,
                    "validate",
                )

            configured_test_size = self.dataset_config.test_ratio
            if configured_test_size > 0.0:
                remaining_fraction = 1.0 - val_size
                if remaining_fraction <= 0.0:
                    raise ValueError(
                        "val_ratio must leave data available for training "
                        "and testing."
                    )
                idx, split_indices["test"] = self._split_or_raise(
                    splitter,
                    idx,
                    (
                        theta[idx]
                        if theta is not None
                        else fidx.index_select(0, idx)
                    ),
                    configured_test_size / remaining_fraction,
                    "test",
                )
            split_indices["train"] = idx
            self._validate_partition_sizes(split_indices)

            # Fit feature transforms on training rows only, then apply them
            # consistently to all partitions.
            self._normalizer = Normalizer(
                self.dataset_config.use_feature_normalization
            )
            self._fit_normalizer_on_indices(self._normalizer, idx)
            if theta is not None:
                theta = (
                    self._normalizer.transform(theta, "theta")
                    .float()
                    .contiguous()
                )
                phi = (
                    self._normalizer.transform(phi, "phi")
                    .float()
                    .contiguous()
                )
            else:
                self.store.set_normalizer(self._normalizer)

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

            if self.dataset_config.mixup_ratio > 0.0:
                self.mixup_plan = self.sampler.build_mixup_plan(
                    idx,
                    fidx.index_select(0, idx),
                    y.index_select(0, idx),
                    self.dataset_config.mixup_ratio,
                    use_beta=self.dataset_config.use_beta,
                    margin=self.dataset_config.mixup_margin,
                    seed=self.seed,
                )

            data = {"data": data["data"]}
            for split_mode, split_idx in split_indices.items():
                self._base_indices[split_mode] = split_idx.clone()
                if (
                    split_mode == "train"
                    and isinstance(
                        self.dataset_config.positive_ratio_train,
                        float,
                    )
                ):
                    self._positive_pools[split_mode] = (
                        self.sampler.prepare_positive_pools(
                            fidx.index_select(0, split_idx),
                            y.index_select(0, split_idx),
                            split_idx,
                        )
                    )
                data[split_mode] = self._empty_mode_plan(
                    positive_ratio_data
                )
        else:
            self._validate_partition_sizes({self.mode: idx})
            if self._canonical_normalization_method(
                self.dataset_config.use_feature_normalization
            ) is None:
                self._fit_normalizer_on_indices(self._normalizer, idx)
            if theta is not None:
                theta = self._normalizer.transform(
                    x=theta,
                    feature_grp="theta",
                )
                phi = self._normalizer.transform(
                    x=phi,
                    feature_grp="phi",
                )
                theta = theta.float().contiguous()
                phi = phi.float().contiguous()
            else:
                self.store.set_normalizer(self._normalizer)
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
        return {
            "order": torch.empty(0, dtype=torch.long),
            "spans": (),
            "context": {
                "batch_size": self.batch_size_ctx,
                "ratio": self.context_ratio,
            },
            "target": {
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
        positive_ratio = self.dataset_config.positive_ratio_train
        if (
            mode != "train"
            or isinstance(positive_ratio, (list, tuple))
            or positive_ratio is None
        ):
            return indices, None

        selected, _, unused, meta = (
            self.sampler.sample_prepared_positive_pools(
            self._positive_pools[mode],
            target_pos_frac=positive_ratio,
            max_pos_reuse_per_epoch=self.dataset_config.max_positive_reuse,
            sticky_frac=0.25,
            epoch=epoch,
            )
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

        self._batch_plan_build_count.add_(1)
        indices, sampling_meta = self._indices_for_epoch(mode, plan_epoch)
        combined_batches, _, _ = self.sampler.build_batches(
            indices,
            batch_size=self.batch_size,
            epoch=plan_epoch,
        )
        combined_batches = self._rebalance_singleton_batch(combined_batches)

        ordered_batches = []
        spans = []
        start = 0
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
            else:
                context_size = 0
            stop = start + combined_batch.numel()
            ordered_batches.append(combined_batch)
            spans.append(
                BatchSpan(
                    start=start,
                    stop=stop,
                    context_size=context_size,
                    context_is_subset=self.context_is_subset,
                )
            )
            start = stop

        mode_data = self.data[mode]
        mode_data["order"] = (
            torch.cat(ordered_batches)
            if ordered_batches
            else torch.empty(0, dtype=torch.long)
        )
        mode_data["spans"] = tuple(spans)
        mode_data["meta"]["num_batches"] = len(spans)
        if sampling_meta is not None:
            mode_data["meta"].update(sampling_meta)
            mode_data["meta"]["num_batches"] = len(spans)
        self._built_epochs[mode] = plan_epoch

    def set_mode(self, mode):
        if mode not in self._MODE_TO_CODE:
            raise ValueError(
                f"Unsupported dataset mode {mode!r}. Expected one of "
                f"{sorted(self._MODE_TO_CODE)}."
            )
        self.mode = mode

        self._iteration_mode.fill_(self._MODE_TO_CODE[mode])

    def set_iteration(self, mode, epoch):
        self.set_mode(mode)
        self._iteration_epoch.fill_(int(epoch))
    
    def batch_plan(self, mode):
        return self.data[mode]["order"], self.data[mode]["spans"]

    def context_indices(self, mode):
        order, spans = self.batch_plan(mode)
        pieces = [
            order[span.start:span.start + span.context_size]
            for span in spans
        ]
        return (
            torch.cat(pieces)
            if pieces
            else torch.empty(0, dtype=torch.long)
        )

    def query_indices(self, mode):
        order, spans = self.batch_plan(mode)
        pieces = [
            order[
                span.start
                if span.context_is_subset
                else span.start + span.context_size
                :span.stop
            ]
            for span in spans
        ]
        return (
            torch.cat(pieces)
            if pieces
            else torch.empty(0, dtype=torch.long)
        )

    def __getitem__(self, request):
        if not isinstance(request, BatchRequest):
            raise TypeError("Dataset indices must be BatchRequest objects.")
        (
            theta,
            phi,
            target,
            file_indices,
        ) = self.store.read_rows(request.indices)
        if (
            target is not None
            and self.mixup_plan.destinations.numel() > 0
        ):
            theta, phi, target = self.mixup_plan.apply(
                request.indices,
                theta,
                phi,
                target,
                self.store,
            )
        context_size = request.context_size
        idx_ctx = request.indices[:context_size]
        query_start = 0 if request.context_is_subset else context_size
        idx_tgt = request.indices[query_start:]

        b_theta_ctx = theta[:context_size].unsqueeze(0)
        b_phi_ctx = phi[:context_size].unsqueeze(0)
        b_y_ctx = (
            target[:context_size].unsqueeze(0)
            if target is not None
            else None
        )
        b_file_idx_ctx = file_indices[:context_size].unsqueeze(0)
        b_theta_tgt = theta[query_start:].unsqueeze(0)
        b_phi_tgt = phi[query_start:].unsqueeze(0)
        b_y_tgt = (
            target[query_start:].unsqueeze(0)
            if target is not None
            else None
        )
        b_file_idx_tgt = file_indices[query_start:].unsqueeze(0)

        return BatchCollection(
            context=ContextSet(
                theta=b_theta_ctx.contiguous(),
                phi=b_phi_ctx.contiguous(),
                y=(
                    b_y_ctx.contiguous()
                    if b_y_ctx is not None
                    else None
                ),
                idx=idx_ctx,
                file_indices=b_file_idx_ctx,
            ),
            query=QuerySet(
                theta=b_theta_tgt.contiguous(),
                phi=b_phi_tgt.contiguous(),
                idx=idx_tgt,
                file_indices=b_file_idx_tgt,
            ),
            target_y=(
                b_y_tgt.contiguous()
                if b_y_tgt is not None
                else None
            ),
        )

    def close(self):
        """Delete all tensors and arrays from memory to free up resources."""
        if getattr(self, "store", None) is not None:
            self.store.close()
            self.store = None
        # Clear main data dictionary
        if hasattr(self, 'data'):
            for mode in self.data:
                for key in self.data[mode]:
                    self.data[mode][key] = None
            self.data = None
        
        # Clear normalizer
        if hasattr(self, '_normalizer'):
            self._normalizer = None
        
        # Clear other attributes that might hold data
        self.files = None

    def __len__(self) -> int:
        return self.data[self.mode]["meta"]["num_batches"]

    @property
    def storage_mode(self):
        return self.store.backend

    @property
    def has_targets(self):
        return self.store.has_targets

    @property
    def available_modes(self):
        return tuple(mode for mode in self.data if mode != "data")

    def has_mode(self, mode):
        return mode in self.data and mode != "data"

    def target_batch_size(self, mode=None):
        mode = self.mode if mode is None else mode
        return self.data[mode]["target"]["batch_size"]

    def sampling_epochs(self, mode="train"):
        return self.data[mode]["meta"]["num_epochs"]

    def positive_fraction(self, mode="train"):
        return self.data[mode]["meta"]["pos_frac"]

    def require_memory_backend(self, consumer):
        if self.store.backend != "memory":
            raise StreamingMaterializationError(
                f"{consumer} requires complete dataset tensors. Configure "
                "model_settings.train.dataset.storage_mode: memory."
            )

    def materialized_tensors(self, consumer="Full-tensor access"):
        self.require_memory_backend(consumer)
        return self.store.materialize()
    
    def num_samples(self) -> int:
        return self.store.num_samples
    
    def get_data(
        self,
        key: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Get all data tensors for a given key ('train','test', 'validate')."""
        if self.store.backend != "memory":
            raise StreamingMaterializationError(
                "get_data() requires complete tensors. Set "
                "storage_mode='memory' for this model path."
            )
        if key not in self.data.keys():
            raise ValueError(f"Invalid key: {key}. Must be one of {list(self.data.keys())}.")
        
        idx = self.data[key]["order"]
        theta = self.data["data"]["theta"]
        phi = self.data["data"]["phi"]
        y = self.data["data"]["y"]

        return (
            theta.index_select(0, idx),
            phi.index_select(0, idx),
            y.index_select(0, idx) if y is not None else None,
        )
    
    def get_positives(self, key: str):
        theta, phi, y = self.get_data(key)
        if y is None:
            raise ValueError("Positive samples require target labels.")
        pos_mask = self.sampler.get_positive_indices(y)
        pos_idx = pos_mask.nonzero(as_tuple=False).view(-1)
        return theta.index_select(0, pos_idx), phi.index_select(0, pos_idx), y.index_select(0, pos_idx)
    
    def get_negatives(self, key: str):
        theta, phi, y = self.get_data(key)
        if y is None:
            raise ValueError("Negative samples require target labels.")
        pos_mask = self.sampler.get_positive_indices(y)
        neg_idx = (~pos_mask).nonzero(as_tuple=False).view(-1)
        return theta.index_select(0, neg_idx), phi.index_select(0, neg_idx), y.index_select(0, neg_idx)
    

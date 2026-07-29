import logging
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from resolve.helpers.data_source import preflight_data_loader
from resolve.helpers.data_store import (
    StreamingDataStore,
    select_storage_backend,
)
from resolve.helpers.iterable_dataset import InMemoryIterableData
from resolve.helpers.loader_lifecycle import (
    ReusableDataLoader,
    shutdown_loader,
)
from resolve.helpers.normalizer import Normalizer


def _freeze_signature(value):
    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                sorted(
                    (
                        repr(key),
                        _freeze_signature(item),
                    )
                    for key, item in value.items()
                )
            ),
        )
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return (
            type(value).__qualname__,
            tuple(_freeze_signature(item) for item in value),
        )
    if isinstance(value, os.PathLike):
        return ("path", os.fspath(value))
    return (type(value).__qualname__, repr(value))


class DataLoaderManager:
    """Own validated data, deterministic batch plans, and one reusable loader.

    A returned loader must be fully consumed, or its iterator explicitly
    closed, before requesting another epoch or mode from this manager.
    """

    def __init__(self, mode, config_file, normalizer=None):
        self.mode = mode
        self.config_file = config_file
        self.dataloader = None
        self._normalizer = None
        self.dataset = None
        self._preflight()
        if normalizer is not None:
            self._store_external_normalizer(normalizer)

    # ------------- helpers -------------
    def _preflight(self):
        specification, data_source = preflight_data_loader(
            self.mode,
            self.config_file,
        )
        self._install_preflight(specification, data_source)

    def _install_preflight(self, specification, data_source):
        self._specification = specification
        self._data_source = data_source
        self._configuration_signature = _freeze_signature(self.config_file)
        self._source_signature = self._source_manifest(specification)
        self.files = [
            str(path) for path in self._data_source.paths
        ]
        self.parameters = {
            parameter.name: {
                "key": parameter.dataset_key,
                "selected_labels": list(parameter.selected_labels),
            }
            for parameter in self._specification.parameters
        }
        self.positive_condition = list(
            self._specification.positive_condition
        )

    @staticmethod
    def _source_manifest(specification):
        paths = sorted(
            specification.data_directory.glob(
                f"*.{specification.file_format}"
            )
        )
        manifest = []
        for path in paths:
            try:
                stat = path.stat()
            except OSError:
                manifest.append((str(path), None))
                continue
            manifest.append(
                (
                    str(path),
                    stat.st_dev,
                    stat.st_ino,
                    stat.st_size,
                    stat.st_mtime_ns,
                    stat.st_ctime_ns,
                )
            )
        return tuple(manifest)

    def _preflight_is_current(self):
        return (
            _freeze_signature(self.config_file)
            == self._configuration_signature
            and self._source_manifest(self._specification)
            == self._source_signature
        )

    def _current_preflight(self):
        if self._preflight_is_current():
            return self._specification, self._data_source
        return preflight_data_loader(
            self.mode,
            self.config_file,
        )

    @staticmethod
    def _canonical_normalization_method(method):
        return None if method in (None, "none") else method

    @property
    def normalizer(self):
        return self._normalizer

    def _configured_normalization_method(self, specification=None):
        specification = (
            self._specification
            if specification is None
            else specification
        )
        return specification.dataset.use_feature_normalization

    def _validate_external_normalizer(
        self,
        normalizer,
        specification,
        data_source,
    ):
        if self.mode == "train":
            raise ValueError(
                "Training managers create and fit their own normalizer; do "
                "not inject one."
            )
        if not isinstance(normalizer, Normalizer):
            raise TypeError("normalizer must be a Normalizer instance.")

        configured_method = self._configured_normalization_method(
            specification
        )
        if self._canonical_normalization_method(
            normalizer.method
        ) != self._canonical_normalization_method(configured_method):
            raise ValueError(
                f"Injected normalizer method {normalizer.method!r} does not "
                f"match configured method {configured_method!r}."
            )
        if self._canonical_normalization_method(configured_method) is not None:
            normalizer.validate_fitted()
            normalizer.validate_schema(
                "theta",
                data_source.selected_labels("theta"),
            )
            normalizer.validate_schema(
                "phi",
                data_source.selected_labels("phi"),
            )

    def _store_external_normalizer(self, normalizer):
        self._validate_external_normalizer(
            normalizer,
            self._specification,
            self._data_source,
        )
        self._normalizer = normalizer

    def set_dataset(self, normalizer=None):
        specification, data_source = self._current_preflight()
        storage_selection = select_storage_backend(
            data_source,
            specification.dataset,
        )
        logging.getLogger(__name__).info(
            "Selected %s dataloader backend (estimated peak %.1f MiB, "
            "budget %.1f MiB): %s.",
            storage_selection.backend,
            storage_selection.estimated_peak_bytes / 1024**2,
            storage_selection.memory_budget_bytes / 1024**2,
            storage_selection.reason,
        )
        external_normalizer = (
            normalizer if normalizer is not None else self._normalizer
        )
        if self.mode != "train" and external_normalizer is not None:
            self._validate_external_normalizer(
                external_normalizer,
                specification,
                data_source,
            )

        configured_method = self._configured_normalization_method(
            specification
        )
        if (
            self.mode != "train"
            and self._canonical_normalization_method(configured_method)
            is not None
            and external_normalizer is None
        ):
            raise ValueError(
                f"{self.mode.capitalize()} data using "
                f"{configured_method!r} normalization requires a fitted "
                "training normalizer. Pass it to DataLoaderManager(..., "
                "normalizer=training_manager.normalizer) or "
                "set_dataset(normalizer=...)."
            )

        cache_directory = specification.dataset.cache_directory
        if cache_directory is None:
            configured_output = self.config_file.get(
                "path_settings",
                {},
            ).get("path_out_model")
            cache_directory = (
                Path(configured_output).expanduser() / ".resolve-cache"
                if configured_output
                else Path.home() / ".cache" / "resolve"
            )
        data_store = (
            StreamingDataStore(
                data_source,
                chunk_rows=specification.dataset.stream_chunk_rows,
                cache_directory=cache_directory,
            )
            if storage_selection.backend == "streaming"
            else None
        )
        try:
            replacement = InMemoryIterableData(
                    data_source=data_source,
                    batch_size=specification.batch_size,
                    dataset_config=specification.dataset,
                    positive_condition=specification.positive_condition,
                    normalizer=(
                        None if self.mode == "train" else external_normalizer
                    ),
                    mode=self.mode,
                    data_store=data_store,
                )
        except Exception:
            if data_store is not None:
                data_store.close()
            raise
        self._dispose_loader(close_dataset=True)
        self._install_preflight(specification, data_source)
        self.dataset = replacement
        self.storage_selection = storage_selection
        self._normalizer = self.dataset._normalizer

    def _loader_options(self):
        settings = self._specification.loader
        num_workers = settings.num_workers

        options = {
            "batch_size": None,
            "num_workers": num_workers,
            "pin_memory": settings.pin_memory,
            "persistent_workers": (
                settings.persistent_workers
                if num_workers > 0
                else False
            ),
        }
        if num_workers > 0:
            options["prefetch_factor"] = settings.prefetch_factor
        return options

    def _dispose_loader(self, close_dataset=False):
        loader = self.dataloader
        dataset = self.dataset
        if loader is not None:
            shutdown_loader(loader)
            dataset = loader.dataset
            self.dataloader = None

        if close_dataset and dataset is not None:
            dataset.close()
            if self.dataset is dataset:
                self.dataset = None

    def set_loader(self, epoch, mode=None, shuffle=True):
        """Return the cached loader configured for one complete iteration.

        Finish or close the current iterator before changing epoch or mode.
        Evaluation plans always use epoch zero inside the dataset.
        """

        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a non-negative integer.")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a boolean.")
        if (
            self.dataloader is not None
            and self.dataloader.iteration_active
        ):
            raise RuntimeError(
                "Cannot change DataLoader epoch or mode while an iterator is "
                "active. Finish or close the current iterator first."
            )
        if self.dataset is None:
            self.set_dataset()

        mode = self.mode if mode is None else mode
        plan_epoch = (
            epoch
            if shuffle
            else self.dataset._built_epochs.get(mode, 0)
        )
        self.dataset.build_batches(plan_epoch, mode=mode)
        self.dataset.set_iteration(mode, plan_epoch)

        if self.dataloader is None:
            self.dataloader = ReusableDataLoader(
                self.dataset,
                **self._loader_options(),
            )

        return self.dataloader
    
    def close_loader(self):
        self._dispose_loader(close_dataset=True)
    

from torch.utils.data import DataLoader
from resolve.helpers.data_source import preflight_data_loader
from resolve.helpers.iterable_dataset import InMemoryIterableData
from resolve.helpers.normalizer import Normalizer


class DataLoaderManager:
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
        self._specification, self._data_source = preflight_data_loader(
            self.mode,
            self.config_file,
        )
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
    def _canonical_normalization_method(method):
        return None if method in (None, "none") else method

    @property
    def normalizer(self):
        return self._normalizer

    def _configured_normalization_method(self):
        return self._specification.dataset.use_feature_normalization

    def _store_external_normalizer(self, normalizer):
        if self.mode == "train":
            raise ValueError(
                "Training managers create and fit their own normalizer; do "
                "not inject one."
            )
        if not isinstance(normalizer, Normalizer):
            raise TypeError("normalizer must be a Normalizer instance.")

        configured_method = self._configured_normalization_method()
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
                self._data_source.selected_labels("theta"),
            )
            normalizer.validate_schema(
                "phi",
                self._data_source.selected_labels("phi"),
            )
        self._normalizer = normalizer

    def set_dataset(self, normalizer=None):
        self._preflight()
        if normalizer is not None:
            self._store_external_normalizer(normalizer)
        elif self.mode != "train" and self._normalizer is not None:
            self._store_external_normalizer(self._normalizer)

        configured_method = self._configured_normalization_method()
        if (
            self.mode != "train"
            and self._canonical_normalization_method(configured_method)
            is not None
            and self._normalizer is None
        ):
            raise ValueError(
                f"{self.mode.capitalize()} data using "
                f"{configured_method!r} normalization requires a fitted "
                "training normalizer. Pass it to DataLoaderManager(..., "
                "normalizer=training_manager.normalizer) or "
                "set_dataset(normalizer=...)."
            )

        self._dispose_loader(close_dataset=True)
        self.dataset = InMemoryIterableData(
                data_source=self._data_source,
                batch_size=self._specification.batch_size,
                dataset_config=self._specification.dataset,
                positive_condition=self._specification.positive_condition,
                normalizer=(
                    None if self.mode == "train" else self._normalizer
                ),
                mode=self.mode
            )
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
            iterator = getattr(loader, "_iterator", None)
            if iterator is not None:
                iterator._shutdown_workers()
                loader._iterator = None
            dataset = loader.dataset
            self.dataloader = None

        if close_dataset and dataset is not None:
            dataset.close()
            if self.dataset is dataset:
                self.dataset = None

    def set_loader(self, epoch, mode=None, shuffle=True):
        if self.dataset is None:
            self.set_dataset()

        mode = self.mode if mode is None else mode
        plan_epoch = (
            epoch
            if shuffle
            else self.dataset._built_epochs.get(mode, 0)
        )
        self.dataset.set_iteration(mode, plan_epoch)
        self.dataset.build_batches(plan_epoch, mode=mode)

        if self.dataloader is None:
            self.dataloader = DataLoader(
                self.dataset,
                **self._loader_options(),
            )

        return self.dataloader
    
    def close_loader(self):
        self._dispose_loader(close_dataset=True)
    

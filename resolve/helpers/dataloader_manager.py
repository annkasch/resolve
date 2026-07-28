from pathlib import Path
import collections
from torch.utils.data import DataLoader
from resolve.helpers.iterable_dataset import InMemoryIterableData
from resolve.helpers.normalizer import Normalizer


ContextSet = collections.namedtuple("ContextSet", ("theta", "phi", "y"))
QuerySet   = collections.namedtuple("QuerySet",   ("theta", "phi"))

BatchCollection = collections.namedtuple(
    "BatchCollection",
    ("context", "query", "target_y")
)

def running_average(batch_sum, batch_count, mean, I):
    mean = mean + (batch_sum - batch_count * mean) / (I + batch_count)
    I += batch_count
    return mean

class DataLoaderManager:
    def __init__(self, mode, config_file, normalizer=None):
        self.mode = mode
        self.config_file = config_file
        
        self.files = self._get_hdf5_files(Path(self.config_file["path_settings"][f"path_to_files_{self.mode}"]))
        self.dataloader = None
        self._normalizer = None

        # base parameter spec
        sim = config_file["simulation_settings"]

        self.parameters = {
            "phi": {
                "key": "features/values",
                "selected_labels": sim["phi_labels"],
            },
            "theta": {
                "key": "features/values",
                "selected_labels": sim["theta_labels"],
            },
            "target": {
                "key": "labels/values",
                "selected_labels": sim["target_labels"],
            },
        }

        self.positive_condition  = self.config_file["simulation_settings"]["signal_condition"]

        self.dataset = None
        if normalizer is not None:
            self._store_external_normalizer(normalizer)

    # ------------- helpers -------------
    def _get_hdf5_files(self, path_to_files):
        return sorted(str(p) for p in path_to_files.glob(f"*.{self.config_file['simulation_settings']['file_format']}"))

    @staticmethod
    def _canonical_normalization_method(method):
        return None if method in (None, "none") else method

    @property
    def normalizer(self):
        return self._normalizer

    def _configured_normalization_method(self):
        return self.config_file["model_settings"]["train"]["dataset"].get(
            "use_feature_normalization",
            None,
        )

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
        self._normalizer = normalizer

    def set_dataset(self, normalizer=None):
        dataset_config = self.config_file["model_settings"]["train"]["dataset"]
        if normalizer is not None:
            self._store_external_normalizer(normalizer)

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
                files=self.files,
                batch_size=self.config_file["model_settings"]["train"]["batch_size"],
                parameter_config=self.parameters,
                dataset_config=dataset_config,
                positive_condition=self.positive_condition,
                normalizer=(
                    None if self.mode == "train" else self._normalizer
                ),
                mode=self.mode
            )
        self._normalizer = self.dataset._normalizer

    def _loader_options(self):
        settings = self.config_file["model_settings"]["dataloader"]
        num_workers = int(settings["dataloader_number_of_workers"])
        if num_workers < 0:
            raise ValueError(
                "dataloader_number_of_workers must be non-negative."
            )

        options = {
            "batch_size": None,
            "num_workers": num_workers,
            "pin_memory": bool(
                settings.get("dataloader_pin_memory", False)
            ),
            "persistent_workers": (
                bool(
                    settings.get(
                        "dataloader_persistent_workers",
                        False,
                    )
                )
                if num_workers > 0
                else False
            ),
        }
        if num_workers > 0:
            options["prefetch_factor"] = settings.get(
                "dataloader_prefetch_factor",
                None,
            )
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

    def set_loader(self, epoch, mode="train", shuffle=True):
        if self.dataset is None:
            self.set_dataset()

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
    

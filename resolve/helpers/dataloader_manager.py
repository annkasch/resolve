import h5py
from pathlib import Path
import collections
import torch
from torch.utils.data import DataLoader
from resolve.helpers.iterable_dataset import InMemoryIterableData
from resolve.helpers.normalizer import Normalizer
from resolve.utilities import utilities as utils
utils.set_random_seed(42)


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
    def __init__(self, mode, config_file):
        self.mode = mode
        self.config_file = config_file
        
        self.files = self._get_hdf5_files(Path(self.config_file["path_settings"][f"path_to_files_{self.mode}"]))
        self.dataloader = None

        # base parameter spec
        sim = config_file["simulation_settings"]

        self.parameters = {
            "phi":    {"key": "features/values",  "selected_labels": sim["phi_labels"],    "size": len(sim["phi_labels"]),    "selected_indices": None},
            "theta":  {"key": "features/values",  "selected_labels": sim["theta_labels"],  "size": len(sim["theta_labels"]),  "selected_indices": None},
            "target": {"key": "labels/values",    "selected_labels": sim["target_labels"], "size": len(sim["target_labels"]), "selected_indices": None},
        }

        if self.files[0].endswith(('.h5', '.hdf5')):
            with h5py.File(self.files[0], "r") as f:
                for k in self.parameters:
                    labels= f[self.parameters[k]["key"]].attrs["labels"].astype(str)
                    indices = [labels.tolist().index(name) for name in self.parameters[k]["selected_labels"]]
                    self.parameters[k]["selected_indices"] = indices

        self.positive_condition  = self.config_file["simulation_settings"]["signal_condition"]

        self.dataset = None
    # ------------- helpers -------------
    def _get_hdf5_files(self, path_to_files):
        return sorted(str(p) for p in path_to_files.glob(f"*.{self.config_file['simulation_settings']['file_format']}"))

    def set_dataset(self, normalizer=Normalizer()):
        dataset_config = self.config_file["model_settings"]["train"]["dataset"]

        self.dataset = InMemoryIterableData(
                files=self.files,
                batch_size=self.config_file["model_settings"]["train"]["batch_size"],
                parameter_config=self.parameters,
                dataset_config=dataset_config,
                positive_condition=self.positive_condition,
                normalizer=normalizer,
                mode=self.mode
            )

    def set_loader(self, epoch, mode="train", shuffle=True):
        if self.dataset is None:
            self.set_dataset()
            self.dataset.set_mode(mode)
        else:
            # shuffle if provided
            self.dataset.set_mode(mode)
            if shuffle is True:
                self.dataset.build_batches(epoch)
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=None,  # required for IterableDataset
            num_workers=self.config_file["model_settings"]["dataloader"]["dataloader_number_of_workers"],
            prefetch_factor=self.config_file["model_settings"]["dataloader"]["dataloader_prefetch_factor"],
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.config_file["model_settings"]["dataloader"]["dataloader_persistent_workers"]
        )

        return self.dataloader
    
    def close_loader(self):
        self.dataloader.dataset.close()
    

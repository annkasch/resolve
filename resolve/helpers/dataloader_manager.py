import numpy as np
from pathlib import Path
from resolve.utilities import utilities as utils
from torch.utils.data import DataLoader
import collections
from resolve.helpers.iterable_dataset import InMemoryIterableData
from resolve.helpers.normalizer import Normalizer
utils.set_random_seed(42)
import torch

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
            "phi":    {"key": "phi",    "label_key": "phi_labels",    "selected_labels": sim["phi_labels"],    "size": len(sim["phi_labels"]),      "selected_indices": None},
            "theta":  {"key": "theta",  "label_key": "theta_headers", "selected_labels": sim["theta_labels"],  "size": len(sim["theta_labels"]),  "selected_indices": None},
            "target": {"key": "target", "label_key": "target_headers","selected_labels": sim["target_labels"], "size": len(sim["target_labels"]), "selected_indices": None},
        }
        if self.files[0].endswith(('.h5', '.hdf5')):
            self.parameters["phi"]["selected_indices"] = utils.find_selected_indices(self.files[0],self.parameters["phi"])
            self.parameters["target"]["selected_indices"] = utils.find_selected_indices(self.files[0],self.parameters["target"])
            self.parameters["theta"]["selected_indices"] = utils.find_selected_indices(self.files[0],self.parameters["theta"])

        self.positive_condition  = self.config_file["simulation_settings"]["signal_condition"]

        #self.positive_condition_function = np.full(len(positive_cond), None) 
        #for i, cond_str in enumerate(positive_cond):
        #        self.positive_condition_function[i] = utils.parse_condition(cond_str) 

        self.dataset = None
    # ------------- helpers -------------
    def _get_hdf5_files(self, path_to_files):
        return sorted(str(p) for p in path_to_files.glob(f"*.{self.config_file['simulation_settings']['file_format']}"))

    def set_dataset(self, normalizer=Normalizer(), shuffle = "global"):
        dataset_config = self.config_file["model_settings"]["train"]["dataset"]

        self.dataset = InMemoryIterableData(
                files=self.files,
                batch_size=self.config_file["model_settings"]["train"]["batch_size"],
                parameter_config=self.parameters,
                shuffle=shuffle,   # reshuffles every epoch
                seed=42,
                dataset_config=dataset_config,
                positive_condition=self.positive_condition,
                normalizer=normalizer,
                mode=self.mode
            )

    def set_loader(self, mode="train", shuffle=None):
        if self.dataset is None:
            # Pass shuffle only if explicitly provided
            if shuffle is not None:
                self.set_dataset(shuffle=shuffle)
            else:
                self.set_dataset()
        else:
            # Only change shuffle if provided
            if shuffle is not None:
                self.dataset.shuffle = shuffle
            
        self.dataset.set_mode(mode)
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
    


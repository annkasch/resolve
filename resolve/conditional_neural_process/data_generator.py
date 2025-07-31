import torch
import collections
import numpy as np
import os
#from imblearn.over_sampling import SMOTE
from collections import Counter
from tqdm import tqdm
import gc
from IPython.display import display, Image
import re
import sys
import yaml
import itertools
import h5py
from torch.utils.data import DataLoader, IterableDataset
from ..utilities import utilities as utils
import random

utils.set_random_seed(42)

class HDF5Dataset(IterableDataset):
    def __init__(self, hdf5_dir, 
                batch_size=3000, 
                files_per_batch=20,
                parameters = {'phi': {'key': "phi",'label_key': "phi_labels",'selected_labels': None},
                              'theta': {'key': "theta",'label_key': "theta_headers",'selected_labels': None},
                              'target': {'key': "target",'label_key': "target_headers",'selected_labels': None}}
        ):
        """
        - hdf5_dir: Directory containing HDF5 files.
        - batch_size: Number of samples per batch (3,400).
        - files_per_batch: Number of files used in each batch (34).
        """
        super().__init__()
        self.hdf5_dir = hdf5_dir
        self.batch_size = batch_size
        self.files_per_batch = files_per_batch
        self.rows_per_file = batch_size // files_per_batch
        self.parameters = parameters
        self.phi_selected_indices= None
        self.theta_selected_indices= None
        self.target_selected_indices= None

        # List and sort all HDF5 files
        self.files = sorted([os.path.join(hdf5_dir, f) for f in os.listdir(hdf5_dir) if f.endswith(".h5")])
        self.num_files = len(self.files)
        # Total row cycles per file to complete an epoch

        max_rows_per_file, self.nrows = utils.get_max_number_of_rows(self.files, self.parameters['target']['key'])
        self.total_batches = max_rows_per_file // self.rows_per_file  # maxrows / k rows per batch = c cycles per full dataset pass

    def shuffle_files(self):
        """Shuffle the file order at the start of each full dataset pass (epoch)."""
        random.shuffle(self.files)
    
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.nrows

    def __iter__(self):
        batch_idx = 0

        while batch_idx < self.total_batches:
            self.shuffle_files()

            for i in range(0, len(self.files), self.files_per_batch):
                if i == 0 and batch_idx == 0:
                    self.phi_selected_indices = utils.read_selected_indices(self.files[0], self.parameters['phi'])

                    if self.parameters['theta']['selected_labels']:
                        self.theta_selected_indices = utils.read_selected_indices(self.files[0], self.parameters['theta'])
                    else:
                        self.theta_selected_indices = None

                    labels = self.parameters['target']['selected_labels']
                    if labels:
                        if "columns[" in labels:
                            start, end = utils.parse_slice_string(labels)
                            self.target_selected_indices = [np.arange(start, end)]
                        else:
                            self.target_selected_indices = utils.read_selected_indices(self.files[0], self.parameters['target'])
                    else:
                        self.target_selected_indices = None

                batch = []
                selected_files = self.files[i:i + self.files_per_batch]
                start_idx = batch_idx * self.rows_per_file
                end_idx = start_idx + self.rows_per_file

                for file in selected_files:
                    with h5py.File(file, "r") as hdf:
                        phi = hdf[self.parameters['phi']['key']][start_idx:end_idx, self.phi_selected_indices]

                        if self.theta_selected_indices is not None:
                            theta_data = hdf[self.parameters['theta']['key']]
                            if theta_data.ndim == 1:
                                theta = theta_data[self.theta_selected_indices]
                                theta = np.tile(theta, (phi.shape[0], 1))
                            else:
                                theta = theta_data[start_idx:end_idx, self.theta_selected_indices]
                            features = np.hstack([theta, phi])
                        else:
                            features = phi

                        target_data = hdf[self.parameters['target']['key']]
                        if target_data.ndim > 1 and self.target_selected_indices is not None:
                            target = target_data[start_idx:end_idx, self.target_selected_indices]
                        else:
                            target = target_data[start_idx:end_idx]
                            target = target.reshape(-1, 1)

                        file_data = np.hstack([features, target])
                        batch.append(file_data)

                batch = np.vstack(batch)
                np.random.shuffle(batch)
                yield torch.tensor(batch, dtype=torch.float32)

            batch_idx += 1


CNPRegressionDescription = collections.namedtuple(
    "CNPRegressionDescription", ("query", "target_y")
)

class DataGeneration:
    def __init__(self, mode, config_file, path_to_files, batch_size, files_per_batch):
        self._context_ratio = config_file["cnp_settings"]["context_ratio"]
        self._batch_size = batch_size
        self.path_to_files = path_to_files
        self.dataloader = None
        self.config_file = config_file
        self.files_per_batch = files_per_batch
        self.feature_mean = None
        self.feature_std = None
        self.use_normalization = config_file["feature_settings"]["use_normalization"]

        _phi_key = "phi"
        _theta_key = "theta"
        _target_key = "target"

        self._names_theta = config_file["simulation_settings"]["theta_headers"]
        _names_phi = config_file["simulation_settings"]["phi_labels"]
        self._names_target = config_file["simulation_settings"]["target_headers"]

        self.feature_size, self.target_size = utils.get_feature_and_label_size(config_file)


        if not any(f.endswith(".h5") for f in os.listdir(path_to_files)):
            utils.convert_all_csv_to_hdf5(config_file)

        self.parameters = {
            'phi': {'key': _phi_key, 'label_key': "phi_labels", 'selected_labels': _names_phi},
            'theta': {'key': _theta_key, 'label_key': "theta_headers", 'selected_labels': self._names_theta},
            'target': {'key': _target_key, 'label_key': "target_headers", 'selected_labels': self._names_target}
        }

        if mode.startswith("training"):
            files = self._get_hdf5_files()
            if "phase2" not in mode:
                signal_condition = config_file["simulation_settings"]["signal_condition"]
                for file in tqdm(files, total=len(files), desc="Data Processing in Progress"):
                    self.split_and_mixup_augment(
                        file,
                        config_file["cnp_settings"]["split_ratio"],
                        config_file["cnp_settings"]["training"]["phase1"]["use_beta"],
                        signal_condition,
                        config_file["cnp_settings"]["training"]["phase1"]["mixup_ratio"]
                    )

                self.parameters["phi"]["key"] = "phi_train" # eventually altered training set according to mixup/ mixup_ratio
                self.parameters["target"]["key"] = "target_train" # eventually altered training set according to mixup/ mixup_ratio
            else:
                self.parameters["phi"]["key"] = "phi_train_raw" # unaltered training set
                self.parameters["target"]["key"] = "target_train_raw" # unaltered training set

        elif mode == "validation":
            self.parameters["phi"]["key"] = "phi_val"
            self.parameters["target"]["key"] = "target_val"
        elif mode == "testing":
            self.parameters["phi"]["key"] = "phi_test"
            self.parameters["target"]["key"] = "target_test"
        elif mode == "prediction":
            self.parameters["phi"]["key"] = "phi"
            self.parameters["target"]["key"] = "target"

    def _get_hdf5_files(self):
        return sorted(
            os.path.join(self.path_to_files, f)
            for f in os.listdir(self.path_to_files)
            if f.endswith(".h5")
        )

    def set_loader(self):
        if (self.feature_mean is None or self.feature_std is None) and self.use_normalization != False:
            self.compute_feature_stats()
            self.config_file["feature_settings"]["x_mean"] = self.feature_mean.numpy().tolist()
            self.config_file["feature_settings"]["x_std"] = self.feature_std.numpy().tolist()
            
        dataset = HDF5Dataset(self.path_to_files, self._batch_size, files_per_batch=self.files_per_batch, parameters=self.parameters)
        self.dataloader = DataLoader(dataset, batch_size=None, num_workers=self.config_file["cnp_settings"]["number_of_walkers"], prefetch_factor=2) 
        # write the feature mean and std from the training set to the config file

    def split_and_mixup_augment(self, filename, split_ratio, use_beta, condition_strings, mixup_ratio=0.):

        combined = [
            f"split_ratio='{split_ratio}'",
            f"condition='{condition_strings}'",
            f"use_beta={use_beta}",
            f"mixup_ratio={mixup_ratio}"
        ]

        with h5py.File(filename, "a") as f:  # Open in append mode

            # Check if split + mixup was already done with identical metadata
            if all(key in f for key in ("phi_train", "target_train", "training_metadata")):
                existing_metadata = [s.decode("utf-8") for s in f["training_metadata"][:]]
                if existing_metadata == combined:
                    #print("Skipping split and mixup augmentation — already exists with matching metadata.")
                    #test = 1
                    return
            
            phi = np.array(f[self.parameters["phi"]["key"]])  # Feature data
            target = np.array(f[self.parameters["target"]["key"]])  # Labels
            has_weights = "weights" in f  # Check if "weights" dataset exists
            weights = np.array(f["weights"]) if has_weights else None

            max_retries = 1000
            retry_count = 0
            nsignals = [0]
            while np.all(nsignals) == 0 is 0 and retry_count < max_retries:
                retry_count += 1
                (phi_train, phi_val, phi_test,
                target_train, target_val, target_test,
                weights_train, weights_val, weights_test) = self.split_data(phi, target, weights, split_ratio)
                nsignals = np.sum(target_train == 1, axis=0)

            phi_train_phase1 = None
            retry_count = 0
            while phi_train_phase1 is None and retry_count < max_retries:
                retry_count += 1

                if mixup_ratio > 0.:
                    all_indices = np.arange(target_train.shape[0])
                    all_indices_phase1 = np.random.choice(all_indices, size=int(len(all_indices) * mixup_ratio), replace=False)
                    all_indices_phase2 = np.setdiff1d(all_indices, all_indices_phase1)
                    
                    all_target_names = [label.decode("utf-8") for label in f[self.parameters["target"]["label_key"]][:]]
                    phi_train_phase1, target_train_phase1, weights_train_phase1 = self.apply_mixup(
                        phi_train[all_indices_phase1],
                        target_train[all_indices_phase1],
                        weights_train[all_indices_phase1] if has_weights else None,
                        all_target_names,
                        condition_strings,
                        use_beta) #apply mixup returns None, None, None if no signal samples found

                    if phi_train_phase1 is not None:
                        # Concatenate original data
                        if len(all_indices_phase2) > 0:
                            phi_train_phase1 = np.concatenate((phi_train[all_indices_phase2],phi_train_phase1), axis=0)
                            target_train_phase1 = np.concatenate((target_train[all_indices_phase2],target_train_phase1), axis=0)
                            if has_weights:
                                weights_train_phase1 = np.concatenate((weights_train[all_indices_phase2], weights_train_phase1), axis=0)
                        
                        # Shuffle everything using the same permutation
                        permutation = np.random.permutation(phi_train_phase1.shape[0])
                        phi_train_phase1 = phi_train_phase1[permutation]
                        target_train_phase1 = target_train_phase1[permutation]
                        if has_weights:
                            weights_train_phase1 = weights_train_phase1[permutation]
                else:
                    break  # Exit loop if split was successful
                    
            if phi_train_phase1 is None:
                target = np.array(f[self.parameters["target"]["key"]])  # Labels
                count_ones_per_column = np.sum(target == 1, axis=0)
                print("Number of label==1 per column:", count_ones_per_column)
                raise RuntimeError(f"Could not create training data with signal for mixup after {max_retries} retries in {filename}. Check your data and conditions.")
            # Store new datasets in the same file
            # Define data splits and corresponding variable names
            datasets = {
                "phi_train_raw": phi_train,
                "phi_train": phi_train_phase1 if mixup_ratio > 0. else phi_train,
                "phi_val": phi_val,
                "phi_test": phi_test,
                "target_train_raw": target_train,
                "target_train": target_train_phase1 if mixup_ratio > 0. else target_train,
                "target_val": target_val,
                "target_test": target_test,
            }

            if has_weights:
                datasets.update({
                    "weights_train_raw": weights_train,
                    "weights_train": weights_train_phase1 if mixup_ratio > 0. else weights_train,
                    "weights_val": weights_val,
                    "weights_test": weights_test,
                })

            # Write datasets to file (delete if they already exist)
            for name, data in datasets.items():
                if name in f:
                    del f[name]
                f.create_dataset(name, data=data, compression="gzip")
            
            if "training_metadata" in f:
                del f["training_metadata"]
            f.create_dataset("training_metadata", data=np.array(combined, dtype="S"))

    def split_data(self, phi, target, weights, split_ratio):
        # Shuffle indices
        N = phi.shape[0]
        indices = np.arange(N)
        np.random.seed(42)   # for reproducibility
        np.random.shuffle(indices)

        # Compute split sizes
        n_train = int(split_ratio[0] * N)
        n_val = int(split_ratio[1] * N)

        # Apply split
        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        phi_train, target_train = phi[train_idx], target[train_idx]
        phi_val, target_val = phi[val_idx], target[val_idx]
        phi_test, target_test = phi[test_idx], target[test_idx]

        if weights != None:
            weights_train = weights[train_idx]
            weights_val = weights[val_idx]
            weights_test = weights[test_idx]
        else:
            weights_train = weights_val = weights_test = None    
        
        return phi_train, phi_val, phi_test, target_train, target_val, target_test, weights_train, weights_val, weights_test
    
    def apply_mixup(self, phi, target, weights, all_target_names, condition_strings, use_beta):
        all_indices = np.arange(target.shape[0])
        # Convert conditions to apply on NumPy target array
        conditions = np.ones(target.shape[0], dtype=bool)  # Start with all True

        for cond_str in condition_strings:
            col_idx, cond_func = utils.parse_condition(cond_str, all_target_names)  # Get index and condition
            if np.ndim(target) > 1:
                conditions &= cond_func(target[:, col_idx])  # Apply condition to the correct dimension
            else:
                conditions &= cond_func(target[:])
        
        # Identify background (0) and signal (1) indices
        signal_indices = np.where(conditions)[0]
        if len(signal_indices) == 0:
            #print(f"Error: No signal samples found for conditions: {condition_strings}")
            return None, None, None
        
        # Background indices are those NOT in signal_indices
        background_indices = np.setdiff1d(all_indices, signal_indices)
        if len(background_indices) == 0 or len(signal_indices) == 0:
            #print("Dataset must contain both signal (1) and background (0) samples.")
            return None, None, None

        # Randomly pair each background sample with a signal sample
        sampled_background_indices = np.random.choice(background_indices, size=len(all_indices), replace=True)
        sampled_signal_indices = np.random.choice(signal_indices, size=len(sampled_background_indices), replace=True)

        # Generate mixup ratios
        mixup_margin = self.config_file["cnp_settings"]["training"]["phase1"]["mixup_margin"]
        if use_beta and isinstance(use_beta, (list, tuple)) and len(use_beta) == 2:
            if mixup_margin > 0.:
                # Use uniform distribution if margin > 0
                alpha = np.random.beta(1., 1., size=(len(sampled_background_indices), 1))
            else:
                alpha = np.random.beta(use_beta[0], use_beta[1], size=(len(sampled_background_indices), 1))
        else:
            alpha = np.random.beta(1., 1., size=(len(sampled_background_indices), 1))

        # Perform mixup augmentation
        phi_mixup= alpha * phi[sampled_signal_indices] + (1 - alpha) * phi[sampled_background_indices]
        
        # If weights are present, compute and concatenate
        weights_mixup = None
        if weights is not None:
            weights_mixup = weights.reshape(-1, 1) if weights.ndim == 1 else weights
            weights_mixup = alpha * weights_mixup[sampled_signal_indices] + (1 - alpha) * weights_mixup[sampled_background_indices]

        alpha = np.where(alpha >= 1. - mixup_margin, 1., np.where(alpha <= mixup_margin, 0., alpha))
        target_mixup = target.reshape(-1, 1) if target.ndim == 1 else target
        target_mixup = alpha * target_mixup[sampled_signal_indices] + (1 - alpha) * target_mixup[sampled_background_indices]

        return phi_mixup, target_mixup, weights_mixup


    def compute_feature_stats(self, use_indices=True):
        """
        Computes global statistics (mean/std or min/max) for normalization of features (theta + phi),
        using the feature key currently set in self.parameters['phi']['key'].

        Supports both z-score and min-max normalization, depending on:
            self.parameters['normalization'] = 'zscore' or 'minmax'

        Stores:
            For z-score:
                self.feature_mean: torch.Tensor of shape (1, D)
                self.feature_std:  torch.Tensor of shape (1, D)
            For min-max:
                self.feature_min: torch.Tensor of shape (1, D)
                self.feature_max: torch.Tensor of shape (1, D)
        """
        phi_key = self.parameters["phi"]["key"]
        normalization = self.use_normalization

        file_list = sorted([
            os.path.join(self.path_to_files, f)
            for f in os.listdir(self.path_to_files)
            if f.endswith(".h5")
        ])

        sum_features = None
        sum_squared = None
        total_count = 0

        global_min = None
        global_max = None

        for file in tqdm(file_list, desc="Computing global feature stats"):
            with h5py.File(file, "r") as f:
                if phi_key not in f:
                    print(f"Warning: '{phi_key}' not found in {file}. Skipping.")
                    continue

                phi = np.array(f[phi_key])
 
                if use_indices == True:
                    phi_indices = utils.read_selected_indices(file, self.parameters['phi'])
                    phi = phi[:, phi_indices]

                if use_indices == True and len(self.parameters['theta']['selected_labels']) > 0:
                    theta_indices = utils.read_selected_indices(file, self.parameters['theta'])
                    theta = np.array(f[self.parameters['theta']['key']])
                    theta = np.array(f[self.parameters['theta']['key']])[theta_indices]
                else:
                    theta = np.array(f[self.parameters['theta']['key']])

                if len(theta.shape) == 1:
                        theta = np.tile(theta, (phi.shape[0], 1))
                else:
                        theta = theta[:, theta_indices]

                if theta.shape[1] > 0: 
                    features = np.hstack([theta, phi])
                else:
                    features = phi

                if normalization == 'zscore':
                    if sum_features is None:
                        sum_features = np.sum(features, axis=0)
                        sum_squared = np.sum(features ** 2, axis=0)
                    else:
                        sum_features += np.sum(features, axis=0)
                        sum_squared += np.sum(features ** 2, axis=0)

                elif normalization == 'minmax':
                    if global_min is None:
                        global_min = np.min(features, axis=0)
                        global_max = np.max(features, axis=0)
                    else:
                        global_min = np.minimum(global_min, np.min(features, axis=0))
                        global_max = np.maximum(global_max, np.max(features, axis=0))

                total_count += features.shape[0]

        if total_count == 0:
            raise ValueError(f"No samples found using key '{phi_key}'")

        if normalization == 'zscore':
            mean = sum_features / total_count
            std = np.sqrt((sum_squared / total_count) - (mean ** 2) + 1e-8)

            self.feature_mean = torch.tensor(mean, dtype=torch.float32).unsqueeze(0)
            self.feature_std = torch.tensor(std, dtype=torch.float32).unsqueeze(0)

            print("Z-score feature mean/std computation completed.")

        elif normalization == 'minmax':
            # Store min as "mean"
            self.feature_mean = torch.tensor(global_min, dtype=torch.float32).unsqueeze(0)
            # Store (max - min) as "std"
            self.feature_std = torch.tensor(global_max - global_min, dtype=torch.float32).unsqueeze(0)

            print("Min-Max feature min/max computation completed.")

        else:
            raise ValueError(f"Unsupported normalization method: {normalization}")
    
    def format_batch_for_cnp(self,batch, context_is_subset=True):
        """
        Formats a batch into the query format required for CNP training with dynamic batch splitting.
        Parameters:
        - batch (torch.Tensor): Input batch of shape (batch_size, feature_dim).
        - total_batch_size (int): Expected full batch size (default: 3000).
        - context_ratio (float): Ratio of context points (default: 1/3).
        - target_ratio (float): Ratio of target points (default: 2/3).

        Returns:
        - CNPRegressionDescription(query=((batch_context_x, batch_context_y), batch_target_x), target_y=batch_target_y)
        """

        batch_size = batch.shape[0]  # Actual batch size (may be < 3000)
        
        # Dynamically compute num_context and num_target
        num_context = int(batch_size * self._context_ratio)
        num_target = batch_size - num_context  # Ensure it sums to batch_size
        
        # Shuffle the batch to ensure randomness
        batch = batch[torch.randperm(batch.shape[0])]
        
        # Split batch into input (X) and target (Y) features
        batch_x = batch[:,:self.feature_size]  # All features except last column (input features)
        # Z-score normalization for input features
        if self.use_normalization != False:
            batch_x = (batch_x - self.feature_mean) / (self.feature_std + 1e-8)  # Avoid division by zero

        batch_y = batch[:,self.feature_size:self.feature_size+self.target_size]   # Last column is the target (output values)

        if context_is_subset:
            # **Context is taken as the first num_context points from target**
            batch_target_x = batch_x  # Target is the entire batch
            batch_target_y = batch_y  # Target outputs are the entire batch

            batch_context_x = batch_target_x[:num_context]  # Context is a subset of target
            batch_context_y = batch_target_y[:num_context]  # Context outputs
        else:
            batch_context_x = batch_x[:num_context]  # Context inputs
            batch_context_y = batch_y[:num_context]  # Context outputs
            batch_target_x = batch_x[num_context:num_context + num_target]  # Target inputs
            batch_target_y = batch_y[num_context:num_context + num_target]  # Target outputs

        # Ensure y tensors have correct dimensions (convert from 1D to 2D if needed)
        batch_context_y = batch_context_y.view(-1, 1) if batch_context_y.ndim == 1 else batch_context_y
        batch_target_y = batch_target_y.view(-1, 1) if batch_target_y.ndim == 1 else batch_target_y
        
        if batch_context_x.dim() == 2:  # Convert from [N, D] → [1, N, D]
            batch_context_x = batch_context_x.unsqueeze(0)
        if batch_context_y.dim() == 2:  # Convert from [N, 1] → [1, N, 1]
            batch_context_y = batch_context_y.unsqueeze(0)

        if batch_target_x.dim() == 2:  # Convert from [N, D] → [1, N, D]
            batch_target_x = batch_target_x.unsqueeze(0)
        if batch_target_y.dim() == 2:  # Convert from [N, 1] → [1, N, 1]
            batch_target_y = batch_target_y.unsqueeze(0)
        # Construct the query tuple
        query = ((batch_context_x, batch_context_y), batch_target_x)
        
        # Return the properly formatted object
        return CNPRegressionDescription(query=query, target_y=batch_target_y)

    def get_dataloader(self):
        return self.dataloader
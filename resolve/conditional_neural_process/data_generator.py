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
        self.epoch_counter = 0  # Tracks row block
        self.total_batches = 0
        self.parameters = parameters
        self.phi_selected_indices= None
        self.theta_selected_indices= None
        self.target_selected_indices= None

        # List and sort all HDF5 files
        self.files = sorted([os.path.join(hdf5_dir, f) for f in os.listdir(hdf5_dir) if f.endswith(".h5")])
        self.num_files = len(self.files)
        self.dataset_size =0 
        # Total row cycles per file to complete an epoch
        self.nrows = self.get_max_number_of_rows()
        self.total_cycles_per_epoch = self.nrows // self.rows_per_file  # nrows / k rows per batch = c cycles per full dataset pass

    def shuffle_files(self):
        """Shuffle the file order at the start of each full dataset pass (epoch)."""
        random.shuffle(self.files)
        self.epoch_counter = 0  # Reset row counter

    def get_max_number_of_rows(self):
        max_rows = 0
        self.dataset_size = 0 
        for file in self.files:
            try:
                with h5py.File(file, "r") as hdf:
                        if self.parameters['target']['key'] in hdf:
                            num_rows = hdf[self.parameters['target']['key']].shape[0]
                            self.dataset_size += num_rows
                            # Update max row count if this file has more rows
                            if num_rows > max_rows:
                                max_rows = num_rows
            except (OSError, IOError) as e:
                print(f"Skipping corrupted or unreadable file: {file}")
                print(f"Reason: {e}")
                continue
            if num_rows==0:
                print(f"WARNING! {file} has row size 0. Either no data or target key doesn't match.")
        if max_rows == 0:
                raise ValueError("ERROR! Data is either empty or target key doesn't match.")
        return max_rows
    
    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return self.dataset_size

    def __iter__(self):
        #print(f"Starting structured HDF5 loading for row block {self.epoch_counter}...")
        self.total_batches = 0
        cycle_idx = 0
        used_rows = 0  # Track number of rows used

        while cycle_idx < self.total_cycles_per_epoch:
            for i in range(0, len(self.files), self.files_per_batch):  # Loop over file chunks
                if i == 0  and self.epoch_counter==0:
                    self.phi_selected_indices=utils.read_selected_indices(self.files[0],self.parameters['phi'])
                    if len(self.parameters['theta']['selected_labels'])> 0:
                        self.theta_selected_indices=utils.read_selected_indices(self.files[0],self.parameters['theta'])
                    if len(self.parameters['target']['selected_labels'])> 0 and "columns[" not in self.parameters['target']['selected_labels']:
                        self.target_selected_indices=utils.read_selected_indices(self.files[0],self.parameters['target'])
                    elif len(self.parameters['target']['selected_labels'])> 0 and "columns[" in self.parameters['target']['selected_labels']:
                        start, end = utils.parse_slice_string(self.parameters['target']['selected_labels'])
                        self.target_selected_indices=[np.arange(start,end)]
                    else:
                        self.target_selected_indices = 0

                batch = []
                selected_files = self.files[i:i + self.files_per_batch]

                # Select the next sequential k rows per file
                start_idx = self.epoch_counter * self.rows_per_file
                end_idx = start_idx + self.rows_per_file

                for j, file in enumerate(selected_files):
                    with h5py.File(file, "r") as hdf:
                        phi=hdf[self.parameters['phi']['key']][start_idx:end_idx, self.phi_selected_indices]
                        if self.theta_selected_indices != None:
                            if  len(hdf[self.parameters['theta']['key']][:].shape) == 1:
                                theta = hdf[self.parameters['theta']['key']][self.theta_selected_indices]
                                theta = np.tile(theta, (phi.shape[0], 1))
                            else:
                                theta = hdf[self.parameters['theta']['key']][start_idx:end_idx, self.theta_selected_indices]
                            features = np.hstack([theta, phi])
                        else:
                            features = phi

                        if np.ndim(hdf[self.parameters['target']['key']]) > 1:
                            target=hdf[self.parameters['target']['key']][start_idx:end_idx, self.target_selected_indices]
                        else:
                            target=hdf[self.parameters['target']['key']][start_idx:end_idx]
                            target=target.reshape(-1, 1)

                        # Stack rows from this file

                        file_data = np.hstack([features, target])

                        batch.extend(file_data.tolist())
                        used_rows += len(file_data)
                # end loop over single file
                # Yield batch of batch-size shuffled samples
                random.shuffle(batch)
                yield torch.tensor(batch, dtype=torch.float32)

                self.total_batches += 1
            # end loop over file chunk
            cycle_idx += 1

            # Move to next row block
            self.epoch_counter += 1
            self.total_batches += cycle_idx+1
            # end loop of row chunk after all files read in

            # If all files and rows (from k*i to k*(i+1)) are read, reshuffle files for the row block
            if self.epoch_counter >= self.total_cycles_per_epoch:
                self.shuffle_files()
                break

CNPRegressionDescription = collections.namedtuple(
    "CNPRegressionDescription", ("query", "target_y")
)

class DataGeneration(object):
    """
    """
    def __init__(
        self,
        mode,
        config_file,
        path_to_files,
        batch_size,
        files_per_batch
    ):
        self._context_ratio = config_file["cnp_settings"]["context_ratio"]
        self._batch_size = batch_size
        self.path_to_files = path_to_files
        self.dataloader="None"
        self.config_file=config_file
        self.files_per_batch = files_per_batch
        self.feature_mean = None
        self.feature_std = None
        self.use_normalization = config_file["feature_settings"]["use_normalization"]

        _phi_key="phi"
        _theta_key="theta"
        _target_key="target"
        self._names_theta=config_file["simulation_settings"]["theta_headers"]
        _names_phi=config_file["simulation_settings"]["phi_labels"]
        self._names_target =config_file["simulation_settings"]["target_headers"]
        self.feature_size,self.target_size=utils.get_feature_and_label_size(config_file)
        
        if not any(f.endswith(".h5") for f in os.listdir(path_to_files)):
            utils.convert_all_csv_to_hdf5(config_file)
        self.parameters={'phi': {'key': _phi_key,'label_key': "phi_labels",'selected_labels': _names_phi}, 
                        'theta': {'key': _theta_key,'label_key': "theta_headers",'selected_labels': self._names_theta}, 
                        'target': {'key': _target_key,'label_key': "target_headers",'selected_labels': self._names_target}}

        if mode == "training":
                signal_condition = config_file["simulation_settings"]["signal_condition"]
                files = sorted([os.path.join(path_to_files, f) for f in os.listdir(path_to_files) if f.endswith(".h5")])
                for file in tqdm(files, total=len(files), desc="Data Processing in Progress"):
                    self.split_and_mixup_augment(file,config_file["cnp_settings"]["split_ratio"] ,config_file["cnp_settings"]["use_beta"],signal_condition, config_file["cnp_settings"]["mixup_ratio"])
                self.parameters["phi"]["key"]="phi_train"
                self.parameters["target"]["key"]="target_train"
        elif mode == "validation":
                self.parameters["phi"]["key"]="phi_val"
                self.parameters["target"]["key"]="target_val"
        elif mode == "testing":
                self.parameters["phi"]["key"]="phi_test"
                self.parameters["target"]["key"]="target_test"
        elif mode == "prediction":
                self.parameters["phi"]["key"]="phi"
                self.parameters["target"]["key"]="target"

    def set_loader(self):
        if (self.feature_mean is None or self.feature_std is None) and self.use_normalization != False:
            self.compute_feature_stats()
            self.config_file["feature_settings"]["x_mean"] = self.feature_mean.numpy().tolist()
            self.config_file["feature_settings"]["x_std"] = self.feature_std.numpy().tolist()
            
        dataset = HDF5Dataset(self.path_to_files, self._batch_size, files_per_batch=self.files_per_batch, parameters=self.parameters)
        self.dataloader = DataLoader(dataset, batch_size=None, num_workers=self.config_file["cnp_settings"]["number_of_walkers"], prefetch_factor=2) 
        # write the feature mean and std from the training set to the config file


    def split_and_mixup_augment(self, filename, split_ratio, use_beta, condition_strings, mixup_ratio=0.):
        """
        Splits the dataset into train/validation/test sets and optionally applies mixup augmentation.

        This function reads feature, label, and optional weight data from an HDF5 file, splits it into
        60% training, 20% validation, and 20% test sets, and optionally augments the training set
        using mixup (a weighted combination of signal and background events).
        The signal events used in mixup are selected based on logical conditions applied to the
        label columns.

        Args:
            filename (str): Path to the HDF5 file containing datasets.
            use_beta (list or tuple or None): If provided, specifies the Beta distribution parameters
                `[alpha, beta]` for sampling mixup weights. If None, mixup weights are sampled uniformly.
            mixup_ratio (float): Fraction (between 0 and 1) of training samples to replace with mixup-augmented
                samples. Set to 0.0 to disable mixup.
            condition_strings (list of str): Logical conditions to define which events are considered "signal".
                Example: ['Class==1', 'Energy>0.5']

        Returns:
            None. The function modifies the input HDF5 file in-place by creating or overwriting the following datasets:
                - 'phi_train', 'phi_val', 'phi_test'
                - 'target_train', 'target_val', 'target_test'
                - 'weights_train', 'weights_val', 'weights_test' (if weights exist)
                - 'signal_condition' (records the applied condition strings)
        """
        combined = [
            f"split_ratio='{split_ratio}'"
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
                    return
            
            phi = np.array(f[self.parameters["phi"]["key"]])  # Feature data
            target = np.array(f[self.parameters["target"]["key"]])  # Labels
            has_weights = "weights" in f  # Check if "weights" dataset exists
            weights = np.array(f["weights"]) if has_weights else None

            splits = self.split_data(phi, target, weights, split_ratio)
            (phi_train, phi_val, phi_test,
            target_train, target_val, target_test,
            weights_train, weights_val, weights_test) = splits
            

            if mixup_ratio > 0.:
                # Identify background (0) and signal (1) indices
                background_indices = np.where(target_train == 0)[0]
                all_target_names = [label.decode("utf-8") for label in f[self.parameters["target"]["label_key"]][:]]

                # Function to parse condition strings dynamically
                def parse_condition(condition_str, columns):
                    """
                    Parses condition strings like 'BBH Events==1' or 'some name>=value'
                    and returns (column index, condition lambda).
                    """
                    # Supported operators, ordered by length to match longest first
                    operators = ['==', '!=', '>=', '<=', '>', '<']

                    # Try each operator and see if it's in the string
                    for op in operators:
                        if op in condition_str:
                            parts = condition_str.split(op)
                            if len(parts) != 2:
                                raise ValueError(f"Invalid condition format: {condition_str}")
                            column_name = parts[0].strip()
                            value_str = parts[1].strip()
                            break
                    else:
                        raise ValueError(f"No valid operator found in: {condition_str}")

                    if column_name not in columns:
                        raise ValueError(f"Column '{column_name}' not found in target!")

                    column_idx = columns.index(column_name)
                    
                    # Try to convert value to number
                    try:
                        value = float(value_str) if '.' in value_str else int(value_str)
                    except ValueError:
                        value = f'"{value_str}"'  # Quote string for eval

                    # Return column index and lambda condition
                    return column_idx, lambda x: eval(f"x {op} {value}", {"x": x})
                # Convert conditions to apply on NumPy target array
                conditions = np.ones(target_train.shape[0], dtype=bool)  # Start with all True

                for cond_str in condition_strings:
                    col_idx, cond_func = parse_condition(cond_str, all_target_names)  # Get index and condition

                    if np.ndim(target_train) > 1:
                        conditions &= cond_func(target_train[:, col_idx])  # Apply condition to the correct dimension
                    else:
                        conditions &= cond_func(target_train[:])
                
                # Find matching indices
                signal_indices = np.where(conditions)[0]
                if len(signal_indices) == 0:
                    raise ValueError(f"No signal samples found for conditions: {condition_strings} in file {filename}")
                
                # All indices in the dataset
                all_indices = np.arange(target_train.shape[0])
                all_indices_new = np.random.choice(all_indices, size=int(len(all_indices) * (1.-mixup_ratio)), replace=True)
                # Background indices are those NOT in signal_indices
                background_indices = np.setdiff1d(all_indices, signal_indices)
                background_indices = np.random.choice(background_indices, size=int(len(all_indices) * mixup_ratio), replace=True)

                if len(background_indices) == 0 or len(signal_indices) == 0:
                    raise ValueError("Dataset must contain both signal (1) and background (0) samples.")

                # Randomly pair each background sample with a signal sample
                sampled_signal_indices = np.random.choice(signal_indices, size=len(background_indices), replace=True)

                # Generate mixup ratios
                mixup_margin = self.config_file["cnp_settings"]["mixup_margin"]
                if use_beta and isinstance(use_beta, (list, tuple)) and len(use_beta) == 2:
                    if mixup_margin > 0.:
                        # Use uniform distribution if margin > 0
                        alpha = np.random.beta(1., 1., size=(len(background_indices), 1))
                    else:
                        alpha = np.random.beta(use_beta[0], use_beta[1], size=(len(background_indices), 1))
                else:
                    alpha = np.random.beta(1., 1., size=(len(background_indices), 1))

                # Perform mixup augmentation
                phi_train_mix = alpha * phi_train[sampled_signal_indices] + (1 - alpha) * phi_train[background_indices]
                
                # If weights are present, compute and concatenate
                if has_weights:
                    weights_train = weights_train.reshape(-1, 1) if weights_train.ndim == 1 else weights_train
                    weights_train_mix = alpha * weights_train[sampled_signal_indices] + (1 - alpha) * weights_train[background_indices]
                    weights_train = np.concatenate((weights_train[all_indices_new], weights_train_mix), axis=0)

                alpha = np.where(alpha >= 1. - mixup_margin, 1., np.where(alpha <= mixup_margin, 0., alpha))
                target_train = target_train.reshape(-1, 1) if target_train.ndim == 1 else target_train
                target_train_mix = alpha * target_train[sampled_signal_indices] + (1 - alpha) * target_train[background_indices]

                # Concatenate original data
                phi_train = np.concatenate((phi_train[all_indices_new], phi_train_mix), axis=0)
                target_train = np.concatenate((target_train[all_indices_new], target_train_mix), axis=0)

                

                # Shuffle everything using the same permutation
                permutation = np.random.permutation(phi_train.shape[0])
                phi_train = phi_train[permutation]
                target_train = target_train[permutation]
                if has_weights:
                    weights_train = weights_train[permutation]
            
            # Store new datasets in the same file
            # Define data splits and corresponding variable names
            datasets = {
                "phi_train": phi_train,
                "phi_val": phi_val,
                "phi_test": phi_test,
                "target_train": target_train,
                "target_val": target_val,
                "target_test": target_test,
            }

            if has_weights:
                datasets.update({
                    "weights_train": weights_train,
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
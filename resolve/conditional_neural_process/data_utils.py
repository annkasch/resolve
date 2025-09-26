import torch
import numpy as np
import os
from tqdm import tqdm
import h5py
from ..utilities import utilities as utils
utils.set_random_seed(42)

def mixup_augment_data(filename,use_beta,condition_strings, seed=42):
        """
        Augments an imbalanced dataset using the "mixup" method for HDF5 files.

        Each background event is combined with a randomly drawn signal event using a weighted sum.
        The ratio is drawn from either a uniform distribution or a beta distribution.

        Args:
            filename (str): Path to the HDF5 file.
            use_beta (list or None): Distribution from which the ratio is drawn.
                - `None`: Uniform distribution in [0,1].
                - `[z1, z2]`: Beta distribution B(z1, z2).
            config_file (dict): Preloaded YAML config dictionary.

        Returns:
            None: Updates the existing HDF5 file with new datasets.
        """
        np.random.seed(seed)  # Set the seed for reproducibility
        with h5py.File(filename, "a") as f:  # Open in append mode
            # Check if mixup datasets already exist
            if "phi_mixedup" in f and "target_mixedup" in f:
                if "signal_condition" in f:
                    existing_conditions = [s.decode("utf-8") for s in f["signal_condition"][:]]
                    if existing_conditions == condition_strings:
                        #print("skip mixup augmentation, already exists")
                        return
            phi = np.array(f["phi"])  # Feature data
            target = np.array(f["target"])  # Labels
            has_weights = "weights" in f  # Check if "weights" dataset exists
            weights = np.array(f["weights"]) if has_weights else None

            # Identify background (0) and signal (1) indices
            background_indices = np.where(target == 0)[0]
            all_target_names = f["target_headers"][:]
            all_target_names=[label.decode("utf-8") for label in all_target_names]

            # Function to parse condition strings dynamically
            '''
            def parse_condition(condition_str, columns):
                print(condition_str, columns)
                """Parses condition strings and returns (column index, condition lambda)."""
                match = re.match(r"(\S+)\s*(==|!=|<=|>=|<|>)\s*(\S+)", condition_str)
                if not match:
                    raise ValueError(f"Invalid condition format: {condition_str}")

                column_name, operator, value = match.groups()
                if column_name not in columns:
                    raise ValueError(f"Column {column_name} not found in target!")
                
                column_idx = columns.index(column_name)  # Get the column index

                # Convert condition string to a lambda function
                return column_idx, lambda x: eval(f"x {operator} {value}", {"x": x})
            '''

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
            conditions = np.ones(target.shape[0], dtype=bool)  # Start with all True

            for cond_str in condition_strings:
                col_idx, cond_func = parse_condition(cond_str, all_target_names)  # Get index and condition

                if np.ndim(target) > 1:
                    conditions &= cond_func(target[:, col_idx])  # Apply condition to the correct dimension
                else:
                    conditions &= cond_func(target[:])
            
            # Find matching indices
            signal_indices = np.where(conditions)[0]
            if len(signal_indices) == 0:
                ValueError(f"No signal samples found for conditions: {condition_strings} in file {filename}")
                return
            
            # All indices in the dataset
            all_indices = np.arange(target.shape[0])

            # Background indices are those NOT in signal_indices
            background_indices = np.setdiff1d(all_indices, signal_indices)
        
            if len(background_indices) == 0 or len(signal_indices) == 0:
                #print(f"Dataset must contain both signal (1) and background (0) samples. {filename}")
                #shutil.move(filename, './binary-black-hole/in/data/lf/v1.1/run/run/run/zero_signal')
                #return
                raise ValueError("Dataset must contain both signal (1) and background (0) samples.")

            # Randomly pair each background sample with a signal sample
            sampled_signal_indices = np.random.choice(signal_indices, size=len(background_indices), replace=True)

            # Generate mixup ratios
            if use_beta and isinstance(use_beta, (list, tuple)) and len(use_beta) == 2:
                alpha = np.random.beta(use_beta[0], use_beta[1], size=(len(background_indices), 1))
            else:
                alpha = np.random.uniform(0, 1, size=(len(background_indices), 1))

            # Perform mixup augmentation
            phi_mixedup = alpha * phi[sampled_signal_indices] + (1 - alpha) * phi[background_indices]
            target_mixedup = alpha * target[sampled_signal_indices] + (1 - alpha) * target[background_indices]

            # Apply mixup to weights if they exist
            weights_mixedup = None
            if has_weights:
                weights_mixedup = alpha * weights[sampled_signal_indices] + (1 - alpha) * weights[background_indices]

            # Store new datasets in the same file
            if "phi_mixedup" in f:
                del f["phi_mixedup"]
            f.create_dataset("phi_mixedup", data=phi_mixedup, compression="gzip")

            if "target_mixedup" in f:
                del f["target_mixedup"]
            f.create_dataset("target_mixedup", data=target_mixedup, compression="gzip")

            if has_weights:
                if "weights_mixedup" in f:
                    del f["weights_mixedup"]
                f.create_dataset("weights_mixedup", data=weights_mixedup, compression="gzip")
            
            if "signal_condition" in f:
                del f["signal_condition"]
            f.create_dataset("signal_condition", data=np.array(condition_strings, dtype="S"))

def mixup_augment_data_with_norm(filename,parameters,use_beta,condition_strings,feature_mean=None, feature_std=None, seed=42):
        """
        Augments an imbalanced dataset using the "mixup" method for HDF5 files.

        Each background event is combined with a randomly drawn signal event using a weighted sum.
        The ratio is drawn from either a uniform distribution or a beta distribution.

        Args:
            filename (str): Path to the HDF5 file.
            use_beta (list or None): Distribution from which the ratio is drawn.
                - `None`: Uniform distribution in [0,1].
                - `[z1, z2]`: Beta distribution B(z1, z2).
            config_file (dict): Preloaded YAML config dictionary.

        Returns:
            None: Updates the existing HDF5 file with new datasets.
        """
        np.random.seed(seed)  # Set the seed for reproducibility

        with h5py.File(filename, "a") as f:  # Open in append mode
            """
            # Check if mixup datasets already exist
            if "phi_mixedup" in f and "target_mixedup" in f:
                if "signal_condition" in f:
                    existing_conditions = [s.decode("utf-8") for s in f["signal_condition"][:]]
                    if existing_conditions == condition_strings:
                        if [s.decode("utf-8") for s in f["phi_labels_mixedup"][:]] == parameters['phi']['selected_labels']:
                            #print("skip mixup augmentation, already exists")
                            #return
            """
            #phi_selected_indices=utils.read_selected_indices(filename,parameters['phi'])
            #phi=f[parameters['phi']['key']][:, phi_selected_indices]
            phi=f[parameters['phi']['key']]

            #if feature_mean != None and feature_std is not None:
            #    ntheta=len(parameters['theta']['selected_labels'])
            #    mu = feature_mean.numpy()[:,ntheta:]
            #    std = feature_std.numpy()[:,ntheta:]
            #    phi = (phi - mu) / std

            target = np.array(f["target"])  # Labels
            has_weights = "weights" in f  # Check if "weights" dataset exists
            weights = np.array(f["weights"]) if has_weights else None

            # Identify background (0) and signal (1) indices
            background_indices = np.where(target == 0)[0]
            all_target_names = f["target_headers"][:]
            all_target_names=[label.decode("utf-8") for label in all_target_names]

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
            conditions = np.ones(target.shape[0], dtype=bool)  # Start with all True

            for cond_str in condition_strings:
                col_idx, cond_func = parse_condition(cond_str, all_target_names)  # Get index and condition

                if np.ndim(target) > 1:
                    conditions &= cond_func(target[:, col_idx])  # Apply condition to the correct dimension
                else:
                    conditions &= cond_func(target[:])
            
            # Find matching indices
            signal_indices = np.where(conditions)[0]
            if len(signal_indices) == 0:
                ValueError(f"No signal samples found for conditions: {condition_strings} in file {filename}")
                return
            
            # All indices in the dataset
            all_indices = np.arange(target.shape[0])

            # Background indices are those NOT in signal_indices
            background_indices = np.setdiff1d(all_indices, signal_indices)
        
            if len(background_indices) == 0 or len(signal_indices) == 0:
                #print(f"Dataset must contain both signal (1) and background (0) samples. {filename}")
                #shutil.move(filename, './binary-black-hole/in/data/lf/v1.1/run/run/run/zero_signal')
                #return
                raise ValueError("Dataset must contain both signal (1) and background (0) samples.")

            # Randomly pair each background sample with a signal sample
            sampled_signal_indices = np.random.choice(signal_indices, size=len(background_indices), replace=True)

            # Generate mixup ratios
            if use_beta and isinstance(use_beta, (list, tuple)) and len(use_beta) == 2:
                alpha = np.random.beta(use_beta[0], use_beta[1], size=(len(background_indices), 1))
            else:
                alpha = np.random.uniform(0, 1, size=(len(background_indices), 1))

            # Perform mixup augmentation
            phi_mixedup = alpha * phi[sampled_signal_indices] + (1 - alpha) * phi[background_indices]
            target_mixedup = alpha * target[sampled_signal_indices] + (1 - alpha) * target[background_indices]

            # Apply mixup to weights if they exist
            weights_mixedup = None
            if has_weights:
                weights_mixedup = alpha * weights[sampled_signal_indices] + (1 - alpha) * weights[background_indices]

            # Store new datasets in the same file
            try:
                if "phi_mixedup" in f:
                    del f["phi_mixedup"]
            except Exception as e:
                print(f"Warning: Could not delete 'phi_mixedup': {e}")
            f.create_dataset("phi_mixedup", data=phi_mixedup, compression="gzip")
            
            try:
                if "phi_labels_mixedup" in f:
                    del f["phi_labels_mixedup"]
            except Exception as e:
                print(f"Warning: Could not delete 'phi_labels_mixedup': {e}")
            #f.create_dataset("phi_labels_mixedup", data=parameters['phi']['selected_labels'], compression="gzip")

            if "target_mixedup" in f:
                del f["target_mixedup"]
            f.create_dataset("target_mixedup", data=target_mixedup, compression="gzip")

            if has_weights:
                if "weights_mixedup" in f:
                    del f["weights_mixedup"]
                f.create_dataset("weights_mixedup", data=weights_mixedup, compression="gzip")
            
            if "signal_condition" in f:
                del f["signal_condition"]
            f.create_dataset("signal_condition", data=np.array(condition_strings, dtype="S"))


def compute_feature_stats(path_to_files, parameters, normalization):

    """
    Computes normalization statistics for features (theta + phi) using the feature key currently set in parameters['phi']['key'].

    Parameters:
        normalization_mode (str): Either "zscore" or "minmax"

    Stores:
        - feature_mean: torch.Tensor(1, D)
        - feature_std:  torch.Tensor(1, D)
          where:
            - for z-score: mean and std
            - for min-max: min and (max - min)
    """
    phi_key = parameters['phi']['key']

    file_list = sorted([
        os.path.join(path_to_files, f)
        for f in os.listdir(path_to_files)
        if f.endswith(".h5")
    ])

    sum_features = None
    sum_squared = None
    min_features = None
    max_features = None
    total_count = 0

    for file in tqdm(file_list, desc="Computing global feature stats"):
        with h5py.File(file, "r") as f:
            if phi_key not in f:
                print(f"Warning: '{phi_key}' not found in {file}. Skipping.")
                continue

            # Read selected phi features
            phi_indices = utils.read_selected_indices(file, parameters['phi'])
            phi = np.array(f[phi_key][:, phi_indices])

            # Read and broadcast theta
            if parameters['theta']['selected_labels']:
                theta_indices = utils.read_selected_indices(file, parameters['theta'])
                theta_data = f[parameters['theta']['key']]
                if len(theta_data.shape) == 1:
                    theta = np.tile(theta_data[theta_indices], (phi.shape[0], 1))
                else:
                    theta = theta_data[:, theta_indices]
                features = np.hstack([theta, phi])
            else:
                features = phi

            # Initialize accumulators on first pass
            if sum_features is None:
                sum_features = np.sum(features, axis=0)
                sum_squared = np.sum(features ** 2, axis=0)
                min_features = np.min(features, axis=0)
                max_features = np.max(features, axis=0)
            else:
                sum_features += np.sum(features, axis=0)
                sum_squared += np.sum(features ** 2, axis=0)
                min_features = np.minimum(min_features, np.min(features, axis=0))
                max_features = np.maximum(max_features, np.max(features, axis=0))

            total_count += features.shape[0]

    if total_count == 0:
        raise ValueError(f"No samples found using key '{phi_key}'")

    if normalization == "zscore" or normalization == "standard":
        mean = sum_features / total_count
        std = np.sqrt((sum_squared / total_count) - (mean ** 2) + 1e-8)
    elif normalization == "minmax":
        mean = min_features
        std = max_features - min_features + 1e-8  # Add epsilon to avoid division by 0

    feature_mean = torch.tensor(mean, dtype=torch.float32).unsqueeze(0)
    feature_std = torch.tensor(std, dtype=torch.float32).unsqueeze(0)

    return feature_mean,feature_std
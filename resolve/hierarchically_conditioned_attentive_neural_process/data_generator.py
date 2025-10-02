import torch
import collections
import numpy as np
import os
#from imblearn.over_sampling import SMOTE
from collections import Counter
from tqdm import tqdm
import gc

from pathlib import Path

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
import h5py
import h5py
from torch.utils.data import DataLoader, IterableDataset
from ..utilities import utilities as utils
import math
from torch.utils.data import get_worker_info
utils.set_random_seed(42)
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict

class _LRUBytesBoundCache:
    def __init__(self, max_bytes=512 * 1024**2):  # ~512MB default
        self.max_bytes = max_bytes
        self.bytes = 0
        self.store = OrderedDict()

    def _nbytes(self, arr):
        # torch.Tensor or np.ndarray support
        if hasattr(arr, 'nbytes'):
            return int(arr.nbytes)
        if hasattr(arr, 'element_size') and hasattr(arr, 'nelement'):
            return int(arr.element_size() * arr.nelement())
        return 0

    def get(self, key):
        if key in self.store:
            val = self.store.pop(key)
            self.store[key] = val
            return val
        return None

    def put(self, key, value):
        nb = self._nbytes(value)
        if nb > self.max_bytes:
            return  # refuse single huge entries
        # evict until there is room
        while self.bytes + nb > self.max_bytes and self.store:
            _, ev = self.store.popitem(last=False)  # LRU
            self.bytes -= self._nbytes(ev)
        self.store[key] = value
        self.bytes += nb

class HDF5Dataset(IterableDataset):
    def __init__(self, files, batch_size=3000, files_per_batch=20, parameters=None, mode='train',enable_slice_cache=False, slice_cache_max_bytes=512*1024**2):
        super().__init__()
        self.files = files
        self.batch_size = batch_size
        self.files_per_batch = 1 if (mode == "predict" or mode == "test") else files_per_batch
        self._file_cache = {}
        self.enable_slice_cache = enable_slice_cache
        self._slice_cache = _LRUBytesBoundCache(slice_cache_max_bytes) if enable_slice_cache else None
        self.parameters = parameters
        self.mode = mode


        self.num_files = len(self.files)

        max_rows_per_file, self.nrows = utils.get_max_number_of_rows(
            self.files, self.parameters['target']['key']
        )
        if max_rows_per_file < self.batch_size:
            self.batch_size = max_rows_per_file
        self.rows_per_file = self.batch_size // files_per_batch
        self.total_cycles = math.ceil(max_rows_per_file / self.rows_per_file)

        self.phi_selected_indices = self.parameters['phi']['selected_indices']

        if self.parameters['theta']['selected_labels']:
            self.theta_selected_indices = self.parameters['theta']['selected_indices']
        else:
            self.theta_selected_indices = None

        labels = self.parameters['target']['selected_labels']
        if labels:
            if "columns[" in labels:
                start, end = utils.parse_slice_string(labels)
                self.target_selected_indices = np.arange(start, end)
            else:
                self.target_selected_indices = self.parameters['target']['selected_indices']
        else:
            self.target_selected_indices = None

        self._file_cache = {}

    def __len__(self):
        return self.nrows

    def _get_file_handle(self, file_path):
        # Lazy open per-worker file handles
        if file_path not in self._file_cache:
            self._file_cache[file_path] = h5py.File(file_path, "r")
        return self._file_cache[file_path]
    
    def _read_file(self, hdf, start_idx, end_idx, file_path=None):
        if self.enable_slice_cache and file_path is not None:
            key = (file_path, int(start_idx), int(end_idx))
            cached = self._slice_cache.get(key)
            if cached is not None:
                return cached

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
            target = target_data[start_idx:end_idx].reshape(-1, 1)

        out = np.hstack([features, target])

        if self.enable_slice_cache and file_path is not None:
            self._slice_cache.put(key, out)

        return out

    def _train_iter(self):
        rng = np.random.default_rng()
        worker_info = get_worker_info()

        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        all_cycle_indices = list(range(self.total_cycles))
        worker_cycle_indices = all_cycle_indices[worker_id::num_workers]

        for cycle_idx in worker_cycle_indices:
            file_indices = list(range(self.num_files))
            rng.shuffle(file_indices)

            for i in range(0, self.num_files, self.files_per_batch):
                selected_files = [self.files[j] for j in file_indices[i:i + self.files_per_batch]]

                start_idx = cycle_idx * self.rows_per_file
                end_idx = start_idx + self.rows_per_file
                
                batch = []
                for file_path in selected_files:
                    hdf = self._get_file_handle(file_path)
                    batch.append(self._read_file(hdf, start_idx, end_idx, file_path=file_path))

                batch = np.vstack(batch)
                rng.shuffle(batch)
                yield torch.tensor(batch, dtype=torch.float32)

    def _predict_iter(self):
        try:
            worker_info = get_worker_info()
            worker_id = worker_info.id if worker_info else 0
            num_workers = worker_info.num_workers if worker_info else 1

            self.subcycles_per_file = self.total_cycles  # total_cycles = rows_per_file per file

            # Split files among workers
            files_for_worker = self.files[worker_id::num_workers]

            for file_path in files_for_worker:
                hdf = self._get_file_handle(file_path)
                file_idx = self.files.index(file_path)

                for subcycle_idx in range(self.subcycles_per_file):
                    start_idx = subcycle_idx * self.rows_per_file
                    end_idx = start_idx + self.rows_per_file
                    batch = self._read_file(hdf, start_idx, end_idx, file_path=file_path)
                    file_completed = (subcycle_idx == self.subcycles_per_file - 1)

                    yield torch.tensor(batch, dtype=torch.float32), worker_id, file_idx, file_completed
                    if file_completed:
                        # Close the file handle after processing the last subcycle
                        hdf.close()
                        del self._file_cache[file_path]

        finally:
            self.close()
                

    def __iter__(self):
        if self.mode !='predict' and self.mode !='test':
            return self._train_iter()
        else:
            return self._predict_iter()
    
    def __del__(self):
        for f in list(self._file_cache.values()):
            try: f.close()
            except: pass
        self._file_cache.clear()
        
    def close(self):
        for f in list(self._file_cache.values()):
            try: f.close()
            except: pass
        self._file_cache.clear()
        gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

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

class DataGeneration:
    def __init__(self, mode, config_file, path_to_files, batch_size, files_per_batch):
        self.mode = mode
        self.config_file = config_file
        self.files = self._get_hdf5_files(Path(path_to_files))
        self._batch_size = batch_size
        self.files_per_batch = files_per_batch
        self.dataloader = None
        self._context_ratio = config_file["cnp_settings"]["context_ratio"]

        # ensure HDF5 exists (only once)
        if not any(self.files):
            utils.convert_all_csv_to_hdf5(config_file)

        # base parameter spec
        sim = config_file["simulation_settings"]
        
        self.parameters = {
            "phi":    {"key": "phi",    "label_key": "phi_labels",    "selected_labels": sim["phi_labels"],    "size": len(sim["phi_labels"]),      "selected_indices": None},
            "theta":  {"key": "theta",  "label_key": "theta_headers", "selected_labels": sim["theta_headers"],  "size": len(sim["theta_headers"]),  "selected_indices": None},
            "target": {"key": "target", "label_key": "target_headers","selected_labels": sim["target_headers"], "size": len(sim["target_headers"]), "selected_indices": None},
        }
        self.parameters["phi"]["selected_indices"] = utils.read_selected_indices(self.files[0],self.parameters["phi"])
        self.parameters["target"]["selected_indices"] = utils.read_selected_indices(self.files[0],self.parameters["target"])
        self.parameters["theta"]["selected_indices"] = utils.read_selected_indices(self.files[0],self.parameters["theta"])

        # optional Phase-1 preprocessing + signal-rate computation
        if mode == "train":
            self._phase1_preprocess_and_signal_rate()

    # ------------- helpers -------------
    def _get_hdf5_files(self, path_to_files):
        return sorted(str(p) for p in path_to_files.glob("*.h5"))

    def _configure_parameter_keys(self, mode="train"):
        """Map parameter keys depending on mode."""
        if mode != "predict":
            self.parameters["phi"]["key"] = f"{mode}/phi"
            self.parameters["target"]["key"] = f"{mode}/target"

    def _phase1_preprocess_and_signal_rate(self):
        """Split, (optionally) mixup-augment, then compute overall signal rate on selected targets."""

        cnp = self.config_file["cnp_settings"]
        split_ratio  = cnp["split_ratio"]
        use_beta     = cnp["training"]["phase1"]["use_beta"]
        mixup_ratio  = cnp["training"]["phase1"]["mixup_ratio"]
        signal_cond  = self.config_file["simulation_settings"]["signal_condition"]

        # run preprocessing once per file
        for f in tqdm(self.files, total=len(self.files), desc="Data Processing in Progress"):
            self.split_and_mixup_augment(f, split_ratio, use_beta, signal_cond, mixup_ratio)

        # after augmentation, training keys point to the altered sets
        self._configure_parameter_keys(mode="train")

        # compute signal rate with a stable accumulator (fewer openings, cached indices)
        total_pos = None
        total_cnt = 0
        selected_idx = None

        for f in self.files:
            with h5py.File(f, "r") as h5:
                tgt = h5[self.parameters["target"]["key"]][...]  # shape (N, K)
                tgt = tgt[:, self.parameters["target"]["selected_indices"]]                       # select columns once

                # count positives per selected target column
                pos = np.sum(tgt == 1, axis=0)                  # shape (K_sel,)
                n = tgt.shape[0]

                if total_pos is None:
                    total_pos = pos.astype(np.float64)
                else:
                    total_pos += pos
                total_cnt += n
            h5.close()

        # avoid division by zero
        self.signal_rate = (total_pos / max(total_cnt, 1.0)) if total_pos is not None else 0.0
        print("Overall signal rate in training data:", self.signal_rate)
    
    def set_loader(self, mode="train"):
        self._configure_parameter_keys(mode)
        dataset = HDF5Dataset(
            self.files,
            batch_size=self._batch_size,
            files_per_batch=self.files_per_batch,
            parameters=self.parameters,
            mode=mode
        )

        self.dataloader = DataLoader(
            dataset,
            batch_size=None,  # required for IterableDataset
            num_workers=self.config_file["cnp_settings"]["dataloader_number_of_workers"],
            prefetch_factor=self.config_file["cnp_settings"]["dataloader_prefetch_factor"],
            pin_memory=self.config_file["cnp_settings"]["dataloader_pin_memory"],
            persistent_workers=self.config_file["cnp_settings"]["dataloader_persistent_workers"]
        )
        return self.dataloader
    
    def close_loader(self):
        self.dataloader.dataset.close()
    
    def split_and_mixup_augment(self, filename, split_ratio, use_beta, condition_strings, mixup_ratio=0.):

        training_metadata = {
            "split_ratio": split_ratio,
            "condition": condition_strings,
            "use_beta": use_beta,
            "mixup_ratio": mixup_ratio
        }

        with h5py.File(filename, "a") as f:  # Open in append mode
            
            # Check if split + mixup was already done with identical metadata
            if all(key in f for key in ("train", "validate", "test", "train_meta")):
                grp = f["train_meta"]  # or whichever group you used
                attrs = dict(grp.attrs)       # convert to dict for easy comparison

                # check all expected keys exist and match
                for k, v in training_metadata.items():
                    if k in attrs:
                        break
                    elif not (attrs[k] == v).all() if hasattr(attrs[k], "all") else attrs[k] == v:
                        break

                return  # skip further processing
            
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

            if phi_train_phase1 is None and mixup_ratio > 0.:
                target = np.array(f[self.parameters["target"]["key"]])  # Labels
                count_ones_per_column = np.sum(target == 1, axis=0)
                print("Number of label==1 per column:", count_ones_per_column)
                raise RuntimeError(f"Could not create training data with signal for mixup after {max_retries} retries in {filename}. Check your data and conditions.")

            # Store new datasets in the same file
            # Define data splits and corresponding variable names
            datasets = {
                "train":{
                    "phi": phi_train,
                    "target": target_train
                },
                "train_mix":{
                    "phi": phi_train_phase1 if mixup_ratio > 0. else phi_train,
                    "target": target_train_phase1 if mixup_ratio > 0. else target_train,
                },
                "validate":
                {
                    "phi": phi_val,
                    "target": target_val,
                },
                "test":{
                    "phi": phi_test,
                    "target": target_test
                }
            }

            if has_weights:
                datasets.update({
                "train":{
                    "weights": weights_train
                },
                "train_mix":{                    
                    "weights": weights_train_phase1 if mixup_ratio > 0. else weights_train,
                },
                "validate":{
                    "weights": weights_val,
                },
                "test":{
                    "weights": weights_test,
                }})

            # Write datasets to file (delete if they already exist)
            for g, k in datasets.items():
                if g in f:
                    del f[g]
                grp = f.require_group(g)
                for n, data in k.items():
                    grp.create_dataset(n, data=data, compression="gzip")
            
            if "train_meta" in f:
                del f["train_meta"]
            grp = f.require_group("train_meta")
            for k, v in training_metadata.items():
                grp.attrs[k] = v
        f.close()


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

    def get_dataloader(self):
        return self.dataloader
    
class Normalizer:
    def __init__(self, method: str = "zscore", eps: float = 1e-8):
        self.method = method
        self.eps = eps
        self.scaler = StandardScaler(copy=False, with_mean=False, with_std=False)
    
    def fit_from_files(self, file_list, parameters, chunk_size=131072):
        """
        Stream over HDF5 files and compute normalization stats for [theta | phi].
        Supports:
        - self.method == 'zscore'  -> StandardScaler.partial_fit (streaming)
        - self.method == 'minmax'  -> stream gmin/gmax, then instantiate MinMaxScaler
        Side effects:
        - self.mean  : [1, D] torch.float32
        - self.scale : [1, D] torch.float32  (std for zscore, (max-min) for minmax)
        - self.sklearn_scaler : fitted sklearn scaler object (StandardScaler or MinMaxScaler)
        """
        if self.method == None:
            return
            
        self.scaler = StandardScaler(copy=False, with_mean=True, with_std=True)
        
        assert self.method in ("zscore", "minmax"), f"Unsupported method: {self.method}"

        phi_key   = parameters["phi"]["key"]
        theta_key = parameters["theta"]["key"]
        theta_sel = parameters["theta"]["selected_labels"]
        phi_sel   = parameters["phi"]["selected_labels"]

        gmin = None
        gmax = None
        D = None
        total_rows = 0


        with tqdm(total=len(file_list), desc="Computing global feature stats") as pbar:
            for path in file_list:
                if not path.endswith(".h5"):
                    pbar.update(1); continue
                try:
                    with h5py.File(path, "r") as f:
                        if phi_key not in f or theta_key not in f:
                            print(f"Warning: keys '{phi_key}' or '{theta_key}' not found in {path}. Skipping.")
                            pbar.update(1); continue

                        d_phi   = f[phi_key]
                        d_theta = f[theta_key]

                        # column selections
                        phi_idx   = parameters["phi"]["selected_indices"]   if phi_sel   else None
                        theta_idx = parameters["theta"]["selected_indices"] if theta_sel else None

                        # dims without loading all
                        phi_dim = d_phi.shape[1] if d_phi.ndim == 2 else 1
                        sel_phi_dim = (len(phi_idx) if phi_idx is not None and len(phi_idx) > 0 else phi_dim)

                        if d_theta.ndim == 1:
                            theta_raw = np.array(d_theta, dtype=np.float64, copy=False)
                            if theta_idx: theta_raw = theta_raw[theta_idx]
                            theta_cols = theta_raw.shape[0]
                            per_file_theta = theta_raw  # broadcast later
                            per_row_theta = None
                        else:
                            theta_cols = (len(theta_idx) if theta_idx else d_theta.shape[1])
                            per_file_theta = None
                            per_row_theta = d_theta

                        cur_D = sel_phi_dim + theta_cols
                        if D is None:
                            D = cur_D
                        elif cur_D != D:
                            raise ValueError(f"Feature dimension changed ({cur_D} vs {D}) in {path}")

                        # chunk loop
                        N = d_phi.shape[0]
                        for start in range(0, N, chunk_size):
                            stop = min(start + chunk_size, N)

                            # phi slice
                            if d_phi.ndim == 2:
                                phi_chunk = np.array(d_phi[start:stop, :], dtype=np.float64, copy=False)
                                if phi_idx is not None and len(phi_idx) > 0:
                                    phi_chunk = phi_chunk[:, phi_idx]
                            else:
                                phi_chunk = np.array(d_phi[start:stop], dtype=np.float64, copy=False).reshape(-1, 1)

                            # theta slice/broadcast
                            if per_file_theta is not None:
                                theta_chunk = np.tile(per_file_theta, (phi_chunk.shape[0], 1))
                            else:
                                theta_chunk = np.array(per_row_theta[start:stop, :], dtype=np.float64, copy=False)
                                if theta_idx: theta_chunk = theta_chunk[:, theta_idx]

                            feats = np.hstack((theta_chunk, phi_chunk)) if theta_cols > 0 else phi_chunk

                            # filter non-finite
                            mask = np.isfinite(feats).all(axis=1)
                            if not mask.all():
                                feats = feats[mask]
                            if feats.size == 0:
                                continue

                            if self.method == "zscore":
                                self.scaler.partial_fit(feats)  # streaming update

                            mn = feats.min(axis=0)
                            mx = feats.max(axis=0)
                            gmin = mn if gmin is None else np.minimum(gmin, mn)
                            gmax = mx if gmax is None else np.maximum(gmax, mx)

                            total_rows += feats.shape[0]

                    f.close()
                    pbar.update(1)
                except OSError as e:
                    print(f"Warning: failed to open {path}: {e}")
                    pbar.update(1); continue

        if total_rows == 0:
            raise ValueError(f"No samples found using key '{phi_key}'")
            
        self.data_min_ = gmin.astype(np.float64, copy=False)
        self.data_max_ = gmax.astype(np.float64, copy=False)
        self.data_range_ = (self.data_max_ - self.data_min_)
        
        # finalize MinMaxScaler from streamed min/max
        if self.method == "minmax":
            # avoid zero range to prevent div-by-zero; matches sklearn behavior
            zero = self.data_range_ == 0.0
            if np.any(zero):
                self.data_range_[zero] = 1.0

            self.scaler.n_features_in_ = self.data_min_.shape[0]
            self.scaler.mean_ = self.data_min_
            self.scaler.scale_ = self.data_range_
            self.scaler.n_samples_seen_= total_rows

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaler.transform(x)
    
    def inverse_transform(self, x: torch.Tensor) -> torch.Tensor:
        if self.scaler == None:
                raise RuntimeError("Normalizer not set")
        return self.scaler.inverse_transform(x)

    def to(self, device):
        if self.mean is not None: self.mean = self.mean.to(device)
        if self.scale is not None: self.scale = self.scale.to(device)
        return self

    def dump_to_config(self, cfg):
        cfg["feature_settings"]["x_mean"] = self.mean.cpu().numpy().tolist()
        cfg["feature_settings"]["x_std"]  = self.scale.cpu().numpy().tolist()
        return cfg

    def load_from_config(self, cfg):
        self.mean  = torch.tensor(cfg["feature_settings"]["x_mean"], dtype=torch.float32)
        self.scale = torch.tensor(cfg["feature_settings"]["x_std"],  dtype=torch.float32)
        return self

class BatchFormatter:
    def __init__(self, parameter, context_ratio: float, normalizer: Normalizer):
        self.theta_size = parameter["theta"]["size"]
        self.F = self.theta_size + parameter["phi"]["size"]
        self.T = parameter["target"]["size"]
        self.context_ratio = context_ratio
        self.norm = normalizer
        self.x_query = None

    def __call__(self, batch: torch.Tensor, context_is_subset: bool = True):
        # split X/Y
        x = batch[:, :self.F]
        y = batch[:, self.F:self.F+self.T]

        B = x.shape[0]
        n_ctx = int(B * self.context_ratio)
        
        if context_is_subset:
            x_ctx, y_ctx = x[:n_ctx], y[:n_ctx]
            self.x_query, y_tgt = x, y
            
        else:
            x_ctx, y_ctx = x[:n_ctx], y[:n_ctx]
            self.x_query, y_tgt = x[n_ctx:], y[n_ctx:]
        
        x_ctx = torch.tensor(self.norm.transform(x_ctx), dtype=torch.float32)
        x_tgt = torch.tensor(self.norm.transform(self.x_query), dtype=torch.float32)

        # shape them and slice theta/phi
        def ensure_3d(a): return a.unsqueeze(0) if a.dim()==2 else a
        x_ctx, y_ctx, x_tgt, y_tgt = map(ensure_3d, (x_ctx, y_ctx, x_tgt, y_tgt))

        return BatchCollection(
            context=ContextSet(theta=x_ctx[:, :, :self.theta_size], phi=x_ctx[:, :, self.theta_size:], y=y_ctx),
            query=QuerySet(theta=x_tgt[:, :, :self.theta_size], phi=x_tgt[:, :, self.theta_size:]),
            target_y=y_tgt,
        )
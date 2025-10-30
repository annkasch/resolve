import os
import math
from tokenize import group
from turtle import mode, update
import h5py
import pandas as pd
import numpy as np
from resolve.helpers import splitter
import torch
from torch.utils.data import IterableDataset, get_worker_info
from typing import List, Optional, Sequence, Tuple, Dict, Union
from sklearn.model_selection import train_test_split
from resolve.utilities import utilities as utils
import collections
from resolve.helpers.normalizer import Normalizer
import operator
import functools
from resolve.helpers.sampler import Sampler
from resolve.helpers.splitter import Splitter

ContextSet = collections.namedtuple("ContextSet", ("theta", "phi", "y"))
QuerySet   = collections.namedtuple("QuerySet",   ("theta", "phi"))

BatchCollection = collections.namedtuple(
    "BatchCollection",
    ("context", "query", "target_y")
)

class InMemoryIterableData(IterableDataset):
    def __init__(self, files: Sequence[str], batch_size: int = 1000, shuffle: bool = "global",
                 seed: int = 42, as_float32: bool = True, device: Optional[torch.device] = None,
                 parameter_config: Dict = None, dataset_config: Dict = None, positive_condition: Optional[List]=None,
                 normalizer: Optional[Normalizer]=Normalizer(), mode: Optional[str] = "train") -> None:
        super().__init__()
        
        self.files, self.batch_size, self.shuffle = list(files), int(batch_size), shuffle

        self.seed, self.as_float32, self.device = int(seed), bool(as_float32), device
        self.parameter_config, self.dataset_config = (parameter_config or {}), dataset_config
        
        self.mode = mode
        self._normalizer = normalizer
        self.sampler = Sampler(self.batch_size, positive_condition, shuffle=self.shuffle, seed=self.seed, device=self.device)

        # load all data into memory
        theta, phi, y, fidx = self._load_data_to_mem(self.files, self.as_float32, self.parameter_config)
        
        self.data = self._set_data(theta, phi, y, fidx)
        
        # get positive indices
        pos_mask = self.sampler.get_positive_indices(y)
        positive_ratio_data = pos_mask.sum(dim=0)/y.shape[0]
        print("positives ratio ", positive_ratio_data)
        self.state = None
        self.set_batch_schedule(target_pos_frac=self.dataset_config.get("positive_ratio_train", None), max_pos_reuse_per_epoch = self.dataset_config.get("max_positive_reuse",0.), sticky_frac = 0.25, seed=self.dataset_config.get("seed",12345))

    def _set_data(self, theta: torch.Tensor, phi: torch.Tensor, y: torch.Tensor, fidx: torch.Tensor):
        data = {}
        if self.mode == "train":
            # apply normalization
            self._normalizer = Normalizer(self.dataset_config.get("use_feature_normalization", None))
            theta, phi = self._normalizer.fit_transform_as_f32(self.as_float32, theta=theta, phi=phi)

            def make_empty_like(*tensors):
                return [torch.empty_like(t) for t in tensors]

            theta_test, phi_test, y_test, fidx_test = make_empty_like(theta, phi, y, fidx)
            theta_val,  phi_val,  y_val, fidx_val  = make_empty_like(theta, phi, y, fidx)

            # split into training, validation and testing data
            val_size = 1 - self.dataset_config.get('train_ratio',0.6)
            splitter = Splitter(self.shuffle, seed=42)
            if val_size > 0.:
                theta, theta_val, phi, phi_val, y, y_val, fidx, fidx_val = splitter.train_test_split(theta, phi, y, fidx, groups=theta, test_size=val_size)
                
                test_size = self.dataset_config.get('test_ratio',0.2)/val_size
                if test_size > 0. :
                    theta_val, theta_test, phi_val, phi_test, y_val, y_test, fidx_val, fidx_test = splitter.train_test_split(theta_val, phi_val, y_val, fidx_val, groups=theta_val, test_size=test_size)
            
            if self.dataset_config and self.dataset_config.get('mixup_ratio', 0.) > 0.0:

                theta, phi, y, fidx = self.sampler.mix_by_file_chunks(
                        theta, phi, y, fidx,self.dataset_config.get('mixup_ratio'),
                        use_beta=self.dataset_config.get('use_beta', None),
                        margin=float(self.dataset_config.get('mixup_margin', 0.0))
                    )


            data = {
                "train": {"theta": theta, "phi": phi, "y": y, "file_indices": fidx, "batches": None},
                "test": {"theta": theta_test, "phi": phi_test, "y": y_test, "file_indices": fidx_test, "batches": None},
                "validate": {"theta": theta_val, "phi": phi_val, "y": y_val, "file_indices": fidx_val, "batches": None}
            }
        elif self.mode == "test":
            theta = self._normalizer.transform(x=theta, feature_grp="theta")
            phi = self._normalizer.transform(x=phi, feature_grp="phi")
            if self.as_float32:
                theta = theta.float(); phi = phi.float()
            data = {
                "test": {"theta": theta.contiguous(), "phi": phi.contiguous(), "y": y, "file_indices": fidx, "batches": None},
            }
        elif self.mode == "predict":
            theta = self._normalizer.transform(x=theta, feature_grp="theta")
            phi = self._normalizer.transform(x=phi, feature_grp="phi")
            if self.as_float32:
                theta = theta.float(); phi = phi.float()
            data = {
                "predict": {"theta": theta.contiguous(), "phi": phi.contiguous(), "y": y, "file_indices": fidx, "batches": None},
            }

        return data
    
    def set_batch_schedule(self,
        target_pos_frac: float,
        max_pos_reuse_per_epoch: int = 0,     # 0 => no reuse; >0 => cap per epoch
        sticky_frac: float = 0.25,            # keep 25% of last epoch's negs
        seed: int | None = None,          # reproducible positive order
    ):

        # train sets the batch size and needs to be processed first
        if target_pos_frac != None:
            """
            self.data["train"]["batches"], self.state, self.meta = self.sampler.build_batches_with_target_pos_frac(
                self.data["train"]["theta"],
                self.data["train"]["y"],
                target_pos_frac=target_pos_frac,
                max_pos_reuse_per_epoch=max_pos_reuse_per_epoch,   # cap reuse; set 0 for no reuse
                sticky_frac=sticky_frac,
                unused_neg_subset=self.state
            )
            """
            self.data["train"]["batches"], self.state, self.meta = self.sampler._plan_batches(
                y=self.data["train"]["y"],
                target_pos_frac=target_pos_frac,
                max_pos_reuse_per_epoch=max_pos_reuse_per_epoch,   # cap reuse; set 0 for no reuse
                sticky_frac=sticky_frac,
                last_neg_subset=self.state,
                seed=seed
            )
        else:
            self.data["train"]["batches"], self.state, self.meta = self.sampler.build_batches(self.data["train"]["phi"].shape[0])
            
    @staticmethod
    def _read_in_from_file(file_path: str, parameter_config: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
        if file_path.endswith(('.h5', '.hdf5')):
            with h5py.File(file_path, 'r') as hdf:
                phi = hdf[parameter_config['phi']['key']][:,parameter_config['phi']['selected_indices']]
                
                theta = hdf[parameter_config['theta']['key']]
                
                if len(parameter_config['theta']['selected_indices']) != 0:
                    if theta.ndim == 1:
                        theta_vec = theta[parameter_config['theta']['selected_indices']]             # (T,)
                        # broadcast, then copy once during final assembly
                        theta = torch.from_numpy(theta_vec).unsqueeze(0).expand(phi.shape[0], -1)
                    else:
                        theta = theta[:, parameter_config['theta']['selected_indices']]
                else:
                    theta = torch.from_numpy(theta)

                tgt_ds = hdf[parameter_config['target']['key']]
                if tgt_ds.ndim > 1 and parameter_config['target']['selected_indices'] != None:
                    y = tgt_ds[:, parameter_config['target']['selected_indices']]
                else:
                    y = tgt_ds[:].reshape(-1, 1)

            phi = torch.from_numpy(phi)
            y = torch.from_numpy(y)

        elif file_path.endswith('.csv'):
            # --- CSV reading using column names (selected_labels) ---
            df = pd.read_csv(file_path)

            def select_labels(df: pd.DataFrame, labels: Union[str, List[str]]) -> pd.DataFrame:
                """Select one or multiple columns by name."""
                if isinstance(labels, str):
                    return df[[labels]]
                elif isinstance(labels, list):
                    return df[labels]
                else:
                    raise ValueError(f"Invalid label type: {type(labels)}")

            # Extract φ, θ, and y by column labels
            phi = select_labels(df, parameter_config['phi']['selected_labels'])
            theta = select_labels(df, parameter_config['theta']['selected_labels'])
            y = select_labels(df, parameter_config['target']['selected_labels'])

            # Convert to torch tensors
            phi = torch.tensor(phi.values, dtype=torch.float32)
            theta = torch.tensor(theta.values, dtype=torch.float32)
            y = torch.tensor(y.values, dtype=torch.float32)

            # Ensure y has shape (N, 1)
            if y.ndim == 1:
                y = y.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        return theta, phi, y

    def _load_data_to_mem(self, files: Sequence[str], as_f32: bool, cfg: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Thetas, Phis, ys, file_inds = [], [], [], []
        for i, fp in enumerate(files):
            if not os.path.exists(fp): raise FileNotFoundError(fp)
            Thetai, Phii, yi = self._read_in_from_file(fp, cfg)
            Thetas.append(Thetai); Phis.append(Phii); ys.append(yi)
            file_inds.append(torch.full((Phii.size(0),), i, dtype=torch.long))
        Theta, Phi, y = torch.cat(Thetas, 0), torch.cat(Phis, 0), torch.cat(ys, 0)
        file_index = torch.cat(file_inds, 0)
        if as_f32:
            Theta = Theta.float(); Phi = Phi.float(); y = y.float() if y.dtype == torch.float64 else y
        return Theta.contiguous(), Phi.contiguous(), y.contiguous(), file_index

    def set_mode(self, mode):
        self.mode = mode
    
    def set_normalizer(self, normalizer: Normalizer):
        """Attach a Normalizer instance to this object."""
        self._normalizer = normalizer
    
    def set_normalizer(self, method: str):
        """Attach a Normalizer instance to this object."""
        self._normalizer = Normalizer(method)



    def _compute_worker_slice(self, n: int) -> Tuple[int, int]:
        info = get_worker_info()
        if info is None: return 0, n
        per = int(math.ceil(n / info.num_workers)); s = info.id * per; e = min(s + per, n); return s, e

    def __iter__(self):
        if self.mode == "predict":
            return self._predict_iter()
        else:
            return self._train_iter()

    def _train_iter(self):
        """Iterator for train/validate/test. Uses precomputed batch-index plans if present."""
        # tensors
        store = self.data[self.mode]
        phi   = store["phi"]
        theta = store.get("theta")
        y     = store.get("y")
        dev   = self.device

        batches = store.get("batches", None)
        total_batches = len(batches)
        b_start, b_end = self._compute_worker_slice(total_batches)  # reuse same helper; it just slices a range
        if b_start >= b_end:
            return iter(())

        for b in range(b_start, b_end):
            idx = batches[b]  # 1D LongTensor of indices for this batch
            phib = phi.index_select(0, idx)
            thetab = theta.index_select(0, idx) if theta is not None else None
            yb     = y.index_select(0, idx)     if y     is not None else None
            if dev is not None:
                phib   = phib.to(dev, non_blocking=True)
                if thetab is not None: thetab = thetab.to(dev, non_blocking=True)
                if yb is not None:     yb     = yb.to(dev, non_blocking=True)
            yield self._format_batch(thetab, phib, yb)

    def _predict_iter(self):
        """Iterator for prediction mode where we process one file at a time from memory."""
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        # Get data for current mode
        theta = self.data[self.mode]["theta"]
        phi = self.data[self.mode]["phi"]
        y = self.data[self.mode]["y"]
        file_indices = self.data[self.mode]["file_indices"]

        # Get unique file indices
        unique_files = torch.unique(file_indices)
        
        # Split files among workers
        files_for_worker = unique_files[worker_id::num_workers]

        for file_idx in files_for_worker:
            # Get mask for current file
            file_mask = (file_indices == file_idx)

            # Get data for current file
            file_theta = theta[file_mask]
            file_phi = phi[file_mask]
            file_y = y[file_mask]

            # Process file in batches
            n = file_phi.shape[0]
            for start_idx in range(0, n, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n)
                
                # Extract batch
                theta_batch = file_theta[start_idx:end_idx]
                phi_batch = file_phi[start_idx:end_idx]
                y_batch = file_y[start_idx:end_idx]

                # Move to device if needed
                if self.device is not None:
                    theta_batch = theta_batch.to(self.device, non_blocking=True)
                    phi_batch = phi_batch.to(self.device, non_blocking=True)
                    y_batch = y_batch.to(self.device, non_blocking=True)

                # Format and yield batch
                batch = self._format_batch(theta_batch, phi_batch, y_batch)
                yield batch, file_idx.item(), end_idx >= n
    
    def _format_batch(self, theta, phi, y):

        n_ctx = int(phi.shape[0] * self.dataset_config.get("context_ratio", 0.33))
        
        if self.dataset_config.get("context_is_subset", True):
            theta_ctx, phi_ctx, y_ctx = theta[:n_ctx], phi[:n_ctx], y[:n_ctx]
            self.theta_query, self.phi_query, y_tgt = theta, phi, y
            
        else:
            theta_ctx, phi_ctx, y_ctx = phi[:n_ctx], y[:n_ctx]
            self.theta_query, self.phi_query, y_tgt = theta[n_ctx:], phi[n_ctx:], y[n_ctx:]

        def ensure_3d(a): return a.unsqueeze(0) if a.dim()==2 else a
        theta_ctx, phi_ctx, y_ctx, self.theta_query, self.phi_query, y_tgt = map(ensure_3d, (theta_ctx, phi_ctx, y_ctx, self.theta_query, self.phi_query, y_tgt))
        
        return BatchCollection(
            context=ContextSet(theta=theta_ctx, phi=phi_ctx, y=y_ctx),
            query=QuerySet(theta=self.theta_query, phi=self.phi_query),
            target_y=y_tgt,
        )
    
    def get(self, i: int, dev=None):
        """
        Return the i-th yielded batch for THIS worker under the current epoch/seed.
        Supports negative i (e.g., -1 is the last batch of this worker's shard).
        """
        # pull tensors for current mode
        mode = self.mode
        phi   = self.data[mode]["phi"]
        theta = self.data[mode].get("theta", None)
        y     = self.data[mode].get("y", None)

        idx = self.data[model]["batches"][i]
        theta_batch, phi_batch, y_batch = theta.index_select(0, idx), phi.index_select(0, idx), y.index_select(0, idx)
        batch = self._format_batch(theta_batch, phi_batch, y_batch)
        return batch


    def close(self):
        """Delete all tensors and arrays from memory to free up resources."""
        # Clear main data dictionary
        if hasattr(self, 'data'):
            for mode in self.data:
                for key in self.data[mode]:
                    self.data[mode][key] = None
            self.data = None
        
        # Clear unused data
        for attr in ['_theta_unused', '_phi_unused', '_y_unused', '_fidx_unused']:
            if hasattr(self, attr):
                setattr(self, attr, None)
        
        # Clear normalizer
        if hasattr(self, '_normalizer'):
            self._normalizer = None
        
        # Clear other attributes that might hold data
        self.files = None

    def __len__(self) -> int: 
        if self.data[self.mode]["batches"]!=None:
            return len(self.data[self.mode]["batches"])
        else:
            return int(math.ceil(self.data[self.mode]["phi"].shape[0]/self.batch_size))

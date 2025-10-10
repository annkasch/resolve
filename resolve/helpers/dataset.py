import os
import math
import h5py
import torch
from torch.utils.data import IterableDataset, get_worker_info
from typing import List, Optional, Sequence, Tuple, Dict
from sklearn.model_selection import train_test_split
import collections
from resolve.helpers.normalizer import Normalizer

ContextSet = collections.namedtuple("ContextSet", ("theta", "phi", "y"))
QuerySet   = collections.namedtuple("QuerySet",   ("theta", "phi"))

BatchCollection = collections.namedtuple(
    "BatchCollection",
    ("context", "query", "target_y")
)

class InMemoryIterableData(IterableDataset):
    """Load all HDF5 rows into memory, reshuffle each epoch, shard across workers.
    Optional: build a balanced USED/UNUSED split once from (X,y)."""
    def __init__(self, files: Sequence[str], batch_size: int = 1000, shuffle: bool = True,
                 seed: int = 42, as_float32: bool = True, device: Optional[torch.device] = None,
                 parameter_config: Dict = None, dataset_config: Dict = None, positive_function: Optional[List]=None,
                 normalizer: Optional[Normalizer]=Normalizer(), mode: Optional[str] = "train") -> None:
        super().__init__()
        
        self.files, self.batch_size, self.shuffle = list(files), int(batch_size), bool(shuffle)
        self.seed, self.as_float32, self.device = int(seed), bool(as_float32), device
        self.parameter_config, self.dataset_config = (parameter_config or {}), dataset_config
        
        self.mode = mode
        self._normalizer = normalizer

        # load all data into memory
        theta, phi, y, fidx = self._load_all(self.files, self.as_float32, self.parameter_config, positive_function)
        # get positive indices
        pos_mask = self._get_positive_indices(y,positive_function)
        self.positive_ratio_data = pos_mask.sum(dim=0)/y.shape[0]
        print("positives ratio ", self.positive_ratio_data)
        
        self.data = self._set_data(theta, phi, y, fidx, pos_mask, positive_function)

    
    def _set_data(self, theta: torch.Tensor, phi: torch.Tensor, y: torch.Tensor, fidx: torch.Tensor, pos_mask, positive_function: Optional[List]=None):
        data = {}
        if self.mode == "train":
            # apply normalization
            self._normalizer = Normalizer(self.dataset_config.get("use_feature_normalization", None))
            theta, phi = self.fit_transform(self.as_float32, theta=theta, phi=phi)

            # split into training, validation and testing data
            test_size = 1 - self.dataset_config.get('train_ratio',0.6)
            theta_train, theta_test, phi_train, phi_test, y_train, y_test, fidx_train, fidx_test, pos_mask_train, _ = train_test_split(theta, phi, y, fidx, pos_mask, test_size=test_size, random_state=42)
            test_size = self.dataset_config.get('val_ratio',0.2)/test_size
            theta_val, theta_test, phi_val, phi_test, y_val, y_test, fidx_val, fidx_test = train_test_split(theta_test, phi_test, y_test, fidx_test, test_size=test_size, random_state=42)

            if self.dataset_config and self.dataset_config.get('mixup_ratio', 0.) > 0.0:
                theta_train, phi_train, y_train, fidx_train = self._mix_by_file_chunks(
                        theta_train, phi_train, y_train, fidx_train,
                        positive_fn=positive_function,
                        use_beta=self.dataset_config.get('use_beta', None),
                        margin=float(self.dataset_config.get('mixup_margin', 0.0)),
                        seed=int(self.dataset_config.get('seed', 12345))
                    )
            
            # apply positives oversampling if required
            if  self.dataset_config.get('positive_ratio_train', None) and self.dataset_config.get('mixup_ratio', 0.)==0.:
                theta_train, phi_train, y_train, self._theta_unused, self._phi_unused, self._y_unused, fidx_train, self._fidx_unused = self._sample_balanced_dataset(
                    theta_train, phi_train, y_train, fidx_train, pos_mask=pos_mask_train,
                    ratio=float(self.dataset_config.get('positive_ratio_train', None)),
                    used_size=self.dataset_config.get('used_size'),
                    replace=bool(self.dataset_config.get('with_replacement', False)),
                    seed=int(self.dataset_config.get('seed', 12345)),
                )
            else:
                theta_train, phi_train, y_train, fidx_train, self._theta_unused, self._phi_unused, self._y_unused, self._fidx_unused = theta_train, phi_train, y_train, fidx_train, None, None, None, None
            
            data = {
                "train": {"theta": theta_train, "phi": phi_train, "y": y_train, "file_indices": fidx_train},
                "validate": {"theta": theta_test, "phi": phi_test, "y": y_test, "file_indices": fidx_test},
                "test": {"theta": theta_val, "phi": phi_val, "y": y_val, "file_indices": fidx_val}
            }
        elif self.mode == "predict":
            theta = self._normalizer.transform(x=theta, feature_grp="theta")
            phi = self._normalizer.transform(x=phi, feature_grp="phi")
            if self.as_float32:
                theta = theta.float(); phi = phi.float()
            data = {
                "predict": {"theta": theta.contiguous(), "phi": phi.contiguous(), "y": y, "file_indices": fidx},
            }

        return data
        

    @staticmethod
    def _read_one(file_path: str, parameter_config: Dict) -> Tuple[torch.Tensor, torch.Tensor]:
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

        return theta, phi, y

    def _load_all(self, files: Sequence[str], as_f32: bool, cfg: Dict, positive_function: Optional[List]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Thetas, Phis, ys, file_inds = [], [], [], []
        for i, fp in enumerate(files):
            if not os.path.exists(fp): raise FileNotFoundError(fp)
            Thetai, Phii, yi = self._read_one(fp, cfg)
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

    def fit_transform(self, as_f32,**feature_groups):
        """
        Fit and transform multiple feature groups (e.g., theta, phi).
        
        Example:
            theta, phi = self.fit_and_transform(theta=theta, phi=phi)
        """
        if self._normalizer is None:
            raise ValueError("Normalizer not set")

        transformed = {}
        for name, data in feature_groups.items():
            transformed[name] = self._normalizer.fit_transform(data, name)
            if as_f32:
                transformed[name] = transformed[name].float();
                transformed[name].contiguous()
        return tuple(transformed.values())

    @staticmethod
    def _get_positive_indices(y: torch.Tensor, positive_fn) -> Tuple[torch.Tensor, torch.Tensor]:
        y2 = y if y.ndim > 1 else y.unsqueeze(1)

        if callable(positive_fn):
            pos = torch.as_tensor(positive_fn(y2), dtype=torch.bool)
        else:
            ms = [torch.as_tensor(fn(y2[:,i]), dtype=torch.bool) for i, fn in enumerate(positive_fn) if fn is not None]
            pos = ms[0].clone(); [pos.__ior__(m) for m in ms[1:]]

        return pos

    @staticmethod
    def _sample_balanced_dataset(theta: torch.Tensor, phi: torch.Tensor, y: torch.Tensor, file_index: Optional[torch.Tensor]=None, pos_mask: torch.Tensor=None, *, ratio: float,
                  used_size: Optional[int] = None, replace: bool = False, seed: int = 12345) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        if not (0.0 < ratio < 1.0): raise ValueError('positive_ratio in (0,1)')
        
        pos_idx, neg_idx = pos_mask.nonzero(as_tuple=False).view(-1), (~pos_mask).nonzero(as_tuple=False).view(-1)
        nP, nN = pos_idx.numel(), neg_idx.numel()

        g = torch.Generator().manual_seed(seed)
        
        def take(idx, k, rep):
            n = idx.numel();
            if n == 0 or k <= 0: return idx.new_empty((0,), dtype=idx.dtype)
            if rep: return idx[torch.randint(n, (k,), generator=g)]
            if k >= n: return idx if k==n else torch.cat([idx, idx[torch.randint(n,(k-n,),generator=g)]])
            return idx[torch.randperm(n, generator=g)[:k]]

        if used_size is None:
            s1, b1 = nP, int(round(nP*(1-ratio)/ratio))
            b2, s2 = nN, int(round(nN*ratio/(1-ratio)))
            S, B = (s1, b1) if b1 <= nN else (s2, b2)
        else:
            total = int(used_size); S, B = int(round(total*ratio)), total-int(round(total*ratio))

        used = torch.cat([take(pos_idx,S,replace), take(neg_idx,B,replace)])
        used = used[torch.randperm(used.numel(), generator=g)]
        
        mask = torch.ones(phi.size(0), dtype=torch.bool); mask[torch.unique(used) if replace else used] = False
        
        unused = mask.nonzero(as_tuple=False).view(-1)
        
        theta_used, phi_used, y_used = theta.index_select(0, used).contiguous(), phi.index_select(0, used).contiguous(), y.index_select(0, used).contiguous()
        theta_unused, phi_unused, y_unused = theta.index_select(0, unused).contiguous(), phi.index_select(0, unused).contiguous(), y.index_select(0, unused).contiguous()
        
        fi_used = file_index.index_select(0, used) if file_index is not None else None
        fi_unused = file_index.index_select(0, unused) if file_index is not None else None
        
        return theta_used, phi_used, y_used, theta_unused, phi_unused, y_unused, fi_used, fi_unused

    # --- simple per-file mixup helper used inside _load_all ---
    def _mix_by_file_chunks(self, theta: torch.Tensor, phi: torch.Tensor, y: torch.Tensor, fidx: torch.Tensor, *,
                         positive_fn: Optional[List]=None, use_beta: Optional[Tuple[float, float]]=(1.,1.),
                         margin: float=0.0, seed: int=12345) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply _mix_bg_sig separately to chunks of data that share the same file ID.
        
        Args:
            theta: Input tensor of shape (N, d_theta)
            phi: Input tensor of shape (N, d_phi)
            y: Target tensor of shape (N,) or (N, d_y)
            fidx: File index tensor of shape (N,)
            positive_fn: Function(s) to identify positive samples
            use_beta: Parameters for Beta distribution for mixing coefficients
            margin: Margin for mixing coefficient clipping
            seed: Random seed
        
        Returns:
            Tuple of mixed (theta, phi, y) tensors
        """
        # Get unique file IDs and their indices
        unique_ids = torch.unique(fidx)
        
        # Initialize output tensors
        theta_mixed = []
        phi_mixed = []
        y_mixed = []
        file_idx = []
        
        # Process each file chunk separately
        for file_id in unique_ids:
            # Get mask for current file ID
            mask = (fidx == file_id)
            
            # Extract data for current file
            theta_chunk = theta[mask]
            phi_chunk = phi[mask]
            y_chunk = y[mask]
            
            # Apply mixing to this chunk
            theta_m, phi_m, y_m = self._mix_negatives_positives(theta_chunk, phi_chunk, y_chunk,
                                                    positive_fn=positive_fn,
                                                    use_beta=use_beta,
                                                    margin=margin,
                                                    seed=seed,
                                                    mix_ratio=self.dataset_config.get('mixup_ratio'))

            theta_mixed.append(theta_m)
            phi_mixed.append(phi_m)
            y_mixed.append(y_m)
            file_idx.append(torch.full((phi_m.size(0),), file_id, dtype=torch.long))
        
        # Concatenate all chunks back together
        return (torch.cat(theta_mixed, dim=0),
                torch.cat(phi_mixed, dim=0),
                torch.cat(y_mixed, dim=0),
                torch.cat(file_idx, dim=0))

    def _mix_negatives_positives(self, theta: torch.Tensor, phi: torch.Tensor, y: torch.Tensor, *, positive_fn: Optional[List]=None,
                    use_beta: Optional[Tuple[float, float]]=(1.,1.), margin: float=0.0,
                    seed: int=12345, mix_ratio: float=1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Mix negative and positive samples, replacing a portion of the original samples with mixed versions.
        
        Args:
            theta, phi, y: Input tensors
            positive_fn: Function(s) to identify positive samples
            use_beta: Parameters for Beta distribution
            margin: Margin for mixing coefficient clipping
            seed: Random seed
            mix_ratio: Ratio of samples to replace with mixed versions (0.0 to 1.0).
                      For example, 0.5 means 50% of samples will be replaced with mixed versions.
        """
        # Work on copies
        pos = self._get_positive_indices(y, positive_fn)

        if pos.sum() == 0 or (~pos).sum() == 0:
            raise ValueError("No positives found")
        
        pos_idx = pos.nonzero(as_tuple=False).view(-1)
        neg_idx = (~pos).nonzero(as_tuple=False).view(-1)
        
        # Initialize random generator
        g = torch.Generator().manual_seed(seed)
        
        # Initialize output tensors with original data
        thetam = theta.clone()
        phim = phi.clone()
        yi = y if y.ndim > 1 else y.unsqueeze(1)
        ym = yi.clone()
        
        # Determine how many samples to mix
        total_samples = y.shape[0]
        n_mix = int(total_samples * mix_ratio)
        
        if n_mix > 0:
            # Randomly select indices to replace with mixed samples
            replace_idx = torch.randperm(total_samples, generator=g)[:n_mix]
            
            # For each selected index, pick a negative and positive sample to mix
            neg_choices = torch.randint(high=neg_idx.numel(), size=(n_mix,), generator=g)
            pos_choices = torch.randint(high=pos_idx.numel(), size=(n_mix,), generator=g)
            
            neg_samples = neg_idx[neg_choices]
            pos_samples = pos_idx[pos_choices]
            
            # Generate mixing coefficients
            if use_beta and isinstance(use_beta, list) and len(use_beta) == 2:
                a = torch.distributions.Beta(use_beta[0], use_beta[1]).sample((n_mix, 1))
            else:
                a = torch.rand((n_mix, 1), generator=g)
            if margin > 0:
                a = torch.where(a >= 1. - margin, torch.ones_like(a), 
                              torch.where(a <= margin, torch.zeros_like(a), a))
            
            # Create mixed samples
            theta_neg, theta_pos = theta.index_select(0, neg_samples), theta.index_select(0, pos_samples)
            phi_neg, phi_pos = phi.index_select(0, neg_samples), phi.index_select(0, pos_samples)
            y_neg, y_pos = yi.index_select(0, neg_samples), yi.index_select(0, pos_samples)
            
            # Apply mixing
            theta_mixed = a * theta_pos + (1. - a) * theta_neg
            phi_mixed = a * phi_pos + (1. - a) * phi_neg
            y_mixed = a * y_pos + (1. - a) * y_neg
            
            # Replace selected indices with mixed samples
            thetam.index_copy_(0, replace_idx, theta_mixed)
            phim.index_copy_(0, replace_idx, phi_mixed)
            ym.index_copy_(0, replace_idx, y_mixed)
        
        return thetam, phim, ym

    def __len__(self) -> int: return self.data[self.mode]["phi"].shape[0]

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
        
    def _epoch_seed(self) -> int:
        if not hasattr(self, '_epoch_counter'): self._epoch_counter = 0
        s = self.seed + self._epoch_counter; self._epoch_counter += 1; return s

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
        """Iterator for training/validation/testing modes where data is already loaded in memory."""
        n = self.data[self.mode]["phi"].shape[0]
        perm = torch.randperm(n, generator=(torch.Generator().manual_seed(self._epoch_seed()))) if self.shuffle else torch.arange(n)
        s, e = self._compute_worker_slice(n)
        if s >= e: return iter(())

        shard = perm[s:e]; bs = self.batch_size; theta, phi, y = self.data[self.mode]["theta"], self.data[self.mode]["phi"], self.data[self.mode]["y"]
        for i in range(s, e, bs):
            idx = shard[i - s: i - s + bs]
            thetab, phib, yb = theta.index_select(0, idx), phi.index_select(0,idx), y.index_select(0, idx)
            if self.device is not None:
                thetab, phib, yb = thetab.to(self.device, non_blocking=True), phib.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
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


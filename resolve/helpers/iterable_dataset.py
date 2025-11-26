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

ContextSet = collections.namedtuple("ContextSet", ("theta", "phi", "y", "idx"))
QuerySet   = collections.namedtuple("QuerySet",   ("theta", "phi", "idx"))

BatchCollection = collections.namedtuple(
    "BatchCollection",
    ("context", "query", "target_y")
)

class InMemoryIterableData(IterableDataset):
    def __init__(self, files: Sequence[str], batch_size: int = 1000,
                 parameter_config: Dict = None, dataset_config: Dict = None, positive_condition: Optional[List]=None,
                 normalizer: Optional[Normalizer]=Normalizer(), mode: Optional[str] = "train") -> None:
        super().__init__()
        
        self.files, self.shuffle, self.seed = list(files), dataset_config["shuffle_dataset"], dataset_config["seed"]
        self.parameter_config, self.dataset_config = (parameter_config or {}), dataset_config
        self.batch_size_tgt = math.ceil(batch_size*(1.-self.dataset_config.get("context_ratio", 1./3.)))
        self.batch_size_ctx = batch_size - self.batch_size_tgt
        self.context_ratio = round(self.batch_size_ctx/batch_size, int(math.log10(batch_size)))

        self.mode = mode
        self._normalizer = normalizer
        self.sampler = Sampler(positive_condition, shuffle=self.shuffle, seed=self.seed)
        self.sampler._epoch_counter = -1

        self.state = None
        self.meta = {}

        # load all data into memory
        theta, phi, y, fidx = self._load_data_to_mem(self.files, self.parameter_config)
        
        self.theta_to_id = self.sampler.get_unique_ids(theta)

        self.data = self._set_data(theta, phi, y, fidx)
        self.build_batches(0)
        
        if not isinstance(self.dataset_config.get('positive_ratio_train'), list) and self.dataset_config.get("positive_ratio_train", None) is not None:
            self.set_batch_schedule(target_pos_frac=self.dataset_config.get("positive_ratio_train", None), max_pos_reuse_per_epoch = self.dataset_config.get("max_positive_reuse",0.), sticky_frac = 0.25, seed=self.dataset_config.get("seed",12345))
        
    def make_empty_like(self,*tensors):
                return [torch.empty_like(t) for t in tensors]
    
    def _set_data(self, theta: torch.Tensor, phi: torch.Tensor, y: torch.Tensor, fidx: torch.Tensor):
        data = {}
        pos_mask = self.sampler.get_positive_indices(y)
        positive_ratio_data = pos_mask.sum(dim=0)/y.shape[0]
        splitter = Splitter(self.shuffle, seed=42)
        idx =  torch.arange(phi.shape[0])
        if self.mode == "train":
            # apply normalization
            self._normalizer = Normalizer(self.dataset_config.get("use_feature_normalization", None))
            theta, phi = self._normalizer.fit_transform_as_f32(theta=theta, phi=phi)

            data = {"data": {"theta": theta, "phi": phi, "y": y, "file_indices": fidx}}

            # split into training, validation and testing data
            val_size = self.dataset_config.get('val_ratio',0.2)
            if val_size > 0.:
                idx, idx_val = splitter.train_test_split(idx, groups=theta[idx], test_size=val_size)
                if self.context_ratio >0. :
                    idx_val, idx_val_ctx = splitter.train_test_split(idx_val, groups=theta[idx_val], test_size=self.context_ratio)
                data.update({"validate": {"target": {"indices": idx_val, "batch_size": self.batch_size_tgt}}})
                
            test_size = self.dataset_config.get('test_ratio',0.2)/ (1.-val_size)
            if test_size > 0. :
                idx, idx_test = splitter.train_test_split(idx, groups=theta[idx], test_size=test_size)
                if self.context_ratio >0.:
                    idx_test, idx_test_ctx = splitter.train_test_split(idx_test, groups=theta[idx_test], test_size=self.context_ratio)
                data.update({"test": {"target":{"indices": idx_test, "batch_size": self.batch_size_tgt}}})
                
            # split training data into context and target data
            if self.context_ratio >0. :
                idx, idx_ctx = splitter.train_test_split(idx, groups=theta[idx], test_size=self.context_ratio)

            # apply mixup to context data only
            if self.dataset_config and self.dataset_config.get('mixup_ratio', 0.) > 0.0:

                if self.context_ratio >0.:
                    theta[idx_ctx], phi[idx_ctx], y[idx_ctx], fidx[idx_ctx] = self.sampler.mix_by_file_chunks(
                            theta[idx_ctx], phi[idx_ctx], y[idx_ctx], fidx[idx_ctx], self.dataset_config.get('mixup_ratio'),
                            use_beta=self.dataset_config.get('use_beta', None),
                            margin=float(self.dataset_config.get('mixup_margin', 0.0))
                        )
                else:
                    theta[idx], phi[idx], y[idx], fidx[idx] = self.sampler.mix_by_file_chunks(
                            theta[idx], phi[idx], y[idx], fidx[idx],self.dataset_config.get('mixup_ratio'),
                            use_beta=self.dataset_config.get('use_beta', None),
                            margin=float(self.dataset_config.get('mixup_margin', 0.0))
                        )

            data.update({"train": {"target":{"indices": idx, "batch_size": self.batch_size_tgt}}})

            if self.context_ratio > 0.:
                data["train"].update({"context": {"indices": idx_ctx, "batch_size": self.batch_size_ctx}})
                if val_size > 0.:
                    data["validate"].update({"context":{"indices": idx_val_ctx, "batch_size": self.batch_size_ctx}})
                if test_size > 0.:
                    data["test"].update({"context":{"indices": idx_test_ctx, "batch_size": self.batch_size_ctx}})
        else:
            theta = self._normalizer.transform(x=theta, feature_grp="theta")
            phi = self._normalizer.transform(x=phi, feature_grp="phi")
            theta = theta.float().contiguous(); phi = phi.float().contiguous()
            data = {"data": {"theta": theta, "phi": phi, "y": y, "file_indices": fidx}}
            
            if self.context_ratio >0.:
                idx, idx_ctx = splitter.train_test_split(idx, groups=theta[idx], test_size=self.context_ratio)
                data.update({f"{self.mode}":{"context":{"indices": idx_ctx, "batch_size": self.batch_size_ctx}}})
            data[f"{self.mode}"].update({"target":{"indices": idx, "batch_size": self.batch_size_tgt}})
            
        return data
    
    def build_batches(self, epoch):
        if self.sampler._epoch_counter == epoch: return
        for k in self.data.keys():
            if k == "data": 
                continue
            perm = None
            for t in self.data[k].keys():
                self.data[k][t]["batches"], _, self.data[k][t]["meta"], perm = self.sampler.build_batches(self.data[k][t]["indices"], batch_size=self.data[k][t]["batch_size"], randperm=perm)
        self.sampler._epoch_counter = epoch

    def set_batch_schedule(self,
        target_pos_frac: float,
        max_pos_reuse_per_epoch: int = 0,     # 0 => no reuse; >0 => cap per epoch
        sticky_frac: float = 0.25,            # keep 25% of last epoch's negs
    ):

        idx_ctx = self.data[self.mode]["context"]["indices"] if "context" in self.data[self.mode].keys() else None
        idx_tgt = self.data[self.mode]["target"]["indices"]
        # train sets the batch size and needs to be processed first
        if self.mode == "train" and target_pos_frac != None and self.dataset_config.get('mixup_ratio', 0.) == 0.0:
            if idx_ctx is not None:
                self.data[self.mode]["context"]["batches"], self.state, self.meta = self.sampler.build_batches_with_posneg_ratio_groupaware(
                    self.data["data"]["file_indices"][idx_ctx],
                    self.data["data"]["y"][idx_ctx],
                    idx_ctx,
                    self.data["data"]["y"],
                    target_pos_frac=target_pos_frac,
                    batch_size=self.batch_size_ctx,
                    max_pos_reuse_per_epoch=max_pos_reuse_per_epoch,   # cap reuse; set 0 for no reuse
                    sticky_frac=sticky_frac,
                    unused_neg_subset=self.state
                )
                batch_size_tgt = math.ceil(self.data[self.mode]["target"]["indices"].shape[0]/self.meta["num_batches"])

                rperm = torch.arange(self.meta["num_batches"]) if self.sampler.shuffle == "batch_wise" else None
                self.data[self.mode]["target"]["batches"], _, meta, _ = self.sampler.build_batches(self.data[self.mode]["target"]["indices"], batch_size=batch_size_tgt, randperm=rperm)
                self.meta["batch_size"] = meta["batch_size"]
                print(self.meta)
            else:
                self.data[self.mode]["target"]["batches"], self.state, self.meta = self.sampler.build_batches_with_posneg_ratio_groupaware(
                    self.data["data"]["file_indices"][idx_tgt],
                    self.data["data"]["y"][idx_tgt],
                    idx_tgt,
                    target_pos_frac=target_pos_frac,
                    batch_size=self.batch_size_tgt,
                    max_pos_reuse_per_epoch=max_pos_reuse_per_epoch,   # cap reuse; set 0 for no reuse
                    sticky_frac=sticky_frac,
                    unused_neg_subset=self.state
                )
                print(self.meta)
            
        else:
            self.data[self.mode]["target"]["batches"], self.state, self.meta, rperm = self.sampler.build_batches(idx_tgt, batch_size=self.batch_size_tgt)
            if idx_ctx is not None: self.data[self.mode]["context"]["batches"] = self.sampler.build_batches(idx_ctx, batch_size=self.batch_size_ctx, randperm=rperm)[0]
        self.sampler._epoch_counter += 1 

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

        # ensure float32 on CPU
        phi = phi.contiguous().to(torch.float32)
        theta = theta.contiguous().to(torch.float32)
        y = y.contiguous().to(torch.float32)
        return theta, phi, y

    def _load_data_to_mem(self, files: Sequence[str], cfg: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        Thetas, Phis, ys, file_inds = [], [], [], []
        for i, fp in enumerate(files):
            if not os.path.exists(fp): raise FileNotFoundError(fp)
            Thetai, Phii, yi = self._read_in_from_file(fp, cfg)
            Thetas.append(Thetai); Phis.append(Phii); ys.append(yi)
            file_inds.append(torch.full((Phii.size(0),), i, dtype=torch.long))
        Theta, Phi, y, fidx = torch.cat(Thetas, 0).contiguous(), torch.cat(Phis, 0).contiguous(), torch.cat(ys, 0).contiguous(),torch.cat(file_inds, 0).contiguous()
        return Theta, Phi, y, fidx

    def set_mode(self, mode):
        self.mode = mode
    
    def set_normalizer(self, method_or_obj):
        if isinstance(method_or_obj, Normalizer):
            self._normalizer = method_or_obj
        else:
            self._normalizer = Normalizer(method_or_obj)

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

        batches_tgt = self.data[self.mode]["target"].get("batches", None)
        total_batches = len(batches_tgt)
        b_start, b_end = self._compute_worker_slice(total_batches)  # reuse same helper; it just slices a range
        if b_start >= b_end:
            return iter(())
        theta = self.data["data"]["theta"]
        phi   = self.data["data"]["phi"]
        y     = self.data["data"]["y"]

        for b in range(b_start, b_end):
            idx_tgt = self.data[self.mode]["target"]["batches"][b]
            b_phi_tgt = phi.index_select(0, idx_tgt).unsqueeze(0)
            b_theta_tgt = theta.index_select(0, idx_tgt).unsqueeze(0)
            b_y_tgt     = y.index_select(0, idx_tgt).unsqueeze(0)

            if self.context_ratio > 0.:
                idx_ctx = self.data[self.mode]["context"]["batches"][b]
                b_phi_ctx = phi.index_select(0, idx_ctx).unsqueeze(0)
                b_theta_ctx = theta.index_select(0, idx_ctx).unsqueeze(0)
                b_y_ctx     = y.index_select(0, idx_ctx).unsqueeze(0)
            else:
                b_theta_ctx, b_phi_ctx, b_y_ctx, idx_ctx = torch.empty(0), torch.empty(0), torch.empty(0),torch.empty(0) 


            batch = BatchCollection(
                context=ContextSet(theta=b_theta_ctx.contiguous(), phi=b_phi_ctx.contiguous(), y=b_y_ctx.contiguous(), idx=idx_ctx),
                query=QuerySet(theta=b_theta_tgt.contiguous(), phi=b_phi_tgt.contiguous(), idx=idx_tgt),
                target_y=b_y_tgt.contiguous(),
            )
            yield batch

    
    def __getitem__(self, index):
        """
        Return the i-th yielded batch.
        """
        idx_tgt = self.data[self.mode]["target"]["batches"][index]
        b_phi_tgt = self.data["data"]["phi"].index_select(0, idx_tgt).unsqueeze(0)
        b_theta_tgt = self.data["data"]["theta"].index_select(0, idx_tgt).unsqueeze(0) if self.data[self.mode]["target"].get("theta",None) is not None else None
        b_y_tgt     = self.data["data"]["y"].index_select(0, idx_tgt).unsqueeze(0)     if self.data[self.mode]["target"].get("y",None)     is not None else None

        if self.context_ratio > 0.:
            idx_ctx = self.data[self.mode]["context"]["batches"][index]
            b_phi_ctx = self.data["data"]["phi"].index_select(0, idx_ctx).unsqueeze(0)
            b_theta_ctx = self.data["data"]["theta"].index_select(0, idx_ctx).unsqueeze(0) if self.data[self.mode]["context"].get("theta",None) is not None else None
            b_y_ctx     = self.data["data"]["y"].index_select(0, idx_ctx).unsqueeze(0)     if self.data[self.mode]["context"].get("y",None)     is not None else None
        else:
            b_theta_ctx, b_phi_ctx, b_y_ctx, idx_ctx = torch.empty(0), torch.empty(0), torch.empty(0),torch.empty(0) 


        batch = BatchCollection(
            context=ContextSet(theta=b_theta_ctx.contiguous(), phi=b_phi_ctx.contiguous(), y=b_y_ctx.contiguous(), idx=idx_ctx),
            query=QuerySet(theta=b_theta_tgt.contiguous(), phi=b_phi_tgt.contiguous(), idx=idx_tgt),
            target_y=b_y_tgt.contiguous(),
        )

        yield batch


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
            file_mask_t = (file_indices[1] == file_idx)
            file_mask_ctx = (file_indices[0] == file_idx)

            # Get data for current file
            file_theta_t = theta[1][file_mask_t]
            file_phi_t = phi[1][file_mask_t]
            file_y_t = y[1][file_mask_t]
            file_theta_ctx = theta[0][file_mask_ctx]
            file_phi_ctx = phi[0][file_mask_ctx]
            file_y_ctx = y[0][file_mask_ctx]

            # Process file in batches
            n = file_phi_t.shape[0]

            ratio_ctx = self.context_ratio/(1.-self.context_ratio)
            for start_idx in range(0, n, self.batch_size_tgt):
                end_idx = min(start_idx + self.batch_size_tgt, n)
                start_idx_ctx = start_idx*ratio_ctx
                end_idx_ctx = end_idx*ratio_ctx
                
                # Extract batch
                theta_t = file_theta_t[start_idx:end_idx]
                phi_t = file_phi_t[start_idx:end_idx]
                y_t = file_y_t[start_idx:end_idx]
                theta_ctx = file_theta_ctx[start_idx_ctx:end_idx_ctx]
                phi_ctx = file_phi_ctx[start_idx_ctx:end_idx_ctx]
                y_ctx = file_y_ctx[start_idx_ctx:end_idx_ctx]

                def ensure_3d(a): return a.unsqueeze(0) if a.dim()==2 else a
                theta_ctx, phi_ctx, y_ctx  = map(ensure_3d, (theta_ctx, phi_ctx, y_ctx))
                theta_t, phi_t, y_t = map(ensure_3d, (theta_t, phi_t, y_t))

                ctx_theta_cell = self.sampler.to_cell(theta_ctx,1).to(torch.float32)
                qry_theta_cell = self.sampler.to_cell(theta_t,1).to(torch.float32)
        
                batch = BatchCollection(
                    context=ContextSet(theta=theta_ctx.contiguous(), phi=phi_ctx.contiguous(), y=y_ctx.contiguous(), theta_cell=ctx_theta_cell.contiguous()),
                    query=QuerySet(theta=theta_t.contiguous(), phi=phi_t.contiguous(), theta_cell=qry_theta_cell.contiguous()),
                    target_y=y_t.contiguous(),
                )
                
                yield batch, file_idx.item(), end_idx >= n
    
    def _format_batch(self, theta, phi, y):

        n_ctx = int(phi.shape[0] * self.context_ratio)
        
        if self.dataset_config.get("context_is_subset", True):
            theta_ctx, phi_ctx, y_ctx = theta[:n_ctx], phi[:n_ctx], y[:n_ctx]
            self.theta_query, self.phi_query, y_tgt = theta, phi, y
            
        else:
            theta_ctx, phi_ctx, y_ctx = theta[:n_ctx], phi[:n_ctx], y[:n_ctx]
            self.theta_query, self.phi_query, y_tgt = theta[n_ctx:], phi[n_ctx:], y[n_ctx:]

        def ensure_3d(a): return a.unsqueeze(0) if a.dim()==2 else a
        theta_ctx, phi_ctx, y_ctx  = map(ensure_3d, (theta_ctx, phi_ctx, y_ctx))
        self.theta_query, self.phi_query, y_tgt = map(ensure_3d, (self.theta_query, self.phi_query, y_tgt))

        ctx_theta_cell = self.sampler.to_cell(theta_ctx[0],1).to(torch.float32)
        qry_theta_cell = self.sampler.to_cell(self.theta_query[0],1).to(torch.float32)
        
        return BatchCollection(
            context=ContextSet(theta=theta_ctx.contiguous(), phi=phi_ctx.contiguous(), y=y_ctx.contiguous(), theta_cell=ctx_theta_cell.contiguous()),
            query=QuerySet(theta=self.theta_query.contiguous(), phi=self.phi_query.contiguous(), theta_cell=qry_theta_cell.contiguous()),
            target_y=y_tgt.contiguous(),
        )

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
        store = self.data[self.mode]["target"]

        if store.get("batches") is not None:
            return len(store["batches"])
        return int(math.ceil(self.data["data"]["phi"][store["indices"]].shape[-2] / self.batch_size_tgt))
    
    def num_samples(self) -> int:
        return self.data["data"]["phi"].shape[-2]
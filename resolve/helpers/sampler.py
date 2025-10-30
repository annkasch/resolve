import torch
from typing import List, Optional, Sequence, Tuple, Dict, Union, Any
import operator
import functools
import numpy as np
import math

class Sampler():
    def __init__(self, batch_size: int,  positive_condition: str, shuffle="global", seed: Optional[int] = None):
        super().__init__()
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.positive_fn = self.positive_function(positive_condition) if positive_condition else None
        self.seed = seed
    
    def build_batches(self, n_samples: int):
        if self.shuffle == "global":
            perm = torch.randperm(n_samples, generator=torch.Generator().manual_seed(self._epoch_seed()))
        elif self.shuffle == "batch_wise":
            perm = torch.cat([torch.arange(i * self.batch_size, min((i + 1) * self.batch_size, n_samples)) for i in torch.randperm((n_samples + self.batch_size - 1) // self.batch_size, generator=torch.Generator().manual_seed(self._epoch_seed()))])
        else:
            perm = torch.arange(n_samples)

        batches = torch.split(perm, split_size_or_sections=self.batch_size)

        meta = {"batch_size": self.batch_size, "num_batches": len(batches),
                "pos_frac": None, "num_epochs": 1}

        return batches, None, meta

    def _epoch_seed(self) -> int:
        if not hasattr(self, '_epoch_counter'): self._epoch_counter = 0
        s = self.seed + self._epoch_counter; self._epoch_counter += 1; return s

    def sample_positives_negatives(
        self,
        pos_idx: torch.Tensor,
        neg_idx: torch.Tensor,
        n: int,
        nP_tot: int = 0,     # 0 => no reuse; >0 => cap per epoch
        max_pos_reuse_per_epoch: int = 0,
        sticky_frac: float = 0.25,            # keep 25% of last epoch's negs
        unused_neg_subset: torch.Tensor | None = None,
        seed: int | None = None,          # reproducible positive order
    ):


        # positives: build pool with reuse cap, then shuffle with seed
        pos_pool = torch.empty(0, dtype=torch.long)
        if nP_tot > 0:
            pos_pool = (pos_idx.repeat_interleave(pos_idx.numel()*max_pos_reuse_per_epoch) if max_pos_reuse_per_epoch > 0 else pos_idx)
            pos_pool = pos_pool[:nP_tot]
            if seed is not None and pos_pool.numel() > 1:
                g = torch.Generator().manual_seed(seed)
                perm = torch.randperm(pos_pool.numel(), generator=g)
                pos_pool = pos_pool[perm]

        unused_mask = torch.isin(neg_idx, unused_neg_subset) if unused_neg_subset != None else torch.ones_like(neg_idx, dtype=torch.bool)
        last_neg_subset = neg_idx[~unused_mask]
        
        Nneed = n - nP_tot
        keep = int(sticky_frac * Nneed) if last_neg_subset.numel() > 0 else 0

        new_block = torch.empty(0, dtype=torch.long)
        if Nneed > keep:
            take_new = Nneed - keep

            if unused_neg_subset is None:
                base = neg_idx
            else:
                base = unused_neg_subset

            if seed is not None and base.numel() > 1:
                g = torch.Generator().manual_seed(seed+1)
                perm = torch.randperm(base.numel(), generator=g)
                base = base[perm]
            new_block = base[:take_new] 
            keep = Nneed - take_new + keep

        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed + 2)
        perm = torch.randperm(last_neg_subset.numel(), generator=g)
        sticky = last_neg_subset[perm[:keep]] if keep else None

        neg_plan = torch.cat([sticky, new_block]) if sticky!=None else new_block

        if last_neg_subset.numel() > 0:
            union_used = torch.unique(
                torch.cat([neg_plan, last_neg_subset])
            )
        else:
            union_used = neg_plan.to().long()

        used_mask = torch.isin(neg_idx.long(), union_used)
        not_used_mask = ~used_mask
        remaining_negatives = neg_idx[not_used_mask]

        return pos_pool, neg_plan, remaining_negatives

    @staticmethod
    def _as_key(t: torch.Tensor) -> Tuple:
        """Turn a 1D/ND row tensor into a Python hashable key."""
        return tuple(t.tolist()) if t.ndim > 0 else (t.item(),)

    def build_batches_with_posneg_ratio_groupaware(
        self,
        theta: torch.Tensor,                     # shape [N] or [N, d]
        y: torch.Tensor,  
        target_pos_frac: float,
        max_pos_reuse_per_epoch: int = 0,
        sticky_frac: float = 0.25,
        unused_neg_subset: Dict[Any, torch.Tensor] | None = None,  # keyed by group key
    ):
        # precompute inverse once
        if theta.ndim == 1:
            _, inverse = torch.unique(theta, return_inverse=True)
        else:
            _, inverse = torch.unique(theta, dim=0, return_inverse=True)

        num_groups = inverse.max().item() + 1

        pos = self.get_positive_indices(y)
        pos_idx = pos.nonzero(as_tuple=False).view(-1)
        neg_idx = (~pos).nonzero(as_tuple=False).view(-1)

        pos_gid = inverse[pos_idx]
        neg_gid = inverse[neg_idx]

        reuse = max(1, max_pos_reuse_per_epoch)
        n_tmp = pos_idx.numel()*reuse/target_pos_frac if target_pos_frac > 0. else neg_idx.numel()
        n = int(round(n_tmp/num_groups))
        nN_min = max(2, min(4, int(0.05 * n)))

        batches, unused = [], []
        nP_tot = 0
        for gi in range(num_groups):
            pos_g = pos_idx[pos_gid == gi]
            neg_g = neg_idx[neg_gid == gi]
            

            unused_neg_g = None
            if unused_neg_subset is not None:
                # if you must, precompute its inverse too
                inv_unused = inverse[unused_neg_subset]
                unused_neg_g = unused_neg_subset[inv_unused == gi]

            gseed = None if self.seed is None else self.seed + gi

            nP_max = min(pos_g.numel()*reuse, n) if target_pos_frac > 0. else 0
            nP_tot += nP_max

            pos_pool, neg_plan, rem = self.sample_positives_negatives(
                pos_idx=pos_g, neg_idx=neg_g,
                n=n+nN_min, nP_tot=nP_max,
                max_pos_reuse_per_epoch=max_pos_reuse_per_epoch,
                sticky_frac=sticky_frac,
                unused_neg_subset=unused_neg_g,
                seed=gseed,
            )

            selected = torch.cat([pos_pool, neg_plan])
            # don’t sort—preserve randomness, save time
            b_size = min(n+nN_min, self.batch_size)
            batches.extend(selected.split(b_size))
            unused.append(rem)

        unused = torch.cat(unused) if unused else neg_idx.new_empty((0,), dtype=torch.long)
        nepochs = self.epochs_until_full_coverage(neg_idx.numel(), int((n+nN_min)*(1-target_pos_frac))*len(batches), sticky_frac)
        meta = {"batch_size": b_size, "num_batches": len(batches),
                "pos_frac": (nP_tot / max(1, (b_size * len(batches)))), "num_epochs": nepochs}
        return batches, unused, meta
    
    @staticmethod
    def epochs_until_full_coverage(n_neg_total: int,
                                n_neg_per_epoch: int,
                                sticky_frac: float = 0.25) -> int:
        """
        Estimate the number of epochs required until all negatives have been seen once.

        Args:
            n_neg_total: Total number of available negative samples in the dataset.
            n_neg_per_epoch: Number of negative samples used per epoch (sum over all batches).
            sticky_frac: Fraction of negatives carried over (reused) between epochs, e.g. 0.25.

        Returns:
            int: Estimated number of epochs until all negatives have been seen at least once.
        """
        if n_neg_per_epoch <= 0:
            raise ValueError("n_neg_per_epoch must be > 0")
        if not (0.0 <= sticky_frac < 1.0):
            raise ValueError("sticky_frac must be in [0, 1)")

        if n_neg_total <= n_neg_per_epoch:
            # You already use all negatives in one epoch
            return 1

        # Derived from coverage formula:
        # E >= 1 + (N_total / N_per_epoch - 1) / (1 - sticky)
        epochs = 1 + (n_neg_total / n_neg_per_epoch - 1) / (1.0 - sticky_frac)
        return math.ceil(epochs)

    def build_batches_with_posneg_ratio(self,
        y: torch.Tensor,
        target_pos_frac: float,
        max_pos_reuse_per_epoch: int = 0,     # 0 => no reuse; >0 => cap per epoch
        sticky_frac: float = 0.25,            # keep 25% of last epoch's negs
        last_neg_subset: torch.Tensor | None = None,
        seed: int | None = None,          # reproducible positive order
    ):
        assert self.batch_size >= 2
        n_total = y.shape[0]

        pos_mask = self.get_positive_indices(y)
        pos_idx, neg_idx = pos_mask.nonzero(as_tuple=False).view(-1), (~pos_mask).nonzero(as_tuple=False).view(-1)

        nP, nN = pos_idx.numel(), neg_idx.numel()

        if nP == 0 or nN == 0:
            raise ValueError("Both classes required.")
        # batches to roughly cover n_total rows
        M = max(1, math.ceil(n_total /self.batch_size))

        # epoch positive budget
        reuse = max(1, max_pos_reuse_per_epoch)
        Pmax = nP * reuse
        Pneed = int(round((nN / 1.-target_pos_frac) * target_pos_frac))

        Ptot = min(Pneed, Pmax)
        if Ptot == 0:
            Nneed = nN
        else:   
            Nneed = int(round((Ptot / target_pos_frac) * (1.-target_pos_frac)))
        n_total = Ptot + Nneed
        M = max(1, math.ceil(n_total / self.batch_size))

        # per-batch positive counts (balanced rounding, at least 0, at most B-1)
        avgk = Ptot / M
        kfloor = int(math.floor(avgk))
        rem = Ptot - kfloor * M
        k_list = [min(self.batch_size-1, max(0, kfloor + (1 if b < rem else 0))) for b in range(M)]

        # positives: build pool with reuse cap, then shuffle with seed
        pos_pool = (pos_idx.repeat_interleave(reuse) if max_pos_reuse_per_epoch > 0 else pos_idx)
        pos_pool = pos_pool[:Ptot]
        if seed is not None and pos_pool.numel() > 1:
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(pos_pool.numel(), generator=g)
            pos_pool = pos_pool[perm]

        # chunk positives
        pos_chunks, pptr = [], 0
        for k in k_list:
            pos_chunks.append(pos_pool[pptr:pptr+k]); pptr += k

        # negatives: sticky + seeded shuffle (no replacement in plan)
        Nneed = sum(self.batch_size - k for k in k_list)
        sticky = (last_neg_subset[: int(sticky_frac * Nneed)]
                if (last_neg_subset is not None and Nneed > 0) else torch.empty(0, dtype=torch.long))

        # choose remaining negatives by seeded shuffle excluding sticky
        if Nneed > sticky.numel():
            take_new = Nneed - sticky.numel()

            if sticky.numel() == 0:
                base = neg_idx
            else:
                # Exclude sticky *by value*, not by position
                mask = ~torch.isin(neg_idx, sticky)
                base = neg_idx[mask]

            if seed is not None and base.numel() > 1:
                g = torch.Generator().manual_seed(seed+1)
                perm = torch.randperm(base.numel(), generator=g)
                base = base[perm]

            new_block = base[:take_new]
            neg_plan = torch.cat((sticky, new_block))
        else:
            neg_plan = sticky

        # chunk negatives
        neg_chunks, nptr = [], 0
        for k in k_list:
            need = self.batch_size - k
            neg_chunks.append(neg_plan[nptr:nptr+need]); nptr += need

        # assemble batches
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(seed+2)

        batches = []
        for b in range(M):
            batch = torch.cat((pos_chunks[b], neg_chunks[b]))
            perm = torch.randperm(batch.numel(), generator=g)
            batches.append(batch[perm])

        nepochs = self.epochs_until_full_coverage(nN, Nneed, sticky_frac)

        state = neg_plan
        meta  = {"batch_size": self.batch_size, "num_batches": M,
                "pos_frac": (Ptot / max(1, (self.batch_size * M))), "num_epochs": nepochs}
        return batches, state, meta

    def positive_function(self, positive_condition):
        positive_fn = np.full(len(positive_condition), None) 
        for i, cond_str in enumerate(positive_condition):
                positive_fn[i] = self._parse_condition(cond_str) 
        return positive_fn

    @staticmethod
    def _compare(x, op, value):
            """Top-level helper that is picklable."""
            return op(x, value)

    def _parse_condition(self, condition_str):
            ops = {
                        '==': operator.eq,
                        '!=': operator.ne,
                        '>=': operator.ge,
                        '<=': operator.le,
                        '>': operator.gt,
                        '<': operator.lt
                    }

            for op in ops:
                if op in condition_str:
                    value_str = condition_str.split(op)[1].strip()
                    break
            value = float(value_str) if '.' in value_str else int(value_str)
            
            return functools.partial(self._compare, op=ops[op], value=value)

    def get_positive_indices(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            y2 = y if y.ndim > 1 else y.unsqueeze(1)

            if callable(self.positive_fn):
                pos = torch.as_tensor(self.positive_fn(y2), dtype=torch.bool)
            else:
                ms = [torch.as_tensor(fn(y2[:,i]), dtype=torch.bool) for i, fn in enumerate(self.positive_fn) if fn is not None]
                pos = ms[0].clone(); [pos.__ior__(m) for m in ms[1:]]

            return pos
    
    def mix_by_file_chunks(self, theta: torch.Tensor, phi: torch.Tensor, y: torch.Tensor, fidx: torch.Tensor, mixup_ratio: float, *,
                        use_beta: Optional[Tuple[float, float]]=(1.,1.),
                        margin: float=0.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        
        Returns:
            Tuple of mixed (theta, phi, y) tensors
        """
        # Get unique file IDs and their indices
        unique_ids = torch.unique(fidx)
        
        # Initialize output tensors
        theta_mixed, phi_mixed, y_mixed, file_idx = [], [], [], []
        
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
                                                    use_beta=use_beta,
                                                    margin=margin,
                                                    seed=self.seed,
                                                    mix_ratio=mixup_ratio)

            theta_mixed.append(theta_m)
            phi_mixed.append(phi_m)
            y_mixed.append(y_m)
            file_idx.append(torch.full((phi_m.size(0),), file_id, dtype=torch.long))
        
        # Concatenate all chunks back together
        return (torch.cat(theta_mixed, dim=0),
                torch.cat(phi_mixed, dim=0),
                torch.cat(y_mixed, dim=0),
                torch.cat(file_idx, dim=0))

    def _mix_negatives_positives(self, theta: torch.Tensor, phi: torch.Tensor, y: torch.Tensor, *,
                    use_beta: Optional[Tuple[float, float]]=(1.,1.), margin: float=0.0,
                    seed: int=42, mix_ratio: float=1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        pos = self.get_positive_indices(y)

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
import torch
from typing import Optional, Tuple
import operator
import functools
import numpy as np
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PositiveGroupPool:
    group: int
    positives: torch.Tensor
    negatives: torch.Tensor
    fixed_negative_order: torch.Tensor


@dataclass(frozen=True)
class PositiveSamplingPools:
    groups: tuple[PositiveGroupPool, ...]


@dataclass(frozen=True)
class MixupPlan:
    destinations: torch.Tensor
    negative_sources: torch.Tensor
    positive_sources: torch.Tensor
    coefficients: torch.Tensor

    @classmethod
    def empty(cls):
        return cls(
            destinations=torch.empty(0, dtype=torch.long),
            negative_sources=torch.empty(0, dtype=torch.long),
            positive_sources=torch.empty(0, dtype=torch.long),
            coefficients=torch.empty((0, 1), dtype=torch.float32),
        )

    def apply(self, requested_indices, theta, phi, target, store):
        if self.destinations.numel() == 0 or requested_indices.numel() == 0:
            return theta, phi, target
        positions = torch.searchsorted(
            self.destinations,
            requested_indices,
        )
        valid = positions < self.destinations.numel()
        matches = torch.zeros_like(valid)
        matches[valid] = (
            self.destinations.index_select(0, positions[valid])
            == requested_indices[valid]
        )
        batch_positions = torch.nonzero(
            matches,
            as_tuple=False,
        ).view(-1)
        if batch_positions.numel() == 0:
            return theta, phi, target

        plan_positions = positions.index_select(0, batch_positions)
        negative_sources = self.negative_sources.index_select(
            0,
            plan_positions,
        )
        positive_sources = self.positive_sources.index_select(
            0,
            plan_positions,
        )
        source_indices = torch.cat(
            (negative_sources, positive_sources)
        )
        (
            source_theta,
            source_phi,
            source_target,
            _source_files,
        ) = store.read_rows(source_indices)
        count = batch_positions.numel()
        coefficients = self.coefficients.index_select(
            0,
            plan_positions,
        ).to(dtype=theta.dtype)
        theta = theta.clone()
        phi = phi.clone()
        target = target.clone()
        theta.index_copy_(
            0,
            batch_positions,
            coefficients * source_theta[count:]
            + (1.0 - coefficients) * source_theta[:count],
        )
        phi.index_copy_(
            0,
            batch_positions,
            coefficients * source_phi[count:]
            + (1.0 - coefficients) * source_phi[:count],
        )
        target.index_copy_(
            0,
            batch_positions,
            coefficients * source_target[count:]
            + (1.0 - coefficients) * source_target[:count],
        )
        return theta, phi, target

class Sampler():
    def __init__(self, positive_condition: str, seed, shuffle="global"):
        super().__init__()
        self.shuffle = shuffle
        self.positive_fn = self.positive_function(positive_condition) if positive_condition else None
        self.seed = seed

    def _seed_for(self, epoch=0, stream=0, group=0) -> int:
        modulus = 2**63 - 1
        return int(
            (
                int(self.seed)
                + 1_000_003 * int(epoch)
                + 97_409 * int(stream)
                + 7_919 * int(group)
            )
            % modulus
        )

    def _generator(self, epoch=0, stream=0, group=0):
        return torch.Generator().manual_seed(
            self._seed_for(epoch, stream, group)
        )

    def build_batches(self, idx_array, batch_size: int, epoch=0):
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        n_samples = idx_array.shape[0]
        if n_samples == 0:
            return (), 0, None

        if self.shuffle == "global":
            perm_idx = torch.randperm(
                n_samples,
                generator=self._generator(epoch, stream=1),
            )
            batches = torch.split(idx_array[perm_idx], batch_size)
        elif self.shuffle == "batch_wise":
            unshuffled_batches = torch.split(idx_array, batch_size)
            batch_order = torch.randperm(
                len(unshuffled_batches),
                generator=self._generator(epoch, stream=2),
            )
            batches = tuple(
                unshuffled_batches[index] for index in batch_order.tolist()
            )
        else:
            batches = torch.split(idx_array, batch_size)

        return batches, len(batches), None
    def sample_positives_negatives(
        self,
        pos_idx: torch.Tensor,
        neg_idx: torch.Tensor,
        n: int,
        nP_tot: int = 0,     # 0 => no reuse; >0 => cap per epoch
        max_pos_reuse_per_epoch: int = 0,
        sticky_frac: float = 0.25,            # keep 25% of last epoch's negs
        seed: int | None = None,          # reproducible positive order
        epoch: int = 0,
        group: int = 0,
    ):
        if not 0.0 <= sticky_frac < 1.0:
            raise ValueError("sticky_frac must be in [0, 1).")

        base_seed = self.seed if seed is None else seed
        local_sampler = self if base_seed == self.seed else Sampler(
            positive_condition=None,
            seed=base_seed,
            shuffle=self.shuffle,
        )

        pos_pool = pos_idx.new_empty((0,), dtype=torch.long)
        if nP_tot > 0:
            reuse = max(1, max_pos_reuse_per_epoch)
            pos_pool = pos_idx.repeat_interleave(reuse)
            if pos_pool.numel() > 1:
                pos_pool = pos_pool[
                    torch.randperm(
                        pos_pool.numel(),
                        generator=local_sampler._generator(
                            epoch,
                            stream=10,
                            group=group,
                        ),
                    )
                ]
            pos_pool = pos_pool[:nP_tot]

        negatives_needed = max(0, n - pos_pool.numel())
        negative_count = min(negatives_needed, neg_idx.numel())
        fixed_order = neg_idx
        if negative_count:
            fixed_order = neg_idx[
                torch.randperm(
                    neg_idx.numel(),
                    generator=local_sampler._generator(
                        0,
                        stream=11,
                        group=group,
                    ),
                )
            ]
            stride = max(
                1,
                round(negative_count * (1.0 - sticky_frac)),
            )
            start = (int(epoch) * stride) % neg_idx.numel()
            positions = (
                torch.arange(negative_count, dtype=torch.long) + start
            ) % neg_idx.numel()
            neg_plan = fixed_order.index_select(0, positions)
        else:
            neg_plan = neg_idx.new_empty((0,), dtype=torch.long)

        remaining_mask = torch.ones(
            fixed_order.numel(),
            dtype=torch.bool,
        )
        if negative_count:
            remaining_mask[positions] = False
        remaining_negatives = fixed_order[remaining_mask]
        return pos_pool, neg_plan, remaining_negatives

    def prepare_positive_pools(
        self,
        groups: torch.Tensor,
        y: torch.Tensor,
        indices: torch.Tensor,
        *,
        seed: int | None = None,
    ) -> PositiveSamplingPools:
        base_seed = self.seed if seed is None else seed
        local_sampler = self if base_seed == self.seed else Sampler(
            positive_condition=None,
            seed=base_seed,
            shuffle=self.shuffle,
        )
        positive = self.get_positive_indices(y)
        pools = []
        if groups.ndim == 1:
            _unique_groups, inverse = torch.unique(
                groups,
                sorted=True,
                return_inverse=True,
            )
        else:
            _unique_groups, inverse = torch.unique(
                groups,
                dim=0,
                sorted=True,
                return_inverse=True,
            )
        for group_number in range(_unique_groups.shape[0]):
            group_mask = inverse == group_number
            group_indices = indices[group_mask]
            group_positive = positive[group_mask]
            positives = group_indices[group_positive]
            negatives = group_indices[~group_positive]
            fixed_negative_order = (
                negatives[
                    torch.randperm(
                        negatives.numel(),
                        generator=local_sampler._generator(
                            0,
                            stream=11,
                            group=group_number,
                        ),
                    )
                ]
                if negatives.numel() > 1
                else negatives
            )
            pools.append(
                PositiveGroupPool(
                    group=group_number,
                    positives=positives,
                    negatives=negatives,
                    fixed_negative_order=fixed_negative_order,
                )
            )
        return PositiveSamplingPools(tuple(pools))

    def sample_prepared_positive_pools(
        self,
        pools: PositiveSamplingPools,
        target_pos_frac: float,
        max_pos_reuse_per_epoch: int = 0,
        sticky_frac: float = 0.25,
        epoch: int = 0,
    ):
        if not 0.0 <= target_pos_frac < 1.0:
            raise ValueError("target_pos_frac must be in [0, 1).")
        if not pools.groups:
            empty = torch.empty(0, dtype=torch.long)
            return empty, 0, empty, {
                "num_epochs": 1,
                "pos_frac": 0.0,
                "num_batches": 0,
            }

        reuse = max(1, max_pos_reuse_per_epoch)
        total_positives = sum(
            pool.positives.numel() for pool in pools.groups
        )
        total_negatives = sum(
            pool.negatives.numel() for pool in pools.groups
        )
        estimate = (
            total_positives * reuse / target_pos_frac
            if target_pos_frac > 0.0
            else total_negatives
        )
        per_group = int(round(estimate / len(pools.groups)))
        minimum_negatives = max(2, min(4, int(0.05 * per_group)))
        group_size = per_group + minimum_negatives
        selected_groups = []
        unused_groups = []
        selected_positive_count = 0
        selected_negative_count = 0

        for group_number, pool in enumerate(pools.groups):
            positive_count = (
                min(pool.positives.numel() * reuse, per_group)
                if target_pos_frac > 0.0
                else 0
            )
            positive_pool = pool.positives.repeat_interleave(reuse)
            if positive_pool.numel() > 1:
                positive_pool = positive_pool[
                    torch.randperm(
                        positive_pool.numel(),
                        generator=self._generator(
                            epoch,
                            stream=10,
                            group=group_number,
                        ),
                    )
                ]
            positive_pool = positive_pool[:positive_count]

            negatives_needed = max(
                0,
                group_size - positive_pool.numel(),
            )
            negative_count = min(
                negatives_needed,
                pool.fixed_negative_order.numel(),
            )
            if negative_count:
                stride = max(
                    1,
                    round(negative_count * (1.0 - sticky_frac)),
                )
                start = (
                    int(epoch) * stride
                ) % pool.fixed_negative_order.numel()
                positions = (
                    torch.arange(negative_count, dtype=torch.long) + start
                ) % pool.fixed_negative_order.numel()
                negative_plan = pool.fixed_negative_order.index_select(
                    0,
                    positions,
                )
                unused_mask = torch.ones(
                    pool.fixed_negative_order.numel(),
                    dtype=torch.bool,
                )
                unused_mask[positions] = False
                unused = pool.fixed_negative_order[unused_mask]
            else:
                negative_plan = pool.negatives.new_empty((0,))
                unused = pool.negatives

            selected = torch.cat((positive_pool, negative_plan))
            if selected.numel() > 1:
                selected = selected[
                    torch.randperm(
                        selected.numel(),
                        generator=self._generator(
                            epoch,
                            stream=12,
                            group=group_number,
                        ),
                    )
                ]
            selected_groups.append(selected)
            unused_groups.append(unused)
            selected_positive_count += positive_pool.numel()
            selected_negative_count += negative_plan.numel()

        selected = torch.cat(selected_groups)
        unused = torch.cat(unused_groups)
        num_epochs = self.epochs_until_full_coverage(
            unused.numel(),
            selected_negative_count,
            sticky_frac,
        )
        return selected, group_size, unused, {
            "num_epochs": num_epochs,
            "pos_frac": selected_positive_count / max(1, selected.numel()),
            "num_batches": {},
        }

    def groupaware_pos_sampling(
        self,
        theta: torch.Tensor,                     # shape [N] or [N, d]
        y: torch.Tensor,
        idx: torch.Tensor,
        target_pos_frac: float,
        max_pos_reuse_per_epoch: int = 0,
        sticky_frac: float = 0.25,
        seed=None,
        epoch: int = 0,
    ):
        pools = self.prepare_positive_pools(
            theta,
            y,
            idx,
            seed=seed,
        )
        return self.sample_prepared_positive_pools(
            pools,
            target_pos_frac,
            max_pos_reuse_per_epoch=max_pos_reuse_per_epoch,
            sticky_frac=sticky_frac,
            epoch=epoch,
        )

    def build_mixup_plan(
        self,
        indices,
        groups,
        y,
        mixup_ratio,
        *,
        use_beta=(1.0, 1.0),
        margin=0.0,
        seed=None,
    ):
        if mixup_ratio <= 0.0:
            return MixupPlan.empty()
        base_seed = self.seed if seed is None else seed
        positive = self.get_positive_indices(y)
        destinations = []
        negative_sources = []
        positive_sources = []
        coefficients = []
        for group in torch.unique(groups, sorted=True).tolist():
            group_mask = groups == group
            group_indices = indices[group_mask]
            group_positive = positive[group_mask]
            positives = group_indices[group_positive]
            negatives = group_indices[~group_positive]
            if positives.numel() == 0 or negatives.numel() == 0:
                raise ValueError(
                    f"Mixup group {group} requires both positive and "
                    "negative samples."
                )
            group_seed = (
                int(base_seed) + 7_919 * (int(group) + 1)
            ) % (2**63 - 1)
            generator = torch.Generator().manual_seed(group_seed)
            count = int(group_indices.numel() * mixup_ratio)
            if count == 0:
                continue
            destination = group_indices[
                torch.randperm(
                    group_indices.numel(),
                    generator=generator,
                )[:count]
            ]
            negative = negatives[
                torch.randint(
                    negatives.numel(),
                    (count,),
                    generator=generator,
                )
            ]
            positive_source = positives[
                torch.randint(
                    positives.numel(),
                    (count,),
                    generator=generator,
                )
            ]
            if use_beta and len(use_beta) == 2:
                beta_rng = np.random.default_rng(group_seed + 3)
                coefficient = torch.as_tensor(
                    beta_rng.beta(
                        use_beta[0],
                        use_beta[1],
                        size=(count, 1),
                    ),
                    dtype=torch.float32,
                )
            else:
                coefficient = torch.rand(
                    (count, 1),
                    generator=generator,
                )
            if margin > 0.0:
                coefficient = torch.where(
                    coefficient >= 1.0 - margin,
                    torch.ones_like(coefficient),
                    torch.where(
                        coefficient <= margin,
                        torch.zeros_like(coefficient),
                        coefficient,
                    ),
                )
            destinations.append(destination)
            negative_sources.append(negative)
            positive_sources.append(positive_source)
            coefficients.append(coefficient)

        if not destinations:
            return MixupPlan.empty()
        destination = torch.cat(destinations)
        order = torch.argsort(destination)
        return MixupPlan(
            destinations=destination.index_select(0, order),
            negative_sources=torch.cat(negative_sources).index_select(
                0,
                order,
            ),
            positive_sources=torch.cat(positive_sources).index_select(
                0,
                order,
            ),
            coefficients=torch.cat(coefficients).index_select(0, order),
        )
    
    @staticmethod
    def epochs_until_full_coverage(n_unused: int,
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
        if n_unused == 0 or n_unused == None: return 1
        n_neg_total = n_unused + n_neg_per_epoch
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
        return math.ceil(epochs)+1

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

            for symbol, comparison in ops.items():
                if symbol in condition_str:
                    value_str = condition_str.split(symbol, 1)[1].strip()
                    break
            else:
                raise ValueError(
                    f"Unsupported positive condition: {condition_str!r}."
                )
            value = float(value_str)

            return functools.partial(
                self._compare,
                op=comparison,
                value=value,
            )

    def get_positive_indices(self, y: torch.Tensor) -> torch.Tensor:
            y2 = y if y.ndim > 1 else y.unsqueeze(1)

            if callable(self.positive_fn):
                pos = torch.as_tensor(self.positive_fn(y2), dtype=torch.bool)
            else:
                ms = [torch.as_tensor(fn(y2[:,i]), dtype=torch.bool) for i, fn in enumerate(self.positive_fn) if fn is not None]
                pos = ms[0].clone(); [pos.__ior__(m) for m in ms[1:]]

            return pos
    
    def mix_by_file_chunks(self, theta: torch.Tensor, phi: torch.Tensor, y: torch.Tensor, fidx: torch.Tensor, mixup_ratio: float, *,
                        use_beta: Optional[Tuple[float, float]]=(1.,1.),
                        margin: float=0.0, seed: int | None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            base_seed = self.seed if seed is None else seed
            file_seed = (
                int(base_seed) + 7_919 * (int(file_id.item()) + 1)
            ) % (2**63 - 1)
            theta_m, phi_m, y_m = self._mix_negatives_positives(theta_chunk, phi_chunk, y_chunk,
                                                    use_beta=use_beta,
                                                    margin=margin,
                                                    seed=file_seed,
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
                    seed: int=None, mix_ratio: float=1.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        seed = seed if seed is not None else self.seed
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
            if (
                use_beta
                and isinstance(use_beta, (list, tuple))
                and len(use_beta) == 2
            ):
                beta_rng = np.random.default_rng(seed + 3)
                a = torch.as_tensor(
                    beta_rng.beta(
                        use_beta[0],
                        use_beta[1],
                        size=(n_mix, 1),
                    ),
                    dtype=phi.dtype,
                    device=phi.device,
                )
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

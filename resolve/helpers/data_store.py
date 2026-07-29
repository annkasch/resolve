from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator

import torch

from resolve.helpers.data_config import DatasetSettings
from resolve.helpers.data_schema import ValidatedDataSource


_FALLBACK_MEMORY_BUDGET_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class DataChunk:
    start: int
    stop: int
    theta: torch.Tensor
    phi: torch.Tensor
    target: torch.Tensor | None
    file_indices: torch.Tensor


@dataclass(frozen=True)
class StorageSelection:
    backend: str
    estimated_peak_bytes: int
    memory_budget_bytes: int
    reason: str


class DataStore(ABC):
    """Backend-neutral row storage used by the dataloader pipeline."""

    backend: str

    @property
    @abstractmethod
    def num_samples(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def has_targets(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def iter_chunks(self, chunk_rows: int) -> Iterator[DataChunk]:
        raise NotImplementedError

    @abstractmethod
    def read_rows(
        self,
        indices: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        raise NotImplementedError

    @abstractmethod
    def materialize(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


class InMemoryDataStore(DataStore):
    backend = "memory"

    def __init__(
        self,
        theta: torch.Tensor,
        phi: torch.Tensor,
        target: torch.Tensor | None,
        file_indices: torch.Tensor,
    ):
        self._theta = theta
        self._phi = phi
        self._target = target
        self._file_indices = file_indices

    @classmethod
    def from_source(
        cls,
        source: ValidatedDataSource,
        *,
        chunk_rows: int,
    ) -> InMemoryDataStore:
        return cls(*source.load(chunk_rows=chunk_rows))

    @property
    def num_samples(self) -> int:
        return 0 if self._phi is None else self._phi.shape[0]

    @property
    def has_targets(self) -> bool:
        return self._target is not None

    def replace_tensors(
        self,
        theta: torch.Tensor,
        phi: torch.Tensor,
        target: torch.Tensor | None,
        file_indices: torch.Tensor,
    ) -> None:
        self._theta = theta
        self._phi = phi
        self._target = target
        self._file_indices = file_indices

    def iter_chunks(self, chunk_rows: int) -> Iterator[DataChunk]:
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be greater than zero.")
        for start in range(0, self.num_samples, chunk_rows):
            stop = min(start + chunk_rows, self.num_samples)
            yield DataChunk(
                start=start,
                stop=stop,
                theta=self._theta[start:stop],
                phi=self._phi[start:stop],
                target=(
                    self._target[start:stop]
                    if self._target is not None
                    else None
                ),
                file_indices=self._file_indices[start:stop],
            )

    def read_rows(self, indices):
        indices = indices.to(dtype=torch.long, device="cpu")
        return (
            self._theta.index_select(0, indices),
            self._phi.index_select(0, indices),
            (
                self._target.index_select(0, indices)
                if self._target is not None
                else None
            ),
            self._file_indices.index_select(0, indices),
        )

    def materialize(self):
        return (
            self._theta,
            self._phi,
            self._target,
            self._file_indices,
        )

    def close(self):
        self._theta = None
        self._phi = None
        self._target = None
        self._file_indices = None


def physical_memory_bytes() -> int | None:
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    if not isinstance(page_size, int) or not isinstance(page_count, int):
        return None
    if page_size <= 0 or page_count <= 0:
        return None
    return page_size * page_count


def memory_budget_bytes(settings: DatasetSettings) -> int:
    if settings.memory_budget_bytes is not None:
        return settings.memory_budget_bytes
    physical = physical_memory_bytes()
    if physical is None:
        return _FALLBACK_MEMORY_BUDGET_BYTES
    return max(1, int(physical * settings.memory_budget_fraction))


def estimate_in_memory_peak_bytes(
    source: ValidatedDataSource,
    settings: DatasetSettings,
) -> int:
    rows = source.num_samples
    feature_columns = (
        source.feature_count("theta") + source.feature_count("phi")
    )
    target_columns = (
        source.feature_count("target") if source.has_targets else 0
    )
    tensor_bytes = rows * (
        (feature_columns + target_columns) * torch.float32.itemsize
        + torch.int64.itemsize
    )
    index_bytes = rows * torch.int64.itemsize * 2
    normalization_bytes = (
        rows * feature_columns * torch.float32.itemsize
        if settings.use_feature_normalization not in (None, "none")
        else 0
    )
    mixup_bytes = (
        rows
        * (feature_columns + target_columns)
        * torch.float32.itemsize
        if settings.mixup_ratio > 0.0
        else 0
    )
    reader_temporary_bytes = min(rows, settings.stream_chunk_rows) * (
        feature_columns + target_columns
    ) * 8
    csv_overhead = (
        sum(file_spec.path.stat().st_size for file_spec in source.files) * 3
        if source.file_format == "csv"
        else 0
    )
    return int(
        tensor_bytes
        + index_bytes
        + normalization_bytes
        + mixup_bytes
        + reader_temporary_bytes
        + csv_overhead
    )


def select_storage_backend(
    source: ValidatedDataSource,
    settings: DatasetSettings,
) -> StorageSelection:
    estimate = estimate_in_memory_peak_bytes(source, settings)
    budget = memory_budget_bytes(settings)
    if settings.storage_mode == "memory":
        return StorageSelection(
            "memory",
            estimate,
            budget,
            "storage_mode explicitly requests memory",
        )
    if settings.storage_mode == "streaming":
        backend = "streaming"
        reason = "storage_mode explicitly requests streaming"
    elif estimate <= budget:
        backend = "memory"
        reason = "estimated peak fits the configured memory budget"
    else:
        backend = "streaming"
        reason = "estimated peak exceeds the configured memory budget"

    if backend == "streaming" and settings.shuffle_dataset == "batch_wise":
        raise ValueError(
            "shuffle_dataset='batch_wise' requires storage_mode='memory', "
            f"but the estimated peak is {estimate:,} bytes and the budget is "
            f"{budget:,} bytes. Increase memory_budget_bytes, reduce the "
            "dataset, or use global/disabled shuffling."
        )
    return StorageSelection(backend, estimate, budget, reason)

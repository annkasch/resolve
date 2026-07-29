from __future__ import annotations

import os
import hashlib
import json
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np
import torch

from resolve.helpers.data_config import DatasetSettings
from resolve.helpers.data_schema import ValidatedDataSource


_FALLBACK_MEMORY_BUDGET_BYTES = 2 * 1024**3
_CSV_CACHE_SCHEMA_VERSION = 1


class StreamingMaterializationError(RuntimeError):
    pass


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

    @property
    @abstractmethod
    def metadata_targets(self) -> torch.Tensor | None:
        raise NotImplementedError

    @property
    @abstractmethod
    def metadata_file_indices(self) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def set_normalizer(self, normalizer) -> None:
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

    @property
    def metadata_targets(self):
        return self._target

    @property
    def metadata_file_indices(self):
        return self._file_indices

    def set_normalizer(self, normalizer):
        self._normalizer = normalizer

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


class StreamingDataStore(DataStore):
    backend = "streaming"

    def __init__(
        self,
        source: ValidatedDataSource,
        *,
        chunk_rows: int,
        cache_directory: Path | None = None,
    ):
        self.source = source
        self.chunk_rows = chunk_rows
        self._normalizer = None
        self._handles = {}
        self._handle_pid = os.getpid()
        self.cache_path = None
        if source.file_format == "csv":
            if cache_directory is None:
                raise ValueError(
                    "CSV streaming requires a cache directory."
                )
            self.cache_path = _prepare_csv_cache(
                source,
                cache_directory,
                chunk_rows=chunk_rows,
            )
        self._file_offsets = [0]
        for file_spec in source.files:
            self._file_offsets.append(
                self._file_offsets[-1] + file_spec.row_count
            )
        self._file_offsets_tensor = torch.tensor(
            self._file_offsets[1:-1],
            dtype=torch.long,
        )
        self._metadata_targets, self._metadata_file_indices = (
            self._load_metadata()
        )

    @property
    def num_samples(self):
        return self.source.num_samples

    @property
    def has_targets(self):
        return self.source.has_targets

    @property
    def metadata_targets(self):
        return self._metadata_targets

    @property
    def metadata_file_indices(self):
        return self._metadata_file_indices

    def set_normalizer(self, normalizer):
        self._normalizer = normalizer

    def _ensure_process_handles(self):
        current_pid = os.getpid()
        if current_pid != self._handle_pid:
            for handle in self._handles.values():
                try:
                    handle.close()
                except Exception:
                    pass
            self._handles = {}
            self._handle_pid = current_pid

    def _handle(self, path):
        self._ensure_process_handles()
        key = str(path)
        handle = self._handles.get(key)
        if handle is None or not handle.id.valid:
            handle = h5py.File(path, "r")
            self._handles[key] = handle
        return handle

    def _load_metadata(self):
        file_indices = torch.empty(self.num_samples, dtype=torch.long)
        targets = (
            torch.empty(
                (
                    self.num_samples,
                    self.source.feature_count("target"),
                ),
                dtype=torch.float32,
            )
            if self.has_targets
            else None
        )
        if self.cache_path is not None:
            with h5py.File(self.cache_path, "r") as cache:
                for start in range(0, self.num_samples, self.chunk_rows):
                    stop = min(start + self.chunk_rows, self.num_samples)
                    file_indices[start:stop].copy_(
                        torch.from_numpy(
                            np.asarray(cache["file_indices"][start:stop])
                        )
                    )
                    if targets is not None:
                        targets[start:stop].copy_(
                            torch.from_numpy(
                                np.asarray(cache["target"][start:stop])
                            )
                        )
            return targets, file_indices

        for file_index, file_spec in enumerate(self.source.files):
            start = self._file_offsets[file_index]
            stop = self._file_offsets[file_index + 1]
            file_indices[start:stop].fill_(file_index)
            if targets is None:
                continue
            target_selection = file_spec.selection("target")
            with h5py.File(file_spec.path, "r") as hdf:
                dataset = hdf[target_selection.dataset_key]
                for local_start in range(
                    0,
                    file_spec.row_count,
                    self.chunk_rows,
                ):
                    local_stop = min(
                        local_start + self.chunk_rows,
                        file_spec.row_count,
                    )
                    if dataset.ndim == 1:
                        values = np.asarray(
                            dataset[local_start:local_stop]
                        ).reshape(-1, 1)
                    else:
                        values = np.asarray(
                            dataset[
                                local_start:local_stop,
                                list(target_selection.physical_indices),
                            ]
                        )
                        values = values[
                            :,
                            target_selection.configured_order,
                        ]
                    from resolve.helpers.data_readers import (
                        _as_finite_float32,
                    )

                    values = _as_finite_float32(
                        values,
                        file_spec,
                        target_selection,
                        row_offset=local_start + 1,
                    )
                    global_start = start + local_start
                    global_stop = start + local_stop
                    targets[global_start:global_stop].copy_(
                        torch.from_numpy(values)
                    )
        return targets, file_indices

    def _read_cached(self, indices):
        handle = self._handle(self.cache_path)
        unique, inverse = torch.unique(
            indices,
            sorted=True,
            return_inverse=True,
        )
        physical = unique.numpy()
        theta = torch.from_numpy(np.asarray(handle["theta"][physical]))
        phi = torch.from_numpy(np.asarray(handle["phi"][physical]))
        target = (
            torch.from_numpy(np.asarray(handle["target"][physical]))
            if self.has_targets
            else None
        )
        file_indices = torch.from_numpy(
            np.asarray(handle["file_indices"][physical])
        )
        return (
            theta.index_select(0, inverse),
            phi.index_select(0, inverse),
            target.index_select(0, inverse) if target is not None else None,
            file_indices.index_select(0, inverse),
        )

    def _read_native_hdf5_file(self, file_index, local_indices):
        file_spec = self.source.files[file_index]
        handle = self._handle(file_spec.path)
        unique, inverse = torch.unique(
            local_indices,
            sorted=True,
            return_inverse=True,
        )
        physical_rows = unique.numpy()
        arrays = {}
        by_dataset = {}
        for selection in file_spec.columns:
            by_dataset.setdefault(selection.dataset_key, []).append(selection)

        from resolve.helpers.data_readers import _as_finite_float32

        for dataset_key, selections in by_dataset.items():
            dataset = handle[dataset_key]
            if dataset.ndim == 1:
                selection = selections[0]
                values = self._read_bounded_rows(
                    dataset,
                    physical_rows,
                    None,
                ).reshape(-1, 1)
                arrays[selection.name] = _as_finite_float32(
                    values,
                    file_spec,
                    selection,
                    row_offset=1,
                )
                continue

            union_indices = sorted(
                {
                    physical_index
                    for selection in selections
                    for physical_index in selection.physical_indices
                }
            )
            loaded = self._read_bounded_rows(
                dataset,
                physical_rows,
                union_indices,
            )
            physical_to_loaded = {
                physical: position
                for position, physical in enumerate(union_indices)
            }
            for selection in selections:
                positions = [
                    physical_to_loaded[index]
                    for index in selection.physical_indices
                ]
                values = loaded[:, positions]
                values = values[:, selection.configured_order]
                arrays[selection.name] = _as_finite_float32(
                    values,
                    file_spec,
                    selection,
                    row_offset=1,
                )
        return tuple(
            (
                torch.from_numpy(arrays[name]).index_select(0, inverse)
                if name in arrays
                else None
            )
            for name in ("theta", "phi", "target")
        )

    def _read_bounded_rows(self, dataset, rows, columns):
        output_shape = (
            (len(rows),)
            if columns is None
            else (len(rows), len(columns))
        )
        output = np.empty(output_shape, dtype=dataset.dtype)
        if not len(rows):
            return output
        buckets = rows // self.chunk_rows
        for bucket in np.unique(buckets):
            positions = np.flatnonzero(buckets == bucket)
            selected_rows = rows[positions]
            start = int(selected_rows[0])
            stop = int(selected_rows[-1]) + 1
            if columns is None:
                block = np.asarray(dataset[start:stop])
            else:
                block = np.asarray(dataset[start:stop, columns])
            output[positions] = block[selected_rows - start]
        return output

    def read_rows(self, indices):
        indices = torch.as_tensor(indices, dtype=torch.long, device="cpu")
        if indices.ndim != 1:
            raise ValueError("Row indices must be one-dimensional.")
        if indices.numel() == 0:
            theta = torch.empty(
                (0, self.source.feature_count("theta")),
                dtype=torch.float32,
            )
            phi = torch.empty(
                (0, self.source.feature_count("phi")),
                dtype=torch.float32,
            )
            target = (
                torch.empty(
                    (0, self.source.feature_count("target")),
                    dtype=torch.float32,
                )
                if self.has_targets
                else None
            )
            return theta, phi, target, torch.empty(0, dtype=torch.long)
        if indices.min().item() < 0 or indices.max().item() >= self.num_samples:
            raise IndexError("Streaming row index is out of range.")

        if self.cache_path is not None:
            theta, phi, target, file_indices = self._read_cached(indices)
        else:
            file_ids = torch.bucketize(
                indices,
                self._file_offsets_tensor,
                right=True,
            )
            theta = torch.empty(
                (indices.numel(), self.source.feature_count("theta")),
                dtype=torch.float32,
            )
            phi = torch.empty(
                (indices.numel(), self.source.feature_count("phi")),
                dtype=torch.float32,
            )
            target = (
                torch.empty(
                    (indices.numel(), self.source.feature_count("target")),
                    dtype=torch.float32,
                )
                if self.has_targets
                else None
            )
            file_indices = file_ids.clone()
            for file_id in torch.unique(file_ids, sorted=True).tolist():
                request_positions = torch.nonzero(
                    file_ids == file_id,
                    as_tuple=False,
                ).view(-1)
                local_indices = indices.index_select(
                    0,
                    request_positions,
                ) - self._file_offsets[file_id]
                file_theta, file_phi, file_target = (
                    self._read_native_hdf5_file(file_id, local_indices)
                )
                theta.index_copy_(0, request_positions, file_theta)
                phi.index_copy_(0, request_positions, file_phi)
                if target is not None:
                    target.index_copy_(0, request_positions, file_target)

        if self._normalizer is not None:
            theta = self._normalizer.transform(theta, "theta")
            phi = self._normalizer.transform(phi, "phi")
        return theta, phi, target, file_indices

    def iter_chunks(self, chunk_rows):
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be greater than zero.")
        for start in range(0, self.num_samples, chunk_rows):
            stop = min(start + chunk_rows, self.num_samples)
            theta, phi, target, file_indices = self.read_rows(
                torch.arange(start, stop)
            )
            yield DataChunk(
                start,
                stop,
                theta,
                phi,
                target,
                file_indices,
            )

    def materialize(self):
        raise StreamingMaterializationError(
            "The streaming backend does not materialize the complete dataset. "
            "Set storage_mode='memory' for full-tensor access."
        )

    def close(self):
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles = {}
        self._metadata_targets = None
        self._metadata_file_indices = None


def _cache_manifest(source):
    def source_identity(file_spec):
        stat = file_spec.path.stat()
        return {
            "path": str(file_spec.path.resolve()),
            "device": stat.st_dev,
            "inode": stat.st_ino,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "rows": file_spec.row_count,
        }

    return {
        "schema_version": _CSV_CACHE_SCHEMA_VERSION,
        "files": [source_identity(file_spec) for file_spec in source.files],
        "labels": {
            name: list(source.selected_labels(name))
            for name in ("theta", "phi", "target")
            if name != "target" or source.has_targets
        },
    }


def _prepare_csv_cache(source, cache_directory, *, chunk_rows):
    manifest = _cache_manifest(source)
    encoded = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    cache_key = hashlib.sha256(encoded).hexdigest()
    cache_directory = Path(cache_directory).expanduser()
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_path = cache_directory / f"csv-{cache_key}.h5"
    if cache_path.exists():
        try:
            with h5py.File(cache_path, "r") as cache:
                if (
                    cache.attrs.get("cache_key") == cache_key
                    and bool(cache.attrs.get("complete", False))
                ):
                    return cache_path
        except OSError:
            pass

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".csv-{cache_key}-",
        suffix=".tmp",
        dir=cache_directory,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        with h5py.File(temporary_path, "w") as cache:
            cache.attrs["cache_key"] = cache_key
            cache.attrs["manifest"] = encoded.decode("utf-8")
            theta = cache.create_dataset(
                "theta",
                shape=(source.num_samples, source.feature_count("theta")),
                dtype=np.float32,
                chunks=(
                    min(chunk_rows, source.num_samples),
                    source.feature_count("theta"),
                ),
            )
            phi = cache.create_dataset(
                "phi",
                shape=(source.num_samples, source.feature_count("phi")),
                dtype=np.float32,
                chunks=(
                    min(chunk_rows, source.num_samples),
                    source.feature_count("phi"),
                ),
            )
            target = (
                cache.create_dataset(
                    "target",
                    shape=(
                        source.num_samples,
                        source.feature_count("target"),
                    ),
                    dtype=np.float32,
                    chunks=(
                        min(chunk_rows, source.num_samples),
                        source.feature_count("target"),
                    ),
                )
                if source.has_targets
                else None
            )
            file_indices = cache.create_dataset(
                "file_indices",
                shape=(source.num_samples,),
                dtype=np.int64,
                chunks=(min(chunk_rows, source.num_samples),),
            )

            from resolve.helpers.data_readers import iter_data_source_chunks

            for chunk in iter_data_source_chunks(
                source,
                chunk_rows=chunk_rows,
            ):
                theta[chunk.start:chunk.stop] = chunk.theta.numpy()
                phi[chunk.start:chunk.stop] = chunk.phi.numpy()
                file_indices[chunk.start:chunk.stop] = (
                    chunk.file_indices.numpy()
                )
                if target is not None:
                    target[chunk.start:chunk.stop] = chunk.target.numpy()
            cache.attrs["complete"] = True
            cache.flush()
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return cache_path


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

from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import torch

from resolve.helpers.data_schema import DataFileSpec, ValidatedDataSource
from resolve.helpers.data_store import DataChunk


def _as_finite_float32(
    values: np.ndarray,
    file_spec: DataFileSpec,
    selection,
    *,
    row_offset: int,
) -> np.ndarray:
    if np.iscomplexobj(values):
        raise ValueError(
            f"Complex values are not supported in {str(file_spec.path)!r}, "
            f"column {selection.selected_labels[0]!r}."
        )
    with np.errstate(over="ignore", invalid="ignore"):
        converted = np.asarray(values, dtype=np.float32)
    invalid = ~np.isfinite(converted)
    if invalid.any():
        row, column = np.argwhere(invalid)[0]
        label = selection.selected_labels[column]
        value = np.asarray(values)[row, column]
        raise ValueError(
            f"Non-finite numeric value {value!r} in "
            f"{str(file_spec.path)!r}, row {row + row_offset}, "
            f"column {label!r}."
        )
    return converted


def load_data_source(
    source: ValidatedDataSource,
    *,
    chunk_rows: int = 65_536,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
]:
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be greater than zero.")
    total_rows = source.num_samples
    theta = torch.empty(
        (total_rows, source.feature_count("theta")),
        dtype=torch.float32,
    )
    phi = torch.empty(
        (total_rows, source.feature_count("phi")),
        dtype=torch.float32,
    )
    target = (
        torch.empty(
            (total_rows, source.feature_count("target")),
            dtype=torch.float32,
        )
        if source.has_targets
        else None
    )
    file_indices = torch.empty(total_rows, dtype=torch.long)

    loaded_rows = 0
    for chunk in iter_data_source_chunks(source, chunk_rows=chunk_rows):
        if chunk.start != loaded_rows:
            raise ValueError(
                "Data-source chunks are not contiguous: expected row "
                f"{loaded_rows}, got {chunk.start}."
            )
        theta[chunk.start:chunk.stop].copy_(chunk.theta)
        phi[chunk.start:chunk.stop].copy_(chunk.phi)
        file_indices[chunk.start:chunk.stop].copy_(chunk.file_indices)
        if target is not None:
            target[chunk.start:chunk.stop].copy_(chunk.target)
        loaded_rows = chunk.stop

    if loaded_rows != total_rows:
        raise ValueError(
            f"Validated source described {total_rows} rows but loaded "
            f"{loaded_rows}."
        )
    return theta, phi, target, file_indices


def iter_data_source_chunks(
    source: ValidatedDataSource,
    *,
    chunk_rows: int,
):
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be greater than zero.")
    global_start = 0
    for file_index, file_spec in enumerate(source.files):
        reader = (
            _iter_csv_file_chunks
            if source.file_format == "csv"
            else _iter_hdf5_file_chunks
        )
        file_rows = 0
        for arrays in reader(file_spec, chunk_rows):
            row_count = arrays["theta"].shape[0]
            row_counts = {
                name: values.shape[0] for name, values in arrays.items()
            }
            if len(set(row_counts.values())) != 1:
                raise ValueError(
                    f"Inconsistent row counts while loading "
                    f"{str(file_spec.path)!r}: {row_counts}."
                )
            start = global_start + file_rows
            stop = start + row_count
            yield DataChunk(
                start=start,
                stop=stop,
                theta=_as_float_tensor(arrays["theta"]),
                phi=_as_float_tensor(arrays["phi"]),
                target=(
                    _as_float_tensor(arrays["target"])
                    if "target" in arrays
                    else None
                ),
                file_indices=torch.full(
                    (row_count,),
                    file_index,
                    dtype=torch.long,
                ),
            )
            file_rows += row_count
        if file_rows != file_spec.row_count:
            raise ValueError(
                f"File {str(file_spec.path)!r} contained {file_rows} rows "
                f"during loading; preflight found {file_spec.row_count}."
            )
        global_start += file_rows


def _load_hdf5_file(file_spec: DataFileSpec) -> dict[str, np.ndarray]:
    return next(
        _iter_hdf5_file_chunks(file_spec, file_spec.row_count)
    )


def _iter_hdf5_file_chunks(file_spec, chunk_rows):
    try:
        with h5py.File(file_spec.path, "r") as hdf:
            by_dataset = {}
            for selection in file_spec.columns:
                by_dataset.setdefault(selection.dataset_key, []).append(
                    selection
                )
            for start in range(0, file_spec.row_count, chunk_rows):
                stop = min(start + chunk_rows, file_spec.row_count)
                arrays = {}
                for dataset_key, selections in by_dataset.items():
                    dataset = hdf[dataset_key]
                    if dataset.ndim == 1:
                        values = np.asarray(dataset[start:stop]).reshape(-1, 1)
                        selection = selections[0]
                        arrays[selection.name] = _as_finite_float32(
                            values,
                            file_spec,
                            selection,
                            row_offset=start + 1,
                        )
                        continue

                    union_indices = sorted(
                        {
                            physical_index
                            for selection in selections
                            for physical_index in selection.physical_indices
                        }
                    )
                    loaded = np.asarray(
                        dataset[start:stop, union_indices]
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
                            row_offset=start + 1,
                        )
                yield arrays
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(
            f"Failed to load HDF5 values from {str(file_spec.path)!r}: "
            f"{error}"
        ) from error


def _load_csv_file(file_spec: DataFileSpec) -> dict[str, np.ndarray]:
    return next(_iter_csv_file_chunks(file_spec, file_spec.row_count))


def _iter_csv_file_chunks(file_spec, chunk_rows):
    physical_indices = sorted(
        {
            index
            for selection in file_spec.columns
            for index in selection.physical_indices
        }
    )
    try:
        frames = pd.read_csv(
            file_spec.path,
            usecols=physical_indices,
            skip_blank_lines=False,
            chunksize=chunk_rows,
        )
    except Exception as error:
        raise ValueError(
            f"Failed to load CSV values from {str(file_spec.path)!r}: {error}"
        ) from error

    absolute_to_loaded = {
        absolute: loaded
        for loaded, absolute in enumerate(physical_indices)
    }
    row_offset = 2
    for frame in frames:
        arrays = {}
        for selection in file_spec.columns:
            loaded_indices = [
                absolute_to_loaded[index]
                for index in selection.physical_indices
            ]
            selected = frame.iloc[:, loaded_indices]
            numeric = selected.apply(pd.to_numeric, errors="coerce")
            missing_or_invalid = numeric.isna()
            if missing_or_invalid.to_numpy().any():
                row, column = np.argwhere(missing_or_invalid.to_numpy())[0]
                label = selected.columns[column]
                value = selected.iat[row, column]
                description = (
                    "Missing numeric value"
                    if pd.isna(value)
                    else f"Failed to parse numeric value {value!r}"
                )
                raise ValueError(
                    f"{description} in {str(file_spec.path)!r}, "
                    f"row {row + row_offset}, column {label!r}."
                )
            values = numeric.to_numpy()[:, selection.configured_order]
            arrays[selection.name] = _as_finite_float32(
                values,
                file_spec,
                selection,
                row_offset=row_offset,
            )
        yield arrays
        row_offset += len(frame)


def _as_float_tensor(values: np.ndarray) -> torch.Tensor:
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return torch.as_tensor(
        np.asarray(values, dtype=np.float32),
        dtype=torch.float32,
    ).contiguous()

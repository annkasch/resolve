from __future__ import annotations

import h5py
import numpy as np
import pandas as pd
import torch

from resolve.helpers.data_schema import DataFileSpec, ValidatedDataSource


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
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor,
]:
    theta_parts = []
    phi_parts = []
    target_parts = []
    file_indices = []

    for file_index, file_spec in enumerate(source.files):
        arrays = (
            _load_csv_file(file_spec)
            if source.file_format == "csv"
            else _load_hdf5_file(file_spec)
        )
        theta = _as_float_tensor(arrays["theta"])
        phi = _as_float_tensor(arrays["phi"])
        target = (
            _as_float_tensor(arrays["target"])
            if "target" in arrays
            else None
        )
        row_counts = {
            "theta": theta.shape[0],
            "phi": phi.shape[0],
        }
        if target is not None:
            row_counts["target"] = target.shape[0]
        if len(set(row_counts.values())) != 1:
            raise ValueError(
                f"Inconsistent row counts while loading "
                f"{str(file_spec.path)!r}: {row_counts}."
            )

        theta_parts.append(theta)
        phi_parts.append(phi)
        if target is not None:
            target_parts.append(target)
        file_indices.append(
            torch.full(
                (phi.shape[0],),
                file_index,
                dtype=torch.long,
            )
        )

    return (
        torch.cat(theta_parts, dim=0).contiguous(),
        torch.cat(phi_parts, dim=0).contiguous(),
        (
            torch.cat(target_parts, dim=0).contiguous()
            if target_parts
            else None
        ),
        torch.cat(file_indices, dim=0).contiguous(),
    )


def _load_hdf5_file(file_spec: DataFileSpec) -> dict[str, np.ndarray]:
    arrays = {}
    try:
        with h5py.File(file_spec.path, "r") as hdf:
            for selection in file_spec.columns:
                dataset = hdf[selection.dataset_key]
                if selection.source_ndim == 1:
                    values = dataset[:].reshape(-1, 1)
                else:
                    values = dataset[:, list(selection.physical_indices)]
                    values = values[:, selection.configured_order]
                arrays[selection.name] = _as_finite_float32(
                    np.asarray(values),
                    file_spec,
                    selection,
                    row_offset=1,
                )
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(
            f"Failed to load HDF5 values from {str(file_spec.path)!r}: "
            f"{error}"
        ) from error
    return arrays


def _load_csv_file(file_spec: DataFileSpec) -> dict[str, np.ndarray]:
    physical_indices = sorted(
        {
            index
            for selection in file_spec.columns
            for index in selection.physical_indices
        }
    )
    try:
        frame = pd.read_csv(
            file_spec.path,
            usecols=physical_indices,
            skip_blank_lines=False,
        )
    except Exception as error:
        raise ValueError(
            f"Failed to load CSV values from {str(file_spec.path)!r}: {error}"
        ) from error

    absolute_to_loaded = {
        absolute: loaded
        for loaded, absolute in enumerate(physical_indices)
    }
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
                f"row {row + 2}, column {label!r}."
            )
        values = numeric.to_numpy()[:, selection.configured_order]
        arrays[selection.name] = _as_finite_float32(
            values,
            file_spec,
            selection,
            row_offset=2,
        )
    return arrays


def _as_float_tensor(values: np.ndarray) -> torch.Tensor:
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return torch.as_tensor(
        np.asarray(values, dtype=np.float32),
        dtype=torch.float32,
    ).contiguous()

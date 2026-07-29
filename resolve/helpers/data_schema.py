from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch

from resolve.helpers.data_config import (
    LoaderSpecification,
    ParameterSpec,
    ValidationIssue,
)


@dataclass(frozen=True)
class ColumnSelection:
    name: str
    dataset_key: str | None
    selected_labels: tuple[str, ...]
    physical_indices: tuple[int, ...]
    configured_order: tuple[int, ...]
    source_ndim: int


@dataclass(frozen=True)
class DataFileSpec:
    path: Path
    row_count: int | None
    columns: tuple[ColumnSelection, ...]

    def selection(self, name: str) -> ColumnSelection:
        return next(item for item in self.columns if item.name == name)


@dataclass(frozen=True)
class ValidatedDataSource:
    file_format: str
    files: tuple[DataFileSpec, ...]

    @property
    def paths(self) -> tuple[Path, ...]:
        return tuple(item.path for item in self.files)

    @property
    def num_samples(self) -> int:
        row_counts = [item.row_count for item in self.files]
        if any(row_count is None for row_count in row_counts):
            raise ValueError(
                "Validated data source is missing a file row count."
            )
        return sum(row_counts)

    def feature_count(self, name: str) -> int:
        return len(self.selected_labels(name))

    def selected_labels(self, name: str) -> tuple[str, ...]:
        if not self.files:
            raise ValueError("Cannot resolve labels from an empty data source.")
        return self.files[0].selection(name).selected_labels

    @property
    def has_targets(self) -> bool:
        return bool(
            self.files
            and any(
                selection.name == "target"
                for selection in self.files[0].columns
            )
        )

    def load(
        self,
        *,
        chunk_rows: int = 65_536,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
    ]:
        from resolve.helpers.data_readers import load_data_source

        return load_data_source(self, chunk_rows=chunk_rows)


def inspect_data_source(
    specification: LoaderSpecification,
    issues: list[ValidationIssue],
) -> ValidatedDataSource | None:
    extension = specification.file_format
    paths = tuple(
        sorted(specification.data_directory.glob(f"*.{extension}"))
    )
    if not paths:
        issues.append(
            ValidationIssue(
                str(specification.data_directory),
                f"contains no '*.{extension}' files",
            )
        )
        return None

    files = []
    for path in paths:
        if not path.is_file():
            issues.append(ValidationIssue(str(path), "is not a regular file"))
            continue
        if not os.access(path, os.R_OK):
            issues.append(ValidationIssue(str(path), "is not readable"))
            continue
        inspector = (
            _inspect_csv_file
            if extension == "csv"
            else _inspect_hdf5_file
        )
        file_spec = inspector(path, specification.parameters, issues)
        if file_spec is not None:
            files.append(file_spec)

    if len(files) != len(paths):
        return None
    return ValidatedDataSource(extension, tuple(files))


def _duplicate_labels(labels: Sequence[str]) -> list[str]:
    return sorted({label for label in labels if labels.count(label) > 1})


def _selection(
    parameter: ParameterSpec,
    labels: Sequence[str],
    *,
    dataset_key: str | None,
    source_ndim: int,
) -> ColumnSelection:
    physical = [labels.index(label) for label in parameter.selected_labels]
    sorted_pairs = sorted(enumerate(physical), key=lambda item: item[1])
    configured_order = [0] * len(sorted_pairs)
    for loaded_position, (configured_position, _) in enumerate(sorted_pairs):
        configured_order[configured_position] = loaded_position
    return ColumnSelection(
        name=parameter.name,
        dataset_key=dataset_key,
        selected_labels=parameter.selected_labels,
        physical_indices=tuple(item[1] for item in sorted_pairs),
        configured_order=tuple(configured_order),
        source_ndim=source_ndim,
    )


def _inspect_csv_file(
    path: Path,
    parameters: Sequence[ParameterSpec],
    issues: list[ValidationIssue],
) -> DataFileSpec | None:
    start = len(issues)
    try:
        with path.open("r", newline="", encoding="utf-8-sig") as input_file:
            reader = csv.reader(input_file)
            labels = next(reader, None)
            row_count = sum(1 for _row in reader)
    except (OSError, UnicodeError, csv.Error) as error:
        issues.append(
            ValidationIssue(str(path), f"cannot read header: {error}")
        )
        return None

    if not labels:
        issues.append(ValidationIssue(str(path), "has no CSV header"))
        return None
    if row_count == 0:
        issues.append(ValidationIssue(str(path), "contains no data rows"))
    duplicates = _duplicate_labels(labels)
    if duplicates:
        issues.append(
            ValidationIssue(
                str(path),
                f"CSV header has duplicate labels: {duplicates}",
            )
        )

    columns = []
    for parameter in parameters:
        missing = [
            label
            for label in parameter.selected_labels
            if label not in labels
        ]
        if missing:
            issues.append(
                ValidationIssue(
                    str(path),
                    f"CSV header is missing {parameter.name} labels: {missing}",
                )
            )
            continue
        columns.append(
            _selection(
                parameter,
                labels,
                dataset_key=None,
                source_ndim=2,
            )
        )

    if len(issues) != start:
        return None
    return DataFileSpec(path, row_count, tuple(columns))


def _decode_hdf5_labels(
    dataset: h5py.Dataset,
    path: Path,
    dataset_key: str,
    issues: list[ValidationIssue],
) -> tuple[str, ...] | None:
    if "labels" not in dataset.attrs:
        issues.append(
            ValidationIssue(
                f"{path}:{dataset_key}",
                "has no 'labels' attribute",
            )
        )
        return None
    try:
        raw_labels = np.atleast_1d(dataset.attrs["labels"]).tolist()
        labels = tuple(
            value.decode("utf-8")
            if isinstance(value, bytes)
            else str(value)
            for value in raw_labels
        )
    except (UnicodeError, TypeError, ValueError) as error:
        issues.append(
            ValidationIssue(
                f"{path}:{dataset_key}",
                f"has invalid labels metadata: {error}",
            )
        )
        return None

    expected = 1 if dataset.ndim == 1 else dataset.shape[1]
    if len(labels) != expected:
        issues.append(
            ValidationIssue(
                f"{path}:{dataset_key}",
                f"has {expected} columns but {len(labels)} labels",
            )
        )
    duplicates = _duplicate_labels(labels)
    if duplicates:
        issues.append(
            ValidationIssue(
                f"{path}:{dataset_key}",
                f"has duplicate labels: {duplicates}",
            )
        )
    return labels


def _inspect_hdf5_file(
    path: Path,
    parameters: Sequence[ParameterSpec],
    issues: list[ValidationIssue],
) -> DataFileSpec | None:
    start = len(issues)
    columns = []
    row_counts = {}
    schemas = {}
    try:
        with h5py.File(path, "r") as hdf:
            for dataset_key in dict.fromkeys(
                parameter.dataset_key for parameter in parameters
            ):
                if dataset_key not in hdf:
                    issues.append(
                        ValidationIssue(
                            str(path),
                            f"is missing HDF5 dataset {dataset_key!r}",
                        )
                    )
                    continue
                dataset = hdf[dataset_key]
                if not isinstance(dataset, h5py.Dataset):
                    issues.append(
                        ValidationIssue(
                            f"{path}:{dataset_key}",
                            "must be an HDF5 dataset",
                        )
                    )
                    continue
                if dataset.ndim not in (1, 2):
                    issues.append(
                        ValidationIssue(
                            f"{path}:{dataset_key}",
                            f"must be 1D or 2D, got shape {dataset.shape}",
                        )
                    )
                    continue
                if dataset.shape[0] == 0:
                    issues.append(
                        ValidationIssue(
                            f"{path}:{dataset_key}",
                            "contains no data rows",
                        )
                    )
                if (
                    not np.issubdtype(dataset.dtype, np.number)
                    or np.issubdtype(dataset.dtype, np.complexfloating)
                ):
                    issues.append(
                        ValidationIssue(
                            f"{path}:{dataset_key}",
                            "must have a real numeric dtype, "
                            f"got {dataset.dtype}",
                        )
                    )
                labels = _decode_hdf5_labels(
                    dataset,
                    path,
                    dataset_key,
                    issues,
                )
                schemas[dataset_key] = (dataset, labels)

            for parameter in parameters:
                schema = schemas.get(parameter.dataset_key)
                if schema is None:
                    continue
                dataset, labels = schema
                row_counts[parameter.name] = dataset.shape[0]
                if labels is None:
                    continue
                missing = [
                    label
                    for label in parameter.selected_labels
                    if label not in labels
                ]
                if missing:
                    issues.append(
                        ValidationIssue(
                            f"{path}:{parameter.dataset_key}",
                            f"is missing requested {parameter.name} labels: {missing}",
                        )
                    )
                    continue
                if dataset.ndim == 1 and len(parameter.selected_labels) != 1:
                    issues.append(
                        ValidationIssue(
                            f"{path}:{parameter.dataset_key}",
                            f"is 1D but {len(parameter.selected_labels)} "
                            f"{parameter.name} labels were requested",
                        )
                    )
                    continue
                columns.append(
                    _selection(
                        parameter,
                        labels,
                        dataset_key=parameter.dataset_key,
                        source_ndim=dataset.ndim,
                    )
                )
    except (OSError, ValueError) as error:
        issues.append(
            ValidationIssue(
                str(path),
                f"cannot inspect HDF5 file: {error}",
            )
        )
        return None

    if row_counts and len(set(row_counts.values())) != 1:
        issues.append(
            ValidationIssue(
                str(path),
                f"has inconsistent row counts: {row_counts}",
            )
        )
    if len(issues) != start:
        return None
    row_count = next(iter(row_counts.values()))
    return DataFileSpec(path, row_count, tuple(columns))

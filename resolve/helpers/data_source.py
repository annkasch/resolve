from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import pandas as pd
import torch


_SUPPORTED_FORMATS = {"csv", "h5", "hdf5"}
_SUPPORTED_MODES = {"train", "test", "inference"}
_SUPPORTED_NORMALIZATION = {None, "none", "zscore", "minmax"}
_SUPPORTED_SHUFFLE = {False, "global", "batch_wise"}
_CONDITION_PATTERN = re.compile(
    r"^\s*(?:==|!=|>=|<=|>|<)\s*"
    r"-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*$"
)


@dataclass(frozen=True)
class ValidationIssue:
    location: str
    message: str

    def __str__(self) -> str:
        return f"{self.location}: {self.message}"


class DataValidationError(ValueError):
    """Aggregated configuration and data-source validation failure."""

    def __init__(self, issues: Sequence[ValidationIssue]):
        self.issues = tuple(issues)
        details = "\n".join(f"  - {issue}" for issue in self.issues)
        super().__init__(
            f"Data loader preflight found {len(self.issues)} issue(s):\n"
            f"{details}"
        )


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    dataset_key: str
    selected_labels: tuple[str, ...]


@dataclass(frozen=True)
class DatasetSettings:
    seed: int
    shuffle_dataset: str | bool
    val_ratio: float
    test_ratio: float
    context_ratio: float
    context_is_subset: bool
    mixup_ratio: float
    mixup_margin: float
    use_beta: tuple[float, float] | None
    positive_ratio_train: float | tuple[float, ...] | None
    max_positive_reuse: int
    use_feature_normalization: str | None


@dataclass(frozen=True)
class LoaderSettings:
    num_workers: int
    prefetch_factor: int | None
    pin_memory: bool
    persistent_workers: bool


@dataclass(frozen=True)
class LoaderSpecification:
    mode: str
    batch_size: int
    parameters: tuple[ParameterSpec, ...]
    positive_condition: tuple[str, ...]
    dataset: DatasetSettings
    loader: LoaderSettings
    data_directory: Path
    file_format: str

    def parameter(self, name: str) -> ParameterSpec:
        return next(item for item in self.parameters if item.name == name)


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

    def selected_labels(self, name: str) -> tuple[str, ...]:
        if not self.files:
            raise ValueError("Cannot resolve labels from an empty data source.")
        return self.files[0].selection(name).selected_labels

    def load(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        theta_parts = []
        phi_parts = []
        target_parts = []
        file_indices = []

        for file_index, file_spec in enumerate(self.files):
            if self.file_format == "csv":
                arrays = _load_csv_file(file_spec)
            else:
                arrays = _load_hdf5_file(file_spec)

            theta = _as_float_tensor(arrays["theta"])
            phi = _as_float_tensor(arrays["phi"])
            target = _as_float_tensor(arrays["target"])
            row_counts = {
                "theta": theta.shape[0],
                "phi": phi.shape[0],
                "target": target.shape[0],
            }
            if len(set(row_counts.values())) != 1:
                raise ValueError(
                    f"Inconsistent row counts while loading "
                    f"{str(file_spec.path)!r}: {row_counts}."
                )

            theta_parts.append(theta)
            phi_parts.append(phi)
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
            torch.cat(target_parts, dim=0).contiguous(),
            torch.cat(file_indices, dim=0).contiguous(),
        )


def preflight_data_loader(
    mode: str,
    config: Mapping[str, Any],
) -> tuple[LoaderSpecification, ValidatedDataSource]:
    issues: list[ValidationIssue] = []
    specification = _validate_configuration(mode, config, issues)
    source = None
    if specification is not None:
        source = _inspect_data_source(specification, issues)

    if issues:
        raise DataValidationError(issues)
    if specification is None or source is None:
        raise RuntimeError("Preflight failed without reporting an issue.")
    return specification, source


def _mapping(
    value: Any,
    location: str,
    issues: list[ValidationIssue],
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        issues.append(ValidationIssue(location, "must be a mapping"))
        return None
    return value


def _required(
    mapping: Mapping[str, Any] | None,
    key: str,
    location: str,
    issues: list[ValidationIssue],
) -> Any:
    if mapping is None:
        return None
    if key not in mapping:
        issues.append(ValidationIssue(f"{location}.{key}", "is required"))
        return None
    return mapping[key]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(
    value: Any,
    location: str,
    issues: list[ValidationIssue],
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    maximum_inclusive: bool = True,
) -> float | None:
    if not _is_number(value):
        issues.append(ValidationIssue(location, "must be a number"))
        return None
    number = float(value)
    if minimum is not None and number < minimum:
        issues.append(
            ValidationIssue(location, f"must be at least {minimum}")
        )
    if maximum is not None:
        invalid = (
            number > maximum
            if maximum_inclusive
            else number >= maximum
        )
        if invalid:
            comparison = "at most" if maximum_inclusive else "less than"
            issues.append(
                ValidationIssue(location, f"must be {comparison} {maximum}")
            )
    return number


def _integer(
    value: Any,
    location: str,
    issues: list[ValidationIssue],
    *,
    minimum: int | None = None,
) -> int | None:
    if not isinstance(value, int) or isinstance(value, bool):
        issues.append(ValidationIssue(location, "must be an integer"))
        return None
    if minimum is not None and value < minimum:
        issues.append(
            ValidationIssue(location, f"must be at least {minimum}")
        )
    return value


def _boolean(
    value: Any,
    location: str,
    issues: list[ValidationIssue],
) -> bool | None:
    if not isinstance(value, bool):
        issues.append(ValidationIssue(location, "must be a boolean"))
        return None
    return value


def _labels(
    value: Any,
    location: str,
    issues: list[ValidationIssue],
) -> tuple[str, ...] | None:
    if not isinstance(value, (list, tuple)):
        issues.append(
            ValidationIssue(location, "must be a nonempty list of labels")
        )
        return None
    labels = tuple(value)
    if not labels:
        issues.append(ValidationIssue(location, "must not be empty"))
        return None
    invalid = [label for label in labels if not isinstance(label, str) or not label]
    if invalid:
        issues.append(
            ValidationIssue(location, "labels must be nonempty strings")
        )
        return None
    duplicates = sorted(
        {label for label in labels if labels.count(label) > 1}
    )
    if duplicates:
        issues.append(
            ValidationIssue(
                location,
                f"contains duplicate labels: {duplicates}",
            )
        )
        return None
    return labels


def _validate_configuration(
    mode: str,
    config: Mapping[str, Any],
    issues: list[ValidationIssue],
) -> LoaderSpecification | None:
    mode_is_valid = isinstance(mode, str) and mode in _SUPPORTED_MODES
    if not mode_is_valid:
        issues.append(
            ValidationIssue(
                "mode",
                f"must be one of {sorted(_SUPPORTED_MODES)}, got {mode!r}",
            )
        )

    root = _mapping(config, "config", issues)
    path_settings = _mapping(
        _required(root, "path_settings", "config", issues),
        "path_settings",
        issues,
    )
    simulation = _mapping(
        _required(root, "simulation_settings", "config", issues),
        "simulation_settings",
        issues,
    )
    model = _mapping(
        _required(root, "model_settings", "config", issues),
        "model_settings",
        issues,
    )
    dataloader = _mapping(
        _required(model, "dataloader", "model_settings", issues),
        "model_settings.dataloader",
        issues,
    )
    train = _mapping(
        _required(model, "train", "model_settings", issues),
        "model_settings.train",
        issues,
    )
    dataset = _mapping(
        _required(train, "dataset", "model_settings.train", issues),
        "model_settings.train.dataset",
        issues,
    )

    path_key = f"path_to_files_{mode}"
    raw_path = _required(path_settings, path_key, "path_settings", issues)
    data_directory = None
    if raw_path is not None:
        if not isinstance(raw_path, (str, os.PathLike)):
            issues.append(
                ValidationIssue(
                    f"path_settings.{path_key}",
                    "must be a filesystem path",
                )
            )
        else:
            data_directory = Path(raw_path).expanduser()
            if not data_directory.exists():
                issues.append(
                    ValidationIssue(
                        f"path_settings.{path_key}",
                        f"directory does not exist: {data_directory}",
                    )
                )
            elif not data_directory.is_dir():
                issues.append(
                    ValidationIssue(
                        f"path_settings.{path_key}",
                        f"is not a directory: {data_directory}",
                    )
                )
            elif not os.access(data_directory, os.R_OK):
                issues.append(
                    ValidationIssue(
                        f"path_settings.{path_key}",
                        f"directory is not readable: {data_directory}",
                    )
                )

    raw_format = _required(
        simulation,
        "file_format",
        "simulation_settings",
        issues,
    )
    file_format = raw_format.lower() if isinstance(raw_format, str) else None
    if file_format not in _SUPPORTED_FORMATS:
        issues.append(
            ValidationIssue(
                "simulation_settings.file_format",
                f"must be one of {sorted(_SUPPORTED_FORMATS)}",
            )
        )

    parameter_specs = []
    for name, dataset_key in (
        ("phi", "features/values"),
        ("theta", "features/values"),
        ("target", "labels/values"),
    ):
        config_key = f"{name}_labels" if name != "target" else "target_labels"
        selected = _labels(
            _required(
                simulation,
                config_key,
                "simulation_settings",
                issues,
            ),
            f"simulation_settings.{config_key}",
            issues,
        )
        if selected is not None:
            parameter_specs.append(
                ParameterSpec(name, dataset_key, selected)
            )

    raw_conditions = _required(
        simulation,
        "signal_condition",
        "simulation_settings",
        issues,
    )
    positive_condition = None
    if not isinstance(raw_conditions, (list, tuple)):
        issues.append(
            ValidationIssue(
                "simulation_settings.signal_condition",
                "must be a list of comparison expressions",
            )
        )
    else:
        positive_condition = tuple(raw_conditions)
        invalid_conditions = [
            condition
            for condition in positive_condition
            if not isinstance(condition, str)
            or _CONDITION_PATTERN.fullmatch(condition) is None
        ]
        if invalid_conditions:
            issues.append(
                ValidationIssue(
                    "simulation_settings.signal_condition",
                    "entries must look like '== 1', '< 0.5', or '>= 1e-3'",
                )
            )
        target_spec = next(
            (
                item
                for item in parameter_specs
                if item.name == "target"
            ),
            None,
        )
        if (
            target_spec is not None
            and len(positive_condition) != len(target_spec.selected_labels)
        ):
            issues.append(
                ValidationIssue(
                    "simulation_settings.signal_condition",
                    "must contain one condition per target label",
                )
            )

    batch_size = _integer(
        _required(train, "batch_size", "model_settings.train", issues),
        "model_settings.train.batch_size",
        issues,
        minimum=1,
    )

    seed = _integer(
        _required(dataset, "seed", "model_settings.train.dataset", issues),
        "model_settings.train.dataset.seed",
        issues,
    )
    shuffle = _required(
        dataset,
        "shuffle_dataset",
        "model_settings.train.dataset",
        issues,
    )
    shuffle_is_valid = (
        shuffle is False or shuffle in ("global", "batch_wise")
    )
    if not shuffle_is_valid:
        issues.append(
            ValidationIssue(
                "model_settings.train.dataset.shuffle_dataset",
                "must be 'global', 'batch_wise', or false",
            )
        )

    val_ratio = _number(
        dataset.get("val_ratio", 0.2) if dataset is not None else None,
        "model_settings.train.dataset.val_ratio",
        issues,
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )
    test_ratio = _number(
        dataset.get("test_ratio", 0.2) if dataset is not None else None,
        "model_settings.train.dataset.test_ratio",
        issues,
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )
    if (
        val_ratio is not None
        and test_ratio is not None
        and val_ratio + test_ratio >= 1.0
    ):
        issues.append(
            ValidationIssue(
                "model_settings.train.dataset",
                "val_ratio + test_ratio must be less than 1",
            )
        )

    context_ratio = _number(
        dataset.get("context_ratio", 1.0 / 3.0)
        if dataset is not None
        else None,
        "model_settings.train.dataset.context_ratio",
        issues,
        minimum=0.0,
        maximum=1.0,
        maximum_inclusive=False,
    )
    context_is_subset = _boolean(
        dataset.get("context_is_subset", True)
        if dataset is not None
        else None,
        "model_settings.train.dataset.context_is_subset",
        issues,
    )
    mixup_ratio = _number(
        dataset.get("mixup_ratio", 0.0) if dataset is not None else None,
        "model_settings.train.dataset.mixup_ratio",
        issues,
        minimum=0.0,
        maximum=1.0,
    )
    mixup_margin = _number(
        dataset.get("mixup_margin", 0.0) if dataset is not None else None,
        "model_settings.train.dataset.mixup_margin",
        issues,
        minimum=0.0,
        maximum=0.5,
    )

    raw_beta = dataset.get("use_beta") if dataset is not None else None
    use_beta = None
    if raw_beta is not None:
        if (
            not isinstance(raw_beta, (list, tuple))
            or len(raw_beta) != 2
            or not all(_is_number(value) and value > 0 for value in raw_beta)
        ):
            issues.append(
                ValidationIssue(
                    "model_settings.train.dataset.use_beta",
                    "must be null or two positive numbers",
                )
            )
        else:
            use_beta = (float(raw_beta[0]), float(raw_beta[1]))

    positive_ratio = (
        dataset.get("positive_ratio_train")
        if dataset is not None
        else None
    )
    validated_positive_ratio = None
    if positive_ratio is not None:
        values = (
            positive_ratio
            if isinstance(positive_ratio, (list, tuple))
            else (positive_ratio,)
        )
        if not values or any(
            not _is_number(value) or not 0.0 <= float(value) < 1.0
            for value in values
        ):
            issues.append(
                ValidationIssue(
                    "model_settings.train.dataset.positive_ratio_train",
                    "must be null, a ratio in [0, 1), or a nonempty list of such ratios",
                )
            )
        elif isinstance(positive_ratio, (list, tuple)):
            validated_positive_ratio = tuple(
                float(value) for value in values
            )
        else:
            validated_positive_ratio = float(positive_ratio)

    max_positive_reuse = _integer(
        dataset.get("max_positive_reuse", 0)
        if dataset is not None
        else None,
        "model_settings.train.dataset.max_positive_reuse",
        issues,
        minimum=0,
    )
    normalization = (
        dataset.get("use_feature_normalization")
        if dataset is not None
        else None
    )
    normalization_is_valid = (
        normalization is None
        or normalization in ("none", "zscore", "minmax")
    )
    if not normalization_is_valid:
        issues.append(
            ValidationIssue(
                "model_settings.train.dataset.use_feature_normalization",
                "must be null, 'none', 'zscore', or 'minmax'",
            )
        )

    num_workers = _integer(
        _required(
            dataloader,
            "dataloader_number_of_workers",
            "model_settings.dataloader",
            issues,
        ),
        "model_settings.dataloader.dataloader_number_of_workers",
        issues,
        minimum=0,
    )
    raw_prefetch = _required(
        dataloader,
        "dataloader_prefetch_factor",
        "model_settings.dataloader",
        issues,
    )
    prefetch_factor = None
    if raw_prefetch is not None:
        prefetch_factor = _integer(
            raw_prefetch,
            "model_settings.dataloader.dataloader_prefetch_factor",
            issues,
            minimum=1,
        )
    pin_memory = _boolean(
        _required(
            dataloader,
            "dataloader_pin_memory",
            "model_settings.dataloader",
            issues,
        ),
        "model_settings.dataloader.dataloader_pin_memory",
        issues,
    )
    persistent_workers = _boolean(
        _required(
            dataloader,
            "dataloader_persistent_workers",
            "model_settings.dataloader",
            issues,
        ),
        "model_settings.dataloader.dataloader_persistent_workers",
        issues,
    )

    required_values = (
        mode_is_valid,
        data_directory is not None,
        file_format in _SUPPORTED_FORMATS,
        len(parameter_specs) == 3,
        positive_condition is not None,
        batch_size is not None,
        seed is not None,
        shuffle_is_valid,
        val_ratio is not None,
        test_ratio is not None,
        context_ratio is not None,
        context_is_subset is not None,
        mixup_ratio is not None,
        mixup_margin is not None,
        max_positive_reuse is not None,
        normalization_is_valid,
        num_workers is not None,
        pin_memory is not None,
        persistent_workers is not None,
    )
    if not all(required_values):
        return None

    return LoaderSpecification(
        mode=mode,
        batch_size=batch_size,
        parameters=tuple(parameter_specs),
        positive_condition=positive_condition,
        dataset=DatasetSettings(
            seed=seed,
            shuffle_dataset=shuffle,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            context_ratio=context_ratio,
            context_is_subset=context_is_subset,
            mixup_ratio=mixup_ratio,
            mixup_margin=mixup_margin,
            use_beta=use_beta,
            positive_ratio_train=validated_positive_ratio,
            max_positive_reuse=max_positive_reuse,
            use_feature_normalization=normalization,
        ),
        loader=LoaderSettings(
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        ),
        data_directory=data_directory,
        file_format=file_format,
    )


def _inspect_data_source(
    specification: LoaderSpecification,
    issues: list[ValidationIssue],
) -> ValidatedDataSource | None:
    extension = specification.file_format
    paths = tuple(sorted(specification.data_directory.glob(f"*.{extension}")))
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
        if extension == "csv":
            file_spec = _inspect_csv_file(
                path,
                specification.parameters,
                issues,
            )
        else:
            file_spec = _inspect_hdf5_file(
                path,
                specification.parameters,
                issues,
            )
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
            has_data_row = any(row for row in reader)
    except (OSError, UnicodeError, csv.Error) as error:
        issues.append(ValidationIssue(str(path), f"cannot read header: {error}"))
        return None

    if not labels:
        issues.append(ValidationIssue(str(path), "has no CSV header"))
        return None
    if not has_data_row:
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
    return DataFileSpec(path, None, tuple(columns))


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
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
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
    schemas: dict[str, tuple[h5py.Dataset, tuple[str, ...] | None]] = {}
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
                if not np.issubdtype(dataset.dtype, np.number):
                    issues.append(
                        ValidationIssue(
                            f"{path}:{dataset_key}",
                            f"must have a numeric dtype, got {dataset.dtype}",
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
            ValidationIssue(str(path), f"cannot inspect HDF5 file: {error}")
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
                arrays[selection.name] = np.asarray(values)
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
        frame = pd.read_csv(file_spec.path, usecols=physical_indices)
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
        invalid = numeric.isna() & ~selected.isna()
        if invalid.to_numpy().any():
            row, column = np.argwhere(invalid.to_numpy())[0]
            label = selected.columns[column]
            value = selected.iat[row, column]
            raise ValueError(
                f"Failed to parse numeric value {value!r} in "
                f"{str(file_spec.path)!r}, row {row + 2}, column {label!r}."
            )
        values = numeric.to_numpy()[:, selection.configured_order]
        arrays[selection.name] = values
    return arrays


def _as_float_tensor(values: np.ndarray) -> torch.Tensor:
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return torch.as_tensor(
        np.asarray(values, dtype=np.float32),
        dtype=torch.float32,
    ).contiguous()

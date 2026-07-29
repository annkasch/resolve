from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SUPPORTED_FORMATS = {"csv", "h5", "hdf5"}
SUPPORTED_MODES = {"train", "test", "inference"}
CONDITION_PATTERN = re.compile(
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


def _mapping(value, location, issues):
    if not isinstance(value, Mapping):
        issues.append(ValidationIssue(location, "must be a mapping"))
        return None
    return value


def _required(mapping, key, location, issues):
    if mapping is None:
        return None
    if key not in mapping:
        issues.append(ValidationIssue(f"{location}.{key}", "is required"))
        return None
    return mapping[key]


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _number(
    value,
    location,
    issues,
    *,
    minimum=None,
    maximum=None,
    maximum_inclusive=True,
):
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


def _integer(value, location, issues, *, minimum=None):
    if not isinstance(value, int) or isinstance(value, bool):
        issues.append(ValidationIssue(location, "must be an integer"))
        return None
    if minimum is not None and value < minimum:
        issues.append(
            ValidationIssue(location, f"must be at least {minimum}")
        )
    return value


def _boolean(value, location, issues):
    if not isinstance(value, bool):
        issues.append(ValidationIssue(location, "must be a boolean"))
        return None
    return value


def _labels(value, location, issues):
    if not isinstance(value, (list, tuple)):
        issues.append(
            ValidationIssue(location, "must be a nonempty list of labels")
        )
        return None
    labels = tuple(value)
    if not labels:
        issues.append(ValidationIssue(location, "must not be empty"))
        return None
    if any(not isinstance(label, str) or not label for label in labels):
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


def _parameter_specs(mode, simulation, issues):
    parameters = []
    for name in ("phi", "theta"):
        config_key = f"{name}_labels"
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
            parameters.append(
                ParameterSpec(name, "features/values", selected)
            )

    target_value = (
        simulation.get("target_labels")
        if mode == "inference" and simulation is not None
        else _required(
            simulation,
            "target_labels",
            "simulation_settings",
            issues,
        )
    )
    if mode == "inference" and target_value in (None, []):
        target_labels = None
    else:
        target_labels = _labels(
            target_value,
            "simulation_settings.target_labels",
            issues,
        )
    if target_labels is not None:
        parameters.append(
            ParameterSpec("target", "labels/values", target_labels)
        )
    return parameters


def _positive_conditions(mode, simulation, target_spec, issues):
    raw_conditions = (
        simulation.get("signal_condition")
        if mode == "inference" and simulation is not None
        else _required(
            simulation,
            "signal_condition",
            "simulation_settings",
            issues,
        )
    )
    if target_spec is None:
        if raw_conditions not in (None, [], ()):
            issues.append(
                ValidationIssue(
                    "simulation_settings.signal_condition",
                    "must be omitted when inference targets are omitted",
                )
            )
        return ()
    if mode == "inference" and raw_conditions in (None, [], ()):
        return ()
    if not isinstance(raw_conditions, (list, tuple)):
        issues.append(
            ValidationIssue(
                "simulation_settings.signal_condition",
                "must be a list of comparison expressions",
            )
        )
        return ()

    conditions = tuple(raw_conditions)
    if any(
        not isinstance(condition, str)
        or CONDITION_PATTERN.fullmatch(condition) is None
        for condition in conditions
    ):
        issues.append(
            ValidationIssue(
                "simulation_settings.signal_condition",
                "entries must look like '== 1', '< 0.5', or '>= 1e-3'",
            )
        )
    if len(conditions) != len(target_spec.selected_labels):
        issues.append(
            ValidationIssue(
                "simulation_settings.signal_condition",
                "must contain one condition per target label",
            )
        )
    return conditions


def _validate_path(mode, path_settings, issues):
    path_key = f"path_to_files_{mode}"
    raw_path = _required(path_settings, path_key, "path_settings", issues)
    if raw_path is None:
        return None
    if not isinstance(raw_path, (str, os.PathLike)):
        issues.append(
            ValidationIssue(
                f"path_settings.{path_key}",
                "must be a filesystem path",
            )
        )
        return None

    path = Path(raw_path).expanduser()
    if not path.exists():
        issues.append(
            ValidationIssue(
                f"path_settings.{path_key}",
                f"directory does not exist: {path}",
            )
        )
    elif not path.is_dir():
        issues.append(
            ValidationIssue(
                f"path_settings.{path_key}",
                f"is not a directory: {path}",
            )
        )
    elif not os.access(path, os.R_OK):
        issues.append(
            ValidationIssue(
                f"path_settings.{path_key}",
                f"directory is not readable: {path}",
            )
        )
    return path


def _validate_dataset_settings(mode, dataset, target_spec, issues):
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
    shuffle_valid = (
        shuffle is False or shuffle in ("global", "batch_wise")
    )
    if not shuffle_valid:
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
    if (
        mode == "inference"
        and target_spec is None
        and context_ratio is not None
        and context_ratio > 0.0
    ):
        issues.append(
            ValidationIssue(
                "model_settings.train.dataset.context_ratio",
                "must be 0 for inference data without target labels",
            )
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
            or not all(
                _is_number(value) and value > 0
                for value in raw_beta
            )
        ):
            issues.append(
                ValidationIssue(
                    "model_settings.train.dataset.use_beta",
                    "must be null or two positive numbers",
                )
            )
        else:
            use_beta = (float(raw_beta[0]), float(raw_beta[1]))

    raw_positive_ratio = (
        dataset.get("positive_ratio_train")
        if dataset is not None
        else None
    )
    positive_ratio = None
    if raw_positive_ratio is not None:
        values = (
            raw_positive_ratio
            if isinstance(raw_positive_ratio, (list, tuple))
            else (raw_positive_ratio,)
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
        elif isinstance(raw_positive_ratio, (list, tuple)):
            positive_ratio = tuple(float(value) for value in values)
        else:
            positive_ratio = float(raw_positive_ratio)

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
    normalization_valid = (
        normalization is None
        or normalization in ("none", "zscore", "minmax")
    )
    if not normalization_valid:
        issues.append(
            ValidationIssue(
                "model_settings.train.dataset.use_feature_normalization",
                "must be null, 'none', 'zscore', or 'minmax'",
            )
        )

    valid = all(
        (
            seed is not None,
            shuffle_valid,
            val_ratio is not None,
            test_ratio is not None,
            context_ratio is not None,
            context_is_subset is not None,
            mixup_ratio is not None,
            mixup_margin is not None,
            max_positive_reuse is not None,
            normalization_valid,
        )
    )
    if not valid:
        return None
    return DatasetSettings(
        seed=seed,
        shuffle_dataset=shuffle,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        context_ratio=context_ratio,
        context_is_subset=context_is_subset,
        mixup_ratio=mixup_ratio,
        mixup_margin=mixup_margin,
        use_beta=use_beta,
        positive_ratio_train=positive_ratio,
        max_positive_reuse=max_positive_reuse,
        use_feature_normalization=normalization,
    )


def _validate_loader_settings(dataloader, issues):
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
    if any(
        value is None
        for value in (num_workers, pin_memory, persistent_workers)
    ):
        return None
    return LoaderSettings(
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )


def validate_configuration(
    mode: str,
    config: Mapping[str, Any],
    issues: list[ValidationIssue],
) -> LoaderSpecification | None:
    mode_valid = isinstance(mode, str) and mode in SUPPORTED_MODES
    if not mode_valid:
        issues.append(
            ValidationIssue(
                "mode",
                f"must be one of {sorted(SUPPORTED_MODES)}, got {mode!r}",
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

    data_directory = _validate_path(mode, path_settings, issues)
    raw_format = _required(
        simulation,
        "file_format",
        "simulation_settings",
        issues,
    )
    file_format = (
        raw_format.lower() if isinstance(raw_format, str) else None
    )
    format_valid = file_format in SUPPORTED_FORMATS
    if not format_valid:
        issues.append(
            ValidationIssue(
                "simulation_settings.file_format",
                f"must be one of {sorted(SUPPORTED_FORMATS)}",
            )
        )

    parameters = _parameter_specs(mode, simulation, issues)
    target_spec = next(
        (
            parameter
            for parameter in parameters
            if parameter.name == "target"
        ),
        None,
    )
    positive_condition = _positive_conditions(
        mode,
        simulation,
        target_spec,
        issues,
    )
    batch_size = _integer(
        _required(train, "batch_size", "model_settings.train", issues),
        "model_settings.train.batch_size",
        issues,
        minimum=1,
    )
    dataset_settings = _validate_dataset_settings(
        mode,
        dataset,
        target_spec,
        issues,
    )
    loader_settings = _validate_loader_settings(dataloader, issues)
    expected_parameters = (
        2 if mode == "inference" and target_spec is None else 3
    )

    if not all(
        (
            mode_valid,
            data_directory is not None,
            format_valid,
            len(parameters) == expected_parameters,
            batch_size is not None,
            dataset_settings is not None,
            loader_settings is not None,
        )
    ):
        return None
    return LoaderSpecification(
        mode=mode,
        batch_size=batch_size,
        parameters=tuple(parameters),
        positive_condition=positive_condition,
        dataset=dataset_settings,
        loader=loader_settings,
        data_directory=data_directory,
        file_format=file_format,
    )

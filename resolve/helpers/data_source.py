"""Compatibility facade for validated dataloader sources."""

from __future__ import annotations

from typing import Any, Mapping

from resolve.helpers.data_config import (
    DataValidationError,
    DatasetSettings,
    LoaderSettings,
    LoaderSpecification,
    ParameterSpec,
    ValidationIssue,
    validate_configuration,
)
from resolve.helpers.data_schema import (
    ColumnSelection,
    DataFileSpec,
    ValidatedDataSource,
    inspect_data_source,
)
from resolve.helpers.data_store import (
    DataChunk,
    DataStore,
    InMemoryDataStore,
    StreamingDataStore,
    StreamingMaterializationError,
    StorageSelection,
    estimate_in_memory_peak_bytes,
    memory_budget_bytes,
    select_storage_backend,
)


def preflight_data_loader(
    mode: str,
    config: Mapping[str, Any],
) -> tuple[LoaderSpecification, ValidatedDataSource]:
    issues = []
    specification = validate_configuration(mode, config, issues)
    source = (
        inspect_data_source(specification, issues)
        if specification is not None
        else None
    )

    if issues:
        raise DataValidationError(issues)
    if specification is None or source is None:
        raise RuntimeError("Preflight failed without reporting an issue.")
    return specification, source


__all__ = [
    "ColumnSelection",
    "DataChunk",
    "DataStore",
    "DataFileSpec",
    "DataValidationError",
    "DatasetSettings",
    "InMemoryDataStore",
    "StreamingDataStore",
    "StreamingMaterializationError",
    "LoaderSettings",
    "LoaderSpecification",
    "ParameterSpec",
    "StorageSelection",
    "ValidatedDataSource",
    "ValidationIssue",
    "estimate_in_memory_peak_bytes",
    "memory_budget_bytes",
    "preflight_data_loader",
    "select_storage_backend",
]

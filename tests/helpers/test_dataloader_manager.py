import copy
import csv
import inspect
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

import resolve.helpers.data_readers as data_readers
import resolve.helpers.data_source as data_source
import resolve.helpers.dataloader_manager as dataloader_manager_module
import resolve.helpers.iterable_dataset as iterable_dataset_module
from resolve.helpers.batch_requests import BatchRequest, BatchRequestSampler
from resolve.helpers import (
    BatchCollection,
    ContextSet,
    DataValidationError,
    QuerySet,
)
from resolve.helpers.dataloader_manager import DataLoaderManager
from resolve.helpers.data_store import (
    DataStore,
    InMemoryDataStore,
    StreamingDataStore,
    StreamingMaterializationError,
    select_storage_backend,
)
from resolve.helpers.iterable_dataset import InMemoryIterableData
from resolve.helpers.normalizer import Normalizer
from resolve.helpers.sampler import Sampler


FEATURE_LABELS = ("theta_value", "phi_value", "unused")
TARGET_LABELS = ("signal",)


def _make_rows(start: int, count: int = 6):
    return [
        {
            "theta_value": float(10 + row_id),
            "phi_value": float(100 + row_id),
            "unused": float(-row_id),
            "signal": float(row_id % 2),
        }
        for row_id in range(start, start + count)
    ]


def _write_csv(path: Path, rows):
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[*FEATURE_LABELS, *TARGET_LABELS],
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_hdf5(
    path: Path,
    rows,
    feature_labels=FEATURE_LABELS,
    target_rows=None,
    target_ndim=2,
):
    target_rows = rows if target_rows is None else target_rows
    feature_values = np.asarray(
        [[row[label] for label in feature_labels] for row in rows],
        dtype=np.float32,
    )
    target_values = np.asarray(
        [[row[label] for label in TARGET_LABELS] for row in target_rows],
        dtype=np.float32,
    )
    if target_ndim == 1:
        target_values = target_values[:, 0]

    with h5py.File(path, "w") as output:
        features = output.create_group("features")
        feature_dataset = features.create_dataset("values", data=feature_values)
        feature_dataset.attrs["labels"] = np.asarray(feature_labels, dtype="S")

        labels = output.create_group("labels")
        target_dataset = labels.create_dataset("values", data=target_values)
        target_dataset.attrs["labels"] = np.asarray(TARGET_LABELS, dtype="S")


def _make_config(data_directory: Path, file_format: str, context_ratio=0.5):
    return {
        "path_settings": {
            "path_to_files_train": str(data_directory),
            "path_to_files_test": str(data_directory),
            "path_to_files_inference": str(data_directory),
        },
        "simulation_settings": {
            "file_format": file_format,
            "theta_labels": ["theta_value"],
            "phi_labels": ["phi_value"],
            "target_labels": ["signal"],
            "signal_condition": ["== 1"],
        },
        "model_settings": {
            "dataloader": {
                "dataloader_number_of_workers": 0,
                "dataloader_prefetch_factor": None,
                "dataloader_pin_memory": False,
                "dataloader_persistent_workers": False,
            },
            "train": {
                "batch_size": 10,
                "dataset": {
                    "seed": 13,
                    "shuffle_dataset": False,
                    "val_ratio": 0.0,
                    "test_ratio": 0.0,
                    "context_ratio": context_ratio,
                    "context_is_subset": False,
                    "mixup_ratio": 0.0,
                    "positive_ratio_train": None,
                    "use_feature_normalization": None,
                },
            },
        },
    }


@pytest.fixture(params=("h5", "csv"))
def loader_case(tmp_path, request):
    file_format = request.param
    data_directory = tmp_path / file_format
    data_directory.mkdir()

    rows = _make_rows(0) + _make_rows(6)
    extension = "h5" if file_format == "h5" else "csv"
    writer = _write_hdf5 if file_format == "h5" else _write_csv
    writer(data_directory / f"part_0.{extension}", rows[:6])
    writer(data_directory / f"part_1.{extension}", rows[6:])

    manager = DataLoaderManager(
        mode="train",
        config_file=_make_config(data_directory, file_format),
    )
    return manager, rows


def test_loader_emits_expected_values_and_batch_contract(loader_case):
    manager, rows = loader_case

    batches = list(manager.set_loader(epoch=0, mode="train"))
    data = manager.dataset.data["data"]

    expected_theta = torch.tensor(
        [[row["theta_value"]] for row in rows],
        dtype=torch.float32,
    )
    expected_phi = torch.tensor(
        [[row["phi_value"]] for row in rows],
        dtype=torch.float32,
    )
    expected_targets = torch.tensor(
        [[row["signal"]] for row in rows],
        dtype=torch.float32,
    )

    torch.testing.assert_close(data["theta"], expected_theta)
    torch.testing.assert_close(data["phi"], expected_phi)
    torch.testing.assert_close(data["y"], expected_targets)
    torch.testing.assert_close(
        data["file_indices"],
        torch.tensor([0] * 6 + [1] * 6),
    )

    assert len(batches) == 2
    assert [batch.query.idx.numel() for batch in batches] == [5, 1]
    assert [batch.context.idx.numel() for batch in batches] == [5, 1]

    for batch in batches:
        context_size = batch.context.idx.numel()
        query_size = batch.query.idx.numel()

        assert type(batch) is BatchCollection
        assert type(batch.context) is ContextSet
        assert type(batch.query) is QuerySet
        assert batch._fields == ("context", "query", "target_y")
        assert batch.context._fields == (
            "theta",
            "phi",
            "y",
            "idx",
            "file_indices",
        )
        assert batch.query._fields == (
            "theta",
            "phi",
            "idx",
            "file_indices",
        )

        assert batch.context.theta.shape == (1, context_size, 1)
        assert batch.context.phi.shape == (1, context_size, 1)
        assert batch.context.y.shape == (1, context_size, 1)
        assert batch.context.idx.shape == (context_size,)
        assert batch.context.file_indices.shape == (1, context_size)

        assert batch.query.theta.shape == (1, query_size, 1)
        assert batch.query.phi.shape == (1, query_size, 1)
        assert batch.query.idx.shape == (query_size,)
        assert batch.query.file_indices.shape == (1, query_size)
        assert batch.target_y.shape == (1, query_size, 1)

        torch.testing.assert_close(
            batch.context.theta,
            data["theta"].index_select(0, batch.context.idx).unsqueeze(0),
        )
        torch.testing.assert_close(
            batch.context.phi,
            data["phi"].index_select(0, batch.context.idx).unsqueeze(0),
        )
        torch.testing.assert_close(
            batch.context.y,
            data["y"].index_select(0, batch.context.idx).unsqueeze(0),
        )
        torch.testing.assert_close(
            batch.context.file_indices,
            data["file_indices"]
            .index_select(0, batch.context.idx)
            .unsqueeze(0),
        )

        torch.testing.assert_close(
            batch.query.theta,
            data["theta"].index_select(0, batch.query.idx).unsqueeze(0),
        )
        torch.testing.assert_close(
            batch.query.phi,
            data["phi"].index_select(0, batch.query.idx).unsqueeze(0),
        )
        torch.testing.assert_close(
            batch.query.file_indices,
            data["file_indices"].index_select(0, batch.query.idx).unsqueeze(0),
        )
        torch.testing.assert_close(
            batch.target_y,
            data["y"].index_select(0, batch.query.idx).unsqueeze(0),
        )

    context_indices = torch.cat([batch.context.idx for batch in batches])
    query_indices = torch.cat([batch.query.idx for batch in batches])
    expected_context_indices = manager.dataset.context_indices("train")
    expected_query_indices = manager.dataset.query_indices("train")

    assert sorted(context_indices.tolist()) == sorted(
        expected_context_indices.tolist()
    )
    assert sorted(query_indices.tolist()) == sorted(expected_query_indices.tolist())
    assert set(context_indices.tolist()).isdisjoint(query_indices.tolist())
    assert sorted(torch.cat((context_indices, query_indices)).tolist()) == list(
        range(12)
    )


def test_normalizer_is_fit_only_on_training_rows(tmp_path):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    rows = _make_rows(0, count=12)
    for row_id, row in enumerate(rows):
        row["theta_value"] = float(2**row_id)
        row["phi_value"] = float(1_000 + row_id**3)
    _write_csv(data_directory / "part_0.csv", rows)

    config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.0,
    )
    dataset_config = config["model_settings"]["train"]["dataset"]
    dataset_config["val_ratio"] = 0.25
    dataset_config["test_ratio"] = 0.25
    dataset_config["use_feature_normalization"] = "zscore"

    manager = DataLoaderManager(mode="train", config_file=config)
    manager.set_dataset()

    train_indices = manager.dataset.query_indices("train")
    raw_theta = np.asarray([row["theta_value"] for row in rows])
    raw_phi = np.asarray([row["phi_value"] for row in rows])
    expected_theta_mean = raw_theta[train_indices.tolist()].mean()
    expected_phi_mean = raw_phi[train_indices.tolist()].mean()
    normalizer = manager.dataset._normalizer
    theta_scaler = normalizer._get_scaler("theta")
    phi_scaler = normalizer._get_scaler("phi")

    assert int(theta_scaler.n_samples_seen_) == train_indices.numel()
    assert int(phi_scaler.n_samples_seen_) == train_indices.numel()
    np.testing.assert_allclose(theta_scaler.mean_, [expected_theta_mean])
    np.testing.assert_allclose(phi_scaler.mean_, [expected_phi_mean])

    expected_theta = torch.from_numpy(
        ((raw_theta - theta_scaler.mean_[0]) / theta_scaler.scale_[0]).astype(
            np.float32
        )
    ).unsqueeze(1)
    expected_phi = torch.from_numpy(
        ((raw_phi - phi_scaler.mean_[0]) / phi_scaler.scale_[0]).astype(np.float32)
    ).unsqueeze(1)
    torch.testing.assert_close(
        manager.dataset.data["data"]["theta"],
        expected_theta,
    )
    torch.testing.assert_close(
        manager.dataset.data["data"]["phi"],
        expected_phi,
    )


def test_hdf5_resolves_columns_by_label_in_each_file(tmp_path):
    data_directory = tmp_path / "h5"
    data_directory.mkdir()
    rows = _make_rows(0) + _make_rows(6)

    _write_hdf5(
        data_directory / "part_0.h5",
        rows[:6],
        feature_labels=("theta_value", "phi_value", "unused"),
    )
    _write_hdf5(
        data_directory / "part_1.h5",
        rows[6:],
        feature_labels=("phi_value", "theta_value", "unused"),
    )

    manager = DataLoaderManager(
        mode="train",
        config_file=_make_config(
            data_directory,
            file_format="h5",
            context_ratio=0.0,
        ),
    )
    manager.set_dataset()

    expected_theta = torch.tensor(
        [[row["theta_value"]] for row in rows],
        dtype=torch.float32,
    )
    expected_phi = torch.tensor(
        [[row["phi_value"]] for row in rows],
        dtype=torch.float32,
    )

    torch.testing.assert_close(manager.dataset.data["data"]["theta"], expected_theta)
    torch.testing.assert_close(manager.dataset.data["data"]["phi"], expected_phi)


def test_hdf5_returns_columns_in_configured_order(tmp_path):
    data_directory = tmp_path / "h5"
    data_directory.mkdir()
    rows = _make_rows(0)
    for row_id, row in enumerate(rows):
        row["theta_second"] = float(1_000 + row_id)
    _write_hdf5(
        data_directory / "part_0.h5",
        rows,
        feature_labels=(
            "theta_value",
            "theta_second",
            "phi_value",
            "unused",
        ),
    )
    config = _make_config(
        data_directory,
        file_format="h5",
        context_ratio=0.0,
    )
    config["simulation_settings"]["theta_labels"] = [
        "theta_second",
        "theta_value",
    ]

    manager = DataLoaderManager(mode="train", config_file=config)
    manager.set_dataset()

    expected_theta = torch.tensor(
        [
            [row["theta_second"], row["theta_value"]]
            for row in rows
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(
        manager.dataset.data["data"]["theta"],
        expected_theta,
    )


@pytest.mark.parametrize(
    ("configured_labels", "error_pattern"),
    [
        (["missing"], "missing requested .* labels"),
        (["theta_value", "theta_value"], "duplicate labels"),
    ],
)
def test_hdf5_rejects_invalid_configured_labels(
    tmp_path,
    configured_labels,
    error_pattern,
):
    data_directory = tmp_path / "h5"
    data_directory.mkdir()
    _write_hdf5(data_directory / "part_0.h5", _make_rows(0))
    config = _make_config(
        data_directory,
        file_format="h5",
        context_ratio=0.0,
    )
    config["simulation_settings"]["theta_labels"] = configured_labels

    with pytest.raises(DataValidationError, match=error_pattern):
        DataLoaderManager(mode="train", config_file=config)


def test_hdf5_rejects_duplicate_schema_labels(tmp_path):
    data_directory = tmp_path / "h5"
    data_directory.mkdir()
    _write_hdf5(
        data_directory / "part_0.h5",
        _make_rows(0),
        feature_labels=("theta_value", "theta_value", "phi_value"),
    )
    with pytest.raises(DataValidationError, match="has duplicate labels"):
        DataLoaderManager(
            mode="train",
            config_file=_make_config(
                data_directory,
                file_format="h5",
                context_ratio=0.0,
            ),
        )


@pytest.mark.parametrize(
    ("mutation", "error_pattern"),
    [
        ("remove_labels", "has no 'labels' attribute"),
        ("remove_dataset", "is missing .*dataset"),
    ],
)
def test_hdf5_rejects_missing_schema_metadata(
    tmp_path,
    mutation,
    error_pattern,
):
    data_directory = tmp_path / "h5"
    data_directory.mkdir()
    file_path = data_directory / "part_0.h5"
    _write_hdf5(file_path, _make_rows(0))
    with h5py.File(file_path, "a") as output:
        if mutation == "remove_labels":
            del output["features/values"].attrs["labels"]
        else:
            del output["labels"]

    with pytest.raises(DataValidationError, match=error_pattern):
        DataLoaderManager(
            mode="train",
            config_file=_make_config(
                data_directory,
                file_format="h5",
                context_ratio=0.0,
            ),
        )


def test_hdf5_reads_one_dimensional_targets_as_columns(tmp_path):
    data_directory = tmp_path / "h5"
    data_directory.mkdir()
    rows = _make_rows(0)
    _write_hdf5(
        data_directory / "part_0.h5",
        rows,
        target_ndim=1,
    )
    manager = DataLoaderManager(
        mode="train",
        config_file=_make_config(
            data_directory,
            file_format="h5",
            context_ratio=0.0,
        ),
    )

    manager.set_dataset()

    expected_targets = torch.tensor(
        [[row["signal"]] for row in rows],
        dtype=torch.float32,
    )
    assert manager.dataset.data["data"]["y"].shape == (len(rows), 1)
    torch.testing.assert_close(
        manager.dataset.data["data"]["y"],
        expected_targets,
    )


def test_hdf5_rejects_inconsistent_row_counts(tmp_path):
    data_directory = tmp_path / "h5"
    data_directory.mkdir()
    rows = _make_rows(0)
    _write_hdf5(
        data_directory / "part_0.h5",
        rows,
        target_rows=rows[:-1],
    )
    with pytest.raises(DataValidationError, match="inconsistent row counts"):
        DataLoaderManager(
            mode="train",
            config_file=_make_config(
                data_directory,
                file_format="h5",
                context_ratio=0.0,
            ),
        )


def test_preflight_aggregates_config_and_multifile_schema_issues(tmp_path):
    data_directory = tmp_path / "h5"
    data_directory.mkdir()
    first_path = data_directory / "part_0.h5"
    second_path = data_directory / "part_1.h5"
    _write_hdf5(first_path, _make_rows(0))
    _write_hdf5(
        second_path,
        _make_rows(6),
        target_rows=_make_rows(6)[:-1],
    )
    with h5py.File(first_path, "a") as output:
        del output["features/values"].attrs["labels"]

    config = _make_config(
        data_directory,
        file_format="h5",
        context_ratio=0.0,
    )
    config["model_settings"]["train"]["dataset"]["context_ratio"] = 1.25
    config["model_settings"]["dataloader"][
        "dataloader_number_of_workers"
    ] = -1

    with pytest.raises(DataValidationError) as error:
        DataLoaderManager(mode="train", config_file=config)

    messages = [str(issue) for issue in error.value.issues]
    assert len(messages) == 4
    assert any("context_ratio" in message for message in messages)
    assert any("dataloader_number_of_workers" in message for message in messages)
    assert any(str(first_path) in message and "labels" in message for message in messages)
    assert any(
        str(second_path) in message and "row counts" in message
        for message in messages
    )


def test_preflight_aggregates_hdf5_dimension_and_dtype_issues(tmp_path):
    data_directory = tmp_path / "h5"
    data_directory.mkdir()
    dimensional_path = data_directory / "part_0.h5"
    dtype_path = data_directory / "part_1.h5"
    _write_hdf5(dimensional_path, _make_rows(0))
    _write_hdf5(dtype_path, _make_rows(6))

    with h5py.File(dimensional_path, "a") as output:
        del output["features/values"]
        dataset = output["features"].create_dataset(
            "values",
            data=np.zeros((6, 3, 1), dtype=np.float32),
        )
        dataset.attrs["labels"] = np.asarray(FEATURE_LABELS, dtype="S")
    with h5py.File(dtype_path, "a") as output:
        del output["labels/values"]
        dataset = output["labels"].create_dataset(
            "values",
            data=np.full((6, 1), "signal", dtype="S8"),
        )
        dataset.attrs["labels"] = np.asarray(TARGET_LABELS, dtype="S")

    with pytest.raises(DataValidationError) as error:
        DataLoaderManager(
            mode="train",
            config_file=_make_config(
                data_directory,
                file_format="h5",
                context_ratio=0.0,
            ),
        )

    message = str(error.value)
    assert str(dimensional_path) in message
    assert "must be 1D or 2D" in message
    assert str(dtype_path) in message
    assert "numeric dtype" in message


def test_preflight_rejects_invalid_public_mode(tmp_path):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0))

    with pytest.raises(DataValidationError) as error:
        DataLoaderManager(
            mode="validate",
            config_file=_make_config(data_directory, "csv"),
        )

    message = str(error.value)
    assert "mode: must be one of" in message
    assert "path_to_files_validate" in message


@pytest.mark.parametrize(
    ("setup", "file_format", "error_pattern"),
    [
        ("missing", "csv", "directory does not exist"),
        ("empty", "csv", "contains no '\\*\\.csv' files"),
        ("empty", "parquet", "file_format"),
    ],
)
def test_preflight_rejects_invalid_source_location_or_format(
    tmp_path,
    setup,
    file_format,
    error_pattern,
):
    data_directory = tmp_path / setup
    if setup == "empty":
        data_directory.mkdir()
    config = _make_config(data_directory, file_format)

    with pytest.raises(DataValidationError, match=error_pattern):
        DataLoaderManager(mode="train", config_file=config)


def test_preflight_aggregates_malformed_worker_settings(tmp_path):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0))
    config = _make_config(data_directory, "csv")
    settings = config["model_settings"]["dataloader"]
    settings["dataloader_number_of_workers"] = "two"
    settings["dataloader_prefetch_factor"] = 0
    settings["dataloader_pin_memory"] = 1
    settings["dataloader_persistent_workers"] = "yes"

    with pytest.raises(DataValidationError) as error:
        DataLoaderManager(mode="train", config_file=config)

    assert len(error.value.issues) == 4
    message = str(error.value)
    assert "dataloader_number_of_workers" in message
    assert "dataloader_prefetch_factor" in message
    assert "dataloader_pin_memory" in message
    assert "dataloader_persistent_workers" in message


def test_preflight_rejects_invalid_ratios_and_mixup_settings(tmp_path):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0))
    config = _make_config(data_directory, "csv")
    settings = config["model_settings"]["train"]["dataset"]
    settings["val_ratio"] = 0.7
    settings["test_ratio"] = 0.4
    settings["mixup_ratio"] = -0.1
    settings["mixup_margin"] = 0.75
    settings["use_beta"] = [0.0, 1.0]
    settings["positive_ratio_train"] = [0.2, 1.0]
    settings["max_positive_reuse"] = -1

    with pytest.raises(DataValidationError) as error:
        DataLoaderManager(mode="train", config_file=config)

    message = str(error.value)
    assert "val_ratio + test_ratio" in message
    assert "mixup_ratio" in message
    assert "mixup_margin" in message
    assert "use_beta" in message
    assert "positive_ratio_train" in message
    assert "max_positive_reuse" in message


def test_preflight_rejects_duplicate_csv_headers(tmp_path):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    (data_directory / "part_0.csv").write_text(
        "theta_value,theta_value,phi_value,signal\n"
        "1,2,3,0\n",
        encoding="utf-8",
    )

    with pytest.raises(DataValidationError, match="duplicate labels"):
        DataLoaderManager(
            mode="train",
            config_file=_make_config(data_directory, "csv"),
        )


@pytest.mark.parametrize("file_format", ("csv", "h5"))
def test_preflight_rejects_files_without_data_rows(tmp_path, file_format):
    data_directory = tmp_path / file_format
    data_directory.mkdir()
    if file_format == "csv":
        (data_directory / "part_0.csv").write_text(
            "theta_value,phi_value,unused,signal\n",
            encoding="utf-8",
        )
    else:
        with h5py.File(data_directory / "part_0.h5", "w") as output:
            features = output.create_group("features").create_dataset(
                "values",
                shape=(0, len(FEATURE_LABELS)),
                dtype=np.float32,
            )
            features.attrs["labels"] = np.asarray(
                FEATURE_LABELS,
                dtype="S",
            )
            targets = output.create_group("labels").create_dataset(
                "values",
                shape=(0, len(TARGET_LABELS)),
                dtype=np.float32,
            )
            targets.attrs["labels"] = np.asarray(
                TARGET_LABELS,
                dtype="S",
            )

    with pytest.raises(DataValidationError, match="contains no data rows"):
        DataLoaderManager(
            mode="train",
            config_file=_make_config(data_directory, file_format),
        )


def test_dataset_rejects_split_that_cannot_leave_training_rows(tmp_path):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0, count=1))
    config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.0,
    )
    config["model_settings"]["train"]["dataset"]["val_ratio"] = 0.5
    manager = DataLoaderManager(mode="train", config_file=config)

    with pytest.raises(
        DataValidationError,
        match=r"dataset split\.validate.*cannot be created",
    ):
        manager.set_dataset()


def test_preflight_does_not_load_csv_values(tmp_path, monkeypatch):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0))

    def fail_if_called(_file_spec):
        raise AssertionError("value loader ran during preflight")

    monkeypatch.setattr(data_readers, "_load_csv_file", fail_if_called)
    manager = DataLoaderManager(
        mode="train",
        config_file=_make_config(data_directory, "csv"),
    )

    assert manager.dataset is None


def test_unchanged_dataset_setup_reuses_preflight_schema(
    tmp_path,
    monkeypatch,
):
    data_directory = tmp_path / "h5"
    data_directory.mkdir()
    _write_hdf5(data_directory / "part_0.h5", _make_rows(0))
    manager = DataLoaderManager(
        mode="train",
        config_file=_make_config(data_directory, "h5"),
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("unchanged schema was inspected again")

    monkeypatch.setattr(
        dataloader_manager_module,
        "preflight_data_loader",
        fail_if_called,
    )

    manager.set_dataset()

    assert manager.dataset.num_samples() == 6


def test_configuration_change_invalidates_cached_preflight(
    tmp_path,
    monkeypatch,
):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0))
    config = _make_config(data_directory, "csv")
    manager = DataLoaderManager(mode="train", config_file=config)
    original_preflight = dataloader_manager_module.preflight_data_loader
    calls = []

    def tracked_preflight(*args, **kwargs):
        calls.append((args, kwargs))
        return original_preflight(*args, **kwargs)

    monkeypatch.setattr(
        dataloader_manager_module,
        "preflight_data_loader",
        tracked_preflight,
    )
    config["model_settings"]["train"]["dataset"]["context_ratio"] = 1.0

    with pytest.raises(DataValidationError, match="context_ratio"):
        manager.set_dataset()

    assert len(calls) == 1


def test_source_change_invalidates_cached_preflight(tmp_path, monkeypatch):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    path = data_directory / "part_0.csv"
    _write_csv(path, _make_rows(0))
    manager = DataLoaderManager(
        mode="train",
        config_file=_make_config(data_directory, "csv"),
    )
    original_preflight = dataloader_manager_module.preflight_data_loader
    calls = []

    def tracked_preflight(*args, **kwargs):
        calls.append((args, kwargs))
        return original_preflight(*args, **kwargs)

    monkeypatch.setattr(
        dataloader_manager_module,
        "preflight_data_loader",
        tracked_preflight,
    )
    _write_csv(path, _make_rows(0, count=7))

    manager.set_dataset()

    assert len(calls) == 1
    assert manager.dataset.num_samples() == 7


def test_csv_value_parse_error_reports_file_row_and_column(tmp_path):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    path = data_directory / "part_0.csv"
    _write_csv(path, _make_rows(0))
    rows = path.read_text(encoding="utf-8").splitlines()
    rows[2] = rows[2].replace("101.0", "not-a-number")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    manager = DataLoaderManager(
        mode="train",
        config_file=_make_config(data_directory, "csv"),
    )

    with pytest.raises(
        ValueError,
        match=r"row 3, column 'phi_value'",
    ):
        manager.set_dataset()


@pytest.mark.parametrize("value", ("", "nan"))
def test_csv_missing_value_reports_file_row_and_column(tmp_path, value):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    path = data_directory / "part_0.csv"
    _write_csv(path, _make_rows(0))
    rows = path.read_text(encoding="utf-8").splitlines()
    rows[2] = rows[2].replace("101.0", value)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    manager = DataLoaderManager(
        mode="train",
        config_file=_make_config(data_directory, "csv"),
    )

    with pytest.raises(
        ValueError,
        match=r"Missing numeric value.*row 3, column 'phi_value'",
    ):
        manager.set_dataset()


@pytest.mark.parametrize("value", ("inf", "-inf", "1e300"))
def test_csv_nonfinite_value_reports_file_row_and_column(tmp_path, value):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    path = data_directory / "part_0.csv"
    _write_csv(path, _make_rows(0))
    rows = path.read_text(encoding="utf-8").splitlines()
    rows[2] = rows[2].replace("101.0", value)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    manager = DataLoaderManager(
        mode="train",
        config_file=_make_config(data_directory, "csv"),
    )

    with pytest.raises(
        ValueError,
        match=r"Non-finite numeric value.*row 3, column 'phi_value'",
    ):
        manager.set_dataset()


@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf, 1e300))
def test_hdf5_nonfinite_value_reports_file_row_and_column(tmp_path, value):
    data_directory = tmp_path / "h5"
    data_directory.mkdir()
    path = data_directory / "part_0.h5"
    _write_hdf5(path, _make_rows(0))
    with h5py.File(path, "a") as output:
        if np.isfinite(value):
            values = output["features/values"][:].astype(np.float64)
            values[1, 1] = value
            del output["features/values"]
            dataset = output["features"].create_dataset(
                "values",
                data=values,
            )
            dataset.attrs["labels"] = np.asarray(
                FEATURE_LABELS,
                dtype="S",
            )
        else:
            output["features/values"][1, 1] = value
    manager = DataLoaderManager(
        mode="train",
        config_file=_make_config(data_directory, "h5"),
    )

    with pytest.raises(
        ValueError,
        match=r"Non-finite numeric value.*row 2, column 'phi_value'",
    ):
        manager.set_dataset()


def test_hdf5_preflight_rejects_complex_dtype(tmp_path):
    data_directory = tmp_path / "h5"
    data_directory.mkdir()
    path = data_directory / "part_0.h5"
    _write_hdf5(path, _make_rows(0))
    with h5py.File(path, "a") as output:
        values = output["features/values"][:].astype(np.complex64)
        del output["features/values"]
        dataset = output["features"].create_dataset("values", data=values)
        dataset.attrs["labels"] = np.asarray(FEATURE_LABELS, dtype="S")

    with pytest.raises(DataValidationError, match="real numeric dtype"):
        DataLoaderManager(
            mode="train",
            config_file=_make_config(data_directory, "h5"),
        )


def test_dataset_replacement_revalidates_before_loading(tmp_path, monkeypatch):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0))
    config = _make_config(data_directory, "csv")
    manager = DataLoaderManager(mode="train", config_file=config)
    manager.set_dataset()
    original_dataset = manager.dataset

    def fail_if_called(_source):
        raise AssertionError("replacement loaded invalid data")

    monkeypatch.setattr(
        data_source.ValidatedDataSource,
        "load",
        fail_if_called,
    )
    config["model_settings"]["train"]["dataset"]["context_ratio"] = 1.0

    with pytest.raises(DataValidationError, match="context_ratio"):
        manager.set_dataset()

    assert manager.dataset is original_dataset
    assert manager.dataset.data is not None


def test_dataset_replacement_preserves_working_loader_on_load_failure(tmp_path):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    path = data_directory / "part_0.csv"
    _write_csv(path, _make_rows(0))
    manager = DataLoaderManager(
        mode="train",
        config_file=_make_config(
            data_directory,
            file_format="csv",
            context_ratio=0.0,
        ),
    )
    original_loader = manager.set_loader(epoch=0)
    original_dataset = manager.dataset
    rows = path.read_text(encoding="utf-8").splitlines()
    rows[2] = rows[2].replace("101.0", "not-a-number")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Failed to parse numeric value"):
        manager.set_dataset()

    assert manager.dataloader is original_loader
    assert manager.dataset is original_dataset
    assert manager.dataset.data is not None
    assert list(manager.dataloader)


def test_warmup_ratio_schedule_remains_deferred_until_scalar_phase(tmp_path):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    rows = _make_rows(0, count=20)
    _write_csv(data_directory / "part_0.csv", rows)
    config = _make_config(data_directory, "csv", context_ratio=0.0)
    config["model_settings"]["train"]["dataset"][
        "positive_ratio_train"
    ] = [0.2, 0.1]

    manager = DataLoaderManager(mode="train", config_file=config)
    manager.set_dataset()

    assert manager.dataset.dataset_config.positive_ratio_train == (0.2, 0.1)
    assert manager.dataset.query_indices("train").numel() == len(
        rows
    )


def test_loader_batch_types_are_canonical_and_dead_apis_are_removed(
    loader_case,
):
    manager, _rows = loader_case
    batch = next(iter(manager.set_loader(epoch=0, mode="train")))

    assert type(batch) is BatchCollection
    assert iterable_dataset_module.BatchCollection is BatchCollection
    assert iterable_dataset_module.ContextSet is ContextSet
    assert iterable_dataset_module.QuerySet is QuerySet
    assert not hasattr(dataloader_manager_module, "BatchCollection")
    assert not hasattr(dataloader_manager_module, "ContextSet")
    assert not hasattr(dataloader_manager_module, "QuerySet")
    assert not hasattr(dataloader_manager_module, "running_average")

    assert "make_empty_like" not in InMemoryIterableData.__dict__
    assert "__iter__" not in InMemoryIterableData.__dict__
    assert "__getitem__" in InMemoryIterableData.__dict__
    assert "set_normalizer" not in InMemoryIterableData.__dict__
    assert not hasattr(manager.dataset, "theta_to_id")
    assert not hasattr(manager.dataset, "nepochs")

    assert not hasattr(Sampler, "get_unique_ids")
    assert not hasattr(Sampler, "to_cell")
    assert not hasattr(Sampler, "_as_key")
    assert not hasattr(Sampler, "build_batches_with_posneg_ratio")
    assert "randperm" not in inspect.signature(Sampler.build_batches).parameters
    assert "unused_neg_subset" not in inspect.signature(
        Sampler.sample_positives_negatives
    ).parameters
    assert "unused_neg_subset" not in inspect.signature(
        Sampler.groupaware_pos_sampling
    ).parameters
    assert not hasattr(Normalizer, "fit_transform_as_f32")


def test_low_level_dataset_requires_validated_data_source():
    with pytest.raises(TypeError, match="ValidatedDataSource"):
        InMemoryIterableData(data_source=[])


def test_storage_configuration_defaults_to_automatic_memory_selection(
    tmp_path,
):
    data_directory = tmp_path / "h5"
    data_directory.mkdir()
    _write_hdf5(data_directory / "part_0.h5", _make_rows(0))
    manager = DataLoaderManager(
        mode="train",
        config_file=_make_config(data_directory, "h5", context_ratio=0.0),
    )

    settings = manager._specification.dataset
    selection = select_storage_backend(manager._data_source, settings)

    assert settings.storage_mode == "auto"
    assert settings.memory_budget_bytes is None
    assert settings.memory_budget_fraction == 0.25
    assert settings.stream_chunk_rows == 65_536
    assert settings.cache_directory is None
    assert selection.backend == "memory"
    assert selection.estimated_peak_bytes > 0
    assert selection.memory_budget_bytes > selection.estimated_peak_bytes


def test_automatic_storage_selection_uses_configured_memory_budget(tmp_path):
    data_directory = tmp_path / "h5"
    data_directory.mkdir()
    _write_hdf5(data_directory / "part_0.h5", _make_rows(0))
    config = _make_config(data_directory, "h5", context_ratio=0.0)
    config["model_settings"]["train"]["dataset"][
        "memory_budget_bytes"
    ] = 1
    manager = DataLoaderManager(mode="train", config_file=config)

    selection = select_storage_backend(
        manager._data_source,
        manager._specification.dataset,
    )

    assert selection.backend == "streaming"
    assert selection.memory_budget_bytes == 1
    assert "exceeds" in selection.reason


def test_batch_wise_storage_rejects_streaming_selection(tmp_path):
    data_directory = tmp_path / "h5"
    data_directory.mkdir()
    _write_hdf5(data_directory / "part_0.h5", _make_rows(0))
    config = _make_config(data_directory, "h5", context_ratio=0.0)
    dataset = config["model_settings"]["train"]["dataset"]
    dataset["shuffle_dataset"] = "batch_wise"
    dataset["memory_budget_bytes"] = 1
    manager = DataLoaderManager(mode="train", config_file=config)

    with pytest.raises(ValueError, match="requires storage_mode='memory'"):
        manager.set_dataset()


@pytest.mark.parametrize(
    ("key", "value", "pattern"),
    [
        ("storage_mode", "disk", "storage_mode"),
        ("memory_budget_bytes", 0, "memory_budget_bytes"),
        ("memory_budget_fraction", 0.0, "memory_budget_fraction"),
        ("stream_chunk_rows", 0, "stream_chunk_rows"),
        ("cache_directory", 12, "cache_directory"),
    ],
)
def test_preflight_rejects_invalid_storage_configuration(
    tmp_path,
    key,
    value,
    pattern,
):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0))
    config = _make_config(data_directory, "csv")
    config["model_settings"]["train"]["dataset"][key] = value

    with pytest.raises(DataValidationError, match=pattern):
        DataLoaderManager(mode="train", config_file=config)


@pytest.mark.parametrize("file_format", ("h5", "csv"))
def test_preallocated_reader_loads_bounded_chunks_without_tensor_cat(
    tmp_path,
    monkeypatch,
    file_format,
):
    data_directory = tmp_path / file_format
    data_directory.mkdir()
    rows = _make_rows(0, count=7)
    extension = "h5" if file_format == "h5" else "csv"
    writer = _write_hdf5 if file_format == "h5" else _write_csv
    writer(data_directory / f"part_0.{extension}", rows)
    config = _make_config(data_directory, file_format, context_ratio=0.0)
    config["model_settings"]["train"]["dataset"]["stream_chunk_rows"] = 2
    manager = DataLoaderManager(mode="train", config_file=config)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("preallocated source loading must not concatenate")

    monkeypatch.setattr(data_readers.torch, "cat", fail_if_called)
    theta, phi, target, file_indices = manager._data_source.load(
        chunk_rows=2
    )

    assert theta.shape == (7, 1)
    assert phi.shape == (7, 1)
    assert target.shape == (7, 1)
    assert file_indices.shape == (7,)
    torch.testing.assert_close(
        theta[:, 0],
        torch.tensor([row["theta_value"] for row in rows]),
    )


def test_hdf5_reader_reads_shared_feature_dataset_once_per_chunk(
    tmp_path,
    monkeypatch,
):
    data_directory = tmp_path / "h5"
    data_directory.mkdir()
    _write_hdf5(data_directory / "part_0.h5", _make_rows(0, count=6))
    config = _make_config(data_directory, "h5", context_ratio=0.0)
    manager = DataLoaderManager(mode="train", config_file=config)
    original_getitem = h5py.Dataset.__getitem__
    feature_reads = []

    def record_getitem(dataset, selection):
        if dataset.name == "/features/values":
            feature_reads.append(selection)
        return original_getitem(dataset, selection)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", record_getitem)
    manager._data_source.load(chunk_rows=2)

    assert len(feature_reads) == 3


def test_in_memory_store_implements_shared_data_store_contract(loader_case):
    manager, _rows = loader_case
    manager.set_dataset()

    assert isinstance(manager.dataset.store, DataStore)
    assert isinstance(manager.dataset.store, InMemoryDataStore)
    assert manager.dataset.store.backend == "memory"
    assert manager.dataset.store.num_samples == manager.dataset.num_samples()
    first = next(manager.dataset.store.iter_chunks(3))
    assert first.start == 0
    assert first.stop == 3
    assert first.theta.shape[0] == 3


def _collect_batches(manager, epoch, mode):
    return [
        {
            "context_idx": batch.context.idx.clone(),
            "query_idx": batch.query.idx.clone(),
            "context_theta": batch.context.theta.clone(),
            "context_phi": batch.context.phi.clone(),
            "context_y": (
                batch.context.y.clone()
                if batch.context.y is not None
                else None
            ),
            "query_theta": batch.query.theta.clone(),
            "query_phi": batch.query.phi.clone(),
            "target_y": (
                batch.target_y.clone()
                if batch.target_y is not None
                else None
            ),
            "context_files": batch.context.file_indices.clone(),
            "query_files": batch.query.file_indices.clone(),
        }
        for batch in manager.set_loader(epoch=epoch, mode=mode)
    ]


def _assert_batch_sequences_equal(first, second):
    assert len(first) == len(second)
    for left, right in zip(first, second):
        assert left.keys() == right.keys()
        for key in left:
            if left[key] is None:
                assert right[key] is None
            else:
                torch.testing.assert_close(left[key], right[key])


@pytest.mark.parametrize("file_format", ("h5", "csv"))
@pytest.mark.parametrize("normalization", (None, "zscore", "minmax"))
def test_streaming_and_memory_backends_emit_identical_batches(
    tmp_path,
    file_format,
    normalization,
):
    data_directory = tmp_path / file_format
    data_directory.mkdir()
    rows = _make_rows(0, count=24)
    extension = "h5" if file_format == "h5" else "csv"
    writer = _write_hdf5 if file_format == "h5" else _write_csv
    writer(data_directory / f"part_0.{extension}", rows[:11])
    writer(data_directory / f"part_1.{extension}", rows[11:])
    base = _make_config(data_directory, file_format, context_ratio=0.3)
    dataset = base["model_settings"]["train"]["dataset"]
    dataset["shuffle_dataset"] = "global"
    dataset["val_ratio"] = 0.2
    dataset["test_ratio"] = 0.2
    dataset["use_feature_normalization"] = normalization
    dataset["stream_chunk_rows"] = 4
    dataset["cache_directory"] = str(tmp_path / "cache")

    memory_config = copy.deepcopy(base)
    memory_config["model_settings"]["train"]["dataset"][
        "storage_mode"
    ] = "memory"
    streaming_config = copy.deepcopy(base)
    streaming_config["model_settings"]["train"]["dataset"][
        "storage_mode"
    ] = "streaming"
    memory = DataLoaderManager("train", memory_config)
    streaming = DataLoaderManager("train", streaming_config)

    for mode in ("train", "validate", "test"):
        memory_batches = _collect_batches(memory, 3, mode)
        streaming_batches = _collect_batches(streaming, 3, mode)
        _assert_batch_sequences_equal(memory_batches, streaming_batches)

    assert memory.normalizer.feature_labels == streaming.normalizer.feature_labels
    for feature_group in ("theta", "phi"):
        left = memory.normalizer.scalers[feature_group]
        right = streaming.normalizer.scalers[feature_group]
        if normalization == "zscore":
            np.testing.assert_allclose(left.mean_, right.mean_)
            np.testing.assert_allclose(left.var_, right.var_)
        elif normalization == "minmax":
            np.testing.assert_allclose(left.data_min_, right.data_min_)
            np.testing.assert_allclose(left.data_max_, right.data_max_)


def test_csv_streaming_cache_is_reused_and_invalidated(tmp_path):
    data_directory = tmp_path / "csv"
    cache_directory = tmp_path / "cache"
    data_directory.mkdir()
    source_path = data_directory / "part_0.csv"
    _write_csv(source_path, _make_rows(0))
    config = _make_config(data_directory, "csv", context_ratio=0.0)
    dataset = config["model_settings"]["train"]["dataset"]
    dataset["storage_mode"] = "streaming"
    dataset["cache_directory"] = str(cache_directory)
    dataset["stream_chunk_rows"] = 2

    first = DataLoaderManager("train", config)
    first.set_dataset()
    first_path = first.dataset.store.cache_path
    first_stat = first_path.stat()
    first.close_loader()

    second = DataLoaderManager("train", config)
    second.set_dataset()
    second_path = second.dataset.store.cache_path

    assert second_path == first_path
    assert second_path.stat().st_mtime_ns == first_stat.st_mtime_ns
    assert not list(cache_directory.glob("*.tmp"))
    second.close_loader()

    _write_csv(source_path, _make_rows(0, count=7))
    third = DataLoaderManager("train", config)
    third.set_dataset()

    assert third.dataset.store.cache_path != first_path
    assert third.dataset.store.cache_path.exists()


def test_streaming_store_refuses_full_materialization(tmp_path):
    data_directory = tmp_path / "h5"
    data_directory.mkdir()
    _write_hdf5(data_directory / "part_0.h5", _make_rows(0))
    config = _make_config(data_directory, "h5", context_ratio=0.0)
    config["model_settings"]["train"]["dataset"][
        "storage_mode"
    ] = "streaming"
    manager = DataLoaderManager("train", config)
    manager.set_dataset()

    assert isinstance(manager.dataset.store, StreamingDataStore)
    with pytest.raises(StreamingMaterializationError, match="complete dataset"):
        manager.dataset.store.materialize()
    with pytest.raises(StreamingMaterializationError, match="get_data"):
        manager.dataset.get_data("train")


def test_streaming_native_hdf5_reads_are_bounded_by_chunk_size(
    tmp_path,
    monkeypatch,
):
    data_directory = tmp_path / "h5"
    data_directory.mkdir()
    _write_hdf5(data_directory / "part_0.h5", _make_rows(0, count=20))
    config = _make_config(data_directory, "h5", context_ratio=0.0)
    dataset = config["model_settings"]["train"]["dataset"]
    dataset["storage_mode"] = "streaming"
    dataset["stream_chunk_rows"] = 4
    manager = DataLoaderManager("train", config)
    manager.set_dataset()
    spans = []
    original = manager.dataset.store._read_bounded_rows

    def tracked(dataset, rows, columns):
        if len(rows):
            buckets = rows // manager.dataset.store.chunk_rows
            for bucket in np.unique(buckets):
                selected = rows[buckets == bucket]
                spans.append(int(selected[-1] - selected[0] + 1))
        return original(dataset, rows, columns)

    monkeypatch.setattr(
        manager.dataset.store,
        "_read_bounded_rows",
        tracked,
    )
    manager.dataset.store.read_rows(
        torch.tensor([19, 0, 7, 8, 3, 7], dtype=torch.long)
    )

    assert spans
    assert max(spans) <= 4


def test_auto_backend_uses_streaming_when_budget_is_exceeded(tmp_path):
    data_directory = tmp_path / "h5"
    data_directory.mkdir()
    _write_hdf5(data_directory / "part_0.h5", _make_rows(0))
    config = _make_config(data_directory, "h5", context_ratio=0.0)
    config["model_settings"]["train"]["dataset"][
        "memory_budget_bytes"
    ] = 1
    manager = DataLoaderManager("train", config)

    batches = list(manager.set_loader(0, "train"))

    assert manager.storage_selection.backend == "streaming"
    assert manager.dataset.store.backend == "streaming"
    assert batches


def test_streaming_external_loader_uses_training_normalizer(tmp_path):
    train_directory = tmp_path / "train"
    test_directory = tmp_path / "test"
    train_directory.mkdir()
    test_directory.mkdir()
    _write_hdf5(train_directory / "part_0.h5", _make_rows(0))
    test_rows = _make_rows(6)
    _write_hdf5(test_directory / "part_0.h5", test_rows)
    config = _make_config(train_directory, "h5", context_ratio=0.0)
    config["path_settings"]["path_to_files_test"] = str(test_directory)
    dataset = config["model_settings"]["train"]["dataset"]
    dataset["use_feature_normalization"] = "zscore"
    dataset["storage_mode"] = "streaming"

    training = DataLoaderManager("train", config)
    training.set_dataset()
    external = DataLoaderManager(
        "test",
        config,
        normalizer=training.normalizer,
    )
    batch = next(iter(external.set_loader(0)))
    expected_theta = (
        torch.tensor(
            [[row["theta_value"]] for row in test_rows],
            dtype=torch.float32,
        )
        - 12.5
    ) / torch.tensor([1.707825127659933])

    torch.testing.assert_close(
        batch.query.theta.squeeze(0),
        expected_theta,
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("file_format", ("h5", "csv"))
def test_streaming_inference_supports_unlabeled_sources(
    tmp_path,
    file_format,
):
    data_directory = tmp_path / file_format
    data_directory.mkdir()
    rows = _make_rows(0)
    if file_format == "h5":
        path = data_directory / "part_0.h5"
        feature_values = np.asarray(
            [[row[label] for label in FEATURE_LABELS] for row in rows],
            dtype=np.float32,
        )
        with h5py.File(path, "w") as output:
            dataset = output.create_group("features").create_dataset(
                "values",
                data=feature_values,
            )
            dataset.attrs["labels"] = np.asarray(FEATURE_LABELS, dtype="S")
    else:
        path = data_directory / "part_0.csv"
        with path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=FEATURE_LABELS)
            writer.writeheader()
            writer.writerows(
                {label: row[label] for label in FEATURE_LABELS}
                for row in rows
            )
    config = _make_config(data_directory, file_format, context_ratio=0.0)
    del config["simulation_settings"]["target_labels"]
    del config["simulation_settings"]["signal_condition"]
    dataset = config["model_settings"]["train"]["dataset"]
    dataset["storage_mode"] = "streaming"
    dataset["cache_directory"] = str(tmp_path / "cache")
    manager = DataLoaderManager("inference", config)

    batch = next(iter(manager.set_loader(0)))

    assert batch.target_y is None
    assert batch.context.y is None
    assert batch.query.theta.shape == (1, len(rows), 1)


def test_streaming_hdf5_reports_malformed_feature_on_indexed_read(tmp_path):
    data_directory = tmp_path / "h5"
    data_directory.mkdir()
    path = data_directory / "part_0.h5"
    _write_hdf5(path, _make_rows(0))
    with h5py.File(path, "a") as output:
        output["features/values"][2, 0] = np.inf
    config = _make_config(data_directory, "h5", context_ratio=0.0)
    config["model_settings"]["train"]["dataset"][
        "storage_mode"
    ] = "streaming"
    manager = DataLoaderManager("train", config)

    with pytest.raises(ValueError, match="Non-finite numeric value"):
        manager.set_dataset()


def test_default_csv_cache_uses_model_output_directory(tmp_path):
    data_directory = tmp_path / "csv"
    output_directory = tmp_path / "model-output"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0))
    config = _make_config(data_directory, "csv", context_ratio=0.0)
    config["path_settings"]["path_out_model"] = str(output_directory)
    config["model_settings"]["train"]["dataset"][
        "storage_mode"
    ] = "streaming"
    manager = DataLoaderManager("train", config)
    manager.set_dataset()

    assert manager.dataset.store.cache_path.parent == (
        output_directory / ".resolve-cache"
    )


def test_main_process_sampler_sends_one_combined_request_per_batch(
    loader_case,
    monkeypatch,
):
    manager, _rows = loader_case
    loader = manager.set_loader(epoch=0, mode="train")
    assert isinstance(loader.sampler, BatchRequestSampler)
    requests = tuple(loader.sampler)
    assert requests
    assert all(isinstance(request, BatchRequest) for request in requests)
    mode_plan = manager.dataset.data["train"]
    assert set(mode_plan).issuperset({"order", "spans", "context", "target"})
    assert "indices" not in mode_plan["context"]
    assert "batches" not in mode_plan["context"]
    assert "indices" not in mode_plan["target"]
    assert "batches" not in mode_plan["target"]
    assert all(
        request.indices.untyped_storage().data_ptr()
        == mode_plan["order"].untyped_storage().data_ptr()
        for request in requests
    )

    calls = []
    original = manager.dataset.store.read_rows

    def tracked(indices):
        calls.append(indices.clone())
        return original(indices)

    monkeypatch.setattr(manager.dataset.store, "read_rows", tracked)
    batches = list(loader)

    assert len(calls) == len(batches) == len(requests)
    for call, request, batch in zip(calls, requests, batches):
        torch.testing.assert_close(call, request.indices)
        assert request.context_size == batch.context.idx.numel()
        if request.context_is_subset:
            torch.testing.assert_close(request.indices, batch.query.idx)
        else:
            torch.testing.assert_close(
                request.indices,
                torch.cat((batch.context.idx, batch.query.idx)),
            )


def test_workers_do_not_rebuild_epoch_plans(tmp_path):
    data_directory = tmp_path / "h5"
    data_directory.mkdir()
    _write_hdf5(data_directory / "part_0.h5", _make_rows(0, count=30))
    config = _make_config(data_directory, "h5", context_ratio=0.25)
    dataset = config["model_settings"]["train"]["dataset"]
    dataset["storage_mode"] = "streaming"
    dataset["shuffle_dataset"] = "global"
    dataloader = config["model_settings"]["dataloader"]
    dataloader["dataloader_number_of_workers"] = 1
    dataloader["dataloader_prefetch_factor"] = 2
    dataloader["dataloader_persistent_workers"] = True
    manager = DataLoaderManager("train", config)
    loader = manager.set_loader(4, "train")
    builds_before_iteration = int(
        manager.dataset._batch_plan_build_count.item()
    )

    batches = list(loader)

    assert batches
    assert int(manager.dataset._batch_plan_build_count.item()) == (
        builds_before_iteration
    )


def test_streaming_persistent_workers_match_zero_worker_order_and_values(
    tmp_path,
):
    data_directory = tmp_path / "h5"
    data_directory.mkdir()
    rows = _make_rows(0, count=40)
    _write_hdf5(data_directory / "part_0.h5", rows[:17])
    _write_hdf5(data_directory / "part_1.h5", rows[17:])
    base = _make_config(data_directory, "h5", context_ratio=0.25)
    dataset = base["model_settings"]["train"]["dataset"]
    dataset["storage_mode"] = "streaming"
    dataset["shuffle_dataset"] = "global"
    dataset["use_feature_normalization"] = "zscore"
    dataset["stream_chunk_rows"] = 8

    zero_config = copy.deepcopy(base)
    workers_config = copy.deepcopy(base)
    worker_settings = workers_config["model_settings"]["dataloader"]
    worker_settings["dataloader_number_of_workers"] = 1
    worker_settings["dataloader_prefetch_factor"] = 2
    worker_settings["dataloader_persistent_workers"] = True
    zero = DataLoaderManager("train", zero_config)
    workers = DataLoaderManager("train", workers_config)

    for epoch in (0, 3):
        expected = _collect_batches(zero, epoch, "train")
        actual = _collect_batches(workers, epoch, "train")
        _assert_batch_sequences_equal(expected, actual)

    loader = workers.dataloader
    worker_processes = tuple(loader._iterator._workers)
    worker_pids = [worker.pid for worker in worker_processes]
    _collect_batches(workers, 5, "train")

    assert [worker.pid for worker in loader._iterator._workers] == worker_pids
    workers.close_loader()
    assert all(not worker.is_alive() for worker in worker_processes)


@pytest.mark.parametrize("injection", ("constructor", "set_dataset"))
def test_external_loader_uses_fitted_training_normalizer(tmp_path, injection):
    train_directory = tmp_path / "train"
    test_directory = tmp_path / "test"
    train_directory.mkdir()
    test_directory.mkdir()
    train_rows = _make_rows(0)
    test_rows = _make_rows(6)
    _write_csv(train_directory / "part_0.csv", train_rows)
    _write_csv(test_directory / "part_0.csv", test_rows)

    config = _make_config(
        train_directory,
        file_format="csv",
        context_ratio=0.0,
    )
    config["path_settings"]["path_to_files_test"] = str(test_directory)
    config["model_settings"]["train"]["dataset"][
        "use_feature_normalization"
    ] = "zscore"

    train_manager = DataLoaderManager(mode="train", config_file=config)
    train_manager.set_dataset()
    training_normalizer = train_manager.normalizer

    if injection == "constructor":
        test_manager = DataLoaderManager(
            mode="test",
            config_file=config,
            normalizer=training_normalizer,
        )
    else:
        test_manager = DataLoaderManager(mode="test", config_file=config)
        test_manager.set_dataset(normalizer=training_normalizer)

    batches = list(test_manager.set_loader(epoch=0, mode="test"))

    raw_theta = np.asarray(
        [[row["theta_value"]] for row in test_rows],
        dtype=np.float32,
    )
    raw_phi = np.asarray(
        [[row["phi_value"]] for row in test_rows],
        dtype=np.float32,
    )
    theta_scaler = training_normalizer._get_scaler("theta")
    phi_scaler = training_normalizer._get_scaler("phi")
    expected_theta = torch.from_numpy(theta_scaler.transform(raw_theta)).float()
    expected_phi = torch.from_numpy(phi_scaler.transform(raw_phi)).float()

    assert batches
    assert test_manager.normalizer is training_normalizer
    torch.testing.assert_close(
        test_manager.dataset.data["data"]["theta"],
        expected_theta,
    )
    torch.testing.assert_close(
        test_manager.dataset.data["data"]["phi"],
        expected_phi,
    )


def test_external_loader_defaults_to_its_manager_mode(tmp_path):
    data_directory = tmp_path / "test"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0))
    manager = DataLoaderManager(
        mode="test",
        config_file=_make_config(
            data_directory,
            file_format="csv",
            context_ratio=0.0,
        ),
    )

    batches = list(manager.set_loader(epoch=0))

    assert batches
    assert manager.dataset.mode == "test"


@pytest.mark.parametrize(
    "test_theta_labels",
    (
        ["theta_b", "theta_a"],
        ["theta_a"],
    ),
)
def test_external_loader_rejects_mismatched_feature_schema(
    tmp_path,
    test_theta_labels,
):
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    rows = [
        {
            "theta_a": float(index),
            "theta_b": float(index + 100),
            "phi_value": float(index + 200),
            "signal": float(index % 2),
        }
        for index in range(12)
    ]
    _write_hdf5(
        data_directory / "part_0.h5",
        rows,
        feature_labels=("theta_a", "theta_b", "phi_value"),
    )
    train_config = _make_config(
        data_directory,
        file_format="h5",
        context_ratio=0.0,
    )
    train_config["simulation_settings"]["theta_labels"] = [
        "theta_a",
        "theta_b",
    ]
    train_config["model_settings"]["train"]["dataset"][
        "use_feature_normalization"
    ] = "zscore"
    train_manager = DataLoaderManager(
        mode="train",
        config_file=train_config,
    )
    train_manager.set_dataset()

    test_config = copy.deepcopy(train_config)
    test_config["simulation_settings"]["theta_labels"] = test_theta_labels

    with pytest.raises(ValueError, match="feature labels"):
        DataLoaderManager(
            mode="test",
            config_file=test_config,
            normalizer=train_manager.normalizer,
        )


@pytest.mark.parametrize("mode", ("test", "inference"))
def test_external_loader_requires_normalizer_when_enabled(tmp_path, mode):
    data_directory = tmp_path / mode
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0))
    config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.0,
    )
    config["model_settings"]["train"]["dataset"][
        "use_feature_normalization"
    ] = "zscore"
    manager = DataLoaderManager(mode=mode, config_file=config)

    with pytest.raises(ValueError, match="requires a fitted training normalizer"):
        manager.set_loader(epoch=0, mode=mode)

    assert manager.dataset is None


def test_external_loader_rejects_unfitted_normalizer(tmp_path):
    data_directory = tmp_path / "test"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0))
    config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.0,
    )
    config["model_settings"]["train"]["dataset"][
        "use_feature_normalization"
    ] = "zscore"

    with pytest.raises(ValueError, match="requires fitted scaler state"):
        DataLoaderManager(
            mode="test",
            config_file=config,
            normalizer=Normalizer("zscore"),
        )


def test_external_loader_rejects_mismatched_normalizer(tmp_path):
    data_directory = tmp_path / "data"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0))
    train_config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.0,
    )
    train_config["model_settings"]["train"]["dataset"][
        "use_feature_normalization"
    ] = "zscore"
    train_manager = DataLoaderManager(mode="train", config_file=train_config)
    train_manager.set_dataset()

    test_config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.0,
    )
    test_config["model_settings"]["train"]["dataset"][
        "use_feature_normalization"
    ] = "minmax"

    with pytest.raises(ValueError, match="does not match configured method"):
        DataLoaderManager(
            mode="test",
            config_file=test_config,
            normalizer=train_manager.normalizer,
        )


def test_training_loader_rejects_injected_normalizer(tmp_path):
    data_directory = tmp_path / "train"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0))
    config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.0,
    )

    with pytest.raises(ValueError, match="create and fit their own normalizer"):
        DataLoaderManager(
            mode="train",
            config_file=config,
            normalizer=Normalizer(),
        )


def test_external_loader_without_normalization_initializes_automatically(tmp_path):
    data_directory = tmp_path / "test"
    data_directory.mkdir()
    rows = _make_rows(0)
    _write_csv(data_directory / "part_0.csv", rows)
    config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.0,
    )
    manager = DataLoaderManager(mode="test", config_file=config)

    batches = list(manager.set_loader(epoch=0, mode="test"))

    expected_theta = torch.tensor(
        [[row["theta_value"]] for row in rows],
        dtype=torch.float32,
    )
    expected_phi = torch.tensor(
        [[row["phi_value"]] for row in rows],
        dtype=torch.float32,
    )
    assert batches
    assert manager.normalizer is not None
    torch.testing.assert_close(
        manager.dataset.data["data"]["theta"],
        expected_theta,
    )
    torch.testing.assert_close(
        manager.dataset.data["data"]["phi"],
        expected_phi,
    )


def test_zero_context_preserves_batch_shapes_and_index_dtypes(tmp_path):
    data_directory = tmp_path / "test"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0))
    manager = DataLoaderManager(
        mode="test",
        config_file=_make_config(
            data_directory,
            file_format="csv",
            context_ratio=0.0,
        ),
    )

    batch = next(iter(manager.set_loader(epoch=0, mode="test")))

    assert batch.context.theta.shape == (1, 0, 1)
    assert batch.context.phi.shape == (1, 0, 1)
    assert batch.context.y.shape == (1, 0, 1)
    assert batch.context.idx.shape == (0,)
    assert batch.context.file_indices.shape == (1, 0)
    assert batch.context.theta.dtype == batch.query.theta.dtype
    assert batch.context.phi.dtype == batch.query.phi.dtype
    assert batch.context.y.dtype == batch.target_y.dtype
    assert batch.context.idx.dtype == batch.query.idx.dtype == torch.long
    assert batch.context.file_indices.dtype == torch.long


@pytest.mark.parametrize("file_format", ("csv", "h5"))
def test_inference_loader_accepts_unlabeled_data(tmp_path, file_format):
    data_directory = tmp_path / file_format
    data_directory.mkdir()
    rows = _make_rows(0)
    if file_format == "csv":
        path = data_directory / "part_0.csv"
        with path.open("w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=FEATURE_LABELS)
            writer.writeheader()
            writer.writerows(
                {
                    label: row[label]
                    for label in FEATURE_LABELS
                }
                for row in rows
            )
    else:
        path = data_directory / "part_0.h5"
        feature_values = np.asarray(
            [[row[label] for label in FEATURE_LABELS] for row in rows],
            dtype=np.float32,
        )
        with h5py.File(path, "w") as output:
            features = output.create_group("features").create_dataset(
                "values",
                data=feature_values,
            )
            features.attrs["labels"] = np.asarray(
                FEATURE_LABELS,
                dtype="S",
            )

    config = _make_config(
        data_directory,
        file_format=file_format,
        context_ratio=0.0,
    )
    del config["simulation_settings"]["target_labels"]
    del config["simulation_settings"]["signal_condition"]
    manager = DataLoaderManager(mode="inference", config_file=config)

    batches = list(manager.set_loader(epoch=0))

    assert batches
    assert "target" not in manager.parameters
    assert manager.dataset.data["data"]["y"] is None
    assert all(batch.target_y is None for batch in batches)
    assert all(batch.context.y is None for batch in batches)
    assert sorted(
        torch.cat([batch.query.idx for batch in batches]).tolist()
    ) == list(range(len(rows)))
    theta, phi, target = manager.dataset.get_data("inference")
    assert theta.shape == (len(rows), 1)
    assert phi.shape == (len(rows), 1)
    assert target is None


def test_unlabeled_inference_rejects_positive_context_ratio(tmp_path):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    path = data_directory / "part_0.csv"
    path.write_text(
        "theta_value,phi_value,unused\n1,2,3\n",
        encoding="utf-8",
    )
    config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.5,
    )
    del config["simulation_settings"]["target_labels"]
    del config["simulation_settings"]["signal_condition"]

    with pytest.raises(DataValidationError, match="must be 0"):
        DataLoaderManager(mode="inference", config_file=config)


@pytest.mark.parametrize("mode", ("train", "test"))
def test_labeled_modes_still_require_target_configuration(tmp_path, mode):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0))
    config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.0,
    )
    del config["simulation_settings"]["target_labels"]

    with pytest.raises(DataValidationError, match="target_labels"):
        DataLoaderManager(mode=mode, config_file=config)


def test_context_ratio_is_preserved_for_small_batches(tmp_path):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    rows = _make_rows(0)
    _write_csv(data_directory / "part_0.csv", rows)
    config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.5,
    )
    config["model_settings"]["train"]["batch_size"] = 4

    manager = DataLoaderManager(mode="train", config_file=config)
    batches = list(manager.set_loader(epoch=0, mode="train"))

    assert manager.dataset.context_ratio == 0.5
    assert all(batch.context.idx.numel() > 0 for batch in batches)


def _index_plan(manager, epoch, mode="train"):
    return [
        (
            batch.context.idx.tolist(),
            batch.query.idx.tolist(),
        )
        for batch in manager.set_loader(epoch=epoch, mode=mode)
    ]


def test_epoch_plan_depends_only_on_seed_and_requested_epoch(tmp_path):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0, count=30))
    config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.3,
    )
    config["model_settings"]["train"]["batch_size"] = 8
    config["model_settings"]["train"]["dataset"]["shuffle_dataset"] = "global"

    sequential = DataLoaderManager(mode="train", config_file=config)
    _index_plan(sequential, 0)
    _index_plan(sequential, 1)
    expected = _index_plan(sequential, 5)

    resumed = DataLoaderManager(mode="train", config_file=config)
    direct = _index_plan(resumed, 5)

    assert direct == expected
    assert _index_plan(resumed, 5) == direct
    assert _index_plan(resumed, 4) != direct


@pytest.mark.parametrize("mode", ("validate", "test"))
def test_evaluation_batch_plan_is_fixed_across_training_epochs(tmp_path, mode):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0, count=40))
    config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.25,
    )
    dataset_config = config["model_settings"]["train"]["dataset"]
    dataset_config["shuffle_dataset"] = "global"
    dataset_config["val_ratio"] = 0.2
    dataset_config["test_ratio"] = 0.2

    manager = DataLoaderManager(mode="train", config_file=config)
    _index_plan(manager, 0, "train")
    expected = _index_plan(manager, 0, mode)
    _index_plan(manager, 7, "train")

    assert _index_plan(manager, 7, mode) == expected


@pytest.mark.parametrize("shuffle", ("global", "batch_wise", False))
def test_paired_batches_cover_uneven_dataset_without_drops(
    tmp_path,
    shuffle,
):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    rows = _make_rows(0, count=23)
    _write_csv(data_directory / "part_0.csv", rows)
    config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.25,
    )
    config["model_settings"]["train"]["batch_size"] = 8
    config["model_settings"]["train"]["dataset"]["shuffle_dataset"] = shuffle
    manager = DataLoaderManager(mode="train", config_file=config)

    batches = list(manager.set_loader(epoch=3, mode="train"))

    assert len(manager.dataset.data["train"]["spans"]) == len(batches)
    assert all(batch.context.idx.numel() > 0 for batch in batches)
    assert all(batch.query.idx.numel() > 0 for batch in batches)

    context_indices = torch.cat([batch.context.idx for batch in batches])
    query_indices = torch.cat([batch.query.idx for batch in batches])
    assert set(context_indices.tolist()).isdisjoint(query_indices.tolist())
    assert sorted(
        torch.cat((context_indices, query_indices)).tolist()
    ) == list(range(len(rows)))


def test_context_is_subset_of_query_when_configured(tmp_path):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    rows = _make_rows(0, count=12)
    _write_csv(data_directory / "part_0.csv", rows)
    config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.5,
    )
    config["model_settings"]["train"]["dataset"]["context_is_subset"] = True
    manager = DataLoaderManager(mode="train", config_file=config)

    batches = list(manager.set_loader(epoch=0, mode="train"))

    for batch in batches:
        assert set(batch.context.idx.tolist()).issubset(
            batch.query.idx.tolist()
        )
        torch.testing.assert_close(
            batch.target_y,
            manager.dataset.data["data"]["y"]
            .index_select(0, batch.query.idx)
            .unsqueeze(0),
        )

    query_indices = torch.cat([batch.query.idx for batch in batches])
    assert sorted(query_indices.tolist()) == list(range(len(rows)))


def test_positive_sampling_plan_is_resume_deterministic(tmp_path):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    rows = _make_rows(0, count=40)
    for row_id, row in enumerate(rows):
        row["signal"] = float(row_id % 10 == 0)
    _write_csv(data_directory / "part_0.csv", rows)
    config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.2,
    )
    dataset_config = config["model_settings"]["train"]["dataset"]
    dataset_config["shuffle_dataset"] = "global"
    dataset_config["positive_ratio_train"] = 0.25
    dataset_config["max_positive_reuse"] = 2

    sequential = DataLoaderManager(mode="train", config_file=config)
    _index_plan(sequential, 0)
    _index_plan(sequential, 1)
    expected = _index_plan(sequential, 4)

    resumed = DataLoaderManager(mode="train", config_file=config)
    assert _index_plan(resumed, 4) == expected
    assert _index_plan(resumed, 3) != expected


def test_zero_worker_loader_normalizes_multiprocessing_options(tmp_path):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0))
    config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.0,
    )
    dataloader_config = config["model_settings"]["dataloader"]
    dataloader_config["dataloader_prefetch_factor"] = 4
    dataloader_config["dataloader_persistent_workers"] = True
    manager = DataLoaderManager(mode="train", config_file=config)

    loader = manager.set_loader(epoch=0, mode="train")

    assert loader.num_workers == 0
    assert loader.prefetch_factor is None
    assert loader.persistent_workers is False
    assert list(loader)


@pytest.mark.parametrize(
    ("configured_pin_memory", "cuda_available"),
    ((False, True), (True, False)),
)
def test_loader_honors_configured_pin_memory(
    tmp_path,
    monkeypatch,
    configured_pin_memory,
    cuda_available,
):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0))
    config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.0,
    )
    config["model_settings"]["dataloader"][
        "dataloader_pin_memory"
    ] = configured_pin_memory
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    manager = DataLoaderManager(mode="train", config_file=config)

    loader = manager.set_loader(epoch=0, mode="train")

    assert loader.pin_memory is configured_pin_memory


def test_loader_instance_is_reused_across_epochs_and_modes(tmp_path):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0, count=30))
    config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.25,
    )
    dataset_config = config["model_settings"]["train"]["dataset"]
    dataset_config["shuffle_dataset"] = "global"
    dataset_config["val_ratio"] = 0.2
    manager = DataLoaderManager(mode="train", config_file=config)

    loader = manager.set_loader(epoch=0, mode="train")
    epoch_zero = list(loader)
    epoch_three_loader = manager.set_loader(epoch=3, mode="train")
    epoch_three = list(epoch_three_loader)
    validation_loader = manager.set_loader(epoch=3, mode="validate")
    validation = list(validation_loader)

    assert epoch_three_loader is loader
    assert validation_loader is loader
    assert [
        batch.query.idx.tolist() for batch in epoch_zero
    ] != [
        batch.query.idx.tolist() for batch in epoch_three
    ]
    assert validation


def test_loader_rejects_plan_changes_during_active_iteration(tmp_path):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    _write_csv(
        data_directory / "part_0.csv",
        _make_rows(0, count=30),
    )
    config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.25,
    )
    config["model_settings"]["train"]["dataset"]["shuffle_dataset"] = "global"
    manager = DataLoaderManager(mode="train", config_file=config)
    loader = manager.set_loader(epoch=0)
    iterator = iter(loader)
    next(iterator)

    with pytest.raises(RuntimeError, match="iterator is active"):
        manager.set_loader(epoch=1)

    iterator.close()
    assert manager.set_loader(epoch=1) is loader
    assert list(loader)


def test_unavailable_mode_does_not_change_current_iteration_state(tmp_path):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    _write_csv(
        data_directory / "part_0.csv",
        _make_rows(0, count=20),
    )
    manager = DataLoaderManager(
        mode="train",
        config_file=_make_config(
            data_directory,
            file_format="csv",
            context_ratio=0.25,
        ),
    )
    loader = manager.set_loader(epoch=2, mode="train")
    expected = [
        batch.query.idx.tolist()
        for batch in loader
    ]
    previous_mode = manager.dataset.mode
    previous_iteration_mode = manager.dataset._iteration_mode.item()
    previous_iteration_epoch = manager.dataset._iteration_epoch.item()

    with pytest.raises(ValueError, match="has no data"):
        manager.set_loader(epoch=3, mode="validate")

    assert manager.dataset.mode == previous_mode
    assert manager.dataset._iteration_mode.item() == previous_iteration_mode
    assert manager.dataset._iteration_epoch.item() == previous_iteration_epoch
    assert len(loader) == len(expected)
    assert [
        batch.query.idx.tolist()
        for batch in loader
    ] == expected


@pytest.mark.parametrize("epoch", (-1, 1.5, "1", True))
def test_loader_rejects_invalid_epochs(loader_case, epoch):
    manager, _rows = loader_case

    with pytest.raises(ValueError, match="non-negative integer"):
        manager.set_loader(epoch=epoch)


def test_loader_rejects_non_boolean_shuffle(loader_case):
    manager, _rows = loader_case

    with pytest.raises(TypeError, match="shuffle must be a boolean"):
        manager.set_loader(epoch=0, shuffle="yes")


def test_shuffle_false_reuses_current_training_plan(tmp_path):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0, count=24))
    config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.25,
    )
    config["model_settings"]["train"]["dataset"]["shuffle_dataset"] = "global"
    manager = DataLoaderManager(mode="train", config_file=config)

    expected = _index_plan(manager, 2)
    actual = [
        (
            batch.context.idx.tolist(),
            batch.query.idx.tolist(),
        )
        for batch in manager.set_loader(
            epoch=9,
            mode="train",
            shuffle=False,
        )
    ]

    assert actual == expected


def test_persistent_worker_observes_epoch_and_mode_updates(tmp_path):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0, count=40))
    config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.25,
    )
    dataset_config = config["model_settings"]["train"]["dataset"]
    dataset_config["shuffle_dataset"] = "global"
    dataset_config["val_ratio"] = 0.2
    dataloader_config = config["model_settings"]["dataloader"]
    dataloader_config["dataloader_number_of_workers"] = 1
    dataloader_config["dataloader_prefetch_factor"] = 2
    dataloader_config["dataloader_persistent_workers"] = True
    manager = DataLoaderManager(mode="train", config_file=config)

    loader = manager.set_loader(epoch=0, mode="train")
    epoch_zero = [
        batch.query.idx.tolist() for batch in loader
    ]
    workers = tuple(loader._iterator._workers)
    worker_pids = [worker.pid for worker in workers]

    same_loader = manager.set_loader(epoch=4, mode="train")
    epoch_four = [
        batch.query.idx.tolist() for batch in same_loader
    ]
    validation = [
        batch.query.idx.tolist()
        for batch in manager.set_loader(epoch=4, mode="validate")
    ]

    direct_config = copy.deepcopy(config)
    direct_dataloader = direct_config["model_settings"]["dataloader"]
    direct_dataloader["dataloader_number_of_workers"] = 0
    direct_dataloader["dataloader_persistent_workers"] = False
    direct = DataLoaderManager(mode="train", config_file=direct_config)
    expected_epoch_four = [
        batch.query.idx.tolist()
        for batch in direct.set_loader(epoch=4, mode="train")
    ]
    expected_validation = [
        batch.query.idx.tolist()
        for batch in direct.set_loader(epoch=4, mode="validate")
    ]

    assert same_loader is loader
    assert [worker.pid for worker in loader._iterator._workers] == worker_pids
    assert epoch_four != epoch_zero
    assert epoch_four == expected_epoch_four
    assert validation == expected_validation

    manager.close_loader()
    assert manager.dataloader is None
    assert manager.dataset is None
    assert all(not worker.is_alive() for worker in workers)
    manager.close_loader()


def test_closing_nonpersistent_iterator_stops_its_workers(tmp_path):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0, count=40))
    config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.0,
    )
    dataloader_config = config["model_settings"]["dataloader"]
    dataloader_config["dataloader_number_of_workers"] = 1
    dataloader_config["dataloader_prefetch_factor"] = 2
    dataloader_config["dataloader_persistent_workers"] = False
    manager = DataLoaderManager(mode="train", config_file=config)
    loader = manager.set_loader(epoch=0, mode="train")
    iterator = iter(loader)
    next(iterator)
    inner_iterator = iterator._iterator
    workers = tuple(inner_iterator._workers)

    assert all(worker.is_alive() for worker in workers)

    iterator.close()

    assert iterator._iterator is None
    assert loader.iteration_active is False
    assert all(not worker.is_alive() for worker in workers)

    next_iterator = iter(manager.set_loader(epoch=1, mode="train"))
    next(next_iterator)
    next_workers = tuple(next_iterator._iterator._workers)
    assert {worker.pid for worker in next_workers}.isdisjoint(
        worker.pid for worker in workers
    )

    next_iterator.close()
    manager.close_loader()
    assert all(not worker.is_alive() for worker in next_workers)


def test_replacing_dataset_invalidates_cached_loader(tmp_path):
    data_directory = tmp_path / "csv"
    data_directory.mkdir()
    _write_csv(data_directory / "part_0.csv", _make_rows(0))
    config = _make_config(
        data_directory,
        file_format="csv",
        context_ratio=0.0,
    )
    manager = DataLoaderManager(mode="train", config_file=config)
    old_loader = manager.set_loader(epoch=0, mode="train")
    old_dataset = manager.dataset

    manager.set_dataset()

    assert manager.dataloader is None
    assert manager.dataset is not old_dataset
    assert old_loader.dataset.data is None

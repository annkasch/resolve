import csv
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from resolve.helpers.dataloader_manager import DataLoaderManager
from resolve.helpers.normalizer import Normalizer


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
    expected_context_indices = manager.dataset.data["train"]["context"]["indices"]
    expected_query_indices = manager.dataset.data["train"]["target"]["indices"]

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

    train_indices = manager.dataset.data["train"]["target"]["indices"]
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
        (["missing"], "missing requested labels"),
        (["theta_value", "theta_value"], "contain duplicates"),
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

    manager = DataLoaderManager(mode="train", config_file=config)

    with pytest.raises(ValueError, match=error_pattern):
        manager.set_dataset()


def test_hdf5_rejects_duplicate_schema_labels(tmp_path):
    data_directory = tmp_path / "h5"
    data_directory.mkdir()
    _write_hdf5(
        data_directory / "part_0.h5",
        _make_rows(0),
        feature_labels=("theta_value", "theta_value", "phi_value"),
    )
    manager = DataLoaderManager(
        mode="train",
        config_file=_make_config(
            data_directory,
            file_format="h5",
            context_ratio=0.0,
        ),
    )

    with pytest.raises(ValueError, match="has duplicate labels"):
        manager.set_dataset()


@pytest.mark.parametrize(
    ("mutation", "error_pattern"),
    [
        ("remove_labels", "has no 'labels' attribute"),
        ("remove_dataset", "is missing dataset"),
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

    manager = DataLoaderManager(
        mode="train",
        config_file=_make_config(
            data_directory,
            file_format="h5",
            context_ratio=0.0,
        ),
    )

    with pytest.raises(ValueError, match=error_pattern):
        manager.set_dataset()


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
    manager = DataLoaderManager(
        mode="train",
        config_file=_make_config(
            data_directory,
            file_format="h5",
            context_ratio=0.0,
        ),
    )

    with pytest.raises(ValueError, match="Inconsistent row counts"):
        manager.set_dataset()


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

    assert len(manager.dataset.data["train"]["context"]["batches"]) == len(
        manager.dataset.data["train"]["target"]["batches"]
    )
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

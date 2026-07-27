import csv
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from resolve.helpers.dataloader_manager import DataLoaderManager


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


def _write_hdf5(path: Path, rows, feature_labels=FEATURE_LABELS):
    feature_values = np.asarray(
        [[row[label] for label in feature_labels] for row in rows],
        dtype=np.float32,
    )
    target_values = np.asarray(
        [[row[label] for label in TARGET_LABELS] for row in rows],
        dtype=np.float32,
    )

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


@pytest.mark.xfail(
    strict=True,
    reason="HDF5 column indices are currently resolved from only the first file",
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


@pytest.mark.xfail(
    strict=True,
    reason="Context ratio precision is derived from the batch-size magnitude",
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

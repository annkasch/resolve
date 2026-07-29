import json
import subprocess
import sys
from pathlib import Path

from resolve.benchmarks.dataloader_workers import choose_worker_defaults
from resolve.benchmarks.dataloader_hybrid import default_scenarios


def _case(workers, persistent, rate):
    return {
        "num_workers": workers,
        "persistent_workers": persistent,
        "prefetch_factor": 2 if workers else None,
        "workloads": {
            "consumer_5ms": {
                "median_batches_per_second": rate,
            }
        },
    }


def test_default_selection_keeps_zero_workers_below_threshold():
    recommendation = choose_worker_defaults(
        [
            _case(0, False, 100.0),
            _case(1, True, 109.0),
            _case(2, True, 108.0),
        ]
    )

    assert recommendation["num_workers"] == 0
    assert recommendation["persistent_workers"] is False
    assert recommendation["prefetch_factor"] is None


def test_default_selection_uses_lowest_worker_count_near_fastest():
    recommendation = choose_worker_defaults(
        [
            _case(0, False, 100.0),
            _case(1, True, 116.0),
            _case(2, True, 120.0),
            _case(4, True, 121.0),
        ]
    )

    assert recommendation["num_workers"] == 1
    assert recommendation["persistent_workers"] is True
    assert recommendation["prefetch_factor"] == 2


def test_benchmark_cli_smoke_run(tmp_path):
    output = tmp_path / "benchmark.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "resolve.benchmarks.dataloader_workers",
            "--rows",
            "32",
            "--features",
            "4",
            "--batch-size",
            "8",
            "--warmups",
            "0",
            "--epochs",
            "1",
            "--consumer-delay-ms",
            "0",
            "--workers",
            "0",
            "--in-process",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["workload"]["rows"] == 32
    assert result["cases"][0]["num_workers"] == 0
    assert result["cases"][0]["workloads"]["loader_only"][
        "batches_per_epoch"
    ] == 4
    assert "Wrote" in completed.stdout


def test_hybrid_benchmark_matrix_covers_required_efficiency_paths():
    scenarios = default_scenarios()
    names = {scenario.name for scenario in scenarios}

    assert {scenario.rows for scenario in scenarios} >= {250_000, 5_000_000}
    assert {scenario.storage_mode for scenario in scenarios} == {
        "memory",
        "streaming",
    }
    assert {scenario.source_format for scenario in scenarios} == {"csv", "h5"}
    assert any(scenario.compressed for scenario in scenarios)
    assert {
        scenario.cache_state
        for scenario in scenarios
        if scenario.source_format == "csv"
    } == {"cold", "hot"}
    assert "streaming-normalized-sampled-mixup" in names
    assert any(scenario.num_workers > 0 for scenario in scenarios)


def test_hybrid_benchmark_cli_smoke_run(tmp_path):
    output = tmp_path / "hybrid.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "resolve.benchmarks.dataloader_hybrid",
            "--quick",
            "--in-process",
            "--rows",
            "32",
            "--large-rows",
            "64",
            "--features",
            "4",
            "--batch-size",
            "8",
            "--warmups",
            "0",
            "--epochs",
            "1",
            "--consumer-delay-ms",
            "0",
            "--historical-baseline",
            "1",
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    names = {
        item["scenario"]["name"]
        for item in result["results"]
    }
    assert names == {
        "memory-hdf5",
        "streaming-hdf5",
        "streaming-csv-cold-cache",
        "streaming-csv-hot-cache",
    }
    assert all(
        item["measurement"]["workloads"]["loader_only"][
            "batches_per_epoch"
        ] == 4
        for item in result["results"]
    )
    assert "Wrote" in completed.stdout


def test_committed_hybrid_benchmark_meets_acceptance_thresholds():
    result_path = (
        Path(__file__).parents[1]
        / "benchmarks"
        / "results"
        / "dataloader_hybrid.json"
    )
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["acceptance"][
        "within_five_percent_fast_path_regression"
    ]
    streaming = [
        item["measurement"]
        for item in result["results"]
        if item["measurement"].get("backend") == "streaming"
    ]
    assert streaming
    assert all(
        measurement["within_streaming_budget_plus_20_percent"]
        for measurement in streaming
    )
    large = next(
        item
        for item in result["results"]
        if item["scenario"]["rows"] == 5_000_000
    )
    assert large["measurement"]["baseline_adjusted_peak_rss_mb"] <= (
        large["measurement"]["memory_budget_mb"] * 1.2
    )

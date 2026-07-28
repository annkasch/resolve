import json
import subprocess
import sys

from resolve.benchmarks.dataloader_workers import choose_worker_defaults


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

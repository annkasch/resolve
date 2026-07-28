from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import platform
import resource
import statistics
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch

from resolve.helpers import DataLoaderManager


@dataclass(frozen=True)
class BenchmarkCase:
    num_workers: int
    persistent_workers: bool
    prefetch_factor: int | None


def choose_worker_defaults(cases: Sequence[dict]) -> dict:
    baseline = next(
        (
            case
            for case in cases
            if case["num_workers"] == 0
        ),
        None,
    )
    if baseline is None:
        raise ValueError("Benchmark results require a zero-worker baseline.")

    baseline_rate = baseline["workloads"]["consumer_5ms"][
        "median_batches_per_second"
    ]
    persistent = [
        case
        for case in cases
        if case["num_workers"] > 0 and case["persistent_workers"]
    ]
    if not persistent:
        return _recommendation(0, baseline_rate, baseline_rate)

    fastest_rate = max(
        case["workloads"]["consumer_5ms"]["median_batches_per_second"]
        for case in persistent
    )
    if fastest_rate < baseline_rate * 1.10:
        return _recommendation(0, baseline_rate, fastest_rate)

    eligible = [
        case
        for case in persistent
        if case["workloads"]["consumer_5ms"]["median_batches_per_second"]
        >= fastest_rate * 0.95
    ]
    selected = min(eligible, key=lambda case: case["num_workers"])
    return _recommendation(
        selected["num_workers"],
        baseline_rate,
        fastest_rate,
    )


def _recommendation(
    num_workers: int,
    baseline_rate: float,
    fastest_rate: float,
) -> dict:
    return {
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "prefetch_factor": 2 if num_workers > 0 else None,
        "baseline_batches_per_second": baseline_rate,
        "fastest_persistent_batches_per_second": fastest_rate,
        "minimum_required_improvement_percent": 10.0,
        "selection_window_percent": 5.0,
    }


def _make_config(
    data_directory: Path,
    theta_labels: Sequence[str],
    phi_labels: Sequence[str],
    batch_size: int,
    seed: int,
    case: BenchmarkCase,
) -> dict:
    return {
        "path_settings": {
            "path_to_files_train": str(data_directory),
            "path_to_files_test": str(data_directory),
            "path_to_files_inference": str(data_directory),
        },
        "simulation_settings": {
            "file_format": "h5",
            "theta_labels": list(theta_labels),
            "phi_labels": list(phi_labels),
            "target_labels": ["target"],
            "signal_condition": ["== 1"],
        },
        "model_settings": {
            "dataloader": {
                "dataloader_number_of_workers": case.num_workers,
                "dataloader_prefetch_factor": case.prefetch_factor,
                "dataloader_pin_memory": False,
                "dataloader_persistent_workers": case.persistent_workers,
            },
            "train": {
                "batch_size": batch_size,
                "dataset": {
                    "seed": seed,
                    "shuffle_dataset": False,
                    "val_ratio": 0.0,
                    "test_ratio": 0.0,
                    "context_ratio": 0.0,
                    "context_is_subset": False,
                    "mixup_ratio": 0.0,
                    "mixup_margin": 0.0,
                    "use_beta": None,
                    "positive_ratio_train": None,
                    "max_positive_reuse": 0,
                    "use_feature_normalization": None,
                },
            },
        },
    }


def _write_synthetic_hdf5(
    path: Path,
    rows: int,
    features: int,
    seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if features < 2:
        raise ValueError("features must be at least 2.")
    theta_count = max(1, features // 3)
    theta_labels = tuple(
        f"theta_{index}" for index in range(theta_count)
    )
    phi_labels = tuple(
        f"phi_{index}" for index in range(features - theta_count)
    )
    feature_labels = (*theta_labels, *phi_labels)
    generator = np.random.default_rng(seed)

    with h5py.File(path, "w") as output:
        features_group = output.create_group("features")
        feature_values = features_group.create_dataset(
            "values",
            shape=(rows, features),
            dtype=np.float32,
            chunks=(min(rows, 16_384), features),
        )
        feature_values.attrs["labels"] = np.asarray(
            feature_labels,
            dtype="S",
        )
        for start in range(0, rows, 16_384):
            stop = min(rows, start + 16_384)
            feature_values[start:stop] = generator.normal(
                size=(stop - start, features)
            ).astype(np.float32)

        labels_group = output.create_group("labels")
        targets = labels_group.create_dataset(
            "values",
            data=generator.integers(
                0,
                2,
                size=(rows, 1),
                dtype=np.int8,
            ),
        )
        targets.attrs["labels"] = np.asarray(["target"], dtype="S")
    return theta_labels, phi_labels


def _consume_epoch(manager, epoch: int, consumer_delay: float) -> dict:
    started = time.perf_counter()
    first_batch_seconds = None
    batches = 0
    for _batch in manager.set_loader(epoch=epoch, mode="train"):
        if first_batch_seconds is None:
            first_batch_seconds = time.perf_counter() - started
        batches += 1
        if consumer_delay:
            time.sleep(consumer_delay)
    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": elapsed,
        "batches": batches,
        "first_batch_seconds": first_batch_seconds,
    }


def _rss_megabytes(who: int) -> float:
    peak = resource.getrusage(who).ru_maxrss
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    return peak / divisor


def _measure_case(
    data_directory: str,
    theta_labels: tuple[str, ...],
    phi_labels: tuple[str, ...],
    batch_size: int,
    seed: int,
    warmups: int,
    epochs: int,
    case: BenchmarkCase,
    consumer_delay: float,
) -> dict:
    config = _make_config(
        Path(data_directory),
        theta_labels,
        phi_labels,
        batch_size,
        seed,
        case,
    )
    manager = DataLoaderManager(mode="train", config_file=config)
    preparation_started = time.perf_counter()
    manager.set_dataset()
    preparation_seconds = time.perf_counter() - preparation_started

    cold_start_seconds = None
    for epoch in range(warmups):
        result = _consume_epoch(manager, epoch, consumer_delay)
        if cold_start_seconds is None:
            cold_start_seconds = result["first_batch_seconds"]

    measured = []
    for epoch in range(warmups, warmups + epochs):
        result = _consume_epoch(manager, epoch, consumer_delay)
        if cold_start_seconds is None:
            cold_start_seconds = result["first_batch_seconds"]
        measured.append(result)

    manager.close_loader()
    epoch_seconds = [item["elapsed_seconds"] for item in measured]
    batch_rates = [
        item["batches"] / item["elapsed_seconds"]
        for item in measured
    ]
    return {
        "dataset_preparation_seconds": preparation_seconds,
        "cold_start_seconds": cold_start_seconds,
        "epoch_seconds": epoch_seconds,
        "total_measured_epoch_seconds": sum(epoch_seconds),
        "median_epoch_seconds": statistics.median(epoch_seconds),
        "median_batches_per_second": statistics.median(batch_rates),
        "batches_per_epoch": measured[0]["batches"],
        "parent_peak_rss_mb": _rss_megabytes(resource.RUSAGE_SELF),
        "child_peak_rss_mb": (
            _rss_megabytes(resource.RUSAGE_CHILDREN)
            if case.num_workers > 0
            else 0.0
        ),
    }


def _case_process(connection, arguments):
    try:
        connection.send({"result": _measure_case(**arguments)})
    except BaseException:
        connection.send({"error": traceback.format_exc()})
    finally:
        connection.close()


def _measure_isolated(arguments: dict) -> dict:
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_case_process,
        args=(child, arguments),
    )
    process.start()
    child.close()
    process.join()
    if not parent.poll():
        raise RuntimeError(
            f"Benchmark subprocess exited with code {process.exitcode} "
            "without returning a result."
        )
    message = parent.recv()
    parent.close()
    if "error" in message:
        raise RuntimeError(message["error"])
    if process.exitcode != 0:
        raise RuntimeError(
            f"Benchmark subprocess exited with code {process.exitcode}."
        )
    return message["result"]


def _benchmark_cases(worker_counts: Sequence[int]) -> list[BenchmarkCase]:
    available = max(1, os.cpu_count() or 1)
    normalized = sorted(
        {0}.union(
            {
                min(max(0, worker_count), available)
                for worker_count in worker_counts
            }
        )
    )
    cases = []
    for worker_count in normalized:
        if worker_count == 0:
            cases.append(BenchmarkCase(0, False, None))
        else:
            cases.extend(
                (
                    BenchmarkCase(worker_count, False, 2),
                    BenchmarkCase(worker_count, True, 2),
                )
            )
    return cases


def run_benchmark(args: argparse.Namespace) -> dict:
    cases = _benchmark_cases(args.workers)
    workloads = (
        ("loader_only", 0.0),
        ("consumer_5ms", args.consumer_delay_ms / 1000.0),
    )
    with tempfile.TemporaryDirectory(
        prefix="resolve-dataloader-benchmark-"
    ) as temporary_directory:
        data_directory = Path(temporary_directory)
        theta_labels, phi_labels = _write_synthetic_hdf5(
            data_directory / "synthetic.h5",
            rows=args.rows,
            features=args.features,
            seed=args.seed,
        )
        results = []
        for case in cases:
            case_result = asdict(case)
            case_result["workloads"] = {}
            for workload_name, consumer_delay in workloads:
                arguments = {
                    "data_directory": str(data_directory),
                    "theta_labels": theta_labels,
                    "phi_labels": phi_labels,
                    "batch_size": args.batch_size,
                    "seed": args.seed,
                    "warmups": args.warmups,
                    "epochs": args.epochs,
                    "case": case,
                    "consumer_delay": consumer_delay,
                }
                measure = (
                    _measure_case(**arguments)
                    if args.in_process
                    else _measure_isolated(arguments)
                )
                case_result["workloads"][workload_name] = measure
            results.append(case_result)

    output = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cpu_count": os.cpu_count(),
        },
        "workload": {
            "rows": args.rows,
            "features": args.features,
            "targets": 1,
            "batch_size": args.batch_size,
            "warmup_epochs": args.warmups,
            "measured_epochs": args.epochs,
            "consumer_delay_ms": args.consumer_delay_ms,
            "seed": args.seed,
        },
        "cases": results,
    }
    output["recommended_defaults"] = choose_worker_defaults(results)
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark RESOLVE DataLoader worker configurations."
    )
    parser.add_argument("--rows", type=int, default=250_000)
    parser.add_argument("--features", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--consumer-delay-ms", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[0, 1, 2, 4],
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/dataloader_workers.json"),
    )
    parser.add_argument(
        "--in-process",
        action="store_true",
        help="Run cases in this process; intended for smoke tests.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.rows <= 0:
        raise ValueError("rows must be greater than zero.")
    if args.features < 2:
        raise ValueError("features must be at least 2.")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be greater than zero.")
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative.")
    if args.epochs <= 0:
        raise ValueError("epochs must be greater than zero.")
    if args.consumer_delay_ms < 0:
        raise ValueError("consumer-delay-ms must be non-negative.")

    result = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    recommendation = result["recommended_defaults"]
    print(
        f"Wrote {args.output}. Recommended num_workers="
        f"{recommendation['num_workers']}, persistent_workers="
        f"{recommendation['persistent_workers']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

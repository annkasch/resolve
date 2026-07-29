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
import pandas as pd
import torch

from resolve.helpers import DataLoaderManager


@dataclass(frozen=True)
class HybridScenario:
    name: str
    rows: int
    source_format: str
    storage_mode: str
    shuffle: str | bool = "global"
    compressed: bool = False
    normalization: str | None = None
    positive_ratio: float | None = None
    mixup_ratio: float = 0.0
    num_workers: int = 0
    persistent_workers: bool = False
    pin_cuda_transfer: bool = False
    cache_state: str | None = None
    warmups_override: int | None = None
    epochs_override: int | None = None


def default_scenarios(
    rows: int = 250_000,
    large_rows: int = 5_000_000,
    *,
    include_cuda: bool = False,
) -> tuple[HybridScenario, ...]:
    scenarios = [
        HybridScenario(
            "memory-hdf5",
            rows,
            "h5",
            "memory",
            shuffle=False,
        ),
        HybridScenario(
            "streaming-hdf5",
            rows,
            "h5",
            "streaming",
            shuffle=False,
        ),
        HybridScenario(
            "streaming-hdf5-workers",
            rows,
            "h5",
            "streaming",
            num_workers=2,
            persistent_workers=True,
        ),
        HybridScenario(
            "streaming-hdf5-compressed",
            rows,
            "h5",
            "streaming",
            compressed=True,
            warmups_override=0,
            epochs_override=1,
        ),
        HybridScenario(
            "streaming-csv-cold-cache",
            rows,
            "csv",
            "streaming",
            cache_state="cold",
            warmups_override=0,
            epochs_override=2,
        ),
        HybridScenario(
            "streaming-csv-hot-cache",
            rows,
            "csv",
            "streaming",
            cache_state="hot",
            warmups_override=1,
            epochs_override=2,
        ),
        HybridScenario(
            "streaming-normalized-sampled-mixup",
            rows,
            "h5",
            "streaming",
            normalization="zscore",
            positive_ratio=0.25,
            mixup_ratio=0.25,
        ),
        HybridScenario(
            "streaming-hdf5-5m",
            large_rows,
            "h5",
            "streaming",
            normalization="zscore",
            warmups_override=0,
            epochs_override=1,
        ),
    ]
    if include_cuda:
        scenarios.append(
            HybridScenario(
                "streaming-hdf5-pinned-cuda",
                rows,
                "h5",
                "streaming",
                num_workers=2,
                persistent_workers=True,
                pin_cuda_transfer=True,
            )
        )
    return tuple(scenarios)


def _labels(features):
    theta_count = max(1, features // 3)
    theta = tuple(f"theta_{index}" for index in range(theta_count))
    phi = tuple(
        f"phi_{index}" for index in range(features - theta_count)
    )
    return theta, phi


def _write_hdf5(path, rows, features, seed, *, compressed):
    theta_labels, phi_labels = _labels(features)
    labels = (*theta_labels, *phi_labels)
    generator = np.random.default_rng(seed)
    compression = "gzip" if compressed else None
    with h5py.File(path, "w") as output:
        feature_values = output.create_group("features").create_dataset(
            "values",
            shape=(rows, features),
            dtype=np.float32,
            chunks=(min(rows, 65_536), features),
            compression=compression,
            compression_opts=1 if compressed else None,
        )
        feature_values.attrs["labels"] = np.asarray(labels, dtype="S")
        target_values = output.create_group("labels").create_dataset(
            "values",
            shape=(rows, 1),
            dtype=np.int8,
            chunks=(min(rows, 65_536), 1),
            compression=compression,
            compression_opts=1 if compressed else None,
        )
        target_values.attrs["labels"] = np.asarray(["target"], dtype="S")
        for start in range(0, rows, 65_536):
            stop = min(start + 65_536, rows)
            feature_values[start:stop] = generator.normal(
                size=(stop - start, features)
            ).astype(np.float32)
            target_values[start:stop] = generator.integers(
                0,
                2,
                size=(stop - start, 1),
                dtype=np.int8,
            )
    return theta_labels, phi_labels


def _write_csv(path, rows, features, seed):
    theta_labels, phi_labels = _labels(features)
    labels = (*theta_labels, *phi_labels)
    generator = np.random.default_rng(seed)
    first = True
    for start in range(0, rows, 65_536):
        stop = min(start + 65_536, rows)
        values = generator.normal(
            size=(stop - start, features)
        ).astype(np.float32)
        frame = pd.DataFrame(values, columns=labels)
        frame["target"] = generator.integers(
            0,
            2,
            size=stop - start,
            dtype=np.int8,
        )
        frame.to_csv(path, mode="w" if first else "a", header=first, index=False)
        first = False
    return theta_labels, phi_labels


def _make_config(
    scenario,
    data_directory,
    cache_directory,
    theta_labels,
    phi_labels,
    batch_size,
    seed,
):
    return {
        "path_settings": {
            "path_to_files_train": str(data_directory),
            "path_to_files_test": str(data_directory),
            "path_to_files_inference": str(data_directory),
            "path_out_model": str(data_directory / "out"),
        },
        "simulation_settings": {
            "file_format": scenario.source_format,
            "theta_labels": list(theta_labels),
            "phi_labels": list(phi_labels),
            "target_labels": ["target"],
            "signal_condition": ["== 1"],
        },
        "model_settings": {
            "dataloader": {
                "dataloader_number_of_workers": scenario.num_workers,
                "dataloader_prefetch_factor": (
                    2 if scenario.num_workers else None
                ),
                "dataloader_pin_memory": scenario.pin_cuda_transfer,
                "dataloader_persistent_workers": (
                    scenario.persistent_workers
                ),
            },
            "train": {
                "batch_size": batch_size,
                "dataset": {
                    "seed": seed,
                    "shuffle_dataset": scenario.shuffle,
                    "val_ratio": 0.0,
                    "test_ratio": 0.0,
                    "context_ratio": 0.0,
                    "context_is_subset": False,
                    "mixup_ratio": scenario.mixup_ratio,
                    "mixup_margin": 0.0,
                    "use_beta": [0.4, 0.8],
                    "positive_ratio_train": scenario.positive_ratio,
                    "max_positive_reuse": 2,
                    "use_feature_normalization": scenario.normalization,
                    "storage_mode": scenario.storage_mode,
                    "memory_budget_bytes": 512 * 1024**2,
                    "stream_chunk_rows": 65_536,
                    "cache_directory": str(cache_directory),
                },
            },
        },
    }


def _peak_rss_mb(who=resource.RUSAGE_SELF):
    peak = resource.getrusage(who).ru_maxrss
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    return peak / divisor


def _consume(manager, epoch, delay, cuda_transfer):
    started = time.perf_counter()
    cpu_started = time.process_time()
    first_batch = None
    batches = 0
    for batch in manager.set_loader(epoch, "train"):
        if first_batch is None:
            first_batch = time.perf_counter() - started
        if cuda_transfer:
            batch.query.theta.to("cuda", non_blocking=True)
            batch.query.phi.to("cuda", non_blocking=True)
        if delay:
            time.sleep(delay)
        batches += 1
    if cuda_transfer:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "elapsed_seconds": elapsed,
        "cpu_seconds": time.process_time() - cpu_started,
        "first_batch_seconds": first_batch,
        "batches": batches,
    }


def _measure(
    scenario,
    data_directory,
    cache_directory,
    theta_labels,
    phi_labels,
    batch_size,
    seed,
    warmups,
    epochs,
    consumer_delay_ms,
):
    if scenario.pin_cuda_transfer and not torch.cuda.is_available():
        return {"skipped": "CUDA is unavailable"}
    baseline_rss = _peak_rss_mb()
    config = _make_config(
        scenario,
        Path(data_directory),
        Path(cache_directory),
        theta_labels,
        phi_labels,
        batch_size,
        seed,
    )
    manager = DataLoaderManager("train", config)
    cpu_started = time.process_time()
    started = time.perf_counter()
    manager.set_dataset()
    preparation_seconds = time.perf_counter() - started
    preparation_cpu_seconds = time.process_time() - cpu_started
    selection = manager.storage_selection

    workloads = {}
    for name, delay in (
        ("loader_only", 0.0),
        ("consumer_5ms", consumer_delay_ms / 1000.0),
    ):
        for epoch in range(warmups):
            _consume(
                manager,
                epoch,
                delay,
                scenario.pin_cuda_transfer,
            )
        measured = [
            _consume(
                manager,
                warmups + epoch,
                delay,
                scenario.pin_cuda_transfer,
            )
            for epoch in range(epochs)
        ]
        elapsed = [item["elapsed_seconds"] for item in measured]
        workloads[name] = {
            "first_batch_seconds": measured[0]["first_batch_seconds"],
            "epoch_seconds": elapsed,
            "median_epoch_seconds": statistics.median(elapsed),
            "median_batches_per_second": statistics.median(
                item["batches"] / item["elapsed_seconds"]
                for item in measured
            ),
            "median_cpu_seconds": statistics.median(
                item["cpu_seconds"] for item in measured
            ),
            "batches_per_epoch": measured[0]["batches"],
        }

    manager.close_loader()
    peak_rss = _peak_rss_mb()
    child_peak = (
        _peak_rss_mb(resource.RUSAGE_CHILDREN)
        if scenario.num_workers
        else 0.0
    )
    adjusted_peak = max(0.0, peak_rss - baseline_rss)
    budget_mb = selection.memory_budget_bytes / 1024**2
    return {
        "backend": selection.backend,
        "estimated_peak_mb": selection.estimated_peak_bytes / 1024**2,
        "memory_budget_mb": budget_mb,
        "preparation_seconds": preparation_seconds,
        "preparation_cpu_seconds": preparation_cpu_seconds,
        "baseline_rss_mb": baseline_rss,
        "parent_peak_rss_mb": peak_rss,
        "baseline_adjusted_peak_rss_mb": adjusted_peak,
        "child_peak_rss_mb": child_peak,
        "within_streaming_budget_plus_20_percent": (
            adjusted_peak <= budget_mb * 1.2
            if selection.backend == "streaming"
            else None
        ),
        "workloads": workloads,
    }


def _child(connection, arguments):
    try:
        connection.send({"result": _measure(**arguments)})
    except BaseException:
        connection.send({"error": traceback.format_exc()})
    finally:
        connection.close()


def _isolated(arguments):
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe(duplex=False)
    process = context.Process(target=_child, args=(child, arguments))
    process.start()
    child.close()
    process.join()
    if not parent.poll():
        raise RuntimeError(
            f"Scenario exited with {process.exitcode} without a result."
        )
    message = parent.recv()
    parent.close()
    if "error" in message:
        raise RuntimeError(message["error"])
    return message["result"]


def run_benchmark(args):
    scenarios = list(
        default_scenarios(
            args.rows,
            args.large_rows,
            include_cuda=args.cuda_transfer,
        )
        if not args.quick
        else (
            HybridScenario(
                "memory-hdf5",
                args.rows,
                "h5",
                "memory",
                shuffle=False,
            ),
            HybridScenario(
                "streaming-hdf5",
                args.rows,
                "h5",
                "streaming",
                shuffle=False,
            ),
            HybridScenario(
                "streaming-csv-cold-cache",
                args.rows,
                "csv",
                "streaming",
                cache_state="cold",
            ),
            HybridScenario(
                "streaming-csv-hot-cache",
                args.rows,
                "csv",
                "streaming",
                cache_state="hot",
            ),
        )
    )
    if args.scenario:
        requested = set(args.scenario)
        available = {scenario.name for scenario in scenarios}
        missing = requested - available
        if missing:
            raise ValueError(
                f"Unknown scenarios {sorted(missing)}; available: "
                f"{sorted(available)}."
            )
        scenarios = [
            scenario
            for scenario in scenarios
            if scenario.name in requested
        ]
    with tempfile.TemporaryDirectory(
        prefix="resolve-hybrid-benchmark-"
    ) as temporary:
        root = Path(temporary)
        sources = {}
        results = []
        for scenario in scenarios:
            source_key = (
                scenario.rows,
                scenario.source_format,
                scenario.compressed,
            )
            if source_key not in sources:
                source_directory = root / (
                    f"{scenario.source_format}-{scenario.rows}-"
                    f"{'compressed' if scenario.compressed else 'plain'}"
                )
                source_directory.mkdir()
                suffix = scenario.source_format
                path = source_directory / f"synthetic.{suffix}"
                writer = (
                    _write_csv
                    if scenario.source_format == "csv"
                    else _write_hdf5
                )
                if scenario.source_format == "csv":
                    labels = writer(
                        path,
                        scenario.rows,
                        args.features,
                        args.seed,
                    )
                else:
                    labels = writer(
                        path,
                        scenario.rows,
                        args.features,
                        args.seed,
                        compressed=scenario.compressed,
                    )
                sources[source_key] = (source_directory, labels)
            source_directory, labels = sources[source_key]
            cache_directory = root / (
                f"cache-{scenario.rows}-{scenario.source_format}"
            )
            arguments = {
                "scenario": scenario,
                "data_directory": str(source_directory),
                "cache_directory": str(cache_directory),
                "theta_labels": labels[0],
                "phi_labels": labels[1],
                "batch_size": args.batch_size,
                "seed": args.seed,
                "warmups": (
                    args.warmups
                    if scenario.warmups_override is None
                    else scenario.warmups_override
                ),
                "epochs": (
                    args.epochs
                    if scenario.epochs_override is None
                    else scenario.epochs_override
                ),
                "consumer_delay_ms": args.consumer_delay_ms,
            }
            measurement = (
                _measure(**arguments)
                if args.in_process
                else _isolated(arguments)
            )
            results.append(
                {"scenario": asdict(scenario), "measurement": measurement}
            )

    memory = next(
        (
            item
            for item in results
            if item["scenario"]["name"] == "memory-hdf5"
        ),
        None,
    )
    fast_rate = (
        memory["measurement"]["workloads"]["loader_only"][
            "median_batches_per_second"
        ]
        if memory is not None
        else None
    )
    return {
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
            "cuda_available": torch.cuda.is_available(),
        },
        "settings": {
            "rows": args.rows,
            "large_rows": args.large_rows,
            "features": args.features,
            "batch_size": args.batch_size,
            "warmups": args.warmups,
            "epochs": args.epochs,
            "consumer_delay_ms": args.consumer_delay_ms,
            "seed": args.seed,
        },
        "acceptance": (
            {
                "memory_fast_path_batches_per_second": fast_rate,
                "historical_baseline_batches_per_second": (
                    args.historical_baseline
                ),
                "memory_fast_path_regression_percent": (
                    100.0
                    * (args.historical_baseline - fast_rate)
                    / args.historical_baseline
                ),
                "within_five_percent_fast_path_regression": (
                    fast_rate >= args.historical_baseline * 0.95
                ),
            }
            if fast_rate is not None
            else None
        ),
        "results": results,
    }


def _parser():
    parser = argparse.ArgumentParser(
        description="Benchmark RESOLVE memory and streaming dataloaders."
    )
    parser.add_argument("--rows", type=int, default=250_000)
    parser.add_argument("--large-rows", type=int, default=5_000_000)
    parser.add_argument("--features", type=int, default=14)
    parser.add_argument("--batch-size", type=int, default=1_000)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=7)
    parser.add_argument("--consumer-delay-ms", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--historical-baseline",
        type=float,
        default=17_946.0,
    )
    parser.add_argument("--cuda-transfer", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--in-process", action="store_true")
    parser.add_argument(
        "--scenario",
        action="append",
        help="Run only the named scenario; repeat to select multiple.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Replace selected scenarios in an existing output file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/dataloader_hybrid.json"),
    )
    return parser


def main(argv: Sequence[str] | None = None):
    args = _parser().parse_args(argv)
    for name in ("rows", "large_rows", "features", "batch_size", "epochs"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive.")
    if args.features < 2:
        raise ValueError("features must be at least 2.")
    if args.warmups < 0:
        raise ValueError("warmups must be non-negative.")
    result = run_benchmark(args)
    if args.resume and args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        merged = {
            item["scenario"]["name"]: item
            for item in previous.get("results", [])
        }
        merged.update(
            {
                item["scenario"]["name"]: item
                for item in result["results"]
            }
        )
        ordered_names = [
            scenario.name
            for scenario in default_scenarios(
                args.rows,
                args.large_rows,
                include_cuda=args.cuda_transfer,
            )
        ]
        result["results"] = [
            merged[name] for name in ordered_names if name in merged
        ]
        memory = next(
            item
            for item in result["results"]
            if item["scenario"]["name"] == "memory-hdf5"
        )
        fast_rate = memory["measurement"]["workloads"]["loader_only"][
            "median_batches_per_second"
        ]
        result["acceptance"] = {
            "memory_fast_path_batches_per_second": fast_rate,
            "historical_baseline_batches_per_second": (
                args.historical_baseline
            ),
            "memory_fast_path_regression_percent": (
                100.0
                * (args.historical_baseline - fast_rate)
                / args.historical_baseline
            ),
            "within_five_percent_fast_path_regression": (
                fast_rate >= args.historical_baseline * 0.95
            ),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.output}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# DataLoader Worker Benchmark

This benchmark compares RESOLVE's in-memory iterable dataset with zero and
multiple PyTorch DataLoader workers. It measures both raw loader throughput and
throughput while a simulated consumer spends 5 ms on each batch.

Run the committed workload from the repository root:

```bash
python -m resolve.benchmarks.dataloader_workers
```

The default run uses 250,000 synthetic rows, 14 features, one target, batches
of 1,000, two warmup epochs, and seven measured epochs. Worker counts are
`0`, `1`, `2`, and `4`, capped by the available CPUs. Multiprocessing cases
run both with and without persistent workers.

Each case runs in an isolated process. The JSON result includes dataset
preparation time, cold time to the first batch, per-epoch duration, median
batches per second, total measured time, and peak RSS for the loader parent
and worker children.

The recommended default uses the persistent-worker results from the simulated
consumer workload. Multiprocessing must improve throughput over zero workers
by at least 10%. When it does, the lowest worker count within 5% of the fastest
result is selected. Timing thresholds are intentionally not enforced in CI.

## Committed Result

The committed result was generated on Darwin arm64 with 10 logical CPUs,
Python 3.11.13, NumPy 1.26.4, and PyTorch 2.9.0. Zero workers reached a median
156.2 batches/s with the 5 ms consumer. The fastest persistent configuration
used two workers and reached 133.1 batches/s, so it did not clear the 10%
improvement threshold.

The example configurations therefore use:

```yaml
dataloader_number_of_workers: 0
dataloader_prefetch_factor: null
dataloader_persistent_workers: false
```

See `results/dataloader_workers.json` for all raw epoch timings and RSS values.

## Hybrid Storage Benchmark

Run the complete memory/streaming matrix with:

```bash
python -m resolve.benchmarks.dataloader_hybrid
```

The matrix covers 250,000-row memory and streaming paths, native compressed
and uncompressed HDF5, cold and hot CSV caches, persistent workers,
normalization, positive sampling, batch-local mixup, and a 5-million-row
streaming workload. Add `--cuda-transfer` to include pinned, non-blocking CUDA
transfer when CUDA is available.

Each scenario runs in an isolated process and reports preparation wall/CPU
time, first-batch latency, steady epoch throughput, baseline-adjusted parent
peak RSS, child peak RSS, and the configured streaming-budget acceptance
check. The output also compares the in-memory zero-worker fast path with the
historical pre-refactor rate and records whether it remains within the 5%
regression allowance.

The fast-path and worker cases use the configured warmup and measured epoch
counts. Compressed HDF5 and the 5-million-row case use one measured epoch, and
CSV cache cases use two, because exact global random access makes those cases
orders of magnitude slower. The per-scenario overrides are recorded in the
JSON rather than hidden from the comparison.

For a small functional run suitable for development:

```bash
python -m resolve.benchmarks.dataloader_hybrid \
  --quick --rows 1000 --large-rows 2000 --warmups 0 --epochs 1
```

Timing thresholds remain excluded from CI because they depend on the host.
Committed full results are stored in `results/dataloader_hybrid.json`.

The committed Darwin arm64 run passed both acceptance checks:

- The 250k-row in-memory zero-worker fast path reached about 18,498
  batches/second, 3.1% faster than the historical 17,946 batches/second
  baseline.
- Every streaming case stayed within its configured 512 MiB budget plus 20%.
  The 5-million-row normalized case added about 312 MiB of peak RSS and
  sustained about 49 batches/second with exact row-level global shuffling.

Uncompressed sequential HDF5 streaming reached about 1,462 batches/second.
Compressed random-access HDF5 fell to about 8.8 batches/second, confirming
that compression is unsuitable for this shuffle pattern. CSV cache preparation
dropped from about 1.05 seconds cold to 0.67 seconds hot on the 250k workload.

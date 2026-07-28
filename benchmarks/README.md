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

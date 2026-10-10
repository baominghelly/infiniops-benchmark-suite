# Unified benchmark runner guide

`scripts/run_benchmark.sh` is the single public entry point for all supported
accelerator platforms. It selects a platform adapter from the requested device,
applies the shared operator-data delivery contract, and forwards all remaining
arguments to that platform's runner.

The deployed shared entry point is:

```text
/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh
```

## Device selection

Pass the device as the first argument:

```bash
/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh metax-c550
```

Or set `DEVICE` in the environment:

```bash
DEVICE=metax-c550 \
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k \
/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh
```

Additional arguments are forwarded to the selected platform runner. Therefore
an explicit device index can still be passed after the device:

```bash
/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh nvidia-a100 3
```

If `DEVICE` is set, the first non-option argument is not consumed as a device
and is forwarded directly:

```bash
DEVICE=nvidia-a100 \
/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh 3
```

List supported devices:

```bash
/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh --list
```

Resolve and print the selected adapter and output contract without launching a
benchmark:

```bash
/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh --print metax-c550
```

## Supported devices

| Canonical device | Accepted aliases | Platform runner | Output directory |
|---|---|---|---|
| `nvidia-a100` | `nvidia`, `a100` | `run_a100_benchmark.sh` | `/data-aisoft/infinibench_data/ops/nvidia-a100/` |
| `metax-c550` | `metax`, `c550` | `run_metax_benchmark.sh` | `/data-aisoft/infinibench_data/ops/metax-c550/` |
| `ascend-910c` | `ascend`, `910c` | `run_ascend_benchmark.sh` | `/data-aisoft/infinibench_data/ops/ascend-910c/` |
| `moore-s5000` | `moore` | `run_moore_benchmark.sh` | `/data-aisoft/infinibench_data/ops/moore-s5000/` |
| `iluvatar-tg150` | `iluvatar`, `tianshu` | `run_iluvatar_benchmark.sh` | `/data-aisoft/infinibench_data/ops/iluvatar-tg150/` |
| `cambricon-mlu590` | `cambricon`, `mlu` | `run_cambricon_benchmark.sh` | `/data-aisoft/infinibench_data/ops/cambricon-mlu590/` |
| `hygon-bw1000` | `hygon`, `dcu` | `run_hygon_benchmark.sh` | `/data-aisoft/infinibench_data/ops/hygon-bw1000/` |

The shared entry point exports the canonical `DEVICE_ID` and:

```text
OUTPUT_ROOT=/data-aisoft/infinibench_data/ops
```

Platform runners own their container names, source roots, build caches, GPU
monitoring commands, runtime preflight, and compilation flags. This keeps one
stable public entry point while avoiding an unsafe monolithic script that mixes
seven incompatible hardware toolchains.

Every platform runner performs a container preflight before acquiring its lock
or creating output files. If its existing benchmark container is in the
`created` or `exited` state, the runner starts it and waits for Docker to report
it running. A missing container, a non-runnable state, or a failed start is a
hard error. Set `AUTO_START_CONTAINER=0` when diagnosing host/container state to
restore fail-fast behavior without attempting `docker start`.

## Common workflow

Normal run:

```bash
DEVICE=metax-c550 \
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k \
/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh
```

Backfill run:

```bash
DEVICE=moore-s5000 \
BACKFILL_DATE=2026-09-04 \
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k \
/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh
```

Development quick check:

```bash
DEVICE=moore-s5000 \
BENCHMARK_ITERATIONS=1 \
WORKER_BATCH_SIZE=8 \
WORKER_BATCH_TIMEOUT=240 \
OUTPUT_ROOT=/data-aisoft/dev_infinibench_data \
DTYPES=float32 \
PROFILES=decode-4k \
/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh
```

`BENCHMARK_ITERATIONS` selects fixed-count timing. The default `0` keeps the
normal adaptive timing behavior. Set it to `1` for development smoke testing:
each successful InfiniOps and PyTorch timing call is executed once. These
latencies are not comparable with adaptive benchmark results and must not be
loaded into the production latency dashboard. The fixed count is recorded in
`results.json` as `benchmark_iterations`.

`WORKER_BATCH_SIZE` groups that many operator wrappers into one worker process
to reduce Python/container startup cost during development validation; it is
currently honored by the MetaX runner. The default `1` preserves
one-process-per-operator isolation for formal runs. Use
`WORKER_BATCH_TIMEOUT` to override a batch timeout. Its default is 20 seconds
multiplied by the batch size, so the default batch size of one preserves the
formal per-operator timeout. Batch mode currently records one result per wrapper, but a fatal
worker failure can affect the wrappers in that batch, so production comparison
runs should retain the default.

`MIN_RUN_TIME` is the minimum measurement time in seconds for each successful
InfiniOps or PyTorch timing. The default remains `0.02` for comparable full
runs and is ignored when `BENCHMARK_ITERATIONS` is greater than zero. Do not
mix fixed-count or shortened measurements into the same latency comparison as
standard runs. The effective values are recorded in each `results.json` as
`min_run_time_s` and `benchmark_iterations`, and the complete harness command
is recorded in `benchmark-launcher.log`.

`OUTPUT_ROOT` is the delivery root. The platform runner appends
`<device>/<run-date>`, so the example above writes to:

```text
/data-aisoft/dev_infinibench_data/moore-s5000/<execution-date>/
```

For all current runners, a normal output directory is:

```text
/data-aisoft/infinibench_data/ops/<device>/<execution-date>/
```

A backfill output directory is:

```text
/data-aisoft/infinibench_data/ops/<device>/backfill-<target-date>-<execution-date>/
```

Directory names do not include the time of day. The complete
`EXECUTION_TIME=YYYYMMDD-HHMMSS` remains in the runner logs and `benchmark.done`
metadata. A non-empty deterministic run directory is rejected by default;
pass `REUSE_RUN_DIR=1` only when replacing a failed or superseded attempt.

Every exported operator row includes two provenance columns:

```text
infiniops_commit
infinirt_commit
```

The values are the full Git commit IDs used by the platform runner. Older or
direct exporter invocations that do not provide these values export `NA`.

## Operator category exclusions

Full operator runs benchmark every semantic category by default. The `special`
category is retained because it contains many successful InfiniOps rows even
though a significant number of PyTorch reference calls reject FP16/BF16 or
complex outputs. For a diagnostic run that must avoid those reference failures,
explicitly exclude the category:

```bash
python3 full_operator_benchmark.py ... --exclude-categories special
```

The exclusion applies after profile selection, so both full and small profiles
would skip operators such as `fft_fft` and `special_gammaln`.

Each dtype/profile output directory records what was not attempted in:

```text
excluded-operators.json
```

The file contains the excluded category codes and wrapper names. The default
unfiltered run has an empty exclusion list.

The benchmark also has an operator-level dtype policy. PyTorch does not provide
usable Half or BFloat16 reference paths for the 53 FFT and special-math wrappers
from the MetaX failure retry. These wrappers are skipped automatically for
`float16` and `bfloat16`, including when they are selected by `--ops` or a small
profile. They remain in the normal benchmark set for `float32`, including the
complex-output FFT wrappers that need future InfiniOps complex-dtype support.
Their names are recorded in the same manifest under
`dtype_excluded_operators`, and `results.json` reports
`dtype_excluded_wrapper_count`.

The same low-precision policy also covers 69 additional non-special wrappers,
principally linalg decomposition and solve operators, whose Half/BFloat16
PyTorch reference calls were rejected across the complete September 21 runs.
They remain available for Float32 comparison.

The five-platform pilot also records known platform gaps in the same manifest.
Platform runners pass their canonical `--platform` ID to the harness. The
harness then skips PyTorch backend kernels that are known to be unavailable
for that platform and skips exact `platform + dtype` combinations for which
both the PyTorch reference and the InfiniOps-only fallback are unavailable.
These are reported separately as `platform_excluded_operators` and
`fallback_excluded_operators`; they are not counted as operator failures.

Loss benchmarks use elementwise `reduction=none` so the reference and explicit
InfiniOps output have the input shape. `linspace` and `logspace` derive
`steps` from the active profile, and `masked_select` uses an all-true mask so
its explicit output storage has a deterministic size.

```bash
python3 full_operator_benchmark.py ... --exclude-categories none
```

When a PyTorch reference call rejects a special math function, the harness also
tries an InfiniOps-only path. If the selected implementation can run, the row is
exported with InfiniOps latency, `pt_latency_us=NA`,
`reference=unavailable`, and `validation=infini_only`. The original PyTorch
error remains in `reference_error` in `results.json` and `results.csv`. FFT
wrappers are excluded from this fallback because their output dtype and shape
cannot be derived safely from the input schema alone. If the fallback itself
fails, the case remains in `diagnostics.csv` with the original reference error
attached.

If a platform runner fails after some operators have completed, it exports the
available successful rows to a partial CSV:

```text
<device>.partial.csv
diagnostics.partial.csv
report.partial.md
partial-manifest.json
```

The partial exporter reads both completed `results.json` files and interrupted
`results.jsonl` checkpoints, ignores a truncated final JSONL line, and dedupes
rows using the normal exporter rules. Partial CSVs are useful for debugging and
coverage inspection, but are not complete benchmark deliveries and must not be
treated as full-run latency comparisons.

## Why platform adapters remain

The unified entry point centralizes device selection and delivery paths, but
hardware execution remains in platform adapters because the platforms use
different container images, monitoring tools, source layouts, compilers, runtime
 APIs, and build caches. For example:

- NVIDIA uses `nvidia-smi`;
- MetaX uses `mx-smi`;
- Ascend uses NPU tooling;
- Moore uses `mthreads-gmi`;
- Hygon uses ROCm/Hygon tooling;
- Cambricon uses MLU tooling;
- Iluvatar/Tianshu uses its deployed stack.

Keeping these adapters separate avoids accidental cross-platform breakage while
still giving agents and frontend workflows one command to call.

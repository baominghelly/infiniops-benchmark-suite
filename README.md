# InfiniOps Benchmark Suite

Standalone benchmark harness and report tooling for InfiniOps operator coverage
and PyTorch comparisons on accelerator runtimes.

## Contents

- `scripts/full_operator_benchmark.py`: schema-driven per-operator benchmark
  harness for correctness and latency comparisons.
- `scripts/export_full_operator_report.py`: merges result JSON files into the
  platform CSV, diagnostics CSV, and Markdown report.
- `scripts/classify_operator_semantics.py`: detailed semantic classification.
- `scripts/classify_operator_semantics_9cat.py`: compact nine-category
  classification used by report exports.
- Platform runners under `scripts/run_*_benchmark.sh`.
- `docs/run_nvidia_benchmark.md`: complete NVIDIA A100 runner operations guide.
- `docs/run_metax_benchmark.md`: complete MetaX runner operations guide.
- `docs/run_hygon_benchmark.md`: complete Hygon BW1000 runner operations guide.
- `docs/run_moore_benchmark.md`: complete Moore Threads MTT S5000 runner
  operations guide.
- `docs/run_cambricon_benchmark.md`: complete Cambricon MLU590 runner
  operations guide.
- `docs/run_tianshu58_benchmark.md`: complete Tianshu/Iluvatar TG150 runner
  operations guide with device ID `iluvatar-tg150`.
- Regression tests under `tests/`.

The exported platform CSV contains:

```text
platform,op_name,dtype,shape_config,ic_latency_us,ic_tflops,
pt_latency_us,pt_tflops,date,tester,remarks,category_code
```

`category_code` uses:

```text
math, logic, reduction, linalg, neural, indexing, special, dense, sparse
```

## Tests

Run the software regression tests from the repository root:

```bash
python -m unittest discover -s tests -p 'test_*.py'
```

The tests do not require an accelerator. They cover result parsing, report
schema generation, operator recipes, and semantic category mapping.

## Accelerator Runs

Accelerator runs require an already prepared environment containing InfiniOps,
InfiniRT, PyTorch, the accelerator runtime, and generated InfiniOps metadata.
Platform-specific runners may require environment variables or arguments for
the container name and physical device index. Inspect the runner header before
using it on a new machine.

`run_a100_benchmark.sh` is a host-side orchestrator for the A100 machine. It
uses host `docker`, `nvidia-smi`, and `/proc` access to enforce GPU resource
gates and attribute compute processes. Source preparation, compilation,
benchmarking, and export run inside the configured Docker container. See
[docs/run_nvidia_benchmark.md](docs/run_nvidia_benchmark.md) for source
backfill, GPU thresholds, output contracts, monitoring, and troubleshooting.

`run_metax_benchmark.sh` is a host-side orchestrator. It uses host `docker`,
`mx-smi`, and `/proc` access to select a device and attribute GPU processes,
but compilation and benchmark execution always run inside the configured Docker
container through `docker exec`.

See [docs/run_metax_benchmark.md](docs/run_metax_benchmark.md) for the complete
MetaX runner guide, including source backfill, GPU resource gates, output
contracts, monitoring commands, and troubleshooting.

`run_hygon_benchmark.sh` is the corresponding host-side orchestrator for the
Hygon BW1000/DCU machine. It uses host `docker`, `rocm-smi`, and `/proc` access
for HCU selection and process attribution. See
[docs/run_hygon_benchmark.md](docs/run_hygon_benchmark.md) for the complete
Hygon runner guide, including InfiniRT installation, source backfill, GPU
resource gates, output contracts, and troubleshooting.

See [docs/run_moore_benchmark.md](docs/run_moore_benchmark.md) for the complete
Moore Threads MTT S5000 runner guide, including source backfill, MUSA device
selection, GPU resource gates, output contracts, and troubleshooting.

On the MetaX machine, deploy the suite under the shared NFS entry:

```text
/data-aisoft/infinibench-scripts/
```

The frontend or another user's agent should invoke the host-side runner from
that shared path rather than from a user's home directory. Keep the scripts tree
read-only to benchmark callers; operator result files still belong under
`/data-aisoft/infinibench_data/ops/`.

### Source update and backfill

By default the runner prepares both source trees inside the container before
compilation:

```text
/workspace/InfiniOps    -> https://github.com/InfiniTensor/InfiniOps.git
/workspace/InfiniRT     -> https://github.com/InfiniTensor/InfiniRT.git
```

If a repository is absent, it is cloned with full history. If it already exists,
the runner refuses to continue when the work tree is dirty, sets `origin` to the
configured URL, fetches tags and branches, and fast-forwards the configured
branch (`master` by default). The URL and branch can be overridden with
`INFINIOPS_REPO_URL`, `INFINIRT_REPO_URL`, `INFINIOPS_BRANCH`, and
`INFINIRT_BRANCH`. Because direct GitHub access can be intermittent, the default
repository list tries GitHub directly, then `gh-proxy.com`, then `ghproxy.net`.
Provide a comma-separated `INFINIOPS_REPO_URLS` or `INFINIRT_REPO_URLS` list to
change the complete fallback chain. URL probes use
`SOURCE_PROBE_TIMEOUT_SECONDS=30` and clone/fetch operations use
`SOURCE_OPERATION_TIMEOUT_SECONDS=900`. `UPDATE_SOURCES=0` skips this stage and
requires both repositories to already exist.

The runner has two time semantics. The execution time defaults to the current
wall clock and can be overridden with `EXECUTION_TIME=YYYYMMDD-HHMMSS`. The
optional `BACKFILL_DATE=YYYY-MM-DD` is the historical source date. A normal run
writes to `<execution-time>/`; a backfill writes to
`backfill-<target-date>-<execution-time>/`. For backfill runs, the CSV `date`
column uses the target date rather than the execution date.

For a date backfill, both repositories select the last commit on their configured
remote branch whose commit time is no later than 23:59:59 Asia/Shanghai on that
date, check out that commit, and then build:

```bash
BACKFILL_DATE=2026-09-11 \
DTYPES=float16,float32,bfloat16 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_metax_benchmark.sh
```

For reproducible runs tied to exact tested revisions, set `INFINIOPS_REF` and/or
`INFINIRT_REF` to a commit, tag, or branch. Explicit refs take precedence over
`BACKFILL_DATE`; set both refs when a matched pair is required. The selected
commits are recorded in `benchmark-launcher.log` and `benchmark.done`.

Because source backfill and compilation mutate shared container workspaces, the
MetaX runner takes a machine-level lock. A second invocation on the same machine
exits with code 2 instead of switching source versions under an active run.

The MetaX runner accepts a comma-separated `DTYPES` list and runs all requested
dtypes in one invocation. It also accepts a comma-separated `PROFILES` list.
Its workflow defaults are to compile InfiniOps, automatically choose the
lowest-numbered idle MetaX GPU, run three dtypes, and use one coverage profile:

```bash
DTYPES=float16,float32,bfloat16 PROFILES=decode-4k \
  /data-aisoft/infinibench-scripts/scripts/run_metax_benchmark.sh
```

Idle selection requires no running process, no `unavailable_reason`, and at most
5% utilization by default. Automatic selection is refreshed after compilation
so a device that becomes occupied during the build is not used. Pass a physical
index as the positional argument, or set `PHYSICAL_GPU`, when an explicit device
is required. Set `AUTO_SELECT_GPU=0` to make an explicit index mandatory.

For functional smoke runs that may share a device, tune the resource gate:

```bash
GPU_MAX_UTILIZATION_PERCENT=40 \
GPU_MIN_FREE_MEMORY_PERCENT=40 \
GPU_ALLOW_OCCUPIED=1 \
PROFILES=decode-4k-small \
/data-aisoft/infinibench-scripts/scripts/run_metax_benchmark.sh
```

`GPU_MAX_UTILIZATION_PERCENT` defaults to 5, `GPU_MIN_FREE_MEMORY_PERCENT`
defaults to 0, and `GPU_ALLOW_OCCUPIED` defaults to 0. Values are percentages.
Allowing an occupied device makes a run suitable for pipeline validation, but
latency numbers from that run should not be compared with exclusive benchmarks.

Compilation uses a persistent scikit-build directory at
`/workspace/.cache/infiniops-build` and defaults to 64 parallel CMake jobs for
the 128-core MetaX host. Override these with `INFINIOPS_BUILD_DIR` and
`CMAKE_BUILD_PARALLEL_LEVEL`. Set `BUILD_INFINIOPS=0` only for a runtime-only
run against an already installed InfiniOps package.

Since every profile uses the same 495-operator set with a different shape,
`decode-4k` alone is sufficient for operator-coverage runs. Add
`decode-8k` or `prefill-4k` only when shape-specific performance is needed.

For development smoke runs, select the reduced profile:

```bash
PROFILES=decode-4k-small \
  /data-aisoft/infinibench-scripts/scripts/run_metax_benchmark.sh
```

`decode-4k-small` uses a `[1, 128]` default shape and an 18-operator
representative set spanning all nine semantic categories. Override it with
`METAX_DECODE_4K_SMALL_OPS` when needed.

Canonical raw and final results are written directly to:

```text
/data-aisoft/infinibench_data/ops/metax-c550/<date>/
```

The runner then exports one combined CSV and diagnostics table at that
date-root level. Rows are distinguished by their `dtype` column.

The host path `/data-aisoft/infinibench_data/ops` and the same path inside the
container must refer to the same bind mount. `OUTPUT_ROOT` and
`CONTAINER_OUTPUT_ROOT` remain available for environments that cannot mount
that canonical path.

`RUN_DATE` and `CSV_DATE` are intentionally not accepted by the MetaX runner.
The output directory and CSV date are derived from `EXECUTION_TIME` and
`BACKFILL_DATE`.

The general harness command is:

```bash
python3 scripts/full_operator_benchmark.py \
  --metadata /path/to/torch_ops_metadata.json \
  --output-dir /path/to/results \
  --device cuda \
  --dtype float16 \
  --sha "$INFINIOPS_COMMIT" \
  --min-run-time 0.02 \
  --timeout 20
```

Some accelerator builds expose a CUDA-compatible API even when the underlying
hardware is not NVIDIA. Use the device name expected by the deployed PyTorch
build.

## Notes

- Result CSV and JSON files are intentionally not committed.
- Platform runners may contain deployment-specific defaults; review and adjust
  them before running in a new environment.
- Latency results are only comparable when the accelerator is idle and the same
  runtime, dtype, shape, timer, and synchronization policy are used.

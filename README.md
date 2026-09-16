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

The MetaX runner accepts a comma-separated `DTYPES` list and runs all requested
dtypes in one invocation. It also accepts a comma-separated `PROFILES` list.
The workflow defaults are three dtypes and one coverage profile:

```bash
DTYPES=float16,float32,bfloat16 PROFILES=decode-4k scripts/run_metax_benchmark.sh 0
```

Since every profile uses the same 495-operator set with a different shape,
`decode-4k` alone is sufficient for operator-coverage runs. Add
`decode-8k` or `prefill-4k` only when shape-specific performance is needed.

For development smoke runs, select the reduced profile:

```bash
PROFILES=decode-4k-small scripts/run_metax_benchmark.sh 0
```

`decode-4k-small` uses a `[1, 128]` default shape and an 18-operator
representative set spanning all nine semantic categories. Override it with
`METAX_DECODE_4K_SMALL_OPS` when needed.

Canonical raw and final results are written directly to:

```text
/data-aisoft/infinibench_data/metax-c550/<date>/
```

The runner then exports one combined CSV and diagnostics table at that
date-root level. Rows are distinguished by their `dtype` column.

The host path `/data-aisoft/infinibench_data` and the same path inside the
container must refer to the same bind mount. `OUTPUT_ROOT` and
`CONTAINER_OUTPUT_ROOT` remain available for environments that cannot mount
that canonical path.

`RUN_DATE` is also used as the output directory name. When it is a test label
rather than a date, set `CSV_DATE` explicitly, for example
`CSV_DATE=2026/09/15`.

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

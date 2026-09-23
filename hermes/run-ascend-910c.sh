#!/usr/bin/env bash
set -Eeuo pipefail

export DEVICE="${BENCHMARK_DEVICE:-ascend-910c}"
runner="${BENCHMARK_RUNNER:-/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh}"
test -x "$runner"
exec "$runner"

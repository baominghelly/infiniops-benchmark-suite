#!/usr/bin/env bash
set -euo pipefail

runner="${BENCHMARK_RUNNER:-/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh}"
device="${BENCHMARK_DEVICE:-nvidia-a100}"
dtypes="${DTYPES:-float16,float32,bfloat16}"
profiles="${PROFILES:-decode-4k}"
infiniops_ref="${INFINIOPS_REF:-}"
infinirt_ref="${INFINIRT_REF:-}"
state_root="${HERMES_STATE_ROOT:-/data-aisoft/baoming/hermes/runs}"
benchmark_output_root="${BENCHMARK_OUTPUT_ROOT:-/data-aisoft/infinibench_data/ops/nvidia-a100}"
execution_time="${EXECUTION_TIME:-$(date +%Y%m%d-%H%M%S)}"
execution_date="${execution_time:0:8}"
run_root="$state_root/$execution_date"
attempt_log="$run_root/attempts.tsv"
lock_file="$state_root/nvidia-a100.lock"

# Hermes itself runs in a private PID namespace so its s6 supervision tree
# stays healthy. GPU process attribution requires host PIDs, so dispatch a
# short-lived launcher with --pid=host for the actual benchmark.
if [[ "${BENCHMARK_DIRECT:-0}" != "1" ]]; then
  image="${BENCHMARK_HELPER_IMAGE:-infinibench/hermes-agent:ca29a769}"
  exec docker run --rm --pid host \
    --name "baoming_hermes_a100_benchmark_${execution_time}" \
    --entrypoint /bin/bash \
    -e BENCHMARK_DIRECT=1 \
    -e TZ="${TZ:-Asia/Shanghai}" \
    -e EXECUTION_TIME="$execution_time" \
    -e DTYPES="$dtypes" \
    -e PROFILES="$profiles" \
    -e INFINIOPS_REF="$infiniops_ref" \
    -e INFINIRT_REF="$infinirt_ref" \
    -e BUILD_INFINIOPS="${BUILD_INFINIOPS:-1}" \
    -e UPDATE_SOURCES="${UPDATE_SOURCES:-1}" \
    -e REUSE_RUN_DIR="${REUSE_RUN_DIR:-0}" \
    -e DRY_RUN="${DRY_RUN:-0}" \
    -e HERMES_STATE_ROOT="$state_root" \
    -v /data-aisoft/baoming/hermes/data:/opt/data \
    -v /data-aisoft/baoming/hermes/runs:/data-aisoft/baoming/hermes/runs \
    -v /data-aisoft/infinibench-scripts:/data-aisoft/infinibench-scripts:ro \
    -v /data-aisoft/infinibench_data:/data-aisoft/infinibench_data \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -v /usr/bin/nvidia-smi:/usr/bin/nvidia-smi:ro \
    "$image" /opt/data/scripts/run-nvidia-a100-benchmark.sh
fi

mkdir -p "$run_root"

record_attempt() {
  local attempt="$1"
  local start="$2"
  local end="${3:-RUNNING}"
  local exit_code="${4:--}"
  local mode="$5"
  local reason="$6"
  if [[ ! -s "$attempt_log" ]]; then
    printf 'attempt\tstart\tend\texit_code\tresource_mode\treason\n' >"$attempt_log"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$attempt" "$start" "$end" "$exit_code" "$mode" "$reason" >>"$attempt_log"
}

write_outcome() {
  local status="$1"
  local exit_code="$2"
  local console="$3"
  local final_mode="$4"
  if [[ "$status" == "success" ]]; then
    outcome_file="$run_root/success.json"
    rm -f -- "$run_root/failure.json"
  else
    outcome_file="$run_root/failure.json"
    rm -f -- "$run_root/success.json"
  fi
  EXECUTION_TIME="$execution_time" \
  BENCHMARK_OUTPUT_ROOT="$benchmark_output_root" \
    python3 - "$run_root" "$status" "$exit_code" "$console" "$execution_date" "$final_mode" <<'PY'
import datetime
import json
import os
import pathlib
import sys

run_root, status, exit_code, console, execution_date, final_mode = sys.argv[1:7]
payload = {
    "status": status,
    "final_exit_code": int(exit_code),
    "final_console": sys.argv[3],
    "final_mode": final_mode,
    "execution_date": sys.argv[5],
    "execution_time": os.environ["EXECUTION_TIME"],
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "benchmark_output": os.environ["BENCHMARK_OUTPUT_ROOT"] + "/" + execution_date,
}
pathlib.Path(run_root).joinpath(
    "success.json" if status == "success" else "failure.json"
).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY
}

exec 9>"$lock_file"
if ! flock -n 9; then
  echo "another NVIDIA A100 Hermes workflow is active: $lock_file" >&2
  exit 2
fi

for command in docker nvidia-smi flock; do
  command -v "$command" >/dev/null || {
    echo "required command is unavailable: $command" >&2
    exit 2
  }
done
test -x "$runner" || {
  echo "benchmark runner is not executable: $runner" >&2
  exit 2
}
docker info >/dev/null 2>&1 || {
  echo "Docker socket is unavailable" >&2
  exit 2
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'runner=%s\ndevice=%s\ndtypes=%s\nprofiles=%s\nexecution_time=%s\nstate_root=%s\n' \
    "$runner" "$device" "$dtypes" "$profiles" "$execution_time" "$state_root"
  exit 0
fi

start="$(date --iso-8601=seconds)"
attempt=1
mode=exclusive
console="$run_root/attempt-1-exclusive.log"
record_attempt "$attempt" "$start" "" "-" "$mode" started
set +e
env \
  EXECUTION_TIME="$execution_time" \
  DTYPES="$dtypes" \
  PROFILES="$profiles" \
  INFINIOPS_REF="$infiniops_ref" \
  INFINIRT_REF="$infinirt_ref" \
  BUILD_INFINIOPS="${BUILD_INFINIOPS:-1}" \
  UPDATE_SOURCES="${UPDATE_SOURCES:-1}" \
  REUSE_RUN_DIR="${REUSE_RUN_DIR:-0}" \
  GPU_ALLOW_OCCUPIED=0 \
  GPU_MAX_UTILIZATION_PERCENT="${GPU_MAX_UTILIZATION_PERCENT:-5}" \
  GPU_MIN_FREE_MEMORY_PERCENT="${GPU_MIN_FREE_MEMORY_PERCENT:-0}" \
  "$runner" "$device" >"$console" 2>&1
first_exit=$?
set -e
end="$(date --iso-8601=seconds)"
first_reason="exclusive attempt exited $first_exit"

if [[ $first_exit -eq 0 ]]; then
  record_attempt "$attempt" "$start" "$end" 0 "$mode" success
  write_outcome success 0 "$console" "$mode"
  cat "$console"
  exit 0
fi

record_attempt "$attempt" "$start" "$end" "$first_exit" "$mode" "$first_reason"

retry_needed=0
if grep -Eqi \
  'no idle NVIDIA GPU|has a running process|utilization exceeds|free memory is below|foreign process' \
  "$console"; then
  retry_needed=1
fi

if [[ $retry_needed -ne 1 ]]; then
  write_outcome failure "$first_exit" "$console" "$mode"
  cat "$console"
  exit "$first_exit"
fi

attempt=2
mode=shared-flow-validation
console="$run_root/attempt-2-shared.log"
start="$(date --iso-8601=seconds)"
record_attempt "$attempt" "$start" "" "-" "$mode" "exclusive gate failed; retrying with GPU_ALLOW_OCCUPIED=1"
set +e
env \
  EXECUTION_TIME="$execution_time" \
  DTYPES="$dtypes" \
  PROFILES="$profiles" \
  INFINIOPS_REF="$infiniops_ref" \
  INFINIRT_REF="$infinirt_ref" \
  BUILD_INFINIOPS=0 \
  UPDATE_SOURCES=1 \
  REUSE_RUN_DIR=1 \
  GPU_ALLOW_OCCUPIED=1 \
  GPU_MAX_UTILIZATION_PERCENT=50 \
  GPU_MIN_FREE_MEMORY_PERCENT=50 \
  "$runner" "$device" >"$console" 2>&1
second_exit=$?
set -e
end="$(date --iso-8601=seconds)"
if [[ $second_exit -eq 0 ]]; then
  record_attempt "$attempt" "$start" "$end" 0 "$mode" "success; latency is not comparable with exclusive runs"
  write_outcome success 0 "$console" "$mode"
else
  record_attempt "$attempt" "$start" "$end" "$second_exit" "$mode" "shared retry failed"
  write_outcome failure "$second_exit" "$console" "$mode"
fi
cat "$console"
exit "$second_exit"

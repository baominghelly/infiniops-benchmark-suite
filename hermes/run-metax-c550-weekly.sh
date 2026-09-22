#!/usr/bin/env bash
set -Eeuo pipefail

umask 077
export TZ="${TZ:-Asia/Shanghai}"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

runner="${BENCHMARK_RUNNER:-/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh}"
device="${BENCHMARK_DEVICE:-metax-c550}"
dtypes="${DTYPES:-float16,float32,bfloat16}"
profiles="${PROFILES:-decode-4k}"
infiniops_ref="${INFINIOPS_REF:-}"
infinirt_ref="${INFINIRT_REF:-}"
container="${CONTAINER:-baoming_infiniops_metax_data}"
state_root="${SCHEDULE_STATE_ROOT:-/data/libaoming/infinibench-schedule/metax-c550}"
build_dir="${INFINIOPS_BUILD_DIR:-/data/libaoming/infinibench-build/metax-c550}"
infiniops_source_root="${INFINIOPS_SOURCE_ROOT:-/data/libaoming/infiniops-source/InfiniOps}"
infinirt_source_root="${INFINIRT_SOURCE_ROOT:-/data/libaoming/infiniops-source/InfiniRT}"
benchmark_output_root="${BENCHMARK_OUTPUT_ROOT:-/data-aisoft/infinibench_data/ops/metax-c550}"
execution_time="${EXECUTION_TIME:-$(date +%Y%m%d-%H%M%S)}"
execution_date="${execution_time:0:8}"
run_root="$state_root/runs/$execution_date"
attempt_log="$run_root/attempts.tsv"
console="$run_root/benchmark-console.log"
lock_file="$state_root/metax-c550.lock"

mkdir -p "$run_root"
chmod 700 "$state_root" "$run_root"

for command in flock python3; do
  command -v "$command" >/dev/null || {
    echo "required command is unavailable: $command" >&2
    exit 2
  }
done

exec 9>"$lock_file"
if ! flock -n 9; then
  echo "another MetaX C550 scheduled benchmark is active: $lock_file" >&2
  exit 2
fi
test -x "$runner" || {
  echo "benchmark runner is not executable: $runner" >&2
  exit 2
}

record_attempt() {
  local start="$1"
  local end="${2:-RUNNING}"
  local exit_code="${3:--}"
  local category="${4:-started}"
  if [[ ! -s "$attempt_log" ]]; then
    printf 'start\tend\texit_code\tcategory\tconsole\n' >"$attempt_log"
  fi
  printf '%s\t%s\t%s\t%s\t%s\n' \
    "$start" "$end" "$exit_code" "$category" "$console" >>"$attempt_log"
}

classify_failure() {
  local text
  text="$(cat "$console" 2>/dev/null || true)"
  if grep -Eq 'container not found|container is not running' <<<"$text"; then
    printf '%s\n' benchmark_container_down
  elif grep -Eq 'Docker socket is unavailable|Cannot connect to the Docker daemon' <<<"$text"; then
    printf '%s\n' docker_down
  elif grep -Eq 'no idle MetaX GPU found' <<<"$text"; then
    printf '%s\n' no_idle_gpu
  elif grep -Eq 'has a running process|foreign process|foreign_process_seen=1' <<<"$text"; then
    printf '%s\n' foreign_process
  elif grep -Eq 'output directory already exists and is non-empty' <<<"$text"; then
    printf '%s\n' output_dir_nonempty
  elif grep -Eq 'repository has local changes' <<<"$text"; then
    printf '%s\n' source_dirty
  elif grep -Eqi 'no space left|disk quota exceeded' <<<"$text"; then
    printf '%s\n' disk_full
  elif grep -Eq 'mx-smi .*failed|mx-smi-error' <<<"$text"; then
    printf '%s\n' gpu_tool_failure
  elif grep -Eq 'another MetaX benchmark runner is active' <<<"$text"; then
    printf '%s\n' runner_busy
  else
    printf '%s\n' unknown
  fi
}

write_outcome() {
  local status="$1"
  local exit_code="$2"
  local category="${3:-}"
  local reason="${4:-}"
  local needs_llm=0
  if [[ "$status" == "failure" && "$category" == "unknown" ]]; then
    needs_llm=1
  fi

  rm -f -- "$run_root/success.json" "$run_root/failure.json"
  EXECUTION_TIME="$execution_time" \
  BENCHMARK_OUTPUT_ROOT="$benchmark_output_root" \
    python3 - "$run_root" "$status" "$exit_code" "$category" "$reason" "$needs_llm" "$console" <<'PY'
import datetime
import json
import os
import pathlib
import sys

run_root, status, exit_code, category, reason, needs_llm, console = sys.argv[1:8]
payload = {
    "status": status,
    "device": "metax-c550",
    "exit_code": int(exit_code),
    "failure_category": category or None,
    "failure_reason": reason or None,
    "needs_llm": needs_llm == "1",
    "execution_time": os.environ["EXECUTION_TIME"],
    "execution_date": os.environ["EXECUTION_TIME"][:8],
    "benchmark_output": os.environ["BENCHMARK_OUTPUT_ROOT"] + "/" + os.environ["EXECUTION_TIME"][:8],
    "console": console,
    "recorded_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
name = "success.json" if status == "success" else "failure.json"
pathlib.Path(run_root, name).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'runner=%s\ndevice=%s\ncontainer=%s\ndtypes=%s\nprofiles=%s\nexecution_time=%s\nstate_root=%s\nbuild_dir=%s\ninfiniops_source_root=%s\ninfinirt_source_root=%s\nbenchmark_output_root=%s\n' \
    "$runner" "$device" "$container" "$dtypes" "$profiles" "$execution_time" "$state_root" "$build_dir" "$infiniops_source_root" "$infinirt_source_root" "$benchmark_output_root"
  DEVICE="$device" OUTPUT_ROOT="${benchmark_output_root%/*}" "$runner" --print
  exit 0
fi

start="$(date --iso-8601=seconds)"
record_attempt "$start"
: >"$console"

set +e
env \
  TZ="$TZ" \
  DEVICE="$device" \
  CONTAINER="$container" \
  EXECUTION_TIME="$execution_time" \
  DTYPES="$dtypes" \
  PROFILES="$profiles" \
  INFINIOPS_REF="$infiniops_ref" \
  INFINIRT_REF="$infinirt_ref" \
  INFINIOPS_BUILD_DIR="$build_dir" \
  INFINIOPS_SOURCE_ROOT="$infiniops_source_root" \
  INFINIRT_SOURCE_ROOT="$infinirt_source_root" \
  BUILD_INFINIOPS="${BUILD_INFINIOPS:-1}" \
  UPDATE_SOURCES="${UPDATE_SOURCES:-1}" \
  REUSE_RUN_DIR="${REUSE_RUN_DIR:-0}" \
  GPU_ALLOW_OCCUPIED="${GPU_ALLOW_OCCUPIED:-0}" \
  GPU_MAX_UTILIZATION_PERCENT="${GPU_MAX_UTILIZATION_PERCENT:-5}" \
  GPU_MIN_FREE_MEMORY_PERCENT="${GPU_MIN_FREE_MEMORY_PERCENT:-0}" \
  "$runner" >"$console" 2>&1
exit_code=$?
set -e

end="$(date --iso-8601=seconds)"
if [[ "$exit_code" -eq 0 ]]; then
  category=success
  reason=""
  record_attempt "$start" "$end" "$exit_code" "$category"
  write_outcome success "$exit_code" "" ""
else
  category="$(classify_failure)"
  reason="benchmark exited with code $exit_code"
  record_attempt "$start" "$end" "$exit_code" "$category"
  write_outcome failure "$exit_code" "$category" "$reason"
fi

cat "$console"
exit "$exit_code"

#!/usr/bin/env bash
set -euo pipefail

default_container=infiniops-ascend910c-245
default_device_id=ascend-910c
default_soc_version=Ascend910_9362

container="${CONTAINER:-$default_container}"
requested_npu="${1:-${PHYSICAL_NPU:-}}"
auto_select_npu="${AUTO_SELECT_NPU:-1}"
gpu_max_utilization="${GPU_MAX_UTILIZATION_PERCENT:-5}"
gpu_min_free_memory="${GPU_MIN_FREE_MEMORY_PERCENT:-90}"
gpu_allow_occupied="${GPU_ALLOW_OCCUPIED:-0}"
physical_npu=""
npu_selection=""
npu_candidates=""
npu_busy=""
npu_rejected=""

execution_time="${EXECUTION_TIME:-$(date +%Y%m%d-%H%M%S)}"
execution_date="${execution_time:0:8}"
tester="${TESTER:-baoming}"
build_infiniops="${BUILD_INFINIOPS:-1}"
build_ascend_custom="${BUILD_ASCEND_CUSTOM:-ON}"
soc_version="${SOC_VERSION:-$default_soc_version}"
update_sources="${UPDATE_SOURCES:-1}"
backfill_date="${BACKFILL_DATE:-}"
reuse_run_dir="${REUSE_RUN_DIR:-0}"
infiniops_ref="${INFINIOPS_REF:-}"
infinirt_ref="${INFINIRT_REF:-}"
device_id="${DEVICE_ID:-$default_device_id}"
output_csv_name="${OUTPUT_CSV_NAME:-infiniops-$device_id.csv}"
output_root="${OUTPUT_ROOT:-/data-aisoft/infinibench_data/ops}"
dtypes_input="${DTYPES:-float16}"
read -r -a dtypes <<<"${dtypes_input//,/ }"
for dtype in "${dtypes[@]}"; do
  case "$dtype" in
    float16|float32|bfloat16) ;;
    *)
      echo "invalid DTYPES entry: $dtype; expected float16, float32, or bfloat16" >&2
      exit 2
      ;;
  esac
done
profiles_input="${PROFILES:-decode-4k}"
read -r -a profiles <<<"${profiles_input//,/ }"
for profile in "${profiles[@]}"; do
  case "$profile" in
    decode-4k|decode-4k-small|decode-8k|prefill-4k) ;;
    *)
      echo "invalid PROFILES entry: $profile; expected decode-4k, decode-4k-small, decode-8k, or prefill-4k" >&2
      exit 2
      ;;
  esac
done

case "$gpu_allow_occupied" in
  0|1) ;;
  *)
    echo "invalid GPU_ALLOW_OCCUPIED: $gpu_allow_occupied; expected 0 or 1" >&2
    exit 2
    ;;
esac
for gpu_threshold in "$gpu_max_utilization" "$gpu_min_free_memory"; do
  if [[ ! "$gpu_threshold" =~ ^[0-9]+$ ]] || (( gpu_threshold > 100 )); then
    echo "invalid GPU threshold: $gpu_threshold; expected an integer between 0 and 100" >&2
    exit 2
  fi
done

if [[ -n "${RUN_DATE:-}" || -n "${CSV_DATE:-}" ]]; then
  echo "RUN_DATE and CSV_DATE are no longer accepted; use optional EXECUTION_TIME and/or BACKFILL_DATE" >&2
  exit 2
fi
execution_time_iso="${execution_time:0:4}-${execution_time:4:2}-${execution_time:6:2} ${execution_time:9:2}:${execution_time:11:2}:${execution_time:13:2}"
if [[ ! "$execution_time" =~ ^[0-9]{8}-[0-9]{6}$ ]] || ! date -d "$execution_time_iso" >/dev/null 2>&1; then
  echo "invalid EXECUTION_TIME: $execution_time; expected YYYYMMDD-HHMMSS" >&2
  exit 2
fi
if [[ -n "$backfill_date" ]]; then
  run_date="backfill-${backfill_date//-/}-${execution_date}"
  csv_date="${backfill_date//-//}"
else
  run_date="$execution_date"
  csv_date="${execution_time:0:4}/${execution_time:4:2}/${execution_time:6:2}"
fi

case "$update_sources" in
  0|1) ;;
  *)
    echo "invalid UPDATE_SOURCES: $update_sources; expected 0 or 1" >&2
    exit 2
    ;;
esac
case "$reuse_run_dir" in
  0|1) ;;
  *)
    echo "invalid REUSE_RUN_DIR: $reuse_run_dir; expected 0 or 1" >&2
    exit 2
    ;;
esac
if [[ -n "$backfill_date" ]]; then
  if [[ ! "$backfill_date" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] || ! date -d "$backfill_date" >/dev/null 2>&1; then
    echo "invalid BACKFILL_DATE: $backfill_date; expected YYYY-MM-DD" >&2
    exit 2
  fi
fi
if [[ "$update_sources" != "1" && ( -n "$backfill_date" || -n "$infiniops_ref" || -n "$infinirt_ref" ) ]]; then
  echo "UPDATE_SOURCES=0 cannot be combined with BACKFILL_DATE, INFINIOPS_REF, or INFINIRT_REF" >&2
  exit 2
fi

host_tools_root="${BENCHMARK_TOOLS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
output_host="$output_root/$device_id/$run_date"
container_tools_root="${CONTAINER_BENCHMARK_ROOT:-/data-aisoft/infinibench-scripts}"
source_root="${INFINIOPS_SOURCE_ROOT:-/workspace/InfiniOps}"
infinirt_source_root="${INFINIRT_SOURCE_ROOT:-/workspace/InfiniRT}"
infiniops_repo_urls_input="${INFINIOPS_REPO_URLS:-${INFINIOPS_REPO_URL:-https://github.com/InfiniTensor/InfiniOps.git,https://gh-proxy.com/https://github.com/InfiniTensor/InfiniOps.git,https://ghproxy.net/https://github.com/InfiniTensor/InfiniOps.git}}"
infinirt_repo_urls_input="${INFINIRT_REPO_URLS:-${INFINIRT_REPO_URL:-https://github.com/InfiniTensor/InfiniRT.git,https://gh-proxy.com/https://github.com/InfiniTensor/InfiniRT.git,https://ghproxy.net/https://github.com/InfiniTensor/InfiniRT.git}}"
source_probe_timeout="${SOURCE_PROBE_TIMEOUT_SECONDS:-30}"
source_operation_timeout="${SOURCE_OPERATION_TIMEOUT_SECONDS:-900}"
infiniops_branch="${INFINIOPS_BRANCH:-master}"
infinirt_branch="${INFINIRT_BRANCH:-master}"
for timeout_value in "$source_probe_timeout" "$source_operation_timeout"; do
  if [[ ! "$timeout_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid source timeout: $timeout_value; expected a positive integer" >&2
    exit 2
  fi
done
build_dir="${INFINIOPS_BUILD_DIR:-/workspace/.cache/infiniops-build}"
cmake_build_parallel_level="${CMAKE_BUILD_PARALLEL_LEVEL:-32}"
output_container_root="${CONTAINER_OUTPUT_ROOT:-$output_root}"
output_container="$output_container_root/$device_id/$run_date"
runner_lock="$output_root/$device_id/.infinibench-runner.lock"
runner_lock_container="$output_container_root/$device_id/.infinibench-runner.lock"
harness="$container_tools_root/scripts/full_operator_benchmark.py"
exporter="$container_tools_root/scripts/export_full_operator_report.py"
partial_exporter="$container_tools_root/scripts/export_partial_results.py"
launcher_log="$output_host/benchmark-launcher.log"
monitor_log="$output_host/benchmark-monitor.log"
foreign_flag="$output_host/foreign-process.flag"
done_file="$output_host/benchmark.done"
npu_logical_mapping="$(npu-smi info -m 2>&1)"

target_ops=abs,add,addmm,amax,amin,argmax,bitwise_and,bitwise_not,bitwise_or,bmm,cat,ceil,clamp,cos,cumsum,div,eq,exp,floor,gather,ge,gelu,gt,hardtanh,index_select,le,log,log_softmax,lt,matmul,maximum,mean,minimum,mm,mul,ne,neg,pow,reciprocal,relu,round,rsqrt,sigmoid,sign,silu,sin,softmax,sort,sqrt,sub,sum,tanh,topk
small_ops=abs,add,avg_pool2d,bmm,gather,fft_fft,gelu,matmul,mm,mul,relu,sigmoid,silu,softmax,special_gammaln,sparse_sampled_addmm,sum,where
decode_4k_ops="${ASCEND_DECODE_4K_OPS:-all}"
decode_4k_small_ops="${ASCEND_DECODE_4K_SMALL_OPS:-$small_ops}"
decode_8k_ops="${ASCEND_DECODE_8K_OPS:-$target_ops}"
prefill_4k_ops="${ASCEND_PREFILL_4K_OPS:-$target_ops}"

if ! docker inspect "$container" >/dev/null 2>&1; then
  echo "container not found: $container" >&2
  exit 2
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" != "true" ]]; then
  echo "container is not running: $container" >&2
  exit 2
fi
container_id="$(docker inspect -f '{{.Id}}' "$container")"

# Backfill changes shared source trees, so serialize all runner executions on
# this machine. The lock is released automatically when this process exits.
docker exec "$container" mkdir -p "$(dirname "$runner_lock_container")"
docker exec "$container" touch "$runner_lock_container"
docker exec "$container" chmod 0666 "$runner_lock_container"
exec 9>"$runner_lock"
if ! flock -n 9; then
  echo "another Ascend benchmark runner is active: $runner_lock" >&2
  exit 2
fi

# A run directory is deterministic by execution date, or by backfill target
# date plus execution date. Refuse to mix stale checkpoints into a new attempt.
# REUSE_RUN_DIR=1 replaces that fixed directory only after the lock is held.
if [[ -e "$output_host" && "$reuse_run_dir" != "1" ]]; then
  if [[ -n "$(find "$output_host" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "output directory already exists and is non-empty: $output_host" >&2
    echo "set REUSE_RUN_DIR=1 to replace it for the same execution/backfill date pair" >&2
    exit 2
  fi
fi
if [[ "$reuse_run_dir" == "1" && -e "$output_container" ]]; then
  case "$output_container" in
    /data-aisoft/infinibench_data/ops/*/*) ;;
    *)
      echo "refusing to reset untrusted output path: $output_container" >&2
      exit 2
      ;;
  esac
  docker exec "$container" rm -rf -- "$output_container"
fi

# Create the run directory from the container so a shared frontend/agent
# account does not need write permission on the parent NFS directory. The
# launcher writes logs from the host after this directory is prepared.
docker exec "$container" mkdir -p "$output_container"
docker exec "$container" chmod 01777 "$output_container"
mkdir -p "$output_host"
: >"$launcher_log"
: >"$monitor_log"
: >"$foreign_flag"
: >"$done_file"

stage="init"
exit_code=""
monitor_pid=""

write_done() {
  if [[ -s "$foreign_flag" ]]; then
    foreign_process_seen=1
  else
    foreign_process_seen=0
  fi
  {
    echo "exit_code=${exit_code:-1}"
    echo "stage=$stage"
    echo "foreign_process_seen=$foreign_process_seen"
    echo "end=$(date --iso-8601=seconds)"
    echo "container=$container"
    echo "physical_npu=${physical_npu:-}"
    echo "npu_selection=$npu_selection"
    echo "gpu_max_utilization_percent=$gpu_max_utilization"
    echo "gpu_min_free_memory_percent=$gpu_min_free_memory"
    echo "gpu_allow_occupied=$gpu_allow_occupied"
    echo "execution_time=$execution_time"
    echo "execution_date=$execution_date"
    echo "run_date=$run_date"
    echo "csv_date=$csv_date"
    echo "source_backfill_date=$backfill_date"
    echo "infiniops_requested_ref=$infiniops_ref"
    echo "infinirt_requested_ref=$infinirt_ref"
    echo "sha=${sha:-}"
    echo "infinirt_sha=${infinirt_sha:-}"
  } >"$done_file"
}

on_exit() {
  local shell_exit_code=$?
  if [[ ! -s "$done_file" ]]; then
    exit_code="${exit_code:-$shell_exit_code}"
    write_done
  fi
  if [[ -n "$monitor_pid" ]]; then
    wait "$monitor_pid" 2>/dev/null || true
  fi
}
trap on_exit EXIT

docker exec "$container" test -f "$harness"
docker exec "$container" test -f "$exporter"
docker exec "$container" test -f "$partial_exporter"

# Verify that the host and container output roots resolve to the same bind
# mount. The container owns the run directory on NFS, so create and remove the
# marker from the container. A host-side unlink can fail with EPERM in a sticky
# directory when the runner is invoked by another shared account.
output_path_marker=".benchmark-output-path-check-$$-$(date +%s%N)"
docker exec "$container" touch "$output_container/$output_path_marker"
if ! test -f "$output_host/$output_path_marker"; then
  docker exec "$container" rm -f -- "$output_container/$output_path_marker"
  echo "OUTPUT_ROOT and CONTAINER_OUTPUT_ROOT do not point to the same mounted directory" >&2
  exit 2
fi
docker exec "$container" rm -f -- "$output_container/$output_path_marker"

stage="source-update"
prepare_source_repo() {
  local label="$1"
  local path="$2"
  local urls_input="$3"
  local branch="$4"
  local requested_ref="$5"
  local source_date="$6"
  local cutoff requested_commit status_output resolved_ref
  local -a urls
  local url selected_url clone_target

  read -r -a urls <<<"${urls_input//,/ }"
  if [[ "${#urls[@]}" -eq 0 ]]; then
    echo "$label repository URL list is empty" | tee -a "$launcher_log" >&2
    return 2
  fi

  echo "=== source $label prepare start=$(date --iso-8601=seconds) ===" | tee -a "$launcher_log"
  echo "repo_urls=$urls_input branch=$branch path=$path" | tee -a "$launcher_log"

  if ! docker exec "$container" test -d "$path/.git"; then
    if docker exec "$container" test -e "$path"; then
      echo "$label source path exists but is not a Git repository: $path" | tee -a "$launcher_log" >&2
      return 2
    fi
    echo "action=clone" | tee -a "$launcher_log"
    if ! docker exec "$container" mkdir -p "$(dirname "$path")"; then
      return 2
    fi
    clone_target="${path}.clone-$$"
    docker exec "$container" test ! -e "$clone_target"
    for url in "${urls[@]}"; do
      echo "clone_candidate=$url" | tee -a "$launcher_log"
      if ! docker exec "$container" timeout "$source_probe_timeout" git ls-remote "$url" HEAD >/dev/null 2>&1; then
        echo "clone_candidate_unreachable=$url" | tee -a "$launcher_log"
        continue
      fi
      if docker exec "$container" timeout "$source_operation_timeout" git clone --branch "$branch" "$url" "$clone_target" 2>&1 | tee -a "$launcher_log"; then
        docker exec "$container" mv "$clone_target" "$path"
        selected_url="$url"
        break
      fi
      echo "clone_candidate_failed=$url" | tee -a "$launcher_log" >&2
      docker exec "$container" rm -rf -- "$clone_target"
    done
    if [[ -z "$selected_url" ]]; then
      echo "$label clone failed for all repository URLs" | tee -a "$launcher_log" >&2
      return 2
    fi
  else
    echo "action=fetch" | tee -a "$launcher_log"
    if ! status_output="$(docker exec "$container" git -c "safe.directory=$path" -C "$path" status --porcelain=v1)"; then
      printf '%s\n' "$status_output" | tee -a "$launcher_log" >&2
      return 2
    fi
    if [[ -n "$status_output" ]]; then
      echo "$label repository has local changes; refusing to overwrite:" | tee -a "$launcher_log" >&2
      printf '%s\n' "$status_output" | tee -a "$launcher_log" >&2
      return 2
    fi
    for url in "${urls[@]}"; do
      echo "fetch_candidate=$url" | tee -a "$launcher_log"
      if ! docker exec "$container" timeout "$source_probe_timeout" git ls-remote "$url" HEAD >/dev/null 2>&1; then
        echo "fetch_candidate_unreachable=$url" | tee -a "$launcher_log"
        continue
      fi
      if ! docker exec "$container" git -c "safe.directory=$path" -C "$path" remote set-url origin "$url" 2>&1 | tee -a "$launcher_log"; then
        continue
      fi
      if docker exec "$container" timeout "$source_operation_timeout" git -c "safe.directory=$path" -C "$path" fetch --prune --tags origin 2>&1 | tee -a "$launcher_log"; then
        selected_url="$url"
        break
      fi
      echo "fetch_candidate_failed=$url" | tee -a "$launcher_log" >&2
    done
    if [[ -z "$selected_url" ]]; then
      echo "$label fetch failed for all repository URLs" | tee -a "$launcher_log" >&2
      return 2
    fi
  fi

  echo "$label selected_repo=$selected_url" | tee -a "$launcher_log"

  if ! docker exec "$container" git -c "safe.directory=$path" -C "$path" show-ref --verify --quiet "refs/remotes/origin/$branch"; then
    echo "$label remote branch not found: origin/$branch" | tee -a "$launcher_log" >&2
    return 2
  fi

  if [[ -n "$requested_ref" ]]; then
    echo "version=explicit ref=$requested_ref" | tee -a "$launcher_log"
    if ! resolved_ref="$(docker exec "$container" git -c "safe.directory=$path" -C "$path" rev-parse --verify "$requested_ref^{commit}")"; then
      echo "$label ref was not found after fetch: $requested_ref" | tee -a "$launcher_log" >&2
      return 2
    fi
    if ! docker exec "$container" git -c "safe.directory=$path" -C "$path" checkout -q "$resolved_ref" 2>&1 | tee -a "$launcher_log"; then
      return 2
    fi
  elif [[ -n "$source_date" ]]; then
    cutoff="${source_date}T23:59:59+08:00"
    echo "version=backfill date=$source_date cutoff=$cutoff" | tee -a "$launcher_log"
    if ! requested_commit="$(docker exec "$container" git -c "safe.directory=$path" -C "$path" rev-list -1 --before="$cutoff" "refs/remotes/origin/$branch")"; then
      echo "$label failed to resolve a commit before $cutoff" | tee -a "$launcher_log" >&2
      return 2
    fi
    if [[ -z "$requested_commit" ]]; then
      echo "no $label commit exists before $cutoff on origin/$branch" | tee -a "$launcher_log" >&2
      return 2
    fi
    echo "$label selected_commit=$requested_commit" | tee -a "$launcher_log"
    if ! docker exec "$container" git -c "safe.directory=$path" -C "$path" checkout -q "$requested_commit" 2>&1 | tee -a "$launcher_log"; then
      return 2
    fi
  else
    echo "version=latest" | tee -a "$launcher_log"
    if docker exec "$container" git -c "safe.directory=$path" -C "$path" show-ref --verify --quiet "refs/heads/$branch"; then
      if ! docker exec "$container" git -c "safe.directory=$path" -C "$path" checkout -q "$branch" 2>&1 | tee -a "$launcher_log"; then
        return 2
      fi
      if ! docker exec "$container" git -c "safe.directory=$path" -C "$path" merge --ff-only "refs/remotes/origin/$branch" 2>&1 | tee -a "$launcher_log"; then
        return 2
      fi
    else
      if ! docker exec "$container" git -c "safe.directory=$path" -C "$path" checkout -q -b "$branch" "refs/remotes/origin/$branch" 2>&1 | tee -a "$launcher_log"; then
        return 2
      fi
    fi
  fi

  echo "$label commit=$(docker exec "$container" git -c "safe.directory=$path" -C "$path" rev-parse HEAD)" | tee -a "$launcher_log"
  echo "=== source $label prepare end=$(date --iso-8601=seconds) ===" | tee -a "$launcher_log"
}

if [[ "$update_sources" == "1" ]]; then
  prepare_source_repo "InfiniOps" "$source_root" "$infiniops_repo_urls_input" "$infiniops_branch" "$infiniops_ref" "$backfill_date"
  prepare_source_repo "InfiniRT" "$infinirt_source_root" "$infinirt_repo_urls_input" "$infinirt_branch" "$infinirt_ref" "$backfill_date"
else
  {
    echo "source_update=disabled"
    docker exec "$container" test -d "$source_root/.git"
    docker exec "$container" test -d "$infinirt_source_root/.git"
  } | tee -a "$launcher_log"
fi

npu_location_for_logical() {
  local target_npu="$1"
  python3 -c '
import sys

target = int(sys.argv[1])
lines = sys.argv[2].splitlines()
header = ""
for line in lines:
    if "NPU ID" in line and "Chip Logic ID" in line:
        header = line
        break
if not header:
    raise SystemExit("npu-smi mapping did not contain a chip logic header")

labels = ["NPU ID", "Chip ID", "Chip Logic ID"]
if "Chip Phy-ID" in header:
    labels.append("Chip Phy-ID")
labels.append("Chip Name")
starts = [header.find(label) for label in labels]
starts.append(len(header))

def values(line):
    return [line[starts[index]:starts[index + 1]].strip() for index in range(len(labels))]

for line in lines:
    columns = values(line)
    if len(columns) < len(labels) or not columns[2].isdigit():
        continue
    if int(columns[2]) != target:
        continue
    if not columns[0].isdigit() or not columns[1].isdigit():
        continue
    print(columns[0], columns[1], columns[-1])
    break
else:
    raise SystemExit(f"logical NPU {target} was not found in npu-smi mapping")
' "$target_npu" "$npu_logical_mapping"
}

npu_pids_from_table() {
  local target_npu="$1"
  local npu_location target_card target_chip
  npu_location="$(npu_location_for_logical "$target_npu")"
  target_card="${npu_location%% *}"
  target_chip="${npu_location#* }"
  target_chip="${target_chip%% *}"
  sed -n '/Process id/,$p' | awk -F '|' -v target_card="$target_card" -v target_chip="$target_chip" '
    function trim(text) { gsub(/^[[:space:]]+|[[:space:]]+$/, "", text); return text }
    NF >= 5 && $2 ~ /^[[:space:]]*[0-9]+[[:space:]]+[0-9]+[[:space:]]*$/ {
      split(trim($2), chip, /[[:space:]]+/)
      if (chip[1] + 0 == target_card + 0 && chip[2] + 0 == target_chip + 0) print trim($3)
    }
  '
}

npu_resource_for_logical() {
  local target_npu="$1"
  local npu_location target_card target_chip usage aicore hbm_used
  npu_location="$(npu_location_for_logical "$target_npu")"
  target_card="${npu_location%% *}"
  target_chip="${npu_location#* }"
  target_chip="${target_chip%% *}"
  if ! usage="$(npu-smi info -t usages -i "$target_card" -c "$target_chip" 2>&1)"; then
    printf '%s\n' "$usage" >&2
    return 2
  fi
  aicore="$(awk -F: '/Aicore Usage Rate/ { gsub(/[^0-9]/, "", $2); print $2; exit }' <<<"$usage")"
  hbm_used="$(awk -F: '/HBM Usage Rate/ { gsub(/[^0-9]/, "", $2); print $2; exit }' <<<"$usage")"
  if [[ -z "$aicore" || -z "$hbm_used" ]]; then
    echo "NPU $target_npu usage information was not found" >&2
    return 2
  fi
  printf '%s\n%s\n' "$aicore" "$((100 - hbm_used))"
}

select_idle_npu() {
  local npu_table npu_id npu_location target_card target_chip pids utilization free_memory reason
  local -a logical_npus candidates rejected busy_npus
  if ! npu_table="$(npu-smi info 2>&1)"; then
    echo "npu-smi info failed:" >&2
    printf '%s\n' "$npu_table" >&2
    return 2
  fi
  mapfile -t logical_npus < <(python3 -c '
import sys

lines = sys.argv[1].splitlines()
header = ""
for line in lines:
    if "NPU ID" in line and "Chip Logic ID" in line:
        header = line
        break
if not header:
    raise SystemExit("npu-smi mapping did not contain a chip logic header")
labels = ["NPU ID", "Chip ID", "Chip Logic ID"]
if "Chip Phy-ID" in header:
    labels.append("Chip Phy-ID")
labels.append("Chip Name")
starts = [header.find(label) for label in labels]
starts.append(len(header))

def values(line):
    return [line[starts[index]:starts[index + 1]].strip() for index in range(len(labels))]

for line in lines:
    columns = values(line)
    if len(columns) == len(labels) and columns[2].isdigit() and "Ascend" in columns[-1]:
        print(columns[2])
 ' "$npu_logical_mapping")
  if [[ "${#logical_npus[@]}" -eq 0 ]]; then
    echo "npu-smi mapping did not contain any Ascend chips" >&2
    return 2
  fi

  for npu_id in "${logical_npus[@]}"; do
    reason=""
    pids="$(npu_pids_from_table "$npu_id" <<<"$npu_table")"
    if [[ -n "$pids" ]]; then
      busy_npus+=("$npu_id")
      if [[ "$gpu_allow_occupied" != "1" ]]; then
        reason="busy"
      fi
    fi
    if ! npu_resource_output="$(npu_resource_for_logical "$npu_id")"; then
      reason="${reason:+$reason,}usage_query_failed"
      rejected+=("$npu_id($reason)")
      continue
    fi
    mapfile -t npu_resource <<<"$npu_resource_output"
    utilization="${npu_resource[0]}"
    free_memory="${npu_resource[1]}"
    if (( utilization > gpu_max_utilization )); then
      reason="${reason:+$reason,}utilization=$utilization"
    fi
    if (( free_memory < gpu_min_free_memory )); then
      reason="${reason:+$reason,}free_memory=${free_memory}%"
    fi
    if [[ -n "$reason" ]]; then
      rejected+=("$npu_id($reason)")
    else
      candidates+=("$npu_id")
    fi
  done

  if [[ "${#candidates[@]}" -eq 0 ]]; then
    printf 'no idle Ascend NPU found; rejected: %s\n' \
      "${rejected[*]:-unknown}" >&2
    return 2
  fi
  physical_npu="${candidates[0]}"
  npu_candidates="$(IFS=,; echo "${candidates[*]}")"
  npu_busy="$(IFS=,; echo "${busy_npus[*]:--}")"
  npu_rejected="$(IFS=';'; echo "${rejected[*]:--}")"
  npu_selection="auto"
}

assert_npu_idle() {
  local utilization free_memory
  process_snapshot="$(npu-smi info 2>&1)"
  process_pids="$(npu_pids_from_table "$physical_npu" <<<"$process_snapshot")"
  if [[ -n "$process_pids" && "$gpu_allow_occupied" != "1" ]]; then
    printf '%s\n' "$process_snapshot" >&2
    echo "physical Ascend NPU $physical_npu has a running process" >&2
    return 2
  fi
  if ! npu_resource_output="$(npu_resource_for_logical "$physical_npu")"; then
    return 2
  fi
  mapfile -t npu_resource <<<"$npu_resource_output"
  utilization="${npu_resource[0]}"
  free_memory="${npu_resource[1]}"
  if [[ -z "$utilization" || "$utilization" =~ [^0-9] ]] || (( utilization > gpu_max_utilization )); then
    echo "physical Ascend NPU $physical_npu AICore utilization exceeds ${gpu_max_utilization}%: $utilization" >&2
    return 2
  fi
  if [[ -z "$free_memory" || "$free_memory" =~ [^0-9.] ]] || python3 -c 'import sys; raise SystemExit(float(sys.argv[1]) >= float(sys.argv[2]))' "$free_memory" "$gpu_min_free_memory"; then
    echo "physical Ascend NPU $physical_npu free HBM is below ${gpu_min_free_memory}%: $free_memory" >&2
    return 2
  fi
}

if [[ -n "$requested_npu" ]]; then
  if [[ ! "$requested_npu" =~ ^[0-9]+$ ]]; then
    echo "invalid physical NPU index: $requested_npu" >&2
    exit 2
  fi
  physical_npu="$requested_npu"
  npu_selection="explicit"
elif [[ "$auto_select_npu" == "1" ]]; then
  if ! select_idle_npu; then
    exit 2
  fi
else
  echo "usage: [CONTAINER=<name>] [AUTO_SELECT_NPU=0] [PHYSICAL_NPU=<index>] $0 [<physical-npu-index>]" >&2
  exit 2
fi

{
  echo "npu_selection=$npu_selection"
  echo "physical_npu=$physical_npu"
  echo "npu_candidates=$npu_candidates"
  echo "npu_busy=$npu_busy"
  echo "npu_rejected=$npu_rejected"
} >>"$launcher_log"

stage="build"
if [[ "$build_infiniops" == "1" ]]; then
  docker exec "$container" mkdir -p "$build_dir"
  {
    echo "=== build start=$(date --iso-8601=seconds) ==="
    echo "build_dir=$build_dir"
    echo "cmake_build_parallel_level=$cmake_build_parallel_level"
    echo "soc_version=$soc_version"
    echo "build_ascend_custom=$build_ascend_custom"
  } >>"$launcher_log"
  build_exit=0
  docker exec \
    -e CMAKE_BUILD_PARALLEL_LEVEL="$cmake_build_parallel_level" \
    "$container" python3 -m pip install \
      --no-build-isolation \
      --no-deps \
      --force-reinstall \
      "$source_root" \
      "--config-settings=build-dir=$build_dir" \
      --config-settings=cmake.define.AUTO_DETECT_DEVICES=OFF \
      --config-settings=cmake.define.AUTO_DETECT_BACKENDS=OFF \
      --config-settings=cmake.define.WITH_CPU=ON \
      --config-settings=cmake.define.WITH_ASCEND=ON \
      --config-settings=cmake.define.WITH_TORCH=ON \
      --config-settings=cmake.define.WITH_LINKED=OFF \
      "--config-settings=cmake.define.INFINI_RT_SOURCE_DIR=$infinirt_source_root" \
      "--config-settings=cmake.define.BUILD_ASCEND_CUSTOM=$build_ascend_custom" \
      "--config-settings=cmake.define.SOC_VERSION=$soc_version" \
      2>&1 | tee -a "$launcher_log" || build_exit=$?
  if [[ "$build_exit" -ne 0 ]]; then
    echo "build_exit=$build_exit" | tee -a "$launcher_log"
    exit "$build_exit"
  fi
  echo "=== build end=$(date --iso-8601=seconds) ===" >>"$launcher_log"
fi

# Compilation can take long enough for the initially selected device to become
# occupied. Refresh automatic selection immediately before benchmark preflight.
# An explicit NPU request is never changed automatically.
if [[ "$npu_selection" == "auto" ]]; then
  previous_npu="$physical_npu"
  if ! select_idle_npu; then
    exit 2
  fi
  {
    echo "npu_reselection=auto"
    echo "previous_npu=$previous_npu"
    echo "physical_npu=$physical_npu"
    echo "npu_candidates=$npu_candidates"
    echo "npu_busy=$npu_busy"
    echo "npu_rejected=$npu_rejected"
  } >>"$launcher_log"
fi

metadata="${INFINIOPS_METADATA:-$source_root/generated/torch_ops_metadata.json}"
docker exec "$container" test -f "$metadata"
sha="$(docker exec "$container" git -c "safe.directory=$source_root" -C "$source_root" rev-parse HEAD)"
infinirt_sha="$(docker exec "$container" git -c "safe.directory=$infinirt_source_root" -C "$infinirt_source_root" rev-parse HEAD)"

stage="preflight"
if ! assert_npu_idle; then
  exit 2
fi

# infini/ops.so reaches libtorch_python.so through a transitive dependency, so
# preserve the image CANN paths and add the installed Torch library paths.
container_ld_library_path="$({
  docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$container" |
    sed -n 's/^LD_LIBRARY_PATH=//p'
} | tail -n 1)"
torch_library_path="$(docker exec "$container" python3 -c '
import importlib.util
from pathlib import Path

paths = []
for package in ("torch", "torch_npu"):
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        raise SystemExit(f"cannot locate {package}")
    paths.append(str(Path(spec.origin).resolve().parent / "lib"))
print(":".join(paths))
')"
runtime_ld_library_path="$torch_library_path"
if [[ -n "$container_ld_library_path" ]]; then
  runtime_ld_library_path+=":$container_ld_library_path"
fi
npu_resource_output="$(npu_resource_for_logical "$physical_npu")"
mapfile -t npu_resource <<<"$npu_resource_output"
npu_free_memory_percent="${npu_resource[1]}"

{
  echo "start=$(date --iso-8601=seconds)"
  echo "container=$container"
  echo "container_id=$container_id"
  echo "physical_npu=$physical_npu"
  echo "gpu_max_utilization_percent=$gpu_max_utilization"
  echo "gpu_min_free_memory_percent=$gpu_min_free_memory"
  echo "gpu_allow_occupied=$gpu_allow_occupied"
  echo "gpu_free_memory_percent=$npu_free_memory_percent"
  echo "execution_time=$execution_time"
  echo "run_date=$run_date"
  echo "csv_date=$csv_date"
  echo "source_update=$update_sources"
  echo "source_backfill_date=$backfill_date"
  echo "infiniops_repo_urls=$infiniops_repo_urls_input"
  echo "infiniops_branch=$infiniops_branch"
  echo "infiniops_requested_ref=$infiniops_ref"
  echo "infinirt_repo_urls=$infinirt_repo_urls_input"
  echo "infinirt_branch=$infinirt_branch"
  echo "infinirt_requested_ref=$infinirt_ref"
  echo "source_root=$source_root"
  echo "sha=$sha"
  echo "infinirt_sha=$infinirt_sha"
  echo "metadata=$metadata"
  echo "build_infiniops=$build_infiniops"
  echo "build_ascend_custom=$build_ascend_custom"
  echo "soc_version=$soc_version"
  echo "dtypes=$dtypes_input"
  echo "profiles=$profiles_input"
  echo "decode_4k_ops=$decode_4k_ops"
  echo "decode_4k_small_ops=$decode_4k_small_ops"
  echo "decode_8k_ops=$decode_8k_ops"
  echo "prefill_4k_ops=$prefill_4k_ops"
  echo "torch_library_path=$torch_library_path"
  echo "=== host preflight ==="
  npu-smi info
  printf '%s\n' "$process_snapshot"
  echo "=== container preflight ==="
  docker exec \
    -e ASCEND_RT_VISIBLE_DEVICES="$physical_npu" \
    -e LD_LIBRARY_PATH="$runtime_ld_library_path" \
    "$container" python3 -c \
    "import torch, torch_npu, infini.ops; assert torch.npu.is_available(); assert torch.npu.device_count() == 1; print('torch', torch.__version__); print('torch_npu', torch_npu.__version__); print('npu_count', torch.npu.device_count()); print('infini_ops', infini.ops.__file__)"
} >>"$launcher_log" 2>&1

stage="npu-select"

monitor() {
  local snapshot pids pid cgroup cmdline
  while [[ ! -s "$done_file" ]]; do
    echo "sample=$(date --iso-8601=seconds)"
    if ! snapshot="$(npu-smi info 2>&1)"; then
      echo "monitor_error=$snapshot"
      echo "npu-smi-error" >>"$foreign_flag"
      docker exec "$container" pkill -f "$harness" 2>/dev/null || true
      break
    fi
    printf '%s\n' "$snapshot"
    pids="$(npu_pids_from_table "$physical_npu" <<<"$snapshot")"
    if [[ -z "$pids" ]]; then
      sleep 2
      continue
    fi
    for pid in $pids; do
      if [[ ! -e "/proc/$pid" ]]; then
        echo "process_owner=exited_before_attribution pid=$pid"
        continue
      fi
      cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
      if [[ -z "$cmdline" ]]; then
        echo "process_owner=unattributed_empty_cmdline pid=$pid cgroup=$(tr '\n' ' ' <"/proc/$pid/cgroup" 2>/dev/null || true)"
        continue
      fi
      cgroup="$(tr '\n' ' ' <"/proc/$pid/cgroup" 2>/dev/null || true)"
      if [[ "$cgroup" == *"$container_id"* ]]; then
        echo "process_owner=benchmark_container pid=$pid cmdline=$cmdline"
      else
        if [[ "$gpu_allow_occupied" == "1" ]]; then
          echo "process_owner=foreign_allowed pid=$pid cgroup=$cgroup cmdline=$cmdline"
        else
          echo "process_owner=foreign pid=$pid cgroup=$cgroup cmdline=$cmdline"
          if [[ ! -s "$foreign_flag" ]]; then
            echo "$pid" >>"$foreign_flag"
            docker exec "$container" pkill -f "$harness" 2>/dev/null || true
          fi
        fi
      fi
    done
    sleep 2
  done
}

run_profile() {
  local dtype="$1"
  local profile="$2"
  local selected_ops="${3:-}"
  local output="$output_container/$dtype/$profile"
  local -a command=(
    python3 "$harness"
    --metadata "$metadata"
    --output-dir "$output"
    --device npu
    --dtype "$dtype"
    --sha "$sha"
    --min-run-time 0.02
    --timeout 20
    --resume
  )
  if [[ -n "$selected_ops" && "$selected_ops" != "all" ]]; then
    command+=(--ops "$selected_ops")
  fi
  {
    echo "=== dtype=$dtype profile=$profile start=$(date --iso-8601=seconds) ==="
    printf 'command='
    printf '%q ' "${command[@]}"
    echo
  } >>"$launcher_log"
  docker exec \
    -e ASCEND_RT_VISIBLE_DEVICES="$physical_npu" \
    -e LD_LIBRARY_PATH="$runtime_ld_library_path" \
    -e INFINIOPS_BENCHMARK_PROFILE="$profile" \
    "$container" "${command[@]}" 2>&1 | tee -a "$launcher_log"
  return "${PIPESTATUS[0]}"
}

stage="benchmark"
monitor >>"$monitor_log" 2>&1 &
monitor_pid=$!

exit_code=0
exporter_sources=()
for dtype in "${dtypes[@]}"; do
  echo "=== dtype=$dtype start=$(date --iso-8601=seconds) ===" >>"$launcher_log"
  for profile in "${profiles[@]}"; do
    case "$profile" in
      decode-4k) selected_ops="$decode_4k_ops" ;;
      decode-4k-small) selected_ops="$decode_4k_small_ops" ;;
      decode-8k) selected_ops="$decode_8k_ops" ;;
      prefill-4k) selected_ops="$prefill_4k_ops" ;;
    esac
    run_profile "$dtype" "$profile" "$selected_ops" || exit_code=$?
    result_path="$output_container/$dtype/$profile/results.json"
    if docker exec "$container" test -f "$result_path"; then
      exporter_sources+=("$result_path")
    else
      echo "missing_result=$result_path" | tee -a "$launcher_log"
      exit_code=1
    fi
  done
  echo "=== dtype=$dtype end=$(date --iso-8601=seconds) ===" >>"$launcher_log"
done

stage="export"
if [[ "${#exporter_sources[@]}" -gt 0 ]]; then
  docker exec \
    -e LD_LIBRARY_PATH="$runtime_ld_library_path" \
    "$container" python3 "$exporter" \
    "${exporter_sources[@]}" \
    --output "$output_container/$output_csv_name" \
    --diagnostics "$output_container/diagnostics.csv" \
    --report "$output_container/report.md" \
    --platform ascend \
    --date "$csv_date" \
    --infiniops-commit "$sha" \
    --infinirt-commit "$infinirt_sha" \
    --tester "$tester" 2>&1 | tee -a "$launcher_log"
  exporter_exit=${PIPESTATUS[0]}
  if [[ "$exporter_exit" -ne 0 ]]; then
    exit_code="$exporter_exit"
  fi
else
  echo "no result JSON files were available for export" | tee -a "$launcher_log"
  exit_code=1
fi

if [[ "$exit_code" -ne 0 || -s "$foreign_flag" ]]; then
  if docker exec "$container" python3 -c '
import pathlib, sys
root = pathlib.Path(sys.argv[1])
found = any(path.name in {"results.json", "results.jsonl"} for path in root.rglob("*") if path.is_file())
raise SystemExit(0 if found else 1)
' "$output_container"; then
    if [[ -s "$foreign_flag" ]]; then
      partial_failure_category=foreign_process
      partial_failure_reason="foreign process detected during benchmark"
    else
      partial_failure_category=benchmark_failure
      partial_failure_reason="runner exit code $exit_code"
    fi
    stage="partial-export"
    echo "=== partial export start=$(date --iso-8601=seconds) ===" | tee -a "$launcher_log"
    partial_exit=0
    docker exec "$container" python3 "$partial_exporter" \
      --output-root "$output_container" \
      --output-name "$output_csv_name" \
      --platform ascend \
      --date "$csv_date" \
      --tester "$tester" \
      --infiniops-commit "$sha" \
      --infinirt-commit "$infinirt_sha" \
      --failure-category "$partial_failure_category" \
      --failure-reason "$partial_failure_reason" \
      2>&1 | tee -a "$launcher_log" || partial_exit=$?
    if [[ "$partial_exit" -eq 0 ]]; then
      echo "partial_csv=$output_container/${output_csv_name%.csv}.partial.csv" | tee -a "$launcher_log"
    else
      echo "partial_export_failed=$partial_exit" | tee -a "$launcher_log"
    fi
  else
    echo "partial_export_skipped=no complete or checkpoint result files" | tee -a "$launcher_log"
  fi
fi

{
  echo "=== post-run npu-smi ==="
  npu-smi info
  echo "end=$(date --iso-8601=seconds)"
} >>"$launcher_log" 2>&1

write_done
wait "$monitor_pid"
trap - EXIT
if [[ "$foreign_process_seen" != 0 ]]; then
  exit 3
fi
exit "$exit_code"

#!/usr/bin/env bash
set -euo pipefail

container="${CONTAINER:-baoming_infiniops}"
requested_gpu="${1:-${PHYSICAL_GPU:-}}"
auto_select_gpu="${AUTO_SELECT_GPU:-1}"
gpu_max_utilization="${GPU_MAX_UTILIZATION_PERCENT:-5}"
gpu_min_free_memory="${GPU_MIN_FREE_MEMORY_PERCENT:-0}"
gpu_allow_occupied="${GPU_ALLOW_OCCUPIED:-0}"
physical_gpu=""
gpu_selection=""
gpu_candidates=""
gpu_busy=""
gpu_rejected=""
gpu_uuid=""
sha=""
infinirt_sha=""

execution_time="${EXECUTION_TIME:-$(date +%Y%m%d-%H%M%S)}"
execution_date="${execution_time:0:8}"
tester="${TESTER:-baoming}"
build_infiniops="${BUILD_INFINIOPS:-1}"
update_sources="${UPDATE_SOURCES:-1}"
backfill_date="${BACKFILL_DATE:-}"
reuse_run_dir="${REUSE_RUN_DIR:-0}"
infiniops_ref="${INFINIOPS_REF:-}"
infinirt_ref="${INFINIRT_REF:-}"
device_id="${DEVICE_ID:-nvidia-a100}"
output_root="${OUTPUT_ROOT:-/data-aisoft/infinibench_data/ops}"
min_run_time="${MIN_RUN_TIME:-0.02}"
benchmark_iterations="${BENCHMARK_ITERATIONS:-0}"
dtypes_input="${DTYPES:-float16,float32,bfloat16}"
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
if [[ ! "$min_run_time" =~ ^[0-9]+([.][0-9]+)?$ ]] || ! awk -v value="$min_run_time" 'BEGIN { exit !(value > 0) }'; then
  echo "invalid MIN_RUN_TIME: $min_run_time; expected a positive number of seconds" >&2
  exit 2
fi
if [[ ! "$benchmark_iterations" =~ ^[0-9]+$ ]]; then
  echo "invalid BENCHMARK_ITERATIONS: $benchmark_iterations; expected a non-negative integer" >&2
  exit 2
fi
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
cmake_build_parallel_level="${CMAKE_BUILD_PARALLEL_LEVEL:-16}"
cuda_architectures="${CUDA_ARCHITECTURES:-80}"
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

target_ops=abs,add,addmm,amax,amin,argmax,bitwise_and,bitwise_not,bitwise_or,bmm,cat,ceil,clamp,cos,cumsum,div,eq,exp,floor,gather,ge,gelu,gt,hardtanh,index_select,le,log,log_softmax,lt,matmul,maximum,mean,minimum,mm,mul,ne,neg,pow,reciprocal,relu,round,rsqrt,sigmoid,sign,silu,sin,softmax,sort,sqrt,sub,sum,tanh,topk
small_ops=abs,add,avg_pool2d,bmm,gather,fft_fft,gelu,matmul,mm,mul,relu,sigmoid,silu,softmax,special_gammaln,sparse_sampled_addmm,sum,where
decode_4k_ops="${NVIDIA_DECODE_4K_OPS:-all}"
decode_4k_small_ops="${NVIDIA_DECODE_4K_SMALL_OPS:-$small_ops}"
decode_8k_ops="${NVIDIA_DECODE_8K_OPS:-$target_ops}"
prefill_4k_ops="${NVIDIA_PREFILL_4K_OPS:-$target_ops}"

"$host_tools_root/scripts/ensure_benchmark_container.sh" "$container"
container_id="$(docker inspect -f '{{.Id}}' "$container")"

# Backfill changes shared source trees, so serialize all runner executions on
# this machine. The lock is released automatically when this process exits.
docker exec "$container" mkdir -p "$(dirname "$runner_lock_container")"
docker exec "$container" touch "$runner_lock_container"
docker exec "$container" chmod 0666 "$runner_lock_container"
exec 9>"$runner_lock"
if ! flock -n 9; then
  echo "another NVIDIA A100 benchmark runner is active: $runner_lock" >&2
  exit 2
fi

# A date-keyed output directory is deterministic. Refuse to mix a new attempt
# with stale checkpoint files. REUSE_RUN_DIR=1 explicitly replaces the fixed
# directory for a retry after the machine-level lock has been acquired.
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

write_done_on_exit() {
  local exit_code=$?
  local foreign_process_seen=0
  if [[ -s "$foreign_flag" ]]; then
    foreign_process_seen=1
  fi
  {
    echo "exit_code=$exit_code"
    echo "foreign_process_seen=$foreign_process_seen"
    echo "end=$(date --iso-8601=seconds)"
    echo "container=$container"
    echo "physical_gpu=${physical_gpu:-}"
    echo "gpu_uuid=${gpu_uuid:-}"
    echo "gpu_max_utilization_percent=$gpu_max_utilization"
    echo "gpu_min_free_memory_percent=$gpu_min_free_memory"
    echo "gpu_allow_occupied=$gpu_allow_occupied"
    echo "execution_time=$execution_time"
    echo "execution_date=$execution_date"
    echo "run_date=$run_date"
    echo "source_update=$update_sources"
    echo "csv_date=$csv_date"
    echo "source_backfill_date=$backfill_date"
    echo "infiniops_requested_ref=$infiniops_ref"
    echo "infinirt_requested_ref=$infinirt_ref"
    echo "sha=${sha:-}"
    echo "infinirt_sha=${infinirt_sha:-}"
  } >"$done_file"
}
trap write_done_on_exit EXIT

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

select_idle_gpu_once() {
  local gpu_table process_table selection_output
  if ! gpu_table="$(nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.total,memory.used --format=csv,noheader,nounits 2>&1)"; then
    echo "nvidia-smi GPU query failed:" >&2
    printf '%s\n' "$gpu_table" >&2
    return 2
  fi
  if ! process_table="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits 2>&1)"; then
    echo "nvidia-smi process query failed:" >&2
    printf '%s\n' "$process_table" >&2
    return 2
  fi

  selection_output="$(
    {
      printf '%s\n' "$gpu_table"
      printf '\n__NVIDIA_SMI_PROCESS_TABLE__\n'
      printf '%s\n' "$process_table"
    } | python3 -c '
import csv
import io
import sys

raw = sys.stdin.read()
marker = "__NVIDIA_SMI_PROCESS_TABLE__"
if marker not in raw:
    raise SystemExit("nvidia-smi output did not contain the process-table marker")
gpu_text, process_text = raw.split(marker, 1)

devices = []
for row in csv.reader(io.StringIO(gpu_text)):
    if not row or len(row) != 5:
        continue
    index, uuid, utilization, total_memory, used_memory = [
        column.strip() for column in row
    ]
    if not index.isdigit():
        continue
    devices.append((int(index), uuid, utilization, total_memory, used_memory))
if not devices:
    raise SystemExit("nvidia-smi output did not contain any NVIDIA GPUs")

busy_uuids = {
    line.strip().split(",", 1)[0].strip()
    for line in process_text.splitlines()
    if line.strip()
}
uuid_to_index = {uuid: index for index, uuid, _, _, _ in devices}
busy = sorted(
    index for uuid, index in uuid_to_index.items() if uuid in busy_uuids
)

eligible = []
rejected = []
max_utilization = int(sys.argv[1])
min_free_memory = float(sys.argv[2])
allow_occupied = sys.argv[3] == "1"

for index, uuid, utilization, total_memory, used_memory in devices:
    reasons = []
    if uuid in busy_uuids and not allow_occupied:
        reasons.append("busy")
    try:
        utilization_value = int(utilization)
    except ValueError:
        reasons.append(f"invalid_utilization={utilization}")
    else:
        if utilization_value > max_utilization:
            reasons.append(f"utilization={utilization_value}")
    try:
        total_memory_value = int(total_memory)
        used_memory_value = int(used_memory)
    except ValueError:
        reasons.append(
            f"invalid_memory={total_memory},{used_memory}"
        )
    else:
        if total_memory_value <= 0:
            reasons.append(f"invalid_total_memory={total_memory}")
        else:
            free_memory = round(
                (total_memory_value - used_memory_value)
                * 100.0
                / total_memory_value,
                2,
            )
            if free_memory < min_free_memory:
                reasons.append(f"free_memory={free_memory:.2f}%")
    if reasons:
        rejected.append(str(index) + "(" + ",".join(reasons) + ")")
    else:
        eligible.append(index)

if not eligible:
    raise SystemExit(
        "no idle NVIDIA GPU found; rejected: "
        + (", ".join(rejected) if rejected else "unknown")
    )

print(eligible[0])
print(",".join(str(item) for item in eligible))
print(",".join(str(item) for item in busy) or "-")
print(";".join(rejected) or "-")
' "$gpu_max_utilization" "$gpu_min_free_memory" "$gpu_allow_occupied"
  )" || return 2

  mapfile -t selection_rows <<<"$selection_output"
  if [[ "${#selection_rows[@]}" -ne 4 ]]; then
    echo "unexpected idle-GPU selector output: $selection_output" >&2
    return 2
  fi
  physical_gpu="${selection_rows[0]}"
  gpu_candidates="${selection_rows[1]}"
  gpu_busy="${selection_rows[2]}"
  gpu_rejected="${selection_rows[3]}"
}

select_idle_gpu() {
  local sample current_candidates candidate persistent_candidates
  persistent_candidates=""
  for sample in 1 2 3; do
    if ! select_idle_gpu_once; then
      return 2
    fi
    current_candidates="$gpu_candidates"
    if [[ "$sample" -eq 1 ]]; then
      persistent_candidates="$current_candidates"
    else
      local next_candidates=""
      for candidate in ${persistent_candidates//,/ }; do
        if [[ ",$current_candidates," == *",$candidate,"* ]]; then
          next_candidates="${next_candidates:+$next_candidates,}$candidate"
        fi
      done
      persistent_candidates="$next_candidates"
    fi
    if [[ -z "$persistent_candidates" ]]; then
      echo "no NVIDIA GPU remained idle across all three selection samples" >&2
      return 2
    fi
    if [[ "$sample" -lt 3 ]]; then
      sleep 2
    fi
  done
  gpu_candidates="$persistent_candidates"
  physical_gpu="${gpu_candidates%%,*}"
  gpu_selection="auto"
}

if [[ -n "$requested_gpu" ]]; then
  if [[ ! "$requested_gpu" =~ ^[0-9]+$ ]]; then
    echo "invalid physical GPU index: $requested_gpu" >&2
    exit 2
  fi
  physical_gpu="$requested_gpu"
  gpu_selection="explicit"
elif [[ "$auto_select_gpu" == "1" ]]; then
  if ! select_idle_gpu; then
    exit 2
  fi
else
  echo "usage: [CONTAINER=<name>] [AUTO_SELECT_GPU=0] [PHYSICAL_GPU=<index>] $0 [<physical-gpu-index>]" >&2
  exit 2
fi

{
  echo "gpu_selection=$gpu_selection"
  echo "physical_gpu=$physical_gpu"
  echo "gpu_candidates=$gpu_candidates"
  echo "gpu_busy=$gpu_busy"
  echo "gpu_rejected=$gpu_rejected"
} >>"$launcher_log"

if [[ "$build_infiniops" == "1" ]]; then
  docker exec "$container" mkdir -p "$build_dir"
  {
    echo "=== build start=$(date --iso-8601=seconds) ==="
    echo "build_dir=$build_dir"
    echo "cmake_build_parallel_level=$cmake_build_parallel_level"
    echo "cuda_architectures=$cuda_architectures"
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
      --config-settings=cmake.define.WITH_NVIDIA=ON \
      --config-settings=cmake.define.WITH_TORCH=ON \
      --config-settings=cmake.define.WITH_LINKED=OFF \
      "--config-settings=cmake.define.INFINI_RT_SOURCE_DIR=$infinirt_source_root" \
      "--config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=$cuda_architectures" \
      2>&1 | tee -a "$launcher_log" || build_exit=$?
  if [[ "$build_exit" -ne 0 ]]; then
    echo "build_exit=$build_exit" | tee -a "$launcher_log"
    exit "$build_exit"
  fi
  echo "=== build end=$(date --iso-8601=seconds) ===" >>"$launcher_log"
fi

# Compilation can take long enough for the initially selected device to become
# occupied. Refresh automatic selection immediately before benchmark preflight.
# An explicit GPU request is never changed automatically.
if [[ "$gpu_selection" == "auto" ]]; then
  previous_gpu="$physical_gpu"
  if ! select_idle_gpu; then
    exit 2
  fi
  {
    echo "gpu_reselection=auto"
    echo "previous_gpu=$previous_gpu"
    echo "physical_gpu=$physical_gpu"
    echo "gpu_candidates=$gpu_candidates"
    echo "gpu_busy=$gpu_busy"
    echo "gpu_rejected=$gpu_rejected"
  } >>"$launcher_log"
fi

metadata="${INFINIOPS_METADATA:-$source_root/generated/torch_ops_metadata.json}"
docker exec "$container" test -f "$metadata"
sha="$(docker exec "$container" git -c "safe.directory=$source_root" -C "$source_root" rev-parse HEAD)"
infinirt_sha="$(docker exec "$container" git -c "safe.directory=$infinirt_source_root" -C "$infinirt_source_root" rev-parse HEAD)"

if ! gpu_status="$(nvidia-smi -i "$physical_gpu" --query-gpu=index,uuid,utilization.gpu,memory.total,memory.used --format=csv,noheader,nounits 2>&1)"; then
  printf '%s\n' "$gpu_status" >&2
  echo "nvidia-smi failed for physical GPU $physical_gpu" >&2
  exit 2
fi
read -r status_index gpu_uuid utilization total_memory_mib used_memory_mib <<<"$(awk -F',' '{print $1, $2, $3, $4, $5}' <<<"$gpu_status")"
if [[ -z "$gpu_uuid" ]]; then
  echo "physical NVIDIA GPU $physical_gpu was not found" >&2
  exit 2
fi
if ! all_processes="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>&1)"; then
  printf '%s\n' "$all_processes" >&2
  echo "nvidia-smi process query failed" >&2
  exit 2
fi
process_snapshot="$(awk -F ', ' -v uuid="$gpu_uuid" '$1 == uuid {print}' <<<"$all_processes")"
if [[ -n "$process_snapshot" && "$gpu_allow_occupied" != "1" ]]; then
  printf '%s\n' "$process_snapshot" >&2
  echo "physical NVIDIA GPU $physical_gpu has a running process" >&2
  exit 2
fi
if [[ -z "$utilization" || "$utilization" =~ [^0-9] || "$utilization" -gt "$gpu_max_utilization" ]]; then
  echo "physical NVIDIA GPU $physical_gpu utilization exceeds ${gpu_max_utilization}%: $utilization" >&2
  exit 2
fi
if [[ -z "$total_memory_mib" || "$total_memory_mib" =~ [^0-9] || -z "$used_memory_mib" || "$used_memory_mib" =~ [^0-9] ]] || (( total_memory_mib <= 0 )); then
  echo "physical NVIDIA GPU $physical_gpu has invalid memory data: total=${total_memory_mib:-} MiB used=${used_memory_mib:-} MiB" >&2
  exit 2
fi
free_memory="$(awk -v total="$total_memory_mib" -v used="$used_memory_mib" 'BEGIN { printf "%.2f", (total-used)*100/total }')"
if [[ -z "$free_memory" || "$free_memory" =~ [^0-9.] ]] || python3 -c 'import sys; raise SystemExit(float(sys.argv[1]) >= float(sys.argv[2]))' "$free_memory" "$gpu_min_free_memory"; then
  echo "physical NVIDIA GPU $physical_gpu free memory is below ${gpu_min_free_memory}%: $free_memory%" >&2
  exit 2
fi

{
  echo "start=$(date --iso-8601=seconds)"
  echo "container=$container"
  echo "container_id=$container_id"
  echo "physical_gpu=$physical_gpu"
  echo "gpu_uuid=$gpu_uuid"
  echo "gpu_max_utilization_percent=$gpu_max_utilization"
  echo "gpu_min_free_memory_percent=$gpu_min_free_memory"
  echo "gpu_allow_occupied=$gpu_allow_occupied"
  echo "gpu_free_memory_percent=$free_memory"
  echo "execution_time=$execution_time"
  echo "execution_date=$execution_date"
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
  echo "dtypes=$dtypes_input"
  echo "profiles=$profiles_input"
  echo "decode_4k_ops=$decode_4k_ops"
  echo "decode_4k_small_ops=$decode_4k_small_ops"
  echo "decode_8k_ops=$decode_8k_ops"
  echo "prefill_4k_ops=$prefill_4k_ops"
  echo "=== host preflight ==="
  nvidia-smi -i "$physical_gpu"
  printf '%s\n' "$process_snapshot"
  echo "=== container preflight ==="
  docker exec -e CUDA_VISIBLE_DEVICES="$physical_gpu" "$container" python3 -c \
    "import torch, infini.ops; assert torch.cuda.is_available(); print('torch', torch.__version__); print('cuda_count', torch.cuda.device_count()); print('device', torch.cuda.get_device_name(0))"
} >>"$launcher_log" 2>&1

monitor() {
  local snapshot pids pid cgroup
  while [[ ! -s "$done_file" ]]; do
    echo "sample=$(date --iso-8601=seconds)"
    if ! snapshot="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits 2>&1)"; then
      echo "monitor_error=$snapshot"
      echo "nvidia-smi-error" >>"$foreign_flag"
      docker exec "$container" pkill -f "$harness" 2>/dev/null || true
      break
    fi
    printf '%s\n' "$snapshot"
    snapshot="$(awk -F ', ' -v uuid="$gpu_uuid" '$1 == uuid {print}' <<<"$snapshot")"
    if [[ -z "$snapshot" ]]; then
      sleep 2
      continue
    fi
    pids="$(awk -F ', ' '{print $2}' <<<"$snapshot")"
    for pid in $pids; do
      if [[ ! -e "/proc/$pid" ]]; then
        echo "process_owner=exited_before_attribution pid=$pid"
        continue
      fi
      cgroup="$(tr '\n' ' ' <"/proc/$pid/cgroup" 2>/dev/null || true)"
      if [[ "$cgroup" == *"$container_id"* ]]; then
        echo "process_owner=benchmark_container pid=$pid cmdline=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
      else
        if [[ "$gpu_allow_occupied" == "1" ]]; then
          echo "process_owner=foreign_allowed pid=$pid cgroup=$cgroup cmdline=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
        else
          echo "process_owner=foreign pid=$pid cgroup=$cgroup cmdline=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
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
    --device cuda
    --dtype "$dtype"
    --platform "$device_id"
    --sha "$sha"
    --min-run-time "$min_run_time"
    --benchmark-iterations "$benchmark_iterations"
    --timeout 20
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
    -e CUDA_VISIBLE_DEVICES="$physical_gpu" \
    -e INFINIOPS_BENCHMARK_PROFILE="$profile" \
    "$container" "${command[@]}" 2>&1 | tee -a "$launcher_log"
  return "${PIPESTATUS[0]}"
}

monitor >>"$monitor_log" 2>&1 &
monitor_pid=$!

exit_code=0
exporter_sources=()
for dtype in "${dtypes[@]}"; do
  if [[ "$exit_code" -ne 0 || -s "$foreign_flag" ]]; then
    break
  fi
  echo "=== dtype=$dtype start=$(date --iso-8601=seconds) ===" >>"$launcher_log"
  for profile in "${profiles[@]}"; do
    case "$profile" in
      decode-4k) selected_ops="$decode_4k_ops" ;;
      decode-4k-small) selected_ops="$decode_4k_small_ops" ;;
      decode-8k) selected_ops="$decode_8k_ops" ;;
      prefill-4k) selected_ops="$prefill_4k_ops" ;;
    esac
    if ! run_profile "$dtype" "$profile" "$selected_ops"; then
      exit_code=$?
      break
    fi
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

if [[ "$exit_code" -ne 0 || -s "$foreign_flag" ]]; then
  echo "benchmark failed or saw a foreign process; skipping export" | tee -a "$launcher_log"
elif [[ "${#exporter_sources[@]}" -gt 0 ]]; then
  docker exec "$container" python3 "$exporter" \
    "${exporter_sources[@]}" \
    --output "$output_container/infiniops-nvidia-a100.csv" \
    --diagnostics "$output_container/diagnostics.csv" \
    --report "$output_container/report.md" \
    --platform nvidia \
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
    echo "=== partial export start=$(date --iso-8601=seconds) ===" | tee -a "$launcher_log"
    partial_exit=0
    docker exec "$container" python3 "$partial_exporter" \
      --output-root "$output_container" \
      --output-name "infiniops-nvidia-a100.csv" \
      --platform nvidia \
      --date "$csv_date" \
      --tester "$tester" \
      --infiniops-commit "$sha" \
      --infinirt-commit "$infinirt_sha" \
      --failure-category "$partial_failure_category" \
      --failure-reason "$partial_failure_reason" \
      2>&1 | tee -a "$launcher_log" || partial_exit=$?
    if [[ "$partial_exit" -eq 0 ]]; then
      echo "partial_csv=$output_container/infiniops-nvidia-a100.partial.csv" | tee -a "$launcher_log"
    else
      echo "partial_export_failed=$partial_exit" | tee -a "$launcher_log"
    fi
  else
    echo "partial_export_skipped=no complete or checkpoint result files" | tee -a "$launcher_log"
  fi
fi

{
  echo "=== post-run nvidia-smi ==="
  nvidia-smi -i "$physical_gpu"
  echo "end=$(date --iso-8601=seconds)"
} >>"$launcher_log" 2>&1

if [[ -s "$foreign_flag" ]]; then
  foreign_process_seen=1
else
  foreign_process_seen=0
fi
{
  echo "exit_code=$exit_code"
  echo "foreign_process_seen=$foreign_process_seen"
  echo "end=$(date --iso-8601=seconds)"
  echo "container=$container"
  echo "physical_gpu=$physical_gpu"
  echo "gpu_uuid=$gpu_uuid"
  echo "gpu_max_utilization_percent=$gpu_max_utilization"
  echo "gpu_min_free_memory_percent=$gpu_min_free_memory"
  echo "gpu_allow_occupied=$gpu_allow_occupied"
  echo "execution_time=$execution_time"
  echo "execution_date=$execution_date"
  echo "run_date=$run_date"
  echo "csv_date=$csv_date"
  echo "source_update=$update_sources"
  echo "source_backfill_date=$backfill_date"
  echo "infiniops_requested_ref=$infiniops_ref"
  echo "infinirt_requested_ref=$infinirt_ref"
  echo "sha=$sha"
  echo "infinirt_sha=$infinirt_sha"
} >"$done_file"

wait "$monitor_pid"
if [[ "$foreign_process_seen" != 0 ]]; then
  exit 3
fi
exit "$exit_code"

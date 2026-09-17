#!/usr/bin/env bash
set -euo pipefail

container="${CONTAINER:-infinibench-stack-20260724-iluvatar}"
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

execution_time="${EXECUTION_TIME:-$(date +%Y%m%d-%H%M%S)}"
execution_date="${execution_time:0:8}"
tester="${TESTER:-baoming}"
build_infiniops="${BUILD_INFINIOPS:-1}"
update_sources="${UPDATE_SOURCES:-1}"
backfill_date="${BACKFILL_DATE:-}"
reuse_run_dir="${REUSE_RUN_DIR:-0}"
infiniops_ref="${INFINIOPS_REF:-}"
infinirt_ref="${INFINIRT_REF:-}"
device_id="${DEVICE_ID:-iluvatar-tg150}"
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
profiles_input="${PROFILES:-decode-4k,decode-8k,prefill-4k}"
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
source_root="${INFINIOPS_SOURCE_ROOT:-/workspace/src/infinibench-git/InfiniOps}"
infinirt_source_root="${INFINIRT_SOURCE_ROOT:-/workspace/src/infinibench-git/InfiniRT}"
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
build_workspace="${INFINIBENCH_BUILD_WORKSPACE:-/workspace/.cache/infinibench-iluvatar}"
iluvatar_arch="${ILUVATAR_ARCH:-ivcore20}"
cmake_build_parallel_level="${CMAKE_BUILD_PARALLEL_LEVEL:-16}"
infiniops_build_jobs="${INFINIOPS_BUILD_JOBS:-16}"
infiniops_torch_jobs="${INFINIOPS_TORCH_COMPILE_JOBS:-2}"
infiniops_binding_jobs="${INFINIOPS_BINDING_COMPILE_JOBS:-2}"
patch_dir="${ILUVATAR_PATCH_DIR:-$container_tools_root/patches/iluvatar-tg150}"
output_container_root="${CONTAINER_OUTPUT_ROOT:-$output_root}"
output_container="$output_container_root/$device_id/$run_date"
runner_lock="$output_root/$device_id/.infinibench-runner.lock"
runner_lock_container="$output_container_root/$device_id/.infinibench-runner.lock"
harness="$container_tools_root/scripts/full_operator_benchmark.py"
exporter="$container_tools_root/scripts/export_full_operator_report.py"
launcher_log="$output_host/benchmark-launcher.log"
monitor_log="$output_host/benchmark-monitor.log"
foreign_flag="$output_host/foreign-process.flag"
done_file="$output_host/benchmark.done"

target_ops=abs,add,addmm,amax,amin,argmax,bitwise_and,bitwise_not,bitwise_or,bmm,cat,ceil,clamp,cos,cumsum,div,eq,exp,floor,gather,ge,gelu,gt,hardtanh,index_select,le,log,log_softmax,lt,matmul,maximum,mean,minimum,mm,mul,ne,neg,pow,reciprocal,relu,round,rsqrt,sigmoid,sign,silu,sin,softmax,sort,sqrt,sub,sum,tanh,topk
small_ops=abs,add,avg_pool2d,bmm,gather,fft_fft,gelu,matmul,mm,mul,relu,sigmoid,silu,softmax,special_gammaln,sparse_sampled_addmm,sum,where
decode_4k_ops="${ILUVATAR_DECODE_4K_OPS:-all}"
decode_4k_small_ops="${ILUVATAR_DECODE_4K_SMALL_OPS:-$small_ops}"
decode_8k_ops="${ILUVATAR_DECODE_8K_OPS:-$target_ops}"
prefill_4k_ops="${ILUVATAR_PREFILL_4K_OPS:-$target_ops}"

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
  echo "another Iluvatar benchmark runner is active: $runner_lock" >&2
  exit 2
fi

# A date-only run directory is deterministic: one normal execution date, or one
# backfill target date plus one execution date. Refuse to mix a new attempt
# with stale checkpoint files. REUSE_RUN_DIR=1 explicitly replaces the fixed
# directory for a retry.
if [[ -e "$output_host" ]] && [[ "$reuse_run_dir" != "1" ]]; then
  if [[ -n "$(find "$output_host" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
    echo "output directory already exists and is non-empty: $output_host" >&2
    echo "set REUSE_RUN_DIR=1 to replace it for the same normal execution date or backfill/execution date pair" >&2
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

docker exec "$container" test -f "$harness"
docker exec "$container" test -f "$exporter"

write_benchmark_done() {
  local result_code="$1"
  {
    echo "exit_code=$result_code"
    echo "foreign_process_seen=${foreign_process_seen:-0}"
    echo "end=$(date --iso-8601=seconds)"
    echo "container=$container"
    echo "container_id=$container_id"
    echo "physical_gpu=${physical_gpu:-}"
    echo "gpu_uuid=${gpu_uuid:-}"
    echo "gpu_selection=${gpu_selection:-}"
    echo "gpu_max_utilization_percent=$gpu_max_utilization"
    echo "gpu_min_free_memory_percent=$gpu_min_free_memory"
    echo "gpu_allow_occupied=$gpu_allow_occupied"
    echo "gpu_free_memory_percent=${free_memory_percent:-}"
    echo "execution_time=$execution_time"
    echo "run_date=$run_date"
    echo "csv_date=$csv_date"
    echo "backfill_date=${backfill_date:-}"
    echo "source_update=$update_sources"
    echo "infiniops_requested_ref=$infiniops_ref"
    echo "infinirt_requested_ref=$infinirt_ref"
    echo "sha=${sha:-}"
    echo "infinirt_sha=${infinirt_sha:-}"
    echo "output_dir=$output_container"
  } >"$done_file"
}

on_runner_exit() {
  local result_code=$?
  if [[ ! -s "$done_file" ]]; then
    write_benchmark_done "$result_code"
  fi
}
trap on_runner_exit EXIT

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

select_idle_gpu() {
  local gpu_table process_table selection_output
  if ! gpu_table="$(docker exec "$container" /usr/local/corex-host/bin/ixsmi \
    --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,memory.free \
    --format=csv,noheader,nounits 2>&1)"; then
    echo "ixsmi GPU query failed:" >&2
    printf '%s\n' "$gpu_table" >&2
    return 2
  fi
  if ! process_table="$(docker exec "$container" /usr/local/corex-host/bin/ixsmi \
    --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits 2>&1)"; then
    echo "ixsmi process query failed:" >&2
    printf '%s\n' "$process_table" >&2
    return 2
  fi

  selection_output="$(
    {
      printf '%s\n' "$gpu_table"
      printf '\n__IXSMI_PROCESS_TABLE__\n'
      printf '%s\n' "$process_table"
    } | python3 -c '
import csv
import sys

raw = sys.stdin.read()
marker = "__IXSMI_PROCESS_TABLE__"
if marker not in raw:
    raise SystemExit("ixsmi output did not contain the process-table marker")
gpu_text, process_text = raw.split(marker, 1)
devices = []
for row in csv.reader(gpu_text.splitlines()):
    if len(row) != 6:
        continue
    index, uuid, utilization, used_memory, total_memory, free_memory = [value.strip() for value in row]
    try:
        devices.append(
            (
                int(index),
                uuid,
                int(utilization),
                int(used_memory),
                int(total_memory),
                int(free_memory),
            )
        )
    except ValueError:
        raise SystemExit(f"invalid ixsmi GPU row: {row}")
if not devices:
    raise SystemExit("ixsmi output did not contain any Iluvatar devices")

busy_uuid = set()
for row in csv.reader(process_text.splitlines()):
    if row:
        busy_uuid.add(row[0].strip())
uuid_to_index = {uuid: index for index, uuid, *_ in devices}
busy = sorted(uuid_to_index[uuid] for uuid in busy_uuid if uuid in uuid_to_index)

max_utilization = int(sys.argv[1])
min_free_memory = float(sys.argv[2])
allow_occupied = sys.argv[3] == "1"

eligible = []
rejected = []
for index, uuid, utilization, used_memory, total_memory, free_memory in sorted(devices):
    reasons = []
    if index in busy and not allow_occupied:
        reasons.append("busy")
    if utilization > max_utilization:
        reasons.append(f"utilization={utilization}")
    if total_memory <= 0:
        reasons.append("invalid_memory")
    else:
        free_memory_percent = round(free_memory * 100.0 / total_memory, 2)
        if free_memory_percent < min_free_memory:
            reasons.append(f"free_memory={free_memory_percent:.2f}%")
    if reasons:
        rejected.append(str(index) + "(" + ",".join(reasons) + ")")
    else:
        eligible.append(index)

if not eligible:
    raise SystemExit(
        "no idle Iluvatar GPU found; rejected: "
        + (", ".join(rejected) if rejected else "unknown")
    )

selected = eligible[0]
print(selected)
print(",".join(str(item) for item in eligible))
print(",".join(str(item) for item in sorted(busy)) or "-")
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
  if [[ -z "$build_workspace" || "$build_workspace" == / || "$build_workspace" == /workspace ]]; then
    echo "refusing unsafe INFINIBENCH_BUILD_WORKSPACE: $build_workspace" >&2
    exit 2
  fi
  docker exec "$container" rm -rf -- "$build_workspace"
  docker exec "$container" mkdir -p "$build_workspace/InfiniOps-source" "$build_workspace/InfiniRT-build"

  {
    echo "=== InfiniRT build start=$(date --iso-8601=seconds) ==="
    echo "source=$infinirt_source_root"
    echo "build_dir=$build_workspace/InfiniRT-build"
    echo "install=$build_workspace/InfiniRT-install"
  } >>"$launcher_log"
  build_exit=0
  docker exec \
    -e CUDA_HOME=/usr/local/corex \
    -e CUDA_PATH=/usr/local/corex \
    -e CUDAToolkit_ROOT=/usr/local/corex \
    -e CPATH=/usr/local/corex/include:/usr/local/include/python3.10 \
    -e CPLUS_INCLUDE_PATH=/usr/local/corex/include:/usr/local/include/python3.10 \
    -e LD_LIBRARY_PATH=/usr/local/corex-host/lib64:/usr/local/corex/lib64:/usr/local/openmpi/lib:/usr/local/lib \
    "$container" cmake \
      -S "$infinirt_source_root" \
      -B "$build_workspace/InfiniRT-build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$build_workspace/InfiniRT-install" \
      -DCMAKE_INSTALL_LIBDIR=lib \
      -DCMAKE_CUDA_ARCHITECTURES="$iluvatar_arch" \
      -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
      -DWITH_CPU=ON \
      -DWITH_ILUVATAR=ON 2>&1 | tee -a "$launcher_log" || build_exit=$?
  if [[ "$build_exit" -eq 0 ]]; then
    docker exec \
      -e CMAKE_BUILD_PARALLEL_LEVEL="$cmake_build_parallel_level" \
      "$container" cmake --build "$build_workspace/InfiniRT-build" 2>&1 | tee -a "$launcher_log" || build_exit=$?
  fi
  if [[ "$build_exit" -eq 0 ]]; then
    docker exec "$container" cmake --install "$build_workspace/InfiniRT-build" 2>&1 | tee -a "$launcher_log" || build_exit=$?
  fi
  if [[ "$build_exit" -ne 0 ]]; then
    echo "infinirt_build_exit=$build_exit" | tee -a "$launcher_log"
    exit "$build_exit"
  fi
  echo "=== InfiniRT build end=$(date --iso-8601=seconds) ===" >>"$launcher_log"

  docker exec "$container" bash -c \
    'git -C "$1" archive HEAD | tar -x -C "$2"' \
    sh "$source_root" "$build_workspace/InfiniOps-source"
  for patch_name in infiniops-iluvatar-corex45-driver-target.patch infiniops-iluvatar-cvta.patch; do
    patch_file="$patch_dir/$patch_name"
    docker exec "$container" test -f "$patch_file"
    echo "apply_patch=$patch_file" | tee -a "$launcher_log"
    if docker exec "$container" bash -c \
      'patch -d "$1" -p1 --dry-run --forward < "$2"' \
      sh "$build_workspace/InfiniOps-source" "$patch_file" >/dev/null 2>&1; then
      docker exec "$container" bash -c \
        'patch -d "$1" -p1 --forward < "$2"' \
        sh "$build_workspace/InfiniOps-source" "$patch_file" 2>&1 | tee -a "$launcher_log"
    elif docker exec "$container" bash -c \
      'patch -d "$1" -p1 --dry-run -R < "$2"' \
      sh "$build_workspace/InfiniOps-source" "$patch_file" >/dev/null 2>&1; then
      echo "patch_already_applied=$patch_name" | tee -a "$launcher_log"
    else
      echo "compatibility patch does not apply: $patch_file" | tee -a "$launcher_log" >&2
      exit 2
    fi
  done

  {
    echo "=== InfiniOps build start=$(date --iso-8601=seconds) ==="
    echo "source=$build_workspace/InfiniOps-source"
    echo "cmake_build_parallel_level=$cmake_build_parallel_level"
    echo "infinirt_root=$build_workspace/InfiniRT-install"
  } >>"$launcher_log"
  build_exit=0
  docker exec \
    -e CMAKE_BUILD_PARALLEL_LEVEL="$cmake_build_parallel_level" \
    -e CUDA_HOME=/usr/local/corex \
    -e CUDA_PATH=/usr/local/corex \
    -e CUDAToolkit_ROOT=/usr/local/corex \
    -e INFINI_RT_ROOT="$build_workspace/InfiniRT-install" \
    -e CPATH=/usr/local/corex/include:/usr/local/include/python3.10 \
    -e CPLUS_INCLUDE_PATH=/usr/local/corex/include:/usr/local/include/python3.10 \
    -e LD_LIBRARY_PATH="$build_workspace/InfiniRT-install/lib:/usr/local/corex-host/lib64:/usr/local/corex/lib64:/usr/local/corex/lib64/python3/dist-packages/torch/lib:/usr/local/openmpi/lib:/usr/local/lib" \
    "$container" bash -c \
      'cd "$1" && scripts/dev/build.sh iluvatar --jobs "$2" --torch-jobs "$3" --binding-jobs "$4"' \
      sh "$build_workspace/InfiniOps-source" \
      "$infiniops_build_jobs" \
      "$infiniops_torch_jobs" \
      "$infiniops_binding_jobs" 2>&1 | tee -a "$launcher_log" || build_exit=$?
  if [[ "$build_exit" -ne 0 ]]; then
    echo "infiniops_build_exit=$build_exit" | tee -a "$launcher_log"
    exit "$build_exit"
  fi
  echo "=== InfiniOps build end=$(date --iso-8601=seconds) ===" >>"$launcher_log"
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

infini_root="${INFINIOPS_INSTALL_ROOT:-$build_workspace/InfiniOps-source/build-iluvatar/install}"
infini_lib="$infini_root/infini"
python_path="$infini_root:/usr/local/corex-4.5.0.20260615/lib64/python3/dist-packages"
ld_library_path="$infini_lib:$build_workspace/InfiniRT-install/lib:/opt/infini/lib:/opt/infini-rt/lib:/usr/local/corex-host/lib64:/usr/local/corex/lib64:/usr/local/corex/lib64/python3/dist-packages/torch/lib:/usr/local/openmpi/lib:/usr/local/lib"
metadata="${INFINIOPS_METADATA:-$infini_root/infini/torch_ops_metadata.json}"
docker exec "$container" test -f "$metadata"
sha="$(docker exec "$container" git -c "safe.directory=$source_root" -C "$source_root" rev-parse HEAD)"
infinirt_sha="$(docker exec "$container" git -c "safe.directory=$infinirt_source_root" -C "$infinirt_source_root" rev-parse HEAD)"

gpu_uuid="$(
  docker exec "$container" /usr/local/corex-host/bin/ixsmi --query-gpu=index,uuid \
    --format=csv,noheader,nounits |
  awk -F',' -v id="$physical_gpu" '
    {gsub(/ /, "", $1); if ($1 == id) {gsub(/ /, "", $2); selected=$2}}
    END {if (selected != "") print selected}
  '
)"
if [[ -z "$gpu_uuid" ]]; then
  echo "physical Iluvatar GPU $physical_gpu was not found" >&2
  exit 2
fi

for sample in 1 2 3; do
  status="$(
    docker exec "$container" /usr/local/corex-host/bin/ixsmi \
      --query-gpu=index,utilization.gpu,memory.used,memory.total,memory.free \
      --format=csv,noheader,nounits |
    awk -F',' -v id="$physical_gpu" '
      {gsub(/ /, "", $1); if ($1 == id) {for (i=2; i<=5; i++) gsub(/ /, "", $i); selected=$2 "," $3 "," $4 "," $5}}
      END {if (selected != "") print selected}
    '
  )"
  utilization="$(cut -d',' -f1 <<<"$status")"
  used_memory_mib="$(cut -d',' -f2 <<<"$status")"
  total_memory_mib="$(cut -d',' -f3 <<<"$status")"
  free_memory_mib="$(cut -d',' -f4 <<<"$status")"
  free_memory_percent="$(
    awk -v free="$free_memory_mib" -v total="$total_memory_mib" \
      'BEGIN { if (total <= 0) exit 1; printf "%.2f", free * 100.0 / total }'
  )" || {
    echo "physical Iluvatar GPU $physical_gpu has invalid memory information" >&2
    exit 2
  }
  processes="$(
    docker exec "$container" /usr/local/corex-host/bin/ixsmi --query-compute-apps=gpu_uuid,pid \
      --format=csv,noheader,nounits |
    awk -F',' -v uuid="$gpu_uuid" '{gsub(/ /, "", $1); if ($1 == uuid) print $0}' |
    wc -l
  )"
  {
    echo "precheck_sample=$sample util=$utilization used_memory_mib=$used_memory_mib total_memory_mib=$total_memory_mib free_memory_mib=$free_memory_mib free_memory_percent=$free_memory_percent processes=$processes"
  } >>"$launcher_log"
  if [[ -z "$utilization" || "$utilization" =~ [^0-9] || "$utilization" -gt "$gpu_max_utilization" ]] ||
     [[ -z "$free_memory_percent" || "$free_memory_percent" =~ [^0-9.] ]] ||
     python3 -c 'import sys; raise SystemExit(not (float(sys.argv[1]) < float(sys.argv[2])))' \
       "$free_memory_percent" "$gpu_min_free_memory" ||
     [[ "$processes" -gt 0 && "$gpu_allow_occupied" != "1" ]]; then
    echo "physical Iluvatar GPU $physical_gpu does not satisfy GPU gate (util=$utilization%, free_memory=$free_memory_percent%, processes=$processes)" >&2
    exit 2
  fi
  sleep 2
done

{
  echo "start=$(date --iso-8601=seconds)"
  echo "container=$container"
  echo "container_id=$container_id"
  echo "physical_gpu=$physical_gpu"
  echo "gpu_max_utilization_percent=$gpu_max_utilization"
  echo "gpu_min_free_memory_percent=$gpu_min_free_memory"
  echo "gpu_allow_occupied=$gpu_allow_occupied"
  echo "gpu_free_memory_percent=$free_memory_percent"
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
  echo "dtypes=$dtypes_input"
  echo "profiles=$profiles_input"
  echo "decode_4k_ops=$decode_4k_ops"
  echo "decode_4k_small_ops=$decode_4k_small_ops"
  echo "decode_8k_ops=$decode_8k_ops"
  echo "prefill_4k_ops=$prefill_4k_ops"
  echo "gpu_uuid=$gpu_uuid"
  echo "=== host preflight ==="
  docker exec "$container" /usr/local/corex-host/bin/ixsmi -i "$physical_gpu"
  echo "=== container preflight ==="
  docker exec \
    -e CUDA_VISIBLE_DEVICES="$physical_gpu" \
    -e PYTHONPATH="$python_path" \
    -e LD_LIBRARY_PATH="$ld_library_path" \
    "$container" python3 -c \
    "import torch, infini.ops; assert torch.cuda.is_available(); print('torch', torch.__version__); print('cuda_count', torch.cuda.device_count()); print('device', torch.cuda.get_device_name(0))"
} >>"$launcher_log" 2>&1

monitor() {
  local snapshot rows pid cgroup cmdline
  while [[ ! -s "$done_file" ]]; do
    echo "sample=$(date --iso-8601=seconds)"
    if ! snapshot="$(docker exec "$container" /usr/local/corex-host/bin/ixsmi \
      --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory \
      --format=csv,noheader,nounits 2>&1)"; then
      echo "monitor_error=$snapshot"
      echo "ixsmi-error" >>"$foreign_flag"
      docker exec "$container" pkill -f "$harness" 2>/dev/null || true
      break
    fi
    rows="$(
      printf '%s\n' "$snapshot" |
      awk -F',' -v uuid="$gpu_uuid" '{gsub(/ /, "", $1); if ($1 == uuid) print $0}'
    )"
    if [[ -z "$rows" ]]; then
      echo "gpu_rows=none"
    else
      while IFS= read -r row; do
        echo "gpu_rows=$row"
        pid="$(echo "$row" | cut -d',' -f2 | tr -d ' ')"
        if [[ ! -e "/proc/$pid" ]]; then
          echo "process_owner=exited_before_attribution pid=$pid"
          continue
        fi
        cgroup="$(tr '\n' ' ' <"/proc/$pid/cgroup" 2>/dev/null || true)"
        cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
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
      done <<<"$rows"
    fi
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
    --sha "$sha"
    --min-run-time 0.02
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
    -e PYTHONPATH="$python_path" \
    -e LD_LIBRARY_PATH="$ld_library_path" \
    "$container" "${command[@]}" 2>&1 | tee -a "$launcher_log"
  return "${PIPESTATUS[0]}"
}

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

if [[ "${#exporter_sources[@]}" -gt 0 ]]; then
  docker exec "$container" python3 "$exporter" \
    "${exporter_sources[@]}" \
    --output "$output_container/infiniops-iluvatar-biv150.csv" \
    --diagnostics "$output_container/diagnostics.csv" \
    --report "$output_container/report.md" \
    --platform iluvatar \
    --date "$csv_date" \
    --tester "$tester" 2>&1 | tee -a "$launcher_log"
  exporter_exit=${PIPESTATUS[0]}
  if [[ "$exporter_exit" -ne 0 ]]; then
    exit_code="$exporter_exit"
  fi
else
  echo "no result JSON files were available for export" | tee -a "$launcher_log"
  exit_code=1
fi

{
  echo "=== post-run ixsmi ==="
  docker exec "$container" /usr/local/corex-host/bin/ixsmi -i "$physical_gpu"
  echo "end=$(date --iso-8601=seconds)"
} >>"$launcher_log" 2>&1

if [[ -s "$foreign_flag" ]]; then
  foreign_process_seen=1
else
  foreign_process_seen=0
fi
write_benchmark_done "$exit_code"

wait "$monitor_pid"
if [[ "$foreign_process_seen" != 0 ]]; then
  exit 3
fi
exit "$exit_code"

#!/usr/bin/env bash
set -uo pipefail

container="${CONTAINER:-baoming_infiniops_hygon}"
physical_gpu="${1:-${PHYSICAL_GPU:-}}"
if [[ -z "$physical_gpu" ]]; then
  echo "usage: [BUILD_INFINIOPS=1] $0 <physical-hcu-index>" >&2
  exit 2
fi

run_date="${RUN_DATE:-$(date +%Y-%m-%d)}"
tester="${TESTER:-baoming}"
build_infiniops="${BUILD_INFINIOPS:-0}"
build_infinirt="${BUILD_INFINIRT:-0}"
hygon_arch="${HYGON_ARCH:-gfx906}"
worker_timeout="${WORKER_TIMEOUT:-60}"
worker_batch_size="${WORKER_BATCH_SIZE:-8}"
worker_batch_timeout="${WORKER_BATCH_TIMEOUT:-120}"

host_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
container_root="${CONTAINER_BENCHMARK_ROOT:-/workspace/InfiniOps_benchmark}"
runtime_ld_library_path="/opt/dtk/cuda/cuda/lib64:/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/dtk/rccl/lib:/opt/ucx/lib:/opt/mpi/lib:/opt/hwloc/lib"
source_root="${INFINIOPS_SOURCE_ROOT:-$container_root/InfiniOps}"
infinirt_source_root="${INFINIRT_SOURCE_ROOT:-$container_root/InfiniRT}"
infinirt_install_root="${INFINI_RT_ROOT:-$container_root/infini-rt-install}"
output_host="${OUTPUT_ROOT:-$host_root/results}/$run_date"
output_container="$container_root/results/$run_date"
harness="$container_root/scripts/full_operator_benchmark.py"
exporter="$container_root/scripts/export_full_operator_report.py"
audit="$host_root/scripts/hygon_gpu_audit.py"
launcher_log="$output_host/benchmark-launcher.log"
monitor_log="$output_host/benchmark-monitor.log"
foreign_flag="$output_host/foreign-process.flag"
done_file="$output_host/benchmark.done"
csv_date="${run_date//-//}"

target_ops=abs,add,addmm,amax,amin,argmax,bitwise_and,bitwise_not,bitwise_or,bmm,cat,ceil,clamp,cos,cumsum,div,eq,exp,floor,gather,ge,gelu,gt,hardtanh,index_select,le,log,log_softmax,lt,matmul,maximum,mean,minimum,mm,mul,ne,neg,pow,reciprocal,relu,round,rsqrt,sigmoid,sign,silu,sin,softmax,sort,sqrt,sub,sum,tanh,topk

mkdir -p "$output_host"
: >"$launcher_log"
: >"$monitor_log"
: >"$foreign_flag"
: >"$done_file"

if ! docker inspect "$container" >/dev/null 2>&1; then
  echo "container not found: $container" >&2
  exit 2
fi
if [[ "$(docker inspect -f '{{.State.Running}}' "$container")" != "true" ]]; then
  echo "container is not running: $container" >&2
  exit 2
fi

container_id="$(docker inspect -f '{{.Id}}' "$container")"
if ! python3 "$audit" --gpu "$physical_gpu" --assert-free; then
  echo "selected HCU $physical_gpu is already occupied" >&2
  exit 2
fi

if [[ "$build_infinirt" == "1" ]]; then
  docker exec -e HYGON_ARCH="$hygon_arch" -e LD_LIBRARY_PATH="$runtime_ld_library_path" "$container" cmake \
    -S "$infinirt_source_root" \
    -B "$container_root/build/infinirt-hygon" \
    -DCMAKE_INSTALL_PREFIX="$infinirt_install_root" \
    -DWITH_CPU=ON -DWITH_HYGON=ON -DINFINI_RT_BUILD_TESTING=OFF \
    2>&1 | tee -a "$launcher_log"
  build_status="${PIPESTATUS[0]}"
  if [[ "$build_status" -ne 0 ]]; then
    echo "exit_code=$build_status" >"$done_file"
    exit "$build_status"
  fi
  docker exec -e HYGON_ARCH="$hygon_arch" -e LD_LIBRARY_PATH="$runtime_ld_library_path" "$container" cmake \
    --build "$container_root/build/infinirt-hygon" --parallel 8 \
    2>&1 | tee -a "$launcher_log"
  build_status="${PIPESTATUS[0]}"
  if [[ "$build_status" -ne 0 ]]; then
    echo "exit_code=$build_status" >"$done_file"
    exit "$build_status"
  fi
  docker exec -e HYGON_ARCH="$hygon_arch" -e LD_LIBRARY_PATH="$runtime_ld_library_path" "$container" cmake \
    --install "$container_root/build/infinirt-hygon" 2>&1 | tee -a "$launcher_log"
  build_status="${PIPESTATUS[0]}"
  if [[ "$build_status" -ne 0 ]]; then
    echo "exit_code=$build_status" >"$done_file"
    exit "$build_status"
  fi
fi

if [[ "$build_infiniops" == "1" ]]; then
  if ! docker exec -e LD_LIBRARY_PATH="$runtime_ld_library_path" "$container" python3 -c "import clang" >/dev/null 2>&1; then
    docker exec -e LD_LIBRARY_PATH="$runtime_ld_library_path" "$container" python3 -m pip install libclang 2>&1 | tee -a "$launcher_log"
    build_status="${PIPESTATUS[0]}"
    if [[ "$build_status" -ne 0 ]]; then
      echo "exit_code=$build_status" >"$done_file"
      exit "$build_status"
    fi
  fi
  build_command=(
    python3 -m pip install --no-build-isolation --no-deps --force-reinstall "$source_root"
    --config-settings=cmake.define.AUTO_DETECT_DEVICES=OFF
    --config-settings=cmake.define.AUTO_DETECT_BACKENDS=OFF
    --config-settings=cmake.define.WITH_CPU=ON
    --config-settings=cmake.define.WITH_HYGON=ON
    --config-settings=cmake.define.WITH_TORCH=ON
    --config-settings=cmake.define.WITH_LINKED=OFF
    --config-settings=cmake.define.INFINI_RT_ROOT="$infinirt_install_root"
  )
  docker exec \
    -e HYGON_ARCH="$hygon_arch" \
    -e LD_LIBRARY_PATH="$runtime_ld_library_path" \
    -e SKBUILD_BUILD_DIR="$container_root/build/infiniops-hygon" \
    -e PIP_NO_CACHE_DIR=1 \
    "$container" "${build_command[@]}" 2>&1 | tee -a "$launcher_log"
  build_status="${PIPESTATUS[0]}"
  if [[ "$build_status" -ne 0 ]]; then
    echo "exit_code=$build_status" >"$done_file"
    exit "$build_status"
  fi
fi

metadata="$(docker exec -e LD_LIBRARY_PATH="$runtime_ld_library_path" "$container" python3 -c \
  "from pathlib import Path; import infini; print(Path(infini.__file__).resolve().parent / 'torch_ops_metadata.json')")"
sha="$(docker exec "$container" git -c "safe.directory=$source_root" -C "$source_root" rev-parse HEAD)"
docker exec "$container" test -f "$metadata"
docker exec "$container" test -f "$harness"
docker exec "$container" test -f "$exporter"
docker exec "$container" mkdir -p "$output_container"

{
  echo "start=$(date --iso-8601=seconds)"
  echo "container=$container"
  echo "container_id=$container_id"
  echo "physical_gpu=$physical_gpu"
  echo "run_date=$run_date"
  echo "source_root=$source_root"
  echo "infinirt_source_root=$infinirt_source_root"
  echo "infinirt_install_root=$infinirt_install_root"
  echo "sha=$sha"
  echo "metadata=$metadata"
  echo "build_infiniops=$build_infiniops"
  echo "build_infinirt=$build_infinirt"
  echo "hygon_arch=$hygon_arch"
  echo "worker_timeout=$worker_timeout"
  echo "worker_batch_size=$worker_batch_size"
  echo "worker_batch_timeout=$worker_batch_timeout"
  echo "profiles=decode-4k(full),decode-8k(targeted),prefill-4k(targeted)"
  echo "=== preflight ==="
  python3 "$audit" --gpu "$physical_gpu" || true
  docker exec \
    -e HIP_VISIBLE_DEVICES="$physical_gpu" \
    -e HYGON_ARCH="$hygon_arch" \
    -e LD_LIBRARY_PATH="$runtime_ld_library_path" \
    "$container" python3 -c \
    "import torch, infini.ops; assert torch.cuda.is_available(); print('torch', torch.__version__); print('hip', torch.version.hip); print('gpu_count', torch.cuda.device_count()); print('gpu_name', torch.cuda.get_device_name(0))"
} >>"$launcher_log" 2>&1

monitor() {
  python3 "$audit" \
    --gpu "$physical_gpu" \
    --container-id "$container_id" \
    --watch \
    --done-file "$done_file" \
    --foreign-flag "$foreign_flag" >>"$monitor_log" 2>&1
}

run_profile() {
  local profile="$1"
  local selected_ops="${2:-}"
  local output="$output_container/$profile"
  local -a command=(
    python3 "$harness"
    --metadata "$metadata"
    --output-dir "$output"
    --device cuda
    --dtype float16
    --sha "$sha"
    --min-run-time 0.02
    --timeout "$worker_timeout"
    --worker-batch-size "$worker_batch_size"
    --batch-timeout "$worker_batch_timeout"
  )
  if [[ -n "$selected_ops" ]]; then
    command+=(--ops "$selected_ops")
  fi
  {
    echo "=== profile=$profile start=$(date --iso-8601=seconds) ==="
    printf 'command='
    printf '%q ' "${command[@]}"
    echo
  } >>"$launcher_log"
  docker exec \
    -e HIP_VISIBLE_DEVICES="$physical_gpu" \
    -e HYGON_ARCH="$hygon_arch" \
    -e LD_LIBRARY_PATH="$runtime_ld_library_path" \
    -e INFINIOPS_BENCHMARK_PROFILE="$profile" \
    "$container" "${command[@]}" 2>&1 | tee -a "$launcher_log"
  return "${PIPESTATUS[0]}"
}

monitor &
monitor_pid=$!

exit_code=0
run_profile decode-4k || exit_code=$?
if [[ "$exit_code" -eq 0 ]]; then
  run_profile decode-8k "$target_ops" || exit_code=$?
fi
if [[ "$exit_code" -eq 0 ]]; then
  run_profile prefill-4k "$target_ops" || exit_code=$?
fi
if [[ -s "$foreign_flag" ]]; then
  exit_code=3
  docker exec "$container" pkill -f "$harness" 2>/dev/null || true
fi

if [[ "$exit_code" -eq 0 ]]; then
  docker exec -e LD_LIBRARY_PATH="$runtime_ld_library_path" "$container" python3 "$exporter" \
    "$output_container/decode-4k/results.json" \
    "$output_container/decode-8k/results.json" \
    "$output_container/prefill-4k/results.json" \
    --output "$output_container/infiniops-hygon-bw1000.csv" \
    --diagnostics "$output_container/diagnostics.csv" \
    --report "$output_container/report.md" \
    --platform hygon \
    --date "$csv_date" \
    --tester "$tester" 2>&1 | tee -a "$launcher_log"
  exit_code=${PIPESTATUS[0]}
fi

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
  echo "sha=$sha"
} >"$done_file"

wait "$monitor_pid" || true
exit "$exit_code"

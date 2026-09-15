#!/usr/bin/env bash
set -uo pipefail

container=libaoming_accel_test_nvidia_20260714_145051
physical_gpu="${1:?physical GPU index is required}"
sha=01f63f4beaa1ca0e178c9b728c1706a4065657cf
base=/workspace/InfiniOps_perf_logs/full-operator-2026-07-30/a100
harness_dir="$base/harness-v5"
output="$base/linalg-lu-solve-fp32-v5"
metadata=/workspace/InfiniOps_weekly_2026-07-27_01f63f4b/install-a100-retry2/infini/torch_ops_metadata.json
python_path=/workspace/InfiniOps_weekly_2026-07-27_01f63f4b/install-a100-retry2
launcher_log=/tmp/infiniops-full-a100-lu-v5-launcher.log
monitor_log=/tmp/infiniops-full-a100-lu-v5-monitor.log
foreign_flag=/tmp/infiniops-full-a100-lu-v5-foreign.flag
done_file=/tmp/infiniops-full-a100-lu-v5.done
container_id="$(docker inspect --format '{{.Id}}' "$container")"
gpu_uuid="$(nvidia-smi -i "$physical_gpu" --query-gpu=uuid --format=csv,noheader,nounits)"

: >"$launcher_log"
: >"$monitor_log"
: >"$foreign_flag"
: >"$done_file"

{
  echo "start=$(date --iso-8601=seconds)"
  echo "physical_gpu=$physical_gpu"
  echo "gpu_uuid=$gpu_uuid"
  echo "sha=$sha"
  nvidia-smi -i "$physical_gpu"
} >>"$launcher_log" 2>&1

monitor() {
  while [[ ! -s "$done_file" ]]; do
    rows="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits | awk -F ', ' -v uuid="$gpu_uuid" '$1 == uuid {print $2 ", " $3 ", " $4}')"
    if [[ -z "$rows" ]]; then
      echo "sample=$(date --iso-8601=seconds) gpu_rows=none" >>"$monitor_log"
    else
      while IFS= read -r row; do
        pid="${row%%,*}"
        cgroup="$(tr '\n' ' ' <"/proc/$pid/cgroup" 2>/dev/null || true)"
        if [[ "$cgroup" == *"$container_id"* ]]; then
          echo "sample=$(date --iso-8601=seconds) process_owner=benchmark_container row=$row" >>"$monitor_log"
        else
          echo "sample=$(date --iso-8601=seconds) process_owner=foreign row=$row cgroup=$cgroup" >>"$monitor_log"
          echo "$pid" >>"$foreign_flag"
        fi
      done <<<"$rows"
    fi
    sleep 2
  done
}

docker exec "$container" mkdir -p "$harness_dir" "$output"
docker cp /tmp/benchmark_infiniops_full_v5.py "$container:$harness_dir/benchmark_infiniops_full.py"
docker cp /tmp/full_operator_benchmark_v5.py "$container:$harness_dir/full_operator_benchmark.py"

monitor &
monitor_pid=$!
set -o pipefail
docker exec -e PYTHONPATH="$python_path" -e CUDA_VISIBLE_DEVICES="$physical_gpu" "$container" \
  python3 "$harness_dir/benchmark_infiniops_full.py" \
  --metadata "$metadata" \
  --output-dir "$output" \
  --device cuda \
  --dtype float32 \
  --sha "$sha" \
  --min-run-time 0.01 \
  --timeout 20 \
  --ops linalg_lu_solve 2>&1 | tee -a "$launcher_log"
exit_code=${PIPESTATUS[0]}

{
  nvidia-smi -i "$physical_gpu"
  echo "end=$(date --iso-8601=seconds)"
  echo "exit_code=$exit_code"
} >>"$launcher_log" 2>&1
foreign_process_seen=0
[[ -s "$foreign_flag" ]] && foreign_process_seen=1
{
  echo "exit_code=$exit_code"
  echo "foreign_process_seen=$foreign_process_seen"
  echo "end=$(date --iso-8601=seconds)"
} >"$done_file"
wait "$monitor_pid"
exit "$exit_code"

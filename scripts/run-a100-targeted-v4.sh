#!/usr/bin/env bash
set -uo pipefail

container=libaoming_accel_test_nvidia_20260714_145051
physical_gpu="${1:?physical GPU index is required}"
sha=01f63f4beaa1ca0e178c9b728c1706a4065657cf
base=/workspace/InfiniOps_perf_logs/full-operator-2026-07-30/a100
harness_dir="$base/harness-v4"
metadata=/workspace/InfiniOps_weekly_2026-07-27_01f63f4b/install-a100-retry2/infini/torch_ops_metadata.json
python_path=/workspace/InfiniOps_weekly_2026-07-27_01f63f4b/install-a100-retry2
fp16_ops=histogram,internal_add_relu,internal_batch_norm_with_update,max_unpool2d,max_unpool3d,mkldnn_adaptive_avg_pool2d,slow_conv3d,slow_conv3d_forward
fp32_ops=cholesky_solve,histc,internal_linalg_det,internal_linalg_eigh,internal_linalg_slogdet,internal_linalg_solve_ex,internal_linalg_svd,inverse,linalg_det,linalg_eig,linalg_eigh,linalg_eigvals,linalg_eigvalsh,linalg_householder_product,linalg_inv,linalg_inv_ex,linalg_ldl_factor,linalg_ldl_factor_ex,linalg_ldl_solve,linalg_lstsq,linalg_lu,linalg_lu_solve,linalg_matrix_norm,linalg_pinv,linalg_slogdet,linalg_tensorinv,linalg_tensorsolve,lu_solve,nanquantile,orgqr,ormqr,quantile,slogdet
launcher_log=/tmp/infiniops-full-a100-targeted-v4-launcher.log
monitor_log=/tmp/infiniops-full-a100-targeted-v4-monitor.log
foreign_flag=/tmp/infiniops-full-a100-targeted-v4-foreign.flag
done_file=/tmp/infiniops-full-a100-targeted-v4.done
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
  echo "container=$container"
  echo "sha=$sha"
  echo "fp16_ops=$fp16_ops"
  echo "fp32_ops=$fp32_ops"
  echo "=== pre-run nvidia-smi ==="
  nvidia-smi -i "$physical_gpu"
} >>"$launcher_log" 2>&1

monitor() {
  while [[ ! -s "$done_file" ]]; do
    {
      echo "sample=$(date --iso-8601=seconds)"
      rows="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits | awk -F ', ' -v uuid="$gpu_uuid" '$1 == uuid {print $2 ", " $3 ", " $4}')"
      if [[ -z "$rows" ]]; then
        echo "gpu_rows=none"
      else
        while IFS= read -r row; do
          echo "gpu_rows=$row"
          pid="${row%%,*}"
          if [[ ! -e "/proc/$pid" ]]; then
            echo "process_owner=exited_before_attribution pid=$pid"
            continue
          fi
          cgroup="$(tr '\n' ' ' <"/proc/$pid/cgroup" 2>/dev/null || true)"
          cmdline="$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)"
          if [[ "$cgroup" == *"$container_id"* ]]; then
            echo "process_owner=benchmark_container pid=$pid cmdline=$cmdline"
          else
            echo "process_owner=foreign pid=$pid cgroup=$cgroup cmdline=$cmdline"
            echo "$pid" >>"$foreign_flag"
          fi
        done <<<"$rows"
      fi
    } >>"$monitor_log" 2>&1
    sleep 2
  done
}

run_case() {
  local dtype="$1"
  local name="$2"
  local ops="$3"
  local output="$base/$name"
  docker exec "$container" mkdir -p "$output"
  docker exec -e PYTHONPATH="$python_path" -e CUDA_VISIBLE_DEVICES="$physical_gpu" "$container" \
    python3 "$harness_dir/benchmark_infiniops_full.py" \
    --metadata "$metadata" \
    --output-dir "$output" \
    --device cuda \
    --dtype "$dtype" \
    --sha "$sha" \
    --min-run-time 0.01 \
    --timeout 20 \
    --ops "$ops"
}

docker exec "$container" mkdir -p "$harness_dir"
docker cp /tmp/benchmark_infiniops_full_v4.py "$container:$harness_dir/benchmark_infiniops_full.py"
docker cp /tmp/full_operator_benchmark_v4.py "$container:$harness_dir/full_operator_benchmark.py"

monitor &
monitor_pid=$!

set -o pipefail
run_case float16 targeted-fp16-v4 "$fp16_ops" 2>&1 | tee -a "$launcher_log"
fp16_exit=${PIPESTATUS[0]}
run_case float32 targeted-fp32-v4 "$fp32_ops" 2>&1 | tee -a "$launcher_log"
fp32_exit=${PIPESTATUS[0]}

{
  echo "=== post-run nvidia-smi ==="
  nvidia-smi -i "$physical_gpu"
  echo "end=$(date --iso-8601=seconds)"
  echo "fp16_exit=$fp16_exit"
  echo "fp32_exit=$fp32_exit"
} >>"$launcher_log" 2>&1

foreign_process_seen=0
[[ -s "$foreign_flag" ]] && foreign_process_seen=1
{
  echo "fp16_exit=$fp16_exit"
  echo "fp32_exit=$fp32_exit"
  echo "foreign_process_seen=$foreign_process_seen"
  echo "end=$(date --iso-8601=seconds)"
} >"$done_file"

wait "$monitor_pid"
if [[ "$fp16_exit" -ne 0 ]]; then
  exit "$fp16_exit"
fi
exit "$fp32_exit"

#!/usr/bin/env bash
set -uo pipefail

container=libaoming_accel_test_nvidia_20260714_145051
physical_gpu="${1:?physical GPU index is required}"
sha=01f63f4beaa1ca0e178c9b728c1706a4065657cf
base=/workspace/InfiniOps_perf_logs/full-operator-2026-07-30/a100
output="$base/shape-probe-v3"
harness_dir="$base/harness-v3"
metadata=/workspace/InfiniOps_weekly_2026-07-27_01f63f4b/install-a100-retry2/infini/torch_ops_metadata.json
python_path=/workspace/InfiniOps_weekly_2026-07-27_01f63f4b/install-a100-retry2
ops=adaptive_avg_pool3d_backward,adaptive_max_pool2d_backward,adaptive_max_pool3d_backward,bucketize,cholesky_solve,col2im,cross,cudnn_convolution,fft_hfft,fft_hfft2,fft_hfftn,fft_irfft,fft_irfft2,fft_irfftn,fractional_max_pool2d_backward,fractional_max_pool3d_backward,glu_backward,histogram,hspmm,im2col,index,internal_add_relu,internal_addmm_activation,internal_batch_norm_with_update,internal_chunk_cat,internal_conv_depthwise2d,internal_convert_indices_from_coo_to_csr,internal_fft_c2c,internal_fft_c2r,internal_fft_r2c,internal_int_mm,internal_linalg_det,internal_linalg_eigh,internal_linalg_slogdet,internal_linalg_solve_ex,internal_log_softmax_backward_data,internal_scaled_mm,internal_slow_conv2d_backward,internal_slow_conv2d_forward,internal_softmax_backward_data,inverse,linalg_cross,linalg_det,linalg_eig,linalg_eigh,linalg_eigvals,linalg_eigvalsh,linalg_householder_product,linalg_inv,linalg_inv_ex,linalg_ldl_factor,linalg_ldl_factor_ex,linalg_ldl_solve,linalg_lstsq,linalg_lu_solve,linalg_matmul,linalg_matrix_power,linalg_pinv,linalg_slogdet,linalg_tensorinv,linalg_tensorsolve,linspace,logspace,lu_solve,matrix_power,max_pool2d_with_indices_backward,max_pool3d_with_indices_backward,max_unpool2d,max_unpool3d,mkldnn_adaptive_avg_pool2d,multi_margin_loss_backward,multilabel_margin_loss,multilabel_margin_loss_backward,multilabel_margin_loss_forward,nanquantile,nll_loss_backward,nll_loss_forward,nll_loss2d,nll_loss2d_backward,nll_loss2d_forward,orgqr,ormqr,quantile,reflection_pad2d,reflection_pad2d_backward,reflection_pad3d,reflection_pad3d_backward,replication_pad2d,replication_pad2d_backward,replication_pad3d,replication_pad3d_backward,slogdet,slow_conv3d,slow_conv3d_forward,slow_conv_transpose2d,slow_conv_transpose3d,sparse_sampled_addmm,special_multigammaln,sspaddmm,thnn_conv2d
launcher_log=/tmp/infiniops-full-a100-shape-v3-launcher.log
monitor_log=/tmp/infiniops-full-a100-shape-v3-monitor.log
foreign_flag=/tmp/infiniops-full-a100-shape-v3-foreign.flag
done_file=/tmp/infiniops-full-a100-shape-v3.done
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
  echo "dtype=float16"
  echo "ops=$ops"
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

docker exec "$container" mkdir -p "$harness_dir" "$output"
docker cp /tmp/benchmark_infiniops_full_v3.py "$container:$harness_dir/benchmark_infiniops_full.py"
docker cp /tmp/full_operator_benchmark_v3.py "$container:$harness_dir/full_operator_benchmark.py"

monitor &
monitor_pid=$!

set -o pipefail
docker exec -e PYTHONPATH="$python_path" -e CUDA_VISIBLE_DEVICES="$physical_gpu" "$container" \
  python3 "$harness_dir/benchmark_infiniops_full.py" \
  --metadata "$metadata" \
  --output-dir "$output" \
  --device cuda \
  --dtype float16 \
  --sha "$sha" \
  --min-run-time 0.01 \
  --timeout 20 \
  --ops "$ops" 2>&1 | tee -a "$launcher_log"
exit_code=${PIPESTATUS[0]}

{
  echo "=== post-run nvidia-smi ==="
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

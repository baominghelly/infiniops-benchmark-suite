from pathlib import Path
import importlib.util
import unittest


class RunMooreBenchmarkContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = Path(__file__).parents[1].joinpath(
            "scripts", "run_moore_benchmark.sh"
        ).read_text(encoding="utf-8")
        cls.audit = Path(__file__).parents[1].joinpath(
            "scripts", "moore_gpu_audit.py"
        ).read_text(encoding="utf-8")
        cls.guide = Path(__file__).parents[1].joinpath(
            "docs", "run_moore_benchmark.md"
        ).read_text(encoding="utf-8")

    def test_uses_shared_delivery_and_container_local_source_paths(self):
        self.assertIn('output_root="${OUTPUT_ROOT:-/data-aisoft/infinibench_data/ops}"', self.script)
        self.assertIn('output_host="$output_root/$device_id/$run_date"', self.script)
        self.assertIn('output_container_root="${CONTAINER_OUTPUT_ROOT:-$output_root}"', self.script)
        self.assertIn('output_container="$output_container_root/$device_id/$run_date"', self.script)
        self.assertIn('container_tools_root="${CONTAINER_BENCHMARK_ROOT:-/data-aisoft/infinibench-scripts}"', self.script)
        self.assertIn('device_id="${DEVICE_ID:-moore-s5000}"', self.script)
        self.assertIn('source_root="${INFINIOPS_SOURCE_ROOT:-/workspace/InfiniOps}"', self.script)
        self.assertIn('infinirt_source_root="${INFINIRT_SOURCE_ROOT:-/workspace/InfiniRT}"', self.script)
        self.assertIn('build_cache_root="${INFINIOPS_BUILD_CACHE_ROOT:-/workspace/.cache/infinibench-moore}"', self.script)
        self.assertNotIn("/data-aisoft/infinibench-src", self.script)
        self.assertNotIn("/data-aisoft/infinibench-cache", self.script)

    def test_date_backfill_and_legacy_inputs_are_contracted(self):
        self.assertIn('execution_time="${EXECUTION_TIME:-$(date +%Y%m%d-%H%M%S)}"', self.script)
        self.assertIn('execution_date="${execution_time:0:8}"', self.script)
        self.assertIn('backfill_date="${BACKFILL_DATE:-}"', self.script)
        self.assertIn("RUN_DATE and CSV_DATE are no longer accepted", self.script)
        self.assertIn('run_date="$execution_date"', self.script)
        self.assertIn('run_date="backfill-${backfill_date//-/}-${execution_date}"', self.script)
        self.assertIn('csv_date="${backfill_date//-//}"', self.script)
        self.assertIn('csv_date="${execution_time:0:4}/${execution_time:4:2}/${execution_time:6:2}"', self.script)
        self.assertIn("<EXECUTION_DATE>/", self.guide)
        self.assertIn("backfill-<YYYYMMDD>-<EXECUTION_DATE>/", self.guide)

    def test_existing_run_directory_is_not_reused_silently(self):
        self.assertIn('reuse_run_dir="${REUSE_RUN_DIR:-0}"', self.script)
        self.assertIn('case "$reuse_run_dir" in', self.script)
        self.assertIn("output directory already exists and is non-empty", self.script)
        self.assertIn("REUSE_RUN_DIR=1 to replace it", self.script)
        self.assertIn('case "$output_container" in', self.script)
        self.assertIn("refusing to reset untrusted output path", self.script)
        self.assertIn('docker exec "$container" rm -rf -- "$output_container"', self.script)
        self.assertIn("set `REUSE_RUN_DIR=1`", self.guide)

    def test_source_fallback_lock_and_output_marker_are_contracted(self):
        self.assertIn("https://github.com/InfiniTensor/InfiniOps.git,https://gh-proxy.com/https://github.com/InfiniTensor/InfiniOps.git,https://ghproxy.net/https://github.com/InfiniTensor/InfiniOps.git", self.script)
        self.assertIn("https://github.com/InfiniTensor/InfiniRT.git,https://gh-proxy.com/https://github.com/InfiniTensor/InfiniRT.git,https://ghproxy.net/https://github.com/InfiniTensor/InfiniRT.git", self.script)
        self.assertIn("status --porcelain=v1", self.script)
        self.assertIn("merge --ff-only", self.script)
        self.assertIn('rev-list -1 --before="$cutoff"', self.script)
        self.assertIn('runner_lock="$output_root/$device_id/.infinibench-runner.lock"', self.script)
        self.assertIn("flock -n 9", self.script)
        self.assertIn('docker exec "$container" touch "$output_container/$output_path_marker"', self.script)
        self.assertIn('docker exec "$container" rm -f -- "$output_container/$output_path_marker"', self.script)

    def test_historical_compatibility_patches_are_selected_by_apply_check(self):
        self.assertIn("moore-paged-attention-low-smem-fallback.patch", self.script)
        self.assertIn("moore-paged-attention-low-smem-fallback-20260904.patch", self.script)
        self.assertIn("compat_patch_candidates=(", self.script)
        self.assertIn("apply --check \"$compat_patch_candidate\"", self.script)
        self.assertIn('echo "compat_patch_selected=$compat_patch"', self.script)

    def test_gpu_gate_preflight_and_occupied_mode_are_contracted(self):
        self.assertIn('gpu_max_utilization="${GPU_MAX_UTILIZATION_PERCENT:-5}"', self.script)
        self.assertIn('gpu_min_free_memory="${GPU_MIN_FREE_MEMORY_PERCENT:-0}"', self.script)
        self.assertIn('gpu_allow_occupied="${GPU_ALLOW_OCCUPIED:-0}"', self.script)
        self.assertIn("--max-utilization \"$gpu_max_utilization\"", self.script)
        self.assertIn("--min-free-memory \"$gpu_min_free_memory\"", self.script)
        self.assertIn("gpu_reselection=auto", self.script)
        self.assertIn('echo "gpu_max_utilization_percent=$gpu_max_utilization"', self.script)
        self.assertIn('echo "gpu_min_free_memory_percent=$gpu_min_free_memory"', self.script)
        self.assertIn('echo "gpu_allow_occupied=$gpu_allow_occupied"', self.script)
        self.assertIn("latency_comparable=$((1-gpu_allow_occupied))", self.script)
        self.assertIn("gpu_resource_mode=", self.script)
        self.assertIn("shared-process-validation", self.script)
        self.assertIn("foreign_allowed", self.audit)
        self.assertIn("allow_foreign", self.audit)
        self.assertIn("_free_memory_percent", self.audit)

    def test_moore_build_device_and_done_contracts_are_present(self):
        self.assertIn("--config-settings=cmake.define.WITH_MOORE=ON", self.script)
        self.assertIn('"--config-settings=cmake.define.CMAKE_C_COMPILER=$source_root/scripts/mcc_wrapper.sh"', self.script)
        self.assertIn("--device musa", self.script)
        self.assertIn("-e MUSA_VISIBLE_DEVICES=\"$physical_gpu\"", self.script)
        self.assertIn("--output \"$output_container/infiniops-moore-s5000.csv\"", self.script)
        self.assertIn('echo "exit_code=$exit_code"', self.script)
        self.assertIn('echo "foreign_process_seen=$foreign_process_seen"', self.script)
        self.assertIn('echo "execution_time=$execution_time"', self.script)
        self.assertIn('echo "csv_date=$csv_date"', self.script)
        self.assertIn('echo "backfill_date=$backfill_date"', self.script)
        self.assertIn('echo "infiniops_requested_ref=$infiniops_ref"', self.script)
        self.assertIn('echo "infinirt_requested_ref=$infinirt_ref"', self.script)
        self.assertIn('echo "infiniops_commit=$sha"', self.script)
        self.assertIn('echo "infinirt_commit=$infinirt_sha"', self.script)

    def test_moore_audit_free_memory_and_occupied_policy_behavior(self):
        audit_path = Path(__file__).parents[1].joinpath("scripts", "moore_gpu_audit.py")
        spec = importlib.util.spec_from_file_location("moore_gpu_audit_test", audit_path)
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)

        gpu = {
            "Index": "0",
            "Utilization": {"Gpu": "3%"},
            "FB Memory Usage": {"Total": "81920MiB", "Used": "20480MiB"},
        }
        self.assertEqual(audit._free_memory_percent(gpu), 75.0)

        audit.gpu_snapshot = lambda: [gpu]
        audit.process_snapshot = lambda: [{"gpu": "0", "pid": "123"}]
        with self.assertRaisesRegex(RuntimeError, "no idle Moore GPU"):
            audit.select_idle_gpu(
                samples=1,
                interval=0,
                max_utilization=5,
                min_free_memory=50,
                allow_occupied=False,
            )
        self.assertEqual(
            audit.select_idle_gpu(
                samples=1,
                interval=0,
                max_utilization=5,
                min_free_memory=50,
                allow_occupied=True,
            ),
            "0",
        )


if __name__ == "__main__":
    unittest.main()

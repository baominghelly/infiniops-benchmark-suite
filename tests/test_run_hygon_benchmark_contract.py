from pathlib import Path
import unittest


class RunHygonBenchmarkContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = Path(__file__).parents[1].joinpath(
            "scripts", "run_hygon_benchmark.sh"
        ).read_text(encoding="utf-8")
        cls.guide = Path(__file__).parents[1].joinpath(
            "docs", "run_hygon_benchmark.md"
        ).read_text(encoding="utf-8")

    def test_uses_shared_tools_and_direct_output_paths(self):
        self.assertIn('host_tools_root="${BENCHMARK_TOOLS_ROOT:-/data-aisoft/infinibench-scripts}"', self.script)
        self.assertIn('container_tools_root="${CONTAINER_BENCHMARK_ROOT:-/data-aisoft/infinibench-scripts}"', self.script)
        self.assertIn(
            'output_root="${OUTPUT_ROOT:-/data-aisoft/infinibench_data/ops}"',
            self.script,
        )
        self.assertIn('output_container_root="${CONTAINER_OUTPUT_ROOT:-$output_root}"', self.script)
        self.assertIn('device_id="${DEVICE_ID:-hygon-bw1000}"', self.script)
        self.assertIn('output_host="$output_root/$device_id/$run_date"', self.script)
        self.assertIn('output_container="$output_container_root/$device_id/$run_date"', self.script)

    def test_date_backfill_and_ref_semantics_are_present(self):
        self.assertIn('execution_time="${EXECUTION_TIME:-$(date +%Y%m%d-%H%M%S)}"', self.script)
        self.assertIn('execution_date="${execution_time:0:8}"', self.script)
        self.assertIn('backfill_date="${BACKFILL_DATE:-}"', self.script)
        self.assertIn("RUN_DATE and CSV_DATE are no longer accepted", self.script)
        self.assertIn(
            'run_date="backfill-${backfill_date//-/}-${execution_date}"',
            self.script,
        )
        self.assertIn('run_date="$execution_date"', self.script)
        self.assertIn('csv_date="${backfill_date//-//}"', self.script)
        self.assertIn('infiniops_ref="${INFINIOPS_REF:-}"', self.script)
        self.assertIn('infinirt_ref="${INFINIRT_REF:-}"', self.script)
        self.assertIn('--before="$cutoff"', self.script)

    def test_repo_urls_lock_marker_and_done_contract_are_present(self):
        self.assertIn("https://github.com/InfiniTensor/InfiniOps.git,https://gh-proxy.com/https://github.com/InfiniTensor/InfiniOps.git,https://ghproxy.net/https://github.com/InfiniTensor/InfiniOps.git", self.script)
        self.assertIn("https://github.com/InfiniTensor/InfiniRT.git,https://gh-proxy.com/https://github.com/InfiniTensor/InfiniRT.git,https://ghproxy.net/https://github.com/InfiniTensor/InfiniRT.git", self.script)
        self.assertIn("status --porcelain=v1", self.script)
        self.assertIn("merge --ff-only", self.script)
        self.assertIn('runner_lock="$output_root/$device_id/.infinibench-runner.lock"', self.script)
        self.assertIn("flock -n 9", self.script)
        self.assertIn('docker exec "$container" touch "$output_container/$output_path_marker"', self.script)
        self.assertIn('docker exec "$container" rm -f -- "$output_container/$output_path_marker"', self.script)
        self.assertIn('echo "execution_time=$execution_time"', self.script)
        self.assertIn('echo "execution_date=$execution_date"', self.script)
        self.assertIn('echo "run_date=$run_date"', self.script)
        self.assertIn('echo "csv_date=$csv_date"', self.script)
        self.assertIn('echo "source_backfill_date=$backfill_date"', self.script)
        self.assertIn('echo "sha=$sha"', self.script)
        self.assertIn('echo "infinirt_sha=$infinirt_sha"', self.script)

    def test_fixed_run_directory_is_not_reused_silently(self):
        self.assertIn('reuse_run_dir="${REUSE_RUN_DIR:-0}"', self.script)
        self.assertIn("output directory already exists and is non-empty", self.script)
        self.assertIn("REUSE_RUN_DIR=1 to replace it", self.script)
        self.assertIn('case "$output_container" in', self.script)
        self.assertIn("refusing to reset untrusted output path", self.script)
        self.assertIn("deletes and recreates only that run directory", self.guide)
        self.assertIn("never moved or deleted", self.guide)

    def test_hygon_build_and_dcu_paths_are_used(self):
        self.assertIn("--config-settings=cmake.define.WITH_HYGON=ON", self.script)
        self.assertIn('"--config-settings=cmake.define.INFINI_RT_ROOT=$infinirt_install_root"', self.script)
        self.assertIn("-e HIP_VISIBLE_DEVICES=\"$physical_gpu\"", self.script)
        self.assertIn("python3 \"$gpu_audit\" \"${audit_args[@]}\"", self.script)
        self.assertIn("audit_args=(", self.script)
        self.assertIn("--output \"$output_container/infiniops-hygon-bw1000.csv\"", self.script)

    def test_gpu_resource_gate_is_parameterized(self):
        self.assertIn('gpu_max_utilization="${GPU_MAX_UTILIZATION_PERCENT:-5}"', self.script)
        self.assertIn('gpu_min_free_memory="${GPU_MIN_FREE_MEMORY_PERCENT:-0}"', self.script)
        self.assertIn('gpu_allow_occupied="${GPU_ALLOW_OCCUPIED:-0}"', self.script)
        self.assertIn('--max-utilization "$gpu_max_utilization"', self.script)
        self.assertIn('--min-free-memory "$gpu_min_free_memory"', self.script)
        self.assertIn('echo "gpu_max_utilization_percent=$gpu_max_utilization"', self.script)
        self.assertIn('echo "gpu_min_free_memory_percent=$gpu_min_free_memory"', self.script)
        self.assertIn('echo "gpu_allow_occupied=$gpu_allow_occupied"', self.script)


if __name__ == "__main__":
    unittest.main()

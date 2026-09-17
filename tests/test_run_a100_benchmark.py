import unittest
from pathlib import Path


class RunA100BenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_a100_benchmark.sh"
        ).read_text(encoding="utf-8")
        cls.guide = (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "run_nvidia_benchmark.md"
        ).read_text(encoding="utf-8")

    def test_uses_shared_paths_and_new_date_semantics(self):
        self.assertIn('device_id="${DEVICE_ID:-nvidia-a100}"', self.script)
        self.assertIn(
            'output_root="${OUTPUT_ROOT:-/data-aisoft/infinibench_data/ops}"',
            self.script,
        )
        self.assertIn(
            'container_tools_root="${CONTAINER_BENCHMARK_ROOT:-/data-aisoft/infinibench-scripts}"',
            self.script,
        )
        self.assertIn('execution_date="${execution_time:0:8}"', self.script)
        self.assertIn('run_date="$execution_date"', self.script)
        self.assertIn(
            'run_date="backfill-${backfill_date//-/}-${execution_date}"',
            self.script,
        )
        self.assertIn("<EXECUTION_DATE>/", self.guide)
        self.assertIn("backfill-<YYYYMMDD>-<EXECUTION_DATE>/", self.guide)
        self.assertIn("RUN_DATE and CSV_DATE are no longer accepted", self.script)
        self.assertIn("--output \"$output_container/infiniops-nvidia-a100.csv\"", self.script)

    def test_existing_date_directory_is_not_reused_silently(self):
        self.assertIn('reuse_run_dir="${REUSE_RUN_DIR:-0}"', self.script)
        self.assertIn("invalid REUSE_RUN_DIR", self.script)
        self.assertIn("output directory already exists and is non-empty", self.script)
        self.assertIn("REUSE_RUN_DIR=1 to replace it", self.script)
        self.assertIn('case "$output_container" in', self.script)
        self.assertIn("refusing to reset untrusted output path", self.script)
        self.assertIn("explicitly set", self.guide)
        self.assertIn("`REUSE_RUN_DIR=1`", self.guide)

    def test_source_updates_are_safe_and_support_backfill(self):
        self.assertIn("https://github.com/InfiniTensor/InfiniOps.git", self.script)
        self.assertIn("https://gh-proxy.com/https://github.com/InfiniTensor/InfiniOps.git", self.script)
        self.assertIn("https://ghproxy.net/https://github.com/InfiniTensor/InfiniOps.git", self.script)
        self.assertIn("status --porcelain=v1", self.script)
        self.assertIn("repository has local changes; refusing to overwrite", self.script)
        self.assertIn("merge --ff-only", self.script)
        self.assertIn('cutoff="${source_date}T23:59:59+08:00"', self.script)
        self.assertIn("rev-list -1 --before", self.script)
        self.assertIn('infiniops_ref="${INFINIOPS_REF:-}"', self.script)
        self.assertIn('infinirt_ref="${INFINIRT_REF:-}"', self.script)

    def test_uses_container_execution_locking_and_nvidia_preflight(self):
        self.assertIn("flock -n 9", self.script)
        self.assertIn('runner_lock="$output_root/$device_id/.infinibench-runner.lock"', self.script)
        self.assertIn('docker exec "$container" touch "$output_container/$output_path_marker"', self.script)
        self.assertIn('docker exec "$container" rm -f -- "$output_container/$output_path_marker"', self.script)
        self.assertIn("--config-settings=cmake.define.WITH_NVIDIA=ON", self.script)
        self.assertIn("--config-settings=cmake.define.CMAKE_CUDA_ARCHITECTURES=$cuda_architectures", self.script)
        self.assertIn(
            "nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.total,memory.used",
            self.script,
        )
        self.assertIn("nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory", self.script)
        self.assertIn("write_done_on_exit", self.script)
        self.assertIn("echo \"gpu_uuid=${gpu_uuid:-}\"", self.script)
        self.assertNotIn("mx-smi", self.script)

    def test_gpu_resource_gates_are_parameterized(self):
        self.assertIn('gpu_max_utilization="${GPU_MAX_UTILIZATION_PERCENT:-5}"', self.script)
        self.assertIn('gpu_min_free_memory="${GPU_MIN_FREE_MEMORY_PERCENT:-0}"', self.script)
        self.assertIn('gpu_allow_occupied="${GPU_ALLOW_OCCUPIED:-0}"', self.script)
        self.assertIn("invalid GPU_ALLOW_OCCUPIED", self.script)
        self.assertIn("free_memory < min_free_memory", self.script)
        self.assertIn("gpu_free_memory_percent=$free_memory", self.script)
        self.assertIn("process_owner=foreign_allowed", self.script)
        self.assertIn("gpu_allow_occupied=$gpu_allow_occupied", self.script)


if __name__ == "__main__":
    unittest.main()

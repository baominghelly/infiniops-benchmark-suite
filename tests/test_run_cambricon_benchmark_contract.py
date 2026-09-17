from pathlib import Path
import unittest


class RunCambriconBenchmarkContractTests(unittest.TestCase):
    def test_runner_uses_shared_paths_and_new_date_semantics(self):
        script = Path(__file__).parents[1].joinpath(
            "scripts", "run_cambricon_benchmark.sh"
        ).read_text(encoding="utf-8")

        self.assertIn(
            'output_root="${OUTPUT_ROOT:-/data-aisoft/infinibench_data/ops}"',
            script,
        )
        self.assertIn(
            'container_tools_root="${CONTAINER_BENCHMARK_ROOT:-/data-aisoft/infinibench-scripts}"',
            script,
        )
        self.assertIn('device_id="${DEVICE_ID:-cambricon-mlu590}"', script)
        self.assertIn('output_host="$output_root/$device_id/$run_dir"', script)
        self.assertIn('output_container="$output_container_root/$device_id/$run_dir"', script)
        self.assertIn('execution_date="${execution_time:0:8}"', script)
        self.assertIn('run_dir="backfill-${backfill_date//-/}-${execution_date}"', script)
        self.assertIn('run_dir="$execution_date"', script)
        self.assertIn('csv_date="${backfill_date//-//}"', script)
        self.assertIn("RUN_DATE and CSV_DATE are no longer accepted", script)
        self.assertIn("--output \"$output_container/infiniops-cambricon-mlu590.csv\"", script)

    def test_existing_date_directory_is_not_reused_silently(self):
        script = Path(__file__).parents[1].joinpath(
            "scripts", "run_cambricon_benchmark.sh"
        ).read_text(encoding="utf-8")
        guide = Path(__file__).parents[1].joinpath(
            "docs", "run_cambricon_benchmark.md"
        ).read_text(encoding="utf-8")

        self.assertIn('reuse_run_dir="${REUSE_RUN_DIR:-0}"', script)
        self.assertIn("output directory already exists and is non-empty", script)
        self.assertIn("set REUSE_RUN_DIR=1 to replace it", script)
        self.assertIn('case "$output_container" in', script)
        self.assertIn("refusing to reset untrusted output path", script)
        self.assertIn("To replace a failed", guide)
        self.assertIn("set `REUSE_RUN_DIR=1`", guide)

    def test_runner_prepares_sources_with_mirrors_lock_and_mlu_contract(self):
        script = Path(__file__).parents[1].joinpath(
            "scripts", "run_cambricon_benchmark.sh"
        ).read_text(encoding="utf-8")

        self.assertIn("https://github.com/InfiniTensor/InfiniOps.git", script)
        self.assertIn("https://gh-proxy.com/https://github.com/InfiniTensor/InfiniOps.git", script)
        self.assertIn("https://ghproxy.net/https://github.com/InfiniTensor/InfiniOps.git", script)
        self.assertIn("status --porcelain=v1", script)
        self.assertIn("merge --ff-only", script)
        self.assertIn('rev-list -1 --before="$cutoff"', script)
        self.assertIn("flock -n 9", script)
        self.assertIn("--config-settings=cmake.define.WITH_CAMBRICON=ON", script)
        self.assertIn("--device mlu", script)
        self.assertIn("-e MLU_VISIBLE_DEVICES=", script)
        self.assertIn("cambricon_mlu_audit.py", script)
        self.assertIn('gpu_max_utilization="${GPU_MAX_UTILIZATION_PERCENT:-5}"', script)
        self.assertIn('gpu_min_free_memory="${GPU_MIN_FREE_MEMORY_PERCENT:-0}"', script)
        self.assertIn('gpu_allow_occupied="${GPU_ALLOW_OCCUPIED:-0}"', script)
        self.assertIn("--min-free-memory-percent", script)
        self.assertIn("--allow-occupied", script)
        self.assertIn("latency_comparable=$((1-gpu_allow_occupied))", script)
        self.assertIn("echo \"foreign_process_seen=$foreign_process_seen\"", script)
        self.assertIn("echo \"source_backfill_date=$backfill_date\"", script)
        self.assertIn("echo \"infiniops_requested_ref=$infiniops_ref\"", script)
        self.assertIn("echo \"infinirt_requested_ref=$infinirt_ref\"", script)
        self.assertIn("echo \"gpu_max_utilization_percent=$gpu_max_utilization\"", script)
        self.assertIn("echo \"gpu_min_free_memory_percent=$gpu_min_free_memory\"", script)
        self.assertIn("echo \"gpu_allow_occupied=$gpu_allow_occupied\"", script)
        self.assertIn("echo \"exit_code=$exit_code\"", script)
        self.assertIn("echo \"execution_time=$execution_time\"", script)
        self.assertIn("echo \"csv_date=$csv_date\"", script)
        self.assertIn("echo \"source_backfill_date=$backfill_date\"", script)


if __name__ == "__main__":
    unittest.main()

from pathlib import Path
import unittest


class RunIluvatarBenchmarkContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = Path(__file__).parents[1].joinpath(
            "scripts", "run_iluvatar_benchmark.sh"
        ).read_text(encoding="utf-8")

    def test_uses_shared_paths_and_current_date_semantics(self):
        self.assertIn(
            'container_tools_root="${CONTAINER_BENCHMARK_ROOT:-/data-aisoft/infinibench-scripts}"',
            self.script,
        )
        self.assertIn('device_id="${DEVICE_ID:-iluvatar-tg150}"', self.script)
        self.assertIn(
            'patch_dir="${ILUVATAR_PATCH_DIR:-$container_tools_root/patches/iluvatar-tg150}"',
            self.script,
        )
        self.assertIn(
            'output_root="${OUTPUT_ROOT:-/data-aisoft/infinibench_data/ops}"',
            self.script,
        )
        self.assertIn('output_host="$output_root/$device_id/$run_date"', self.script)
        self.assertIn('output_container="$output_container_root/$device_id/$run_date"', self.script)
        self.assertIn('execution_time="${EXECUTION_TIME:-$(date +%Y%m%d-%H%M%S)}"', self.script)
        self.assertIn('execution_date="${execution_time:0:8}"', self.script)
        self.assertIn('backfill_date="${BACKFILL_DATE:-}"', self.script)
        self.assertIn("RUN_DATE and CSV_DATE are no longer accepted", self.script)
        self.assertIn('run_date="$execution_date"', self.script)
        self.assertIn(
            'run_date="backfill-${backfill_date//-/}-${execution_date}"',
            self.script,
        )
        self.assertIn('csv_date="${backfill_date//-//}"', self.script)

    def test_date_only_directory_reuse_contract_is_present(self):
        self.assertIn('reuse_run_dir="${REUSE_RUN_DIR:-0}"', self.script)
        self.assertIn("output directory already exists and is non-empty", self.script)
        self.assertIn("REUSE_RUN_DIR=1 to replace it", self.script)
        self.assertIn('case "$output_container" in', self.script)
        self.assertIn("refusing to reset untrusted output path", self.script)

    def test_document_declares_date_only_output_directories(self):
        guide = Path(__file__).parents[1].joinpath(
            "docs", "run_tianshu58_benchmark.md"
        ).read_text(encoding="utf-8")
        self.assertIn("<EXECUTION_DATE>/", guide)
        self.assertIn(
            "backfill-<TARGET_YYYYMMDD>-<EXECUTION_YYYYMMDD>/",
            guide,
        )
        self.assertIn("does not contain hours, minutes, or seconds", guide)
        self.assertIn("To replace a failed or superseded attempt", guide)
        self.assertIn("set `REUSE_RUN_DIR=1`", guide)

    def test_source_preparation_contract_is_present(self):
        self.assertIn(
            "https://github.com/InfiniTensor/InfiniOps.git,"
            "https://gh-proxy.com/https://github.com/InfiniTensor/InfiniOps.git,"
            "https://ghproxy.net/https://github.com/InfiniTensor/InfiniOps.git",
            self.script,
        )
        self.assertIn(
            "https://github.com/InfiniTensor/InfiniRT.git,"
            "https://gh-proxy.com/https://github.com/InfiniTensor/InfiniRT.git,"
            "https://ghproxy.net/https://github.com/InfiniTensor/InfiniRT.git",
            self.script,
        )
        self.assertIn("status --porcelain=v1", self.script)
        self.assertIn("merge --ff-only", self.script)
        self.assertIn('rev-list -1 --before="$cutoff"', self.script)
        self.assertIn('infiniops_ref="${INFINIOPS_REF:-}"', self.script)
        self.assertIn('infinirt_ref="${INFINIRT_REF:-}"', self.script)

    def test_gpu_resource_gate_contract_is_present(self):
        self.assertIn('gpu_max_utilization="${GPU_MAX_UTILIZATION_PERCENT:-5}"', self.script)
        self.assertIn('gpu_min_free_memory="${GPU_MIN_FREE_MEMORY_PERCENT:-0}"', self.script)
        self.assertIn('gpu_allow_occupied="${GPU_ALLOW_OCCUPIED:-0}"', self.script)
        self.assertIn("--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,memory.free", self.script)
        self.assertIn("if index in busy and not allow_occupied:", self.script)
        self.assertIn("if free_memory_percent < min_free_memory:", self.script)
        self.assertIn("process_owner=foreign_allowed", self.script)
        self.assertIn('[[ "$processes" -gt 0 && "$gpu_allow_occupied" != "1" ]]', self.script)

    def test_lock_marker_and_done_contract_are_present(self):
        self.assertIn('runner_lock="$output_root/$device_id/.infinibench-runner.lock"', self.script)
        self.assertIn("flock -n 9", self.script)
        self.assertIn('docker exec "$container" touch "$output_container/$output_path_marker"', self.script)
        self.assertIn('docker exec "$container" rm -f -- "$output_container/$output_path_marker"', self.script)
        self.assertIn('echo "exit_code=$result_code"', self.script)
        self.assertIn('echo "foreign_process_seen=${foreign_process_seen:-0}"', self.script)
        self.assertIn('echo "gpu_max_utilization_percent=$gpu_max_utilization"', self.script)
        self.assertIn('echo "gpu_min_free_memory_percent=$gpu_min_free_memory"', self.script)
        self.assertIn('echo "gpu_allow_occupied=$gpu_allow_occupied"', self.script)
        self.assertIn('echo "execution_time=$execution_time"', self.script)
        self.assertIn('echo "csv_date=$csv_date"', self.script)
        self.assertIn('echo "backfill_date=${backfill_date:-}"', self.script)
        self.assertIn('echo "sha=${sha:-}"', self.script)
        self.assertIn('echo "infinirt_sha=${infinirt_sha:-}"', self.script)

    def test_document_declares_occupied_gpu_mode_non_comparable(self):
        guide = Path(__file__).parents[1].joinpath(
            "docs", "run_tianshu58_benchmark.md"
        ).read_text(encoding="utf-8")
        self.assertIn("GPU_ALLOW_OCCUPIED=1", guide)
        self.assertIn("only for validating end-to-end workflow behavior", guide)
        self.assertIn("not comparable with exclusive runs", guide)


if __name__ == "__main__":
    unittest.main()

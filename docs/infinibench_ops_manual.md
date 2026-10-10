# Manual InfiniBench global-run deployment

This pilot deliberately has no enabled scheduler. Codex is not part of the
runtime path.

## 1. Create a global run

Run on the A100 host or inside `baoming_hermes_agent`:

```bash
python3 /opt/data/scripts/create_global_run.py \
  --global-run-id 20260922-manual \
  --execution-time 20260922-200000 \
  --schedule-mode manual \
  --infiniops-ref <commit-or-branch> \
  --infinirt-ref <commit-or-branch>
```

The creator resolves refs once and writes:

```text
/data-aisoft/infinibench_data/ops/.runs/<global-run-id>/global-manifest.json
```

Retries must reuse the same manifest. Workers never resolve `master` themselves.

## 2. NVIDIA worker

```bash
docker exec baoming_hermes_agent \
  python3 /opt/data/scripts/run_platform_worker.py \
    --global-run-id 20260922-manual \
    --device nvidia-a100 \
    --adapter /opt/data/scripts/run-nvidia-a100-benchmark.sh
```

Add `--dry-run` to inspect the command without changing state.

## 3. MetaX worker

Run on `metax49`:

```bash
python3 /data-aisoft/baoming/infinibench-workers/bin/run_platform_worker.py \
  --global-run-id 20260922-manual \
  --device metax-c550 \
  --adapter /data-aisoft/baoming/infinibench-workers/metax-c550/bin/run-metax-c550-weekly.sh
```

The worker state is written under:

```text
/data-aisoft/baoming/infinibench-workers/<device>/<global-run-id>/
```

Canonical benchmark data remains under:

```text
/data-aisoft/infinibench_data/ops/<device>/<execution-date>/
```

## Safety

- A worker refuses a non-empty canonical output unless
  `--replace-existing-output` is explicitly passed.
- A succeeded worker refuses to run again.
- The per-platform lock prevents two workers from writing one device.
- The global manifest freezes both source commits and the execution timestamp.

## Heartbeat and retry policy

While a worker runs, it refreshes `heartbeat_at` in
`worker-manifest.json` every 30 seconds. A heartbeat older than 120 seconds is
reported as stale by `infinibenchctl status`.

On failure, the worker writes a `failure_policy` object:

```json
{
  "failure_code": "container_cuda_lost",
  "auto_retry": true,
  "max_attempts": 16,
  "max_auto_retries": 16,
  "retry_delay_base_seconds": 300,
  "retry_delay_multiplier": 2.0,
  "retry_delay_cap_seconds": 3600,
  "retry_delay_seconds": 300,
  "retry_window_seconds": 72000,
  "recommended_action": "restart_benchmark_container_then_retry_without_rebuild",
  "preserve_partial_results": false,
  "allow_replace_existing_output": true
}
```

`max_attempts` is the total-attempt limit. A transient failure is retried
after 5, 10, 20, 40, and then at most 60-minute intervals. A new attempt is
started only within 20 hours after the global manifest was created, so retries
do not spill into the next daily run; an already-running attempt is not killed
by that window. `max_auto_retries` remains as a compatibility field for
deployed workers that have not yet learned `max_attempts`.

Known transient classes include container CUDA loss, benchmark-container
down state, source-network failure, and GPU-busy rejection. Timeouts remain
manual because they can indicate a hung accelerator or worker.
Unknown build, benchmark, configuration, and operator-stop failures require
manual review. If a failed output contains any `results.json` or
`results.jsonl`, `allow_replace_existing_output` is forced to false. Known
transient failures remain retryable and use `RESUME=1`; complete profile
results are skipped and incomplete `results.jsonl` checkpoints are passed to
the benchmark harness with `--resume`.

## Periodic MetaX worker takeover

MetaX C550 runs this cron entry every five minutes:

```bash
*/5 * * * * /data-aisoft/baoming/infinibench-workers/metax-c550/bin/periodic_worker.sh
```

The launcher uses a host-level `flock`, reads immutable global manifests from
the shared runs directory, and exits immediately when a worker has a fresh
heartbeat. If the heartbeat is stale, or a transient failure's retry delay has
elapsed, it starts the worker with `--resume-existing-output`. The worker still
uses its per-run lease, so multiple cron invocations cannot benchmark one device
concurrently.

Pause or stop all workers from the A100 control host with:

```bash
/data-aisoft/baoming/infinibench-ops/bin/infinibenchctl stop --all
```

The periodic launcher honors the same shared pause marker and will not launch a
new worker while the system is paused. Remove only the MetaX cron entry with
`crontab -e` on `metax49` if that platform must stop independently.

## Daily six-platform pilot

The A100 daily launcher creates one immutable manifest for:

```text
nvidia-a100
metax-c550
ascend-910c
moore-s5000
iluvatar-tg150
hygon-bw1100
```

It uses `decode-4k`, all three dtypes, and the same frozen InfiniOps/InfiniRT
commits for every platform. Moore defaults to shared-process validation because
that machine hosts short model-test jobs; its latency is marked non-comparable.
Ascend245 and both BW1100 nodes do not permit user crontabs, so A100 invokes
their periodic workers over SSH every five minutes through
`ops/periodic_ascend_remote.sh` and `ops/periodic_hygon_remote.sh`.

Hygon BW1100 uses one logical daily platform (`hygon-bw1100`) and two physical
nodes (`hygon-bw1100-01`/`dcu1` and `hygon-bw1100-02`/`dcu2`). Both nodes
write to `ops/hygon-bw1100/<YYYYMMDD>/` and share the logical worker lease and
heartbeat. The worker manifest records the node that owns each attempt. A
healthy heartbeat makes the other node exit; a stale heartbeat or retryable
failure allows either node to take the shared lock and resume checkpoints.
Node 1 uses container `baoming_infiniops_hygon`; node 2 uses
`baoming_infiniops_hygon_dcu2`. Both pin `HYGON_ARCH=gfx938`.

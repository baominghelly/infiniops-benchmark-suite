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

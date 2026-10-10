# NVIDIA A100 and Cambricon Hermes deployment

This deployment keeps Hermes state under `/data-aisoft/baoming/hermes`.

## Layout

- Hermes home: `/data-aisoft/baoming/hermes/data`
- Hermes workflow script: `/data-aisoft/baoming/hermes/data/scripts/run-nvidia-a100-benchmark.sh`
- Workflow attempt records: `/data-aisoft/baoming/hermes/runs`
- Workspace: `/data-aisoft/baoming/hermes/workspace`
- Canonical benchmark output: `/data-aisoft/infinibench_data/ops/nvidia-a100/<execution-date>/`
- Cambricon collector script: `/data-aisoft/baoming/hermes/data/scripts/run-cambricon-result-collector.sh`
- Cambricon collector state: `/data-aisoft/baoming/hermes/runs/cambricon-result-collector/`

The scheduled job is `nvidia-a100-weekly-benchmark`. It runs `0 2 * * 1` in
Asia/Shanghai and invokes the shared entry:

```text
/data-aisoft/infinibench-scripts/scripts/run_benchmark.sh nvidia-a100
```

The first attempt is exclusive. It retries in shared GPU mode only when the
failure is an idle-resource gate or foreign-process condition. Shared-mode
results are flow-validation only and their latency is not comparable.

## Known-failure Doctor

The benchmark wrapper writes one of:

```text
/data-aisoft/baoming/hermes/runs/<execution-date>/success.json
/data-aisoft/baoming/hermes/runs/<execution-date>/failure.json
```

`nvidia-a100-doctor` runs as a Hermes no-agent cron job. It classifies the
latest failed run without an LLM and writes:

```text
doctor-state.json
doctor-actions.jsonl
retry-request.json
```

Implemented deterministic cases are:

- `container_cuda_lost`: verify host GPU health, container state, no active
  benchmark process, restart the benchmark container, verify CUDA, and schedule
  a no-rebuild retry.
- `no_idle_gpu` / `foreign_process`: preserve logs, never kill the foreign
  process, and schedule a later exclusive retry.
- `disk_full`: prune only old Hermes-owned run logs under the Hermes runs root,
  subject to retention and byte caps.
- `source_dirty`: export Git status and diff for human review; never reset or
  clean the source tree.
- `output_dir_nonempty`: schedule an explicit `REUSE_RUN_DIR=1` retry.

Unknown failures and failed deterministic recovery write `needs_llm=true` to
Doctor state. `nvidia-a100-llm-monitor.py` emits stable `NO_LLM_NEEDED` output
unless that flag is true, so known failures do not invoke a model.

## Cambricon result collector

The Cambricon MLU590 host is outside the internal network. Its
`/data-aisoft/infinibench_data/ops/cambricon-mlu590` directory is local disk,
not the shared A100 NFS volume. The `cambricon-mlu590-result-collector` Hermes
no-agent cron runs every five minutes and pulls completed runs back to:

```text
/data-aisoft/infinibench_data/ops/cambricon-mlu590/<run-dir>/
```

A run is archived only when both conditions are true:

1. `benchmark.done` is non-empty.
2. at least one non-empty top-level `infiniops-*.csv` exists.

The collector transfers through `ssh | tar` into a staging directory, writes
`.collector-manifest.sha256` and `.collector-success.json`, validates the
completion metadata and CSV, and only then renames the staging directory to its
canonical name. It never overwrites an existing destination. Completed runs
without a CSV are recorded once under
`/data-aisoft/baoming/hermes/runs/cambricon-result-collector/skipped/` rather
than copied as empty result directories.

The collector uses a dedicated private key at:

```text
/data-aisoft/baoming/hermes/data/ssh/cambricon_result_collector
```

The matching public key is installed for the Cambricon `libaoming` account.
The key is not reused for interactive administration.

Manual dry run and manual collection:

```bash
COLLECTOR_DRY_RUN=1 \
  /data-aisoft/baoming/hermes/data/scripts/run-cambricon-result-collector.sh

/data-aisoft/baoming/hermes/data/scripts/run-cambricon-result-collector.sh
```

Manual Doctor dry run:

```bash
docker exec baoming_hermes_agent \
  python3 /opt/data/scripts/nvidia_a100_doctor.py
```

Manual apply pass:

```bash
docker exec baoming_hermes_agent \
  python3 /opt/data/scripts/nvidia_a100_doctor.py --apply --execute-retry
```

## Operations

Check health and cron:

```bash
docker inspect baoming_hermes_agent --format '{{.State.Status}} {{.State.Health.Status}}'
docker exec baoming_hermes_agent hermes cron status
docker exec baoming_hermes_agent hermes cron list
```

Dry-run the workflow without starting a benchmark:

```bash
docker exec -e DRY_RUN=1 baoming_hermes_agent \
  /opt/data/scripts/run-nvidia-a100-benchmark.sh
```

Run the workflow manually:

```bash
docker exec baoming_hermes_agent \
  /opt/data/scripts/run-nvidia-a100-benchmark.sh
```

Recreate the Hermes container from this directory:

```bash
/data-aisoft/baoming/hermes/data/deploy/recreate-baoming-hermes.sh
```

`docker-compose.yml` is a declarative reference for the same configuration.

## LLM API and Triage

Configure one provider in the new Hermes instance. For example, OpenRouter:

```bash
docker exec -it baoming_hermes_agent \
  hermes auth add openrouter --type api-key
docker exec baoming_hermes_agent hermes config set model.provider openrouter
docker restart baoming_hermes_agent
```

For an OpenAI-compatible internal gateway, configure a named provider and put its
key in `/opt/data/.env`; do not place keys in this repository or in cron text.
After a model test succeeds, create the gated LLM triage cron:

```bash
docker exec baoming_hermes_agent hermes cron create \
  --name nvidia-a100-llm-triage \
  --monitor-script scripts/nvidia_a100_llm_monitor.py \
  --continuity \
  --deliver local \
  "*/5 * * * *" \
  "You are the NVIDIA A100 benchmark reliability engineer. The monitor output is either NO_LLM_NEEDED or a JSON triage request. For NO_LLM_NEEDED, reply with one line and take no action. For a triage request, inspect the listed Doctor state, action history, and logs. Diagnose the failure and write a note and patch under /data-aisoft/baoming/hermes/workspace. Do not kill external GPU processes, do not modify shared platform runners, and do not reset or clean source trees."
```

# InfiniBench operations manual

## Deployment paths

Control command:

```text
/data-aisoft/baoming/infinibench-ops/bin/infinibenchctl
```

Daily schedule:

```text
/data-aisoft/baoming/infinibench-ops/scheduled_daily.sh
```

The schedule is installed as an A100 host crontab entry. The host clock is UTC,
so Beijing 02:00 is represented as 18:00 UTC.

## Start one run manually

Small smoke run on both deployed platforms:

```bash
/data-aisoft/baoming/infinibench-ops/bin/infinibenchctl run-once \
  --device nvidia-a100 \
  --device metax-c550 \
  --profile decode-4k-small
```

Full run:

```bash
/data-aisoft/baoming/infinibench-ops/bin/infinibenchctl run-once \
  --device nvidia-a100 \
  --device metax-c550 \
  --profile decode-4k
```

Add `--dry-run` to inspect parameters without creating a manifest or workers.
The controller refuses to launch while a global pause marker exists.

## Check status

```bash
/data-aisoft/baoming/infinibench-ops/bin/infinibenchctl status
```

Worker detail:

```bash
find /data-aisoft/baoming/infinibench-workers \
  -name worker-manifest.json -print
```

## Pause future runs

```bash
/data-aisoft/baoming/infinibench-ops/bin/infinibenchctl pause
```

This prevents new `run-once` calls and daily launches, but does not kill an
active benchmark.

## Stop active workers

```bash
/data-aisoft/baoming/infinibench-ops/bin/infinibenchctl stop --all
```

Stop one global run:

```bash
/data-aisoft/baoming/infinibench-ops/bin/infinibenchctl stop \
  --run-id 20260923-daily
```

Stopping writes a pause marker. It terminates only benchmark processes whose
command line contains that run's exact output directory; it does not stop the
shared platform containers.

## Cancel a run

```bash
/data-aisoft/baoming/infinibench-ops/bin/infinibenchctl cancel-run \
  20260923-daily
```

This writes:

```text
/data-aisoft/infinibench_data/ops/.runs/<run-id>/control.json
```

and stops its active workers. Use `--no-stop` only when the run should be
marked canceled without sending termination signals.

## Resume scheduling

```bash
/data-aisoft/baoming/infinibench-ops/bin/infinibenchctl resume
```

This removes the global pause marker. It does not resurrect a canceled run.

## Daily schedule

The A100 crontab entry runs at 18:00 UTC, equal to 02:00 Beijing time:

```text
0 18 * * * /data-aisoft/baoming/infinibench-ops/scheduled_daily.sh >> /data-aisoft/baoming/infinibench-ops/logs/daily.log 2>&1
```

It creates one global manifest for that Beijing date, then launches NVIDIA and
MetaX workers. Results are written under:

```text
/data-aisoft/infinibench_data/ops/nvidia-a100/<YYYYMMDD>/
/data-aisoft/infinibench_data/ops/metax-c550/<YYYYMMDD>/
```

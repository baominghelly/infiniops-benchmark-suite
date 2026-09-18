"""Export successful rows from complete and interrupted benchmark runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from .export_full_operator_report import _dedupe, _load_rows, main as export_main
except ImportError:
    from export_full_operator_report import _dedupe, _load_rows, main as export_main


def _partial_csv_name(output_name: str) -> str:
    path = Path(output_name)
    if path.suffix != ".csv":
        raise ValueError("--output-name must end with .csv")
    return path.with_suffix(".partial.csv").name


def _write_manifest(
    path: Path,
    *,
    sources: list[Path],
    rows: list[dict[str, Any]],
    successes: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    output_root: Path,
    output_name: str,
    args: argparse.Namespace,
) -> None:
    payload: dict[str, Any] = {
        "result_status": "partial",
        "platform": args.platform,
        "date": args.date,
        "tester": args.tester,
        "infiniops_commit": args.infiniops_commit,
        "infinirt_commit": args.infinirt_commit,
        "failure_category": args.failure_category,
        "failure_reason": args.failure_reason,
        "output_root": str(output_root),
        "partial_csv": str(output_root / _partial_csv_name(output_name)),
        "partial_diagnostics": str(output_root / "diagnostics.partial.csv"),
        "partial_report": str(output_root / "report.partial.md"),
        "sources": [str(source) for source in sources],
        "source_count": len(sources),
        "loaded_rows": len(rows),
        "unique_rows": len(successes) + len(failures),
        "unique_success_rows": len(successes),
        "unique_non_success_rows": len(failures),
        "status_counts": dict(
            sorted(Counter(str(row.get("status", "unknown")) for row in failures).items())
        ),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-name", required=True)
    parser.add_argument("--platform", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--tester", default="baoming")
    parser.add_argument("--infiniops-commit", default="NA")
    parser.add_argument("--infinirt-commit", default="NA")
    parser.add_argument("--failure-category", default="unknown")
    parser.add_argument("--failure-reason", default="")
    args = parser.parse_args(argv)

    output_root = args.output_root.resolve()
    if not output_root.is_dir():
        raise SystemExit(f"partial output root does not exist: {output_root}")

    sources = sorted(
        path
        for path in output_root.rglob("*")
        if path.is_file() and path.name in {"results.json", "results.jsonl"}
    )
    if not sources:
        raise SystemExit("no complete or checkpoint result files were found")

    rows = _load_rows(sources)
    successes = _dedupe(rows, successful=True)
    failures = _dedupe(rows, successful=False)
    partial_output = output_root / _partial_csv_name(args.output_name)
    partial_diagnostics = output_root / "diagnostics.partial.csv"
    partial_report = output_root / "report.partial.md"

    exporter_exit = export_main(
        [str(source) for source in sources]
        + [
            "--output",
            str(partial_output),
            "--diagnostics",
            str(partial_diagnostics),
            "--report",
            str(partial_report),
            "--platform",
            args.platform,
            "--date",
            args.date,
            "--tester",
            args.tester,
            "--infiniops-commit",
            args.infiniops_commit,
            "--infinirt-commit",
            args.infinirt_commit,
        ]
    )
    if exporter_exit != 0:
        return exporter_exit

    _write_manifest(
        output_root / "partial-manifest.json",
        sources=sources,
        rows=rows,
        successes=successes,
        failures=failures,
        output_root=output_root,
        output_name=args.output_name,
        args=args,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

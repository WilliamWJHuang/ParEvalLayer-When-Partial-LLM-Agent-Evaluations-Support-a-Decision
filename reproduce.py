"""Cross-platform entry point for the ParEvalLayer public artifact.

The default command runs a synthetic stream through the decision adapter. It
does not download data. Benchmark-backed pipelines run only when selected with
``--pipeline`` and require the corresponding compact task tables under
``analysis/public_backtest``.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "outputs"
IGNORED_PATTERNS = (
    ".git/*",
    ".venv/*",
    "__pycache__/*",
    "*/__pycache__/*",
    "*.pyc",
    "outputs/*",
    "analysis/public_backtest/*",
    ".env",
    ".env.*",
    "credentials*",
    "secrets*",
    "*.pem",
    "*.key",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the synthetic ParEvalLayer demo, contract checks, a selected "
            "benchmark pipeline, or manifest verification. No command downloads data."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Run synthetic schema, adapter, and malformed-stream contract checks.",
    )
    mode.add_argument(
        "--pipeline",
        choices=["tau", "terminal", "all"],
        help="Run an explicit benchmark-backed pipeline using packaged compact data.",
    )
    mode.add_argument(
        "--verify",
        action="store_true",
        help="Verify files listed in MANIFEST.sha256.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory for generated outputs (default: outputs/).",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "MANIFEST.sha256",
        help="Manifest used by --verify.",
    )
    parser.add_argument(
        "--strict-files",
        action="store_true",
        help="With --verify, also fail on unexpected files not matched by built-in ignores.",
    )
    return parser.parse_args()


def run(argv: list[str]) -> None:
    subprocess.run(argv, cwd=ROOT, check=True)


def synthetic_demo(output_dir: Path) -> None:
    run(
        [
            sys.executable,
            "scripts/run_stream_decision_adapter.py",
            "--source",
            "generic",
            "--stream-path",
            "examples/synthetic_task_outcome_stream.jsonl",
            "--threshold",
            "0.05",
            "--budget-caps",
            "0.50",
            "1.00",
            "--skip-equivalence-check",
            "--out-dir",
            str(output_dir / "synthetic_demo"),
        ]
    )


def contract_check(output_dir: Path) -> None:
    run(
        [
            sys.executable,
            "scripts/test_stream_adapter_contract.py",
            "--out-dir",
            str(output_dir / "contract_check"),
        ]
    )
    run([sys.executable, "scripts/test_compare.py"])


def benchmark_pipeline(suite: str, output_dir: Path) -> None:
    required = {
        "tau": ROOT / "analysis/public_backtest/tau2_ready/public_tau2_task_rows.csv",
        "terminal": ROOT
        / "analysis/public_backtest/terminal_bench/terminal_bench_task_rows.csv",
    }
    selected = required if suite == "all" else {suite: required[suite]}
    missing = [path for path in selected.values() if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path.relative_to(ROOT)}" for path in missing)
        raise FileNotFoundError(
            "The selected pipeline requires compact public task tables:\n" + formatted
        )
    run(
        [
            sys.executable,
            "scripts/run_runner_replay_pipeline.py",
            "--suite",
            suite,
            "--out-dir",
            str(output_dir / f"pipeline_{suite}"),
        ]
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_manifest(path: Path) -> dict[Path, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")
    entries: dict[Path, str] = {}
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"Malformed manifest line {line_number}: {raw}")
        relative = Path(parts[1].lstrip("*"))
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe manifest path on line {line_number}: {relative}")
        entries[relative] = parts[0].lower()
    if not entries:
        raise ValueError(f"Manifest has no file entries: {path}")
    return entries


def ignored(relative: Path) -> bool:
    posix = relative.as_posix()
    return any(fnmatch.fnmatch(posix, pattern) for pattern in IGNORED_PATTERNS)


def verify_manifest(path: Path, strict_files: bool) -> None:
    entries = parse_manifest(path)
    failures: list[str] = []
    for relative, expected in entries.items():
        candidate = ROOT / relative
        if not candidate.is_file():
            failures.append(f"missing: {relative.as_posix()}")
            continue
        observed = sha256_file(candidate)
        if observed != expected:
            failures.append(
                f"hash mismatch: {relative.as_posix()} expected {expected} got {observed}"
            )
    if strict_files:
        manifest_relative = path.resolve().relative_to(ROOT.resolve())
        allowed = set(entries) | {manifest_relative}
        for candidate in ROOT.rglob("*"):
            if not candidate.is_file():
                continue
            relative = candidate.relative_to(ROOT)
            if relative not in allowed and not ignored(relative):
                failures.append(f"unexpected: {relative.as_posix()}")
    if failures:
        raise SystemExit("Manifest verification failed:\n" + "\n".join(failures))
    suffix = " with strict file inventory" if strict_files else ""
    print(f"Verified {len(entries)} manifest entries{suffix}.")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if args.verify:
        verify_manifest(args.manifest.resolve(), args.strict_files)
    elif args.check:
        contract_check(output_dir)
    elif args.pipeline:
        benchmark_pipeline(args.pipeline, output_dir)
    else:
        synthetic_demo(output_dir)


if __name__ == "__main__":
    main()

# Reproducing the artifact

This repository has two execution levels. The offline path tests the decision contract with synthetic inputs. The public replay path requires source-specific task records and may take substantially longer.

## Environment

Use Python 3.12. On Windows PowerShell, create and activate an isolated environment, then install the pinned core dependencies:

```text
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, use:

```text
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Install optional figure dependencies only when regenerating figures:

```text
python -m pip install -r requirements-figures.txt
```

Run commands from the repository root. The scripts use paths relative to that root.

## Offline smoke path

Run the synthetic example:

```text
python reproduce.py
```

Run the synthetic schema, adapter, and malformed-stream contract checks:

```text
python reproduce.py --check
```

Generated files are written under `outputs/`. Removing that directory does not remove source inputs.

## Verify packaged files

When a release manifest is present, verify its recorded hashes with:

```text
python reproduce.py --verify
```

Hash verification checks byte identity. It does not establish source licensing, scientific validity, or independent execution.

## Compact-data pipelines

Read [DATA.md](DATA.md) before adding benchmark records. The data guide identifies which compact tables may be distributed and which must be rebuilt locally from the original source.

When the required compact inputs are present, run:

```text
python reproduce.py --pipeline tau
python reproduce.py --pipeline terminal
python reproduce.py --pipeline all
```

The tau and Terminal-Bench pipelines rebuild task-outcome streams, decision records, traces, and equivalence reports. They start from completed task outcomes. They do not execute agents or validate a live stopping guarantee.

With both compact benchmark tables present, run the fuller adapter verification separately:

```text
python scripts/verify_stream_adapter.py --suite all --out-dir outputs/adapter_verification
```

This command checks source-specific batch equivalence, prefix isolation under future-row mutation, budget accounting, trace integrity, audit-field confinement, and malformed inputs.

## Main public-record replay

The cross-source policy sweep reads these compact inputs when available:

```text
analysis/public_backtest/swebench_smoke/swebench_task_rows.csv
analysis/public_backtest/swebench_smoke/swebench_pair_overlap_summary.csv
analysis/public_backtest/appworld_feasibility/appworld_task_rows.csv
analysis/public_backtest/osworld_verified/osworld_verified_task_rows.csv
analysis/public_backtest/tau2_ready/public_tau2_task_rows.csv
```

Run a selected source first:

```text
python scripts/run_main_policy_sensitivity.py --sources swe_lite
python scripts/run_main_policy_sensitivity.py --sources swe_verified
python scripts/run_main_policy_sensitivity.py --sources appworld
python scripts/run_main_policy_sensitivity.py --sources osworld
python scripts/run_main_policy_sensitivity.py --sources tau
```

Run all five sources only after every required input is present:

```text
python scripts/run_main_policy_sensitivity.py --sources swe_lite swe_verified appworld tau osworld
python scripts/summarize_main_policy_sensitivity.py
```

The raw sweep can be large. The summary step produces the compact operating curves, minimum-budget table, descriptive pair-level intervals, and supporting diagnostics used by the paper workflow.

## Diagnostics

These commands use compact task tables and, in some cases, outputs from the main sweep:

```text
python scripts/run_paired_test_diagnostics.py
python scripts/run_sensitivity_diagnostics.py
```

Use `python <script> --help` to restrict sources, caps, simulation counts, or output directories. Lower replay counts are useful for a smoke run, but their outputs must not replace the recorded paper analyses.

The sensitivity script expects the summarized main-policy outputs and other recorded diagnostic inputs. It will stop if a prerequisite is missing. This is preferable to silently substituting another result.

## Source-specific builders and replays

The source adapters remain available so that readers can rebuild compact tables under the upstream terms:

| Source | Builder or converter | Replay |
| --- | --- | --- |
| SWE-bench | `scripts/run_swebench_conversion_smoke.py` | `scripts/run_swebench_count_budget_replay.py` |
| AppWorld | `scripts/build_appworld_feasibility.py` | `scripts/run_appworld_count_budget_replay.py` |
| OSWorld-Verified | `scripts/build_osworld_verified_matrix.py` | `scripts/run_main_policy_sensitivity.py --sources osworld` |
| tau-bench | `scripts/build_public_tau2_table.py` | `scripts/run_public_tau2_replay.py` |
| Terminal-Bench | `scripts/build_terminal_bench_matrix.py` | `scripts/run_terminal_bench_resource_replay.py` |

Inspect each command before a full rebuild:

```text
python scripts/build_public_tau2_table.py --help
python scripts/build_appworld_feasibility.py --help
python scripts/build_osworld_verified_matrix.py --help
python scripts/build_terminal_bench_matrix.py --help
```

These builders may require external repositories, released bundles, network access, or benchmark-specific tools. Acquisition is outside the offline smoke path. AppWorld records must be handled under its output-license terms and are not released here as plain derived task rows.

## Terminal-Bench resource replay

With `analysis/public_backtest/terminal_bench/terminal_bench_task_rows.csv` present, run:

```text
python scripts/run_terminal_bench_resource_replay.py
```

This analysis uses completed-record task counts and wall-clock fields. Resource fractions are retrospective unless the denominator is fixed before a live run. Sparse token or cost fields should not be promoted into a cross-system cost claim.

## Figures

Figure scripts are optional:

```text
python scripts/make_system_figure.py
python scripts/make_paper_figures.py
```

Figure generation requires the corresponding derived CSV files. A successful plot command shows that the available inputs are structurally usable. It does not reproduce a paper result unless the input hashes and expected values also match the recorded targets.

## Interpreting a successful run

A clean smoke run establishes that the packaged interface and checks execute in the current environment. A completed public replay establishes that the available compact records can be processed by the recorded code. Neither outcome proves that a live evaluation can stop safely at the same budget. See [METHOD.md](METHOD.md) for the inference boundary.

# ParEvalLayer

ParEvalLayer compares scored results from two agents on the same tasks. Give it a CSV of paired scores and an improvement threshold. It produces a readable report and structured decision records: `promote`, `reject`, or `abstain` under the specified policy.

Use your existing benchmark or evaluation harness to run and score the agents. ParEvalLayer reads those results; it can also replay selected prefixes of the file to examine decisions under partial evaluation budgets.

This repository accompanies *ParEvalLayer: When Partial LLM-Agent Evaluations Support a Decision* by Wei-Jung Huang and Bonan Shen. It contains the decision-layer code, contract checks, source adapters, replay analyses, and small synthetic examples. The paper and PDF are not included here.

## Quick start

Clone the repository, or download and extract its ZIP:

```text
git clone https://github.com/WilliamWJHuang/ParEvalLayer-When-Partial-LLM-Agent-Evaluations-Support-a-Decision.git ParEvalLayer
cd ParEvalLayer
```

Use Python 3.12. On Windows PowerShell, create and activate an isolated environment, then install the pinned core dependencies:

```text
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, use `python3.12 -m venv .venv` followed by `. .venv/bin/activate`.

Run the paired-score example:

```text
python compare.py --input examples/paired_results.csv --tested candidate --comparison baseline --threshold 0.05
```

Read `outputs/comparison/report.md` for the comparison. The same directory contains `summary.csv`, decision records, the normalized task stream, and the run configuration. The example uses synthetic scores and requires no downloads or API keys.

## Compare your own agents

Export one row per task, with both systems scored by the same procedure on the same scale. Higher scores must mean better performance.

```csv
task_id,tested_score,comparison_score
task_001,0.8,0.6
task_002,0.7,0.5
task_003,0.9,0.7
```

Then run:

```text
python compare.py --input my_results.csv --tested my-agent --comparison baseline --threshold 0.05 --output-dir outputs/my-comparison
```

`--threshold` is the required improvement in score units. For scores in `[0, 1]`, `0.05` means five percentage points. Scores are not rescaled. The optional `group` column identifies task categories for the coverage check.

| Output | Meaning under the specified policy |
| --- | --- |
| `promote` | The observed mean improvement exceeds the threshold and passes the bootstrap tail check. |
| `reject` | The observed mean improvement is at or below the threshold and passes the opposite tail check. This does not necessarily mean the candidate is worse. |
| `abstain` | Required task groups are insufficiently represented, or the tail check does not resolve the comparison. |

The default uses all rows. To replay partial budgets, specify `--budget-caps 0.5 1.0`. Rows are processed in file order, and each cap uses `ceil(cap * number_of_rows)` paired tasks. Fix the order and policy before inspecting outcomes if using the result to support a decision. Checks at several caps are unadjusted and do not provide an anytime-valid stopping rule.

See [the own-results guide](docs/USING_YOUR_RESULTS.md) for data preparation, output fields, limitations, and common errors. Existing output directories must be empty so that one run cannot silently replace another.

## Check the installation

```text
python reproduce.py --check
```

This runs the CSV workflow tests and the stream contract checks. The original JSONL example remains available through `python reproduce.py`.

The synthetic checks and Terminal-Bench interface checks passed in a fresh local environment. See [docs/VALIDATION.md](docs/VALIDATION.md) for the tested paths and reproduction limits.

## What the evidence supports

The reported study replays completed public benchmark records. It hides later task outcomes, applies a fixed-cap policy to the visible prefix, and checks terminal decisions against the decision from the completed shared record. These analyses describe the selected released records and simulated reveal orders.

The current evidence does not establish an anytime-valid live stopping rule, a population guarantee for future agent systems, or a universal budget reduction. The nominal bootstrap tail check is evaluated at fixed caps. Repeated inspection during a live run would require a time-uniform method and a protocol fixed before outcome access. See [docs/METHOD.md](docs/METHOD.md) for the estimand and statistical boundaries.

## Public replay data

The repository includes compact OSWorld-Verified and Terminal-Bench tables with source and license notices. AppWorld, SWE-bench experiment records, and tau-bench trajectories must be obtained separately. Their adapters and replay code are included. Follow [docs/DATA.md](docs/DATA.md) for the acquisition, redistribution, and derivation details.

Verify packaged hashes and run the included Terminal-Bench pipeline:

```text
python reproduce.py --verify
python reproduce.py --pipeline terminal
```

The `--pipeline tau` and `--pipeline all` options additionally require the separately acquired tau-bench inputs.

Availability of an input does not imply that every paper result can be regenerated from the repository alone. [docs/REPRODUCING.md](docs/REPRODUCING.md) separates the offline smoke path from source acquisition and full replay commands.

## Repository map

- `reproduce.py` runs the synthetic example, checks, hash verification, and supported compact-data pipelines.
- `compare.py` compares your own paired scores from a CSV file.
- `scripts/` contains the decision layer, stream adapter, source builders, replay analyses, diagnostics, and figure code.
- `examples/` contains synthetic paired scores and a task-outcome stream.
- `analysis/public_backtest/` contains compact inputs or derived outputs whose release conditions have been checked.
- `docs/METHOD.md` defines the method, estimand, dependence, controls, and limitations.
- `docs/REPRODUCING.md` gives practical execution paths.
- `docs/DATA.md` records data provenance and redistribution status.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff), including the paper title and authors.

## License

Project code and documentation are released under the [MIT License](LICENSE). Third-party data and notices retain their original terms, including the OSWorld dataset's MIT declaration and the Terminal-Bench leaderboard dataset's Apache 2.0 declaration. See [docs/DATA.md](docs/DATA.md) for provenance and redistribution details.

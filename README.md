# ParEvalLayer

ParEvalLayer turns partial task outcomes from a paired LLM-agent evaluation into an auditable decision record. A policy fixed before a budget cap specifies the required improvement, task order, coverage rule, uncertainty check, and unresolved-comparison target. For each cap, the generated trace records `continue` at earlier prefixes and `promote`, `reject`, or `abstain` at the selected prefix.

This repository accompanies *ParEvalLayer: When Partial LLM-Agent Evaluations Support a Decision* by Wei-Jung Huang and Bonan Shen. It contains the decision-layer code, contract checks, source adapters, replay analyses, and small synthetic examples. The paper and PDF are not included here.

## What the evidence supports

The reported study replays completed public benchmark records. It hides later task outcomes, applies a fixed-cap policy to the visible prefix, and checks terminal decisions against the decision from the completed shared record. These analyses describe the selected released records and simulated reveal orders.

The current evidence does not establish an anytime-valid live stopping rule, a population guarantee for future agent systems, or a universal budget reduction. The nominal bootstrap tail check is evaluated at fixed caps. Repeated inspection during a live run would require a time-uniform method and a protocol fixed before outcome access. See [docs/METHOD.md](docs/METHOD.md) for the estimand and statistical boundaries.

## Quick start

Use Python 3.12. On Windows PowerShell, create and activate an isolated environment, then install the pinned core dependencies:

```text
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS or Linux, use `python3.12 -m venv .venv` followed by `. .venv/bin/activate`.

Run the synthetic example:

```text
python reproduce.py
```

Run the contract checks:

```text
python reproduce.py --check
```

The synthetic path requires no benchmark download. It writes generated files under `outputs/`.

The synthetic checks and Terminal-Bench interface checks passed in a fresh local environment. See [docs/VALIDATION.md](docs/VALIDATION.md) for the tested paths and reproduction limits.

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
- `scripts/` contains the decision layer, stream adapter, source builders, replay analyses, diagnostics, and figure code.
- `examples/` contains a synthetic task-outcome stream.
- `analysis/public_backtest/` contains compact inputs or derived outputs whose release conditions have been checked.
- `docs/METHOD.md` defines the method, estimand, dependence, controls, and limitations.
- `docs/REPRODUCING.md` gives practical execution paths.
- `docs/DATA.md` records data provenance and redistribution status.

## Citation

Citation metadata are provided in [CITATION.cff](CITATION.cff), including the paper title and authors.

## License

The [original MIT notice](licenses/Legacy-code-MIT.txt) is retained for code originating in the earlier reproducibility package. A project-wide code license for this standalone release has not yet been selected. Data and third-party materials retain their own terms, as described in [docs/DATA.md](docs/DATA.md).

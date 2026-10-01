# Data sources and redistribution

The release separates code from benchmark records. The synthetic stream in `examples/` was constructed for the contract test and contains no benchmark data. The real-data files listed in [data_manifest.json](data_manifest.json) are copied without numerical changes from the project analyses. Each entry records a SHA-256 checksum and the applicable source notice.

The builders convert released outcomes into paired task tables. They do not execute agents. Raw trajectories, screenshots, prompts, downloaded repositories, and unpacked benchmark bundles are not included.

## Included data

### OSWorld-Verified

`analysis/public_backtest/osworld_verified/` contains the compact task-outcome table and the archive and pair audits. The [trajectory dataset card](https://huggingface.co/datasets/xlangai/ubuntu_osworld_verified_trajs/blob/19510c66fd59669c61347452d5308f5fa6f68de3/README.md) declares MIT for `xlangai/ubuntu_osworld_verified_trajs`. Its original text and requested paper citation are preserved in `licenses/OSWorld-dataset-card.md`, with the MIT terms in `licenses/OSWorld-MIT.txt`.

The dataset supplies no separate copyright notice. None has been inferred or assigned here. This notice concerns the trajectory dataset; the OSWorld code repository has a separate license. The bundled CSVs are transformed outcome records and extraction audits produced by `build_osworld_verified_matrix.py`. No trajectory media or upstream code is bundled.

The compact table can be used directly with `run_main_policy_sensitivity.py --sources osworld`. Rebuilding it from upstream requires the Hugging Face file tree and HTTP access to the source archives. The builder reads result maps without needing a complete local copy of the trajectory media.

### Terminal-Bench 2.0

`analysis/public_backtest/terminal_bench/` contains the compact task table and its extraction audits. `analysis/public_backtest/terminal_bench_resource_replay/` contains the recorded aggregate resource results, minimum-budget table, and replay configuration audit. These are existing analysis outputs, not results from a new experimental run during packaging.

The [leaderboard dataset card](https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard/blob/572b2614be2c0cb2527e14f5b1e4026f1072e6c1/README.md) specifies Apache 2.0. The publisher is Harbor Framework. A copy of the card and the full license are stored in `licenses/`. The tables were transformed from the released result files by `build_terminal_bench_matrix.py`; the replay aggregates were produced by `run_terminal_bench_resource_replay.py`.

This subset is used for the paper's wall-clock analysis. Its task-name prefixes are not native benchmark groups. Token and monetary-cost fields are incomplete. The resource replay treats the subset separately from the main analyses with benchmark-provided groups.

## Inputs obtained separately

| Source | Release decision | Source and local destination |
| --- | --- | --- |
| AppWorld | Excluded, including derived tables. The protected-output license restricts public redistribution of protected content and derivatives to encrypted form. | [Encrypted leaderboard bundles](https://github.com/StonyBrookNLP/appworld-leaderboard/tree/8da928a6840d53e83e7cb328590f3d7727fbf2f4). Local builder expects `analysis/public_backtest/appworld_leaderboard_repo/`. |
| SWE-bench experiment records | Excluded. No license for the separate experiment submissions was verified. The harness license alone does not settle the rights to these records. | [Experiment repository snapshot](https://github.com/SWE-bench/experiments/tree/2f15350cd32becc4569e0d826361048555b605c0). Local converter expects `analysis/public_backtest/swebench_experiments_repo/`. |
| tau-bench trajectories | Excluded. The repository's MIT license was not verified to cover the externally hosted trajectory files used by the analysis. | [Submission metadata and source links](https://github.com/sierra-research/tau2-bench/tree/main/web/leaderboard/public/submissions). Local builder expects `analysis/public_backtest/raw_tau2/`. |

For AppWorld, follow the upstream [download and sharing instructions](https://github.com/StonyBrookNLP/appworld#readme), including the license inside each output bundle. Local parsing uses `build_appworld_feasibility.py`; do not add the resulting plaintext tables to this repository. The original conversion used `appworld==0.1.3.post1` to unpack the bundles. AppWorld is an optional acquisition dependency, separate from the core environment.

For SWE-bench, `run_swebench_conversion_smoke.py` reads the released `evaluation/lite` and `evaluation/verified` result metadata. After obtaining the records under appropriate terms, run it from the repository root to build `swebench_smoke/swebench_task_rows.csv` and `swebench_pair_overlap_summary.csv`.

For tau-bench, `build_public_tau2_table.py` reads local submission metadata and trajectories. A missing input raises an error unless `--download` is explicitly supplied. Its defaults select four submission directories dated 2026-02-26 and the airline, retail, and telecom domains. Inspect the source links and terms before choosing to download them. The primary analysis uses failure imputation; neutral and drop variants are separate sensitivity inputs.

## Rebuilding and provenance

The acquisition code includes upstream URLs that use `main`. Upstream records may change. A new download is not necessarily the input used in the paper. The checksums in `data_manifest.json` identify the exact compact files in this release; a dataset-card commit identifies the license evidence, not a claim that every original result was fetched at that commit.

See [REPRODUCING.md](REPRODUCING.md) for execution commands. Rebuilding all paper figures requires additional inputs and intermediate tables that are not shipped here. The default synthetic example and the included-data paths remain usable without those files.

The repository ignores new files under `analysis/public_backtest/` by default and permits only the reviewed files named in `.gitignore`. Keep downloaded or locally regenerated records out of commits unless their redistribution terms have been checked separately.

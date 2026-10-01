# Changes made for this release

The release keeps the analysis scripts in their original `scripts/` layout so that their imports and relative input paths continue to work. The six decision-layer modules retain the original decision calculations. Source-specific replay and diagnostic scripts are included separately from the small runnable example.

The packaging changes are limited to entry points, output locations, and documentation:

- `reproduce.py` runs the synthetic example, contract checks, selected compact-data pipelines, and checksum verification. Child-process failures stop the command.
- Figure scripts write to `outputs/figures/` instead of folders belonging to the earlier paper package. Their plotting calculations and selection rules are retained.
- The tau-bench builder requires `--download` before fetching a missing file. Previously, a missing file could trigger a download even when that option was absent.
- Stale references to an earlier paper and private analyses were removed from generated report text.

The source figures retain their original selection. In particular, the legacy multi-source figure code is not a standalone regeneration of every ParEvalLayer paper figure. It requires the corresponding external inputs and recorded intermediate tables. The release does not substitute synthetic outputs for empirical results.

Private experiments, unrelated papers, presentation drafts, downloaded repositories, virtual environments, large raw traces, and internal revision notes are outside this release. No experiment result was recomputed or altered to improve the reported findings during cleanup.

# How to compare two LLM agents using paired evaluation scores

This tutorial starts with scores that an evaluation harness has already produced. You will compare a candidate with a baseline, inspect a partial task budget, and read the resulting decision report. ParEvalLayer does not execute the agents or score their responses.

The example uses eight synthetic task pairs. Its purpose is to demonstrate the interface, not to estimate real agent performance.

## 1. Install and inspect the input

Follow the [README installation instructions](../README.md#quick-start), then run the commands below from the repository root. The example requires no API keys or benchmark downloads.

Open [examples/paired_results.csv](../examples/paired_results.csv). Its first four rows are:

```csv
task_id,tested_score,comparison_score,group
illustrative_01,0.82,0.64,domain_a
illustrative_02,0.74,0.61,domain_b
illustrative_03,0.91,0.70,domain_a
illustrative_04,0.68,0.57,domain_b
```

Each row pairs the two systems on one task. `tested_score` is the candidate's score, `comparison_score` is the baseline's, and `group` supplies a task category for the coverage check. Both score columns must use the same scale, with higher values indicating better outcomes.

## 2. Compare half the tasks and the complete file

```text
python compare.py --input examples/paired_results.csv --tested candidate --comparison baseline --threshold 0.05 --budget-caps 0.5 1.0 --output-dir outputs/tutorial
```

The threshold specifies a required improvement of `0.05` score units. The two caps select the first four and all eight paired tasks, respectively. The program preserves CSV order. It does not choose the rows with the largest score differences.

The console output contains:

```text
cap  n  mean_difference  action   reason
0.5  4  0.1575           promote  supported_above_threshold
1    8  0.17             promote  supported_above_threshold
```

The actual console separates fields with tabs. These values were generated from the bundled CSV with the command above, using the default `alpha=0.05`, 500 bootstrap samples, and seed `20260613`.

For the first cap, the four differences are `0.18`, `0.13`, `0.21`, and `0.11`, giving a mean of `0.1575`. Both task groups are represented. The mean exceeds `0.05` and passes the policy's bootstrap tail check, so the action is `promote`. The complete file gives a mean difference of `0.17` and the same action.

These favorable synthetic differences make the example easy to inspect. The actions are not evidence that the method will resolve a real comparison this way or that four tasks are sufficient for a scientific conclusion.

## 3. Inspect the report and machine-readable records

Open `outputs/tutorial/report.md`. It explains the decision and lists the selected caps. Use `summary.csv` for a compact table, `decisions.jsonl` for structured decisions, and `run.json` to recover the input hash, settings, and dependency versions. `stream.jsonl` contains the normalized events, and `trace.jsonl` records the prefix traces.

The three terminal actions refer to the chosen improvement threshold:

- `promote`: the mean improvement is above the threshold and passes the corresponding tail check.
- `reject`: the mean improvement is at or below the threshold and passes the opposite check. A small positive improvement can still receive this action.
- `abstain`: task-group coverage is insufficient, or the bootstrap check leaves the comparison unresolved. Abstention does not establish equivalence.

Each cap is a separate, fixed retrospective comparison. Bootstrap tail fractions are policy checks, not p-values or confidence intervals. Multiple caps are unadjusted, and the tool does not provide an anytime-valid rule for stopping a running evaluator. See [METHOD.md](METHOD.md) for the statistical assumptions and scope.

## 4. Replace the example with your own results

Export one row per task with `task_id`, `tested_score`, and `comparison_score`. Include `group` if your task categories should be checked for coverage. Then run:

```text
python compare.py --input my_results.csv --tested my-agent --comparison baseline --threshold 0.05 --output-dir outputs/my-comparison
```

Choose the threshold in your own score units before inspecting the comparison. The command above uses the full file because no partial caps are specified. Fix any reveal order and partial cap in advance if you intend to use a prefix for a decision.

Use one prespecified paired result per task. Define how to handle missing runs and repeated trials before creating the CSV; do not select tasks based on which system won. The bootstrap resamples task differences individually and does not model clustering among related tasks. The [own-results guide](USING_YOUR_RESULTS.md) gives the full input contract, dependence caveats, and error explanations.

Choose a new output directory for a second run. The tool refuses to overwrite a nonempty directory so that the previous report remains available for inspection.

# Comparing your own agents

ParEvalLayer accepts task scores that your evaluation harness has already produced. It does not execute agents, define benchmark tasks, or grade responses. You need a candidate system, a comparison system, and results for both on the same tasks.

For a complete example with expected output, start with [How to compare two LLM agents using paired evaluation scores](TUTORIAL.md).

## Prepare paired scores

Save a UTF-8 CSV with these columns:

| Column | Required | Content |
| --- | --- | --- |
| `task_id` | Yes | A unique, nonempty task identifier. |
| `tested_score` | Yes | The candidate's finite numeric score. |
| `comparison_score` | Yes | The comparison system's finite numeric score. |
| `group` | No | A nonempty task category. If the column is omitted, every task belongs to `all`. |

```csv
task_id,group,tested_score,comparison_score
web_001,web,0.8,0.6
code_001,code,0.7,0.5
web_002,web,0.9,0.7
code_002,code,0.8,0.6
```

These values illustrate the format only. The small example is not evidence about real agent performance.

Use the same scorer and score scale for both systems, with higher scores indicating better outcomes. Binary success scores and numeric rubric scores can be represented. A threshold of `0.05` has a different meaning on a 0 to 1 scale than on a 0 to 100 scale. The importer neither normalizes scores nor checks that the scoring procedure is comparable.

Each row must contain both scores. Resolve missing runs according to a rule fixed before inspecting the comparison; do not silently drop difficult tasks or replace missing scores with zero. If tasks have repeated trials, define the task-level aggregation before preparing this file. The current bootstrap resamples task differences individually and does not model dependence among repeated trials or related tasks.

At least two paired tasks are required by the CSV interface. That is an input check, not evidence that two tasks suffice for a reliable scientific conclusion. Small samples and identical differences can yield degenerate bootstrap distributions.

## Run a comparison

After installing the dependencies in the README, run from the repository directory:

```text
python compare.py --input my_results.csv --tested my-agent-v2 --comparison my-agent-v1 --threshold 0.05 --output-dir outputs/my-comparison
```

The system names must be distinct. They label the scores you supply; the program cannot verify which agent produced them. The threshold is a nonnegative required improvement in the original score units. The reported difference is the mean of `tested_score - comparison_score` across the selected tasks, with equal weight per task.

The default cap is `1.0`, which uses all rows. To examine the first half and then the complete file:

```text
python compare.py --input my_results.csv --tested my-agent-v2 --comparison my-agent-v1 --threshold 0.05 --budget-caps 0.5 1.0 --output-dir outputs/prefix-comparison
```

The importer preserves file order. It does not sort by score, shuffle, or balance groups. A task cap selects `ceil(cap * N)` rows, where `N` is the number of paired tasks in the CSV. For each group with `N_g` tasks, a prefix of `k` tasks requires at least `max(1, floor(k * N_g / N))` observations from that group. Missing coverage leads to `abstain`. Define categories from task metadata, not from observed outcomes.

The complete file supplies the task count and group counts. Scores after a selected prefix are excluded from that prefix's decision. This is retrospective replay of a completed paired file. It is not a service that interrupts a running evaluator when a decision appears.

Optional controls include `--alpha` (default `0.05`), `--bootstrap-samples` (default `500`), and `--seed` (default `20260613`). Use `python compare.py --help` for all options. A fixed seed makes the calculation repeatable; it does not remove Monte Carlo error or correct an unsuitable statistical model.

## Read the outputs

| File | Contents |
| --- | --- |
| `report.md` | Human-readable comparison, settings, and interpretation. |
| `summary.csv` | One summary row per requested cap. |
| `decisions.jsonl` | Structured terminal decision records. |
| `trace.jsonl` | Prefix traces for the requested caps. Earlier prefixes are marked `continue`. |
| `stream.jsonl` | The paired CSV converted to the task-outcome stream format. |
| `run.json` | Input hash, run settings, and dependency versions. |

`promote` requires a mean difference strictly above the threshold and a bootstrap fraction at or below the threshold no larger than `alpha`. `reject` requires a mean difference at or below the threshold and the opposite bootstrap fraction no larger than `alpha`. Otherwise the result is `abstain`; insufficient group coverage also causes abstention.

The bootstrap fractions are policy checks, not reported p-values or calibrated confidence levels. A `reject` result can occur for a candidate that improves on the baseline but falls short of the chosen threshold. An `abstain` result does not establish equivalence.

Fix the threshold, task order, coverage categories, and selected cap before viewing outcomes when using the policy for a decision. Trying several caps or settings and retaining the most favorable result introduces selection. No multiple-look adjustment is applied. The output does not certify a live stopping rule, future-agent performance, or generalization to a new task distribution. See [METHOD.md](METHOD.md) for the method's scope.

## Common errors

| Error | What to check |
| --- | --- |
| Missing or unknown column | Use the column names listed above; remove dataframe index columns. |
| Duplicate task | Keep one prespecified paired result per task. |
| Missing or nonfinite score | Resolve the source result; `NaN`, infinity, and blank scores are rejected. |
| Blank group | Fill each group cell, or omit the column if no grouping is intended. |
| Nonempty output directory | Choose a new `--output-dir`; prior outputs are preserved. |

Quote paths or system labels that contain spaces. Input and output paths may be absolute. To call the tool from another working directory, use an absolute path to `compare.py` as well.

For an existing event-producing evaluator, the lower-level JSONL adapter remains in `scripts/run_stream_decision_adapter.py`. Its manifest declares the run and system identities. Events must agree with those identities; a supplied event source must match the manifest source.

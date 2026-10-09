# Use your own results

You need two scores for each task: one from the agent you want to test and one from the agent you want to compare against. Run and score both agents with your existing tools, then give those scores to ParEvalLayer.

For a worked example first, see [Compare two LLM agents from a CSV file](TUTORIAL.md).

## Prepare the CSV

Save the file as UTF-8 CSV with these columns:

| Column | Required | What to put here |
| --- | --- | --- |
| `task_id` | Yes | A different, nonempty name or ID for each task. |
| `tested_score` | Yes | The score from the agent you want to test. |
| `comparison_score` | Yes | The other agent's score on the same task. |
| `group` | No | A task category. If you leave out the column, all tasks belong to one group called `all`. If you include it, fill in every row. |

For example:

```csv
task_id,group,tested_score,comparison_score
web_001,web,0.8,0.6
code_001,code,0.7,0.5
web_002,web,0.9,0.7
code_002,code,0.8,0.6
```

These are made-up scores to show the format. For your own results, score both agents in the same way. Higher numbers must mean better performance. You can use pass/fail scores such as `0` and `1`, or another numeric scale. Blank scores, `NaN`, and infinity are rejected.

ParEvalLayer uses the scores as written. An improvement of `0.05` means something different on a 0 to 1 scale than on a 0 to 100 scale. The program cannot check whether your scoring methods are comparable.

Keep one row per task, with both scores present. Decide how to handle missing runs before looking at which agent won. Do not quietly remove difficult tasks or fill missing scores with zero. If you ran each task several times, decide how to combine those attempts into a task score before creating the CSV. The calculation does not account for links between repeated attempts or closely related tasks.

The file must contain at least two tasks. That is a requirement for running the program, not a promise that two tasks support a reliable conclusion. With very few tasks, the check has little information to work with. If every task has the same score difference, sampling those differences again will show no variation.

## Run the comparison

After following the [installation instructions](../README.md#quick-start), run this from the repository folder:

```text
python compare.py --input my_results.csv --tested my-agent-v2 --comparison my-agent-v1 --threshold 0.05 --output-dir outputs/my-comparison
```

Use two different names for the agents. Those names label the score columns; the program cannot verify which agent produced the scores.

`--threshold` is the improvement you want to assess, in the original score units. It must be zero or greater. The report subtracts the comparison agent's score from the tested agent's score on each task, then takes the average. Every task has equal weight.

## Compare part of the file

The default `--budget-caps 1.0` uses all rows. To compare the first half and then the full file, run:

```text
python compare.py --input my_results.csv --tested my-agent-v2 --comparison my-agent-v1 --threshold 0.05 --budget-caps 0.5 1.0 --output-dir outputs/prefix-comparison
```

Rows stay in the order you saved them. The program does not sort by score, shuffle tasks, or move categories around. It multiplies the requested fraction by the total number of rows and rounds up to a whole task, with at least one task used.

If you supply a `group` column, every category must be represented before the tool can decide. The required counts depend on the size of each category and how many rows you use. If a category is short of its required count, the answer is `abstain`. Choose categories from the tasks' descriptions or other known properties, not from their scores. The [exact counting rules](METHOD.md#csv-counting-and-decision-rules) are in the method notes.

The full file tells the tool how many tasks and groups exist. For a comparison using only the first part, later scores do not affect that answer. This lets you examine completed results at different task counts; it does not control a running evaluation.

Choose the row order, fraction of tasks, and improvement threshold before viewing scores if you plan to use the answer for a decision. Trying many settings and keeping a favorable answer can change how often you make a wrong decision. These checks do not correct for the extra attempts, establish when it is safe to stop a live run, or guarantee the same result on future tasks or agents.

## Read the output files

Start with `report.md`. It explains the result in words. The other files are useful if you want to check the run or use its output in your own code:

| File | What it contains |
| --- | --- |
| `summary.csv` | A short table with one row for each requested fraction of tasks. |
| `decisions.jsonl` | Detailed results, stored as one JSON record per line. |
| `trace.jsonl` | The checks made as more tasks are included. Earlier steps say `continue` until the requested fraction is reached. |
| `stream.jsonl` | Your CSV converted into task records for the comparison code. |
| `run.json` | The input file's identifying hash, the settings, and package versions. |

An average improvement above the threshold can produce `promote`; one at or below it can produce `reject`. Both answers also have to pass the built-in check. Otherwise the result is `abstain`. Missing tasks from a required group also produce `abstain`.

`reject` does not necessarily mean the tested agent is worse. It can mean an improvement is smaller than you required. `abstain` does not prove the agents are equal. The check's numbers are not a probability that one agent is better or a guaranteed range for the true improvement. The [method notes](METHOD.md) explain exactly how those numbers are used.

## Other settings

Use `python compare.py --help` to see all options. These settings control the additional check:

| Option | Default | Purpose |
| --- | --- | --- |
| `--alpha` | `0.05` | Sets the cutoff used by the check. |
| `--bootstrap-samples` | `500` | Sets how many times to draw a sample of the task differences. |
| `--seed` | `20260613` | Makes those random draws repeatable for the same inputs and settings. |

The [method notes](METHOD.md#csv-counting-and-decision-rules) explain these settings in more detail. A fixed seed helps repeat a run; it does not remove the variation caused by random sampling or make an unsuitable calculation valid.

## Common errors

| Error | What to check |
| --- | --- |
| Missing or unknown column | Use the column names listed above. Remove any extra index column added by your export tool. |
| Duplicate task | Keep one row per task, using a rule chosen before you inspect the results. |
| Missing or nonfinite score | Check the original result. Blank scores, `NaN`, and infinity are rejected. |
| Blank group | Fill in every group cell, or leave out the whole column. |
| Nonempty output directory | Choose a new `--output-dir` to preserve earlier reports. |

Put quotes around paths or agent names containing spaces. You can use full file paths for both input and output. To run the command from another folder, use the full path to `compare.py` too.

If your evaluator already writes one task result at a time, see `scripts/run_stream_decision_adapter.py` for the JSONL interface. Its first record identifies the run and the two systems. Later records must agree with those names; if they include a source name, it must match too.

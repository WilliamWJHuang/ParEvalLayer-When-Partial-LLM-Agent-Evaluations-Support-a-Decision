# Compare two LLM agents from a CSV file

This example shows what to put in the file, which command to run, and how to read the answer. It uses eight tasks with made-up scores. You will compare a new agent, called `candidate`, with another agent, called `baseline`.

The scores are already in the file. ParEvalLayer will read them; it will not run the agents or grade their work.

## 1. Open the example file

Follow the [installation instructions](../README.md#quick-start), then run the commands below from the repository folder. You will not need API keys or extra data downloads.

Open [examples/paired_results.csv](../examples/paired_results.csv). The first four rows look like this:

```csv
task_id,tested_score,comparison_score,group
illustrative_01,0.82,0.64,domain_a
illustrative_02,0.74,0.61,domain_b
illustrative_03,0.91,0.70,domain_a
illustrative_04,0.68,0.57,domain_b
```

Each row has the two agents' scores for the same task. `tested_score` belongs to the candidate, and `comparison_score` belongs to the baseline. The `group` column names the task category. Both score columns use the same scale, with higher scores meaning better performance.

## 2. Run the comparison

```text
python compare.py --input examples/paired_results.csv --tested candidate --comparison baseline --threshold 0.05 --budget-caps 0.5 1.0 --output-dir outputs/tutorial
```

This asks two questions: what does the comparison say after the first four tasks, and what does it say after all eight? `--budget-caps 0.5 1.0` sets those amounts. `--threshold 0.05` asks whether the candidate's average improvement is greater than `0.05` score units.

Tasks stay in file order. The tool does not pick the rows where the candidate did best.

You should see:

```text
cap  n  mean_difference  action   reason
0.5  4  0.1575           promote  supported_above_threshold
1    8  0.17             promote  supported_above_threshold
```

The actual output separates the columns with tabs. `cap` is the requested fraction of tasks, `n` is the number used, and `mean_difference` is the average of candidate score minus baseline score.

For the first four tasks, the differences are `0.18`, `0.13`, `0.21`, and `0.11`. Their average is `0.1575`, which is above `0.05`. Both task categories are included, and the tool's additional check supports the result, so the answer is `promote`. Using all eight tasks gives an average difference of `0.17` and the same answer.

These numbers were produced by the command above with the default settings: `alpha=0.05`, 500 samples for the check, and seed `20260613`. The made-up scores make the example easy to follow. They do not show how a real agent will perform or how many tasks a real comparison needs.

## 3. Read the report

Open `outputs/tutorial/report.md`. Start with the results table, then read the explanation below it. The result labels mean:

- `promote`: the average improvement is above your threshold, and the check supports that answer.
- `reject`: the average improvement is at or below your threshold, and the check supports that answer. A small positive improvement can still get this label.
- `abstain`: the check cannot settle the comparison, or too few tasks from a required category have been used. This does not mean the two agents are equally good.

The check samples the task differences again to see how much their average varies. Its numbers are not the probability that one agent is better or a guaranteed range for the true improvement. [How the checks work](METHOD.md) explains the calculation.

If you want to use the output in another program, `summary.csv` holds the short table and `decisions.jsonl` holds the detailed results. `run.json` records the input file's identifying hash, settings, and package versions. `stream.jsonl` and `trace.jsonl` let you inspect the converted task records and the checks made as more rows are included.

## 4. Try your own scores

Create a CSV with `task_id`, `tested_score`, and `comparison_score`. Add `group` if you want the comparison to require tasks from each category. Then run:

```text
python compare.py --input my_results.csv --tested my-agent --comparison baseline --threshold 0.05 --output-dir outputs/my-comparison
```

This command uses the full file. Choose an improvement threshold that makes sense on your score scale before looking at the answer. If you plan to use only part of the file, decide the task order and number of tasks in advance too.

The tool reads completed results. It does not tell you when it is safe to stop an evaluation that is still running. Trying several amounts of data or settings and keeping the answer you like can be misleading; the checks do not correct for those extra attempts.

Use one result from each agent for each task. Decide how to handle missing runs and repeated attempts before preparing the file. Closely related tasks or repeated attempts on the same task need care because the check treats task differences as separate observations. See [Use your own results](USING_YOUR_RESULTS.md) for the full input requirements and examples of errors.

For another run, choose a new output directory. ParEvalLayer refuses to overwrite a nonempty directory, so your earlier report stays available.

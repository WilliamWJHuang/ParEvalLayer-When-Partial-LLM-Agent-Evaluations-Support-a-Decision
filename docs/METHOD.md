# Method and evidence boundary

## Decision target

ParEvalLayer compares a tested system, `A`, with a comparison system, `B`, on tasks for which both systems have usable outcomes. For task `i`, the paired difference is

```text
D_i = Y_i,A - Y_i,B
```

In the benchmark analyses, each normalized outcome lies in `[0, 1]`. The generic stream validator accepts any finite numeric outcome, so a caller using another scale must choose a threshold in that same scale. The completed-record margin is the mean paired difference over the shared usable tasks. For a threshold `delta`, the completed-record decision is `promote` when that margin exceeds `delta`, and `reject` otherwise.

At a fixed budget cap, the policy uses only the paired outcomes visible in the current prefix. Before the cap it emits `continue`. At the cap it emits:

- `promote` when the observed margin is above the threshold and the promote-side bootstrap tail is at most the configured cutoff;
- `reject` when the observed margin is at or below the threshold and the reject-side bootstrap tail is at most the configured cutoff;
- `abstain` when coverage or uncertainty is insufficient.

Completed-record fields are used after the decision for replay auditing. They must not enter the prefix decision.

## Inputs and outputs

The stream adapter reads a manifest followed by task-outcome events. The manifest states the source, run, tested and comparison systems, budget measure and denominator, bootstrap seed, task and group counts, coverage mode, and ordering. Each task event identifies its index, run, role, system, task, group, outcome, and resource object. Cost and execution-time values inside the resource object are optional. A pair becomes available only after both systems have a usable outcome for the same task.

For each configured cap, the adapter writes a terminal decision record and a prefix trace. The record includes the policy settings, observed task count, coverage status, paired margin, action, unresolved reason when applicable, and event-stream metadata. Source-specific modes can also compare the stream result with the batch decision driver.

## Coverage rule

The primary replay policy requires visible tasks from every benchmark-provided group in the shared pair overlap. The required count for each group grows with the current task budget. Repository, difficulty, application, and domain labels are used where the source provides them. A final decision that fails the configured coverage rule is counted as a coverage failure.

These labels are coarse metadata. Passing the rule does not establish behavioral coverage, distributional representativeness, or robustness to a different grouping.

## Estimands and units

The main empirical target is agreement with a completed released record under a fixed policy. For each threshold and budget, the analyses report:

- the false-promotion rate among cases classified as negative by the completed record;
- the false-rejection rate among cases classified as positive by the completed record;
- the fraction of replay cases with a non-abstaining action despite failed task-group coverage;
- the fraction of cases that remain unresolved.

A replay case is an ordered system pair, a simulated reveal order, and a budget cap. Ordered directions answer different one-sided questions, but they are statistically dependent. Cases reuse systems, tasks, and outcomes across pairs and reveal orders. The reported pair and replay counts therefore do not represent the number of independent experimental units. The public-record summaries are descriptive for a finite suite. Clustered or hierarchical inference would require enough independently sampled systems, tasks, and repeated runs to support the chosen population.

## Uncertainty and multiplicity

The bootstrap resamples visible paired task differences at one fixed cap. Its tail value is a policy input and an empirical uncertainty check. It is not an anytime-valid probability for repeated live looks.

The study examines multiple benchmarks, thresholds, caps, task orders, uncertainty cutoffs, and policy variants. These comparisons are operating-characteristic summaries. They do not form a multiplicity-adjusted family of confirmatory hypothesis tests and do not support a p-value based rejection claim.

Monte Carlo intervals over reveal-order seeds quantify simulation variation under the recorded replay procedure. They are not population confidence intervals for future systems or benchmark tasks.

## Dependence and repeated runs

The fixed released record is the empirical target. When a source contains repeated runs for the same system and task, the paper averages those outcomes before pairing. This removes run-level variation from the replay target. It does not show that a live rerun would reach the same action.

Full-record task counts and resource totals may be used retrospectively to define replay fractions. A live service cannot know a future realized denominator unless that denominator is part of a prespecified execution plan. Live use therefore requires an operational cap defined from information available before or during execution.

## Retrospective boundary

The adapter and verification scripts can test schema validity, malformed inputs, batch equivalence, prefix isolation under future-row mutation, resource accounting, trace integrity, and separation of completed-record audit fields. These checks cover the implementation boundary. They do not show that an upstream benchmark record is free of data contamination or scoring error.

The current public records and policy variants were available during method development. The packaged analyses are retrospective and should be read as descriptive or exploratory evidence for the selected released records.

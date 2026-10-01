---
license: apache-2.0
---

---

# SUBMISSIONS CLOSED
- All PRs opened before May 14th have been reviewed and merged if valid. If you have already run a job, you can use the same files and trajectories in the new submission process once it is announced.
- We are working on a new submission process for the Terminal Bench 2.0 Leaderboard. The new process will enforce the policies outlined in [this blog post](https://www.tbench.ai/news/leaderboard-integrity-update). Check back by end of June for an update.

---

# Terminal-Bench 2.0 Leaderboard Submissions

This repository accepts leaderboard submissions for [Terminal-Bench 2.0](https://terminal-bench.org).

## How to Submit

1. [Fork this repository](https://huggingface.co/docs/hub/en/repositories-next-steps#duplicating-with-the-git-history-fork)
2. Create a new branch for your submission
3. Add your submission (a job or folder of jobs) under `submissions/terminal-bench/2.0/<agent>__<model(s)>/`
4. Open a Pull Request

## Submission Structure

```text
submissions/
  terminal-bench/
    2.0/
      <agent>__<model>/
        metadata.yaml       # Required: agent and model info
        <job-folder>/       # One or more job directories
          config.json
          <trial-1>/result.json
          <trial-2>/result.json
          ...
```

## Required: metadata.yaml

Each submission must include a `metadata.yaml` file with the following fields:

```yaml
agent_url: https://...         # Required: link to agent repo/docs
agent_display_name: "My Agent" # Required: display name for leaderboard
agent_org_display_name: "Org"  # Required: organization name

models:                              # Required: list of models used
  - model_name: gpt-5                # Required: model identifier
    model_provider: openai           # Required: provider (openai, anthropic, etc.)
    model_display_name: "GPT-5"      # Required
    model_org_display_name: "OpenAI" # Required
  # - Other models if your agent used multiple
```

## Job Directory Requirements

Each job directory must contain all of the contents of your run.

### Validation Rules

Your submission will be automatically validated. To pass:

- `timeout_multiplier` must equal `1.0`
- No agent timeout overrides (`override_timeout_sec`, `max_timeout_sec`)
- No verifier timeout overrides
- No resource overrides (`override_cpus`, `override_memory_mb`, `override_storage_mb`)
- All trial directories must have valid `result.json` files
- Trial directories must contain other artifacts from the run
- Each task must be evaluated with a minimum of five trials. We recommend the `-k 5` flag for convenience.
- Agents cannot access the Terminal-Bench website or GitHub repository (reward hacking)

## Submission Process

1. **Open PR**: When you open a Pull Request, our bot will automatically validate your submission
2. **Fix Issues**: If validation fails, the bot will comment with specific errors to fix
3. **Merge**: Once validation passes, a maintainer will review and merge your PR
4. **Import**: After merge, results are automatically imported to the leaderboard

## Questions?

Open an issue in this repository or contact <alexgshaw64@gmail.com>.

# Can LLMs compress a network by reasoning about its weights?

6.S978 final project. An LLM is given a trained teacher MLP's weights as text and must write the weights of a
student with **half the hidden units** that makes the same predictions, with **no training, no code
execution, and no training data**. We compare against magnitude pruning and standard distillation, and vary
model capability, teacher size, and the harness.

## Setup

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env   # then paste your OpenRouter key into .env
```

Teachers and baselines are already in `artifacts/teachers/` (regenerate with `python -m weightedit.prepare`, ~10 s).

## Running

```bash
.venv/bin/python -m weightedit.run exp1 --list        # print planned runs, no API calls
.venv/bin/python -m weightedit.run exp1 --dry-run     # whole pipeline with an offline mock model
.venv/bin/python -m weightedit.run exp0 --models gpt-4o
.venv/bin/python -m weightedit.run exp1
.venv/bin/python -m weightedit.run exp1_depth --models o3
.venv/bin/python -m weightedit.run exp2
.venv/bin/python -m weightedit.analyze                # summary.csv, tables, figures in results/
```

Each episode is saved to `results/runs/<model>__<arch>__<condition>__<harness>__s<seed>.json` (full prompts,
responses, reasoning if returned, token usage, cost, every student, and metrics). Saved runs are skipped, so
runs resume after interruption, and runs with identical settings are shared between experiments (e.g. exp2's
one-shot runs reuse exp1's h16 seeds 0–2). `--rerun-failed` retries runs that ended in an API error;
`--limit N` caps new episodes; `--workers` sets concurrency (default 4).

**Before running:** fill in the OpenRouter ids for `sol` and `astra` in `weightedit/models.py`, and check the
other ids and `max_tokens` values. GPT-4 has an 8k context, so the larger teachers may fail for it; that is
recorded as an invalid run rather than retried.

## Experiments

| | models | teacher | condition | harness | seeds | runs per model |
|---|---|---|---|---|---|---|
| exp0 (pilot) | one | h16 | task only / weights only / weights + task | one shot | 5 | 15 |
| exp1 | all | h8, h16, h32, h64 | weights only | one shot | 5 | 20 |
| exp1_depth | best | h16 vs h16-16 | weights only | one shot | 5 | 10 (5 new) |
| exp2 | all | h16 | weights only | one shot / up to 5 turns | 3 | 6 (3 new) |

## Design details

- **Data**: OpenML `vowel` (990 examples). 12 inputs (speaker code, sex, 10 acoustic features), standardized with
  training statistics; 11 classes. Random 60/20/20 train/val/test split (seed 0). OOD inputs: 1000 samples from
  N(0, 2²) in standardized space.
- **Teachers**: one-hidden-layer ReLU MLPs (width 8/16/32/64) plus a 16-16 two-layer teacher, Adam, 6000 steps.
  The h16 teacher has 395 parameters, its student 203.
- **What the LLM sees**: teacher weights rounded to 3 significant figures, the student architecture and
  required JSON shapes. The task description is included only in the `task_only` and `weights_and_task`
  conditions. Model temperature is left at the provider default; the run seed is passed to the API.
- **Multi-turn harness**: up to 5 attempts. After each one, the model is told its student's agreement with the
  teacher on the validation split (or why its answer could not be used). The reported student is the attempt
  with the best validation agreement; test inputs are never used for feedback. Stops early at 100%.
- **Baselines**: magnitude pruning keeps the hidden units with the largest ‖incoming‖·‖outgoing‖. Distillation
  trains a fresh half-width student on the teacher's soft labels over the training inputs (Adam, batch 64,
  3000 steps, 3 seeds); its value at 3000 steps is the distillation ceiling.
- **Metrics** (all against the full-precision teacher): test and OOD agreement and KL(teacher‖student), test
  accuracy, error agreement (on test inputs the teacher gets wrong, how often the student makes the same
  prediction), % of distillation ceiling, equivalent distillation steps (interpolated on the mean distillation
  curve; empty if the student beats the ceiling), and whether each student hidden unit is copied from a teacher
  unit (cosine ≥ 0.99 and norm within 5%) or new.

## Baselines (test agreement with the teacher)

| teacher | params | student params | teacher test acc | magnitude pruning | distillation @3000 |
|---|---|---|---|---|---|
| h8 | 203 | 107 | 0.803 | 0.237 | 0.700 |
| h16 | 395 | 203 | 0.934 | 0.414 | 0.800 |
| h32 | 779 | 395 | 0.934 | 0.525 | 0.919 |
| h64 | 1547 | 779 | 0.929 | 0.662 | 0.929 |
| h16-16 | 667 | 275 | 0.909 | 0.278 | 0.850 |

Rounding the teacher to 3 significant figures changes a few predictions (h16: 0 on test, 4 of 1000 OOD inputs);
the exact counts per split are in each teacher's `meta.json` under `rounding_check`.

## Layout

```
data/vowel.arff            OpenML vowel dataset
artifacts/teachers/<arch>/ teacher.json, meta.json, baselines.json
weightedit/
  data.py        loading, splits, OOD inputs
  mlp.py         numpy MLP, Adam training, rounding/formatting
  teachers.py    architectures and teacher training
  baselines.py   magnitude pruning, distillation curve
  metrics.py     fidelity, error agreement, equivalent steps, unit origins
  prompts.py     prompts for each condition, feedback, answer parsing
  llm.py         OpenRouter client (stdlib only) and mock client
  models.py      model short names -> OpenRouter ids
  harness.py     one-shot and multi-turn episodes
  prepare.py     train teachers + compute baselines
  run.py         experiment runner
  analyze.py     tables, CSV, figures
```

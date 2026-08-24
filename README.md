# Tiny Scaling Laws Research Track

Three beginner CPU experiments with a real character-level Transformer:

1. Scale **model capacity**.
2. Scale **training updates**.
3. Scale **available training data**.

Each command trains the models, evaluates held-out examples, saves a JSON
receipt, creates one PNG with pretraining and RL curves, and opens the PNG. No
model or dataset is downloaded. The first run installs CPU PyTorch; later runs
take only a few seconds on a normal laptop.

## Level 1 — Model scaling

Question: does increasing Transformer width reduce pretraining loss and RL
reward gap when data and update counts stay fixed?

macOS or Linux:

```bash
git clone https://github.com/vukrosic/tiny-scaling-laws-skool-lab.git && cd tiny-scaling-laws-skool-lab && ./first_win.sh
```

Windows PowerShell:

```powershell
git clone https://github.com/vukrosic/tiny-scaling-laws-skool-lab.git; cd tiny-scaling-laws-skool-lab; .\first_win.bat
```

Reference endpoints from one three-seed macOS run:

```text
844 parameters:   pretraining loss 3.386 | RL reward gap 85.8%
8,904 parameters: pretraining loss 2.197 | RL reward gap  7.1%
```

## Level 2 — Training-budget scaling

Question: with model and data fixed, what happens when training receives more
optimizer updates?

```bash
./training_budget.sh
```

Windows: `.\training_budget.bat`

Reference endpoints:

```text
Pretraining: 5 updates → 4.004 loss; 60 updates → 2.000 loss
RL:         10 updates → 84.4% gap; 100 updates → 0.3% gap
```

This is a compute-budget sweep. More updates mean more computation.

## Level 3 — Data scaling

Question: with model and update budget fixed, what happens when the learner can
sample from a larger training pool?

```bash
./data_scaling.sh
```

Windows: `.\data_scaling.bat`

Reference endpoints:

```text
Pretraining: 116 training characters → 2.934 loss; 34k → 2.580 loss
RL:          16 unique prompts → 66.6% gap; 512 → 12.1% gap
```

This experiment holds the number of optimizer updates fixed. It changes the
available data pool, not the number of sampled training batches.

## What changes and what stays fixed

| Level | Changed variable | Fixed controls | Not fixed |
|---|---|---|---|
| 1. Model capacity | Transformer residual width and parameter count | Data and splits, context length, sampled batches within each seed, 30 pretraining updates, 50 RL updates, batch size 32, learning rates, evaluator, and seeds | Computation and runtime: wider models cost more per update |
| 2. Training budget | Number of optimizer updates | Width 16, data and splits, batch size 32, learning rates, evaluator, and seeds | Computation and runtime: more updates deliberately use more compute |
| 3. Data size | Available pretraining characters or unique RL prompts | Width 16, 30 pretraining updates, 50 RL updates, batch size 32, validation/evaluation sets, learning rates, and seeds | The sampled training pool, which is the variable being tested |

Level 1 is **update-matched, not compute-matched**: every model receives the
same number of updates, but a larger model performs more operations per update.

## Submit each experiment in Skool

Before running, write a prediction. After running, upload the generated PNG and
complete [`RESEARCH_TEMPLATE.md`](RESEARCH_TEMPLATE.md). The scientific task is
not merely running code: identify the independent variable, fixed controls,
result, limitation, and smallest useful next experiment.

## What the model does

All studies use a one-block, one-head, decoder-only Transformer that processes
individual characters.

Pretraining predicts the next character in synthetic sentences such as:

```text
Ada follows the blue owl near the lake.
```

RL is a one-step contextual-bandit task:

```text
prompt: Copy:facbed=
reward: 1 for generating f; 0 for a, b, c, d, e, g, or h
```

The RL metric is reward gap, `1 - mean reward`; lower is better. This is exact
reward-based policy optimization over eight possible character actions. It is
not RLHF, PPO, or GRPO.

## Scientific limits

These are small controlled sweeps, not universal neural scaling laws. Four
scales and three seeds cannot establish an asymptotic law. The fitted log-log
slopes are descriptive. Error bars show one standard deviation across seeds.

The main lesson is that “scaling” is incomplete unless you say what changes:
model capacity, optimization budget, or available data.

## Verify everything

```bash
./first_win.sh --no-open
./training_budget.sh --no-open
./data_scaling.sh --no-open
.venv/bin/python -m unittest discover -s tests -v
```

This repository is derived from
[`tiny-scaling-laws-lab`](https://github.com/vukrosic/tiny-scaling-laws-lab).
Generated results and virtual environments are excluded from Git.

MIT License.

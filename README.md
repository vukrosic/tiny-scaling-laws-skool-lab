# Tiny Scaling Laws Research Track

Train a real character-level Transformer from scratch and run three controlled
CPU studies. The levels progress from reproducing a result to modifying and
designing an experiment.

| Level | Research variable | Your task |
|---|---|---|
| 1 | Model capacity | Reproduce and explain the supplied sweep |
| 2 | Training updates | Change one budget value and compare the result |
| 3 | Available data | Choose a controlled data point and defend your conclusion |

Every command trains the models, evaluates held-out examples, saves a PNG and
JSON receipt, and opens the graph. No model or dataset is downloaded. The first
run installs CPU PyTorch; warm experiments take only a few seconds on a normal
laptop.

For every level: **predict → run → inspect → report**. Use
[`RESEARCH_TEMPLATE.md`](RESEARCH_TEMPLATE.md) and copy exact measurements from
the JSON receipt.

You do not need to edit Python for these three levels. Command options define
the experimental points. For example, `--rl-budgets 10,25,50,150` compares four
RL update counts, while `--rl-data-sizes 16,64,128,512` compares four training
prompt pools. Use `--image` and `--receipt` to preserve a new run instead of
overwriting the default result.

## Level 1 — Reproduce a model-capacity sweep

**Question:** Does increasing Transformer width reduce held-out pretraining
loss and RL reward gap?

- **Change:** residual width and parameter count.
- **Keep fixed:** data and splits, context length, batches within each seed, 30
  pretraining updates, 50 RL updates, batch size 32, learning rates, evaluator,
  and three seeds.
- **Not fixed:** computation and runtime. Wider models cost more per update, so
  this is update-matched, not compute-matched.

Before running, complete this sentence:

```text
I predict increasing model width will ______ because ______.
```

macOS or Linux:

```bash
git clone https://github.com/vukrosic/tiny-scaling-laws-skool-lab.git && cd tiny-scaling-laws-skool-lab && ./first_win.sh
```

Windows PowerShell:

```powershell
git clone https://github.com/vukrosic/tiny-scaling-laws-skool-lab.git; cd tiny-scaling-laws-skool-lab; .\first_win.bat
```

The command creates `my_scaling_laws.png` and `my_scaling_laws.json`. Lower is
better on both panels. Vertical bars show ±1 standard deviation over three
seeds.

Reference endpoints from one macOS run:

```text
844 parameters:   pretraining loss 3.386 | RL reward gap 85.8%
8,904 parameters: pretraining loss 2.197 | RL reward gap  7.1%
```

**Finish Level 1:** Upload the graph and report the two endpoint comparisons,
whether they support your prediction, and one limitation.

Example conclusion format—replace these reference values with your own:

```text
Increasing capacity from 844 to 8,904 parameters reduced validation loss
from 3.386 to 2.197 and RL reward gap from 85.8% to 7.1% in this setup.
This supports my prediction, but four tiny models do not establish a
universal scaling law.
```

## Level 2 — Modify the training-budget sweep

**Question:** With model width and data fixed, what happens when training gets
more optimizer updates?

- **Change:** pretraining or RL optimizer-update count.
- **Keep fixed:** width 16, data and splits, batch size 32, learning rates,
  evaluator, and three seeds.
- **Not fixed:** computation and runtime. More updates deliberately use more
  compute.

Run the supplied sweep:

```bash
./training_budget.sh
```

Windows: `.\training_budget.bat`

It creates `my_training_budget_scaling.png` and
`my_training_budget_scaling.json`.

Reference endpoints:

```text
Pretraining: 5 updates → 4.004 loss; 60 updates → 2.000 loss
RL:         10 updates → 84.4% gap; 100 updates → 0.3% gap
```

Now change one value: replace the 100-update RL endpoint with 150 while keeping
the other RL budgets fixed.

```bash
./training_budget.sh --rl-budgets 10,25,50,150 \
  --image results/level2.png --receipt results/level2.json
```

The same arguments work with `.\training_budget.bat` on Windows.

This command independently starts each RL run from the same initial policy and
trains it for 10, 25, 50, or 150 updates. You are testing whether the curve is
still improving after 50 updates, not continuing the already-trained
100-update model.

Example prediction:

```text
If the policy is still undertrained at 50 updates, 150 updates should reduce
the reward gap further. If learning is already saturated, the change should
be small.
```

**Finish Level 2:** Compare the default and modified RL curves. Report whether
the additional updates reduced the reward gap and whether the improvement was
large enough to justify the added computation.

## Level 3 — Design a data-size sweep

**Question:** With model width and update counts fixed, what happens when the
model can sample from a larger training pool?

- **Change:** available pretraining characters or unique RL prompts.
- **Keep fixed:** width 16, 30 pretraining updates, 50 RL updates, batch size
  32, validation/evaluation sets, learning rates, and three seeds.
- **Not fixed:** the training pool, which is the independent variable.

Run the supplied sweep:

```bash
./data_scaling.sh
```

Windows: `.\data_scaling.bat`

It creates `my_data_scaling.png` and `my_data_scaling.json`.

Reference endpoints:

```text
Pretraining: 116 training characters → 2.934 loss; 34k → 2.580 loss
RL:          16 unique prompts → 66.6% gap; 512 → 12.1% gap
```

Now choose a new intermediate RL data size. Keep the maximum at 512 so the
held-out split stays unchanged. For example, replace 256 prompts with 128:

```bash
./data_scaling.sh --rl-data-sizes 16,64,128,512 \
  --image results/level3.png --receipt results/level3.json
```

The same arguments work with `.\data_scaling.bat` on Windows.

Every point still receives 50 RL updates. Only the pool it can sample from
changes. A reasonable prediction is that the 128-prompt result will fall
between the 64- and 512-prompt results if additional prompt diversity helps.
That outcome is not guaranteed; a noisy or flat point is also a result to
report.

**Finish Level 3:** Explain where the new point falls, whether the curve shows
diminishing returns, and what this four-point experiment cannot establish.

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

The rule is: **output the first character after `Copy:`**. The reward function,
not the model, identifies `f` as correct because it is the first character
after the colon. The model begins with about `1/8` probability on each of the
eight actions. For this prompt, expected reward equals
`P(f | Copy:facbed=)`, so optimization increases that probability. Across 512
training prompts with different first characters, the model learns the copying
policy; evaluation uses 256 separate prompts.

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

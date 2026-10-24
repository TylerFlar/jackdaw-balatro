# Training demo: MaskablePPO on jackdaw

This folder trains [sb3-contrib](https://sb3-contrib.readthedocs.io/)'s MaskablePPO on the
simulator's `BalatroVectorEnv` (Rust engine, flat `Discrete(500)` actions with masks), then plays
the trained policy against two baselines on the same runs: a random legal action at every step,
and `jackdaw.cli.smart_agent`, the repo's hand-written heuristic player.

## Results

The demo run trained for 30.0M environment steps in 40 min on 6 CPU threads for torch
and 2 for the envs. Every agent then played the same 1,000 runs (seeds 10,000,000 to
10,000,999, Red Deck, stake 1), which training and validation never saw:

| Agent | Mean ante reached | Blinds beaten per run | Win rate | Mean episode reward | Mean episode length (steps) |
|---|---:|---:|---:|---:|---:|
| MaskablePPO, greedy | 2.08 | 4.52 ± 0.13 | 0% | +1.23 | 65 |
| MaskablePPO, sampled | 1.99 | 4.18 ± 0.12 | 0% | +1.11 | 58 |
| `smart_agent` | 2.31 | 5.38 ± 0.13 | 0% | +1.48 | 50 |
| Random legal action | 1.00 | 0.001 ± 0.001 | 0% | +0.02 | 9 |

- **Sampled** draws each action from the policy's masked distribution, as in training. **Greedy**
  takes the most likely legal action.
- Blinds beaten: the mean and its standard error. The reward is the training reward (below),
  summed over the episode.
- Episodes end when the run is lost or won, or after 2,000 steps.

From beating essentially no blind, PPO learns to clear about four and a half per run, and
its best runs reach ante 8 (`smart_agent`'s best: ante 7). The hand-written `smart_agent`
is still better on average, by almost a blind per run, and no agent wins a run. Learning
hadn't stopped: the training rollouts gained 0.3 blinds per episode over the last 10M
steps.

Playing the most likely action (greedy) is the stronger way to use the policy here. The
previous demo's greedy policy looped on free moves until the step cap; the cap on free
moves in a row (below) ended that, and no episode reaches the step cap now.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/learning_curve_dark.png">
  <img src="results/learning_curve.png" alt="Learning curves over 30M environment steps. Blinds beaten per episode rises from 0 to about 3 in the first 3M steps, then slowly to about 4.1 by 30M; the greedy evaluation sits above the sampled one and the training rollouts, and smart_agent's 5.4 stays above all three. Ante reached and episode reward follow the same pattern; episodes settle around 60 steps.">
</picture>

The four panels show the training rollouts (episodes that ended in each rollout, smoothed), and
evaluations every 2M steps on 64 fixed validation seeds, sampled and greedy. The gray lines are
the baselines on the test seeds. The same values are in
[`results/learning_curve.csv`](results/learning_curve.csv), one row per evaluation.

`results/` holds the summary (`eval_results.json`), the curves, the run's settings
(`config.json`) and the trained model (`maskable_ppo_balatro.zip`, without Adam's state, 2.5 MB).

## Install

The scripts need the `train` dependency group: stable-baselines3, sb3-contrib, tensorboard, torch
and matplotlib. It is a dependency group of this repository, not a dependency of the `jackdaw`
package, so neither `pip install jackdaw` nor a plain `uv sync` installs it:

```bash
uv sync --dev --group train
```

`uv sync` installs exactly the groups it is given, so a later plain `uv sync` (or
`uv sync --dev`) uninstalls the training packages again. Pass `--group train` every time, or use
`uv sync --all-groups`. torch comes from PyTorch's CPU wheel index, which `pyproject.toml` sets:
the demo trains a small MLP on the CPU, and the default Linux wheels would pull in several GB of
CUDA libraries.

## Train

```bash
uv run --group train python examples/train_maskable_ppo.py
```

The defaults are the demo run's settings. Training is seeded (the envs' runs, the action-table
sampler, torch, numpy), but floating-point results can still differ with the hardware and the
thread count. Everything goes to `runs/maskable_ppo/<run>/`, which is gitignored:

| Path | What |
|---|---|
| `tb/` | TensorBoard logs: `uv run --group train tensorboard --logdir runs/maskable_ppo` |
| `checkpoints/` | the model every 5M steps |
| `best_model.zip` | the model with the most blinds beaten on the 64 validation seeds (sampled) |
| `final_model.zip` | the model at the end of training |
| `config.json` | the command-line settings and the action space's version |

Create a file named `STOP` in the run directory to end training early: the script still saves
`final_model.zip`. `--resume <model.zip>` continues from a saved model. `--help` lists the
hyperparameters.

Besides SB3's usual `rollout/` and `train/` scalars, the logs have:

- `balatro/ante_reached`, `balatro/blinds_beaten`, `balatro/win_rate`: means over the training
  episodes that ended in each rollout;
- `eval/` and `eval_sampled/`: every 2M steps, 64 episodes on fixed validation seeds, playing the
  most likely legal action (`eval/`) or sampling from the policy (`eval_sampled/`).

## Evaluate and plot

```bash
uv run --group train python examples/evaluate_agents.py \
    --model examples/results/maskable_ppo_balatro.zip --out examples/results/eval_results.json
uv run --group train python examples/plot_learning_curve.py runs/maskable_ppo/<run> \
    --baselines examples/results/eval_results.json --out examples/results/learning_curve
```

`evaluate_agents.py` plays one episode per test seed for each agent, with the reward and
episode cap from the `config.json` next to the model, prints the table above and writes the JSON
summary. The first command reproduces the table from the committed model in under a minute.

## How the training is set up

**Environments.** One `BalatroVectorEnv` of 32 envs on 2 worker threads, with same-step
autoreset (`AutoresetMode.SAME_STEP`), behind `BalatroSB3VecEnv` (`balatro_sb3.py`), a small
Stable-Baselines3 `VecEnv`, and `VecMonitor` for the episode statistics. Sub-env `i` starts from
seed `seed + i` and then plays runs `TRAIN<seed>_<i>_<n>`, so no two envs replay the same run. The
step that ends an episode returns the next episode's first observation and mask, with the final
observation in `infos[i]["terminal_observation"]` (SB3 bootstraps truncated episodes from it).
MaskablePPO reads the masks with `env_method("action_masks")`, which the adapter answers from the
vector env: the masks of the observations it just returned, so the policy always picks from the
legal actions of the state it sees.

**Reward.** `ShapedReward(score_progress=0.1, blind_beaten=0.5, boss_beaten=0.3, loss=0)`: the
built-in dense reward (computed in Rust) with heavier progress terms and no loss penalty.
`step_cost` (0.001), `shop_step_cost` (0.002), `hands_left` (0.01) and `win` (0.5) keep their
defaults. With the default weights, putting off the end of a lost run saves
`loss * (1 - gamma)` per step, which equals `step_cost` at `gamma = 0.995`, so stalling is free.
In a pilot with the defaults (before the free-move cap below), the greedy policy learned to
alternate sorting and swapping cards until the 2,000-step cap and beat no blind. Pilots with the
same settings otherwise:

| Reward | Blinds beaten on the validation seeds (greedy / sampled) |
|---|---|
| `ShapedReward()`, after 0.3M steps | 0.00 / 0.00, greedy episodes of 1,627 steps |
| `ShapedReward(loss=0)`, after 1M steps | 1.30 / 1.02 |
| the demo's reward, after 1M steps | 2.56 / 1.55 |

**Free moves.** The envs allow at most 10 free moves (swaps and sorts) in a row, the default
`max_free_moves`. The previous demo was trained without the cap, and its greedy policy looped:
in some states its most likely action was a free move (mostly swapping two cards in the hand,
or two jokers in the shop), and from the next state another one. 48% of its greedy episodes ran
into the 2,000-step cap, and it beat 2.65 blinds per run against 3.74 sampled. With the cap, no
episode reaches the step cap.

**Hyperparameters.**

| | |
|---|---|
| Policy | `MultiInputPolicy`: the dict observation flattened (614 features), separate policy and value MLPs of 2 x 256 ReLU units |
| Rollouts | 32 envs x 256 steps = 8,192 transitions per update |
| Updates | 4 epochs, minibatches of 2,048, Adam at 3e-4 (constant) |
| PPO | gamma 0.995, GAE lambda 0.95, clip 0.2, entropy 0.01, value 0.5, max grad norm 0.5 |
| Episodes | Red Deck, stake 1, ended by a loss, a win or 2,000 steps |
| Threads | `--threads 6` caps torch's and the BLAS libraries' thread pools; `--env-threads 2` the vector env's |

MaskablePPO trained at 12.4K steps/s on average (30.0M steps in 40 min), against 7.8K for
the previous demo, which ran 32 `BalatroGymnasiumEnv`s in SB3's `DummyVecEnv`. torch's
OpenMP threads spin between calls, so training keeps its cores busy throughout. With
`KMP_BLOCKTIME=0` in the environment they sleep instead, at the cost of speed.

**Refusals.** An action the engine refuses (the masks and the engine disagree: an env bug) ends
its episode as truncated and is flagged in `infos[i]["balatro/refused"]`, so a bug can't stop a
long run. None happened in the demo run or its evaluations.

**The action space's version.** A trained policy is tied to the masks it was trained with: when
a release makes more moves legal, table slots the policy never saw unmasked appear, and it may
take them often. (An earlier demo model, trained before joker swaps became legal at blind select
and on the cash-out screen, spent 97% of its sampled moves on those swaps on the newer engine,
and its blinds beaten per run fell from 4.1 to 2.2.) `config.json` records
`jackdaw.env.ACTION_VERSION`, and `evaluate_agents.py` refuses a model trained on another
version. Retrain (or fine-tune) a policy when it changes.

## Files

| File | What |
|---|---|
| `train_maskable_ppo.py` | training |
| `evaluate_agents.py` | the evaluation against the baselines |
| `plot_learning_curve.py` | the learning curves from the TensorBoard logs |
| `balatro_sb3.py` | what the scripts share: the SB3 adapter, callbacks, the evaluation loop |
| `results/` | the demo run's summary, curves, settings and model |

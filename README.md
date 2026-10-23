<div align="center">
  <h1>Jackdaw</h1>
  <div><img src="docs/jackdaw.png" alt="Jackdaw Logo" width="300" height="300"/></div>
  <br/>
  <a href="https://tylerflar.github.io/jackdaw-balatro/"><img src="https://img.shields.io/badge/docs-GitHub%20Pages-blue" alt="Documentation"/></a>
</div>

---

Jackdaw is a Balatro simulator built for RL research. It features a 1:1 Python reimplementation of the Balatro engine, a Gymnasium-style environment with entity-based observations and a factored action space, and a validation bridge to play against live Balatro via [BalatroBot](https://github.com/coder/balatrobot).

## Motivation

With [BalatroBot](https://github.com/coder/balatrobot) already existing, why build a separate simulator?

While BalatroBot lets you input actions with no latency (minus network time), this can cause race conditions and actually crash the game. Parallelizing is also important for RL training — you can run multiple instances of BalatroBot, but it's more overhead and complexity than just running multiple simulators in Python. Finally, having a pure Python implementation allows for easier debugging, introspection, and customization of the game logic, which is really helpful for research.

## Install

### Add to your project

```bash
uv add git+https://github.com/TylerFlar/jackdaw-balatro.git
```

### For development

```bash
git clone https://github.com/TylerFlar/jackdaw-balatro
cd jackdaw-balatro
uv sync --dev
```

## Quick Start

```python
from jackdaw.env import BalatroEnvironment, DirectAdapter

env = BalatroEnvironment(adapter_factory=DirectAdapter)
obs, mask, info = env.reset()

while not info.get("done"):
    action = your_model.act(obs, mask)
    obs, terminated, truncated, mask, info = env.step(action)
```

## Gymnasium Wrapper

`BalatroGymnasiumEnv` exposes the game as a flat `Discrete(500)` action space with an `action_masks()` method, so it works with action-masking RL libraries such as sb3-contrib's MaskablePPO. Jackdaw doesn't depend on any RL library; install the one you use in your own project.

See the [RL Project Guide](https://tylerflar.github.io/jackdaw-balatro/rl-guide) for details on the gymnasium wrapper, reward shaping, and writing custom agents.

## Live Validation

Swap the adapter to play against real Balatro via [BalatroBot](https://github.com/coder/balatrobot) — the interface is identical. `LiveAdapter` keeps a shadow run in the simulator, checks every live answer against it, and takes observations and masks from it, so an agent sees exactly what it saw in training:

```python
from jackdaw.bridge import LiveAdapter, LiveBackend
from jackdaw.env import BalatroEnvironment

env = BalatroEnvironment(
    adapter_factory=lambda: LiveAdapter(LiveBackend("127.0.0.1", 12346))
)
# Same obs and actions; the mask only offers what BalatroBot can execute
```

`jackdaw lockstep` plays organic games on one or more BalatroBot instances this way and writes a repro bundle for every divergence; `jackdaw lockstep replay` reruns one on the simulator. See [Lockstep against the live game](https://tylerflar.github.io/jackdaw-balatro/live).

## Architecture

```
jackdaw/
  engine/           Deterministic game simulator (30 modules)
    game.py           step() — core state transition function
    run_init.py       initialize_run() — seeded game setup
    runner.py         simulate_run() — full game with an agent
    scoring.py        14-phase scoring pipeline
    jokers.py         All 150 joker effects
    rng.py            Bit-exact 3-layer PRNG (matches LuaJIT 2.1)
    shop.py           Shop population, buy/sell/reroll
    actions.py        Action types, GamePhase enum

  env/              RL environment
    balatro_env.py    BalatroEnvironment — factored action interface
    gymnasium_wrapper.py  BalatroGymnasiumEnv — SB3-compatible flat Discrete + action masking
    observation.py    Entity-based encoding (235-dim global + variable entities)
    vector_env.py     BalatroVectorEnv — many envs stepped in parallel in Rust
    action_space.py   23-type factored action space
    game_interface.py GameAdapter protocol (DirectAdapter, OracleAdapter, BridgeAdapter)
    agents.py         Agent protocol + RandomAgent baseline

  bridge/           BalatroBot backends; LiveAdapter, the deviation catalog, lockstep
  cli/              CLI tools & 250+ validation scenarios
```

## CLI

```bash
jackdaw validate                          # Run all ~250 validation scenarios
jackdaw validate --category jokers        # Joker scenarios only
jackdaw validate --scenario joker_jolly   # Single scenario
jackdaw validate --host 127.0.0.1 --port 12346
jackdaw lockstep --instances 4 --seeds 200  # Organic games on 4 BalatroBot instances
jackdaw lockstep replay BUNDLE            # Rerun a divergence on the simulator
jackdaw serve                             # BalatroBot's API on the simulator
```

> One of the biggest ways to contribute is by writing custom validation scenarios! See the [validation docs](https://tylerflar.github.io/jackdaw-balatro/validation) to get started.

## Development

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --dev          # install with dev dependencies
pytest                 # run tests
pytest --cov=jackdaw   # with coverage
pytest -m benchmark    # performance benchmarks
pytest -m live         # live validation (needs BalatroBot)
pytest -m lockstep     # long sim-against-sim lockstep runs over jackdaw serve
ruff check .           # lint
ruff format .          # format
```

## License

MIT

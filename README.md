# Jackdaw

Bit-exact headless Balatro v1.0.1o simulator for reinforcement learning
research. 1,400+ runs/sec, pure Python, no game client needed.

## Install
```bash
pip install jackdaw
```

## Quick Start
```python
from jackdaw.env import BalatroEnvironment, DirectAdapter, balatro_game_spec

spec = balatro_game_spec()
env = BalatroEnvironment(adapter_factory=DirectAdapter)
obs, mask, info = env.reset()

# Your model, your training loop, your evaluation — Jackdaw provides the env
while not info.get("done"):
    action = your_model.act(obs, mask)   # FactoredAction
    obs, mask, info = env.step(action)
```

## Live Play

Swap the adapter to play against real Balatro via balatrobot — the
interface is identical:
```python
from jackdaw.bridge import LiveBackend, BridgeAdapter

env = BalatroEnvironment(
    adapter_factory=lambda: BridgeAdapter(LiveBackend("127.0.0.1", 12346))
)
# Same obs, same masks, same FactoredAction
```

## CLI
```bash
jackdaw serve          # Start bridge server
jackdaw play           # Play interactively
jackdaw validate crash --count 200      # Crash test 200 random seeds
jackdaw validate benchmark --count 1000 # Benchmark throughput
jackdaw validate seed --seed TESTSEED   # Validate specific seed
```

## Architecture

```
jackdaw/
  engine/           Deterministic game simulator (29 modules)
    actions.py        Action types, legal-action generation, GamePhase enum
    game.py           step() — the core state transition function
    run_init.py       initialize_run() — set up a new seeded game
    runner.py         simulate_run() — run a full game with an agent
    scoring.py        14-phase scoring pipeline
    jokers.py         All 150 joker effects
    rng.py            Bit-exact pseudorandom system (matches Lua)
    shop.py           Shop population, buy/sell/reroll

  bridge/           Validation bridge for cross-checking against live Balatro

  env/              RL environment — observation encoding, action space, rewards
    balatro_env.py    BalatroEnvironment: Gymnasium-style env wrapper
    observation.py    Entity-based observation encoding (211-dim global + variable entities)
    action_space.py   21-type factored action space with pointer network targeting
    game_interface.py GameAdapter protocol (DirectAdapter, BridgeAdapter)
    rewards.py        Dense/sparse reward shaping
    agents.py         Agent protocol + RandomAgent baseline
```

## Validation

```bash
# Run the test suite (~1,400 tests)
pytest

# Crash-test 200 random games
jackdaw validate crash --count 200

# Benchmark throughput
jackdaw validate benchmark --count 1000

# Compare against live Balatro via balatrobot (requires balatrobot running)
jackdaw validate seed --seed TESTSEED --back b_red --stake 1
```

## Development

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync              # install dependencies
pytest               # run tests
ruff check           # lint
```

## License

MIT

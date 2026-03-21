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
jackdaw validate                          # Run all validation scenarios
jackdaw validate --category jokers        # Run only joker scenarios
jackdaw validate --scenario joker_joker   # Run a single scenario
jackdaw validate --host 127.0.0.1 --port 12346  # Custom balatrobot connection
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
# Run the test suite
pytest

# Validate engine against live Balatro (requires balatrobot running)
jackdaw validate                          # All ~250 scenarios
jackdaw validate --category jokers        # All 150 joker scenarios
jackdaw validate --category tarots        # All 22 tarot scenarios
jackdaw validate --category planets       # All 13 planet scenarios
jackdaw validate --category spectrals     # All 18 spectral scenarios
jackdaw validate --category boss_blinds   # All 28 boss blind scenarios
jackdaw validate --category modifiers     # All 20 modifier scenarios
```

Scenarios use the balatrobot `add`/`set` debug API to inject specific game
state (jokers, consumables, card modifiers) on both the sim engine and the
live game, then compare results after each action.

## Development

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync              # install dependencies
pytest               # run tests
ruff check           # lint
```

## License

MIT

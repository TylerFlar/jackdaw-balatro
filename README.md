# Jackdaw

Headless [Balatro](https://www.playbalatro.com/) simulator for reinforcement learning research.

Jackdaw replicates Balatro's game logic as a deterministic state machine —
scoring, 150 joker effects, shop generation, deck mechanics, seeded RNG —
producing bit-identical outcomes to the original game given the same seed
and actions.

## Quick Start

```bash
uv sync
```

```python
from jackdaw.engine.runner import simulate_run, random_agent

# Simulate a complete game with a random agent
result = simulate_run("b_red", 1, "MYSEED", random_agent)
print(f"Won: {result['won']}, Ante: {result['round_resets']['ante']}")
```

For manual step-through control:

```python
from jackdaw.engine.runner import simulate_run, random_agent
from jackdaw.engine.run_init import initialize_run
from jackdaw.engine.game import step
from jackdaw.engine.actions import get_legal_actions, GamePhase, SelectBlind

gs = initialize_run("b_red", 1, "MYSEED")
gs["phase"] = GamePhase.BLIND_SELECT
gs["blind_on_deck"] = "Small"
gs["jokers"] = []
gs["consumables"] = []

actions = get_legal_actions(gs)  # [SelectBlind(), SkipBlind()]
step(gs, actions[0])             # gs["phase"] is now "selecting_hand"
```

## Architecture

```
jackdaw/
  engine/         Deterministic game simulator (29 modules)
    actions.py      Action types, legal-action generation, GamePhase enum
    game.py         step() — the core state transition function
    run_init.py     initialize_run() — set up a new seeded game
    runner.py       simulate_run() — run a full game with an agent
    scoring.py      14-phase scoring pipeline
    jokers.py       All 150 joker effects
    rng.py          Bit-exact pseudorandom system (matches Lua)
    shop.py         Shop population, buy/sell/reroll
    ...             blinds, cards, consumables, decks, packs, pools, tags, vouchers
  bridge/         Validation bridge for cross-checking against live Balatro
  env/            Gymnasium environment wrapper (planned)
  agents/         Agent implementations (planned)
```

## Validation

```bash
# Run the test suite
uv run pytest

# Crash-test 200 random games
uv run scripts/validate.py crash --count 200

# Benchmark throughput
uv run scripts/validate.py benchmark --count 1000

# Compare against live Balatro via balatrobot (requires balatrobot running)
uv run scripts/validate.py seed --seed TESTSEED --back b_red --stake 1
```

## Development

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync              # install dependencies
uv run pytest        # run tests
uv run ruff check    # lint
```

See `docs/balatro-internals/` for reverse-engineered Balatro source documentation.

## License

MIT

# Jackdaw

Headless [Balatro](https://www.playbalatro.com/) simulator for reinforcement learning research.

## Architecture

```
jackdaw/
  engine/     Deterministic game simulator (no rendering, no UI)
  env/        Gymnasium-compatible RL environment wrapper
  agents/     Agent implementations (heuristic, MCTS, learned)
  bridge/     Validation bridge for cross-checking against Balatro
```

**Engine** replicates Balatro's game logic — scoring, joker effects, shop
generation, deck mechanics, RNG — as a pure state machine driven by player
actions. Given the same seed and action sequence, it produces identical
outcomes to the original game.

**Env** wraps the engine as a Gymnasium environment with observation/action
spaces suitable for RL training.

**Agents** consume the environment and learn to play.

## Development

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

## Status

Early development. See `docs/balatro-internals/` for reverse-engineered Balatro
source documentation and `docs/balatro-internals/engine-plan.md` for the build plan.

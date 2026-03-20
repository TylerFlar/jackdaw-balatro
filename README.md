# Jackdaw

Headless [Balatro](https://www.playbalatro.com/) simulator for reinforcement learning research.

Jackdaw replicates Balatro's game logic as a deterministic state machine —
scoring, 150 joker effects, shop generation, deck mechanics, seeded RNG —
producing bit-identical outcomes to the original game given the same seed
and actions. Includes a full RL training pipeline with observation encoding,
factored action space, transformer policy network, and PPO training loop.

## Quick Start

```bash
uv sync
```

```python
from jackdaw.engine import simulate_run, random_agent

# Simulate a complete game with a random agent
result = simulate_run("b_red", 1, "MYSEED", random_agent)
print(f"Won: {result['won']}, Ante: {result['round_resets']['ante']}")
```

For manual step-through control:

```python
from jackdaw.engine import initialize_run, step, get_legal_actions, SelectBlind

gs = initialize_run("b_red", 1, "MYSEED")
actions = get_legal_actions(gs)   # [SelectBlind(), SkipBlind()]
gs = step(gs, SelectBlind())      # now in SELECTING_HAND phase
```

## RL Training

Install with training dependencies:

```bash
uv sync --extra train
```

Evaluate the built-in heuristic agent:

```python
from jackdaw.env.agents import evaluate_agent, HeuristicAgent

result = evaluate_agent(HeuristicAgent(), n_episodes=100)
print(f"Win rate: {result.win_rate:.1%}, Avg ante: {result.avg_ante:.1f}")
```

Train a policy with PPO:

```python
from jackdaw.env.training import train_ppo, PPOConfig

result = train_ppo(PPOConfig(total_timesteps=100_000))
```

Run baseline evaluations:

```bash
uv run python scripts/baselines.py
```

See [docs/env-architecture.md](docs/env-architecture.md) for the full env
module design, and [docs/profiling-results.md](docs/profiling-results.md)
for performance benchmarks.

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
    ...               blinds, cards, consumables, decks, packs, pools, tags, vouchers

  bridge/           Validation bridge for cross-checking against live Balatro

  env/              RL environment — observation encoding, action space, agents, training
    observation.py    Entity-based observation encoding (211-dim global + variable entities)
    action_space.py   21-type factored action space with pointer network targeting
    game_interface.py GameAdapter protocol (DirectAdapter, BridgeAdapter)
    rewards.py        Dense/sparse reward shaping
    agents/           Agent implementations (RandomAgent, HeuristicAgent)
    policy/           Transformer policy network
      entity_encoder.py  Per-type MLPs + center key embeddings
      transformer.py     [CLS]-conditioned self-attention core
      action_heads.py    Type/entity/card/value heads
      policy.py          BalatroPolicy: full network + sampling + evaluation
    training/         PPO training loop
      ppo.py            SyncVectorEnv, RolloutBuffer, PPOTrainer, train_ppo()

  agents/           Engine-level agent implementations
```

## Validation

```bash
# Run the test suite (~380 tests)
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

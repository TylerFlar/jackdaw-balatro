# Balatro RL Environment Architecture

## Overview

```
                        ┌─────────────────────────────────────────────────────┐
                        │                  Agent / Policy                     │
                        │  (HeuristicAgent, RandomAgent, BalatroPolicy)       │
                        └──────────────────────┬──────────────────────────────┘
                                               │ FactoredAction
                        ┌──────────────────────▼──────────────────────────────┐
                        │                  GameAdapter                        │
                        │  Unified interface: reset / step / get_legal_actions │
                        └──────┬──────────────────────────────────┬───────────┘
                               │                                  │
                    ┌──────────▼──────────┐          ┌────────────▼────────────┐
                    │   DirectAdapter     │          │    BridgeAdapter        │
                    │   (zero overhead)   │          │    (JSON-RPC layer)     │
                    └──────────┬──────────┘          └──────┬─────────┬────────┘
                               │                           │         │
                    ┌──────────▼──────────┐    ┌───────────▼───┐ ┌───▼──────────┐
                    │    Engine           │    │  SimBackend   │ │ LiveBackend  │
                    │  (in-process Lua)   │    │  (in-process) │ │ (HTTP proxy) │
                    └─────────────────────┘    └───────────────┘ └──────────────┘
                                                                        │
                                                                 ┌──────▼──────┐
                                                                 │  Balatro    │
                                                                 │  (real game)│
                                                                 └─────────────┘
```

## Observation Space

Entity-based encoding following the AlphaStar pattern. Each entity type gets a
fixed-width feature vector; the number of entities varies per timestep.

### Global Context (90 features)

| Range  | Features     | Count |
|--------|-------------|-------|
| 0–5    | Phase one-hot (BLIND_SELECT, SELECTING_HAND, ROUND_EVAL, SHOP, PACK_OPENING, GAME_OVER) | 6 |
| 6–9    | Blind on deck one-hot (None, Small, Big, Boss) | 4 |
| 10–29  | Scalars: ante, round, dollars, hands_left, discards_left, hand_size, joker_slots, consumable_slots, blind_chips, chips_scored, score_fraction, reroll_cost, free_rerolls, interest_cap, discount_percent, skips, boss_blind_key, deck_remaining, discard_pile_size, meta_flags | 20 |
| 30–89  | Hand levels: 12 hand types × 5 (level, chips, mult, played, visible) | 60 |

### Playing Card (14 features)

| Index | Feature | Encoding |
|-------|---------|----------|
| 0 | rank_id | ordinal /14 |
| 1 | suit | ordinal 0–3, /3 |
| 2 | chip_value | /11 |
| 3 | enhancement | ordinal 0–8, /8 |
| 4 | edition | ordinal 0–4, /4 |
| 5 | seal | ordinal 0–4, /4 |
| 6 | debuffed | 0/1 |
| 7 | face_down | 0/1 |
| 8 | is_face_card | 0/1 |
| 9 | is_scoring | 0/1 |
| 10 | bonus_chips | log-scaled |
| 11 | times_played | log-scaled |
| 12 | position | /20 |
| 13 | reserved | 0 |

### Joker (15 features)

| Index | Feature | Encoding |
|-------|---------|----------|
| 0 | center_key_id | /NUM_CENTER_KEYS |
| 1 | rarity proxy | cost /20 |
| 2 | edition | ordinal /4 |
| 3 | sell_value | log-scaled |
| 4 | eternal | 0/1 |
| 5 | perishable | 0/1 |
| 6 | perish_tally | /5 |
| 7 | rental | 0/1 |
| 8 | debuffed | 0/1 |
| 9 | position | /20 |
| 10 | ability_mult | log-scaled |
| 11 | ability_x_mult | raw |
| 12 | ability_chips | log-scaled |
| 13 | ability_extra | log-scaled |
| 14 | condition_met | 0/1 |

### Consumable (7 features)

center_key_id, card_set, sell_value, can_use, needs_targets, max_targets, min_targets

### Shop Item (9 features)

center_key_id, card_set, cost, affordable, has_slot, edition, eternal, perishable, rental

## Action Space (21 Types)

| ID | Name | Targets |
|----|------|---------|
| 0 | PlayHand | card_target (1–5 hand cards) |
| 1 | Discard | card_target (1–5 hand cards) |
| 2 | SelectBlind | none |
| 3 | SkipBlind | none |
| 4 | CashOut | none |
| 5 | Reroll | none |
| 6 | NextRound | none |
| 7 | SkipPack | none |
| 8 | BuyCard | entity_target (shop index) |
| 9 | SellJoker | entity_target (joker index) |
| 10 | SellConsumable | entity_target (consumable index) |
| 11 | UseConsumable | entity_target + card_target |
| 12 | RedeemVoucher | entity_target (voucher index) |
| 13 | OpenBooster | entity_target (booster index) |
| 14 | PickPackCard | entity_target (pack card index) |
| 15 | SwapJokersLeft | entity_target (joker index) |
| 16 | SwapJokersRight | entity_target (joker index) |
| 17 | SwapHandLeft | entity_target (hand card index) |
| 18 | SwapHandRight | entity_target (hand card index) |
| 19 | SortHandRank | none |
| 20 | SortHandSuit | none |

### Action Sampling (Autoregressive)

The policy network samples actions in three stages:

1. **Type**: Categorical over 21 types, masked by `type_mask`
2. **Entity**: Pointer network (scaled dot-product attention) over entities, masked by `pointer_masks[action_type]`
3. **Cards**: Independent Bernoulli per hand card, with min/max constraint enforcement

## Reward Function

Two wrappers are provided:

### DenseRewardWrapper (for training)

| Component | Weight | Description |
|-----------|--------|-------------|
| Terminal | win=+10, loss=-1 | Scaled by ante progress |
| Ante advance | +1 per blind beaten | Type-weighted (Boss > Small/Big) |
| Scoring efficiency | 0.01 | score / blind_chips ratio |
| Economy | 0.01 | dollars / interest_threshold |
| Hand improvement | 0.001 | log(new_best / old_best) |
| Wasted action | -0.001 | Sort/swap/bankrupt reroll penalty |

### SparseRewardWrapper (for evaluation)

Terminal win/loss only.

## Policy Network

```
Observation
    │
    ▼
EntityEncoder (per-type MLPs + center key embeddings + type embeddings)
    │
    ▼
TransformerCore ([CLS] + entities → self-attention → global_repr + entity_reprs)
    │
    ▼
ActionHeads
    ├── type_head: Linear(E, 21) → Categorical
    ├── entity_query/key: pointer attention → Categorical
    ├── card_scorer: Linear(E, 1) per card → Bernoulli
    └── value_head: MLP(E) → scalar
```

## How to Train

```python
from jackdaw.env.training import train_ppo, PPOConfig

config = PPOConfig(
    total_timesteps=1_000_000,
    num_envs=8,
    back_keys="b_red",
    embed_dim=128,
    num_layers=3,
)
result = train_ppo(config)
print(f"Win rate: {result.final_eval.win_rate:.1%}")
```

## How to Evaluate

```python
from jackdaw.env.agents import evaluate_agent, HeuristicAgent

result = evaluate_agent(HeuristicAgent(), n_episodes=100)
print(result.summary())
# {'n_episodes': 100, 'win_rate': 0.45, 'avg_ante': 5.2, ...}
```

## How to Validate Against Real Balatro

```python
from jackdaw.env.live_env import LiveBalatroEnv, validate_episode
from jackdaw.env.game_interface import DirectAdapter

env = DirectAdapter()
live_env = LiveBalatroEnv(host="127.0.0.1", port=12346)

def agent(adapter):
    legal = adapter.get_legal_actions()
    return legal[0] if legal else None

result = validate_episode(env, live_env, seed="VALIDATE_1", agent=agent)
print(f"OK: {result.ok}, Divergences: {len(result.divergences)}")
```

## Performance Characteristics

| Operation | Typical Latency |
|-----------|----------------|
| Engine step | ~0.1–0.5 ms |
| Observation encoding | ~0.05–0.2 ms |
| Action mask computation | ~0.02–0.1 ms |
| Policy forward (CPU, B=8) | ~5–15 ms |
| Policy forward (GPU, B=8) | ~1–3 ms |
| Full episode (heuristic) | ~200–1000 ms |
| Bridge round-trip overhead | ~10–50% vs direct |

# Balatro RL Environment Architecture

## Overview

The `jackdaw.env` module wraps the deterministic engine into a complete RL
training pipeline: observation encoding, factored action space, reward
shaping, transformer policy network, and PPO training loop.

```
                        +-----------------------------------------------------+
                        |                  Agent / Policy                      |
                        |  (HeuristicAgent, RandomAgent, BalatroPolicy)        |
                        +------------------------+----------------------------+
                                                 | FactoredAction
                        +------------------------v----------------------------+
                        |                  GameAdapter                        |
                        |  Unified interface: reset / step / get_legal_actions |
                        +--------+-------------------------------+------------+
                                 |                               |
                      +----------v----------+       +------------v-----------+
                      |   DirectAdapter     |       |    BridgeAdapter       |
                      |   (zero overhead)   |       |    (JSON-RPC layer)    |
                      +----------+----------+       +------+--------+--------+
                                 |                         |        |
                      +----------v----------+  +-----------v--+ +---v-----------+
                      |    Engine           |  |  SimBackend  | | LiveBackend   |
                      |  (in-process)       |  |  (in-process)| | (HTTP proxy)  |
                      +---------------------+  +--------------+ +------+--------+
                                                                       |
                                                                +------v------+
                                                                |  Balatro    |
                                                                |  (real game)|
                                                                +-------------+
```

### Training pipeline

```
SyncVectorEnv (N parallel _EnvInstance)
     |
     | obs, masks, infos
     v
collate_policy_inputs() --> padded batch tensors
     |
     v
BalatroPolicy.sample_action()
     |  type --> entity --> cards (autoregressive)
     v
FactoredAction per env
     |
     | factored_to_engine_action()
     v
_EnvInstance.step()
     |  engine step + encode_observation + get_action_mask + reward
     v
RolloutBuffer.add()
     |
     | after N steps: compute_returns() (GAE)
     v
PPOTrainer._update()
     |  evaluate_actions() + clipped surrogate loss + entropy bonus
     v
optimizer.step()
```

## Observation Space

Entity-based encoding following the AlphaStar pattern. Each entity type gets
a fixed-width feature vector; the number of entities varies per timestep.
All values are float32 with explicit normalization.

### Global Context (211 features)

| Range | Features | Count |
|-------|----------|-------|
| 0-5 | Phase one-hot (BLIND_SELECT, SELECTING_HAND, ROUND_EVAL, SHOP, PACK_OPENING, GAME_OVER) | 6 |
| 6-9 | Blind on deck one-hot (None, Small, Big, Boss) | 4 |
| 10-29 | Scalars: ante, round, dollars, hands_left, discards_left, hand_size, joker_slots, consumable_slots, blind_chips, chips_scored, score_fraction, reroll_cost, free_rerolls, interest_cap, discount_percent, skips, boss_blind_key, deck_remaining, discard_pile_size, meta_flags | 20 |
| 30-89 | Hand levels: 12 hand types x 5 (level, chips, mult, played, visible) | 60 |
| 90-121 | Vouchers owned: 32-dim binary (all voucher keys) | 32 |
| 122-129 | Blind effect: boss, disabled, mult, debuff_suit (4), debuff_face | 8 |
| 130-132 | Round position: small/big/boss one-hot | 3 |
| 133-134 | Round progress: hands_played, discards_used this round | 2 |
| 135-158 | Awarded tags: 24-dim binary (all tag keys) | 24 |
| 159-210 | Discard pile: 4 suits x 13 ranks histogram | 52 |

Normalization strategy:
- Ordinals: divided by max value (e.g., rank /14, suit /3)
- Unbounded values: `log2(1 + |x|)` with sign preservation
- Fractions: clamped to [0, 1] or [0, 10] before dividing
- Binary: 0/1 flags

### Playing Card (14 features)

| Index | Feature | Encoding |
|-------|---------|----------|
| 0 | rank_id | ordinal /14 |
| 1 | suit | ordinal 0-3, /3 |
| 2 | chip_value | /11 |
| 3 | enhancement | ordinal 0-8, /8 |
| 4 | edition | ordinal 0-4, /4 |
| 5 | seal | ordinal 0-4, /4 |
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
| 0 | center_key_id | /NUM_CENTER_KEYS (~299) |
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

center_key_id, card_set, sell_value, can_use, needs_targets, max_targets,
min_targets.

### Shop Item (9 features)

center_key_id, card_set, cost, affordable, has_slot, edition, eternal,
perishable, rental.

## Action Space (21 Types)

Factored action space with autoregressive sampling: type -> entity -> cards.

| ID | Name | Targets | Engine equivalent |
|----|------|---------|-------------------|
| 0 | PlayHand | card_target (1-5 hand cards) | PlayHand |
| 1 | Discard | card_target (1-5 hand cards) | Discard |
| 2 | SelectBlind | none | SelectBlind |
| 3 | SkipBlind | none | SkipBlind |
| 4 | CashOut | none | CashOut |
| 5 | Reroll | none | Reroll |
| 6 | NextRound | none | NextRound |
| 7 | SkipPack | none | SkipPack |
| 8 | BuyCard | entity_target (shop index) | BuyCard, BuyAndUse |
| 9 | SellJoker | entity_target (joker index) | SellCard(area=jokers) |
| 10 | SellConsumable | entity_target (consumable index) | SellCard(area=consumables) |
| 11 | UseConsumable | entity_target + card_target | UseConsumable |
| 12 | RedeemVoucher | entity_target (voucher index) | RedeemVoucher |
| 13 | OpenBooster | entity_target (booster index) | OpenBooster |
| 14 | PickPackCard | entity_target (pack card index) | PickPackCard |
| 15 | SwapJokersLeft | entity_target (joker index) | ReorderJokers |
| 16 | SwapJokersRight | entity_target (joker index) | ReorderJokers |
| 17 | SwapHandLeft | entity_target (hand card index) | ReorderHand |
| 18 | SwapHandRight | entity_target (hand card index) | ReorderHand |
| 19 | SortHandRank | none | SortHand(mode=rank) |
| 20 | SortHandSuit | none | SortHand(mode=suit) |

100% forward coverage verified empirically: every legal engine action converts
to a factored action (see [action-coverage.md](action-coverage.md)).

### Action sampling (autoregressive)

1. **Type**: Categorical over 21 types, masked by `type_mask`
2. **Entity**: Pointer network (scaled dot-product attention) over entities,
   masked by `pointer_masks[action_type]`. Entity targets are grouped by
   action type for batched computation.
3. **Cards**: Independent Bernoulli per hand card, with min/max constraint
   enforcement via top-k clamping.

### Pointer mask construction

Entity types are concatenated into a single sequence:
`[hand | jokers | consumables | shop | pack]`. Pointer masks index into this
sequence using `shop_splits` to place BuyCard, RedeemVoucher, and OpenBooster
masks at the correct offsets within the shop region.

## Reward Function

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
    |
    v
EntityEncoder
    |-- hand_proj:  Linear(D_PLAYING_CARD, E)
    |-- joker_proj: Linear(D_JOKER, E) + center_key_embedding(NUM_KEYS, E)
    |-- cons_proj:  Linear(D_CONSUMABLE, E)
    |-- shop_proj:  Linear(D_SHOP, E)
    |-- pack_proj:  Linear(D_PLAYING_CARD, E)
    |-- type_embed: Embedding(5, E)  (entity type IDs)
    |
    v
TransformerCore
    |-- [CLS] token (learned) + global_context projection
    |-- entity tokens (from EntityEncoder)
    |-- nn.TransformerEncoder (num_layers, num_heads, dropout)
    |-- output: global_repr = [CLS], entity_reprs = rest
    |
    v
ActionHeads
    |-- type_head:     Linear(E, 21) + masked softmax --> Categorical
    |-- entity_query:  Linear(E, E)  }
    |-- entity_key:    Linear(E, E)  } --> scaled dot-product attention --> Categorical
    |-- card_scorer:   Linear(E, 1) per card position --> Bernoulli
    |-- value_head:    MLP(E, E, 1) --> state value scalar
```

Default hyperparameters: embed_dim=128, num_heads=4, num_layers=3, dropout=0.1.

## Training Loop

CleanRL-style PPO with custom components for variable-length observations.

### Components

- **SyncVectorEnv**: Synchronous parallel envs returning lists (not stacked
  arrays) to handle variable-length entity lists.
- **_EnvInstance**: Single env wrapping DirectAdapter + DenseRewardWrapper
  with auto-reset and episode tracking.
- **RolloutBuffer**: Stores raw Observation/ActionMask objects per step.
  Only collates into padded tensors when generating minibatches, keeping
  per-step storage lightweight.
- **PPOTrainer**: Orchestrates rollout collection, GAE computation, and
  PPO updates with clipped surrogate loss, clipped value loss, and entropy
  bonus.

### PPO hyperparameters (defaults)

| Parameter | Value |
|-----------|-------|
| num_envs | 8 |
| num_steps | 128 |
| update_epochs | 4 |
| num_minibatches | 4 |
| gamma | 0.99 |
| gae_lambda | 0.95 |
| clip_coef | 0.2 |
| ent_coef | 0.01 |
| vf_coef | 0.5 |
| max_grad_norm | 0.5 |
| learning_rate | 2.5e-4 |

## Performance

All CPU, single-threaded. See [profiling-results.md](profiling-results.md)
for full breakdown.

| Metric | Value |
|--------|-------|
| Single-env step (engine + obs + mask + reward) | 61us |
| Env overhead excluding engine | 36us |
| Training SPS (tiny: embed=32, 1L, 2 envs) | 1,075 |
| Training SPS (small: embed=64, 2L, 4 envs) | 1,085 |
| Rollout collection | 60-62% of training time |
| PPO update | 20-38% of training time |

## Baseline Performance

See [baselines.md](baselines.md) for full results.

| Agent | Win Rate | Avg Ante |
|-------|----------|----------|
| RandomAgent (200 episodes) | 0.0% | 1.0 |
| HeuristicAgent (200 episodes) | 0.0% | 2.2 |

0% win rate is expected — beating ante 8 boss (100K chips) requires joker
synergy exploitation beyond what simple heuristics achieve. The RL agent's
goal is to learn these scaling strategies.

## Known Limitations and Future Work

From the [observation audit](observation-audit.md):

| Gap | Importance | Status |
|-----|-----------|--------|
| Vouchers owned | HIGH | Implemented (32-dim binary) |
| Boss blind effects | HIGH | Implemented (8-dim features) |
| Discard pile histogram | HIGH | Implemented (52-dim suit x rank) |
| Tags | MEDIUM | Implemented (24-dim binary) |
| Round position in ante | MEDIUM | Implemented (3-dim one-hot) |
| Joker trigger state (per-joker counters) | LOW | Deferred — center_key_id is sufficient |
| Targeting cards (idol/mail/ancient/castle) | LOW | Deferred |
| Shop/pool rates | LOW | Deferred |

Remaining engineering work:
- GPU training support (config change — policy already device-aware)
- SubprocVectorEnv for parallel env stepping
- Vectorize remaining per-item entity log-prob loop in evaluate_actions
- Curriculum learning (start easy, increase stake)
- Self-play or population-based training

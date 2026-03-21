---
layout: default
title: RL Environment
---

# RL Environment

Gymnasium-compatible environment for training RL agents on Balatro.

## Interface

```python
from jackdaw.env import BalatroEnvironment, DirectAdapter

env = BalatroEnvironment(
    adapter_factory=DirectAdapter,
    back_keys=["b_red"],
    stakes=[1],
    max_steps=10_000,
)

obs, mask, info = env.reset()
while not info.get("done"):
    action = agent.act(obs, mask)
    obs, mask, info = env.step(action)
```

| Return | Type | Description |
|--------|------|-------------|
| `obs` | `dict[str, np.ndarray]` | Entity-based observation |
| `mask` | `dict[str, np.ndarray]` | Legal action masks |
| `info` | `dict` | Phase, ante, round, reward, done flag |

## Adapters

| Adapter | Speed | Use case |
|---------|-------|----------|
| `DirectAdapter` | ~1,400 runs/sec | Training (in-process) |
| `BridgeAdapter` | ~2 runs/sec | Validation (live Balatro via BalatroBot) |

## Action Space

21 factored action types. Each action has three components:

| Component | Description |
|-----------|-------------|
| `action_type` | What to do (0–20) |
| `entity_target` | Index into jokers, shop cards, etc. |
| `card_target` | Binary mask over hand cards |

Action types: PlayHand, Discard, SelectBlind, SkipBlind, CashOut, Reroll, NextRound, SkipPack, BuyCard, SellJoker, SellConsumable, UseConsumable, RedeemVoucher, OpenBooster, PickPackCard, SwapJokersLeft/Right, SwapHandLeft/Right, SortHandRank, SortHandSuit.

The `mask` dict tells you which actions are legal — always respect it.

## Observations

Entity-based encoding (variable-length, no padding):

| Key | Shape | Description |
|-----|-------|-------------|
| `global` | `(211,)` | Ante, round, dollars, hand levels, blind info, deck composition, vouchers |
| `hand_cards` | `(H, D)` | Cards in hand (rank, suit, enhancement, edition, seal) |
| `jokers` | `(J, D)` | Owned jokers (center key, edition, cost, flags) |
| `consumables` | `(C, D)` | Owned consumables |
| `shop_cards` | `(S, D)` | Shop offerings |
| `pack_cards` | `(K, D)` | Cards in open booster pack |

Works naturally with transformers, set encoders, or pointer networks.

## Validation

250+ scenarios across 7 categories validate the engine against live Balatro:

| Category | Count |
|----------|-------|
| Jokers | 150+ |
| Boss blinds | 28 |
| Tarots | 22 |
| Modifiers | 20 |
| Spectrals | 18 |
| Planets | 13 |
| Tags | varies |

```bash
jackdaw validate                        # all scenarios
jackdaw validate --category jokers      # specific category
jackdaw validate --scenario joker_joker # single scenario
```

## CLI

```bash
jackdaw validate [--category CAT] [--scenario NAME] [--host HOST] [--port PORT] [--delay SEC]
```

---
layout: default
title: Python API
---

# The Python API

The game engine is written in Rust (`crates/jackdaw-core`) and runs in-process through the
`jackdaw._core` extension. `jackdaw.core` is its Python face. It reproduces Balatro 1.0.1o
exactly, down to the RNG, and checks itself against the game's Lua source (see
[the parity ledger](https://github.com/TylerFlar/jackdaw-balatro/tree/main/docs/parity)).

For reinforcement learning, start with the [RL Project Guide](rl-guide): the environments wrap
this API. Use `jackdaw.core` directly for scripting, analysis, search, or your own environment.

## A run

```python
from jackdaw.core import Game
from jackdaw.engine.actions import SelectBlind, PlayHand, Discard, CashOut, BuyCard, NextRound

game = Game.new("b_red", stake=1, seed="TUTORIAL")   # back key, stake 1-8, seed
game.step(SelectBlind())                              # the round starts, the hand is dealt
game.step(PlayHand(card_indices=(0, 2, 3)))           # hand indices
print(game.phase, game.chips, game.hands_left)
```

- **Creating a run.** `Game.new(back, stake, seed=None, challenge=None)`. Without a seed it
  draws one the way the game does. For a challenge, pass its id (`"c_omelette_1"`); the back is
  then the challenge deck.
- **Stepping.** `step(action)` takes an action dataclass from `jackdaw.engine.actions`, or its
  tape form (`["PlayHand", [0, 2, 3]]`), and plays until the game waits for the next decision.
- **Errors.** An action the rules refuse raises `IllegalActionError`, and the run is unchanged.
  `NotPortedError` would mean a missing rule; nothing raises it now that every rule is ported.

The actions are `SelectBlind`, `SkipBlind`, `RerollBoss`, `PlayHand`, `Discard`, `SortHand`,
`SwapHandLeft`/`Right`, `SwapJokersLeft`/`Right`, `CashOut`, `BuyCard`, `BuyAndUse`,
`SellCard`, `UseConsumable`, `RedeemVoucher`, `OpenBooster`, `PickPackCard`, `SkipPack`,
`Reroll` and `NextRound`.

## What is legal

```python
for action in game.legal_actions():
    print(action)
```

- **Order.** `legal_actions()` lists the actions in a fixed order.
- **Plays and discards.** `PlayHand` and `Discard` appear once each, with no cards, standing
  for any selection of 1-5 hand cards (Cerulean Bell's forced card must be included).
- **Consumables.** A consumable that targets cards carries a default selection.
- **Checking one action.** `game.check(action)` returns why the rules refuse an action, or
  `None`, without changing the run.

`legal_actions(Legality.BALATROBOT)` keeps only what the live
[BalatroBot](balatrobot) also executes. Use it for agents that will play the real game. It
drops sim-only actions such as `RerollBoss` and `BuyAndUse`, and moves the bot refuses.

## Reading the run

Scalars are properties: `phase`, `seed`, `ante`, `round`, `dollars`, `chips`, `hands_left`,
`discards_left`, `won`. Areas and the rest are read-only views:

| Property | Type |
|---|---|
| `hand`, `jokers`, `consumables`, `deck`, `discard_pile` | `tuple[CardView, ...]` |
| `shop_jokers`, `shop_vouchers`, `shop_boosters`, `pack_cards` | `tuple[CardView, ...]` |
| `blind` | `BlindView` (`key`, `chips` to beat, `dollars`, `boss`, `disabled`) or `None` |
| `hand_levels` | mapping hand name → `HandLevel` (`level`, `chips`, `mult`, `played`, `visible`) |
| `tags` | held tag keys |
| `last_score` | `Score` (`hand_type`, `chips`, `mult`, `total`, `dollars`, `debuffed`) or `None` |

A `CardView` has:
- `key` (the center: `"j_joker"`, `"c_fool"`, `"m_glass"`, `"c_base"`...);
- for playing cards, `front` (`"H_A"`), `suit` and `rank`;
- `edition`, `seal`, `debuff` and `face_down`;
- `cost` and `sell_cost`;
- the stickers;
- a read-only `ability`.

Views are frozen: they describe the run when they were read.

## Exact previews and search

```python
score = game.preview_play((0, 1, 2, 3, 4))   # the real play on a copy
print(score.hand_type, score.total)

branch = game.clone()                        # an independent copy, a few microseconds
branch.step(Discard(card_indices=(5, 6, 7)))
```

`preview_play` runs the actual step on a copy, so it includes every joker, enhancement, edition,
seal, the boss and the RNG. The run itself is unchanged. `clone()` is cheap enough for tree
search. `scoring_cards(positions)` gives the hand those cards make and which of them would score
(`("Pair", [1, 3])`), without playing them.

## Building a state

```python
from jackdaw.core import GameBuilder

game = (
    GameBuilder("b_red", stake=2, seed="DEMO0001")
    .boss("bl_hook")
    .select_blind()
    .joker("j_blueprint")
    .joker("j_joker", edition="polychrome")
    .card("H_A", enhancement="m_glass", seal="Red")   # into the hand during a round
    .money(50)
    .build()
)
```

The additions go through the same operations as BalatroBot's debug commands `add` and `set`
(`Game.debug_add`, `Game.debug_set`, `Game.debug_restock_shop`).

## Saving and comparing runs

- `to_dict()` returns the canonical snapshot (`jackdaw.snapshot/1`,
  [schema](snapshot-schema)). It is plain JSON-compatible data and is also the save format;
  `Game.from_dict(snapshot)` restores it exactly.
- `state_hash()` is the SHA-256 digest of the canonical snapshot. Equal states have equal hashes.
- `replace_state(snapshot)` loads an edited snapshot into the run at the same decision point,
  keeping what a snapshot doesn't carry (the card-area settings BalatroBot's view reports, the
  hand's highlights). The live lockstep uses it to copy live values into its shadow run.

## BalatroBot

- `bot_view()` returns BalatroBot v1.5.2's `gamestate` view of the run, as the live bot would
  send it, except that text fields carry names.
- `bot_request(action)` returns the `(method, params)` request that performs an action live, or
  `None` for a sim-only action.
- `jackdaw serve` runs a BalatroBot-compatible JSON-RPC server on the simulator.
- `jackdaw.bridge.LiveAdapter` plays the live game in lockstep with a `Game` as its shadow, and
  `jackdaw lockstep` does it at scale: see [Lockstep against the live game](live).

See [BalatroBot on the simulator](balatrobot).

## Observations

`observation_buffer()` returns the RL observation (encoder v1) as raw float32 data, and
`jackdaw.env.encode_observation(game)` wraps it as an `Observation`; `jackdaw.env.get_action_mask(game)`
gives the action mask. The environments in `jackdaw.env` (`BalatroEnvironment`,
`BalatroGymnasiumEnv`, `BalatroVectorEnv`) build on these; see the [RL Project Guide](rl-guide).

## Speed

Typical costs on a desktop CPU, release build:

| Operation | Time |
|---|---|
| A play step | 4–8 µs |
| `clone()` | 1–2 µs |
| A new run | about 27 µs |
| Encoding an observation | 1 µs, 2.5–3 µs with its numpy arrays |

The single-threaded environments step at about 130K steps/s. `BalatroVectorEnv` steps at
about 580K steps/s on 8 threads (256 envs).

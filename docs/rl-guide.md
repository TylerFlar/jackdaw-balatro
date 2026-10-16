# RL Project Guide

How to set up an RL project using Jackdaw as the environment.

## Install

```bash
uv add git+https://github.com/TylerFlar/jackdaw-balatro.git
```

## The Loop

Every RL project using Jackdaw follows the same pattern:

```python
from jackdaw.env import BalatroEnvironment, DirectAdapter

env = BalatroEnvironment(adapter_factory=DirectAdapter)
obs, mask, info = env.reset()

terminated = False
truncated = False

while not (terminated or truncated): 
    action = agent.act(obs, mask, info)
    obs, terminated, truncated, mask, info = env.step(action)
```

That's the whole interface. The rest of this page explains what `obs`, `mask`, and `action` actually look like so you can wire them into your model.

## Observations

`obs` is a `GameObservation` with two fields:

```python
obs.global_context   # np.ndarray, shape (235,)
obs.entities         # dict[str, np.ndarray]
```

### Global Context — `(235,)` float32

A fixed-size vector encoding the full game state that isn't tied to a specific card. Includes:

- Game phase (one-hot, 6 phases)
- Ante, round, dollars (normalized)
- Hands left, discards left
- Hand levels for all 12 poker hand types (level + chips + mult each)
- Current blind info (type, target score, boss blind key)
- Deck composition (count per rank, count per suit)
- Owned vouchers (binary, 32 voucher slots)
- Strategic features (best hand in current hand, playable hand types)

### Entities — variable-length

Each entity type is a 2D array with shape `(N, D)` where N varies per timestep:

| Key | Feature dim | What each row encodes |
|-----|-------------|----------------------|
| `hand_card` | 15 | rank, suit, enhancement, edition, seal, position, chips, scoring flags |
| `joker` | 15 | center_key (integer ID), edition, cost, sell_value, rarity, position, ability flags |
| `consumable` | 7 | center_key, card_set (tarot/planet/spectral), cost, sell_value, can_use flag |
| `shop_item` | 9 | center_key, card_set, cost, edition, rarity, affordable flag |
| `pack_card` | 15 | same encoding as hand_card |

When an entity type is empty (e.g. no jokers owned), the array has shape `(0, D)`.

This encoding works naturally with transformers, set encoders, or pointer networks — no padding, no fixed-size limitations.

## Action Masks

`mask` is a `GameActionMask` that tells you what's legal right now:

```python
mask.type_mask        # np.ndarray, shape (21,) bool — which action types are legal
mask.card_mask        # np.ndarray, shape (N_hand,) bool — which hand cards can be selected
mask.entity_masks     # dict[int, np.ndarray] — per action type, which entities are legal targets
mask.min_card_select  # int — minimum cards to select (for play/discard)
mask.max_card_select  # int — maximum cards to select
```

Always respect the masks. If `type_mask[i]` is False, don't emit action type `i`.

## Actions

Actions are `FactoredAction` with three components:

```python
from jackdaw.env import FactoredAction

action = FactoredAction(
    action_type=0,                  # int: which action (0-20)
    card_target=(0, 2, 4),          # tuple[int, ...] | None: hand card indices
    entity_target=None,             # int | None: entity index
)
```

### The 21 Action Types

| ID | Name | Targets | When |
|----|------|---------|------|
| 0 | PlayHand | card_target (1-5 cards) | Playing phase |
| 1 | Discard | card_target (1-5 cards) | Playing phase |
| 2 | SelectBlind | — | Blind select |
| 3 | SkipBlind | — | Blind select |
| 4 | CashOut | — | Round eval |
| 5 | Reroll | — | Shop |
| 6 | NextRound | — | Shop |
| 7 | SkipPack | — | Pack opening |
| 8 | BuyCard | entity_target (shop index) | Shop |
| 9 | SellJoker | entity_target (joker index) | Blind select / Play / Shop / Pack opening |
| 10 | SellConsumable | entity_target (consumable index) | Blind select / Play / Shop / Pack opening |
| 11 | UseConsumable | entity_target + card_target | Any phase but game over |
| 12 | RedeemVoucher | entity_target (shop index) | Shop |
| 13 | OpenBooster | entity_target (shop index) | Shop |
| 14 | PickPackCard | entity_target + card_target | Pack opening |
| 15 | SwapJokersLeft | entity_target (joker index) | Play / Shop |
| 16 | SwapJokersRight | entity_target (joker index) | Play / Shop |
| 17 | SwapHandLeft | entity_target (card index) | Playing phase |
| 18 | SwapHandRight | entity_target (card index) | Playing phase |
| 19 | SortHandRank | — | Playing phase |
| 20 | SortHandSuit | — | Playing phase |

`UseConsumable` and `PickPackCard` take a `card_target` when the consumable works on hand cards (The Magician, Death, Aura...). Bound it with `get_consumable_target_info(card)`, which returns `(min_cards, max_cards, needs_targets)`. Leave `card_target` as `None` to get the engine's default selection: the lowest-index legal cards.

## Environment Config

```python
env = BalatroEnvironment(
    adapter_factory=DirectAdapter,
    back_keys=["b_red"],       # deck backs to sample from on reset
    stakes=[1],                # stakes to sample from on reset
    max_steps=10_000,          # steps before truncation
    seed_prefix="TRAIN",       # prefix for deterministic seed generation
    end_on_win=True,           # end the episode when ante 8 is beaten
)
```

With `end_on_win=True` (the default) a won run ends the episode, as vanilla stops at its win screen. Pass `end_on_win=False` to keep playing into endless mode until the run is lost or truncated.

## Reward

`BalatroEnvironment` does **not** compute rewards for you. `step()` returns `terminated` and `truncated` bools, and `info` carries the game state before and after the action:

```python
info["prev_state"]  # GameState snapshot taken before this step
info["state"]       # GameState snapshot taken after this step
info["raw_state"]   # the live engine state dict
```

`GameState` is a small frozen snapshot: `phase`, `ante`, `round`, `dollars`, `hands_left`, `discards_left`, `hand_size`, `joker_slots`, `consumable_slots`, `blind_on_deck`, `blind_chips`, `chips`, `won`, `done`. Compare `prev_state` with `state` for before/after deltas. `raw_state` is the engine's own dict, which the next step mutates in place, so read from it immediately and don't hold on to it.

The easiest route is a reward function from `jackdaw.env.rewards` (see [Rewards](#rewards)). Wrap each step in a `Transition`:

```python
from jackdaw.env import Transition, blind_progress_reward

obs, terminated, truncated, mask, info = env.step(action)
reward = blind_progress_reward(Transition.from_step(action, terminated, truncated, info))
```

The environment also tracks per-episode stats:

```python
env.episode_length   # steps taken this episode
env.episode_won      # True if the run was won
env.episode_ante     # highest ante reached
```

## Demo: Random Agent

Here's a complete working example using the built-in `RandomAgent`:

```python
from jackdaw.env import BalatroEnvironment, DirectAdapter, RandomAgent

env = BalatroEnvironment(adapter_factory=DirectAdapter)
agent = RandomAgent()

for episode in range(10):
    obs, mask, info = env.reset()
    agent.reset()

    terminated = False
    truncated = False

    while not (terminated or truncated): 
        action = agent.act(
            obs={"global": obs.global_context, **obs.entities},
            action_mask=mask,
            info=info,
        )
        obs, terminated, truncated, mask, info = env.step(action)

    print(f"Episode {episode}: ante={env.episode_ante}, won={env.episode_won}, steps={env.episode_length}")
```

## Demo: Custom Agent Skeleton

The `Agent` protocol requires two methods:

```python
from jackdaw.env import Agent, FactoredAction, ActionMask
import numpy as np

class MyAgent:
    def reset(self) -> None:
        """Called at the start of each episode."""
        pass

    def act(self, obs: dict, action_mask: ActionMask, info: dict) -> FactoredAction:
        """Select an action given observation and mask."""
        # 1. Which action types are legal?
        legal_types = np.nonzero(action_mask.type_mask)[0]

        # 2. Pick an action type (your model goes here)
        action_type = int(legal_types[0])  # placeholder

        # 3. Fill in targets based on action type
        entity_target = None
        card_target = None

        # Entity-targeted actions (buy, sell, use, etc.)
        if action_type in action_mask.entity_masks:
            legal_entities = np.nonzero(action_mask.entity_masks[action_type])[0]
            entity_target = int(legal_entities[0])  # placeholder

        # Card-selecting actions (play hand, discard)
        if action_type in (0, 1):  # PlayHand or Discard
            legal_cards = np.nonzero(action_mask.card_mask)[0]
            n = min(len(legal_cards), action_mask.max_card_select)
            card_target = tuple(int(i) for i in legal_cards[:n])

        return FactoredAction(
            action_type=action_type,
            card_target=card_target,
            entity_target=entity_target,
        )
```

## Gymnasium Wrapper (SB3 / MaskablePPO)

If you want to use standard RL libraries like Stable-Baselines3 instead of writing a custom training loop, Jackdaw provides `BalatroGymnasiumEnv` — a `gymnasium.Env` subclass with a flat `Discrete(500)` action space and action masking for [MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).

### Install

```bash
uv sync --extra train   # installs sb3-contrib, stable-baselines3, torch, tensorboard
```

### How It Works

The wrapper sits between SB3 and the factored `BalatroEnvironment`:

1. **Action flattening**: Each step, it enumerates all legal `FactoredAction` instances from the `ActionMask` into a flat list. The agent picks an index into this list. Card-selecting actions enumerate all legal card combinations up to a budget of 200 per action: PlayHand and Discard, plus each targeting Tarot/Spectral used from the consumable slots or picked from a pack. If the combinatorial space exceeds the budget, subsets are randomly sampled.

2. **Action masking**: The `action_masks()` method returns a `bool[500]` array — `True` for indices that map to a legal action, `False` for padding. MaskablePPO uses this to zero out logits for illegal actions.

3. **Observation space**: `gymnasium.spaces.Dict` with zero-padded entity arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `global` | `(235,)` | Fixed-size game state vector |
| `hand_card` | `(8, 15)` | Hand cards, zero-padded to max 8 |
| `joker` | `(5, 15)` | Jokers, zero-padded to max 5 |
| `consumable` | `(2, 7)` | Consumables, zero-padded to max 2 |
| `shop_item` | `(10, 9)` | Shop items, zero-padded to max 10 |
| `pack_card` | `(5, 15)` | Pack cards, zero-padded to max 5 |
| `entity_counts` | `(5,)` | Actual count per entity type |

### Basic Usage

```python
from jackdaw.env import BalatroGymnasiumEnv, DirectAdapter, ShapedReward

env = BalatroGymnasiumEnv(
    adapter_factory=DirectAdapter,
    reward_fn=ShapedReward(),   # dense multi-signal reward, see Rewards below
    max_steps=10_000,
)

obs, info = env.reset()
mask = env.action_masks()  # bool[500]

# Pick a legal action
import numpy as np
legal = np.nonzero(mask)[0]
action = int(np.random.choice(legal))
obs, reward, terminated, truncated, info = env.step(action)
```

### Training with MaskablePPO

```python
from sb3_contrib import MaskablePPO
from jackdaw.env import BalatroGymnasiumEnv, DirectAdapter, ShapedReward

env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, reward_fn=ShapedReward())

model = MaskablePPO("MultiInputPolicy", env, verbose=1, tensorboard_log="runs/balatro_ppo")
model.learn(total_timesteps=500_000)
model.save("balatro_ppo")
```

Or use the included training script:

```bash
python scripts/train_ppo.py --total-timesteps 500000 --seed 42
tensorboard --logdir runs/balatro_ppo
```

The training script logs Balatro-specific metrics to tensorboard: `balatro/mean_ante_reached`, `balatro/max_ante_reached`, `balatro/mean_rounds_beaten`, and `balatro/win_rate`.

### Rewards

The reward is pluggable: pass any callable `reward_fn(transition) -> float`. Three examples ship in `jackdaw.env.rewards`, from sparse to dense:

| Reward | Signal |
|--------|--------|
| `sparse_reward` (default) | +1 when the episode ends with the run won, -1 otherwise |
| `blind_progress_reward` | +1 per blind beaten, -1 when the episode ends short of a win |
| `ShapedReward()` | Dense multi-signal reward with tunable weights (table below) |

`reward_shaping=True` is shorthand for `reward_fn=ShapedReward()`.

#### Writing your own

A reward function receives a `Transition`:

```python
t.prev, t.curr          # GameState snapshots before / after the action
t.action                # the FactoredAction taken
t.terminated, t.truncated, t.done
t.raw_state             # live engine dict after the step (read now, don't keep)

# Derived events
t.blind_beaten          # this step's hand cleared a blind
t.boss_beaten           # ... and it was the boss
t.blind_ante            # ante of the blind being played
t.chips_scored          # chips this step added toward the blind
t.run_won               # this step won the run (fires once)
```

Prefer the derived events to raw counters. `round` increments when a blind is *selected*, not beaten, and `ante` advances on the play that beats the boss, so the obvious checks fire on the wrong steps.

```python
from jackdaw.env import BalatroGymnasiumEnv, DirectAdapter, Transition

def money_aware_reward(t: Transition) -> float:
    reward = 1.0 if t.blind_beaten else 0.0
    reward += 0.01 * max(t.curr.dollars - t.prev.dollars, 0)  # money earned
    if t.done and not t.curr.won:
        reward -= 1.0
    return reward

env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, reward_fn=money_aware_reward)
```

A reward function that keeps per-episode state can define a `reset()` method, which the env calls on every reset. Build one instance per env, for example inside the env factory of a vectorized setup.

#### ShapedReward

| Field | Trigger | Default |
|-------|---------|---------|
| `step_cost` | Every action | -0.001 |
| `shop_step_cost` | Every action taken in the shop (instead) | -0.002 |
| `blind_beaten` | The hand that clears a blind, × (ante / 8) | +0.15 |
| `boss_beaten` | Extra when that blind is a boss, × (ante / 8) | +0.10 |
| `hands_left` | Per hand left after the clearing hand | +0.01 |
| `score_progress` | × min(chips scored / blind target, 1) | +0.02 |
| `win` | The run is won | +0.5 |
| `loss` | The episode ends short of a win | -0.2 |

`ante` is the ante of the blind just beaten. Override any weight by keyword: `ShapedReward(win=1.0, step_cost=0.0)`. The penalties (`step_cost`, `shop_step_cost`, `loss`) are positive magnitudes that get subtracted, so write `loss=0.5`, not `-0.5`. The step cost discourages stalling in shops and menus, while the blind-beaten and score-progress signals give credit for partial progress even when the agent fails to clear a blind.

By default, beating ante 8 ends the episode (`end_on_win=True`), so `run_won` and `done` arrive on the same step. With `end_on_win=False`, play continues into endless mode: `run_won` still fires once on the winning step, and a run that dies later in endless mode still counts as won.

### Episode-End Metrics

At episode end, the `info` dict returned by `step()` includes:

```python
info["balatro/ante_reached"]    # int — highest ante reached this episode
info["balatro/rounds_beaten"]   # int — blinds beaten (skipped blinds don't count)
info["balatro/won"]             # bool — whether the run was won
```

These are tracked the same way whatever reward function is in use, and are useful for custom callbacks or logging.

## Live Validation Adapter

To validate your agent against real Balatro, swap the adapter:

```python
from jackdaw.bridge import LiveBackend, BridgeAdapter

env = BalatroEnvironment(
    adapter_factory=lambda: BridgeAdapter(LiveBackend("127.0.0.1", 12346))
)
# Same observations and actions; the mask is narrowed to what BalatroBot executes
```

You must have Balatro running with BalatroBot configured to listen on the same host and port.

The simulator follows vanilla, which allows a few things BalatroBot's endpoints refuse: using a consumable at blind select, on the cash-out screen or while a pack is open, selling while a pack is open or at blind select, and using a card-targeting consumable in the shop. Over a `BridgeAdapter` those actions are masked out (and `SimBackend` refuses them the same way), so an agent is never offered a move the bot can't make.

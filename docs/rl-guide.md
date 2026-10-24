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

`DirectAdapter` runs the game on the Rust engine (`jackdaw.core.Game`) in process, the adapter to train on. `LiveAdapter` plays the live game instead (see [Live Validation Adapter](#live-validation-adapter)).

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
mask.type_mask        # np.ndarray, shape (23,) bool — which action types are legal
mask.card_mask        # np.ndarray, shape (N_hand,) bool — which hand cards can be selected
mask.entity_masks     # dict[int, np.ndarray] — per action type, which entities are legal targets
mask.min_card_select  # int — minimum cards to select (for play/discard)
mask.max_card_select  # int — maximum cards to select
mask.forced_card      # int | None — hand index every play/discard must include (Cerulean Bell)
```

Always respect the masks. If `type_mask[i]` is False, don't emit action type `i`. Against the Cerulean Bell, a play or discard that leaves out `forced_card` is illegal, as in vanilla, where the forced card can't be deselected.

## Actions

Actions are `FactoredAction` with three components:

```python
from jackdaw.env import FactoredAction

action = FactoredAction(
    action_type=0,                  # int: which action (0-22)
    card_target=(0, 2, 4),          # tuple[int, ...] | None: hand card indices
    entity_target=None,             # int | None: entity index
)
```

### The 23 Action Types

| ID | Name | Targets | When |
|----|------|---------|------|
| 0 | PlayHand | card_target (1-5 cards) | Playing phase |
| 1 | Discard | card_target (1-5 cards) | Playing phase |
| 2 | SelectBlind | — | Blind select |
| 3 | SkipBlind | — | Blind select |
| 4 | CashOut | — | Round eval |
| 5 | Reroll | — | Shop |
| 6 | NextRound | — | Shop |
| 7 | SkipPack | — | Pack opening (an Arcana or Spectral pack once its hand is dealt) |
| 8 | BuyCard | entity_target (shop index) | Shop |
| 9 | SellJoker | entity_target (joker index) | Blind select / Play / Round eval / Shop / Pack opening |
| 10 | SellConsumable | entity_target (consumable index) | Blind select / Play / Round eval / Shop / Pack opening |
| 11 | UseConsumable | entity_target + card_target | Any phase but game over |
| 12 | RedeemVoucher | entity_target (shop index) | Shop |
| 13 | OpenBooster | entity_target (shop index) | Shop |
| 14 | PickPackCard | entity_target + card_target | Pack opening |
| 15 | SwapJokersLeft | entity_target (joker index) | Any phase but game over |
| 16 | SwapJokersRight | entity_target (joker index) | Any phase but game over |
| 17 | SwapHandLeft | entity_target (card index) | Playing phase / Pack opening (an Arcana or Spectral pack's dealt hand) |
| 18 | SwapHandRight | entity_target (card index) | Playing phase / Pack opening (an Arcana or Spectral pack's dealt hand) |
| 19 | SortHandRank | — | Playing phase |
| 20 | SortHandSuit | — | Playing phase |
| 21 | RerollBoss | — | Blind select, with Director's Cut or Retcon ($10) |
| 22 | BuyAndUse | entity_target (shop index) | Shop: buy a consumable and use it at once |

Types 21 and 22 were added after the first 21, so ids 0-20 keep their meaning; a policy with one logit per type needs two more outputs.

The swaps follow the game's drag and drop: the jokers can be reordered whenever the game waits for a move, the hand whenever it holds cards (while choosing a hand, and in an Arcana or Spectral pack, where the order decides which card Death copies and the order the cards go back into the deck). The sort buttons exist only while choosing a hand. Consumables can be dragged in the game too, but their order changes nothing, so there is no action for it. Under BALATROBOT legality (the live bot) the jokers can't be reordered at blind select or on the cash-out screen: BalatroBot's `rearrange` refuses there.

`UseConsumable` and `PickPackCard` take a `card_target` when the consumable works on hand cards (The Magician, Death, Aura...). Bound it with `get_consumable_target_info(info["game"], "consumables", index)` (or `"pack_cards"`), which returns `(min_cards, max_cards, needs_targets)`. Leave `card_target` as `None` to get the engine's default selection: the lowest-index legal cards.

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
info["game"]        # the run itself, a jackdaw.core.Game
```

`GameState` is a small frozen snapshot: `phase`, `ante`, `round`, `dollars`, `hands_left`, `discards_left`, `hand_size`, `joker_slots`, `consumable_slots`, `blind_on_deck`, `blind_chips`, `chips`, `won`, `done`. Compare `prev_state` with `state` for before/after deltas. `info["game"]` is the live run, which the next step moves on, so read from it immediately or keep `info["game"].freeze()`.

A step the Rust engine can't take yet leaves the run unchanged and ends the episode as truncated, with the reason in `info["not_ported"]`.

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
            selected = [int(i) for i in legal_cards[:n]]
            forced = action_mask.forced_card  # Cerulean Bell
            if forced is not None and forced not in selected:
                selected[-1] = forced
            card_target = tuple(sorted(selected))

        return FactoredAction(
            action_type=action_type,
            card_target=card_target,
            entity_target=entity_target,
        )
```

## Gymnasium Wrapper (SB3 / MaskablePPO)

If you want to use standard RL libraries like Stable-Baselines3 instead of writing a custom training loop, Jackdaw provides `BalatroGymnasiumEnv` — a `gymnasium.Env` subclass with a flat `Discrete(500)` action space and action masking for [MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html).

### How It Works

The wrapper sits between SB3 and the factored `BalatroEnvironment`:

1. **Action flattening**: Each step, it enumerates all legal `FactoredAction` instances from the `ActionMask` into a flat list. The agent picks an index into this list. Card-selecting actions enumerate all legal card combinations up to a budget of 200 per action: PlayHand and Discard, plus each targeting Tarot/Spectral used from the consumable slots or picked from a pack. If the combinatorial space exceeds the budget, subsets are randomly sampled, as is the whole list when it exceeds 500 actions. The sampler is a xoshiro256** generator seeded by `reset(seed=...)`, the same in Python and Rust.

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

### With MaskablePPO

Jackdaw doesn't depend on any RL library. Install [sb3-contrib](https://sb3-contrib.readthedocs.io/) in your own project, then:

```python
from sb3_contrib import MaskablePPO
from jackdaw.env import BalatroGymnasiumEnv, DirectAdapter, ShapedReward

env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, reward_fn=ShapedReward())

model = MaskablePPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)
model.save("balatro_ppo")
```

The repository's [`examples/`](https://github.com/TylerFlar/jackdaw-balatro/tree/main/examples) trains this setup for real: 32 envs in SB3's `DummyVecEnv`, TensorBoard logs, checkpoints, and an evaluation against a random agent and the hand-written `smart_agent` on the same seeds. In a clone, `uv sync --group train` installs what it needs.

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
t.game                  # the run after the step, a jackdaw.core.Game

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

Mind the discount factor: a policy that puts off the end of a lost run gains `loss * (1 - gamma)` per step from the discounted `loss`, and free moves (sorting or swapping cards) can put it off forever. With the default weights that gain matches `step_cost` at `gamma = 0.995` and exceeds it below, so stalling costs nothing or pays. A higher `gamma`, a smaller `loss` or a larger `step_cost` removes the incentive; the [training demo](https://github.com/TylerFlar/jackdaw-balatro/tree/main/examples) uses `loss=0`.

By default, beating ante 8 ends the episode (`end_on_win=True`), so `run_won` and `done` arrive on the same step. With `end_on_win=False`, play continues into endless mode: `run_won` still fires once on the winning step, and a run that dies later in endless mode still counts as won.

### Episode-End Metrics

At episode end, the `info` dict returned by `step()` includes:

```python
info["balatro/ante_reached"]    # int — highest ante reached this episode
info["balatro/rounds_beaten"]   # int — blinds beaten (skipped blinds don't count)
info["balatro/won"]             # bool — whether the run was won
info["balatro/not_ported"]      # str — only when the Rust engine couldn't take the step
```

These are tracked the same way whatever reward function is in use, and are useful for custom callbacks or logging.

## Vector Environment

`BalatroVectorEnv` is a `gymnasium.vector.VectorEnv` of `num_envs` Gym environments stepped together in Rust: one `step` runs them all on worker threads with the GIL released and writes the observations and action masks straight into numpy arrays. On a 16-core machine it runs about 350K steps per second on 8 threads, against about 75K for one `BalatroGymnasiumEnv`.

```python
import numpy as np
from jackdaw.env import BalatroVectorEnv

envs = BalatroVectorEnv(num_envs=256, num_threads=8, reward_shaping=True)
obs, info = envs.reset(seed=0)
masks = info["action_mask"]                    # (256, 500) bool, also envs.action_masks()

for _ in range(1000):
    actions = np.array([np.random.choice(np.flatnonzero(m)) if m.any() else 0 for m in masks])
    obs, rewards, terminated, truncated, info = envs.step(actions)
    masks = info["action_mask"]
```

- Sub-env `i` plays exactly as `BalatroGymnasiumEnv(DirectAdapter, seed_prefix=f"{seed_prefix}_{i}")` with the same arguments; `reset(seed=s)` gives it seed `s + i`. The results don't depend on the number of threads.
- Episodes reset on the step after they end (gymnasium's `AutoresetMode.NEXT_STEP`): that step ignores the env's action and returns the new episode's first observation with reward 0. Until then the ended env's mask is all False.
- The rewards are computed in Rust, so only the built-in ones are available: `sparse_reward` (the default), `blind_progress_reward` and `ShapedReward` with any weights.
- The episode-end metrics come as arrays with gymnasium's `_key` masks: `info["balatro/won"]` and `info["_balatro/won"]`, and so on.
- `copy=False` returns views of buffers the next step overwrites, skipping a copy per step.

## Live Validation Adapter

To validate your agent against real Balatro, swap the adapter:

```python
from jackdaw.bridge import LiveAdapter, LiveBackend
from jackdaw.env import BalatroEnvironment

env = BalatroEnvironment(
    adapter_factory=lambda: LiveAdapter(LiveBackend("127.0.0.1", 12346), bundle_dir="bundles")
)
# Same observations and actions; the mask is narrowed to what BalatroBot executes
```

You must have Balatro running with BalatroBot configured to listen on the same host and port. `LiveAdapter` (also what `BridgeAdapter(backend)` gives) keeps a shadow run in the Rust engine: every action is taken by the shadow first, then by the game, and the game's answer is compared with the shadow. `info["game"]` is the shadow, and the observation and mask come from it, exactly as in training. An unexplained difference raises `DivergenceError` and writes a repro bundle; see [Lockstep against the live game](live). With `SimBackend()` in place of the `LiveBackend`, the same adapter plays BalatroBot's API on the simulator, in process.

The simulator follows vanilla, which allows a few things BalatroBot's endpoints refuse: using a consumable at blind select, on the cash-out screen or while a pack is open, selling while a pack is open or at blind select, and using a card-targeting consumable in the shop. Over a `LiveAdapter` those actions are masked out (and `SimBackend`, which runs BalatroBot's endpoints on the simulator, refuses them the same way), so an agent is never offered a move the bot can't make. `Game.legal_actions(Legality.BALATROBOT)` gives the same list from the Rust engine. See [BalatroBot on the simulator](balatrobot) for the full policy and `jackdaw serve`.

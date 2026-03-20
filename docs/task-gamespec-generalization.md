# Task: Generalize RL Pipeline with GameSpec Abstraction

## Goal

Refactor the RL pipeline so the neural network, training loop, and policy
architecture are fully game-agnostic. Game-specific logic (Balatro observations,
actions, rewards) lives behind a `GameSpec` interface that any card game can
implement.

The key insight: the current architecture already follows the right pattern
(variable-length entities → transformer → factored actions). The coupling is
in hardcoded constants and data schemas, not in the neural net design.

## Architecture After Refactoring

```
┌─────────────────────────────────────────────────────────────────┐
│ Generic RL Layer (game-agnostic)                                │
│   policy/transformer.py  — TransformerCore                      │
│   policy/entity_encoder.py — EntityEncoder (parametric)         │
│   policy/action_heads.py — ActionHeads (parametric)             │
│   policy/policy.py — GenericPolicy                              │
│   training/ppo.py — PPOTrainer, RolloutBuffer                   │
│   training/curriculum.py — CurriculumManager                    │
├─────────────────────────────────────────────────────────────────┤
│ GameSpec Interface                                              │
│   game_spec.py — GameSpec protocol                              │
│   Defines: entity types, action types, feature dims, masks      │
├─────────────────────────────────────────────────────────────────┤
│ Balatro Implementation (game-specific)                          │
│   balatro/spec.py — BalatroGameSpec(GameSpec)                    │
│   balatro/observation.py — encode_observation (Balatro-specific) │
│   balatro/actions.py — action masking, conversion               │
│   balatro/rewards.py — reward shaping                           │
│   balatro/features.py — ★ EXTENSION POINT for enhanced features │
│   env/game_interface.py — GameAdapter (already exists)          │
└─────────────────────────────────────────────────────────────────┘
```

The **Balatro Implementation** layer is where game-specific feature
engineering goes. After this refactoring, you can add Balatro-specific
observation improvements (hand type detection, scoring potential,
blind-relative strength) in `balatro/features.py` without touching
any generic RL code.

## GameSpec Protocol

Create `jackdaw/env/game_spec.py`:

```python
@dataclass
class EntityTypeSpec:
    """Describes one type of entity in the game."""
    name: str                  # e.g. "hand_card", "minion", "spell"
    feature_dim: int           # dimension of raw feature vector
    max_count: int             # max entities of this type (for pre-allocation)
    has_catalog_id: bool       # whether entities have a categorical ID (like center_key)
    catalog_size: int = 0      # number of unique IDs if has_catalog_id

@dataclass
class ActionTypeSpec:
    """Describes one type of action in the game."""
    name: str                  # e.g. "play_hand", "summon_creature"
    needs_entity_target: bool  # requires selecting an entity
    needs_card_select: bool    # requires multi-selecting cards
    entity_type_index: int = -1  # which entity type the pointer targets (-1 = none)

@dataclass
class GameSpec:
    """Complete specification of a game's RL interface."""
    name: str
    entity_types: list[EntityTypeSpec]
    action_types: list[ActionTypeSpec]
    global_feature_dim: int    # dimension of global context vector
    max_card_select: int = 5   # maximum cards selectable in one action

    @property
    def num_entity_types(self) -> int:
        return len(self.entity_types)

    @property
    def num_action_types(self) -> int:
        return len(self.action_types)

    @property
    def needs_entity_set(self) -> frozenset[int]:
        return frozenset(i for i, a in enumerate(self.action_types) if a.needs_entity_target)

    @property
    def needs_cards_set(self) -> frozenset[int]:
        return frozenset(i for i, a in enumerate(self.action_types) if a.needs_card_select)
```

## GameEnvironment Protocol

Create alongside GameSpec — this is the runtime interface that wraps
a game instance and produces observations/masks:

```python
@dataclass
class GameObservation:
    """Game-agnostic observation container."""
    global_context: np.ndarray                    # (D_global,)
    entities: dict[str, np.ndarray]               # entity_name -> (N_i, D_i)

@dataclass
class GameActionMask:
    """Game-agnostic action mask container."""
    type_mask: np.ndarray                         # (num_action_types,) bool
    card_mask: np.ndarray                         # (max_hand,) bool
    entity_masks: dict[int, np.ndarray]           # action_type -> (N_entity,) bool
    min_card_select: int = 0
    max_card_select: int = 5

class GameEnvironment(Protocol):
    """What the RL pipeline needs from any game."""

    @property
    def spec(self) -> GameSpec: ...

    def reset(self, **kwargs) -> tuple[GameObservation, GameActionMask, dict]: ...

    def step(self, action: FactoredAction) -> tuple[
        GameObservation, float, bool, bool, GameActionMask, dict
    ]: ...
```

## What Changes in Each File

### 1. entity_encoder.py — Parametric Entity Types

Current: 5 hardcoded entity types (HAND=0, JOKER=1, CONSUMABLE=2, SHOP=3, PACK=4)

Change to:
- Accept `entity_types: list[EntityTypeSpec]` in `__init__`
- Dynamically create one MLP per entity type based on `feature_dim`
- Conditionally create catalog embedding based on `has_catalog_id`
- Forward takes `entities: dict[str, Tensor]` + `masks: dict[str, Tensor]`
  instead of 5 named arguments

Keep the same pattern: per-type MLP → concat → shared sequence.

### 2. action_heads.py — Parametric Action Types

Current: `NUM_ACTION_TYPES=21`, `NEEDS_ENTITY=frozenset(range(8,19))`,
`NEEDS_CARDS=frozenset({0, 1})`

Change to:
- Accept `game_spec: GameSpec` in `__init__`
- Use `game_spec.num_action_types` for type head output dim
- Use `game_spec.needs_entity_set` and `game_spec.needs_cards_set`
- No other changes needed — pointer attention and card logits are generic

### 3. policy.py — GenericPolicy + Parametric Collation

Current: `BalatroPolicy` hardcodes entity types, `collate_policy_inputs`
hardcodes 5 entity arrays and 21-action pointer masks.

Change to:
- Rename to `CardGamePolicy` (or keep `BalatroPolicy` as alias)
- Accept `game_spec: GameSpec` in `__init__` (pass to sub-modules)
- `collate_policy_inputs` iterates `game_spec.entity_types` instead
  of hardcoding hand/joker/consumable/shop/pack
- Pointer mask construction driven by `ActionTypeSpec.entity_type_index`

### 4. observation.py — Split into Generic Container + Balatro Encoder

Current: `encode_observation()` is 800 lines of Balatro-specific feature
extraction, plus `Observation` dataclass with 5 named entity arrays.

Split into:
- **Generic** `GameObservation` dataclass (in `game_spec.py`) — just
  `global_context` + `entities: dict[str, ndarray]`
- **Balatro-specific** `encode_observation()` stays in `jackdaw/env/observation.py`
  (or move to `jackdaw/env/balatro/observation.py`) — returns `GameObservation`

The current `Observation` dataclass becomes a convenience wrapper that
constructs `GameObservation`:
```python
def to_game_observation(self) -> GameObservation:
    return GameObservation(
        global_context=self.global_context,
        entities={
            "hand_card": self.hand_cards,
            "joker": self.jokers,
            "consumable": self.consumables,
            "shop_item": self.shop_cards,
            "pack_card": self.pack_cards,
        },
    )
```

### 5. action_space.py — Split into Generic + Balatro

Current: `ActionType` enum (21 Balatro types), `FactoredAction`,
`ActionMask`, `get_action_mask()`, factored↔engine conversion.

Split into:
- **Generic** `FactoredAction` dataclass — keep as-is (action_type int +
  entity_target + card_target). This is game-agnostic.
- **Generic** `GameActionMask` — same pattern as current `ActionMask` but
  driven by `GameSpec.num_action_types`
- **Balatro-specific** `ActionType` enum, `get_action_mask()`,
  `factored_to_engine_action()` — stay in Balatro module

### 6. training/ppo.py — Use GameSpec

Current: `_EnvInstance` calls `encode_observation()` and `get_action_mask()`
directly.

Change to:
- `_EnvInstance` wraps a `GameEnvironment` instead of a `GameAdapter`
- `PPOConfig` takes `game_spec: GameSpec` (instead of inferring from imports)
- Policy construction uses `game_spec` to determine dimensions
- Everything else stays the same — PPO math doesn't care about the game

### 7. Balatro GameSpec Implementation

Create `jackdaw/env/balatro_spec.py` (or similar):

```python
def balatro_game_spec() -> GameSpec:
    return GameSpec(
        name="balatro",
        entity_types=[
            EntityTypeSpec("hand_card", D_PLAYING_CARD, max_count=8, has_catalog_id=False),
            EntityTypeSpec("joker", D_JOKER, max_count=5, has_catalog_id=True, catalog_size=NUM_CENTER_KEYS),
            EntityTypeSpec("consumable", D_CONSUMABLE, max_count=2, has_catalog_id=True, catalog_size=NUM_CENTER_KEYS),
            EntityTypeSpec("shop_item", D_SHOP, max_count=10, has_catalog_id=True, catalog_size=NUM_CENTER_KEYS),
            EntityTypeSpec("pack_card", D_PLAYING_CARD, max_count=5, has_catalog_id=False),
        ],
        action_types=[
            ActionTypeSpec("play_hand", needs_entity_target=False, needs_card_select=True),
            ActionTypeSpec("discard", needs_entity_target=False, needs_card_select=True),
            ActionTypeSpec("select_blind", needs_entity_target=False, needs_card_select=False),
            # ... all 21 Balatro actions
        ],
        global_feature_dim=D_GLOBAL,  # 211
    )
```

## Extension Point: Balatro Feature Enhancements

After this refactoring, improving Balatro observations is purely local:

```
jackdaw/env/observation.py  (or balatro/observation.py)
    ↓ modify encode_observation()
    ↓ add hand-type detection, scoring potential, etc.
    ↓ increase D_GLOBAL or add new entity features
    ↓ update balatro_game_spec() dimensions to match
```

Examples of Balatro-specific features to add AFTER generalization:
- Hand poker type indicators (pair? flush? straight?) → add to D_PLAYING_CARD or D_GLOBAL
- Score-to-blind ratio (how close am I to beating this blind?) → add to D_GLOBAL
- Joker synergy features (which jokers combo with current hand?) → add to D_JOKER
- Economy trajectory (am I on track for interest?) → add to D_GLOBAL
- Discard efficiency (expected hand improvement from discarding) → add to D_GLOBAL

Each of these just changes the Balatro observation encoder and spec dimensions.
The generic policy, encoder, and training loop adapt automatically via GameSpec.

## File Layout After Refactoring

```
jackdaw/env/
    game_spec.py              # NEW: GameSpec, EntityTypeSpec, ActionTypeSpec, GameObservation, GameActionMask, GameEnvironment protocol
    game_interface.py          # KEEP: GameAdapter protocol (Balatro engine interface)
    observation.py             # MODIFY: encode_observation returns GameObservation, keep Balatro logic here
    action_space.py            # MODIFY: FactoredAction stays generic; ActionType enum, masks, conversions stay Balatro-specific
    balatro_spec.py            # NEW: balatro_game_spec() factory
    balatro_env.py             # NEW: BalatroEnvironment(GameEnvironment) wrapping GameAdapter + observation + masks + rewards
    rewards.py                 # KEEP: Balatro reward shaping (game-specific, behind GameEnvironment.step)

    policy/
        policy.py              # MODIFY: accept GameSpec, parametric collation
        entity_encoder.py      # MODIFY: accept list[EntityTypeSpec]
        action_heads.py        # MODIFY: accept GameSpec for action count + entity/card sets
        transformer.py         # KEEP: already generic (just parametrize global_dim)

    training/
        ppo.py                 # MODIFY: _EnvInstance wraps GameEnvironment, PPOConfig takes GameSpec
        curriculum.py          # KEEP: game-agnostic
        sweep.py               # KEEP: game-agnostic
```

## Migration Strategy

Do this incrementally to keep tests passing at each step:

### Phase 1: Create GameSpec + GameObservation (additive, no breaking changes)
1. Create `game_spec.py` with all dataclasses and protocols
2. Create `balatro_spec.py` with `balatro_game_spec()`
3. Add `to_game_observation()` to existing `Observation` class
4. All tests pass — nothing uses GameSpec yet

### Phase 2: Parametrize Neural Net (modify policy/, keep backward compat)
1. Add `game_spec` param to EntityEncoder, ActionHeads, Policy `__init__`
2. Keep old positional args as defaults for backward compat
3. When `game_spec` is provided, use it; otherwise fall back to hardcoded
4. Update policy construction in PPOTrainer to pass game_spec
5. All tests pass — existing code uses defaults

### Phase 3: Create BalatroEnvironment (new wrapper, parallel to _EnvInstance)
1. Create `balatro_env.py` with `BalatroEnvironment(GameEnvironment)`
2. It wraps GameAdapter + encode_observation + get_action_mask + DenseRewardWrapper
3. Update `_EnvInstance` to accept `GameEnvironment` instead of raw components
4. All tests pass — same behavior, new interface

### Phase 4: Remove Hardcoded Defaults (clean up)
1. Remove fallback paths in EntityEncoder, ActionHeads, Policy
2. Make `game_spec` required (not optional)
3. Update all tests to pass `balatro_game_spec()`
4. Clean up imports

## Testing

- All 1,562+ existing tests must pass at each phase
- Add new tests for GameSpec construction and validation
- Add test that `balatro_game_spec()` dimensions match current hardcoded values
- Training smoke test: `--quick` produces same results before and after

## Success Criteria

- Policy network has ZERO Balatro imports
- EntityEncoder, ActionHeads, TransformerCore have ZERO Balatro imports
- PPOTrainer has ZERO Balatro imports (gets game via GameSpec/GameEnvironment)
- Balatro-specific code isolated to observation.py, action_space.py, balatro_spec.py, balatro_env.py, rewards.py
- Adding a new entity feature to Balatro observation requires changing ONLY observation.py + balatro_spec.py
- A hypothetical Hearthstone port would implement HearthstoneGameSpec + HearthstoneEnvironment, reuse all policy/ and training/ code unchanged

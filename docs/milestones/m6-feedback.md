# M6 Feedback: Joker Framework + Simple/Conditional Jokers

## 1. Implementation coverage

**71 joker handlers registered and tested** out of ~150 total in the source.
This covers all jokers that fire during the scoring pipeline (Phases 5, 7, 8, 9)
with deterministic or pre-computed state:

- 3 simple unconditional: Joker, Misprint, Stuntman
- 17 hand-type conditional (poker_hands containment): Jolly/Zany/Mad/Crazy/Droll,
  Sly/Wily/Clever/Devious/Crafty, Duo/Trio/Family/Order/Tribe
- 2 scoring_name-based: Supernova, Card Sharp
- 9 suit-conditional (individual/play): Greedy/Lusty/Wrathful/Gluttonous,
  Arrowhead, Onyx Agate, Rough Gem, Bloodstone, Ancient
- 11 rank-conditional: Fibonacci, Scholar, Walkie Talkie, Even Steven, Odd Todd,
  Scary Face, Smiley Face, Photograph, Triboulet, The Idol, Hack (retrigger)
- 21 game-state-dependent: Half, Abstract, Acrobat, Mystic Summit, Banner,
  Blue Joker, Erosion, Stone Joker, Steel Joker, Bull, Driver's License,
  Blackboard, Joker Stencil, Flower Pot, Seeing Double, Bootstraps,
  Fortune Teller, Loyalty Card, Raised Fist, Shoot the Moon, Baron
- 7 economy: Golden Ticket, Business, Reserved Parking, Faceless, Mail,
  Trading, To Do List
- 1 boss reaction: Matador
- 2 copy/delegation: Blueprint, Brainstorm

**314 joker-related tests** (226 handler unit + 13 pipeline integration + 75 oracle).

## 2. scoring_name vs poker_hands containment

ALL hand-type jokers use `poker_hands` containment (not `scoring_name`):

- **t_mult group** (Jolly/Zany/Mad/Crazy/Droll): `next(poker_hands[type])`
- **t_chips group** (Sly/Wily/Clever/Devious/Crafty): `next(poker_hands[type])`
- **xMult group** (Duo/Trio/Family/Order/Tribe): `next(poker_hands[type])`

Only **Supernova** and **Card Sharp** use `scoring_name`.

**Surprise:** The user prompt initially categorized Jolly et al. as "Category A
(scoring_name)" and Duo et al. as "Category B (poker_hands)".  Source
verification showed ALL of them use poker_hands containment.  This means
j_jolly triggers on a Full House (which contains Pair via downward
propagation), not just when the detected hand IS a Pair.  This is the single
most common source of joker scoring bugs.

The Lua oracle confirms: `full_house_jolly` scores 960 (40+40=80 chips,
4+8=12 mult) because Jolly fires on the Pair sub-hand within Full House.

## 3. RNG-dependent jokers

Three probabilistic jokers implemented: Bloodstone, Business, Reserved Parking.
All follow the same pattern as Lucky Card:

- Handler accepts `ctx.rng` (PseudoRandom instance) and `ctx.probabilities_normal`
- With RNG: rolls `rng.random(seed_key)` and checks `< normal/odds`
- Without RNG: returns None (no effect), suitable for EV-mode where the
  expected value can be computed externally

Each has specific seed keys matching the source: `'bloodstone'`, `'business'`,
`'parking'`.

Testing approach:
- **High probability test**: `probabilities_normal=1000` guarantees trigger
- **No RNG test**: confirms None return
- **Deterministic test**: same seed produces same result
- **Non-matching suit/rank test**: confirms the probability check is gated
  behind the card condition, not evaluated unconditionally

Misprint also uses RNG: `rng.random('misprint', min_val=0, max_val=23)`.

## 4. Blueprint/Brainstorm copy mechanism

The dispatch table approach works cleanly with Blueprint/Brainstorm:

```python
target = _find_right_neighbor(card, ctx)  # or _find_leftmost for Brainstorm
new_ctx = replace(ctx, blueprint=bp, blueprint_card=ctx.blueprint_card or card)
return calculate_joker(target, new_ctx)
```

Key design decisions:
- Uses `dataclasses.replace` to create a shallow copy of JokerContext with
  incremented blueprint counter — immutable context per delegation
- Loop prevention: `blueprint > len(jokers) + 1` matches source exactly
- `blueprint_card` tracks the original copier (first in chain)
- Blueprint reads the target's current `card.ability` state — no separate
  accumulation.  A Blueprint copying a Green Joker that accumulated +20 mult
  will use that +20, not start from scratch.

Edge cases tested:
- Two adjacent Blueprints: right one has no target → left copies right → None
- Blueprint→Brainstorm→Joker chain: works, counter increments each hop
- Blueprint↔Brainstorm infinite loop: terminates via counter cap
- Debuffed target: returns None
- No joker list: returns None

## 5. Joker ordering

**Confirmed: order matters for mixed additive/multiplicative effects.**

Lua oracle validation (both at 100% match):

| Order | Phase 9 Computation | mult | Total |
|---|---|---|---|
| j_joker (+4) → j_blackboard (×3) | (2+4)×3 = 18 | 18 | 576 |
| j_blackboard (×3) → j_joker (+4) | (2×3)+4 = 10 | 10 | 320 |

Two xMult jokers commute: j_duo (×2) → j_blackboard (×3) and reversed both
give ×6.  But additive + multiplicative do NOT commute.

Edition effects also order correctly:
- Foil (+50 chips) applies BEFORE the joker's own effect in Phase 9a
- Polychrome (×1.5 mult) applies AFTER in Phase 9d
- Holo (+10 mult) is additive, applies in Phase 9a with Foil

## 6. Performance

| Pipeline | µs/hand | hands/sec | Notes |
|---|---|---|---|
| `score_hand_base` (no jokers) | 15 | 65,000 | M5 baseline |
| `score_hand` (5 jokers) | 164 | 6,100 | Pair + 5 diverse jokers |

The ~10× slowdown comes from:
- Building JokerContext (55-field dataclass) per call
- N_jokers × N_scoring_cards iterations for individual effects
- N_jokers iterations for repetition checks
- N_jokers × N_jokers for joker-on-joker in Phase 9c

At 6.1K hands/sec, scoring 218 subsets takes ~36ms.  This is adequate for
RL training (target: <100ms per action).  The main optimization opportunity
is reducing JokerContext construction overhead — either by pre-building a
shared context and mutating only the per-card fields, or by switching to
a slotted class.

## 7. JokerContext size

**55 fields** currently:
- 22 phase flags (bool/optional, exactly one set per call)
- 10 context data (cards, names, poker_hands, RNG, etc.)
- 5 joker meta-flags (smeared, pareidolia, probabilities, etc.)
- 16 game state (money, deck counts, tallies, etc.)
- 2 Blueprint fields (counter, card reference)

This is large but mirrors the source's `context` table which accumulates
all these fields dynamically.  The key difference: Lua's table is sparse
(only set fields exist), while our dataclass allocates all 55 fields.

**Restructuring recommendation for M7:** Split into sub-structures:

```python
@dataclass
class GameSnapshot:
    """Pre-computed game state, built once per score_hand call."""
    joker_count: int
    money: int
    deck_cards_remaining: int
    ...

@dataclass
class JokerContext:
    phase: str  # 'individual', 'joker_main', 'repetition', etc.
    cardarea: str | None
    other_card: Card | None
    game: GameSnapshot  # shared reference
    ...
```

This would reduce per-call allocation (GameSnapshot built once, shared
across all JokerContext instances in a scoring pass) while keeping the
flat field access pattern for handler simplicity.

## 8. Total joker coverage

**71 of ~150 jokers implemented** (47%).

Remaining ~79 jokers by category:
- **Scaling jokers** (~15): Green Joker, Red Card, Campfire, Ride the Bus,
  Runner, Ice Cream, Obelisk, Castle, Wee Joker, Vampire, etc.
  These mutate `card.ability` fields between hands.
- **End-of-round economy** (~10): Golden Joker, Cloud 9, Rocket, Delayed
  Gratification, etc.  Need round lifecycle.
- **Sell/buy triggers** (~5): Egg, Riff-raff, Luchador, etc.
- **Consumable interaction** (~8): Constellation, Cartomancer, Astronomer, etc.
- **Destruction/creation** (~10): Cavendish, Glass Joker, Ceremonial Dagger,
  Madness, etc.  Need card destruction pipeline.
- **Joker-on-joker** (~5): Baseball Card, Mime, Dusk, Sock and Buskin.
  Need `other_joker` context.
- **Complex state** (~15): Mr. Bones, Chicot, Luchador, Perkeo, Showman, etc.
  Need game state mutations beyond scoring.
- **Copy variants** (~3): already done (Blueprint, Brainstorm).
  Sock and Buskin (held retrigger) still needed.

## Modules built/updated in M6

| Module | New/Updated | Tests | Purpose |
|---|---|---|---|
| `jokers.py` | New | 226 | JokerContext + JokerResult + dispatch + 71 handlers |
| `scoring.py` | Updated | 13+75 | score_hand with Phases 5, 7b-d, 8b-c, 9 |
| Lua scoring oracle | Updated | 40 | 8 joker scenarios (6 joker types) |
| Integration tests | New | 13 | Full pipeline with joker effects |

**M6 total: ~280 new tests, 1,324 cumulative.**

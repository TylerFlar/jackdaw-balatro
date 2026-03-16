# M5 Feedback: Base Scoring Pipeline

## 1. Lua scoring oracle results

**7 scenarios validated, 100% match** on all 5 metrics (chips, mult, total,
hand_type, debuffed) across all 7 test cases = 35 cross-validation assertions.

| Scenario | Chips | Mult | Total | Feature Tested |
|---|---|---|---|---|
| pair_aces_basic | 32 | 2 | 64 | Baseline |
| three_kings_foil | 110 | 3 | 330 | Foil +50 chips |
| flush_glass | 71 | 8 | 568 | Glass ×2 mult |
| full_house_steel_held | 80 | 6 | 480 | Steel ×1.5 held |
| pair_aces_red_seal | 43 | 2 | 86 | Red Seal retrigger |
| pair_aces_flint | 27 | 1 | 27 | The Flint halving |
| pair_eye_debuffed | 0 | 0 | 0 | The Eye blocks repeat |

The Lua oracle script stubs all UI/animation functions and runs the source's
`eval_card` and `evaluate_poker_hand` directly.  Card objects are Lua tables
with all 8 scoring methods matching `card.lua:976-1089`.

## 2. Surprises in scoring order

**Edition effects are split across the pipeline.** The source applies edition
chip_mod and mult_mod (Foil/Holo) BEFORE x_mult_mod (Polychrome) within a
single card's evaluation.  But all three are applied within the same effect
block (lines 759-776), not separated into different phases.  This means
Foil's +50 chips and Holo's +10 mult are additive, while Polychrome's x1.5
is multiplicative — and this ordering is per-card, not global.

**Red Seal retriggers re-evaluate everything** — not just the enhancement
effect.  Each rep runs the full `eval_card` which calls `get_chip_bonus`,
`get_chip_mult`, `get_chip_x_mult`, `get_edition`, AND `get_p_dollars`.
So a Red Seal on a Foil Glass Card would: rep 1: +5 chips, x2 mult,
+50 chips, x1.5 edition.  Rep 2: same again.  The x2 and x1.5 compound
multiplicatively across reps.

**Phase 7 vs 8 ordering matters for multiplicative effects.** Scored cards
(Phase 7) apply their effects first, then held cards (Phase 8).  A Steel
Card in hand multiplies the mult AFTER all scored card effects have been
applied.  This means Steel Card's x1.5 multiplies a larger base.

## 3. Boss blinds with state machine dependencies

The following boss blinds have effects that go BEYOND scoring and require
the state machine (press_play, drawn_to_hand, etc.):

| Boss | Method | Effect |
|---|---|---|
| The Hook | press_play | Discards 2 random cards from hand |
| The Tooth | press_play | Lose $1 per card played |
| The Serpent | press_play | Draw 3 extra cards after play/discard |
| Cerulean Bell | drawn_to_hand | Force-select a random card |
| Crimson Heart | drawn_to_hand | Debuff a random joker each hand |
| The Water | set_blind | Remove all discards for the round |
| The Needle | set_blind | Only 1 hand per round |
| The Manacle | set_blind | −1 hand size |
| Amber Acorn | set_blind | Flip and shuffle all jokers |
| The Wheel | stay_flipped | Probabilistic card flip |
| The House | stay_flipped | Flip all cards on first hand |
| The Mark | stay_flipped | Flip face cards face-down |
| The Fish | stay_flipped | Flip cards after each play |

These are all implemented as Blind methods returning side-effect descriptors,
but the game loop must call them at the right time.

## 4. Lucky Card handling

**Deferred to RNG-dependent path.** Lucky Card's `get_chip_mult` and
`get_p_dollars` accept optional `rng` + `probabilities_normal` parameters:

- **With RNG** (simulation mode): does the actual roll
  (`pseudorandom('lucky_mult') < normal/5` for +20 mult,
  `pseudorandom('lucky_money') < normal/15` for +$20)
- **Without RNG** (default): returns 0 for Lucky Card

The oracle tests don't use Lucky Card because the probabilistic roll would
require matching the exact RNG state.  Lucky Card testing will be added when
we have a controlled game state where the RNG stream positions are known.

For the optimizer/EV calculator, the expected value of Lucky Card is
`normal/5 × 20 = 4.0` mult and `normal/15 × 20 ≈ 1.33` dollars at
`probabilities.normal = 1`.

## 5. Performance

| Metric | Result |
|---|---|
| `score_hand_base` per hand | **30 µs/hand** (33,430 hands/sec) |
| 218-subset enumeration | ~6.5 ms (30 µs × 218) |
| Bottleneck | `evaluate_hand` inside `score_hand_base` (~23 µs of the 30 µs) |

At 33K hands/sec, a full game action (evaluate 218 subsets × score each)
takes ~6.5ms.  This is adequate for RL training — the joker system will
be the next bottleneck.

## 6. Code reuse vs rewrite

**100% rewritten from Balatro source** — no code from any previous repo.
The entire engine was built by reading the Lua source, documenting the
behavior in the source-map docs, then implementing in Python with
comprehensive testing.

The approach of building from source documentation (M1) rather than
porting code line-by-line proved effective.  The scoring pipeline doc
(scoring-pipeline.md) was the primary reference for implementation, with
the Lua oracle providing ground-truth validation.

## Modules built/updated in M5

| Module | New/Updated | Tests | Purpose |
|---|---|---|---|
| `scoring.py` | Updated | 29+23+35+37 | eval_card + score_hand_base + ScoreResult |
| `blind.py` | Updated | 149 | debuff_card/hand + modify_hand + press_play + drawn_to_hand + disable + get_new_boss |
| `hand_levels.py` | Existing | 39 | HandLevels (from M4) |
| `card.py` scoring methods | Updated | 57+87 | 8 scoring methods + is_face + is_suit |
| Lua scoring oracle | New | 35 | LuaJIT ground truth |
| Integration tests | New | 37 | Full pipeline coverage |

**M5 total: ~200 new tests, 1,045 cumulative.**

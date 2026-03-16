# M7 Feedback: Complete Joker System

## 1. Final joker coverage

**150/150 jokers registered** — every joker key in centers.json has a handler
in `_REGISTRY`. Coverage audit test (`test_all_150_registered`) enforces this.

Breakdown by handler type:
- **71 simple/conditional** (M6): scoring bonuses, suit/rank/hand-type checks
- **10 scaling** (per-hand mutation): Green Joker, Ride the Bus, Square, Ice Cream, etc.
- **12 xMult scaling** (cross-round): Campfire, Hologram, Vampire, Obelisk, etc.
- **5 retrigger**: Sock and Buskin, Hanging Chad, Dusk, Seltzer, Mime
- **2 copy/delegation**: Blueprint, Brainstorm
- **2 joker-on-joker**: Baseball Card, Swashbuckler
- **11 card creation** (stubbed as descriptors): Certificate, Marble, DNA, etc.
- **7 destructive/rule-modifying**: Gros Michel, Chicot, Midas Mask, Hiker, etc.
- **8 end-of-round economy**: Golden Joker, Cloud 9, Rocket, Egg, Gift Card, etc.
- **7 active remaining** (Castle, Runner, Ramen, Mr. Bones, Burnt, Turtle Bean, Perkeo)
- **14 passive/meta** (no-op handlers): Four Fingers, Shortcut, Pareidolia, etc.
- **5 dollar bonus** (separate `_DOLLAR_REGISTRY`): Golden, Cloud 9, Rocket, Satellite, Delayed Grat

## 2. Hardest joker interactions to port

**Blueprint/Brainstorm copy chain.** The `not context.blueprint` guard is critical:
scaling jokers (Green, Ride the Bus, etc.) skip mutation when called via Blueprint.
Without this guard, Blueprint would double-mutate the target's ability state.
The Lua oracle caught this discrepancy during cross-validation.

**Vampire's individual_hand_end timing.** Vampire strips enhancements AFTER
Phase 7-8 scoring (so the Glass Card's x2 mult fires first), but BEFORE
Phase 9 (so Vampire's accumulated xMult applies in joker_main). This required
adding a new Phase 8d (individual_hand_end) to the pipeline.

**Obelisk's most-played check.** Iterating HandLevels required accessing
`_hands.items()` directly since HandLevels doesn't implement `__iter__`.
The "is most played" logic is inverted from intuition: reset when the
current hand IS the most-played, increment when it's NOT.

**Glass Card destruction + reaction ordering.** Phase 11 destroys cards,
then notifies jokers. Caino and Glass Joker react to the destruction list,
gaining xMult for future hands (not the current one). The xMult gained
doesn't affect the current hand's score since Phase 12 already computed.

## 3. Side-effect descriptors vs inline execution

Card creation, joker destruction, and blind disabling are returned as
**side-effect descriptors** in `JokerResult.extra`, NOT executed inline:

```python
# Card creation
JokerResult(extra={"create": {"type": "Tarot", "key": "car"}})

# Joker destruction
JokerResult(extra={"destroy_random_joker": True})

# Blind disabling
JokerResult(extra={"disable_blind": True})
```

This design means:
- Handlers are **pure functions** of (card, context) → result
- The scoring pipeline and state machine (M12) interpret descriptors
- Side effects can be replayed, logged, or simulated without mutation
- Blueprint correctly copies effects without executing side effects

**Impact on M12 state machine:** The state machine must implement a
descriptor interpreter that processes each side effect type:
- `create` → call card factory with pool generation
- `destroy_random_joker` → select and remove (needs Madness RNG)
- `disable_blind` → call blind.disable()
- `set_hands`/`set_discards` → modify round state

## 4. Scaling joker state persistence

All scaling state lives on `card.ability` (a plain dict):
- `ability["mult"]` — Green Joker, Ride the Bus, Flash Card, etc.
- `ability["extra"]["chips"]` — Square Joker, Ice Cream, Wee Joker
- `ability["x_mult"]` — Campfire, Hologram, Constellation, Ramen, etc.
- `ability["caino_xmult"]` — Caino (separate from x_mult)
- `ability["yorick_discards"]` — Yorick countdown
- `ability["invis_rounds"]` — Invisible Joker counter

Since `ability` is a plain `dict[str, Any]`, serialization is trivial:
`json.dumps(card.ability)` captures all accumulated state. No special
save/load handling needed beyond what M3's Card class already supports.

## 5. Performance

| Benchmark | µs/hand | hands/sec | Target | Status |
|---|---|---|---|---|
| 0 jokers | 17 | 58,000 | ~30 (M5 baseline) | Better |
| 5 simple jokers | 159 | 6,300 | ~164 (M6) | Same |
| 5 complex jokers | 328 | 3,100 | < 300 | Close |
| Retrigger chain (5 Red Seal Kings + Sock) | 99 | 10,100 | < 500 | Better |
| 218-subset enum (5 jokers) | 17 ms | 12,600 | < 80 ms | 4× faster |

The 0-joker path improved from M5's 30µs to 17µs due to the GameSnapshot
refactoring (built once, shared by reference). The 218-subset enumeration
at 17ms is well within the RL training target of <100ms per action.

The 5-complex benchmark (328µs) includes Blueprint delegation, Green Joker
mutation, Sock and Buskin retrigger, Baseball Card Phase 9c, and Ice Cream
after-decay — the most expensive realistic joker combination.

## 6. Known inaccuracies and deferred edge cases

**Card creation pool generation (M10).** The 11 card-creation jokers return
descriptors but don't actually create cards. Pool generation (weighted
random selection from valid cards) is deferred to M10.

**Consumable slot checks.** Card-creation jokers should check
`consumeable_buffer < consumeable_limit` before creating. Currently
the descriptors don't include this check — the state machine must verify.

**Vampire timing.** Our Vampire fires in `individual_hand_end` (Phase 8d),
which is after Phase 8 (held cards). The source fires it in a similar
position, but the exact timing relative to held-card joker effects may
differ in edge cases involving Steel Card + Vampire.

**Probabilistic jokers without RNG.** Bloodstone, Business, Reserved
Parking, Misprint, 8 Ball, and Hallucination return None when `rng` is
not provided. For EV calculation, these need expected-value substitution
at the caller level.

**Popcorn, Turtle Bean, Campfire resets.** These fire in `end_of_round`
or `setting_blind` contexts which are outside the scoring pipeline.
The `on_end_of_round` lifecycle function handles round-end effects,
but blind-setting effects need the state machine.

## 7. Glass Card destruction + Caino/Glass Joker ordering

**Matches the source.** The pipeline processes destruction in this order:

1. **Phase 7** — Glass Card's x2 mult is applied during scoring
2. **Phase 12** — Final score computed (Glass's scoring effect included)
3. **Phase 11** — Glass Card rolls for shatter (1/4 chance at normal prob)
4. **Phase 11b** — Destroyed cards collected into list
5. **Phase 11c** — All jokers notified with `cards_destroyed` context
6. Caino counts face cards in destroyed list → `caino_xmult += 1`
7. Glass Joker counts glass cards → `x_mult += 0.75`

The xMult gains from Caino and Glass Joker apply to **future hands only**,
not the current hand's score. This is confirmed by the Lua oracle:
`glass_king_caino` shows total=360 (computed before destruction),
with `caino_xmult=2` (gained +1 for future use).

**Lua oracle cross-validation:** 24 scenarios × 5 metrics = 120 assertions,
100% match. Includes multi-hand sequences (Green Joker 4 hands, Ice Cream
decay), retrigger combinations (Red Seal + Dusk), Blueprint delegation,
Baseball Card Phase 9c, and Glass destruction + Caino reaction.

## Modules built/updated in M7

| Module | New/Updated | Tests | Purpose |
|---|---|---|---|
| `jokers.py` | Updated | 226+35+44+22+11+30+26+30+33 | 150 handlers + GameSnapshot + dollar registry |
| `scoring.py` | Updated | 13+75+14 | Full 14-phase pipeline + score_hand |
| Lua scoring oracle | Updated | 120 | 24 scenarios (9 new complex interactions) |
| test_jokers_scaling.py | New | 35 | Scaling joker multi-hand tests |
| test_jokers_xmult.py | New | 44 | xMult accumulation tests |
| test_jokers_retrigger.py | New | 22 | Retrigger pipeline tests |
| test_jokers_j2j.py | New | 11 | Joker-on-joker (Baseball) tests |
| test_jokers_create.py | New | 30 | Card creation descriptor tests |
| test_jokers_destructive.py | New | 26 | Destruction/rule-mod tests |
| test_jokers_endround.py | New | 30 | End-of-round economy tests |
| test_jokers_coverage.py | New | 33 | 150/150 audit + passive tests |
| test_scoring_phases.py | New | 14 | Phase 10-14 integration tests |

**M7 total: ~570 new tests, 1,614 cumulative.**

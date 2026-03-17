# M9 Feedback: Consumables, Vouchers & Economy

## 1. Consumable coverage

**85 consumables registered** across four categories:

| Category | Count | Notes |
|---|---|---|
| Tarots | 22 | All 22 from the Lua source (Fool through World) |
| Planets | 13 | 12 standard + Black Hole (all-hands level-up) |
| Spectrals | 18 | All 18 including Aura, Ectoplasm, Hex, Wraith, Ankh, Soul |
| Vouchers | 32 | All 32 via `apply_voucher` (separate dispatch, not consumable registry) |
| **Total** | **85** | |

Coverage audit: `test_consumables.py` has 219 unit tests; `test_vouchers.py` has 76 tests;
`test_consumables_integration.py` adds 36 cross-system integration tests.

## 2. Consumables with pool-generation dependencies (deferred to M10)

The following ConsumableResult `create` descriptors return intent but do NOT
generate actual cards — the pool selection logic is deferred to M10:

| Consumable | `create` descriptor type | Pool logic needed |
|---|---|---|
| The Fool | `type="Tarot_Planet"` | Copy last used tarot/planet key |
| The High Priestess | `type="Planet"`, count=2 | `get_current_pool("Planet")` |
| The Judgement | `type="Joker"`, count=1 | `get_current_pool("Joker")`, rarity-weighted |
| Wraith | `type="Joker"`, rarity=3 (Rare) | `get_current_pool("Joker")`, rare-only |
| Soul | `type="Joker"`, rarity=4 (Legendary) | Fixed legendary pool |
| Ankh | `copy_of=chosen_joker` | Exact copy, no pool needed (✓ done) |
| Familiar | `type="PlayingCard"`, face_only, count=3 | Face-rank filter, random suit/enh |
| Grim | `type="PlayingCard"`, ace_only, count=2 | Ace-only, random suit/enh |
| Incantation | `type="PlayingCard"`, number_only, count=4 | Number-rank filter |
| Sigil / Ouija | `change_suit` / `change_rank` | Already fully applied (no pool) |
| Immolate | `dollars=+20` | Already applied (no pool) |

The M12 state machine must resolve `create` descriptors by calling
`get_current_pool` (M10) and then the card factory.

## 3. Voucher rate multipliers — exact source values

From `card.lua:1891` (`apply_to_run`), the tarot/planet rate is **set** (not
multiplied) via `G.GAME.tarot_rate = 4 * center_table.extra`:

| Voucher | `extra` | `tarot_rate` result |
|---|---|---|
| Tarot Merchant | 2.4 | `4 × 2.4 = 9.6` (SET) |
| Tarot Tycoon | 8 | `4 × 8 = 32` (SET) |
| Planet Merchant | 2.4 | `4 × 2.4 = 9.6` (SET, planet_rate) |
| Planet Tycoon | 8 | `4 × 8 = 32` (SET, planet_rate) |
| Hone | 2 | `edition_rate = 2` (SET) |
| Glow Up | 4 | `edition_rate = 4` (SET) |

**Common confusion:** The "1.5×" or "2.4×" multiplier phrasing in the wiki
refers to the increased shop weight (weighted pool draw), not a runtime
multiplier on an existing value. `apply_to_run` always **replaces** the
rate field — applying Tarot Tycoon after Tarot Merchant gives rate 32, not
9.6 × 3.33.

**`poll_edition` rate effect:** With `edition_rate=2` (Hone), the Foil
threshold moves from `> 0.96` to `> 0.92`, approximately doubling the Foil
chance from 2% to 4%. With `edition_rate=4` (Glow Up), it moves to `> 0.84`
(~4× more Foil chances vs base). Negative chance ignores `edition_rate` —
only `mod` affects it.

## 4. Hieroglyph / Petroglyph — one-time vs per-round

Both are **one-time permanent mutations** applied to `round_resets` when the
voucher is purchased (via `apply_voucher`). They do NOT fire on every round.

`round_resets` is the per-run baseline that the state machine resets to at
the start of each round. Hieroglyph mutates it once:

```
round_resets.ante     -= 1   (also decrements blind_ante)
round_resets.hands    -= 1   (permanent reduction)
```

Petroglyph:
```
round_resets.ante     -= 1   (also decrements blind_ante)
round_resets.discards -= 1   (permanent reduction)
```

**Edge case — double application:** Applying Hieroglyph twice gives
`hands -= 2`. The Lua source does not guard against duplicate voucher
application — that's enforced by marking the voucher as "used" in
`used_vouchers`. The `check_voucher_prerequisites` and
`get_available_voucher_pool` helpers enforce this by excluding already-used
voucher keys.

**`blind_ante` initialisation:** If `round_resets.blind_ante` is not set
before the first Hieroglyph/Petroglyph application, we retroactively set it
to `ante + extra` before decrementing, matching the Lua source's implicit
assumption that `blind_ante` was previously equal to `ante` before the
decrement.

## 5. Death tarot edge cases

**Stone Card (no rank):** The Death handler (`_death`) checks
`if c.base is not None` before touching rank/suit. Stone Cards set
`ability.effect = "Stone Card"` but still have `base` set (they just have
a fixed nominal). Death will copy the nominal rank of a Stone Card normally
— `card.lua:1111` does not special-case Stone.

**Sort-id ordering:** Death picks `rightmost = max(highlighted, key=lambda
c: c.sort_id)`. Since `sort_id` is assigned sequentially at construction
time, the card created *later* is treated as rightmost. In tests, always
create the target card *before* the source to ensure the source is rightmost.

**`copy_card` applies:** rank, suit, enhancement, edition, seal. The ability
dict is fully reset by `card.enhance(source.center_key)` which calls
`set_ability`. This means ability state (e.g. Glass `x_mult`, Lucky `mult`,
Steel `h_x_mult`) is reset to the prototype defaults for the new center.

**What Death does NOT copy:** `eternal`, `perishable`, `rental` sticker
flags — these are purchase-time properties, not intrinsic card identity.

## 6. To the Moon: +$1 to interest_amount, not +$1 per $5

To the Moon adds `+1` to `G.GAME.interest_amount` (a per-bracket multiplier),
not a flat $1 bonus. The interest formula is:

```
interest = interest_amount × min(floor(effective_money / 5), interest_cap / 5)
```

With one copy of To the Moon (`interest_amount=2`) and `$25` in the bank:
- `min(25//5=5, 25//5=5) = 5` brackets
- `interest = 2 × 5 = $10` (vs $5 base)

With two copies (`interest_amount=3`) and `$25`:
- `interest = 3 × 5 = $15`

`Card.add_to_deck` handles the accumulation:
```python
if name == "To the Moon":
    game_state["interest_amount"] += extra  # extra=1 per copy
```

The `interest_cap` (25/$50/$100 from Seed Money/Money Tree) limits brackets,
not the raw dollar amount. At `interest_cap=25` the maximum brackets are 5,
so `interest_amount=3` gives a max of `$15/round`.

## 7. Performance impact of consumable system

Consumable use is a pre-round operation; it does not execute during the
scoring hot path. The consumable dispatch (registry lookup → handler call)
is O(1). Planet cards call `hand_levels.level_up()` which is also O(1).

The scoring pipeline benchmarks from M7 are unchanged:

| Benchmark | µs/hand |
|---|---|
| 0 jokers | 17 |
| 5 simple jokers | 159 |

Planet-leveled hands add zero overhead to scoring — `HandLevels.get()` is a
single dict lookup regardless of level.

## 8. Lua oracle coverage for consumable effects

No new Lua oracle scenarios were added in M9. The existing 24-scenario oracle
(`test_scoring_oracle.py`, 120 assertions) covers post-enhancement scoring
(Glass Card, Steel Card, etc.) which validates that the consumable → score
pipeline is correct end-to-end.

**Gaps for M10 oracle work:**
- Planet card → leveled hand → exact chip/mult values for non-trivial hands
- Black Hole followed by multi-hand scoring sequences
- Wheel of Fortune → joker edition → scoring with edition in Phase 9a

The `test_consumables_integration.py` tests substitute for oracle coverage
by using exact arithmetic verification against known Balatro scoring formulas.

## Modules built/updated in M9

| Module | New/Updated | Tests | Purpose |
|---|---|---|---|
| `consumables.py` | Updated | 219 | 6 new spectral handlers (Aura, Ectoplasm, Hex, Wraith, Ankh, Soul); `poll_edition` extracted |
| `card_utils.py` | New | 45 | Full `poll_edition` (normal + guaranteed mode, rate/mod params) |
| `vouchers.py` | New | 76 | All 32 vouchers: prerequisites, pool, `get_next_voucher_key`, `apply_voucher` |
| `economy.py` | New | 41 | `RoundEarnings`, `calculate_round_earnings`, `calculate_discard_cost` |
| `test_consumables_integration.py` | New | 36 | 10 cross-system integration scenarios |

**M9 total: ~417 new tests, ~2,031 cumulative.**

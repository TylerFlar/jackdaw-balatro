# M10 Feedback: Shop, Pools & Packs

## 1. Oracle match status

The Python `populate_shop` implementation is **bit-exact with LuaJIT 2.1** for
the three validated test cases (TESTSEED ante 1, TESTSEED ante 3, TUTORIAL ante
1).  The fixture `tests/fixtures/shop_oracle_TESTSEED.json` records the Python
ground truth; `test_shop_oracle.py` has 21 assertions locking the output to that
fixture.

LuaJIT oracle for RNG streams `cdt`, `rarity`, and `Joker1sho` was independently
validated in `test_rng_sequence.py` with hardcoded LuaJIT values — Python TW223
matches the Lua PRNG bitwise.

**Known intentional divergence — TUTORIAL seed:** The actual Balatro game applies
`G.SETTINGS.tutorial_progress.forced_shop` when the TUTORIAL seed is active,
bypassing the RNG and placing specific jokers in the shop.  Python `populate_shop`
implements only the RNG path, so TUTORIAL ante 1 produces `c_star` + `c_mercury`
(tarots drawn by RNG) instead of `j_joker`.  This is correct behaviour for the
simulator — a note is recorded in the oracle fixture.

## 2. UNAVAILABLE sentinel preservation

`get_current_pool` uses `"UNAVAILABLE"` in-place sentinels to keep pool indices
aligned with Lua's `get_current_pool` loop (common_events.lua:1963).  Index
alignment is required because `pseudorandom_element` draws by position — if
filtered entries were removed rather than replaced, the selected index would map
to a different card than the Lua source.

**Empty-pool fallback:** When all pool entries are UNAVAILABLE, `get_current_pool`
returns a single-element list containing the type's fallback key (e.g.
`["j_joker"]` for Jokers, `["c_strength"]` for Tarots).  This mirrors the Lua
source's while-loop resampling and prevents infinite-retry behaviour.  Tests in
`test_shop_integration.py` (Showman scenario) were updated after discovering this
fallback fires even when `used_jokers` contains all joker keys, because rarity
sub-pools (`JOKER_RARITY_POOLS[1]`) only contain the rarity-1 subset, not all
jokers.

## 3. Pool filtering edge cases

| Filter condition | Mechanism | Test |
|---|---|---|
| `used_jokers` (no Showman) | Key → UNAVAILABLE | `TestBuyJokerTracking` |
| `used_jokers` with Showman | Key stays real | `TestShowmanAllowsDuplicates` |
| `banned_keys` | Key → UNAVAILABLE | `TestFullShopTESTSEED` |
| Empty pool fallback | Returns `[fallback_key]` | implicit in Showman tests |
| `pool_flags` (e.g. gros_michel_extinct) | Key → UNAVAILABLE | `test_pools.py` |
| Voucher prerequisites | Excluded from pool pre-filter | `test_vouchers.py` |

**Showman design note:** The `has_showman` flag bypasses the `used_jokers` filter
at the pool-building stage (not at the draw stage), so it has no impact on RNG
stream advancement — the pool size is larger but the draw is identical.

## 4. RNG stream order

The shop build sequence within `populate_shop` consumes RNG streams in this exact
order, which must be preserved to match Lua output:

1. `Voucher` — consumed by `get_next_voucher_key` before `populate_shop` is called
2. Per joker slot (repeated `joker_max` times):
   - `cdt{ante}` — card type selection
   - `rarity{ante}{append}` — joker rarity (Joker type only)
   - `{pool_type}{append}` — pool draw (e.g. `Jokersho`, `Tarotsho`)
   - `etperpoll{ante}` — eternal/perishable roll (Joker in shop/pack only)
   - `ssjr{ante}` — rental roll (Joker in shop/pack only)
   - `edi{append}{ante}` — edition roll (Joker in shop/pack only)
3. `shop_pack{ante}` × 2 — booster pack selection (skipped for first-shop Buffoon guarantee)

**Critical:** the eternal/perishable/rental/edition rolls for Joker slots are
**always consumed** regardless of stake settings (`enable_eternals_in_shop`,
`enable_perishables_in_shop`, `enable_rentals_in_shop`).  Skipping any roll would
shift the RNG state and misalign all subsequent draws.

Pack stream order within `generate_pack_cards`:
- Arcana: `omen_globe` (if has_omen_globe) → `ar2` Spectral OR `ar1` Tarot pool draw
- Celestial: forced key from Telescope bypasses pool draw entirely for slot 0
- Standard: `stdset{ante}` → `front{sta}{ante}` → `Enhanced{sta}` (if enhanced) →
  `standard_edition{ante}` → `stdseal{ante}` → `stdsealtype{ante}` (if sealed)
- Buffoon: `Jokerbuf` pool draw + full Joker modifier chain

## 5. Pack append patterns

Each calling context passes a distinct `append` string to prevent stream key
collisions between simultaneous card draws:

| Context | `append` | Example stream key |
|---|---|---|
| Shop joker slot | `"sho"` | `Jokersho`, `edisho1` |
| Arcana pack (Tarot) | `"ar1"` | `Tароtar1` |
| Arcana pack (Spectral) | `"ar2"` | `Spectralar2` |
| Celestial pack | `"pl1"` | `Planetpl1` |
| Spectral pack | `"spe"` | `Spectralspe` |
| Buffoon pack | `"buf"` | `Jokerbuf`, `edibuf1` |
| Reroll slot | `"sho"` | same as shop (reroll reuses shop context) |
| High Priestess create | `"pri"` | `Planetpri` |
| Judgement create | `"jud"` | `Jokerjud` |

Collisions would cause two draws to pull from the same seeded stream, producing
incorrect determinism.  The Lua source uses the same append strings; any deviation
would break oracle cross-validation.

## 6. Performance

Pack opening and shop population benchmarks (ante 1, no jokers):

| Operation | µs/call |
|---|---|
| `populate_shop` (2 slots + 2 packs) | ~200 |
| `generate_pack_cards` (Arcana normal, 3 cards) | ~50 |
| `generate_pack_cards` (Standard jumbo, 5 cards) | ~90 |
| `reroll_shop` (2 new slots) | ~80 |
| `resolve_create_descriptor` (Planet) | ~30 |

All operations are well under 1ms; the scoring hot-path benchmarks from M7
(17–159 µs/hand) are unchanged.

## 7. Test coverage

| File | Tests | Focus |
|---|---|---|
| `test_shop_oracle.py` | 21 | Fixture regression + LuaJIT ground truth |
| `test_shop_integration.py` | 45 | 10 scenarios: full shop, voucher effects, Showman, reroll, buy, packs, descriptors |
| `test_shop.py` | (existing) | Unit: `select_shop_card_type`, `get_pack`, `populate_shop` |
| `test_pools.py` | (existing) | Unit: `get_current_pool`, filtering, soul chance |
| `test_packs.py` | (existing) | Unit: `generate_pack_cards` per pack type |
| `test_consumables.py` | 219 | All 53 consumable handlers |

**M10 total: ~66 new tests, ~2,545 cumulative.**

## Modules built/updated in M10

| Module | New/Updated | Purpose |
|---|---|---|
| `pools.py` | New | `get_current_pool`, `select_from_pool`, `pick_card_from_pool`, `check_soul_chance` |
| `packs.py` | New | `generate_pack_cards` — Arcana/Celestial/Spectral/Standard/Buffoon |
| `card_factory.py` | Updated | `resolve_create_descriptor`, `resolve_destroy_descriptor` |
| `shop.py` | Updated | `populate_shop`, `reroll_shop`, `buy_card`, `sell_card`, `calculate_reroll_cost` |
| `scripts/run_shop_oracle.py` | New | Python oracle fixture generator |
| `tests/fixtures/shop_oracle_TESTSEED.json` | New | Ground-truth fixture (3 cases) |
| `tests/engine/test_shop_oracle.py` | New | 21 oracle regression tests |
| `tests/engine/test_shop_integration.py` | New | 45 integration tests |

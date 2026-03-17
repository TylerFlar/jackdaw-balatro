# M11 Feedback — Tags, Backs, Stakes & Run Initialization

## 1. Does the run initialization RNG consumption order match Lua?

Yes. The order in `initialize_run` matches `game.lua:2177-2180` exactly:

1. `get_new_boss()` — seed `'boss'`
2. `get_next_voucher_key()` — seed `'Voucher'`
3. `get_next_tag_key()` — seed `'Tag' + str(ante)` (Small)
4. `get_next_tag_key()` — seed `'Tag' + str(ante)` (Big)

Then deck shuffle (`'nr' + str(ante)`), then targeting cards (`'idol'`, `'mail'`, `'anc'`, `'cas'` + ante).

Known-seed integration tests lock in specific values (e.g. TESTSEED → boss=bl_head, voucher=v_blank, tags=tag_economy/tag_investment) so any RNG order regression breaks determinism tests immediately.

## 2. Tag system coverage — all 24 tags implemented?

Yes. All 24 tags fire in their correct context via `Tag.apply()`:

| Context              | Tags |
|----------------------|------|
| immediate            | economy, garbage, handy, orbital, skip, top_up |
| new_blind_choice     | boss, buffoon, charm, ethereal, meteor, standard |
| eval                 | investment |
| tag_add              | double |
| round_start_bonus    | juggle |
| store_joker_create   | uncommon, rare |
| shop_start           | d_six |
| store_joker_modify   | foil, holo, polychrome, negative |
| shop_final_pass      | coupon |
| voucher_add          | voucher |

86 unit tests + 51 generation tests = 137 tag tests total.

## 3. Back.apply_to_run integration — did refactoring scoring.py break anything?

No. Scoring Phase 10 was refactored from `if back_key == "b_plasma"` to `Back(back_key).trigger_effect("final_scoring_step", ...)`. All 34 back tests pass. The scoring integration tests (Plasma deck averaging, non-Plasma unaffected, breakdown message) all validate the refactored path.

Bug found and fixed: Abandoned Deck's `remove_faces` config flag was not propagated to `starting_params.no_faces`, causing `build_deck` to include face cards. Fixed by explicitly propagating `back.config.remove_faces` → `sp["no_faces"]` in `initialize_run`.

## 4. Challenge system status — data extracted? Application tested?

All 20 challenges extracted from `challenges.lua` into `CHALLENGES` dict in `jackdaw/engine/challenges.py`. Application function handles all 6 steps:

1. Starting jokers (with eternal/pinned/edition flags)
2. Starting consumables
3. Starting vouchers (mark used + apply effects)
4. Rule modifiers (override starting_params)
5. Custom rules (no_reward, no_shop_jokers, inflation, debuff_played_cards, etc.)
6. Restrictions (banned_cards with nested ids arrays, banned_tags, banned_other)

97 tests covering every challenge's data integrity, all custom rule types, voucher effects (Zodiac rates, Money Tree cap), and integration through `initialize_run`.

## 5. Perishable/rental processing edge cases?

Covered in `round_lifecycle.py`:

- Perishable: full 5-round countdown tested (5→4→3→2→1→0→debuff)
- Already-at-0 cards stay stable (no double-debuff)
- Rental fires even on debuffed cards (matches Lua: `calculate_rental` checks `ability.rental`, not `debuff`)
- Rental can push dollars negative (no floor check in Lua)
- Both perishable + rental on same card: both fire, including on the debuff round
- 18 tests total

## 6. Starting params → round_resets → current_round chain — any gotchas?

Two gotchas discovered:

1. **Voucher ordering**: Crystal Ball (from Magic Deck) sets `game_state["consumable_slots"] += 1`. If card area limits are set from `starting_params` *after* voucher application, the voucher's effect gets overwritten. Fixed by initializing card area limits from starting_params first, then applying vouchers.

2. **Challenge voucher overwrite**: Challenge starting vouchers (like Grabber) modify `round_resets.hands` directly, but the `starting_params → round_resets` transfer at step 6 overwrites this. This matches the Lua behavior — challenge vouchers are applied for their passive/flag effects, not their round_resets mutations.

## 7. How many total tests now?

**2,934 tests** (including 11 RNG performance tests).

Breakdown of new M11 tests:

| File | Tests | What |
|------|-------|------|
| test_tags.py | 86 | Tag.apply for all 24 tags |
| test_tag_generation.py | 51 | generate_blind_tags, assign_ante_blinds |
| test_back.py | 34 | Back class, apply_to_run, trigger_effect |
| test_stakes.py | 50 | apply_stake_modifiers |
| test_run_init.py | 63 | init_game_object, initialize_run, start_round |
| test_run_init_integration.py | 39 | Full chain integration across 8 deck types |
| test_challenges.py | 97 | 20 challenge definitions + apply_challenge |
| test_round_lifecycle.py | 34 | Perishable/rental + targeting card reset |
| **Total new** | **454** | |

## New modules

| Module | Purpose |
|--------|---------|
| `jackdaw/engine/tags.py` | Tag class, TagResult, 24 tag effects, generate_blind_tags, assign_ante_blinds |
| `jackdaw/engine/back.py` | Back class, apply_to_run (16 decks), trigger_effect (Plasma, Anaglyph) |
| `jackdaw/engine/stakes.py` | apply_stake_modifiers (8 stakes), DEFAULT_STARTING_PARAMS |
| `jackdaw/engine/run_init.py` | init_game_object, initialize_run, start_round, get_starting_params |
| `jackdaw/engine/challenges.py` | 20 challenge definitions, apply_challenge |
| `jackdaw/engine/round_lifecycle.py` | process_round_end_cards, reset_round_targets |

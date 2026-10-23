# Gap register

Known behaviour gaps between jackdaw and Balatro 1.0.1o, confirmed during the planning audit
against both the Lua source and the Python engine. Line refs are into `balatro_source/`; refs
marked `ledger:` were added when the gap was linked to its ledger rows (the plan gave none).

Every gap is fixed in the Python oracle first, with a regression test that cites the Lua lines,
then in Rust (in the same PR if that subsystem is already ported). While a fix is pending, Rust
stays oracle-compatible behind an `// AUDIT-Gxx` marker. The ledger rows that carry a gap id in
their `gap` column (status `gap`) show where the logic lives; see [README.md](README.md).

Columns: `python fix` and `rust` take the PR or commit and code location of each fix, `test` the
regression test id. Status is one of `open`, `in-progress`, `python-fixed` (fixed in the Python
oracle with its regression test; the Rust port follows the fixed oracle), `fixed` (both engines,
test landed) or `wontfix` (with the reason in the summary).

## Gameplay gaps

| id | area | summary | lua | python fix | rust | test | status |
|---|---|---|---|---|---|---|---|
| G01 | setup | Green Deck economy broken: the values are stored at the top level, but `economy.py` reads `modifiers`. Result: $1/hand, no $/discard, interest still paid. | back.lua:269-277 |  |  |  | open |
| G02 | setup | Magic Deck (2× The Fool) and Ghost Deck (Hex) start without their consumables. | back.lua:184-196 |  |  |  | open |
| G03 | setup | Challenge starting jokers and consumables are never created (17/20 challenges), and there is no `pinned` logic. Ledger check: in the 1.0.1o source 12/20 challenges start with jokers (only `c_bram_poker_1` also with consumables) and 3 with vouchers; the 12 are linked. | game.lua:2067-2088; cardarea.lua:528 |  |  |  | open |
| G04 | setup | Missing challenge rules: `chips_dollar_cap`, `flipped_cards`, `minus_hand_size_per_X_dollar`, `all_eternal`, `debuff_played_cards`, `set_eternal_ante` / `set_joker_slots_ante`, `inflation`. | common_events.lua:2134; misc_functions.lua:684; cardarea.lua:601; cardarea.lua:245; state_events.lua:1077; state_events.lua:240; button_callbacks.lua:2455; card.lua:1800/1858 |  |  |  | open |
| G05 | setup | Challenge bans are not applied to the boss, tag and voucher pools or to the Soul roll. | common_events.lua:2030/2088/2359 |  |  |  | open |
| G06 | setup | Anaglyph Deck: the `eval` trigger (Double Tag after a boss) is never called. | state_events.lua:1163 |  |  |  | open |
| G07 | blinds | `Blind:disable()` is incomplete, and Chicot only sets a flag. Missing: The Wall halved, Violet Vessel divided by 3, ending the round at once if chips already beat the new target, The Manacle's draw, flipping face-down cards. | blind.lua:356-406; card.lua:596/2492 |  |  |  | open |
| G08 | blinds | Verdant Leaf is never disabled by selling a joker. | card.lua:1616 |  |  |  | open |
| G09 | blinds | The Ox uses the live play counts instead of the frozen `most_played_poker_hand` (updated only at boss defeat), and zeroes money after scoring instead of before. | blind.lua:560; state_events.lua:129 |  |  |  | open |
| G10 | blinds | Amber Acorn shuffles 3× with `aajk`; Python shuffles once. | blind.lua:196 | `game.py::_handle_select_blind`: the flip runs before the setting_blind context (Riff-raff's new jokers stay face up), then three `shuffle('aajk')` calls run on the board as the setting_blind effects left it, whether or not Chicot disables the blind |  | `tests/engine/audit/test_blinds.py::test_g10_amber_acorn_shuffles_three_times`, `::test_g10_jokers_made_at_setting_blind_are_shuffled_but_face_up` | python-fixed |
| G11 | blinds | Crimson Heart re-picks only when `prepped`; Python also re-picks after discards. Its debuff should toggle passives and be cleared at round end. | blind.lua:588-602, 211 |  |  |  | open |
| G12 | blinds | The defeated boss is never reset, so selling Luchador in the shop "disables" The Manacle and gives a permanent +1 hand size. | blind.lua:336 | `game.py::_round_won` runs Blind:defeat after the earnings: The Manacle's +1, then `gs["blind"] = Blind.empty()` and set_blind's debuff pass over playing cards and jokers (Crimson Heart's victim freed). A run starts with the empty blind (`run_init.py`). `Blind.prepped` is a real field (true at set time, false for The Fish). The env's score-progress reward now reads the target from the pre-step state (`env/rewards.py`) |  | `tests/engine/audit/test_blinds.py::test_g12_the_run_starts_with_the_empty_blind`, `::test_g12_a_defeated_boss_is_reset_for_the_shop`, `::test_g12_selling_luchador_in_the_shop_disables_nothing`, `::test_g12_defeat_lifts_crimson_heart_joker_debuff` | python-fixed |
| G13 | blinds | Boss debuffs are not re-checked after a suit/rank/enhancement change, or when Pareidolia or Smeared Joker join or leave. | card.lua:143/365/561; card.lua:641/698 | `game.py::blind_debuff_card` after every consumable suit/rank/enhancement change and Death's copy, and in Vampire/Midas Mask (`jokers.py`); `game.py::reapply_blind_debuffs` (set_blind(nil, true)) from every joker/consumable `add_to_deck`/`remove_from_deck` and as set_blind's own pass, now before the setting_blind context. Draws no longer re-check debuffs, so cards created mid-round (Familiar, Marble's stone) keep vanilla's state. Death's final `debuff = other.debuff` is G30 |  | `tests/engine/audit/test_blinds.py::test_g13_suit_change_lifts_a_suit_debuff_at_once`, `::test_g13_enhancement_change_rechecks_the_debuff`, `::test_g13_rank_change_rechecks_a_face_debuff`, `::test_g13_selling_pareidolia_rechecks_every_card`, `::test_g13_adding_pareidolia_rechecks_every_card`, `::test_g13_cards_created_mid_round_are_not_debuffed_by_a_draw`, `::test_g13_vampire_rechecks_the_card_it_strips` | python-fixed |
| G14 | blinds | `first_hand_drawn` (Certificate) must run before `drawn_to_hand` (Cerulean Bell). | game.lua:3226 |  |  |  | open |
| G15 | debuffs | `set_debuff` must call `remove_from_deck`/`add_to_deck`, and expired perishables must stay debuffed and lose their passives. The end-of-round order must be perishable tick, then dollar bonus. | card.lua:526-538, 2283; state_events.lua:99-110 | `card.py`: `Card.added_to_deck` (new snapshot field) makes add/remove idempotent; `Card.set_debuff(should, game_state)` keeps perished cards debuffed and toggles a joker's passives with `from_debuff` (a Negative's slot is kept and queued, then freed when the card goes, card.lua:4733); clones reset the flag. `game.py::_joker_end_of_round_effects` runs end_of_round → rental → perishable per joker, then the dollar bonuses; `round_lifecycle.calculate_rental` / `calculate_perishable`. The blind-side set_debuff callers are G11/G12/G07 |  | `tests/engine/audit/test_debuffs.py::test_g15_expiring_perishable_loses_its_passive`, `::test_g15_perished_joker_stays_debuffed`, `::test_g15_dollar_bonus_comes_after_the_perishable_tick`, `::test_g15_rent_is_charged_for_a_joker_that_destroys_itself`, `::test_g15_debuffed_negative_joker_frees_its_slot_when_sold` | python-fixed |
| G16 | jokers | Blueprint/Brainstorm guards are wrong: Hiker, DNA, Perkeo and Burnt Joker should be copyable; Trading Card should not be. | card.lua:3067/3501/2413/2748/2802 |  |  |  | open |
| G17 | jokers | Satellite never pays; `planet_types_used` is never written. | card.lua:1667 |  |  |  | open |
| G18 | jokers | Bull and Bootstraps must see `dollar_buffer` (money earned earlier in the same hand). To Do List pays in `before`. | card.lua:3936/4046/3491 |  |  |  | open |
| G19 | jokers | Hologram (`playing_card_added`) is missing for Certificate, Marble Joker, DNA, Cryptid and Familiar/Grim/Incantation. | card.lua:1217/1338/2481/2600; common_events.lua:923 |  |  |  | open |
| G20 | jokers | Mime retriggers only held cards that have an effect during scoring; Mime and Red Seal retrigger Gold-card $ and Blue Seal at round end. | state_events.lua:171-233, 813-829 |  |  |  | open |
| G21 | jokers | The Pareidolia, Smeared Joker and Four Fingers flags are not passed to Ride the Bus, Caino, Flower Pot and Seeing Double, to the discard contexts used by Faceless Joker and Castle, or to Burnt Joker's hand detection. | ledger: card.lua:2623/2673/2749/2814/2858/3525/3807/3840 |  |  |  | open |
| G22 | jokers | 8 Ball rolls even with full slots (desyncs the stream); Hallucination needs a buffer between copies. | card.lua:3106/2335 |  |  |  | open |
| G23 | jokers | Oops! All 6s is ignored by The Wheel of Fortune and by the end-of-round roll for Gros Michel/Cavendish. | ledger: card.lua:1470/3020 |  |  |  | open |
| G24 | jokers | `getting_sliced` and `joker_buffer` are not modelled (Ceremonial Dagger, Madness, Riff-raff, Burglar, Cartomancer). | ledger: card.lua:2503-2578 |  |  |  | open |
| G25 | jokers | A Mr. Bones save still pays the blind reward. | state_events.lua:1139 |  |  |  | open |
| G26 | consumables | The Wheel of Fortune, Aura and Hex set bare edition flags without the bonus or repricing; they must go through `set_edition` + `set_cost`. | card.lua:387-462 |  |  |  | open |
| G27 | consumables | Ectoplasm's hand-size penalty grows by one each use (`ecto_minus`). | card.lua:1494 |  |  |  | open |
| G28 | consumables | Ankh refuses to work when joker slots are full (`check_use`). Ankh and Invisible Joker copies drop Negative. | card.lua:1581/1445/2384 |  |  |  | open |
| G29 | consumables | `copy_card` gives copies a fresh `sort_id`; Python deep copies keep the old one, which changes later picks that sort by `sort_id`. | common_events.lua:2157 |  |  |  | open |
| G30 | consumables | Death/DNA copies must carry `perma_bonus`, `played_this_ante` and the debuff state. | common_events.lua:2161 |  |  |  | open |
| G31 | consumables | The Soul's edition roll uses the `sou` append; Python uses `soul`. | card.lua:1418 |  |  |  | open |
| G32 | consumables | A debuffed Purple Seal makes no Tarot. | card.lua:2243 |  |  |  | open |
| G33 | consumables | Sticker eligibility: `eternal_compat` / `perishable_compat`; Eternal and Perishable are mutually exclusive; the Perishable roll is capped at 0.7. | card.lua:506-519; common_events.lua:2141 |  |  |  | open |
| G34 | scoring | The `glass` roll happens even after a joker destroyed the card, and `destroying_card` runs for debuffed cards too. | state_events.lua:950 |  |  |  | open |
| G35 | scoring | A hand becomes `visible` when played, not when leveled. | state_events.lua:578; common_events.lua:464 | `hand_levels.py`: `HandLevels.level_up` no longer reveals a hand; `record_play` (evaluate_play's counters) sets `visible`. Secret hands leveled by Planet X/Ceres/Eris or Black Hole stay out of the To Do List, Telescope, Obelisk and Orbital pools until played |  | `tests/engine/audit/test_hand_mechanics.py::test_g35_leveling_a_secret_hand_keeps_it_hidden`, `::test_g35_black_hole_keeps_secret_hands_hidden`, `::test_g35_playing_a_secret_hand_reveals_it` | python-fixed |
| G36 | scoring | The round must end when hand, deck and play are all empty, and the game must end when hand size ≤ 0 with an empty hand; Python soft-locks instead. | game.lua:3056; state_events.lua:356 |  |  |  | open |
| G37 | scoring | Growing the hand size mid-round draws immediately. | cardarea.lua:100 | `game.py::change_hand_size` (G.hand:change_size): growing to a limit above 1 while the hand holds cards in SELECTING_HAND draws `delta` cards through the boss flip check, then sorts; every passive and boss hand-size change goes through it. Round-end processing now runs outside SELECTING_HAND (vanilla's NEW_ROUND). A fill draw with nothing to draw no longer re-sorts the hand (state_events.lua:362) |  | `tests/engine/audit/test_round_flow.py::test_g37_growing_the_hand_size_mid_round_draws_the_difference`, `::test_g37_hand_size_draws_use_the_boss_flip_rules`, `::test_g37_shrinking_the_hand_size_draws_nothing` | python-fixed |
| G38 | scoring | Cards added to the deck go to the bottom. | cardarea.lua:33 | `game.py::_emplace_in_deck` (index 0 = bottom) for bought playing cards, Standard pack picks and the out-of-round create paths. Playing cards created during an Arcana/Spectral pack (Familiar/Grim/Incantation, Cryptid) belong in the pack's dealt hand in vanilla, then return with it; that routing is not modelled |  | `tests/engine/audit/test_hand_mechanics.py::test_g38_bought_playing_card_goes_to_the_bottom_of_the_deck`, `::test_g38_standard_pack_pick_goes_to_the_bottom_of_the_deck` | python-fixed |
| G39 | scoring | SortHand: vanilla sorts `desc` / `suit desc` (Python sorts ascending), and the chosen mode persists in `G.hand.config.sort` for later draws. | cardarea.lua:577; button_callbacks.lua:36 | `game.py::_handle_sort_hand` sorts `desc` / `suit desc` by `get_nominal` and stores the mode in `gs["hand_sort"]` (new snapshot field `game.hand_sort`); `_sort_hand` uses it at every vanilla auto-sort (draws, pack deals, Certificate). Cards created into the hand (Familiar/Grim/Incantation, Cryptid, DNA) are appended unsorted, as `G.hand:emplace` does |  | `tests/engine/audit/test_hand_mechanics.py::test_g39_sort_by_rank_is_descending`, `::test_g39_sort_by_suit_is_suit_descending`, `::test_g39_sort_mode_persists_for_later_draws`, `::test_g39_cards_created_into_the_hand_are_not_sorted_in` | python-fixed |
| G40 | tags | Orbital Tag: Lua rolls its hand when the blind-select screen is built, from visible hands, three per ante; a Double Tag copy keeps that hand. | UI_definitions.lua:1506; tag.lua:324 |  |  |  | open |
| G41 | tags | Boss Tag needs bans and `boss_rerolled`. | ledger: tag.lua:284-300 |  |  |  | open |
| G42 | tags | Coupon Tag needs its `shop_free` guard. | ledger: tag.lua:448-464 |  |  |  | open |
| G43 | tags | Packs opened by a tag must fire `open_booster` and the inflation counter. | ledger: tag.lua:209-282 |  |  |  | open |
| G44 | shop | Hallucination's Tarot is created before the pack contents (event order). | card.lua:1721 |  |  |  | open |
| G45 | actions | Missing action: boss reroll for Director's Cut/Retcon. | button_callbacks.lua:2784 |  |  |  | open |
| G46 | actions | Missing action: buy-and-use. | button_callbacks.lua:77/2407 |  |  |  | open |
| G47 | shop | The Overstock refill ignores Illusion and shop tags. | ledger: card.lua:1885; common_events.lua:1097-1118 |  |  |  | open |
| G48 | flow | `won` is reset by a later endless-mode loss. | ledger: state_events.lua:112-117 |  |  |  | open |
| G49 | scoring | Played cards score in hand order: highlighted cards are sorted by position. Python uses the click order of `card_indices`. | state_events.lua:463 | `game.py::_handle_play_hand` plays `hand[i] for i in sorted(indices)`; discards were already sorted |  | `tests/engine/audit/test_hand_mechanics.py::test_g49_played_cards_score_in_hand_order`, `::test_g49_discarded_cards_follow_hand_order` | python-fixed |
| G50 | scoring | `set_base` keeps `suit_nominal_original` and `original_value` across suit/rank changes. | card.lua:109-140 | `card.py::Card.set_base`: a non-initial call keeps the old `suit_nominal_original` and leaves `original_value` unset (vanilla nils it); Checkered Deck now goes through `Card.change_suit` (`deck_builder.py`), which also fixes its stale `card_key`. DNA/Cryptid copies (copy_card) still follow G29/G30 |  | `tests/engine/audit/test_hand_mechanics.py::test_g50_fresh_card_records_original_value`, `::test_g50_suit_change_keeps_original_suit_tiebreak`, `::test_g50_rank_change_keeps_original_suit_tiebreak`, `::test_g50_checkered_deck_keeps_original_suit_tiebreak` | python-fixed |
| G51 | scoring | `get_blind_amount` gives NaN, not inf, at extreme antes. Follow Lua in both engines. | ledger: misc_functions.lua:919-954 | `data/blind_scaling.py::get_blind_amount` returns NaN once the formula overflows (ante 39+); `Blind.beaten_by` is the literal `chips - blind.chips >= 0`, false for a NaN target; disable's /2 and /3 keep NaN |  | `tests/engine/audit/test_blinds.py::test_g51_overflowed_blind_amount_is_nan`, `::test_g51_nan_blind_can_never_be_beaten`, `::test_g51_nan_round_plays_to_game_over_without_crashing`, `::test_g51_disabling_a_nan_wall_keeps_nan` | python-fixed |

## Not gameplay, but fix them

| id | area | summary | lua | python fix | rust | test | status |
|---|---|---|---|---|---|---|---|
| B01 | env | The four dead observation features (`four_fingers`, `shortcut`, `smeared`, `splash`); `obs_version=2` fixes them. | — |  |  |  | open |
| B02 | env | Gym observations drop rows beyond the max counts, but the action table still indexes them. | — |  |  |  | open |
| B03 | bridge | `SimBackend`: `set shop` crashes. | — |  |  |  | open |
| B04 | bridge | `SimBackend`: `add` double-counts `d_size`. | — |  |  |  | open |
| B05 | bridge | `SimBackend`: a `rearrange` permutation applies only its first adjacent swap. | — |  |  |  | open |
| B06 | bridge | `SimBackend`: `add` can't add vouchers or packs. | — |  |  |  | open |
| B07 | bridge | `SimBackend`: the BalatroBot view mismatches listed in the plan's view section (hand `example` lists, `highlighted_limit`, card limits, labels, playing-card costs, `modifier.enhancement`, omitted false fields and `[]` for empty tables, `used_vouchers`, blind `score`, the MENU view, leftover shop areas). | — |  |  |  | open |
| B08 | docs | The README's `from jackdaw.bridge import BridgeAdapter` fails. | — |  |  |  | open |
| B09 | tooling | `extract_prototypes.lua` writes `%.15g` (the extracts need `%.17g`). | — |  |  |  | open |
| B10 | python | Drop the global `sort_id` counter (it should be per run). | — |  |  |  | open |
| B11 | python | Drop value-equal Cards (Lua cards have reference identity). | — |  |  |  | open |
| B12 | python | Drop big-int chips (Lua numbers are doubles). | — |  |  |  | open |
| B13 | python | `pseudohash` iterates code points; Lua iterates bytes. | — |  |  |  | open |
| B14 | python | `card_area._card_nominal` (used by `get_highest`) breaks ties with the raw `sort_id`. | — |  |  |  | open |

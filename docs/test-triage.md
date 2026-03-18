# Test Triage — Classification Audit

Every test function classified into one of four buckets:

- **ORACLE** — validates against Lua ground-truth fixtures (irreplaceable)
- **EDGE CASE** — tests tricky behavior not caught by broad integration tests
- **INTEGRATION** — exercises multiple systems together
- **CUT — REDUNDANT** — same code path already exercised by an oracle or integration test

---

## Summary

| File | ORACLE | EDGE | INTEG | CUT | Total |
|------|--------|------|-------|-----|-------|
| test_scoring_oracle.py | 115 | 0 | 0 | 0 | 115 |
| test_hand_eval_oracle.py | 19 | 0 | 0 | 0 | 19 |
| test_rng_oracle.py | 15 | 0 | 0 | 0 | 15 |
| test_shop_oracle.py | 13 | 0 | 0 | 0 | 13 |
| test_cross_validation.py | 40 | 0 | 0 | 0 | 40 |
| test_rng_sequence.py | 8 | 0 | 0 | 0 | 8 |
| test_scoring_jokers.py | 0 | 4 | 9 | 0 | 13 |
| test_scoring_phases.py | 0 | 5 | 9 | 0 | 14 |
| test_scoring_pipeline.py | 0 | 5 | 4 | 14 | 23 |
| test_scoring_integration.py | 0 | 6 | 10 | 21 | 37 |
| test_scoring.py | 0 | 5 | 0 | 22 | 27 |
| test_card_scoring.py | 0 | 8 | 0 | 48 | 56 |
| test_jokers.py | 0 | 26 | 0 | 94 | 120 |
| test_jokers_coverage.py | 0 | 8 | 3 | 8 | 19 |
| test_jokers_create.py | 0 | 10 | 0 | 14 | 24 |
| test_jokers_endround.py | 0 | 4 | 5 | 15 | 24 |
| test_jokers_j2j.py | 3 | 4 | 0 | 2 | 9 |
| test_jokers_destructive.py | 1 | 10 | 3 | 3 | 17 |
| test_jokers_retrigger.py | 6 | 0 | 6 | 6 | 18 |
| test_jokers_scaling.py | 1 | 14 | 1 | 7 | 23 |
| test_jokers_xmult.py | 0 | 26 | 0 | 5 | 31 |
| test_hand_eval.py | 0 | 15 | 10 | 50 | 75 |
| test_hand_eval_perf.py | 0 | 0 | 7 | 0 | 7 |
| test_evaluate_hand.py | 0 | 4 | 8 | 9 | 21 |
| test_hands.py | 0 | 2 | 0 | 21 | 23 |
| test_hand_levels.py | 0 | 6 | 0 | 32 | 38 |
| test_blind.py | 0 | 30 | 10 | 55 | 95 |
| test_blind_scaling.py | 0 | 3 | 0 | 20 | 23 |
| test_enums.py | 0 | 0 | 0 | 46 | 46 |
| test_card.py | 0 | 25 | 5 | 45 | 75 |
| test_card_area.py | 0 | 4 | 5 | 16 | 25 |
| test_card_factory.py | 0 | 15 | 10 | 30 | 55 |
| test_card_factory_resolvers.py | 0 | 8 | 5 | 20 | 33 |
| test_card_utils.py | 0 | 12 | 5 | 20 | 37 |
| test_data_integration.py | 0 | 0 | 7 | 20 | 27 |
| test_shop.py | 0 | 10 | 15 | 40 | 65 |
| test_shop_actions.py | 0 | 10 | 8 | 20 | 38 |
| test_shop_integration.py | 0 | 5 | 20 | 10 | 35 |
| test_pools.py | 0 | 15 | 10 | 30 | 55 |
| test_actions.py | 0 | 8 | 12 | 49 | 69 |
| test_back.py | 0 | 3 | 5 | 12 | 20 |
| test_challenges.py | 0 | 5 | 5 | 27 | 37 |
| test_consumables.py | 0 | 5 | 5 | 10 | 20 |
| test_consumables_integration.py | 0 | 5 | 20 | 5 | 30 |
| test_deck_builder.py | 0 | 3 | 5 | 23 | 31 |
| test_economy.py | 0 | 5 | 8 | 17 | 30 |
| test_packs.py | 0 | 5 | 15 | 30 | 50 |
| test_prototypes.py | 0 | 0 | 0 | 35 | 35 |
| test_rng.py | 0 | 5 | 0 | 20 | 25 |
| test_run_init.py | 0 | 5 | 15 | 20 | 40 |
| test_run_init_integration.py | 0 | 3 | 17 | 0 | 20 |
| test_round_lifecycle.py | 0 | 8 | 10 | 7 | 25 |
| test_stakes.py | 0 | 3 | 10 | 22 | 35 |
| test_tags.py | 0 | 5 | 5 | 10 | 20 |
| test_tag_generation.py | 0 | 3 | 10 | 7 | 20 |
| test_vouchers.py | 0 | 5 | 10 | 15 | 30 |
| test_runner.py | 0 | 0 | 18 | 0 | 18 |
| test_validator.py | 0 | 0 | 10 | 0 | 10 |
| test_full_validation.py | 0 | 0 | 30 | 0 | 30 |
| test_profile.py | 0 | 3 | 5 | 7 | 15 |
| test_game.py | 0 | 8 | 25 | 7 | 40 |
| test_balatrobot_adapter.py | 0 | 0 | 20 | 0 | 20 |
| test_mechanics_checklist.py | 0 | 5 | 5 | 0 | 10 |
| **TOTAL** | **221** | **~420** | **~470** | **~935** | **~2046** |

---

## File-by-File Classification

---

### test_scoring_oracle.py — ALL ORACLE (115)

All 115 parametrized tests (23 scenarios × 5 assertions: total, chips, mult, hand_type, debuffed) validate Python scoring pipeline against Lua ground-truth fixtures. **Irreplaceable.**

---

### test_hand_eval_oracle.py — ALL ORACLE (19)

- `TestFixtureOracle.test_all_detected_hands_match` — ORACLE
- `TestFixtureOracle.test_all_populated_hands_match` — ORACLE
- `TestFixtureOracle.test_individual_hand` (×17 parametrized) — ORACLE
- `TestLiveOracle.test_live_matches_fixture` — ORACLE
- `TestLiveOracle.test_all_hands_match_python` — ORACLE

---

### test_rng_oracle.py — ALL ORACLE (15)

All tests cross-validate pseudohash, pseudoseed, predict_seed sequences against Lua fixtures. **Irreplaceable.**

---

### test_shop_oracle.py — ALL ORACLE (13)

All tests validate shop population (voucher, joker keys, stickers, boosters) against LuaJIT fixtures. **Irreplaceable.**

---

### test_cross_validation.py — ALL ORACLE (40)

All tests validate run_init, scoring, shop, economy, hand_levels, targeting, consumables against oracle fixtures. **Irreplaceable.**

---

### test_rng_sequence.py — ALL ORACLE (8)

Tests simulate real Balatro RNG sequence (deck shuffle, boss selection, tag generation, shop population) against LuaJIT ground truth. **Irreplaceable.**

---

### test_scoring_jokers.py (13)

| Test | Bucket | Reason |
|------|--------|--------|
| `test_pair_of_aces_with_joker` | INTEGRATION | Full pipeline with j_joker |
| `test_no_jokers_matches_base` | INTEGRATION | Pipeline equivalence check |
| `test_flush_of_hearts` (lusty) | INTEGRATION | Per-card joker × 5 cards |
| `test_greedy_and_lusty` | INTEGRATION | Two per-card jokers stacking |
| `test_two_xmult_commutative` | EDGE CASE | xMult ordering proof |
| `test_two_xmult_reversed` | EDGE CASE | Reversed xMult order, same result |
| `test_additive_then_multiplicative` | EDGE CASE | Add→mult ordering matters |
| `test_multiplicative_then_additive` | EDGE CASE | Mult→add gives different result |
| `test_blueprint_copies_joker` | INTEGRATION | Blueprint dispatch in pipeline |
| `test_foil_edition_adds_chips` | INTEGRATION | Joker edition ordering |
| `test_polychrome_edition_multiplies_after` | INTEGRATION | Multiplicative edition after effect |
| `test_holo_edition_adds_mult` | INTEGRATION | Additive mult edition |
| `test_foil_and_polychrome_on_different_jokers` | INTEGRATION | Multiple joker editions |

---

### test_scoring_phases.py (14)

| Test | Bucket | Reason |
|------|--------|--------|
| `test_averages_chips_and_mult` (Plasma) | INTEGRATION | Phase 10 back card |
| `test_non_plasma_no_effect` | INTEGRATION | Control test |
| `test_plasma_with_joker` | INTEGRATION | Joker + Plasma stacking |
| `test_glass_scores_x2_then_may_shatter` | EDGE CASE | Glass shatter probability |
| `test_glass_guaranteed_shatter` | EDGE CASE | Forced shatter (RNG control) |
| `test_caino_gains_xmult_from_shattered_face` | EDGE CASE | Joker reaction to destruction |
| `test_glass_joker_gains_xmult_from_shatter` | EDGE CASE | Glass Joker shatter callback |
| `test_ice_cream_decays_after_scoring` | INTEGRATION | Phase 13 decay |
| `test_ice_cream_self_destructs` | EDGE CASE | Self-destruct threshold |
| `test_saves_losing_hand` (Mr. Bones) | INTEGRATION | Phase 13 save mechanic |
| `test_no_save_when_winning` | INTEGRATION | Control |
| `test_no_save_with_hands_remaining` | INTEGRATION | Control |
| `test_five_jokers_with_retrigger_and_destruction` | INTEGRATION | Complex multi-phase |
| `test_seltzer_retrigger_with_after_decay` | INTEGRATION | Retrigger + decay sequence |

---

### test_scoring_pipeline.py (23)

| Test | Bucket | Reason |
|------|--------|--------|
| `test_score` (Pair of Aces) | CUT | Oracle covers pair_aces_basic |
| `test_scoring_cards` | CUT | Implicit in oracle tests |
| `test_records_play` | CUT | Level tracking tested in hand_levels |
| `test_score` (Flush L3) | CUT | Oracle covers flush scenarios |
| `test_glass_non_scoring_card_no_effect` | EDGE CASE | Non-scoring glass ignored |
| `test_glass_in_pair` | CUT | Oracle covers flush_glass |
| `test_mult_adds` | CUT | Oracle covers via scoring scenarios |
| `test_bonus_chips` | CUT | Oracle covers via scoring scenarios |
| `test_held_x_mult` (Steel) | CUT | Oracle covers full_house_steel_held |
| `test_multiple_steel` | EDGE CASE | Stacking held x_mult (x2.25) |
| `test_foil_on_scored_card` | CUT | Oracle covers three_kings_foil |
| `test_holo_on_scored_card` | CUT | Oracle covers holo scenarios |
| `test_polychrome_on_scored_card` | CUT | Oracle covers polychrome |
| `test_double_chips` (Red seal) | CUT | Oracle covers pair_aces_red_seal |
| `test_glass_with_red_seal` | EDGE CASE | Glass x2 twice = x4 |
| `test_held_steel_with_red_seal` | EDGE CASE | Held x_mult retrigger |
| `test_halving` (Flint) | CUT | Oracle covers pair_aces_flint |
| `test_the_eye_repeat` | CUT | Oracle covers pair_eye_debuffed |
| `test_psychic_too_few_cards` | EDGE CASE | Boss debuff (card count) |
| `test_gold_seal` (dollars) | CUT | Tested in economy integration |
| `test_no_cards` (empty hand) | INTEGRATION | Edge: empty hand → NULL |
| `test_has_breakdown` | INTEGRATION | Output structure verification |
| `test_breakdown_shows_final` | INTEGRATION | Breakdown detail |
| `test_no_hand` (empty) | INTEGRATION | Edge case: empty hand |

---

### test_scoring_integration.py (37)

| Test | Bucket | Reason |
|------|--------|--------|
| `test_bonus_card` | CUT | Oracle covers bonus scenarios |
| `test_mult_card` | CUT | Oracle covers mult scenarios |
| `test_wild_card` | CUT | Neutral enhancement, trivial |
| `test_glass_card` | CUT | Oracle covers glass |
| `test_steel_card_held` | CUT | Oracle covers steel |
| `test_stone_card` | CUT | Covered by card_scoring edge cases |
| `test_gold_card_enhancement` | EDGE CASE | Gold h_dollars ≠ p_dollars distinction |
| `test_foil` | CUT | Oracle covers foil |
| `test_holographic` | CUT | Oracle covers holo |
| `test_polychrome` | CUT | Oracle covers poly |
| `test_basic_retrigger` | CUT | Oracle covers red_seal |
| `test_red_seal_glass` | EDGE CASE | Seal + enhancement stacking |
| `test_red_seal_polychrome` | EDGE CASE | Seal + edition stacking |
| `test_red_seal_holo` | CUT | Additive mult retrigger, oracle covers |
| `test_held_steel_red_seal` | CUT | Oracle covers |
| `test_mult_card_with_holo` | CUT | Enhancement + edition add, oracle covers |
| `test_glass_with_polychrome` | EDGE CASE | x2 × x1.5 = x3 stacking |
| `test_bonus_with_foil` | CUT | Chip stacking, trivial |
| `test_the_flint` | CUT | Oracle covers pair_aces_flint |
| `test_the_eye_first_allowed` | CUT | Oracle covers |
| `test_the_eye_repeat_blocked` | CUT | Oracle covers pair_eye_debuffed |
| `test_the_mouth_locks` | EDGE CASE | Mouth locking (different from Eye) |
| `test_the_psychic_5_cards_ok` | CUT | Oracle covers implicitly |
| `test_the_psychic_fewer_blocked` | CUT | Pipeline test covers this |
| `test_suit_debuff_goad` | INTEGRATION | Per-card suit debuff in pipeline |
| `test_face_debuff_plant` | INTEGRATION | Per-card rank debuff in pipeline |
| `test_debuffed_card_no_chips` | INTEGRATION | Card-level debuff suppression |
| `test_debuffed_card_no_edition` | EDGE CASE | Debuff suppresses edition |
| `test_debuffed_card_no_seal` | INTEGRATION | Debuff suppresses seal |
| `test_debuffed_card_no_glass_xmult` | INTEGRATION | Debuff suppresses glass |
| `test_scoring_throughput` | INTEGRATION | Performance benchmark |
| `test_pair_aces_level1` | CUT | Oracle has exact same value |
| `test_three_kings_foil` | CUT | Oracle has exact same value |
| `test_flush_glass` | CUT | Oracle has exact same value |
| `test_full_house_steel_held` | CUT | Oracle has exact same value |
| `test_pair_red_seal` | CUT | Oracle has exact same value |
| `test_pair_flint` | CUT | Oracle has exact same value |

---

### test_scoring.py (27)

Tests `eval_card()` context dispatch — a thin wrapper. Most paths are exercised by pipeline/oracle tests.

| Test | Bucket | Reason |
|------|--------|--------|
| `test_plain_ace_of_spades` | CUT | Pipeline tests exercise played cards |
| `test_plain_five` | CUT | Same |
| `test_bonus_card` | CUT | Pipeline tests bonus |
| `test_stone_card` | CUT | Pipeline tests stone |
| `test_mult_card` | CUT | Pipeline tests mult |
| `test_glass_card` | CUT | Pipeline tests glass |
| `test_gold_seal_dollars` | CUT | Economy tests cover |
| `test_foil_edition` | CUT | Pipeline tests foil |
| `test_holo_edition` | CUT | Pipeline tests holo |
| `test_polychrome_edition` | CUT | Pipeline tests poly |
| `test_debuffed_returns_empty` | CUT | Pipeline debuff tests cover |
| `test_combined_bonus_foil` | CUT | Integration tests cover stacking |
| `test_combined_mult_holo` | CUT | Integration tests cover |
| `test_zero_chips_not_included` | EDGE CASE | Zero-suppression in return dict |
| `test_plain_card_no_effect` (held) | CUT | Held pipeline covers |
| `test_steel_card` (held) | CUT | Pipeline covers steel held |
| `test_debuffed` (held) | CUT | Pipeline covers |
| `test_red_seal` (repetition) | EDGE CASE | Repetition-only mode |
| `test_no_seal` | CUT | Trivial default |
| `test_gold_seal_no_repetition` | EDGE CASE | Gold seal NOT a retrigger |
| `test_debuffed_red_seal` | CUT | Debuff tests cover |
| `test_repetition_only_ignores_play_context` | EDGE CASE | Flag precedence edge case |
| `test_foil_joker` (edition_only) | CUT | Joker pipeline covers |
| `test_holo_joker` | CUT | Same |
| `test_polychrome_joker` | CUT | Same |
| `test_no_edition_joker` | CUT | Trivial default |
| `test_debuffed_joker_edition` | EDGE CASE | Debuff suppresses joker edition |

---

### test_card_scoring.py (56)

Unit tests for Card scoring methods. Most paths are exercised by pipeline tests that call these methods.

**KEEP — EDGE CASE (8):**
- `test_stone_card` — Stone overrides nominal → 50 (special case)
- `test_stone_card_with_perma_bonus` — Stone + perma_bonus stacking
- `test_lucky_card_without_rng` — Lucky needs RNG, returns 0
- `test_x_mult_of_1_returns_0` — Threshold: x_mult ≤ 1 treated as 0
- `test_negative` (edition) — Negative has no scoring bonuses
- `test_edition_has_card_ref` — Edition dict includes self-reference
- `test_gold_card_enhancement` — Gold Card: h_dollars ≠ p_dollars
- `test_gold_seal_plus_gold_card` — Gold Seal + Gold Card: only $3

**CUT — REDUNDANT (48):**
All other tests (basic nominal chips, basic mult, basic x_mult, basic h_x_mult, basic edition lookups, debuff suppression per-method, seal basics, combined scoring basics) — every path exercised by pipeline integration tests and oracle tests.

---

### test_jokers.py (120)

**KEEP — EDGE CASE (26):**
- `TestRegistry` (3) — Registry mechanics (overwrite, sorted)
- `TestDispatch` (5) — Context routing, unhandled returns None
- `TestDebuff` (2) — Debuffed joker returns None
- `TestMisprintHandler` (4) — RNG-dependent random mult (0-23), deterministic seed
- `TestCardSharpHandler` (4) — Round counting, reset between blinds, second-play trigger
- `TestGreedyJokerHandler.test_wild_card_triggers` — Wild Card suit matching
- `TestGreedyJokerHandler.test_smeared_hearts_triggers` — Smeared suit grouping
- `TestGreedyJokerHandler.test_smeared_clubs_no_effect` — Smeared color boundary
- `TestBloodstoneHandler` (3) — Probabilistic trigger + deterministic seed
- `TestAncientJokerHandler.test_no_ancient_suit_no_effect` — Missing context guard
- `TestScaryFaceHandler.test_pareidolia_every_card` — Pareidolia makes all cards face
- `TestEvenStevenHandler.test_ace_no_effect` — Ace excluded from even
- `TestOddToddHandler.test_ace_triggers` — Ace special-cased as odd

**CUT — REDUNDANT (94):**
All simple flat-bonus joker tests where the joker is also present in scoring integration/oracle tests:
- `TestJokerHandler` (4) — j_joker +4 mult: oracle test `pair_aces_joker` validates end-to-end
- `TestStuntmanHandler` (2) — j_stuntman flat chips: covered by scoring pipeline
- `TestJollyHandler` (3) — j_jolly hand-type mult: oracle test `full_house_jolly` covers
- `TestZanyHandler` (2) — j_zany same pattern as Jolly
- `TestMadHandler` (1) — j_mad same pattern
- `TestCrazyHandler` (1) — j_crazy same pattern
- `TestDrollHandler` (1) — j_droll same pattern
- `TestSlyHandler` (1) — j_sly chips variant, same pattern
- `TestWilyHandler` (1) — j_wily same
- `TestCleverHandler` (1) — j_clever same
- `TestDeviousHandler` (1) — j_devious same
- `TestCraftyHandler` (1) — j_crafty same
- `TestDuoHandler` (2) — j_duo xMult: oracle test `pair_duo_xmult` covers
- `TestTrioHandler` (1) — j_trio same pattern
- `TestFamilyHandler` (1) — j_family same
- `TestOrderHandler` (1) — j_order same
- `TestTribeHandler` (1) — j_tribe same
- `TestSupernovaHandler` (3) — Stateless lookup, covered by pipeline
- `TestLustyJokerHandler` (2) — oracle `flush_hearts_lusty` covers
- `TestWrathfulJokerHandler` (2) — Same pattern as Greedy/Lusty
- `TestGluttenousJokerHandler` (2 of 3) — Same pattern (keep smeared test)
- `TestArrowheadHandler` (2 of 3) — Same pattern (keep wild test)
- `TestOnyxAgateHandler` (2) — Same pattern
- `TestRoughGemHandler` (2) — Same pattern
- `TestScholarHandler` (2) — Same rank-check pattern
- `TestWalkieTalkieHandler` (3) — Same rank-check pattern
- `TestFibonacciHandler` (7) — Rank membership tests; any integration test with Fibonacci would catch
- `TestSmileyFaceHandler` (1) — Same face-check pattern as Scary Face
- `TestEvenStevenHandler` (3 of 5) — Basic even check (keep Ace exclusion)
- `TestOddToddHandler` (4 of 5) — Basic odd check (keep Ace inclusion)
- `TestScaryFaceHandler` (4 of 5) — Basic face check (keep Pareidolia)
- `TestJokerResult` (2), `TestJokerContext` (2) — Dataclass construction, trivial
- `TestUnregistered` (1) — Trivial None return
- Remaining suit-specific "no_effect" / "joker_main_no_effect" tests — negative cases for simple conditionals

---

### test_jokers_coverage.py (19)

| Test | Bucket | Reason |
|------|--------|--------|
| `test_all_150_registered` | INTEGRATION | Audit: all jokers have handlers |
| `test_no_extra_registrations` | INTEGRATION | No phantom handlers |
| `test_noop_in_all_contexts` (×14) | INTEGRATION | Passive jokers truly passive |
| `test_castle_matching_suit_accumulates` | EDGE CASE | State mutation per discard |
| `test_castle_non_matching_no_effect` | CUT | Simple negative case |
| `test_castle_joker_main_returns` | EDGE CASE | Accumulated value returned |
| `test_runner_straight_accumulates` | EDGE CASE | State mutation per straight |
| `test_runner_non_straight_no_effect` | CUT | Simple negative case |
| `test_runner_joker_main_returns` | EDGE CASE | Accumulated value returned |
| `test_ramen_discard_decrements` | EDGE CASE | x_mult decay |
| `test_ramen_self_destructs_at_1` | EDGE CASE | Threshold self-destruction |
| `test_ramen_joker_main_gives_xmult` | CUT | Simple return |
| `test_mr_bones_game_over_saves` | EDGE CASE | Save mechanic |
| `test_mr_bones_normal_context_no_effect` | CUT | Simple negative |
| `test_burnt_first_discard_levels_up` | EDGE CASE | Conditional on discards_used=0 |
| `test_burnt_second_discard_no_effect` | CUT | Simple negative |
| `test_turtle_bean_end_of_round_decays` | CUT | Simple decay (same as Ramen) |
| `test_turtle_bean_self_destructs_at_zero` | CUT | Same pattern as Ramen |
| `test_perkeo_ending_shop_creates` | CUT | Simple trigger |
| `test_perkeo_other_context_no_effect` | CUT | Simple negative |

---

### test_jokers_create.py (24)

| Test | Bucket | Reason |
|------|--------|--------|
| `test_certificate_first_hand_drawn_creates` | CUT | Simple trigger, no tricky logic |
| `test_certificate_other_context_no_effect` | CUT | Simple negative |
| `test_marble_setting_blind_creates` | CUT | Simple trigger |
| `test_marble_other_context_no_effect` | CUT | Simple negative |
| `test_dna_first_hand_single_card` | EDGE CASE | Single-card + first-hand condition |
| `test_dna_first_hand_multiple_cards_no_effect` | EDGE CASE | Multiple cards blocked |
| `test_dna_second_hand_no_effect` | EDGE CASE | Only first hand |
| `test_dna_blueprint_does_not_copy` | EDGE CASE | Blueprint prevents mutation |
| `test_riff_raff_creates_two_jokers` | EDGE CASE | Slot-limited creation |
| `test_riff_raff_one_slot_creates_one` | EDGE CASE | Partial slot handling |
| `test_riff_raff_no_slots_no_effect` | EDGE CASE | Zero slot guard |
| `test_cartomancer_setting_blind_creates_tarot` | CUT | Simple trigger |
| `test_8_ball_rank_8_high_probability` | EDGE CASE | Probabilistic + rank check |
| `test_8_ball_rank_8_no_rng_no_effect` | CUT | RNG guard, trivial |
| `test_8_ball_non_8_no_effect` | CUT | Simple negative |
| `test_vagabond_low_money_creates_tarot` | CUT | Simple threshold |
| `test_vagabond_exact_threshold` | EDGE CASE | Boundary: exactly $4 |
| `test_vagabond_over_threshold_no_effect` | CUT | Simple negative |
| `test_superposition_ace_and_straight` | CUT | Simple dual condition |
| `test_superposition_no_ace_no_effect` | CUT | Simple negative |
| `test_superposition_ace_no_straight_no_effect` | CUT | Simple negative |
| `test_seance_matching_hand` | CUT | Simple trigger |
| `test_sixth_sense_rank_6_first_hand_single_card` | CUT | Similar to DNA pattern |
| `test_hallucination_high_probability_creates` | CUT | Simple probabilistic |

---

### test_jokers_endround.py (24)

| Test | Bucket | Reason |
|------|--------|--------|
| `test_golden_always_four_dollars` | CUT | Flat $4, trivial |
| `test_golden_debuffed_no_dollars` | CUT | Debuff suppression, tested broadly |
| `test_golden_via_on_end_of_round` | INTEGRATION | Orchestration function |
| `test_cloud9_three_nines` | CUT | Simple counting |
| `test_cloud9_zero_nines` | CUT | Simple zero case |
| `test_cloud9_no_tally_field` | EDGE CASE | Missing field handling |
| `test_rocket_base_dollars` | CUT | Simple return |
| `test_rocket_after_boss_increases` | EDGE CASE | State increment on boss beaten |
| `test_rocket_after_two_bosses` | EDGE CASE | Stacking across bosses |
| `test_rocket_non_boss_no_increase` | CUT | Simple negative |
| `test_satellite_three_planet_types` | CUT | Simple counting |
| `test_satellite_no_planets` | CUT | Simple zero case |
| `test_delayed_grat_zero_used_three_remaining` | CUT | Simple formula |
| `test_delayed_grat_one_used_no_effect` | CUT | Simple condition |
| `test_delayed_grat_zero_discards_left_no_effect` | CUT | Simple condition |
| `test_delayed_grat_via_on_end_of_round` | INTEGRATION | Orchestration function |
| `test_egg_sell_value_increases` | CUT | Simple increment |
| `test_egg_accumulates` | CUT | Simple accumulation |
| `test_gift_card_all_jokers_increase` | INTEGRATION | Multi-joker mutation |
| `test_gift_card_accumulates_across_rounds` | INTEGRATION | Cross-round state |
| `test_invisible_counts_rounds` | CUT | Simple counter |
| `test_invisible_sell_before_threshold_no_effect` | CUT | Simple threshold |
| `test_invisible_sell_at_threshold_duplicates` | EDGE CASE | Threshold-triggered duplication |
| `test_diet_cola_sell_creates_tag` | CUT | Simple sell effect |
| `test_space_joker_high_probability_levels_up` | EDGE CASE | Probabilistic level-up |
| `test_space_joker_no_rng_no_effect` | CUT | RNG guard |
| `test_space_joker_other_context_no_effect` | CUT | Context guard |
| `test_multiple_dollar_jokers` | INTEGRATION | Sum across jokers |
| `test_self_destruct_collected` | CUT | Implicit in pipeline tests |
| `test_egg_mutation_in_eor` | CUT | Implicit in orchestration |

---

### test_jokers_j2j.py (9)

| Test | Bucket | Reason |
|------|--------|--------|
| `test_uncommon_joker_triggers` | EDGE CASE | Rarity check logic |
| `test_common_joker_no_effect` | CUT | Simple negative |
| `test_rare_joker_no_effect` | CUT | Simple negative |
| `test_does_not_trigger_on_self` | EDGE CASE | Self-exclusion guard |
| `test_pipeline_three_uncommon_jokers` | ORACLE | Validates 648 against fixture |
| `test_pipeline_no_uncommon_no_effect` | ORACLE | Validates 192 against fixture |
| `test_pipeline_blueprint_not_uncommon` | ORACLE | Blueprint rarity interaction, fixture |
| `test_swashbuckler_basic_sell_sum` | EDGE CASE | Sell cost summation |
| `test_swashbuckler_excludes_self` | EDGE CASE | Self-exclusion |
| `test_swashbuckler_excludes_debuffed` | CUT (already implicit) | But keep for debuff clarity |
| `test_swashbuckler_pipeline` | ORACLE (partial) | Pipeline validates score |

---

### test_jokers_destructive.py (17)

| Test | Bucket | Reason |
|------|--------|--------|
| `test_gros_michel_joker_main_gives_mult` | CUT | Simple flat mult |
| `test_gros_michel_end_of_round_high_probability_destroys` | EDGE CASE | Probabilistic self-destruction + pool_flag |
| `test_gros_michel_end_of_round_survives` | EDGE CASE | Survival path |
| `test_gros_michel_pipeline_gives_mult` | ORACLE | Validates 544 against fixture |
| `test_cavendish_joker_main_gives_xmult` | CUT | Simple xMult |
| `test_cavendish_end_of_round_survives` | EDGE CASE | Survival path |
| `test_cavendish_end_of_round_high_probability_destroys` | EDGE CASE | Destruction path |
| `test_chicot_disables_boss_blind` | EDGE CASE | Boss blind disable mechanic |
| `test_chicot_non_boss_no_effect` | EDGE CASE | Boss-only guard |
| `test_chicot_already_disabled_no_effect` | EDGE CASE | Idempotency |
| `test_chicot_does_not_self_destruct` | CUT | Simple flag check |
| `test_luchador_sell_during_boss` | EDGE CASE | Sell-triggered disable |
| `test_luchador_sell_during_non_boss` | EDGE CASE | Non-boss guard |
| `test_burglar_setting_blind` | INTEGRATION | Rule modification (+3 hands, 0 discards) |
| `test_midas_mask_converts_face_cards` | INTEGRATION | Card mutation |
| `test_midas_mask_pareidolia_converts_all` | EDGE CASE | Pareidolia interaction |
| `test_hiker_adds_perma_bonus` | INTEGRATION | Permanent card mutation |
| `test_hiker_blueprint_does_not_mutate` | EDGE CASE | Blueprint prevents mutation |
| (remaining negative/control tests) | CUT | Simple negatives |

---

### test_jokers_retrigger.py (18)

| Test | Bucket | Reason |
|------|--------|--------|
| `test_unit_face_card_triggers` (sock_buskin) | CUT | Oracle pipeline test covers |
| `test_unit_non_face_no_effect` | CUT | Oracle pipeline covers |
| `test_pipeline_three_kings` | ORACLE | Validates 270 against fixture |
| `test_pipeline_pareidolia_all_retriggered` | ORACLE | Validates with Pareidolia |
| `test_unit_first_card` (hanging_chad) | CUT | Pipeline covers |
| `test_unit_second_card_no_effect` | CUT | Pipeline covers |
| `test_pipeline_pair_of_aces` (hanging_chad) | ORACLE | Validates 108 |
| `test_unit_last_hand` (dusk) | CUT | Pipeline covers |
| `test_unit_not_last_hand` | CUT | Pipeline covers |
| `test_pipeline_last_hand` (dusk) | ORACLE | Validates 108 |
| `test_red_seal_plus_sock` | ORACLE | Validates 120 (additive retriggers) |
| `test_unit_always_retriggers` (seltzer) | INTEGRATION | Retrigger mechanic |
| `test_after_decrements` | INTEGRATION | Decay mechanic |
| `test_self_destructs_at_zero` | INTEGRATION | Self-destruct threshold |
| `test_ten_hands_then_destruct` | INTEGRATION | Full lifecycle |
| `test_pipeline_all_cards_retriggered` (seltzer) | ORACLE | Validates 108 |
| `test_pipeline_steel_card_doubled` (mime) | INTEGRATION | Held + retrigger |
| `test_pipeline_two_steel_with_mime` | INTEGRATION | Multiple held + retrigger |
| `test_pipeline_pair_of_threes` (hack) | INTEGRATION | Rank retrigger |
| `test_pipeline_pair_of_sixes_no_retrigger` (hack) | INTEGRATION | Non-matching rank |

---

### test_jokers_scaling.py (23)

| Test | Bucket | Reason |
|------|--------|--------|
| `test_green_hand_1_adds_1` | EDGE CASE | Per-hand accumulation |
| `test_green_hand_2_adds_2` | EDGE CASE | Stacking verification |
| `test_green_discard_subtracts` | EDGE CASE | Discard penalty |
| `test_green_discard_clamps_to_zero` | EDGE CASE | Clamping at 0 |
| `test_green_joker_main_returns_accumulated` | CUT | Simple return |
| `test_green_blueprint_does_not_mutate` | EDGE CASE | Blueprint mutation prevention |
| `test_ride_the_bus_three_no_face_hands` | EDGE CASE | Consecutive no-face tracking |
| `test_ride_the_bus_face_card_resets` | EDGE CASE | Reset on face card |
| `test_ride_the_bus_joker_main_returns` | CUT | Simple return |
| `test_ride_the_bus_zero_mult_returns_none` | EDGE CASE | Zero optimization |
| `test_spare_trousers_two_pair_triggers` | CUT | Simple trigger |
| `test_spare_trousers_full_house_triggers` | CUT | Same pattern |
| `test_spare_trousers_flush_no_effect` | CUT | Simple negative |
| `test_spare_trousers_accumulates` | EDGE CASE | State accumulation |
| `test_square_four_cards_adds` | EDGE CASE | Exact count condition |
| `test_square_five_cards_no_effect` | EDGE CASE | Off-by-one boundary |
| `test_square_joker_main_returns` | CUT | Simple return |
| `test_ice_cream_after_one_hand` | CUT | Simple decay |
| `test_ice_cream_after_twenty_hands` | EDGE CASE | Full decay cycle |
| `test_ice_cream_hand_twenty_one_self_destructs` | EDGE CASE | Self-destruct threshold |
| `test_ice_cream_joker_main_returns` | CUT | Simple return |
| `test_ice_cream_in_scoring_pipeline` | ORACLE | Full pipeline validates state across hands |
| `test_popcorn_end_of_round_decrements` | CUT | Simple decay |
| `test_popcorn_self_destructs_at_zero` | EDGE CASE | Self-destruct threshold |
| `test_flash_card_accumulates` | CUT | Simple accumulation |
| `test_red_card_accumulates` | CUT | Simple accumulation |
| `test_wee_joker_score_one_two` | EDGE CASE | Rank-specific accumulation |
| `test_wee_joker_three_twos_across_hands` | EDGE CASE | Cross-hand state |
| `test_lucky_cat_lucky_trigger_accumulates` | EDGE CASE | Lucky trigger tracking |

---

### test_jokers_xmult.py (31)

All xMult jokers have stateful behavior (accumulation, resets, formulas). These are mostly **EDGE CASE** because:
- State mutation logic is per-joker and tricky
- Reset conditions (boss beaten, end-of-round, most-played hand) differ
- No oracle coverage for most xMult jokers

**KEEP — EDGE CASE (26):**
- `TestCampfire` (6) — Sell accumulation, boss reset, non-boss no-reset
- `TestHologram` (2) — Card-added accumulation
- `TestConstellation` (3) — Planet-used accumulation, tarot excluded
- `TestGlassJoker` (2) — Glass-destroyed accumulation
- `TestCaino` (5) — Face-destroyed, dual trigger with Glass Joker
- `TestVampire` (5) — Enhancement stripping, debuffed immunity
- `TestObelisk` (3) — Most-played reset logic
- `TestMadness` (2) — Boss guard + joker destruction signal
- `TestHitTheRoad` (3) — Jack-discard tracking + end-of-round reset
- `TestThrowback` (3) — Formula-based (skips × 0.25)
- `TestYorick` (4) — 23-discard threshold + reset counter
- `TestCeremonialDagger` (4) — Destroy neighbor + eternal immunity

**CUT — REDUNDANT (5):**
- `test_campfire_no_sells_no_effect` — Trivial default
- `test_hologram_add_three_cards` — Same pattern as add_one
- `test_vampire_base_card_no_effect` — Simple negative
- `test_vampire_accumulates_across_hands` — Repetitive accumulation (one test suffices)
- `test_glass_joker_non_glass_destroyed_no_effect` — Simple negative

---

### test_hand_eval.py (75)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 15 | Stone Card augmentation (3), Splash behavior (2), Pareidolia + debuff (3), Wild + Smeared combo (1), downward propagation (4), empty hand (1), Shortcut two-separated-gaps (1) |
| INTEGRATION | 10 | evaluate_poker_hand for all 12 hand types (covers detection + scoring card selection), joker flag integration (5) |
| CUT | 50 | Individual get_flush/get_straight/get_x_same/get_highest tests that duplicate what evaluate_poker_hand already exercises; basic hand detection already covered by oracle; simple flag tests already covered by integration |

---

### test_hand_eval_perf.py — ALL INTEGRATION (7)

Performance benchmarks. **Keep all.** These protect RL training throughput.

---

### test_evaluate_hand.py (21)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 4 | Stone card added as augmentation (2), scoring_cards preserve left-to-right order (1), empty hand (1) |
| INTEGRATION | 8 | Full pipeline detection with joker modifiers, multiple modifiers combo, debuffed joker |
| CUT | 9 | Basic detection (Flush, Full House, Pair, High Card, Straight) already covered by test_hand_eval.py::TestEvaluatePokerHand and oracle |

---

### test_hands.py (23)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 2 | `test_mult_minimum_is_1` (clamping), `test_chips_minimum_is_0` (clamping) |
| CUT | 21 | Enum membership (covered by any test using HandType), HAND_BASE values (covered by oracle + hand_levels), HAND_ORDER (covered by hand_eval priority tests), level formula (covered by hand_levels tests) |

---

### test_hand_levels.py (38)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 6 | `test_level_down` (The Arm), `test_level_down_minimum_zero`, `test_mult_minimum_1`, `test_level_up_makes_secret_visible`, `test_secret_hands_become_visible` (Black Hole), `test_reset_round_counts` |
| CUT | 32 | Init defaults (covered by run_init tests), get() values (covered by scoring oracle), level_up formula verification (same values tested in test_hands.py and oracle), dict access, repr, most_played default, record_play basics |

---

### test_blind.py (95)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 30 | The Eye state tracking (6), The Mouth locking (4), The Psychic blocking (4), The Flint rounding + minimums (5), debuff_card for Plant + Pareidolia (1), Pillar played_this_ante (2), Verdant Leaf conditional (3), disabled clears debuffs (3), press_play Hook unique indices (2) |
| INTEGRATION | 10 | get_new_boss determinism + ante filtering (6), get_ante_blinds structure (2), disable() side effects (Wall halves, Needle restores) (2) |
| CUT | 55 | Basic Small/Big/Boss chip values (covered by blind_scaling + oracle), basic get_type() returns, basic debuff_config lookups (suit-based debuffs: Goad/Head/Club/Window — all same pattern, one example suffices), non-debuff bosses returning None, Plasma scaling (covered by test_blind_scaling), empty blind, basic boss state, repr, basic dollars values |

---

### test_blind_scaling.py (23)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 3 | `test_ante_0` (returns base 100), `test_ante_negative` (returns base 100), `test_unknown_scaling_defaults_to_1` |
| CUT | 20 | All hardcoded antes 1-8 at scaling 1/2/3 (oracle validates these through cross_validation), exponential formula (oracle covers), BLIND_MULT constants (trivial), get_blind_target calculations (test_blind.py also tests these), Plasma doubling (test_blind.py covers) |

---

### test_enums.py — ALL CUT (46)

Every test verifies enum membership, count, or value. If any enum member were missing, dozens of other tests would fail with NameError/KeyError. The RANK_CHIPS and SUIT_NOMINAL values are validated by oracle tests. **Fully redundant.**

---

### test_card.py (75)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 25 | is_face debuffed + from_boss (4), is_face Pareidolia + debuffed combos (4), is_suit Wild Card debuffed (2), is_suit Stone never matches (1), is_suit Smeared color boundaries (4), deep copy isolation (4), perma_bonus preservation (2), set_cost discount minimum (1), rental override (2), astronomer gating (1) |
| INTEGRATION | 5 | set_ability_by_key comprehensive (joker, enhancement, nested extra), set_cost compound scenario, CardBase computed fields |
| CUT | 45 | Basic CardBase fields (id, nominal, suit_nominal — tested by oracle), basic set_ability lookups (tested by card_factory), basic sticker setters (tested by card_factory), basic set_cost arithmetic (tested by shop), sort_id auto-increment (tested by deck_builder), repr, basic edition fields |

---

### test_card_area.py (25)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 4 | `test_remove_also_unhighlights`, `test_no_duplicate_highlights`, `test_card_limit_minimum_zero`, `test_has_space_negative_bonus` |
| INTEGRATION | 5 | draw_card between areas (4), deterministic shuffle (1) |
| CUT | 16 | Basic add/remove/len (covered by game.py integration), basic draw (covered by run_init), basic sort (covered by game.py), basic highlighting add/remove/clear |

---

### test_card_factory.py (55)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 15 | Eternal/perishable mutual exclusivity (3), rental logic (3), edition assignment gating by card type (3), soul forced_key no modifiers (1), modifier stream consumption (2), area-specific seed keys (3) |
| INTEGRATION | 10 | create_card full pipeline (key determination, stickers, edition, cost) (5), card_from_control conversion (3), factory isolation (2) |
| CUT | 30 | Basic create_playing_card (52 cards — trivial), basic create_joker fields (covered by test_card.py + test_prototypes.py), basic create_consumable (covered by shop tests), basic create_voucher (covered by voucher tests), basic card_from_control fields, duplicate isolation tests |

---

### test_card_factory_resolvers.py (33)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 8 | destroy_random_joker eternal exclusion (2), all-eternal fallback (1), empty list fallback (1), copy_of deep copy (1), missing rank/suit returns None (2), unknown type returns None (1) |
| INTEGRATION | 5 | resolve_create for each type (Tarot, Planet, Spectral, Joker, PlayingCard) with full pipeline |
| CUT | 20 | Basic create returns correct type (covered by shop/pack tests), determinism checks (covered by oracle tests), known seed returns specific card (covered by shop_oracle), disable_blind returns None (trivial) |

---

### test_card_utils.py (37)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 12 | Boundary thresholds (Negative/Poly/Holo/Foil exact boundaries — 6), no_neg flag prevents Negative (2), mod=0 collapse (1), rate+mod compound (1), guaranteed+no_neg (1), x_mult_of_1_returns_0 (1) |
| INTEGRATION | 5 | Oracle with real RNG (determinism, stream advance, distribution) |
| CUT | 20 | Basic threshold ranges (covered by the boundary tests), basic guaranteed mode (covered by shop tests), basic rate parameter effects (same pattern as boundary tests), distribution frequency tests (statistical, not deterministic) |

---

### test_data_integration.py (27)

| Bucket | Count | Details |
|--------|-------|---------|
| INTEGRATION | 7 | Cross-reference validation (suit refs, voucher prereqs, enhancement gates, tag requires, blind fields, hand/planet mapping, pool keys) |
| CUT | 20 | Standard 52-card deck checks (covered by deck_builder), Abandoned/Erratic deck (covered by deck_builder), all 150 jokers creation (covered by prototypes), rarity pool integrity (covered by pools), blind scaling (covered by blind_scaling), hand data completeness (covered by hands), performance benchmarks (covered by full_validation) |

---

### test_shop.py (65)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 10 | Modified rates (Tarot Merchant, Planet Merchant, Magic Trick, Spectral) (5), Illusion modifier distribution (3), first shop guarantee bypass on ban (1), streams always consumed (1) |
| INTEGRATION | 15 | populate_shop structure + cards + determinism (8), type selection coverage (3), weighted pack selection (4) |
| CUT | 40 | Known seed type selection (covered by shop_oracle), default rate distribution (statistical, fragile), pack weight distribution (statistical), basic populate_shop card properties (covered by shop_integration), determinism (covered by oracle), Illusion known seeds (low value) |

---

### test_shop_actions.py (38)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 10 | Free reroll doesn't increment cost (1), negative free_rerolls clamped (1), temp_reroll_cost override (1), inflation recalculates remaining (1), Negative edition bypasses slot limit (1), eternal rejection (3), Chaos the Clown free_rerolls (2) |
| INTEGRATION | 8 | buy_card full flow (2), sell_card full flow (2), reroll_shop full flow (2), cost escalation (1), Flash Card notification (1) |
| CUT | 20 | Basic cost calculation (covered by shop_integration), basic insufficient funds (covered by shop_integration), basic card movement (covered by game.py), basic dollars deduction (covered by shop_integration), basic cards_purchased counter, basic used_jokers tracking |

---

### test_shop_integration.py (35)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 5 | Showman allows duplicates (3), Telescope forces most-played planet (2) |
| INTEGRATION | 20 | Full shop TESTSEED (8), Tarot/Planet merchant rates (4), reroll mechanics (6), full cycle (4) |
| CUT | 10 | Basic structure checks (covered by shop.py), basic determinism (covered by oracle), basic card properties |

---

### test_pools.py (55)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 15 | Enhancement gates (Lucky Cat, Steel, Stone, Glass locked) (7), pool flags (Gros Michel extinction, Cavendish) (5), all-banned fallback (2), soul/black_hole always unavailable (1) |
| INTEGRATION | 10 | select_from_pool with resample (3), pick_card_from_pool forwarding (4), check_soul_chance thresholds (3) |
| CUT | 30 | Basic pool sizes (covered by prototypes), basic rarity thresholds (covered by card_factory), basic banned key filtering (same pattern tested 4x), basic used_joker filtering (covered by shop tests), basic seed key construction (covered by shop_oracle), basic determinism |

---

### test_actions.py (69)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 8 | Boss blind no skip (1), no hands → no PlayHand (1), no discards → no Discard (1), empty hand → nothing (1), Negative edition bypasses slot limit (1), eternal not sellable (1), free reroll available (1), no choices remaining in pack (1) |
| INTEGRATION | 12 | Legal action generation for all 6 phases (BLIND_SELECT, SELECTING_HAND, SHOP, PACK_OPENING, ROUND_EVAL, GAME_OVER) |
| CUT | 49 | All 17 action type construction tests (frozen dataclass creation doesn't need testing), all frozen/hashable tests (Python guarantees this for frozen dataclasses), ActionUnion membership (compile-time type), basic legal action positive cases (covered by game.py integration which calls legal_actions) |

---

### test_back.py (20)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 3 | Plasma floors odd total (1), back_with_list_config normalization (1), Anaglyph boss_defeated only (1) |
| INTEGRATION | 5 | apply_to_run for complex decks (Magic, Zodiac, Painted), Plasma scoring phase 10, Anaglyph double tag |
| CUT | 12 | Simple delta returns (Red +1 discard, Blue +1 hand, etc.) — covered by run_init_integration which applies these and checks final state |

---

### test_challenges.py (37)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 5 | Fragile expands nested IDs (1), Non-Perishable bans perishable jokers (1), Mad World editions (1), Luxury Tax hand_size rule (1), Golden Needle overrides red deck (1) |
| INTEGRATION | 5 | Integration with initialize_run (3), banned keys expanded (2) |
| CUT | 27 | Basic data presence (20 challenges, required keys), basic modifier values (covered by integration tests that apply them), basic starting joker/voucher/consumable lists (low-value data spot checks) |

---

### test_consumables.py (~20)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 5 | Debuffed returns None (1), cards_in_play blocks (1), unregistered returns None (1), overwrite (1), spectral handler (1) |
| INTEGRATION | 5 | Dispatch returns ConsumableResult (2), multiple highlighted (1), registry sorted (1), custom values (1) |
| CUT | 10 | Basic registry add (trivial), basic defaults (trivial), basic dispatch structure |

---

### test_consumables_integration.py (~30)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 5 | Death copies enhancement+edition (1), Wheel of Fortune RNG (1), Liquidation 50% discount (1), rental before interest (1), no_interest modifier (1) |
| INTEGRATION | 20 | Chariot→Steel→scoring chain (3), Mercury→level_up→scoring (3), Black Hole→all levels (5), Strength→rank change→scoring (3), Hanged Man→destroy→deck shrink (3), Voucher hands chain (3) |
| CUT | 5 | Baseline "without consumable" control tests (low value, the positive test suffices) |

---

### test_deck_builder.py (31)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 3 | Checkered only 2 suits + suit_nominal updated (2), Erratic deterministic (1) |
| INTEGRATION | 5 | Standard 52-card structure (2), Abandoned 40-card no-faces (2), Erratic duplicates (1) |
| CUT | 23 | Basic 4×13 checks (covered by data_integration and run_init), basic rank presence (covered by data_integration), basic enhancement (trivial default), sort order (trivial), playing_card indices (trivial) |

---

### test_economy.py (~30)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 5 | no_extra_hand_money modifier (1), interest brackets (2), discard_cost modifier (1), negative earnings capped (1) |
| INTEGRATION | 8 | Full round earnings (blind + hands + interest) (3), Green Deck modifiers (2), rental integration (2), no_interest (1) |
| CUT | 17 | Basic defaults (trivial), basic blind reward values (covered by blind.py), basic hand bonus (covered by integration), basic discard cost 0 (trivial) |

---

### test_packs.py (~50)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 5 | Omen Globe introduces Spectrals (1), Telescope forces planet (1), mega pack choose > 1 (1), standard pack enhancement/seal presence (2) |
| INTEGRATION | 15 | generate_pack_cards for all 5 types (5), card count matches config (5), choose matches proto (5) |
| CUT | 30 | Basic return structure (trivial), basic type checks (trivial), basic card_count parametrized repeats (one per type suffices), determinism (covered by oracle) |

---

### test_prototypes.py — ALL CUT (35)

Every test verifies static data counts or spot-checks against Lua values. These are fully covered by:
- Oracle tests (which validate end-to-end behavior depending on correct prototypes)
- Card factory tests (which create from prototypes)
- Pool tests (which filter prototypes)

If a prototype count changed, oracle and pool tests would fail.

---

### test_rng.py (25)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 5 | Empty string hash (1), case sensitivity (1), long string range (1), stream independence (1), predict_seed stateless (1) |
| CUT | 20 | Basic pseudohash match (covered by rng_oracle), basic determinism (covered by oracle), basic PseudoRandom sequence (covered by rng_oracle and rng_sequence) |

---

### test_run_init.py (~40)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 5 | Fresh dict each call (1), bosses_used initialized (1), round_resets defaults (1), targeting cards populated (1), challenge overrides (1) |
| INTEGRATION | 15 | init_game_object structure (3), Magic/Zodiac/Painted deck effects (4), stake modifier application (3), challenge application (3), start_round (2) |
| CUT | 20 | Basic defaults (covered by run_init_integration), basic top_level_keys (structure check), basic starting params (covered by integration), basic Green/Plasma deck (covered by run_init_integration) |

---

### test_run_init_integration.py — ALL INTEGRATION (20)

Full chain initialize_run → start_round with real decks and stakes. Tests reproducibility and cross-cutting concerns. **Keep all.**

---

### test_round_lifecycle.py (~25)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 8 | Perishable countdown 1→0 becomes debuffed (1), full 5-round countdown (1), already at 0 no change (1), debuffed rental still fires (1), glass card destruction probability (2), rental+perishable on same joker (1), non-perishable unaffected (1) |
| INTEGRATION | 10 | process_round_end_cards orchestration (5), reset_round_targets (3), multiple rentals stacking (2) |
| CUT | 7 | Basic countdown decrements (one example suffices), basic rental cost (covered by economy), basic non-rental (trivial negative) |

---

### test_stakes.py (~35)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 3 | Stake 1 no modifiers (1), Stake 6 overrides Stake 3 scaling (1), Stake 8 all modifiers present (1) |
| INTEGRATION | 10 | Each stake level applied incrementally (8 stakes tested), inheritance chain verification (2) |
| CUT | 22 | Basic individual modifier values (covered by run_init_integration which applies stakes and checks final state), idempotency tests (trivial), basic unchanged fields |

---

### test_tags.py (~20)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 5 | Economy capped at max (1), negative dollars → zero (1), wrong context returns None (1), config is independent copy (1), double tag duplication (1) |
| INTEGRATION | 5 | Tag.apply for Economy, Garbage, Handy, Skip, Investment |
| CUT | 10 | Basic construction (trivial), basic triggered starts false (trivial), basic repr, basic zero cases (trivial negatives) |

---

### test_tag_generation.py (~20)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 3 | Different seeds differ (1), min_ante filtering (1), tag fallback (1) |
| INTEGRATION | 10 | assign_ante_blinds known seed (5), structure validation (3), game_state mutation (2) |
| CUT | 7 | Basic determinism (covered by rng_sequence + cross_validation), basic structure fields (covered by integration) |

---

### test_vouchers.py (~30)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 5 | Unknown key returns False (1), in-shop vouchers excluded (1), prerequisite chain unlocks dependent (1), Retcon requires Directors Cut (1), Antimatter requires Blank (1) |
| INTEGRATION | 10 | check_prerequisites for real vouchers (5), get_available_pool filtering (3), apply_voucher effects (2) |
| CUT | 15 | Basic no-requires passes (trivial), basic requires-missing (one example suffices), basic base vouchers always available (covered by pool tests), basic used voucher excluded (covered by shop tests) |

---

### test_runner.py — ALL INTEGRATION (18)

Crash resistance (100 seeds × 2 agents), determinism, performance, all decks/stakes. **Keep all.**

---

### test_validator.py — ALL INTEGRATION (10)

State comparison and validation reporting. **Keep all.**

---

### test_full_validation.py — ALL INTEGRATION (30)

Crash resistance, determinism, performance benchmarks, all decks/stakes. **Keep all.**

---

### test_profile.py (15)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 3 | Legendary bypasses unlock (1), no profile = no filter (1), known locked joker (1) |
| INTEGRATION | 5 | Fresh profile unlock counts (2), default profile all unlocked (1), pool filtering with profile (1), apply profile to game_state (1) |
| CUT | 7 | Basic discovery counts (trivial), basic all-items-discovered (trivial), basic locked count (covered by integration) |

---

### test_game.py (~40)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 8 | Wrong phase raises (2), empty indices raises (1), too many cards raises (1), eternal blocks sale (1), round won transitions (1), boss skip raises (1), skips counter (1) |
| INTEGRATION | 25 | step() for all action types (SelectBlind, SkipBlind, PlayHand, Discard, CashOut, SellCard, NextRound, SortHand, ReorderJokers, Reroll) + mini game full sequence |
| CUT | 7 | Basic field checks after step (e.g., "blind created with chips > 0" — implicit in any round test) |

---

### test_balatrobot_adapter.py — ALL INTEGRATION (20)

Action↔RPC conversion bridge. **Keep all** — external interface contract.

---

### test_mechanics_checklist.py — ALL KEEP (10)

| Bucket | Count | Details |
|--------|-------|---------|
| EDGE CASE | 5 | Purple seal on discard → Tarot, Gold seal end-of-round, Blue seal → Planet, Double tag duplication, card flipping |
| INTEGRATION | 5 | Boss press_play (Hook, Tooth, Fish), seal effects, Endless mode |

---

## Candidates for Immediate Deletion (Highest Confidence)

These files/tests are fully redundant with no unique value:

1. **test_enums.py** (46 tests) — Enum membership. Any missing member → NameError elsewhere.
2. **test_prototypes.py** (35 tests) — Static data counts. Oracle + factory tests cover.
3. **test_scoring.py** (22 CUT tests) — eval_card() wrapper. Pipeline tests exercise all paths.
4. **test_card_scoring.py** (48 CUT tests) — Card method units. Pipeline/oracle tests call these methods.
5. **test_hands.py** (21 CUT tests) — HandType data. Oracle + hand_levels tests cover.
6. **test_hand_levels.py** (32 CUT tests) — Init/get/level_up basics. Run_init + scoring oracle cover.
7. **test_blind_scaling.py** (20 CUT tests) — Hardcoded tables. Cross_validation oracle covers.
8. **test_actions.py** (49 CUT tests) — Frozen dataclass construction + hashability.
9. **test_jokers.py** (94 CUT tests) — Simple flat-bonus joker units. Oracle/pipeline tests cover.

**Estimated reduction: ~935 tests (~46% of total) are CUT candidates.**

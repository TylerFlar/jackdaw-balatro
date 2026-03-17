"""Tests for jackdaw.engine.challenges — CHALLENGES data + apply_challenge.

Coverage
--------
* CHALLENGES dict has exactly 20 entries, all with required keys.
* get_challenge returns the right challenge or None.
* apply_challenge: banned keys populated (Fragile bans j_blueprint-related items).
* apply_challenge: starting jokers with eternal/pinned flags stored.
* apply_challenge: forced modifiers override starting_params.
* apply_challenge: custom rules set modifiers (no_reward, no_shop_jokers, inflation).
* apply_challenge: starting vouchers marked used + effects applied.
* apply_challenge: starting consumables appended.
* apply_challenge: nested ids arrays in banned_cards expanded.
* Specific challenges: Omelette, Bram Poker, Jokerless, Golden Needle, Cruelty.
* Integration with initialize_run.
"""

from __future__ import annotations

from typing import Any

import pytest

from jackdaw.engine.challenges import CHALLENGES, apply_challenge, get_challenge
from jackdaw.engine.run_init import init_game_object

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_gs() -> dict[str, Any]:
    """A minimal game_state suitable for apply_challenge."""
    return init_game_object()


# ---------------------------------------------------------------------------
# CHALLENGES data integrity
# ---------------------------------------------------------------------------


class TestChallengesData:
    def test_exactly_20_challenges(self):
        assert len(CHALLENGES) == 20

    @pytest.mark.parametrize("cid", list(CHALLENGES.keys()))
    def test_required_keys_present(self, cid):
        ch = CHALLENGES[cid]
        for key in (
            "id",
            "name",
            "rules",
            "jokers",
            "consumeables",
            "vouchers",
            "deck",
            "restrictions",
        ):
            assert key in ch, f"{cid} missing key {key!r}"

    @pytest.mark.parametrize("cid", list(CHALLENGES.keys()))
    def test_rules_has_custom_and_modifiers(self, cid):
        rules = CHALLENGES[cid]["rules"]
        assert "custom" in rules
        assert "modifiers" in rules

    @pytest.mark.parametrize("cid", list(CHALLENGES.keys()))
    def test_restrictions_has_three_categories(self, cid):
        r = CHALLENGES[cid]["restrictions"]
        assert "banned_cards" in r
        assert "banned_tags" in r
        assert "banned_other" in r

    def test_all_ids_match_keys(self):
        for cid, ch in CHALLENGES.items():
            assert ch["id"] == cid


# ---------------------------------------------------------------------------
# get_challenge
# ---------------------------------------------------------------------------


class TestGetChallenge:
    def test_found(self):
        ch = get_challenge("c_omelette_1")
        assert ch is not None
        assert ch["name"] == "The Omelette"

    def test_not_found(self):
        assert get_challenge("c_nonexistent") is None


# ---------------------------------------------------------------------------
# apply_challenge — banned keys
# ---------------------------------------------------------------------------


class TestBannedKeys:
    def test_omelette_bans_money_cards(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_omelette_1"], gs)
        banned = gs["banned_keys"]
        assert banned.get("v_seed_money") is True
        assert banned.get("j_to_the_moon") is True
        assert banned.get("j_golden") is True

    def test_fragile_bans_standard_packs_with_ids(self):
        """Fragile has nested ids array for p_standard_normal_1."""
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_fragile_1"], gs)
        banned = gs["banned_keys"]
        assert banned.get("p_standard_normal_1") is True
        assert banned.get("p_standard_mega_1") is True
        assert banned.get("p_standard_jumbo_2") is True
        assert banned.get("tag_standard") is True

    def test_jokerless_bans_buffoon_packs_with_ids(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_jokerless_1"], gs)
        banned = gs["banned_keys"]
        assert banned.get("p_buffoon_normal_1") is True
        assert banned.get("p_buffoon_mega_1") is True
        assert banned.get("c_judgement") is True
        assert banned.get("v_antimatter") is True

    def test_jokerless_bans_tags(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_jokerless_1"], gs)
        banned = gs["banned_keys"]
        for tag in (
            "tag_rare",
            "tag_uncommon",
            "tag_holo",
            "tag_polychrome",
            "tag_negative",
            "tag_foil",
            "tag_buffoon",
            "tag_top_up",
        ):
            assert banned.get(tag) is True, f"{tag} not banned"

    def test_jokerless_bans_blinds(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_jokerless_1"], gs)
        banned = gs["banned_keys"]
        assert banned.get("bl_final_acorn") is True
        assert banned.get("bl_final_leaf") is True

    def test_non_perishable_bans_perishable_jokers(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_non_perishable_1"], gs)
        banned = gs["banned_keys"]
        assert banned.get("j_gros_michel") is True
        assert banned.get("j_ice_cream") is True
        assert banned.get("j_ramen") is True


# ---------------------------------------------------------------------------
# apply_challenge — starting jokers
# ---------------------------------------------------------------------------


class TestStartingJokers:
    def test_omelette_five_eggs(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_omelette_1"], gs)
        jokers = gs["challenge_jokers"]
        assert len(jokers) == 5
        assert all(j["id"] == "j_egg" for j in jokers)

    def test_knife_edge_eternal_pinned(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_knife_1"], gs)
        jokers = gs["challenge_jokers"]
        assert len(jokers) == 1
        assert jokers[0]["id"] == "j_ceremonial"
        assert jokers[0]["eternal"] is True
        assert jokers[0]["pinned"] is True

    def test_mad_world_editions(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_mad_world_1"], gs)
        jokers = gs["challenge_jokers"]
        pareidolia = jokers[0]
        assert pareidolia["id"] == "j_pareidolia"
        assert pareidolia["edition"] == "negative"
        assert pareidolia["eternal"] is True

    def test_no_jokers_no_key(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_jokerless_1"], gs)
        assert "challenge_jokers" not in gs


# ---------------------------------------------------------------------------
# apply_challenge — modifiers override starting_params
# ---------------------------------------------------------------------------


class TestModifiers:
    def test_blast_off_hands_and_discards(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_blast_off_1"], gs)
        sp = gs["starting_params"]
        assert sp["hands"] == 2
        assert sp["discards"] == 2
        assert sp["joker_slots"] == 4

    def test_five_card_draw(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_five_card_1"], gs)
        sp = gs["starting_params"]
        assert sp["hand_size"] == 5
        assert sp["joker_slots"] == 7
        assert sp["discards"] == 6

    def test_golden_needle(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_golden_needle_1"], gs)
        sp = gs["starting_params"]
        assert sp["hands"] == 1
        assert sp["discards"] == 6
        assert sp["dollars"] == 10

    def test_jokerless_zero_slots(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_jokerless_1"], gs)
        assert gs["starting_params"]["joker_slots"] == 0

    def test_cruelty_three_slots(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_cruelty_1"], gs)
        assert gs["starting_params"]["joker_slots"] == 3


# ---------------------------------------------------------------------------
# apply_challenge — custom rules
# ---------------------------------------------------------------------------


class TestCustomRules:
    def test_omelette_no_reward_all_blinds(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_omelette_1"], gs)
        nr = gs["modifiers"]["no_blind_reward"]
        assert nr["Small"] is True
        assert nr["Big"] is True
        assert nr["Boss"] is True

    def test_omelette_no_interest(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_omelette_1"], gs)
        assert gs["modifiers"]["no_interest"] is True
        assert gs["modifiers"]["no_extra_hand_money"] is True

    def test_bram_poker_no_shop_jokers(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_bram_poker_1"], gs)
        assert gs["joker_rate"] == 0

    def test_inflation_flag(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_inflation_1"], gs)
        assert gs["modifiers"]["inflation"] is True

    def test_double_nothing_debuff(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_double_nothing_1"], gs)
        assert gs["modifiers"]["debuff_played_cards"] is True

    def test_xray_flipped_cards(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_xray_1"], gs)
        assert gs["modifiers"]["flipped_cards"] == 4

    def test_cruelty_no_reward_small_and_big(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_cruelty_1"], gs)
        nr = gs["modifiers"]["no_blind_reward"]
        assert nr["Small"] is True
        assert nr["Big"] is True
        assert nr.get("Boss") is not True

    def test_golden_needle_discard_cost(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_golden_needle_1"], gs)
        assert gs["modifiers"]["discard_cost"] == 1

    def test_non_perishable_all_eternal(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_non_perishable_1"], gs)
        assert gs["modifiers"]["all_eternal"] is True

    def test_luxury_tax_hand_size_rule(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_luxury_1"], gs)
        assert gs["modifiers"]["minus_hand_size_per_X_dollar"] == 5
        assert gs["starting_params"]["hand_size"] == 10


# ---------------------------------------------------------------------------
# apply_challenge — starting vouchers
# ---------------------------------------------------------------------------


class TestStartingVouchers:
    def test_bram_poker_vouchers(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_bram_poker_1"], gs)
        assert gs["used_vouchers"].get("v_magic_trick") is True
        assert gs["used_vouchers"].get("v_illusion") is True
        # playing_card_rate should be set by v_magic_trick
        assert gs["playing_card_rate"] > 0

    def test_rich_get_richer_vouchers(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_rich_1"], gs)
        assert gs["used_vouchers"].get("v_seed_money") is True
        assert gs["used_vouchers"].get("v_money_tree") is True
        assert gs["interest_cap"] == 100  # Money Tree sets cap to 100

    def test_blast_off_vouchers(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_blast_off_1"], gs)
        assert gs["used_vouchers"].get("v_planet_merchant") is True
        assert gs["used_vouchers"].get("v_planet_tycoon") is True
        assert gs["planet_rate"] > 4  # Merchant + Tycoon boost rate


# ---------------------------------------------------------------------------
# apply_challenge — starting consumables
# ---------------------------------------------------------------------------


class TestStartingConsumables:
    def test_bram_poker_consumables(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_bram_poker_1"], gs)
        cons = gs.get("starting_consumables", [])
        assert "c_empress" in cons
        assert "c_emperor" in cons

    def test_challenge_without_consumables_no_key(self):
        gs = _fresh_gs()
        apply_challenge(CHALLENGES["c_omelette_1"], gs)
        # Omelette has empty consumeables — shouldn't add anything
        assert gs.get("starting_consumables", []) == []


# ---------------------------------------------------------------------------
# Integration: apply_challenge via initialize_run
# ---------------------------------------------------------------------------


class TestIntegrationWithInitializeRun:
    def test_omelette_through_initialize_run(self):
        from jackdaw.engine.run_init import initialize_run

        gs = initialize_run("b_red", 1, "OMELETTE_TEST", challenge=CHALLENGES["c_omelette_1"])
        # No reward flags set
        assert gs["modifiers"]["no_blind_reward"]["Small"] is True
        # Banned keys populated
        assert gs["banned_keys"].get("j_golden") is True
        # 5 eggs in challenge_jokers
        assert len(gs["challenge_jokers"]) == 5

    def test_jokerless_through_initialize_run(self):
        from jackdaw.engine.run_init import initialize_run

        gs = initialize_run("b_red", 1, "JOKERLESS_TEST", challenge=CHALLENGES["c_jokerless_1"])
        assert gs["joker_rate"] == 0
        assert gs["joker_slots"] == 0
        assert gs["round_resets"]["hands"] == 4  # unmodified

    def test_golden_needle_through_initialize_run(self):
        from jackdaw.engine.run_init import initialize_run

        gs = initialize_run("b_red", 1, "GOLDEN_TEST", challenge=CHALLENGES["c_golden_needle_1"])
        # Red Deck +1 discard is overridden by challenge modifier discards=6
        assert gs["round_resets"]["discards"] == 6
        assert gs["round_resets"]["hands"] == 1
        assert gs["dollars"] == 10

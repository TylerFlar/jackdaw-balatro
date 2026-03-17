"""Tests for jackdaw.engine.run_init — initialize_run, init_game_object, start_round.

Coverage
--------
* init_game_object: all key fields present with correct defaults.
* initialize_run with Red Deck, stake 1: discards=4 (3+1), hands=4.
* initialize_run with Black Deck, stake 5: hands=3, discards=2, joker_slots=6.
* initialize_run with Zodiac Deck: 3 vouchers, rates modified.
* initialize_run with Magic Deck: Crystal Ball active, 2 Fools, consumable_slots=3.
* initialize_run with Nebula Deck: consumable_slots=1 (-1 delta), Telescope voucher.
* initialize_run with Plasma Deck: ante_scaling=2.
* initialize_run with Green Deck: money_per_hand, money_per_discard, no_interest.
* start_round: hands_left/discards_left from round_resets.
* start_round with Juggle Tag temp_handsize: applied then cleared.
* start_round with round_bonus: next_hands and discards applied then cleared.
* Full chain: initialize_run → start_round → verify state.
* Challenge: starting vouchers, custom rules, banned keys.
* Targeting cards: idol, mail, ancient, castle reset with valid values.
* Deck: correct size, shuffled.
"""

from __future__ import annotations

from typing import Any

import pytest

from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.run_init import (
    get_starting_params,
    init_game_object,
    initialize_run,
    start_round,
)

# ---------------------------------------------------------------------------
# get_starting_params
# ---------------------------------------------------------------------------


class TestGetStartingParams:
    def test_defaults(self):
        sp = get_starting_params()
        assert sp["dollars"] == 4
        assert sp["hand_size"] == 8
        assert sp["discards"] == 3
        assert sp["hands"] == 4
        assert sp["reroll_cost"] == 5
        assert sp["joker_slots"] == 5
        assert sp["ante_scaling"] == 1
        assert sp["consumable_slots"] == 2
        assert sp["no_faces"] is False
        assert sp["erratic_suits_and_ranks"] is False

    def test_returns_fresh_dict(self):
        a = get_starting_params()
        b = get_starting_params()
        a["hands"] = 99
        assert b["hands"] == 4


# ---------------------------------------------------------------------------
# init_game_object
# ---------------------------------------------------------------------------


class TestInitGameObject:
    def test_top_level_keys(self):
        gs = init_game_object()
        expected = {
            "won",
            "round_scores",
            "joker_usage",
            "consumeable_usage",
            "hand_usage",
            "last_tarot_planet",
            "win_ante",
            "stake",
            "modifiers",
            "starting_params",
            "banned_keys",
            "round",
            "probabilities",
            "bosses_used",
            "pseudorandom",
            "starting_deck_size",
            "ecto_minus",
            "pack_size",
            "skips",
            "STOP_USE",
            "edition_rate",
            "joker_rate",
            "tarot_rate",
            "planet_rate",
            "spectral_rate",
            "playing_card_rate",
            "consumeable_buffer",
            "joker_buffer",
            "discount_percent",
            "interest_cap",
            "interest_amount",
            "inflation",
            "hands_played",
            "unused_discards",
            "perishable_rounds",
            "rental_rate",
            "blind",
            "chips",
            "dollars",
            "max_jokers",
            "bankrupt_at",
            "current_boss_streak",
            "base_reroll_cost",
            "blind_on_deck",
            "sort",
            "previous_round",
            "tags",
            "tag_tally",
            "pool_flags",
            "used_jokers",
            "used_vouchers",
            "current_round",
            "round_resets",
            "round_bonus",
            "shop",
        }
        assert expected.issubset(gs.keys()), f"Missing keys: {expected - gs.keys()}"

    def test_defaults(self):
        gs = init_game_object()
        assert gs["won"] is False
        assert gs["win_ante"] == 8
        assert gs["stake"] == 1
        assert gs["dollars"] == 0
        assert gs["chips"] == 0
        assert gs["round"] == 0
        assert gs["joker_rate"] == 20
        assert gs["base_reroll_cost"] == 5

    def test_bosses_used_initialized(self):
        gs = init_game_object()
        assert len(gs["bosses_used"]) > 0
        assert all(v == 0 for v in gs["bosses_used"].values())

    def test_round_resets_defaults(self):
        gs = init_game_object()
        rr = gs["round_resets"]
        assert rr["ante"] == 1
        assert rr["blind_choices"]["Small"] == "bl_small"
        assert rr["blind_choices"]["Big"] == "bl_big"

    def test_current_round_defaults(self):
        gs = init_game_object()
        cr = gs["current_round"]
        assert cr["hands_left"] == 0
        assert cr["discards_left"] == 0
        assert cr["reroll_cost"] == 5
        assert cr["idol_card"]["suit"] == "Spades"
        assert cr["idol_card"]["rank"] == "Ace"

    def test_fresh_dict_each_call(self):
        a = init_game_object()
        b = init_game_object()
        a["dollars"] = 999
        assert b["dollars"] == 0


# ---------------------------------------------------------------------------
# initialize_run — Red Deck
# ---------------------------------------------------------------------------


class TestInitializeRunRedDeck:
    """Red Deck (+1 discard), stake 1 = discards=4, hands=4."""

    @pytest.fixture()
    def gs(self):
        return initialize_run("b_red", 1, "TEST_RED")

    def test_discards(self, gs):
        assert gs["starting_params"]["discards"] == 4  # 3 + 1

    def test_hands(self, gs):
        assert gs["starting_params"]["hands"] == 4

    def test_round_resets_discards(self, gs):
        assert gs["round_resets"]["discards"] == 4

    def test_dollars(self, gs):
        assert gs["dollars"] == 4

    def test_deck_size(self, gs):
        assert len(gs["deck"]) == 52


# ---------------------------------------------------------------------------
# initialize_run — Black Deck + stake 5
# ---------------------------------------------------------------------------


class TestInitializeRunBlackDeckStake5:
    """Black Deck (-1 hand, +1 joker slot), stake 5 (-1 discard)."""

    @pytest.fixture()
    def gs(self):
        return initialize_run("b_black", 5, "TEST_BLACK5")

    def test_hands(self, gs):
        assert gs["starting_params"]["hands"] == 3  # 4 - 1

    def test_discards(self, gs):
        assert gs["starting_params"]["discards"] == 2  # 3 - 1 (stake 5)

    def test_joker_slots(self, gs):
        assert gs["starting_params"]["joker_slots"] == 6  # 5 + 1
        assert gs["joker_slots"] == 6

    def test_round_resets_hands(self, gs):
        assert gs["round_resets"]["hands"] == 3

    def test_round_resets_discards(self, gs):
        assert gs["round_resets"]["discards"] == 2

    def test_stake_modifiers(self, gs):
        mods = gs["modifiers"]
        assert mods["enable_eternals_in_shop"] is True
        assert mods["scaling"] == 2  # stake 5 gets scaling=2 from stake 3


# ---------------------------------------------------------------------------
# initialize_run — Zodiac Deck
# ---------------------------------------------------------------------------


class TestInitializeRunZodiacDeck:
    """Zodiac Deck has 3 starting vouchers: tarot_merchant, planet_merchant, overstock_norm."""

    @pytest.fixture()
    def gs(self):
        return initialize_run("b_zodiac", 1, "TEST_ZODIAC")

    def test_three_vouchers_in_used(self, gs):
        uv = gs["used_vouchers"]
        assert uv.get("v_tarot_merchant") is True
        assert uv.get("v_planet_merchant") is True
        assert uv.get("v_overstock_norm") is True

    def test_tarot_rate_modified(self, gs):
        assert gs["tarot_rate"] == 9.6  # 4 * 2.4

    def test_planet_rate_modified(self, gs):
        assert gs["planet_rate"] == 9.6

    def test_shop_joker_max_increased(self, gs):
        assert gs["shop"]["joker_max"] == 3  # 2 + 1


# ---------------------------------------------------------------------------
# initialize_run — Magic Deck
# ---------------------------------------------------------------------------


class TestInitializeRunMagicDeck:
    """Magic Deck: Crystal Ball voucher + 2 Fools."""

    @pytest.fixture()
    def gs(self):
        return initialize_run("b_magic", 1, "TEST_MAGIC")

    def test_crystal_ball_voucher_active(self, gs):
        assert gs["used_vouchers"].get("v_crystal_ball") is True

    def test_consumable_slots_three(self, gs):
        # Crystal Ball adds +1 to consumable_slots
        assert gs["consumable_slots"] == 3

    def test_two_fools_in_starting_consumables(self, gs):
        assert gs["starting_consumables"] == ["c_fool", "c_fool"]


# ---------------------------------------------------------------------------
# initialize_run — Nebula Deck
# ---------------------------------------------------------------------------


class TestInitializeRunNebulaDeck:
    """Nebula Deck: -1 consumable slot, Telescope voucher."""

    @pytest.fixture()
    def gs(self):
        return initialize_run("b_nebula", 1, "TEST_NEBULA")

    def test_consumable_slots_one(self, gs):
        assert gs["consumable_slots"] == 1  # 2 - 1

    def test_telescope_voucher(self, gs):
        assert gs["used_vouchers"].get("v_telescope") is True


# ---------------------------------------------------------------------------
# initialize_run — Plasma Deck
# ---------------------------------------------------------------------------


class TestInitializeRunPlasmaDeck:
    @pytest.fixture()
    def gs(self):
        return initialize_run("b_plasma", 1, "TEST_PLASMA")

    def test_ante_scaling(self, gs):
        assert gs["starting_params"]["ante_scaling"] == 2


# ---------------------------------------------------------------------------
# initialize_run — Green Deck
# ---------------------------------------------------------------------------


class TestInitializeRunGreenDeck:
    @pytest.fixture()
    def gs(self):
        return initialize_run("b_green", 1, "TEST_GREEN")

    def test_money_per_hand(self, gs):
        assert gs["money_per_hand"] == 2

    def test_money_per_discard(self, gs):
        assert gs["money_per_discard"] == 1

    def test_no_interest(self, gs):
        assert gs["no_interest"] is True

    def test_dollars_unchanged(self, gs):
        # Green Deck has no dollars_delta
        assert gs["dollars"] == 4


# ---------------------------------------------------------------------------
# initialize_run — Ghost Deck
# ---------------------------------------------------------------------------


class TestInitializeRunGhostDeck:
    @pytest.fixture()
    def gs(self):
        return initialize_run("b_ghost", 1, "TEST_GHOST")

    def test_spectral_rate(self, gs):
        assert gs["spectral_rate"] == 2

    def test_hex_starting_consumable(self, gs):
        assert gs["starting_consumables"] == ["c_hex"]


# ---------------------------------------------------------------------------
# initialize_run — Yellow Deck (starting money)
# ---------------------------------------------------------------------------


class TestInitializeRunYellowDeck:
    def test_extra_dollars(self):
        gs = initialize_run("b_yellow", 1, "TEST_YELLOW")
        assert gs["dollars"] == 14  # 4 + 10


# ---------------------------------------------------------------------------
# initialize_run — structure and determinism
# ---------------------------------------------------------------------------


class TestInitializeRunStructure:
    @pytest.fixture()
    def gs(self):
        return initialize_run("b_red", 1, "STRUCT_TEST")

    def test_rng_present(self, gs):
        from jackdaw.engine.rng import PseudoRandom

        assert isinstance(gs["rng"], PseudoRandom)

    def test_hand_levels_present(self, gs):
        from jackdaw.engine.hand_levels import HandLevels

        assert isinstance(gs["hand_levels"], HandLevels)

    def test_boss_is_valid(self, gs):
        from jackdaw.engine.data.prototypes import BLINDS

        boss = gs["round_resets"]["blind_choices"]["Boss"]
        assert boss in BLINDS

    def test_tags_are_valid(self, gs):
        from jackdaw.engine.data.prototypes import TAGS

        tags = gs["round_resets"]["blind_tags"]
        assert tags["Small"] in TAGS
        assert tags["Big"] in TAGS

    def test_deterministic(self):
        gs1 = initialize_run("b_red", 1, "DETERM")
        gs2 = initialize_run("b_red", 1, "DETERM")
        # Compare non-object fields
        assert gs1["dollars"] == gs2["dollars"]
        assert gs1["starting_params"] == gs2["starting_params"]
        assert gs1["round_resets"]["blind_choices"] == gs2["round_resets"]["blind_choices"]
        assert gs1["round_resets"]["blind_tags"] == gs2["round_resets"]["blind_tags"]


# ---------------------------------------------------------------------------
# initialize_run — targeting cards
# ---------------------------------------------------------------------------


class TestTargetingCards:
    @pytest.fixture()
    def gs(self):
        return initialize_run("b_red", 1, "TARGET_TEST")

    def test_idol_card_has_rank_and_suit(self, gs):
        idol = gs["current_round"]["idol_card"]
        assert "rank" in idol
        assert "suit" in idol
        assert idol["rank"] in [r.value for r in Rank]
        assert idol["suit"] in [s.value for s in Suit]

    def test_mail_card_has_rank(self, gs):
        mail = gs["current_round"]["mail_card"]
        assert "rank" in mail
        assert mail["rank"] in [r.value for r in Rank]

    def test_ancient_card_has_suit(self, gs):
        ancient = gs["current_round"]["ancient_card"]
        assert "suit" in ancient
        assert ancient["suit"] in [s.value for s in Suit]

    def test_castle_card_has_suit(self, gs):
        castle = gs["current_round"]["castle_card"]
        assert "suit" in castle
        assert castle["suit"] in [s.value for s in Suit]


# ---------------------------------------------------------------------------
# start_round — basic
# ---------------------------------------------------------------------------


class TestStartRound:
    def test_hands_left_from_round_resets(self):
        gs = initialize_run("b_red", 1, "START_ROUND")
        start_round(gs)
        assert gs["current_round"]["hands_left"] == gs["round_resets"]["hands"]

    def test_discards_left_from_round_resets(self):
        gs = initialize_run("b_red", 1, "START_ROUND")
        start_round(gs)
        assert gs["current_round"]["discards_left"] == gs["round_resets"]["discards"]

    def test_hands_played_reset(self):
        gs = initialize_run("b_red", 1, "START_ROUND")
        gs["current_round"]["hands_played"] = 5
        start_round(gs)
        assert gs["current_round"]["hands_played"] == 0

    def test_discards_used_reset(self):
        gs = initialize_run("b_red", 1, "START_ROUND")
        gs["current_round"]["discards_used"] = 2
        start_round(gs)
        assert gs["current_round"]["discards_used"] == 0

    def test_reroll_cost_reset(self):
        gs = initialize_run("b_red", 1, "START_ROUND")
        gs["current_round"]["reroll_cost_increase"] = 10
        start_round(gs)
        # After reset, reroll_cost_increase is 0 and cost = base
        assert gs["current_round"]["reroll_cost_increase"] == 0

    def test_hand_levels_round_counts_reset(self):
        gs = initialize_run("b_red", 1, "START_ROUND")
        from jackdaw.engine.data.hands import HandType

        gs["hand_levels"].record_play(HandType.PAIR)
        assert gs["hand_levels"][HandType.PAIR].played_this_round == 1
        start_round(gs)
        assert gs["hand_levels"][HandType.PAIR].played_this_round == 0
        # But total played count is preserved
        assert gs["hand_levels"][HandType.PAIR].played == 1


# ---------------------------------------------------------------------------
# start_round — Juggle Tag temp_handsize
# ---------------------------------------------------------------------------


class TestStartRoundJuggleTag:
    def test_temp_handsize_applied_then_cleared(self):
        gs = initialize_run("b_red", 1, "JUGGLE")
        original_hand_size = gs["hand_size"]
        gs["round_resets"]["temp_handsize"] = 3  # Juggle Tag effect
        start_round(gs)
        # Hand size increased
        assert gs["hand_size"] == original_hand_size + 3
        # temp_handsize cleared
        assert gs["round_resets"]["temp_handsize"] is None

    def test_no_temp_handsize_no_change(self):
        gs = initialize_run("b_red", 1, "NO_JUGGLE")
        original_hand_size = gs["hand_size"]
        start_round(gs)
        assert gs["hand_size"] == original_hand_size


# ---------------------------------------------------------------------------
# start_round — round_bonus
# ---------------------------------------------------------------------------


class TestStartRoundRoundBonus:
    def test_next_hands_applied_then_cleared(self):
        gs = initialize_run("b_red", 1, "BONUS_HANDS")
        gs["round_bonus"]["next_hands"] = 2
        start_round(gs)
        expected = gs["round_resets"]["hands"] + 2
        # next_hands was applied before clearing
        assert gs["current_round"]["hands_left"] == expected
        assert gs["round_bonus"]["next_hands"] == 0

    def test_bonus_discards_applied_then_cleared(self):
        gs = initialize_run("b_red", 1, "BONUS_DISC")
        gs["round_bonus"]["discards"] = 1
        start_round(gs)
        expected = gs["round_resets"]["discards"] + 1
        assert gs["current_round"]["discards_left"] == expected
        assert gs["round_bonus"]["discards"] == 0


# ---------------------------------------------------------------------------
# Full chain: initialize_run → start_round
# ---------------------------------------------------------------------------


class TestFullChain:
    def test_red_deck_full_chain(self):
        gs = initialize_run("b_red", 1, "FULL_CHAIN")
        start_round(gs)

        # Red Deck: +1 discard → discards=4
        assert gs["current_round"]["hands_left"] == 4
        assert gs["current_round"]["discards_left"] == 4

        # Other state is sane
        assert gs["dollars"] == 4
        assert len(gs["deck"]) == 52
        assert gs["stake"] == 1
        assert gs["hand_size"] == 8
        assert gs["joker_slots"] == 5

    def test_black_deck_stake8_full_chain(self):
        gs = initialize_run("b_black", 8, "FULL_BLACK8")
        start_round(gs)

        # Black Deck: -1 hand, +1 joker slot
        # Stake 8: -1 discard (from stake 5), all sticker flags
        assert gs["current_round"]["hands_left"] == 3  # 4-1
        assert gs["current_round"]["discards_left"] == 2  # 3-1
        assert gs["joker_slots"] == 6  # 5+1
        assert gs["modifiers"]["enable_eternals_in_shop"] is True
        assert gs["modifiers"]["enable_perishables_in_shop"] is True
        assert gs["modifiers"]["enable_rentals_in_shop"] is True
        assert gs["modifiers"]["scaling"] == 3


# ---------------------------------------------------------------------------
# Challenge
# ---------------------------------------------------------------------------


class TestChallenge:
    def test_challenge_starting_vouchers(self):
        challenge: dict[str, Any] = {
            "id": "test_challenge",
            "vouchers": [{"id": "v_grabber"}],
        }
        gs = initialize_run("b_red", 1, "CH_VOUCH", challenge=challenge)
        assert gs["used_vouchers"].get("v_grabber") is True
        # Grabber's round_resets.hands effect is overwritten by the
        # starting_params → round_resets transfer (matching Lua behaviour).
        # The voucher is still marked as used for its passive effects.
        assert gs["round_resets"]["hands"] == 4  # unchanged from starting_params

    def test_challenge_custom_rules_no_reward(self):
        challenge: dict[str, Any] = {
            "id": "test_no_reward",
            "rules": {
                "custom": [{"id": "no_reward"}],
            },
        }
        gs = initialize_run("b_red", 1, "CH_NORW", challenge=challenge)
        nr = gs["modifiers"]["no_blind_reward"]
        assert nr["Small"] is True
        assert nr["Big"] is True
        assert nr["Boss"] is True

    def test_challenge_banned_keys(self):
        challenge: dict[str, Any] = {
            "id": "test_banned",
            "restrictions": {
                "banned_cards": [
                    {"id": "j_joker", "ids": ["j_jolly"]},
                ],
                "banned_tags": [
                    {"id": "tag_double"},
                ],
            },
        }
        gs = initialize_run("b_red", 1, "CH_BAN", challenge=challenge)
        assert gs["banned_keys"]["j_joker"] is True
        assert gs["banned_keys"]["j_jolly"] is True
        assert gs["banned_keys"]["tag_double"] is True

    def test_challenge_modifiers(self):
        challenge: dict[str, Any] = {
            "id": "test_mods",
            "rules": {
                "modifiers": [
                    {"id": "hands", "value": 1},
                    {"id": "discards", "value": 1},
                ],
            },
        }
        gs = initialize_run("b_red", 1, "CH_MODS", challenge=challenge)
        # Challenge overrides starting_params.hands to 1
        assert gs["round_resets"]["hands"] == 1
        # Red Deck +1 discard is applied first, then challenge overrides to 1
        assert gs["round_resets"]["discards"] == 1

    def test_challenge_custom_no_shop_jokers(self):
        challenge: dict[str, Any] = {
            "id": "test_no_jokers",
            "rules": {
                "custom": [{"id": "no_shop_jokers"}],
            },
        }
        gs = initialize_run("b_red", 1, "CH_NSJ", challenge=challenge)
        assert gs["joker_rate"] == 0

    def test_challenge_starting_consumables(self):
        challenge: dict[str, Any] = {
            "id": "test_cons",
            "consumeables": [{"id": "c_strength"}, {"id": "c_death"}],
        }
        gs = initialize_run("b_red", 1, "CH_CONS", challenge=challenge)
        assert "c_strength" in gs["starting_consumables"]
        assert "c_death" in gs["starting_consumables"]

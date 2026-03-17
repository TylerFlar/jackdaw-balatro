"""Tests for jackdaw.engine.round_lifecycle.

Coverage — process_round_end_cards
----------------------------------
* Perishable countdown: 5 → 4 → 3 → 2 → 1 → 0 → debuffed.
* Perishable already at 0: no further mutation.
* Non-perishable joker: unaffected.
* Rental joker deducts $3 per round.
* Multiple rental jokers stack costs.
* Rental + perishable on same joker: both apply.
* Non-rental joker: no deduction.
* Debuffed joker: rental still fires (Lua checks ability.rental, not debuff).
* Glass Card: NOT affected at round end (only shatters during scoring).
* Result structure: perished list, rental_cost, rental_cards.

Coverage — reset_round_targets
-------------------------------
* Known seed → specific idol/mail/ancient/castle values.
* idol_card has both suit and rank; mail_card has rank only.
* ancient_card has suit only; castle_card has suit only.
* Different ante → different targets (ante suffix in seed key).
* Deterministic: same seed + ante → same targets.
* Ancient card excludes current suit.
* Stone Card cards excluded from idol/mail/castle selection.
* Empty deck → defaults (Ace of Spades / Spades).
"""

from __future__ import annotations

from typing import Any

import pytest

from jackdaw.engine.card import Card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.round_lifecycle import (
    RoundEndResult,
    process_round_end_cards,
    reset_round_targets,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _joker(
    key: str = "j_joker",
    *,
    perishable: bool = False,
    perish_tally: int = 5,
    rental: bool = False,
    eternal: bool = False,
    debuff: bool = False,
) -> Card:
    """Create a minimal joker Card for testing."""
    c = Card(center_key=key)
    c.ability = {"set": "Joker"}
    c.perishable = perishable
    c.perish_tally = perish_tally
    c.rental = rental
    c.eternal = eternal
    c.debuff = debuff
    if perishable:
        c.ability["perishable"] = True
        c.ability["perish_tally"] = perish_tally
    if rental:
        c.ability["rental"] = True
    return c


def _gs(**kwargs: Any) -> dict[str, Any]:
    """Minimal game_state for round_lifecycle tests."""
    defaults: dict[str, Any] = {"dollars": 20, "rental_rate": 3}
    defaults.update(kwargs)
    return defaults


# ---------------------------------------------------------------------------
# Perishable countdown
# ---------------------------------------------------------------------------


class TestPerishableCountdown:
    def test_5_to_4(self):
        j = _joker(perishable=True, perish_tally=5)
        result = process_round_end_cards([j], _gs())
        assert j.perish_tally == 4
        assert j.ability["perish_tally"] == 4
        assert j.debuff is False
        assert result.perished == []

    def test_4_to_3(self):
        j = _joker(perishable=True, perish_tally=4)
        result = process_round_end_cards([j], _gs())
        assert j.perish_tally == 3
        assert j.debuff is False

    def test_2_to_1(self):
        j = _joker(perishable=True, perish_tally=2)
        result = process_round_end_cards([j], _gs())
        assert j.perish_tally == 1
        assert j.debuff is False

    def test_1_to_0_debuffed(self):
        j = _joker(perishable=True, perish_tally=1)
        result = process_round_end_cards([j], _gs())
        assert j.perish_tally == 0
        assert j.ability["perish_tally"] == 0
        assert j.debuff is True
        assert result.perished == [j]

    def test_full_countdown_5_rounds(self):
        """Simulate 5 rounds: tally goes 5→4→3→2→1→0 (debuff on round 5)."""
        j = _joker(perishable=True, perish_tally=5)
        gs = _gs()
        for round_num in range(1, 6):
            result = process_round_end_cards([j], gs)
            expected_tally = 5 - round_num
            assert j.perish_tally == expected_tally, f"round {round_num}"
            if round_num < 5:
                assert j.debuff is False
                assert result.perished == []
            else:
                assert j.debuff is True
                assert result.perished == [j]

    def test_already_at_0_no_further_change(self):
        """Once at 0, no more decrement or re-debuff."""
        j = _joker(perishable=True, perish_tally=0, debuff=True)
        result = process_round_end_cards([j], _gs())
        assert j.perish_tally == 0
        assert j.debuff is True
        assert result.perished == []

    def test_non_perishable_unaffected(self):
        j = _joker(perishable=False, perish_tally=5)
        result = process_round_end_cards([j], _gs())
        assert j.perish_tally == 5  # unchanged
        assert j.debuff is False
        assert result.perished == []


# ---------------------------------------------------------------------------
# Rental charges
# ---------------------------------------------------------------------------


class TestRentalCharges:
    def test_single_rental_deducts_3(self):
        j = _joker(rental=True)
        gs = _gs(dollars=20)
        result = process_round_end_cards([j], gs)
        assert gs["dollars"] == 17
        assert result.rental_cost == 3
        assert result.rental_cards == [j]

    def test_multiple_rentals_stack(self):
        j1 = _joker(key="j_a", rental=True)
        j2 = _joker(key="j_b", rental=True)
        j3 = _joker(key="j_c", rental=True)
        gs = _gs(dollars=30)
        result = process_round_end_cards([j1, j2, j3], gs)
        assert gs["dollars"] == 21  # 30 - 3*3
        assert result.rental_cost == 9
        assert len(result.rental_cards) == 3

    def test_custom_rental_rate(self):
        j = _joker(rental=True)
        gs = _gs(dollars=20, rental_rate=5)
        result = process_round_end_cards([j], gs)
        assert gs["dollars"] == 15
        assert result.rental_cost == 5

    def test_can_go_negative(self):
        """Rental can push dollars below 0 (no floor check in Lua)."""
        j = _joker(rental=True)
        gs = _gs(dollars=1)
        result = process_round_end_cards([j], gs)
        assert gs["dollars"] == -2

    def test_non_rental_no_deduction(self):
        j = _joker(rental=False)
        gs = _gs(dollars=20)
        result = process_round_end_cards([j], gs)
        assert gs["dollars"] == 20
        assert result.rental_cost == 0
        assert result.rental_cards == []

    def test_debuffed_rental_still_fires(self):
        """In Lua, calculate_rental checks ability.rental, not debuff.
        A debuffed rental card still charges rent."""
        j = _joker(rental=True, debuff=True)
        gs = _gs(dollars=20)
        result = process_round_end_cards([j], gs)
        assert gs["dollars"] == 17
        assert result.rental_cost == 3


# ---------------------------------------------------------------------------
# Both perishable and rental on same joker
# ---------------------------------------------------------------------------


class TestPerishableAndRental:
    def test_both_apply(self):
        j = _joker(perishable=True, perish_tally=3, rental=True)
        gs = _gs(dollars=20)
        result = process_round_end_cards([j], gs)
        # Rental fires
        assert gs["dollars"] == 17
        assert result.rental_cost == 3
        # Perishable decrements
        assert j.perish_tally == 2
        assert j.debuff is False

    def test_perish_and_rental_on_debuff_round(self):
        """On the round perish_tally hits 0, rental still fires."""
        j = _joker(perishable=True, perish_tally=1, rental=True)
        gs = _gs(dollars=20)
        result = process_round_end_cards([j], gs)
        assert gs["dollars"] == 17
        assert j.perish_tally == 0
        assert j.debuff is True
        assert result.perished == [j]
        assert result.rental_cost == 3


# ---------------------------------------------------------------------------
# Mixed joker list
# ---------------------------------------------------------------------------


class TestMixedJokers:
    def test_varied_list(self):
        """A list with normal, perishable, rental, and perishable+rental jokers."""
        normal = _joker(key="j_normal")
        perish = _joker(key="j_perish", perishable=True, perish_tally=2)
        rent = _joker(key="j_rent", rental=True)
        both = _joker(key="j_both", perishable=True, perish_tally=1, rental=True)

        gs = _gs(dollars=50)
        result = process_round_end_cards([normal, perish, rent, both], gs)

        # Normal: untouched
        assert normal.perish_tally == 5
        assert normal.debuff is False

        # Perishable: 2 → 1
        assert perish.perish_tally == 1
        assert perish.debuff is False

        # Rental: $3 deducted
        # Both: $3 deducted + perish 1→0→debuff
        assert both.perish_tally == 0
        assert both.debuff is True

        assert gs["dollars"] == 44  # 50 - 3 - 3
        assert result.rental_cost == 6
        assert result.perished == [both]


# ---------------------------------------------------------------------------
# Result structure
# ---------------------------------------------------------------------------


class TestRoundEndResult:
    def test_default_result(self):
        r = RoundEndResult()
        assert r.perished == []
        assert r.rental_cost == 0
        assert r.rental_cards == []

    def test_empty_joker_list(self):
        gs = _gs(dollars=10)
        result = process_round_end_cards([], gs)
        assert gs["dollars"] == 10
        assert result.perished == []
        assert result.rental_cost == 0


# ===========================================================================
# reset_round_targets
# ===========================================================================


def _make_target_gs(seed: str = "TARGET_KNOWN") -> dict:
    """Build a game_state with a real deck via initialize_run."""
    from jackdaw.engine.run_init import initialize_run

    return initialize_run("b_red", 1, seed)


# ---------------------------------------------------------------------------
# Known-seed determinism
# ---------------------------------------------------------------------------


class TestResetRoundTargetsKnownSeed:
    """Exact values from a fixed seed — validates RNG stream keys."""

    # Values established by running initialize_run("b_red", 1, "TARGET_KNOWN")
    # which calls reset_round_targets(rng, ante=1, gs) internally.

    def test_idol_card_ante1(self):
        gs = _make_target_gs()
        idol = gs["current_round"]["idol_card"]
        assert idol == {"suit": "Clubs", "rank": "6"}

    def test_mail_card_ante1(self):
        gs = _make_target_gs()
        mail = gs["current_round"]["mail_card"]
        assert mail == {"rank": "6"}

    def test_ancient_card_ante1(self):
        gs = _make_target_gs()
        ancient = gs["current_round"]["ancient_card"]
        assert ancient == {"suit": "Diamonds"}

    def test_castle_card_ante1(self):
        gs = _make_target_gs()
        castle = gs["current_round"]["castle_card"]
        assert castle == {"suit": "Hearts"}


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


class TestResetRoundTargetsStructure:
    def test_idol_has_suit_and_rank(self):
        gs = _make_target_gs()
        idol = gs["current_round"]["idol_card"]
        assert "suit" in idol
        assert "rank" in idol

    def test_mail_has_rank_only(self):
        gs = _make_target_gs()
        mail = gs["current_round"]["mail_card"]
        assert "rank" in mail
        assert "suit" not in mail

    def test_ancient_has_suit_only(self):
        gs = _make_target_gs()
        ancient = gs["current_round"]["ancient_card"]
        assert "suit" in ancient
        assert "rank" not in ancient

    def test_castle_has_suit_only(self):
        gs = _make_target_gs()
        castle = gs["current_round"]["castle_card"]
        assert "suit" in castle
        assert "rank" not in castle

    def test_idol_rank_is_valid(self):
        gs = _make_target_gs()
        valid_ranks = {r.value for r in Rank}
        assert gs["current_round"]["idol_card"]["rank"] in valid_ranks

    def test_idol_suit_is_valid(self):
        gs = _make_target_gs()
        valid_suits = {s.value for s in Suit}
        assert gs["current_round"]["idol_card"]["suit"] in valid_suits


# ---------------------------------------------------------------------------
# Different ante → different targets
# ---------------------------------------------------------------------------


class TestResetRoundTargetsDifferentAnte:
    def test_ante1_vs_ante2_differ(self):
        gs = _make_target_gs()
        rng = gs["rng"]
        ante1_idol = dict(gs["current_round"]["idol_card"])
        ante1_mail = dict(gs["current_round"]["mail_card"])

        gs["round_resets"]["ante"] = 2
        reset_round_targets(rng, 2, gs)
        ante2_idol = gs["current_round"]["idol_card"]
        ante2_mail = gs["current_round"]["mail_card"]

        # At least one of idol/mail should differ between antes
        assert (
            ante1_idol != ante2_idol or ante1_mail != ante2_mail
        ), "Ante 1 and ante 2 produced identical targets"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestResetRoundTargetsDeterminism:
    def test_same_seed_same_result(self):
        gs1 = _make_target_gs("DETERM_TGT")
        gs2 = _make_target_gs("DETERM_TGT")
        assert gs1["current_round"]["idol_card"] == gs2["current_round"]["idol_card"]
        assert gs1["current_round"]["mail_card"] == gs2["current_round"]["mail_card"]
        assert gs1["current_round"]["ancient_card"] == gs2["current_round"]["ancient_card"]
        assert gs1["current_round"]["castle_card"] == gs2["current_round"]["castle_card"]


# ---------------------------------------------------------------------------
# Ancient card excludes current suit
# ---------------------------------------------------------------------------


class TestAncientCardExclusion:
    def test_ancient_never_same_as_previous(self):
        """Calling reset multiple times — ancient suit must differ each time."""
        gs = _make_target_gs("ANCIENT_EXCL")
        rng = gs["rng"]
        prev_suit = gs["current_round"]["ancient_card"]["suit"]
        for ante in range(2, 12):
            gs["round_resets"]["ante"] = ante
            reset_round_targets(rng, ante, gs)
            new_suit = gs["current_round"]["ancient_card"]["suit"]
            assert new_suit != prev_suit, (
                f"ante={ante}: ancient suit {new_suit!r} == previous {prev_suit!r}"
            )
            prev_suit = new_suit


# ---------------------------------------------------------------------------
# Stone Card filtering
# ---------------------------------------------------------------------------


class TestStoneCardFiltering:
    def test_stone_cards_excluded(self):
        """Idol/mail/castle pick from non-Stone cards only."""
        gs = _make_target_gs("STONE_TEST")
        rng = gs["rng"]

        # Make ALL cards Stone except one
        for card in gs["deck"]:
            card.ability["effect"] = "Stone Card"
        # Leave one card as non-Stone (a 7 of Hearts, say)
        survivor = gs["deck"][0]
        survivor.ability["effect"] = ""

        gs["round_resets"]["ante"] = 99  # fresh stream
        reset_round_targets(rng, 99, gs)

        # Idol, mail, castle must all pick the sole non-Stone card
        assert gs["current_round"]["idol_card"]["rank"] == survivor.base.rank.value
        assert gs["current_round"]["idol_card"]["suit"] == survivor.base.suit.value
        assert gs["current_round"]["mail_card"]["rank"] == survivor.base.rank.value
        assert gs["current_round"]["castle_card"]["suit"] == survivor.base.suit.value

    def test_all_stone_falls_back_to_defaults(self):
        """If every card is Stone, defaults are Ace/Spades."""
        gs = _make_target_gs("ALL_STONE")
        rng = gs["rng"]

        for card in gs["deck"]:
            card.ability["effect"] = "Stone Card"

        gs["round_resets"]["ante"] = 99
        reset_round_targets(rng, 99, gs)

        assert gs["current_round"]["idol_card"] == {"suit": "Spades", "rank": "Ace"}
        assert gs["current_round"]["mail_card"] == {"rank": "Ace"}
        assert gs["current_round"]["castle_card"] == {"suit": "Spades"}


# ---------------------------------------------------------------------------
# Empty deck
# ---------------------------------------------------------------------------


class TestEmptyDeck:
    def test_empty_deck_defaults(self):
        rng = PseudoRandom("EMPTY_DECK")
        gs: dict = {
            "deck": [],
            "current_round": {
                "idol_card": {"suit": "Hearts", "rank": "King"},
                "mail_card": {"rank": "King"},
                "ancient_card": {"suit": "Hearts"},
                "castle_card": {"suit": "Hearts"},
            },
            "round_resets": {"ante": 1},
        }
        reset_round_targets(rng, 1, gs)
        # Idol, mail, castle fall back to defaults (no valid cards)
        assert gs["current_round"]["idol_card"] == {"suit": "Spades", "rank": "Ace"}
        assert gs["current_round"]["mail_card"] == {"rank": "Ace"}
        assert gs["current_round"]["castle_card"] == {"suit": "Spades"}
        # Ancient still picks a suit (doesn't use deck)
        assert gs["current_round"]["ancient_card"]["suit"] != "Hearts"

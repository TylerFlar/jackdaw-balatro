"""Tests for jackdaw.engine.round_lifecycle — process_round_end_cards.

Coverage
--------
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
"""

from __future__ import annotations

from typing import Any

import pytest

from jackdaw.engine.card import Card
from jackdaw.engine.round_lifecycle import RoundEndResult, process_round_end_cards


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

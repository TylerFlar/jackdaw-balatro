"""Tests for scaling jokers that mutate ability state across hands.

Validates multi-hand sequences, accumulation, reset conditions,
and self-destruction.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.jokers import JokerContext, JokerResult, calculate_joker
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.scoring import score_hand


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


_SL = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}
_RL = {
    "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7",
    "8": "8", "9": "9", "10": "T", "Jack": "J", "Queen": "Q",
    "King": "K", "Ace": "A",
}


def _card(suit: str, rank: str) -> Card:
    c = Card()
    c.set_base(f"{_SL[suit]}_{_RL[rank]}", suit, rank)
    c.set_ability("c_base")
    return c


def _joker(key: str, **ability_kw) -> Card:
    c = Card()
    c.center_key = key
    c.ability = {"name": key, "set": "Joker", **ability_kw}
    return c


def _poker_hands_with(*types: str) -> dict[str, list]:
    all_types = [
        "Flush Five", "Flush House", "Five of a Kind", "Straight Flush",
        "Four of a Kind", "Full House", "Flush", "Straight",
        "Three of a Kind", "Two Pair", "Pair", "High Card",
    ]
    return {t: [["p"]] if t in types else [] for t in all_types}


def _small_blind() -> Blind:
    return Blind.create("bl_small", ante=1)


# ============================================================================
# Green Joker: +1 mult per hand, -1 per discard
# ============================================================================

class TestGreenJoker:
    def _make(self):
        return _joker(
            "j_green_joker", mult=0,
            extra={"hand_add": 1, "discard_sub": 1},
        )

    def test_hand_1_adds_1(self):
        j = self._make()
        ctx = JokerContext(
            before=True,
            scoring_hand=[_card("Hearts", "5"), _card("Spades", "5")],
        )
        calculate_joker(j, ctx)
        assert j.ability["mult"] == 1

    def test_hand_2_adds_2(self):
        j = self._make()
        # Hand 1
        ctx = JokerContext(
            before=True,
            scoring_hand=[_card("Hearts", "5"), _card("Spades", "5")],
        )
        calculate_joker(j, ctx)
        # Hand 2
        calculate_joker(j, ctx)
        assert j.ability["mult"] == 2

    def test_discard_subtracts(self):
        j = self._make()
        # Play 2 hands: mult = 2
        ctx_b = JokerContext(
            before=True,
            scoring_hand=[_card("Hearts", "5"), _card("Spades", "5")],
        )
        calculate_joker(j, ctx_b)
        calculate_joker(j, ctx_b)
        assert j.ability["mult"] == 2

        # Discard (fires on last card)
        last = _card("Hearts", "3")
        ctx_d = JokerContext(
            discard=True, other_card=last, full_hand=[last],
        )
        calculate_joker(j, ctx_d)
        assert j.ability["mult"] == 1

    def test_discard_clamps_to_zero(self):
        j = self._make()
        last = _card("Hearts", "3")
        ctx_d = JokerContext(
            discard=True, other_card=last, full_hand=[last],
        )
        calculate_joker(j, ctx_d)
        assert j.ability["mult"] == 0

    def test_joker_main_returns_accumulated(self):
        j = self._make()
        # Play 3 hands
        ctx_b = JokerContext(
            before=True,
            scoring_hand=[_card("Hearts", "5"), _card("Spades", "5")],
        )
        for _ in range(3):
            calculate_joker(j, ctx_b)
        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.mult_mod == 3

    def test_blueprint_does_not_mutate(self):
        j = self._make()
        ctx = JokerContext(
            before=True, blueprint=1,
            scoring_hand=[_card("Hearts", "5"), _card("Spades", "5")],
        )
        calculate_joker(j, ctx)
        assert j.ability["mult"] == 0  # unchanged


# ============================================================================
# Ride the Bus: +1 mult per consecutive no-face hand, reset on face
# ============================================================================

class TestRideTheBus:
    def _make(self):
        return _joker("j_ride_the_bus", mult=0, extra=1)

    def test_three_no_face_hands(self):
        j = self._make()
        no_face_hand = [_card("Hearts", "5"), _card("Spades", "5")]
        ctx = JokerContext(before=True, scoring_hand=no_face_hand)
        for _ in range(3):
            calculate_joker(j, ctx)
        assert j.ability["mult"] == 3

    def test_face_card_resets(self):
        j = self._make()
        no_face = [_card("Hearts", "5"), _card("Spades", "5")]
        ctx_nf = JokerContext(before=True, scoring_hand=no_face)
        for _ in range(3):
            calculate_joker(j, ctx_nf)
        assert j.ability["mult"] == 3

        face = [_card("Hearts", "King"), _card("Spades", "5")]
        ctx_f = JokerContext(before=True, scoring_hand=face)
        calculate_joker(j, ctx_f)
        assert j.ability["mult"] == 0

    def test_joker_main_returns_accumulated(self):
        j = self._make()
        ctx = JokerContext(
            before=True,
            scoring_hand=[_card("Hearts", "5"), _card("Spades", "5")],
        )
        for _ in range(3):
            calculate_joker(j, ctx)
        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.mult_mod == 3

    def test_zero_mult_returns_none(self):
        j = self._make()
        assert calculate_joker(j, JokerContext(joker_main=True)) is None


# ============================================================================
# Spare Trousers: +2 mult when Two Pair or Full House
# ============================================================================

class TestSpareTrousers:
    def _make(self):
        return _joker("j_trousers", mult=0, extra=2)

    def test_two_pair_triggers(self):
        j = self._make()
        ph = _poker_hands_with("Two Pair", "Pair")
        ctx = JokerContext(before=True, poker_hands=ph)
        calculate_joker(j, ctx)
        assert j.ability["mult"] == 2

    def test_full_house_triggers(self):
        j = self._make()
        ph = _poker_hands_with("Full House", "Three of a Kind", "Pair")
        ctx = JokerContext(before=True, poker_hands=ph)
        calculate_joker(j, ctx)
        assert j.ability["mult"] == 2

    def test_flush_no_effect(self):
        j = self._make()
        ph = _poker_hands_with("Flush")
        ctx = JokerContext(before=True, poker_hands=ph)
        calculate_joker(j, ctx)
        assert j.ability["mult"] == 0

    def test_accumulates(self):
        j = self._make()
        ph = _poker_hands_with("Two Pair", "Pair")
        ctx = JokerContext(before=True, poker_hands=ph)
        calculate_joker(j, ctx)
        calculate_joker(j, ctx)
        assert j.ability["mult"] == 4


# ============================================================================
# Square Joker: +4 chips when exactly 4 cards played
# ============================================================================

class TestSquareJoker:
    def _make(self):
        return _joker("j_square", extra={"chips": 0, "chip_mod": 4})

    def test_four_cards_adds(self):
        j = self._make()
        hand = [_card("Hearts", "5")] * 4
        ctx = JokerContext(before=True, full_hand=hand)
        calculate_joker(j, ctx)
        assert j.ability["extra"]["chips"] == 4

    def test_five_cards_no_effect(self):
        j = self._make()
        hand = [_card("Hearts", "5")] * 5
        ctx = JokerContext(before=True, full_hand=hand)
        calculate_joker(j, ctx)
        assert j.ability["extra"]["chips"] == 0

    def test_joker_main_returns_accumulated(self):
        j = self._make()
        hand = [_card("Hearts", "5")] * 4
        ctx = JokerContext(before=True, full_hand=hand)
        for _ in range(3):
            calculate_joker(j, ctx)
        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.chip_mod == 12


# ============================================================================
# Ice Cream: starts +100 chips, -5 per hand, self-destructs at 0
# ============================================================================

class TestIceCream:
    def _make(self):
        return _joker("j_ice_cream", extra={"chips": 100, "chip_mod": 5})

    def test_after_one_hand(self):
        j = self._make()
        ctx = JokerContext(after=True)
        calculate_joker(j, ctx)
        assert j.ability["extra"]["chips"] == 95

    def test_after_twenty_hands_reaches_zero(self):
        j = self._make()
        ctx = JokerContext(after=True)
        for _ in range(19):
            calculate_joker(j, ctx)
        assert j.ability["extra"]["chips"] == 5

    def test_hand_twenty_one_self_destructs(self):
        j = self._make()
        ctx = JokerContext(after=True)
        for _ in range(19):
            calculate_joker(j, ctx)
        # chips=5, chip_mod=5 → 5-5 <= 0 → remove
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.remove is True

    def test_joker_main_returns_current_chips(self):
        j = self._make()
        # Score 3 hands
        for _ in range(3):
            calculate_joker(j, JokerContext(after=True))
        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.chip_mod == 85  # 100 - 3*5

    def test_in_scoring_pipeline(self):
        """Full pipeline: Ice Cream contributes chips then decays."""
        j = self._make()
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        r1 = score_hand(
            played, [], [j], HandLevels(), _small_blind(),
            PseudoRandom("TEST"),
        )
        # Pair: 32 chips, 2 mult. Phase 9: +100 chip_mod → 132. Score: 132×2=264.
        assert r1.chips == 132.0
        assert r1.total == 264
        # After Phase 10: chips decremented
        assert j.ability["extra"]["chips"] == 95

        # Second hand
        reset_sort_id_counter()
        r2 = score_hand(
            played, [], [j], HandLevels(), _small_blind(),
            PseudoRandom("TEST"),
        )
        assert r2.chips == 127.0  # 32 + 95
        assert j.ability["extra"]["chips"] == 90


# ============================================================================
# Popcorn: starts +20 mult, -4 per round, self-destructs
# ============================================================================

class TestPopcorn:
    def _make(self):
        return _joker("j_popcorn", mult=20, extra=4)

    def test_end_of_round_decrements(self):
        j = self._make()
        ctx = JokerContext(end_of_round=True)
        calculate_joker(j, ctx)
        assert j.ability["mult"] == 16

    def test_self_destructs_at_zero(self):
        j = self._make()
        ctx = JokerContext(end_of_round=True)
        for _ in range(4):
            calculate_joker(j, ctx)
        assert j.ability["mult"] == 4
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.remove is True

    def test_joker_main_returns_current(self):
        j = self._make()
        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.mult_mod == 20


# ============================================================================
# Flash Card: +2 mult per reroll
# ============================================================================

class TestFlashCard:
    def test_accumulates(self):
        j = _joker("j_flash", mult=0, extra=2)
        ctx = JokerContext(reroll_shop=True)
        for _ in range(5):
            calculate_joker(j, ctx)
        assert j.ability["mult"] == 10
        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.mult_mod == 10


# ============================================================================
# Red Card: +3 mult per booster skip
# ============================================================================

class TestRedCard:
    def test_accumulates(self):
        j = _joker("j_red_card", mult=0, extra=3)
        ctx = JokerContext(skipping_booster=True)
        for _ in range(4):
            calculate_joker(j, ctx)
        assert j.ability["mult"] == 12
        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.mult_mod == 12


# ============================================================================
# Wee Joker: +8 chips per 2-rank card scored
# ============================================================================

class TestWeeJoker:
    def _make(self):
        return _joker("j_wee", extra={"chips": 0, "chip_mod": 8})

    def test_score_one_two(self):
        j = self._make()
        two = _card("Hearts", "2")
        ctx = JokerContext(individual=True, cardarea="play", other_card=two)
        calculate_joker(j, ctx)
        assert j.ability["extra"]["chips"] == 8

    def test_three_twos_across_two_hands(self):
        """Score 3 twos across 2 hands → +24 chips accumulated."""
        j = self._make()
        # Hand 1: 2 twos
        for _ in range(2):
            ctx = JokerContext(
                individual=True, cardarea="play",
                other_card=_card("Hearts", "2"),
            )
            calculate_joker(j, ctx)
        assert j.ability["extra"]["chips"] == 16

        # Hand 2: 1 two
        ctx = JokerContext(
            individual=True, cardarea="play",
            other_card=_card("Spades", "2"),
        )
        calculate_joker(j, ctx)
        assert j.ability["extra"]["chips"] == 24

    def test_non_two_no_effect(self):
        j = self._make()
        ctx = JokerContext(
            individual=True, cardarea="play",
            other_card=_card("Hearts", "5"),
        )
        calculate_joker(j, ctx)
        assert j.ability["extra"]["chips"] == 0

    def test_joker_main_returns_accumulated(self):
        j = self._make()
        for _ in range(3):
            ctx = JokerContext(
                individual=True, cardarea="play",
                other_card=_card("Hearts", "2"),
            )
            calculate_joker(j, ctx)
        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.chip_mod == 24


# ============================================================================
# Lucky Cat: +0.25 xMult per Lucky Card trigger
# ============================================================================

class TestLuckyCat:
    def _make(self):
        return _joker("j_lucky_cat", x_mult=1, extra=0.25)

    def _lucky_card(self) -> Card:
        c = _card("Hearts", "5")
        c.set_ability("m_lucky")
        c.lucky_trigger = True
        return c

    def test_lucky_trigger_accumulates(self):
        j = self._make()
        lc = self._lucky_card()
        ctx = JokerContext(individual=True, cardarea="play", other_card=lc)
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(1.25)

    def test_four_triggers(self):
        j = self._make()
        for _ in range(4):
            lc = self._lucky_card()
            ctx = JokerContext(individual=True, cardarea="play", other_card=lc)
            calculate_joker(j, ctx)
        assert j.ability["x_mult"] == pytest.approx(2.0)

    def test_joker_main_returns_when_above_1(self):
        j = self._make()
        # x_mult = 1, no effect yet
        assert calculate_joker(j, JokerContext(joker_main=True)) is None

        # Trigger once → x_mult = 1.25
        lc = self._lucky_card()
        ctx = JokerContext(individual=True, cardarea="play", other_card=lc)
        calculate_joker(j, ctx)

        result = calculate_joker(j, JokerContext(joker_main=True))
        assert result is not None
        assert result.Xmult_mod == pytest.approx(1.25)

    def test_non_lucky_no_effect(self):
        j = self._make()
        normal = _card("Hearts", "5")
        ctx = JokerContext(individual=True, cardarea="play", other_card=normal)
        calculate_joker(j, ctx)
        assert j.ability["x_mult"] == 1  # unchanged

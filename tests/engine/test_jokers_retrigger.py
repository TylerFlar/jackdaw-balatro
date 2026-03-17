"""Tests for retrigger jokers that cause cards to be re-evaluated.

Tests go through the full score_hand pipeline to verify retriggers
actually compound scoring effects.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.jokers import JokerContext, calculate_joker
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.scoring import score_hand


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


_SL = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}
_RL = {
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "10": "T",
    "Jack": "J",
    "Queen": "Q",
    "King": "K",
    "Ace": "A",
}


def _card(suit: str, rank: str, enhancement: str = "c_base") -> Card:
    c = Card()
    c.set_base(f"{_SL[suit]}_{_RL[rank]}", suit, rank)
    c.set_ability(enhancement)
    return c


def _joker(key: str, **ability_kw) -> Card:
    c = Card()
    c.center_key = key
    c.ability = {"name": key, "set": "Joker", **ability_kw}
    return c


def _small_blind() -> Blind:
    return Blind.create("bl_small", ante=1)


# ============================================================================
# Sock and Buskin: retrigger face cards
# ============================================================================


class TestSockAndBuskin:
    """j_sock_and_buskin: +1 retrigger for face cards scored."""

    def test_unit_face_card_triggers(self):
        j = _joker("j_sock_and_buskin", extra=1)
        king = _card("Hearts", "King")
        ctx = JokerContext(
            repetition=True,
            cardarea="play",
            other_card=king,
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.repetitions == 1

    def test_unit_non_face_no_effect(self):
        j = _joker("j_sock_and_buskin", extra=1)
        five = _card("Hearts", "5")
        ctx = JokerContext(
            repetition=True,
            cardarea="play",
            other_card=five,
        )
        assert calculate_joker(j, ctx) is None

    def test_pipeline_three_kings(self):
        """Three Kings scored: each face card evaluated twice.
        Three of a Kind L1: 30 chips, 3 mult.
        Per card: 10 chips each × 3 Kings × 2 reps = 60 chips.
        Total chips: 30 + 60 = 90, mult = 3. Score: 90 × 3 = 270.
        (Without retrigger: 30 + 30 = 60, score 180.)"""
        played = [
            _card("Hearts", "King"),
            _card("Spades", "King"),
            _card("Clubs", "King"),
            _card("Diamonds", "5"),
            _card("Hearts", "2"),
        ]
        j = _joker("j_sock_and_buskin", extra=1)
        result = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # 3 Kings score. Each King: 10 chips × 2 reps = 20 chips.
        # Non-scoring 5 and 2 don't retrigger.
        assert result.chips == 90.0  # 30 + 3×20
        assert result.total == 270

    def test_pipeline_pareidolia_all_retriggered(self):
        """With Pareidolia, ALL cards are face → all retriggered.
        Pair of 5s: 10 base, 2 mult. Per card: 5 chips × 2 reps = 10 each.
        Total chips: 10 + 10 + 10 = 30. Score: 30 × 2 = 60."""
        played = [_card("Hearts", "5"), _card("Spades", "5")]
        sock = _joker("j_sock_and_buskin", extra=1)
        pareidolia = _joker("j_pareidolia_stub")
        # Pareidolia needs to be a real joker that score_hand detects
        pareidolia.ability = {"name": "Pareidolia", "set": "Joker"}
        result = score_hand(
            played,
            [],
            [sock, pareidolia],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # With Pareidolia, both 5s are "face cards" → retriggered
        assert result.chips == 30.0  # 10 + 5*2 + 5*2
        assert result.total == 60


# ============================================================================
# Hanging Chad: +2 retriggers for FIRST scored card
# ============================================================================


class TestHangingChad:
    """j_hanging_chad: +2 retriggers for first scored card only."""

    def test_unit_first_card(self):
        j = _joker("j_hanging_chad", extra=2)
        first = _card("Hearts", "Ace")
        scoring = [first, _card("Spades", "Ace")]
        ctx = JokerContext(
            repetition=True,
            cardarea="play",
            other_card=first,
            scoring_hand=scoring,
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.repetitions == 2

    def test_unit_second_card_no_effect(self):
        j = _joker("j_hanging_chad", extra=2)
        first = _card("Hearts", "Ace")
        second = _card("Spades", "Ace")
        scoring = [first, second]
        ctx = JokerContext(
            repetition=True,
            cardarea="play",
            other_card=second,
            scoring_hand=scoring,
        )
        assert calculate_joker(j, ctx) is None

    def test_pipeline_pair_of_aces(self):
        """Pair of Aces: first Ace evaluated 3 times (1 base + 2 reps).
        Base: 10 chips, 2 mult.
        First Ace: 11 chips × 3 reps = 33 chips.
        Second Ace: 11 chips × 1 rep = 11 chips.
        Total: 10 + 33 + 11 = 54. Score: 54 × 2 = 108."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        j = _joker("j_hanging_chad", extra=2)
        result = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.chips == 54.0
        assert result.total == 108


# ============================================================================
# Dusk: retrigger all scored cards on last hand
# ============================================================================


class TestDusk:
    """j_dusk: +1 retrigger for all scored cards on last hand."""

    def test_unit_last_hand(self):
        j = _joker("j_dusk", extra=1)
        ctx = JokerContext(
            repetition=True,
            cardarea="play",
            other_card=_card("Hearts", "5"),
            hands_left=0,
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.repetitions == 1

    def test_unit_not_last_hand(self):
        j = _joker("j_dusk", extra=1)
        ctx = JokerContext(
            repetition=True,
            cardarea="play",
            other_card=_card("Hearts", "5"),
            hands_left=2,
        )
        assert calculate_joker(j, ctx) is None

    def test_pipeline_last_hand(self):
        """Last hand: all cards doubled.
        Pair of Aces: 10 base, 2 mult. Each Ace: 11 × 2 = 22 chips.
        Total: 10 + 22 + 22 = 54. Score: 54 × 2 = 108."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        j = _joker("j_dusk", extra=1)
        result = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
            game_state={"hands_left": 0},
        )
        assert result.chips == 54.0
        assert result.total == 108


# ============================================================================
# Red Seal + Sock and Buskin: additive retriggers
# ============================================================================


class TestAdditiveRetriggers:
    """Red Seal (+1) + Sock and Buskin (+1) on face card = 3 total evals."""

    def test_red_seal_plus_sock(self):
        """King with Red Seal + Sock and Buskin:
        1 base + 1 Red Seal retrigger + 1 Sock retrigger = 3 evaluations.
        Pair of Kings: 10 base, 2 mult.
        Red Seal King: 10 chips × 3 reps = 30.
        Normal King: 10 chips × 1 = 10.
        Total: 10 + 30 + 10 = 50. Score: 50 × 2 = 100."""
        k1 = _card("Hearts", "King")
        k1.set_seal("Red")
        k2 = _card("Spades", "King")
        played = [k1, k2]
        j = _joker("j_sock_and_buskin", extra=1)
        result = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # k1: Red Seal (+1 rep) + Sock (+1 rep, is face) = 3 total
        # k2: Sock (+1 rep, is face) = 2 total
        assert result.chips == 10.0 + 30.0 + 20.0  # 60
        assert result.total == 120


# ============================================================================
# Seltzer: retrigger all, decrements, self-destructs
# ============================================================================


class TestSeltzer:
    """j_selzer: +1 retrigger for all cards. Decrements per hand."""

    def test_unit_always_retriggers(self):
        j = _joker("j_selzer", extra=10)
        ctx = JokerContext(
            repetition=True,
            cardarea="play",
            other_card=_card("Hearts", "5"),
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.repetitions == 1

    def test_after_decrements(self):
        j = _joker("j_selzer", extra=10)
        ctx = JokerContext(after=True)
        calculate_joker(j, ctx)
        assert j.ability["extra"] == 9

    def test_self_destructs_at_zero(self):
        j = _joker("j_selzer", extra=1)
        ctx = JokerContext(after=True)
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.remove is True

    def test_ten_hands_then_destruct(self):
        j = _joker("j_selzer", extra=10)
        for i in range(9):
            calculate_joker(j, JokerContext(after=True))
        assert j.ability["extra"] == 1
        result = calculate_joker(j, JokerContext(after=True))
        assert result.remove is True

    def test_pipeline_all_cards_retriggered(self):
        """Pair of Aces with Seltzer: all cards retriggered.
        Base: 10, 2 mult. Each Ace: 11 × 2 = 22. Total: 54 × 2 = 108.
        After scoring: extra decremented 10 → 9."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        j = _joker("j_selzer", extra=10)
        result = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.chips == 54.0
        assert result.total == 108
        assert j.ability["extra"] == 9


# ============================================================================
# Mime: retrigger held cards
# ============================================================================


class TestMime:
    """j_mime: +1 retrigger for held cards."""

    def test_unit_held_card(self):
        j = _joker("j_mime", extra=1)
        ctx = JokerContext(
            repetition=True,
            cardarea="hand",
            other_card=_card("Hearts", "5"),
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.repetitions == 1

    def test_unit_play_area_no_effect(self):
        j = _joker("j_mime", extra=1)
        ctx = JokerContext(
            repetition=True,
            cardarea="play",
            other_card=_card("Hearts", "5"),
        )
        assert calculate_joker(j, ctx) is None

    def test_pipeline_steel_card_doubled(self):
        """Steel Card held + Mime: x1.5 fires twice → mult × 1.5 × 1.5 = ×2.25.
        Pair of Aces: 32 chips, 2 mult.
        Steel held: 2 reps × x1.5 → mult = 2 × 1.5 × 1.5 = 4.5.
        Score: floor(32 × 4.5) = 144."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        held = [_card("Clubs", "3", enhancement="m_steel")]
        j = _joker("j_mime", extra=1)
        result = score_hand(
            played,
            held,
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.mult == pytest.approx(4.5)
        assert result.total == 144

    def test_pipeline_two_steel_with_mime(self):
        """Two Steel Cards + Mime: each fires twice.
        Pair: 32 chips, 2 mult.
        Steel 1: 2 reps × x1.5 → mult × 1.5² = 4.5.
        Steel 2: 2 reps × x1.5 → mult × 1.5² = 10.125.
        Score: floor(32 × 10.125) = 324."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        held = [
            _card("Clubs", "3", enhancement="m_steel"),
            _card("Diamonds", "7", enhancement="m_steel"),
        ]
        j = _joker("j_mime", extra=1)
        result = score_hand(
            played,
            held,
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # 2 × 1.5^4 = 2 × 5.0625 = 10.125
        assert result.mult == pytest.approx(10.125)
        assert result.total == 324


# ============================================================================
# Hack (verify already implemented correctly in repetition context)
# ============================================================================


class TestHackRetrigger:
    """j_hack: verify retrigger works through pipeline."""

    def test_pipeline_pair_of_threes(self):
        """Pair of 3s with Hack: each 3 retriggered.
        Base: 10 chips, 2 mult. Each 3: 3 chips × 2 reps = 6 chips.
        Total: 10 + 6 + 6 = 22. Score: 22 × 2 = 44."""
        played = [_card("Hearts", "3"), _card("Spades", "3")]
        j = _joker("j_hack", extra=1)
        result = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.chips == 22.0
        assert result.total == 44

    def test_pipeline_pair_of_sixes_no_retrigger(self):
        """Pair of 6s: Hack doesn't trigger (6 not in 2-5)."""
        played = [_card("Hearts", "6"), _card("Spades", "6")]
        j = _joker("j_hack", extra=1)
        result = score_hand(
            played,
            [],
            [j],
            HandLevels(),
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.chips == 22.0  # 10 + 6 + 6 (no retrigger)
        assert result.total == 44

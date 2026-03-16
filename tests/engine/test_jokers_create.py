"""Tests for card-creation jokers.

Validates trigger conditions and side-effect descriptors. Actual card
creation is deferred to M10 (pool generation).
"""

from __future__ import annotations

import pytest

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.jokers import JokerContext, calculate_joker
from jackdaw.engine.rng import PseudoRandom


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


_SL = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}
_RL = {
    "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7",
    "8": "8", "9": "9", "10": "T", "Jack": "J", "Queen": "Q",
    "King": "K", "Ace": "A",
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


def _poker_hands_with(*types: str) -> dict[str, list]:
    all_types = [
        "Flush Five", "Flush House", "Five of a Kind", "Straight Flush",
        "Four of a Kind", "Full House", "Flush", "Straight",
        "Three of a Kind", "Two Pair", "Pair", "High Card",
    ]
    return {t: [["p"]] if t in types else [] for t in all_types}


# ============================================================================
# Certificate: create playing card with seal on first_hand_drawn
# ============================================================================

class TestCertificate:
    def test_first_hand_drawn_creates(self):
        j = _joker("j_certificate")
        ctx = JokerContext(first_hand_drawn=True)
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.extra["create"]["type"] == "playing_card"
        assert result.extra["create"]["seal"] is True

    def test_other_context_no_effect(self):
        j = _joker("j_certificate")
        assert calculate_joker(j, JokerContext(joker_main=True)) is None


# ============================================================================
# Marble Joker: add Stone Card on setting_blind
# ============================================================================

class TestMarble:
    def test_setting_blind_creates(self):
        j = _joker("j_marble", extra=1)
        blind = Blind.create("bl_small", ante=1)
        ctx = JokerContext(setting_blind=True, blind=blind)
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.extra["create"]["type"] == "playing_card"
        assert result.extra["create"]["enhancement"] == "m_stone"

    def test_other_context_no_effect(self):
        j = _joker("j_marble", extra=1)
        assert calculate_joker(j, JokerContext(joker_main=True)) is None


# ============================================================================
# DNA: copy first card on first hand (1 card played)
# ============================================================================

class TestDna:
    def test_first_hand_single_card(self):
        j = _joker("j_dna")
        played = [_card("Hearts", "Ace")]
        ctx = JokerContext(before=True, full_hand=played, hands_played=0)
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.extra["create"]["type"] == "playing_card_copy"
        assert result.extra["create"]["source_card"] is played[0]

    def test_first_hand_multiple_cards_no_effect(self):
        j = _joker("j_dna")
        played = [_card("Hearts", "Ace"), _card("Spades", "King")]
        ctx = JokerContext(before=True, full_hand=played, hands_played=0)
        assert calculate_joker(j, ctx) is None

    def test_second_hand_no_effect(self):
        j = _joker("j_dna")
        played = [_card("Hearts", "Ace")]
        ctx = JokerContext(before=True, full_hand=played, hands_played=1)
        assert calculate_joker(j, ctx) is None

    def test_blueprint_does_not_copy(self):
        j = _joker("j_dna")
        played = [_card("Hearts", "Ace")]
        ctx = JokerContext(
            before=True, full_hand=played, hands_played=0, blueprint=1,
        )
        assert calculate_joker(j, ctx) is None


# ============================================================================
# Riff-raff: create Common jokers on setting_blind
# ============================================================================

class TestRiffRaff:
    def test_creates_two_jokers(self):
        j = _joker("j_riff_raff", extra=2)
        ctx = JokerContext(
            setting_blind=True, joker_count=3, joker_slots=5,
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.extra["create"]["type"] == "Joker"
        assert result.extra["create"]["rarity"] == "Common"
        assert result.extra["create"]["count"] == 2

    def test_one_slot_creates_one(self):
        j = _joker("j_riff_raff", extra=2)
        ctx = JokerContext(
            setting_blind=True, joker_count=4, joker_slots=5,
        )
        result = calculate_joker(j, ctx)
        assert result.extra["create"]["count"] == 1

    def test_no_slots_no_effect(self):
        j = _joker("j_riff_raff", extra=2)
        ctx = JokerContext(
            setting_blind=True, joker_count=5, joker_slots=5,
        )
        assert calculate_joker(j, ctx) is None


# ============================================================================
# Cartomancer: create Tarot on setting_blind
# ============================================================================

class TestCartomancer:
    def test_setting_blind_creates_tarot(self):
        j = _joker("j_cartomancer")
        ctx = JokerContext(setting_blind=True)
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.extra["create"]["type"] == "Tarot"


# ============================================================================
# 8 Ball: rank 8 scored → probability → Tarot
# ============================================================================

class TestEightBall:
    def test_rank_8_high_probability(self):
        j = _joker("j_8_ball", extra=4)
        eight = _card("Hearts", "8")
        ctx = JokerContext(
            individual=True, cardarea="play", other_card=eight,
            rng=PseudoRandom("8B"), probabilities_normal=1000.0,
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.extra["create"]["type"] == "Tarot"

    def test_rank_8_no_rng_no_effect(self):
        j = _joker("j_8_ball", extra=4)
        eight = _card("Hearts", "8")
        ctx = JokerContext(
            individual=True, cardarea="play", other_card=eight,
        )
        assert calculate_joker(j, ctx) is None

    def test_non_8_no_effect(self):
        j = _joker("j_8_ball", extra=4)
        five = _card("Hearts", "5")
        ctx = JokerContext(
            individual=True, cardarea="play", other_card=five,
            rng=PseudoRandom("8B"), probabilities_normal=1000.0,
        )
        assert calculate_joker(j, ctx) is None


# ============================================================================
# Vagabond: money ≤ $4 → Tarot
# ============================================================================

class TestVagabond:
    def test_low_money_creates_tarot(self):
        j = _joker("j_vagabond", extra=4)
        ctx = JokerContext(joker_main=True, money=3)
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.extra["create"]["type"] == "Tarot"

    def test_exact_threshold(self):
        j = _joker("j_vagabond", extra=4)
        ctx = JokerContext(joker_main=True, money=4)
        result = calculate_joker(j, ctx)
        assert result is not None

    def test_over_threshold_no_effect(self):
        j = _joker("j_vagabond", extra=4)
        ctx = JokerContext(joker_main=True, money=5)
        assert calculate_joker(j, ctx) is None


# ============================================================================
# Superposition: Ace + Straight → Tarot
# ============================================================================

class TestSuperposition:
    def test_ace_and_straight(self):
        j = _joker("j_superposition")
        ace = _card("Hearts", "Ace")
        scoring = [ace, _card("Hearts", "King")]
        ph = _poker_hands_with("Straight", "High Card")
        ctx = JokerContext(
            joker_main=True, scoring_hand=scoring, poker_hands=ph,
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.extra["create"]["type"] == "Tarot"

    def test_no_ace_no_effect(self):
        j = _joker("j_superposition")
        scoring = [_card("Hearts", "King"), _card("Hearts", "Queen")]
        ph = _poker_hands_with("Straight")
        ctx = JokerContext(
            joker_main=True, scoring_hand=scoring, poker_hands=ph,
        )
        assert calculate_joker(j, ctx) is None

    def test_ace_no_straight_no_effect(self):
        j = _joker("j_superposition")
        ace = _card("Hearts", "Ace")
        scoring = [ace, _card("Hearts", "King")]
        ph = _poker_hands_with("Pair")
        ctx = JokerContext(
            joker_main=True, scoring_hand=scoring, poker_hands=ph,
        )
        assert calculate_joker(j, ctx) is None


# ============================================================================
# Seance: hand matches target → Spectral
# ============================================================================

class TestSeance:
    def test_matching_hand(self):
        j = _joker("j_seance", extra={"poker_hand": "Straight Flush"})
        ph = _poker_hands_with("Straight Flush", "Straight", "Flush")
        ctx = JokerContext(joker_main=True, poker_hands=ph)
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.extra["create"]["type"] == "Spectral"

    def test_non_matching_hand(self):
        j = _joker("j_seance", extra={"poker_hand": "Straight Flush"})
        ph = _poker_hands_with("Flush")
        ctx = JokerContext(joker_main=True, poker_hands=ph)
        assert calculate_joker(j, ctx) is None


# ============================================================================
# Sixth Sense: rank 6, first hand, 1 card → Spectral + destroy
# ============================================================================

class TestSixthSense:
    def test_rank_6_first_hand_single_card(self):
        j = _joker("j_sixth_sense")
        six = _card("Hearts", "6")
        ctx = JokerContext(
            destroying_card=six, full_hand=[six], hands_played=0,
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.remove is True
        assert result.extra["create"]["type"] == "Spectral"

    def test_not_rank_6_no_effect(self):
        j = _joker("j_sixth_sense")
        five = _card("Hearts", "5")
        ctx = JokerContext(
            destroying_card=five, full_hand=[five], hands_played=0,
        )
        assert calculate_joker(j, ctx) is None

    def test_multiple_cards_no_effect(self):
        j = _joker("j_sixth_sense")
        six = _card("Hearts", "6")
        other = _card("Spades", "3")
        ctx = JokerContext(
            destroying_card=six, full_hand=[six, other], hands_played=0,
        )
        assert calculate_joker(j, ctx) is None

    def test_second_hand_no_effect(self):
        j = _joker("j_sixth_sense")
        six = _card("Hearts", "6")
        ctx = JokerContext(
            destroying_card=six, full_hand=[six], hands_played=1,
        )
        assert calculate_joker(j, ctx) is None


# ============================================================================
# Hallucination: open_booster → probability → Tarot
# ============================================================================

class TestHallucination:
    def test_high_probability_creates(self):
        j = _joker("j_hallucination", extra=2)
        ctx = JokerContext(
            open_booster=True,
            rng=PseudoRandom("HAL"),
            probabilities_normal=1000.0,
        )
        result = calculate_joker(j, ctx)
        assert result is not None
        assert result.extra["create"]["type"] == "Tarot"

    def test_no_rng_no_effect(self):
        j = _joker("j_hallucination", extra=2)
        ctx = JokerContext(open_booster=True)
        assert calculate_joker(j, ctx) is None

    def test_other_context_no_effect(self):
        j = _joker("j_hallucination", extra=2)
        ctx = JokerContext(
            joker_main=True,
            rng=PseudoRandom("HAL"),
            probabilities_normal=1000.0,
        )
        assert calculate_joker(j, ctx) is None

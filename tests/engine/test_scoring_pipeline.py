"""Tests for the base scoring pipeline (without joker effects).

Validates Phases 1-4, 6-8, 12 of the scoring pipeline from
state_events.lua:571-1065. Joker effects (Phases 5, 9, 11, 13)
are NOT tested here.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.scoring import ScoreResult, score_hand_base


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


def _card(suit: str, rank: str, enhancement: str = "c_base") -> Card:
    sl = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}
    rl = {
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
    c = Card()
    c.set_base(f"{sl[suit]}_{rl[rank]}", suit, rank)
    c.set_ability(enhancement)
    return c


def _small_blind() -> Blind:
    return Blind.create("bl_small", ante=1)


# ============================================================================
# Basic scoring
# ============================================================================


class TestPairOfAces:
    """Plain pair of Aces, level 1.

    Base: Pair L1 = 10 chips, 2 mult
    Per card: Ace = 11 chips each (2 Aces score)
    Total chips: 10 + 11 + 11 = 32
    Total mult: 2
    Score: floor(32 * 2) = 64
    """

    def test_score(self):
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        levels = HandLevels()
        result = score_hand_base(
            played,
            [],
            levels,
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.hand_type == "Pair"
        assert result.chips == 32.0  # 10 + 11 + 11
        assert result.mult == 2.0
        assert result.total == 64
        assert result.debuffed is False

    def test_scoring_cards(self):
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        levels = HandLevels()
        result = score_hand_base(
            played,
            [],
            levels,
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert len(result.scoring_cards) == 2

    def test_records_play(self):
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        levels = HandLevels()
        score_hand_base(played, [], levels, _small_blind(), PseudoRandom("TEST"))
        assert levels["Pair"].played == 1


class TestFlushLevel3:
    """Flush of 5 Hearts, level 3.

    Base: Flush L3 = 35 + 15*2 = 65 chips, 4 + 2*2 = 8 mult
    Per card: 2+5+8+10+11 = 36 chips from nominals
    Total chips: 65 + 36 = 101
    Total mult: 8
    Score: floor(101 * 8) = 808
    """

    def test_score(self):
        played = [
            _card("Hearts", "2"),
            _card("Hearts", "5"),
            _card("Hearts", "8"),
            _card("Hearts", "Jack"),
            _card("Hearts", "Ace"),
        ]
        levels = HandLevels()
        levels.level_up("Flush", amount=2)  # level 3
        result = score_hand_base(
            played,
            [],
            levels,
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.hand_type == "Flush"
        # Base: 65 chips, 8 mult
        # Per card: 2+5+8+10+11 = 36
        assert result.chips == 101.0
        assert result.mult == 8.0
        assert result.total == 808


# ============================================================================
# Enhancement effects
# ============================================================================


class TestGlassCard:
    """Glass Card in scoring hand: x_mult = 2.0."""

    def test_glass_non_scoring_card_no_effect(self):
        """Glass Card not in scoring pair: x_mult does NOT apply."""
        played = [
            _card("Hearts", "5"),
            _card("Spades", "5"),
            _card("Clubs", "8"),
            _card("Diamonds", "Jack"),
            _card("Hearts", "Ace", enhancement="m_glass"),
        ]
        levels = HandLevels()
        result = score_hand_base(
            played,
            [],
            levels,
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # Pair of 5s scores. Glass Ace is NOT a scoring card → no x_mult.
        assert result.mult == 2.0  # unchanged

    def test_glass_in_pair(self):
        """Glass Card that's part of the scoring pair."""
        glass_5 = _card("Hearts", "5", enhancement="m_glass")
        played = [glass_5, _card("Spades", "5")]
        levels = HandLevels()
        result = score_hand_base(
            played,
            [],
            levels,
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # Pair L1: 10 chips, 2 mult
        # Card 1 (Glass 5): +5 chips, x2 mult
        # Card 2 (normal 5): +5 chips
        # chips: 10 + 5 + 5 = 20
        # mult after card 1: 2 * 2 = 4 (x_mult)
        # After card 2: 4 (no x_mult)
        assert result.chips == 20.0
        assert result.mult == 4.0
        assert result.total == 80


class TestMultCard:
    """Mult Card in scoring hand: +4 mult."""

    def test_mult_adds(self):
        mult_5 = _card("Hearts", "5", enhancement="m_mult")
        played = [mult_5, _card("Spades", "5")]
        levels = HandLevels()
        result = score_hand_base(
            played,
            [],
            levels,
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # Pair L1: 10 chips, 2 mult
        # Card 1 (Mult 5): +5 chips, +4 mult
        # Card 2 (normal 5): +5 chips
        # chips: 10 + 5 + 5 = 20
        # mult: 2 + 4 = 6
        assert result.chips == 20.0
        assert result.mult == 6.0
        assert result.total == 120


class TestBonusCard:
    """Bonus Card: +30 chips."""

    def test_bonus_chips(self):
        bonus_5 = _card("Hearts", "5", enhancement="m_bonus")
        played = [bonus_5, _card("Spades", "5")]
        levels = HandLevels()
        result = score_hand_base(
            played,
            [],
            levels,
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # chips: 10 + (5+30) + 5 = 50
        assert result.chips == 50.0
        assert result.total == 100  # 50 * 2


# ============================================================================
# Held card effects (Phase 8)
# ============================================================================


class TestSteelCard:
    """Steel Card in held cards: h_x_mult = 1.5."""

    def test_held_x_mult(self):
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        held = [_card("Diamonds", "5", enhancement="m_steel")]
        levels = HandLevels()
        result = score_hand_base(
            played,
            held,
            levels,
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # Pair: 10 + 11 + 11 = 32 chips, 2 mult
        # Held Steel: mult *= 1.5 → 3.0
        assert result.chips == 32.0
        assert result.mult == pytest.approx(3.0)
        assert result.total == 96  # floor(32 * 3)

    def test_multiple_steel(self):
        """Two Steel Cards in hand: mult *= 1.5 × 1.5 = 2.25."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        held = [
            _card("Diamonds", "5", enhancement="m_steel"),
            _card("Clubs", "3", enhancement="m_steel"),
        ]
        levels = HandLevels()
        result = score_hand_base(
            played,
            held,
            levels,
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.mult == pytest.approx(2.0 * 1.5 * 1.5)


# ============================================================================
# Edition effects
# ============================================================================


class TestEditions:
    def test_foil_on_scored_card(self):
        """Foil edition: +50 chips."""
        c = _card("Hearts", "Ace")
        c.set_edition({"foil": True})
        played = [c, _card("Spades", "Ace")]
        levels = HandLevels()
        result = score_hand_base(
            played,
            [],
            levels,
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # Pair: 10 base + 11 + 11 = 32 chips from cards
        # Foil edition: +50 chips on first Ace
        # Total chips: 32 + 50 = 82
        assert result.chips == 82.0
        assert result.total == 164  # 82 * 2

    def test_holo_on_scored_card(self):
        """Holo edition: +10 mult."""
        c = _card("Hearts", "Ace")
        c.set_edition({"holo": True})
        played = [c, _card("Spades", "Ace")]
        levels = HandLevels()
        result = score_hand_base(
            played,
            [],
            levels,
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # mult: 2 base + 10 holo = 12
        assert result.mult == 12.0

    def test_polychrome_on_scored_card(self):
        """Polychrome: x1.5 mult."""
        c = _card("Hearts", "Ace")
        c.set_edition({"polychrome": True})
        played = [c, _card("Spades", "Ace")]
        levels = HandLevels()
        result = score_hand_base(
            played,
            [],
            levels,
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # mult: 2 base * 1.5 poly = 3.0
        assert result.mult == pytest.approx(3.0)


# ============================================================================
# Red Seal retrigger
# ============================================================================


class TestRedSealRetrigger:
    """Red Seal: card evaluated twice (base + 1 retrigger)."""

    def test_double_chips(self):
        c = _card("Hearts", "Ace")
        c.set_seal("Red")
        played = [c, _card("Spades", "Ace")]
        levels = HandLevels()
        result = score_hand_base(
            played,
            [],
            levels,
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # Pair: 10 base
        # Red Seal Ace: 11 chips × 2 reps = 22
        # Normal Ace: 11
        # Total chips: 10 + 22 + 11 = 43
        assert result.chips == 43.0
        assert result.total == 86  # 43 * 2

    def test_glass_with_red_seal(self):
        """Glass Card + Red Seal: x2 mult applied twice → x4."""
        c = _card("Hearts", "5", enhancement="m_glass")
        c.set_seal("Red")
        played = [c, _card("Spades", "5")]
        levels = HandLevels()
        result = score_hand_base(
            played,
            [],
            levels,
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # Pair: 10 chips, 2 mult
        # Red Glass 5: rep 1: +5 chips, x2 mult → 4
        #              rep 2: +5 chips, x2 mult → 8
        # Normal 5: +5 chips
        # chips: 10 + 5 + 5 + 5 = 25
        # mult: 2 * 2 * 2 = 8
        assert result.chips == 25.0
        assert result.mult == 8.0
        assert result.total == 200

    def test_held_steel_with_red_seal(self):
        """Steel Card + Red Seal in hand: x1.5 applied twice → x2.25."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        steel = _card("Diamonds", "5", enhancement="m_steel")
        steel.set_seal("Red")
        held = [steel]
        levels = HandLevels()
        result = score_hand_base(
            played,
            held,
            levels,
            _small_blind(),
            PseudoRandom("TEST"),
        )
        # Pair: 32 chips, 2 mult
        # Held Steel+Red: 2 reps × x1.5 = 2 * 1.5 * 1.5 = 4.5
        assert result.mult == pytest.approx(4.5)


# ============================================================================
# Boss blind effects
# ============================================================================


class TestFlintBlind:
    """The Flint: halves base chips and mult (Phase 6)."""

    def test_halving(self):
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        levels = HandLevels()
        blind = Blind.create("bl_flint", ante=1)
        result = score_hand_base(
            played,
            [],
            levels,
            blind,
            PseudoRandom("TEST"),
        )
        # Pair: 10 chips, 2 mult
        # Flint halves: chips = floor(10*0.5+0.5) = 5, mult = max(floor(2*0.5+0.5),1) = 1
        # Per card: 11 + 11 = 22
        # Final chips: 5 + 22 = 27, mult = 1
        assert result.chips == 27.0
        assert result.mult == 1.0
        assert result.total == 27


class TestDebuffedHand:
    """Boss blind blocks the hand entirely."""

    def test_the_eye_repeat(self):
        """The Eye: repeat hand type → debuffed (score = 0)."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        levels = HandLevels()
        blind = Blind.create("bl_eye", ante=1)

        # First Pair: allowed
        r1 = score_hand_base(played, [], levels, blind, PseudoRandom("TEST"))
        assert r1.debuffed is False
        assert r1.total > 0

        # Second Pair: blocked
        reset_sort_id_counter()
        played2 = [_card("Hearts", "King"), _card("Spades", "King")]
        r2 = score_hand_base(played2, [], levels, blind, PseudoRandom("TEST"))
        assert r2.debuffed is True
        assert r2.total == 0

    def test_psychic_too_few_cards(self):
        """The Psychic: < 5 cards → blocked."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        levels = HandLevels()
        blind = Blind.create("bl_psychic", ante=1)
        result = score_hand_base(
            played,
            [],
            levels,
            blind,
            PseudoRandom("TEST"),
        )
        assert result.debuffed is True
        assert result.total == 0


# ============================================================================
# Gold Seal dollars
# ============================================================================


class TestDollars:
    def test_gold_seal(self):
        c = _card("Hearts", "Ace")
        c.set_seal("Gold")
        played = [c, _card("Spades", "Ace")]
        levels = HandLevels()
        result = score_hand_base(
            played,
            [],
            levels,
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.dollars_earned == 3


# ============================================================================
# Empty hand
# ============================================================================


class TestEmptyHand:
    def test_no_cards(self):
        levels = HandLevels()
        result = score_hand_base(
            [],
            [],
            levels,
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.hand_type == "NULL"
        assert result.total == 0


# ============================================================================
# Result structure
# ============================================================================


class TestResultStructure:
    def test_has_breakdown(self):
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        levels = HandLevels()
        result = score_hand_base(
            played,
            [],
            levels,
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert isinstance(result, ScoreResult)
        assert isinstance(result.breakdown, list)
        assert len(result.breakdown) > 0

    def test_breakdown_shows_final(self):
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        levels = HandLevels()
        result = score_hand_base(
            played,
            [],
            levels,
            _small_blind(),
            PseudoRandom("TEST"),
        )
        assert any("Final" in line for line in result.breakdown)

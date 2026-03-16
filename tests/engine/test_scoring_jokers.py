"""Integration tests for the full scoring pipeline with joker effects.

Tests score_hand (Phases 1-9, 12) with real joker handlers to verify
the pipeline correctly orchestrates base scoring + joker dispatch.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.scoring import score_hand


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


_SUIT_LETTER = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}
_RANK_LETTER = {
    "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7",
    "8": "8", "9": "9", "10": "T", "Jack": "J", "Queen": "Q",
    "King": "K", "Ace": "A",
}


def _card(suit: str, rank: str, enhancement: str = "c_base") -> Card:
    c = Card()
    c.set_base(f"{_SUIT_LETTER[suit]}_{_RANK_LETTER[rank]}", suit, rank)
    c.set_ability(enhancement)
    return c


def _joker(center_key: str, *, debuff: bool = False, **ability_kw) -> Card:
    c = Card()
    c.center_key = center_key
    c.debuff = debuff
    c.ability = {"name": center_key, "set": "Joker", **ability_kw}
    return c


def _small_blind() -> Blind:
    return Blind.create("bl_small", ante=1)


# ============================================================================
# Basic joker effects
# ============================================================================

class TestJokerInPipeline:
    """j_joker adds +4 mult in Phase 9."""

    def test_pair_of_aces_with_joker(self):
        """Pair: 10+11+11=32 chips, 2 mult.
        j_joker adds +4 mult → total mult = 6.
        Score: 32 × 6 = 192."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        j = _joker("j_joker", mult=4)
        result = score_hand(
            played, [], [j], HandLevels(), _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.hand_type == "Pair"
        assert result.chips == 32.0
        assert result.mult == 6.0
        assert result.total == 192

    def test_no_jokers_matches_base(self):
        """Without jokers, score_hand matches score_hand_base."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        result = score_hand(
            played, [], [], HandLevels(), _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.total == 64  # 32 × 2


# ============================================================================
# Per-card joker effects (individual context)
# ============================================================================

class TestSuitJokerPerCard:
    """j_lusty_joker adds +3 mult per Heart scored."""

    def test_flush_of_hearts(self):
        """Flush L1: 35 chips, 4 mult.
        Cards: 2+5+8+10+11 = 36 chips.
        j_lusty: +3 mult per Heart scored (5 Hearts) = +15 mult.
        Total: (35+36) chips, (4+15) mult = 71 × 19 = 1349."""
        played = [
            _card("Hearts", "2"), _card("Hearts", "5"),
            _card("Hearts", "8"), _card("Hearts", "Jack"),
            _card("Hearts", "Ace"),
        ]
        j = _joker("j_lusty_joker", extra={"s_mult": 3, "suit": "Hearts"})
        result = score_hand(
            played, [], [j], HandLevels(), _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.hand_type == "Flush"
        assert result.chips == 71.0
        assert result.mult == 19.0
        assert result.total == 1349


class TestMultipleSuitJokers:
    """Two suit jokers trigger on different cards in the same hand."""

    def test_greedy_and_lusty(self):
        """Pair of 5s: one Diamond, one Heart.
        Base: 10+5+5 = 20 chips, 2 mult.
        j_greedy: +3 mult on Diamond 5.
        j_lusty: +3 mult on Heart 5.
        Total mult: 2 + 3 + 3 = 8. Score: 20 × 8 = 160."""
        played = [
            _card("Diamonds", "5"),
            _card("Hearts", "5"),
        ]
        greedy = _joker(
            "j_greedy_joker", extra={"s_mult": 3, "suit": "Diamonds"},
        )
        lusty = _joker(
            "j_lusty_joker", extra={"s_mult": 3, "suit": "Hearts"},
        )
        result = score_hand(
            played, [], [greedy, lusty], HandLevels(), _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.mult == 8.0
        assert result.total == 160


# ============================================================================
# xMult ordering — joker_main (Phase 9, left to right)
# ============================================================================

class TestXMultOrder:
    """Left-to-right joker ordering in Phase 9 matters for mixed add/multiply."""

    def test_two_xmult_commutative(self):
        """j_duo (x2) then j_blackboard (x3): both xMult → order doesn't matter.
        Pair: 10+11+11=32 chips, 2 mult.
        Phase 9: mult × 2 × 3 = 12.
        Score: 32 × 12 = 384."""
        played = [_card("Spades", "Ace"), _card("Clubs", "Ace")]
        held = [_card("Spades", "5")]  # all black for Blackboard
        ph_pair = {"Pair": [["p"]]}  # non-empty for Duo

        duo = _joker("j_duo", x_mult=2, type="Pair")
        bb = _joker("j_blackboard", extra=3)
        result = score_hand(
            played, held, [duo, bb], HandLevels(), _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.mult == pytest.approx(12.0)
        assert result.total == 384

    def test_two_xmult_reversed(self):
        """j_blackboard (x3) then j_duo (x2): still × 6 total."""
        played = [_card("Spades", "Ace"), _card("Clubs", "Ace")]
        held = [_card("Spades", "5")]
        bb = _joker("j_blackboard", extra=3)
        duo = _joker("j_duo", x_mult=2, type="Pair")
        result = score_hand(
            played, held, [bb, duo], HandLevels(), _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.mult == pytest.approx(12.0)
        assert result.total == 384

    def test_additive_then_multiplicative(self):
        """j_joker (+4) then j_blackboard (x3):
        Pair: 32 chips, 2 mult.
        Phase 9: mult = (2 + 4) × 3 = 18.
        Score: 32 × 18 = 576."""
        played = [_card("Spades", "Ace"), _card("Clubs", "Ace")]
        held = [_card("Spades", "5")]
        joker = _joker("j_joker", mult=4)
        bb = _joker("j_blackboard", extra=3)
        result = score_hand(
            played, held, [joker, bb], HandLevels(), _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.mult == pytest.approx(18.0)
        assert result.total == 576

    def test_multiplicative_then_additive(self):
        """j_blackboard (x3) then j_joker (+4):
        Pair: 32 chips, 2 mult.
        Phase 9: mult = (2 × 3) + 4 = 10. DIFFERENT from above!
        Score: 32 × 10 = 320."""
        played = [_card("Spades", "Ace"), _card("Clubs", "Ace")]
        held = [_card("Spades", "5")]
        bb = _joker("j_blackboard", extra=3)
        joker = _joker("j_joker", mult=4)
        result = score_hand(
            played, held, [bb, joker], HandLevels(), _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.mult == pytest.approx(10.0)
        assert result.total == 320


# ============================================================================
# Blueprint copying
# ============================================================================

class TestBlueprintInPipeline:
    """Blueprint copies the joker to its right."""

    def test_blueprint_copies_joker(self):
        """[j_blueprint, j_joker]: Blueprint copies +4 mult, then j_joker adds +4.
        Pair: 32 chips, 2 mult. Phase 9: 2 + 4 + 4 = 10.
        Score: 32 × 10 = 320."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        bp = _joker("j_blueprint")
        joker = _joker("j_joker", mult=4)
        result = score_hand(
            played, [], [bp, joker], HandLevels(), _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.mult == 10.0
        assert result.total == 320


# ============================================================================
# Joker edition effects
# ============================================================================

class TestJokerEdition:
    """Joker editions apply in Phase 9: additive before, multiplicative after."""

    def test_foil_edition_adds_chips(self):
        """Foil joker: +50 chips BEFORE joker effect.
        Pair: 32 chips, 2 mult.
        Phase 9: chips += 50 (Foil) → 82, then j_joker +4 mult → 6.
        Score: 82 × 6 = 492."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        j = _joker("j_joker", mult=4)
        j.set_edition({"foil": True})
        result = score_hand(
            played, [], [j], HandLevels(), _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.chips == 82.0
        assert result.mult == 6.0
        assert result.total == 492

    def test_polychrome_edition_multiplies_after(self):
        """Polychrome joker: x1.5 AFTER joker effect.
        Pair: 32 chips, 2 mult.
        Phase 9: j_joker +4 mult → 6, then Poly x1.5 → 9.
        Score: 32 × 9 = 288."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        j = _joker("j_joker", mult=4)
        j.set_edition({"polychrome": True})
        result = score_hand(
            played, [], [j], HandLevels(), _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.mult == pytest.approx(9.0)
        assert result.total == 288

    def test_holo_edition_adds_mult(self):
        """Holo joker: +10 mult BEFORE joker effect (additive edition).
        Pair: 32 chips, 2 mult.
        Phase 9: mult += 10 (Holo) → 12, then j_joker +4 → 16.
        Score: 32 × 16 = 512."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        j = _joker("j_joker", mult=4)
        j.set_edition({"holo": True})
        result = score_hand(
            played, [], [j], HandLevels(), _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.mult == 16.0
        assert result.total == 512

    def test_foil_and_polychrome_on_different_jokers(self):
        """Foil on first joker (+50 chips), Polychrome on second (x1.5 mult).
        Pair: 32 chips, 2 mult.
        Phase 9 j1: chips += 50 → 82, j_joker +4 → 6.
        Phase 9 j2: j_stuntman +250 chips → 332, then Poly x1.5 → mult 9.
        Score: 332 × 9 = 2988."""
        played = [_card("Hearts", "Ace"), _card("Spades", "Ace")]
        j1 = _joker("j_joker", mult=4)
        j1.set_edition({"foil": True})
        j2 = _joker("j_stuntman", extra={"chip_mod": 250, "h_size": 2})
        j2.set_edition({"polychrome": True})
        result = score_hand(
            played, [], [j1, j2], HandLevels(), _small_blind(),
            PseudoRandom("TEST"),
        )
        assert result.chips == 332.0
        assert result.mult == pytest.approx(9.0)
        assert result.total == 2988

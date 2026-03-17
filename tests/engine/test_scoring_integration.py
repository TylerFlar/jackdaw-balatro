"""Comprehensive integration tests for the base scoring pipeline.

Covers every enhancement, edition, seal, boss blind interaction, and
debuff scenario. All numeric expectations are derived from the source's
scoring order documented in scoring-pipeline.md.
"""

from __future__ import annotations

import time

import pytest

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.hand_levels import HandLevels
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.scoring import score_hand_base


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


def _c(suit: str, rank: str, enh: str = "c_base") -> Card:
    sl = {"H": "Hearts", "D": "Diamonds", "C": "Clubs", "S": "Spades"}
    rl = {
        "2": "2",
        "3": "3",
        "4": "4",
        "5": "5",
        "6": "6",
        "7": "7",
        "8": "8",
        "9": "9",
        "T": "10",
        "J": "Jack",
        "Q": "Queen",
        "K": "King",
        "A": "Ace",
    }
    suit_full = sl.get(suit, suit)
    rank_full = rl.get(rank, rank)
    c = Card()
    key_s = suit if len(suit) == 1 else suit[0]
    rank_to_letter = {"10": "T", "Jack": "J", "Queen": "Q", "King": "K", "Ace": "A"}
    key_r = rank if len(rank) == 1 else rank_to_letter.get(rank_full, rank[0])
    c.set_base(f"{key_s}_{key_r}", suit_full, rank_full)
    c.set_ability(enh)
    return c


def _sb():
    return Blind.create("bl_small", ante=1)


def _score(played, held=None, blind=None, levels=None):
    return score_hand_base(
        played,
        held or [],
        levels or HandLevels(),
        blind or _sb(),
        PseudoRandom("INTEG"),
    )


# ============================================================================
# 1. Every enhancement scored
# ============================================================================


class TestEnhancements:
    """Each enhancement's effect on a scoring Pair of 5s (base: 10 chips, 2 mult)."""

    def test_bonus_card(self):
        """Bonus: +30 chips on the enhanced card."""
        r = _score([_c("H", "5", "m_bonus"), _c("S", "5")])
        # 10 + (5+30) + 5 = 50 chips, 2 mult = 100
        assert r.chips == 50 and r.mult == 2 and r.total == 100

    def test_mult_card(self):
        """Mult Card: +4 mult from the enhanced card."""
        r = _score([_c("H", "5", "m_mult"), _c("S", "5")])
        # 10 + 5 + 5 = 20, mult 2+4 = 6, total 120
        assert r.chips == 20 and r.mult == 6 and r.total == 120

    def test_wild_card(self):
        """Wild Card: no direct scoring effect (only affects suit matching)."""
        r = _score([_c("H", "5", "m_wild"), _c("S", "5")])
        assert r.chips == 20 and r.mult == 2 and r.total == 40

    def test_glass_card(self):
        """Glass Card: x2 mult from the enhanced card."""
        r = _score([_c("H", "5", "m_glass"), _c("S", "5")])
        # chips 20, mult 2*2=4, total 80
        assert r.chips == 20 and r.mult == 4 and r.total == 80

    def test_steel_card_held(self):
        """Steel Card: x1.5 mult when held (not played)."""
        r = _score([_c("H", "5"), _c("S", "5")], held=[_c("D", "3", "m_steel")])
        # 20 chips, mult 2*1.5=3, total 60
        assert r.mult == pytest.approx(3.0) and r.total == 60

    def test_stone_card(self):
        """Stone Card: 50 chips, ignores rank nominal."""
        # Stone Card played as 3rd card — not part of pair, but augmented
        sc = _c("H", "A", "m_stone")
        r = _score([_c("H", "5"), _c("S", "5"), sc])
        # Pair: 10 base + 5 + 5 = 20 from pair. Stone adds 50.
        # chips: 20 + 50 = 70, mult 2, total 140
        assert r.chips == 70 and r.total == 140

    def test_gold_card_enhancement(self):
        """Gold Card: h_dollars=3 (earned at END of round, not when scored).
        p_dollars=0 in config, so no dollars during scoring."""
        r = _score([_c("H", "5", "m_gold"), _c("S", "5")])
        assert r.dollars_earned == 0  # Gold Card $ is h_dollars, not p_dollars
        assert r.total == 40  # no scoring effect


# ============================================================================
# 2. Every edition scored
# ============================================================================


class TestEditions:
    def test_foil(self):
        """Foil: +50 chips from edition."""
        c = _c("H", "5")
        c.set_edition({"foil": True})
        r = _score([c, _c("S", "5")])
        # 10+5+5=20 + 50 foil = 70, mult 2, total 140
        assert r.chips == 70 and r.total == 140

    def test_holographic(self):
        """Holo: +10 mult from edition."""
        c = _c("H", "5")
        c.set_edition({"holo": True})
        r = _score([c, _c("S", "5")])
        # chips 20, mult 2+10=12, total 240
        assert r.mult == 12 and r.total == 240

    def test_polychrome(self):
        """Polychrome: x1.5 mult from edition."""
        c = _c("H", "5")
        c.set_edition({"polychrome": True})
        r = _score([c, _c("S", "5")])
        # chips 20, mult 2*1.5=3, total 60
        assert r.mult == pytest.approx(3.0) and r.total == 60


# ============================================================================
# 3-6. Red Seal retrigger combinations
# ============================================================================


class TestRedSealRetrigger:
    def test_basic_retrigger(self):
        """Red Seal: card scores twice (chips from both reps)."""
        c = _c("H", "A")
        c.set_seal("Red")
        r = _score([c, _c("S", "A")])
        # 10 + 11*2 + 11 = 43, mult 2, total 86
        assert r.chips == 43 and r.total == 86

    def test_red_seal_glass(self):
        """Red + Glass: x2 fires twice = x4 total on that card."""
        c = _c("H", "5", "m_glass")
        c.set_seal("Red")
        r = _score([c, _c("S", "5")])
        # 10 + 5*2 + 5 = 25, mult 2*2*2=8, total 200
        assert r.mult == 8 and r.total == 200

    def test_red_seal_polychrome(self):
        """Red + Polychrome: x1.5 edition fires twice = x2.25."""
        c = _c("H", "5")
        c.set_edition({"polychrome": True})
        c.set_seal("Red")
        r = _score([c, _c("S", "5")])
        # 10+5*2+5=25, mult 2 * 1.5 * 1.5 = 4.5, total 112
        assert r.mult == pytest.approx(4.5) and r.total == 112

    def test_red_seal_holo(self):
        """Red + Holo: +10 mult fires twice = +20."""
        c = _c("H", "5")
        c.set_edition({"holo": True})
        c.set_seal("Red")
        r = _score([c, _c("S", "5")])
        # 10+5*2+5=25, mult 2+10+10=22, total 550
        assert r.mult == 22 and r.total == 550

    def test_held_steel_red_seal(self):
        """Held Steel + Red Seal: x1.5 fires twice = x2.25."""
        steel = _c("D", "3", "m_steel")
        steel.set_seal("Red")
        r = _score([_c("H", "5"), _c("S", "5")], held=[steel])
        assert r.mult == pytest.approx(2.0 * 1.5 * 1.5) and r.total == 90


# ============================================================================
# 7. Enhancement + Edition stacking
# ============================================================================


class TestEnhancementEditionStack:
    def test_mult_card_with_holo(self):
        """Mult Card (+4 mult) + Holo (+10 mult) = +14 total on that card."""
        c = _c("H", "5", "m_mult")
        c.set_edition({"holo": True})
        r = _score([c, _c("S", "5")])
        # 10+5+5=20, mult 2+4+10=16, total 320
        assert r.mult == 16 and r.total == 320

    def test_glass_with_polychrome(self):
        """Glass (x2) + Polychrome (x1.5) on same card."""
        c = _c("H", "5", "m_glass")
        c.set_edition({"polychrome": True})
        r = _score([c, _c("S", "5")])
        # 10+5+5=20, mult: base 2, glass x2 → 4, poly x1.5 → 6
        assert r.mult == pytest.approx(6.0) and r.total == 120

    def test_bonus_with_foil(self):
        """Bonus (+30 chips) + Foil (+50 chips) = +80 chips from that card."""
        c = _c("H", "5", "m_bonus")
        c.set_edition({"foil": True})
        r = _score([c, _c("S", "5")])
        # 10 + (5+30) + 50 + 5 = 100, mult 2, total 200
        assert r.chips == 100 and r.total == 200


# ============================================================================
# 8. Boss blind interactions
# ============================================================================


class TestBossBlindScoring:
    def test_the_flint(self):
        """The Flint: halves base chips/mult before per-card effects."""
        r = _score([_c("H", "A"), _c("S", "A")], blind=Blind.create("bl_flint", ante=1))
        # Pair base: 10 chips, 2 mult → Flint: 5 chips, 1 mult
        # Per card: +11+11 = 5+22 = 27, mult 1
        assert r.chips == 27 and r.mult == 1 and r.total == 27

    def test_the_eye_first_allowed(self):
        blind = Blind.create("bl_eye", ante=1)
        r = _score([_c("H", "A"), _c("S", "A")], blind=blind)
        assert r.debuffed is False and r.total > 0

    def test_the_eye_repeat_blocked(self):
        blind = Blind.create("bl_eye", ante=1)
        _score([_c("H", "A"), _c("S", "A")], blind=blind)  # first Pair
        reset_sort_id_counter()
        r = _score([_c("H", "K"), _c("S", "K")], blind=blind)  # second Pair
        assert r.debuffed is True and r.total == 0

    def test_the_mouth_locks(self):
        blind = Blind.create("bl_mouth", ante=1)
        _score([_c("H", "A"), _c("S", "A")], blind=blind)  # Pair
        reset_sort_id_counter()
        r = _score(
            [_c("H", "2"), _c("H", "5"), _c("H", "8"), _c("H", "J"), _c("H", "A")],
            blind=blind,
        )
        assert r.debuffed is True  # Flush != Pair

    def test_the_psychic_5_cards_ok(self):
        blind = Blind.create("bl_psychic", ante=1)
        r = _score(
            [_c("H", "2"), _c("H", "5"), _c("H", "8"), _c("H", "J"), _c("H", "A")],
            blind=blind,
        )
        assert r.debuffed is False

    def test_the_psychic_fewer_blocked(self):
        blind = Blind.create("bl_psychic", ante=1)
        r = _score([_c("H", "A"), _c("S", "A")], blind=blind)
        assert r.debuffed is True

    def test_suit_debuff_goad(self):
        """The Goad: Spade cards are debuffed (contribute 0)."""
        blind = Blind.create("bl_goad", ante=1)
        s_card = _c("S", "5")
        h_card = _c("H", "5")
        blind.debuff_card(s_card)
        blind.debuff_card(h_card)
        r = _score([s_card, h_card], blind=blind)
        # Pair detected but Spade 5 is debuffed → only Heart 5 scores
        # chips: 10 + 0 + 5 = 15 (debuffed card adds 0)
        assert r.chips == 15

    def test_face_debuff_plant(self):
        """The Plant: face cards debuffed."""
        blind = Blind.create("bl_plant", ante=1)
        k1 = _c("H", "K")
        k2 = _c("S", "K")
        blind.debuff_card(k1)
        blind.debuff_card(k2)
        r = _score([k1, k2], blind=blind)
        # Both Kings debuffed → 0 chip contribution
        assert r.chips == 10  # just base chips, no card chips


# ============================================================================
# 9. Debuffed cards in scoring hand
# ============================================================================


class TestDebuffedCards:
    def test_debuffed_card_no_chips(self):
        c = _c("H", "A")
        c.set_debuff(True)
        r = _score([c, _c("S", "A")])
        # Only non-debuffed Ace contributes chips
        assert r.chips == 10 + 0 + 11  # base + debuffed(0) + normal(11) = 21

    def test_debuffed_card_no_edition(self):
        c = _c("H", "A")
        c.set_edition({"foil": True})
        c.set_debuff(True)
        r = _score([c, _c("S", "A")])
        # Foil on debuffed card: no +50 chips
        assert r.chips == 21  # same as above, no foil

    def test_debuffed_card_no_seal(self):
        c = _c("H", "A")
        c.set_seal("Red")
        c.set_debuff(True)
        r = _score([c, _c("S", "A")])
        # Red Seal on debuffed: no retrigger
        assert r.chips == 21  # no double from Red Seal

    def test_debuffed_card_no_glass_xmult(self):
        c = _c("H", "5", "m_glass")
        c.set_debuff(True)
        r = _score([c, _c("S", "5")])
        assert r.mult == 2  # no x2 from debuffed Glass


# ============================================================================
# 10. Performance
# ============================================================================


class TestPerformance:
    def test_scoring_throughput(self):
        """Score a typical 5-card hand many times."""
        played = [_c("H", "5"), _c("S", "5"), _c("C", "8"), _c("D", "J"), _c("H", "A")]
        held = [_c("S", "3"), _c("D", "7"), _c("C", "K")]
        levels = HandLevels()
        blind = _sb()
        rng = PseudoRandom("BENCH")

        n = 1000
        start = time.perf_counter()
        for _ in range(n):
            score_hand_base(played, held, levels, blind, rng)
        elapsed = time.perf_counter() - start

        rate = n / elapsed
        us_per = elapsed / n * 1_000_000
        print(f"\n  score_hand_base: {us_per:.0f} us/hand ({rate:,.0f} hands/sec)")
        assert elapsed < 5.0


# ============================================================================
# Regression: specific numeric values
# ============================================================================


class TestRegressionValues:
    """Pin exact scores to catch any future rounding or ordering changes."""

    def test_pair_aces_level1(self):
        r = _score([_c("H", "A"), _c("S", "A")])
        assert r.total == 64

    def test_three_kings_foil(self):
        k1 = _c("H", "K")
        k1.set_edition({"foil": True})
        r = _score([k1, _c("S", "K"), _c("C", "K"), _c("D", "5"), _c("H", "2")])
        assert r.total == 330

    def test_flush_glass(self):
        r = _score(
            [
                _c("H", "2"),
                _c("H", "5"),
                _c("H", "8"),
                _c("H", "J"),
                _c("H", "A", "m_glass"),
            ]
        )
        assert r.total == 568

    def test_full_house_steel_held(self):
        r = _score(
            [_c("H", "K"), _c("S", "K"), _c("C", "K"), _c("D", "5"), _c("H", "5")],
            held=[_c("C", "3", "m_steel")],
        )
        assert r.total == 480

    def test_pair_red_seal(self):
        c = _c("H", "A")
        c.set_seal("Red")
        r = _score([c, _c("S", "A")])
        assert r.total == 86

    def test_pair_flint(self):
        r = _score([_c("H", "A"), _c("S", "A")], blind=Blind.create("bl_flint", ante=1))
        assert r.total == 27

"""Tests for Card scoring methods.

Verifies each card-level scoring method against the Lua source behavior
for every relevant enhancement, edition, seal, and debuff state.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.card import Card, reset_sort_id_counter


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


def _playing_card(suit: str, rank: str, enhancement: str = "c_base") -> Card:
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


# ============================================================================
# get_chip_bonus
# ============================================================================


class TestGetChipBonus:
    def test_number_card(self):
        c = _playing_card("Hearts", "7")
        assert c.get_chip_bonus() == 7

    def test_ace(self):
        c = _playing_card("Spades", "Ace")
        assert c.get_chip_bonus() == 11

    def test_face_card(self):
        c = _playing_card("Diamonds", "King")
        assert c.get_chip_bonus() == 10

    def test_ten(self):
        c = _playing_card("Clubs", "10")
        assert c.get_chip_bonus() == 10

    def test_bonus_card(self):
        """Bonus Card: +30 chips via config.bonus."""
        c = _playing_card("Hearts", "5", enhancement="m_bonus")
        assert c.get_chip_bonus() == 5 + 30

    def test_stone_card(self):
        """Stone Card: ignores nominal, returns bonus only (50)."""
        c = _playing_card("Hearts", "Ace", enhancement="m_stone")
        assert c.get_chip_bonus() == 50

    def test_stone_card_with_perma_bonus(self):
        c = _playing_card("Hearts", "Ace", enhancement="m_stone")
        c.ability["perma_bonus"] = 10
        assert c.get_chip_bonus() == 50 + 10

    def test_perma_bonus(self):
        c = _playing_card("Hearts", "5")
        c.ability["perma_bonus"] = 15
        assert c.get_chip_bonus() == 5 + 15

    def test_debuffed(self):
        c = _playing_card("Hearts", "Ace")
        c.debuff = True
        assert c.get_chip_bonus() == 0

    def test_mult_card_no_chip_bonus(self):
        """Mult Card: no extra chip bonus (mult=4 is in get_chip_mult)."""
        c = _playing_card("Hearts", "5", enhancement="m_mult")
        assert c.get_chip_bonus() == 5  # just nominal


# ============================================================================
# get_chip_mult
# ============================================================================


class TestGetChipMult:
    def test_normal_card(self):
        c = _playing_card("Hearts", "5")
        assert c.get_chip_mult() == 0

    def test_mult_card(self):
        """Mult Card: ability.mult = 4."""
        c = _playing_card("Hearts", "5", enhancement="m_mult")
        assert c.get_chip_mult() == 4

    def test_lucky_card_without_rng(self):
        """Lucky Card without RNG: returns 0 (needs actual roll)."""
        c = _playing_card("Hearts", "5", enhancement="m_lucky")
        assert c.get_chip_mult() == 0

    def test_debuffed(self):
        c = _playing_card("Hearts", "5", enhancement="m_mult")
        c.debuff = True
        assert c.get_chip_mult() == 0

    def test_joker_returns_0(self):
        """Jokers use calculate_joker, not get_chip_mult."""
        c = Card()
        c.set_ability("j_joker")
        assert c.get_chip_mult() == 0

    def test_bonus_card_no_mult(self):
        c = _playing_card("Hearts", "5", enhancement="m_bonus")
        assert c.get_chip_mult() == 0


# ============================================================================
# get_chip_x_mult
# ============================================================================


class TestGetChipXMult:
    def test_normal_card(self):
        c = _playing_card("Hearts", "5")
        assert c.get_chip_x_mult() == 0

    def test_glass_card(self):
        """Glass Card: x_mult = 2.0."""
        c = _playing_card("Hearts", "5", enhancement="m_glass")
        assert c.get_chip_x_mult() == 2

    def test_debuffed(self):
        c = _playing_card("Hearts", "5", enhancement="m_glass")
        c.debuff = True
        assert c.get_chip_x_mult() == 0

    def test_joker_returns_0(self):
        c = Card()
        c.set_ability("j_joker")
        assert c.get_chip_x_mult() == 0

    def test_x_mult_of_1_returns_0(self):
        """x_mult <= 1 is treated as 'no bonus'."""
        c = _playing_card("Hearts", "5")
        c.ability["x_mult"] = 1
        assert c.get_chip_x_mult() == 0


# ============================================================================
# get_chip_h_mult
# ============================================================================


class TestGetChipHMult:
    def test_normal_card(self):
        c = _playing_card("Hearts", "5")
        assert c.get_chip_h_mult() == 0

    def test_debuffed(self):
        c = _playing_card("Hearts", "5")
        c.ability["h_mult"] = 5
        c.debuff = True
        assert c.get_chip_h_mult() == 0


# ============================================================================
# get_chip_h_x_mult
# ============================================================================


class TestGetChipHXMult:
    def test_normal_card(self):
        c = _playing_card("Hearts", "5")
        assert c.get_chip_h_x_mult() == 0

    def test_steel_card(self):
        """Steel Card: h_x_mult = 1.5."""
        c = _playing_card("Hearts", "5", enhancement="m_steel")
        assert c.get_chip_h_x_mult() == pytest.approx(1.5)

    def test_debuffed(self):
        c = _playing_card("Hearts", "5", enhancement="m_steel")
        c.debuff = True
        assert c.get_chip_h_x_mult() == 0


# ============================================================================
# get_edition
# ============================================================================


class TestGetEdition:
    def test_no_edition(self):
        c = _playing_card("Hearts", "5")
        assert c.get_edition() is None

    def test_foil(self):
        c = _playing_card("Hearts", "5")
        c.set_edition({"foil": True})
        ed = c.get_edition()
        assert ed is not None
        assert ed["chip_mod"] == 50
        assert "mult_mod" not in ed
        assert "x_mult_mod" not in ed

    def test_holographic(self):
        c = _playing_card("Hearts", "5")
        c.set_edition({"holo": True})
        ed = c.get_edition()
        assert ed is not None
        assert ed["mult_mod"] == 10
        assert "chip_mod" not in ed

    def test_polychrome(self):
        c = _playing_card("Hearts", "5")
        c.set_edition({"polychrome": True})
        ed = c.get_edition()
        assert ed is not None
        assert ed["x_mult_mod"] == pytest.approx(1.5)
        assert "chip_mod" not in ed
        assert "mult_mod" not in ed

    def test_negative(self):
        """Negative edition has no scoring bonuses."""
        c = _playing_card("Hearts", "5")
        c.set_edition({"negative": True})
        ed = c.get_edition()
        assert ed is not None
        assert "chip_mod" not in ed
        assert "mult_mod" not in ed
        assert "x_mult_mod" not in ed

    def test_debuffed(self):
        c = _playing_card("Hearts", "5")
        c.set_edition({"foil": True})
        c.debuff = True
        assert c.get_edition() is None

    def test_edition_has_card_ref(self):
        c = _playing_card("Hearts", "5")
        c.set_edition({"holo": True})
        ed = c.get_edition()
        assert ed["card"] is c


# ============================================================================
# set_edition (updated to populate scoring values)
# ============================================================================


class TestSetEdition:
    def test_foil_populates_chips(self):
        c = Card()
        c.set_edition({"foil": True})
        assert c.edition["chips"] == 50
        assert c.edition["foil"] is True
        assert c.edition["type"] == "foil"

    def test_holo_populates_mult(self):
        c = Card()
        c.set_edition({"holo": True})
        assert c.edition["mult"] == 10
        assert c.edition["type"] == "holo"

    def test_polychrome_populates_x_mult(self):
        c = Card()
        c.set_edition({"polychrome": True})
        assert c.edition["x_mult"] == pytest.approx(1.5)
        assert c.edition["type"] == "polychrome"

    def test_negative_no_scoring(self):
        c = Card()
        c.set_edition({"negative": True})
        assert c.edition["negative"] is True
        assert "chips" not in c.edition
        assert "mult" not in c.edition

    def test_clear_edition(self):
        c = Card()
        c.set_edition({"foil": True})
        c.set_edition(None)
        assert c.edition is None

    def test_empty_dict_clears(self):
        c = Card()
        c.set_edition({"foil": True})
        c.set_edition({})
        assert c.edition is None


# ============================================================================
# get_p_dollars
# ============================================================================


class TestGetPDollars:
    def test_normal_card(self):
        c = _playing_card("Hearts", "5")
        assert c.get_p_dollars() == 0

    def test_gold_seal(self):
        """Gold Seal: +$3 on score."""
        c = _playing_card("Hearts", "5")
        c.set_seal("Gold")
        assert c.get_p_dollars() == 3

    def test_gold_card_enhancement(self):
        """Gold Card enhancement: h_dollars=3, NOT p_dollars.

        Gold Card earns $3 when HELD at end of round, not when scored.
        get_p_dollars should return 0 for Gold Card (p_dollars=0 in config).
        """
        c = _playing_card("Hearts", "5", enhancement="m_gold")
        assert c.get_p_dollars() == 0  # h_dollars, not p_dollars

    def test_lucky_card_without_rng(self):
        """Lucky Card without RNG: returns 0 (needs actual roll)."""
        c = _playing_card("Hearts", "5", enhancement="m_lucky")
        assert c.get_p_dollars() == 0

    def test_gold_seal_plus_gold_card(self):
        """Gold Seal on Gold Card: $3 from seal (p_dollars=0 from Gold Card)."""
        c = _playing_card("Hearts", "5", enhancement="m_gold")
        c.set_seal("Gold")
        assert c.get_p_dollars() == 3

    def test_debuffed(self):
        c = _playing_card("Hearts", "5")
        c.set_seal("Gold")
        c.debuff = True
        assert c.get_p_dollars() == 0


# ============================================================================
# calculate_seal
# ============================================================================


class TestCalculateSeal:
    def test_red_seal_repetition(self):
        c = _playing_card("Hearts", "5")
        c.set_seal("Red")
        result = c.calculate_seal(repetition=True)
        assert result is not None
        assert result["repetitions"] == 1
        assert result["card"] is c

    def test_no_seal(self):
        c = _playing_card("Hearts", "5")
        assert c.calculate_seal(repetition=True) is None

    def test_gold_seal_no_repetition(self):
        """Gold Seal has no repetition effect."""
        c = _playing_card("Hearts", "5")
        c.set_seal("Gold")
        assert c.calculate_seal(repetition=True) is None

    def test_blue_seal_no_repetition(self):
        c = _playing_card("Hearts", "5")
        c.set_seal("Blue")
        assert c.calculate_seal(repetition=True) is None

    def test_purple_seal_no_repetition(self):
        c = _playing_card("Hearts", "5")
        c.set_seal("Purple")
        assert c.calculate_seal(repetition=True) is None

    def test_debuffed(self):
        c = _playing_card("Hearts", "5")
        c.set_seal("Red")
        c.debuff = True
        assert c.calculate_seal(repetition=True) is None

    def test_non_repetition_context(self):
        """Without repetition=True, returns None even for Red Seal."""
        c = _playing_card("Hearts", "5")
        c.set_seal("Red")
        assert c.calculate_seal() is None


# ============================================================================
# Combined scenarios
# ============================================================================


class TestCombinedScoring:
    def test_bonus_card_with_foil(self):
        """Bonus Card + Foil: get_chip_bonus=5+30, get_edition=+50 chips."""
        c = _playing_card("Hearts", "5", enhancement="m_bonus")
        c.set_edition({"foil": True})
        assert c.get_chip_bonus() == 35
        ed = c.get_edition()
        assert ed["chip_mod"] == 50

    def test_mult_card_with_holo(self):
        """Mult Card + Holo: get_chip_mult=4, get_edition=+10 mult."""
        c = _playing_card("Hearts", "5", enhancement="m_mult")
        c.set_edition({"holo": True})
        assert c.get_chip_mult() == 4
        ed = c.get_edition()
        assert ed["mult_mod"] == 10

    def test_glass_with_polychrome(self):
        """Glass + Polychrome: x_mult=2 from Glass, x_mult_mod=1.5 from Poly."""
        c = _playing_card("Hearts", "5", enhancement="m_glass")
        c.set_edition({"polychrome": True})
        assert c.get_chip_x_mult() == 2
        ed = c.get_edition()
        assert ed["x_mult_mod"] == pytest.approx(1.5)

    def test_gold_seal_red_seal_priority(self):
        """Can't have two seals — last set wins."""
        c = _playing_card("Hearts", "5")
        c.set_seal("Gold")
        c.set_seal("Red")
        assert c.seal == "Red"
        assert c.get_p_dollars() == 0  # Red seal has no $ effect
        assert c.calculate_seal(repetition=True)["repetitions"] == 1

    def test_all_zero_for_plain_card(self):
        """Plain card: chips=nominal, mult=0, x_mult=0, edition=None."""
        c = _playing_card("Hearts", "5")
        assert c.get_chip_bonus() == 5
        assert c.get_chip_mult() == 0
        assert c.get_chip_x_mult() == 0
        assert c.get_chip_h_mult() == 0
        assert c.get_chip_h_x_mult() == 0
        assert c.get_edition() is None
        assert c.get_p_dollars() == 0
        assert c.calculate_seal(repetition=True) is None

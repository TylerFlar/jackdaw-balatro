"""Tests for the Card class.

Verifies card creation from prototypes (both string-key and dict APIs),
base field population, ability initialization with post-init fields,
scoring methods, deep-copy isolation, and sort_id auto-increment.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.card import Card, CardBase, reset_sort_id_counter
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.data.prototypes import JOKERS


@pytest.fixture(autouse=True)
def _reset_sort_ids():
    """Reset sort_id counter before each test for determinism."""
    reset_sort_id_counter()


# ============================================================================
# CardBase
# ============================================================================

class TestCardBase:
    """CardBase: playing card identity with computed numeric values."""

    def test_ace_of_spades(self):
        b = CardBase.from_card_key("S_A", "Spades", "Ace")
        assert b.suit is Suit.SPADES
        assert b.rank is Rank.ACE
        assert b.id == 14
        assert b.nominal == 11
        assert b.suit_nominal == 0.04
        assert b.suit_nominal_original == 0.004
        assert b.face_nominal == 0.4
        assert b.original_value is Rank.ACE
        assert b.times_played == 0

    def test_king_of_hearts(self):
        b = CardBase.from_card_key("H_K", "Hearts", "King")
        assert b.id == 13
        assert b.nominal == 10
        assert b.face_nominal == 0.3

    def test_two_of_diamonds(self):
        b = CardBase.from_card_key("D_2", "Diamonds", "2")
        assert b.id == 2
        assert b.nominal == 2
        assert b.face_nominal == 0.0
        assert b.suit_nominal == 0.01

    def test_jack_face_nominal(self):
        b = CardBase.from_card_key("C_J", "Clubs", "Jack")
        assert b.face_nominal == 0.1
        assert b.nominal == 10

    def test_ten_not_face(self):
        b = CardBase.from_card_key("S_T", "Spades", "10")
        assert b.face_nominal == 0.0


# ============================================================================
# Card creation and sort_id
# ============================================================================

class TestCardCreation:
    def test_sort_id_auto_increments(self):
        c1, c2, c3 = Card(), Card(), Card()
        assert (c1.sort_id, c2.sort_id, c3.sort_id) == (1, 2, 3)

    def test_sort_id_reset(self):
        Card()
        Card()
        reset_sort_id_counter()
        assert Card().sort_id == 1

    def test_defaults(self):
        c = Card()
        assert c.base is None
        assert c.center_key == "c_base"
        assert c.ability == {}
        assert c.edition is None
        assert c.seal is None
        assert c.debuff is False
        assert c.eternal is False
        assert c.perish_tally == 5


# ============================================================================
# set_ability — string key API (primary interface)
# ============================================================================

class TestSetAbilityByKey:
    """set_ability(center_key: str) looks up P_CENTERS and populates ability."""

    def test_joker_flat_mult(self):
        """j_joker: config.mult=4, no Xmult."""
        c = Card()
        c.set_ability("j_joker")
        assert c.ability["name"] == "Joker"
        assert c.ability["set"] == "Joker"
        assert c.ability["effect"] == "Mult"
        assert c.ability["mult"] == 4
        assert c.ability["x_mult"] == 1  # no Xmult in config → default 1
        assert c.center_key == "j_joker"
        assert c.base_cost == 2

    def test_greedy_joker_nested_extra(self):
        """j_greedy_joker: config.extra.s_mult=3, config.extra.suit='Diamonds'."""
        c = Card()
        c.set_ability("j_greedy_joker")
        assert c.ability["extra"]["s_mult"] == 3
        assert c.ability["extra"]["suit"] == "Diamonds"
        assert c.ability["effect"] == "Suit Mult"

    def test_duo_xmult(self):
        """j_duo: config.Xmult=2 (top-level) → ability.x_mult=2."""
        c = Card()
        c.set_ability("j_duo")
        assert c.ability["x_mult"] == 2
        assert c.ability["name"] == "The Duo"
        assert c.ability["type"] == "Pair"

    def test_ice_cream_extra_chips(self):
        """j_ice_cream: config.extra.chips=100."""
        c = Card()
        c.set_ability("j_ice_cream")
        assert c.ability["extra"]["chips"] == 100
        assert c.ability["extra"]["chip_mod"] == 5

    def test_hands_played_at_create(self):
        """All cards get hands_played_at_create from game state."""
        c = Card()
        c.set_ability("j_joker", hands_played=42)
        assert c.ability["hands_played_at_create"] == 42

    def test_hands_played_at_create_defaults_zero(self):
        c = Card()
        c.set_ability("j_joker")
        assert c.ability["hands_played_at_create"] == 0

    def test_bonus_card_enhancement(self):
        """m_bonus: config.bonus=30."""
        c = Card()
        c.set_base("D_5", "Diamonds", "5")
        c.set_ability("m_bonus")
        assert c.ability["bonus"] == 30
        assert c.get_chip_bonus() == 5 + 30

    def test_invalid_key_raises(self):
        c = Card()
        with pytest.raises(KeyError, match="Unknown center key"):
            c.set_ability("j_nonexistent_joker")


# ============================================================================
# set_ability — special post-init fields (card.lua:308-337)
# ============================================================================

class TestPostInitFields:
    """Joker-specific fields set after the main ability assignment."""

    def test_invisible_joker(self):
        c = Card()
        c.set_ability("j_invisible")
        assert c.ability["invis_rounds"] == 0

    def test_caino(self):
        c = Card()
        c.set_ability("j_caino")
        assert c.ability["caino_xmult"] == 1

    def test_yorick(self):
        c = Card()
        c.set_ability("j_yorick")
        assert c.ability["yorick_discards"] == c.ability["extra"]["discards"]
        assert c.ability["yorick_discards"] > 0  # should be 23

    def test_loyalty_card(self):
        c = Card()
        c.set_ability("j_loyalty_card")
        assert c.ability["loyalty_remaining"] == c.ability["extra"]["every"]
        assert c.ability["loyalty_remaining"] == 5
        assert c.ability["burnt_hand"] == 0

    def test_non_special_joker_no_extra_fields(self):
        """Normal jokers don't get special post-init fields."""
        c = Card()
        c.set_ability("j_joker")
        assert "invis_rounds" not in c.ability
        assert "caino_xmult" not in c.ability
        assert "yorick_discards" not in c.ability
        assert "loyalty_remaining" not in c.ability


# ============================================================================
# Deep copy isolation
# ============================================================================

class TestDeepCopyIsolation:
    """Mutating one card's extra must not affect another card or the prototype."""

    def test_two_cards_from_same_proto(self):
        c1 = Card()
        c1.set_ability("j_greedy_joker")
        c2 = Card()
        c2.set_ability("j_greedy_joker")

        c1.ability["extra"]["s_mult"] = 999
        assert c2.ability["extra"]["s_mult"] == 3  # unaffected

    def test_prototype_unaffected(self):
        c = Card()
        c.set_ability("j_greedy_joker")
        c.ability["extra"]["s_mult"] = 999
        assert JOKERS["j_greedy_joker"].config["extra"]["s_mult"] == 3

    def test_ice_cream_decay_isolated(self):
        """Simulate Ice Cream chip decay — must not affect other instances."""
        c1 = Card()
        c1.set_ability("j_ice_cream")
        c2 = Card()
        c2.set_ability("j_ice_cream")

        # Simulate decay on c1
        c1.ability["extra"]["chips"] -= 5
        assert c1.ability["extra"]["chips"] == 95
        assert c2.ability["extra"]["chips"] == 100  # unaffected

    def test_scalar_extra_not_shared(self):
        """Jokers with scalar extra (not dict) are also independent."""
        c1 = Card()
        c1.set_ability("j_fibonacci")  # extra = 8 (scalar)
        c2 = Card()
        c2.set_ability("j_fibonacci")
        # Scalars are copied by value, so this is inherently safe
        assert c1.ability["extra"] == 8
        assert c2.ability["extra"] == 8


# ============================================================================
# set_ability — dict API (backward compat)
# ============================================================================

class TestSetAbilityByDict:
    """set_ability(dict) still works for custom/test centers."""

    def test_raw_dict(self):
        c = Card()
        c.set_ability({
            "key": "test_joker", "name": "Test", "set": "Joker",
            "effect": "Mult", "config": {"mult": 7}, "cost": 3,
        })
        assert c.ability["name"] == "Test"
        assert c.ability["mult"] == 7
        assert c.center_key == "test_joker"
        assert c.base_cost == 3

    def test_consumeable_config_ref(self):
        c = Card()
        c.set_ability({
            "key": "c_magician", "name": "The Magician", "set": "Tarot",
            "config": {"mod_conv": "m_lucky", "max_highlighted": 2},
            "consumeable": True,
        })
        assert c.ability["consumeable"]["mod_conv"] == "m_lucky"


# ============================================================================
# perma_bonus preservation
# ============================================================================

class TestPermaBonus:
    def test_preserved_across_set_ability(self):
        c = Card()
        c.set_base("S_5", "Spades", "5")
        c.set_ability("c_base")
        c.ability["perma_bonus"] = 15  # Hiker effect
        c.set_ability("c_base")  # re-apply
        assert c.ability["perma_bonus"] == 15
        assert c.get_chip_bonus() == 5 + 15

    def test_preserved_across_key_change(self):
        c = Card()
        c.set_base("H_3", "Hearts", "3")
        c.set_ability("c_base")
        c.ability["perma_bonus"] = 10
        c.set_ability("m_bonus")  # change to Bonus Card
        assert c.ability["perma_bonus"] == 10
        assert c.get_chip_bonus() == 3 + 30 + 10  # nominal + bonus + perma


# ============================================================================
# Scoring methods
# ============================================================================

class TestScoringMethods:
    def test_is_face_jack(self):
        c = Card()
        c.set_base("H_J", "Hearts", "Jack")
        assert c.is_face() is True

    def test_is_face_ace_is_false(self):
        c = Card()
        c.set_base("D_A", "Diamonds", "Ace")
        assert c.is_face() is False

    def test_is_face_debuffed(self):
        c = Card()
        c.set_base("H_K", "Hearts", "King")
        c.debuff = True
        assert c.is_face() is False

    def test_get_id_stone_card(self):
        c = Card()
        c.set_base("H_5", "Hearts", "5")
        c.set_ability("m_stone")
        assert c.get_id() == -1

    def test_get_chip_x_mult_glass(self):
        c = Card()
        c.set_ability("m_glass")
        assert c.get_chip_x_mult() == 2

    def test_stone_card_ignores_base_nominal(self):
        c = Card()
        c.set_base("H_A", "Hearts", "Ace")  # nominal = 11
        c.set_ability("m_stone")
        # Stone Card uses bonus only (50), not base nominal
        assert c.get_chip_bonus() == 50

    def test_repr(self):
        c = Card()
        c.set_ability("j_joker")
        assert "Joker" in repr(c)


# ============================================================================
# Stickers
# ============================================================================

class TestStickers:
    def test_set_eternal(self):
        c = Card()
        c.set_eternal(True)
        assert c.eternal is True

    def test_set_perishable_resets_tally(self):
        c = Card()
        c.perish_tally = 0
        c.set_perishable(True)
        assert c.perish_tally == 5

    def test_set_edition(self):
        c = Card()
        c.set_edition({"foil": True})
        assert c.edition == {"foil": True}

    def test_set_seal(self):
        c = Card()
        c.set_seal("Red")
        assert c.seal == "Red"

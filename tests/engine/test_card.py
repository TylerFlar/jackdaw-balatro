"""Tests for the Card class.

Verifies card creation from prototypes, base field population,
ability initialization, scoring methods, and sort_id auto-increment.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.card import Card, CardBase, reset_sort_id_counter
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.data.prototypes import JOKERS, PLAYING_CARDS, TAROTS


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
        assert b.suit_nominal == 0.03

    def test_two_of_diamonds(self):
        b = CardBase.from_card_key("D_2", "Diamonds", "2")
        assert b.id == 2
        assert b.nominal == 2
        assert b.face_nominal == 0.0
        assert b.suit_nominal == 0.01

    def test_jack_is_face(self):
        b = CardBase.from_card_key("C_J", "Clubs", "Jack")
        assert b.face_nominal == 0.1
        assert b.nominal == 10

    def test_ten_is_not_face(self):
        b = CardBase.from_card_key("S_T", "Spades", "10")
        assert b.face_nominal == 0.0


# ============================================================================
# Card creation and sort_id
# ============================================================================

class TestCardCreation:
    """Card creation and auto-incrementing sort_id."""

    def test_sort_id_auto_increments(self):
        c1 = Card()
        c2 = Card()
        c3 = Card()
        assert c1.sort_id == 1
        assert c2.sort_id == 2
        assert c3.sort_id == 3

    def test_sort_id_reset(self):
        _ = Card()
        _ = Card()
        reset_sort_id_counter()
        c = Card()
        assert c.sort_id == 1

    def test_defaults(self):
        c = Card()
        assert c.base is None
        assert c.center_key == "c_base"
        assert c.card_key is None
        assert c.ability == {}
        assert c.edition is None
        assert c.seal is None
        assert c.debuff is False
        assert c.playing_card is None
        assert c.facing == "front"
        assert c.cost == 0
        assert c.sell_cost == 0
        assert c.eternal is False
        assert c.perishable is False
        assert c.perish_tally == 5
        assert c.rental is False


# ============================================================================
# Playing card creation
# ============================================================================

class TestPlayingCard:
    """Creating a playing card with base fields."""

    def test_set_base(self):
        c = Card()
        proto = PLAYING_CARDS["H_A"]
        c.set_base("H_A", proto.suit, proto.rank)
        assert c.card_key == "H_A"
        assert c.base is not None
        assert c.base.suit is Suit.HEARTS
        assert c.base.rank is Rank.ACE
        assert c.base.id == 14
        assert c.base.nominal == 11

    def test_chip_bonus_from_base(self):
        c = Card()
        c.set_base("S_7", "Spades", "7")
        c.set_ability({"name": "Default Base", "set": "Default", "config": {}})
        assert c.get_chip_bonus() == 7

    def test_chip_bonus_with_bonus_enhancement(self):
        c = Card()
        c.set_base("D_5", "Diamonds", "5")
        # Bonus Card adds 30 chips via config.bonus
        c.set_ability({
            "name": "Bonus", "set": "Enhanced", "effect": "Bonus Card",
            "config": {"bonus": 30},
        })
        assert c.get_chip_bonus() == 5 + 30  # nominal + bonus

    def test_chip_bonus_debuffed_is_zero(self):
        c = Card()
        c.set_base("H_A", "Hearts", "Ace")
        c.set_ability({"name": "Default Base", "set": "Default", "config": {}})
        c.set_debuff(True)
        assert c.get_chip_bonus() == 0

    def test_perma_bonus_preserved(self):
        """perma_bonus survives set_ability calls (Hiker joker effect)."""
        c = Card()
        c.set_base("S_5", "Spades", "5")
        c.set_ability({"name": "Default Base", "set": "Default", "config": {}})
        c.ability["perma_bonus"] = 15  # as if Hiker added it
        # Re-calling set_ability should preserve perma_bonus
        c.set_ability({"name": "Default Base", "set": "Default", "config": {}})
        assert c.ability["perma_bonus"] == 15
        assert c.get_chip_bonus() == 5 + 15


# ============================================================================
# Joker creation
# ============================================================================

class TestJokerCard:
    """Creating a joker card from prototype data."""

    def test_basic_joker(self):
        proto = JOKERS["j_joker"]
        c = Card()
        c.set_ability({
            "key": proto.key, "name": proto.name, "set": "Joker",
            "effect": proto.effect, "order": proto.order,
            "cost": proto.cost, "config": proto.config,
        })
        assert c.ability["name"] == "Joker"
        assert c.ability["set"] == "Joker"
        assert c.ability["effect"] == "Mult"
        assert c.ability["mult"] == 4
        assert c.ability["x_mult"] == 1  # default, no Xmult in config
        assert c.center_key == "j_joker"
        assert c.base is None  # jokers don't have a base

    def test_greedy_joker_nested_extra(self):
        proto = JOKERS["j_greedy_joker"]
        c = Card()
        c.set_ability({
            "key": proto.key, "name": proto.name, "set": "Joker",
            "effect": proto.effect, "order": proto.order,
            "cost": proto.cost, "config": proto.config,
        })
        assert c.ability["extra"]["s_mult"] == 3
        assert c.ability["extra"]["suit"] == "Diamonds"

    def test_extra_is_deep_copied(self):
        """Mutating a card's extra must not affect the prototype."""
        proto = JOKERS["j_greedy_joker"]
        c = Card()
        c.set_ability({
            "key": proto.key, "name": proto.name, "set": "Joker",
            "config": proto.config,
        })
        c.ability["extra"]["s_mult"] = 999
        assert proto.config["extra"]["s_mult"] == 3  # unchanged

    def test_loyalty_card_extra(self):
        proto = JOKERS["j_loyalty_card"]
        c = Card()
        c.set_ability({
            "key": proto.key, "name": proto.name, "set": "Joker",
            "config": proto.config,
        })
        # Xmult is in config.extra, not config top level
        # The ability.x_mult defaults to 1 (no top-level Xmult)
        assert c.ability["x_mult"] == 1
        # The actual Xmult is accessed via ability.extra during scoring
        assert c.ability["extra"]["Xmult"] == 4
        assert c.ability["extra"]["every"] == 5


# ============================================================================
# Consumable creation
# ============================================================================

class TestConsumableCard:
    """Creating a tarot/planet card."""

    def test_tarot_with_consumeable_config(self):
        proto = TAROTS["c_magician"]
        c = Card()
        c.set_ability({
            "key": proto.key, "name": proto.name, "set": "Tarot",
            "effect": proto.effect, "config": proto.config,
            "consumeable": True,
        })
        assert c.ability["name"] == "The Magician"
        assert c.ability["consumeable"] == proto.config
        assert c.ability["consumeable"]["mod_conv"] == "m_lucky"


# ============================================================================
# Card methods
# ============================================================================

class TestCardMethods:
    """Card scoring and identity methods."""

    def test_is_face_jack(self):
        c = Card()
        c.set_base("H_J", "Hearts", "Jack")
        assert c.is_face() is True

    def test_is_face_ten(self):
        c = Card()
        c.set_base("S_T", "Spades", "10")
        assert c.is_face() is False

    def test_is_face_ace(self):
        c = Card()
        c.set_base("D_A", "Diamonds", "Ace")
        assert c.is_face() is False  # Ace is NOT a face card

    def test_is_face_debuffed(self):
        c = Card()
        c.set_base("H_K", "Hearts", "King")
        c.debuff = True
        assert c.is_face() is False

    def test_get_id(self):
        c = Card()
        c.set_base("S_A", "Spades", "Ace")
        assert c.get_id() == 14

    def test_get_id_stone_card(self):
        c = Card()
        c.set_base("H_5", "Hearts", "5")
        c.set_ability({
            "name": "Stone Card", "set": "Enhanced",
            "effect": "Stone Card", "config": {},
        })
        assert c.get_id() == -1  # Stone cards have random negative id

    def test_get_chip_mult(self):
        c = Card()
        c.set_ability({
            "name": "Mult Card", "set": "Enhanced",
            "effect": "Mult Card", "config": {"mult": 4},
        })
        assert c.get_chip_mult() == 4

    def test_get_chip_x_mult_glass(self):
        c = Card()
        c.set_ability({
            "name": "Glass Card", "set": "Enhanced",
            "effect": "Glass Card", "config": {"Xmult": 2},
        })
        assert c.get_chip_x_mult() == 2

    def test_get_chip_x_mult_default(self):
        c = Card()
        c.set_ability({"name": "Default", "set": "Default", "config": {}})
        assert c.get_chip_x_mult() == 0  # x_mult=1 treated as "no bonus"

    def test_stone_card_chip_bonus(self):
        """Stone Card: uses bonus only, not base nominal."""
        c = Card()
        c.set_base("H_A", "Hearts", "Ace")  # Ace nominal = 11
        c.set_ability({
            "name": "Stone Card", "set": "Enhanced",
            "effect": "Stone Card", "config": {"bonus": 50},
        })
        # Stone Card ignores base nominal, uses only bonus + perma_bonus
        assert c.get_chip_bonus() == 50

    def test_repr_playing_card(self):
        c = Card()
        c.set_base("S_A", "Spades", "Ace")
        assert "Ace" in repr(c) and "Spades" in repr(c)

    def test_repr_joker(self):
        c = Card()
        c.set_ability({"key": "j_joker", "name": "Joker", "set": "Joker", "config": {}})
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
        assert c.perishable is True
        assert c.perish_tally == 5

    def test_set_rental(self):
        c = Card()
        c.set_rental(True)
        assert c.rental is True

    def test_set_edition(self):
        c = Card()
        c.set_edition({"foil": True})
        assert c.edition == {"foil": True}

    def test_set_seal(self):
        c = Card()
        c.set_seal("Red")
        assert c.seal == "Red"

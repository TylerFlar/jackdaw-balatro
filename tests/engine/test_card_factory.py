"""Tests for card factory functions.

Verifies card creation from prototypes, control dicts, and modifier
application for all card types.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.card import reset_sort_id_counter
from jackdaw.engine.card_factory import (
    RANK_LETTER,
    SUIT_LETTER,
    card_from_control,
    create_consumable,
    create_joker,
    create_playing_card,
    create_voucher,
)
from jackdaw.engine.data.enums import Rank, Suit


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


# ============================================================================
# create_playing_card
# ============================================================================

class TestCreatePlayingCard:
    def test_ace_of_spades(self):
        c = create_playing_card(Suit.SPADES, Rank.ACE)
        assert c.base is not None
        assert c.base.suit is Suit.SPADES
        assert c.base.rank is Rank.ACE
        assert c.base.nominal == 11
        assert c.base.id == 14
        assert c.card_key == "S_A"
        assert c.center_key == "c_base"
        assert c.ability["name"] == "Default Base"

    def test_two_of_hearts(self):
        c = create_playing_card(Suit.HEARTS, Rank.TWO)
        assert c.base.nominal == 2
        assert c.card_key == "H_2"

    def test_glass_enhancement(self):
        c = create_playing_card(Suit.DIAMONDS, Rank.KING, enhancement="m_glass")
        assert c.center_key == "m_glass"
        assert c.ability["effect"] == "Glass Card"
        assert c.ability["x_mult"] == 2  # Glass Card has Xmult=2

    def test_gold_enhancement(self):
        c = create_playing_card(Suit.CLUBS, Rank.FIVE, enhancement="m_gold")
        assert c.center_key == "m_gold"
        assert c.ability["effect"] == "Gold Card"

    def test_with_edition(self):
        c = create_playing_card(Suit.HEARTS, Rank.ACE, edition={"foil": True})
        assert c.edition == {"foil": True}

    def test_with_seal(self):
        c = create_playing_card(Suit.SPADES, Rank.QUEEN, seal="Red")
        assert c.seal == "Red"

    def test_with_all_modifiers(self):
        c = create_playing_card(
            Suit.DIAMONDS, Rank.JACK,
            enhancement="m_steel",
            edition={"polychrome": True},
            seal="Gold",
        )
        assert c.center_key == "m_steel"
        assert c.edition == {"polychrome": True}
        assert c.seal == "Gold"
        assert c.base.rank is Rank.JACK

    def test_playing_card_index(self):
        c = create_playing_card(Suit.HEARTS, Rank.THREE, playing_card_index=7)
        assert c.playing_card == 7

    def test_sort_id_auto_assigned(self):
        c1 = create_playing_card(Suit.HEARTS, Rank.TWO)
        c2 = create_playing_card(Suit.HEARTS, Rank.THREE)
        assert c2.sort_id == c1.sort_id + 1

    def test_all_52_cards(self):
        """Create all 52 standard cards and verify uniqueness."""
        cards = []
        for suit in Suit:
            for rank in Rank:
                cards.append(create_playing_card(suit, rank))
        assert len(cards) == 52
        keys = {c.card_key for c in cards}
        assert len(keys) == 52


# ============================================================================
# create_joker
# ============================================================================

class TestCreateJoker:
    def test_basic_joker(self):
        c = create_joker("j_joker")
        assert c.ability["name"] == "Joker"
        assert c.ability["mult"] == 4
        assert c.ability["set"] == "Joker"
        assert c.center_key == "j_joker"
        assert c.base is None
        assert c.base_cost == 2

    def test_greedy_joker(self):
        c = create_joker("j_greedy_joker")
        assert c.ability["extra"]["s_mult"] == 3
        assert c.ability["extra"]["suit"] == "Diamonds"

    def test_with_foil_edition(self):
        c = create_joker("j_joker", edition={"foil": True})
        assert c.edition == {"foil": True}

    def test_eternal(self):
        c = create_joker("j_joker", eternal=True)
        assert c.eternal is True

    def test_perishable(self):
        c = create_joker("j_joker", perishable=True)
        assert c.perishable is True
        assert c.perish_tally == 5

    def test_rental(self):
        c = create_joker("j_joker", rental=True)
        assert c.rental is True

    def test_all_stickers(self):
        c = create_joker(
            "j_joker",
            edition={"negative": True},
            eternal=True,
            rental=True,
        )
        assert c.eternal is True
        assert c.rental is True
        assert c.edition == {"negative": True}

    def test_ice_cream_extra(self):
        c = create_joker("j_ice_cream")
        assert c.ability["extra"]["chips"] == 100

    def test_loyalty_card_post_init(self):
        c = create_joker("j_loyalty_card")
        assert c.ability["loyalty_remaining"] == 5

    def test_hands_played_at_create(self):
        c = create_joker("j_joker", hands_played=42)
        assert c.ability["hands_played_at_create"] == 42


# ============================================================================
# create_consumable
# ============================================================================

class TestCreateConsumable:
    def test_tarot(self):
        c = create_consumable("c_magician")
        assert c.ability["name"] == "The Magician"
        assert c.ability["set"] == "Tarot"
        assert c.center_key == "c_magician"

    def test_planet(self):
        c = create_consumable("c_pluto")
        assert c.ability["name"] == "Pluto"
        assert c.ability["set"] == "Planet"

    def test_spectral(self):
        c = create_consumable("c_aura")
        assert c.ability["name"] == "Aura"
        assert c.ability["set"] == "Spectral"


# ============================================================================
# create_voucher
# ============================================================================

class TestCreateVoucher:
    def test_overstock(self):
        c = create_voucher("v_overstock_norm")
        assert c.ability["name"] == "Overstock"
        assert c.ability["set"] == "Voucher"
        assert c.center_key == "v_overstock_norm"
        assert c.base_cost == 10


# ============================================================================
# card_from_control
# ============================================================================

class TestCardFromControl:
    def test_basic_control(self):
        """Simple card: suit + rank only."""
        c = card_from_control({"s": "S", "r": "A"})
        assert c.base.suit is Suit.SPADES
        assert c.base.rank is Rank.ACE
        assert c.center_key == "c_base"
        assert c.edition is None
        assert c.seal is None

    def test_full_control(self):
        """All fields: enhancement, edition, seal."""
        c = card_from_control({
            "s": "H", "r": "K",
            "e": "m_gold", "d": "holo", "g": "Red",
        })
        assert c.base.suit is Suit.HEARTS
        assert c.base.rank is Rank.KING
        assert c.center_key == "m_gold"
        assert c.ability["effect"] == "Gold Card"
        assert c.edition == {"holo": True}
        assert c.seal == "Red"

    def test_glass_enhancement(self):
        c = card_from_control({"s": "D", "r": "5", "e": "m_glass"})
        assert c.center_key == "m_glass"
        assert c.ability["x_mult"] == 2

    def test_ten_rank(self):
        c = card_from_control({"s": "C", "r": "T"})
        assert c.base.rank is Rank.TEN
        assert c.card_key == "C_T"

    def test_playing_card_index(self):
        c = card_from_control({"s": "H", "r": "2"}, playing_card_index=1)
        assert c.playing_card == 1

    def test_no_enhancement_defaults_to_base(self):
        c = card_from_control({"s": "S", "r": "J"})
        assert c.center_key == "c_base"

    def test_explicit_none_enhancement(self):
        c = card_from_control({"s": "S", "r": "J", "e": None})
        assert c.center_key == "c_base"

    def test_all_suit_letters(self):
        for letter, expected_suit in SUIT_LETTER.items():
            c = card_from_control({"s": letter, "r": "A"})
            assert c.base.suit is expected_suit

    def test_all_rank_letters(self):
        for letter, expected_rank in RANK_LETTER.items():
            c = card_from_control({"s": "H", "r": letter})
            assert c.base.rank is expected_rank


# ============================================================================
# Deep copy isolation across factory
# ============================================================================

class TestFactoryIsolation:
    def test_two_jokers_from_same_key(self):
        c1 = create_joker("j_ice_cream")
        c2 = create_joker("j_ice_cream")
        c1.ability["extra"]["chips"] = 0
        assert c2.ability["extra"]["chips"] == 100

    def test_two_playing_cards_same_enhancement(self):
        c1 = create_playing_card(Suit.HEARTS, Rank.ACE, enhancement="m_bonus")
        c2 = create_playing_card(Suit.SPADES, Rank.ACE, enhancement="m_bonus")
        c1.ability["bonus"] = 999
        assert c2.ability["bonus"] == 30  # original value

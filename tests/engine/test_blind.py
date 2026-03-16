"""Tests for the Blind class.

Verifies chip target calculation, blind type identification, boss state,
reward dollars, and scaling across antes and stake levels.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.blind import Blind
from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.data.blind_scaling import get_blind_amount

# ============================================================================
# Small Blind
# ============================================================================

class TestSmallBlind:
    def test_ante_1_scaling_1(self):
        b = Blind.create("bl_small", ante=1)
        assert b.chips == 300  # 300 × 1.0
        assert b.mult == 1.0
        assert b.boss is False
        assert b.name == "Small Blind"
        assert b.dollars == 3

    def test_ante_3_scaling_1(self):
        b = Blind.create("bl_small", ante=3)
        assert b.chips == 2000  # 2000 × 1.0

    def test_ante_8_scaling_3(self):
        b = Blind.create("bl_small", ante=8, scaling=3)
        assert b.chips == 200_000  # 200000 × 1.0

    def test_get_type(self):
        b = Blind.create("bl_small", ante=1)
        assert b.get_type() == "Small"

    def test_no_boss(self):
        b = Blind.create("bl_small", ante=1)
        assert b.boss is False
        assert b.disabled is False


# ============================================================================
# Big Blind
# ============================================================================

class TestBigBlind:
    def test_ante_1_scaling_1(self):
        b = Blind.create("bl_big", ante=1)
        assert b.chips == 450  # 300 × 1.5
        assert b.mult == 1.5
        assert b.dollars == 4

    def test_ante_5_scaling_2(self):
        b = Blind.create("bl_big", ante=5, scaling=2)
        assert b.chips == 30_000  # 20000 × 1.5

    def test_get_type(self):
        b = Blind.create("bl_big", ante=1)
        assert b.get_type() == "Big"


# ============================================================================
# Boss Blinds
# ============================================================================

class TestBossBlind:
    def test_the_hook(self):
        b = Blind.create("bl_hook", ante=1)
        assert b.chips == 600  # 300 × 2.0
        assert b.mult == 2.0
        assert b.boss is True
        assert b.name == "The Hook"
        assert b.dollars == 5

    def test_the_wall_high_mult(self):
        """The Wall has mult=4 (double the normal boss)."""
        b = Blind.create("bl_wall", ante=1)
        assert b.chips == 1200  # 300 × 4.0
        assert b.mult == 4.0

    def test_the_needle_mult_1(self):
        """The Needle has mult=1 (unusually low for a boss)."""
        b = Blind.create("bl_needle", ante=1)
        assert b.chips == 300  # 300 × 1.0
        assert b.mult == 1.0
        assert b.boss is True

    def test_violet_vessel_mult_6(self):
        """Violet Vessel (showdown) has mult=6."""
        b = Blind.create("bl_final_vessel", ante=1)
        assert b.chips == 1800  # 300 × 6.0
        assert b.dollars == 8

    def test_get_type_boss(self):
        b = Blind.create("bl_hook", ante=1)
        assert b.get_type() == "Boss"


# ============================================================================
# Chip target calculation across antes and scaling
# ============================================================================

class TestChipTargets:
    @pytest.mark.parametrize("ante", range(1, 9))
    def test_small_blind_all_antes_scaling_1(self, ante: int):
        b = Blind.create("bl_small", ante=ante, scaling=1)
        expected = get_blind_amount(ante, scaling=1) * 1  # mult=1.0
        assert b.chips == expected

    @pytest.mark.parametrize("ante", range(1, 9))
    def test_big_blind_all_antes_scaling_1(self, ante: int):
        b = Blind.create("bl_big", ante=ante, scaling=1)
        expected = int(get_blind_amount(ante, scaling=1) * 1.5)
        assert b.chips == expected

    @pytest.mark.parametrize("ante", range(1, 9))
    def test_boss_blind_all_antes_scaling_1(self, ante: int):
        b = Blind.create("bl_hook", ante=ante, scaling=1)
        expected = get_blind_amount(ante, scaling=1) * 2  # mult=2.0
        assert b.chips == expected

    def test_scaling_2_ante_5(self):
        b = Blind.create("bl_hook", ante=5, scaling=2)
        assert b.chips == 40_000  # 20000 × 2.0

    def test_scaling_3_ante_8(self):
        b = Blind.create("bl_hook", ante=8, scaling=3)
        assert b.chips == 400_000  # 200000 × 2.0


# ============================================================================
# Plasma Deck (ante_scaling = 2.0)
# ============================================================================

class TestPlasmaScaling:
    def test_small_blind_plasma(self):
        b = Blind.create("bl_small", ante=1, ante_scaling=2.0)
        assert b.chips == 600  # 300 × 1.0 × 2.0

    def test_boss_blind_plasma(self):
        b = Blind.create("bl_hook", ante=1, ante_scaling=2.0)
        assert b.chips == 1200  # 300 × 2.0 × 2.0

    def test_ante_8_boss_plasma_scaling_3(self):
        b = Blind.create("bl_hook", ante=8, scaling=3, ante_scaling=2.0)
        assert b.chips == 800_000  # 200000 × 2.0 × 2.0


# ============================================================================
# No blind reward (stake modifier)
# ============================================================================

class TestNoBlindReward:
    def test_small_blind_no_reward(self):
        b = Blind.create("bl_small", ante=1, no_blind_reward=True)
        assert b.dollars == 0

    def test_boss_blind_no_reward(self):
        b = Blind.create("bl_hook", ante=1, no_blind_reward=True)
        assert b.dollars == 0

    def test_normal_reward_preserved(self):
        b = Blind.create("bl_small", ante=1, no_blind_reward=False)
        assert b.dollars == 3


# ============================================================================
# Debuff config
# ============================================================================

class TestDebuffConfig:
    def test_suit_debuff(self):
        """The Club debuffs all Clubs."""
        b = Blind.create("bl_club", ante=1)
        assert b.debuff_config.get("suit") == "Clubs"

    def test_the_goad_debuffs_spades(self):
        b = Blind.create("bl_goad", ante=1)
        assert b.debuff_config.get("suit") == "Spades"

    def test_the_head_debuffs_hearts(self):
        b = Blind.create("bl_head", ante=1)
        assert b.debuff_config.get("suit") == "Hearts"

    def test_the_window_debuffs_diamonds(self):
        b = Blind.create("bl_window", ante=1)
        assert b.debuff_config.get("suit") == "Diamonds"

    def test_the_plant_debuffs_face(self):
        b = Blind.create("bl_plant", ante=1)
        assert b.debuff_config.get("is_face") == "face"

    def test_the_psychic_hand_size(self):
        b = Blind.create("bl_psychic", ante=1)
        assert b.debuff_config.get("h_size_ge") == 5

    def test_small_blind_no_debuff(self):
        b = Blind.create("bl_small", ante=1)
        assert b.debuff_config == {}


# ============================================================================
# Empty blind
# ============================================================================

class TestEmptyBlind:
    def test_empty(self):
        b = Blind.empty()
        assert b.key == ""
        assert b.chips == 0
        assert b.boss is False
        assert b.get_type() == ""


# ============================================================================
# Boss state initialization
# ============================================================================

class TestBossState:
    def test_initial_state(self):
        b = Blind.create("bl_hook", ante=1)
        assert b.disabled is False
        assert b.triggered is False
        assert b.hands_used == {}
        assert b.only_hand is None
        assert b.discards_sub is None
        assert b.hands_sub is None

    def test_disable(self):
        b = Blind.create("bl_hook", ante=1)
        b.disabled = True
        assert b.disabled is True

    def test_repr(self):
        b = Blind.create("bl_hook", ante=1)
        r = repr(b)
        assert "The Hook" in r


# ============================================================================
# debuff_card
# ============================================================================

def _card(suit: str, rank: str, enhancement: str = "c_base") -> Card:
    sl = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}
    rl = {
        "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7",
        "8": "8", "9": "9", "10": "T", "Jack": "J", "Queen": "Q",
        "King": "K", "Ace": "A",
    }
    c = Card()
    c.set_base(f"{sl[suit]}_{rl[rank]}", suit, rank)
    c.set_ability(enhancement)
    return c


class TestDebuffCardSuitBlinds:
    """The Goad/Head/Club/Window debuff specific suits."""

    def test_the_goad_debuffs_spades(self):
        reset_sort_id_counter()
        b = Blind.create("bl_goad", ante=1)
        c = _card("Spades", "5")
        b.debuff_card(c)
        assert c.debuff is True

    def test_the_goad_spares_hearts(self):
        reset_sort_id_counter()
        b = Blind.create("bl_goad", ante=1)
        c = _card("Hearts", "5")
        b.debuff_card(c)
        assert c.debuff is False

    def test_the_head_debuffs_hearts(self):
        reset_sort_id_counter()
        b = Blind.create("bl_head", ante=1)
        c = _card("Hearts", "Ace")
        b.debuff_card(c)
        assert c.debuff is True

    def test_the_head_spares_clubs(self):
        reset_sort_id_counter()
        b = Blind.create("bl_head", ante=1)
        c = _card("Clubs", "Ace")
        b.debuff_card(c)
        assert c.debuff is False

    def test_the_club_debuffs_clubs(self):
        reset_sort_id_counter()
        b = Blind.create("bl_club", ante=1)
        c = _card("Clubs", "King")
        b.debuff_card(c)
        assert c.debuff is True

    def test_the_window_debuffs_diamonds(self):
        reset_sort_id_counter()
        b = Blind.create("bl_window", ante=1)
        c = _card("Diamonds", "7")
        b.debuff_card(c)
        assert c.debuff is True

    def test_the_window_spares_spades(self):
        reset_sort_id_counter()
        b = Blind.create("bl_window", ante=1)
        c = _card("Spades", "7")
        b.debuff_card(c)
        assert c.debuff is False


class TestDebuffCardPlant:
    """The Plant debuffs face cards (J/Q/K)."""

    def test_debuffs_king(self):
        reset_sort_id_counter()
        b = Blind.create("bl_plant", ante=1)
        c = _card("Hearts", "King")
        b.debuff_card(c)
        assert c.debuff is True

    def test_debuffs_jack(self):
        reset_sort_id_counter()
        b = Blind.create("bl_plant", ante=1)
        c = _card("Spades", "Jack")
        b.debuff_card(c)
        assert c.debuff is True

    def test_spares_number_card(self):
        reset_sort_id_counter()
        b = Blind.create("bl_plant", ante=1)
        c = _card("Hearts", "7")
        b.debuff_card(c)
        assert c.debuff is False

    def test_spares_ace(self):
        reset_sort_id_counter()
        b = Blind.create("bl_plant", ante=1)
        c = _card("Hearts", "Ace")
        b.debuff_card(c)
        assert c.debuff is False

    def test_with_pareidolia(self):
        """With Pareidolia, ALL cards are face → all debuffed by The Plant."""
        reset_sort_id_counter()
        b = Blind.create("bl_plant", ante=1)
        c = _card("Hearts", "5")
        b.debuff_card(c, pareidolia=True)
        assert c.debuff is True


class TestDebuffCardPillar:
    """The Pillar debuffs cards played earlier this ante."""

    def test_debuffs_played_card(self):
        reset_sort_id_counter()
        b = Blind.create("bl_pillar", ante=1)
        c = _card("Hearts", "5")
        c.ability["played_this_ante"] = True
        b.debuff_card(c)
        assert c.debuff is True

    def test_spares_unplayed_card(self):
        reset_sort_id_counter()
        b = Blind.create("bl_pillar", ante=1)
        c = _card("Hearts", "5")
        # played_this_ante not set → falsy
        b.debuff_card(c)
        assert c.debuff is False


class TestDebuffCardVerdantLeaf:
    """Verdant Leaf debuffs ALL non-joker cards unconditionally."""

    def test_debuffs_all(self):
        reset_sort_id_counter()
        b = Blind.create("bl_final_leaf", ante=1)
        c = _card("Hearts", "5")
        b.debuff_card(c)
        assert c.debuff is True

    def test_spares_joker_area(self):
        reset_sort_id_counter()
        b = Blind.create("bl_final_leaf", ante=1)
        c = _card("Hearts", "5")
        b.debuff_card(c, is_joker_area=True)
        assert c.debuff is False


class TestDebuffCardDisabled:
    """Disabled boss blinds debuff nothing."""

    def test_disabled_suit_blind(self):
        reset_sort_id_counter()
        b = Blind.create("bl_goad", ante=1)
        b.disabled = True
        c = _card("Spades", "5")
        b.debuff_card(c)
        assert c.debuff is False

    def test_disabled_plant(self):
        reset_sort_id_counter()
        b = Blind.create("bl_plant", ante=1)
        b.disabled = True
        c = _card("Hearts", "King")
        b.debuff_card(c)
        assert c.debuff is False

    def test_disabled_verdant_leaf(self):
        reset_sort_id_counter()
        b = Blind.create("bl_final_leaf", ante=1)
        b.disabled = True
        c = _card("Hearts", "5")
        b.debuff_card(c)
        assert c.debuff is False


class TestDebuffCardNonDebuffBlinds:
    """Bosses without debuff config don't debuff cards."""

    def test_the_hook(self):
        reset_sort_id_counter()
        b = Blind.create("bl_hook", ante=1)
        c = _card("Hearts", "5")
        b.debuff_card(c)
        assert c.debuff is False

    def test_the_wall(self):
        reset_sort_id_counter()
        b = Blind.create("bl_wall", ante=1)
        c = _card("Hearts", "King")
        b.debuff_card(c)
        assert c.debuff is False

    def test_small_blind(self):
        reset_sort_id_counter()
        b = Blind.create("bl_small", ante=1)
        c = _card("Spades", "Ace")
        b.debuff_card(c)
        assert c.debuff is False


class TestDebuffCardClearsDebuff:
    """debuff_card should CLEAR debuff if the card doesn't match."""

    def test_previously_debuffed_cleared(self):
        reset_sort_id_counter()
        b = Blind.create("bl_goad", ante=1)
        c = _card("Hearts", "5")
        c.debuff = True  # previously debuffed
        b.debuff_card(c)
        assert c.debuff is False  # Hearts not debuffed by The Goad

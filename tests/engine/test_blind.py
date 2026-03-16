"""Tests for the Blind class.

Verifies chip target calculation, blind type identification, boss state,
reward dollars, and scaling across antes and stake levels.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.blind import Blind, get_ante_blinds, get_new_boss
from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.data.blind_scaling import get_blind_amount
from jackdaw.engine.data.prototypes import BLINDS as ALL_BLINDS
from jackdaw.engine.rng import PseudoRandom

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


# ============================================================================
# debuff_hand
# ============================================================================

class TestDebuffHandEye:
    """The Eye: each hand type can only be used once."""

    def test_first_use_allowed(self):
        b = Blind.create("bl_eye", ante=1)
        assert b.debuff_hand([], {}, "Pair") is False

    def test_second_use_blocked(self):
        b = Blind.create("bl_eye", ante=1)
        b.debuff_hand([], {}, "Pair")
        assert b.debuff_hand([], {}, "Pair") is True

    def test_different_type_allowed(self):
        b = Blind.create("bl_eye", ante=1)
        b.debuff_hand([], {}, "Pair")
        assert b.debuff_hand([], {}, "Flush") is False

    def test_check_mode_no_register(self):
        """check=True: previews without registering the hand type."""
        b = Blind.create("bl_eye", ante=1)
        b.debuff_hand([], {}, "Pair", check=True)
        # Pair should NOT be registered
        assert b.debuff_hand([], {}, "Pair") is False  # first real use

    def test_triggered_on_block(self):
        b = Blind.create("bl_eye", ante=1)
        b.debuff_hand([], {}, "Pair")
        b.debuff_hand([], {}, "Pair")
        assert b.triggered is True

    def test_disabled(self):
        b = Blind.create("bl_eye", ante=1)
        b.disabled = True
        b.debuff_hand([], {}, "Pair")
        assert b.debuff_hand([], {}, "Pair") is False


class TestDebuffHandMouth:
    """The Mouth: only one hand type allowed per round."""

    def test_first_hand_allowed(self):
        b = Blind.create("bl_mouth", ante=1)
        assert b.debuff_hand([], {}, "Flush") is False

    def test_same_type_allowed(self):
        b = Blind.create("bl_mouth", ante=1)
        b.debuff_hand([], {}, "Flush")
        assert b.debuff_hand([], {}, "Flush") is False

    def test_different_type_blocked(self):
        b = Blind.create("bl_mouth", ante=1)
        b.debuff_hand([], {}, "Flush")
        assert b.debuff_hand([], {}, "Pair") is True

    def test_check_mode_no_lock(self):
        b = Blind.create("bl_mouth", ante=1)
        b.debuff_hand([], {}, "Flush", check=True)
        assert b.only_hand is None
        assert b.debuff_hand([], {}, "Pair") is False  # not locked yet


class TestDebuffHandPsychic:
    """The Psychic: must play at least 5 cards (h_size_ge=5)."""

    def test_5_cards_allowed(self):
        b = Blind.create("bl_psychic", ante=1)
        cards = [_card("Hearts", str(i)) for i in range(2, 7)]
        assert b.debuff_hand(cards, {}, "Straight") is False

    def test_4_cards_blocked(self):
        b = Blind.create("bl_psychic", ante=1)
        cards = [_card("Hearts", str(i)) for i in range(2, 6)]
        assert b.debuff_hand(cards, {}, "Pair") is True

    def test_3_cards_blocked(self):
        b = Blind.create("bl_psychic", ante=1)
        cards = [_card("Hearts", "5"), _card("Spades", "5"), _card("Clubs", "5")]
        assert b.debuff_hand(cards, {}, "Three of a Kind") is True

    def test_1_card_blocked(self):
        b = Blind.create("bl_psychic", ante=1)
        cards = [_card("Hearts", "Ace")]
        assert b.debuff_hand(cards, {}, "High Card") is True


class TestDebuffHandNonBlocking:
    """Bosses that don't block hands via debuff_hand."""

    @pytest.mark.parametrize("key", [
        "bl_hook", "bl_wall", "bl_flint", "bl_tooth", "bl_water",
        "bl_needle", "bl_serpent", "bl_manacle", "bl_fish",
        "bl_wheel", "bl_house", "bl_mark", "bl_pillar",
    ])
    def test_non_blocking_bosses(self, key: str):
        b = Blind.create(key, ante=1)
        assert b.debuff_hand([], {}, "Pair") is False

    def test_small_blind(self):
        b = Blind.create("bl_small", ante=1)
        assert b.debuff_hand([], {}, "Pair") is False


# ============================================================================
# modify_hand
# ============================================================================

class TestModifyHand:
    """Blind.modify_hand: The Flint halves chips and mult."""

    def test_the_flint(self):
        b = Blind.create("bl_flint", ante=1)
        mult, chips, modified = b.modify_hand(20.0, 100)
        assert modified is True
        # floor(20 * 0.5 + 0.5) = 10, floor(100 * 0.5 + 0.5) = 50
        assert mult == 10.0
        assert chips == 50

    def test_the_flint_odd_values(self):
        b = Blind.create("bl_flint", ante=1)
        mult, chips, modified = b.modify_hand(7.0, 15)
        # floor(7 * 0.5 + 0.5) = 4, floor(15 * 0.5 + 0.5) = 8
        assert mult == 4.0
        assert chips == 8

    def test_the_flint_minimum_mult_1(self):
        b = Blind.create("bl_flint", ante=1)
        mult, chips, _ = b.modify_hand(1.0, 2)
        assert mult >= 1.0

    def test_the_flint_minimum_chips_0(self):
        b = Blind.create("bl_flint", ante=1)
        mult, chips, _ = b.modify_hand(1.0, 0)
        assert chips >= 0

    def test_the_flint_disabled(self):
        b = Blind.create("bl_flint", ante=1)
        b.disabled = True
        mult, chips, modified = b.modify_hand(20.0, 100)
        assert modified is False
        assert mult == 20.0
        assert chips == 100

    def test_non_flint_no_modify(self):
        b = Blind.create("bl_hook", ante=1)
        mult, chips, modified = b.modify_hand(20.0, 100)
        assert modified is False
        assert mult == 20.0
        assert chips == 100

    def test_small_blind_no_modify(self):
        b = Blind.create("bl_small", ante=1)
        mult, chips, modified = b.modify_hand(20.0, 100)
        assert modified is False


# ============================================================================
# press_play
# ============================================================================


class TestPressPlayHook:
    """The Hook: discard 2 random cards from hand."""

    def test_returns_discard_indices(self):
        reset_sort_id_counter()
        b = Blind.create("bl_hook", ante=1)
        hand = [_card("Hearts", str(i)) for i in range(2, 10)]
        rng = PseudoRandom("TESTSEED")
        result = b.press_play(hand, [], rng=rng)
        assert "discard_indices" in result
        assert len(result["discard_indices"]) == 2
        assert b.triggered is True

    def test_indices_are_valid(self):
        reset_sort_id_counter()
        b = Blind.create("bl_hook", ante=1)
        hand = [_card("Hearts", str(i)) for i in range(2, 7)]
        rng = PseudoRandom("TESTSEED")
        result = b.press_play(hand, [], rng=rng)
        for idx in result["discard_indices"]:
            assert 0 <= idx < len(hand)

    def test_indices_are_unique(self):
        reset_sort_id_counter()
        b = Blind.create("bl_hook", ante=1)
        hand = [_card("Hearts", str(i)) for i in range(2, 10)]
        rng = PseudoRandom("TESTSEED")
        result = b.press_play(hand, [], rng=rng)
        assert len(set(result["discard_indices"])) == 2

    def test_disabled(self):
        reset_sort_id_counter()
        b = Blind.create("bl_hook", ante=1)
        b.disabled = True
        result = b.press_play([], [], rng=PseudoRandom("X"))
        assert result == {}


class TestPressPlayTooth:
    """The Tooth: lose $1 per card played."""

    def test_money_cost(self):
        reset_sort_id_counter()
        b = Blind.create("bl_tooth", ante=1)
        played = [_card("Hearts", "5"), _card("Spades", "King")]
        result = b.press_play([], played)
        assert result["money_cost"] == 2
        assert b.triggered is True


class TestPressPlayOthers:
    """Bosses with no press_play effect."""

    def test_small_blind(self):
        b = Blind.create("bl_small", ante=1)
        assert b.press_play([], []) == {}

    def test_the_wall(self):
        b = Blind.create("bl_wall", ante=1)
        assert b.press_play([], []) == {}


# ============================================================================
# drawn_to_hand
# ============================================================================

class TestDrawnToHandBell:
    """Cerulean Bell: force-select a random card."""

    def test_returns_forced_index(self):
        reset_sort_id_counter()
        b = Blind.create("bl_final_bell", ante=1)
        hand = [_card("Hearts", "5"), _card("Spades", "King"), _card("Clubs", "Ace")]
        rng = PseudoRandom("TESTSEED")
        result = b.drawn_to_hand(hand, rng=rng)
        assert "forced_card_index" in result
        assert 0 <= result["forced_card_index"] < len(hand)

    def test_disabled(self):
        reset_sort_id_counter()
        b = Blind.create("bl_final_bell", ante=1)
        b.disabled = True
        result = b.drawn_to_hand([], rng=PseudoRandom("X"))
        assert result == {}


class TestDrawnToHandCrimsonHeart:
    """Crimson Heart: debuff a random joker."""

    def test_debuffs_one_joker(self):
        reset_sort_id_counter()
        b = Blind.create("bl_final_heart", ante=1)
        from jackdaw.engine.card_factory import create_joker
        jokers = [create_joker("j_joker"), create_joker("j_greedy_joker")]
        rng = PseudoRandom("TESTSEED")
        result = b.drawn_to_hand([], joker_cards=jokers, rng=rng)
        assert "debuffed_joker_index" in result
        # Exactly one joker should be debuffed
        debuffed = [j for j in jokers if j.debuff]
        assert len(debuffed) == 1


# ============================================================================
# stay_flipped
# ============================================================================

class TestStayFlipped:
    def test_the_house_first_hand(self):
        reset_sort_id_counter()
        b = Blind.create("bl_house", ante=1)
        c = _card("Hearts", "5")
        assert b.stay_flipped(c, hands_played=0, discards_used=0) is True

    def test_the_house_after_play(self):
        reset_sort_id_counter()
        b = Blind.create("bl_house", ante=1)
        c = _card("Hearts", "5")
        assert b.stay_flipped(c, hands_played=1, discards_used=0) is False

    def test_the_mark_face_card(self):
        reset_sort_id_counter()
        b = Blind.create("bl_mark", ante=1)
        c = _card("Hearts", "King")
        assert b.stay_flipped(c) is True

    def test_the_mark_number_card(self):
        reset_sort_id_counter()
        b = Blind.create("bl_mark", ante=1)
        c = _card("Hearts", "5")
        assert b.stay_flipped(c) is False

    def test_disabled(self):
        reset_sort_id_counter()
        b = Blind.create("bl_house", ante=1)
        b.disabled = True
        c = _card("Hearts", "5")
        assert b.stay_flipped(c, hands_played=0, discards_used=0) is False

    def test_non_flipping_blind(self):
        b = Blind.create("bl_hook", ante=1)
        c = _card("Hearts", "5")
        assert b.stay_flipped(c) is False


# ============================================================================
# disable
# ============================================================================

class TestDisable:
    def test_sets_disabled(self):
        b = Blind.create("bl_hook", ante=1)
        b.disable()
        assert b.disabled is True

    def test_the_wall_halves_chips(self):
        b = Blind.create("bl_wall", ante=1)
        original_chips = b.chips
        result = b.disable()
        assert b.chips == original_chips // 2
        assert result.get("halve_chips") is True

    def test_violet_vessel_thirds_chips(self):
        b = Blind.create("bl_final_vessel", ante=1)
        original_chips = b.chips
        b.disable()
        assert b.chips == original_chips // 3

    def test_the_water_restore_discards(self):
        b = Blind.create("bl_water", ante=1)
        b.discards_sub = 3
        result = b.disable()
        assert result["restore_discards"] == 3

    def test_the_needle_restore_hands(self):
        b = Blind.create("bl_needle", ante=1)
        b.hands_sub = 3
        result = b.disable()
        assert result["restore_hands"] == 3

    def test_the_manacle_restore_hand_size(self):
        b = Blind.create("bl_manacle", ante=1)
        result = b.disable()
        assert result["restore_hand_size"] == 1

    def test_cerulean_bell_clear_forced(self):
        b = Blind.create("bl_final_bell", ante=1)
        result = b.disable()
        assert result.get("clear_forced") is True

    def test_clears_card_debuffs(self):
        reset_sort_id_counter()
        b = Blind.create("bl_goad", ante=1)
        c = _card("Spades", "5")
        b.debuff_card(c)
        assert c.debuff is True
        b.disable(playing_cards=[c])
        assert c.debuff is False


# ============================================================================
# get_new_boss / get_ante_blinds
# ============================================================================

class TestGetNewBoss:
    """Boss blind selection matching common_events.lua:2338."""

    def test_returns_valid_boss(self):
        bosses_used = {k: 0 for k in ALL_BLINDS if ALL_BLINDS[k].boss}
        rng = PseudoRandom("TESTSEED")
        boss = get_new_boss(1, bosses_used, rng)
        assert boss in ALL_BLINDS
        assert ALL_BLINDS[boss].boss is not None

    def test_deterministic(self):
        bosses_used1 = {k: 0 for k in ALL_BLINDS if ALL_BLINDS[k].boss}
        bosses_used2 = {k: 0 for k in ALL_BLINDS if ALL_BLINDS[k].boss}
        b1 = get_new_boss(1, bosses_used1, PseudoRandom("TESTSEED"))
        b2 = get_new_boss(1, bosses_used2, PseudoRandom("TESTSEED"))
        assert b1 == b2

    def test_increments_usage(self):
        bosses_used = {k: 0 for k in ALL_BLINDS if ALL_BLINDS[k].boss}
        rng = PseudoRandom("TESTSEED")
        boss = get_new_boss(1, bosses_used, rng)
        assert bosses_used[boss] == 1

    def test_avoids_overused_bosses(self):
        """Bosses with higher usage count are excluded in favor of least-used."""
        bosses_used = {k: 0 for k in ALL_BLINDS if ALL_BLINDS[k].boss}
        rng = PseudoRandom("FIXEDSEED")

        # Select many bosses — they should spread across available options
        selected = set()
        for _ in range(20):
            boss = get_new_boss(3, bosses_used, rng)
            selected.add(boss)

        # With 20 selections, we should have used multiple different bosses
        assert len(selected) > 5

    def test_ante_1_excludes_high_min(self):
        """Bosses with boss.min > 1 shouldn't appear at ante 1."""
        bosses_used = {k: 0 for k in ALL_BLINDS if ALL_BLINDS[k].boss}
        rng = PseudoRandom("TESTSEED")

        for _ in range(50):
            boss = get_new_boss(1, bosses_used, rng)
            proto = ALL_BLINDS[boss]
            assert proto.boss["min"] <= 1, (
                f"{boss} has min={proto.boss['min']} but appeared at ante 1"
            )

    def test_showdown_at_ante_8(self):
        """At ante 8 (win_ante), only showdown blinds are eligible."""
        bosses_used = {k: 0 for k in ALL_BLINDS if ALL_BLINDS[k].boss}
        rng = PseudoRandom("TESTSEED")

        boss = get_new_boss(8, bosses_used, rng, win_ante=8)
        proto = ALL_BLINDS[boss]
        assert proto.boss.get("showdown") is True, (
            f"{boss} is not a showdown blind but was selected at ante 8"
        )

    def test_showdown_at_ante_16(self):
        """Ante 16 is also a showdown ante (16 % 8 == 0)."""
        bosses_used = {k: 0 for k in ALL_BLINDS if ALL_BLINDS[k].boss}
        rng = PseudoRandom("TESTSEED")

        boss = get_new_boss(16, bosses_used, rng, win_ante=8)
        proto = ALL_BLINDS[boss]
        assert proto.boss.get("showdown") is True

    def test_non_showdown_at_ante_7(self):
        """Ante 7 is NOT a showdown ante — no showdown blinds."""
        bosses_used = {k: 0 for k in ALL_BLINDS if ALL_BLINDS[k].boss}
        rng = PseudoRandom("TESTSEED")

        for _ in range(20):
            boss = get_new_boss(7, bosses_used, rng, win_ante=8)
            proto = ALL_BLINDS[boss]
            assert not proto.boss.get("showdown"), (
                f"{boss} is showdown but appeared at ante 7"
            )

    def test_banned_keys_excluded(self):
        bosses_used = {k: 0 for k in ALL_BLINDS if ALL_BLINDS[k].boss}
        rng = PseudoRandom("TESTSEED")
        banned = {"bl_hook": True, "bl_club": True}

        for _ in range(30):
            boss = get_new_boss(1, bosses_used, rng, banned_keys=banned)
            assert boss not in banned


class TestGetAnteBlinds:
    def test_structure(self):
        bosses_used = {k: 0 for k in ALL_BLINDS if ALL_BLINDS[k].boss}
        rng = PseudoRandom("TESTSEED")
        result = get_ante_blinds(1, bosses_used, rng)
        assert result["Small"] == "bl_small"
        assert result["Big"] == "bl_big"
        assert result["Boss"] in ALL_BLINDS

    def test_boss_is_valid(self):
        bosses_used = {k: 0 for k in ALL_BLINDS if ALL_BLINDS[k].boss}
        rng = PseudoRandom("TESTSEED")
        result = get_ante_blinds(3, bosses_used, rng)
        assert ALL_BLINDS[result["Boss"]].boss is not None

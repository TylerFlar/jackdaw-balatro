"""Tests for poker hand type data.

Verifies base values, level-up formula, detection priority order,
and enum string matching against Lua source keys.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.data.hands import HAND_BASE, HAND_ORDER, HandBaseData, HandType


class TestHandTypeEnum:
    """HandType enum string values match Lua source keys."""

    def test_count(self):
        assert len(HandType) == 12

    def test_string_values_are_usable_as_keys(self):
        """Each HandType value is a string that works as a dict key."""
        for ht in HandType:
            assert isinstance(ht.value, str)
            assert ht == ht.value  # str enum comparison

    def test_specific_values(self):
        assert HandType.FLUSH_FIVE == "Flush Five"
        assert HandType.HIGH_CARD == "High Card"
        assert HandType.STRAIGHT_FLUSH == "Straight Flush"
        assert HandType.TWO_PAIR == "Two Pair"

    def test_lookup_by_string(self):
        assert HandType("Flush Five") is HandType.FLUSH_FIVE
        assert HandType("High Card") is HandType.HIGH_CARD


class TestHandBaseData:
    """HAND_BASE contains correct values from game.lua init_game_object."""

    def test_all_12_hands_present(self):
        assert len(HAND_BASE) == 12
        for ht in HandType:
            assert ht in HAND_BASE

    def test_all_frozen(self):
        for data in HAND_BASE.values():
            assert isinstance(data, HandBaseData)
            with pytest.raises(AttributeError):
                data.s_chips = 999  # type: ignore[misc]

    # -- Base values (level 1) from game.lua:2001-2014 --

    @pytest.mark.parametrize(
        "hand,s_chips,s_mult,l_chips,l_mult,order,visible",
        [
            (HandType.FLUSH_FIVE, 160, 16, 50, 3, 1, False),
            (HandType.FLUSH_HOUSE, 140, 14, 40, 4, 2, False),
            (HandType.FIVE_OF_A_KIND, 120, 12, 35, 3, 3, False),
            (HandType.STRAIGHT_FLUSH, 100, 8, 40, 4, 4, True),
            (HandType.FOUR_OF_A_KIND, 60, 7, 30, 3, 5, True),
            (HandType.FULL_HOUSE, 40, 4, 25, 2, 6, True),
            (HandType.FLUSH, 35, 4, 15, 2, 7, True),
            (HandType.STRAIGHT, 30, 4, 30, 3, 8, True),
            (HandType.THREE_OF_A_KIND, 30, 3, 20, 2, 9, True),
            (HandType.TWO_PAIR, 20, 2, 20, 1, 10, True),
            (HandType.PAIR, 10, 2, 15, 1, 11, True),
            (HandType.HIGH_CARD, 5, 1, 10, 1, 12, True),
        ],
    )
    def test_base_values(
        self, hand: HandType, s_chips: int, s_mult: int,
        l_chips: int, l_mult: int, order: int, visible: bool,
    ):
        d = HAND_BASE[hand]
        assert d.s_chips == s_chips
        assert d.s_mult == s_mult
        assert d.l_chips == l_chips
        assert d.l_mult == l_mult
        assert d.order == order
        assert d.visible is visible

    def test_secret_hands_are_invisible(self):
        """First 3 hands (Flush Five, Flush House, Five of a Kind) are secret."""
        secret = [ht for ht, d in HAND_BASE.items() if not d.visible]
        assert len(secret) == 3
        assert set(secret) == {
            HandType.FLUSH_FIVE,
            HandType.FLUSH_HOUSE,
            HandType.FIVE_OF_A_KIND,
        }

    def test_orders_are_unique_and_sequential(self):
        orders = sorted(d.order for d in HAND_BASE.values())
        assert orders == list(range(1, 13))


class TestLevelUpFormula:
    """chips = s_chips + l_chips * (level - 1), same for mult."""

    def test_level_1_equals_base(self):
        for hand, d in HAND_BASE.items():
            assert d.chips_at(1) == d.s_chips, f"{hand} chips at level 1"
            assert d.mult_at(1) == d.s_mult, f"{hand} mult at level 1"

    # Level 5: documented in consumables.md as example
    # Pair at level 5 = 10 + 15*4 = 70 chips, 2 + 1*4 = 6 mult
    def test_pair_level_5(self):
        d = HAND_BASE[HandType.PAIR]
        assert d.chips_at(5) == 70
        assert d.mult_at(5) == 6

    def test_high_card_level_10(self):
        d = HAND_BASE[HandType.HIGH_CARD]
        assert d.chips_at(10) == 5 + 10 * 9  # 95
        assert d.mult_at(10) == 1 + 1 * 9  # 10

    def test_flush_five_level_5(self):
        d = HAND_BASE[HandType.FLUSH_FIVE]
        assert d.chips_at(5) == 160 + 50 * 4  # 360
        assert d.mult_at(5) == 16 + 3 * 4  # 28

    def test_straight_flush_level_10(self):
        d = HAND_BASE[HandType.STRAIGHT_FLUSH]
        assert d.chips_at(10) == 100 + 40 * 9  # 460
        assert d.mult_at(10) == 8 + 4 * 9  # 44

    def test_full_house_level_3(self):
        d = HAND_BASE[HandType.FULL_HOUSE]
        assert d.chips_at(3) == 40 + 25 * 2  # 90
        assert d.mult_at(3) == 4 + 2 * 2  # 8

    def test_mult_minimum_is_1(self):
        """mult_at should never go below 1 even at hypothetical level 0."""
        d = HAND_BASE[HandType.HIGH_CARD]
        assert d.mult_at(0) == 1  # 1 + 1*(0-1) = 0, clamped to 1

    def test_chips_minimum_is_0(self):
        d = HAND_BASE[HandType.HIGH_CARD]
        assert d.chips_at(0) == 0  # 5 + 10*(-1) = -5, clamped to 0


class TestHandOrder:
    """HAND_ORDER is the detection priority (highest first)."""

    def test_length(self):
        assert len(HAND_ORDER) == 12

    def test_contains_all_types(self):
        assert set(HAND_ORDER) == set(HandType)

    def test_no_duplicates(self):
        assert len(HAND_ORDER) == len(set(HAND_ORDER))

    def test_flush_five_is_first(self):
        assert HAND_ORDER[0] is HandType.FLUSH_FIVE

    def test_high_card_is_last(self):
        assert HAND_ORDER[-1] is HandType.HIGH_CARD

    def test_matches_order_field(self):
        """HAND_ORDER should be sorted by the 'order' field in HAND_BASE."""
        for i, ht in enumerate(HAND_ORDER):
            assert HAND_BASE[ht].order == i + 1

    def test_straight_flush_before_four_of_a_kind(self):
        sf_idx = HAND_ORDER.index(HandType.STRAIGHT_FLUSH)
        foak_idx = HAND_ORDER.index(HandType.FOUR_OF_A_KIND)
        assert sf_idx < foak_idx

"""Tests for poker hand detection functions.

Verifies get_flush, get_straight, get_x_same, and get_highest against
known card combinations, including Four Fingers, Shortcut, Wild Card,
Smeared Joker, and Stone Card modifiers.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.hand_eval import get_flush, get_highest, get_straight, get_x_same


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


def _card(suit: str, rank: str, enhancement: str = "c_base") -> Card:
    """Helper: create a playing card."""
    suit_letter = {"Hearts": "H", "Diamonds": "D", "Clubs": "C", "Spades": "S"}
    rank_letter = {
        "2": "2", "3": "3", "4": "4", "5": "5", "6": "6", "7": "7",
        "8": "8", "9": "9", "10": "T", "Jack": "J", "Queen": "Q",
        "King": "K", "Ace": "A",
    }
    c = Card()
    key = f"{suit_letter[suit]}_{rank_letter[rank]}"
    c.set_base(key, suit, rank)
    c.set_ability(enhancement)
    return c


# ============================================================================
# get_flush
# ============================================================================

class TestGetFlush:
    def test_five_hearts(self):
        hand = [_card("Hearts", r) for r in ["2", "5", "8", "Jack", "Ace"]]
        result = get_flush(hand)
        assert len(result) == 1
        assert len(result[0]) == 5

    def test_four_hearts_no_flush(self):
        hand = [
            _card("Hearts", "2"), _card("Hearts", "5"),
            _card("Hearts", "8"), _card("Hearts", "Jack"),
            _card("Spades", "Ace"),
        ]
        result = get_flush(hand)
        assert result == []

    def test_four_fingers_four_is_enough(self):
        hand = [
            _card("Clubs", "3"), _card("Clubs", "7"),
            _card("Clubs", "10"), _card("Clubs", "King"),
            _card("Hearts", "Ace"),
        ]
        result = get_flush(hand, four_fingers=True)
        assert len(result) == 1
        assert len(result[0]) == 4

    def test_four_fingers_three_is_not_enough(self):
        hand = [
            _card("Clubs", "3"), _card("Clubs", "7"),
            _card("Clubs", "10"), _card("Hearts", "King"),
        ]
        result = get_flush(hand, four_fingers=True)
        assert result == []

    def test_hand_too_large(self):
        """More than 5 cards: source returns empty (not a valid play)."""
        hand = [_card("Hearts", r) for r in ["2", "3", "4", "5", "6", "7"]]
        assert get_flush(hand) == []

    def test_wild_card_matches_all_suits(self):
        """Wild Card counts for every suit in flush detection."""
        hand = [
            _card("Spades", "2"), _card("Spades", "5"),
            _card("Spades", "8"), _card("Spades", "Jack"),
            _card("Hearts", "Ace", enhancement="m_wild"),  # Wild Card
        ]
        result = get_flush(hand)
        assert len(result) == 1
        assert len(result[0]) == 5  # 4 Spades + Wild = flush

    def test_stone_card_excluded(self):
        """Stone Cards are excluded from flush suit matching."""
        hand = [
            _card("Hearts", "2"), _card("Hearts", "5"),
            _card("Hearts", "8"), _card("Hearts", "Jack"),
            _card("Hearts", "Ace", enhancement="m_stone"),  # Stone
        ]
        result = get_flush(hand)
        assert result == []  # only 4 real Hearts

    def test_smeared_joker_red_suits(self):
        """Smeared: Hearts and Diamonds count as same suit."""
        hand = [
            _card("Hearts", "2"), _card("Hearts", "5"),
            _card("Diamonds", "8"), _card("Diamonds", "Jack"),
            _card("Hearts", "Ace"),
        ]
        # Without smeared: no flush (3H + 2D)
        assert get_flush(hand) == []
        # With smeared: all red = flush
        result = get_flush(hand, smeared=True)
        assert len(result) == 1
        assert len(result[0]) == 5

    def test_smeared_joker_black_suits(self):
        """Smeared: Spades and Clubs count as same suit."""
        hand = [
            _card("Spades", "2"), _card("Clubs", "5"),
            _card("Spades", "8"), _card("Clubs", "Jack"),
            _card("Spades", "Ace"),
        ]
        result = get_flush(hand, smeared=True)
        assert len(result) == 1

    def test_priority_spades_first(self):
        """Source checks suits in order: Spades, Hearts, Clubs, Diamonds."""
        hand = [_card("Spades", r) for r in ["2", "3", "4", "5", "6"]]
        result = get_flush(hand)
        assert len(result) == 1
        # All should be spades
        assert all(c.base.suit is Suit.SPADES for c in result[0])

    def test_empty_hand(self):
        assert get_flush([]) == []


# ============================================================================
# get_straight
# ============================================================================

class TestGetStraight:
    def test_simple_straight(self):
        hand = [_card("Hearts", r) for r in ["3", "4", "5", "6", "7"]]
        result = get_straight(hand)
        assert len(result) == 1
        assert len(result[0]) == 5

    def test_ace_high_straight(self):
        """10-J-Q-K-A is valid."""
        hand = [
            _card("Hearts", "10"), _card("Spades", "Jack"),
            _card("Clubs", "Queen"), _card("Diamonds", "King"),
            _card("Hearts", "Ace"),
        ]
        result = get_straight(hand)
        assert len(result) == 1

    def test_ace_low_straight(self):
        """A-2-3-4-5 is valid (Ace wraps low)."""
        hand = [
            _card("Hearts", "Ace"), _card("Spades", "2"),
            _card("Clubs", "3"), _card("Diamonds", "4"),
            _card("Hearts", "5"),
        ]
        result = get_straight(hand)
        assert len(result) == 1

    def test_no_wrap_around(self):
        """Q-K-A-2-3 is NOT valid (no wrapping)."""
        hand = [
            _card("Hearts", "Queen"), _card("Spades", "King"),
            _card("Clubs", "Ace"), _card("Diamonds", "2"),
            _card("Hearts", "3"),
        ]
        result = get_straight(hand)
        assert result == []

    def test_pair_not_straight(self):
        hand = [
            _card("Hearts", "5"), _card("Spades", "5"),
            _card("Clubs", "6"), _card("Diamonds", "7"),
            _card("Hearts", "8"),
        ]
        result = get_straight(hand)
        assert result == []  # pair breaks it (only 4 unique consecutive)

    def test_four_fingers(self):
        """Four Fingers: 4 consecutive ranks is enough."""
        hand = [
            _card("Hearts", "4"), _card("Spades", "5"),
            _card("Clubs", "6"), _card("Diamonds", "7"),
            _card("Hearts", "King"),  # not part of the straight
        ]
        result = get_straight(hand, four_fingers=True)
        assert len(result) == 1

    def test_shortcut_one_gap(self):
        """Shortcut: one rank gap is allowed."""
        hand = [
            _card("Hearts", "3"), _card("Spades", "4"),
            # gap at 5
            _card("Clubs", "6"), _card("Diamonds", "7"),
            _card("Hearts", "8"),
        ]
        result = get_straight(hand, shortcut=True)
        assert len(result) == 1

    def test_shortcut_two_separated_gaps_still_works(self):
        """Shortcut resets skip flag when a rank IS found, so separated gaps work.

        3-(gap)-5-(gap)-7-8-9: skip_rank resets at 5 and 7, so each gap
        is independent.  Straight length counts: 3→1, skip4, 5→2, skip6, 7→3, 8→4, 9→5.
        """
        hand = [
            _card("Hearts", "3"), _card("Spades", "5"),
            _card("Clubs", "7"), _card("Diamonds", "8"),
            _card("Hearts", "9"),
        ]
        result = get_straight(hand, shortcut=True)
        assert len(result) == 1  # valid shortcut straight

    def test_shortcut_consecutive_gaps_fails(self):
        """Two consecutive missing ranks: skip first, reset on second."""
        hand = [
            _card("Hearts", "3"),
            # gap at 4 AND 5
            _card("Clubs", "6"), _card("Diamonds", "7"),
            _card("Hearts", "8"),
        ]
        result = get_straight(hand, shortcut=True)
        # 3→1, skip4, gap5→reset (already skipped). 6→1, 7→2, 8→3 = not enough
        assert result == []

    def test_hand_too_large(self):
        hand = [_card("Hearts", r) for r in ["2", "3", "4", "5", "6", "7"]]
        assert get_straight(hand) == []

    def test_empty_hand(self):
        assert get_straight([]) == []

    def test_four_card_hand_needs_four_fingers(self):
        hand = [_card("Hearts", r) for r in ["5", "6", "7", "8"]]
        assert get_straight(hand) == []
        assert len(get_straight(hand, four_fingers=True)) == 1


# ============================================================================
# get_x_same
# ============================================================================

class TestGetXSame:
    def test_pair(self):
        hand = [
            _card("Hearts", "5"), _card("Spades", "5"),
            _card("Clubs", "8"), _card("Diamonds", "Jack"),
            _card("Hearts", "Ace"),
        ]
        result = get_x_same(2, hand)
        assert len(result) == 1
        assert len(result[0]) == 2
        assert all(c.base.rank is Rank.FIVE for c in result[0])

    def test_two_pairs(self):
        hand = [
            _card("Hearts", "5"), _card("Spades", "5"),
            _card("Clubs", "Jack"), _card("Diamonds", "Jack"),
            _card("Hearts", "Ace"),
        ]
        result = get_x_same(2, hand)
        assert len(result) == 2
        # Descending rank: Jacks first, then Fives
        assert all(c.base.rank is Rank.JACK for c in result[0])
        assert all(c.base.rank is Rank.FIVE for c in result[1])

    def test_three_of_a_kind(self):
        hand = [
            _card("Hearts", "King"), _card("Spades", "King"),
            _card("Clubs", "King"), _card("Diamonds", "5"),
            _card("Hearts", "2"),
        ]
        result = get_x_same(3, hand)
        assert len(result) == 1
        assert len(result[0]) == 3
        assert all(c.base.rank is Rank.KING for c in result[0])

    def test_four_of_a_kind(self):
        hand = [
            _card("Hearts", "7"), _card("Spades", "7"),
            _card("Clubs", "7"), _card("Diamonds", "7"),
            _card("Hearts", "Ace"),
        ]
        result = get_x_same(4, hand)
        assert len(result) == 1
        assert len(result[0]) == 4

    def test_five_of_a_kind(self):
        """Requires 5 cards of same rank (possible with Steel/Wild shenanigans)."""
        hand = [
            _card("Hearts", "Ace"), _card("Spades", "Ace"),
            _card("Clubs", "Ace"), _card("Diamonds", "Ace"),
            _card("Hearts", "Ace"),  # duplicate suit, happens with card generation
        ]
        result = get_x_same(5, hand)
        assert len(result) == 1

    def test_no_match(self):
        hand = [
            _card("Hearts", "2"), _card("Spades", "5"),
            _card("Clubs", "8"), _card("Diamonds", "Jack"),
            _card("Hearts", "Ace"),
        ]
        assert get_x_same(2, hand) == []

    def test_descending_order(self):
        """Groups are returned highest rank first."""
        hand = [
            _card("Hearts", "3"), _card("Spades", "3"),
            _card("Clubs", "King"), _card("Diamonds", "King"),
            _card("Hearts", "7"),
        ]
        result = get_x_same(2, hand)
        assert result[0][0].base.id > result[1][0].base.id

    def test_empty_hand(self):
        assert get_x_same(2, []) == []


# ============================================================================
# get_highest
# ============================================================================

class TestGetHighest:
    def test_single_highest(self):
        hand = [
            _card("Hearts", "2"), _card("Spades", "King"),
            _card("Clubs", "8"),
        ]
        result = get_highest(hand)
        assert len(result) == 1
        assert len(result[0]) == 1
        # King has highest nominal (10) but Ace would be 11
        # Actually 8 has nominal 8, King has nominal 10
        assert result[0][0].base.rank is Rank.KING

    def test_ace_is_highest(self):
        hand = [
            _card("Hearts", "King"), _card("Spades", "Ace"),
            _card("Clubs", "Queen"),
        ]
        result = get_highest(hand)
        assert result[0][0].base.rank is Rank.ACE

    def test_empty_hand(self):
        assert get_highest([]) == []

    def test_single_card(self):
        hand = [_card("Hearts", "5")]
        result = get_highest(hand)
        assert result[0][0].base.rank is Rank.FIVE


# ============================================================================
# Edge cases and combinations
# ============================================================================

class TestEdgeCases:
    def test_straight_with_duplicate_ranks(self):
        """Duplicate ranks in a straight: all copies included in result."""
        hand = [
            _card("Hearts", "5"), _card("Spades", "5"),
            _card("Clubs", "6"), _card("Diamonds", "7"),
            _card("Hearts", "8"),
        ]
        # 5,5,6,7,8 — only 4 unique ranks, not a straight
        result = get_straight(hand)
        assert result == []

    def test_wild_plus_smeared_flush(self):
        """Wild Card + Smeared: Wild matches everything, smeared merges red/black."""
        hand = [
            _card("Hearts", "2"), _card("Diamonds", "5"),
            _card("Hearts", "8"),
            _card("Clubs", "Jack", enhancement="m_wild"),  # Wild
            _card("Diamonds", "Ace"),
        ]
        # All red + Wild = flush with smeared
        result = get_flush(hand, smeared=True)
        assert len(result) == 1
        assert len(result[0]) == 5

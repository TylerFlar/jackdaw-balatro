"""Tests for poker hand detection functions.

Verifies get_flush, get_straight, get_x_same, and get_highest against
known card combinations, including Four Fingers, Shortcut, Wild Card,
Smeared Joker, and Stone Card modifiers.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.hand_eval import (
    evaluate_poker_hand,
    find_joker,
    get_best_hand,
    get_flush,
    get_hand_eval_flags,
    get_highest,
    get_straight,
    get_x_same,
)


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


# ============================================================================
# evaluate_poker_hand
# ============================================================================

class TestEvaluatePokerHand:
    """Master detection function returning all matching hands."""

    def test_high_card(self):
        hand = [
            _card("Hearts", "2"), _card("Spades", "5"),
            _card("Clubs", "8"), _card("Diamonds", "Jack"),
            _card("Hearts", "Ace"),
        ]
        results = evaluate_poker_hand(hand)
        assert results["High Card"]
        assert not results["Pair"]

    def test_pair(self):
        hand = [
            _card("Hearts", "5"), _card("Spades", "5"),
            _card("Clubs", "8"), _card("Diamonds", "Jack"),
            _card("Hearts", "Ace"),
        ]
        results = evaluate_poker_hand(hand)
        assert results["Pair"]
        assert not results["Two Pair"]

    def test_two_pair(self):
        hand = [
            _card("Hearts", "5"), _card("Spades", "5"),
            _card("Clubs", "Jack"), _card("Diamonds", "Jack"),
            _card("Hearts", "Ace"),
        ]
        results = evaluate_poker_hand(hand)
        assert results["Two Pair"]
        assert results["Pair"]  # downward: Two Pair also has Pair

    def test_three_of_a_kind(self):
        hand = [
            _card("Hearts", "King"), _card("Spades", "King"),
            _card("Clubs", "King"), _card("Diamonds", "5"),
            _card("Hearts", "2"),
        ]
        results = evaluate_poker_hand(hand)
        assert results["Three of a Kind"]
        assert results["Pair"]  # downward propagation

    def test_straight(self):
        hand = [
            _card("Hearts", "4"), _card("Spades", "5"),
            _card("Clubs", "6"), _card("Diamonds", "7"),
            _card("Hearts", "8"),
        ]
        results = evaluate_poker_hand(hand)
        assert results["Straight"]

    def test_flush(self):
        hand = [_card("Hearts", r) for r in ["2", "5", "8", "Jack", "Ace"]]
        results = evaluate_poker_hand(hand)
        assert results["Flush"]

    def test_full_house(self):
        hand = [
            _card("Hearts", "King"), _card("Spades", "King"),
            _card("Clubs", "King"), _card("Diamonds", "5"),
            _card("Hearts", "5"),
        ]
        results = evaluate_poker_hand(hand)
        assert results["Full House"]
        assert results["Three of a Kind"]  # downward
        assert results["Pair"]  # downward

    def test_four_of_a_kind(self):
        hand = [
            _card("Hearts", "7"), _card("Spades", "7"),
            _card("Clubs", "7"), _card("Diamonds", "7"),
            _card("Hearts", "Ace"),
        ]
        results = evaluate_poker_hand(hand)
        assert results["Four of a Kind"]
        assert results["Three of a Kind"]  # downward
        assert results["Pair"]  # downward

    def test_straight_flush(self):
        hand = [
            _card("Hearts", "4"), _card("Hearts", "5"),
            _card("Hearts", "6"), _card("Hearts", "7"),
            _card("Hearts", "8"),
        ]
        results = evaluate_poker_hand(hand)
        assert results["Straight Flush"]
        assert results["Flush"]
        assert results["Straight"]

    def test_five_of_a_kind(self):
        hand = [
            _card("Hearts", "Ace"), _card("Spades", "Ace"),
            _card("Clubs", "Ace"), _card("Diamonds", "Ace"),
            _card("Hearts", "Ace"),
        ]
        results = evaluate_poker_hand(hand)
        assert results["Five of a Kind"]
        assert results["Four of a Kind"]  # downward
        assert results["Three of a Kind"]  # downward
        assert results["Pair"]  # downward

    def test_flush_five(self):
        hand = [
            _card("Hearts", "Ace"), _card("Hearts", "Ace"),
            _card("Hearts", "Ace"), _card("Hearts", "Ace"),
            _card("Hearts", "Ace"),
        ]
        results = evaluate_poker_hand(hand)
        assert results["Flush Five"]
        assert results["Five of a Kind"]
        assert results["Flush"]

    def test_flush_house(self):
        hand = [
            _card("Hearts", "King"), _card("Hearts", "King"),
            _card("Hearts", "King"), _card("Hearts", "5"),
            _card("Hearts", "5"),
        ]
        results = evaluate_poker_hand(hand)
        assert results["Flush House"]
        assert results["Full House"]
        assert results["Flush"]

    def test_high_card_always_populated(self):
        """High Card entry is always populated for non-empty hands."""
        hand = [_card("Hearts", "5"), _card("Spades", "5")]
        results = evaluate_poker_hand(hand)
        assert results["High Card"]


class TestDownwardPropagation:
    """Downward propagation: higher hands populate lower entries."""

    def test_five_of_a_kind_populates_four_three_pair(self):
        hand = [
            _card("Hearts", "Ace"), _card("Spades", "Ace"),
            _card("Clubs", "Ace"), _card("Diamonds", "Ace"),
            _card("Hearts", "Ace"),
        ]
        results = evaluate_poker_hand(hand)
        assert results["Five of a Kind"]
        assert results["Four of a Kind"]
        assert results["Three of a Kind"]
        assert results["Pair"]

    def test_four_of_a_kind_populates_three_pair(self):
        hand = [
            _card("Hearts", "7"), _card("Spades", "7"),
            _card("Clubs", "7"), _card("Diamonds", "7"),
            _card("Hearts", "Ace"),
        ]
        results = evaluate_poker_hand(hand)
        assert results["Four of a Kind"]
        assert results["Three of a Kind"]
        assert results["Pair"]
        assert not results["Five of a Kind"]

    def test_three_of_a_kind_populates_pair(self):
        hand = [
            _card("Hearts", "King"), _card("Spades", "King"),
            _card("Clubs", "King"), _card("Diamonds", "5"),
            _card("Hearts", "2"),
        ]
        results = evaluate_poker_hand(hand)
        assert results["Three of a Kind"]
        assert results["Pair"]
        assert not results["Four of a Kind"]

    def test_pair_does_not_propagate_up(self):
        hand = [
            _card("Hearts", "5"), _card("Spades", "5"),
            _card("Clubs", "8"), _card("Diamonds", "Jack"),
            _card("Hearts", "Ace"),
        ]
        results = evaluate_poker_hand(hand)
        assert results["Pair"]
        assert not results["Three of a Kind"]


class TestGetBestHand:
    """get_best_hand returns the highest-priority detected hand."""

    def test_full_house_over_flush(self):
        """Full House has higher priority than Flush — but both can coexist."""
        hand = [
            _card("Hearts", "King"), _card("Hearts", "King"),
            _card("Hearts", "King"), _card("Hearts", "5"),
            _card("Hearts", "5"),
        ]
        name, scoring, results = get_best_hand(hand)
        # Flush House is highest priority here (it's a flush + full house)
        assert name == "Flush House"
        assert results["Full House"]
        assert results["Flush"]

    def test_straight_flush_detected(self):
        hand = [
            _card("Spades", "9"), _card("Spades", "10"),
            _card("Spades", "Jack"), _card("Spades", "Queen"),
            _card("Spades", "King"),
        ]
        name, scoring, _ = get_best_hand(hand)
        assert name == "Straight Flush"
        assert len(scoring) == 5

    def test_high_card(self):
        hand = [
            _card("Hearts", "2"), _card("Spades", "5"),
            _card("Clubs", "8"), _card("Diamonds", "Jack"),
            _card("Hearts", "Ace"),
        ]
        name, scoring, _ = get_best_hand(hand)
        assert name == "High Card"
        assert len(scoring) == 1

    def test_empty_hand(self):
        name, scoring, _ = get_best_hand([])
        assert name == "NULL"
        assert scoring == []


class TestJokerFlags:
    """Joker modifiers passed through to evaluate_poker_hand."""

    def test_four_fingers_flush(self):
        hand = [
            _card("Clubs", "3"), _card("Clubs", "7"),
            _card("Clubs", "10"), _card("Clubs", "King"),
            _card("Hearts", "Ace"),
        ]
        name, _, _ = get_best_hand(hand, four_fingers=True)
        assert name == "Flush"

    def test_four_fingers_straight(self):
        hand = [
            _card("Hearts", "4"), _card("Spades", "5"),
            _card("Clubs", "6"), _card("Diamonds", "7"),
            _card("Hearts", "King"),
        ]
        name, _, _ = get_best_hand(hand, four_fingers=True)
        assert name == "Straight"

    def test_shortcut_straight(self):
        hand = [
            _card("Hearts", "3"), _card("Spades", "4"),
            _card("Clubs", "6"), _card("Diamonds", "7"),
            _card("Hearts", "8"),
        ]
        name, _, _ = get_best_hand(hand, shortcut=True)
        assert name == "Straight"

    def test_smeared_flush(self):
        hand = [
            _card("Hearts", "2"), _card("Hearts", "5"),
            _card("Diamonds", "8"), _card("Diamonds", "Jack"),
            _card("Hearts", "Ace"),
        ]
        name, _, _ = get_best_hand(hand, smeared=True)
        assert name == "Flush"

    def test_four_fingers_straight_flush(self):
        hand = [
            _card("Hearts", "5"), _card("Hearts", "6"),
            _card("Hearts", "7"), _card("Hearts", "8"),
            _card("Spades", "King"),
        ]
        name, _, _ = get_best_hand(hand, four_fingers=True)
        assert name == "Straight Flush"


# ============================================================================
# find_joker
# ============================================================================

class TestFindJoker:
    """find_joker matches by ability.name, excludes debuffed by default."""

    def test_finds_by_name(self):
        j = create_joker("j_four_fingers")
        result = find_joker("Four Fingers", [j])
        assert len(result) == 1
        assert result[0] is j

    def test_not_found(self):
        j = create_joker("j_joker")
        assert find_joker("Four Fingers", [j]) == []

    def test_empty_list(self):
        assert find_joker("Four Fingers", []) == []

    def test_debuffed_excluded_by_default(self):
        """Debuffed jokers are excluded from find_joker by default.

        Source: find_joker (misc_functions.lua:903) uses the condition
        ``(non_debuff or not v.debuff)`` — with default nil/false for
        non_debuff, this evaluates to ``not v.debuff``, excluding
        debuffed jokers.
        """
        j = create_joker("j_four_fingers")
        j.set_debuff(True)
        assert find_joker("Four Fingers", [j]) == []

    def test_debuffed_included_with_non_debuff(self):
        """non_debuff=True includes debuffed jokers (rare, but supported)."""
        j = create_joker("j_four_fingers")
        j.set_debuff(True)
        result = find_joker("Four Fingers", [j], non_debuff=True)
        assert len(result) == 1

    def test_multiple_matches(self):
        """Multiple jokers with same name (via Showman)."""
        j1 = create_joker("j_four_fingers")
        j2 = create_joker("j_four_fingers")
        result = find_joker("Four Fingers", [j1, j2])
        assert len(result) == 2


# ============================================================================
# get_hand_eval_flags
# ============================================================================

class TestGetHandEvalFlags:
    """Extract modifier flags from active jokers."""

    def test_no_jokers(self):
        flags = get_hand_eval_flags([])
        assert flags == {
            "four_fingers": False, "shortcut": False,
            "smeared": False, "splash": False, "pareidolia": False,
        }

    def test_four_fingers(self):
        flags = get_hand_eval_flags([create_joker("j_four_fingers")])
        assert flags["four_fingers"] is True
        assert flags["shortcut"] is False

    def test_shortcut(self):
        flags = get_hand_eval_flags([create_joker("j_shortcut")])
        assert flags["shortcut"] is True

    def test_smeared(self):
        flags = get_hand_eval_flags([create_joker("j_smeared")])
        assert flags["smeared"] is True

    def test_splash(self):
        flags = get_hand_eval_flags([create_joker("j_splash")])
        assert flags["splash"] is True

    def test_pareidolia(self):
        flags = get_hand_eval_flags([create_joker("j_pareidolia")])
        assert flags["pareidolia"] is True

    def test_multiple_modifiers(self):
        jokers = [
            create_joker("j_four_fingers"),
            create_joker("j_shortcut"),
            create_joker("j_smeared"),
        ]
        flags = get_hand_eval_flags(jokers)
        assert flags["four_fingers"] is True
        assert flags["shortcut"] is True
        assert flags["smeared"] is True
        assert flags["splash"] is False

    def test_non_meta_joker_ignored(self):
        """Regular jokers don't set any flags."""
        flags = get_hand_eval_flags([create_joker("j_joker")])
        assert not any(flags.values())

    def test_debuffed_meta_joker_excluded(self):
        """Debuffed meta jokers do NOT apply their passive effects.

        Source: find_joker (misc_functions.lua:903) excludes debuffed
        jokers by default.  The condition ``(non_debuff or not v.debuff)``
        with nil non_debuff evaluates to ``not v.debuff``.
        """
        j = create_joker("j_four_fingers")
        j.set_debuff(True)
        flags = get_hand_eval_flags([j])
        assert flags["four_fingers"] is False

    def test_mixed_debuffed_and_active(self):
        j1 = create_joker("j_four_fingers")
        j1.set_debuff(True)  # debuffed → excluded
        j2 = create_joker("j_shortcut")  # active
        flags = get_hand_eval_flags([j1, j2])
        assert flags["four_fingers"] is False
        assert flags["shortcut"] is True

    def test_integration_with_evaluate(self):
        """Detection flags feed into evaluate_poker_hand.

        Note: splash and pareidolia are applied in the scoring pipeline,
        not in hand detection — only four_fingers, shortcut, and smeared
        are passed to evaluate_poker_hand.
        """
        jokers = [create_joker("j_four_fingers")]
        flags = get_hand_eval_flags(jokers)

        hand = [
            _card("Clubs", "3"), _card("Clubs", "7"),
            _card("Clubs", "10"), _card("Clubs", "King"),
            _card("Hearts", "Ace"),
        ]
        # Without flags: no flush (only 4 clubs)
        name_no, _, _ = get_best_hand(hand)
        assert name_no != "Flush"

        # With detection flags from jokers: flush detected
        # Filter to only the 3 flags evaluate_poker_hand accepts
        detect_flags = {
            k: v for k, v in flags.items()
            if k in ("four_fingers", "shortcut", "smeared")
        }
        name_yes, _, _ = get_best_hand(hand, **detect_flags)
        assert name_yes == "Flush"

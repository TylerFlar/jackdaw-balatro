"""Integration tests for the complete hand evaluation pipeline.

Tests evaluate_hand() which ties together joker flag extraction,
poker hand detection, and scoring card augmentation (Splash, Stone).
"""

from __future__ import annotations

from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.hand_eval import HandEvalResult, evaluate_hand


def _reset():
    reset_sort_id_counter()


def _card(suit: str, rank: str, enhancement: str = "c_base") -> Card:
    """Helper: create a playing card."""
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


# ============================================================================
# Basic detection
# ============================================================================

class TestBasicDetection:
    def test_flush(self):
        _reset()
        hand = [_card("Hearts", r) for r in ["2", "5", "8", "Jack", "Ace"]]
        result = evaluate_hand(hand)
        assert isinstance(result, HandEvalResult)
        assert result.detected_hand == "Flush"
        assert len(result.scoring_cards) == 5
        assert result.all_played == hand

    def test_full_house(self):
        _reset()
        hand = [
            _card("Hearts", "King"), _card("Spades", "King"),
            _card("Clubs", "King"), _card("Diamonds", "5"),
            _card("Hearts", "5"),
        ]
        result = evaluate_hand(hand)
        assert result.detected_hand == "Full House"
        assert len(result.scoring_cards) == 5

    def test_pair(self):
        _reset()
        hand = [
            _card("Hearts", "5"), _card("Spades", "5"),
            _card("Clubs", "8"),
        ]
        result = evaluate_hand(hand)
        assert result.detected_hand == "Pair"
        assert len(result.scoring_cards) == 2

    def test_high_card(self):
        _reset()
        hand = [
            _card("Hearts", "2"), _card("Spades", "5"),
            _card("Clubs", "8"), _card("Diamonds", "Jack"),
            _card("Hearts", "Ace"),
        ]
        result = evaluate_hand(hand)
        assert result.detected_hand == "High Card"
        assert len(result.scoring_cards) == 1

    def test_straight(self):
        _reset()
        hand = [
            _card("Hearts", "4"), _card("Spades", "5"),
            _card("Clubs", "6"), _card("Diamonds", "7"),
            _card("Hearts", "8"),
        ]
        result = evaluate_hand(hand)
        assert result.detected_hand == "Straight"

    def test_flush_five(self):
        _reset()
        hand = [
            _card("Hearts", "Ace"), _card("Hearts", "Ace"),
            _card("Hearts", "Ace"), _card("Hearts", "Ace"),
            _card("Hearts", "Ace"),
        ]
        result = evaluate_hand(hand)
        assert result.detected_hand == "Flush Five"
        # Should also register lower hands
        assert result.poker_hands["Five of a Kind"]
        assert result.poker_hands["Flush"]
        assert result.poker_hands["Four of a Kind"]
        assert result.poker_hands["Pair"]


# ============================================================================
# Joker modifiers
# ============================================================================

class TestJokerModifiers:
    def test_four_fingers_flush(self):
        """Four Fingers: 4 same-suit cards = Flush."""
        _reset()
        hand = [
            _card("Clubs", "3"), _card("Clubs", "7"),
            _card("Clubs", "10"), _card("Clubs", "King"),
            _card("Hearts", "Ace"),
        ]
        jokers = [create_joker("j_four_fingers")]
        result = evaluate_hand(hand, jokers=jokers)
        assert result.detected_hand == "Flush"

    def test_four_fingers_straight(self):
        _reset()
        hand = [
            _card("Hearts", "4"), _card("Spades", "5"),
            _card("Clubs", "6"), _card("Diamonds", "7"),
            _card("Hearts", "King"),
        ]
        jokers = [create_joker("j_four_fingers")]
        result = evaluate_hand(hand, jokers=jokers)
        assert result.detected_hand == "Straight"

    def test_shortcut_straight(self):
        """Shortcut: gap at rank 4, 3-5-6-7-8 = Straight."""
        _reset()
        hand = [
            _card("Hearts", "3"), _card("Spades", "5"),
            _card("Clubs", "6"), _card("Diamonds", "7"),
            _card("Hearts", "8"),
        ]
        jokers = [create_joker("j_shortcut")]
        result = evaluate_hand(hand, jokers=jokers)
        assert result.detected_hand == "Straight"

    def test_smeared_flush(self):
        """Smeared: 3 Hearts + 2 Diamonds = Flush (all red)."""
        _reset()
        hand = [
            _card("Hearts", "2"), _card("Hearts", "5"),
            _card("Diamonds", "8"), _card("Diamonds", "Jack"),
            _card("Hearts", "Ace"),
        ]
        jokers = [create_joker("j_smeared")]
        result = evaluate_hand(hand, jokers=jokers)
        assert result.detected_hand == "Flush"
        assert len(result.scoring_cards) == 5

    def test_splash_all_cards_score(self):
        """Splash: all 5 played cards become scoring cards, even High Card."""
        _reset()
        hand = [
            _card("Hearts", "2"), _card("Spades", "5"),
            _card("Clubs", "8"), _card("Diamonds", "Jack"),
            _card("Hearts", "Ace"),
        ]
        jokers = [create_joker("j_splash")]
        result = evaluate_hand(hand, jokers=jokers)
        assert result.detected_hand == "High Card"
        assert len(result.scoring_cards) == 5
        # All 5 cards should be in scoring
        assert set(id(c) for c in result.scoring_cards) == set(id(c) for c in hand)

    def test_splash_with_pair(self):
        """Splash + Pair: pair cards score normally, rest added by Splash."""
        _reset()
        hand = [
            _card("Hearts", "5"), _card("Spades", "5"),
            _card("Clubs", "8"), _card("Diamonds", "Jack"),
            _card("Hearts", "Ace"),
        ]
        jokers = [create_joker("j_splash")]
        result = evaluate_hand(hand, jokers=jokers)
        assert result.detected_hand == "Pair"
        assert len(result.scoring_cards) == 5

    def test_debuffed_four_fingers_no_effect(self):
        """Debuffed Four Fingers: does NOT enable 4-card flush."""
        _reset()
        hand = [
            _card("Clubs", "3"), _card("Clubs", "7"),
            _card("Clubs", "10"), _card("Clubs", "King"),
            _card("Hearts", "Ace"),
        ]
        ff = create_joker("j_four_fingers")
        ff.set_debuff(True)
        result = evaluate_hand(hand, jokers=[ff])
        assert result.detected_hand != "Flush"

    def test_no_jokers(self):
        """No jokers = no modifier flags."""
        _reset()
        hand = [_card("Hearts", r) for r in ["2", "5", "8", "Jack", "Ace"]]
        result = evaluate_hand(hand, jokers=None)
        assert result.detected_hand == "Flush"

    def test_multiple_modifiers(self):
        """Four Fingers + Shortcut active simultaneously."""
        _reset()
        hand = [
            _card("Hearts", "3"),
            _card("Hearts", "5"),  # gap at 4
            _card("Hearts", "6"),
            _card("Hearts", "7"),
            _card("Spades", "King"),
        ]
        jokers = [create_joker("j_four_fingers"), create_joker("j_shortcut")]
        result = evaluate_hand(hand, jokers=jokers)
        # 4 hearts + shortcut gap = Straight Flush with Four Fingers
        assert result.detected_hand == "Straight Flush"


# ============================================================================
# Stone Card augmentation
# ============================================================================

class TestStoneCardAugmentation:
    def test_stone_card_added_to_scoring(self):
        """Stone Card played as 5th card with Full House: included in scoring."""
        _reset()
        hand = [
            _card("Hearts", "King"), _card("Spades", "King"),
            _card("Clubs", "King"), _card("Diamonds", "5"),
            _card("Hearts", "3", enhancement="m_stone"),
        ]
        result = evaluate_hand(hand)
        assert result.detected_hand == "Three of a Kind"
        # Stone card should be augmented into scoring
        stone = hand[4]
        assert stone in result.scoring_cards

    def test_stone_in_pair_hand(self):
        """Stone Card doesn't form pairs (get_id returns -1), but is added as a pure."""
        _reset()
        hand = [
            _card("Hearts", "5"), _card("Spades", "5"),
            _card("Clubs", "3", enhancement="m_stone"),
        ]
        result = evaluate_hand(hand)
        assert result.detected_hand == "Pair"
        # 2 pair cards + 1 stone = 3 scoring
        assert len(result.scoring_cards) == 3


# ============================================================================
# Scoring card ordering
# ============================================================================

class TestScoringOrder:
    def test_left_to_right(self):
        """Scoring cards should be in the same order as played_cards."""
        _reset()
        hand = [
            _card("Hearts", "King"), _card("Spades", "King"),
            _card("Clubs", "King"), _card("Diamonds", "5"),
            _card("Hearts", "5"),
        ]
        result = evaluate_hand(hand)
        # Full House: KKK + 55 — should preserve played order
        played_ids = [id(c) for c in hand]
        scoring_ids = [id(c) for c in result.scoring_cards]
        # Scoring cards should be a subsequence of played_cards order
        for i in range(len(scoring_ids) - 1):
            assert played_ids.index(scoring_ids[i]) < played_ids.index(scoring_ids[i + 1])


# ============================================================================
# Result structure
# ============================================================================

class TestResultStructure:
    def test_all_fields_present(self):
        _reset()
        hand = [_card("Hearts", "5"), _card("Spades", "5")]
        result = evaluate_hand(hand)
        assert hasattr(result, "detected_hand")
        assert hasattr(result, "poker_hands")
        assert hasattr(result, "scoring_cards")
        assert hasattr(result, "all_played")
        assert isinstance(result.poker_hands, dict)
        assert "Pair" in result.poker_hands

    def test_all_played_is_copy(self):
        """all_played should be a copy, not a reference."""
        _reset()
        hand = [_card("Hearts", "5")]
        result = evaluate_hand(hand)
        result.all_played.append(_card("Spades", "Ace"))
        assert len(hand) == 1  # original unchanged

    def test_empty_hand(self):
        _reset()
        result = evaluate_hand([])
        assert result.detected_hand == "NULL"
        assert result.scoring_cards == []

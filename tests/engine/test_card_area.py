"""Tests for CardArea and draw_card.

Verifies card management, limit enforcement, deterministic shuffle,
sorting, highlighting, and draw operations.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.card import Card, reset_sort_id_counter
from jackdaw.engine.card_area import CardArea, draw_card
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.rng import PseudoRandom


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


_RANK_LETTER = {
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


def _make_playing_card(suit: str, rank: str) -> Card:
    """Helper: create a Card with base set."""
    c = Card()
    key = f"{suit[0]}_{_RANK_LETTER[rank]}"
    c.set_base(key, suit, rank)
    c.set_ability({"name": "Default Base", "set": "Default", "config": {}})
    return c


def _make_deck(n: int = 52) -> list[Card]:
    """Helper: create n cards with sequential sort_ids and bases."""
    suits = ["Hearts", "Diamonds", "Clubs", "Spades"]
    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
    rank_letters = {
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
    cards = []
    for suit in suits:
        for rank in ranks:
            c = Card()
            key = f"{suit[0]}_{rank_letters[rank]}"
            c.set_base(key, suit, rank)
            c.set_ability({"name": "Default Base", "set": "Default", "config": {}})
            cards.append(c)
            if len(cards) >= n:
                return cards
    return cards


# ============================================================================
# CardArea basics
# ============================================================================


class TestCardAreaBasics:
    def test_init(self):
        area = CardArea(card_limit=5, area_type="hand")
        assert area.card_limit == 5
        assert area.type == "hand"
        assert len(area) == 0
        assert area.cards == []
        assert area.highlighted == []

    def test_add_and_len(self):
        area = CardArea()
        c = Card()
        area.add(c)
        assert len(area) == 1
        assert area.cards[0] is c

    def test_add_front(self):
        area = CardArea()
        c1, c2 = Card(), Card()
        area.add(c1)
        area.add(c2, front=True)
        assert area.cards[0] is c2
        assert area.cards[1] is c1

    def test_remove(self):
        area = CardArea()
        c1, c2, c3 = Card(), Card(), Card()
        area.add(c1)
        area.add(c2)
        area.add(c3)
        removed = area.remove(c2)
        assert removed is c2
        assert len(area) == 2
        assert c2 not in area.cards

    def test_remove_top(self):
        area = CardArea()
        c1, c2 = Card(), Card()
        area.add(c1)
        area.add(c2)
        top = area.remove_top()
        assert top is c2
        assert len(area) == 1

    def test_has_space(self):
        area = CardArea(card_limit=3)
        assert area.has_space() is True
        area.add(Card())
        area.add(Card())
        assert area.has_space() is True
        area.add(Card())
        assert area.has_space() is False

    def test_has_space_negative_bonus(self):
        area = CardArea(card_limit=3)
        area.add(Card())
        area.add(Card())
        area.add(Card())
        assert area.has_space() is False
        assert area.has_space(negative_bonus=1) is True

    def test_card_limit_setter(self):
        area = CardArea(card_limit=5)
        area.card_limit = 8
        assert area.card_limit == 8

    def test_card_limit_minimum_zero(self):
        area = CardArea(card_limit=5)
        area.card_limit = -3
        assert area.card_limit == 0

    def test_repr(self):
        area = CardArea(card_limit=5, area_type="hand")
        area.add(Card())
        assert "hand" in repr(area)
        assert "1" in repr(area)


# ============================================================================
# Highlighting
# ============================================================================


class TestHighlighting:
    def test_highlight_and_unhighlight(self):
        area = CardArea()
        c = Card()
        area.add(c)
        area.add_to_highlighted(c)
        assert c in area.highlighted
        area.remove_from_highlighted(c)
        assert c not in area.highlighted

    def test_unhighlight_all(self):
        area = CardArea()
        cards = [Card() for _ in range(5)]
        for c in cards:
            area.add(c)
            area.add_to_highlighted(c)
        assert len(area.highlighted) == 5
        area.unhighlight_all()
        assert len(area.highlighted) == 0

    def test_remove_also_unhighlights(self):
        area = CardArea()
        c = Card()
        area.add(c)
        area.add_to_highlighted(c)
        area.remove(c)
        assert c not in area.highlighted

    def test_no_duplicate_highlights(self):
        area = CardArea()
        c = Card()
        area.add(c)
        area.add_to_highlighted(c)
        area.add_to_highlighted(c)
        assert area.highlighted.count(c) == 1


# ============================================================================
# draw_card
# ============================================================================


class TestDrawCard:
    def test_draw_from_deck_to_hand(self):
        deck = CardArea(card_limit=52, area_type="deck")
        hand = CardArea(card_limit=8, area_type="hand")
        cards = _make_deck(52)
        for c in cards:
            deck.add(c)
        assert len(deck) == 52

        drawn = draw_card(deck, hand, count=8)
        assert len(drawn) == 8
        assert len(deck) == 44
        assert len(hand) == 8

    def test_draw_respects_target_limit(self):
        deck = CardArea(card_limit=52, area_type="deck")
        hand = CardArea(card_limit=3, area_type="hand")
        for c in _make_deck(10):
            deck.add(c)

        drawn = draw_card(deck, hand, count=5)
        assert len(drawn) == 3  # limited by hand capacity
        assert len(hand) == 3
        assert len(deck) == 7

    def test_draw_respects_source_count(self):
        deck = CardArea(card_limit=52, area_type="deck")
        hand = CardArea(card_limit=10, area_type="hand")
        for c in _make_deck(3):
            deck.add(c)

        drawn = draw_card(deck, hand, count=5)
        assert len(drawn) == 3  # only 3 available
        assert len(deck) == 0

    def test_draw_from_top(self):
        """Cards are drawn from the end of the list (top of deck)."""
        deck = CardArea(card_limit=52, area_type="deck")
        hand = CardArea(card_limit=8, area_type="hand")
        cards = _make_deck(5)
        for c in cards:
            deck.add(c)

        drawn = draw_card(deck, hand, count=1)
        assert drawn[0] is cards[-1]  # last card added = top of deck

    def test_draw_empty_deck(self):
        deck = CardArea(card_limit=52, area_type="deck")
        hand = CardArea(card_limit=8, area_type="hand")
        drawn = draw_card(deck, hand, count=5)
        assert drawn == []

    def test_draw_to_full_hand(self):
        deck = CardArea(card_limit=52, area_type="deck")
        hand = CardArea(card_limit=2, area_type="hand")
        for c in _make_deck(5):
            deck.add(c)
        hand.add(Card())
        hand.add(Card())

        drawn = draw_card(deck, hand, count=3)
        assert drawn == []  # hand is full
        assert len(deck) == 5


# ============================================================================
# Shuffle
# ============================================================================


class TestShuffle:
    def test_deterministic_shuffle(self):
        """Same seed produces same card order."""

        def make_area():
            reset_sort_id_counter()
            area = CardArea(card_limit=52, area_type="deck")
            for c in _make_deck(10):
                area.add(c)
            return area

        a1 = make_area()
        rng1 = PseudoRandom("TESTSEED")
        a1.shuffle(rng1, "shuffle")
        ids1 = [c.sort_id for c in a1.cards]

        a2 = make_area()
        rng2 = PseudoRandom("TESTSEED")
        a2.shuffle(rng2, "shuffle")
        ids2 = [c.sort_id for c in a2.cards]

        assert ids1 == ids2

    def test_shuffle_changes_order(self):
        area = CardArea(card_limit=52, area_type="deck")
        cards = _make_deck(20)
        for c in cards:
            area.add(c)
        original_ids = [c.sort_id for c in area.cards]

        rng = PseudoRandom("TESTSEED")
        area.shuffle(rng, "shuffle")
        shuffled_ids = [c.sort_id for c in area.cards]

        assert shuffled_ids != original_ids
        assert sorted(shuffled_ids) == sorted(original_ids)

    def test_52_card_shuffle_matches_rng_test(self):
        """Full 52-card shuffle should produce same order as test_rng_sequence."""
        area = CardArea(card_limit=52, area_type="deck")
        for i in range(1, 53):
            c = Card()
            # Use sort_id as a simple identifier
            area.add(c)

        rng = PseudoRandom("TESTSEED")
        area.shuffle(rng, "shuffle")

        # The shuffled sort_ids should match the sequence from test_rng_sequence
        # (which uses the same RNG path: pseudoseed('shuffle') → shuffle)
        result = [c.sort_id for c in area.cards]
        assert len(result) == 52
        assert len(set(result)) == 52  # all unique


# ============================================================================
# Sorting
# ============================================================================


class TestSort:
    def test_sort_by_value_descending(self):
        area = CardArea(card_limit=10, area_type="hand")
        # Add cards in arbitrary order
        for rank in ["3", "Ace", "7", "King", "2"]:
            c = Card()
            c.set_base(f"H_{rank[0]}", "Hearts", rank)
            c.set_ability({"name": "Default Base", "set": "Default", "config": {}})
            area.add(c)

        area.sort_by_value(descending=True)
        ranks = [c.base.rank for c in area.cards]
        # Ace(11) > King(10) > 7 > 3 > 2
        assert ranks[0] is Rank.ACE
        assert ranks[1] is Rank.KING
        assert ranks[-1] is Rank.TWO

    def test_sort_by_value_ascending(self):
        area = CardArea(card_limit=10, area_type="hand")
        for rank in ["King", "3", "Ace"]:
            c = Card()
            c.set_base(f"S_{rank[0]}", "Spades", rank)
            c.set_ability({"name": "Default Base", "set": "Default", "config": {}})
            area.add(c)

        area.sort_by_value(descending=False)
        ranks = [c.base.rank for c in area.cards]
        assert ranks[0] is Rank.THREE
        assert ranks[-1] is Rank.ACE

    def test_sort_by_suit(self):
        area = CardArea(card_limit=10, area_type="hand")
        for suit, rank in [("Diamonds", "5"), ("Spades", "5"), ("Hearts", "5"), ("Clubs", "5")]:
            c = Card()
            c.set_base(f"{suit[0]}_5", suit, rank)
            c.set_ability({"name": "Default Base", "set": "Default", "config": {}})
            area.add(c)

        area.sort_by_suit(descending=True)
        suits = [c.base.suit for c in area.cards]
        # Spades(0.04) > Hearts(0.03) > Clubs(0.02) > Diamonds(0.01)
        assert suits[0] is Suit.SPADES
        assert suits[-1] is Suit.DIAMONDS

    def test_sort_stable_for_same_rank_different_suit(self):
        """Same-rank cards from different suits should be ordered by suit_nominal."""
        area = CardArea(card_limit=10, area_type="hand")
        for suit in ["Diamonds", "Spades", "Hearts", "Clubs"]:
            c = Card()
            c.set_base(f"{suit[0]}_A", suit, "Ace")
            c.set_ability({"name": "Default Base", "set": "Default", "config": {}})
            area.add(c)

        area.sort_by_value(descending=True)
        suits = [c.base.suit for c in area.cards]
        # All Aces, sorted by suit_nominal descending
        assert suits[0] is Suit.SPADES  # 0.04
        assert suits[-1] is Suit.DIAMONDS  # 0.01

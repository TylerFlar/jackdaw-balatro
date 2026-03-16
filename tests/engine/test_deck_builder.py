"""Tests for deck building.

Verifies standard 52-card deck, Abandoned/Checkered/Erratic deck mutations,
challenge filtering, and deterministic sort order.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.card import reset_sort_id_counter
from jackdaw.engine.data.enums import Rank, Suit
from jackdaw.engine.deck_builder import build_deck
from jackdaw.engine.rng import PseudoRandom


@pytest.fixture(autouse=True)
def _reset():
    reset_sort_id_counter()


# ============================================================================
# Standard deck
# ============================================================================

class TestStandardDeck:
    def test_52_cards(self):
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_red", rng)
        assert len(cards) == 52

    def test_4_suits_13_ranks(self):
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_red", rng)
        suits = {c.base.suit for c in cards}
        assert suits == {Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS, Suit.SPADES}
        for suit in Suit:
            count = sum(1 for c in cards if c.base.suit is suit)
            assert count == 13, f"{suit}: {count}"

    def test_all_ranks_present(self):
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_red", rng)
        ranks = {c.base.rank for c in cards}
        assert ranks == set(Rank)

    def test_default_enhancement_is_base(self):
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_red", rng)
        for c in cards:
            assert c.center_key == "c_base"

    def test_sorted_deterministically(self):
        """Cards are sorted by suit+rank letter concatenation."""
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_red", rng)
        # First card should be C_2 (Clubs 2), last should be S_T or similar
        # Sort order: C < D < H < S, then 2 < 3 < ... < 9 < A < J < K < Q < T
        # (lexicographic on the rank LETTER, not rank value)
        assert cards[0].card_key == "C_2"
        # Verify monotonic sort
        keys = [c.card_key for c in cards]
        assert keys == sorted(keys)

    def test_playing_card_indices(self):
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_red", rng)
        indices = [c.playing_card for c in cards]
        assert indices == list(range(1, 53))


# ============================================================================
# Abandoned Deck
# ============================================================================

class TestAbandonedDeck:
    def test_40_cards(self):
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_abandoned", rng)
        assert len(cards) == 40

    def test_no_face_cards(self):
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_abandoned", rng)
        for c in cards:
            assert c.base.rank not in (Rank.JACK, Rank.QUEEN, Rank.KING), (
                f"Face card found: {c.base.rank}"
            )

    def test_has_aces(self):
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_abandoned", rng)
        aces = [c for c in cards if c.base.rank is Rank.ACE]
        assert len(aces) == 4

    def test_10_cards_per_suit(self):
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_abandoned", rng)
        for suit in Suit:
            count = sum(1 for c in cards if c.base.suit is suit)
            assert count == 10


# ============================================================================
# Checkered Deck
# ============================================================================

class TestCheckeredDeck:
    def test_52_cards(self):
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_checkered", rng)
        assert len(cards) == 52

    def test_only_two_suits(self):
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_checkered", rng)
        suits = {c.base.suit for c in cards}
        assert suits == {Suit.SPADES, Suit.HEARTS}

    def test_26_per_suit(self):
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_checkered", rng)
        for suit in (Suit.SPADES, Suit.HEARTS):
            count = sum(1 for c in cards if c.base.suit is suit)
            assert count == 26

    def test_no_clubs_or_diamonds(self):
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_checkered", rng)
        for c in cards:
            assert c.base.suit not in (Suit.CLUBS, Suit.DIAMONDS)

    def test_suit_nominal_updated(self):
        """Suit swap must update suit_nominal, not just the enum."""
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_checkered", rng)
        for c in cards:
            if c.base.suit is Suit.SPADES:
                assert c.base.suit_nominal == 0.04
            elif c.base.suit is Suit.HEARTS:
                assert c.base.suit_nominal == 0.03


# ============================================================================
# Erratic Deck
# ============================================================================

class TestErraticDeck:
    def test_52_cards(self):
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_erratic", rng)
        assert len(cards) == 52

    def test_has_duplicates(self):
        """Erratic deck randomizes, so duplicates are expected."""
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_erratic", rng)
        card_keys = [c.card_key for c in cards]
        assert len(set(card_keys)) < 52  # some duplicates

    def test_deterministic_with_seed(self):
        """Same seed produces same cards."""
        cards1 = build_deck("b_erratic", PseudoRandom("TESTSEED"))
        reset_sort_id_counter()
        cards2 = build_deck("b_erratic", PseudoRandom("TESTSEED"))
        keys1 = [c.card_key for c in cards1]
        keys2 = [c.card_key for c in cards2]
        assert keys1 == keys2

    def test_different_seed_different_cards(self):
        cards1 = build_deck("b_erratic", PseudoRandom("TESTSEED"))
        reset_sort_id_counter()
        cards2 = build_deck("b_erratic", PseudoRandom("OTHERSEED"))
        keys1 = [c.card_key for c in cards1]
        keys2 = [c.card_key for c in cards2]
        assert keys1 != keys2

    def test_sorted_after_randomization(self):
        """Even randomized cards are sorted by s+r concatenation."""
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_erratic", rng)
        keys = [c.card_key for c in cards]
        assert keys == sorted(keys)


# ============================================================================
# Challenge filtering
# ============================================================================

class TestChallengeFiltering:
    def test_explicit_card_list(self):
        """Challenge with explicit cards overrides normal generation."""
        challenge = {
            "deck": {
                "cards": [
                    {"s": "H", "r": "A"},
                    {"s": "S", "r": "K"},
                    {"s": "D", "r": "5", "e": "m_glass"},
                ],
            }
        }
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_red", rng, challenge=challenge)
        assert len(cards) == 3
        assert cards[0].card_key == "D_5"  # sorted first (D < H < S)
        assert cards[0].center_key == "m_glass"
        assert cards[1].card_key == "H_A"
        assert cards[2].card_key == "S_K"

    def test_no_ranks(self):
        """Filter out specific ranks."""
        challenge = {
            "deck": {
                "no_ranks": {"J": True, "Q": True, "K": True},
            }
        }
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_red", rng, challenge=challenge)
        assert len(cards) == 40  # same as Abandoned but via challenge
        for c in cards:
            assert c.base.rank not in (Rank.JACK, Rank.QUEEN, Rank.KING)

    def test_yes_suits(self):
        """Only allow specific suits."""
        challenge = {
            "deck": {
                "yes_suits": {"H": True, "S": True},
            }
        }
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_red", rng, challenge=challenge)
        assert len(cards) == 26  # 2 suits × 13 ranks
        for c in cards:
            assert c.base.suit in (Suit.HEARTS, Suit.SPADES)

    def test_global_enhancement(self):
        """Challenge applies enhancement to all cards."""
        challenge = {
            "deck": {
                "enhancement": "m_glass",
            }
        }
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_red", rng, challenge=challenge)
        assert len(cards) == 52
        for c in cards:
            assert c.center_key == "m_glass"

    def test_global_edition_and_seal(self):
        challenge = {
            "deck": {
                "edition": "foil",
                "gold_seal": "Gold",
            }
        }
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_red", rng, challenge=challenge)
        for c in cards:
            assert c.edition == {"foil": True}
            assert c.seal == "Gold"


# ============================================================================
# Starting params override
# ============================================================================

class TestStartingParams:
    def test_no_faces_via_params(self):
        """starting_params.no_faces overrides back config."""
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_red", rng, starting_params={"no_faces": True})
        assert len(cards) == 40

    def test_erratic_via_params(self):
        rng = PseudoRandom("TESTSEED")
        cards = build_deck("b_red", rng, starting_params={"erratic_suits_and_ranks": True})
        assert len(cards) == 52
        card_keys = [c.card_key for c in cards]
        assert len(set(card_keys)) < 52

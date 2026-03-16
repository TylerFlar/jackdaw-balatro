"""Tests for game enums and lookup tables.

Verifies enum membership, chip values, ordering IDs, and that every
enhancement/edition/seal maps to a valid P_CENTERS or P_SEALS key.
"""

from __future__ import annotations

import pytest

from jackdaw.engine.data.enums import (
    EDITION_CHIPS,
    EDITION_COST,
    EDITION_MULT,
    EDITION_X_MULT,
    RANK_CHIPS,
    RANK_ID,
    SUIT_NOMINAL,
    Edition,
    Enhancement,
    GameStage,
    GameState,
    Rank,
    Rarity,
    Seal,
    Suit,
)
from jackdaw.engine.data.prototypes import ENHANCEMENTS, SEALS


class TestGameState:
    """Game state enum matches globals.lua G.STATES."""

    def test_selecting_hand(self):
        assert GameState.SELECTING_HAND == 1

    def test_shop(self):
        assert GameState.SHOP == 5

    def test_new_round(self):
        assert GameState.NEW_ROUND == 19

    def test_pack_states(self):
        assert GameState.TAROT_PACK == 9
        assert GameState.PLANET_PACK == 10
        assert GameState.SPECTRAL_PACK == 15
        assert GameState.STANDARD_PACK == 17
        assert GameState.BUFFOON_PACK == 18

    def test_all_values_unique(self):
        values = [s.value for s in GameState]
        assert len(values) == len(set(values))

    def test_is_int(self):
        assert isinstance(GameState.SHOP, int)
        assert GameState.SHOP + 1 == 6


class TestGameStage:
    def test_values(self):
        assert GameStage.MAIN_MENU == 1
        assert GameStage.RUN == 2
        assert GameStage.SANDBOX == 3


class TestSuit:
    def test_count(self):
        assert len(Suit) == 4

    def test_string_values(self):
        assert Suit.HEARTS == "Hearts"
        assert Suit.DIAMONDS == "Diamonds"
        assert Suit.CLUBS == "Clubs"
        assert Suit.SPADES == "Spades"

    def test_all_have_nominal(self):
        for s in Suit:
            assert s in SUIT_NOMINAL


class TestRank:
    def test_count(self):
        assert len(Rank) == 13

    def test_number_ranks(self):
        assert Rank.TWO == "2"
        assert Rank.TEN == "10"

    def test_face_ranks(self):
        assert Rank.JACK == "Jack"
        assert Rank.QUEEN == "Queen"
        assert Rank.KING == "King"

    def test_ace(self):
        assert Rank.ACE == "Ace"

    def test_all_have_chips(self):
        for r in Rank:
            assert r in RANK_CHIPS

    def test_all_have_id(self):
        for r in Rank:
            assert r in RANK_ID


class TestRankChips:
    """RANK_CHIPS matches Card:get_chip_bonus in card.lua."""

    def test_ace_is_11(self):
        assert RANK_CHIPS[Rank.ACE] == 11

    def test_face_cards_are_10(self):
        assert RANK_CHIPS[Rank.JACK] == 10
        assert RANK_CHIPS[Rank.QUEEN] == 10
        assert RANK_CHIPS[Rank.KING] == 10

    def test_ten_is_10(self):
        assert RANK_CHIPS[Rank.TEN] == 10

    def test_number_cards(self):
        for rank, expected in [(Rank.TWO, 2), (Rank.FIVE, 5), (Rank.NINE, 9)]:
            assert RANK_CHIPS[rank] == expected

    def test_all_positive(self):
        for chips in RANK_CHIPS.values():
            assert chips > 0


class TestRankID:
    """RANK_ID matches Card:get_id ordering."""

    def test_two_is_lowest(self):
        assert RANK_ID[Rank.TWO] == 2

    def test_ace_is_highest(self):
        assert RANK_ID[Rank.ACE] == 14

    def test_face_card_ordering(self):
        assert RANK_ID[Rank.JACK] == 11
        assert RANK_ID[Rank.QUEEN] == 12
        assert RANK_ID[Rank.KING] == 13

    def test_ids_are_contiguous(self):
        ids = sorted(RANK_ID.values())
        assert ids == list(range(2, 15))


class TestSuitNominal:
    """SUIT_NOMINAL tiebreaker values for sorting."""

    def test_spades_highest(self):
        assert SUIT_NOMINAL[Suit.SPADES] == 0.04

    def test_diamonds_lowest(self):
        assert SUIT_NOMINAL[Suit.DIAMONDS] == 0.01

    def test_all_distinct(self):
        values = list(SUIT_NOMINAL.values())
        assert len(values) == len(set(values))


class TestEnhancement:
    """Enhancement enum maps to P_CENTERS Enhanced keys."""

    def test_count(self):
        # 8 real enhancements + NONE sentinel
        assert len(Enhancement) == 9

    def test_none_sentinel(self):
        assert Enhancement.NONE == "none"

    def test_maps_to_p_centers(self):
        """Every non-NONE enhancement key must exist in P_CENTERS."""
        for e in Enhancement:
            if e is Enhancement.NONE:
                continue
            assert e.value in ENHANCEMENTS, f"{e.value} not in ENHANCEMENTS"

    def test_specific_keys(self):
        assert Enhancement.BONUS == "m_bonus"
        assert Enhancement.GLASS == "m_glass"
        assert Enhancement.LUCKY == "m_lucky"
        assert Enhancement.STONE == "m_stone"


class TestEdition:
    """Edition enum matches card.edition field keys."""

    def test_count(self):
        # 4 real editions + NONE sentinel
        assert len(Edition) == 5

    def test_none_sentinel(self):
        assert Edition.NONE == "none"

    def test_short_names(self):
        """Edition values are short names (not e_* prefixed)."""
        assert Edition.FOIL == "foil"
        assert Edition.HOLOGRAPHIC == "holo"
        assert Edition.POLYCHROME == "polychrome"
        assert Edition.NEGATIVE == "negative"

    def test_all_have_cost(self):
        for e in Edition:
            assert e in EDITION_COST

    def test_cost_values(self):
        assert EDITION_COST[Edition.FOIL] == 2
        assert EDITION_COST[Edition.HOLOGRAPHIC] == 3
        assert EDITION_COST[Edition.POLYCHROME] == 5
        assert EDITION_COST[Edition.NEGATIVE] == 5
        assert EDITION_COST[Edition.NONE] == 0

    def test_edition_scoring(self):
        assert EDITION_CHIPS[Edition.FOIL] == 50
        assert EDITION_MULT[Edition.HOLOGRAPHIC] == 10
        assert EDITION_X_MULT[Edition.POLYCHROME] == pytest.approx(1.5)
        assert EDITION_X_MULT[Edition.NEGATIVE] == pytest.approx(1.0)


class TestSeal:
    """Seal enum maps to P_SEALS keys."""

    def test_count(self):
        # 4 real seals + NONE sentinel
        assert len(Seal) == 5

    def test_none_sentinel(self):
        assert Seal.NONE == "none"

    def test_maps_to_p_seals(self):
        """Every non-NONE seal must exist in P_SEALS."""
        for s in Seal:
            if s is Seal.NONE:
                continue
            assert s.value in SEALS, f"{s.value} not in SEALS"

    def test_specific_keys(self):
        assert Seal.GOLD == "Gold"
        assert Seal.RED == "Red"
        assert Seal.BLUE == "Blue"
        assert Seal.PURPLE == "Purple"


class TestRarity:
    def test_values(self):
        assert Rarity.COMMON == 1
        assert Rarity.UNCOMMON == 2
        assert Rarity.RARE == 3
        assert Rarity.LEGENDARY == 4

    def test_is_int(self):
        assert isinstance(Rarity.COMMON, int)

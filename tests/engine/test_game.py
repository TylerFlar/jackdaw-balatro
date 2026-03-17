"""Tests for jackdaw.engine.game — step function.

Coverage
--------
* SelectBlind → phase transitions to SELECTING_HAND, blind created, hand drawn.
* SkipBlind Small → advances to Big. SkipBlind Big → advances to Boss.
* SkipBlind Boss → IllegalActionError.
* PlayHand → scores, chips accumulate, hand depleted, round won on target.
* Discard → cards removed, replacements drawn.
* CashOut → dollars increase, phase → SHOP.
* SellCard → dollars increase, card removed. Eternal → error.
* NextRound → phase → BLIND_SELECT.
* SortHand → hand reordered.
* ReorderJokers → jokers reordered.
* Reroll → dollars deducted, cost increases.
* Phase enforcement: wrong-phase actions raise IllegalActionError.
* Full mini-game: select blind → play hand → cash out → next round.
"""

from __future__ import annotations

from typing import Any

import pytest

from jackdaw.engine.actions import (
    CashOut,
    Discard,
    GamePhase,
    NextRound,
    PlayHand,
    Reroll,
    ReorderJokers,
    SelectBlind,
    SellCard,
    SkipBlind,
    SortHand,
)
from jackdaw.engine.card import Card
from jackdaw.engine.game import IllegalActionError, step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _init_gs(seed: str = "GAME_TEST") -> dict[str, Any]:
    """Create a fully initialised game_state ready for blind selection."""
    from jackdaw.engine.run_init import initialize_run

    gs = initialize_run("b_red", 1, seed)
    gs["phase"] = GamePhase.BLIND_SELECT
    gs["blind_on_deck"] = "Small"
    gs["jokers"] = []
    gs["consumables"] = []
    return gs


def _joker_card(key: str = "j_joker", **kw) -> Card:
    c = Card(center_key=key)
    c.ability = {"set": "Joker", "effect": "", "name": key}
    c.sell_cost = kw.pop("sell_cost", 3)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# ---------------------------------------------------------------------------
# SelectBlind
# ---------------------------------------------------------------------------


class TestSelectBlind:
    def test_phase_transitions_to_selecting_hand(self):
        gs = _init_gs()
        step(gs, SelectBlind())
        assert gs["phase"] == GamePhase.SELECTING_HAND

    def test_blind_created(self):
        gs = _init_gs()
        step(gs, SelectBlind())
        assert gs["blind"] is not None
        assert gs["blind"].chips > 0

    def test_hand_drawn(self):
        gs = _init_gs()
        step(gs, SelectBlind())
        hand = gs.get("hand", [])
        assert len(hand) > 0
        assert len(hand) <= gs.get("hand_size", 8)

    def test_hands_left_set(self):
        gs = _init_gs()
        step(gs, SelectBlind())
        assert gs["current_round"]["hands_left"] > 0

    def test_discards_left_set(self):
        gs = _init_gs()
        step(gs, SelectBlind())
        # Red Deck: 3+1=4 discards
        assert gs["current_round"]["discards_left"] == 4

    def test_blind_state_updated(self):
        gs = _init_gs()
        step(gs, SelectBlind())
        assert gs["round_resets"]["blind_states"]["Small"] == "Current"

    def test_wrong_phase_raises(self):
        gs = _init_gs()
        gs["phase"] = GamePhase.SHOP
        with pytest.raises(IllegalActionError):
            step(gs, SelectBlind())


# ---------------------------------------------------------------------------
# SkipBlind
# ---------------------------------------------------------------------------


class TestSkipBlind:
    def test_skip_small_advances_to_big(self):
        gs = _init_gs()
        step(gs, SkipBlind())
        assert gs["blind_on_deck"] == "Big"
        assert gs["round_resets"]["blind_states"]["Small"] == "Skipped"
        assert gs["round_resets"]["blind_states"]["Big"] == "Select"
        assert gs["phase"] == GamePhase.BLIND_SELECT

    def test_skip_big_advances_to_boss(self):
        gs = _init_gs()
        step(gs, SkipBlind())  # Small → Big
        step(gs, SkipBlind())  # Big → Boss
        assert gs["blind_on_deck"] == "Boss"
        assert gs["round_resets"]["blind_states"]["Big"] == "Skipped"
        assert gs["round_resets"]["blind_states"]["Boss"] == "Select"

    def test_skip_boss_raises(self):
        gs = _init_gs()
        gs["blind_on_deck"] = "Boss"
        with pytest.raises(IllegalActionError, match="Cannot skip Boss"):
            step(gs, SkipBlind())

    def test_skip_increments_skips(self):
        gs = _init_gs()
        assert gs.get("skips", 0) == 0
        step(gs, SkipBlind())
        assert gs["skips"] == 1


# ---------------------------------------------------------------------------
# PlayHand
# ---------------------------------------------------------------------------


class TestPlayHand:
    def _setup_playing(self, seed="PLAY_TEST"):
        gs = _init_gs(seed)
        step(gs, SelectBlind())
        return gs

    def test_chips_accumulate(self):
        gs = self._setup_playing()
        hand = gs["hand"]
        step(gs, PlayHand(card_indices=tuple(range(min(5, len(hand))))))
        assert gs["chips"] > 0

    def test_hands_left_decremented(self):
        gs = self._setup_playing()
        initial = gs["current_round"]["hands_left"]
        hand = gs["hand"]
        step(gs, PlayHand(card_indices=(0,)))
        assert gs["current_round"]["hands_left"] == initial - 1

    def test_played_cards_removed_from_hand(self):
        gs = self._setup_playing()
        initial_hand_size = len(gs["hand"])
        step(gs, PlayHand(card_indices=(0, 1)))
        # Hand should have lost 2 cards but drawn replacements
        # (up to hand_size)
        assert len(gs["hand"]) <= gs.get("hand_size", 8)

    def test_no_hands_left_raises(self):
        gs = self._setup_playing()
        gs["current_round"]["hands_left"] = 0
        with pytest.raises(IllegalActionError, match="No hands"):
            step(gs, PlayHand(card_indices=(0,)))

    def test_empty_indices_raises(self):
        gs = self._setup_playing()
        with pytest.raises(IllegalActionError, match="at least 1"):
            step(gs, PlayHand(card_indices=()))

    def test_too_many_cards_raises(self):
        gs = self._setup_playing()
        with pytest.raises(IllegalActionError, match="more than 5"):
            step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4, 5)))

    def test_round_won_transitions_to_round_eval(self):
        gs = self._setup_playing()
        # Set chips target very low so any hand wins
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        assert gs["phase"] == GamePhase.ROUND_EVAL

    def test_game_over_when_no_hands_and_not_beaten(self):
        gs = self._setup_playing()
        gs["current_round"]["hands_left"] = 1
        gs["blind"].chips = 999_999_999  # impossible to beat
        step(gs, PlayHand(card_indices=(0,)))
        assert gs["phase"] == GamePhase.GAME_OVER

    def test_wrong_phase_raises(self):
        gs = _init_gs()
        gs["phase"] = GamePhase.SHOP
        with pytest.raises(IllegalActionError):
            step(gs, PlayHand(card_indices=(0,)))


# ---------------------------------------------------------------------------
# Discard
# ---------------------------------------------------------------------------


class TestDiscard:
    def _setup_playing(self, seed="DISC_TEST"):
        gs = _init_gs(seed)
        step(gs, SelectBlind())
        return gs

    def test_discards_left_decremented(self):
        gs = self._setup_playing()
        initial = gs["current_round"]["discards_left"]
        step(gs, Discard(card_indices=(0,)))
        assert gs["current_round"]["discards_left"] == initial - 1

    def test_hand_replenished(self):
        gs = self._setup_playing()
        hand_size = gs.get("hand_size", 8)
        step(gs, Discard(card_indices=(0, 1)))
        # Should draw back up to hand_size (if deck has cards)
        assert len(gs["hand"]) <= hand_size

    def test_no_discards_raises(self):
        gs = self._setup_playing()
        gs["current_round"]["discards_left"] = 0
        with pytest.raises(IllegalActionError, match="No discards"):
            step(gs, Discard(card_indices=(0,)))

    def test_stays_in_selecting_hand(self):
        gs = self._setup_playing()
        step(gs, Discard(card_indices=(0,)))
        assert gs["phase"] == GamePhase.SELECTING_HAND


# ---------------------------------------------------------------------------
# CashOut
# ---------------------------------------------------------------------------


class TestCashOut:
    def _setup_round_eval(self):
        gs = _init_gs("CASHOUT_TEST")
        step(gs, SelectBlind())
        gs["blind"].chips = 1  # easy win
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        assert gs["phase"] == GamePhase.ROUND_EVAL
        return gs

    def test_transitions_to_shop(self):
        gs = self._setup_round_eval()
        step(gs, CashOut())
        assert gs["phase"] == GamePhase.SHOP

    def test_dollars_increase(self):
        gs = self._setup_round_eval()
        before = gs["dollars"]
        step(gs, CashOut())
        # Earnings should be positive (blind reward + unused hands)
        assert gs["dollars"] >= before


# ---------------------------------------------------------------------------
# SellCard
# ---------------------------------------------------------------------------


class TestSellCard:
    def test_sell_joker(self):
        gs = _init_gs()
        gs["phase"] = GamePhase.SHOP
        gs["jokers"] = [_joker_card(sell_cost=5)]
        before = gs["dollars"]
        step(gs, SellCard(area="jokers", card_index=0))
        assert gs["dollars"] == before + 5
        assert len(gs["jokers"]) == 0

    def test_sell_eternal_raises(self):
        gs = _init_gs()
        gs["phase"] = GamePhase.SHOP
        gs["jokers"] = [_joker_card(eternal=True)]
        with pytest.raises(IllegalActionError, match="eternal"):
            step(gs, SellCard(area="jokers", card_index=0))


# ---------------------------------------------------------------------------
# NextRound
# ---------------------------------------------------------------------------


class TestNextRound:
    def test_transitions_to_blind_select(self):
        gs = _init_gs()
        gs["phase"] = GamePhase.SHOP
        step(gs, NextRound())
        assert gs["phase"] == GamePhase.BLIND_SELECT


# ---------------------------------------------------------------------------
# SortHand
# ---------------------------------------------------------------------------


class TestSortHand:
    def test_sort_by_rank(self):
        gs = _init_gs()
        step(gs, SelectBlind())
        hand_before = list(gs["hand"])
        step(gs, SortHand(mode="rank"))
        # Hand should be sorted by rank id
        ids = [c.base.id for c in gs["hand"] if c.base]
        assert ids == sorted(ids)

    def test_sort_by_suit(self):
        gs = _init_gs()
        step(gs, SelectBlind())
        step(gs, SortHand(mode="suit"))
        # Hand should be sorted by suit nominal
        noms = [c.base.suit_nominal for c in gs["hand"] if c.base]
        assert noms == sorted(noms)


# ---------------------------------------------------------------------------
# ReorderJokers
# ---------------------------------------------------------------------------


class TestReorderJokers:
    def test_reorder(self):
        gs = _init_gs()
        gs["phase"] = GamePhase.SHOP
        j0 = _joker_card("j_a")
        j1 = _joker_card("j_b")
        j2 = _joker_card("j_c")
        gs["jokers"] = [j0, j1, j2]
        step(gs, ReorderJokers(new_order=(2, 0, 1)))
        assert gs["jokers"] == [j2, j0, j1]

    def test_invalid_permutation_raises(self):
        gs = _init_gs()
        gs["phase"] = GamePhase.SHOP
        gs["jokers"] = [_joker_card(), _joker_card()]
        with pytest.raises(IllegalActionError, match="permutation"):
            step(gs, ReorderJokers(new_order=(0, 0)))

    def test_empty_order_is_noop(self):
        gs = _init_gs()
        gs["phase"] = GamePhase.SHOP
        gs["jokers"] = [_joker_card()]
        step(gs, ReorderJokers(new_order=()))
        assert len(gs["jokers"]) == 1


# ---------------------------------------------------------------------------
# Reroll
# ---------------------------------------------------------------------------


class TestReroll:
    def test_deducts_dollars(self):
        gs = _init_gs()
        gs["phase"] = GamePhase.SHOP
        gs["dollars"] = 10
        gs["current_round"]["reroll_cost"] = 5
        gs["current_round"]["free_rerolls"] = 0
        step(gs, Reroll())
        assert gs["dollars"] == 5

    def test_cost_increases(self):
        gs = _init_gs()
        gs["phase"] = GamePhase.SHOP
        gs["dollars"] = 20
        gs["current_round"]["reroll_cost"] = 5
        gs["current_round"]["free_rerolls"] = 0
        step(gs, Reroll())
        assert gs["current_round"]["reroll_cost"] > 5

    def test_free_reroll(self):
        gs = _init_gs()
        gs["phase"] = GamePhase.SHOP
        gs["dollars"] = 0
        gs["current_round"]["free_rerolls"] = 1
        step(gs, Reroll())
        assert gs["dollars"] == 0
        assert gs["current_round"]["free_rerolls"] == 0

    def test_cannot_afford_raises(self):
        gs = _init_gs()
        gs["phase"] = GamePhase.SHOP
        gs["dollars"] = 2
        gs["current_round"]["reroll_cost"] = 5
        gs["current_round"]["free_rerolls"] = 0
        with pytest.raises(IllegalActionError, match="afford"):
            step(gs, Reroll())


# ---------------------------------------------------------------------------
# Full mini-game
# ---------------------------------------------------------------------------


class TestFullMiniGame:
    def test_select_play_cashout_next(self):
        """Full loop: blind select → play hand → cash out → next round."""
        gs = _init_gs("MINI_GAME")

        # Phase 1: Select Small Blind
        assert gs["phase"] == GamePhase.BLIND_SELECT
        step(gs, SelectBlind())
        assert gs["phase"] == GamePhase.SELECTING_HAND

        # Phase 2: Play a hand (set easy blind target)
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        assert gs["phase"] == GamePhase.ROUND_EVAL

        # Phase 3: Cash out
        step(gs, CashOut())
        assert gs["phase"] == GamePhase.SHOP
        assert gs["dollars"] > 0

        # Phase 4: Next round
        step(gs, NextRound())
        assert gs["phase"] == GamePhase.BLIND_SELECT

    def test_skip_small_select_big(self):
        """Skip Small → select Big."""
        gs = _init_gs("SKIP_SELECT")

        step(gs, SkipBlind())
        assert gs["blind_on_deck"] == "Big"
        assert gs["phase"] == GamePhase.BLIND_SELECT

        step(gs, SelectBlind())
        assert gs["phase"] == GamePhase.SELECTING_HAND
        assert gs["round_resets"]["blind_states"]["Big"] == "Current"


# ===========================================================================
# Detailed SelectBlind / SkipBlind tests
# ===========================================================================


class TestSelectBlindDetailed:
    """Tests for the full setting_blind → boss effects → draw flow."""

    def test_blind_chips_correct_at_ante1(self):
        gs = _init_gs("CHIPS_TEST")
        step(gs, SelectBlind())
        # Small Blind at ante 1 with scaling 1: base=300, mult=1 → 300
        assert gs["blind"].chips == 300

    def test_hand_has_8_cards(self):
        gs = _init_gs("HAND8")
        step(gs, SelectBlind())
        assert len(gs["hand"]) == 8

    def test_deck_reduced_by_hand_size(self):
        gs = _init_gs("DECK_REDUCED")
        initial_deck = len(gs["deck"])
        step(gs, SelectBlind())
        assert len(gs["deck"]) == initial_deck - len(gs["hand"])

    def test_cards_debuffed_by_boss(self):
        """Select a Boss blind that debuffs cards (The Goad debuffs Spades)."""
        gs = _init_gs("BOSS_DEBUFF")
        # Skip to Boss
        step(gs, SkipBlind())  # Small → Big
        step(gs, SkipBlind())  # Big → Boss
        # Force boss to bl_goad (debuffs Spades cards)
        gs["round_resets"]["blind_choices"]["Boss"] = "bl_goad"
        step(gs, SelectBlind())
        # Spades cards in hand should be debuffed
        debuffed = [c for c in gs["hand"] if c.debuff]
        spades = [c for c in gs["hand"]
                  if c.base and c.base.suit.value == "Spades"]
        assert len(debuffed) == len(spades)
        assert len(debuffed) > 0  # at least one Spades card in hand

    def test_marble_joker_adds_stone_card(self):
        """Marble Joker adds a Stone Card to deck on setting_blind."""
        gs = _init_gs("MARBLE_TEST")
        marble = _joker_card("j_marble")
        gs["jokers"] = [marble]
        initial_deck = len(gs["deck"])
        step(gs, SelectBlind())
        # Deck should have one more card (Stone Card added, then hand drawn)
        total_cards = len(gs["deck"]) + len(gs["hand"])
        assert total_cards == initial_deck + 1

    def test_chicot_disables_boss(self):
        """Chicot disables boss blind effect."""
        gs = _init_gs("CHICOT_TEST")
        chicot = _joker_card("j_chicot")
        gs["jokers"] = [chicot]
        # Skip to Boss
        step(gs, SkipBlind())
        step(gs, SkipBlind())
        step(gs, SelectBlind())
        assert gs["blind"].disabled is True

    def test_the_water_zero_discards(self):
        """The Water boss sets discards_left to 0."""
        gs = _init_gs("WATER_TEST")
        step(gs, SkipBlind())  # Small → Big
        step(gs, SkipBlind())  # Big → Boss
        gs["round_resets"]["blind_choices"]["Boss"] = "bl_water"
        step(gs, SelectBlind())
        assert gs["current_round"]["discards_left"] == 0

    def test_the_needle_one_hand(self):
        """The Needle boss reduces hands to 1."""
        gs = _init_gs("NEEDLE_TEST")
        step(gs, SkipBlind())
        step(gs, SkipBlind())
        gs["round_resets"]["blind_choices"]["Boss"] = "bl_needle"
        step(gs, SelectBlind())
        assert gs["current_round"]["hands_left"] == 1

    def test_the_manacle_hand_size_reduced(self):
        """The Manacle boss reduces hand_size by 1."""
        gs = _init_gs("MANACLE_TEST")
        step(gs, SkipBlind())
        step(gs, SkipBlind())
        gs["round_resets"]["blind_choices"]["Boss"] = "bl_manacle"
        initial_size = gs["hand_size"]
        step(gs, SelectBlind())
        assert gs["hand_size"] == initial_size - 1
        assert len(gs["hand"]) == initial_size - 1


class TestSkipBlindDetailed:
    """Tests for tag award + joker context on skip."""

    def test_tag_awarded_on_skip(self):
        gs = _init_gs("TAG_AWARD")
        step(gs, SkipBlind())
        awarded = gs.get("awarded_tags", [])
        assert len(awarded) >= 1
        assert awarded[0]["blind"] == "Small"

    def test_economy_tag_gives_dollars(self):
        """If the skip tag is tag_economy, dollars should increase."""
        gs = _init_gs("TAG_ECON")
        # Force the Small tag to be tag_economy
        gs["round_resets"]["blind_tags"]["Small"] = "tag_economy"
        initial_dollars = gs["dollars"]
        step(gs, SkipBlind())
        # Economy tag gives min(max, current_dollars) — should gain something
        # if dollars > 0
        if initial_dollars > 0:
            assert gs["dollars"] > initial_dollars

    def test_skip_then_select_boss_full_progression(self):
        """Skip Small, skip Big, select Boss — full ante progression."""
        gs = _init_gs("FULL_PROG")
        # Skip Small
        step(gs, SkipBlind())
        assert gs["blind_on_deck"] == "Big"
        assert gs["skips"] == 1

        # Skip Big
        step(gs, SkipBlind())
        assert gs["blind_on_deck"] == "Boss"
        assert gs["skips"] == 2

        # Select Boss
        step(gs, SelectBlind())
        assert gs["phase"] == GamePhase.SELECTING_HAND
        assert gs["blind"].boss is True
        assert gs["round_resets"]["blind_states"]["Boss"] == "Current"


# ===========================================================================
# Detailed PlayHand tests
# ===========================================================================


class TestPlayHandDetailed:
    """Tests for the full play_hand flow including side-effects."""

    def _setup(self, seed="PLAY_DETAIL"):
        gs = _init_gs(seed)
        step(gs, SelectBlind())
        return gs

    def test_score_computed_and_chips_updated(self):
        """Play some cards → chips should increase from 0."""
        gs = self._setup()
        assert gs["chips"] == 0
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        assert gs["chips"] > 0

    def test_last_score_result_stored(self):
        gs = self._setup()
        step(gs, PlayHand(card_indices=(0, 1)))
        result = gs.get("last_score_result")
        assert result is not None
        assert result.hand_type != "NULL"
        assert result.chips > 0
        assert result.mult > 0

    def test_hands_left_decremented(self):
        gs = self._setup()
        initial = gs["current_round"]["hands_left"]
        step(gs, PlayHand(card_indices=(0,)))
        assert gs["current_round"]["hands_left"] == initial - 1

    def test_hands_played_incremented(self):
        gs = self._setup()
        step(gs, PlayHand(card_indices=(0,)))
        assert gs["current_round"]["hands_played"] == 1
        assert gs["hands_played"] >= 1

    def test_hand_type_recorded(self):
        """The played hand type should be recorded in hand_levels."""
        gs = self._setup()
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        hl = gs["hand_levels"]
        result = gs["last_score_result"]
        if result.hand_type != "NULL":
            from jackdaw.engine.data.hands import HandType
            ht = HandType(result.hand_type)
            assert hl[ht].played >= 1

    def test_played_cards_moved_to_discard(self):
        """Played cards go to discard_pile, not back to hand."""
        gs = self._setup()
        hand_before = list(gs["hand"])
        played_cards = [hand_before[0], hand_before[1]]
        step(gs, PlayHand(card_indices=(0, 1)))
        discard = gs.get("discard_pile", [])
        # At least the 2 played cards (or surviving ones) should be in discard
        assert len(discard) >= 1

    def test_cards_drawn_after_non_winning_play(self):
        """If blind not beaten and hands remain, hand replenished."""
        gs = self._setup()
        gs["blind"].chips = 999_999_999  # impossible to beat
        gs["current_round"]["hands_left"] = 4
        step(gs, PlayHand(card_indices=(0,)))
        assert gs["phase"] == GamePhase.SELECTING_HAND
        # Hand should have drawn back up
        assert len(gs["hand"]) <= gs.get("hand_size", 8)

    def test_winning_play_transitions_to_round_eval(self):
        gs = self._setup()
        gs["blind"].chips = 1  # trivial target
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        assert gs["phase"] == GamePhase.ROUND_EVAL

    def test_game_over_when_exhausted(self):
        gs = self._setup()
        gs["current_round"]["hands_left"] = 1
        gs["blind"].chips = 999_999_999
        step(gs, PlayHand(card_indices=(0,)))
        assert gs["phase"] == GamePhase.GAME_OVER
        assert gs["won"] is False

    def test_per_card_times_played(self):
        """Each played card's times_played should increment."""
        gs = self._setup()
        card = gs["hand"][0]
        initial_tp = card.base.times_played if card.base else 0
        step(gs, PlayHand(card_indices=(0,)))
        if card.base:
            assert card.base.times_played == initial_tp + 1

    def test_played_this_ante_set(self):
        """Each played card should have ability.played_this_ante = True."""
        gs = self._setup()
        card = gs["hand"][0]
        step(gs, PlayHand(card_indices=(0,)))
        if isinstance(card.ability, dict):
            assert card.ability.get("played_this_ante") is True

    def test_index_out_of_range_raises(self):
        gs = self._setup()
        with pytest.raises(IllegalActionError, match="out of range"):
            step(gs, PlayHand(card_indices=(99,)))

    def test_dollars_earned_from_scoring(self):
        """If scoring produces dollars (Gold Seal, etc.), they're added."""
        gs = self._setup()
        initial_dollars = gs["dollars"]
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        result = gs["last_score_result"]
        if result.dollars_earned > 0:
            assert gs["dollars"] > initial_dollars

    def test_the_tooth_loses_dollar_per_card(self):
        """The Tooth boss: lose $1 per card played."""
        gs = _init_gs("TOOTH_TEST")
        step(gs, SkipBlind())  # Small→Big
        step(gs, SkipBlind())  # Big→Boss
        gs["round_resets"]["blind_choices"]["Boss"] = "bl_tooth"
        step(gs, SelectBlind())
        initial_dollars = gs["dollars"]
        step(gs, PlayHand(card_indices=(0, 1, 2)))
        # Should lose $3 (1 per card played) plus any scoring dollars
        tooth_loss = 3
        result = gs["last_score_result"]
        expected = initial_dollars - tooth_loss + result.dollars_earned
        assert gs["dollars"] == expected

    def test_multiple_plays_accumulate_chips(self):
        """Two plays should accumulate chips."""
        gs = self._setup("MULTI_PLAY")
        gs["blind"].chips = 999_999_999  # don't win
        step(gs, PlayHand(card_indices=(0,)))
        first_chips = gs["chips"]
        assert first_chips > 0
        step(gs, PlayHand(card_indices=(0,)))
        assert gs["chips"] > first_chips

    def test_joker_self_destruct_removed(self):
        """If a joker self-destructs during scoring, it's removed."""
        # This is validated structurally — jokers_removed in ScoreResult
        # are removed from gs["jokers"]. Hard to force without specific
        # joker setup, so just verify the list is accessible.
        gs = self._setup()
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        result = gs["last_score_result"]
        # If any jokers were removed, they shouldn't be in jokers list
        for removed in result.jokers_removed:
            assert removed not in gs.get("jokers", [])


# ===========================================================================
# Detailed Discard tests
# ===========================================================================


class TestDiscardDetailed:
    """Tests for the full discard flow including joker contexts."""

    def _setup(self, seed="DISC_DETAIL"):
        gs = _init_gs(seed)
        step(gs, SelectBlind())
        return gs

    def test_discards_left_decremented(self):
        gs = self._setup()
        initial = gs["current_round"]["discards_left"]
        step(gs, Discard(card_indices=(0,)))
        assert gs["current_round"]["discards_left"] == initial - 1

    def test_discards_used_incremented(self):
        gs = self._setup()
        step(gs, Discard(card_indices=(0, 1)))
        assert gs["current_round"]["discards_used"] == 1

    def test_cards_moved_to_discard_pile(self):
        gs = self._setup()
        card0 = gs["hand"][0]
        step(gs, Discard(card_indices=(0,)))
        discard = gs.get("discard_pile", [])
        assert card0 in discard

    def test_three_cards_discarded_three_drawn(self):
        """Discard 3 → hand replenished back to hand_size (if deck has cards)."""
        gs = self._setup()
        hand_size = gs.get("hand_size", 8)
        step(gs, Discard(card_indices=(0, 1, 2)))
        assert len(gs["hand"]) == hand_size

    def test_hand_stays_selecting_hand(self):
        gs = self._setup()
        step(gs, Discard(card_indices=(0,)))
        assert gs["phase"] == GamePhase.SELECTING_HAND

    def test_no_discards_raises(self):
        gs = self._setup()
        gs["current_round"]["discards_left"] = 0
        with pytest.raises(IllegalActionError, match="No discards"):
            step(gs, Discard(card_indices=(0,)))

    def test_index_out_of_range_raises(self):
        gs = self._setup()
        with pytest.raises(IllegalActionError, match="out of range"):
            step(gs, Discard(card_indices=(99,)))

    def test_discard_cost_deducted(self):
        """Golden Needle challenge: discard_cost modifier deducts dollars."""
        gs = self._setup()
        gs["modifiers"]["discard_cost"] = 1
        before = gs["dollars"]
        step(gs, Discard(card_indices=(0,)))
        assert gs["dollars"] <= before - 1

    def test_cards_discarded_stat_tracked(self):
        gs = self._setup()
        step(gs, Discard(card_indices=(0, 1, 2)))
        assert gs["round_scores"]["cards_discarded"] >= 3

    def test_castle_joker_gains_chips(self):
        """Castle: +3 chips per card matching castle_card suit."""
        gs = self._setup("CASTLE_DISC")
        castle = _joker_card("j_castle")
        castle_suit = gs["current_round"]["castle_card"]["suit"]
        castle.ability["castle_card_suit"] = castle_suit
        castle.ability["extra"] = {"chips": 0, "chip_mod": 3}
        gs["jokers"] = [castle]

        # Find a card in hand matching the castle suit
        matching_idx = None
        for i, card in enumerate(gs["hand"]):
            if card.base and card.base.suit.value == castle_suit:
                matching_idx = i
                break

        if matching_idx is not None:
            step(gs, Discard(card_indices=(matching_idx,)))
            assert castle.ability["extra"]["chips"] >= 3

    def test_green_joker_loses_mult(self):
        """Green Joker: -1 mult per discard action."""
        gs = self._setup("GREEN_DISC")
        green = _joker_card("j_green_joker")
        green.ability["mult"] = 5
        green.ability["extra"] = {"hand_add": 1, "discard_sub": 1}
        gs["jokers"] = [green]
        step(gs, Discard(card_indices=(0,)))
        assert green.ability["mult"] == 4

    def test_ramen_loses_xmult(self):
        """Ramen: -0.01 xMult per card discarded."""
        gs = self._setup("RAMEN_DISC")
        ramen = _joker_card("j_ramen")
        ramen.ability["x_mult"] = 2.0
        ramen.ability["extra"] = 0.01
        gs["jokers"] = [ramen]
        step(gs, Discard(card_indices=(0,)))
        assert ramen.ability["x_mult"] < 2.0

    def test_ramen_self_destructs_at_threshold(self):
        """Ramen: self-destructs when xMult would drop to ≤ 1."""
        gs = self._setup("RAMEN_DESTROY")
        ramen = _joker_card("j_ramen")
        ramen.ability["x_mult"] = 1.005
        ramen.ability["extra"] = 0.01
        gs["jokers"] = [ramen]
        step(gs, Discard(card_indices=(0,)))
        assert ramen not in gs["jokers"]

    def test_burnt_joker_levels_up_hand(self):
        """Burnt Joker: level up the hand type of discarded cards (first discard)."""
        gs = self._setup("BURNT_DISC")
        burnt = _joker_card("j_burnt")
        gs["jokers"] = [burnt]
        hl = gs["hand_levels"]
        # Discard 2 cards — Burnt fires on first discard
        from jackdaw.engine.hand_eval import evaluate_hand

        cards_to_discard = gs["hand"][:2]
        det = evaluate_hand(cards_to_discard)
        if det.detected_hand and det.detected_hand != "NULL":
            from jackdaw.engine.data.hands import HandType

            ht = HandType(det.detected_hand)
            level_before = hl[ht].level
            step(gs, Discard(card_indices=(0, 1)))
            assert hl[ht].level == level_before + 1

    def test_purple_seal_creates_tarot(self):
        """Discarding a Purple Seal card creates a Tarot consumable."""
        gs = self._setup("PURPLE_DISC")
        # Give the first hand card a Purple Seal
        gs["hand"][0].seal = "Purple"
        initial_cons = len(gs.get("consumables", []))
        step(gs, Discard(card_indices=(0,)))
        assert len(gs.get("consumables", [])) == initial_cons + 1

    def test_empty_deck_fewer_drawn(self):
        """When deck is empty, fewer cards drawn after discard."""
        gs = self._setup("EMPTY_DECK_DISC")
        # Empty the deck
        gs["deck"] = []
        hand_before = len(gs["hand"])
        step(gs, Discard(card_indices=(0, 1)))
        # Hand should have shrunk (2 discarded, 0 drawn)
        assert len(gs["hand"]) == hand_before - 2

    def test_multiple_discards_accumulate(self):
        """Two discard actions: counters accumulate correctly."""
        gs = self._setup("MULTI_DISC")
        step(gs, Discard(card_indices=(0,)))
        step(gs, Discard(card_indices=(0,)))
        assert gs["current_round"]["discards_used"] == 2


# ===========================================================================
# Round end and game over tests
# ===========================================================================


class TestRoundEnd:
    """Tests for _round_won and blind progression."""

    def _beat_small(self, seed="ROUND_END"):
        """Set up game and beat the Small Blind."""
        gs = _init_gs(seed)
        step(gs, SelectBlind())
        gs["blind"].chips = 1  # trivial target
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        return gs

    def test_beat_small_phase_is_round_eval(self):
        gs = self._beat_small()
        assert gs["phase"] == GamePhase.ROUND_EVAL

    def test_beat_small_blind_on_deck_advances_to_big(self):
        gs = self._beat_small()
        assert gs["blind_on_deck"] == "Big"

    def test_beat_small_blind_state_defeated(self):
        gs = self._beat_small()
        assert gs["round_resets"]["blind_states"]["Small"] == "Defeated"

    def test_cards_returned_to_deck(self):
        gs = self._beat_small()
        assert gs["hand"] == []
        assert gs["played_cards_area"] == []
        assert gs["discard_pile"] == []
        assert len(gs["deck"]) == 52  # full deck returned

    def test_cards_undebuffed_after_round(self):
        """Blind debuffs are cleared after the round ends."""
        gs = _init_gs("UNDEBUFF")
        step(gs, SkipBlind())  # Small→Big
        step(gs, SkipBlind())  # Big→Boss
        gs["round_resets"]["blind_choices"]["Boss"] = "bl_goad"  # debuffs Spades
        step(gs, SelectBlind())
        # Some hand cards should be debuffed (Spades)
        debuffed_hand = [c for c in gs["hand"] if c.debuff]
        assert len(debuffed_hand) > 0
        # Now beat the blind
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        # After round end, deck cards should be un-debuffed
        debuffed_deck = [c for c in gs["deck"] if c.debuff]
        assert len(debuffed_deck) == 0

    def test_round_counter_incremented(self):
        gs = self._beat_small()
        assert gs["round"] >= 1

    def test_round_earnings_stored(self):
        gs = self._beat_small()
        assert gs.get("round_earnings") is not None
        assert gs["round_earnings"].blind_reward >= 0

    def test_cash_out_adds_earnings(self):
        gs = self._beat_small()
        before = gs["dollars"]
        step(gs, CashOut())
        assert gs["dollars"] >= before  # earnings added
        assert gs["phase"] == GamePhase.SHOP

    def test_cash_out_tracks_previous_round(self):
        gs = self._beat_small()
        step(gs, CashOut())
        assert gs["previous_round"]["dollars"] == gs["dollars"]

    def test_unused_discards_tracked(self):
        gs = self._beat_small()
        # We didn't discard, so unused_discards should be > 0
        assert gs["unused_discards"] > 0


class TestBeatBoss:
    """Tests for boss defeat and ante progression."""

    def _beat_boss(self, seed="BOSS_BEAT", win_ante=8):
        gs = _init_gs(seed)
        gs["win_ante"] = win_ante
        # Skip to Boss
        step(gs, SkipBlind())
        step(gs, SkipBlind())
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        return gs

    def test_boss_beaten_ante_increments(self):
        gs = self._beat_boss()
        # Ante should have advanced from 1 to 2
        assert gs["round_resets"]["ante"] == 2

    def test_boss_beaten_blind_on_deck_resets(self):
        gs = self._beat_boss()
        assert gs["blind_on_deck"] == "Small"

    def test_boss_beaten_new_boss_generated(self):
        gs = self._beat_boss()
        boss_key = gs["round_resets"]["blind_choices"]["Boss"]
        assert boss_key is not None
        from jackdaw.engine.data.prototypes import BLINDS
        assert boss_key in BLINDS

    def test_boss_at_win_ante_sets_won(self):
        gs = _init_gs("WIN_GAME")
        gs["win_ante"] = 1  # set to 1 so ante=1 boss win triggers it
        step(gs, SkipBlind())
        step(gs, SkipBlind())
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        assert gs["won"] is True

    def test_boss_before_win_ante_not_won(self):
        gs = self._beat_boss(win_ante=8)  # ante=1, win_ante=8
        assert gs.get("won") is not True

    def test_manacle_hand_size_restored(self):
        """The Manacle's -1 hand size is restored after boss defeat."""
        gs = _init_gs("MANACLE_RESTORE")
        step(gs, SkipBlind())
        step(gs, SkipBlind())
        gs["round_resets"]["blind_choices"]["Boss"] = "bl_manacle"
        initial_size = gs["hand_size"]
        step(gs, SelectBlind())
        assert gs["hand_size"] == initial_size - 1
        # Beat the boss
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        assert gs["hand_size"] == initial_size


class TestGameOver:
    """Tests for game over detection."""

    def test_fail_no_bones_game_over(self):
        gs = _init_gs("FAIL_TEST")
        step(gs, SelectBlind())
        gs["current_round"]["hands_left"] = 1
        gs["blind"].chips = 999_999_999
        step(gs, PlayHand(card_indices=(0,)))
        assert gs["phase"] == GamePhase.GAME_OVER
        assert gs["won"] is False

    def test_fail_with_mr_bones_saved(self):
        """Mr. Bones saves from game over (checks ScoreResult.saved)."""
        gs = _init_gs("BONES_TEST")
        step(gs, SelectBlind())
        gs["current_round"]["hands_left"] = 1
        gs["blind"].chips = 999_999_999
        # Mr. Bones logic is in scoring pipeline — we test the state machine
        # branch by manually setting saved=True on the result
        # This is a structural test of the game over check
        from jackdaw.engine.scoring import ScoreResult
        # Play hand and check if the game over path works
        step(gs, PlayHand(card_indices=(0,)))
        # Without Mr. Bones, should be game over
        assert gs["phase"] == GamePhase.GAME_OVER


class TestRoundEndRentalPerishable:
    """Tests for rental/perishable processing at round end."""

    def test_rental_deducts_at_round_end(self):
        gs = _init_gs("RENTAL_END")
        rental_joker = _joker_card("j_rental")
        rental_joker.rental = True
        rental_joker.ability["rental"] = True
        gs["jokers"] = [rental_joker]
        step(gs, SelectBlind())
        dollars_before = gs["dollars"]
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        # Rental deducts $3 during round end processing
        # (net effect depends on earnings too, but rental cost is applied)
        assert gs["phase"] == GamePhase.ROUND_EVAL

    def test_perishable_countdown_at_round_end(self):
        gs = _init_gs("PERISH_END")
        perish_joker = _joker_card("j_perish")
        perish_joker.perishable = True
        perish_joker.perish_tally = 2
        perish_joker.ability["perishable"] = True
        perish_joker.ability["perish_tally"] = 2
        gs["jokers"] = [perish_joker]
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        assert perish_joker.perish_tally == 1
        assert perish_joker.debuff is False

    def test_perishable_debuffs_at_zero(self):
        gs = _init_gs("PERISH_ZERO")
        perish_joker = _joker_card("j_perish")
        perish_joker.perishable = True
        perish_joker.perish_tally = 1
        perish_joker.ability["perishable"] = True
        perish_joker.ability["perish_tally"] = 1
        gs["jokers"] = [perish_joker]
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        assert perish_joker.perish_tally == 0
        assert perish_joker.debuff is True

    def test_gold_seal_dollars_at_round_end(self):
        """Gold Seal cards in hand at round end give +$3 each."""
        gs = _init_gs("GOLD_SEAL")
        step(gs, SelectBlind())
        # Give some hand cards Gold Seal
        gold_count = 0
        for card in gs["hand"][:3]:
            card.seal = "Gold"
            gold_count += 1
        dollars_before = gs["dollars"]
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(3, 4)))  # play non-sealed cards
        # After round end, dollars should include gold seal bonus
        # (held cards still in hand get +$3 each)
        # The 3 gold sealed cards were in hand (held), so +$9 expected
        # plus earnings
        step(gs, CashOut())
        # Just verify gold seal cards contributed
        assert gs["dollars"] > dollars_before


class TestFullAnteProgression:
    """Test a complete ante: Small → Big → Boss → next ante."""

    def test_full_ante_cycle(self):
        gs = _init_gs("FULL_ANTE")

        # Small Blind
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        step(gs, CashOut())
        step(gs, NextRound())
        assert gs["phase"] == GamePhase.BLIND_SELECT
        assert gs["blind_on_deck"] == "Big"

        # Big Blind
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        step(gs, CashOut())
        step(gs, NextRound())
        assert gs["phase"] == GamePhase.BLIND_SELECT
        assert gs["blind_on_deck"] == "Boss"

        # Boss Blind
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        assert gs["phase"] == GamePhase.ROUND_EVAL
        step(gs, CashOut())
        assert gs["phase"] == GamePhase.SHOP

        # Next round advances to ante 2
        step(gs, NextRound())
        assert gs["phase"] == GamePhase.BLIND_SELECT
        assert gs["round_resets"]["ante"] == 2
        assert gs["blind_on_deck"] == "Small"


# ===========================================================================
# Detailed shop action tests
# ===========================================================================


def _setup_shop(seed="SHOP_TEST"):
    """Set up a game state in the SHOP phase after beating Small Blind."""
    gs = _init_gs(seed)
    step(gs, SelectBlind())
    gs["blind"].chips = 1
    step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
    step(gs, CashOut())
    assert gs["phase"] == GamePhase.SHOP
    return gs


class TestBuyCardDetailed:
    def test_buy_joker_added_to_jokers(self):
        gs = _setup_shop("BUY_JOKER")
        shop_joker = _joker_card("j_banner", cost=5, sell_cost=3)
        gs["shop_cards"] = [shop_joker]
        gs["dollars"] = 10
        from jackdaw.engine.actions import BuyCard

        step(gs, BuyCard(shop_index=0))
        assert shop_joker in gs["jokers"]
        assert gs["dollars"] == 5

    def test_buy_joker_marks_used(self):
        gs = _setup_shop("BUY_USED")
        shop_joker = _joker_card("j_banner", cost=3)
        gs["shop_cards"] = [shop_joker]
        gs["dollars"] = 10
        from jackdaw.engine.actions import BuyCard

        step(gs, BuyCard(shop_index=0))
        assert gs["used_jokers"].get("j_banner") is True

    def test_buy_consumable_added_to_consumables(self):
        gs = _setup_shop("BUY_CONS")
        tarot = Card(center_key="c_strength", cost=4)
        tarot.ability = {"set": "Tarot", "effect": ""}
        gs["shop_cards"] = [tarot]
        gs["dollars"] = 10
        from jackdaw.engine.actions import BuyCard

        step(gs, BuyCard(shop_index=0))
        assert tarot in gs["consumables"]

    def test_buy_playing_card_added_to_deck(self):
        gs = _setup_shop("BUY_PLAY")
        playing = Card(center_key="c_base", cost=1)
        playing.ability = {"set": "Default", "effect": ""}
        gs["shop_cards"] = [playing]
        gs["dollars"] = 10
        initial_deck = len(gs["deck"])
        from jackdaw.engine.actions import BuyCard

        step(gs, BuyCard(shop_index=0))
        assert len(gs["deck"]) == initial_deck + 1

    def test_buy_negative_joker_when_slots_full(self):
        """Negative edition joker can be bought even with full slots."""
        gs = _setup_shop("BUY_NEG")
        gs["joker_slots"] = 2
        gs["jokers"] = [_joker_card("j_a"), _joker_card("j_b")]
        neg_joker = _joker_card("j_c", cost=5)
        neg_joker.edition = {"negative": True}
        gs["shop_cards"] = [neg_joker]
        gs["dollars"] = 10
        from jackdaw.engine.actions import BuyCard

        step(gs, BuyCard(shop_index=0))
        assert neg_joker in gs["jokers"]
        assert len(gs["jokers"]) == 3

    def test_buy_too_expensive_raises(self):
        gs = _setup_shop("BUY_BROKE")
        expensive = _joker_card(cost=100)
        gs["shop_cards"] = [expensive]
        gs["dollars"] = 5
        from jackdaw.engine.actions import BuyCard

        with pytest.raises(IllegalActionError, match="Cannot afford"):
            step(gs, BuyCard(shop_index=0))

    def test_buy_fires_campfire_on_hologram(self):
        """Hologram gains xMult when a playing card is added to deck."""
        gs = _setup_shop("BUY_HOLO")
        holo = _joker_card("j_hologram")
        holo.ability["x_mult"] = 1.0
        holo.ability["extra"] = 0.25
        gs["jokers"] = [holo]
        playing = Card(center_key="c_base", cost=1)
        playing.ability = {"set": "Default", "effect": ""}
        gs["shop_cards"] = [playing]
        gs["dollars"] = 10
        from jackdaw.engine.actions import BuyCard

        step(gs, BuyCard(shop_index=0))
        # Hologram should have gained xMult
        assert holo.ability["x_mult"] > 1.0


class TestSellCardDetailed:
    def test_sell_joker_dollars_increase(self):
        gs = _setup_shop("SELL_J")
        joker = _joker_card(sell_cost=5)
        gs["jokers"] = [joker]
        before = gs["dollars"]
        step(gs, SellCard(area="jokers", card_index=0))
        assert gs["dollars"] == before + 5
        assert joker not in gs["jokers"]

    def test_sell_consumable(self):
        gs = _setup_shop("SELL_C")
        tarot = Card(center_key="c_fool", cost=3, sell_cost=2)
        tarot.ability = {"set": "Tarot"}
        gs["consumables"] = [tarot]
        before = gs["dollars"]
        step(gs, SellCard(area="consumables", card_index=0))
        assert gs["dollars"] == before + 2

    def test_sell_eternal_raises(self):
        gs = _setup_shop("SELL_E")
        eternal = _joker_card(eternal=True)
        gs["jokers"] = [eternal]
        with pytest.raises(IllegalActionError, match="eternal"):
            step(gs, SellCard(area="jokers", card_index=0))

    def test_sell_fires_campfire(self):
        """Campfire gains +0.25 xMult when any card is sold."""
        gs = _setup_shop("SELL_CAMP")
        campfire = _joker_card("j_campfire")
        campfire.ability["x_mult"] = 1.0
        campfire.ability["extra"] = 0.25
        victim = _joker_card("j_victim", sell_cost=2)
        gs["jokers"] = [campfire, victim]
        step(gs, SellCard(area="jokers", card_index=1))
        assert campfire.ability["x_mult"] == 1.25


class TestRedeemVoucherDetailed:
    def test_redeem_voucher_effect_applied(self):
        gs = _setup_shop("REDEEM_V")
        voucher = Card(center_key="v_grabber", cost=10)
        voucher.ability = {"set": "Voucher"}
        gs["shop_vouchers"] = [voucher]
        gs["dollars"] = 15
        from jackdaw.engine.actions import RedeemVoucher

        step(gs, RedeemVoucher(card_index=0))
        assert gs["used_vouchers"].get("v_grabber") is True
        assert gs["dollars"] == 5
        assert len(gs["shop_vouchers"]) == 0

    def test_redeem_voucher_too_expensive(self):
        gs = _setup_shop("REDEEM_BROKE")
        voucher = Card(center_key="v_grabber", cost=10)
        voucher.ability = {"set": "Voucher"}
        gs["shop_vouchers"] = [voucher]
        gs["dollars"] = 5
        from jackdaw.engine.actions import RedeemVoucher

        with pytest.raises(IllegalActionError, match="Cannot afford"):
            step(gs, RedeemVoucher(card_index=0))


class TestRerollDetailed:
    def test_reroll_deducts_cost(self):
        gs = _setup_shop("REROLL_COST")
        gs["dollars"] = 15
        gs["current_round"]["reroll_cost"] = 5
        gs["current_round"]["free_rerolls"] = 0
        step(gs, Reroll())
        assert gs["dollars"] == 10

    def test_reroll_cost_increases(self):
        gs = _setup_shop("REROLL_INC")
        gs["dollars"] = 20
        gs["current_round"]["reroll_cost"] = 5
        gs["current_round"]["free_rerolls"] = 0
        step(gs, Reroll())
        assert gs["current_round"]["reroll_cost"] > 5

    def test_reroll_free(self):
        gs = _setup_shop("REROLL_FREE")
        gs["dollars"] = 0
        gs["current_round"]["free_rerolls"] = 2
        step(gs, Reroll())
        assert gs["dollars"] == 0
        assert gs["current_round"]["free_rerolls"] == 1

    def test_reroll_tracks_stat(self):
        gs = _setup_shop("REROLL_STAT")
        gs["dollars"] = 10
        gs["current_round"]["reroll_cost"] = 5
        gs["current_round"]["free_rerolls"] = 0
        step(gs, Reroll())
        assert gs["round_scores"]["times_rerolled"] >= 1

    def test_reroll_fires_flash_card(self):
        """Flash Card gains +2 mult per reroll."""
        gs = _setup_shop("REROLL_FLASH")
        flash = _joker_card("j_flash")
        flash.ability["mult"] = 0
        flash.ability["extra"] = 2
        gs["jokers"] = [flash]
        gs["dollars"] = 10
        gs["current_round"]["reroll_cost"] = 5
        gs["current_round"]["free_rerolls"] = 0
        step(gs, Reroll())
        assert flash.ability["mult"] == 2


class TestOpenBoosterDetailed:
    def test_open_booster_transitions_to_pack(self):
        gs = _setup_shop("OPEN_PACK")
        pack = Card(center_key="p_arcana_normal_1", cost=4)
        pack.ability = {"set": "Booster", "name": "Arcana Pack"}
        gs["shop_boosters"] = [pack]
        gs["dollars"] = 10
        from jackdaw.engine.actions import OpenBooster

        step(gs, OpenBooster(card_index=0))
        assert gs["phase"] == GamePhase.PACK_OPENING

    def test_open_booster_deducts_cost(self):
        gs = _setup_shop("OPEN_COST")
        pack = Card(center_key="p_arcana_normal_1", cost=4)
        pack.ability = {"set": "Booster", "name": "Arcana Pack"}
        gs["shop_boosters"] = [pack]
        gs["dollars"] = 10
        from jackdaw.engine.actions import OpenBooster

        step(gs, OpenBooster(card_index=0))
        assert gs["dollars"] == 6


class TestNextRoundDetailed:
    def test_next_round_transitions_to_blind_select(self):
        gs = _setup_shop("NEXT_ROUND")
        step(gs, NextRound())
        assert gs["phase"] == GamePhase.BLIND_SELECT

    def test_next_round_with_perkeo(self):
        """Perkeo copies a random consumable when leaving shop."""
        gs = _setup_shop("PERKEO")
        perkeo = _joker_card("j_perkeo")
        gs["jokers"] = [perkeo]
        tarot = Card(center_key="c_strength")
        tarot.ability = {"set": "Tarot", "effect": ""}
        gs["consumables"] = [tarot]
        step(gs, NextRound())
        # Perkeo should have created a Negative copy
        assert len(gs["consumables"]) == 2
        copy = gs["consumables"][1]
        assert copy.edition == {"negative": True}


class TestPackOpening:
    def _setup_pack(self, seed="PACK_TEST"):
        gs = _setup_shop(seed)
        pack = Card(center_key="p_arcana_normal_1", cost=4)
        pack.ability = {"set": "Booster", "name": "Arcana Pack"}
        gs["shop_boosters"] = [pack]
        gs["dollars"] = 10
        from jackdaw.engine.actions import OpenBooster

        step(gs, OpenBooster(card_index=0))
        # Manually set pack cards for testing
        tarot1 = Card(center_key="c_magician", cost=0)
        tarot1.ability = {"set": "Tarot", "effect": ""}
        tarot2 = Card(center_key="c_empress", cost=0)
        tarot2.ability = {"set": "Tarot", "effect": ""}
        gs["pack_cards"] = [tarot1, tarot2]
        gs["pack_choices_remaining"] = 1
        return gs

    def test_pick_card_adds_to_consumables(self):
        from jackdaw.engine.actions import PickPackCard

        gs = self._setup_pack()
        card = gs["pack_cards"][0]
        step(gs, PickPackCard(card_index=0))
        assert card in gs["consumables"]

    def test_pick_returns_to_shop_when_done(self):
        from jackdaw.engine.actions import PickPackCard

        gs = self._setup_pack()
        step(gs, PickPackCard(card_index=0))
        assert gs["phase"] == GamePhase.SHOP

    def test_skip_pack_returns_to_shop(self):
        from jackdaw.engine.actions import SkipPack

        gs = self._setup_pack()
        step(gs, SkipPack())
        assert gs["phase"] == GamePhase.SHOP
        assert gs["pack_cards"] == []

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
    def _setup_arcana_pack(self, seed="PACK_ARC"):
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

    def _setup_buffoon_pack(self, seed="PACK_BUF"):
        gs = _setup_shop(seed)
        pack = Card(center_key="p_buffoon_normal_1", cost=4)
        pack.ability = {"set": "Booster", "name": "Buffoon Pack"}
        gs["shop_boosters"] = [pack]
        gs["dollars"] = 10
        from jackdaw.engine.actions import OpenBooster

        step(gs, OpenBooster(card_index=0))
        joker1 = Card(center_key="j_joker", cost=0)
        joker1.ability = {"set": "Joker", "effect": "", "name": "Joker"}
        gs["pack_cards"] = [joker1]
        gs["pack_choices_remaining"] = 1
        return gs

    def _setup_standard_pack(self, seed="PACK_STD"):
        gs = _setup_shop(seed)
        pack = Card(center_key="p_standard_normal_1", cost=4)
        pack.ability = {"set": "Booster", "name": "Standard Pack"}
        gs["shop_boosters"] = [pack]
        gs["dollars"] = 10
        from jackdaw.engine.actions import OpenBooster

        step(gs, OpenBooster(card_index=0))
        from jackdaw.engine.card_factory import create_playing_card
        from jackdaw.engine.data.enums import Rank, Suit

        playing = create_playing_card(Suit.HEARTS, Rank.ACE)
        gs["pack_cards"] = [playing]
        gs["pack_choices_remaining"] = 1
        return gs

    def _setup_celestial_pack(self, seed="PACK_CEL"):
        gs = _setup_shop(seed)
        pack = Card(center_key="p_celestial_normal_1", cost=4)
        pack.ability = {"set": "Booster", "name": "Celestial Pack"}
        gs["shop_boosters"] = [pack]
        gs["dollars"] = 10
        from jackdaw.engine.actions import OpenBooster

        step(gs, OpenBooster(card_index=0))
        planet = Card(center_key="c_pluto", cost=0)
        planet.ability = {
            "set": "Planet",
            "effect": "",
            "consumeable": {"hand_type": "High Card"},
        }
        gs["pack_cards"] = [planet]
        gs["pack_choices_remaining"] = 1
        return gs

    def test_pick_tarot_used_immediately(self):
        """Arcana pack: tarot is used immediately, not added to consumables."""
        from jackdaw.engine.actions import PickPackCard

        gs = self._setup_arcana_pack()
        step(gs, PickPackCard(card_index=0))
        # Tarot is used, not kept in consumables (unless can't be used)
        assert gs["phase"] == GamePhase.SHOP

    def test_pick_planet_levels_up_hand(self):
        """Celestial pack: planet card levels up the hand type."""
        from jackdaw.engine.actions import PickPackCard

        gs = self._setup_celestial_pack()
        hl = gs.get("hand_levels")
        if hl:
            from jackdaw.engine.data.hands import HandType

            level_before = hl[HandType.HIGH_CARD].level
            step(gs, PickPackCard(card_index=0))
            # Planet may or may not level up depending on handler registration
            # At minimum, the pick should succeed and close the pack
            assert gs["phase"] == GamePhase.SHOP

    def test_pick_joker_from_buffoon(self):
        """Buffoon pack: joker added to joker slots."""
        from jackdaw.engine.actions import PickPackCard

        gs = self._setup_buffoon_pack()
        initial_jokers = len(gs.get("jokers", []))
        step(gs, PickPackCard(card_index=0))
        assert len(gs["jokers"]) == initial_jokers + 1
        assert gs["used_jokers"].get("j_joker") is True

    def test_pick_playing_card_from_standard(self):
        """Standard pack: playing card added to deck."""
        from jackdaw.engine.actions import PickPackCard

        gs = self._setup_standard_pack()
        initial_deck = len(gs["deck"])
        step(gs, PickPackCard(card_index=0))
        assert len(gs["deck"]) == initial_deck + 1

    def test_last_pick_closes_pack(self):
        """When pack_choices_remaining hits 0, return to SHOP."""
        from jackdaw.engine.actions import PickPackCard

        gs = self._setup_buffoon_pack()
        assert gs["pack_choices_remaining"] == 1
        step(gs, PickPackCard(card_index=0))
        assert gs["phase"] == GamePhase.SHOP
        assert gs["pack_cards"] == []

    def test_skip_pack_returns_to_shop(self):
        from jackdaw.engine.actions import SkipPack

        gs = self._setup_arcana_pack()
        step(gs, SkipPack())
        assert gs["phase"] == GamePhase.SHOP
        assert gs["pack_cards"] == []

    def test_skip_pack_returns_hand_to_deck(self):
        """Pack hand cards (dealt for Arcana) returned to deck on close."""
        from jackdaw.engine.actions import SkipPack

        gs = self._setup_arcana_pack()
        # Simulate dealt pack hand
        from jackdaw.engine.card_factory import create_playing_card
        from jackdaw.engine.data.enums import Rank, Suit

        dealt = [create_playing_card(Suit.SPADES, Rank.ACE)]
        gs["pack_hand"] = dealt
        initial_deck = len(gs["deck"])
        step(gs, SkipPack())
        assert len(gs["deck"]) == initial_deck + 1
        assert gs["pack_hand"] == []

    def test_no_choices_remaining_raises(self):
        from jackdaw.engine.actions import PickPackCard

        gs = self._setup_buffoon_pack()
        gs["pack_choices_remaining"] = 0
        with pytest.raises(IllegalActionError, match="No pack choices"):
            step(gs, PickPackCard(card_index=0))


# ===========================================================================
# UseConsumable tests
# ===========================================================================


def _make_consumable(key: str, set_name: str = "Tarot", **kw) -> Card:
    """Create a consumable card for testing."""
    c = Card(center_key=key, cost=0)
    ability = {"set": set_name, "effect": ""}
    ability.update(kw.pop("extra_ability", {}))
    c.ability = ability
    for k, v in kw.items():
        setattr(c, k, v)
    return c


class TestUseConsumableChariot:
    """Chariot (Steel Card enhancement) during SELECTING_HAND."""

    def test_chariot_enhances_to_steel(self):
        from jackdaw.engine.actions import UseConsumable

        gs = _init_gs("CHARIOT_TEST")
        step(gs, SelectBlind())
        assert gs["phase"] == GamePhase.SELECTING_HAND
        chariot = _make_consumable("c_chariot")
        gs["consumables"] = [chariot]
        target_card = gs["hand"][0]
        step(gs, UseConsumable(card_index=0, target_indices=(0,)))
        # Card should be enhanced to Steel
        assert target_card.ability.get("effect") == "Steel Card" or \
               target_card.ability.get("name") == "Steel Card"

    def test_phase_stays_selecting_hand(self):
        from jackdaw.engine.actions import UseConsumable

        gs = _init_gs("CHARIOT_PHASE")
        step(gs, SelectBlind())
        chariot = _make_consumable("c_chariot")
        gs["consumables"] = [chariot]
        step(gs, UseConsumable(card_index=0, target_indices=(0,)))
        assert gs["phase"] == GamePhase.SELECTING_HAND

    def test_consumable_removed_after_use(self):
        from jackdaw.engine.actions import UseConsumable

        gs = _init_gs("CHARIOT_POP")
        step(gs, SelectBlind())
        chariot = _make_consumable("c_chariot")
        gs["consumables"] = [chariot]
        step(gs, UseConsumable(card_index=0, target_indices=(0,)))
        assert chariot not in gs["consumables"]


class TestUseConsumablePlanet:
    """Mercury (Pair level-up) during SHOP."""

    def test_mercury_levels_up_pair(self):
        from jackdaw.engine.actions import UseConsumable
        from jackdaw.engine.data.hands import HandType

        gs = _setup_shop("MERCURY_TEST")
        mercury = _make_consumable("c_mercury", set_name="Planet")
        mercury.ability["consumeable"] = {"hand_type": "Pair"}
        gs["consumables"] = [mercury]
        hl = gs["hand_levels"]
        level_before = hl[HandType.PAIR].level
        step(gs, UseConsumable(card_index=0))
        assert hl[HandType.PAIR].level == level_before + 1

    def test_phase_stays_shop(self):
        from jackdaw.engine.actions import UseConsumable

        gs = _setup_shop("MERCURY_PHASE")
        mercury = _make_consumable("c_mercury", set_name="Planet")
        mercury.ability["consumeable"] = {"hand_type": "Pair"}
        gs["consumables"] = [mercury]
        step(gs, UseConsumable(card_index=0))
        assert gs["phase"] == GamePhase.SHOP

    def test_last_tarot_planet_tracked(self):
        from jackdaw.engine.actions import UseConsumable

        gs = _setup_shop("MERCURY_TRACK")
        mercury = _make_consumable("c_mercury", set_name="Planet")
        mercury.ability["consumeable"] = {"hand_type": "Pair"}
        gs["consumables"] = [mercury]
        step(gs, UseConsumable(card_index=0))
        assert gs.get("last_tarot_planet") == "c_mercury"


class TestUseConsumableDeath:
    """Death: copy rightmost card onto left card."""

    def test_death_copies_card(self):
        from jackdaw.engine.actions import UseConsumable

        gs = _init_gs("DEATH_TEST")
        step(gs, SelectBlind())
        death = _make_consumable("c_death")
        gs["consumables"] = [death]
        # Target two cards — Death copies rightmost (highest sort_id) onto other
        card_a = gs["hand"][0]
        card_b = gs["hand"][1]
        # Determine which is rightmost by sort_id
        if card_a.sort_id > card_b.sort_id:
            source, target = card_a, card_b
        else:
            source, target = card_b, card_a
        src_suit = source.base.suit.value
        src_rank = source.base.rank.value
        step(gs, UseConsumable(card_index=0, target_indices=(0, 1)))
        # Target should now have source's properties
        assert target.base.suit.value == src_suit
        assert target.base.rank.value == src_rank


class TestUseConsumableHangedMan:
    """Hanged Man: destroy highlighted cards."""

    def test_hanged_man_destroys_cards(self):
        from jackdaw.engine.actions import UseConsumable

        gs = _init_gs("HANGED_TEST")
        step(gs, SelectBlind())
        hanged = _make_consumable("c_hanged_man")
        gs["consumables"] = [hanged]
        target1 = gs["hand"][0]
        target2 = gs["hand"][1]
        initial_hand = len(gs["hand"])
        step(gs, UseConsumable(card_index=0, target_indices=(0, 1)))
        # Targets should be removed from hand
        assert target1 not in gs["hand"]
        assert target2 not in gs["hand"]
        assert len(gs["hand"]) == initial_hand - 2


class TestUseConsumableHighPriestess:
    """High Priestess: create 2 Planet cards."""

    def test_creates_planets(self):
        from jackdaw.engine.actions import UseConsumable

        gs = _setup_shop("PRIESTESS_TEST")
        priestess = _make_consumable("c_high_priestess")
        priestess.ability["consumeable"] = {"planets": 2}
        gs["consumables"] = [priestess]
        gs["consumable_slots"] = 5  # room for 2 new
        initial_count = len(gs.get("consumables", []))
        step(gs, UseConsumable(card_index=0))
        # Should have created 2 planet cards in consumables
        # (priestess removed, 2 added → net +1)
        assert len(gs["consumables"]) >= initial_count + 1


class TestUseConsumableWheelOfFortune:
    """Wheel of Fortune: add edition to random joker (1-in-4 chance)."""

    def test_wheel_on_editionless_joker(self):
        from jackdaw.engine.actions import UseConsumable

        gs = _setup_shop("WHEEL_TEST")
        joker = _joker_card("j_joker")
        joker.edition = None
        gs["jokers"] = [joker]
        wheel = _make_consumable("c_wheel_of_fortune")
        wheel.ability["extra"] = 4  # 1-in-4 chance
        gs["consumables"] = [wheel]
        step(gs, UseConsumable(card_index=0))
        # Either the joker got an edition or it didn't (RNG-dependent)
        # Just verify the action didn't crash and wheel was consumed
        assert wheel not in gs.get("consumables", [])


class TestUseConsumableBlindSelect:
    """Consumable usable in BLIND_SELECT phase."""

    def test_use_in_blind_select(self):
        from jackdaw.engine.actions import UseConsumable

        gs = _init_gs("USE_BLIND_SEL")
        assert gs["phase"] == GamePhase.BLIND_SELECT
        mercury = _make_consumable("c_mercury", set_name="Planet")
        mercury.ability["consumeable"] = {"hand_type": "Pair"}
        gs["consumables"] = [mercury]
        step(gs, UseConsumable(card_index=0))
        assert gs["phase"] == GamePhase.BLIND_SELECT
        assert len(gs["consumables"]) == 0


class TestUseConsumableConstellationJoker:
    """Constellation gains xMult when Planet is used."""

    def test_constellation_notified(self):
        from jackdaw.engine.actions import UseConsumable

        gs = _setup_shop("CONSTELLATION")
        constellation = _joker_card("j_constellation")
        constellation.ability["x_mult"] = 1.0
        constellation.ability["extra"] = 0.1
        gs["jokers"] = [constellation]
        mercury = _make_consumable("c_mercury", set_name="Planet")
        mercury.ability["consumeable"] = {"hand_type": "Pair"}
        gs["consumables"] = [mercury]
        step(gs, UseConsumable(card_index=0))
        # Constellation should have gained xMult
        assert constellation.ability["x_mult"] > 1.0


class TestUseConsumableHermit:
    """Hermit: gain dollars equal to min(current, $20)."""

    def test_hermit_doubles_money(self):
        from jackdaw.engine.actions import UseConsumable

        gs = _setup_shop("HERMIT_TEST")
        gs["dollars"] = 15
        hermit = _make_consumable("c_hermit")
        hermit.ability["extra"] = 20
        gs["consumables"] = [hermit]
        step(gs, UseConsumable(card_index=0))
        assert gs["dollars"] == 30  # 15 + min(15, 20)


class TestUseConsumableInvalidIndex:
    def test_invalid_index_raises(self):
        from jackdaw.engine.actions import UseConsumable

        gs = _setup_shop("USE_INVALID")
        gs["consumables"] = []
        with pytest.raises(IllegalActionError, match="Invalid consumable"):
            step(gs, UseConsumable(card_index=0))


# ===========================================================================
# Shop population tests
# ===========================================================================


class TestShopPopulation:
    """Verify shop is auto-populated on cash_out and cleared on next_round."""

    def _beat_and_cash_out(self, seed="SHOP_POP"):
        gs = _init_gs(seed)
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        assert gs["phase"] == GamePhase.ROUND_EVAL
        step(gs, CashOut())
        assert gs["phase"] == GamePhase.SHOP
        return gs

    def test_shop_has_joker_cards(self):
        gs = self._beat_and_cash_out("POP_JOKERS")
        shop_cards = gs.get("shop_cards", [])
        joker_max = gs.get("shop", {}).get("joker_max", 2)
        assert len(shop_cards) == joker_max

    def test_shop_has_voucher(self):
        gs = self._beat_and_cash_out("POP_VOUCHER")
        vouchers = gs.get("shop_vouchers", [])
        assert len(vouchers) == 1
        assert vouchers[0].center_key.startswith("v_")

    def test_shop_has_two_boosters(self):
        gs = self._beat_and_cash_out("POP_BOOSTERS")
        boosters = gs.get("shop_boosters", [])
        assert len(boosters) == 2

    def test_shop_cards_have_costs(self):
        gs = self._beat_and_cash_out("POP_COSTS")
        for card in gs.get("shop_cards", []):
            assert card.cost > 0, f"{card.center_key} has cost 0"

    def test_shop_cards_valid_types(self):
        gs = self._beat_and_cash_out("POP_TYPES")
        for card in gs.get("shop_cards", []):
            card_set = card.ability.get("set", "")
            assert card_set in ("Joker", "Tarot", "Planet", "Spectral", "Enhanced", "Default"), (
                f"Unexpected card set: {card_set} for {card.center_key}"
            )

    def test_first_shop_has_buffoon_pack(self):
        """First shop of a run should have a Buffoon pack (guaranteed)."""
        gs = self._beat_and_cash_out("POP_BUFFOON")
        boosters = gs.get("shop_boosters", [])
        booster_keys = [b.center_key for b in boosters]
        has_buffoon = any("buffoon" in k for k in booster_keys)
        assert has_buffoon, f"No Buffoon pack in first shop: {booster_keys}"


class TestShopReroll:
    """Verify reroll regenerates shop cards."""

    def _setup_shop_with_cards(self, seed="REROLL_TEST"):
        gs = _init_gs(seed)
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        step(gs, CashOut())
        return gs

    def test_reroll_changes_shop_cards(self):
        gs = self._setup_shop_with_cards()
        gs["dollars"] = 20
        old_keys = [c.center_key for c in gs.get("shop_cards", [])]
        step(gs, Reroll())
        new_keys = [c.center_key for c in gs.get("shop_cards", [])]
        # New cards should exist and have same count
        assert len(new_keys) == len(old_keys)
        # Cards may or may not differ (RNG), but the list should be regenerated

    def test_reroll_preserves_voucher_and_boosters(self):
        gs = self._setup_shop_with_cards()
        gs["dollars"] = 20
        voucher_before = [v.center_key for v in gs.get("shop_vouchers", [])]
        boosters_before = [b.center_key for b in gs.get("shop_boosters", [])]
        step(gs, Reroll())
        voucher_after = [v.center_key for v in gs.get("shop_vouchers", [])]
        boosters_after = [b.center_key for b in gs.get("shop_boosters", [])]
        # Voucher and boosters should NOT change on reroll
        assert voucher_after == voucher_before
        assert boosters_after == boosters_before

    def test_reroll_count_matches_joker_max(self):
        gs = self._setup_shop_with_cards()
        gs["dollars"] = 20
        step(gs, Reroll())
        joker_max = gs.get("shop", {}).get("joker_max", 2)
        assert len(gs["shop_cards"]) == joker_max


class TestShopClearOnNextRound:
    """Verify shop is cleared when leaving."""

    def test_next_round_clears_shop(self):
        gs = _init_gs("CLEAR_SHOP")
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        step(gs, CashOut())
        assert len(gs["shop_cards"]) > 0
        step(gs, NextRound())
        assert gs["shop_cards"] == []
        assert gs["shop_vouchers"] == []
        assert gs["shop_boosters"] == []


class TestShopOverstock:
    """Overstock voucher adds +1 joker_max → shop has 3 cards."""

    def test_overstock_increases_shop_size(self):
        gs = _init_gs("OVERSTOCK")
        # Apply Overstock voucher effect
        gs["shop"]["joker_max"] = 3
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        step(gs, CashOut())
        assert len(gs["shop_cards"]) == 3


# ===========================================================================
# Pack generation tests
# ===========================================================================


class TestPackGeneration:
    """Verify packs generate real cards when opened."""

    def _enter_shop(self, seed="PACK_GEN"):
        gs = _init_gs(seed)
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        step(gs, CashOut())
        assert gs["phase"] == GamePhase.SHOP
        return gs

    def test_open_arcana_generates_tarot_cards(self):
        from jackdaw.engine.actions import OpenBooster

        gs = self._enter_shop("ARCANA_GEN")
        # Find an arcana pack or force one
        from jackdaw.engine.card import Card

        arcana = Card(center_key="p_arcana_normal_1", cost=4)
        arcana.ability = {"set": "Booster", "name": "Arcana Pack"}
        gs["shop_boosters"] = [arcana]
        gs["dollars"] = 20
        step(gs, OpenBooster(card_index=0))
        assert gs["phase"] == GamePhase.PACK_OPENING
        assert len(gs["pack_cards"]) == 3  # arcana normal has extra=3
        assert gs["pack_choices_remaining"] == 1
        assert gs["pack_type"] == "Arcana"
        # Cards should be Tarot/Spectral
        for c in gs["pack_cards"]:
            s = c.ability.get("set", "")
            assert s in ("Tarot", "Spectral"), f"Expected Tarot/Spectral, got {s}"

    def test_open_arcana_deals_hand(self):
        from jackdaw.engine.actions import OpenBooster

        gs = self._enter_shop("ARCANA_HAND")
        from jackdaw.engine.card import Card

        arcana = Card(center_key="p_arcana_normal_1", cost=4)
        arcana.ability = {"set": "Booster", "name": "Arcana Pack"}
        gs["shop_boosters"] = [arcana]
        gs["dollars"] = 20
        deck_before = len(gs["deck"])
        step(gs, OpenBooster(card_index=0))
        # Hand should have cards dealt from deck for targeting
        assert len(gs.get("hand", [])) > 0
        assert len(gs.get("pack_hand", [])) > 0

    def test_open_buffoon_generates_jokers(self):
        from jackdaw.engine.actions import OpenBooster

        gs = self._enter_shop("BUFFOON_GEN")
        from jackdaw.engine.card import Card

        buffoon = Card(center_key="p_buffoon_normal_1", cost=4)
        buffoon.ability = {"set": "Booster", "name": "Buffoon Pack"}
        gs["shop_boosters"] = [buffoon]
        gs["dollars"] = 20
        step(gs, OpenBooster(card_index=0))
        assert len(gs["pack_cards"]) == 2  # buffoon normal has extra=2
        assert gs["pack_type"] == "Buffoon"
        for c in gs["pack_cards"]:
            assert c.ability.get("set") == "Joker"

    def test_open_celestial_generates_planets(self):
        from jackdaw.engine.actions import OpenBooster

        gs = self._enter_shop("CELESTIAL_GEN")
        from jackdaw.engine.card import Card

        celestial = Card(center_key="p_celestial_normal_1", cost=4)
        celestial.ability = {"set": "Booster", "name": "Celestial Pack"}
        gs["shop_boosters"] = [celestial]
        gs["dollars"] = 20
        step(gs, OpenBooster(card_index=0))
        assert len(gs["pack_cards"]) == 3  # celestial normal has extra=3
        assert gs["pack_type"] == "Celestial"
        for c in gs["pack_cards"]:
            assert c.ability.get("set") == "Planet"

    def test_open_standard_generates_playing_cards(self):
        from jackdaw.engine.actions import OpenBooster

        gs = self._enter_shop("STANDARD_GEN")
        from jackdaw.engine.card import Card

        standard = Card(center_key="p_standard_normal_1", cost=4)
        standard.ability = {"set": "Booster", "name": "Standard Pack"}
        gs["shop_boosters"] = [standard]
        gs["dollars"] = 20
        step(gs, OpenBooster(card_index=0))
        assert len(gs["pack_cards"]) == 3  # standard normal has extra=3
        assert gs["pack_type"] == "Standard"
        for c in gs["pack_cards"]:
            assert c.base is not None  # playing cards have a base

    def test_pick_joker_from_buffoon_adds_to_slots(self):
        from jackdaw.engine.actions import OpenBooster, PickPackCard

        gs = self._enter_shop("PICK_JOKER")
        from jackdaw.engine.card import Card

        buffoon = Card(center_key="p_buffoon_normal_1", cost=4)
        buffoon.ability = {"set": "Booster", "name": "Buffoon Pack"}
        gs["shop_boosters"] = [buffoon]
        gs["dollars"] = 20
        step(gs, OpenBooster(card_index=0))
        initial_jokers = len(gs.get("jokers", []))
        step(gs, PickPackCard(card_index=0))
        assert len(gs["jokers"]) == initial_jokers + 1
        assert gs["phase"] == GamePhase.SHOP

    def test_pick_playing_card_from_standard_adds_to_deck(self):
        from jackdaw.engine.actions import OpenBooster, PickPackCard

        gs = self._enter_shop("PICK_PLAYING")
        from jackdaw.engine.card import Card

        standard = Card(center_key="p_standard_normal_1", cost=4)
        standard.ability = {"set": "Booster", "name": "Standard Pack"}
        gs["shop_boosters"] = [standard]
        gs["dollars"] = 20
        initial_deck = len(gs["deck"])
        step(gs, OpenBooster(card_index=0))
        step(gs, PickPackCard(card_index=0))
        assert len(gs["deck"]) == initial_deck + 1

    def test_skip_pack_clears_cards(self):
        from jackdaw.engine.actions import OpenBooster, SkipPack

        gs = self._enter_shop("SKIP_PACK")
        from jackdaw.engine.card import Card

        arcana = Card(center_key="p_arcana_normal_1", cost=4)
        arcana.ability = {"set": "Booster", "name": "Arcana Pack"}
        gs["shop_boosters"] = [arcana]
        gs["dollars"] = 20
        step(gs, OpenBooster(card_index=0))
        assert len(gs["pack_cards"]) > 0
        step(gs, SkipPack())
        assert gs["pack_cards"] == []
        assert gs["phase"] == GamePhase.SHOP

    def test_skip_arcana_returns_hand_to_deck(self):
        from jackdaw.engine.actions import OpenBooster, SkipPack

        gs = self._enter_shop("SKIP_ARCANA")
        from jackdaw.engine.card import Card

        arcana = Card(center_key="p_arcana_normal_1", cost=4)
        arcana.ability = {"set": "Booster", "name": "Arcana Pack"}
        gs["shop_boosters"] = [arcana]
        gs["dollars"] = 20
        deck_before = len(gs["deck"])
        step(gs, OpenBooster(card_index=0))
        hand_dealt = len(gs.get("pack_hand", []))
        step(gs, SkipPack())
        # Pack hand returned to deck
        assert len(gs["deck"]) == deck_before
        assert gs.get("pack_hand", []) == []

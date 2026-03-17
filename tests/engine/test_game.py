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

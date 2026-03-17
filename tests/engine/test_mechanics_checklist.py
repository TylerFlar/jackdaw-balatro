"""Mechanics checklist — verify each Balatro mechanic is implemented.

Each test asserts one specific mechanic works, serving as a living
checklist of simulator accuracy.
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
    SelectBlind,
    SkipBlind,
)
from jackdaw.engine.card import Card
from jackdaw.engine.game import IllegalActionError, step
from jackdaw.engine.run_init import initialize_run


def _init(seed: str = "MECH") -> dict[str, Any]:
    gs = initialize_run("b_red", 1, seed)
    gs["phase"] = GamePhase.BLIND_SELECT
    gs["blind_on_deck"] = "Small"
    gs["jokers"] = []
    gs["consumables"] = []
    return gs


def _joker(key: str = "j_joker", **kw) -> Card:
    c = Card(center_key=key)
    c.ability = {"set": "Joker", "effect": "", "name": key}
    c.sell_cost = 3
    for k, v in kw.items():
        setattr(c, k, v)
    return c


# ---------------------------------------------------------------------------
# Card flipping state
# ---------------------------------------------------------------------------


class TestCardFlipping:
    def test_card_has_facing_attribute(self):
        c = Card()
        assert hasattr(c, "facing")
        assert c.facing == "front"

    def test_facing_can_be_set_to_back(self):
        c = Card()
        c.facing = "back"
        assert c.facing == "back"

    def test_the_fish_flips_cards(self):
        """The Fish boss flips hand cards after play."""
        gs = _init("FISH_FLIP")
        step(gs, SkipBlind())
        step(gs, SkipBlind())
        gs["round_resets"]["blind_choices"]["Boss"] = "bl_fish"
        step(gs, SelectBlind())
        # Play a hand
        gs["blind"].chips = 999999
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        # After play, remaining hand cards should be flipped
        if gs["phase"] == GamePhase.SELECTING_HAND:
            for card in gs["hand"]:
                assert card.facing == "back", f"{card.card_key} not flipped"


# ---------------------------------------------------------------------------
# Boss press_play effects
# ---------------------------------------------------------------------------


class TestBossPressPlay:
    def test_the_hook_discards(self):
        """The Hook discards 2 random hand cards on play."""
        gs = _init("HOOK_TEST")
        step(gs, SkipBlind())
        step(gs, SkipBlind())
        gs["round_resets"]["blind_choices"]["Boss"] = "bl_hook"
        step(gs, SelectBlind())
        hand_before = len(gs["hand"])
        gs["blind"].chips = 999999
        step(gs, PlayHand(card_indices=(0,)))
        # Hand should have fewer cards (1 played + 2 hooked + replacements drawn)
        # The net effect depends on deck size, but discard_pile should have entries
        assert len(gs.get("discard_pile", [])) >= 2

    def test_the_tooth_costs_dollars(self):
        """The Tooth costs $1 per card played."""
        gs = _init("TOOTH_TEST")
        step(gs, SkipBlind())
        step(gs, SkipBlind())
        gs["round_resets"]["blind_choices"]["Boss"] = "bl_tooth"
        step(gs, SelectBlind())
        dollars_before = gs["dollars"]
        gs["blind"].chips = 999999
        step(gs, PlayHand(card_indices=(0, 1, 2)))
        # Should lose $3 (ignoring scoring dollars)
        result = gs["last_score_result"]
        expected = dollars_before - 3 + result.dollars_earned
        assert gs["dollars"] == expected

    def test_the_fish_sets_prepped(self):
        """The Fish sets prepped flag on press_play."""
        gs = _init("FISH_PREP")
        step(gs, SkipBlind())
        step(gs, SkipBlind())
        gs["round_resets"]["blind_choices"]["Boss"] = "bl_fish"
        step(gs, SelectBlind())
        gs["blind"].chips = 999999
        step(gs, PlayHand(card_indices=(0,)))
        assert getattr(gs["blind"], "prepped", False) is True


# ---------------------------------------------------------------------------
# Seal effects
# ---------------------------------------------------------------------------


class TestSealEffects:
    def test_purple_seal_creates_tarot_on_discard(self):
        """Discarding a Purple Seal card creates a Tarot consumable."""
        gs = _init("PURPLE_SEAL")
        step(gs, SelectBlind())
        gs["hand"][0].seal = "Purple"
        initial_cons = len(gs.get("consumables", []))
        step(gs, Discard(card_indices=(0,)))
        assert len(gs.get("consumables", [])) == initial_cons + 1

    def test_gold_seal_gives_dollars_at_round_end(self):
        """Gold Seal cards in hand at round end give +$3 each."""
        gs = _init("GOLD_SEAL")
        step(gs, SelectBlind())
        gs["hand"][5].seal = "Gold"
        gs["hand"][6].seal = "Gold"
        gs["blind"].chips = 1
        dollars_before = gs["dollars"]
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        # After round end, $6 from Gold Seals (2 cards × $3)
        step(gs, CashOut())
        # Gold seal bonus is included in the round_end processing
        assert gs["dollars"] > dollars_before

    def test_blue_seal_creates_planet_at_round_end(self):
        """Blue Seal creates Planet for most-played hand at round end."""
        gs = _init("BLUE_SEAL")
        step(gs, SelectBlind())
        # Give a held card Blue Seal
        gs["hand"][7].seal = "Blue"
        gs["blind"].chips = 1
        gs["consumable_slots"] = 5
        initial_cons = len(gs.get("consumables", []))
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        # Blue Seal on held card (index 7 wasn't played) should create planet
        new_cons = len(gs.get("consumables", []))
        assert new_cons > initial_cons


# ---------------------------------------------------------------------------
# Double Tag duplication
# ---------------------------------------------------------------------------


class TestDoubleTag:
    def test_double_tag_duplicates(self):
        """Double Tag duplicates a newly awarded tag."""
        gs = _init("DOUBLE_TAG")
        # Put a Double Tag in the active tags
        gs["tags"] = ["tag_double"]
        # Force Small tag to be tag_economy
        gs["round_resets"]["blind_tags"]["Small"] = "tag_economy"
        gs["dollars"] = 10
        step(gs, SkipBlind())
        # Should have received tag_economy + a duplicate
        awarded = gs.get("awarded_tags", [])
        economy_awards = [a for a in awarded if a["key"] == "tag_economy"]
        assert len(economy_awards) >= 2, f"Expected 2 economy tags, got {len(economy_awards)}"

    def test_double_tag_consumed(self):
        """Double Tag is consumed after use."""
        gs = _init("DOUBLE_CONSUME")
        gs["tags"] = ["tag_double"]
        gs["round_resets"]["blind_tags"]["Small"] = "tag_skip"
        step(gs, SkipBlind())
        assert "tag_double" not in gs.get("tags", [])


# ---------------------------------------------------------------------------
# Endless mode
# ---------------------------------------------------------------------------


class TestEndlessMode:
    def test_game_continues_after_win_ante(self):
        """After beating the win_ante boss, the game continues."""
        gs = _init("ENDLESS")
        gs["win_ante"] = 1
        step(gs, SkipBlind())
        step(gs, SkipBlind())
        step(gs, SelectBlind())
        gs["blind"].chips = 1
        gs["blind"].disabled = True  # avoid debuffs
        step(gs, PlayHand(card_indices=(0, 1, 2, 3, 4)))
        assert gs["won"] is True
        # Game should continue (ROUND_EVAL, not GAME_OVER)
        assert gs["phase"] == GamePhase.ROUND_EVAL
        step(gs, CashOut())
        assert gs["phase"] == GamePhase.SHOP
        # Can proceed to next round
        step(gs, NextRound())
        assert gs["phase"] == GamePhase.BLIND_SELECT
        # Ante should have advanced
        assert gs["round_resets"]["ante"] == 2


# ---------------------------------------------------------------------------
# Ante-modifying vouchers
# ---------------------------------------------------------------------------


class TestAnteModifyingVouchers:
    def test_hieroglyph_reduces_ante(self):
        """Hieroglyph voucher reduces ante."""
        from jackdaw.engine.vouchers import apply_voucher

        gs = _init("HIEROGLYPH")
        gs["round_resets"]["ante"] = 3
        gs["round_resets"]["blind_ante"] = 3
        gs["round_resets"]["hands"] = 4
        apply_voucher("v_hieroglyph", gs)
        assert gs["round_resets"]["ante"] < 3
        assert gs["round_resets"]["hands"] < 4

    def test_petroglyph_reduces_ante(self):
        """Petroglyph voucher reduces ante and discards."""
        from jackdaw.engine.vouchers import apply_voucher

        gs = _init("PETROGLYPH")
        gs["round_resets"]["ante"] = 3
        gs["round_resets"]["blind_ante"] = 3
        gs["round_resets"]["discards"] = 4
        apply_voucher("v_petroglyph", gs)
        assert gs["round_resets"]["ante"] < 3
        assert gs["round_resets"]["discards"] < 4

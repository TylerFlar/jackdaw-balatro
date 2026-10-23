"""Consumable fidelity (G26, G27, G28, G31, G23 Wheel of Fortune)."""

from __future__ import annotations

import pytest

from jackdaw.engine.actions import UseConsumable, get_legal_actions
from jackdaw.engine.game import IllegalActionError, step

from ._helpers import give_consumable, give_joker, new_run, select_blind


class TestAnkhCheckUse:
    def test_ankh_cannot_be_used_with_full_joker_slots(self):
        """use_card returns early when Card:check_use says so
        (button_callbacks.lua:2163-2169); Ankh's check_use is true while
        #G.jokers.cards >= G.jokers.config.card_limit (card.lua:1581-1588).
        The card is not consumed and nothing happens."""
        gs = select_blind(new_run("AUDIT_ANKH_FULL"))
        for key in ("j_joker", "j_greedy_joker", "j_lusty_joker", "j_wrathful_joker", "j_mad"):
            give_joker(gs, key)
        assert len(gs["jokers"]) == gs["joker_slots"]
        give_consumable(gs, "c_ankh")
        assert UseConsumable(card_index=0) not in get_legal_actions(gs)
        with pytest.raises(IllegalActionError):
            step(gs, UseConsumable(card_index=0))
        assert [c.center_key for c in gs["consumables"]] == ["c_ankh"]
        assert len(gs["jokers"]) == 5

    def test_ankh_usable_with_a_free_slot(self):
        gs = select_blind(new_run("AUDIT_ANKH_ROOM"))
        for key in ("j_joker", "j_greedy_joker", "j_lusty_joker", "j_wrathful_joker"):
            give_joker(gs, key)
        give_consumable(gs, "c_ankh")
        assert UseConsumable(card_index=0) in get_legal_actions(gs)

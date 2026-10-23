"""Joker fidelity (G16, G17, G18, G21, G24, G25 and related)."""

from __future__ import annotations

from jackdaw.engine.actions import GamePhase

from ._helpers import give_card, give_joker, new_run, play, select_blind


class TestMrBones:
    def test_save_pays_no_blind_reward(self):
        """evaluate_round pays the blind only when G.GAME.chips >=
        G.GAME.blind.chips; a Mr. Bones save shows the blind row with $0
        (state_events.lua:1139-1146).  The other rows are unchanged."""
        gs = select_blind(new_run("AUDIT_BONES"))
        give_joker(gs, "j_mr_bones")
        card = give_card(gs, "S_2")
        gs["current_round"]["hands_left"] = 1
        gs["chips"] = int(gs["blind"].chips * 0.3)
        play(gs, card)
        assert gs["phase"] == GamePhase.ROUND_EVAL
        assert not any(j.center_key == "j_mr_bones" for j in gs["jokers"])
        earnings = gs["round_earnings"]
        assert earnings.blind_reward == 0
        assert earnings.total == (
            earnings.unused_hands_bonus
            + earnings.unused_discards_bonus
            + earnings.joker_dollars
            + earnings.interest
            - earnings.rental_cost
        )

    def test_beaten_blind_still_pays(self):
        gs = select_blind(new_run("AUDIT_BONES2"))
        gs["blind"].chips = 1
        play(gs, gs["hand"][0])
        assert gs["round_earnings"].blind_reward == gs["blind"].dollars > 0

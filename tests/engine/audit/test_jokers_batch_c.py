"""Oracle fixes found while porting joker batch C to Rust (G80-G96)."""

from __future__ import annotations

from typing import Any

from ._joker_helpers import give_card, give_joker, new_run, play, select_blind


def _pillar_round(seed: str) -> dict[str, Any]:
    """A run facing The Pillar, the first boss of ante 1."""
    gs = new_run(seed)
    gs["round_resets"]["blind_choices"]["Boss"] = "bl_pillar"
    gs["blind_on_deck"] = "Boss"
    return select_blind(gs)


class TestBossRecheckSeesThePlayedCards:
    """G80: a joker or consumable joining or leaving the run queues
    ``G.GAME.blind:set_blind(nil, true)`` (card.lua:641, 698), whose debuff
    pass covers every card of ``G.playing_cards`` (blind.lua:207-210), the
    cards being played included. They were played this ante, so The Pillar
    debuffs them (blind.lua:634-636)."""

    def test_seltzer_drunk_rechecks_the_played_cards(self):
        """Seltzer's last hand: its removal (card.lua:3601-3622) re-checks."""
        gs = _pillar_round("AUDIT_G80_SELTZER")
        seltzer = give_joker(gs, "j_selzer")
        seltzer.ability["extra"] = 1
        cards = [give_card(gs, "S_9"), give_card(gs, "H_9")]
        play(gs, *cards)
        assert seltzer not in gs["jokers"]
        assert all(c in gs["discard_pile"] and c.debuff for c in cards)

    def test_eight_ball_tarot_rechecks_the_played_cards(self):
        """8 Ball's Tarot (card.lua:3106-3125): its add_to_deck re-checks."""
        gs = _pillar_round("AUDIT_G80_8BALL")
        give_joker(gs, "j_8_ball")
        gs["probabilities"]["normal"] = 4  # 4 in 4: every roll hits
        cards = [give_card(gs, "S_8"), give_card(gs, "H_8")]
        play(gs, *cards)
        assert len(gs["consumables"]) == 2
        assert all(c in gs["discard_pile"] and c.debuff for c in cards)

    def test_without_a_recheck_the_played_cards_stay_clean(self):
        """No card joined or left: nothing re-checks the cards just played."""
        gs = _pillar_round("AUDIT_G80_PLAIN")
        cards = [give_card(gs, "S_9"), give_card(gs, "H_9")]
        play(gs, *cards)
        assert all(c in gs["discard_pile"] and not c.debuff for c in cards)

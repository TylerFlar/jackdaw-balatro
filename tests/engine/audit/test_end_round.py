"""end_round's bookkeeping (state_events.lua:87-288)."""

from __future__ import annotations

from ._joker_helpers import give_joker, new_run, select_blind, win_round


def test_a_perishing_drunkards_discard_counts_as_unused() -> None:
    """G262: end_round adds the round's discards left to ``unused_discards``
    (state_events.lua:124) before the ease_discard that a perishing
    Drunkard's remove_from_deck queues (card.lua:686-693) has run."""
    gs = select_blind(new_run("AUDIT_UNUSED_DISCARDS"))
    drunkard = give_joker(gs, "j_drunkard", perishable=True)
    drunkard.ability["perish_tally"] = 1
    left = gs["current_round"]["discards_left"]
    before = gs.get("unused_discards", 0)
    win_round(gs)
    assert drunkard.debuff  # the sticker ran out at the round's end
    assert gs["current_round"]["discards_left"] == left - 1
    assert gs["unused_discards"] == before + left

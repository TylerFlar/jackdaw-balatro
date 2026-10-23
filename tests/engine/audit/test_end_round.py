"""Round bookkeeping: end_round (state_events.lua:87-288) and the discard
(state_events.lua:381-445)."""

from __future__ import annotations

from ._joker_helpers import discard, give_card, give_joker, new_run, select_blind, win_round


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


def test_a_purple_seals_tarot_re_debuffs_the_discarded_cards() -> None:
    """G263: the Tarot's add_to_deck queues the boss debuff pass
    (card.lua:641), which runs after the discarded cards reach the discard
    pile: a Wild Card is every suit, so The Club debuffs it there."""
    gs = new_run("AUDIT_PURPLE_CLUB")
    gs["round_resets"]["blind_choices"]["Boss"] = "bl_club"
    gs["blind_on_deck"] = "Boss"
    select_blind(gs)
    wild = give_card(gs, "D_9", enhancement="m_wild", seal="Purple")
    assert not wild.debuff  # add runs no debuff check
    discard(gs, wild)
    assert wild in gs["discard_pile"]
    assert len(gs["consumables"]) == 1
    assert wild.debuff

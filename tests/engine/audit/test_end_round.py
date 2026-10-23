"""Round bookkeeping: end_round (state_events.lua:87-288) and the discard
(state_events.lua:381-445)."""

from __future__ import annotations

from jackdaw.engine.actions import GamePhase, PlayHand
from jackdaw.engine.game import step

from ._joker_helpers import (
    bot_add,
    discard,
    give_card,
    give_joker,
    new_run,
    select_blind,
    win_round,
)


def test_a_lost_round_runs_none_of_end_rounds_events() -> None:
    """G265: losing, end_round sets GAME_OVER and the game pauses in the same
    frame (update_game_over, game.lua:2601, after the event pass at 2509): the
    rent's ease_dollars (card.lua:2271-2276) and Gros Michel's removal event
    (card.lua:3021-3036) never run. The perishable tick (card.lua:2278-2289)
    and the extinction flag (card.lua:3037) happen at once."""
    gs = select_blind(new_run("G265S002"))
    bot_add(gs, "j_gros_michel")
    bot_add(gs, "j_chaos", rental=True, perishable=True)
    gs["jokers"][-1].ability["perish_tally"] = 3
    gs["current_round"]["hands_left"] = 1
    low = min(range(len(gs["hand"])), key=lambda i: gs["hand"][i].base.id)
    dollars = gs["dollars"]
    step(gs, PlayHand((low,)))
    assert gs["phase"] == GamePhase.GAME_OVER
    assert gs["dollars"] == dollars
    assert [j.center_key for j in gs["jokers"]] == ["j_gros_michel", "j_chaos"]
    assert gs["jokers"][1].ability["perish_tally"] == 2
    assert gs["pool_flags"]["gros_michel_extinct"] is True


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

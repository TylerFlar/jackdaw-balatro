"""A lost run: end_round sets GAME_OVER and the game pauses in the same frame, so
nothing end_round queued ever runs (G265, found by the live lockstep)."""

from __future__ import annotations

from jackdaw.core import Game
from jackdaw.engine.actions import GamePhase, PlayHand, SelectBlind

RANKS = "23456789TJQKA"


def _lose(seed: str) -> tuple[Game, float]:
    """Lose the Small Blind with one low card, holding Gros Michel and a rental,
    perishable Chaos the Clown. Returns the run and the money before the play."""
    g = Game.new("b_red", 1, seed)
    g.step(SelectBlind())
    g.debug_add("j_gros_michel")
    g.debug_add("j_chaos", rental=True, perishable=3)
    g.debug_set(hands=1)
    low = min(range(len(g.hand)), key=lambda i: RANKS.index(g.hand[i].front[2:]))
    dollars = g.dollars
    g.step(PlayHand((low,)))
    assert g.phase == GamePhase.GAME_OVER
    return g, dollars


def test_a_lost_round_charges_no_rent_but_ticks_perishables() -> None:
    g, dollars = _lose("G265S002")
    assert g.dollars == dollars  # calculate_rental's ease_dollars is an event
    chaos = next(c for c in g.jokers if c.key == "j_chaos")
    assert chaos.perish_tally == 2  # calculate_perishable changes the card at once


def test_an_extinct_gros_michel_stays_when_the_run_is_lost() -> None:
    g, _ = _lose("G265S002")
    assert [c.key for c in g.jokers] == ["j_gros_michel", "j_chaos"]  # removal: an event
    assert g.to_dict()["sets"]["pool_flags"]["gros_michel_extinct"] is True  # set at once

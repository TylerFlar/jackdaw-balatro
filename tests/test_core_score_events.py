"""Scoring events (``Game.record_scoring``, ``Game.last_play``).

Every ported joker's frozen scenarios replay with recording on: the states are the frozen ones,
so recording changes nothing, and each play's events add up, one by one, to what it scored.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import pytest

from jackdaw.core import Game, Play
from jackdaw.engine.actions import PlayHand, SelectBlind
from tests.parity import frozen

_SCENARIOS = frozen.load("joker_scenarios.json")
# Events that set the chips and mult outright, rather than adding to them.
_SETS = {"level_up", "hand", "blind", "not_allowed", "plasma"}


def _same(a: float, b: float) -> bool:
    return a == b or (math.isnan(a) and math.isnan(b))


def _floor(x: float) -> float:
    return float(math.floor(x)) if math.isfinite(x) else x


def check_play(play: Play, selected: Sequence[int]) -> None:
    """The play's cards are the selected ones, and its events replay to its score."""
    assert sorted(play.played) == sorted(selected)
    assert set(play.scoring) <= set(play.played)
    assert [e.kind for e in play.events].count("hand") == 1
    pair: tuple[float, float] | None = None
    dollars = 0.0
    for ev in play.events:
        dollars += ev.dollars or 0.0
        if ev.kind == "level_up":
            assert pair is None, "a level-up after scoring started"
            continue
        if ev.kind in _SETS:
            pair = (ev.hand_chips, ev.hand_mult)
            continue
        assert pair is not None, f"{ev} before the hand's base"
        chips, mult = pair
        if ev.chips is not None:
            chips += ev.chips
        if ev.mult is not None:
            mult += ev.mult
        if ev.x_mult is not None:
            mult *= ev.x_mult
        if ev.swap:
            chips, mult = mult, chips
        assert _same(chips, ev.hand_chips) and _same(mult, ev.hand_mult), ev
        pair = (chips, mult)
    assert pair is not None
    assert _same(pair[0], play.score.chips) and _same(pair[1], play.score.mult)
    assert _same(_floor(pair[0] * pair[1]), play.score.total)
    assert dollars == pytest.approx(play.score.dollars)


@pytest.mark.parametrize("key", sorted(_SCENARIOS))
def test_recorded_scenarios_reach_the_frozen_states(key: str) -> None:
    kinds: Counter[str] = Counter()
    for case in _SCENARIOS[key]:
        game = frozen.new_game(case["start"], case["setup"])
        game.record_scoring = True
        for action, digest in zip(case["tape"], case["digests"], strict=True):
            selected = (
                [game.hand[i].sort_id for i in action[1]] if action[0] == "PlayHand" else None
            )
            game.step(action)
            assert frozen.short(game.state_hash()) == digest, f"{key}: {action}"
            play = game.last_play
            if selected is None:
                assert play is None, f"{key}: a trace after {action}"
                continue
            assert play is not None, f"{key}: no trace after {action}"
            check_play(play, selected)
            kinds.update(e.kind for e in play.events)
    assert kinds["hand"] > 0


def test_a_glass_card_shatters_and_a_red_seal_retriggers() -> None:
    game = Game.new("b_red", 1, "TRACE1")
    game.record_scoring = True
    game.step(SelectBlind())
    glass = game.debug_add("H_A", enhancement="m_glass", seal="Red")
    joker = game.debug_add("j_joker", edition="holo")
    assert game.preview_play([0]).total > 0  # played on a copy: no trace here
    assert game.last_play is None
    position = next(i for i, c in enumerate(game.hand) if c.sort_id == glass)
    game.step(PlayHand((position,)))
    play = game.last_play
    assert play is not None and play.played == (glass,) and play.scoring == (glass,)
    kinds = [(e.kind, e.card) for e in play.events]
    assert kinds[0] == ("hand", None)
    assert ("retrigger", glass) in kinds  # the Red Seal
    assert kinds.count(("card", glass)) == 4  # chips and x2 mult, twice
    assert ("joker", joker) in kinds and ("edition", joker) in kinds
    check_play(play, [glass])
    game.record_scoring = False
    assert game.last_play is None

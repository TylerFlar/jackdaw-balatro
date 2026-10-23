"""Audit regressions: blinds (boss effects, disable, defeat, debuff re-checks,
blind amounts).

Each test cites the vanilla 1.0.1o Lua it pins (``balatro_source/``).
"""

from __future__ import annotations

import math

import pytest

from jackdaw.engine.actions import GamePhase, PlayHand, SelectBlind
from jackdaw.engine.blind import Blind
from jackdaw.engine.data.blind_scaling import get_blind_amount, get_blind_target
from jackdaw.engine.game import step
from tests.engine.audit._boards import new_run
from tests.parity import canon
from tests.parity.py_export import export_state

# ---------------------------------------------------------------------------
# G51 — get_blind_amount is NaN once the formula overflows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scaling", [1, 2, 3])
@pytest.mark.parametrize("ante", [39, 40, 100])
def test_g51_overflowed_blind_amount_is_nan(scaling: int, ante: int):
    """get_blind_amount (misc_functions.lua:919-954) rounds with
    ``amount - amount % 10^floor(log10(amount) - 1)``.  Once the formula
    overflows (ante 39), amount is inf, log10 gives inf, and Lua's
    ``inf % inf`` is ``inf - floor(inf/inf)*inf`` = NaN, so it returns NaN —
    not inf.  The blind's chips (blind.lua:107) are NaN too."""
    assert math.isnan(get_blind_amount(ante, scaling))
    assert math.isnan(get_blind_target(ante, "Boss", scaling))
    assert math.isnan(Blind.create("bl_small", ante, scaling).chips)


def test_g51_last_finite_ante_is_unchanged():
    assert get_blind_amount(38, 1) == 4.5e288


def test_g51_nan_blind_can_never_be_beaten():
    """The win checks are ``G.GAME.chips - G.GAME.blind.chips >= 0``
    (state_events.lua:96, game.lua:3196): with a NaN target the difference
    is NaN and the comparison is false for any score, inf included."""
    blind = Blind.create("bl_small", 39)
    for chips in (0, 10**300, math.inf):
        assert not blind.beaten_by(chips)


def test_g51_nan_round_plays_to_game_over_without_crashing():
    """A run at ante 39 sets a NaN blind and plays on; when the hands run out
    the round is lost (end_round's game_over, state_events.lua:96-116)."""
    gs = new_run("AUDIT_G51")
    gs["round_resets"]["ante"] = 39
    step(gs, SelectBlind())
    assert math.isnan(gs["blind"].chips)
    gs["chips"] = math.inf
    canon.digest(export_state(gs))  # NaN and inf encode without raising

    while gs["phase"] == GamePhase.SELECTING_HAND:
        step(gs, PlayHand(card_indices=(0,)))

    assert gs["phase"] == GamePhase.GAME_OVER


def test_g51_disabling_a_nan_wall_keeps_nan():
    """Blind:disable halves The Wall (blind.lua:377-380): NaN/2 is NaN."""
    wall = Blind.create("bl_wall", 39)
    wall.disable()
    assert math.isnan(wall.chips)
    vessel = Blind.create("bl_final_vessel", 39)
    vessel.disable()
    assert math.isnan(vessel.chips)

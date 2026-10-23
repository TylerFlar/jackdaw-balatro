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
from jackdaw.engine.rng import PseudoRandom
from tests.engine.audit._boards import force_boss, give_joker, ids, new_run
from tests.parity import canon
from tests.parity.py_export import export_state

# ---------------------------------------------------------------------------
# G10 — Amber Acorn flips at set time and shuffles three times with 'aajk'
# ---------------------------------------------------------------------------


def _three_aajk_shuffles(seed: str, cards: list) -> list:
    """What three ``G.jokers:shuffle('aajk')`` calls leave, from a fresh stream."""
    ref = PseudoRandom(seed)
    order = list(cards)
    for _ in range(3):
        ref.shuffle(order, ref.seed("aajk"))
    return order


def test_g10_amber_acorn_shuffles_three_times():
    """Amber Acorn's set_blind queues three ``G.jokers:shuffle('aajk')`` events
    (blind.lua:195-203); each pseudoseed('aajk') call advances the stream, so
    the board ends in the third shuffle's order with the stream three steps
    on.  The jokers were flipped face down first (blind.lua:190-194)."""
    seed = "AUDIT_G10"
    gs = new_run(seed)
    board = [give_joker(gs, key) for key in ("j_joker", "j_greedy_joker", "j_lusty_joker")]
    force_boss(gs, "bl_final_acorn")

    step(gs, SelectBlind())

    assert ids(gs["jokers"]) == ids(_three_aajk_shuffles(seed, board))
    ref = PseudoRandom(seed)
    for _ in range(3):
        ref.seed("aajk")
    assert gs["rng"].state["aajk"] == ref.state["aajk"]
    assert all(j.facing == "back" for j in gs["jokers"])


def test_g10_jokers_made_at_setting_blind_are_shuffled_but_face_up():
    """The flip runs inside set_blind, before the jokers' setting_blind
    context (state_events.lua:333-337), so Riff-raff's new jokers are not
    flipped; the shuffles run later from queued events (blind.lua:196-203)
    and include them."""
    seed = "AUDIT_G10R"
    gs = new_run(seed)
    riff = give_joker(gs, "j_riff_raff")
    force_boss(gs, "bl_final_acorn")

    step(gs, SelectBlind())

    made = [j for j in gs["jokers"] if j is not riff]
    assert len(made) == 2
    assert riff.facing == "back"
    assert all(j.facing == "front" for j in made)
    assert ids(gs["jokers"]) == ids(_three_aajk_shuffles(seed, gs["jokers"]))


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

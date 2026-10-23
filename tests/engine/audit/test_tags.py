"""Audit regressions: tags (G40-G43)."""

from __future__ import annotations

from jackdaw.engine import (
    CashOut,
    NextRound,
    SelectBlind,
    SkipBlind,
    step,
)
from jackdaw.engine.data.hands import HAND_BASE
from jackdaw.engine.rng import PseudoRandom
from jackdaw.engine.tags import _ORBITAL_HANDS
from tests.engine.audit._helpers import new_run, win_blind

# ---------------------------------------------------------------------------
# G40 — Orbital Tag
# ---------------------------------------------------------------------------

_SECRET = {"Flush Five", "Flush House", "Five of a Kind"}


def _orbital_draws(gs: dict) -> float | None:
    return gs["rng"].state.get("orbital")


def test_orbital_choices_rolled_when_blind_select_is_built():
    """Three Orbital hands per ante, rolled when the blind-select screen is built.

    UI_definitions.lua:1506-1516 (create_UIBox_blind_choice, called for
    Small, Big and Boss by create_UIBox_blind_select): if
    ``G.GAME.orbital_choices[ante][type]`` is unset it is set to
    ``pseudorandom_element(visible hands, pseudoseed('orbital'))``.  The
    screen is built at run start and on every return from the shop
    (game.lua:3258-3298); later builds in an ante find the choices set.
    """
    seed = "AUDIT_G40"
    gs = new_run(seed=seed)
    ref = PseudoRandom(seed)
    visible = [ht.value for ht in _ORBITAL_HANDS if ht.value not in _SECRET]
    expected = {
        blind: ref.element(visible, ref.seed("orbital"))[0] for blind in ("Small", "Big", "Boss")
    }
    assert gs["orbital_choices"] == {1: expected}

    # Back from the shop in the same ante: no new rolls.
    after_init = _orbital_draws(gs)
    step(gs, SelectBlind())
    win_blind(gs)
    step(gs, CashOut())
    step(gs, NextRound())
    assert _orbital_draws(gs) == after_init

    # A new ante rolls its own three (here after the boss).
    gs["blind_on_deck"] = "Boss"
    step(gs, SelectBlind())
    win_blind(gs)
    step(gs, CashOut())
    step(gs, NextRound())
    assert set(gs["orbital_choices"]) == {1, 2}
    for blind in ("Small", "Big", "Boss"):
        ref.element(visible, ref.seed("orbital"))
    assert _orbital_draws(gs) == ref.state["orbital"]


def test_orbital_tag_levels_the_prerolled_hand():
    """The Orbital Tag levels the hand rolled for its blind.

    tag.lua:103-112 (Tag:set_ability): a tag made for a blind takes
    ``orbital_choices[ante][blind_type]``; tag.lua:191-197 levels that hand
    up 3 times when the tag applies.
    """
    gs = new_run(seed="AUDIT_G40_SKIP")
    gs["round_resets"]["blind_tags"]["Small"] = "tag_orbital"
    hand = gs["orbital_choices"][1]["Small"]
    step(gs, SkipBlind())
    levels = {ht.value: gs["hand_levels"][ht].level for ht in HAND_BASE}
    assert levels[hand] == 4
    assert all(lvl == 1 for name, lvl in levels.items() if name != hand)


def test_double_tag_copy_keeps_the_orbital_hand():
    """A Double Tag's copy of an Orbital Tag levels the same hand.

    tag.lua:324-328: the copy is made with ``G.orbital_hand`` set to the
    original tag's ``ability.orbital_hand``, and Tag:set_ability takes it
    (tag.lua:105-106).
    """
    from jackdaw.engine.game import _award_tag

    gs = new_run(seed="AUDIT_G40_DOUBLE")
    _award_tag(gs, "tag_double", source="test")
    gs["round_resets"]["blind_tags"]["Small"] = "tag_orbital"
    hand = gs["orbital_choices"][1]["Small"]
    step(gs, SkipBlind())
    levels = {ht.value: gs["hand_levels"][ht].level for ht in HAND_BASE}
    assert levels[hand] == 7
    assert all(lvl == 1 for name, lvl in levels.items() if name != hand)

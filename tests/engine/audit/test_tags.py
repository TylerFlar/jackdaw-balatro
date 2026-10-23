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


# ---------------------------------------------------------------------------
# G41 — Boss Tag
# ---------------------------------------------------------------------------


def test_boss_tag_reroll_marks_the_ante_rerolled_and_respects_bans():
    """The Boss Tag rerolls through G.FUNCS.reroll_boss, for free.

    tag.lua:284-300 sets G.from_boss_tag and calls G.FUNCS.reroll_boss
    (button_callbacks.lua:2800-2853): ``boss_rerolled = true`` (so
    Director's Cut can no longer reroll this ante, :2784-2787), no $10
    from a tag, and get_new_boss, which drops banned blinds
    (common_events.lua:2359-2361).
    """
    for i in range(6):
        gs = new_run(challenge="c_jokerless_1", seed=f"AUDIT_G41_{i}")
        gs["round_resets"]["ante"] = 8  # showdown pool: 3 of 5 bosses banned
        gs["round_resets"]["blind_tags"]["Small"] = "tag_boss"
        dollars = gs["dollars"]
        step(gs, SkipBlind())
        rr = gs["round_resets"]
        assert rr["boss_rerolled"] is True
        assert rr["blind_choices"]["Boss"] in {"bl_final_vessel", "bl_final_bell"}
        assert gs["dollars"] == dollars


# ---------------------------------------------------------------------------
# G42 — Coupon Tag
# ---------------------------------------------------------------------------


def _pending(gs: dict) -> list[str]:
    from tests.parity.py_export import pending_tags

    return [t["key"] for t in pending_tags(gs.get("awarded_tags", []))]


def _shop_is_free(gs: dict) -> bool:
    return all(c.cost == 0 for c in gs["shop_cards"] + gs["shop_boosters"])


def test_coupon_tag_applies_once_per_shop():
    """Only one Coupon Tag works per shop; a second waits for the next shop.

    tag.lua:447-465: the Coupon Tag applies only ``if G.shop and not
    G.GAME.shop_free``, and sets ``G.GAME.shop_free``; cash_out clears it
    (button_callbacks.lua:2932).  An unapplied tag is not triggered, so it
    stays for the next shop.
    """
    from jackdaw.engine.game import _award_tag

    gs = new_run(seed="AUDIT_G42")
    step(gs, SelectBlind())
    win_blind(gs)
    _award_tag(gs, "tag_coupon", source="test")
    _award_tag(gs, "tag_coupon", source="test")
    step(gs, CashOut())
    assert _shop_is_free(gs)
    assert _pending(gs) == ["tag_coupon"]

    step(gs, NextRound())
    step(gs, SelectBlind())
    win_blind(gs)
    step(gs, CashOut())
    assert _shop_is_free(gs)
    assert _pending(gs) == []

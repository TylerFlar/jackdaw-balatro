"""Audit regressions: actions vanilla allows that the engine lacked (G45, G46)."""

from __future__ import annotations

import pytest

import jackdaw.engine
from jackdaw.engine import (
    IllegalActionError,
    RerollBoss,
    SelectBlind,
    get_legal_actions,
    step,
)
from tests.engine.audit._helpers import new_run, win_blind

# ---------------------------------------------------------------------------
# G45 — RerollBoss (Director's Cut / Retcon)
# ---------------------------------------------------------------------------


def test_reroll_boss_is_exported():
    assert "RerollBoss" in jackdaw.engine.__all__


def test_reroll_boss_needs_a_voucher():
    """The reroll button only exists with Retcon or Director's Cut.

    UI_definitions.lua:1427-1428 (create_UIBox_blind_select) adds it only
    when either voucher is redeemed.
    """
    gs = new_run(seed="AUDIT_G45_NONE")
    gs["dollars"] = 50
    assert RerollBoss() not in get_legal_actions(gs)
    with pytest.raises(IllegalActionError):
        step(gs, RerollBoss())


def test_directors_cut_rerolls_once_per_ante_for_ten_dollars():
    """Director's Cut: one $10 boss reroll per ante.

    button_callbacks.lua:2784-2798 (reroll_boss_button): enabled when
    ``dollars - bankrupt_at - 10 >= 0`` and (Retcon or (Director's Cut and
    not boss_rerolled)); :2800-2853 (reroll_boss): ``boss_rerolled =
    true``, ``ease_dollars(-10)`` and a new get_new_boss ('boss' stream).
    reset_blinds clears boss_rerolled with the next ante
    (common_events.lua:2334).
    """
    gs = new_run(seed="AUDIT_G45_DC")
    gs["used_vouchers"]["v_directors_cut"] = True
    gs["dollars"] = 25
    boss_draws = gs["rng"].state["boss"]
    assert RerollBoss() in get_legal_actions(gs)
    step(gs, RerollBoss())
    rr = gs["round_resets"]
    assert gs["dollars"] == 15
    assert rr["boss_rerolled"] is True
    assert gs["rng"].state["boss"] != boss_draws  # one more 'boss' pick
    assert RerollBoss() not in get_legal_actions(gs)
    with pytest.raises(IllegalActionError):
        step(gs, RerollBoss())

    # Next ante: available again.
    gs["blind_on_deck"] = "Boss"
    step(gs, SelectBlind())
    win_blind(gs)
    assert rr["boss_rerolled"] is False


def test_retcon_rerolls_without_limit_and_credit_card_pays():
    """Retcon: unlimited $10 rerolls; the gate uses dollars - bankrupt_at.

    button_callbacks.lua:2785-2787.
    """
    gs = new_run(challenge="c_golden_needle_1", seed="AUDIT_G45_RETCON")  # Credit Card
    gs["used_vouchers"]["v_retcon"] = True
    gs["dollars"] = 5  # 5 - (-20) - 10 >= 0
    step(gs, RerollBoss())
    step(gs, RerollBoss())
    assert gs["dollars"] == -15
    assert RerollBoss() not in get_legal_actions(gs)  # -15 + 20 - 10 < 0

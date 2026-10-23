"""Audit regressions: actions vanilla allows that the engine lacked (G45, G46)."""

from __future__ import annotations

import pytest

import jackdaw.engine
from jackdaw.engine import (
    BuyAndUse,
    BuyCard,
    CashOut,
    IllegalActionError,
    RerollBoss,
    SelectBlind,
    get_legal_actions,
    step,
)
from jackdaw.engine.card_factory import create_consumable
from jackdaw.engine.data.hands import HandType
from tests.engine.audit._helpers import new_run, play_round, win_blind

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

    # Next ante: available again once cash_out's reset_blinds has run.
    gs["blind_on_deck"] = "Boss"
    step(gs, SelectBlind())
    win_blind(gs)
    assert rr["boss_rerolled"] is True
    step(gs, CashOut())
    assert gs["round_resets"]["boss_rerolled"] is False


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


# ---------------------------------------------------------------------------
# G46 — BuyAndUse
# ---------------------------------------------------------------------------


def _shop_with(gs: dict, key: str) -> int:
    """Put a consumable *key* in the shop (priced) and return its index."""
    card = create_consumable(key)
    card.set_cost()
    gs["shop_cards"].append(card)
    return len(gs["shop_cards"]) - 1


def _fill_consumables(gs: dict) -> None:
    while len(gs["consumables"]) < gs["consumable_slots"]:
        gs["consumables"].append(create_consumable("c_strength"))


def test_buy_and_use_skips_the_slot_check():
    """A usable shop consumable can be bought and used at once, slots full or not.

    button_callbacks.lua:77-89 (can_buy_and_use) enables the button when
    the card is affordable (``cost <= dollars - bankrupt_at`` or free) and
    ``can_use_consumeable()``; buy_from_shop (:2404-2477) with the
    'buy_and_use' id skips check_for_buy_space, never emplaces the card in
    G.consumeables, charges the cost and hands it to use_card.
    """
    gs = new_run(seed="AUDIT_G46")
    play_round(gs)
    gs["dollars"] = 20
    i = _shop_with(gs, "c_mercury")  # Planet: Pair
    _fill_consumables(gs)
    held = list(gs["consumables"])
    planet = gs["shop_cards"][i]

    legal = get_legal_actions(gs)
    assert BuyCard(shop_index=i) not in legal  # no room
    assert BuyAndUse(shop_index=i) in legal

    step(gs, BuyAndUse(shop_index=i))
    assert gs["dollars"] == 20 - planet.cost
    assert gs["hand_levels"][HandType.PAIR].level == 2
    assert [id(c) for c in gs["consumables"]] == [id(c) for c in held]
    assert all(c is not planet for c in gs["shop_cards"])
    assert gs["last_tarot_planet"] == "c_mercury"


def test_buy_and_use_needs_a_usable_card():
    """can_buy_and_use requires can_use_consumeable (card.lua:1523-1579): in the
    shop a card that needs a hand selection (The Magician) is not usable."""
    gs = new_run(seed="AUDIT_G46_NO")
    play_round(gs)
    gs["dollars"] = 20
    i = _shop_with(gs, "c_magician")
    assert BuyAndUse(shop_index=i) not in get_legal_actions(gs)
    with pytest.raises(IllegalActionError):
        step(gs, BuyAndUse(shop_index=i))

"""Legal actions and the steps that take them follow vanilla's button gates
(G201-G203). Line references are into ``balatro_source/``."""

from __future__ import annotations

import pytest

from jackdaw.engine.actions import (
    BuyCard,
    GamePhase,
    OpenBooster,
    SellCard,
    SkipPack,
    get_legal_actions,
)
from jackdaw.engine.card import Card
from jackdaw.engine.card_factory import create_joker
from jackdaw.engine.game import IllegalActionError, step

from ._joker_helpers import give_joker, new_run, select_blind, to_shop, win_round


def _negative_joker_in_shop(seed: str) -> dict:
    gs = to_shop(new_run(seed))
    gs["dollars"] = 100
    card = create_joker("j_joker", edition={"negative": True}, hands_played=gs["hands_played"])
    card.set_cost(ante=gs["round_resets"]["ante"])
    gs["shop_cards"][0] = card
    return gs


class TestNegativeShopJokerNeedsRoom:
    """G201: check_for_buy_space (button_callbacks.lua:2392-2402) lets a Negative
    joker in only while the jokers number fewer than the limit plus one."""

    def test_refused_when_the_jokers_already_exceed_the_limit(self):
        gs = _negative_joker_in_shop("AUDIT_G201_FULL")
        give_joker(gs, "j_greedy_joker")
        gs["joker_slots"] = 0  # Typecast's set_joker_slots_ante (state_events.lua:245-247)
        assert BuyCard(shop_index=0) not in get_legal_actions(gs)
        with pytest.raises(IllegalActionError):
            step(gs, BuyCard(shop_index=0))

    def test_bought_into_the_extra_slot(self):
        gs = _negative_joker_in_shop("AUDIT_G201_ROOM")
        give_joker(gs, "j_greedy_joker")
        gs["joker_slots"] = 1
        assert BuyCard(shop_index=0) in get_legal_actions(gs)
        step(gs, BuyCard(shop_index=0))
        assert [j.center_key for j in gs["jokers"]] == ["j_greedy_joker", "j_joker"]


def _arcana_pack(seed: str) -> dict:
    gs = to_shop(new_run(seed))
    pack = Card()
    pack.set_ability("p_arcana_normal_1")
    pack.set_cost()
    gs["shop_boosters"] = [pack]
    gs["dollars"] = 50
    step(gs, OpenBooster(card_index=0))
    assert gs["phase"] == GamePhase.PACK_OPENING
    return gs


class TestSkipPackNeedsTheDealtHand:
    """G202: can_skip_booster (button_callbacks.lua:2132-2141) lets an Arcana or
    Spectral pack go only once its hand is dealt (or the hand holds no cards)."""

    def test_an_empty_hand_blocks_the_skip(self):
        gs = _arcana_pack("AUDIT_G202_EMPTY")
        gs["deck"].extend(gs["hand"])
        gs["hand"].clear()
        assert SkipPack() not in get_legal_actions(gs)
        with pytest.raises(IllegalActionError):
            step(gs, SkipPack())

    def test_a_dealt_hand_allows_it(self):
        gs = _arcana_pack("AUDIT_G202_DEALT")
        assert gs["hand"]
        assert SkipPack() in get_legal_actions(gs)
        step(gs, SkipPack())
        assert gs["phase"] == GamePhase.SHOP

    def test_no_hand_size_allows_it(self):
        gs = _arcana_pack("AUDIT_G202_NOSIZE")
        gs["deck"].extend(gs["hand"])
        gs["hand"].clear()
        gs["hand_size"] = 0
        assert SkipPack() in get_legal_actions(gs)
        step(gs, SkipPack())
        assert gs["phase"] == GamePhase.SHOP


class TestSellingAtRoundEval:
    """G203: Card:can_sell_card (card.lua:1640-1653) has no state gate, so a joker
    can be sold on the cash-out screen."""

    def test_a_joker_can_be_sold_before_cashing_out(self):
        gs = select_blind(new_run("AUDIT_G203"))
        joker = give_joker(gs, "j_joker")
        win_round(gs)
        assert gs["phase"] == GamePhase.ROUND_EVAL
        action = SellCard(area="jokers", card_index=0)
        assert action in get_legal_actions(gs)
        before = gs["dollars"]
        step(gs, action)
        assert gs["jokers"] == []
        assert gs["dollars"] == before + joker.sell_cost
        assert gs["phase"] == GamePhase.ROUND_EVAL

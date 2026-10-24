"""The bot layer on ``jackdaw.core.Game``: the view, requests, legality and the debug ops."""

from __future__ import annotations

import pytest

from jackdaw.core import Game, IllegalActionError, Legality
from jackdaw.engine.actions import PlayHand, RerollBoss, SelectBlind, SortHand


def test_view_and_requests() -> None:
    g = Game.new("b_red", 1, "FACADE")
    view = g.bot_view()
    assert (view["state"], view["deck"], view["stake"], view["seed"]) == (
        "BLIND_SELECT",
        "RED",
        "WHITE",
        "FACADE",
    )
    assert g.bot_request(SelectBlind()) == ("select", {})
    assert g.bot_request(RerollBoss()) is None
    g.step(SelectBlind())
    assert g.bot_request(PlayHand(card_indices=(1, 0))) == ("play", {"cards": [1, 0]})
    method, params = g.bot_request(SortHand(mode="suit"))
    assert method == "rearrange" and sorted(params["hand"]) == list(range(8))


def test_legality() -> None:
    g = Game.new("b_red", 1, "FACADE_LEGAL")
    vanilla = g.legal_actions()
    bot = g.legal_actions(Legality.BALATROBOT)
    assert g.legal_actions("balatrobot") == bot
    assert SelectBlind() in bot and all(a in vanilla for a in bot)
    with pytest.raises(ValueError):
        g.legal_actions("live")


def test_debug_ops() -> None:
    g = Game.new("b_red", 1, "FACADE_DEBUG")
    g.step(SelectBlind())
    before = len(g.bot_view()["jokers"]["cards"])
    assert isinstance(g.debug_add("j_joker", edition="foil"), int)
    g.debug_add("H_A", seal="Red", enhancement="m_steel")
    g.debug_set(money=99, hands=2)
    view = g.bot_view()
    assert len(view["jokers"]["cards"]) == before + 1
    assert view["hand"]["cards"][-1]["modifier"] == {"seal": "RED", "enhancement": "STEEL"}
    assert (view["money"], view["round"]["hands_left"]) == (99, 2)
    with pytest.raises(IllegalActionError):
        g.debug_restock_shop()
    with pytest.raises(ValueError):
        g.debug_add("j_nope")
    g.apply_setup({"op": "set", "dollars": 7})
    assert g.dollars == 7

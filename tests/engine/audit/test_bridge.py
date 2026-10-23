"""Audit regressions: the in-process BalatroBot backend (B03, B04)."""

from __future__ import annotations

import pytest

from jackdaw.bridge.backend import NOT_ALLOWED, RPCError, SimBackend
from jackdaw.engine import GamePhase
from tests.engine.audit._helpers import play_round


def _started(seed: str) -> SimBackend:
    backend = SimBackend()
    backend.handle("start", {"deck": "RED", "stake": "WHITE", "seed": seed})
    return backend


def test_set_shop_restocks_the_shop():
    """``set`` with ``shop=true`` re-stocks the shop like BalatroBot does.

    third_party/balatrobot/v1.5.2/src/lua/endpoints/set.lua:170-189: only in
    SHOP (else NOT_ALLOWED), it removes G.shop, nils
    ``current_round.used_packs`` and reruns ``G:update_shop()``
    (game.lua:3072-3181): new jokers (create_card_for_shop) and packs
    (get_pack), the same voucher (``current_round.voucher``).
    """
    backend = _started("AUDIT_B03")
    with pytest.raises(RPCError) as exc:
        backend.handle("set", {"shop": True})
    assert exc.value.code == NOT_ALLOWED

    gs = backend._gs
    play_round(gs)
    old_cards = list(gs["shop_cards"])
    voucher = [v.center_key for v in gs["shop_vouchers"]]
    cdt = gs["rng"].state["cdt1"]

    backend.handle("set", {"shop": True})
    assert gs["phase"] == GamePhase.SHOP
    assert len(gs["shop_cards"]) == gs["shop"]["joker_max"]
    assert all(c is not o for c in gs["shop_cards"] for o in old_cards)
    assert [v.center_key for v in gs["shop_vouchers"]] == voucher
    assert len(gs["shop_boosters"]) == 2
    assert gs["rng"].state["cdt1"] != cdt


def test_add_counts_a_jokers_discards_once():
    """``add`` of a d_size joker adds its discards once.

    card.lua:589-592 (Card:add_to_deck): ``round_resets.discards += d_size``
    and ``ease_discard(d_size)`` for the current round; BalatroBot's add
    goes through add_to_deck once.
    """
    backend = _started("AUDIT_B04")
    backend.handle("select", {})
    gs = backend._gs
    left = gs["current_round"]["discards_left"]
    per_round = gs["round_resets"]["discards"]
    backend.handle("add", {"key": "j_drunkard"})
    assert gs["current_round"]["discards_left"] == left + 1
    assert gs["round_resets"]["discards"] == per_round + 1

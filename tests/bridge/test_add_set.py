"""Tests for SimBackend's add/set debug methods (BalatroBot's add.lua, set.lua)."""

from __future__ import annotations

import pytest

from jackdaw.bridge.backend import BAD_REQUEST, INVALID_STATE, RPCError, SimBackend


@pytest.fixture
def backend() -> SimBackend:
    """A SimBackend with an active game in SELECTING_HAND."""
    b = SimBackend()
    b.handle("start", {"deck": "RED", "stake": "WHITE", "seed": "ADD_SET_TEST"})
    b.handle("select", {})  # advance to SELECTING_HAND
    return b


class TestSet:
    def test_set_money(self, backend: SimBackend) -> None:
        assert backend.handle("set", {"money": 100})["money"] == 100
        assert backend.game.dollars == 100

    def test_set_hands(self, backend: SimBackend) -> None:
        assert backend.handle("set", {"hands": 10})["round"]["hands_left"] == 10

    def test_set_discards(self, backend: SimBackend) -> None:
        assert backend.handle("set", {"discards": 7})["round"]["discards_left"] == 7

    def test_set_ante(self, backend: SimBackend) -> None:
        assert backend.handle("set", {"ante": 5})["ante_num"] == 5

    def test_set_round(self, backend: SimBackend) -> None:
        assert backend.handle("set", {"round": 3})["round_num"] == 3

    def test_set_chips(self, backend: SimBackend) -> None:
        assert backend.handle("set", {"chips": 500})["round"]["chips"] == 500

    def test_set_multiple(self, backend: SimBackend) -> None:
        view = backend.handle("set", {"money": 50, "ante": 3, "round": 2})
        assert (view["money"], view["ante_num"], view["round_num"]) == (50, 3, 2)

    def test_set_nothing(self, backend: SimBackend) -> None:
        with pytest.raises(RPCError, match="Must provide at least one field to set") as exc:
            backend.handle("set", {})
        assert exc.value.code == BAD_REQUEST

    def test_set_negative_money(self, backend: SimBackend) -> None:
        with pytest.raises(RPCError, match="Money must be a positive integer"):
            backend.handle("set", {"money": -1})

    def test_set_no_active_run(self) -> None:
        b = SimBackend()
        with pytest.raises(RPCError, match="Can only set during an active run") as exc:
            b.handle("set", {"money": 100})
        assert exc.value.code == INVALID_STATE


def _last(view: dict, area: str) -> dict:
    return view[area]["cards"][-1]


class TestAdd:
    def test_add_joker(self, backend: SimBackend) -> None:
        before = len(backend.view()["jokers"]["cards"])
        view = backend.handle("add", {"key": "j_joker"})
        assert len(view["jokers"]["cards"]) == before + 1
        assert _last(view, "jokers")["key"] == "j_joker"

    def test_add_joker_with_edition(self, backend: SimBackend) -> None:
        view = backend.handle("add", {"key": "j_joker", "edition": "foil"})
        added = _last(view, "jokers")
        assert added["key"] == "j_joker"
        assert added["modifier"]["edition"] == "FOIL"

    def test_add_joker_eternal(self, backend: SimBackend) -> None:
        view = backend.handle("add", {"key": "j_joker", "eternal": True})
        assert _last(view, "jokers")["modifier"]["eternal"] is True

    def test_add_consumable(self, backend: SimBackend) -> None:
        view = backend.handle("add", {"key": "c_magician"})
        assert _last(view, "consumables")["key"] == "c_magician"

    def test_add_playing_card_to_hand(self, backend: SimBackend) -> None:
        # In SELECTING_HAND, cards go to hand
        before = len(backend.view()["hand"]["cards"])
        view = backend.handle("add", {"key": "H_A"})
        assert len(view["hand"]["cards"]) == before + 1
        assert _last(view, "hand")["key"] == "H_A"

    def test_add_playing_card_with_enhancement(self, backend: SimBackend) -> None:
        view = backend.handle("add", {"key": "H_A", "enhancement": "m_glass"})
        added = _last(view, "hand")
        assert added["modifier"]["enhancement"] == "GLASS"
        assert added["label"] == "Glass Card"

    def test_add_playing_card_with_seal(self, backend: SimBackend) -> None:
        view = backend.handle("add", {"key": "H_A", "seal": "Gold"})
        assert _last(view, "hand")["modifier"]["seal"] == "GOLD"

    def test_add_playing_card_with_edition(self, backend: SimBackend) -> None:
        view = backend.handle("add", {"key": "H_A", "edition": "polychrome"})
        assert _last(view, "hand")["modifier"]["edition"] == "POLYCHROME"

    def test_add_invalid_key(self, backend: SimBackend) -> None:
        with pytest.raises(RPCError, match="Invalid card key format") as exc:
            backend.handle("add", {"key": "z_invalid"})
        assert exc.value.code == BAD_REQUEST

    def test_add_no_key(self, backend: SimBackend) -> None:
        with pytest.raises(RPCError, match="Missing required field 'key'"):
            backend.handle("add", {})

    def test_add_no_active_run(self) -> None:
        b = SimBackend()
        with pytest.raises(RPCError, match="requires one of these states") as exc:
            b.handle("add", {"key": "j_joker"})
        assert exc.value.code == INVALID_STATE

    def test_add_invalid_playing_card(self, backend: SimBackend) -> None:
        with pytest.raises(RPCError, match="Invalid card key format"):
            backend.handle("add", {"key": "X_Z"})

    def test_perishable_is_a_round_count(self, backend: SimBackend) -> None:
        with pytest.raises(RPCError, match="Field 'perishable' must be an integer"):
            backend.handle("add", {"key": "j_joker", "perishable": True})
        view = backend.handle("add", {"key": "j_joker", "perishable": 3})
        assert _last(view, "jokers")["modifier"]["perishable"] == 3

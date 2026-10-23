"""Tests for jackdaw.bridge.backend — SimBackend and LiveBackend."""

from __future__ import annotations

import pytest

from jackdaw.bridge.backend import (
    BAD_REQUEST,
    INVALID_STATE,
    NOT_ALLOWED,
    LiveBackend,
    RPCError,
    SimBackend,
    to_bot_add_params,
)

# ============================================================================
# SimBackend
# ============================================================================


@pytest.fixture()
def sim() -> SimBackend:
    return SimBackend()


def _start_run(sim: SimBackend, seed: str = "TEST") -> dict:
    return sim.handle("start", {"deck": "RED", "stake": "WHITE", "seed": seed})


def _beat_blind(sim: SimBackend) -> dict:
    """``set`` the round's chips to the blind's score, then play one card."""
    view = sim.handle("gamestate", None)
    current = next(b for b in view["blinds"].values() if b["status"] == "CURRENT")
    sim.handle("set", {"chips": current["score"]})
    return sim.handle("play", {"cards": [0]})


class TestSimHealth:
    def test_health(self, sim):
        result = sim.handle("health", None)
        assert result == {"status": "ok"}


class TestSimStartAndGamestate:
    def test_start_returns_gamestate(self, sim):
        result = _start_run(sim)

        assert result["state"] == "BLIND_SELECT"
        assert result["deck"] == "RED"
        assert result["stake"] == "WHITE"
        assert result["seed"] == "TEST"
        assert result["ante_num"] == 1
        assert result["money"] == 4
        assert result["won"] is False
        assert isinstance(result["blinds"], dict)
        assert isinstance(result["hands"], dict)

    def test_gamestate_after_start(self, sim):
        _start_run(sim)
        result = sim.handle("gamestate", None)

        assert result["state"] == "BLIND_SELECT"
        assert result["ante_num"] == 1
        assert sim.state == "BLIND_SELECT"
        assert sim.view() == result

    def test_start_different_deck_and_stake(self, sim):
        result = sim.handle("start", {"deck": "BLUE", "stake": "RED", "seed": "X"})

        assert result["deck"] == "BLUE"
        assert result["stake"] == "RED"

    def test_start_needs_the_menu(self, sim):
        _start_run(sim)
        with pytest.raises(RPCError, match="requires one of these states: MENU") as exc_info:
            _start_run(sim)
        assert exc_info.value.code == INVALID_STATE
        sim.handle("menu", {})
        assert _start_run(sim, seed="AGAIN")["seed"] == "AGAIN"

    def test_the_game_is_a_copy(self, sim):
        assert sim.game is None
        _start_run(sim)
        game = sim.game
        assert game is not None and game.seed == "TEST"
        game.step(["SelectBlind"])
        assert sim.state == "BLIND_SELECT"


class TestSimFullRound:
    def test_select_play_cashout_nextround(self, sim):
        _start_run(sim, seed="ROUND_TEST")

        # Select blind
        result = sim.handle("select", {})
        assert result["state"] == "SELECTING_HAND"
        assert result["hand"]["count"] > 0
        assert result["round"]["hands_left"] > 0

        # Should have won the blind
        result = _beat_blind(sim)
        assert result["state"] == "ROUND_EVAL"

        # Cash out
        result = sim.handle("cash_out", {})
        assert result["state"] == "SHOP"

        # Next round
        result = sim.handle("next_round", {})
        assert result["state"] == "BLIND_SELECT"


class TestSimNoActiveRun:
    def test_gamestate_before_start_is_the_menu(self, sim):
        assert sim.handle("gamestate", None)["state"] == "MENU"

    def test_action_before_start(self, sim):
        with pytest.raises(RPCError) as exc_info:
            sim.handle("select", {})
        assert exc_info.value.code == INVALID_STATE
        assert exc_info.value.data == {"name": "INVALID_STATE"}


class TestSimIllegalAction:
    def test_play_during_blind_select(self, sim):
        _start_run(sim)
        # Like balatrobot's dispatcher: wrong state is INVALID_STATE
        with pytest.raises(RPCError, match="requires one of these states") as exc_info:
            sim.handle("play", {"cards": [0, 1, 2]})
        assert exc_info.value.code == INVALID_STATE

    def test_cash_out_during_blind_select(self, sim):
        _start_run(sim)
        with pytest.raises(RPCError) as exc_info:
            sim.handle("cash_out", {})
        assert exc_info.value.code == INVALID_STATE

    def test_six_cards(self, sim):
        _start_run(sim)
        sim.handle("select", {})
        with pytest.raises(RPCError, match="You can only play 5 cards") as exc_info:
            sim.handle("play", {"cards": [0, 1, 2, 3, 4, 5]})
        assert exc_info.value.code == BAD_REQUEST

    def test_check_is_a_dry_run(self, sim):
        _start_run(sim)
        assert sim.check("select") is None
        assert sim.state == "BLIND_SELECT"
        refusal = sim.check("play", {"cards": [0]})
        assert refusal is not None and refusal.code == INVALID_STATE


class TestSimMatchesBalatrobotStates:
    """The engine follows vanilla, which allows more than balatrobot's
    endpoints accept; the sim refuses what the live bot refuses."""

    def test_use_and_sell_refused_at_blind_select(self, sim):
        _start_run(sim)
        for method in ("use", "sell"):
            with pytest.raises(RPCError) as exc_info:
                sim.handle(method, {"consumable": 0})
            assert exc_info.value.code == INVALID_STATE

    def test_use_and_sell_accepted_while_selecting_hand(self, sim):
        _start_run(sim)
        sim.handle("select", {})
        sim.handle("add", {"key": "c_pluto"})
        sim.handle("add", {"key": "c_pluto"})
        sim.handle("use", {"consumable": 0})
        view = sim.handle("sell", {"consumable": 0})
        assert view["consumables"]["cards"] == []

    def test_card_selecting_consumable_only_while_selecting_hand(self, sim):
        _start_run(sim)
        sim.handle("select", {})
        _beat_blind(sim)
        sim.handle("cash_out", {})
        sim.handle("add", {"key": "c_magician"})
        with pytest.raises(RPCError) as exc_info:
            sim.handle("use", {"consumable": 0, "cards": [0]})
        assert exc_info.value.code == INVALID_STATE
        assert exc_info.value.message == (
            "Consumable 'The Magician' requires card selection and can only be used in "
            "SELECTING_HAND state"
        )

    def test_eternal_jokers_are_not_sold(self, sim):
        _start_run(sim)
        sim.handle("select", {})
        sim.handle("add", {"key": "j_joker", "eternal": True})
        with pytest.raises(RPCError) as exc_info:
            sim.handle("sell", {"joker": 0})
        assert exc_info.value.code == NOT_ALLOWED


class TestSimBadParams:
    def test_buy_empty_params(self, sim):
        _start_run(sim)
        sim.handle("select", {})
        _beat_blind(sim)
        sim.handle("cash_out", {})
        with pytest.raises(RPCError) as exc_info:
            sim.handle("buy", {})
        assert exc_info.value.code == BAD_REQUEST

    def test_sell_empty_params(self, sim):
        _start_run(sim)
        sim.handle("select", {})
        with pytest.raises(RPCError) as exc_info:
            sim.handle("sell", {})
        assert exc_info.value.code == BAD_REQUEST

    def test_unknown_method(self, sim):
        with pytest.raises(RPCError) as exc_info:
            sim.handle("nonexistent", {})
        assert exc_info.value.code == BAD_REQUEST
        assert exc_info.value.message == "Unknown method: nonexistent"


class TestSimMenu:
    def test_menu_ends_the_run(self, sim):
        _start_run(sim)
        result = sim.handle("menu", {})
        assert result["state"] == "MENU"
        assert sim.game is None
        assert sim.handle("gamestate", None)["state"] == "MENU"

    def test_menu_without_run(self, sim):
        assert sim.handle("menu", {})["state"] == "MENU"


class TestSimLegality:
    def test_legal_actions_and_requests(self, sim):
        from jackdaw.engine.actions import PlayHand, SelectBlind, SortHand

        assert sim.legal_actions() == []
        _start_run(sim)
        assert SelectBlind() in sim.legal_actions()
        assert sim.request_for(SelectBlind()) == ("select", {})
        sim.handle("select", {})
        method, params = sim.request_for(SortHand(mode="suit"))
        assert method == "rearrange" and sorted(params["hand"]) == list(range(8))
        assert sim.request_for(PlayHand(card_indices=(2, 0))) == ("play", {"cards": [2, 0]})


def test_add_params_use_bot_names():
    assert to_bot_add_params({"key": "H_A", "enhancement": "m_glass", "seal": "Gold"}) == {
        "key": "H_A",
        "enhancement": "GLASS",
        "seal": "GOLD",
    }
    assert to_bot_add_params({"key": "j_joker", "edition": "NEGATIVE"})["edition"] == "NEGATIVE"


# ============================================================================
# RPCError
# ============================================================================


class TestRPCError:
    def test_fields(self):
        err = RPCError(code=-32001, message="test error", data={"key": "val"})
        assert err.code == -32001
        assert err.message == "test error"
        assert err.data == {"key": "val"}
        assert str(err) == "test error"

    def test_default_data(self):
        err = RPCError(code=-32000, message="oops")
        assert err.data == {}


# ============================================================================
# LiveBackend — requires a running balatrobot instance
# ============================================================================


@pytest.mark.live
class TestLiveBackendHealth:
    def test_health(self):
        backend = LiveBackend()
        result = backend.handle("health", None)
        assert result.get("status") == "ok"


@pytest.mark.live
class TestLiveBackendGamestate:
    def test_start_and_gamestate(self):
        backend = LiveBackend()
        backend.handle("menu", None)
        backend.handle("start", {"deck": "RED", "stake": "WHITE", "seed": "LIVE_TEST"})
        result = backend.handle("gamestate", None)
        assert "state" in result

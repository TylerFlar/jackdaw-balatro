"""Tests for src/lua/endpoints/select.lua"""

import httpx

from tests.lua.conftest import (
    api,
    assert_error_response,
    assert_gamestate_response,
    load_fixture,
)


class TestSelectEndpoint:
    """Test basic select endpoint functionality."""

    def test_select_small_blind(self, client: httpx.Client) -> None:
        """Test selecting Small blind in BLIND_SELECT state."""
        gamestate = load_fixture(
            client, "select", "state-BLIND_SELECT--blinds.small.status-SELECT"
        )
        assert gamestate["state"] == "BLIND_SELECT"
        assert gamestate["blinds"]["small"]["status"] == "SELECT"
        response = api(client, "select", {})
        assert_gamestate_response(response, state="SELECTING_HAND")

    def test_select_big_blind(self, client: httpx.Client) -> None:
        """Test selecting Big blind in BLIND_SELECT state."""
        gamestate = load_fixture(
            client, "select", "state-BLIND_SELECT--blinds.big.status-SELECT"
        )
        assert gamestate["state"] == "BLIND_SELECT"
        assert gamestate["blinds"]["big"]["status"] == "SELECT"
        response = api(client, "select", {})
        assert_gamestate_response(response, state="SELECTING_HAND")

    def test_select_boss_blind(self, client: httpx.Client) -> None:
        """Test selecting Boss blind in BLIND_SELECT state."""
        gamestate = load_fixture(
            client, "select", "state-BLIND_SELECT--blinds.boss.status-SELECT"
        )
        assert gamestate["state"] == "BLIND_SELECT"
        assert gamestate["blinds"]["boss"]["status"] == "SELECT"
        response = api(client, "select", {})
        assert_gamestate_response(response, state="SELECTING_HAND")


class TestSelectEndpointStateRequirements:
    """Test select endpoint state requirements."""

    def test_select_from_MENU(self, client: httpx.Client):
        """Test that select fails when not in BLIND_SELECT state."""
        response = api(client, "menu", {})
        assert_gamestate_response(response, state="MENU")
        assert_error_response(
            api(client, "select", {}),
            "INVALID_STATE",
            "Method 'select' requires one of these states: BLIND_SELECT",
        )

"""Tests for src/lua/endpoints/next_round.lua"""

import httpx

from tests.lua.conftest import (
    api,
    assert_error_response,
    assert_gamestate_response,
    load_fixture,
)


class TestNextRoundEndpoint:
    """Test basic next_round endpoint functionality."""

    def test_next_round_from_shop(self, client: httpx.Client) -> None:
        """Test advancing to next round from SHOP state."""
        gamestate = load_fixture(client, "next_round", "state-SHOP")
        assert gamestate["state"] == "SHOP"
        response = api(client, "next_round", {})
        assert_gamestate_response(response, state="BLIND_SELECT")


class TestNextRoundEndpointStateRequirements:
    """Test next_round endpoint state requirements."""

    def test_next_round_from_MENU(self, client: httpx.Client):
        """Test that next_round fails when not in SHOP state."""
        gamestate = load_fixture(client, "next_round", "state-BLIND_SELECT")
        assert gamestate["state"] == "BLIND_SELECT"
        response = api(client, "next_round", {})
        assert_error_response(
            response,
            "INVALID_STATE",
            "Method 'next_round' requires one of these states: SHOP",
        )

# tests/lua/endpoints/test_health.py
# Tests for src/lua/endpoints/health.lua
#
# Tests the health check endpoint:
# - Basic health check functionality
# - Response structure and fields

import httpx

from tests.lua.conftest import (
    api,
    assert_gamestate_response,
    assert_health_response,
    load_fixture,
)


class TestHealthEndpoint:
    """Test basic health endpoint functionality."""

    def test_health_from_MENU(self, client: httpx.Client) -> None:
        """Test that health check returns status ok."""
        response = api(client, "menu", {})
        assert_gamestate_response(response, state="MENU")
        assert_health_response(api(client, "health", {}))

    def test_health_from_BLIND_SELECT(self, client: httpx.Client) -> None:
        """Test that health check returns status ok."""
        save = "state-BLIND_SELECT"
        gamestate = load_fixture(client, "health", save)
        assert gamestate["state"] == "BLIND_SELECT"
        assert_health_response(api(client, "health", {}))

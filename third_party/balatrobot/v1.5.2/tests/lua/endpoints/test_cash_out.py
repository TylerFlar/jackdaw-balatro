"""Tests for src/lua/endpoints/cash_out.lua"""

import httpx

from tests.lua.conftest import (
    api,
    assert_error_response,
    assert_gamestate_response,
    load_fixture,
)


class TestCashOutEndpoint:
    """Test basic cash_out endpoint functionality."""

    def test_cash_out_from_ROUND_EVAL(self, client: httpx.Client) -> None:
        """Test cashing out from ROUND_EVAL state."""
        gamestate = load_fixture(client, "cash_out", "state-ROUND_EVAL")
        assert gamestate["state"] == "ROUND_EVAL"
        response = api(client, "cash_out", {})
        assert_gamestate_response(response, state="SHOP")


class TestCashOutEndpointStateRequirements:
    """Test cash_out endpoint state requirements."""

    def test_cash_out_from_BLIND_SELECT(self, client: httpx.Client):
        """Test that cash_out fails when not in ROUND_EVAL state."""
        gamestate = load_fixture(client, "cash_out", "state-BLIND_SELECT")
        assert gamestate["state"] == "BLIND_SELECT"
        assert_error_response(
            api(client, "cash_out", {}),
            "INVALID_STATE",
            "Method 'cash_out' requires one of these states: ROUND_EVAL",
        )

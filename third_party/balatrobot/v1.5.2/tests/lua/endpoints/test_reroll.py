"""Tests for src/lua/endpoints/reroll.lua"""

import httpx

from tests.lua.conftest import (
    api,
    assert_error_response,
    assert_gamestate_response,
    load_fixture,
)


class TestRerollEndpoint:
    """Test basic reroll endpoint functionality."""

    def test_reroll_from_shop(self, client: httpx.Client) -> None:
        """Test rerolling shop from SHOP state."""
        before = load_fixture(client, "reroll", "state-SHOP")
        assert before["state"] == "SHOP"
        response = api(client, "reroll", {})
        after = assert_gamestate_response(response, state="SHOP")
        assert before["shop"] != after["shop"]

    def test_reroll_insufficient_funds(self, client: httpx.Client) -> None:
        """Test reroll endpoint when player has insufficient funds."""
        gamestate = load_fixture(client, "reroll", "state-SHOP--money-0")
        assert gamestate["state"] == "SHOP"
        assert gamestate["money"] == 0
        assert_error_response(
            api(client, "reroll", {}),
            "NOT_ALLOWED",
            "Not enough dollars to reroll",
        )

    def test_reroll_with_credit_card_joker(self, client: httpx.Client) -> None:
        """Test rerolling when player has Credit Card joker (can go negative)."""
        # Get to shop state with $0
        gamestate = load_fixture(client, "reroll", "state-SHOP--money-0")
        assert gamestate["state"] == "SHOP"
        assert gamestate["money"] == 0

        # Add Credit Card joker (gives +$20 credit)
        response = api(client, "add", {"key": "j_credit_card"})
        gamestate = assert_gamestate_response(response)
        assert any(j["key"] == "j_credit_card" for j in gamestate["jokers"]["cards"])

        # Should be able to reroll (costs $5 by default) even with $0
        response = api(client, "reroll", {})
        gamestate = assert_gamestate_response(response)
        # Money should be negative now
        assert gamestate["money"] < 0


class TestRerollEndpointStateRequirements:
    """Test reroll endpoint state requirements."""

    def test_reroll_from_BLIND_SELECT(self, client: httpx.Client):
        """Test that reroll fails when not in SHOP state."""
        gamestate = load_fixture(client, "reroll", "state-BLIND_SELECT")
        assert gamestate["state"] == "BLIND_SELECT"
        assert_error_response(
            api(client, "reroll", {}),
            "INVALID_STATE",
            "Method 'reroll' requires one of these states: SHOP",
        )

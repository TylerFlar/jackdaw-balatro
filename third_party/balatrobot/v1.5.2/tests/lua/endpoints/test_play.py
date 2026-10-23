"""Tests for src/lua/endpoints/play.lua"""

import httpx

from tests.lua.conftest import (
    api,
    assert_error_response,
    assert_gamestate_response,
    load_fixture,
)


class TestPlayEndpoint:
    """Test basic play endpoint functionality."""

    def test_play_zero_cards(self, client: httpx.Client) -> None:
        """Test play endpoint from BLIND_SELECT state."""
        gamestate = load_fixture(client, "play", "state-SELECTING_HAND")
        assert gamestate["state"] == "SELECTING_HAND"
        assert_error_response(
            api(client, "play", {"cards": []}),
            "BAD_REQUEST",
            "Must provide at least one card to play",
        )

    def test_play_six_cards(self, client: httpx.Client) -> None:
        """Test play endpoint from BLIND_SELECT state."""
        gamestate = load_fixture(client, "play", "state-SELECTING_HAND")
        assert gamestate["state"] == "SELECTING_HAND"
        assert_error_response(
            api(client, "play", {"cards": [0, 1, 2, 3, 4, 5]}),
            "BAD_REQUEST",
            "You can only play 5 cards",
        )

    def test_play_out_of_range_cards(self, client: httpx.Client) -> None:
        """Test play endpoint from BLIND_SELECT state."""
        gamestate = load_fixture(client, "play", "state-SELECTING_HAND")
        assert gamestate["state"] == "SELECTING_HAND"
        assert_error_response(
            api(client, "play", {"cards": [999]}),
            "BAD_REQUEST",
            "Invalid card index: 999",
        )

    def test_play_valid_cards_and_round_active(self, client: httpx.Client) -> None:
        """Test play endpoint from BLIND_SELECT state."""
        gamestate = load_fixture(client, "play", "state-SELECTING_HAND")
        assert gamestate["state"] == "SELECTING_HAND"
        response = api(client, "play", {"cards": [0, 3, 4, 5, 6]})
        gamestate = assert_gamestate_response(response, state="SELECTING_HAND")
        assert gamestate["hands"]["Flush"]["played_this_round"] == 1
        assert gamestate["round"]["chips"] == 260

    def test_play_valid_cards_and_round_won(self, client: httpx.Client) -> None:
        """Test play endpoint from BLIND_SELECT state."""
        gamestate = load_fixture(
            client, "play", "state-SELECTING_HAND--round.chips-200"
        )
        assert gamestate["state"] == "SELECTING_HAND"
        assert gamestate["round"]["chips"] == 200
        response = api(client, "play", {"cards": [0, 3, 4, 5, 6]})
        assert_gamestate_response(response, state="ROUND_EVAL")

    def test_play_valid_cards_and_game_won(self, client: httpx.Client) -> None:
        """Test play endpoint from BLIND_SELECT state."""
        gamestate = load_fixture(
            client,
            "play",
            "state-SELECTING_HAND--ante_num-8--blinds.boss.status-CURRENT--round.chips-1000000",
        )
        assert gamestate["state"] == "SELECTING_HAND"
        assert gamestate["ante_num"] == 8
        assert gamestate["blinds"]["boss"]["status"] == "CURRENT"
        assert gamestate["round"]["chips"] == 1000000
        response = api(client, "play", {"cards": [0, 3, 4, 5, 6]})
        assert_gamestate_response(response, won=True)

    def test_play_valid_cards_and_game_over(self, client: httpx.Client) -> None:
        """Test play endpoint from BLIND_SELECT state."""
        gamestate = load_fixture(
            client, "play", "state-SELECTING_HAND--round.hands_left-1"
        )
        assert gamestate["state"] == "SELECTING_HAND"
        assert gamestate["round"]["hands_left"] == 1
        response = api(client, "play", {"cards": [0]}, timeout=5)
        assert_gamestate_response(response, state="GAME_OVER")


class TestPlayEndpointValidation:
    """Test play endpoint parameter validation."""

    def test_missing_cards_parameter(self, client: httpx.Client):
        """Test that play fails when cards parameter is missing."""
        gamestate = load_fixture(client, "play", "state-SELECTING_HAND")
        assert gamestate["state"] == "SELECTING_HAND"
        assert_error_response(
            api(client, "play", {}),
            "BAD_REQUEST",
            "Missing required field 'cards'",
        )

    def test_invalid_cards_type(self, client: httpx.Client):
        """Test that play fails when cards parameter is not an array."""
        gamestate = load_fixture(client, "play", "state-SELECTING_HAND")
        assert gamestate["state"] == "SELECTING_HAND"
        assert_error_response(
            api(client, "play", {"cards": "INVALID_CARDS"}),
            "BAD_REQUEST",
            "Field 'cards' must be an array",
        )


class TestPlayEndpointStateRequirements:
    """Test play endpoint state requirements."""

    def test_play_from_BLIND_SELECT(self, client: httpx.Client):
        """Test that play fails when not in SELECTING_HAND state."""
        gamestate = load_fixture(client, "play", "state-BLIND_SELECT")
        assert gamestate["state"] == "BLIND_SELECT"
        assert_error_response(
            api(client, "play", {"cards": [0]}),
            "INVALID_STATE",
            "Method 'play' requires one of these states: SELECTING_HAND",
        )

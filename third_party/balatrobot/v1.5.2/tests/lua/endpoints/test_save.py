"""Tests for src/lua/endpoints/save.lua"""

from pathlib import Path

import httpx

from tests.lua.conftest import (
    api,
    assert_error_response,
    assert_path_response,
    load_fixture,
)


class TestSaveEndpoint:
    """Test basic save endpoint functionality."""

    def test_save_from_BLIND_SELECT(self, client: httpx.Client, tmp_path: Path) -> None:
        """Test that save succeeds from BLIND_SELECT state."""
        gamestate = load_fixture(client, "save", "state-BLIND_SELECT")
        assert gamestate["state"] == "BLIND_SELECT"
        temp_file = tmp_path / "save"
        response = api(client, "save", {"path": str(temp_file)})
        assert_path_response(response)
        assert response["result"]["path"] == str(temp_file)
        assert temp_file.exists()
        assert temp_file.stat().st_size > 0

    def test_save_creates_valid_file(
        self, client: httpx.Client, tmp_path: Path
    ) -> None:
        """Test that saved file can be loaded back successfully."""
        gamestate = load_fixture(client, "save", "state-BLIND_SELECT")
        assert gamestate["state"] == "BLIND_SELECT"
        temp_file = tmp_path / "save"
        save_response = api(client, "save", {"path": str(temp_file)})
        assert_path_response(save_response)
        load_response = api(client, "load", {"path": str(temp_file)})
        assert_path_response(load_response)


class TestSaveValidation:
    """Test save endpoint parameter validation."""

    def test_missing_path_parameter(self, client: httpx.Client) -> None:
        """Test that save fails when path parameter is missing."""
        response = api(client, "save", {})
        assert_error_response(
            response,
            "BAD_REQUEST",
            "Missing required field 'path'",
        )

    def test_invalid_path_type(self, client: httpx.Client) -> None:
        """Test that save fails when path is not a string."""
        response = api(client, "save", {"path": 123})
        assert_error_response(
            response,
            "BAD_REQUEST",
            "Field 'path' must be of type string",
        )


class TestSaveStateRequirements:
    """Test save endpoint state requirements."""

    def test_save_from_MENU(self, client: httpx.Client, tmp_path: Path) -> None:
        """Test that save fails when not in an active run."""
        api(client, "menu", {})
        temp_file = tmp_path / "save"
        response = api(client, "save", {"path": str(temp_file)})
        assert_error_response(
            response,
            "INVALID_STATE",
            "Method 'save' requires one of these states: SELECTING_HAND, HAND_PLAYED, DRAW_TO_HAND, GAME_OVER, SHOP, PLAY_TAROT, BLIND_SELECT, ROUND_EVAL, TAROT_PACK, PLANET_PACK, SPECTRAL_PACK, STANDARD_PACK, BUFFOON_PACK, NEW_ROUND",
        )
        assert not temp_file.exists()

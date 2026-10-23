"""Tests for src/lua/endpoints/load.lua"""

from pathlib import Path

import httpx

from tests.lua.conftest import (
    api,
    assert_error_response,
    assert_path_response,
    get_fixture_path,
    load_fixture,
)


class TestLoadEndpoint:
    """Test basic load endpoint functionality."""

    def test_load_from_fixture(self, client: httpx.Client) -> None:
        """Test that load succeeds with a valid fixture file."""
        gamestate = load_fixture(client, "load", "state-BLIND_SELECT")
        assert gamestate["state"] == "BLIND_SELECT"
        fixture_path = get_fixture_path("load", "state-BLIND_SELECT")
        response = api(client, "load", {"path": str(fixture_path)})
        assert_path_response(response)
        assert response["result"]["path"] == str(fixture_path)

    def test_load_save_roundtrip(self, client: httpx.Client, tmp_path: Path) -> None:
        """Test that a loaded fixture can be saved and loaded again."""
        # Load fixture
        gamestate = load_fixture(client, "load", "state-BLIND_SELECT")
        assert gamestate["state"] == "BLIND_SELECT"
        fixture_path = get_fixture_path("load", "state-BLIND_SELECT")
        load_response = api(client, "load", {"path": str(fixture_path)})
        assert_path_response(load_response)

        # Save to temp path
        temp_file = tmp_path / "save"
        save_response = api(client, "save", {"path": str(temp_file)})
        assert_path_response(save_response)
        assert temp_file.exists()

        # Load the saved file back
        load_again_response = api(client, "load", {"path": str(temp_file)})
        assert_path_response(load_again_response)


class TestLoadValidation:
    """Test load endpoint parameter validation."""

    def test_missing_path_parameter(self, client: httpx.Client) -> None:
        """Test that load fails when path parameter is missing."""
        assert_error_response(
            api(client, "load", {}),
            "BAD_REQUEST",
            "Missing required field 'path'",
        )

    def test_invalid_path_type(self, client: httpx.Client) -> None:
        """Test that load fails when path is not a string."""
        assert_error_response(
            api(client, "load", {"path": 123}),
            "BAD_REQUEST",
            "Field 'path' must be of type string",
        )

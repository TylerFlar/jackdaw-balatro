"""
Integration tests for BB_DISPATCHER request routing and validation (JSON-RPC 2.0).

Test classes are organized by validation tier:
- TestDispatcherProtocolValidation: TIER 1 - Protocol structure validation
- TestDispatcherSchemaValidation: TIER 2 - Schema/argument validation
- TestDispatcherStateValidation: TIER 3 - Game state validation
- TestDispatcherExecution: TIER 4 - Endpoint execution and error handling
- TestDispatcherEndpointRegistry: Endpoint registration and discovery
"""

import httpx

from tests.lua.conftest import api

# Request ID counter for malformed request tests only
_test_request_id = 0


class TestDispatcherProtocolValidation:
    """Tests for TIER 1: Protocol Validation.

    Tests verify that dispatcher correctly validates:
    - Request has 'method' field (string)
    - Request has 'params' field (optional, defaults to {})
    - Endpoint exists in registry
    """

    def test_missing_name_field(self, client: httpx.Client) -> None:
        """Test that requests without 'method' field are rejected."""
        global _test_request_id
        _test_request_id += 1
        # Send JSON-RPC request missing 'method' field
        response = client.post(
            "/",
            json={"jsonrpc": "2.0", "params": {}, "id": _test_request_id},
        )
        parsed = response.json()

        assert "error" in parsed
        assert "message" in parsed["error"]
        assert "data" in parsed["error"]
        assert "name" in parsed["error"]["data"]
        assert parsed["error"]["data"]["name"] == "BAD_REQUEST"
        assert "method" in parsed["error"]["message"].lower()

    def test_invalid_name_type(self, client: httpx.Client) -> None:
        """Test that 'method' field must be a string."""
        global _test_request_id
        _test_request_id += 1
        # Send JSON-RPC request with 'method' as integer
        response = client.post(
            "/",
            json={
                "jsonrpc": "2.0",
                "method": 123,
                "params": {},
                "id": _test_request_id,
            },
        )
        parsed = response.json()

        assert "error" in parsed
        assert parsed["error"]["data"]["name"] == "BAD_REQUEST"

    def test_missing_arguments_field(self, client: httpx.Client) -> None:
        """Test that requests without 'params' field succeed (params is optional in JSON-RPC 2.0)."""
        global _test_request_id
        _test_request_id += 1
        # Send JSON-RPC request without 'params' field
        response = client.post(
            "/",
            json={"jsonrpc": "2.0", "method": "health", "id": _test_request_id},
        )
        parsed = response.json()

        # In JSON-RPC 2.0, params is optional - should succeed for health
        assert "result" in parsed
        assert "status" in parsed["result"]
        assert parsed["result"]["status"] == "ok"

    def test_unknown_endpoint(self, client: httpx.Client) -> None:
        """Test that unknown endpoints are rejected."""
        response = api(client, "nonexistent_endpoint", {})

        assert "error" in response
        assert response["error"]["data"]["name"] == "BAD_REQUEST"
        assert "nonexistent_endpoint" in response["error"]["message"]

    def test_valid_health_endpoint_request(self, client: httpx.Client) -> None:
        """Test that valid requests to health endpoint succeed."""
        response = api(client, "health", {})

        # Health endpoint should return success
        assert "result" in response
        assert "status" in response["result"]
        assert response["result"]["status"] == "ok"


class TestDispatcherSchemaValidation:
    """Tests for TIER 2: Schema Validation.

    Tests verify that dispatcher correctly validates arguments against
    endpoint schemas using the Validator module.
    """

    def test_missing_required_field(self, client: httpx.Client) -> None:
        """Test that missing required fields are rejected."""
        # test_endpoint requires 'required_string' and 'required_integer'
        response = api(
            client,
            "test_endpoint",
            {
                "required_integer": 50,
                "required_enum": "option_a",
                # Missing 'required_string'
            },
        )

        assert "error" in response
        assert response["error"]["data"]["name"] == "BAD_REQUEST"
        assert "required_string" in response["error"]["message"].lower()

    def test_invalid_type_string_instead_of_integer(self, client: httpx.Client) -> None:
        """Test that type validation rejects wrong types."""
        response = api(
            client,
            "test_endpoint",
            {
                "required_string": "valid_string",
                "required_integer": "not_an_integer",  # Should be integer
                "required_enum": "option_a",
            },
        )

        assert "error" in response
        assert response["error"]["data"]["name"] == "BAD_REQUEST"
        assert "required_integer" in response["error"]["message"].lower()

    def test_array_item_type_validation(self, client: httpx.Client) -> None:
        """Test that array items are validated for correct type."""
        response = api(
            client,
            "test_endpoint",
            {
                "required_string": "test",
                "required_integer": 50,
                "optional_array_integers": [
                    1,
                    2,
                    "not_integer",
                    4,
                ],  # Should be integers
            },
        )

        assert "error" in response
        assert response["error"]["data"]["name"] == "BAD_REQUEST"

    def test_valid_request_with_all_fields(self, client: httpx.Client) -> None:
        """Test that valid requests with multiple fields pass validation."""
        response = api(
            client,
            "test_endpoint",
            {
                "required_string": "test",
                "required_integer": 50,
                "optional_string": "optional",
                "optional_integer": 42,
                "optional_array_integers": [1, 2, 3],
            },
        )

        # Should succeed and echo back
        assert "result" in response
        assert "success" in response["result"]
        assert response["result"]["success"] is True
        assert "received_args" in response["result"]

    def test_valid_request_with_only_required_fields(
        self, client: httpx.Client
    ) -> None:
        """Test that valid requests with only required fields pass validation."""
        response = api(
            client,
            "test_endpoint",
            {
                "required_string": "test",
                "required_integer": 1,
                "required_enum": "option_c",
            },
        )

        assert "result" in response
        assert "success" in response["result"]
        assert response["result"]["success"] is True


class TestDispatcherStateValidation:
    """Tests for TIER 3: Game State Validation.

    Tests verify that dispatcher enforces endpoint state requirements.
    Note: These tests may pass or fail depending on current game state.
    """

    def test_state_validation_enforcement(self, client: httpx.Client) -> None:
        """Test that endpoints with requires_state are validated."""
        # test_state_endpoint requires SPLASH or MENU state
        response = api(client, "test_state_endpoint", {})

        # Response depends on current game state
        # Either succeeds if in correct state, or fails with INVALID_STATE
        if "error" in response:
            assert response["error"]["data"]["name"] == "INVALID_STATE"
            assert "requires" in response["error"]["message"].lower()
        else:
            assert "result" in response
            assert "success" in response["result"]
            assert response["result"]["state_validated"] is True


class TestDispatcherExecution:
    """Tests for TIER 4: Endpoint Execution and Error Handling.

    Tests verify that dispatcher correctly executes endpoints and
    handles runtime errors with appropriate error codes.
    """

    def test_successful_endpoint_execution(self, client: httpx.Client) -> None:
        """Test that endpoints execute successfully with valid input."""
        response = api(
            client,
            "test_endpoint",
            {
                "required_string": "test",
                "required_integer": 42,
                "required_enum": "option_a",
            },
        )

        assert "result" in response
        assert "success" in response["result"]
        assert response["result"]["success"] is True
        assert "received_args" in response["result"]
        assert response["result"]["received_args"]["required_integer"] == 42

    def test_execution_error_handling(self, client: httpx.Client) -> None:
        """Test that runtime errors are caught and returned properly."""
        response = api(client, "test_error_endpoint", {"error_type": "throw_error"})

        assert "error" in response
        assert response["error"]["data"]["name"] == "INTERNAL_ERROR"
        assert "Intentional test error" in response["error"]["message"]

    def test_execution_error_no_categorization(self, client: httpx.Client) -> None:
        """Test that all execution errors use INTERNAL_ERROR."""
        response = api(client, "test_error_endpoint", {"error_type": "throw_error"})

        # Should always be INTERNAL_ERROR (no categorization)
        assert response["error"]["data"]["name"] == "INTERNAL_ERROR"

    def test_execution_success_when_no_error(self, client: httpx.Client) -> None:
        """Test that endpoints can execute successfully."""
        response = api(client, "test_error_endpoint", {"error_type": "success"})

        assert "result" in response
        assert "success" in response["result"]
        assert response["result"]["success"] is True


class TestDispatcherEndpointRegistry:
    """Tests for endpoint registration and discovery."""

    def test_health_endpoint_is_registered(self, client: httpx.Client) -> None:
        """Test that the health endpoint is properly registered."""
        response = api(client, "health", {})

        assert "result" in response
        assert "status" in response["result"]
        assert response["result"]["status"] == "ok"

    def test_multiple_sequential_requests_to_same_endpoint(
        self, client: httpx.Client
    ) -> None:
        """Test that multiple requests to the same endpoint work."""
        for i in range(3):
            response = api(client, "health", {})

            assert "result" in response
            assert "status" in response["result"]
            assert response["result"]["status"] == "ok"

    def test_requests_to_different_endpoints(self, client: httpx.Client) -> None:
        """Test that requests can be routed to different endpoints."""
        # Request to health endpoint
        response1 = api(client, "health", {})
        assert "result" in response1
        assert "status" in response1["result"]

        # Request to test_endpoint
        response2 = api(
            client,
            "test_endpoint",
            {
                "required_string": "test",
                "required_integer": 25,
                "required_enum": "option_a",
            },
        )
        assert "result" in response2
        assert "success" in response2["result"]

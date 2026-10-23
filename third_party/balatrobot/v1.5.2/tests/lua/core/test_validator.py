# tests/lua/core/test_validator.py
# Comprehensive tests for src/lua/core/validator.lua
#
# Tests validation scenarios through the dispatcher using the test_validation endpoint:
# - Type validation (string, integer, boolean, array, table)
# - Required field validation
# - Array item type validation (integer arrays only)
# - Error codes and messages

import httpx

from tests.lua.conftest import (
    api,
    assert_error_response,
    assert_test_response,
)

# ============================================================================
# Test: Type Validation
# ============================================================================


class TestTypeValidation:
    """Test type validation for all supported types."""

    def test_valid_string_type(self, client: httpx.Client) -> None:
        """Test that valid string type passes validation."""
        response = api(
            client,
            "test_validation",
            {
                "required_field": "test",
                "string_field": "hello",
            },
        )
        assert_test_response(response)

    def test_invalid_string_type(self, client: httpx.Client) -> None:
        """Test that invalid string type fails validation."""
        response = api(
            client,
            "test_validation",
            {
                "required_field": "test",
                "string_field": 123,  # Should be string
            },
        )
        assert_error_response(
            response,
            "BAD_REQUEST",
            "string_field",
        )

    def test_valid_integer_type(self, client: httpx.Client) -> None:
        """Test that valid integer type passes validation."""
        response = api(
            client,
            "test_validation",
            {
                "required_field": "test",
                "integer_field": 42,
            },
        )
        assert_test_response(response)

    def test_invalid_integer_type_float(self, client: httpx.Client) -> None:
        """Test that float fails integer validation."""
        response = api(
            client,
            "test_validation",
            {
                "required_field": "test",
                "integer_field": 42.5,  # Should be integer
            },
        )
        assert_error_response(
            response,
            "BAD_REQUEST",
            "integer_field",
        )

    def test_invalid_integer_type_string(self, client: httpx.Client) -> None:
        """Test that string fails integer validation."""
        response = api(
            client,
            "test_validation",
            {
                "required_field": "test",
                "integer_field": "42",
            },
        )
        assert_error_response(
            response,
            "BAD_REQUEST",
            "integer_field",
        )

    def test_valid_array_type(self, client: httpx.Client) -> None:
        """Test that valid array type passes validation."""
        response = api(
            client,
            "test_validation",
            {
                "required_field": "test",
                "array_field": [1, 2, 3],
            },
        )
        assert_test_response(response)

    def test_invalid_array_type_not_sequential(self, client: httpx.Client) -> None:
        """Test that non-sequential table fails array validation."""
        response = api(
            client,
            "test_validation",
            {
                "required_field": "test",
                "array_field": {"key": "value"},  # Not an array
            },
        )
        assert_error_response(
            response,
            "BAD_REQUEST",
            "array_field",
        )

    def test_invalid_array_type_string(self, client: httpx.Client) -> None:
        """Test that string fails array validation."""
        response = api(
            client,
            "test_validation",
            {
                "required_field": "test",
                "array_field": "not an array",
            },
        )
        assert_error_response(
            response,
            "BAD_REQUEST",
            "array_field",
        )

    def test_valid_boolean_type_true(self, client: httpx.Client) -> None:
        """Test that boolean true passes validation."""
        response = api(
            client,
            "test_validation",
            {
                "required_field": "test",
                "boolean_field": True,
            },
        )
        assert_test_response(response)

    def test_valid_boolean_type_false(self, client: httpx.Client) -> None:
        """Test that boolean false passes validation."""
        response = api(
            client,
            "test_validation",
            {
                "required_field": "test",
                "boolean_field": False,
            },
        )
        assert_test_response(response)

    def test_invalid_boolean_type_string(self, client: httpx.Client) -> None:
        """Test that string fails boolean validation."""
        response = api(
            client,
            "test_validation",
            {
                "required_field": "test",
                "boolean_field": "true",  # Should be boolean, not string
            },
        )
        assert_error_response(
            response,
            "BAD_REQUEST",
            "boolean_field",
        )

    def test_invalid_boolean_type_number(self, client: httpx.Client) -> None:
        """Test that number fails boolean validation."""
        response = api(
            client,
            "test_validation",
            {
                "required_field": "test",
                "boolean_field": 1,  # Should be boolean, not number
            },
        )
        assert_error_response(
            response,
            "BAD_REQUEST",
            "boolean_field",
        )

    def test_valid_table_type(self, client: httpx.Client) -> None:
        """Test that valid table (non-array) passes validation."""
        response = api(
            client,
            "test_validation",
            {
                "required_field": "test",
                "table_field": {"key": "value", "nested": {"data": 123}},
            },
        )
        assert_test_response(response)

    def test_valid_table_type_empty(self, client: httpx.Client) -> None:
        """Test that empty table passes validation."""
        response = api(
            client,
            "test_validation",
            {
                "required_field": "test",
                "table_field": {},
            },
        )
        assert_test_response(response)

    def test_invalid_table_type_array(self, client: httpx.Client) -> None:
        """Test that array fails table validation (arrays should use 'array' type)."""
        response = api(
            client,
            "test_validation",
            {
                "required_field": "test",
                "table_field": [1, 2, 3],  # Array not allowed for 'table' type
            },
        )
        assert_error_response(
            response,
            "BAD_REQUEST",
            "table_field",
        )

    def test_invalid_table_type_string(self, client: httpx.Client) -> None:
        """Test that string fails table validation."""
        response = api(
            client,
            "test_validation",
            {
                "required_field": "test",
                "table_field": "not a table",
            },
        )
        assert_error_response(
            response,
            "BAD_REQUEST",
            "table_field",
        )


# ============================================================================
# Test: Required Field Validation
# ============================================================================


class TestRequiredFields:
    """Test required field validation."""

    def test_required_field_present(self, client: httpx.Client) -> None:
        """Test that request with required field passes."""
        response = api(
            client,
            "test_validation",
            {"required_field": "present"},
        )
        assert_test_response(response)

    def test_required_field_missing(self, client: httpx.Client) -> None:
        """Test that request without required field fails."""
        response = api(
            client,
            "test_validation",
            {},  # Missing required_field
        )
        assert_error_response(
            response,
            "BAD_REQUEST",
            "required_field",
        )

    def test_optional_field_missing(self, client: httpx.Client) -> None:
        """Test that missing optional fields are allowed."""
        response = api(
            client,
            "test_validation",
            {
                "required_field": "present",
                # All other fields are optional
            },
        )
        assert_test_response(response)


# ============================================================================
# Test: Array Item Type Validation
# ============================================================================


class TestArrayItemTypes:
    """Test array item type validation."""

    def test_array_of_integers_valid(self, client: httpx.Client) -> None:
        """Test that array of integers passes."""
        response = api(
            client,
            "test_validation",
            {
                "required_field": "test",
                "array_of_integers": [1, 2, 3],
            },
        )
        assert_test_response(response)

    def test_array_of_integers_invalid_float(self, client: httpx.Client) -> None:
        """Test that array with float items fails integer validation."""
        response = api(
            client,
            "test_validation",
            {
                "required_field": "test",
                "array_of_integers": [1, 2.5, 3],
            },
        )
        assert_error_response(
            response,
            "BAD_REQUEST",
            "array_of_integers",
        )

    def test_array_of_integers_invalid_string(self, client: httpx.Client) -> None:
        """Test that array with string items fails integer validation."""
        response = api(
            client,
            "test_validation",
            {
                "required_field": "test",
                "array_of_integers": [1, "2", 3],
            },
        )
        assert_error_response(
            response,
            "BAD_REQUEST",
            "array_of_integers",
        )


# ============================================================================
# Test: Fail-Fast Behavior
# ============================================================================


class TestFailFastBehavior:
    """Test that validator fails fast on first error."""

    def test_multiple_errors_returns_first(self, client: httpx.Client) -> None:
        """Test that only the first error is returned when multiple errors exist."""
        response = api(
            client,
            "test_validation",
            {
                # Missing required_field (one error)
                "string_field": 123,  # Type error (another error)
                "integer_field": "not an integer",  # Type error (another error)
            },
        )
        # Should get ONE error (fail-fast), not all errors
        # The specific error depends on Lua table iteration order
        assert_error_response(response)
        # Verify it's one of the expected error codes
        assert response["error"]["data"]["name"] in [
            "BAD_REQUEST",
            "BAD_REQUEST",
        ]


# ============================================================================
# Test: Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_arguments_with_only_required_field(
        self, client: httpx.Client
    ) -> None:
        """Test that arguments with only required field passes."""
        response = api(
            client,
            "test_validation",
            {"required_field": "only this"},
        )
        assert_test_response(response)

    def test_all_fields_provided(self, client: httpx.Client) -> None:
        """Test request with multiple valid fields."""
        response = api(
            client,
            "test_validation",
            {
                "required_field": "test",
                "string_field": "hello",
                "integer_field": 42,
                "boolean_field": True,
                "array_field": [1, 2, 3],
                "table_field": {"key": "value"},
                "array_of_integers": [4, 5, 6],
            },
        )
        assert_test_response(response)

    def test_empty_array_when_allowed(self, client: httpx.Client) -> None:
        """Test that empty array passes when no min constraint."""
        response = api(
            client,
            "test_validation",
            {
                "required_field": "test",
                "array_field": [],
            },
        )
        assert_test_response(response)

    def test_empty_string_when_allowed(self, client: httpx.Client) -> None:
        """Test that empty string passes when no min constraint."""
        response = api(
            client,
            "test_validation",
            {
                "required_field": "",  # Empty but present
            },
        )
        assert_test_response(response)

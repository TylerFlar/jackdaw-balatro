"""Tests for the JSON-RPC 2.0 HTTP server."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import HTTPServer
from typing import Any

import pytest

from jackdaw.bridge.backend import RPCError, SimBackend
from jackdaw.cli.serve import RPCHandler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _start_server(backend: Any) -> tuple[HTTPServer, str]:
    """Start a server on a random port, return (server, base_url)."""
    handler = RPCHandler
    handler.backend = backend
    server = HTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://127.0.0.1:{port}"


def _rpc(url: str, method: str, params: dict | None = None, req_id: int = 1) -> dict:
    """Send a JSON-RPC request, return the parsed response."""
    payload: dict[str, Any] = {"jsonrpc": "2.0", "method": method, "id": req_id}
    if params is not None:
        payload["params"] = params
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


def _raw_post(url: str, body: bytes) -> dict:
    """POST raw bytes, return parsed JSON response."""
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


@pytest.fixture()
def server():
    backend = SimBackend()
    srv, url = _start_server(backend)
    yield url
    srv.shutdown()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_health(self, server: str) -> None:
        resp = _rpc(server, "health")
        assert resp["jsonrpc"] == "2.0"
        assert resp["result"] == {"status": "ok"}
        assert resp["id"] == 1


class TestGameplaySequence:
    def test_start_gamestate_play(self, server: str) -> None:
        # Start a run
        resp = _rpc(server, "start", {"deck": "RED", "stake": "WHITE", "seed": "TEST1"})
        assert "result" in resp
        result = resp["result"]
        assert "hand" in result or "state" in result  # has game data

        # Get gamestate
        resp = _rpc(server, "gamestate")
        assert "result" in resp

        # Play first 5 cards
        resp = _rpc(server, "select")  # select blind first
        assert "result" in resp

        resp = _rpc(server, "play", {"cards": [0, 1, 2, 3, 4]})
        assert "result" in resp
        assert "error" not in resp


class TestParseErrors:
    def test_invalid_json(self, server: str) -> None:
        resp = _raw_post(server, b"{not valid json")
        assert resp["error"]["code"] == -32700
        assert resp["id"] is None

    def test_missing_method(self, server: str) -> None:
        resp = _raw_post(server, json.dumps({"jsonrpc": "2.0", "id": 1}).encode())
        assert resp["error"]["code"] == -32600
        assert resp["id"] == 1

    def test_non_string_method(self, server: str) -> None:
        resp = _raw_post(
            server, json.dumps({"jsonrpc": "2.0", "method": 42, "id": 1}).encode()
        )
        assert resp["error"]["code"] == -32600

    def test_non_object_body(self, server: str) -> None:
        resp = _raw_post(server, json.dumps([1, 2, 3]).encode())
        assert resp["error"]["code"] == -32600


class TestRPCErrorPropagation:
    def test_action_without_start(self, server: str) -> None:
        """Action on no active run should give INVALID_STATE (-32002)."""
        resp = _rpc(server, "gamestate")
        assert resp["error"]["code"] == -32002
        assert resp["id"] == 1

    def test_unknown_method(self, server: str) -> None:
        resp = _rpc(server, "nonexistent_method")
        assert "error" in resp


class TestMethodNotAllowed:
    def test_get_returns_405(self, server: str) -> None:
        req = urllib.request.Request(server, method="GET")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 405

    def test_put_returns_405(self, server: str) -> None:
        req = urllib.request.Request(server, data=b"{}", method="PUT")
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 405


class TestResponseFormat:
    def test_id_preserved(self, server: str) -> None:
        resp = _rpc(server, "health", req_id=42)
        assert resp["id"] == 42

    def test_null_id_on_error(self, server: str) -> None:
        resp = _raw_post(server, b"garbage")
        assert resp["id"] is None

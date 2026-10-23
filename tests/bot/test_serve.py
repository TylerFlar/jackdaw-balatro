"""``jackdaw serve``: BalatroBot's HTTP JSON-RPC 2.0 server (server.lua) on the simulator."""

from __future__ import annotations

import subprocess
import sys

import httpx
import pytest

from tests.bot.server import ROOT, jackdaw_serve


@pytest.fixture(scope="module")
def url() -> str:
    with jackdaw_serve() as port:
        yield f"http://127.0.0.1:{port}"


def _rpc(url: str, method: str, params: dict | None = None, id: object = 1) -> dict:
    body = {"jsonrpc": "2.0", "method": method, "id": id}
    if params is not None:
        body["params"] = params
    return httpx.post(url, json=body, timeout=10).json()


def test_a_run_over_http(url: str) -> None:
    assert _rpc(url, "health") == {"jsonrpc": "2.0", "result": {"status": "ok"}, "id": 1}
    _rpc(url, "menu")
    out = _rpc(url, "start", {"deck": "RED", "stake": "WHITE", "seed": "SERVE"}, id="abc")
    assert out["id"] == "abc" and out["result"]["state"] == "BLIND_SELECT"
    assert _rpc(url, "select")["result"]["state"] == "SELECTING_HAND"
    err = _rpc(url, "select", id=7)
    assert err == {
        "jsonrpc": "2.0",
        "error": {
            "code": -32002,
            "message": "Method 'select' requires one of these states: BLIND_SELECT",
            "data": {"name": "INVALID_STATE"},
        },
        "id": 7,
    }
    assert _rpc(url, "menu")["result"]["state"] == "MENU"


def test_envelope_errors(url: str) -> None:
    bad = httpx.post(url, content=b"{not json", timeout=10).json()
    assert bad["error"]["message"] == "Invalid JSON in request body" and "id" not in bad
    out = httpx.post(url, json={"jsonrpc": "1.0", "method": "health", "id": 3}, timeout=10).json()
    assert out["error"]["message"] == "Invalid JSON-RPC version: expected '2.0'"
    assert out["id"] == 3
    out = httpx.post(url, json={"jsonrpc": "2.0", "method": "health"}, timeout=10).json()
    assert out["error"]["message"] == "Invalid Request: 'id' field is required"
    out = _rpc(url, "health", id=1.5)
    assert out["error"]["message"] == "Invalid Request: 'id' must be an integer, not a float"
    out = _rpc(url, "nope")
    assert out["error"] == {
        "code": -32001,
        "message": "Unknown method: nope",
        "data": {"name": "BAD_REQUEST"},
    }


def test_http_errors(url: str) -> None:
    got = httpx.get(url, timeout=10)
    assert got.status_code == 405
    assert got.json()["error"]["message"] == "Method not allowed. Use POST for JSON-RPC requests"
    posted = httpx.post(url + "/other", json={"jsonrpc": "2.0", "method": "health", "id": 1})
    assert posted.status_code == 404
    assert posted.json()["error"]["message"] == "Not found. Use POST to '/' for JSON-RPC requests"
    big = httpx.post(url, content=b"x" * 70000, timeout=10)
    assert big.status_code == 400
    assert big.json()["error"]["message"] == "Request body too large"


def test_rpc_discover(url: str) -> None:
    import json

    from tests.bot.schema import OPENRPC

    out = _rpc(url, "rpc.discover")
    assert out["result"] == json.loads(OPENRPC.read_text(encoding="utf-8"))


def test_screenshot_is_an_internal_error(url: str) -> None:
    out = _rpc(url, "screenshot", {"path": "shot.png"})
    assert out["error"]["code"] == -32000


def test_off_loopback_needs_an_fs_root() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "jackdaw.cli.main", "serve", "--host", "0.0.0.0", "--port", "0"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 1
    assert "fs-root" in proc.stderr or "fs_root" in proc.stderr

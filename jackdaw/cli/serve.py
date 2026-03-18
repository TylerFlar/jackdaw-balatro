"""JSON-RPC 2.0 HTTP server for the jackdaw simulator.

Uses only ``http.server`` from the standard library — zero runtime deps.
Any balatrobot-compatible client can connect to this server and play
against the headless engine with zero code changes.

Usage::

    from jackdaw.bridge.backend import SimBackend
    from jackdaw.cli.serve import run_server

    run_server(SimBackend(), "127.0.0.1", 8080)
"""

from __future__ import annotations

import json
import sys
from functools import partial
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from jackdaw.bridge.backend import Backend, RPCError

# JSON-RPC 2.0 standard error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
INTERNAL_ERROR = -32603


def _make_response(result: Any, req_id: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "result": result, "id": req_id}


def _make_error(code: int, message: str, req_id: Any, data: dict | None = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data:
        err["data"] = data
    return {"jsonrpc": "2.0", "error": err, "id": req_id}


class RPCHandler(BaseHTTPRequestHandler):
    """HTTP request handler that speaks JSON-RPC 2.0."""

    backend: Backend  # set via partial / class attribute

    def do_POST(self) -> None:
        req_id: Any = None
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)

            try:
                body = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError):
                self._send_json(_make_error(PARSE_ERROR, "Parse error", None))
                return

            if not isinstance(body, dict):
                self._send_json(_make_error(INVALID_REQUEST, "Invalid Request", None))
                return

            req_id = body.get("id")
            method = body.get("method")
            if not isinstance(method, str):
                self._send_json(
                    _make_error(INVALID_REQUEST, "Invalid Request: missing method", req_id)
                )
                return

            params = body.get("params")
            if params is not None and not isinstance(params, dict):
                self._send_json(
                    _make_error(INVALID_REQUEST, "Invalid Request: params must be object", req_id)
                )
                return

            result = self.backend.handle(method, params)
            self._send_json(_make_response(result, req_id))
            self.log_message("%s → ok", method)

        except RPCError as exc:
            self._send_json(_make_error(exc.code, exc.message, req_id, exc.data or None))
            self.log_message("%s → error %d", body.get("method", "?"), exc.code)

        except Exception as exc:
            self._send_json(_make_error(INTERNAL_ERROR, str(exc), req_id))
            self.log_message("internal error: %s", exc)

    def _send_json(self, obj: dict[str, Any]) -> None:
        payload = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        self._method_not_allowed()

    def do_PUT(self) -> None:
        self._method_not_allowed()

    def do_DELETE(self) -> None:
        self._method_not_allowed()

    def _method_not_allowed(self) -> None:
        self.send_response(405)
        self.send_header("Allow", "POST")
        self.end_headers()

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: N802
        # Single-line compact log to stderr
        sys.stderr.write(f"[jackdaw] {fmt % args}\n")


def run_server(backend: Backend, host: str = "127.0.0.1", port: int = 8080) -> None:
    """Start the JSON-RPC HTTP server (blocking)."""
    handler = partial(RPCHandler)
    handler.backend = backend  # type: ignore[attr-defined]

    server = HTTPServer((host, port), handler)
    print(f"Jackdaw JSON-RPC server listening on http://{host}:{port}")

    backend_label = type(backend).__name__
    if backend_label == "LiveBackend":
        url = getattr(backend, "_url", "?")
        print(f"Backend: live (proxying to {url})")
    else:
        print("Backend: sim")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()

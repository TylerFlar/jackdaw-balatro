"""Swappable backend interface for the JSON-RPC server.

Two implementations:

- **SimBackend** — runs the jackdaw engine in-process. Fast, headless,
  deterministic, zero runtime deps.
- **LiveBackend** — proxies all requests to a real balatrobot instance
  over HTTP. Requires a running Balatro + balatrobot mod.
"""

from __future__ import annotations

from typing import Any, Protocol

from jackdaw.engine.actions import GamePhase

# ---------------------------------------------------------------------------
# Balatrobot enum → engine value maps
# ---------------------------------------------------------------------------

DECK_FROM_BOT: dict[str, str] = {
    "RED": "b_red",
    "BLUE": "b_blue",
    "YELLOW": "b_yellow",
    "GREEN": "b_green",
    "BLACK": "b_black",
    "MAGIC": "b_magic",
    "NEBULA": "b_nebula",
    "GHOST": "b_ghost",
    "ABANDONED": "b_abandoned",
    "CHECKERED": "b_checkered",
    "ZODIAC": "b_zodiac",
    "PAINTED": "b_painted",
    "ANAGLYPH": "b_anaglyph",
    "PLASMA": "b_plasma",
    "ERRATIC": "b_erratic",
}

STAKE_FROM_BOT: dict[str, int] = {
    "WHITE": 1,
    "RED": 2,
    "GREEN": 3,
    "BLACK": 4,
    "BLUE": 5,
    "PURPLE": 6,
    "ORANGE": 7,
    "GOLD": 8,
}


# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class RPCError(Exception):
    """JSON-RPC error with code, message, and optional data."""

    def __init__(
        self,
        code: int,
        message: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.data = data or {}
        super().__init__(message)


# Error codes
BAD_REQUEST = -32001
INVALID_STATE = -32002
NOT_ALLOWED = -32003


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


class Backend(Protocol):
    """Structural interface for a JSON-RPC backend."""

    def handle(self, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        """Handle a JSON-RPC method call. Returns the result dict."""
        ...


# ---------------------------------------------------------------------------
# SimBackend
# ---------------------------------------------------------------------------

# Methods that map to engine game actions (handled via rpc_to_action + step).
_ACTION_METHODS = frozenset(
    {
        "play",
        "discard",
        "select",
        "skip",
        "buy",
        "sell",
        "use",
        "reroll",
        "next_round",
        "cash_out",
        "pack",
        "rearrange",
    }
)


class SimBackend:
    """Backend that runs the jackdaw engine in-process."""

    def __init__(self) -> None:
        self._gs: dict[str, Any] | None = None

    def handle(self, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        if params is None:
            params = {}

        if method == "health":
            return {"status": "ok"}

        if method == "start":
            return self._handle_start(params)

        if method == "menu":
            self._gs = None
            return {"state": "MENU"}

        if method == "gamestate":
            return self._require_gamestate()

        if method in _ACTION_METHODS:
            return self._handle_action(method, params)

        raise RPCError(BAD_REQUEST, f"Unknown method: {method!r}")

    # -- internal -----------------------------------------------------------

    def _handle_start(self, params: dict[str, Any]) -> dict[str, Any]:
        from jackdaw.engine.run_init import initialize_run

        deck_str = params.get("deck", "RED")
        stake_str = params.get("stake", "WHITE")
        seed = params.get("seed", "DEFAULT")

        back_key = DECK_FROM_BOT.get(deck_str, "b_red")
        stake = STAKE_FROM_BOT.get(stake_str, 1)

        self._gs = initialize_run(back_key, stake, seed)
        self._gs["phase"] = GamePhase.BLIND_SELECT
        self._gs["blind_on_deck"] = "Small"

        return self._serialize()

    def _handle_action(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if self._gs is None:
            raise RPCError(INVALID_STATE, "No active run — call 'start' first")

        from jackdaw.bridge.deserializer import rpc_to_action

        try:
            action = rpc_to_action(method, params)
        except ValueError as exc:
            raise RPCError(BAD_REQUEST, str(exc)) from exc

        if action is None:
            return self._serialize()

        from jackdaw.engine.game import IllegalActionError, step

        try:
            step(self._gs, action)
        except IllegalActionError as exc:
            raise RPCError(NOT_ALLOWED, str(exc)) from exc

        return self._serialize()

    def _require_gamestate(self) -> dict[str, Any]:
        if self._gs is None:
            raise RPCError(INVALID_STATE, "No active run — call 'start' first")
        return self._serialize()

    def _serialize(self) -> dict[str, Any]:
        from jackdaw.bridge.serializer import game_state_to_bot_response

        return game_state_to_bot_response(self._gs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# LiveBackend
# ---------------------------------------------------------------------------


class LiveBackend:
    """Backend that proxies requests to a real balatrobot instance."""

    def __init__(self, host: str = "127.0.0.1", port: int = 12346) -> None:
        self._url = f"http://{host}:{port}"

    def handle(self, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        import httpx

        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": 1,
        }
        resp = httpx.post(self._url, json=payload, timeout=10.0)
        data = resp.json()

        if "error" in data:
            err = data["error"]
            raise RPCError(
                code=err.get("code", -32000),
                message=err.get("message", "Unknown error"),
                data=err.get("data", {}),
            )

        return data.get("result", {})

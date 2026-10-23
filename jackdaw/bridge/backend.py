"""Swappable backend interface for the JSON-RPC server.

Two implementations:

- **SimBackend** — BalatroBot v1.5.2's API on the Rust simulator, in-process
  (``jackdaw._core.BotSession``, ``crates/jackdaw-bot``): the same
  dispatcher, validator, endpoints and gamestate view as ``jackdaw serve``.
  Fast, headless, deterministic.
- **LiveBackend** — proxies all requests to a real balatrobot instance
  over HTTP. Requires a running Balatro + balatrobot mod.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from jackdaw.core import Game, Legality

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


# Error codes (balatrobot src/lua/settings.lua: BB_ERROR_CODES)
INTERNAL_ERROR = -32000
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
# ``add`` arguments: engine names → balatrobot enums
# ---------------------------------------------------------------------------

_ENHANCEMENT_TO_BOT: dict[str, str] = {
    "m_bonus": "BONUS",
    "m_mult": "MULT",
    "m_wild": "WILD",
    "m_glass": "GLASS",
    "m_steel": "STEEL",
    "m_stone": "STONE",
    "m_gold": "GOLD",
    "m_lucky": "LUCKY",
}
_EDITION_TO_BOT: dict[str, str] = {
    "foil": "FOIL",
    "holo": "HOLO",
    "polychrome": "POLYCHROME",
    "negative": "NEGATIVE",
}
_SEAL_TO_BOT: dict[str, str] = {
    "Gold": "GOLD",
    "Red": "RED",
    "Blue": "BLUE",
    "Purple": "PURPLE",
}


def to_bot_add_params(params: dict[str, Any]) -> dict[str, Any]:
    """``add`` arguments with engine names (``m_glass``, ``foil``, ``Gold``) in
    balatrobot's enums (``GLASS``, ``FOIL``, ``GOLD``); bot names pass through."""
    params = dict(params)  # shallow copy
    if "enhancement" in params:
        params["enhancement"] = _ENHANCEMENT_TO_BOT.get(
            params["enhancement"], params["enhancement"]
        )
    if "edition" in params:
        params["edition"] = _EDITION_TO_BOT.get(params["edition"], params["edition"])
    if "seal" in params:
        params["seal"] = _SEAL_TO_BOT.get(params["seal"], params["seal"])
    return params


# ---------------------------------------------------------------------------
# SimBackend
# ---------------------------------------------------------------------------


class SimBackend:
    """BalatroBot v1.5.2 on the Rust simulator, in-process.

    A thin wrapper over ``jackdaw._core.BotSession``: every request runs
    through the port of BalatroBot's dispatcher, validator and endpoint, and
    the result is its gamestate view, so results and errors (codes and
    messages) are what the live bot sends.  The session starts at the main
    menu; ``start`` begins a run.  ``add`` also takes the engine's names
    (``m_glass``, ``foil``, ``Gold``), as :class:`LiveBackend` does.

    ``fs_root`` confines ``save``/``load`` paths; ``debug`` registers
    BalatroBot's test endpoints.  See ``docs/balatrobot.md``.
    """

    def __init__(self, *, fs_root: str | os.PathLike[str] | None = None, debug: bool = False):
        from jackdaw import _core

        self._session = _core.BotSession(None if fs_root is None else os.fspath(fs_root), debug)

    def handle(self, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        if method == "add" and params:
            params = to_bot_add_params(params)
        ok, text = self._session.call(method, json.dumps({} if params is None else params))
        data = json.loads(text)
        if not ok:
            raise RPCError(data["code"], data["message"], {"name": data["name"]})
        return data

    def check(self, method: str, params: dict[str, Any] | None = None) -> RPCError | None:
        """Why the bot would refuse this request now (its argument, state and
        endpoint checks, run as a dry run), or ``None`` if it would run it."""
        ok, text = self._session.check(method, json.dumps({} if params is None else params))
        if ok:
            return None
        data = json.loads(text)
        return RPCError(data["code"], data["message"], {"name": data["name"]})

    @property
    def state(self) -> str:
        """The view's ``state``: ``MENU``, ``BLIND_SELECT``, ..., or ``UNKNOWN``."""
        return self._session.state()

    @property
    def game(self) -> Game | None:
        """A copy of the run in progress (``jackdaw.core.Game``), if any."""
        from jackdaw.core import Game

        inner = self._session.game()
        return None if inner is None else Game(inner)

    def view(self) -> dict[str, Any]:
        """The gamestate view, as ``gamestate`` returns it."""
        return json.loads(self._session.view())

    def legal_actions(self, legality: Legality | str = "balatrobot") -> list[Any]:
        """The run's legal actions (engine dataclasses); BALATROBOT legality by default."""
        from jackdaw.core import Legality, decode_action

        text = self._session.legal_actions(Legality(legality).value)
        return [decode_action(a) for a in json.loads(text)]

    def request_for(self, action: Any) -> tuple[str, dict[str, Any]] | None:
        """The request that performs an engine action in the current run:
        ``(method, params)``, or ``None`` for a sim-only action."""
        from jackdaw.core import encode_action

        out = self._session.translate(json.dumps(encode_action(action)))
        if out is None:
            return None
        method, params = out
        return method, json.loads(params) or {}  # an empty Lua table encodes as []


# ---------------------------------------------------------------------------
# LiveBackend
# ---------------------------------------------------------------------------


class LiveBackend:
    """Backend that sends requests to a BalatroBot server over HTTP (JSON-RPC 2.0).

    ``LiveBackend()`` talks to BalatroBot's default ``http://127.0.0.1:12346``;
    ``LiveBackend(url="http://127.0.0.1:12347")`` (or ``host``/``port``) to
    another instance or to ``jackdaw serve``. ``timeout`` is per request, in
    seconds: the live endpoints answer once the game's animations settle.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 12346,
        *,
        url: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.url = url.rstrip("/") if url else f"http://{host}:{port}"
        self.timeout = timeout
        self._next_id = 0
        self._client: Any = None

    def __repr__(self) -> str:
        return f"LiveBackend({self.url!r})"

    def handle(self, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        import httpx

        rpc_params = params or {}
        if method == "add" and rpc_params:
            rpc_params = to_bot_add_params(rpc_params)

        self._next_id += 1
        payload: dict[str, Any] = {
            "jsonrpc": "2.0",
            "method": method,
            "params": rpc_params,
            "id": self._next_id,
        }
        if self._client is None:
            self._client = httpx.Client(timeout=self.timeout)
        resp = self._client.post(self.url, json=payload)
        data = resp.json()

        if "error" in data:
            err = data["error"]
            raise RPCError(
                code=err.get("code", -32000),
                message=err.get("message", "Unknown error"),
                data=err.get("data", {}),
            )

        return data.get("result", {})

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

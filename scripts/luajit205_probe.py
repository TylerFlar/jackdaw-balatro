"""Ask Balatro's own Lua runtime (LuaJIT 2.0.5, ``lua51.dll``) what it does.

Some of BalatroBot's behaviour depends on the runtime rather than on the Lua
source: the order ``pairs`` visits a table in (which error the validator
reports first, ``G.GAME.hands``' iteration order) and how ``string.format``
rounds.  The game ships LuaJIT 2.0.5, which differs from the LuaJIT 2.1 the
fixtures are generated with on both counts.  This script loads the game's DLL
with ctypes and runs the probes in it.  Local only: it needs a Balatro install
(Windows), so nothing in CI calls it.

Usage::

    .venv/Scripts/python.exe scripts/luajit205_probe.py schema-order
    .venv/Scripts/python.exe scripts/luajit205_probe.py hands-order --source balatro_source
    .venv/Scripts/python.exe scripts/luajit205_probe.py format "%.13f" 6.103515625e-05
    .venv/Scripts/python.exe scripts/luajit205_probe.py run probe.lua

``--lua51`` (or ``BALATRO_DIR``) points at the game; the default is Steam's
standard library folder.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "third_party" / "balatrobot" / "v1.5.2"
DEFAULT_GAME = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Balatro")

ENDPOINTS = [
    "health", "gamestate", "save", "load", "screenshot", "set", "add", "menu", "start",
    "skip", "select", "play", "discard", "cash_out", "next_round", "reroll", "buy", "pack",
    "rearrange", "sell", "use",
    "tests/echo", "tests/endpoint", "tests/error", "tests/state", "tests/validation",
]  # fmt: skip

# Stubs for what an endpoint file touches while it is loaded (not run).
_SCHEMA_ORDER = r"""
local ROOT = ...
local function read(path)
  local f = assert(io.open(ROOT .. path, "rb"))
  local s = f:read("*a")
  f:close()
  return s
end
SMODS = {
  load_file = function(path)
    return loadstring(read(path), '=[SMODS balatrobot "' .. path .. '"]')
  end,
}
G = { STATES = {
  SELECTING_HAND = 1, HAND_PLAYED = 2, DRAW_TO_HAND = 3, GAME_OVER = 4, SHOP = 5,
  PLAY_TAROT = 6, BLIND_SELECT = 7, ROUND_EVAL = 8, TAROT_PACK = 9, PLANET_PACK = 10,
  MENU = 11, TUTORIAL = 12, SPLASH = 13, SANDBOX = 14, SPECTRAL_PACK = 15, DEMO_CTA = 16,
  STANDARD_PACK = 17, BUFFOON_PACK = 18, NEW_ROUND = 19, SMODS_BOOSTER_OPENED = 999,
} }
BB_ERROR_NAMES = {}
sendDebugMessage = function() end
package.preload["nativefs"] = function() return {} end
local out = {}
for name in string.gmatch(%s, "[^,]+") do
  local ep = SMODS.load_file("src/lua/endpoints/" .. name .. ".lua")()
  local keys = {}
  for k in pairs(ep.schema) do keys[#keys + 1] = k end
  out[#out + 1] = ep.name .. ": " .. table.concat(keys, ", ")
end
return table.concat(out, "\n")
"""


class Lua:
    """One LuaJIT state in the game's ``lua51.dll``."""

    def __init__(self, dll: Path) -> None:
        os.add_dll_directory(str(dll.parent))
        lib = ctypes.CDLL(str(dll))
        lib.luaL_newstate.restype = ctypes.c_void_p
        lib.luaL_openlibs.argtypes = [ctypes.c_void_p]
        lib.luaL_loadstring.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.luaL_loadstring.restype = ctypes.c_int
        lib.lua_pcall.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        lib.lua_pcall.restype = ctypes.c_int
        lib.lua_pushstring.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        lib.lua_tolstring.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_size_t),
        ]
        lib.lua_tolstring.restype = ctypes.c_char_p
        self.lib = lib

    def run(self, code: str, *args: str) -> str:
        """Run a chunk with string arguments (``...``); its result as text."""
        lib = self.lib
        state = lib.luaL_newstate()
        lib.luaL_openlibs(state)
        n = ctypes.c_size_t()
        if lib.luaL_loadstring(state, code.encode()) != 0:
            raise RuntimeError(lib.lua_tolstring(state, -1, ctypes.byref(n)).decode())
        for a in args:
            lib.lua_pushstring(state, a.encode())
        if lib.lua_pcall(state, len(args), 1, 0) != 0:
            raise RuntimeError(lib.lua_tolstring(state, -1, ctypes.byref(n)).decode())
        out = lib.lua_tolstring(state, -1, ctypes.byref(n))
        return "nil" if out is None else out.decode()


def schema_order(lua: Lua) -> str:
    """The order ``pairs`` gives each endpoint's schema fields (the validator's order)."""
    names = '"' + ",".join(ENDPOINTS) + '"'
    return lua.run(_SCHEMA_ORDER % names, VENDORED.as_posix() + "/")


def hands_order(lua: Lua, source: Path) -> str:
    """The order ``pairs(G.GAME.hands)`` visits the hands in, on a copy of the ``hands``
    constructor in ``init_game_object`` (game.lua)."""
    text = (source / "game.lua").read_text(encoding="utf-8")
    start = text.index("hands = {", text.index("function Game:init_game_object"))
    depth, i = 0, text.index("{", start)
    while True:
        depth += {"{": 1, "}": -1}.get(text[i], 0)
        i += 1
        if depth == 0:
            break
    table = text[text.index("{", start) : i]
    code = f"local t = {table}\nlocal out = {{}}\nfor k in pairs(t) do out[#out + 1] = k end\n"
    return lua.run(code + 'return table.concat(out, ", ")')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    game = Path(os.environ.get("BALATRO_DIR", DEFAULT_GAME))
    parser.add_argument("--lua51", type=Path, default=game / "lua51.dll")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("schema-order", help="validator field order of every endpoint")
    p_hands = sub.add_parser("hands-order", help="pairs order of G.GAME.hands")
    p_hands.add_argument("--source", type=Path, default=ROOT / "balatro_source")
    p_fmt = sub.add_parser("format", help="string.format(FMT, X) in the game's runtime")
    p_fmt.add_argument("fmt")
    p_fmt.add_argument("x")
    p_run = sub.add_parser("run", help="run a Lua file and print what it returns")
    p_run.add_argument("file", type=Path)
    args = parser.parse_args()

    if not args.lua51.is_file():
        sys.exit(f"no lua51.dll at {args.lua51} (pass --lua51 or set BALATRO_DIR)")
    lua = Lua(args.lua51)
    if args.cmd == "schema-order":
        print(schema_order(lua))
    elif args.cmd == "hands-order":
        print(hands_order(lua, args.source))
    elif args.cmd == "format":
        if not re.fullmatch(r"%[-+ #0]*\d*(\.\d+)?[a-zA-Z]", args.fmt):
            sys.exit(f"not a single format specifier: {args.fmt!r}")
        print(lua.run("local f, x = ...\nreturn string.format(f, tonumber(x))", args.fmt, args.x))
    else:
        print(lua.run(args.file.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()

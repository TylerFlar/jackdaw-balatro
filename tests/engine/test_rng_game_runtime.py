"""The RNG fixture against the game's own Lua runtime (LuaJIT 2.0.5, ``lua51.dll``).

``tests/fixtures/rng_luajit.json`` was generated with LuaJIT 2.1; the shipped
game embeds 2.0.5, whose number formatting goes through the C runtime (G181).
These tests replay the fixture in the game's DLL through ctypes
(``scripts/luajit205_probe.py``): every seeded ``math.random`` draw, and every
``tonumber(string.format("%.13f", x))`` and ``tostring(x)`` answer.

Local only: set ``BALATRO_DIR`` to the game's install folder (the one holding
``lua51.dll``); without it the tests skip.
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "tests" / "fixtures" / "rng_luajit.json"


def _dll() -> Path | None:
    game = os.environ.get("BALATRO_DIR")
    if not game or sys.platform != "win32":
        return None
    dll = Path(game) / "lua51.dll"
    return dll if dll.is_file() else None


pytestmark = pytest.mark.skipif(_dll() is None, reason="BALATRO_DIR (with lua51.dll) not set")


@pytest.fixture(scope="module")
def lua():
    sys.path.insert(0, str(ROOT / "scripts"))
    from luajit205_probe import Lua

    return Lua(_dll())


@pytest.fixture(scope="module")
def data() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _f(h: str) -> float:
    return struct.unpack(">d", bytes.fromhex(h))[0]


def _h(x: float) -> str:
    return struct.pack(">d", x).hex()


def _ask(lua, expr: str, xs: list[float]) -> list[str]:
    out: list[str] = []
    for start in range(0, len(xs), 400):
        lits = ",".join(repr(f"{x:.17g}") for x in xs[start : start + 400])
        code = (
            f"local xs = {{{lits}}}\nlocal out = {{}}\n"
            f"for i, s in ipairs(xs) do local x = tonumber(s); out[i] = {expr} end\n"
            "return table.concat(out, '\\n')"
        )
        out += lua.run(code).split("\n")
    return out


def test_seeded_draws_match_the_game(lua, data) -> None:
    for case in data["seeded"]:
        d = _f(case["d"])
        if not math.isfinite(d):
            continue
        n = len(case["floats"])
        code = (
            f"math.randomseed({d:.17g})\nlocal row = {{}}\n"
            f'for j = 1, {n} do row[j] = string.format("%.17g", math.random()) end\n'
            "row[#row + 1] = tostring(math.random(1, 52))\n"
            "row[#row + 1] = tostring(math.random(1, 7))\n"
            "row[#row + 1] = tostring(math.random(0, 2^30))\n"
            'return table.concat(row, " ")'
        )
        parts = lua.run(code).split(" ")
        assert [_h(float(x)) for x in parts[:n]] == case["floats"], case["d"]
        assert [int(x) for x in parts[n:]] == case["ints"], case["d"]


def test_truncate13_answers_are_the_games(lua, data) -> None:
    cases = [(x, want) for x, want in data["truncate13"] if math.isfinite(_f(x))]
    got = _ask(
        lua,
        'string.format("%.17g", tonumber(string.format("%.13f", x)))',
        [_f(x) for x, _ in cases],
    )
    for (x, want), text in zip(cases, got, strict=True):
        assert _h(float(text)) == want, x


def test_tostring_answers_are_the_games(lua, data) -> None:
    cases = [(x, want) for x, want in data["tostring"] if math.isfinite(_f(x))]
    got = _ask(lua, "tostring(x)", [_f(x) for x, _ in cases])
    for (x, want), text in zip(cases, got, strict=True):
        assert text == want, x

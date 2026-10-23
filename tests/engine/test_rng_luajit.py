"""The Python RNG against LuaJIT itself (``tests/fixtures/rng_luajit.json``).

The fixture holds the bit patterns the game's own RNG functions produce under
LuaJIT 2.1 on x86-64 (``scripts/luajit_rng_oracle.lua``); the Rust engine is
checked against the same file (``crates/jackdaw-core/tests/rng_luajit.rs``).
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from jackdaw.engine.rng import (
    PseudoRandom,
    _luajit_random,
    _luajit_random_int,
    _luajit_seed,
    _select_element,
    _truncate_13,
    generate_starting_seed,
    pseudohash,
)

_FIXTURE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "rng_luajit.json").read_text()
)


def _f(h: str) -> float:
    return struct.unpack(">d", bytes.fromhex(h))[0]


def _h(x: float) -> str:
    return struct.pack(">d", x).hex()


def test_pseudohash() -> None:
    for case in _FIXTURE["pseudohash"]:
        s = bytes(case["bytes"]).decode("utf-8")
        assert _h(pseudohash(s)) == case["hash"], s


def test_truncate13() -> None:
    for x_hex, want in _FIXTURE["truncate13"]:
        assert _h(_truncate_13(_f(x_hex))) == want, x_hex


def test_seeded_draws() -> None:
    for case in _FIXTURE["seeded"]:
        tw = _luajit_seed(_f(case["d"]))
        for want in case["floats"]:
            assert _h(_luajit_random(tw)) == want, case["d"]
        a, b, c = case["ints"]
        assert _luajit_random_int(tw, 1, 52) == a
        assert _luajit_random_int(tw, 1, 7) == b
        assert _luajit_random_int(tw, 0, 2**30) == c


@pytest.mark.parametrize("run", _FIXTURE["streams"], ids=lambda r: r["seed"] + str(r["inject"]))
def test_stream_sequences(run: dict) -> None:
    rng = PseudoRandom(run["seed"])
    for name, value in run["inject"]:
        rng.state[name] = _f(value)
    assert _h(rng.hashed_seed) == run["hashed_seed"]
    for i, op in enumerate(run["ops"]):
        kind, key = op[0], op[1]
        if kind == "seed":
            assert _h(rng.seed(key)) == op[2], (i, op)
        elif kind == "random":
            assert _h(rng.random(key)) == op[2], (i, op)
        else:
            assert rng.random(key, op[2], op[3]) == op[4], (i, op)


def test_predict_seed() -> None:
    rng = PseudoRandom("X")
    for key, p, want in _FIXTURE["predict"]:
        assert _h(rng.predict_seed(key, p)) == want, (key, p)


def test_element() -> None:
    for seed_hex, n, key in _FIXTURE["element"]:
        table = [k * 10 for k in range(1, n + 1)]
        _, got = _select_element(table, _luajit_seed(_f(seed_hex)))
        assert got == key


def test_shuffle() -> None:
    rng = PseudoRandom("X")
    for seed_hex, perm in _FIXTURE["shuffle"]:
        items = [{"sort_id": k} for k in range(1, len(perm) + 1)]
        rng.shuffle(items, _f(seed_hex))
        assert [c["sort_id"] for c in items] == perm


def test_random_string() -> None:
    for seed_hex, want in _FIXTURE["random_string"]:
        assert generate_starting_seed(_f(seed_hex)) == want

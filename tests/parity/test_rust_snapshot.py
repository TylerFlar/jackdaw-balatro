"""The Rust run state holds every snapshot the Python oracle exports.

At every state along every engine tape and along seeded fuzz runs, the
Python snapshot is imported into the Rust ``GameState`` and exported again;
the result must be the same snapshot.  Rust import is strict (unknown or
missing keys and malformed values raise), so state the Rust side doesn't
model can't be dropped silently.
"""

from __future__ import annotations

import json
import random
from typing import Any

import pytest

from jackdaw import _core
from jackdaw.engine.actions import get_legal_actions
from tests.golden.trajectories import _BACKS, _CHALLENGES, _mixed_policy, encode_action
from tests.parity import canon
from tests.parity.harness import load_specs
from tests.parity.pyengine import PyEngine

_SPECS = load_specs()


def rust_roundtrip(snap: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(canon.to_json_compatible(snap))
    return canon.from_json_compatible(json.loads(_core._snapshot_roundtrip(text)))


def _check(snap: dict[str, Any], where: str) -> None:
    back = rust_roundtrip(snap)
    if canon.digest(back) != canon.digest(snap):
        pytest.fail(f"{where}: Rust re-export differs: {canon.diff(snap, back)}")


@pytest.mark.parametrize("spec", _SPECS, ids=[s.name for s in _SPECS])
def test_rust_reexports_tape_states(spec) -> None:
    engine = PyEngine(spec.start, spec.setup)
    for pos in range(len(spec.tape) + 1):
        snap = engine.snapshot()
        _check(snap, f"{spec.name}@{pos}")
        if pos == len(spec.tape) or snap["phase"] == "game_over" or snap["run"]["won"]:
            break
        try:
            engine.step(spec.tape[pos])
        except Exception:  # an illegal action ends the tape
            break


def _fuzz(i: int, steps: int) -> None:
    back = _BACKS[i % len(_BACKS)]
    challenge = _CHALLENGES[i % len(_CHALLENGES)] if i % 5 == 4 else None
    start = {
        "back": "b_challenge" if challenge else back,
        "stake": 1 + (i * 3) % 8,
        "seed": f"RS_FUZZ{i}",
        "challenge": challenge,
    }
    engine = PyEngine(start)
    rng = random.Random(start["seed"])
    eps = (0.05, 0.15, 0.4)[i % 3]
    for pos in range(steps):
        snap = engine.snapshot()
        _check(snap, f"{start['seed']}@{pos}")
        if snap["phase"] == "game_over" or snap["run"]["won"]:
            return
        with engine._ids():
            legal = get_legal_actions(engine.gs)
            if not legal:
                return
            action = encode_action(_mixed_policy(engine.gs, legal, rng, eps))
        try:
            engine.step(action)
        except Exception:
            return


@pytest.mark.parametrize("i", range(20))
def test_rust_reexports_fuzz_states(i: int) -> None:
    _fuzz(i, 400)


def _first_tape_snapshot() -> dict[str, Any]:
    spec = _SPECS[0]
    return PyEngine(spec.start, spec.setup).snapshot()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda s: s.__setitem__("bogus", 1), 'unknown key "bogus"'),
        (lambda s: s["game"].pop("dollars"), 'missing key "dollars"'),
        (lambda s: s.__setitem__("schema", "jackdaw.snapshot/0"), "unsupported schema"),
        (lambda s: s["run"].__setitem__("back", "b_nope"), "unknown center"),
        (lambda s: s["areas"]["deck"][0]["base"].__setitem__("nominal", 99), "disagrees"),
    ],
)
def test_rust_import_rejects_bad_snapshots(mutate, message) -> None:
    snap = _first_tape_snapshot()
    mutate(snap)
    with pytest.raises(ValueError, match=message):
        rust_roundtrip(snap)

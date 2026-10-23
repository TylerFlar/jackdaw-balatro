"""The canonical snapshot is complete: a state rebuilt from it plays identically.

For every state along every engine tape (and along seeded fuzz runs), the
snapshot is exported, a fresh Python state is rebuilt from it, both states
take the same action, and their snapshots must still match.  Any state the
snapshot leaves out that matters for play shows up as a mismatch at the step
that reads it.
"""

from __future__ import annotations

import random

import pytest

from jackdaw.engine.actions import get_legal_actions
from tests.golden.trajectories import _BACKS, _CHALLENGES, _mixed_policy, encode_action
from tests.parity import canon
from tests.parity.harness import load_specs
from tests.parity.pyengine import PyEngine

_SPECS = load_specs()


def _outcome(engine: PyEngine, action: list) -> str | None:
    try:
        engine.step(action)
    except Exception as exc:  # behaviour includes refusing an action
        return type(exc).__name__
    return None


def _transplant_step(engine: PyEngine, action: list, where: str) -> str | None:
    """Step ``engine`` and a copy rebuilt from its snapshot; assert they agree."""
    snap = engine.snapshot()
    rebuilt = PyEngine.from_snapshot(snap)
    assert not canon.diff(snap, rebuilt.snapshot()), f"{where}: export(import(s)) != s"
    r1 = _outcome(engine, action)
    r2 = _outcome(rebuilt, action)
    assert r1 == r2, f"{where} {action}: outcome {r1} != {r2}"
    if r1 is None:
        d = canon.diff(engine.snapshot(), rebuilt.snapshot())
        assert not d, f"{where} {action}: states diverge after one step: {d}"
    return r1


@pytest.mark.parametrize("spec", _SPECS, ids=[s.name for s in _SPECS])
def test_transplant_along_tape(spec) -> None:
    engine = PyEngine(spec.start, spec.setup)
    for pos, action in enumerate(spec.tape):
        snap = engine.snapshot()
        if snap["phase"] == "game_over" or snap["run"]["won"]:
            break
        if _transplant_step(engine, action, f"{spec.name}@{pos}") is not None:
            break


def _fuzz(i: int, steps: int) -> None:
    back = _BACKS[i % len(_BACKS)]
    challenge = _CHALLENGES[i % len(_CHALLENGES)] if i % 5 == 4 else None
    start = {
        "back": "b_challenge" if challenge else back,
        "stake": 1 + (i * 3) % 8,
        "seed": f"RT_FUZZ{i}",
        "challenge": challenge,
    }
    engine = PyEngine(start)
    rng = random.Random(start["seed"])
    eps = (0.05, 0.15, 0.4)[i % 3]
    for pos in range(steps):
        snap = engine.snapshot()
        if snap["phase"] == "game_over" or snap["run"]["won"]:
            return
        with engine._ids():
            legal = get_legal_actions(engine.gs)
            if not legal:
                return
            action = encode_action(_mixed_policy(engine.gs, legal, rng, eps))
        _transplant_step(engine, action, f"{start['seed']}@{pos}")


@pytest.mark.parametrize("i", range(12))
def test_transplant_fuzz(i: int) -> None:
    _fuzz(i, steps=250)


@pytest.mark.slow
@pytest.mark.parametrize("i", range(12, 212))
def test_transplant_fuzz_slow(i: int) -> None:
    _fuzz(i, steps=600)


def test_canonical_encoding_semantics() -> None:
    # ints and floats are the same Lua number; -0.0 is 0.0; NaNs collapse
    assert canon.digest({"a": 1}) == canon.digest({"a": 1.0})
    assert canon.digest(-0.0) == canon.digest(0.0)
    assert canon.digest(float("nan")) == canon.digest(-float("nan"))
    assert canon.digest(True) != canon.digest(1)
    assert canon.digest({"b": 1, "a": 2}) == canon.digest({"a": 2, "b": 1})
    assert canon.digest([1, 2]) != canon.digest([2, 1])
    assert canon.digest("x") != canon.digest(["x"])
    inf_tree = {"x": float("inf"), "y": [float("-inf"), float("nan")]}
    back = canon.from_json_compatible(canon.to_json_compatible(inf_tree))
    assert canon.digest(back) == canon.digest(inf_tree)

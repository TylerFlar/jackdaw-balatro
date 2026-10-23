"""Full lockstep: the Rust engine plays each tape on its own state.

Unlike :mod:`tests.parity.test_rust_step`, which re-imports the oracle's state
before every step, Rust here keeps its own state from the start: a new run
from ``_core._new_run`` (for a spec with setup ops, a new ``_core.Game`` with
the ops applied natively, ``Game.apply_setup``), then every tape action
applied to Rust's previous result.
Each step's snapshot must equal the oracle's. A step Rust can't take yet ends
that tape's lockstep; how far each tape got is the coverage report.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from jackdaw import _core
from tests.parity import canon
from tests.parity.harness import load_specs
from tests.parity.pyengine import PyEngine
from tests.parity.rust_setup import rust_setup, rust_snapshot
from tests.parity.rust_step import plain

_SPECS = load_specs()


def _rust(snap: dict[str, Any], action: list[Any]) -> tuple[str, dict[str, Any] | None]:
    status, text = _core._step_snapshot(
        json.dumps(canon.to_json_compatible(snap)), json.dumps(action)
    )
    if status != "ok":
        return status, None
    return status, canon.from_json_compatible(json.loads(text))


def lockstep(spec: Any) -> tuple[int, str | None, list[str]]:
    """Play ``spec`` in both engines; return (steps matched, why it stopped, first diff)."""
    py = PyEngine(spec.start, spec.setup)
    if spec.setup:
        rs = rust_snapshot(rust_setup(spec.start, spec.setup))
    else:
        s = spec.start
        text = _core._new_run(s["back"], s["stake"], s["seed"], s.get("challenge"))
        rs = canon.from_json_compatible(json.loads(text))
    found = canon.diff(plain(py.snapshot()), rs)
    if found:
        return 0, "setup" if spec.setup else "new run", found
    for pos, action in enumerate(spec.tape):
        if plain(py.snapshot())["phase"] == "game_over":
            return pos, None, []
        status, after = _rust(rs, action)
        try:
            py.step(action)
            py_ok = True
        except Exception:  # the oracle refuses: the tape ends here
            py_ok = False
        if status.startswith("unported"):
            return pos, status, []
        if (status == "ok") != py_ok:
            return pos, f"outcome: rust {status!r}, python ok={py_ok}", []
        if not py_ok:
            return pos, None, []
        found = canon.diff(plain(py.snapshot()), after)
        if found:
            return pos, f"mismatch at step {pos} ({action[0]})", found
        rs = after
    return len(spec.tape), None, []


@pytest.mark.parametrize("spec", _SPECS, ids=[s.name for s in _SPECS])
def test_rust_keeps_step_with_the_oracle(spec) -> None:
    done, why, found = lockstep(spec)
    assert not found, f"{spec.name}: {why}: " + "; ".join(found[:8])

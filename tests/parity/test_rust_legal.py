"""The Rust engine's legal actions equal the oracle's at every tape state.

``jackdaw._core.Game.legal_actions`` lists the actions in the Python engine's
form and order (``jackdaw.engine.actions.get_legal_actions``): marker
``PlayHand`` / ``Discard`` actions and each usable consumable with its default
selection. Every state along every tape is compared.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from jackdaw import _core
from jackdaw.engine.actions import get_legal_actions
from tests.golden.trajectories import encode_action
from tests.parity import canon
from tests.parity.harness import load_specs
from tests.parity.pyengine import PyEngine
from tests.parity.rust_step import plain

_SPECS = load_specs()


def _oracle(engine: PyEngine) -> list[Any]:
    with engine._ids():
        return [encode_action(a) for a in get_legal_actions(engine.gs)]


def _rust(snap: dict[str, Any]) -> list[Any] | None:
    try:
        game = _core.Game.from_snapshot(json.dumps(canon.to_json_compatible(snap)))
    except ValueError:
        return None
    return json.loads(game.legal_actions())


def compare(spec: Any, limit: int | None = None) -> list[str]:
    """Differences along one tape: ``where: missing [...] extra [...]`` (or the order)."""
    engine = PyEngine(spec.start, spec.setup)
    problems: list[str] = []
    for pos, action in enumerate(spec.tape[:limit]):
        snap = plain(engine.snapshot())
        if snap["phase"] == "game_over":
            break
        py = _oracle(engine)
        rs = _rust(snap)
        if rs is not None and rs != py:
            pj = [json.dumps(a) for a in py]
            rj = [json.dumps(a) for a in rs]
            missing = [a for a in pj if a not in rj]
            extra = [a for a in rj if a not in pj]
            what = f"missing {missing[:4]} extra {extra[:4]}" if missing or extra else "order"
            problems.append(f"{spec.name}@{pos} ({snap['phase']}): {what}")
        try:
            engine.step(action)
        except Exception:
            break
    return problems


@pytest.mark.parametrize("spec", _SPECS, ids=[s.name for s in _SPECS])
def test_legal_actions_match_the_oracle(spec) -> None:
    problems = compare(spec)
    assert not problems, "\n".join(problems[:6])

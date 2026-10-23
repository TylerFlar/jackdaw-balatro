"""A few seeded lockstep fuzz cases (:mod:`tests.parity.fuzz`).

The nightly workflow runs a thousand new cases each night; these fixed ones
keep the harness itself working and catch gross regressions in the suite.
Divergences the fuzzer found are kept in ``tests/fixtures/fuzz_regressions.json``
(specs written by ``--save``) and replayed here.
"""

from __future__ import annotations

import json
import random

import pytest

from tests.parity.fuzz import random_case, replay, run_case
from tests.parity.harness import FIXTURES

_CASES = [random_case(i, random.Random(1000 + i), max_steps=300) for i in range(8)]


_REGRESSIONS = json.loads((FIXTURES / "fuzz_regressions.json").read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", _CASES, ids=[c.name for c in _CASES])
def test_rust_keeps_step_on_a_random_run(case) -> None:
    out = run_case(case)
    assert out.divergence is None, f"{out.divergence}: " + "; ".join(out.diff[:8])


@pytest.mark.parametrize("spec", _REGRESSIONS, ids=[s["name"] for s in _REGRESSIONS])
def test_fuzz_regression_replays_in_step(spec) -> None:
    out = replay(spec)
    assert out.divergence is None, f"{out.divergence}: " + "; ".join(out.diff[:8])
    assert out.stop in ("tape end", "tape action refused", "game over", "game over (won)"), out.stop

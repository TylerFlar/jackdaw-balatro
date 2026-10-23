"""A few seeded lockstep fuzz cases (:mod:`tests.parity.fuzz`).

The nightly workflow runs a thousand new cases each night; these fixed ones
keep the harness itself working and catch gross regressions in the suite.
"""

from __future__ import annotations

import random

import pytest

from tests.parity.fuzz import random_case, run_case

_CASES = [random_case(i, random.Random(1000 + i), max_steps=300) for i in range(8)]


@pytest.mark.parametrize("case", _CASES, ids=[c.name for c in _CASES])
def test_rust_keeps_step_on_a_random_run(case) -> None:
    out = run_case(case)
    assert out.divergence is None, f"{out.divergence}: " + "; ".join(out.diff[:8])

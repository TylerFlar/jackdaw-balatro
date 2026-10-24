"""Lockstep fuzz cases, frozen.

The lockstep fuzzer played random runs in both engines and compared every step: random decks,
stakes, challenges, forced bosses and joker boards (BalatroBot ``add``), with a seeded random
policy mixed with the smart agent. Eight seeded cases (``tests/fixtures/fuzz_cases.json``) and
the divergences it found, each fixed since (``tests/fixtures/fuzz_regressions.json``), keep
their tapes, each state's legal-action digest and each step's state digest.
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.parity import frozen

_CASES = frozen.load("fuzz_cases.json")["cases"]
_REGRESSIONS = frozen.load("fuzz_regressions.json")["cases"]


def _replay(case: dict[str, Any]) -> None:
    for _ in frozen.replay_tape(case):
        pass


@pytest.mark.parametrize("case", _CASES, ids=[c["name"] for c in _CASES])
def test_rust_keeps_step_on_a_random_run(case) -> None:
    _replay(case)


@pytest.mark.parametrize("spec", _REGRESSIONS, ids=[s["name"] for s in _REGRESSIONS])
def test_fuzz_regression_replays_in_step(spec) -> None:
    # A saved tape ends at the divergence: the steps up to it, then possibly an action both
    # engines refuse.
    _replay(dict(spec, tape=spec["tape"][: len(spec["digests"])]))

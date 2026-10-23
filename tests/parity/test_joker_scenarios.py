"""Per-joker parity: every joker the Rust engine has ported plays seeded scenarios.

Each ported joker gets a few short seeded runs with only that joker on the
board (:func:`tests.parity.joker_scenarios.scenario`); every step must match
the oracle.  Steps Rust refuses as unported are skipped there, so this gate
grows with the port: a joker joins it as soon as it is ported.
"""

from __future__ import annotations

import pytest

from jackdaw import _core
from tests.parity.joker_scenarios import scenario
from tests.parity.rust_step import Report

_PORTED = _core._ported_jokers()

# Seeds and steps per joker: enough to reach a cash out and the next blind.
_SEEDS = 2
_STEPS = 40


@pytest.mark.parametrize("key", _PORTED)
def test_ported_joker_matches_the_oracle(key: str) -> None:
    report = Report()
    for index in range(_SEEDS):
        scenario((key,), index, steps=_STEPS, report=report)
    found = [
        f"{where}: {'; '.join(diff[:6])}"
        for items in report.mismatches.values()
        for where, diff in items
    ]
    assert not found, "\n".join(found[:5])
    assert sum(report.matched.values()) > 0, "no step was checked"

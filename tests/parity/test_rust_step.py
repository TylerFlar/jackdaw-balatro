"""The phase-3 gate: every tape step the Rust engine can take matches the oracle.

For every state along every engine tape, the Python snapshot is imported into
Rust, both engines take the tape's action, and the snapshots afterwards must
be identical (or both engines refuse the action).  Steps that need logic the
port doesn't have yet are refused by Rust as unported and skipped.
"""

from __future__ import annotations

import pytest

from tests.parity.harness import load_specs
from tests.parity.rust_step import Report, run_spec

_SPECS = load_specs()


@pytest.mark.parametrize("spec", _SPECS, ids=[s.name for s in _SPECS])
def test_rust_steps_match_the_oracle(spec) -> None:
    report = Report()
    run_spec(spec, report)
    found = [
        f"{where}: {'; '.join(diff[:6])}"
        for items in report.mismatches.values()
        for where, diff in items
    ]
    assert not found, "\n".join(found[:5])

"""BalatroBot's ``add`` / ``set`` and the other parity setup ops, natively in Rust.

Every parity spec with setup ops, and the joker boards the fuzzer builds
(:func:`tests.parity.fuzz.random_case`), is set up in both engines from a new
run; the canonical snapshots must be equal before the tape starts.
"""

from __future__ import annotations

import random

import pytest

from tests.parity.fuzz import random_case
from tests.parity.harness import load_specs
from tests.parity.rust_setup import setup_diff

_SPECS = [s for s in load_specs() if s.setup]
_CASES = [c for c in (random_case(i, random.Random(i)) for i in range(400)) if c.setup][:150]


@pytest.mark.parametrize("spec", _SPECS, ids=[s.name for s in _SPECS])
def test_spec_setup_matches_the_oracle(spec) -> None:
    found = setup_diff(spec.start, spec.setup)
    assert not found, f"{spec.name}: " + "; ".join(found[:8])


def test_fuzz_boards_match_the_oracle() -> None:
    failures = []
    for case in _CASES:
        found = setup_diff(case.start, case.setup)
        if found:
            failures.append(f"{case.name} {case.setup}: " + "; ".join(found[:4]))
    assert not failures, f"{len(failures)}/{len(_CASES)} boards differ:\n" + "\n".join(
        failures[:10]
    )

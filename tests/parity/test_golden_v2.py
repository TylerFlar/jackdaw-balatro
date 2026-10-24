"""Engine-agnostic behaviour guard: the Rust engine replays the tapes and the canonical digests
must be golden v2's.

Each trajectory's chain covers every state along the tape: the snapshot and the legal-action
list before each step, the step's action, and the final snapshot (:mod:`tests.parity.harness`).
The fixture was produced by the Python oracle, so a matching chain means the Rust engine plays
every tape exactly as the oracle did.
"""

from __future__ import annotations

import pytest

from tests.parity.golden_v2 import first_divergence, load_fixture
from tests.parity.harness import RustEngine, load_specs, replay

_SPECS = load_specs()
_GOLDEN = load_fixture()["trajectories"]


def test_fixture_covers_every_spec() -> None:
    assert sorted(_GOLDEN) == sorted(s.name for s in _SPECS)


@pytest.mark.parametrize("spec", _SPECS, ids=[s.name for s in _SPECS])
def test_golden_v2(spec) -> None:
    expected = _GOLDEN[spec.name]
    got = replay(spec, RustEngine(spec.start, spec.setup))
    assert all(got.checks.values()), f"scenario check failed: {got.checks}"
    if got.digest != expected["digest"]:
        pytest.fail(
            f"{spec.name}: behaviour changed at {first_divergence(expected, got)}; "
            "if intended, refreeze with scripts/refreeze.py golden_v2"
        )
    assert got.records == expected["records"]
    assert got.checks == expected["checks"]

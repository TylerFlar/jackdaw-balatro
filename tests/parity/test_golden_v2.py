"""Engine-agnostic behaviour guard: replay the tapes and compare canonical digests."""

from __future__ import annotations

import pytest

from tests.parity.golden_v2 import first_divergence, load_fixture
from tests.parity.harness import load_specs, replay
from tests.parity.pyengine import PyEngine

_SPECS = load_specs()
_GOLDEN = load_fixture()["trajectories"]


def test_fixture_covers_every_spec() -> None:
    assert sorted(_GOLDEN) == sorted(s.name for s in _SPECS)


@pytest.mark.parametrize("spec", _SPECS, ids=[s.name for s in _SPECS])
def test_golden_v2(spec) -> None:
    expected = _GOLDEN[spec.name]
    got = replay(spec, PyEngine(spec.start, spec.setup))
    assert all(got.checks.values()), f"scenario check failed: {got.checks}"
    if got.digest != expected["digest"]:
        pytest.fail(
            f"{spec.name}: behaviour changed at {first_divergence(expected, got)}; "
            "if intended, regenerate with scripts/generate_golden_v2.py"
        )
    assert got.records == expected["records"]
    assert got.checks == expected["checks"]

"""Behaviour-equivalence guard: replay pinned trajectories and compare digests.

A failure means observable behaviour changed: game state, RNG stream,
observation encoding, action mask, Gym action table, env info or reward.
Performance work must keep these passing unchanged.  If a change is meant
to alter behaviour, regenerate the fixtures with
``uv run python scripts/generate_golden_trajectories.py`` and explain why.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.golden.trajectories import CHECKPOINT_EVERY, Result, all_cases

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
_DATA = json.loads((FIXTURES / "golden_trajectories.json").read_text())
_EXPECTED = _DATA["trajectories"]
_TAPES = json.loads((FIXTURES / "golden_tapes.json").read_text())
_CASES = all_cases()


def _check(name: str, got: Result) -> None:
    exp = _EXPECTED[name]
    if got.digest == exp["digest"] and got.steps == exp["steps"]:
        return
    first_bad = next(
        (
            (i + 1) * CHECKPOINT_EVERY
            for i, (a, b) in enumerate(zip(got.checkpoints, exp["checkpoints"]))
            if a != b
        ),
        None,
    )
    where = f"diverged by record {first_bad}" if first_bad else "diverged after last checkpoint"
    pytest.fail(
        f"{name}: behaviour changed ({where}; records {got.steps} vs expected {exp['steps']})"
    )


def test_fixture_matches_cases():
    assert {name for name, _, _ in _CASES} == set(_EXPECTED)
    assert _DATA["checkpoint_every"] == CHECKPOINT_EVERY


@pytest.mark.parametrize(("name", "runner", "spec"), _CASES, ids=[c[0] for c in _CASES])
def test_golden(name, runner, spec):
    got = runner(spec, _TAPES.get(name, []))
    for check, ok in got.checks.items():
        assert ok, f"{name}: scenario check {check!r} no longer holds"
    _check(name, got)

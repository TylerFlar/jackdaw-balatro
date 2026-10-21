"""Behaviour-equivalence guard: replay pinned trajectories and compare digests.

A failure means observable behaviour changed: game state, RNG stream,
observation encoding, action mask, Gym action table or reward.  Performance
work must keep these passing unchanged.  If a change is meant to alter
behaviour, regenerate the fixture with
``uv run python scripts/generate_golden_trajectories.py`` and explain why.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.golden.trajectories import (
    CHECKPOINT_EVERY,
    Result,
    engine_specs,
    gym_specs,
    run_engine,
    run_gym,
)

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "golden_trajectories.json"
_DATA = json.loads(FIXTURE.read_text())
_EXPECTED = _DATA["trajectories"]


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
        f"{name}: behaviour changed ({where}; steps {got.steps} vs expected {exp['steps']})"
    )


def test_fixture_matches_specs():
    names = {s.name for s in engine_specs()} | {s.name for s in gym_specs()}
    assert names == set(_EXPECTED)
    assert _DATA["checkpoint_every"] == CHECKPOINT_EVERY


@pytest.mark.parametrize("spec", engine_specs(), ids=lambda s: s.name)
def test_engine_trajectory(spec):
    _check(spec.name, run_engine(spec))


@pytest.mark.parametrize("spec", gym_specs(), ids=lambda s: s.name)
def test_gym_trajectory(spec):
    _check(spec.name, run_gym(spec))

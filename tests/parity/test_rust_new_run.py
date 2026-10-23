"""A run started in Rust matches the oracle's new run.

For every back and stake, and every challenge, ``_core._new_run`` must produce
the same snapshot as the oracle's ``initialize_run`` at the blind-select screen.
"""

from __future__ import annotations

import json

import pytest

from jackdaw import _core
from jackdaw.engine.challenges import CHALLENGES
from jackdaw.engine.data.prototypes import BACKS
from tests.parity import canon
from tests.parity.pyengine import PyEngine
from tests.parity.rust_step import plain

_BACKS = sorted(k for k in BACKS if k != "b_challenge")
_STAKES = range(1, 9)
_CHALLENGES = sorted(CHALLENGES)
_SEEDS = ("NEWRUN01", "7LB2WVPK")


def _check(back: str, stake: int, seed: str, challenge: str | None) -> list[str]:
    start = {"back": back, "stake": stake, "seed": seed, "challenge": challenge}
    py = plain(PyEngine(start).snapshot())
    rs = canon.from_json_compatible(json.loads(_core._new_run(back, stake, seed, challenge)))
    return canon.diff(py, rs)


@pytest.mark.parametrize("back", _BACKS)
@pytest.mark.parametrize("stake", _STAKES)
def test_new_run_matches_the_oracle(back: str, stake: int) -> None:
    for seed in _SEEDS:
        found = _check(back, stake, seed, None)
        assert not found, f"{back} stake {stake} {seed}: " + "; ".join(found[:8])


@pytest.mark.parametrize("challenge", _CHALLENGES)
def test_challenge_run_matches_the_oracle(challenge: str) -> None:
    for seed in _SEEDS:
        found = _check("b_challenge", 1, seed, challenge)
        assert not found, f"{challenge} {seed}: " + "; ".join(found[:8])

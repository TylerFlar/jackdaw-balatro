"""A run started in Rust is the oracle's new run.

For every back and stake, and every challenge, at two seeds, ``Game.new`` must give the
canonical digest the oracle's ``initialize_run`` gave at the blind-select screen
(``tests/fixtures/new_runs.json``).
"""

from __future__ import annotations

import pytest

from jackdaw.core import Game
from tests.parity import frozen

_RUNS = frozen.load("new_runs.json")
_BACKS = sorted({k.split("/")[0] for k in _RUNS if k.startswith("b_")})
_CHALLENGES = sorted({k.split("/")[0] for k in _RUNS if k.startswith("c_")})
_SEEDS = ("NEWRUN01", "7LB2WVPK")


@pytest.mark.parametrize("back", _BACKS)
@pytest.mark.parametrize("stake", range(1, 9))
def test_new_run_matches_the_oracle(back: str, stake: int) -> None:
    for seed in _SEEDS:
        got = Game.new(back, stake, seed).state_hash()
        assert got == _RUNS[f"{back}/{stake}/{seed}"], f"{back} stake {stake} {seed}"


@pytest.mark.parametrize("challenge", _CHALLENGES)
def test_challenge_run_matches_the_oracle(challenge: str) -> None:
    for seed in _SEEDS:
        got = Game.new("b_challenge", 1, seed, challenge).state_hash()
        assert got == _RUNS[f"{challenge}/{seed}"], f"{challenge} {seed}"

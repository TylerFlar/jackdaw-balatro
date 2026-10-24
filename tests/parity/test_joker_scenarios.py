"""Per-joker scenarios: two short seeded runs for every joker, frozen from the lockstep.

Each run starts at the first blind, adds one joker through BalatroBot's ``add`` and plays a
seeded policy: it selects every blind, plays and discards with the golden mixed policy, cashes
out and leaves the shop and packs without buying. Every step was checked against the Python
oracle; ``tests/fixtures/joker_scenarios.json`` keeps the tapes and the state after each step.
The Rust engine must reach the same states, playing on and from each state's snapshot.
"""

from __future__ import annotations

import pytest

from jackdaw import _core
from tests.parity import frozen

_SCENARIOS = frozen.load("joker_scenarios.json")


def test_every_joker_has_scenarios() -> None:
    assert sorted(_SCENARIOS) == sorted(_core._ported_jokers())


@pytest.mark.parametrize("key", sorted(_SCENARIOS))
def test_ported_joker_matches_the_oracle(key: str) -> None:
    steps = 0
    for case in _SCENARIOS[key]:
        for steps, _ in frozen.replay_tape(case):
            pass
    assert steps > 0, "no step was checked"

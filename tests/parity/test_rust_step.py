"""Single-step transplant: every tape step, taken by a run rebuilt from the state's snapshot.

For every state along every engine tape, the Rust run is rebuilt from its canonical snapshot
(``Game.from_dict``) before it takes the tape's action, the way the oracle's states were stepped
in Rust while the oracle existed. The tape's digest chain must still be golden v2's, so each
step from a snapshot reaches the oracle's state (or both refuse the action).
"""

from __future__ import annotations

from typing import Any

import pytest

from tests.parity import frozen
from tests.parity.golden_v2 import first_divergence, load_fixture
from tests.parity.harness import RustEngine, load_specs, replay

_SPECS = load_specs()
_GOLDEN = load_fixture()["trajectories"]


class TransplantEngine(RustEngine):
    """A Rust run rebuilt from its snapshot before every step."""

    def step(self, action: list[Any]) -> None:
        self.game = frozen.transplant(self.game)
        self.game.step(action)


@pytest.mark.parametrize("spec", _SPECS, ids=[s.name for s in _SPECS])
def test_rust_steps_match_the_oracle(spec) -> None:
    expected = _GOLDEN[spec.name]
    got = replay(spec, TransplantEngine(spec.start, spec.setup))
    assert got.digest == expected["digest"], f"{spec.name}: {first_divergence(expected, got)}"

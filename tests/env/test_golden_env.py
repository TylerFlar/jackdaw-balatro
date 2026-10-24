"""The env layer along the parity tapes, against the Python env layer's frozen outputs.

At every state of every parity tape (the Rust engine replaying ``parity_specs.json``) the
observation (encoder v1), the vanilla action mask and the factored-action conversions must
be what the Python env layer computed on the oracle's state (``tests/fixtures/golden_env.json``:
a digest chain per tape, :mod:`tests.parity.frozen`). The Gym action table is built at every
state both in Rust and by ``BalatroGymnasiumEnv._enumerate_actions``, with the same sampler
seed, and must be the same.
"""

from __future__ import annotations

from typing import Any

import pytest

from jackdaw.core import IllegalActionError
from tests.parity import frozen
from tests.parity.harness import load_specs

_SPECS = load_specs()
_GOLDEN = frozen.load("golden_env.json")


def test_fixture_covers_every_spec() -> None:
    assert sorted(_GOLDEN) == sorted(s.name for s in _SPECS)


@pytest.mark.parametrize("spec", _SPECS, ids=[s.name for s in _SPECS])
def test_tape_states_match(spec: Any) -> None:
    game = frozen.new_game(spec.start, spec.setup)
    chain = frozen.Chain()
    for pos in range(len(spec.tape) + 1):
        where = f"{spec.name}@{pos}"
        frozen.push_env_state(game, where, chain, frozen.table_seed(where))
        if game.phase == "game_over" or game.won or pos == len(spec.tape):
            break
        try:
            game.step(spec.tape[pos])
        except IllegalActionError:  # the tape ends here
            break
    frozen.check_chain(_GOLDEN[spec.name], chain, spec.name)

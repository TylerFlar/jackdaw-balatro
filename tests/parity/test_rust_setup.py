"""BalatroBot's ``add`` / ``set`` and the other parity setup ops, natively in Rust.

Every parity spec with setup ops, and 150 joker boards the lockstep fuzzer built, are set up
from a new run with ``Game.apply_setup``; each must reach the canonical digest the oracle's
setup reached (``tests/fixtures/setup_boards.json``).
"""

from __future__ import annotations

import pytest

from tests.parity import frozen
from tests.parity.harness import load_specs

_FROZEN = frozen.load("setup_boards.json")
_SPECS = [s for s in load_specs() if s.setup]


def test_every_spec_setup_is_frozen() -> None:
    assert sorted(_FROZEN["specs"]) == sorted(s.name for s in _SPECS)


@pytest.mark.parametrize("spec", _SPECS, ids=[s.name for s in _SPECS])
def test_spec_setup_matches_the_oracle(spec) -> None:
    game = frozen.new_game(spec.start, spec.setup)
    assert game.state_hash() == _FROZEN["specs"][spec.name]


def test_fuzz_boards_match_the_oracle() -> None:
    failures = [
        f"{board['name']} {board['setup']}"
        for board in _FROZEN["boards"]
        if frozen.new_game(board["start"], board["setup"]).state_hash() != board["digest"]
    ]
    assert not failures, f"{len(failures)}/{len(_FROZEN['boards'])} boards differ:\n" + "\n".join(
        failures[:10]
    )

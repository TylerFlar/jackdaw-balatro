"""``jackdaw.core.Game``: the Rust engine behind a Python run object.

The run's state must be the oracle's (its frozen digests), survive its snapshot round trip,
copy independently, and refuse illegal steps without changing.
"""

from __future__ import annotations

import re

import pytest

from jackdaw.core import Game, IllegalActionError, random_seed
from jackdaw.engine.actions import Discard, GamePhase, PlayHand, SelectBlind, SortHand
from tests.parity import canon, frozen
from tests.parity.golden_v2 import first_divergence, load_fixture
from tests.parity.harness import RustEngine, load_specs, replay


def test_a_new_run_is_the_oracles() -> None:
    game = Game.new("b_blue", 3, "NEWRUN01")
    assert game.state_hash() == frozen.load("new_runs.json")["b_blue/3/NEWRUN01"]
    assert game.state_hash() == canon.digest(game.to_dict())
    assert game.phase is GamePhase.BLIND_SELECT
    assert (game.ante, game.round, game.seed) == (1, 0, "NEWRUN01")


def test_the_snapshot_round_trips() -> None:
    game = Game.new(seed="ROUNDTRIP")
    game.step(SelectBlind())
    again = Game.from_dict(game.to_dict())
    assert again.state_hash() == game.state_hash()


def test_a_clone_is_independent() -> None:
    game = Game.new(seed="CLONES")
    game.step(SelectBlind())
    copy = game.clone()
    before = game.state_hash()
    copy.step(PlayHand(card_indices=(0, 1)))
    assert game.state_hash() == before
    assert copy.state_hash() != before


def test_an_illegal_step_changes_nothing() -> None:
    game = Game.new(seed="ILLEGAL")
    before = game.state_hash()
    with pytest.raises(IllegalActionError):
        game.step(Discard(card_indices=(0,)))  # no hand yet
    assert game.state_hash() == before
    game.step(SelectBlind())
    with pytest.raises(IllegalActionError):
        game.step(PlayHand(card_indices=(0, 1, 2, 3, 4, 5)))
    game.step(SortHand(mode="suit"))
    assert game.phase is GamePhase.SELECTING_HAND


def test_random_seeds_look_like_vanillas() -> None:
    for _ in range(50):
        assert re.fullmatch(r"[1-9A-NP-Z]{8}", random_seed())


@pytest.mark.parametrize("name", ["deck_b_red", "stake_1"])
def test_game_plays_a_tape_as_the_oracle_did(name: str) -> None:
    spec = next(s for s in load_specs() if s.name == name)
    expected = load_fixture()["trajectories"][name]
    got = replay(spec, RustEngine(spec.start, spec.setup))
    assert got.digest == expected["digest"], f"{name}: {first_divergence(expected, got)}"


@pytest.mark.parametrize("name", ["deck_b_erratic", "challenge_c_inflation_1", "jokers_3"])
def test_state_hash_is_the_canonical_digest(name: str) -> None:
    spec = next(s for s in load_specs() if s.name == name)
    game = frozen.new_game(spec.start, spec.setup)
    for action in spec.tape[:40]:
        snap = game.to_dict()
        assert game.state_hash() == canon.digest(snap)
        assert Game.from_dict(snap).state_hash() == canon.digest(snap)
        try:
            game.step(action)
        except IllegalActionError:
            break

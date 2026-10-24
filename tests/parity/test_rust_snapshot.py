"""The Rust run state holds every snapshot it exports, and the canonical encoding's rules.

At every state along every engine tape and along seeded fuzz runs, the snapshot is imported
into the Rust ``GameState`` and exported again; the result must be the same snapshot. Rust
import is strict (unknown or missing keys and malformed values raise), so state the Rust side
doesn't model can't be dropped silently. Every state along these tapes is the oracle's (the
tapes' frozen digests, ``test_golden_v2.py`` and ``test_rust_observation.py``).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from jackdaw import _core
from jackdaw.core import Game, GameBuilder, IllegalActionError
from tests.parity import canon, frozen
from tests.parity.harness import load_specs

_SPECS = load_specs()
_FUZZ = frozen.load("env_fuzz.json")["observation"]


def rust_roundtrip(snap: dict[str, Any]) -> dict[str, Any]:
    text = json.dumps(canon.to_json_compatible(snap))
    return canon.from_json_compatible(json.loads(_core._snapshot_roundtrip(text)))


def _check(game: Game, where: str) -> None:
    text = game._g.snapshot()
    if _core._snapshot_roundtrip(text) == text:
        return
    snap = game.to_dict()
    back = rust_roundtrip(snap)
    if canon.digest(back) != canon.digest(snap):
        pytest.fail(f"{where}: Rust re-export differs: {canon.diff(snap, back)}")


def _replay(name: str, game: Game, tape: list[Any]) -> None:
    for pos, action in enumerate(tape):
        _check(game, f"{name}@{pos}")
        if game.phase == "game_over" or game.won:
            return
        try:
            game.step(action)
        except IllegalActionError:  # the tape ends here
            return
    _check(game, f"{name}@{len(tape)}")


@pytest.mark.parametrize("spec", _SPECS, ids=[s.name for s in _SPECS])
def test_rust_reexports_tape_states(spec) -> None:
    _replay(spec.name, frozen.new_game(spec.start, spec.setup), spec.tape)


@pytest.mark.parametrize("run", _FUZZ, ids=[r["name"] for r in _FUZZ])
def test_rust_reexports_fuzz_states(run: dict[str, Any]) -> None:
    _replay(run["name"], frozen.new_game(run["start"], run["setup"]), run["tape"])


def test_rust_keeps_a_queued_negative_removal() -> None:
    """A debuffed Negative joker keeps its slot and queues the slot's removal
    (card.lua:686-693, 4733-4739); found by the lockstep fuzzer. Crimson Heart debuffs the
    only joker after a hand (blind.lua:588-602)."""
    game = (
        GameBuilder("b_red", 1, "QNEG1")
        .boss("bl_final_heart")
        .select_blind()
        .joker("j_joker", edition="negative")
        .chips(0)
        .build()
    )
    slots = game.to_dict()["game"]["joker_slots"]
    game.step(["PlayHand", [0]])
    (joker,) = game.jokers
    assert joker.debuff and joker.ability.get("queue_negative_removal")
    assert game.to_dict()["game"]["joker_slots"] == slots
    _check(game, "queued negative removal")


def _first_tape_snapshot() -> dict[str, Any]:
    spec = _SPECS[0]
    return frozen.new_game(spec.start, spec.setup).to_dict()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda s: s.__setitem__("bogus", 1), 'unknown key "bogus"'),
        (lambda s: s["game"].pop("dollars"), 'missing key "dollars"'),
        (lambda s: s.__setitem__("schema", "jackdaw.snapshot/0"), "unsupported schema"),
        (lambda s: s["run"].__setitem__("back", "b_nope"), "unknown center"),
        (lambda s: s["areas"]["deck"][0]["base"].__setitem__("nominal", 99), "disagrees"),
    ],
)
def test_rust_import_rejects_bad_snapshots(mutate, message) -> None:
    snap = _first_tape_snapshot()
    mutate(snap)
    with pytest.raises(ValueError, match=message):
        rust_roundtrip(snap)


def test_canonical_encoding_semantics() -> None:
    # ints and floats are the same Lua number; -0.0 is 0.0; NaNs collapse
    assert canon.digest({"a": 1}) == canon.digest({"a": 1.0})
    assert canon.digest(-0.0) == canon.digest(0.0)
    assert canon.digest(float("nan")) == canon.digest(-float("nan"))
    assert canon.digest(True) != canon.digest(1)
    assert canon.digest({"b": 1, "a": 2}) == canon.digest({"a": 2, "b": 1})
    assert canon.digest([1, 2]) != canon.digest([2, 1])
    assert canon.digest("x") != canon.digest(["x"])
    inf_tree = {"x": float("inf"), "y": [float("-inf"), float("nan")]}
    back = canon.from_json_compatible(canon.to_json_compatible(inf_tree))
    assert canon.digest(back) == canon.digest(inf_tree)


def test_the_rust_digest_is_the_canonical_one() -> None:
    """``Game.state_hash`` (``jackdaw_core::canon``) encodes as ``tests/parity/canon.py``."""
    game = frozen.new_game(_SPECS[0].start, _SPECS[0].setup)
    for action in _SPECS[0].tape[:30]:
        assert game.state_hash() == canon.digest(game.to_dict())
        game.step(action)

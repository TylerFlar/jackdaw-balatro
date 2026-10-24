"""Frozen verdicts: what the Python oracle computed, kept as digests the Rust engine must match.

The Rust engine was checked against the Python reference engine (the oracle) step by step
until the oracle was retired. Its verdicts are kept in fixtures under ``tests/fixtures``,
recorded from the last lockstep in which both engines agreed:

* ``golden_v2.json``: digest chains over every state of the parity tapes (``golden_v2``).
* ``golden_env.json``: the same states' observations, action masks and factored-action
  conversions, as the Python env layer computed them.
* ``joker_scenarios.json``, ``fuzz_cases.json``, ``fuzz_regressions.json``: action tapes with
  each step's state digest.
* ``new_runs.json``, ``setup_boards.json``: new runs and parity setups.
* ``env_fuzz.json``, ``env_states.json``: the env layer along fuzz runs and on edited states.
* ``parity_cases.json``: the targeted parity tests' steps (``tests/parity/test_jokers_batch_*``,
  ``test_consumables_rust``, ``test_audit_*``).

A step digest is the first :data:`DIGITS` hex digits of the canonical snapshot digest
(``Game.state_hash``, ``docs/snapshot-schema.md``); a chain folds records with
:func:`tests.parity.canon.chain`, as golden v2 does. A deliberate behaviour change makes the
Rust engine disagree with a frozen verdict; say so in the PR and refreeze the affected
fixture (``scripts/refreeze.py``).
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from jackdaw import _core
from jackdaw.core import Game, IllegalActionError
from tests.parity import canon

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "tests" / "fixtures"

#: Hex digits kept of each digest (64 bits).
DIGITS = 16

#: Records per checkpoint in a chain.
CHECKPOINT_EVERY = 10


def load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def write(name: str, data: Any) -> None:
    """Write a fixture: one top-level entry per line, compact inside."""
    path = FIXTURES / name
    if isinstance(data, dict) and data:
        lines = [
            f"{json.dumps(k)}: {json.dumps(v, separators=(',', ':'))}" for k, v in data.items()
        ]
        text = "{\n" + ",\n".join(lines) + "\n}\n"
    else:
        text = json.dumps(data, separators=(",", ":")) + "\n"
    path.write_text(text, encoding="utf-8")


def short(digest: str) -> str:
    return digest[:DIGITS]


def legal_digest(legal: Sequence[Any]) -> str:
    """The digest of a legal-action list in tape form."""
    return short(canon.digest(list(legal)))


# ---------------------------------------------------------------------------
# The env layer's outputs
# ---------------------------------------------------------------------------

_OBS_FIELDS = ("global_context", "hand_cards", "jokers", "consumables", "shop_cards", "pack_cards")
_NAN32 = np.uint32(0x7FC00000)


def obs_digest(obs: Any) -> str:
    """The digest of an ``Observation``: each array's shape and float32 bits, every NaN alike."""
    h = hashlib.sha256()
    for name in _OBS_FIELDS:
        arr = np.ascontiguousarray(getattr(obs, name), dtype="<f4")
        h.update(struct.pack(">I", arr.ndim) + struct.pack(f">{arr.ndim}I", *arr.shape))
        bits = arr.view("<u4").copy()
        bits[np.isnan(arr)] = _NAN32
        h.update(bits.tobytes())
    return short(h.hexdigest())


def mask_digest(mask: Any) -> str:
    """The digest of an ``ActionMask``: the arrays, the entity types in order, the bounds."""
    h = hashlib.sha256()

    def bools(arr: Any) -> None:
        raw = np.asarray(arr, dtype=np.bool_).astype(np.uint8).tobytes()
        h.update(struct.pack(">I", len(raw)) + raw)

    bools(mask.type_mask)
    bools(mask.card_mask)
    for t, arr in mask.entity_masks.items():
        h.update(struct.pack(">I", int(t)))
        bools(arr)
    forced = -1 if mask.forced_card is None else int(mask.forced_card)
    h.update(struct.pack(">qqq", mask.min_card_select, mask.max_card_select, forced))
    return short(h.hexdigest())


def table_seed(where: str) -> int:
    """The Gym table's sampler seed at a state of a parity tape (``"<spec>@<pos>"``)."""
    return zlib.crc32(where.encode())


def table_parts(table: Sequence[Any]) -> list[tuple[int, int | None, list[int] | None]]:
    """A Gym action table (``FactoredAction`` objects or ``(type, entity, cards)`` tuples) as
    ``(type, entity, cards)`` with list selections."""
    out = []
    for a in table:
        t, e, cards = (
            (a.action_type, a.entity_target, a.card_target) if hasattr(a, "action_type") else a
        )
        out.append((int(t), e, None if cards is None else list(cards)))
    return out


_DEFAULT_AREAS = ("consumables", "pack_cards", "shop_jokers")
_TARGET_AREAS = ("consumables", "pack_cards")


def conversion_samples(table: Sequence[Any]) -> list[tuple[int, int | None, list[int] | None]]:
    """The factored actions whose engine form is checked at a state: every 13th action of the
    Gym table, then each entity type that takes a default selection, with no card target."""
    from jackdaw.env.action_space import ActionType

    samples = table_parts(table)[::13]
    for t in (ActionType.UseConsumable, ActionType.PickPackCard, ActionType.BuyAndUse):
        samples += [(int(t), e, None) for e in range(4)]
    samples.append((int(ActionType.PlayHand), None, None))
    return samples


def conversions(game: Game, table: Sequence[Any]) -> list[Any]:
    """The factored-action conversions at ``game``'s state, as the Rust env layer computes them:
    factored to engine for :func:`conversion_samples`, engine to factored for every legal
    action, the default selections, and the consumables' targeting info."""
    g = game._g
    f2e = []
    for t, e, cards in conversion_samples(table):
        try:
            out = json.loads(g.factored_to_engine(t, e, cards))
        except ValueError as exc:
            out = ["ValueError", str(exc)]
        f2e.append([[t, e, cards], out])
    e2f = [
        [a, list(_core.engine_to_factored(json.dumps(a)))] for a in json.loads(g.legal_actions())
    ]
    sizes = {area: len(json.loads(g.view(area))) for area in _DEFAULT_AREAS}
    defaults = {area: [g.default_targets(area, i) for i in range(sizes[area])] for area in sizes}
    info = {
        area: [list(g.consumable_target_info(area, i)) for i in range(sizes[area])]
        for area in _TARGET_AREAS
    }
    return [f2e, e2f, defaults, info]


def conversions_digest(record: Any) -> str:
    return short(canon.digest(canon.to_json_compatible(record)))


_GYM: Any = None


def push_env_state(game: Game, where: str, chain: Chain, seed: int | None) -> None:
    """Fold one state's env outputs into ``chain``: the observation, the vanilla action mask
    and, given a table seed, the factored-action conversions.

    The observation is encoded from the run and from its snapshot; the Gym action table is
    built in Rust and by ``BalatroGymnasiumEnv._enumerate_actions`` with the same seed, and
    the two must agree.
    """
    global _GYM
    from jackdaw.env.balatro_env import _action_mask_to_game
    from jackdaw.env.game_interface import DirectAdapter
    from jackdaw.env.gymnasium_wrapper import BalatroGymnasiumEnv
    from jackdaw.env.rust_env import action_mask
    from jackdaw.env.rust_observation import encode_game, encode_snapshot_text
    from jackdaw.env.sampler import Xoshiro256

    obs = obs_digest(encode_game(game))
    assert obs_digest(encode_snapshot_text(game._g.snapshot())) == obs, f"{where}: from snapshot"
    mask = action_mask(game)
    chain.push("obs", obs)
    chain.push("mask", mask_digest(mask))
    if seed is None:
        return
    if _GYM is None:  # a wrapper only for its Python action-table builder
        _GYM = BalatroGymnasiumEnv(adapter_factory=DirectAdapter)
    rs_table, rs_state = game._g.action_table(list(Xoshiro256(seed).state))
    _GYM._rng = Xoshiro256(seed)
    py_table = _GYM._enumerate_actions(_action_mask_to_game(mask), {"game": game})
    assert table_parts(py_table) == table_parts(rs_table), f"{where}: action tables differ"
    assert _GYM._rng.state == tuple(rs_state), f"{where}: sampler states differ"
    if game.phase == "game_over":
        chain.push("conv", None)
    else:
        chain.push("conv", conversions_digest(conversions(game, rs_table)))


def state_record(state: Any) -> list[Any]:
    """A ``GameState`` (``info["state"]``) as a JSON list."""
    return [
        state.phase.value, state.ante, state.round, state.dollars, state.hands_left,
        state.discards_left, state.hand_size, state.joker_slots, state.consumable_slots,
        state.blind_on_deck, state.blind_chips, state.chips, bool(state.won), bool(state.done),
    ]  # fmt: skip


def arrays_digest(arrays: dict[str, Any]) -> str:
    """The digest of named arrays (a Gym observation, a mask): names, shapes and bits, every
    float32 NaN alike."""
    h = hashlib.sha256()
    for name, value in arrays.items():
        arr = np.ascontiguousarray(value)
        raw = name.encode()
        h.update(struct.pack(">I", len(raw)) + raw + arr.dtype.str.encode())
        h.update(struct.pack(">I", arr.ndim) + struct.pack(f">{arr.ndim}I", *arr.shape))
        if arr.dtype == np.float32:
            bits = arr.view("<u4").copy()
            bits[np.isnan(arr)] = _NAN32
            arr = bits
        h.update(arr.tobytes())
    return short(h.hexdigest())


def env_reset_record(info: dict[str, Any]) -> list[Any]:
    """What ``BalatroEnvironment.reset`` returns, as a chain record."""
    return [
        obs_digest(info["observation"]),
        mask_digest(info["action_mask"]),
        state_record(info["state"]),
        [int(n) for n in info["shop_splits"]],
    ]


def env_step_record(env: Any, terminated: bool, truncated: bool, info: dict[str, Any]) -> list:
    """What ``BalatroEnvironment.step`` returns, and the episode stats, as a chain record."""
    return [
        obs_digest(info["observation"]),
        mask_digest(info["action_mask"]),
        bool(terminated),
        bool(truncated),
        state_record(info["prev_state"]),
        state_record(info["state"]),
        [int(n) for n in info["shop_splits"]],
        [env.episode_length, bool(env.episode_won), env.episode_ante],
    ]


def gym_record(env: Any, obs: dict[str, Any], info: dict[str, Any], *rest: Any) -> list[Any]:
    """What a ``BalatroGymnasiumEnv`` reset (``rest`` empty) or step (``rest`` the reward,
    terminated and truncated) returns, with the action table, as a chain record."""
    extra = [
        [k, v if isinstance(v, str) else float(v)] for k, v in info.items() if k != "action_mask"
    ]
    return [
        arrays_digest(obs),
        arrays_digest({"action_mask": info["action_mask"]}),
        [float(rest[0]), bool(rest[1]), bool(rest[2])] if rest else None,
        extra,
        conversions_digest(table_parts(env._action_table)),
    ]


def replay_env_run(run: dict[str, Any], *, every: int | None = None) -> int:
    """Replay a frozen env run (``env_fuzz.json``) on the Rust engine: the env outputs and the
    state at each of its states (conversions on every ``every``-th one), against its chain.
    Tape entries are actions or setup ops. Returns the number of states checked."""
    game = new_game(run["start"], run["setup"])
    chain = Chain()
    tape = run["tape"]
    for pos in range(run["states"]):
        where = f"{run['name']}@{pos}"
        seed = table_seed(where) if every and pos % every == 0 else None
        push_env_state(game, where, chain, seed)
        chain.push("state", short(game.state_hash()))
        if pos < len(tape):
            entry = tape[pos]
            if isinstance(entry, dict):
                game.apply_setup(entry)
            else:
                game.step(entry)
    if "refused" in run:
        try:
            game.step(run["refused"])
        except IllegalActionError:
            pass
        else:
            raise AssertionError(f"{run['name']}: {run['refused']} was refused, Rust took it")
    check_chain(run["chain"], chain, run["name"])
    return run["states"]


# ---------------------------------------------------------------------------
# Chains
# ---------------------------------------------------------------------------


@dataclass
class Chain:
    """A digest chain with checkpoints, as golden v2 keeps one per tape."""

    digest: str = ""
    records: int = 0
    checkpoints: list[str] = field(default_factory=list)

    def push(self, kind: str, value: Any) -> None:
        self.digest = canon.chain(self.digest, canon.digest([kind, value]))
        self.records += 1
        if self.records % CHECKPOINT_EVERY == 0:
            self.checkpoints.append(short(self.digest))

    def as_json(self) -> dict[str, Any]:
        return {"digest": self.digest, "records": self.records, "checkpoints": self.checkpoints}


def first_divergence(expected: dict[str, Any], got: Chain) -> str:
    """Where a chain first departs from its frozen record."""
    for i, (a, b) in enumerate(zip(expected["checkpoints"], got.checkpoints, strict=False)):
        if a != b:
            lo = i * CHECKPOINT_EVERY
            return f"records {lo + 1}..{lo + CHECKPOINT_EVERY}"
    return f"after record {len(got.checkpoints) * CHECKPOINT_EVERY} (records {got.records})"


def check_chain(expected: dict[str, Any], got: Chain, where: str) -> None:
    if got.digest != expected["digest"] or got.records != expected["records"]:
        at = first_divergence(expected, got)
        raise AssertionError(f"{where}: differs from the frozen record at {at}")


# ---------------------------------------------------------------------------
# Tapes
# ---------------------------------------------------------------------------


def new_game(start: dict[str, Any], setup: Sequence[dict[str, Any]] = ()) -> Game:
    """A new run with parity setup ops applied natively (``Game.apply_setup``)."""
    game = Game.new(start["back"], start["stake"], start["seed"], start.get("challenge"))
    for op in setup:
        game.apply_setup(op)
    return game


def transplant(game: Game) -> Game:
    """A run rebuilt from ``game``'s snapshot (the snapshot's JSON text, no Python decode)."""
    return Game(_core.Game.from_snapshot(game._g.snapshot()))


def replay_tape(case: dict[str, Any], *, transplanted: bool = True) -> Iterator[tuple[int, Game]]:
    """Play a frozen tape on the Rust engine, checking every step.

    ``case`` holds ``start``, ``setup``, ``tape`` and ``digests`` (the state after each step),
    and optionally ``setup_digest`` (the state before the first step), ``legal`` (each state's
    legal-action digest, before its step) and ``refused`` (the tape's last action, which both
    engines refused). With ``transplanted`` each step is also taken by a run rebuilt from the
    state's snapshot, the way the oracle's states were stepped in Rust while it existed. Yields
    ``(position, game)`` before each step and once more at the end.
    """
    name = case.get("name", case["start"]["seed"])
    game = new_game(case["start"], case["setup"])
    if "setup_digest" in case:
        assert short(game.state_hash()) == case["setup_digest"], f"{name}: setup differs"
    legal = case.get("legal")
    for pos, (action, digest) in enumerate(zip(case["tape"], case["digests"], strict=True)):
        yield pos, game
        where = f"{name}@{pos} {action}"
        if legal is not None:
            got = legal_digest(json.loads(game._g.legal_actions()))
            assert got == legal[pos], f"{where}: legal actions differ"
        if transplanted:
            copy = transplant(game)
            copy.step(action)
            assert short(copy.state_hash()) == digest, f"{where}: the rebuilt run's step differs"
        game.step(action)
        assert short(game.state_hash()) == digest, f"{where}: state differs"
    refused = case.get("refused")
    if refused is not None:
        try:
            game.step(refused)
        except IllegalActionError:
            pass
        else:
            raise AssertionError(f"{name}: {refused} was refused, Rust took it")
    yield len(case["tape"]), game

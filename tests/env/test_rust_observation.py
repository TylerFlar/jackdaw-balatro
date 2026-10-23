"""The Rust observation encoder matches ``encode_observation`` bit for bit.

``jackdaw._core`` ports :mod:`jackdaw.env.observation` (encoder v1) to Rust.
For every state along every parity tape, including the state right after the
tape's setup, the Python encoder reads the live oracle state and the Rust
encoder reads the oracle's canonical snapshot: every array must have the same
shape, dtype and bytes (a NaN matches any NaN).  Where the Rust engine plays a
tape itself and stays in step with the oracle (same state hash), its own run's
observation must match as well.  Seeded fuzz runs, targeted setups and edited
snapshots cover what the tapes don't reach.
"""

from __future__ import annotations

import copy
import random
from typing import Any

import numpy as np
import pytest

from jackdaw.core import Game, IllegalActionError, NotPortedError
from jackdaw.engine import actions as A
from jackdaw.engine.actions import get_legal_actions
from jackdaw.env import rust_observation
from jackdaw.env.observation import (
    D_CONSUMABLE,
    D_GLOBAL,
    D_JOKER,
    D_PLAYING_CARD,
    D_SHOP,
    Observation,
    encode_observation,
)
from tests.golden.trajectories import _BACKS, _CHALLENGES, _mixed_policy, encode_action
from tests.parity import canon
from tests.parity.harness import load_specs
from tests.parity.pyengine import PyEngine

_SPECS = load_specs()
_FIELDS = ("global_context", "hand_cards", "jokers", "consumables", "shop_cards", "pack_cards")


def assert_same(py: Observation, rs: Observation, where: str) -> None:
    """Same arrays: dtype, shape and bits (any NaN matches any NaN)."""
    for name in _FIELDS:
        a = getattr(py, name)
        b = getattr(rs, name)
        assert a.dtype == b.dtype == np.float32, f"{where} {name}: dtype {a.dtype} vs {b.dtype}"
        assert a.shape == b.shape, f"{where} {name}: shape {a.shape} vs {b.shape}"
        if a.tobytes() == b.tobytes():
            continue
        same = (a.view(np.uint32) == b.view(np.uint32)) | (np.isnan(a) & np.isnan(b))
        bad = np.argwhere(~same)
        if bad.size:
            shown = ", ".join(
                f"{tuple(int(x) for x in i)}: py {a[tuple(i)]!r} rust {b[tuple(i)]!r}"
                for i in bad[:5]
            )
            raise AssertionError(f"{where} {name}: {len(bad)} values differ: {shown}")


def _check(engine: PyEngine, where: str) -> tuple[dict[str, Any], Observation]:
    """Compare the encoders on the oracle's state; return its snapshot and observation."""
    snap = engine.snapshot()
    py = encode_observation(engine.gs)
    assert_same(py, rust_observation.encode_snapshot(snap), where)
    return snap, py


def _done(snap: dict[str, Any]) -> bool:
    return snap["phase"] == "game_over" or snap["run"]["won"]


# ---------------------------------------------------------------------------
# Tapes
# ---------------------------------------------------------------------------


def _rust_run(spec: Any, engine: PyEngine) -> Game | None:
    """The Rust engine's own run at the tape's start."""
    if spec.setup:
        return Game.from_dict(engine.snapshot())
    s = spec.start
    try:
        return Game.new(s["back"], s["stake"], s["seed"], s.get("challenge"))
    except (ValueError, NotPortedError):
        return None


@pytest.mark.parametrize("spec", _SPECS, ids=[s.name for s in _SPECS])
def test_tape_states_match(spec: Any) -> None:
    engine = PyEngine(spec.start, spec.setup)
    game = _rust_run(spec, engine)
    for pos in range(len(spec.tape) + 1):
        where = f"{spec.name}@{pos}"
        snap, py = _check(engine, where)
        # The Rust run counts only while it is in the oracle's state; the lockstep tests own
        # engine parity.
        if game is not None and game.state_hash() == canon.digest(snap):
            assert_same(py, rust_observation.encode_game(game), f"{where} (rust run)")
        else:
            game = None
        if _done(snap) or pos == len(spec.tape):
            return
        action = spec.tape[pos]
        try:
            engine.step(action)
        except Exception:  # the oracle refuses the action: the tape ends here
            return
        if game is not None:
            try:
                game.step(action)
            except (IllegalActionError, NotPortedError):
                game = None


# ---------------------------------------------------------------------------
# Seeded fuzz runs and targeted setups
# ---------------------------------------------------------------------------


def _play(
    engine: PyEngine,
    seed: str,
    steps: int,
    *,
    eps: float = 0.2,
    skip_rate: float = 0.0,
    boss: str | None = None,
) -> int:
    """Play the mixed policy, comparing the encoders at every state; return states checked.

    ``skip_rate`` skips blinds that often; ``boss`` is forced at the first blind select.
    """
    rng = random.Random(seed)
    for pos in range(steps):
        snap, _ = _check(engine, f"{seed}@{pos}")
        if _done(snap):
            return pos + 1
        if boss is not None and snap["phase"] == "blind_select":
            engine.apply({"op": "force_boss", "key": boss})
            boss = None
            continue
        with engine._ids():
            legal = get_legal_actions(engine.gs)
            if not legal:
                return pos + 1
            skips = [a for a in legal if isinstance(a, A.SkipBlind)]
            if skips and rng.random() < skip_rate:
                action = encode_action(skips[0])
            else:
                action = encode_action(_mixed_policy(engine.gs, legal, rng, eps))
        try:
            engine.step(action)
        except Exception:  # an action the oracle refuses ends the run
            return pos + 1
    return steps


@pytest.mark.parametrize("i", range(24))
def test_fuzz_states_match(i: int) -> None:
    challenge = _CHALLENGES[(i // 4) % len(_CHALLENGES)] if i % 4 == 3 else None
    start = {
        "back": "b_challenge" if challenge else _BACKS[i % len(_BACKS)],
        "stake": 1 + (i * 5) % 8,
        "seed": f"OBS_FUZZ{i}",
        "challenge": challenge,
    }
    eps = (0.0, 0.05, 0.2)[i % 3]
    _play(PyEngine(start), start["seed"], steps=300, eps=eps, skip_rate=(0.0, 0.2)[i % 2])


_SELECT = {"op": "action", "action": ["SelectBlind"]}

_SCENARIOS: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]] = {
    # Five-card hands with every detection joker: flushes and straights reach the hand types.
    "detection_jokers": (
        [_SELECT,
         *({"op": "add", "key": k} for k in
           ("j_stuntman", "j_merry_andy", "j_four_fingers", "j_shortcut", "j_smeared",
            "j_splash", "j_pareidolia")),
         {"op": "add", "key": "H_A", "enhancement": "m_wild"},
         {"op": "add", "key": "S_5", "enhancement": "m_stone"},
         {"op": "add", "key": "D_9", "enhancement": "m_wild", "edition": "holo"}],
        {"eps": 0.5},
    ),
    # Every consumable gate: hand-size spectrals, slot creators, Aura on editioned cards.
    "consumable_gates": (
        [_SELECT,
         *({"op": "add", "key": k} for k in
           ("c_aura", "c_ouija", "c_sigil", "c_familiar", "c_grim", "c_incantation",
            "c_immolate", "c_cryptid", "c_talisman", "c_deja_vu", "c_trance", "c_medium",
            "c_black_hole", "c_soul", "c_ectoplasm", "c_hex", "c_wraith", "c_ankh", "c_fool",
            "c_emperor", "c_high_priestess", "c_judgement", "c_wheel_of_fortune", "c_death")),
         {"op": "add", "key": "H_K", "edition": "foil"},
         {"op": "add", "key": "D_Q", "edition": "polychrome", "seal": "Red"},
         {"op": "add", "key": "j_joker", "edition": "negative"}],
        {"eps": 0.6},
    ),
    # The jokers whose Python values skip Card:update, with blinds skipped (Throwback).
    "per_frame_jokers": (
        [_SELECT, *({"op": "add", "key": k} for k in
                    ("j_throwback", "j_stencil", "j_swashbuckler", "j_egg"))],
        {"eps": 0.2, "skip_rate": 0.7},
    ),
    # Amber Acorn flips the jokers face down.
    "amber_acorn": (
        [_SELECT, *({"op": "add", "key": k} for k in ("j_joker", "j_stencil", "j_misprint"))],
        {"eps": 0.2, "boss": "bl_final_acorn"},
    ),
    # Suit and face debuffs, then a boss that turns cards face down.
    "plant": ([{"op": "force_boss", "key": "bl_plant"}, _SELECT], {"eps": 0.3}),
    "house": ([{"op": "force_boss", "key": "bl_house"}, _SELECT], {"eps": 0.3}),
}  # fmt: skip


@pytest.mark.parametrize("name", sorted(_SCENARIOS))
def test_scenario_states_match(name: str) -> None:
    setup, play = _SCENARIOS[name]
    for k, seed in enumerate(("OBSCEN1", "OBSCEN2")):
        start = {"back": "b_red", "stake": 1 + 3 * k, "seed": seed + name, "challenge": None}
        engine = PyEngine(start, setup)
        _play(engine, f"{seed}{name}", steps=150, **play)


# ---------------------------------------------------------------------------
# Edited snapshots: rare states, built directly
# ---------------------------------------------------------------------------

# Python keeps these jokers' set_ability values in the live state (no Card:update), which the
# Rust encoder reproduces; a Python state rebuilt from a snapshot holds the per-frame values
# instead, so the edited-snapshot tests leave them out.
_PER_FRAME = {"j_throwback", "j_stencil", "j_swashbuckler"}


def _tape_snapshots(phase: str, count: int) -> list[dict[str, Any]]:
    """Snapshots in ``phase`` along the tapes, a few per tape, without per-frame jokers."""
    out: list[dict[str, Any]] = []
    for spec in _SPECS:
        engine = PyEngine(spec.start, spec.setup)
        taken = 0
        for action in spec.tape:
            snap = engine.snapshot()
            jokers = {c["center"] for c in snap["areas"]["jokers"]}
            if snap["phase"] == phase and not jokers & _PER_FRAME and taken < 2:
                out.append(snap)
                taken += 1
                if len(out) >= count:
                    return out
            if _done(snap):
                break
            try:
                engine.step(action)
            except Exception:
                break
    return out


def _check_snapshot(snap: dict[str, Any], where: str) -> None:
    engine = PyEngine.from_snapshot(snap)
    assert_same(encode_observation(engine.gs), rust_observation.encode_snapshot(snap), where)


def _edits(snap: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    out = []
    s = copy.deepcopy(snap)
    s["blind"]["chips"] = float("nan")
    s["game"]["chips"] = float("inf")
    s["game"]["dollars"] = 10**6
    out.append(("overflow", s))
    s = copy.deepcopy(snap)
    s["game"]["dollars"] = -7
    s["game"]["interest_cap"] = 0
    out.append(("debt", s))
    s = copy.deepcopy(snap)
    for c in s["areas"]["jokers"]:
        c["facing"] = "back"
    for c in s["areas"]["hand"][::2]:
        c["facing"] = "back"
    out.append(("face_down", s))
    s = copy.deepcopy(snap)
    for c in s["areas"]["hand"]:
        c["edition"] = {"foil": True, "chips": 50, "type": "foil"}
    for c in s["areas"]["jokers"]:
        c["debuff"] = True
        c["stickers"].update(perishable=True, perish_tally=2, rental=True)
    out.append(("editions_debuffs", s))
    return out


@pytest.mark.parametrize("phase", ["selecting_hand", "shop", "pack_opening"])
def test_edited_snapshots_match(phase: str) -> None:
    snaps = _tape_snapshots(phase, count=12)
    assert snaps
    for n, snap in enumerate(snaps):
        _check_snapshot(snap, f"{phase}#{n}")
        if snap["blind"] is None:
            continue
        for name, edited in _edits(snap):
            _check_snapshot(edited, f"{phase}#{n} {name}")


# Random boards written into real snapshots: hands built to form groups, flushes and
# straights (of every size, around the five-card limit), every detection joker, consumable
# gates at their limits, and shop slots.

_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
_RANK_LETTERS = dict(zip(_RANKS, "23456789TJQKA", strict=True))
_SUITS = ["Hearts", "Diamonds", "Clubs", "Spades"]
_ENHANCEMENTS = ["c_base"] * 8 + [
    "m_wild", "m_wild", "m_stone", "m_bonus", "m_mult", "m_glass", "m_steel", "m_gold", "m_lucky",
]  # fmt: skip
_EDITIONS = ["foil", "holo", "polychrome", "negative"]
_DETECTION = ["j_four_fingers", "j_shortcut", "j_smeared", "j_splash", "j_pareidolia"]


def _random_hand_spec(rng: random.Random, n: int) -> list[tuple[str, str]]:
    mode = rng.choice(["groups", "flush", "straight", "uniform", "suited_groups"])
    if mode == "suited_groups":  # flush houses and flush fives
        suit = rng.choice(_SUITS)
        ranks = rng.sample(_RANKS, rng.randint(1, 2))
        return [(suit, rng.choice(ranks)) for _ in range(n)]
    if mode == "groups":
        ranks = rng.sample(_RANKS, rng.randint(1, 3))
        return [(rng.choice(_SUITS), rng.choice(ranks)) for _ in range(n)]
    if mode == "flush":
        suits = rng.sample(_SUITS, rng.randint(1, 2))
        return [(rng.choice(suits), rng.choice(_RANKS)) for _ in range(n)]
    if mode == "straight":
        start = rng.randint(-1, 9)  # -1: Ace low
        seq = [_RANKS[(start + k) % 13] for k in range(6) if rng.random() > 0.15]
        suits = rng.sample(_SUITS, rng.randint(1, 4))
        return [(rng.choice(suits), seq[k % len(seq)] if seq else "Ace") for k in range(n)]
    return [(rng.choice(_SUITS), rng.choice(_RANKS)) for _ in range(n)]


class _Cards:
    """Fresh cards with explicit sort ids, exported as snapshot cards."""

    def __init__(self, rng: random.Random, next_id: int) -> None:
        self.rng = rng
        self.next_id = next_id

    def _new(self) -> Any:
        from jackdaw.engine.card import Card

        self.next_id += 1
        return Card(sort_id=self.next_id - 1)

    def playing(self, suit: str, rank: str) -> dict[str, Any]:
        from tests.parity.py_export import export_card

        rng = self.rng
        c = self._new()
        c.set_base(f"{suit[0]}_{_RANK_LETTERS[rank]}", suit, rank)
        c.set_ability(rng.choice(_ENHANCEMENTS))
        if rng.random() < 0.25:
            c.set_edition({rng.choice(_EDITIONS): True})
        if rng.random() < 0.2:
            c.set_seal(rng.choice(["Gold", "Red", "Blue", "Purple"]))
        c.debuff = rng.random() < 0.15
        c.facing = "back" if rng.random() < 0.08 else "front"
        c.ability["perma_bonus"] = rng.choice([0, 0, 0, 5, 10, 25])
        c.base.times_played = rng.choice([0, 0, 1, 3, 12])
        return export_card(c)

    def other(self, key: str, *, cost: int | None = None) -> dict[str, Any]:
        from tests.parity.py_export import export_card

        rng = self.rng
        c = self._new()
        c.set_ability(key)
        c.set_cost()
        if c.ability.get("set") == "Joker":
            if rng.random() < 0.3:
                c.set_edition({rng.choice(_EDITIONS): True})
                c.set_cost()
            c.debuff = rng.random() < 0.15
            c.facing = "back" if rng.random() < 0.05 else "front"
            c.eternal = rng.random() < 0.1
            if not c.eternal and rng.random() < 0.15:
                c.perishable = True
                c.perish_tally = rng.randint(0, 5)
            c.rental = rng.random() < 0.1
            c.ability["mult"] = rng.choice([c.ability["mult"], 4, 7.5])
            c.ability["x_mult"] = rng.choice([c.ability["x_mult"], 1.5, 2.25, 0.1 * 7])
            c.ability["t_chips"] = rng.choice([c.ability["t_chips"], 30])
            extra = c.ability.get("extra")
            if isinstance(extra, (int, float)) and not isinstance(extra, bool):
                c.ability["extra"] = rng.choice([extra, 0.1, 0.7, -2, 1e6])
            elif isinstance(extra, dict):
                for k, v in extra.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        extra[k] = rng.choice([v, 0.1, 0.2, 0.3, -1, 1e16])
        if cost is not None:
            c.cost = cost
        c.sell_cost = rng.choice([c.sell_cost, 0, 1, 13])
        return export_card(c)


def _random_state(base: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    from jackdaw.engine.data.prototypes import (
        BOOSTERS,
        JOKERS,
        PLANETS,
        SPECTRALS,
        TAROTS,
        VOUCHERS,
    )

    snap = copy.deepcopy(base)
    cards = _Cards(rng, snap["sim"]["next_sort_id"])
    areas = snap["areas"]
    n = rng.choice([0, 1, 2, 3, 4, 4, 5, 5, 5, 5, 6, 7, 8, 8, 9, 11, 14])
    areas["hand"] = [cards.playing(s, r) for s, r in _random_hand_spec(rng, n)]
    jokers = [k for k in _DETECTION if rng.random() < 0.4]
    other_jokers = sorted(set(JOKERS) - _PER_FRAME - set(_DETECTION))
    jokers += rng.sample(other_jokers, rng.randint(0, 4))
    areas["jokers"] = [cards.other(k) for k in jokers]
    consumables = sorted(set(TAROTS) | set(PLANETS) | set(SPECTRALS))
    areas["consumables"] = [cards.other(k) for k in rng.sample(consumables, rng.randint(0, 4))]
    shop = sorted(set(JOKERS) - _PER_FRAME) + consumables
    areas["shop_jokers"] = [
        cards.other(k, cost=rng.choice([0, 3, 8, 30])) for k in rng.sample(shop, rng.randint(0, 3))
    ]
    areas["shop_vouchers"] = [
        cards.other(k) for k in rng.sample(sorted(VOUCHERS), rng.randint(0, 2))
    ]
    areas["shop_booster"] = [
        cards.other(k) for k in rng.sample(sorted(BOOSTERS), rng.randint(0, 2))
    ]
    pack_n = rng.choice([0, 0, 2, 3, 5])
    areas["pack_cards"] = [cards.playing(s, r) for s, r in _random_hand_spec(rng, pack_n)]
    if snap["pack"] is not None:
        snap["pack"]["dealt"] = []
    snap["sim"]["next_sort_id"] = cards.next_id
    game = snap["game"]
    game["dollars"] = rng.choice([-20, -3, 0, 4, 5, 24, 25, 26, 99, 1234])
    game["bankrupt_at"] = rng.choice([0, 0, -20])
    game["joker_slots"] = rng.choice([len(jokers), len(jokers) + 1, 5, 1])
    game["consumable_slots"] = rng.choice([len(areas["consumables"]), 1, 2, 3])
    game["interest_cap"] = rng.choice([25, 50, 100, 0])
    game["last_tarot_planet"] = rng.choice([None, "c_fool", "c_pluto", "c_magician"])
    snap["current_round"]["hands_left"] = rng.choice([0, 1, 4])
    for h in snap["hands"].values():
        h["level"] = rng.choice([h["level"], 0, 1, 2, 7])
    return snap


def test_random_states_match() -> None:
    bases = [s for s in _tape_snapshots("selecting_hand", 6) if s["blind"] is not None]
    bases += _tape_snapshots("shop", 3) + _tape_snapshots("pack_opening", 3)
    rng = random.Random("OBS_RANDOM_STATES")
    for n in range(400):
        snap = _random_state(bases[n % len(bases)], rng)
        _check_snapshot(snap, f"random#{n}")


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------


def test_wrapper_arrays() -> None:
    game = Game.new("b_red", 1, "OBS_WRAP")
    game.step(["SelectBlind"])
    obs = rust_observation.encode_game(game)
    assert obs.global_context.shape == (D_GLOBAL,)
    widths = {
        "hand_cards": D_PLAYING_CARD,
        "jokers": D_JOKER,
        "consumables": D_CONSUMABLE,
        "shop_cards": D_SHOP,
        "pack_cards": D_PLAYING_CARD,
    }
    for name, width in widths.items():
        arr = getattr(obs, name)
        assert arr.dtype == np.float32 and arr.ndim == 2 and arr.shape[1] == width
        assert arr.flags.writeable and arr.flags.c_contiguous
    assert obs.hand_cards.shape[0] == 8
    assert obs.pack_cards.shape == (0, D_PLAYING_CARD)
    # Each call returns fresh arrays; the encoder's buffers stay inside the game.
    again = rust_observation.encode_game(game)
    obs.hand_cards[:] = -1.0
    assert not np.array_equal(again.hand_cards, obs.hand_cards)
    assert_same(again, rust_observation.encode_snapshot(game.to_dict()), "wrapper")
    gobs = again.to_game_observation()
    assert gobs.entities["hand_card"] is again.hand_cards

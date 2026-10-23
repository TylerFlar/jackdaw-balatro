"""The Rust observation encoder (encoder v1), against the Python encoder's frozen outputs.

``jackdaw._core`` ports the Python observation encoder bit for bit. That encoder read the
oracle's state; its outputs are kept as digests (:mod:`tests.parity.frozen`). The parity
tapes' states are checked in ``test_golden_env.py``. Here: seeded fuzz runs and scenario boards
(``tests/fixtures/env_fuzz.json``), and edited and random snapshots
(``tests/fixtures/env_states.json``), which reach states the tapes don't.
"""

from __future__ import annotations

import copy
import json
import random
from typing import Any

import numpy as np
import pytest

from jackdaw.core import Game, IllegalActionError
from jackdaw.env import rust_observation
from jackdaw.env.observation import (
    D_CONSUMABLE,
    D_GLOBAL,
    D_JOKER,
    D_PLAYING_CARD,
    D_SHOP,
    Observation,
)
from tests.parity import frozen
from tests.parity.harness import load_specs

_SPECS = load_specs()
_FIELDS = ("global_context", "hand_cards", "jokers", "consumables", "shop_cards", "pack_cards")
_FUZZ = frozen.load("env_fuzz.json")


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
                f"{tuple(int(x) for x in i)}: {a[tuple(i)]!r} vs {b[tuple(i)]!r}" for i in bad[:5]
            )
            raise AssertionError(f"{where} {name}: {len(bad)} values differ: {shown}")


def snapshot_digest(snap: dict[str, Any]) -> str:
    """The observation digest of the state a snapshot describes, encoded in Rust."""
    return frozen.obs_digest(rust_observation.encode_snapshot(snap))


# ---------------------------------------------------------------------------
# Seeded fuzz runs and scenario boards
# ---------------------------------------------------------------------------

_SELECT = {"op": "action", "action": ["SelectBlind"]}

# Each board is played from two seeds (stakes 1 and 4) by the golden mixed policy with the
# given options: ``eps`` its exploration rate, ``skip_rate`` how often it skips a blind and
# ``boss`` a boss forced at the first blind select.
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
    # The jokers whose values come from Card:update, with blinds skipped (Throwback).
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


@pytest.mark.parametrize("run", _FUZZ["observation"], ids=[r["name"] for r in _FUZZ["observation"]])
def test_fuzz_states_match(run: dict[str, Any]) -> None:
    frozen.replay_env_run(run)


@pytest.mark.parametrize("name", sorted(_SCENARIOS))
def test_scenario_states_match(name: str) -> None:
    runs = [r for r in _FUZZ["scenarios"] if r["name"].endswith(name)]
    assert len(runs) == 2
    for run in runs:
        frozen.replay_env_run(run)


# ---------------------------------------------------------------------------
# Edited snapshots: rare states, built directly
# ---------------------------------------------------------------------------

# The jokers whose values come from Card:update. The Python encoder read the oracle's live
# values for them, which a state rebuilt from a snapshot doesn't hold, so the snapshot tests
# leave them out.
_PER_FRAME = {"j_throwback", "j_stencil", "j_swashbuckler"}


def tape_snapshots(phase: str, count: int) -> list[dict[str, Any]]:
    """Snapshots in ``phase`` along the tapes, a few per tape, without per-frame jokers."""
    out: list[dict[str, Any]] = []
    for spec in _SPECS:
        game = frozen.new_game(spec.start, spec.setup)
        taken = 0
        for action in spec.tape:
            if game.phase == phase and taken < 2 and not {j.key for j in game.jokers} & _PER_FRAME:
                out.append(game.to_dict())
                taken += 1
                if len(out) >= count:
                    return out
            if game.phase == "game_over" or game.won:
                break
            try:
                game.step(action)
            except IllegalActionError:
                break
    return out


def snapshot_edits(snap: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
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


def edited_snapshots(phase: str) -> list[tuple[str, dict[str, Any]]]:
    """The tape snapshots in ``phase`` and their edits, labelled."""
    out = []
    for n, snap in enumerate(tape_snapshots(phase, count=12)):
        out.append((f"{phase}#{n}", snap))
        if snap["blind"] is not None:
            out += [(f"{phase}#{n} {name}", s) for name, s in snapshot_edits(snap)]
    return out


@pytest.mark.parametrize("phase", ["selecting_hand", "shop", "pack_opening"])
def test_edited_snapshots_match(phase: str) -> None:
    states = edited_snapshots(phase)
    expected = frozen.load("env_states.json")["edited"][phase]
    assert len(states) == len(expected)
    for (where, snap), digest in zip(states, expected, strict=True):
        assert snapshot_digest(snap) == digest, where


# Random boards written into real snapshots: hands built to form groups, flushes and
# straights (of every size, around the five-card limit), every detection joker, consumable
# gates at their limits, and shop slots. The cards are made by BalatroBot's add in a scratch
# run, then edited.

_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace"]
_RANK_LETTERS = dict(zip(_RANKS, "23456789TJQKA", strict=True))
_SUITS = ["Hearts", "Diamonds", "Clubs", "Spades"]
_ENHANCEMENTS = ["c_base"] * 8 + [
    "m_wild", "m_wild", "m_stone", "m_bonus", "m_mult", "m_glass", "m_steel", "m_gold", "m_lucky",
]  # fmt: skip
_EDITIONS = ["foil", "holo", "polychrome", "negative"]
_DETECTION = ["j_four_fingers", "j_shortcut", "j_smeared", "j_splash", "j_pareidolia"]


def _centers(*sets: str) -> list[str]:
    data = json.loads((frozen.ROOT / "jackdaw/engine/data/centers.json").read_text("utf-8"))
    return sorted(k for k, v in data.items() if v.get("set") in sets)


_JOKERS = _centers("Joker")
_CONSUMABLES = _centers("Tarot", "Planet", "Spectral")
_VOUCHERS = _centers("Voucher")
_BOOSTERS = _centers("Booster")
_AREA_OF = {"j_": "jokers", "c_": "consumables", "v_": "shop_vouchers", "p_": "shop_booster"}


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
    """Fresh cards with explicit sort ids, made by BalatroBot's ``add`` in a scratch run."""

    def __init__(self, rng: random.Random, next_id: int, factory: Game) -> None:
        self.rng = rng
        self.next_id = next_id
        self.factory = factory

    def _make(self, key: str, **params: Any) -> dict[str, Any]:
        game = self.factory.clone()
        sort_id = game.debug_add(key, **params)
        areas = [_AREA_OF[key[:2]]] if key[:2] in _AREA_OF else ["hand", "deck"]
        card = next(
            c for area in areas for c in json.loads(game._g.view(area)) if c["sort_id"] == sort_id
        )
        card["sort_id"] = self.next_id
        self.next_id += 1
        return card

    def playing(self, suit: str, rank: str) -> dict[str, Any]:
        rng = self.rng
        params: dict[str, Any] = {"enhancement": rng.choice(_ENHANCEMENTS)}
        if rng.random() < 0.25:
            params["edition"] = rng.choice(_EDITIONS)
        if rng.random() < 0.2:
            params["seal"] = rng.choice(["Gold", "Red", "Blue", "Purple"])
        c = self._make(f"{suit[0]}_{_RANK_LETTERS[rank]}", **params)
        c["added_to_deck"] = False
        c["debuff"] = rng.random() < 0.15
        c["facing"] = "back" if rng.random() < 0.08 else "front"
        c["ability"]["perma_bonus"] = rng.choice([0, 0, 0, 5, 10, 25])
        c["base"]["times_played"] = rng.choice([0, 0, 1, 3, 12])
        return c

    def other(self, key: str, *, owned: bool, cost: int | None = None) -> dict[str, Any]:
        rng = self.rng
        joker = key.startswith("j_")
        params: dict[str, Any] = {}
        if joker:
            if rng.random() < 0.3:
                params["edition"] = rng.choice(_EDITIONS)
            params["eternal"] = rng.random() < 0.1
            if not params["eternal"] and rng.random() < 0.15:
                params["perishable"] = rng.randint(0, 5)
            params["rental"] = rng.random() < 0.1
        c = self._make(key, **params)
        c["added_to_deck"] = owned
        if joker:
            c["debuff"] = rng.random() < 0.15
            c["facing"] = "back" if rng.random() < 0.05 else "front"
            a = c["ability"]
            a["mult"] = rng.choice([a["mult"], 4, 7.5])
            a["x_mult"] = rng.choice([a["x_mult"], 1.5, 2.25, 0.1 * 7])
            a["t_chips"] = rng.choice([a["t_chips"], 30])
            extra = a.get("extra")
            if isinstance(extra, (int, float)) and not isinstance(extra, bool):
                a["extra"] = rng.choice([extra, 0.1, 0.7, -2, 1e6])
            elif isinstance(extra, dict):
                for k, v in extra.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        extra[k] = rng.choice([v, 0.1, 0.2, 0.3, -1, 1e16])
        costs = c["costs"]
        if cost is not None:
            costs["cost"] = cost
        costs["sell_cost"] = rng.choice([costs["sell_cost"], 0, 1, 13])
        return c


def random_state(base: dict[str, Any], rng: random.Random, factory: Game) -> dict[str, Any]:
    snap = copy.deepcopy(base)
    cards = _Cards(rng, snap["sim"]["next_sort_id"], factory)
    areas = snap["areas"]
    n = rng.choice([0, 1, 2, 3, 4, 4, 5, 5, 5, 5, 6, 7, 8, 8, 9, 11, 14])
    areas["hand"] = [cards.playing(s, r) for s, r in _random_hand_spec(rng, n)]
    jokers = [k for k in _DETECTION if rng.random() < 0.4]
    other_jokers = sorted(set(_JOKERS) - _PER_FRAME - set(_DETECTION))
    jokers += rng.sample(other_jokers, rng.randint(0, 4))
    areas["jokers"] = [cards.other(k, owned=True) for k in jokers]
    areas["consumables"] = [
        cards.other(k, owned=True) for k in rng.sample(_CONSUMABLES, rng.randint(0, 4))
    ]
    shop = sorted(set(_JOKERS) - _PER_FRAME) + _CONSUMABLES
    areas["shop_jokers"] = [
        cards.other(k, owned=False, cost=rng.choice([0, 3, 8, 30]))
        for k in rng.sample(shop, rng.randint(0, 3))
    ]
    areas["shop_vouchers"] = [
        cards.other(k, owned=False) for k in rng.sample(_VOUCHERS, rng.randint(0, 2))
    ]
    areas["shop_booster"] = [
        cards.other(k, owned=False) for k in rng.sample(_BOOSTERS, rng.randint(0, 2))
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


def random_states(count: int = 400) -> list[dict[str, Any]]:
    bases = [s for s in tape_snapshots("selecting_hand", 6) if s["blind"] is not None]
    bases += tape_snapshots("shop", 3) + tape_snapshots("pack_opening", 3)
    factory = Game.new("b_red", 1, "OBS_CARDS")
    factory.step(["SelectBlind"])
    rng = random.Random("OBS_RANDOM_STATES")
    return [random_state(bases[n % len(bases)], rng, factory) for n in range(count)]


def test_random_states_match() -> None:
    expected = frozen.load("env_states.json")["random"]
    states = random_states(len(expected))
    for n, (snap, digest) in enumerate(zip(states, expected, strict=True)):
        assert snapshot_digest(snap) == digest, f"random#{n}"


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

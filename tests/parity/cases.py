"""Targeted parity cases on the Rust engine, checked against the oracle's frozen verdicts.

The targeted parity tests (``test_jokers_batch_*``, ``test_consumables_rust``,
``test_audit_*``) build a state, take a few steps and assert on the result. While the Python
oracle existed, each of those steps was also taken by the oracle and the two snapshots were
compared. ``tests/fixtures/parity_cases.json`` keeps the verdicts: for every run a test builds,
in order, each step's action, the canonical digest of the state before and after it (``null``
after a step both engines refused) and the digests of the legal-action lists before and after
it, plus the legal-action checks some tests make between steps (:mod:`tests.parity.frozen`).

A :class:`Run` builds its state with the Rust API: ``Game.apply_setup`` (BalatroBot's ``add``
and ``set``, a forced boss, actions) and edits of the canonical snapshot for values the tests
write directly. Each step must start from the oracle's state and reach the oracle's next state,
playing on and from a snapshot. A few tests built their state with oracle internals the Rust API
has no counterpart for (a joker added without BalatroBot's edition poll, an enhancement set in
place, a synthesized booster pack); they call :meth:`Run.oracle`, and the fixture holds the
difference between the state built natively and the oracle's, applied before the next check.

``JACKDAW_REFREEZE_CASES=1`` rewrites the fixture's digests from the Rust engine instead of
checking them, keeping each run's actions and differences (``scripts/refreeze.py cases``).
"""

from __future__ import annotations

import atexit
import copy
import json
import os
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from jackdaw.core import CardView, Game, IllegalActionError
from jackdaw.engine.data import load
from tests.parity import canon, frozen

FIXTURE = "parity_cases.json"
_REFREEZE = bool(os.environ.get("JACKDAW_REFREEZE_CASES"))

_frozen: dict[str, list[list[dict[str, Any]]]] | None = None
_refrozen: dict[str, list[list[dict[str, Any]]]] = {}
_runs: dict[str, list[Run]] = {}


def _current_test() -> str:
    return os.environ.get("PYTEST_CURRENT_TEST", "").rsplit(" (", 1)[0]


def _recorded(test: str, index: int) -> list[dict[str, Any]]:
    global _frozen
    if _frozen is None:
        _frozen = frozen.load(FIXTURE)
    runs = _frozen.get(test)
    assert runs is not None and index < len(runs), f"{test}: no frozen run #{index}"
    return runs[index]


def finish_test(test: str) -> list[Run]:
    """The runs a test built (``tests/parity/conftest.py`` checks them afterwards)."""
    return _runs.pop(test, [])


def _write_refrozen() -> None:
    """Merge the runs refrozen in this session into the fixture."""
    data = frozen.load(FIXTURE)
    data.update(_refrozen)
    frozen.write(FIXTURE, {test: data[test] for test in sorted(data)})


if _REFREEZE:
    atexit.register(_write_refrozen)


# ---------------------------------------------------------------------------
# Snapshot patches
# ---------------------------------------------------------------------------


def apply_patch(snap: dict[str, Any], patch: Sequence[Sequence[Any]]) -> dict[str, Any]:
    """Apply a frozen difference: ``[path, value]`` sets and ``[path]`` deletions."""
    for edit in patch:
        path = edit[0]
        node: Any = snap
        for key in path[:-1]:
            node = node[key]
        if len(edit) == 1:
            del node[path[-1]]
        elif not path:
            return canon.from_json_compatible(edit[1])
        else:
            node[path[-1]] = canon.from_json_compatible(edit[1])
    return snap


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------

AREAS = (
    "hand",
    "deck",
    "discard",
    "play",
    "jokers",
    "consumables",
    "shop_jokers",
    "shop_vouchers",
    "shop_booster",
    "pack_cards",
)
_RANK_LETTER = {"10": "T", "Jack": "J", "Queen": "Q", "King": "K", "Ace": "A"}


class CardRef:
    """A card of a run by its sort id: each attribute reads the run's current state."""

    __slots__ = ("run", "sort_id")

    def __init__(self, run: Run, sort_id: int) -> None:
        self.run = run
        self.sort_id = sort_id

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CardRef) and other.sort_id == self.sort_id

    def __hash__(self) -> int:
        return self.sort_id

    def __repr__(self) -> str:
        return f"CardRef({self.sort_id})"

    @property
    def data(self) -> dict[str, Any]:
        """The card's snapshot entry."""
        area = self.area
        assert area is not None, f"card {self.sort_id} is gone"
        return next(c for c in self.run.view(area) if c["sort_id"] == self.sort_id)

    @property
    def area(self) -> str | None:
        """The area that holds the card, or ``None`` once it is destroyed or sold."""
        for area in AREAS:
            if any(c["sort_id"] == self.sort_id for c in self.run.view(area)):
                return area
        return None

    @property
    def view(self) -> CardView:
        return CardView.from_snapshot(self.data)

    @property
    def key(self) -> str:
        return self.data["center"]

    @property
    def front(self) -> str | None:
        return self.data["front"]

    @property
    def set(self) -> str:
        """The center's set: ``Joker``, ``Tarot``..., ``Enhanced`` or ``Default``."""
        return load("centers")[self.key]["set"]

    @property
    def ability(self) -> dict[str, Any]:
        return self.data["ability"]

    @property
    def base(self) -> dict[str, Any] | None:
        return self.data["base"]

    @property
    def edition(self) -> dict[str, Any] | None:
        return self.data["edition"]

    @property
    def seal(self) -> str | None:
        return self.data["seal"]

    @property
    def debuff(self) -> bool:
        return self.data["debuff"]

    @property
    def costs(self) -> dict[str, Any]:
        return self.data["costs"]

    @property
    def stickers(self) -> dict[str, Any]:
        return self.data["stickers"]


def front_key(suit: str, rank: str) -> str:
    """The ``add`` key of a playing card (``"S_T"``)."""
    return f"{suit[0]}_{_RANK_LETTER.get(rank, rank)}"


# ---------------------------------------------------------------------------
# The boss's debuff check (blind.lua:624-652)
# ---------------------------------------------------------------------------


def _is_suit(card: dict[str, Any], suit: str, smeared: bool) -> bool:
    """``card:is_suit(suit, true)``, as the boss asks it (card.lua:4064-4086)."""
    if card["center"] == "m_stone":
        return False
    if card["center"] == "m_wild":
        return True
    own = card["base"]["suit"]
    if smeared and (own in ("Hearts", "Diamonds")) == (suit in ("Hearts", "Diamonds")):
        return True
    return own == suit


def _is_face(card: dict[str, Any], pareidolia: bool) -> bool:
    """``card:is_face(true)`` (card.lua:964-970): a Stone Card's id is never a face's."""
    if pareidolia:
        return True
    return card["center"] != "m_stone" and card["base"]["id"] in (11, 12, 13)


def boss_debuffs(snap: dict[str, Any], card: dict[str, Any]) -> bool:
    """The debuff ``G.GAME.blind:debuff_card(card)`` gives a playing card."""
    blind = snap["blind"]
    if blind is None:
        return bool(card["debuff"])
    found = {j["center"] for j in snap["areas"]["jokers"] if not j["debuff"]}
    key, disabled = blind["key"], blind["disabled"]
    debuffed = False
    if blind["boss"] and not disabled:
        config = load("blinds").get(key, {}).get("debuff") or {}
        if "suit" in config and _is_suit(card, config["suit"], "j_smeared" in found):
            debuffed = True
        elif config.get("is_face") == "face" and _is_face(card, "j_pareidolia" in found):
            debuffed = True
        elif key == "bl_pillar" and card["ability"].get("played_this_ante"):
            debuffed = True
    if key == "bl_final_leaf" and not disabled:
        debuffed = True
    return debuffed or bool(card["ability"].get("perma_debuff"))


# ---------------------------------------------------------------------------
# A run
# ---------------------------------------------------------------------------


class Run:
    """A Rust run built for a test, checked against the oracle's run at the same place."""

    def __init__(self, start: dict[str, Any], setup: Sequence[dict[str, Any]] = ()) -> None:
        self.test = _current_test()
        runs = _runs.setdefault(self.test, [])
        self.index = len(runs)
        runs.append(self)
        self.events = _recorded(self.test, self.index)
        self.pos = 0
        self.pending: list[str] = []
        self.synced = False
        # Refreezing: the events with their digests rewritten as the run makes its checks.
        self.out: list[dict[str, Any]] | None = None
        if _REFREEZE:
            self.out = copy.deepcopy(self.events)
            _refrozen.setdefault(self.test, []).append(self.out)
        self.game = frozen.new_game(start, setup)

    @property
    def where(self) -> str:
        return f"{self.test} run {self.index} check {max(self.pos - 1, 0)}"

    # -- reading -------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        return self.game.to_dict()

    def view(self, name: str) -> Any:
        """One section of the snapshot (``Game._view``): an area, ``blind``, ``hands``..."""
        return self.game._view(name)

    def cards(self, area: str) -> list[CardRef]:
        return [CardRef(self, c["sort_id"]) for c in self.view(area)]

    def keys(self, area: str) -> list[str]:
        return [c["center"] for c in self.view(area)]

    def indices(self, cards: Sequence[CardRef]) -> list[int]:
        """Hand positions of ``cards``, ascending."""
        ids = [c["sort_id"] for c in self.view("hand")]
        return sorted(ids.index(card.sort_id) for card in cards)

    # -- building ------------------------------------------------------------

    def op(self, op: dict[str, Any]) -> None:
        """A parity setup op (``Game.apply_setup``)."""
        self.game.apply_setup(op)

    def add(self, key: str, **params: Any) -> CardRef:
        """BalatroBot's ``add`` as the oracle ran it: a joker, a consumable, or a playing card
        (into the hand during a round, else the deck). Returns the new card.

        The oracle's ``add`` made a playing card with ``hands_played_at_create`` 0; Rust stamps
        the run's ``G.GAME.hands_played``, as ``Card:init`` does (card.lua:337). The frozen
        cases keep the oracle's states, so the count is written back to 0."""
        card = CardRef(self, self.game.debug_add(key, **params))
        if card.front is not None and card.ability["hands_played_at_create"]:
            with self.edit_card(card) as data:
                data["ability"]["hands_played_at_create"] = 0
        return card

    @contextmanager
    def edit(self) -> Iterator[dict[str, Any]]:
        """Edit the canonical snapshot; the run continues from the edited state."""
        snap = self.snapshot()
        yield snap
        self.game = Game.from_dict(snap)

    def write(self, path: str, value: Any) -> None:
        """Write one snapshot value by its dotted path (``"current_round.discards_left"``)."""
        with self.edit() as snap:
            *parents, last = path.split(".")
            node = snap
            for key in parents:
                node = node[key]
            node[last] = value

    def read(self, path: str) -> Any:
        node: Any = self.snapshot()
        for key in path.split("."):
            node = node[key]
        return node

    @contextmanager
    def edit_card(self, card: CardRef) -> Iterator[dict[str, Any]]:
        """Edit one card's snapshot entry."""
        with self.edit() as snap:
            yield next(
                c for area in AREAS for c in snap["areas"][area] if c["sort_id"] == card.sort_id
            )

    def move(self, source: str, target: str, *, to_front: bool = False) -> None:
        """Move every card of ``source`` to the end (or the front) of ``target``. Deck and
        discard cards are written face up, as the snapshot keeps them."""
        with self.edit() as snap:
            areas = snap["areas"]
            moved = areas[source]
            if target in ("deck", "discard"):
                for card in moved:
                    card["facing"] = "front"
            areas[target] = moved + areas[target] if to_front else areas[target] + moved
            areas[source] = []

    def boss_check(self, *cards: CardRef) -> None:
        """The boss's debuff check on playing cards made after its set-time pass, as a card dealt
        this round met it (blind.lua:207-210, 624-652)."""
        with self.edit() as snap:
            ids = {c.sort_id for c in cards}
            for area in ("hand", "deck", "discard", "play"):
                for card in snap["areas"][area]:
                    if card["sort_id"] in ids:
                        card["debuff"] = boss_debuffs(snap, card)

    def level_up(self, hand: str, amount: int = 1) -> None:
        """``level_up_hand`` without its card animations (common_events.lua:450-468)."""
        base = load("hands")[hand]
        with self.edit() as snap:
            h = snap["hands"][hand]
            h["level"] = max(0, h["level"] + amount)
            h["mult"] = max(base["s_mult"] + base["l_mult"] * (h["level"] - 1), 1)
            h["chips"] = max(base["s_chips"] + base["l_chips"] * (h["level"] - 1), 0)

    def recheck(self) -> None:
        """``G.GAME.blind:set_blind(nil, true)``: the boss's debuff pass over every playing card
        (blind.lua:207-210)."""
        with self.edit() as snap:
            for area in ("hand", "deck", "discard", "play"):
                for card in snap["areas"][area]:
                    card["debuff"] = boss_debuffs(snap, card)

    def oracle(self, what: str) -> None:
        """The oracle built part of the next checked state with internals the Rust API has no
        counterpart for; the fixture holds the difference."""
        self.pending.append(what)

    def sync(self) -> None:
        """Bring the state to the oracle's now, for a test that reads it before the next check
        (nothing may change in between)."""
        self._resolve(self.events[self.pos], self._out_event(self.pos))
        self.synced = True

    # -- checks --------------------------------------------------------------

    def _next(self, kind: str) -> tuple[dict[str, Any], dict[str, Any] | None]:
        assert self.pos < len(self.events), f"{self.where}: the oracle's run made no more checks"
        event = self.events[self.pos]
        assert kind in event, f"{self.where}: a {kind} check, the oracle made {sorted(event)}"
        out = self._out_event(self.pos)
        self.pos += 1
        return event, out

    def _out_event(self, pos: int) -> dict[str, Any] | None:
        return None if self.out is None else self.out[pos]

    def _check(self, event: dict[str, Any], out: dict[str, Any] | None, key: str, got: Any) -> bool:
        """Compare a frozen value with the run's, or rewrite it when refreezing."""
        if out is not None:
            out[key] = got
            return True
        return got == event[key]

    def _resolve(self, event: dict[str, Any], out: dict[str, Any] | None) -> None:
        """Bring the native state to the oracle's at this check."""
        if self.synced:
            self.synced = False
        elif "patch" in event:
            self.game = Game.from_dict(apply_patch(self.snapshot(), event["patch"]))
        self.pending = []
        key = "pre" if "pre" in event else "state"
        got = frozen.short(self.game.state_hash())
        same = self._check(event, out, key, got)
        assert same, f"{self.where}: the state differs from the oracle's"

    def _legal(self) -> str | None:
        """The legal actions' digest, ``None`` at game over (as the oracle recorded it)."""
        if self.game.phase == "game_over":
            return None
        return frozen.legal_digest(json.loads(self.game._g.legal_actions()))

    def step(self, action: Sequence[Any]) -> bool:
        """Take ``action`` (tape form) where the oracle took it: True if both engines took it,
        False if both refused it."""
        action = json.loads(json.dumps(list(action)))
        event, out = self._next("action")
        where = f"{self.where} {action}"
        assert event["action"] == action, f"{where}: the oracle's action was {event['action']}"
        self._resolve(event, out)
        legal = list(event["legal"])
        if out is not None:
            out["legal"] = legal = [self._legal(), None]
        elif legal[0] is not None:
            assert self._legal() == legal[0], f"{where}: the legal actions differ"
        rebuilt = frozen.transplant(self.game)
        try:
            self.game.step(action)
        except IllegalActionError:
            assert self._check(event, out, "post", None), f"{where}: refused; the oracle took it"
            try:
                rebuilt.step(action)
            except IllegalActionError:
                return False
            raise AssertionError(f"{where}: refused, but taken from the snapshot") from None
        post = frozen.short(self.game.state_hash())
        taken = out is not None or event["post"] is not None
        assert taken, f"{where}: taken; the oracle refused it"
        assert self._check(event, out, "post", post), f"{where}: the state differs"
        rebuilt.step(action)
        assert frozen.short(rebuilt.state_hash()) == post, f"{where}: from the snapshot"
        if out is not None:
            legal[1] = self._legal()
        elif legal[1] is not None:
            assert self._legal() == legal[1], f"{where}: the legal actions after it differ"
        return True

    def legal(self) -> list[Any]:
        """The legal actions (tape form), where the oracle's list was compared with Rust's."""
        event, out = self._next("state")
        self._resolve(event, out)
        legal = json.loads(self.game._g.legal_actions())
        got = frozen.legal_digest(legal)
        assert self._check(event, out, "legal", got), f"{self.where}: legal actions differ"
        return legal

    def bot_legal(self) -> list[Any]:
        """The legal actions under BALATROBOT legality (tape form)."""
        return json.loads(self.game._g.legal_actions("balatrobot"))

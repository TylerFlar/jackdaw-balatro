"""Copying live values into the shadow run: snapshot edits.

A reconcile (``jackdaw.bridge.deviations``) answers a known difference between
the live game and the shadow run by making the shadow agree with the live
view. It never changes the shadow directly: it returns *edits* to the shadow's
canonical snapshot (``docs/snapshot-schema.md``), which the adapter applies
with :meth:`jackdaw.core.Game.replace_state` and records in the action log, so
``jackdaw lockstep replay`` reproduces the shadow exactly. The edits:

- ``{"op": "set", "path": [...], "value": v}``: a snapshot field;
- ``{"op": "card", "sort_id": n, "path": [...], "value": v}``: a field of the
  card with that sort id, wherever it is;
- ``{"op": "arrange", "areas": {"hand": [ids], "deck": [ids]}}``: those areas
  rebuilt, in that order, from the cards they hold between them.

Only what the snapshot carries can be reconciled. A known consequence of the
deviation that needs more (a joker's debuff, whose passives would have to be
re-applied; a card vanilla makes and the live game doesn't) raises
:class:`Unrepairable`: the divergence stays, classified under the deviation.
Live values that don't fit the deviation at all raise :class:`ReconcileError`:
something else is going on, and the divergence is unclassified.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from jackdaw.bridge.viewdiff import FieldDiff, IdMap


class ReconcileError(Exception):
    """The live values don't fit the deviation: they can't be copied into the shadow."""


class Unrepairable(ReconcileError):
    """A known consequence of the deviation that the snapshot can't carry."""


# ---------------------------------------------------------------------------
# Applying edits
# ---------------------------------------------------------------------------


def _walk(root: Any, path: list[Any]) -> tuple[Any, Any]:
    node = root
    for key in path[:-1]:
        node = node[key]
    return node, path[-1]


def find_card(snapshot: dict[str, Any], sort_id: int) -> dict[str, Any]:
    """The card with ``sort_id`` in any area of a snapshot."""
    for cards in snapshot["areas"].values():
        for card in cards:
            if card["sort_id"] == sort_id:
                return card
    raise ReconcileError(f"no card #{sort_id} in the shadow run")


def apply_edits(snapshot: dict[str, Any], edits: Iterable[dict[str, Any]]) -> None:
    """Apply reconcile edits to a snapshot, in place."""
    for edit in edits:
        op = edit["op"]
        if op == "set":
            node, key = _walk(snapshot, list(edit["path"]))
            node[key] = edit["value"]
        elif op == "card":
            node, key = _walk(find_card(snapshot, edit["sort_id"]), list(edit["path"]))
            node[key] = edit["value"]
        elif op == "arrange":
            areas: dict[str, list[int]] = edit["areas"]
            pool = {c["sort_id"]: c for a in areas for c in snapshot["areas"][a]}
            wanted = [sid for ids in areas.values() for sid in ids]
            if sorted(wanted) != sorted(pool):
                raise ReconcileError(
                    f"arrange: the areas {sorted(areas)} hold other cards than the edit names"
                )
            for a, ids in areas.items():
                snapshot["areas"][a] = [pool[sid] for sid in ids]
        else:
            raise ValueError(f"unknown reconcile edit {op!r}")


# ---------------------------------------------------------------------------
# What a reconcile sees
# ---------------------------------------------------------------------------


@dataclass
class ReconcileContext:
    """The step a reconcile repairs: the shadow's snapshot after the step (edits
    are computed against it), both views, the differences assigned to the
    deviation, the cards the diff paired by position, and the id pairing so far."""

    snapshot: dict[str, Any]
    sim_view: dict[str, Any]
    live_view: dict[str, Any]
    diffs: list[FieldDiff]
    cards: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]
    ids: IdMap
    #: Sort ids of every card the shadow had before the step.
    known_ids: set[int] = field(default_factory=set)

    def paired(self, area: str) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        """``(live card, sim card)`` pairs of a view area, by position."""
        return self.cards.get(area, [])

    def card_at(self, path: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        """The ``(live, sim)`` card pair a diff path points into
        (``jokers.cards[2].cost.sell`` -> the third joker pair)."""
        area, _, rest = path.partition(".cards[")
        if not rest:
            return None
        index = int(rest.split("]", 1)[0])
        pairs = self.paired(area)
        return pairs[index] if index < len(pairs) else None


# ---------------------------------------------------------------------------
# Playing cards
# ---------------------------------------------------------------------------

# Rank letter -> (value, id, nominal, face_nominal); suit letter -> (suit,
# suit_nominal, suit_nominal_original), as Card:set_base sets them (card.lua:97-145).
_RANKS: dict[str, tuple[str, int, int, float]] = {
    **{str(n): (str(n), n, n, 0) for n in range(2, 10)},
    "T": ("10", 10, 10, 0),
    "J": ("Jack", 11, 10, 0.1),
    "Q": ("Queen", 12, 10, 0.2),
    "K": ("King", 13, 10, 0.3),
    "A": ("Ace", 14, 11, 0.4),
}
_SUITS: dict[str, tuple[str, float, float]] = {
    "D": ("Diamonds", 0.01, 0.001),
    "C": ("Clubs", 0.02, 0.002),
    "H": ("Hearts", 0.03, 0.003),
    "S": ("Spades", 0.04, 0.004),
}
#: The view's seal names -> the snapshot's.
SEALS: dict[str, str] = {"RED": "Red", "BLUE": "Blue", "GOLD": "Gold", "PURPLE": "Purple"}


def front_base(key: str, old: dict[str, Any], *, created: bool) -> dict[str, Any]:
    """The ``base`` a playing card gets from ``Card:set_base`` for the front ``key``
    (``H_A``): a card created with it (``initial``: its original suit and rank
    recorded) or one converted to it (the original suit kept, the original rank
    dropped)."""
    suit_letter, _, rank_letter = key.partition("_")
    if suit_letter not in _SUITS or rank_letter not in _RANKS:
        raise ReconcileError(f"not a playing card front: {key!r}")
    suit, suit_nominal, suit_original = _SUITS[suit_letter]
    value, rank_id, nominal, face = _RANKS[rank_letter]
    base = dict(old)
    base.update(
        suit=suit,
        value=value,
        id=rank_id,
        nominal=nominal,
        face_nominal=face,
        suit_nominal=suit_nominal,
        times_played=0,
    )
    if created:
        base["suit_nominal_original"] = suit_original
        base["original_value"] = value
    else:
        base["original_value"] = None
    return base


def table(card: dict[str, Any], key: str) -> dict[str, Any]:
    """A card's ``modifier`` / ``state`` / ``cost`` table (``[]`` when empty)."""
    value = card.get(key) or {}
    return value if isinstance(value, dict) else {}


def _content(card: dict[str, Any]) -> tuple[Any, ...]:
    """What identifies a playing card's face: front and modifiers."""
    m = table(card, "modifier")
    return (card.get("key"), m.get("seal"), m.get("edition"), m.get("enhancement"))


def arrange_playing_cards(
    rctx: ReconcileContext,
    *,
    fronts: bool = False,
    debuffs: bool = False,
    forced: bool = False,
) -> list[dict[str, Any]]:
    """Edits that give the shadow's hand and deck the live game's cards, in the live
    order: each live card is the shadow card its id was paired with before, else an
    unpaired shadow card with the same face. With ``fronts``, cards still unpaired
    are paired in creation order and take the live front (a card new this step as
    created with it, an old one as converted); with ``debuffs`` a card takes the live
    debuff. Hand cards take the live facing; with ``forced``, Cerulean Bell's forced
    card is the one the live game highlights."""
    live_hand = rctx.live_view.get("hand", {}).get("cards") or []
    live_deck = rctx.live_view.get("cards", {}).get("cards") or []
    sim_cards = {
        c["id"]: c
        for area in ("hand", "cards")
        for c in rctx.sim_view.get(area, {}).get("cards") or []
    }
    live_cards = list(live_hand) + list(live_deck)
    if len(live_cards) != len(sim_cards):
        raise ReconcileError(
            f"the live hand and deck hold {len(live_cards)} cards, the shadow's {len(sim_cards)}"
        )
    assign: dict[int, int] = {}
    used: set[int] = set()
    for lc in live_cards:
        sid = rctx.ids.live_to_sim.get(lc["id"])
        if sid in sim_cards and sid not in used:
            assign[lc["id"]] = sid
            used.add(sid)
    for lc in live_cards:
        if lc["id"] in assign:
            continue
        same = sorted(
            sid
            for sid, sc in sim_cards.items()
            if sid not in used and sid not in rctx.ids.sim_to_live and _content(sc) == _content(lc)
        )
        if same:
            assign[lc["id"]] = same[0]
            used.add(same[0])
    left_live = sorted(lc["id"] for lc in live_cards if lc["id"] not in assign)
    left_sim = sorted(sid for sid in sim_cards if sid not in used)
    if left_live or left_sim:
        if not fronts or len(left_live) != len(left_sim):
            raise ReconcileError(
                f"no shadow card for live cards {left_live}, no live card for shadow cards "
                f"{left_sim}"
            )
        for lid, sid in zip(left_live, left_sim, strict=True):
            assign[lid] = sid
            used.add(sid)
    edits: list[dict[str, Any]] = [
        {
            "op": "arrange",
            "areas": {
                "hand": [assign[c["id"]] for c in live_hand],
                "deck": [assign[c["id"]] for c in live_deck],
            },
        }
    ]
    by_live = {c["id"]: c for c in live_cards}
    for lid, sid in assign.items():
        lc, sc = by_live[lid], sim_cards[sid]
        card = find_card(rctx.snapshot, sid)
        if lc.get("key") != sc.get("key"):
            if not fronts:
                raise ReconcileError(
                    f"live card #{lid} is {lc.get('key')}, shadow #{sid} {sc.get('key')}"
                )
            created = sid not in rctx.known_ids
            base = front_base(lc["key"], card["base"], created=created)
            edits.append({"op": "card", "sort_id": sid, "path": ["front"], "value": lc["key"]})
            edits.append({"op": "card", "sort_id": sid, "path": ["base"], "value": base})
        if _content(lc)[1:] != _content(sc)[1:]:
            raise ReconcileError(f"live card #{lid} and shadow #{sid} differ in their modifiers")
        live_debuff = bool(table(lc, "state").get("debuff"))
        if debuffs and live_debuff != card["debuff"]:
            edits.append({"op": "card", "sort_id": sid, "path": ["debuff"], "value": live_debuff})
    for lc in live_hand:
        sid = assign[lc["id"]]
        state = table(lc, "state")
        facing = "back" if state.get("hidden") else "front"
        if find_card(rctx.snapshot, sid)["facing"] != facing:
            edits.append({"op": "card", "sort_id": sid, "path": ["facing"], "value": facing})
        if forced:
            want = True if state.get("highlight") else None
            have = find_card(rctx.snapshot, sid)["ability"].get("forced_selection")
            if bool(have) != bool(want):
                edits.append(
                    {
                        "op": "card",
                        "sort_id": sid,
                        "path": ["ability", "forced_selection"],
                        "value": want,
                    }
                )
    return edits

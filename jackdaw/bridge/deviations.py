"""The deviation catalog: known differences between the live game and the shadow run.

The live game BalatroBot drives runs Steamodded, and BalatroBot itself drives it
in ways a player can't; the simulator follows vanilla Balatro 1.0.1o. Every
known difference is an entry here, with the id of the document that describes
it:

- ``S01``-``S13`` and ``B01``-``B04``: ``docs/parity/smods-diffs.md``;
- ``BB-*``: ``docs/balatrobot.md`` (the command-space policy, the view, the
  timing differences) and ``docs/parity/gaps.md`` (``B21``).

Each entry has a mode that says what the live lockstep
(:class:`jackdaw.bridge.live.LiveAdapter`) does about it:

- ``ignore``: a difference on the entry's paths is dropped from the diff (and
  counted, so a run's summary shows how often it fired);
- ``reconcile``: the live value is copied into the shadow run (a snapshot edit,
  :mod:`jackdaw.bridge.reconcile`), where the snapshot can carry it; the step
  then has to compare clean;
- ``veto``: the action is not taken in lockstep; the adapter refuses it before
  anything reaches the live game (:class:`~jackdaw.bridge.live.VetoedActionError`)
  and leaves it out of its legal actions.

A difference is only explained by an entry that is *active* for the step: its
trigger fired (the step is one the deviation can affect) or, for a sticky
entry, fired earlier in the run. A difference an active entry explains but
can't repair (a reconcile the snapshot can't carry, a veto the predicate
missed) is a **classified** divergence: the run stops, under the entry's id.
Anything else is **unclassified**: a new divergence to triage
(``docs/live.md``).
"""

from __future__ import annotations

import enum
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from jackdaw.bridge.reconcile import (
    SEALS,
    ReconcileContext,
    ReconcileError,
    Unrepairable,
    arrange_playing_cards,
    table,
)
from jackdaw.bridge.viewdiff import ABSENT, TEXT_FIELDS, FieldDiff, matches_any, path_pattern

if TYPE_CHECKING:
    from jackdaw.bridge.backend import RPCError
    from jackdaw.core import Game


class Mode(enum.StrEnum):
    """What the lockstep does about a deviation."""

    IGNORE = "ignore"
    RECONCILE = "reconcile"
    VETO = "veto"


# ---------------------------------------------------------------------------
# What a deviation sees
# ---------------------------------------------------------------------------


@dataclass
class RunMemory:
    """What the catalog remembers over one run."""

    #: Sticky deviations that fired: active for the rest of the run.
    sticky: set[str] = field(default_factory=set)
    #: Antes in which Certificate fired (S03).
    certificate_antes: set[int] = field(default_factory=set)
    #: Sim ids of jokers Invisible Joker or Ankh copied (S05).
    copies: set[int] = field(default_factory=set)
    #: What each cash_out paid in the shadow, in order (BB-CASHOUT-RACE).
    payouts: list[float] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "sticky": sorted(self.sticky),
            "certificate_antes": sorted(self.certificate_antes),
            "copies": sorted(self.copies),
            "payouts": list(self.payouts),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> RunMemory:
        return cls(
            set(data.get("sticky", ())),
            set(data.get("certificate_antes", ())),
            set(data.get("copies", ())),
            list(data.get("payouts", ())),
        )


@dataclass
class StepContext:
    """One lockstep step: the request, the shadow run before and after it (``post``
    is None before the shadow has taken it), the shadow's refusal if it refused."""

    method: str | None
    params: dict[str, Any]
    action: Any = None
    pre: Game | None = None
    pre_view: dict[str, Any] | None = None
    post: Game | None = None
    post_view: dict[str, Any] | None = None
    memory: RunMemory = field(default_factory=RunMemory)
    refusal: RPCError | None = None
    _snapshots: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)

    @property
    def pre_snapshot(self) -> dict[str, Any]:
        if "pre" not in self._snapshots:
            assert self.pre is not None
            self._snapshots["pre"] = self.pre.to_dict()
        return self._snapshots["pre"]

    @property
    def post_snapshot(self) -> dict[str, Any]:
        if "post" not in self._snapshots:
            assert self.post is not None
            self._snapshots["post"] = self.post.to_dict()
        return self._snapshots["post"]


# ---------------------------------------------------------------------------
# Reading views and snapshots
# ---------------------------------------------------------------------------


def _cards(view: dict[str, Any] | None, area: str) -> list[dict[str, Any]]:
    if not view:
        return []
    a = view.get(area)
    if not isinstance(a, dict):
        return []
    return list(a.get("cards") or [])


def _owns(view: dict[str, Any] | None, key: str, *, active: bool = False) -> bool:
    """Whether a joker with this key is held (``active``: and not debuffed)."""
    return any(
        c.get("key") == key and not (active and table(c, "state").get("debuff"))
        for c in _cards(view, "jokers")
    )


def _enhancement(card: dict[str, Any]) -> Any:
    return table(card, "modifier").get("enhancement")


def _debuffed(card: dict[str, Any]) -> bool:
    return bool(table(card, "state").get("debuff"))


def _boss_active(snapshot: dict[str, Any], key: str) -> bool:
    blind = snapshot.get("blind") or {}
    return blind.get("key") == key and not blind.get("disabled")


def _forced_position(snapshot: dict[str, Any]) -> int | None:
    for i, c in enumerate(snapshot["areas"]["hand"]):
        if c["ability"].get("forced_selection"):
            return i
    return None


def _indices(params: dict[str, Any], key: str) -> list[int]:
    value = params.get(key)
    if not isinstance(value, list):
        return []
    return [int(i) for i in value if isinstance(i, (int, float))]


def _card_key(view: dict[str, Any] | None, area: str, index: Any) -> str | None:
    cards = _cards(view, area)
    if isinstance(index, (int, float)) and 0 <= index < len(cards):
        return cards[int(index)].get("key")
    return None


def _state(view: dict[str, Any] | None) -> str | None:
    return None if not view else view.get("state")


def _ante(view: dict[str, Any] | None) -> int:
    return int((view or {}).get("ante_num", 0))


def _limit(view: dict[str, Any] | None, area: str) -> Any:
    a = (view or {}).get(area)
    return a.get("limit") if isinstance(a, dict) else None


# ---------------------------------------------------------------------------
# The entry type
# ---------------------------------------------------------------------------

Trigger = Callable[[StepContext], bool]
Veto = Callable[[StepContext], "str | None"]
Reconcile = Callable[[ReconcileContext, StepContext], "list[dict[str, Any]]"]


@dataclass(frozen=True)
class Deviation:
    """One known difference between the live game and the simulator."""

    id: str
    title: str
    #: Where it is described (``docs/...#anchor``).
    source: str
    mode: Mode
    #: View paths it can make differ (``viewdiff.path_pattern`` globs).
    paths: tuple[str, ...] = ()
    #: When it can affect a step; None: always.
    trigger: Trigger | None = None
    #: Once triggered, it stays active for the rest of the run.
    sticky: bool = False
    #: A value check on a difference it would explain.
    accepts: Callable[[FieldDiff, StepContext], bool] | None = None
    #: ``veto``: why the action is not taken (None: it is).
    veto: Veto | None = None
    #: ``veto``: the shadow refused the request for this reason.
    refusal: Trigger | None = None
    #: ``reconcile``: the snapshot edits that copy the live values.
    reconcile: Reconcile | None = None
    #: Records what later steps need (RunMemory).
    observe: Callable[[StepContext], None] | None = None
    #: False: the paths are never compared at all.
    compared: bool = True
    note: str = ""

    def patterns(self) -> tuple[re.Pattern[str], ...]:
        return tuple(path_pattern(p) for p in self.paths)

    def active(self, ctx: StepContext) -> bool:
        if self.id in ctx.memory.sticky:
            return True
        if self.trigger is None:
            return True
        try:
            return bool(self.trigger(ctx))
        except (KeyError, IndexError, TypeError, ValueError, AssertionError):
            return False

    def explains(self, diff: FieldDiff, ctx: StepContext) -> bool:
        if not self.paths or not matches_any(diff.path, self.patterns()):
            return False
        return self.accepts is None or bool(self.accepts(diff, ctx))

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "title": self.title, "source": self.source, "mode": str(self.mode)}


# ---------------------------------------------------------------------------
# Steamodded (docs/parity/smods-diffs.md)
# ---------------------------------------------------------------------------

_SMODS = "docs/parity/smods-diffs.md"
_PLAYING = ("hand", "cards")  # the view's hand and deck areas


def _s01_trigger(ctx: StepContext) -> bool:
    return ctx.method == "select" and _owns(ctx.pre_view, "j_marble")


def _arrange(**kw: bool) -> Reconcile:
    def run(rctx: ReconcileContext, ctx: StepContext) -> list[dict[str, Any]]:
        return arrange_playing_cards(rctx, **kw)

    return run


def _s02_reconcile(rctx: ReconcileContext, ctx: StepContext) -> list[dict[str, Any]]:
    if ctx.method == "select":
        return arrange_playing_cards(rctx, forced=True)
    # Vanilla's Certificate fires again after a consumable empties the hand: the shadow
    # holds a card (and two draws) the live game doesn't.
    raise Unrepairable("vanilla's Certificate fired a second time this round")


def _s02_trigger(ctx: StepContext) -> bool:
    if not _owns(ctx.pre_view, "j_certificate", active=True):
        return False
    snap = ctx.pre_snapshot
    if ctx.method == "select":
        return (
            snap["game"].get("blind_on_deck") == "Boss"
            and (snap["round_resets"].get("blind_choices") or {}).get("Boss") == "bl_final_bell"
        )
    # A consumable that empties the hand before the first play sends the game back
    # to DRAW_TO_HAND, where vanilla's Certificate fires again (game.lua:2544-2548).
    cr = snap["current_round"]
    return (
        ctx.method in ("use", "pack")
        and _state(ctx.pre_view) == "SELECTING_HAND"
        and cr.get("hands_played") == 0
        and cr.get("discards_used") == 0
    )


def _s03_observe(ctx: StepContext) -> None:
    if ctx.method == "select" and _owns(ctx.pre_view, "j_certificate", active=True):
        ctx.memory.certificate_antes.add(_ante(ctx.pre_view))


def _s03_trigger(ctx: StepContext) -> bool:
    if ctx.post is None or _ante(ctx.post_view) not in ctx.memory.certificate_antes:
        return False
    pack = ctx.post_snapshot.get("pack") or {}
    return pack.get("kind") == "Standard"


def _s03_reconcile(rctx: ReconcileContext, ctx: StepContext) -> list[dict[str, Any]]:
    edits = []
    for diff in rctx.diffs:
        pair = rctx.card_at(diff.path)
        if pair is None:
            raise ReconcileError(f"{diff.path}: not a pack card")
        live, sim = pair
        seal = table(live, "modifier").get("seal")
        value = None if seal is None else SEALS.get(seal)
        if seal is not None and value is None:
            raise ReconcileError(f"unknown seal {seal!r}")
        edits.append({"op": "card", "sort_id": sim["id"], "path": ["seal"], "value": value})
    return edits


def _s04_trigger(ctx: StepContext) -> bool:
    # In a round; or the deal, when setting the blind changes the hand size (Madness destroys
    # a Juggler, Turtle Bean...: vanilla deals to the old limit, Steamodded stops at the new).
    if ctx.post is None or _state(ctx.pre_view) not in ("SELECTING_HAND", "BLIND_SELECT"):
        return False
    if _limit(ctx.pre_view, "hand") != _limit(ctx.post_view, "hand"):
        return True
    return _boss_active(ctx.pre_snapshot, "bl_manacle") and not _boss_active(
        ctx.post_snapshot, "bl_manacle"
    )


def _s05_copies(ctx: StepContext) -> bool:
    """Invisible Joker sold or Ankh used: this step may copy a joker."""
    if ctx.method == "sell" and "joker" in ctx.params:
        return _card_key(ctx.pre_view, "jokers", ctx.params["joker"]) == "j_invisible"
    if ctx.method == "use":
        return _card_key(ctx.pre_view, "consumables", ctx.params.get("consumable")) == "c_ankh"
    if ctx.method == "pack":
        return _card_key(ctx.pre_view, "pack", ctx.params.get("card")) == "c_ankh"
    return False


def _s05_observe(ctx: StepContext) -> None:
    if ctx.post_view is None or not _s05_copies(ctx):
        return
    before = {c.get("id") for c in _cards(ctx.pre_view, "jokers")}
    ctx.memory.copies.update(
        c["id"] for c in _cards(ctx.post_view, "jokers") if c.get("id") not in before
    )


def _s05_trigger(ctx: StepContext) -> bool:
    return _s05_copies(ctx) or bool(ctx.memory.copies)


def _s05_accepts(diff: FieldDiff, ctx: StepContext) -> bool:
    """The joker slots, or the price of a joker that is one of the copies."""
    if diff.path == "jokers.limit":
        return bool(ctx.memory.copies)
    match = re.match(r"jokers\.cards\[(\d+)\]", diff.path)
    jokers = _cards(ctx.post_view, "jokers")
    if match is None or int(match.group(1)) >= len(jokers):
        return False
    return jokers[int(match.group(1))].get("id") in ctx.memory.copies


def _s05_reconcile(rctx: ReconcileContext, ctx: StepContext) -> list[dict[str, Any]]:
    edits = []
    for diff in rctx.diffs:
        if diff.path == "jokers.limit":
            edits.append({"op": "set", "path": ["game", "joker_slots"], "value": diff.live})
            continue
        pair = rctx.card_at(diff.path)
        if pair is None:
            raise ReconcileError(f"{diff.path}: not a joker")
        live, sim = pair
        cost = table(live, "cost")
        field_name = {"buy": "cost", "sell": "sell_cost"}.get(diff.path.rsplit(".", 1)[-1])
        if field_name is None:
            raise ReconcileError(f"{diff.path}: not a price")
        edits.append(
            {
                "op": "card",
                "sort_id": sim["id"],
                "path": ["costs", field_name],
                "value": cost[diff.path.rsplit(".", 1)[-1]],
            }
        )
    return edits


def _s06_trigger(ctx: StepContext) -> bool:
    """Seeing Double's suit tally (card.lua:3840-3870) differs only when the scoring
    hand's one suited, non-debuffed card is a single Wild Card."""
    if ctx.method != "play" or not _owns(ctx.pre_view, "j_seeing_double", active=True):
        return False
    assert ctx.pre is not None
    hand = _cards(ctx.pre_view, "hand")
    scored = ctx.pre.scoring_cards(_indices(ctx.params, "cards"))
    if scored is None:
        return False
    suited = [
        hand[i] for i in scored[1] if not _debuffed(hand[i]) and _enhancement(hand[i]) != "STONE"
    ]
    return len(suited) == 1 and _enhancement(suited[0]) == "WILD"


def _set_chips(rctx: ReconcileContext, ctx: StepContext) -> list[dict[str, Any]]:
    (diff,) = rctx.diffs
    return [{"op": "set", "path": ["game", "chips"], "value": diff.live}]


def _s07_veto(ctx: StepContext) -> str | None:
    if ctx.method != "play" or not _owns(ctx.pre_view, "j_four_fingers", active=True):
        return None
    hand = _cards(ctx.pre_view, "hand")
    wild = [i for i in _indices(ctx.params, "cards") if 0 <= i < len(hand)]
    if sum(_enhancement(hand[i]) == "WILD" for i in wild) >= 3:
        return "Four Fingers and three or more Wild Cards: the flush suit order differs (S07)"
    return None


_S08_KEYS = frozenset({"c_sigil", "c_grim", "c_familiar", "c_incantation"})


def _s08_trigger(ctx: StepContext) -> bool:
    if ctx.method == "use":
        return _card_key(ctx.pre_view, "consumables", ctx.params.get("consumable")) in _S08_KEYS
    if ctx.method == "pack":
        return _card_key(ctx.pre_view, "pack", ctx.params.get("card")) in _S08_KEYS
    return False


def _s09_trigger(ctx: StepContext) -> bool:
    return (
        ctx.post is not None
        and _state(ctx.pre_view) == "SELECTING_HAND"
        and _state(ctx.post_view) in ("ROUND_EVAL", "GAME_OVER")
        and any("perishable" in table(c, "modifier") for c in _cards(ctx.pre_view, "jokers"))
    )


def _s09_reconcile(rctx: ReconcileContext, ctx: StepContext) -> list[dict[str, Any]]:
    edits = []
    for diff in rctx.diffs:
        if not diff.path.endswith(".modifier.perishable"):
            # A joker that expired in one game only: its debuff (and Turtle Bean's hand
            # size) would need its passives removed, which a snapshot edit can't do.
            raise Unrepairable(f"{diff.path}: only a perishable tally can be copied")
        pair = rctx.card_at(diff.path)
        if pair is None:
            raise ReconcileError(f"{diff.path}: not a joker")
        live, sim = pair
        tally = table(live, "modifier").get("perishable", 0)
        edits.append(
            {
                "op": "card",
                "sort_id": sim["id"],
                "path": ["stickers", "perish_tally"],
                "value": tally,
            }
        )
    return edits


def _s10_accepts(diff: FieldDiff, ctx: StepContext) -> bool:
    try:
        delta = abs(float(diff.live) - float(diff.sim))
    except (TypeError, ValueError):
        return False
    return delta <= max(1.0, abs(float(diff.live)) * 1e-12)


_S11_KEY = re.compile(r"p_(buffoon_normal|arcana_mega|celestial_mega)_[12]")


def _s11_accepts(diff: FieldDiff, ctx: StepContext) -> bool:
    a, b = diff.sim, diff.live
    if not (isinstance(a, str) and isinstance(b, str)):
        return False
    ma, mb = _S11_KEY.fullmatch(a), _S11_KEY.fullmatch(b)
    return bool(ma and mb and ma.group(1) == mb.group(1))


def _lost_round(ctx: StepContext) -> bool:
    """The step lost the run at the end of a round."""
    return (
        ctx.post is not None
        and _state(ctx.pre_view) == "SELECTING_HAND"
        and _state(ctx.post_view) == "GAME_OVER"
    )


def _s14_accepts(diff: FieldDiff, ctx: StepContext) -> bool:
    """The live deck limit is the run's playing card count."""
    snap = ctx.post_snapshot if ctx.post is not None else ctx.pre_snapshot
    areas = snap["areas"]
    playing = sum(len(areas[a]) for a in ("deck", "hand", "discard", "play"))
    return diff.live == playing


def _s13_veto(ctx: StepContext) -> str | None:
    if ctx.method not in ("play", "discard") or ctx.pre is None:
        return None
    if not _boss_active(ctx.pre_snapshot, "bl_final_heart"):
        return None
    jokers = _cards(ctx.pre_view, "jokers")
    if len(jokers) >= 2 and all(_debuffed(c) for c in jokers):
        return "Crimson Heart with no joker left to pick: vanilla errors, Steamodded re-picks (S13)"
    return None


# ---------------------------------------------------------------------------
# BalatroBot (smods-diffs.md B01-B04)
# ---------------------------------------------------------------------------


def _refused(*fragments: str) -> Trigger:
    def match(ctx: StepContext) -> bool:
        return ctx.refusal is not None and any(f in ctx.refusal.message for f in fragments)

    return match


def _b02_veto(ctx: StepContext) -> str | None:
    if ctx.method not in ("play", "discard", "use") or ctx.pre is None:
        return None
    if ctx.method == "use" and "cards" not in ctx.params:
        return None
    forced = _forced_position(ctx.pre_snapshot)
    if forced is None or forced in _indices(ctx.params, "cards"):
        return None
    return (
        f"Cerulean Bell's forced card (hand position {forced}) is not among the cards: "
        f"BalatroBot would add it and drop the last one (B02)"
    )


def _b03_trigger(ctx: StepContext) -> bool:
    return ctx.method == "add"


def _b03_reconcile(rctx: ReconcileContext, ctx: StepContext) -> list[dict[str, Any]]:
    edits = []
    for diff in rctx.diffs:
        pair = rctx.card_at(diff.path)
        if pair is None:
            raise ReconcileError(f"{diff.path}: not a playing card")
        live, sim = pair
        edits.append(
            {
                "op": "card",
                "sort_id": sim["id"],
                "path": ["debuff"],
                "value": bool(table(live, "state").get("debuff")),
            }
        )
    return edits


def _b03_edition_observe(ctx: StepContext) -> None:
    if ctx.method == "add" and ctx.params.get("edition"):
        ctx.memory.sticky.add("B03-EDITION")


#: A view edition -> the snapshot's edition table (Card:set_edition, card.lua:387-462).
_EDITIONS: dict[str, dict[str, Any]] = {
    "FOIL": {"foil": True, "chips": 50, "type": "foil"},
    "HOLO": {"holo": True, "mult": 10, "type": "holo"},
    "POLYCHROME": {"polychrome": True, "x_mult": 1.5, "type": "polychrome"},
    "NEGATIVE": {"negative": True, "type": "negative"},
}


def _b03_edition_reconcile(rctx: ReconcileContext, ctx: StepContext) -> list[dict[str, Any]]:
    edits = []
    done: set[int] = set()
    for diff in rctx.diffs:
        pair = rctx.card_at(diff.path)
        if pair is None:
            raise ReconcileError(f"{diff.path}: not a shop or pack card")
        live, sim = pair
        if sim["id"] in done:
            continue
        done.add(sim["id"])
        edition = table(live, "modifier").get("edition")
        if edition is not None and edition not in _EDITIONS:
            raise ReconcileError(f"unknown edition {edition!r}")
        cost = table(live, "cost")
        edits += [
            {
                "op": "card",
                "sort_id": sim["id"],
                "path": ["edition"],
                "value": None if edition is None else dict(_EDITIONS[edition]),
            },
            {"op": "card", "sort_id": sim["id"], "path": ["costs", "cost"], "value": cost["buy"]},
            {
                "op": "card",
                "sort_id": sim["id"],
                "path": ["costs", "sell_cost"],
                "value": cost["sell"],
            },
        ]
    return edits


# ---------------------------------------------------------------------------
# The command space, timing and the view (docs/balatrobot.md, gaps.md)
# ---------------------------------------------------------------------------

_PACK_TAGS = frozenset({"tag_charm", "tag_meteor", "tag_buffoon", "tag_ethereal", "tag_standard"})


def _skip_pack_veto(ctx: StepContext) -> str | None:
    if ctx.method != "skip" or ctx.pre is None:
        return None
    snap = ctx.pre_snapshot
    slot = snap["game"].get("blind_on_deck")
    tag = (snap["round_resets"].get("blind_tags") or {}).get(slot)
    if tag in _PACK_TAGS:
        return (
            f"skipping for {tag} opens a pack; the live `skip` waits for BLIND_SELECT "
            f"and never answers"
        )
    return None


def _outside_shop(ctx: StepContext) -> bool:
    if ctx.post is None:
        return _state(ctx.pre_view) != "SHOP"
    state = _state(ctx.post_view)
    if state == "SHOP":
        return False
    if state == "SMODS_BOOSTER_OPENED":
        return (ctx.post_snapshot.get("pack") or {}).get("return_phase") != "shop"
    return True


def _boss_beaten(ctx: StepContext) -> bool:
    """The play that beat a boss blind: the round is being evaluated."""
    return (
        ctx.method == "play"
        and ctx.post is not None
        and _state(ctx.post_view) == "ROUND_EVAL"
        and bool((ctx.pre_snapshot.get("blind") or {}).get("boss"))
    )


#: The view fields a joker's passive changes when Card:add_to_deck gives it back
#: (card.lua:586-604), by the jokers that have one.
_PASSIVES: dict[str, frozenset[str]] = {
    "hand.limit": frozenset(
        {"j_juggler", "j_troubadour", "j_merry_andy", "j_stuntman", "j_turtle_bean"}
    ),
    "round.discards_left": frozenset({"j_drunkard", "j_merry_andy"}),
    "round.reroll_cost": frozenset({"j_chaos"}),
}

_CARD_STATE = re.compile(r"(cards|jokers)\.cards\[(\d+)\]\.state\.(debuff|hidden)")


def _boss_defeat_accepts(diff: FieldDiff, ctx: StepContext) -> bool:
    """What Blind:defeat hasn't undone yet: a card the boss had debuffed before the
    play, in any area (the view leaves out the discard pile), or a joker it had turned
    face down (Amber Acorn); The Manacle's lost hand size; under Crimson Heart, the
    passive its debuffed joker hasn't got back."""
    areas = ctx.pre_snapshot["areas"].values()
    debuffed = {c["sort_id"]: c["center"] for cards in areas for c in cards if c.get("debuff")}
    match = _CARD_STATE.fullmatch(diff.path)
    if match is not None:
        cards = _cards(ctx.post_view, match.group(1))
        i = int(match.group(2))
        if diff.live is not True or diff.sim is not ABSENT or i >= len(cards):
            return False
        if match.group(3) == "debuff":
            return cards[i].get("id") in debuffed
        jokers = ctx.pre_snapshot["areas"]["jokers"]
        face_down = {c["sort_id"] for c in jokers if c["facing"] == "back"}
        return match.group(1) == "jokers" and cards[i].get("id") in face_down
    if diff.path == "hand.limit" and _boss_active(ctx.pre_snapshot, "bl_manacle"):
        return diff.sim == diff.live + 1
    if not _boss_active(ctx.pre_snapshot, "bl_final_heart"):
        return False
    keys = _PASSIVES.get(diff.path, frozenset())
    return any(center in keys for center in debuffed.values())


def _payout_observe(ctx: StepContext) -> None:
    if ctx.method == "cash_out" and ctx.pre_view and ctx.post_view:
        ctx.memory.payouts.append(ctx.post_view["money"] - ctx.pre_view["money"])


def _cashout_race_accepts(diff: FieldDiff, ctx: StepContext) -> bool:
    """The live game paid the previous round's total instead of this round's."""
    payouts = ctx.memory.payouts  # this cash_out's is the last one
    if len(payouts) < 2 or not ctx.pre_view:
        return False
    try:
        paid = diff.live - ctx.pre_view["money"]
    except TypeError:
        return False
    return paid == payouts[-2] != payouts[-1]


def _set_money(rctx: ReconcileContext, ctx: StepContext) -> list[dict[str, Any]]:
    (diff,) = rctx.diffs
    return [{"op": "set", "path": ["game", "dollars"], "value": diff.live}]


# ---------------------------------------------------------------------------
# The catalog
# ---------------------------------------------------------------------------

CATALOG: tuple[Deviation, ...] = (
    Deviation(
        "S01",
        "Marble Joker's Stone joins the deck after the round's 'nr' shuffle, at the bottom",
        f"{_SMODS}#s01",
        Mode.RECONCILE,
        paths=_PLAYING,
        trigger=_s01_trigger,
        reconcile=_arrange(),
        note="The 'nr' stream is drawn as often, so only the round's deal order differs: the "
        "shadow's hand and deck take the live order (cards matched by id, then by face).",
    ),
    Deviation(
        "S02",
        "Certificate's first_hand_drawn runs after Cerulean Bell's drawn_to_hand, once a round",
        f"{_SMODS}#s02",
        Mode.RECONCILE,
        paths=_PLAYING,
        trigger=_s02_trigger,
        reconcile=_s02_reconcile,
        note="Cerulean Bell may force another card: the shadow takes the live hand and forced "
        "card. A second Certificate card vanilla makes after a consumable empties the hand "
        "can't be removed from the shadow: classified, the run stops.",
    ),
    Deviation(
        "S03",
        "Certificate's seal roll also draws from 'stdseal'..ante",
        f"{_SMODS}#s03",
        Mode.RECONCILE,
        paths=("pack.cards[*].modifier.seal",),
        trigger=_s03_trigger,
        observe=_s03_observe,
        reconcile=_s03_reconcile,
        note="Only Standard pack seals read that stream: in an ante where Certificate fired, a "
        "Standard pack's cards take the live seals.",
    ),
    Deviation(
        "S04",
        "Hand-size changes apply at once and refill the hand; The Manacle's disable draws 1",
        f"{_SMODS}#s04",
        Mode.RECONCILE,
        paths=_PLAYING,
        trigger=_s04_trigger,
        reconcile=_arrange(),
        note="The same cards, drawn from the same deck order: the shadow's hand and deck take "
        "the live split.",
    ),
    Deviation(
        "S04-GAME-OVER",
        "A lost round's hand-size changes: Steamodded applies them, vanilla's never run",
        f"{_SMODS}#s04",
        Mode.IGNORE,
        paths=("hand.limit",),
        trigger=_lost_round,
        note="end_round queues them (Turtle Bean's decay, a perishing Juggler) and the game pauses "
        "at GAME_OVER before they run (gaps.md G265); Steamodded's change_size has already "
        "changed the limit. The run is over: the difference is dropped.",
    ),
    Deviation(
        "S05",
        "copy_card prices the copy before copying abilities; a stripped Negative keeps its slot",
        f"{_SMODS}#s05",
        Mode.RECONCILE,
        paths=("jokers.cards[*].cost.buy", "jokers.cards[*].cost.sell", "jokers.limit"),
        trigger=_s05_trigger,
        accepts=_s05_accepts,
        observe=_s05_observe,
        reconcile=_s05_reconcile,
        note="After Invisible Joker or Ankh copies a joker, the copy's price and the joker "
        "slots take the live values (for the rest of the run: the slot follows the copy).",
    ),
    Deviation(
        "S06",
        "Seeing Double counts one Wild Card as every suit",
        f"{_SMODS}#s06",
        Mode.RECONCILE,
        paths=("round.chips",),
        trigger=_s06_trigger,
        reconcile=_set_chips,
        note="Only the hand's score differs (X2): the shadow takes the live chips. If the "
        "blind falls in one game only, the run stops (classified).",
    ),
    Deviation(
        "S07",
        "get_flush tries the suits in a different order",
        f"{_SMODS}#s07",
        Mode.VETO,
        veto=_s07_veto,
        note="Different cards score, with their own effects and rolls: not reconcilable, so "
        "plays with Four Fingers and three or more Wild Cards are not taken.",
    ),
    Deviation(
        "S08",
        "Sigil, Grim, Familiar and Incantation roll suits (and ranks) in another order",
        f"{_SMODS}#s08",
        Mode.RECONCILE,
        paths=_PLAYING,
        trigger=_s08_trigger,
        reconcile=_arrange(fronts=True, debuffs=True),
        note="The streams are drawn as often: the created or converted cards take the live "
        "fronts (Card:set_base) and the hand the live order.",
    ),
    Deviation(
        "S09",
        "Perishable and rental stickers run before the joker; debuffed jokers don't tick",
        f"{_SMODS}#s09",
        Mode.RECONCILE,
        paths=(
            "jokers.cards[*].modifier.perishable",
            "jokers.cards[*].state.debuff",
            "hand.limit",
        ),
        trigger=_s09_trigger,
        reconcile=_s09_reconcile,
        note="A perishable tally takes the live value. A joker that expires in one game only "
        "(its debuff, Turtle Bean's hand size) can't be copied: classified.",
    ),
    Deviation(
        "S10",
        "xMult is applied as m + m*(X-1): last-bit differences, +-1 chip",
        f"{_SMODS}#s10",
        Mode.RECONCILE,
        paths=("round.chips",),
        trigger=lambda ctx: ctx.method == "play",
        accepts=_s10_accepts,
        reconcile=_set_chips,
        note="A play's score within 1 chip of the live one: the shadow takes the live chips.",
    ),
    Deviation(
        "S11",
        "Unseeded pack variants: the first shop's Buffoon pack, Charm and Meteor mega packs",
        f"{_SMODS}#s11",
        Mode.IGNORE,
        paths=("packs.cards[*].key",),
        accepts=_s11_accepts,
        note="math.random(1, 2) picks the variant (_1 or _2); the contents are the same. The "
        "simulator always takes _1.",
    ),
    Deviation(
        "S12",
        "pseudoseed returns unseeded numbers while the game is paused",
        f"{_SMODS}#s12",
        Mode.IGNORE,
        note="BalatroBot never pauses the game: nothing to drop.",
    ),
    Deviation(
        "S13",
        "Crimson Heart with no joker left to pick: Steamodded re-picks, vanilla errors",
        f"{_SMODS}#s13",
        Mode.VETO,
        veto=_s13_veto,
        note="Plays and discards with Crimson Heart and every joker debuffed are not taken.",
    ),
    Deviation(
        "S14",
        "The deck's card limit follows the playing cards every frame",
        f"{_SMODS}#s14",
        Mode.IGNORE,
        paths=("cards.limit",),
        accepts=_s14_accepts,
        note="Vanilla only lowers G.deck.config.card_limit to #G.playing_cards: a card that joins "
        "outside the deck (Certificate, a Standard pick) leaves it. Only the view reads it.",
    ),
    Deviation(
        "B01",
        "The pack endpoint skips the UI's pick gates",
        f"{_SMODS}#b01",
        Mode.VETO,
        refusal=_refused(
            "vanilla greys this pick out", "Cannot select joker, joker slots are full"
        ),
        note="Enforced by the simulator's pack endpoint (NOT_ALLOWED) and BALATROBOT legality: "
        "greyed picks and a Negative joker at full slots are never sent.",
    ),
    Deviation(
        "B02",
        "play/discard/use under Cerulean Bell: the forced card is added, the last one dropped",
        f"{_SMODS}#b02",
        Mode.VETO,
        veto=_b02_veto,
        note="A request leaving out the forced card plays another hand than the action names "
        "(docs/balatrobot.md: `use` drops the forced card live, the simulator keeps it).",
    ),
    Deviation(
        "B03",
        "The debug add endpoint: the boss checks an added enhanced card at once",
        f"{_SMODS}#b03",
        Mode.RECONCILE,
        paths=("hand.cards[*].state.debuff", "cards.cards[*].state.debuff"),
        trigger=_b03_trigger,
        reconcile=_b03_reconcile,
        note="gaps.md B19: SMODS.create_card applies the enhancement with a second set_ability, "
        "which lets the boss debuff the card; the added card takes the live debuff.",
    ),
    Deviation(
        "B03-EDITION",
        "The debug add endpoint skips the edition poll when an edition is given",
        f"{_SMODS}#b03",
        Mode.RECONCILE,
        paths=(
            "shop.cards[*].modifier.edition",
            "shop.cards[*].cost.buy",
            "shop.cards[*].cost.sell",
            "pack.cards[*].modifier.edition",
            "pack.cards[*].cost.buy",
            "pack.cards[*].cost.sell",
        ),
        observe=_b03_edition_observe,
        trigger=lambda ctx: False,
        reconcile=_b03_edition_reconcile,
        note="gaps.md B15: the simulator still draws from 'edi'..ante, so after such an `add` "
        "(sticky for the run) a shop or pack card takes the live edition and prices. An owned "
        "card's edition can't be copied: classified.",
    ),
    Deviation(
        "B04",
        "BalatroBot's runtime settings and hooks",
        f"{_SMODS}#b04",
        Mode.IGNORE,
        note="Timing, speed and read-only hooks: no gameplay effect, nothing to drop.",
    ),
    Deviation(
        "BB-SIM-ONLY",
        "Sim-only moves: no endpoint, or vanilla allows what the bot refuses",
        "docs/balatrobot.md#command-space-policy",
        Mode.VETO,
        refusal=lambda ctx: ctx.refusal is not None and _vanilla_legal(ctx),
        note="RerollBoss, BuyAndUse, use/sell outside SELECTING_HAND and SHOP, a card-selecting "
        "consumable in the shop, Aura through `use`, a Negative joker bought at full slots, "
        "joker swaps at blind select and round eval. BALATROBOT legality leaves them out.",
    ),
    Deviation(
        "BB-ETERNAL",
        "Selling an eternal joker: BalatroBot allows it, vanilla doesn't",
        "docs/balatrobot.md#command-space-policy",
        Mode.VETO,
        refusal=_refused("the card is eternal"),
        note="Enforced by the simulator's sell endpoint (NOT_ALLOWED).",
    ),
    Deviation(
        "BB-SKIP-PACK",
        "skip with a tag that opens a pack: the live request never answers",
        "docs/balatrobot.md#differences-from-the-live-game",
        Mode.VETO,
        veto=_skip_pack_veto,
        note="The live skip waits for BLIND_SELECT while the tag's pack is open.",
    ),
    Deviation(
        "BB-NONFINITE",
        "A NaN or infinite number in the view: the live server sends nothing",
        "docs/balatrobot.md#the-view",
        Mode.VETO,
        refusal=_refused("Failed to encode response"),
        note="The shadow takes the step first; when its view can't be encoded the step is not "
        "sent (the live client would time out with the step taken).",
    ),
    Deviation(
        "BB-TEXT",
        "Text fields carry names in the simulator, localized descriptions live",
        "docs/balatrobot.md#the-view",
        Mode.IGNORE,
        paths=TEXT_FIELDS,
        compared=False,
        note="value.effect, a blind's effect and tag_effect, used_vouchers' descriptions: "
        "never compared.",
    ),
    Deviation(
        "BB-SHOP-AREAS",
        "The shop's areas in the view after leaving the shop",
        "docs/parity/gaps.md",
        Mode.IGNORE,
        paths=("shop", "vouchers", "packs"),
        trigger=_outside_shop,
        note="gaps.md B21: the simulator reports the removed areas (no cards, the last count) "
        "from a reading of the code. Outside the shop they carry nothing; a difference is "
        "counted, which settles B21.",
    ),
    Deviation(
        "BB-BOSS-DEFEAT",
        "The play that beats a boss answers before Blind:defeat has undone the boss",
        "docs/balatrobot.md#differences-from-the-live-game",
        Mode.IGNORE,
        paths=(
            "cards.cards[*].state.debuff",
            "jokers.cards[*].state.debuff",
            "jokers.cards[*].state.hidden",
            *_PASSIVES,
        ),
        trigger=_boss_beaten,
        accepts=_boss_defeat_accepts,
        note="play can answer before evaluate_round's Blind:defeat runs (from a run's second "
        "round, see BB-CASHOUT-RACE): The Manacle's hand size and Amber Acorn's face-down "
        "jokers are still there. Its set_blind(nil), which lifts the debuffs (and gives "
        "Crimson Heart's joker its passive back), runs in a non-blockable event 0.95 x 1.3 s "
        "after it (blind.lua:296-343). The next answer has them: only what the boss did is "
        "dropped.",
    ),
    Deviation(
        "BB-CASHOUT-RACE",
        "A cash_out sent as soon as play answers can pay the previous round's total",
        "docs/balatrobot.md#differences-from-the-live-game",
        Mode.RECONCILE,
        paths=("money",),
        trigger=lambda ctx: ctx.method == "cash_out",
        accepts=_cashout_race_accepts,
        observe=_payout_observe,
        reconcile=_set_money,
        note="From a run's second round on, play finds the last round's cash-out button and "
        "answers before the round's total is set (common_events.lua:1087); cash_out then pays "
        "the last total. LiveBackend waits before cash_out so it doesn't happen; when it does, "
        "the shadow takes the live money.",
    ),
    Deviation(
        "BB-REARRANGE-ORDER",
        "rearrange also writes ability.order and center.order",
        "docs/balatrobot.md#differences-from-the-live-game",
        Mode.IGNORE,
        note="No rule reads them and the view doesn't show them: nothing to drop.",
    ),
    Deviation(
        "BB-SORT-MODE",
        "SortHand goes out as a rearrange permutation; the sort mode stays live",
        "docs/balatrobot.md#command-space-policy",
        Mode.IGNORE,
        note="The shadow takes the bot's move (the permutation), so later draws are placed "
        "the same way in both: nothing to drop.",
    ),
)


def _vanilla_legal(ctx: StepContext) -> bool:
    """The action is in the shadow's vanilla legal list (a PlayHand / Discard by type)."""
    if ctx.action is None or ctx.pre is None:
        return False
    legal = ctx.pre.legal_actions()
    if type(ctx.action).__name__ in ("PlayHand", "Discard"):
        return any(type(a) is type(ctx.action) for a in legal)
    return ctx.action in legal


def by_id(catalog: Iterable[Deviation] = CATALOG) -> dict[str, Deviation]:
    return {d.id: d for d in catalog}


# ---------------------------------------------------------------------------
# Classifying a step's differences
# ---------------------------------------------------------------------------


@dataclass
class Classification:
    """A step's differences, by what the catalog makes of them."""

    ignored: dict[str, list[FieldDiff]] = field(default_factory=dict)
    reconcile: dict[str, list[FieldDiff]] = field(default_factory=dict)
    #: Explained by a veto entry whose predicate missed the action.
    classified: dict[str, list[FieldDiff]] = field(default_factory=dict)
    unclassified: list[FieldDiff] = field(default_factory=list)
    #: Every active entry (the candidates a bundle lists).
    active: list[str] = field(default_factory=list)


def classify(
    diffs: Iterable[FieldDiff], ctx: StepContext, catalog: Iterable[Deviation] = CATALOG
) -> Classification:
    """Sort a step's differences by the active catalog entries that explain them."""
    entries = [d for d in catalog if d.compared and d.active(ctx)]
    out = Classification(active=[d.id for d in entries])
    for diff in diffs:
        owners = [d for d in entries if d.explains(diff, ctx)]
        ignore = next((d for d in owners if d.mode is Mode.IGNORE), None)
        if ignore is not None:
            out.ignored.setdefault(ignore.id, []).append(diff)
            continue
        fix = next((d for d in owners if d.mode is Mode.RECONCILE), None)
        if fix is not None:
            out.reconcile.setdefault(fix.id, []).append(diff)
            continue
        if owners:
            out.classified.setdefault(owners[0].id, []).append(diff)
            continue
        out.unclassified.append(diff)
    return out


def triggered(ctx: StepContext, catalog: Iterable[Deviation] = CATALOG) -> list[str]:
    """The entries whose trigger fired for this step (sticky ones included)."""
    return [d.id for d in catalog if d.trigger is not None and d.active(ctx)]


def update_memory(ctx: StepContext, catalog: Iterable[Deviation] = CATALOG) -> None:
    """Record what later steps need: observe hooks, and sticky entries that fired."""
    for d in catalog:
        if d.observe is not None:
            d.observe(ctx)
        if d.sticky and d.trigger is not None and d.active(ctx):
            ctx.memory.sticky.add(d.id)


def veto_reason(ctx: StepContext, catalog: Iterable[Deviation] = CATALOG) -> tuple[str, str] | None:
    """``(entry id, reason)`` of the first veto entry that refuses this step."""
    for d in catalog:
        if d.mode is not Mode.VETO or d.veto is None:
            continue
        try:
            reason = d.veto(ctx)
        except (KeyError, IndexError, TypeError, ValueError, AssertionError):
            reason = None
        if reason:
            return d.id, reason
    return None


def refusal_entry(ctx: StepContext, catalog: Iterable[Deviation] = CATALOG) -> str | None:
    """The veto entry a shadow refusal falls under, if any."""
    for d in catalog:
        if d.mode is Mode.VETO and d.refusal is not None and d.refusal(ctx):
            return d.id
    return None

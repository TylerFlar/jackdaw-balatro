"""Field diff of two BalatroBot gamestate views: the live game's and the shadow run's.

Both views come out of BalatroBot's ``gamestate.lua`` (live) or its port
(``crates/jackdaw-bot/src/view.rs``) and go through the same rxi JSON encoder,
so every number is printed with ``%.14g`` on both sides and compares exactly.

- **Structure.** Every field is compared, present or absent, recursively. A
  Lua empty table is sent as ``[]`` whatever it holds, so ``[]`` and ``{}`` are
  equal. Booleans only equal booleans.
- **Text fields are not compared.** The live game fills them from localized
  descriptions, the simulator with names (see :data:`TEXT_FIELDS`).
- **Cards are matched one to one.** A card's ``id`` is its ``sort_id``, a
  per-run counter in the simulator and a session-wide one in the live game, so
  the two ids never agree. Cards are paired by position within their area and
  compared on content; :class:`IdMap` remembers each pairing (live id <-> sim
  id) and reports a pairing that contradicts an earlier one (a card that was
  matched to another card before), which content alone can't see.

Paths name fields the way the view nests them: ``money``, ``round.chips``,
``hand.cards[3].modifier.seal``, ``blinds.boss.name``, ``hands.Pair.level``.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


class _Absent:
    """A field one view has and the other doesn't."""

    _instance: _Absent | None = None

    def __new__(cls) -> _Absent:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "<absent>"

    def __reduce__(self) -> str:
        return "ABSENT"


ABSENT: Any = _Absent()

#: The card areas of a view (``extract_area``): compared card by card.
AREAS: tuple[str, ...] = (
    "jokers",
    "consumables",
    "cards",
    "hand",
    "shop",
    "vouchers",
    "packs",
    "pack",
)

#: Text fields: localized descriptions live, names in the simulator. Never compared
#: (deviation ``V-TEXT``, docs/balatrobot.md "Text fields carry names").
TEXT_FIELDS: tuple[str, ...] = (
    "*.cards[*].value.effect",
    "blinds.*.effect",
    "blinds.*.tag_effect",
    "used_vouchers.*",
)


def path_pattern(pattern: str) -> re.Pattern[str]:
    """A path glob as a regex: ``*`` is one key, ``[*]`` one index, ``**`` anything
    (including nothing), and a pattern also matches every path below it."""
    out = []
    i = 0
    while i < len(pattern):
        if pattern.startswith("**", i):
            out.append(".*")
            i += 2
        elif pattern.startswith("[*]", i):
            out.append(r"\[\d+\]")
            i += 3
        elif pattern[i] == "*":
            out.append(r"[^.\[\]]+")
            i += 1
        else:
            out.append(re.escape(pattern[i]))
            i += 1
    return re.compile("".join(out) + r"(?:[.\[].*)?")


def matches_any(path: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(p.fullmatch(path) for p in patterns)


_TEXT = tuple(path_pattern(p) for p in TEXT_FIELDS)


@dataclass(frozen=True)
class FieldDiff:
    """One field that differs: ``sim`` and ``live`` values (:data:`ABSENT` when
    missing). ``where`` names the cards for a card field."""

    path: str
    sim: Any
    live: Any
    where: str = ""

    def to_json(self) -> dict[str, Any]:
        out = {"path": self.path, "sim": _jsonable(self.sim), "live": _jsonable(self.live)}
        if self.where:
            out["where"] = self.where
        return out

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> FieldDiff:
        return cls(
            data["path"],
            _from_jsonable(data.get("sim")),
            _from_jsonable(data.get("live")),
            data.get("where", ""),
        )

    def __str__(self) -> str:
        where = f" ({self.where})" if self.where else ""
        return f"{self.path}: sim {_short(self.sim)}, live {_short(self.live)}{where}"


def _jsonable(v: Any) -> Any:
    return "<absent>" if v is ABSENT else v


def _from_jsonable(v: Any) -> Any:
    return ABSENT if v == "<absent>" else v


def _short(v: Any) -> str:
    text = repr(v)
    return text if len(text) <= 80 else text[:77] + "..."


@dataclass
class IdMap:
    """The one-to-one pairing of live card ids with sim card ids."""

    live_to_sim: dict[int, int] = field(default_factory=dict)
    sim_to_live: dict[int, int] = field(default_factory=dict)

    def conflict(self, live_id: Any, sim_id: Any) -> str | None:
        """Why pairing these ids contradicts an earlier pairing, or None."""
        known_sim = self.live_to_sim.get(live_id)
        if known_sim is not None and known_sim != sim_id:
            return f"live card #{live_id} was matched to sim card #{known_sim}"
        known_live = self.sim_to_live.get(sim_id)
        if known_live is not None and known_live != live_id:
            return f"sim card #{sim_id} was matched to live card #{known_live}"
        return None

    def commit(self, pairs: Iterable[tuple[int, int]]) -> None:
        for live_id, sim_id in pairs:
            self.live_to_sim[live_id] = sim_id
            self.sim_to_live[sim_id] = live_id

    def to_json(self) -> list[list[int]]:
        return [[lv, s] for lv, s in sorted(self.live_to_sim.items())]

    @classmethod
    def from_json(cls, pairs: Iterable[Iterable[int]]) -> IdMap:
        out = cls()
        out.commit((int(lv), int(s)) for lv, s in pairs)
        return out


@dataclass
class ViewDiff:
    """The result of :func:`diff_views`."""

    diffs: list[FieldDiff] = field(default_factory=list)
    #: (live id, sim id) of every positionally paired card.
    pairs: list[tuple[int, int]] = field(default_factory=list)
    #: area -> [(live card, sim card)] of the paired cards.
    cards: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = field(default_factory=dict)

    def __bool__(self) -> bool:
        return bool(self.diffs)


def _is_table(v: Any) -> bool:
    return isinstance(v, (dict, list))


def _as_dict(v: Any) -> dict[str, Any] | None:
    """A Lua table as a dict: ``[]`` is the empty table; a non-empty list is None."""
    if isinstance(v, dict):
        return v
    if isinstance(v, list) and not v:
        return {}
    return None


def _equal_scalar(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    return a == b


def _card_name(card: Any) -> str:
    if not isinstance(card, dict):
        return "?"
    return f"{card.get('key', '?')} #{card.get('id', '?')}"


def _area_summary(area: Any) -> Any:
    """An area one view lacks, as a diff value."""
    if isinstance(area, dict):
        return f"area with {len(area.get('cards') or [])} cards"
    return area


class _Differ:
    def __init__(self, ids: IdMap | None) -> None:
        self.ids = ids
        self.out = ViewDiff()

    def add(self, path: str, sim: Any, live: Any, where: str = "") -> None:
        self.out.diffs.append(FieldDiff(path, sim, live, where))

    def value(self, path: str, s: Any, lv: Any, where: str = "") -> None:
        # A text field's content isn't compared; whether both views have it is.
        if s is not ABSENT and lv is not ABSENT and matches_any(path, _TEXT):
            return
        if s is ABSENT or lv is ABSENT:
            if not (s is ABSENT and lv is ABSENT):
                self.add(path, s, lv, where)
            return
        if _is_table(s) and _is_table(lv):
            sd, ld = _as_dict(s), _as_dict(lv)
            if sd is not None and ld is not None:
                for k in sorted(set(sd) | set(ld), key=str):
                    self.value(f"{path}.{k}", sd.get(k, ABSENT), ld.get(k, ABSENT), where)
                return
            if isinstance(s, list) and isinstance(lv, list):
                for i in range(max(len(s), len(lv))):
                    a = s[i] if i < len(s) else ABSENT
                    b = lv[i] if i < len(lv) else ABSENT
                    self.value(f"{path}[{i}]", a, b, where)
                return
            self.add(path, s, lv, where)
            return
        if not _equal_scalar(s, lv):
            self.add(path, s, lv, where)

    def card(self, path: str, s: dict[str, Any], lv: dict[str, Any]) -> None:
        where = f"sim {_card_name(s)}, live {_card_name(lv)}"
        for k in sorted(set(s) | set(lv)):
            if k == "id":
                continue
            self.value(f"{path}.{k}", s.get(k, ABSENT), lv.get(k, ABSENT), where)
        live_id, sim_id = lv.get("id"), s.get("id")
        if self.ids is not None:
            why = self.ids.conflict(live_id, sim_id)
            if why is not None:
                self.add(f"{path}.id", sim_id, live_id, why)
        self.out.pairs.append((live_id, sim_id))

    def area(self, name: str, s: Any, lv: Any) -> None:
        if s is ABSENT or lv is ABSENT:
            self.add(name, _area_summary(s), _area_summary(lv))
            return
        sd, ld = _as_dict(s), _as_dict(lv)
        if sd is None or ld is None:
            self.value(name, s, lv)
            return
        for k in sorted(set(sd) | set(ld)):
            if k != "cards":
                self.value(f"{name}.{k}", sd.get(k, ABSENT), ld.get(k, ABSENT))
        sc = sd.get("cards") or []
        lc = ld.get("cards") or []
        pairs = self.out.cards.setdefault(name, [])
        for i in range(max(len(sc), len(lc))):
            path = f"{name}.cards[{i}]"
            if i >= len(sc) or i >= len(lc):
                a = sc[i] if i < len(sc) else ABSENT
                b = lc[i] if i < len(lc) else ABSENT
                self.add(
                    path,
                    a if a is ABSENT else _card_name(a),
                    b if b is ABSENT else _card_name(b),
                )
                continue
            if not (isinstance(sc[i], dict) and isinstance(lc[i], dict)):
                self.value(path, sc[i], lc[i])
                continue
            self.card(path, sc[i], lc[i])
            pairs.append((lc[i], sc[i]))


def diff_views(sim: dict[str, Any], live: dict[str, Any], ids: IdMap | None = None) -> ViewDiff:
    """Every field in which the live view differs from the sim view (text fields aside).

    With ``ids``, a card pairing that contradicts an earlier one is a difference too
    (path ``<area>.cards[i].id``); the pairings seen are returned for
    :meth:`IdMap.commit` once the step is accepted.
    """
    d = _Differ(ids)
    for key in sorted(set(sim) | set(live)):
        s, lv = sim.get(key, ABSENT), live.get(key, ABSENT)
        if key in AREAS:
            d.area(key, s, lv)
        else:
            d.value(key, s, lv)
    return d.out

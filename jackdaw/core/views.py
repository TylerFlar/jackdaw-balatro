"""Read-only views of a run: :attr:`Game.hand <jackdaw.core.Game.hand>`, ``jokers``, ``blind``...

Each view is built from one section of the canonical snapshot
(``docs/snapshot-schema.md``) that the Rust engine exports on its own, so
reading one is cheap compared with :meth:`Game.to_dict`, and its values are the
snapshot's.  Views are frozen: they describe the run when they were read.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class CardView:
    """A card: a playing card, joker, consumable, voucher or booster."""

    sort_id: int
    """The card's identity in the run (``G.sort_id`` when it was made)."""
    key: str
    """Its center: ``"c_base"`` or an enhancement (``"m_glass"``) for playing cards, else
    ``"j_joker"``, ``"c_fool"``, ``"v_overstock_norm"``, ``"p_arcana_normal_1"``..."""
    front: str | None
    """A playing card's front, ``"H_A"`` (suit letter, rank letter); else ``None``."""
    suit: str | None
    """``"Hearts"``, ``"Diamonds"``, ``"Clubs"`` or ``"Spades"`` for playing cards."""
    rank: str | None
    """``"2"`` .. ``"10"``, ``"Jack"``, ``"Queen"``, ``"King"``, ``"Ace"`` for playing cards."""
    edition: str | None
    """``"foil"``, ``"holo"``, ``"polychrome"``, ``"negative"`` or ``None``."""
    seal: str | None
    """``"Red"``, ``"Blue"``, ``"Gold"``, ``"Purple"`` or ``None``."""
    debuff: bool
    face_down: bool
    cost: float
    sell_cost: float
    eternal: bool
    perishable: bool
    perish_tally: float | None
    rental: bool
    pinned: bool
    ability: Mapping[str, Any]
    """``self.ability`` as the snapshot writes it (read-only)."""

    @classmethod
    def from_snapshot(cls, d: Mapping[str, Any]) -> CardView:
        base = d.get("base") or {}
        edition = d.get("edition")
        stickers = d.get("stickers") or {}
        costs = d.get("costs") or {}
        return cls(
            sort_id=int(d["sort_id"]),
            key=d["center"],
            front=d.get("front"),
            suit=base.get("suit"),
            rank=base.get("value"),
            edition=edition.get("type") if edition else None,
            seal=d.get("seal"),
            debuff=bool(d.get("debuff")),
            face_down=d.get("facing") == "back",
            cost=costs.get("cost", 0),
            sell_cost=costs.get("sell_cost", 0),
            eternal=bool(stickers.get("eternal")),
            perishable=bool(stickers.get("perishable")),
            perish_tally=stickers.get("perish_tally"),
            rental=bool(stickers.get("rental")),
            pinned=bool(stickers.get("pinned")),
            ability=MappingProxyType(dict(d.get("ability") or {})),
        )


@dataclass(frozen=True, slots=True)
class BlindView:
    """The blind being played (``G.GAME.blind``)."""

    key: str
    """``"bl_small"``, ``"bl_big"`` or a boss (``"bl_hook"``...)."""
    chips: float
    """The score to beat."""
    mult: float
    dollars: float
    """The reward for beating it."""
    boss: bool
    disabled: bool

    @classmethod
    def from_snapshot(cls, d: Mapping[str, Any]) -> BlindView:
        return cls(
            key=d["key"],
            chips=d["chips"],
            mult=d["mult"],
            dollars=d["dollars"],
            boss=bool(d["boss"]),
            disabled=bool(d["disabled"]),
        )


@dataclass(frozen=True, slots=True)
class HandLevel:
    """One poker hand's level and base score (``G.GAME.hands[name]``)."""

    level: float
    chips: float
    mult: float
    played: float
    visible: bool


@dataclass(frozen=True, slots=True)
class Score:
    """What a play scored (the snapshot's ``sim.last_score``)."""

    hand_type: str
    chips: float
    mult: float
    total: float
    dollars: float
    """The money the hand paid as it scored (seals, Lucky Cards, jokers)."""
    debuffed: bool
    """The boss refused the hand (The Psychic, The Eye...), so it scored nothing."""

    @classmethod
    def from_snapshot(cls, d: Mapping[str, Any]) -> Score:
        return cls(
            hand_type=d["hand_type"],
            chips=d["chips"],
            mult=d["mult"],
            total=d["total"],
            dollars=d["dollars"],
            debuffed=bool(d["debuffed"]),
        )


@dataclass(frozen=True, slots=True)
class ScoreEvent:
    """One step of a hand's scoring (:attr:`Game.last_play <jackdaw.core.Game.last_play>`)."""

    kind: str
    """``level_up``, ``hand``, ``blind``, ``card``, ``held``, ``edition``, ``joker``,
    ``retrigger``, ``debuffed``, ``not_allowed``, ``plasma`` or ``destroyed``
    (``docs/api.md``)."""
    card: int | None
    """The card the event belongs to (sort id), where the game shows it: the scoring or held
    card for its own bonus, the joker for a joker's effect."""
    target: int | None
    """The card it happens to, when that's another one: the card a joker acts on or a
    retrigger repeats, the joker Baseball Card boosts."""
    chips: float | None
    """Chips added."""
    mult: float | None
    """Mult added."""
    x_mult: float | None
    """Mult multiplied."""
    dollars: float | None
    """Money paid, after the hand."""
    level: float | None
    """The hand's level (``level_up``, ``hand``)."""
    swap: bool
    shattered: bool
    """The destroyed card was a Glass Card."""
    hand_chips: float
    """The hand's chips after the event."""
    hand_mult: float
    """The hand's mult after the event."""

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> ScoreEvent:
        return cls(
            kind=d["kind"],
            card=d.get("card"),
            target=d.get("target"),
            chips=d.get("chips"),
            mult=d.get("mult"),
            x_mult=d.get("x_mult"),
            dollars=d.get("dollars"),
            level=d.get("level"),
            swap=bool(d.get("swap", False)),
            shattered=bool(d.get("shattered", False)),
            hand_chips=d["hand_chips"],
            hand_mult=d["hand_mult"],
        )


@dataclass(frozen=True, slots=True)
class Play:
    """How a played hand scored, event by event (:attr:`Game.last_play
    <jackdaw.core.Game.last_play>`)."""

    hand: str
    level: float
    """The hand's level when it was played, before any level-up."""
    played: tuple[int, ...]
    """The played cards (sort ids), in play order."""
    scoring: tuple[int, ...]
    """The played cards that score."""
    events: tuple[ScoreEvent, ...]
    score: Score
    """What it scored: the last event's chips and mult, and their product."""

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> Play:
        return cls(
            hand=d["hand"],
            level=d["level"],
            played=tuple(d["played"]),
            scoring=tuple(d["scoring"]),
            events=tuple(ScoreEvent.from_json(e) for e in d["events"]),
            score=Score.from_snapshot({**d, "hand_type": d["hand"]}),
        )


def hand_levels(d: Mapping[str, Any]) -> Mapping[str, HandLevel]:
    """The ``hands`` section as a read-only mapping by hand name."""
    return MappingProxyType(
        {
            name: HandLevel(
                level=h["level"],
                chips=h["chips"],
                mult=h["mult"],
                played=h["played"],
                visible=bool(h["visible"]),
            )
            for name, h in d.items()
        }
    )

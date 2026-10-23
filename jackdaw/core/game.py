"""A run in the Rust engine, with Python actions and snapshots."""

from __future__ import annotations

import dataclasses
import enum
import json
import math
import random
from collections.abc import Mapping
from typing import Any

from jackdaw import _core
from jackdaw.core.views import BlindView, CardView, HandLevel, Score, hand_levels
from jackdaw.engine import actions as _actions
from jackdaw.engine.actions import GamePhase

IllegalActionError = _core.IllegalActionError
NotPortedError = _core.NotPortedError


class Legality(enum.StrEnum):
    """Which actions count as legal (``Game.legal_actions``).

    ``VANILLA`` is the game's own rules (Balatro 1.0.1o).  ``BALATROBOT`` keeps
    the vanilla actions that BalatroBot v1.5.2 also executes: the action's
    request passes the bot's argument, state and endpoint checks (run as a dry
    run in ``jackdaw-bot``).  See ``docs/balatrobot.md``.
    """

    VANILLA = "vanilla"
    BALATROBOT = "balatrobot"


_ACTION_TYPES: dict[str, type] = {
    cls.__name__: cls
    for cls in vars(_actions).values()
    if isinstance(cls, type)
    and dataclasses.is_dataclass(cls)
    and cls.__module__ == _actions.__name__
}


def encode_action(action: Any) -> list[Any]:
    """An action's tape form: ``["PlayHand", [0, 2, 3]]``. A list passes through."""
    if isinstance(action, list):
        return action
    out: list[Any] = [type(action).__name__]
    for f in dataclasses.fields(action):
        v = getattr(action, f.name)
        out.append(list(v) if isinstance(v, tuple) else v)
    return out


def decode_action(entry: list[Any]) -> Any:
    """The action dataclass for a tape entry."""
    cls = _ACTION_TYPES[entry[0]]
    return cls(*(tuple(v) if isinstance(v, list) else v for v in entry[1:]))


def random_seed(rng: random.Random | None = None) -> str:
    """A new seed the way vanilla's ``random_string(8)`` makes one
    (misc_functions.lua:270-277): digits 1-9, and letters without ``O``."""
    r = rng or random.SystemRandom()
    out = []
    for _ in range(8):
        if r.random() > 0.7:
            out.append(chr(r.randint(ord("1"), ord("9"))))
        elif r.random() > 0.45:
            out.append(chr(r.randint(ord("A"), ord("N"))))
        else:
            out.append(chr(r.randint(ord("P"), ord("Z"))))
    return "".join(out)


def _to_json(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return {"$f": "nan" if math.isnan(obj) else ("inf" if obj > 0 else "-inf")}
    if isinstance(obj, dict):
        return {k: _to_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json(v) for v in obj]
    return obj


def _from_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        if len(obj) == 1 and "$f" in obj:
            return float(obj["$f"])
        return {k: _from_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_from_json(v) for v in obj]
    return obj


class Game:
    """A Balatro run in the Rust engine.

    Create one with :meth:`new` or :meth:`from_dict`; advance it with
    :meth:`step`. Every step either completes or raises and leaves the game
    unchanged: :class:`IllegalActionError` for an action the rules refuse,
    :class:`NotPortedError` for game logic the Rust engine doesn't have yet.
    """

    __slots__ = ("_g",)

    def __init__(self, inner: _core.Game) -> None:
        self._g = inner

    @classmethod
    def new(
        cls,
        back: str = "b_red",
        stake: int = 1,
        seed: str | None = None,
        challenge: str | None = None,
    ) -> Game:
        """A new run at the blind-select screen. Without a seed, a random one."""
        return cls(_core.Game.new(back, stake, seed or random_seed(), challenge))

    @classmethod
    def from_dict(cls, snapshot: dict[str, Any]) -> Game:
        """The run a canonical snapshot describes."""
        return cls(_core.Game.from_snapshot(json.dumps(_to_json(snapshot))))

    def to_dict(self) -> dict[str, Any]:
        """The canonical snapshot (``jackdaw.snapshot/1``)."""
        return _from_json(json.loads(self._g.snapshot()))

    def step(self, action: Any) -> None:
        """Take one action (a dataclass from :mod:`jackdaw.engine.actions`, or its
        tape form) and play until the game waits for the next decision."""
        self._g.step(json.dumps(encode_action(action)))

    def legal_actions(self, legality: Legality | str = Legality.VANILLA) -> list[Any]:
        """The legal actions (dataclasses from :mod:`jackdaw.engine.actions`), in the
        Python engine's order. ``PlayHand`` / ``Discard`` appear once each, with no cards,
        standing for any selection; a consumable that targets cards carries its default
        targets."""
        text = self._g.legal_actions(Legality(legality).value)
        return [decode_action(a) for a in json.loads(text)]

    def bot_view(self) -> dict[str, Any]:
        """BalatroBot's ``gamestate`` view of the run (``jackdaw-bot``), as the bot
        sends it: text fields carry names, false flags and empty areas are left out."""
        return json.loads(self._g.bot_view())

    def bot_request(self, action: Any) -> tuple[str, dict[str, Any]] | None:
        """The BalatroBot request that performs ``action`` here: ``(method, params)``, or
        ``None`` for a sim-only action (``RerollBoss``, ``BuyAndUse``). ``SortHand`` and the
        swaps become ``rearrange`` permutations."""
        out = self._g.bot_translate(json.dumps(encode_action(action)))
        if out is None:
            return None
        method, params = out
        return method, json.loads(params) or {}  # an empty Lua table encodes as []

    def apply_setup(self, op: dict[str, Any]) -> None:
        """Apply one parity setup operation (``tests/parity/harness.py``)."""
        self._g.apply_setup(json.dumps(op))

    def debug_add(
        self,
        key: str,
        *,
        seal: str | None = None,
        edition: str | None = None,
        enhancement: str | None = None,
        eternal: bool = False,
        perishable: int | None = None,
        rental: bool = False,
    ) -> int:
        """BalatroBot's ``add`` with the engine's names (``seal="Red"``,
        ``edition="foil"``, ``enhancement="m_steel"``, ``perishable`` in rounds): a
        joker, consumable, voucher, booster or playing card (``"H_A"``). Returns the
        card's sort id. Debug only; see :mod:`jackdaw.engine.debug`."""
        return self._g.debug_add(key, seal, edition, enhancement, eternal, perishable, rental)

    def debug_set(self, **values: float) -> None:
        """BalatroBot's ``set``: ``money``, ``chips``, ``ante``, ``round``, ``hands``,
        ``discards``, written directly. Debug only."""
        for field, value in values.items():
            self._g.debug_set(field, float(value))

    def debug_restock_shop(self) -> None:
        """``set {shop: true}``: a new shop as on arrival (SHOP only). Debug only."""
        self._g.debug_restock_shop()

    def clone(self) -> Game:
        """An independent copy of the run."""
        return Game(self._g.clone())

    def freeze(self) -> Game:
        """A copy that later steps of this run don't change (the env hands out its live run;
        freeze it to keep a state)."""
        return Game(self._g.clone())

    def state_hash(self) -> str:
        """The SHA-256 hex digest of the canonical snapshot (``tests/parity/canon.py``)."""
        return self._g.state_hash()

    def observation_buffer(self) -> tuple[bytearray, tuple[int, int, int, int, int]]:
        """The current state's observation (encoder v1) as raw data: native-endian float32
        values (global context, then hand, joker, consumable, shop and pack rows) and the row
        counts. :func:`jackdaw.env.rust_observation.encode_game` wraps it as an
        ``Observation``."""
        return self._g.observation()

    @property
    def phase(self) -> GamePhase:
        return GamePhase(self._g.phase)

    @property
    def seed(self) -> str:
        return self._g.seed

    @property
    def ante(self) -> int:
        return int(self._g.ante)

    @property
    def round(self) -> int:
        return int(self._g.round)

    @property
    def dollars(self) -> int:
        return int(self._g.dollars)

    @property
    def chips(self) -> float:
        return self._g.chips

    @property
    def hands_left(self) -> int:
        return int(self._g.hands_left)

    @property
    def discards_left(self) -> int:
        return int(self._g.discards_left)

    @property
    def won(self) -> bool:
        return self._g.won

    # -- read-only views (jackdaw.core.views) -------------------------------------

    def _view(self, name: str) -> Any:
        return _from_json(json.loads(self._g.view(name)))

    def _cards(self, area: str) -> tuple[CardView, ...]:
        return tuple(CardView.from_snapshot(c) for c in self._view(area))

    @property
    def hand(self) -> tuple[CardView, ...]:
        """The cards in hand, in hand order."""
        return self._cards("hand")

    @property
    def jokers(self) -> tuple[CardView, ...]:
        return self._cards("jokers")

    @property
    def consumables(self) -> tuple[CardView, ...]:
        return self._cards("consumables")

    @property
    def deck(self) -> tuple[CardView, ...]:
        """The draw pile; cards are drawn from the end."""
        return self._cards("deck")

    @property
    def discard_pile(self) -> tuple[CardView, ...]:
        return self._cards("discard")

    @property
    def shop_jokers(self) -> tuple[CardView, ...]:
        """The shop's cards for sale (jokers, consumables, playing cards)."""
        return self._cards("shop_jokers")

    @property
    def shop_vouchers(self) -> tuple[CardView, ...]:
        return self._cards("shop_vouchers")

    @property
    def shop_boosters(self) -> tuple[CardView, ...]:
        return self._cards("shop_booster")

    @property
    def pack_cards(self) -> tuple[CardView, ...]:
        """The open pack's cards to choose from."""
        return self._cards("pack_cards")

    @property
    def blind(self) -> BlindView | None:
        """The blind being played, or ``None`` before the first one."""
        d = self._view("blind")
        return BlindView.from_snapshot(d) if d else None

    @property
    def hand_levels(self) -> Mapping[str, HandLevel]:
        """Each poker hand's level and base score, by name (``"Flush"``...)."""
        return hand_levels(self._view("hands"))

    @property
    def tags(self) -> tuple[str, ...]:
        """The held tags' keys (``"tag_double"``...)."""
        return tuple(self._view("tags"))

    @property
    def last_score(self) -> Score | None:
        """What the last play scored, or ``None`` before any play."""
        d = self._view("last_score")
        return Score.from_snapshot(d) if d else None

    def check(self, action: Any) -> str | None:
        """Why the rules refuse ``action`` here, or ``None`` when it is legal. The run is
        unchanged either way (the action is tried on a copy)."""
        trial = self._g.clone()
        try:
            trial.step(json.dumps(encode_action(action)))
        except IllegalActionError as exc:
            return str(exc)
        return None

    def preview_play(self, cards: Any) -> Score:
        """What playing these hand cards (indices) would score now: the real play on a copy,
        with hand evaluation, jokers, enhancements, editions, seals, the boss and the RNG. The
        run is unchanged. Raises :class:`IllegalActionError` for a selection the rules refuse."""
        trial = Game(self._g.clone())
        trial.step(_actions.PlayHand(card_indices=tuple(cards)))
        score = trial.last_score
        assert score is not None, "a play records its score"
        return score

    def __repr__(self) -> str:
        return (
            f"Game(seed={self.seed!r}, ante={self.ante}, round={self.round}, "
            f"phase={self.phase.value!r}, dollars={self.dollars})"
        )

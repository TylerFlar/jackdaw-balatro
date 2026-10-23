"""A stand-in for the live game: the simulator's BalatroBot session, made to deviate.

:class:`FakeLive` answers requests the way ``jackdaw serve`` does (a
``jackdaw._core.BotSession``), and can then be made to differ from the shadow
the way the modded live game would: renumber card ids (live ids are a
session-wide counter), change its own run after a request (``after``: a
Steamodded deviation, applied with ``Game.replace_state``), change only an
answer (``perturb``), or fail a request (``fail``).
"""

from __future__ import annotations

import copy
import json
import random
from collections.abc import Callable
from typing import Any

from jackdaw import _core
from jackdaw.bridge.backend import RPCError
from jackdaw.bridge.reconcile import front_base
from jackdaw.bridge.viewdiff import AREAS
from jackdaw.core import Game

After = Callable[[str, dict[str, Any], Game, dict[str, Any]], bool]
Perturb = Callable[[str, dict[str, Any], dict[str, Any]], dict[str, Any]]


def renumber(view: dict[str, Any], f: Callable[[int], int]) -> dict[str, Any]:
    """The view with every card id mapped through ``f``."""
    view = copy.deepcopy(view)
    for area in AREAS:
        a = view.get(area)
        if isinstance(a, dict):
            for card in a.get("cards") or []:
                card["id"] = f(card["id"])
    return view


def live_ids(i: int) -> int:
    """Ids like the live game's: larger, sparse, still in creation order."""
    return 7 * i + 1000


class FakeLive:
    """BalatroBot on the simulator, with hooks that make it deviate from the shadow."""

    def __init__(
        self,
        *,
        ids: Callable[[int], int] | None = live_ids,
        after: After | None = None,
        perturb: Perturb | None = None,
        fail: Callable[[str, dict[str, Any]], Exception | None] | None = None,
        game: Game | None = None,
    ) -> None:
        self.session = _core.BotSession() if game is None else _core.BotSession.from_game(game._g)
        self.ids = ids
        self.after = after
        self.perturb = perturb
        self.fail = fail
        self.calls: list[tuple[str, dict[str, Any]]] = []
        #: The last answer before ids were renumbered (what ``after`` sees as "before").
        self.previous: dict[str, Any] = {}

    def handle(self, method: str, params: dict[str, Any] | None) -> dict[str, Any]:
        params = dict(params or {})
        self.calls.append((method, params))
        if self.fail is not None:
            exc = self.fail(method, params)
            if exc is not None:
                raise exc
        ok, text = self.session.call(method, json.dumps(params))
        if not ok:
            err = json.loads(text)
            raise RPCError(err["code"], err["message"], {"name": err["name"]})
        view = json.loads(text)
        if self.after is not None and method not in ("gamestate", "health", "menu", "start"):
            inner = self.session.game()
            if inner is not None:
                game = Game(inner)
                if self.after(method, params, game, self.previous):
                    self.session = _core.BotSession.from_game(game._g)
                    view = json.loads(self.session.view())
        if method != "health":
            self.previous = view
        if self.ids is not None:
            view = renumber(view, self.ids)
        if self.perturb is not None:
            view = self.perturb(method, params, view)
        return view

    @property
    def game(self) -> Game:
        inner = self.session.game()
        assert inner is not None
        return Game(inner)


# ---------------------------------------------------------------------------
# Deviations a test can make the fake live game show
# ---------------------------------------------------------------------------


def owns(game: Game, key: str) -> bool:
    return any(c["key"] == key for c in game.bot_view()["jokers"]["cards"])


def reshuffle_round(seed: int = 1) -> After:
    """S01: after ``select`` with Marble Joker, the round is dealt from another order."""

    def after(method: str, params: dict[str, Any], game: Game, before: dict[str, Any]) -> bool:
        if method != "select" or not owns(game, "j_marble"):
            return False
        snap = game.to_dict()
        hand, deck = snap["areas"]["hand"], snap["areas"]["deck"]
        cards = hand + deck
        random.Random(seed).shuffle(cards)
        snap["areas"]["hand"], snap["areas"]["deck"] = cards[: len(hand)], cards[len(hand) :]
        game.replace_state(snap)
        return True

    return after


def resuit(created_only: bool, mapping: dict[str, str] | None = None) -> After:
    """S08: after Sigil (all hand cards) or Familiar/Grim/Incantation (the cards they
    made), the live game's cards have other suits."""
    swap = mapping or {"S": "H", "H": "S", "C": "D", "D": "C"}

    def after(method: str, params: dict[str, Any], game: Game, before: dict[str, Any]) -> bool:
        if method != "use":
            return False
        used = before["consumables"]["cards"][params["consumable"]]["key"]
        if used not in ("c_sigil", "c_familiar", "c_grim", "c_incantation"):
            return False
        old_ids = {c["id"] for a in ("hand", "cards") for c in before[a]["cards"]}
        snap = game.to_dict()
        for card in snap["areas"]["hand"]:
            new = card["sort_id"] not in old_ids
            if created_only and not new:
                continue
            suit, _, rank = card["front"].partition("_")
            front = f"{swap[suit]}_{rank}"
            card["front"] = front
            card["base"] = front_base(front, card["base"], created=new)
        game.replace_state(snap)
        return True

    return after


def score_off_by_one(method: str, params: dict[str, Any], game: Game, before: dict) -> bool:
    """S10: a play scores one chip more live (while the round goes on)."""
    if method != "play" or game.phase.value != "selecting_hand":
        return False
    game.debug_set(chips=game.chips + 1)
    return True

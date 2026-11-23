"""A game built from the player's view, kept across a run (:class:`ViewTracker`)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from jackdaw.core.game import (
    Game,
    IllegalActionError,
    NotPortedError,
    decode_action,
    encode_action,
)


def _factored_parts(action: Any) -> tuple[int, int | None, list[int] | None] | None:
    """``(action_type, entity_target, card_target)`` of a factored action (jackdaw.env's
    ``FactoredAction``, its mapping, or the tuple itself); ``None`` for anything else."""
    if hasattr(action, "action_type"):
        t = action.action_type
        e = getattr(action, "entity_target", None)
        cards = getattr(action, "card_target", None)
    elif isinstance(action, Mapping) and "action_type" in action:
        t, e, cards = action["action_type"], action.get("entity_target"), action.get("card_target")
    elif (
        isinstance(action, tuple)
        and len(action) == 3
        and isinstance(action[0], int)
        and not isinstance(action[0], bool)
    ):
        t, e, cards = action
    else:
        return None
    return int(t), None if e is None else int(e), None if cards is None else [int(c) for c in cards]


def engine_action(game: Game, action: Any) -> Any:
    """The engine action ``action`` stands for in ``game``: a factored action (``FactoredAction``,
    its mapping or ``(action_type, entity_target, card_target)``) as the environments read it
    there, and an engine action (a :mod:`jackdaw.engine.actions` dataclass or its tape form) as
    it is. Raises ``ValueError`` for a factored action with no engine action here."""
    parts = _factored_parts(action)
    if parts is None:
        return action
    return decode_action(json.loads(game._g.factored_to_engine(*parts)))


class ViewTracker:
    """A game built from the player's view, kept across a run: give it each view and each
    action taken, and it builds each view's game on the last one, stepped with the action since
    (:meth:`Game.from_player_view`'s shadow). That game knows what the run's actions taught
    and no view lists (the consumables used, the cards played this ante), and keeps each card a
    player follows under one sort id. One tracker per run, or :meth:`reset` between runs.

    The same code works with ``BalatroGymnasiumEnv`` and Jackdaw Arena's ``ArenaEnv``, which
    both have ``player_view()`` and ``step_factored()``::

        tracker = ViewTracker()
        obs, info = env.reset()
        game = tracker.update(env.player_view())
        while not done:
            action = plan(game)              # a factored action, or an engine action
            obs, reward, terminated, truncated, info = env.step_factored(action)
            done = terminated or truncated
            tracker.step(action)
            if not done:
                game = tracker.update(env.player_view())

    (The envs' ``view_game()`` does this for you.) ``seed`` is the seed of the first game's
    stand-ins; later games carry on its RNG.
    """

    def __init__(self, seed: str | None = None) -> None:
        self.seed = seed
        self._game: Game | None = None
        self._shadow: Game | None = None

    def reset(self) -> None:
        """Forget the run: the next :meth:`update` starts another."""
        self._game = self._shadow = None

    def update(self, view: Mapping[str, Any]) -> Game:
        """The game for ``view``: what the view shows, and the rest from the last game, stepped
        with the actions :meth:`step` gave since. A copy: step it as you like."""
        base = self._shadow if self._shadow is not None else self._game
        if base is not None and int(view["game"]["round"]) < base.round:
            base = None  # a new run
        try:
            game = Game.from_player_view(view, base, seed=self.seed if base is None else None)
        except ValueError:
            if base is None:
                raise
            # The shadow is another run's.
            game = Game.from_player_view(view, None, seed=self.seed)
        self._game, self._shadow = game, None
        return game.clone()

    def step(self, action: Any) -> None:
        """The action taken since the last :meth:`update`: a factored action (as the envs'
        ``step_factored`` takes it) or an engine action. The next game builds on the last one
        stepped with it; an action that game refuses (what it can't see differed) is dropped,
        and the next game builds on the last one as it was."""
        base = self._shadow if self._shadow is not None else self._game
        if base is None:
            raise RuntimeError("ViewTracker.step before update(): give it a view first")
        shadow = base.clone()
        try:
            shadow._g._step_tracked(json.dumps(encode_action(engine_action(shadow, action))))
        except (IllegalActionError, NotPortedError, ValueError):
            return
        self._shadow = shadow

    @property
    def game(self) -> Game | None:
        """The last game :meth:`update` built (a copy), or ``None`` before the first."""
        return None if self._game is None else self._game.clone()

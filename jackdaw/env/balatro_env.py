"""BalatroEnvironment: GameEnvironment implementation for Balatro.

Wraps a :class:`GameAdapter` together with observation encoding and action
masking behind the game-agnostic :class:`GameEnvironment` protocol.  This is
the bridge between the Balatro engine and the generic RL training loop.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from jackdaw.bridge.live import VetoedActionError
from jackdaw.core.game import IllegalActionError, NotPortedError
from jackdaw.env.action_space import (
    ACTION_VERSION,
    DEFAULT_MAX_FREE_MOVES,
    FREE_ACTION_TYPES,
    ActionMask,
    FactoredAction,
)
from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.game_interface import (
    DirectAdapter,
    GameAdapter,
    GameState,
    game_state_from_parts,
)
from jackdaw.env.game_spec import GameActionMask, GameObservation, GameSpec
from jackdaw.env.observation import Observation
from jackdaw.env.rust_env import action_mask_from_parts
from jackdaw.env.rust_env import factored_to_engine_action as rust_factored_to_engine
from jackdaw.env.rust_observation import observation_from_buffer


def _rust_view(
    parts: tuple[Any, ...],
) -> tuple[Observation, ActionMask, GameState, tuple[int, int, int]]:
    """``jackdaw._core``'s ``env_view`` as the env's Python objects."""
    obs_data, rows, mask_parts, state_parts, splits = parts
    return (
        observation_from_buffer(obs_data, rows),
        action_mask_from_parts(mask_parts),
        game_state_from_parts(state_parts),
        splits,
    )


class BalatroEnvironment:
    """Game-agnostic environment wrapper for Balatro.

    Implements the :class:`~jackdaw.env.game_spec.GameEnvironment` protocol
    so the generic RL pipeline can interact with Balatro without importing
    any Balatro-specific modules.

    The run is the Rust engine's: ``info["game"]`` is the adapter's live
    :class:`jackdaw.core.Game`, which the next step moves on (read it, or
    ``.freeze()`` a copy).  For before/after comparisons (rewards), use the
    :class:`~jackdaw.env.game_interface.GameState` snapshots in
    ``info["prev_state"]`` and ``info["state"]``.

    With :class:`~jackdaw.env.game_interface.DirectAdapter` (the default
    choice for training) a step is one call into ``jackdaw._core``.  Any other
    adapter with a ``game`` takes its own ``step``, then the env reads the run.

    A step the Rust engine can't take yet (``NotPortedError``) leaves the run
    unchanged and ends the episode as truncated, with the reason in
    ``info["not_ported"]``.  So does an action the engine refuses (``info["refused"]``),
    unless ``on_refusal`` is ``"raise"``.

    With :class:`~jackdaw.bridge.live.LiveAdapter` (the live game through
    BalatroBot) ``info["game"]`` is the adapter's shadow run and masks follow
    BALATROBOT legality (the adapter's ``legality``); a move the deviation
    catalog vetoes ends the episode as truncated, with the reason in
    ``info["vetoed"]``.

    Parameters
    ----------
    adapter_factory:
        Callable that creates a fresh :class:`GameAdapter` instance (one with a
        ``game``).
    back_keys:
        List of deck back keys to sample from on each reset.
    stakes:
        List of stakes to sample from on each reset.
    max_steps:
        Maximum steps before truncation.
    seed_prefix:
        Prefix for deterministic seed generation.
    end_on_win:
        End the episode (``terminated``) when the run is won, the way
        vanilla stops at its win screen.  False plays on into endless mode
        until the run is lost or truncated.
    max_free_moves:
        After this many free moves in a row (joker and hand swaps, hand sorts:
        :data:`~jackdaw.env.action_space.FREE_ACTION_TYPES`) the mask allows none
        until another action, so a policy can't stall on them.  ``None``: no cap.
    on_refusal:
        ``"truncate"`` (default) ends the episode on an action the engine refuses;
        ``"raise"`` raises its ``IllegalActionError``.

    :attr:`action_version` is the action space's
    :data:`~jackdaw.env.action_space.ACTION_VERSION`: save it with a trained policy.
    """

    action_version: int = ACTION_VERSION

    def __init__(
        self,
        adapter_factory: Callable[[], GameAdapter],
        back_keys: list[str] | None = None,
        stakes: list[int] | None = None,
        max_steps: int = 10_000,
        seed_prefix: str = "TRAIN",
        end_on_win: bool = True,
        max_free_moves: int | None = DEFAULT_MAX_FREE_MOVES,
        on_refusal: str = "truncate",
    ) -> None:
        if on_refusal not in ("truncate", "raise"):
            raise ValueError(f"on_refusal must be 'truncate' or 'raise', not {on_refusal!r}")
        if max_free_moves is not None and max_free_moves < 0:
            raise ValueError("max_free_moves must be None or at least 0")
        self.max_free_moves = max_free_moves
        self.on_refusal = on_refusal
        self._adapter_factory = adapter_factory
        self._adapter = adapter_factory()
        self._back_keys = back_keys or ["b_red"]
        self._stakes = stakes or [1]
        self._max_steps = max_steps
        self._seed_prefix = seed_prefix
        self._end_on_win = end_on_win
        self._spec = balatro_game_spec()

        self._episode_count = 0
        self._step_count = 0
        # Free moves since the last other action.
        self._free_moves = 0
        # The Rust engine's summary of the last state (the next step's prev_state).
        self._state: GameState | None = None

        # Episode tracking
        self.episode_length: int = 0
        self.episode_won: bool = False
        self.episode_ante: int = 1

    @property
    def spec(self) -> GameSpec:
        return self._spec

    def reset(
        self,
        **kwargs: object,
    ) -> tuple[GameObservation, GameActionMask, dict[str, object]]:
        """Reset the environment and return initial observation + mask."""
        seed = str(kwargs.get("seed", f"{self._seed_prefix}_{self._episode_count}"))
        self._episode_count += 1
        self._step_count = 0
        self._free_moves = 0
        self.episode_length = 0
        self.episode_won = False
        self.episode_ante = 1

        self._adapter = self._adapter_factory()
        back_key = (
            random.choice(self._back_keys) if len(self._back_keys) > 1 else self._back_keys[0]
        )
        stake = random.choice(self._stakes) if len(self._stakes) > 1 else self._stakes[0]
        self._adapter.reset(back_key, stake, seed)

        game = getattr(self._adapter, "game", None)
        if game is None:
            raise TypeError(
                f"{type(self._adapter).__name__} has no game: BalatroEnvironment plays a "
                "jackdaw.core.Game (DirectAdapter, LiveAdapter)"
            )
        obs, mask, state, splits = _rust_view(
            game._g.env_view(self._legality(), self._free_moves_allowed())
        )
        self._state = state
        info: dict[str, object] = {
            "game": game,
            "state": state,
            "shop_splits": splits,
            "observation": obs,
            "action_mask": mask,
        }
        return obs.to_game_observation(), _action_mask_to_game(mask), info

    def step(
        self,
        action: FactoredAction,
    ) -> tuple[GameObservation, bool, bool, GameActionMask, dict[str, object]]:
        """Execute an action and return (obs, terminated, truncated, mask, info).

        Convert, step, encode, mask and summarise in one call into ``jackdaw._core``
        (or through a custom adapter's ``step``)."""
        if self._state is None:
            raise RuntimeError("BalatroEnvironment: call reset() first")
        adapter = self._adapter
        game = adapter.game  # type: ignore[attr-defined]
        legality = self._legality()
        prev_state = self._state
        not_ported: str | None = None
        vetoed: str | None = None
        refused: str | None = None
        # The mask after this move counts it among the free moves in a row (or ends the run of
        # them); the count is kept once the step is through.
        free_moves = self._free_moves + 1 if action.action_type in FREE_ACTION_TYPES else 0
        free = self.max_free_moves is None or free_moves < self.max_free_moves
        try:
            if type(adapter) is DirectAdapter:
                parts = game._g.env_step(
                    action.action_type, action.entity_target, action.card_target, legality, free
                )
            else:
                adapter.step(rust_factored_to_engine(action, game))
                # Read the run again: an adapter may move its game to a new state object.
                parts = game._g.env_view(legality, free)
        except NotPortedError as exc:
            not_ported = str(exc)
            parts = game._g.env_view(legality, free)
        except VetoedActionError as exc:
            vetoed = str(exc)
            parts = game._g.env_view(legality, free)
        except IllegalActionError as exc:
            if self.on_refusal == "raise":
                raise
            refused = str(exc)
            parts = game._g.env_view(legality, free)
        obs, mask, state, splits = _rust_view(parts)
        self._free_moves = free_moves
        self._state = state
        self.episode_length += 1
        self._step_count += 1

        terminated = state.done or (self._end_on_win and state.won)
        truncated = (
            self._step_count >= self._max_steps
            or not_ported is not None
            or vetoed is not None
            or refused is not None
        )
        self.episode_ante = state.ante
        self.episode_won = state.won

        info: dict[str, object] = {
            "game": game,
            "prev_state": prev_state,
            "state": state,
            "shop_splits": splits,
            "observation": obs,
            "action_mask": mask,
        }
        if not_ported is not None:
            info["not_ported"] = not_ported
        if vetoed is not None:
            info["vetoed"] = vetoed
        if refused is not None:
            info["refused"] = refused
        return obs.to_game_observation(), terminated, truncated, _action_mask_to_game(mask), info

    def _free_moves_allowed(self) -> bool:
        """The mask may offer free moves: the cap on them in a row isn't reached."""
        return self.max_free_moves is None or self._free_moves < self.max_free_moves

    def _legality(self) -> str:
        """The legality the adapter's moves follow: ``balatrobot`` for an adapter that
        plays through BalatroBot (:class:`~jackdaw.bridge.live.LiveAdapter`)."""
        return str(getattr(self._adapter, "legality", "vanilla"))


def _action_mask_to_game(mask: ActionMask) -> GameActionMask:
    """Convert a Balatro ActionMask to a game-agnostic GameActionMask."""
    return GameActionMask(
        type_mask=mask.type_mask,
        card_mask=mask.card_mask,
        entity_masks=mask.entity_masks,
        min_card_select=mask.min_card_select,
        max_card_select=mask.max_card_select,
        forced_card=mask.forced_card,
    )

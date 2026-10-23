"""Gymnasium wrapper for BalatroEnvironment compatible with SB3's MaskablePPO.

Flattens the factored action space into a ``Discrete(MAX_ACTIONS)`` space by
enumerating all legal :class:`FactoredAction` instances each step.  Exposes an
``action_masks()`` method so ``sb3_contrib.MaskablePPO`` can mask illegal slots.

The reward is pluggable: pass ``reward_fn`` (see :mod:`jackdaw.env.rewards`).

With :class:`~jackdaw.env.game_interface.DirectAdapter` the whole step (the
engine, the observation, the mask, the action table and the built-in rewards)
runs in Rust (``jackdaw._core.GymEnv``); the Python engine's adapters take the
Python path, which the Rust one matches bit for bit.

Example::

    from jackdaw.env import BalatroGymnasiumEnv, DirectAdapter, ShapedReward

    env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, reward_fn=ShapedReward())
    obs, info = env.reset()
    mask = env.action_masks()
"""

from __future__ import annotations

import math
import operator
import secrets
from collections.abc import Callable, Sequence
from functools import lru_cache
from itertools import combinations
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from jackdaw import _core
from jackdaw.core import Game
from jackdaw.env.action_space import ActionType, get_consumable_target_info
from jackdaw.env.balatro_env import BalatroEnvironment
from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.game_interface import DirectAdapter, GameAdapter, game_state_from_parts
from jackdaw.env.game_spec import FactoredAction, GameActionMask, GameObservation
from jackdaw.env.rewards import RewardFn, ShapedReward, Transition, sparse_reward
from jackdaw.env.rust_gym import (
    EpisodeStarts,
    RustActionTable,
    factored_from_parts,
    gym_obs,
    rust_reward,
    sampler_seed,
)
from jackdaw.env.sampler import Xoshiro256, subsample

MAX_ACTIONS: int = 500
CARD_COMBO_BUDGET: int = 200

# Pre-compute entity layout from spec
_SPEC = balatro_game_spec()
_ENTITY_INFO: list[tuple[str, int, int]] = [
    (et.name, et.max_count, et.feature_dim) for et in _SPEC.entity_types
]
# Action types that take no targets at all
_SIMPLE_TYPES: frozenset[int] = frozenset(
    i
    for i, at in enumerate(_SPEC.action_types)
    if not at.needs_entity_target and not at.needs_card_select
)
# Entity-only action types (entity target, no card select)
_ENTITY_ONLY_TYPES: frozenset[int] = frozenset(
    i
    for i, at in enumerate(_SPEC.action_types)
    if at.needs_entity_target and not at.needs_card_select
)
# Entity + optional card select — the game_state list each entity indexes.
# A targeting Tarot/Spectral takes card targets; anything else takes none.
_TARGETED_ENTITY_SOURCES: dict[int, str] = {
    ActionType.UseConsumable: "consumables",
    ActionType.PickPackCard: "pack_cards",
}
# Card-only action types (card select, no entity) — PlayHand, Discard
_CARD_ONLY_TYPES: frozenset[int] = frozenset(
    i
    for i, at in enumerate(_SPEC.action_types)
    if at.needs_card_select and not at.needs_entity_target
)


def _subsample(items: Sequence[Any], budget: int, rng: Xoshiro256) -> Sequence[Any]:
    """Return *items* if within budget, else a random subsample (order kept)."""
    return subsample(items, budget, rng)


def _selection_sizes(n_cards: int, min_select: Any, max_select: Any) -> tuple[int, int]:
    """``(first, stop)`` of the selection-size range, validated like ``range()``."""
    upper = min(n_cards, max_select)
    lower = min(min_select, upper)
    return operator.index(lower), operator.index(upper + 1)


def _card_combos(
    legal_cards: tuple[int, ...], first: int, stop: int, required: int | None = None
) -> list[tuple[int, ...]]:
    """Enumerate all card-index combinations with sizes in ``range(first, stop)``.

    With *required* (Cerulean Bell's forced card), only the combinations
    that include it.
    """
    result: list[tuple[int, ...]] = []
    for k in range(first, stop):
        result.extend(combinations(legal_cards, k))
    if required is not None:
        result = [combo for combo in result if required in combo]
    return result


# FactoredAction is frozen, so identical actions can be shared across steps
# instead of being rebuilt (hundreds per step for PlayHand/Discard).  Only
# modest selection spaces are cached, which bounds the cache's memory
# (at most _CARD_ACTIONS_CACHE entries of _CACHE_MAX_COMBOS actions).
_CACHE_MAX_COMBOS = 1024
_CARD_ACTIONS_CACHE = 128


@lru_cache(maxsize=_CARD_ACTIONS_CACHE)
def _cached_card_actions(
    action_type: int,
    entity_target: int | None,
    legal_cards: tuple[int, ...],
    first: int,
    stop: int,
    required: int | None,
) -> tuple[FactoredAction, ...]:
    return tuple(
        FactoredAction(action_type=action_type, entity_target=entity_target, card_target=combo)
        for combo in _card_combos(legal_cards, first, stop, required)
    )


def _card_actions(
    action_type: int,
    entity_target: int | None,
    legal_cards: tuple[int, ...],
    min_select: Any,
    max_select: Any,
    rng: Xoshiro256,
    required: int | None = None,
) -> Sequence[FactoredAction]:
    """Card-selecting actions for one action type/entity, subsampled to budget.

    *required* restricts them to the selections that include that card.
    """
    n = len(legal_cards)
    first, stop = _selection_sizes(n, min_select, max_select)
    if sum(math.comb(n, k) for k in range(max(first, 0), stop)) <= _CACHE_MAX_COMBOS:
        actions = _cached_card_actions(
            action_type, entity_target, legal_cards, first, stop, required
        )
        return _subsample(actions, CARD_COMBO_BUDGET, rng)
    # Large selection spaces: sample the combinations first, then build only
    # the sampled actions (same draw: the population size is identical).
    combos = _subsample(_card_combos(legal_cards, first, stop, required), CARD_COMBO_BUDGET, rng)
    return [
        FactoredAction(action_type=action_type, entity_target=entity_target, card_target=combo)
        for combo in combos
    ]


@lru_cache(maxsize=1024)
def _target_action(action_type: int, entity_target: int | None = None) -> FactoredAction:
    """The single :class:`FactoredAction` with no card selection."""
    return FactoredAction(action_type=action_type, entity_target=entity_target)


class BalatroGymnasiumEnv(gymnasium.Env):
    """Gymnasium wrapper exposing a flat Discrete action space with action masking.

    Parameters
    ----------
    adapter_factory:
        Callable that creates a fresh :class:`GameAdapter`.
    back_keys, stakes, max_steps, seed_prefix, end_on_win:
        Forwarded to :class:`BalatroEnvironment`.  ``end_on_win`` (default
        True) ends the episode when the run is won; pass False to play on
        into endless mode.
    reward_fn:
        ``reward_fn(transition) -> float``, called once per step with a
        :class:`~jackdaw.env.rewards.Transition`.  If it has a ``reset()``
        method, that is called on every :meth:`reset`.  See
        :mod:`jackdaw.env.rewards` for the built-in examples and how to
        write your own.
    reward_shaping:
        Shorthand when ``reward_fn`` is not given: True selects
        :class:`~jackdaw.env.rewards.ShapedReward`, False (default)
        :func:`~jackdaw.env.rewards.sparse_reward`.

    On the Rust engine (:class:`~jackdaw.env.game_interface.DirectAdapter`) a step
    the engine can't take yet ends the episode as truncated, with the reason in
    ``info["balatro/not_ported"]``; :attr:`game` is a copy of the current run.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        adapter_factory: Callable[[], GameAdapter],
        back_keys: list[str] | None = None,
        stakes: list[int] | None = None,
        max_steps: int = 10_000,
        seed_prefix: str = "TRAIN",
        reward_shaping: bool = False,
        reward_fn: RewardFn | None = None,
        end_on_win: bool = True,
    ) -> None:
        super().__init__()
        if reward_fn is not None and reward_shaping:
            raise ValueError("Pass either reward_fn or reward_shaping=True, not both")
        if reward_fn is None:
            reward_fn = ShapedReward() if reward_shaping else sparse_reward
        self.reward_fn: RewardFn = reward_fn

        # The Rust engine runs the whole step in jackdaw._core.GymEnv; the Python
        # engine's adapters go through BalatroEnvironment.
        self._core: Any = None
        self._inner: BalatroEnvironment | None = None
        if isinstance(adapter_factory(), DirectAdapter):
            builtin = rust_reward(reward_fn)
            # A custom reward (or an override of _compute_reward) is computed in Python.
            self._python_reward = (
                builtin is None
                or type(self)._compute_reward is not BalatroGymnasiumEnv._compute_reward
            )
            kind, weights = ("none", None) if self._python_reward else builtin  # type: ignore[misc]
            self._starts = EpisodeStarts(back_keys, stakes, seed_prefix)
            back, stake, run_seed = self._starts.back_keys[0], self._starts.stakes[0], "INIT"
            self._core = _core.GymEnv(
                back, stake, run_seed, None, secrets.randbits(64), max_steps, end_on_win,
                kind, weights,
            )  # fmt: skip
        else:
            self._inner = BalatroEnvironment(
                adapter_factory=adapter_factory,
                back_keys=back_keys,
                stakes=stakes,
                max_steps=max_steps,
                seed_prefix=seed_prefix,
                end_on_win=end_on_win,
            )

        # Observation space
        obs_spaces: dict[str, spaces.Space] = {
            "global": spaces.Box(
                -np.inf, np.inf, shape=(_SPEC.global_feature_dim,), dtype=np.float32
            ),
        }
        max_counts: list[int] = []
        for name, max_count, feat_dim in _ENTITY_INFO:
            obs_spaces[name] = spaces.Box(
                -np.inf, np.inf, shape=(max_count, feat_dim), dtype=np.float32
            )
            max_counts.append(max_count)
        obs_spaces["entity_counts"] = spaces.Box(
            low=0,
            high=np.array(max_counts, dtype=np.float32),
            shape=(len(_ENTITY_INFO),),
            dtype=np.float32,
        )
        self.observation_space = spaces.Dict(obs_spaces)

        # Action space
        self.action_space = spaces.Discrete(MAX_ACTIONS)

        # Internal state.  The table's subsampling draws from a seeded xoshiro256**
        # (jackdaw.env.sampler), the generator the Rust table uses too.
        self._action_table: Sequence[FactoredAction] = []
        if self._core is not None:
            self._action_table = RustActionTable(self._core)
        self._rng = Xoshiro256(secrets.randbits(64))
        self._episode_max_ante: int = 1
        self._episode_blinds_beaten: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        if self._core is not None:
            return self._rust_reset(seed)
        if seed is not None:
            self._rng = Xoshiro256(seed)

        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = str(seed)

        assert self._inner is not None
        game_obs, game_mask, info = self._inner.reset(**kwargs)
        self._episode_max_ante = self._inner.episode_ante
        self._episode_blinds_beaten = 0
        reward_reset = getattr(self.reward_fn, "reset", None)
        if callable(reward_reset):
            reward_reset()
        self._action_table = self._enumerate_actions(game_mask, info)
        obs = self._build_obs(game_obs)
        return obs, {"action_mask": self.action_masks()}

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self._core is not None:
            return self._rust_step(action)
        assert self._inner is not None
        factored = self._action_table[action]
        game_obs, terminated, truncated, game_mask, info = self._inner.step(factored)

        transition = Transition.from_step(factored, terminated, truncated, info)
        reward = self._compute_reward(transition)

        # Episode stats, tracked whatever the reward function
        self._episode_max_ante = max(self._episode_max_ante, transition.curr.ante)
        if transition.blind_beaten:
            self._episode_blinds_beaten += 1

        # Rebuild action table for next step
        if not (terminated or truncated):
            self._action_table = self._enumerate_actions(game_mask, info)
        else:
            self._action_table = []

        obs = self._build_obs(game_obs)
        step_info: dict[str, Any] = {"action_mask": self.action_masks()}
        if terminated or truncated:
            step_info["balatro/ante_reached"] = self._episode_max_ante
            step_info["balatro/rounds_beaten"] = self._episode_blinds_beaten
            step_info["balatro/won"] = self._inner.episode_won
        return obs, reward, terminated, truncated, step_info

    def action_masks(self) -> np.ndarray:
        """Return bool mask of shape ``(MAX_ACTIONS,)`` for MaskablePPO."""
        mask = np.zeros(MAX_ACTIONS, dtype=bool)
        mask[: len(self._action_table)] = True
        return mask

    @property
    def game(self) -> Game:
        """A copy of the current run (Rust engine only)."""
        if self._core is None:
            raise AttributeError("the Python engine's run is env._inner._adapter.raw_state")
        return Game(self._core.game())

    # ------------------------------------------------------------------
    # The Rust engine
    # ------------------------------------------------------------------

    def _rust_reset(self, seed: int | None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        back, stake, run_seed = self._starts.next(None if seed is None else str(seed))
        rng_seed = None if seed is None else sampler_seed(seed)
        data, _ = self._core.reset(back, stake, run_seed, None, rng_seed)
        self._episode_max_ante = 1
        self._episode_blinds_beaten = 0
        reward_reset = getattr(self.reward_fn, "reset", None)
        if callable(reward_reset):
            reward_reset()
        return gym_obs(data), {"action_mask": self.action_masks()}

    def _rust_step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        data, reward, terminated, truncated, _, episode, not_ported = self._core.step(int(action))
        if self._python_reward:
            prev, curr, fa = self._core.last_transition()
            transition = Transition(
                prev=game_state_from_parts(prev),
                curr=game_state_from_parts(curr),
                action=factored_from_parts(fa),
                terminated=terminated,
                truncated=truncated,
                game=Game(self._core.game()),
            )
            reward = self._compute_reward(transition)
        info: dict[str, Any] = {"action_mask": self.action_masks()}
        if terminated or truncated:
            ante, rounds, won = episode
            self._episode_max_ante = ante
            self._episode_blinds_beaten = rounds
            info["balatro/ante_reached"] = ante
            info["balatro/rounds_beaten"] = rounds
            info["balatro/won"] = won
            if not_ported is not None:
                info["balatro/not_ported"] = not_ported
        return gym_obs(data), reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_reward(self, transition: Transition) -> float:
        """Reward for one step.  Delegates to :attr:`reward_fn`; prefer
        passing a ``reward_fn`` over overriding this."""
        return float(self.reward_fn(transition))

    def _build_obs(self, game_obs: GameObservation) -> dict[str, np.ndarray]:
        obs: dict[str, np.ndarray] = {
            "global": game_obs.global_context.astype(np.float32),
        }
        counts: list[int] = []
        for name, max_count, feat_dim in _ENTITY_INFO:
            arr = game_obs.entities.get(name)
            padded = np.zeros((max_count, feat_dim), dtype=np.float32)
            if arr is not None and arr.shape[0] > 0:
                n = min(arr.shape[0], max_count)
                padded[:n] = arr[:n]
                counts.append(n)
            else:
                counts.append(0)
            obs[name] = padded
        obs["entity_counts"] = np.array(counts, dtype=np.float32)
        return obs

    def _enumerate_actions(
        self, mask: GameActionMask, info: dict[str, Any]
    ) -> list[FactoredAction]:
        actions: list[FactoredAction] = []
        type_mask = mask.type_mask
        legal_cards: tuple[int, ...] | None = None

        for t in range(len(type_mask)):
            if not type_mask[t]:
                continue

            # --- Simple (no targets) ---
            if t in _SIMPLE_TYPES:
                actions.append(_target_action(t))

            # --- Entity-only ---
            elif t in _ENTITY_ONLY_TYPES:
                if t in mask.entity_masks:
                    for idx in np.nonzero(mask.entity_masks[t])[0].tolist():
                        actions.append(_target_action(t, idx))

            # --- Card-only (PlayHand / Discard) ---
            elif t in _CARD_ONLY_TYPES:
                if legal_cards is None:
                    legal_cards = tuple(np.nonzero(mask.card_mask)[0].tolist())
                min_select, max_select = mask.min_card_select, mask.max_card_select
                actions.extend(
                    _card_actions(
                        t, None, legal_cards, min_select, max_select, self._rng, mask.forced_card
                    )
                )

            # --- Entity + optional cards (UseConsumable / PickPackCard) ---
            elif t in _TARGETED_ENTITY_SOURCES:
                if t in mask.entity_masks:
                    legal_entities = np.nonzero(mask.entity_masks[t])[0].tolist()
                    area = _TARGETED_ENTITY_SOURCES[t]
                    game = info.get("game")  # a Rust run (LiveAdapter's shadow)
                    source = info.get("raw_state", {}).get(area, [])
                    for e_idx in legal_entities:
                        if game is not None:
                            try:
                                target_info = game._g.consumable_target_info(area, e_idx)
                            except ValueError:  # past the end of the area
                                continue
                        elif e_idx < len(source):
                            target_info = get_consumable_target_info(source[e_idx])
                        else:
                            continue
                        min_cards, max_cards, needs = target_info
                        if needs:
                            if legal_cards is None:
                                legal_cards = tuple(np.nonzero(mask.card_mask)[0].tolist())
                            actions.extend(
                                _card_actions(
                                    t, e_idx, legal_cards, min_cards, max_cards, self._rng
                                )
                            )
                        else:
                            actions.append(_target_action(t, e_idx))

        # Subsample to budget if needed
        if len(actions) > MAX_ACTIONS:
            actions = _subsample(actions, MAX_ACTIONS, self._rng)

        return actions

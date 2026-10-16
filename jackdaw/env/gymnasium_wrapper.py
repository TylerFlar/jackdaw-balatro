"""Gymnasium wrapper for BalatroEnvironment compatible with SB3's MaskablePPO.

Flattens the factored action space into a ``Discrete(MAX_ACTIONS)`` space by
enumerating all legal :class:`FactoredAction` instances each step.  Exposes an
``action_masks()`` method so ``sb3_contrib.MaskablePPO`` can mask illegal slots.

The reward is pluggable: pass ``reward_fn`` (see :mod:`jackdaw.env.rewards`).

Example::

    from jackdaw.env import BalatroGymnasiumEnv, DirectAdapter, ShapedReward

    env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, reward_fn=ShapedReward())
    obs, info = env.reset()
    mask = env.action_masks()
"""

from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces

from jackdaw.env.action_space import ActionType, get_consumable_target_info
from jackdaw.env.balatro_env import BalatroEnvironment
from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.game_interface import GameAdapter
from jackdaw.env.game_spec import FactoredAction, GameActionMask, GameObservation
from jackdaw.env.rewards import RewardFn, ShapedReward, Transition, sparse_reward

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


def _subsample(items: list[Any], budget: int, rng: np.random.Generator) -> list[Any]:
    """Return *items* if within budget, else a random subsample."""
    if len(items) <= budget:
        return items
    indices = rng.choice(len(items), size=budget, replace=False)
    return [items[i] for i in sorted(indices)]


def _card_combos(
    legal_cards: np.ndarray,
    min_select: int,
    max_select: int,
) -> list[tuple[int, ...]]:
    """Enumerate all card-index combinations within the selection range."""
    upper = min(len(legal_cards), max_select)
    lower = min(min_select, upper)
    result: list[tuple[int, ...]] = []
    for k in range(lower, upper + 1):
        result.extend(tuple(int(c) for c in combo) for combo in combinations(legal_cards, k))
    return result


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
        self._inner = BalatroEnvironment(
            adapter_factory=adapter_factory,
            back_keys=back_keys,
            stakes=stakes,
            max_steps=max_steps,
            seed_prefix=seed_prefix,
            end_on_win=end_on_win,
        )
        if reward_fn is not None and reward_shaping:
            raise ValueError("Pass either reward_fn or reward_shaping=True, not both")
        if reward_fn is None:
            reward_fn = ShapedReward() if reward_shaping else sparse_reward
        self.reward_fn: RewardFn = reward_fn

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

        # Internal state
        self._action_table: list[FactoredAction] = []
        self._rng = np.random.default_rng()
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
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = str(seed)

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

        for t in range(len(type_mask)):
            if not type_mask[t]:
                continue

            # --- Simple (no targets) ---
            if t in _SIMPLE_TYPES:
                actions.append(FactoredAction(action_type=t))

            # --- Entity-only ---
            elif t in _ENTITY_ONLY_TYPES:
                if t in mask.entity_masks:
                    legal = np.nonzero(mask.entity_masks[t])[0]
                    for idx in legal:
                        actions.append(FactoredAction(action_type=t, entity_target=int(idx)))

            # --- Card-only (PlayHand / Discard) ---
            elif t in _CARD_ONLY_TYPES:
                legal_cards = np.nonzero(mask.card_mask)[0]
                combos = _card_combos(legal_cards, mask.min_card_select, mask.max_card_select)
                combos = _subsample(combos, CARD_COMBO_BUDGET, self._rng)
                for combo in combos:
                    actions.append(FactoredAction(action_type=t, card_target=combo))

            # --- Entity + optional cards (UseConsumable / PickPackCard) ---
            elif t in _TARGETED_ENTITY_SOURCES:
                if t in mask.entity_masks:
                    legal_entities = np.nonzero(mask.entity_masks[t])[0]
                    source = info.get("raw_state", {}).get(_TARGETED_ENTITY_SOURCES[t], [])
                    for e_idx in legal_entities:
                        e_idx_int = int(e_idx)
                        if e_idx_int < len(source):
                            card = source[e_idx_int]
                            min_cards, max_cards, needs = get_consumable_target_info(card)
                            if needs:
                                legal_cards = np.nonzero(mask.card_mask)[0]
                                combos = _card_combos(legal_cards, min_cards, max_cards)
                                combos = _subsample(combos, CARD_COMBO_BUDGET, self._rng)
                                for combo in combos:
                                    actions.append(
                                        FactoredAction(
                                            action_type=t,
                                            entity_target=e_idx_int,
                                            card_target=combo,
                                        )
                                    )
                            else:
                                actions.append(
                                    FactoredAction(action_type=t, entity_target=e_idx_int)
                                )

        # Subsample to budget if needed
        if len(actions) > MAX_ACTIONS:
            actions = _subsample(actions, MAX_ACTIONS, self._rng)

        return actions

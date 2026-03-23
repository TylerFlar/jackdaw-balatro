"""Thin environment wrapper for factored-policy training.

Unlike ``BalatroGymnasiumEnv`` which flattens actions into ``Discrete(500)``,
this wrapper returns the raw ``ActionMask`` and accepts ``FactoredAction``
directly.  No action enumeration overhead.

**Shop index remapping**: The observation encodes all shop items as a single
concatenated array (shop_cards + shop_vouchers + shop_boosters), but the
engine's entity masks use sub-list indices.  This wrapper remaps entity masks
to use global shop_item indices on output, and remaps actions back to sub-list
indices on input, so the pointer network always operates in observation space.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from jackdaw.engine.game import IllegalActionError
from jackdaw.env.action_space import ActionType, get_consumable_target_info
from jackdaw.env.balatro_env import BalatroEnvironment
from jackdaw.env.balatro_spec import balatro_game_spec
from jackdaw.env.game_interface import GameAdapter
from jackdaw.env.game_spec import FactoredAction, GameActionMask, GameObservation

_SPEC = balatro_game_spec()
_ENTITY_INFO: list[tuple[str, int, int]] = [
    (et.name, et.max_count, et.feature_dim) for et in _SPEC.entity_types
]

# Shop sub-list action types and their offsets within the concatenated
# shop_item observation array: shop_cards[0:n_cards] + vouchers[n_cards:n_cards+n_vouchers]
# + boosters[n_cards+n_vouchers:]
_SHOP_ACTION_TYPES = {
    ActionType.BuyCard: "cards",
    ActionType.RedeemVoucher: "vouchers",
    ActionType.OpenBooster: "boosters",
}

SHOP_ITEM_MAX = _SPEC.entity_types[3].max_count  # 10


def _remap_shop_masks(
    game_mask: GameActionMask,
    shop_splits: tuple[int, int, int],
) -> GameActionMask:
    """Remap shop sub-list entity masks to global shop_item indices.

    BuyCard indices 0..n_cards-1   → global 0..n_cards-1
    RedeemVoucher indices 0..n_v-1 → global n_cards..n_cards+n_v-1
    OpenBooster indices 0..n_b-1   → global n_cards+n_v..n_cards+n_v+n_b-1
    """
    n_cards, n_vouchers, n_boosters = shop_splits
    total = n_cards + n_vouchers + n_boosters

    new_entity_masks = dict(game_mask.entity_masks)

    offsets = {
        ActionType.BuyCard: 0,
        ActionType.RedeemVoucher: n_cards,
        ActionType.OpenBooster: n_cards + n_vouchers,
    }

    for atype, offset in offsets.items():
        if atype not in new_entity_masks:
            continue
        sub_mask = new_entity_masks[atype]
        global_mask = np.zeros(total, dtype=bool)
        n = min(len(sub_mask), total - offset)
        global_mask[offset : offset + n] = sub_mask[:n]
        new_entity_masks[atype] = global_mask

    return GameActionMask(
        type_mask=game_mask.type_mask,
        card_mask=game_mask.card_mask,
        entity_masks=new_entity_masks,
        min_card_select=game_mask.min_card_select,
        max_card_select=game_mask.max_card_select,
    )


def _unmap_shop_action(
    action: FactoredAction,
    shop_splits: tuple[int, int, int],
) -> FactoredAction:
    """Convert global shop_item entity_target back to sub-list index."""
    at = action.action_type
    et = action.entity_target

    if at not in _SHOP_ACTION_TYPES or et is None:
        return action

    n_cards, n_vouchers, _ = shop_splits

    offsets = {
        ActionType.BuyCard: 0,
        ActionType.RedeemVoucher: n_cards,
        ActionType.OpenBooster: n_cards + n_vouchers,
    }
    offset = offsets.get(at)
    if offset is None:
        return action
    local = et - offset

    return FactoredAction(
        action_type=at,
        card_target=action.card_target,
        entity_target=max(0, local),
    )


def _fix_consumable_targets(
    action: FactoredAction,
    raw_state: dict[str, Any],
) -> FactoredAction:
    """Validate and fix card targets for UseConsumable actions.

    The card selection head may produce the wrong number of targets for a
    given consumable.  This function adjusts the targets to satisfy the
    consumable's requirements so the engine doesn't reject the action.
    """
    if action.action_type != ActionType.UseConsumable:
        return action
    et = action.entity_target
    if et is None:
        return action

    consumables = raw_state.get("consumables", [])
    if et >= len(consumables):
        return action

    card = consumables[et]
    min_cards, max_cards, needs_targets = get_consumable_target_info(card)

    if not needs_targets:
        # Consumable doesn't need card targets — strip them
        if action.card_target is not None:
            return FactoredAction(
                action_type=action.action_type,
                entity_target=et,
                card_target=None,
            )
        return action

    # Consumable needs card targets — validate count
    ct = action.card_target or ()
    hand = raw_state.get("hand", [])
    # Filter to valid indices
    valid = tuple(i for i in ct if i < len(hand))

    if len(valid) < min_cards:
        # Too few — pad with random legal hand card indices
        available = [i for i in range(len(hand)) if i not in valid]
        needed = min_cards - len(valid)
        extra = available[:needed]
        valid = tuple(sorted(valid + tuple(extra)))

    if len(valid) > max_cards:
        valid = valid[:max_cards]

    if len(valid) < min_cards:
        # Still not enough cards available — can't satisfy, strip targets
        return FactoredAction(
            action_type=action.action_type,
            entity_target=et,
            card_target=None,
        )

    return FactoredAction(
        action_type=action.action_type,
        entity_target=et,
        card_target=valid,
    )


class FactoredBalatroEnv:
    """Wraps :class:`BalatroEnvironment` for factored-policy training.

    Returns padded observation dicts and remapped :class:`GameActionMask`
    where shop-targeting actions use global ``shop_item`` indices matching
    the observation array.

    Parameters
    ----------
    adapter_factory:
        Callable that creates a fresh :class:`GameAdapter`.
    reward_shaping:
        If True, use the same dense multi-signal reward as the gymnasium wrapper.
    back_keys, stakes, max_steps, seed_prefix:
        Forwarded to :class:`BalatroEnvironment`.
    """

    def __init__(
        self,
        adapter_factory: Callable[[], GameAdapter],
        reward_shaping: bool = True,
        back_keys: list[str] | None = None,
        stakes: list[int] | None = None,
        max_steps: int = 10_000,
        seed_prefix: str = "TRAIN",
    ) -> None:
        self._inner = BalatroEnvironment(
            adapter_factory=adapter_factory,
            back_keys=back_keys,
            stakes=stakes,
            max_steps=max_steps,
            seed_prefix=seed_prefix,
        )
        self._reward_shaping = reward_shaping
        self._shop_splits: tuple[int, int, int] = (0, 0, 0)
        self.max_episode_steps = 500
        self._step_count = 0

        # Reward-shaping trackers
        self._prev_ante: int = 1
        self._prev_round: int = 0
        self._prev_chips: int = 0
        self._episode_max_ante: int = 1
        self._episode_max_round: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, **kwargs: Any) -> tuple[dict[str, np.ndarray], GameActionMask, dict[str, Any]]:
        """Reset and return (obs_dict, action_mask, info)."""
        game_obs, game_mask, info = self._inner.reset(**kwargs)
        self._step_count = 0
        self._prev_ante = self._inner.episode_ante
        self._prev_round = 0
        self._prev_chips = 0
        self._episode_max_ante = 1
        self._episode_max_round = 0
        self._shop_splits = info.get("shop_splits", (0, 0, 0))
        game_mask = _remap_shop_masks(game_mask, self._shop_splits)
        obs = self._build_obs(game_obs)
        return obs, game_mask, info

    def step(
        self, action: FactoredAction
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, GameActionMask, dict[str, Any]]:
        """Execute action. Returns (obs, reward, terminated, truncated, mask, info)."""
        # Remap global shop index → sub-list index for the engine
        engine_action = _unmap_shop_action(action, self._shop_splits)
        # Fix UseConsumable card targets to match consumable requirements
        engine_action = _fix_consumable_targets(engine_action, self._inner._adapter.raw_state)
        try:
            game_obs, terminated, truncated, game_mask, info = self._inner.step(engine_action)
        except IllegalActionError:
            # Rare: masked action was sampled via float-precision leak.
            # Return current (unchanged) state with a small penalty.
            game_obs, game_mask, info = self._inner.reobserve()
            self._shop_splits = info.get("shop_splits", (0, 0, 0))
            game_mask = _remap_shop_masks(game_mask, self._shop_splits)
            obs = self._build_obs(game_obs)
            return obs, -0.1, False, False, game_mask, info
        self._shop_splits = info.get("shop_splits", (0, 0, 0))
        game_mask = _remap_shop_masks(game_mask, self._shop_splits)

        self._step_count += 1
        if not terminated and self._step_count >= self.max_episode_steps:
            truncated = True

        reward = self._compute_reward(info, terminated, truncated)
        obs = self._build_obs(game_obs)

        if terminated or truncated:
            info["balatro/ante_reached"] = self._episode_max_ante
            info["balatro/rounds_beaten"] = self._episode_max_round
            info["balatro/won"] = self._inner.episode_won

        return obs, reward, terminated, truncated, game_mask, info

    @property
    def episode_won(self) -> bool:
        return self._inner.episode_won

    @property
    def episode_ante(self) -> int:
        return self._inner.episode_ante

    @property
    def episode_length(self) -> int:
        return self._inner.episode_length

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

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

    def _compute_reward(self, info: dict[str, Any], terminated: bool, truncated: bool) -> float:
        """Same reward logic as gymnasium_wrapper._compute_reward."""
        if not self._reward_shaping:
            if terminated or truncated:
                return 1.0 if self._inner.episode_won else -1.0
            return 0.0

        gs: dict[str, Any] = info.get("raw_state", {})
        phase = gs.get("phase")

        reward = -0.002 if phase == "shop" else -0.001

        ante = gs.get("round_resets", {}).get("ante", 1)
        round_num = gs.get("round", 0)
        chips = gs.get("chips", 0)
        ante_scale = ante / 8.0

        if round_num > self._prev_round:
            reward += 0.15 * max(ante / 8, 0.5)
            if ante > self._prev_ante:
                reward += 0.1 * ante_scale
            hands_left = gs.get("current_round", {}).get("hands_left", 0)
            reward += 0.01 * hands_left

        blind = gs.get("blind")
        blind_target = getattr(blind, "chips", 0) if blind is not None else 0
        if blind_target > 0 and chips > self._prev_chips:
            chip_delta = chips - self._prev_chips
            reward += 0.02 * min(chip_delta / blind_target, 1.0)

        if terminated or truncated:
            reward += 0.5 if self._inner.episode_won else -0.2

        self._prev_round = round_num
        self._prev_ante = ante
        self._prev_chips = chips
        self._episode_max_ante = max(self._episode_max_ante, ante)
        self._episode_max_round = max(self._episode_max_round, round_num)

        return reward

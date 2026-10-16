"""Tests for the Gymnasium wrapper around BalatroEnvironment."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from jackdaw.env.action_space import ActionType
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.gymnasium_wrapper import MAX_ACTIONS, BalatroGymnasiumEnv
from jackdaw.env.rewards import ShapedReward, Transition, sparse_reward


@pytest.fixture()
def env() -> BalatroGymnasiumEnv:
    return BalatroGymnasiumEnv(adapter_factory=DirectAdapter, max_steps=200)


# ------------------------------------------------------------------
# Space definitions
# ------------------------------------------------------------------


class TestSpaces:
    def test_action_space_is_discrete(self, env: BalatroGymnasiumEnv) -> None:
        assert isinstance(env.action_space, spaces.Discrete)
        assert env.action_space.n == MAX_ACTIONS

    def test_observation_space_keys(self, env: BalatroGymnasiumEnv) -> None:
        assert isinstance(env.observation_space, spaces.Dict)
        expected = {
            "global",
            "hand_card",
            "joker",
            "consumable",
            "shop_item",
            "pack_card",
            "entity_counts",
        }
        assert set(env.observation_space.spaces.keys()) == expected


# ------------------------------------------------------------------
# Reset
# ------------------------------------------------------------------


class TestReset:
    def test_reset_returns_obs_and_info(self, env: BalatroGymnasiumEnv) -> None:
        obs, info = env.reset(seed=42)
        assert isinstance(obs, dict)
        assert isinstance(info, dict)

    def test_obs_matches_observation_space(self, env: BalatroGymnasiumEnv) -> None:
        obs, _ = env.reset(seed=42)
        assert env.observation_space.contains(obs)

    def test_action_mask_in_info(self, env: BalatroGymnasiumEnv) -> None:
        _, info = env.reset(seed=42)
        assert "action_mask" in info
        mask = info["action_mask"]
        assert mask.shape == (MAX_ACTIONS,)
        assert mask.dtype == bool

    def test_at_least_one_legal_action(self, env: BalatroGymnasiumEnv) -> None:
        _, info = env.reset(seed=42)
        assert info["action_mask"].any()


# ------------------------------------------------------------------
# Step
# ------------------------------------------------------------------


class TestStep:
    def test_step_returns_five_tuple(self, env: BalatroGymnasiumEnv) -> None:
        _, info = env.reset(seed=42)
        action = int(np.nonzero(info["action_mask"])[0][0])
        result = env.step(action)
        assert len(result) == 5

    def test_step_obs_matches_space(self, env: BalatroGymnasiumEnv) -> None:
        _, info = env.reset(seed=42)
        action = int(np.nonzero(info["action_mask"])[0][0])
        obs, reward, terminated, truncated, step_info = env.step(action)
        if not (terminated or truncated):
            assert env.observation_space.contains(obs)

    def test_step_reward_is_float(self, env: BalatroGymnasiumEnv) -> None:
        _, info = env.reset(seed=42)
        action = int(np.nonzero(info["action_mask"])[0][0])
        _, reward, *_ = env.step(action)
        assert isinstance(reward, float)

    def test_mid_episode_reward_is_zero(self, env: BalatroGymnasiumEnv) -> None:
        _, info = env.reset(seed=42)
        action = int(np.nonzero(info["action_mask"])[0][0])
        _, reward, terminated, truncated, _ = env.step(action)
        if not (terminated or truncated):
            assert reward == 0.0


# ------------------------------------------------------------------
# Action masks
# ------------------------------------------------------------------


class TestActionMasks:
    def test_action_masks_method_exists(self, env: BalatroGymnasiumEnv) -> None:
        assert callable(getattr(env, "action_masks", None))

    def test_action_masks_shape_and_dtype(self, env: BalatroGymnasiumEnv) -> None:
        env.reset(seed=42)
        mask = env.action_masks()
        assert mask.shape == (MAX_ACTIONS,)
        assert mask.dtype == bool

    def test_padding_slots_are_false(self, env: BalatroGymnasiumEnv) -> None:
        env.reset(seed=42)
        mask = env.action_masks()
        n_legal = len(env._action_table)
        assert mask[:n_legal].all()
        assert not mask[n_legal:].any()

    def test_action_table_within_budget(self, env: BalatroGymnasiumEnv) -> None:
        env.reset(seed=42)
        assert len(env._action_table) <= MAX_ACTIONS


# ------------------------------------------------------------------
# Integration: random episodes
# ------------------------------------------------------------------


class TestRandomEpisodes:
    @pytest.mark.slow
    def test_ten_random_episodes(self) -> None:
        env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, max_steps=500)
        rng = np.random.default_rng(123)

        for ep in range(10):
            obs, info = env.reset(seed=ep)
            assert env.observation_space.contains(obs)
            mask = info["action_mask"]

            for _ in range(500):
                legal = np.nonzero(mask)[0]
                assert len(legal) > 0, "No legal actions but episode not done"
                action = int(rng.choice(legal))
                obs, reward, terminated, truncated, info = env.step(action)
                mask = info["action_mask"]
                if terminated or truncated:
                    break
            assert terminated or truncated, f"Episode {ep} did not finish in 500 steps"

    def test_single_episode_completes(self, env: BalatroGymnasiumEnv) -> None:
        rng = np.random.default_rng(0)
        obs, info = env.reset(seed=0)
        mask = info["action_mask"]

        for _ in range(200):
            legal = np.nonzero(mask)[0]
            if len(legal) == 0:
                break
            action = int(rng.choice(legal))
            obs, reward, terminated, truncated, info = env.step(action)
            mask = info["action_mask"]
            if terminated or truncated:
                break


class TestPackPickTargets:
    def test_targeting_pick_enumerates_each_selection(self) -> None:
        """A pack Tarot that works on hand cards is offered once per legal
        selection, so the agent chooses which dealt card it hits."""
        from jackdaw.engine.actions import GamePhase, SelectBlind
        from jackdaw.engine.card_factory import create_consumable
        from jackdaw.engine.game import step
        from jackdaw.env.action_space import get_action_mask

        env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter)
        env.reset(seed=5)
        gs = env._inner._adapter.raw_state
        step(gs, SelectBlind())  # deal a hand to target
        gs["phase"] = GamePhase.PACK_OPENING
        gs["pack_type"] = "Arcana"
        gs["pack_cards"] = [create_consumable("c_justice")]  # exactly 1 target
        gs["pack_choices_remaining"] = 1
        env._action_table = env._enumerate_actions(get_action_mask(gs), {"raw_state": gs})

        picks = [fa for fa in env._action_table if fa.action_type == ActionType.PickPackCard]
        assert sorted(fa.card_target for fa in picks) == [(i,) for i in range(len(gs["hand"]))]

        chosen = gs["hand"][3]
        env.step(env._action_table.index(next(fa for fa in picks if fa.card_target == (3,))))
        assert chosen.center_key == "m_glass"


# ------------------------------------------------------------------
# Rewards
# ------------------------------------------------------------------


def _step_beating_every_blind(env: BalatroGymnasiumEnv) -> tuple[float, bool, bool, dict]:
    """Take one step down the main line (select, play, cash out, leave the
    shop), shrinking each blind's target so the first hand beats it."""
    table = env._action_table
    for wanted in (
        ActionType.SelectBlind,
        ActionType.CashOut,
        ActionType.NextRound,
        ActionType.SkipPack,
    ):
        for i, fa in enumerate(table):
            if fa.action_type == wanted:
                _, reward, terminated, truncated, info = env.step(i)
                return reward, terminated, truncated, info
    env._inner._adapter.raw_state["blind"].chips = 1
    play = next(i for i, fa in enumerate(table) if fa.action_type == ActionType.PlayHand)
    _, reward, terminated, truncated, info = env.step(play)
    return reward, terminated, truncated, info


class TestRewardFn:
    def test_default_is_sparse(self) -> None:
        env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter)
        assert env.reward_fn is sparse_reward

    def test_reward_shaping_selects_shaped_reward(self) -> None:
        env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, reward_shaping=True)
        assert isinstance(env.reward_fn, ShapedReward)

    def test_reward_fn_and_reward_shaping_conflict(self) -> None:
        with pytest.raises(ValueError):
            BalatroGymnasiumEnv(
                adapter_factory=DirectAdapter, reward_shaping=True, reward_fn=sparse_reward
            )

    def test_custom_reward_fn_sees_each_transition(self) -> None:
        seen: list[Transition] = []

        def reward_fn(t: Transition) -> float:
            seen.append(t)
            return 7.0

        env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, reward_fn=reward_fn)
        env.reset(seed=1)
        reward, *_ = _step_beating_every_blind(env)  # SelectBlind
        assert reward == 7.0
        [t] = seen
        assert t.action.action_type == ActionType.SelectBlind
        assert t.prev.phase.value == "blind_select"
        assert t.curr.phase.value == "selecting_hand"

    def test_reward_fn_reset_called_on_env_reset(self) -> None:
        class Stateful:
            resets = 0

            def reset(self) -> None:
                self.resets += 1

            def __call__(self, t: Transition) -> float:
                return 0.0

        fn = Stateful()
        env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, reward_fn=fn)
        env.reset(seed=1)
        env.reset(seed=2)
        assert fn.resets == 2

    def test_blind_bonus_paid_on_the_clearing_hand_not_the_select(self) -> None:
        env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, reward_shaping=True)
        env.reset(seed=3)
        select_reward, *_ = _step_beating_every_blind(env)
        clear_reward, *_ = _step_beating_every_blind(env)
        assert select_reward == pytest.approx(-0.001)
        assert clear_reward > 0.05

    @pytest.mark.parametrize("reward_shaping", [False, True])
    def test_metrics_count_blinds_beaten_for_any_reward(self, reward_shaping: bool) -> None:
        env = BalatroGymnasiumEnv(
            adapter_factory=DirectAdapter, max_steps=30, reward_shaping=reward_shaping
        )
        env.reset(seed=4)
        for _ in range(30):
            _, terminated, truncated, info = _step_beating_every_blind(env)
            if terminated or truncated:
                break
        assert truncated
        # 4 steps per blind (select, play, cash out, leave shop): 30 steps
        # beat 8 blinds — two full antes plus two blinds of ante 3.
        assert info["balatro/rounds_beaten"] == 8
        assert info["balatro/ante_reached"] == 3

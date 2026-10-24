"""Tests for the Gymnasium wrapper around BalatroEnvironment.

Most run on ``DirectAdapter`` (the whole step in Rust); the ones that set up a state by
editing the run use :class:`tests.env.generic.GenericAdapter`, whose steps go through
``BalatroEnvironment`` and the wrapper's Python action table.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from jackdaw.core import Game
from jackdaw.engine.actions import CashOut, GamePhase, OpenBooster, PlayHand, SelectBlind
from jackdaw.env.action_space import ActionType, get_action_mask
from jackdaw.env.balatro_env import _action_mask_to_game
from jackdaw.env.game_interface import DirectAdapter
from jackdaw.env.gymnasium_wrapper import MAX_ACTIONS, BalatroGymnasiumEnv
from jackdaw.env.rewards import ShapedReward, Transition, sparse_reward
from tests.env.generic import GenericAdapter


@pytest.fixture(params=[DirectAdapter, GenericAdapter], ids=["direct", "generic"])
def env(request: pytest.FixtureRequest) -> BalatroGymnasiumEnv:
    return BalatroGymnasiumEnv(adapter_factory=request.param, max_steps=200)


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

    def test_game_is_a_copy_of_the_run(self, env: BalatroGymnasiumEnv) -> None:
        env.reset(seed=42)
        copy = env.game
        assert isinstance(copy, Game) and copy.phase == GamePhase.BLIND_SELECT
        copy.step(SelectBlind())
        assert env.game.phase == GamePhase.BLIND_SELECT


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
    @pytest.mark.parametrize("adapter", [DirectAdapter, GenericAdapter])
    def test_ten_random_episodes(self, adapter: type) -> None:
        env = BalatroGymnasiumEnv(adapter_factory=adapter, max_steps=500)
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
        env = BalatroGymnasiumEnv(adapter_factory=GenericAdapter)
        env.reset(seed=5)
        game = env._inner._adapter.game
        _open_arcana_pack(game)
        _make_pack(game, "c_justice")  # exactly 1 target
        _rebuild_table(env, game)

        hand = game.hand
        picks = [fa for fa in env._action_table if fa.action_type == ActionType.PickPackCard]
        assert sorted(fa.card_target for fa in picks) == [(i,) for i in range(len(hand))]

        chosen = hand[3].sort_id
        env.step(env._action_table.index(next(fa for fa in picks if fa.card_target == (3,))))
        card = next(c for c in game.deck + game.hand if c.sort_id == chosen)
        assert card.key == "m_glass"


class TestCeruleanBell:
    """Every play and discard must include the Bell's forced card; the
    table used to offer ones without it, which the engine rejects (#31)."""

    @staticmethod
    def _bell_env(seed: int = 7) -> BalatroGymnasiumEnv:
        env = BalatroGymnasiumEnv(adapter_factory=GenericAdapter)
        env.reset(seed=seed)
        game = env._inner._adapter.game
        game.apply_setup({"op": "force_boss", "key": "bl_final_bell"})
        game.step(SelectBlind())
        _rebuild_table(env, game)
        return env

    def test_selections_are_exactly_those_with_the_forced_card(self) -> None:
        from itertools import combinations

        env = self._bell_env()
        game = env._inner._adapter.game
        forced = get_action_mask(game).forced_card
        assert forced is not None
        others = [i for i in range(len(game.hand)) if i != forced]
        expected = sorted(
            tuple(sorted((forced, *rest))) for k in range(5) for rest in combinations(others, k)
        )
        for t in (ActionType.PlayHand, ActionType.Discard):
            got = sorted(fa.card_target for fa in env._action_table if fa.action_type == t)
            assert got == expected

    def test_random_policy_plays_through_the_bell(self) -> None:
        env = self._bell_env()
        game = env._inner._adapter.game
        rng = np.random.default_rng(0)
        mask = env.action_masks()
        selections = 0
        for _ in range(500):
            if game.phase != GamePhase.SELECTING_HAND:
                break
            action = int(rng.choice(np.flatnonzero(mask)))
            if env._action_table[action].action_type in (ActionType.PlayHand, ActionType.Discard):
                selections += 1
            *_, info = env.step(action)  # used to raise IllegalActionError
            mask = info["action_mask"]
        assert game.phase != GamePhase.SELECTING_HAND
        assert selections > 1


# ------------------------------------------------------------------
# Rewards
# ------------------------------------------------------------------


def _step_beating_every_blind(env: BalatroGymnasiumEnv) -> tuple[float, bool, bool, dict]:
    """Take one step down the main line (select, play, cash out, leave the
    shop), setting each round's score so the first hand beats the blind."""
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
    game = env._inner._adapter.game
    game.debug_set(chips=game.blind.chips)
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

    @pytest.mark.parametrize("adapter", [DirectAdapter, GenericAdapter])
    def test_custom_reward_fn_sees_each_transition(self, adapter: type) -> None:
        seen: list[Transition] = []

        def reward_fn(t: Transition) -> float:
            seen.append(t)
            return 7.0

        env = BalatroGymnasiumEnv(adapter_factory=adapter, reward_fn=reward_fn)
        env.reset(seed=1)
        select = next(i for i, fa in enumerate(env._action_table) if fa.action_type == 2)
        _, reward, *_ = env.step(select)
        assert reward == 7.0
        [t] = seen
        assert t.action.action_type == ActionType.SelectBlind
        assert t.prev.phase.value == "blind_select"
        assert t.curr.phase.value == "selecting_hand"
        assert isinstance(t.game, Game) and t.game.round == 1

    @pytest.mark.parametrize("adapter", [DirectAdapter, GenericAdapter])
    def test_reward_fn_reset_called_on_env_reset(self, adapter: type) -> None:
        class Stateful:
            resets = 0

            def reset(self) -> None:
                self.resets += 1

            def __call__(self, t: Transition) -> float:
                return 0.0

        fn = Stateful()
        env = BalatroGymnasiumEnv(adapter_factory=adapter, reward_fn=fn)
        env.reset(seed=1)
        env.reset(seed=2)
        assert fn.resets == 2

    def test_blind_bonus_paid_on_the_clearing_hand_not_the_select(self) -> None:
        env = BalatroGymnasiumEnv(adapter_factory=GenericAdapter, reward_shaping=True)
        env.reset(seed=3)
        select_reward, *_ = _step_beating_every_blind(env)
        clear_reward, *_ = _step_beating_every_blind(env)
        assert select_reward == pytest.approx(-0.001)
        assert clear_reward > 0.05

    @pytest.mark.parametrize("reward_shaping", [False, True])
    def test_metrics_count_blinds_beaten_for_any_reward(self, reward_shaping: bool) -> None:
        env = BalatroGymnasiumEnv(
            adapter_factory=GenericAdapter, max_steps=30, reward_shaping=reward_shaping
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


# ------------------------------------------------------------------
# Helpers: states the policy doesn't reach on its own
# ------------------------------------------------------------------


def _open_arcana_pack(game: Game) -> None:
    """Beat the first blind, cash out and open an Arcana pack (its hand is dealt)."""
    game.step(SelectBlind())
    game.debug_set(chips=game.blind.chips)
    game.step(PlayHand(card_indices=(0,)))
    game.step(CashOut())
    game.debug_add("p_arcana_normal_1")
    game.debug_set(money=50)
    game.step(OpenBooster(card_index=len(game.shop_boosters) - 1))
    assert game.phase == GamePhase.PACK_OPENING and game.hand


def _make_pack(game: Game, key: str) -> None:
    """Make the open pack's only card ``key``: a card BalatroBot's ``add`` makes in a copy
    of the run, put in the pack under the first pack card's sort id."""
    scratch = game.clone()
    made = scratch.debug_add(key)
    snap = game.to_dict()
    card = next(c for c in scratch.to_dict()["areas"]["consumables"] if c["sort_id"] == made)
    card["sort_id"] = snap["areas"]["pack_cards"][0]["sort_id"]
    snap["areas"]["pack_cards"] = [card]
    game.replace_state(snap)


def _rebuild_table(env: BalatroGymnasiumEnv, game: Game) -> None:
    """The Gym table of the edited run, as the next step would build it."""
    mask = _action_mask_to_game(get_action_mask(game))
    env._action_table = env._enumerate_actions(mask, {"game": game})

"""Tests for the agent protocol and built-in agents.

Covers:
- RandomAgent completes episodes without errors
- Agent protocol compliance
"""

from __future__ import annotations

from jackdaw.core import Game
from jackdaw.engine.actions import CashOut, GamePhase, OpenBooster, PlayHand, SelectBlind
from jackdaw.env.action_space import (
    ActionType,
    FactoredAction,
    factored_to_engine_action,
    get_action_mask,
)
from jackdaw.env.agents import Agent, RandomAgent
from jackdaw.env.game_interface import DirectAdapter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = "TEST_AGENTS_42"
BACK = "b_red"
STAKE = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_agent_episode(
    agent: Agent,
    seed: str = SEED,
    max_steps: int = 5000,
) -> dict:
    """Run one episode with the given agent. Returns the episode's outcome."""
    adapter = DirectAdapter()
    agent.reset()
    adapter.reset(BACK, STAKE, seed)

    actions = 0
    for _ in range(max_steps):
        if adapter.done:
            break
        if adapter.won and adapter.game.phase == GamePhase.SHOP:
            break

        game = adapter.game
        legal = adapter.get_legal_actions()
        if not legal:
            break

        mask = get_action_mask(game)
        info = {"game": game, "legal_actions": legal}
        fa = agent.act({}, mask, info)

        adapter.step(factored_to_engine_action(fa, game))
        actions += 1

    return {"actions": actions, "won": adapter.won, "done": adapter.done}


def _arcana_pack(key: str) -> Game:
    """A run in an Arcana pack (its hand dealt) whose only card is ``key``."""
    game = Game.new(BACK, STAKE, SEED)
    game.step(SelectBlind())
    game.debug_set(chips=game.blind.chips)
    game.step(PlayHand(card_indices=(0,)))
    game.step(CashOut())
    game.debug_add("p_arcana_normal_1")
    game.debug_set(money=50)
    game.step(OpenBooster(card_index=len(game.shop_boosters) - 1))
    scratch = game.clone()
    made = scratch.debug_add(key)
    snap = game.to_dict()
    card = next(c for c in scratch.to_dict()["areas"]["consumables"] if c["sort_id"] == made)
    card["sort_id"] = snap["areas"]["pack_cards"][0]["sort_id"]
    snap["areas"]["pack_cards"] = [card]
    return Game.from_dict(snap)


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_random_agent_satisfies_protocol(self):
        assert isinstance(RandomAgent(), Agent)


# ---------------------------------------------------------------------------
# RandomAgent tests
# ---------------------------------------------------------------------------


class TestRandomAgent:
    def test_completes_episode(self):
        """RandomAgent runs to completion without errors."""
        agent = RandomAgent()
        result = _run_agent_episode(agent, seed="RANDOM_1")
        assert result["actions"] > 0
        assert result["done"] or result["won"]

    def test_multiple_seeds(self):
        """RandomAgent completes multiple episodes with different seeds."""
        agent = RandomAgent()
        for i in range(5):
            result = _run_agent_episode(agent, seed=f"RANDOM_{i}")
            assert result["actions"] > 0

    def test_produces_valid_factored_actions(self):
        """Every action from RandomAgent can be converted to engine action."""
        adapter = DirectAdapter()
        agent = RandomAgent()
        agent.reset()
        adapter.reset(BACK, STAKE, SEED)

        for _ in range(100):
            if adapter.done:
                break
            if adapter.won and adapter.game.phase == GamePhase.SHOP:
                break

            game = adapter.game
            legal = adapter.get_legal_actions()
            if not legal:
                break

            mask = get_action_mask(game)
            fa = agent.act({}, mask, {"game": game, "legal_actions": legal})

            assert isinstance(fa, FactoredAction)
            assert 0 <= fa.action_type < 23
            # Must be convertible without error
            adapter.step(factored_to_engine_action(fa, game))

    def test_chooses_targets_for_a_targeting_pack_pick(self):
        """A pack Tarot that works on hand cards gets a card_target within
        its bounds (The Magician: 1-2 cards), like a tray use does."""
        game = _arcana_pack("c_magician")
        mask = get_action_mask(game)
        mask.type_mask[:] = False
        mask.type_mask[ActionType.PickPackCard] = True

        fa = RandomAgent().act({}, mask, {"game": game})
        assert fa.entity_target == 0
        assert fa.card_target is not None and 1 <= len(fa.card_target) <= 2

    def test_keeps_the_cerulean_bell_forced_card(self):
        """Every play/discard includes the Bell's forced card (#31)."""
        game = Game.new(BACK, STAKE, SEED)
        game.apply_setup({"op": "force_boss", "key": "bl_final_bell"})
        game.step(SelectBlind())
        mask = get_action_mask(game)
        assert mask.forced_card is not None

        agent = RandomAgent()
        for at in (ActionType.PlayHand, ActionType.Discard):
            mask.type_mask[:] = False
            mask.type_mask[at] = True
            for _ in range(50):
                fa = agent.act({}, mask, {"game": game})
                assert mask.forced_card in fa.card_target
                assert 1 <= len(fa.card_target) <= 5

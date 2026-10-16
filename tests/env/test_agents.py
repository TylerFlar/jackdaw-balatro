"""Tests for the agent protocol and built-in agents.

Covers:
- RandomAgent completes episodes without errors
- Agent protocol compliance
"""

from __future__ import annotations

from jackdaw.engine.actions import GamePhase
from jackdaw.env.action_space import (
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
    """Run one episode with the given agent. Returns final raw_state."""
    adapter = DirectAdapter()
    agent.reset()
    adapter.reset(BACK, STAKE, seed)

    actions = 0
    for _ in range(max_steps):
        if adapter.done:
            break
        phase = adapter.raw_state.get("phase")
        if adapter.won and phase == GamePhase.SHOP:
            break

        gs = adapter.raw_state
        legal = adapter.get_legal_actions()
        if not legal:
            break

        mask = get_action_mask(gs)
        info = {"raw_state": gs, "legal_actions": legal}
        fa = agent.act({}, mask, info)

        engine_action = factored_to_engine_action(fa, gs)
        adapter.step(engine_action)
        actions += 1

    gs = adapter.raw_state
    gs["_actions_taken"] = actions
    gs["_won"] = adapter.won
    gs["_done"] = adapter.done
    return gs


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
        gs = _run_agent_episode(agent, seed="RANDOM_1")
        assert gs["_actions_taken"] > 0
        assert gs["_done"] or gs["_won"]

    def test_multiple_seeds(self):
        """RandomAgent completes multiple episodes with different seeds."""
        agent = RandomAgent()
        for i in range(5):
            gs = _run_agent_episode(agent, seed=f"RANDOM_{i}")
            assert gs["_actions_taken"] > 0

    def test_produces_valid_factored_actions(self):
        """Every action from RandomAgent can be converted to engine action."""
        adapter = DirectAdapter()
        agent = RandomAgent()
        agent.reset()
        adapter.reset(BACK, STAKE, SEED)

        for _ in range(100):
            if adapter.done:
                break
            phase = adapter.raw_state.get("phase")
            if adapter.won and phase == GamePhase.SHOP:
                break

            gs = adapter.raw_state
            legal = adapter.get_legal_actions()
            if not legal:
                break

            mask = get_action_mask(gs)
            info = {"raw_state": gs, "legal_actions": legal}
            fa = agent.act({}, mask, info)

            assert isinstance(fa, FactoredAction)
            assert 0 <= fa.action_type < 21
            # Must be convertible without error
            engine_action = factored_to_engine_action(fa, gs)
            adapter.step(engine_action)

    def test_chooses_targets_for_a_targeting_pack_pick(self):
        """A pack Tarot that works on hand cards gets a card_target within
        its bounds (The Magician: 1-2 cards), like a tray use does."""
        from jackdaw.engine.actions import SelectBlind
        from jackdaw.engine.card_factory import create_consumable
        from jackdaw.env.action_space import ActionType

        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        adapter.step(SelectBlind())  # deal a hand to target
        gs = adapter.raw_state
        gs["phase"] = GamePhase.PACK_OPENING
        gs["pack_type"] = "Arcana"
        gs["pack_cards"] = [create_consumable("c_magician")]
        gs["pack_choices_remaining"] = 1
        mask = get_action_mask(gs)
        mask.type_mask[:] = False
        mask.type_mask[ActionType.PickPackCard] = True

        fa = RandomAgent().act({}, mask, {"raw_state": gs})
        assert fa.entity_target == 0
        assert fa.card_target is not None and 1 <= len(fa.card_target) <= 2

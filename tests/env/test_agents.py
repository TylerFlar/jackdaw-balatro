"""Tests for the agent protocol and built-in agents.

Covers:
- RandomAgent completes episodes without errors
- HeuristicAgent beats RandomAgent on average ante reached (20 episodes each)
- EngineAgent wrapping random_agent produces valid actions
- evaluate_agent returns sensible statistics
"""

from __future__ import annotations

from jackdaw.engine.actions import GamePhase
from jackdaw.env.action_space import (
    FactoredAction,
    factored_to_engine_action,
    get_action_mask,
)
from jackdaw.env.agents import (
    Agent,
    EngineAgent,
    EvalResult,
    HeuristicAgent,
    RandomAgent,
    evaluate_agent,
)
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

    def test_heuristic_agent_satisfies_protocol(self):
        assert isinstance(HeuristicAgent(), Agent)

    def test_engine_agent_satisfies_protocol(self):
        from jackdaw.engine.runner import random_agent

        assert isinstance(EngineAgent(random_agent), Agent)


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


# ---------------------------------------------------------------------------
# HeuristicAgent tests
# ---------------------------------------------------------------------------


class TestHeuristicAgent:
    def test_completes_episode(self):
        """HeuristicAgent runs to completion without errors."""
        agent = HeuristicAgent()
        gs = _run_agent_episode(agent, seed="HEURISTIC_1")
        assert gs["_actions_taken"] > 0
        assert gs["_done"] or gs["_won"]

    def test_multiple_seeds(self):
        """HeuristicAgent completes multiple episodes."""
        agent = HeuristicAgent()
        for i in range(5):
            gs = _run_agent_episode(agent, seed=f"HEURISTIC_{i}")
            assert gs["_actions_taken"] > 0

    def test_beats_random_on_ante(self):
        """HeuristicAgent reaches higher average ante than RandomAgent.

        Run 20 episodes each and compare average ante reached.
        The heuristic should consistently outperform random.
        """
        n = 20
        seeds = [f"COMPARE_{i}" for i in range(n)]

        random_result = evaluate_agent(RandomAgent(), n_episodes=n, seeds=seeds, max_steps=5000)
        heuristic_result = evaluate_agent(
            HeuristicAgent(), n_episodes=n, seeds=seeds, max_steps=5000
        )

        assert heuristic_result.avg_ante > random_result.avg_ante, (
            f"Heuristic avg ante {heuristic_result.avg_ante:.2f} "
            f"<= Random avg ante {random_result.avg_ante:.2f}"
        )


# ---------------------------------------------------------------------------
# EngineAgent tests
# ---------------------------------------------------------------------------


class TestEngineAgent:
    def test_wraps_random_agent(self):
        """EngineAgent wrapping engine random_agent produces valid actions."""
        from jackdaw.engine.runner import random_agent

        agent = EngineAgent(random_agent)
        gs = _run_agent_episode(agent, seed="ENGINE_RANDOM_1")
        assert gs["_actions_taken"] > 0
        assert gs["_done"] or gs["_won"]

    def test_wraps_smart_agent(self):
        """EngineAgent wrapping smart_agent produces valid actions."""
        from jackdaw.cli.smart_agent import smart_agent

        agent = EngineAgent(smart_agent)
        gs = _run_agent_episode(agent, seed="ENGINE_SMART_1")
        assert gs["_actions_taken"] > 0
        assert gs["_done"] or gs["_won"]

    def test_action_type_valid(self):
        """EngineAgent always produces FactoredActions with valid types."""
        from jackdaw.engine.runner import greedy_play_agent

        adapter = DirectAdapter()
        agent = EngineAgent(greedy_play_agent)
        agent.reset()
        adapter.reset(BACK, STAKE, SEED)

        for _ in range(50):
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

            engine_action = factored_to_engine_action(fa, gs)
            adapter.step(engine_action)


# ---------------------------------------------------------------------------
# evaluate_agent tests
# ---------------------------------------------------------------------------


class TestEvaluateAgent:
    def test_returns_eval_result(self):
        result = evaluate_agent(RandomAgent(), n_episodes=3, max_steps=2000)
        assert isinstance(result, EvalResult)
        assert result.n_episodes == 3
        assert len(result.episodes) == 3

    def test_sensible_statistics(self):
        result = evaluate_agent(RandomAgent(), n_episodes=5, max_steps=2000)
        assert 0.0 <= result.win_rate <= 1.0
        assert result.avg_ante >= 1.0
        assert result.avg_rounds >= 0.0
        assert result.avg_actions > 0.0

    def test_summary_dict(self):
        result = evaluate_agent(RandomAgent(), n_episodes=3, max_steps=2000)
        s = result.summary()
        assert "win_rate" in s
        assert "avg_ante" in s
        assert "avg_rounds" in s
        assert "n_episodes" in s
        assert s["n_episodes"] == 3.0

    def test_with_explicit_seeds(self):
        seeds = ["EVAL_A", "EVAL_B", "EVAL_C"]
        result = evaluate_agent(RandomAgent(), n_episodes=3, seeds=seeds, max_steps=2000)
        assert result.episodes[0].seed == "EVAL_A"
        assert result.episodes[1].seed == "EVAL_B"
        assert result.episodes[2].seed == "EVAL_C"

    def test_episode_logs_populated(self):
        result = evaluate_agent(RandomAgent(), n_episodes=2, max_steps=2000)
        for ep in result.episodes:
            assert ep.ante_reached >= 1
            assert ep.actions_taken > 0
            assert isinstance(ep.won, bool)

    def test_heuristic_via_evaluate(self):
        """HeuristicAgent completes episodes through evaluate_agent."""
        result = evaluate_agent(HeuristicAgent(), n_episodes=3, max_steps=5000)
        assert result.n_episodes == 3
        assert all(ep.actions_taken > 0 for ep in result.episodes)

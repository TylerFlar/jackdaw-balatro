"""Agent protocol and built-in agents for the Gymnasium environment.

Provides:

- :class:`Agent` — Protocol that all agents must satisfy
- :class:`RandomAgent` — uniform random baseline
- :class:`HeuristicAgent` — ported smart_agent with game knowledge
- :class:`EngineAgent` — wraps old-style ``(game_state, legal_actions) -> Action`` callables
- :func:`evaluate_agent` — multi-episode evaluation harness
"""

from __future__ import annotations

import random as _random
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from jackdaw.engine.actions import Action, GamePhase
from jackdaw.env.action_space import (
    ActionMask,
    ActionType,
    FactoredAction,
    engine_action_to_factored,
    factored_to_engine_action,
    get_action_mask,
)
from jackdaw.env.agents.heuristic import HeuristicAgent
from jackdaw.env.game_interface import DirectAdapter, GameAdapter

# ---------------------------------------------------------------------------
# Agent Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Agent(Protocol):
    """Gymnasium-compatible agent protocol.

    Receives encoded observations and action masks (the same interface
    a policy network would use), and returns a :class:`FactoredAction`.
    """

    def act(self, obs: dict, action_mask: ActionMask, info: dict) -> FactoredAction:
        """Select an action given observation and mask."""
        ...

    def reset(self) -> None:
        """Called at episode start."""
        ...


# ---------------------------------------------------------------------------
# RandomAgent
# ---------------------------------------------------------------------------


class RandomAgent:
    """Uniform random agent — picks randomly from legal actions.

    Samples a legal action type, then fills in entity/card targets
    as needed.  Always produces valid :class:`FactoredAction` instances.
    """

    def reset(self) -> None:
        pass

    def act(self, obs: dict, action_mask: ActionMask, info: dict) -> FactoredAction:
        # Pick a random legal action type
        legal_types = np.nonzero(action_mask.type_mask)[0]
        if len(legal_types) == 0:
            return FactoredAction(action_type=ActionType.SelectBlind)

        at = int(_random.choice(legal_types))

        entity_target: int | None = None
        card_target: tuple[int, ...] | None = None

        # Entity target
        if at in action_mask.entity_masks:
            mask = action_mask.entity_masks[at]
            legal_entities = np.nonzero(mask)[0]
            if len(legal_entities) > 0:
                entity_target = int(_random.choice(legal_entities))

        # Card target for PlayHand, Discard
        if at in (ActionType.PlayHand, ActionType.Discard):
            legal_cards = np.nonzero(action_mask.card_mask)[0]
            if len(legal_cards) > 0:
                n = min(len(legal_cards), action_mask.max_card_select)
                count = _random.randint(action_mask.min_card_select, n)
                selected = sorted(_random.sample(list(legal_cards), count))
                card_target = tuple(int(i) for i in selected)

        # Card target for UseConsumable (if it needs targets)
        if at == ActionType.UseConsumable and entity_target is not None:
            gs = info.get("raw_state", {})
            consumables = gs.get("consumables", [])
            if entity_target < len(consumables):
                from jackdaw.env.action_space import get_consumable_target_info

                card = consumables[entity_target]
                min_cards, max_cards, needs = get_consumable_target_info(card)
                if needs:
                    legal_cards = np.nonzero(action_mask.card_mask)[0]
                    if len(legal_cards) >= min_cards:
                        n = min(len(legal_cards), max_cards)
                        count = _random.randint(min_cards, n)
                        selected = sorted(_random.sample(list(legal_cards), count))
                        card_target = tuple(int(i) for i in selected)

        return FactoredAction(
            action_type=at,
            card_target=card_target,
            entity_target=entity_target,
        )


# ---------------------------------------------------------------------------
# EngineAgent — adapts old-style engine agents
# ---------------------------------------------------------------------------


class EngineAgent:
    """Adapts an engine-protocol agent to the Gymnasium agent protocol.

    The engine protocol is ``(game_state, legal_actions) -> Action``.
    This wrapper extracts ``raw_state`` from ``info``, reconstructs
    legal actions, calls the engine agent, and converts the result to
    a :class:`FactoredAction`.

    Parameters
    ----------
    engine_agent:
        Callable with signature ``(game_state, legal_actions) -> Action``.
    """

    def __init__(
        self,
        engine_agent: Callable[[dict[str, Any], list[Action]], Action],
    ) -> None:
        self._agent = engine_agent

    def reset(self) -> None:
        pass

    def act(self, obs: dict, action_mask: ActionMask, info: dict) -> FactoredAction:
        gs = info["raw_state"]
        legal_actions = info["legal_actions"]

        engine_action = self._agent(gs, legal_actions)
        try:
            return engine_action_to_factored(engine_action, gs)
        except ValueError:
            # Engine action can't be cleanly mapped (e.g. complex permutation,
            # empty reorder). Fall back using RandomAgent logic.
            fallback = RandomAgent()
            return fallback.act(obs, action_mask, info)


# ---------------------------------------------------------------------------
# Evaluation harness
# ---------------------------------------------------------------------------


@dataclass
class EpisodeLog:
    """Log for a single evaluation episode."""

    seed: str
    won: bool
    ante_reached: int
    rounds: int
    actions_taken: int
    total_reward: float = 0.0


@dataclass
class EvalResult:
    """Aggregate statistics from evaluating an agent over multiple episodes."""

    episodes: list[EpisodeLog] = field(default_factory=list)

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    @property
    def win_rate(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(1 for e in self.episodes if e.won) / len(self.episodes)

    @property
    def avg_ante(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.ante_reached for e in self.episodes) / len(self.episodes)

    @property
    def avg_rounds(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.rounds for e in self.episodes) / len(self.episodes)

    @property
    def avg_actions(self) -> float:
        if not self.episodes:
            return 0.0
        return sum(e.actions_taken for e in self.episodes) / len(self.episodes)

    def summary(self) -> dict[str, float]:
        return {
            "n_episodes": float(self.n_episodes),
            "win_rate": self.win_rate,
            "avg_ante": self.avg_ante,
            "avg_rounds": self.avg_rounds,
            "avg_actions": self.avg_actions,
        }


def evaluate_agent(
    agent: Agent,
    *,
    n_episodes: int = 100,
    seeds: list[str] | None = None,
    back_key: str = "b_red",
    stake: int = 1,
    max_steps: int = 5000,
    adapter_factory: Callable[[], GameAdapter] | None = None,
) -> EvalResult:
    """Run an agent for *n_episodes* episodes and collect statistics.

    Parameters
    ----------
    agent:
        Agent satisfying the :class:`Agent` protocol.
    n_episodes:
        Number of episodes to run.
    seeds:
        Explicit seed list. If None, generates ``EVAL_0``, ``EVAL_1``, etc.
    back_key:
        Deck back key.
    stake:
        Stake level.
    max_steps:
        Max steps per episode.
    adapter_factory:
        Callable returning a fresh GameAdapter. Defaults to DirectAdapter.

    Returns
    -------
    EvalResult
        Aggregate statistics and per-episode logs.
    """
    if seeds is None:
        seeds = [f"EVAL_{i}" for i in range(n_episodes)]
    if len(seeds) < n_episodes:
        seeds = seeds + [f"EVAL_{i}" for i in range(len(seeds), n_episodes)]

    result = EvalResult()

    for i in range(n_episodes):
        adapter = adapter_factory() if adapter_factory else DirectAdapter()
        agent.reset()
        adapter.reset(back_key, stake, seeds[i])

        actions_taken = 0
        for _ in range(max_steps):
            if adapter.done:
                break
            phase = adapter.raw_state.get("phase")
            if adapter.won and phase == GamePhase.SHOP:
                break

            gs = adapter.raw_state
            legal_actions = adapter.get_legal_actions()
            if not legal_actions:
                break

            mask = get_action_mask(gs)
            info = {"raw_state": gs, "legal_actions": legal_actions}
            fa = agent.act({}, mask, info)

            engine_action = factored_to_engine_action(fa, gs)
            adapter.step(engine_action)
            actions_taken += 1

        gs = adapter.raw_state
        rr = gs.get("round_resets", {})
        result.episodes.append(
            EpisodeLog(
                seed=seeds[i],
                won=adapter.won,
                ante_reached=rr.get("ante", 1),
                rounds=gs.get("round", 0),
                actions_taken=actions_taken,
            )
        )

    return result


__all__ = [
    "Agent",
    "EngineAgent",
    "EpisodeLog",
    "EvalResult",
    "HeuristicAgent",
    "RandomAgent",
    "evaluate_agent",
]

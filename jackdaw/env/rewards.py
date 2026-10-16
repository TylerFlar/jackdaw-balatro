"""Reward functions for Balatro RL.

Jackdaw doesn't prescribe a reward: :class:`BalatroGymnasiumEnv` accepts
any callable ``reward_fn(t: Transition) -> float`` and calls it once per
step.  A :class:`Transition` carries snapshots of the game before and after
the action, plus derived events that are easy to get wrong from raw state.
``round`` increments when a blind is *selected*, not beaten, and the ante
advances on the play that beats the boss.

Three examples ship with the package, from sparse to dense:

* :func:`sparse_reward` — +1 for a won run, -1 otherwise, at episode end.
* :func:`blind_progress_reward` — +1 per blind beaten, -1 on a loss.
* :class:`ShapedReward` — a dense multi-signal reward with tunable weights.

Writing your own::

    from jackdaw.env import BalatroGymnasiumEnv, DirectAdapter, Transition

    def my_reward(t: Transition) -> float:
        reward = 1.0 if t.blind_beaten else 0.0
        reward += 0.01 * max(t.curr.dollars - t.prev.dollars, 0)
        return reward

    env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter, reward_fn=my_reward)

A reward function with per-episode state can define a ``reset()`` method;
the env calls it on every reset.  Build one instance per env, e.g. inside
the env factory of a vectorized setup, so episodes don't share state.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from jackdaw.engine.actions import GamePhase
from jackdaw.env.game_interface import GameState
from jackdaw.env.game_spec import FactoredAction

# ---------------------------------------------------------------------------
# Transition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Transition:
    """One environment step, as seen by a reward function."""

    prev: GameState
    """Snapshot taken before the action."""

    curr: GameState
    """Snapshot taken after the action."""

    action: FactoredAction
    """The action that was taken."""

    terminated: bool
    """The run ended on this step (game over)."""

    truncated: bool
    """The episode hit its step limit on this step."""

    raw_state: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)
    """The live engine state after the step.  The next step mutates it in
    place: read what you need now, don't keep a reference."""

    @classmethod
    def from_step(
        cls,
        action: FactoredAction,
        terminated: bool,
        truncated: bool,
        info: dict[str, Any],
    ) -> Transition:
        """Build a transition from :meth:`BalatroEnvironment.step` output,
        for computing rewards in a custom training loop."""
        return cls(
            prev=info["prev_state"],
            curr=info["state"],
            action=action,
            terminated=terminated,
            truncated=truncated,
            raw_state=info.get("raw_state", {}),
        )

    @property
    def done(self) -> bool:
        """The episode is over (terminated or truncated)."""
        return self.terminated or self.truncated

    @property
    def blind_beaten(self) -> bool:
        """This step cleared a blind: the hand just played ended the round.

        Also true when Mr. Bones saves a losing final hand, since the run
        carries on to the next blind either way.
        """
        return (
            self.prev.phase == GamePhase.SELECTING_HAND and self.curr.phase == GamePhase.ROUND_EVAL
        )

    @property
    def boss_beaten(self) -> bool:
        """This step cleared a boss blind."""
        return self.blind_beaten and self.prev.blind_on_deck == "Boss"

    @property
    def blind_ante(self) -> int:
        """The ante of the blind being played.  On the step that beats a
        boss, ``curr.ante`` has already moved on to the next ante."""
        return self.prev.ante

    @property
    def chips_scored(self) -> float:
        """Chips this step added toward the current blind (0 outside a
        hand play; selecting a blind resets the count, which isn't
        scoring)."""
        if self.prev.phase != GamePhase.SELECTING_HAND:
            return 0
        delta = self.curr.chips - self.prev.chips
        return delta if delta > 0 else 0  # NaN (inf - inf) reads as 0

    @property
    def run_won(self) -> bool:
        """This step won the run (beat the final boss).  Fires once — with
        ``end_on_win=False`` play continues into endless mode after it."""
        return self.curr.won and not self.prev.won


RewardFn = Callable[[Transition], float]
"""Signature of a reward function: ``reward_fn(transition) -> float``."""


# ---------------------------------------------------------------------------
# Example reward functions
# ---------------------------------------------------------------------------


def sparse_reward(t: Transition) -> float:
    """+1 when an episode ends with the run won, -1 otherwise, else 0.

    With ``end_on_win=False`` a run won at ante 8 stays won even if play
    carries on into endless mode and dies later.
    """
    if t.done:
        return 1.0 if t.curr.won else -1.0
    return 0.0


def blind_progress_reward(t: Transition) -> float:
    """+1 for every blind beaten, -1 when an episode ends short of a win.

    Denser than :func:`sparse_reward` and still impossible to farm:
    blinds only advance by beating them.
    """
    reward = 1.0 if t.blind_beaten else 0.0
    if t.done and not t.curr.won:
        reward -= 1.0
    return reward


@dataclass(frozen=True)
class ShapedReward:
    """Dense multi-signal reward with tunable weights.

    ==============  ==============================================  =======
    Field           Trigger                                         Default
    ==============  ==============================================  =======
    step_cost       every action                                    -0.001
    shop_step_cost  every action taken in the shop, instead         -0.002
    blind_beaten    the hand that clears a blind, x ante / 8        +0.15
    boss_beaten     extra when that blind is a boss, x ante / 8     +0.10
    hands_left      per hand left after the clearing hand           +0.01
    score_progress  x chips scored / blind target, capped at 1      +0.02
    win             the run is won                                  +0.5
    loss            the episode ends short of a win                 -0.2
    ==============  ==============================================  =======

    ``ante`` is the ante of the blind just beaten.  Penalties are given as
    positive magnitudes.  Override any weight by keyword, e.g.
    ``ShapedReward(win=1.0, step_cost=0.0)``.  Stateless, so one instance
    can serve any number of envs.
    """

    step_cost: float = 0.001
    shop_step_cost: float = 0.002
    blind_beaten: float = 0.15
    boss_beaten: float = 0.10
    hands_left: float = 0.01
    score_progress: float = 0.02
    win: float = 0.5
    loss: float = 0.2

    def __call__(self, t: Transition) -> float:
        # Cost of the action, doubled in the shop to discourage stalling
        # there.  Keyed on where the action was taken.
        reward = -(self.shop_step_cost if t.prev.phase == GamePhase.SHOP else self.step_cost)

        if t.blind_beaten:
            ante_scale = t.blind_ante / 8.0
            reward += self.blind_beaten * ante_scale
            if t.boss_beaten:
                reward += self.boss_beaten * ante_scale
            reward += self.hands_left * t.curr.hands_left

        # Progress toward the blind.  An overflowed target (endless ante
        # 39+) can't be progressed against.
        target = t.curr.blind_chips
        if 0 < target < math.inf and t.chips_scored:
            reward += self.score_progress * min(t.chips_scored / target, 1.0)

        if t.run_won:
            reward += self.win
        if t.done and not t.curr.won:
            reward -= self.loss

        return reward

"""Live and bridge-validated environment variants.

Provides :class:`LiveBalatroEnv` (plays against real Balatro via balatrobot)
and :class:`SimBridgeBalatroEnv` (full serialization round-trip in-process),
plus a :func:`validate_episode` function for step-by-step comparison.

Both classes satisfy the :class:`~jackdaw.env.game_interface.GameAdapter`
protocol — the only difference from :class:`DirectAdapter` is the backend.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from jackdaw.engine.actions import Action, GamePhase
from jackdaw.env.game_interface import (
    BridgeAdapter,
    GameAdapter,
    GameState,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection errors
# ---------------------------------------------------------------------------


class BalatrobotConnectionError(ConnectionError):
    """Raised when balatrobot is unreachable.

    The error message includes instructions for starting balatrobot.
    """


# ---------------------------------------------------------------------------
# LiveBalatroEnv
# ---------------------------------------------------------------------------


class LiveBalatroEnv:
    """GameAdapter that plays against real Balatro via balatrobot.

    Wraps ``BridgeAdapter(LiveBackend(host, port))`` with connection
    management: health checks, retries, and clear error messages.

    Parameters
    ----------
    host:
        balatrobot host address.
    port:
        balatrobot port.
    timeout:
        HTTP request timeout in seconds.
    retries:
        Number of retry attempts for transient failures.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 12346,
        timeout: float = 10.0,
        retries: int = 3,
    ) -> None:
        from jackdaw.bridge.backend import LiveBackend

        self._host = host
        self._port = port
        self._timeout = timeout
        self._retries = retries
        self._backend = LiveBackend(host, port)
        self._adapter = BridgeAdapter(self._backend)

    # -- Connection management ----------------------------------------------

    def health_check(self) -> bool:
        """Check if balatrobot is reachable. Returns True if healthy."""
        try:
            result = self._backend.handle("health", {})
            return result.get("status") == "ok"
        except Exception:
            return False

    def _ensure_connected(self) -> None:
        """Verify balatrobot is reachable, raising a clear error if not."""
        if not self.health_check():
            raise BalatrobotConnectionError(
                f"Cannot reach balatrobot at {self._host}:{self._port}. "
                "Ensure Balatro is running with the balatrobot mod enabled, "
                "or start the jackdaw JSON-RPC server with: "
                "python -m jackdaw.cli serve --live"
            )

    def _with_retry(self, fn: Callable[[], Any]) -> Any:
        """Execute *fn* with retry logic for transient failures."""
        last_exc: Exception | None = None
        for attempt in range(self._retries):
            try:
                return fn()
            except BalatrobotConnectionError:
                raise
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "Attempt %d/%d failed: %s",
                    attempt + 1,
                    self._retries,
                    exc,
                )
        assert last_exc is not None
        raise BalatrobotConnectionError(
            f"All {self._retries} attempts failed. Last error: {last_exc}"
        ) from last_exc

    # -- GameAdapter interface ----------------------------------------------

    def reset(
        self,
        back_key: str,
        stake: int,
        seed: str,
        *,
        challenge: dict[str, Any] | None = None,
    ) -> GameState:
        self._ensure_connected()
        return self._with_retry(
            lambda: self._adapter.reset(back_key, stake, seed, challenge=challenge)
        )

    def step(self, action: Action) -> GameState:
        return self._with_retry(lambda: self._adapter.step(action))

    def get_legal_actions(self) -> list[Action]:
        return self._adapter.get_legal_actions()

    @property
    def raw_state(self) -> dict[str, Any]:
        return self._adapter.raw_state

    @property
    def done(self) -> bool:
        return self._adapter.done

    @property
    def won(self) -> bool:
        return self._adapter.won


# ---------------------------------------------------------------------------
# SimBridgeBalatroEnv
# ---------------------------------------------------------------------------


class SimBridgeBalatroEnv:
    """GameAdapter that routes through the full bridge serialization path.

    Uses ``BridgeAdapter(SimBackend())`` — the engine runs in-process but
    every state transition goes through JSON-RPC serialization and
    deserialization.  Useful for catching bridge round-trip bugs without
    needing a live Balatro instance.
    """

    def __init__(self) -> None:
        from jackdaw.bridge.backend import SimBackend

        self._backend = SimBackend()
        self._adapter = BridgeAdapter(self._backend)

    # -- GameAdapter interface (delegate) -----------------------------------

    def reset(
        self,
        back_key: str,
        stake: int,
        seed: str,
        *,
        challenge: dict[str, Any] | None = None,
    ) -> GameState:
        return self._adapter.reset(back_key, stake, seed, challenge=challenge)

    def step(self, action: Action) -> GameState:
        return self._adapter.step(action)

    def get_legal_actions(self) -> list[Action]:
        return self._adapter.get_legal_actions()

    @property
    def raw_state(self) -> dict[str, Any]:
        return self._adapter.raw_state

    @property
    def done(self) -> bool:
        return self._adapter.done

    @property
    def won(self) -> bool:
        return self._adapter.won


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


@dataclass
class StepDivergence:
    """Record of a single divergence between two envs."""

    step: int
    field: str
    expected: Any
    actual: Any
    context: str = ""


@dataclass
class ValidationResult:
    """Result of validating an episode across two environments."""

    seed: str
    steps: int
    divergences: list[StepDivergence] = field(default_factory=list)
    ref_won: bool = False
    test_won: bool = False
    ref_done: bool = False
    test_done: bool = False

    @property
    def ok(self) -> bool:
        """True if no divergences were found."""
        return len(self.divergences) == 0


def _compare_states(
    step_idx: int,
    ref_state: GameState,
    test_state: GameState,
    divergences: list[StepDivergence],
    *,
    atol: float = 0.0,
) -> None:
    """Compare two GameState snapshots field-by-field."""
    fields = [
        "phase",
        "ante",
        "round",
        "dollars",
        "hands_left",
        "discards_left",
        "hand_size",
        "joker_slots",
        "consumable_slots",
        "blind_on_deck",
        "blind_chips",
        "chips",
        "won",
        "done",
    ]
    for f in fields:
        ref_val = getattr(ref_state, f)
        test_val = getattr(test_state, f)
        if isinstance(ref_val, float) and isinstance(test_val, float):
            if abs(ref_val - test_val) > atol:
                divergences.append(
                    StepDivergence(
                        step=step_idx,
                        field=f,
                        expected=ref_val,
                        actual=test_val,
                    )
                )
        elif ref_val != test_val:
            divergences.append(
                StepDivergence(
                    step=step_idx,
                    field=f,
                    expected=ref_val,
                    actual=test_val,
                )
            )


def _compare_legal_masks(
    step_idx: int,
    ref_legal: list[Action],
    test_legal: list[Action],
    divergences: list[StepDivergence],
) -> None:
    """Compare legal action lists by type (order-independent)."""
    ref_types = sorted(type(a).__name__ for a in ref_legal)
    test_types = sorted(type(a).__name__ for a in test_legal)
    if ref_types != test_types:
        divergences.append(
            StepDivergence(
                step=step_idx,
                field="legal_action_types",
                expected=ref_types,
                actual=test_types,
                context=f"ref has {len(ref_legal)} actions, test has {len(test_legal)}",
            )
        )


def validate_episode(
    env: GameAdapter,
    live_env: GameAdapter,
    seed: str,
    agent: Callable[[GameAdapter], Action | None],
    *,
    back_key: str = "b_red",
    stake: int = 1,
    max_steps: int = 5000,
    atol: float = 0.0,
) -> ValidationResult:
    """Run the same agent on both envs with the same seed, compare step-by-step.

    Parameters
    ----------
    env:
        Reference environment (typically DirectAdapter).
    live_env:
        Test environment (LiveBalatroEnv or SimBridgeBalatroEnv).
    seed:
        RNG seed for both envs.
    agent:
        Callable that takes a GameAdapter and returns the next Action,
        or None to stop. The agent should be deterministic and only
        read from the reference env to choose actions.
    back_key:
        Deck back key.
    stake:
        Stake level.
    max_steps:
        Maximum steps before aborting.
    atol:
        Absolute tolerance for floating-point comparisons.

    Returns
    -------
    ValidationResult
        Step count, divergences, and terminal flags.
    """
    result = ValidationResult(seed=seed, steps=0)

    ref_state = env.reset(back_key, stake, seed)
    test_state = live_env.reset(back_key, stake, seed)
    _compare_states(0, ref_state, test_state, result.divergences, atol=atol)

    for i in range(1, max_steps + 1):
        if env.done or live_env.done:
            break

        # Check for won + SHOP (run complete but not GAME_OVER yet)
        ref_phase = env.raw_state.get("phase")
        if env.won and ref_phase == GamePhase.SHOP:
            break

        # Compare legal actions
        ref_legal = env.get_legal_actions()
        test_legal = live_env.get_legal_actions()
        _compare_legal_masks(i, ref_legal, test_legal, result.divergences)

        if not ref_legal:
            break

        # Agent picks action based on reference env
        action = agent(env)
        if action is None:
            break

        ref_state = env.step(action)
        test_state = live_env.step(action)
        result.steps = i

        _compare_states(i, ref_state, test_state, result.divergences, atol=atol)

        # Early exit on too many divergences
        if len(result.divergences) > 100:
            logger.warning("Stopping validation early: >100 divergences")
            break

    result.ref_won = env.won
    result.test_won = live_env.won
    result.ref_done = env.done
    result.test_done = live_env.done

    return result

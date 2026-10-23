"""Scenario-based validation — compare jackdaw engine against live Balatro.

Each scenario plays on a lockstep (:class:`~jackdaw.bridge.live.LiveAdapter`):
it sets up a game state with BalatroBot's ``add``/``set`` debug commands and
executes actions, each of which goes to the shadow run (the simulator) and to
the live game, with the live view compared against the shadow's after every
request. A scenario is ``fn(ls, *, delay) -> ScenarioResult``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ScenarioResult:
    """Outcome of a single scenario run."""

    passed: bool
    diffs: list[str] = field(default_factory=list)
    details: str = ""
    sub_results: list[tuple[str, ScenarioResult]] = field(default_factory=list)
    #: Not run to the end (a request the deviation catalog vetoes); not a failure.
    skipped: bool = False
    #: The repro bundle of a divergence.
    bundle: str = ""
    #: Differences the deviation catalog explained, by entry id.
    explained: dict[str, int] = field(default_factory=dict)


@dataclass
class Scenario:
    """A single validation scenario."""

    name: str
    category: str
    description: str
    run: Callable[..., ScenarioResult]


# ---------------------------------------------------------------------------
# Global registry
# ---------------------------------------------------------------------------

_REGISTRY: list[Scenario] = []


def register(
    name: str,
    category: str,
    description: str,
) -> Callable:
    """Decorator that registers a scenario function."""

    def decorator(fn: Callable) -> Callable:
        _REGISTRY.append(Scenario(name=name, category=category, description=description, run=fn))
        return fn

    return decorator


def get_all_scenarios() -> list[Scenario]:
    """Return all registered scenarios (triggers imports to populate registry)."""
    # Import all scenario modules to trigger registration
    from jackdaw.cli.scenarios import (  # noqa: F401
        boss_blinds,
        jokers,
        modifiers,
        planets,
        spectrals,
        tags,
        tarots,
    )

    return list(_REGISTRY)


def get_scenarios(
    category: str | None = None,
    name: str | None = None,
) -> list[Scenario]:
    """Return filtered scenarios."""
    all_scenarios = get_all_scenarios()
    if name:
        return [s for s in all_scenarios if s.name == name]
    if category:
        return [s for s in all_scenarios if s.category == category]
    return all_scenarios

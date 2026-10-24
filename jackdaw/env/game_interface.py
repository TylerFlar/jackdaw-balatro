"""Unified game interface layer for agents and Gymnasium environments.

Provides a single abstraction over the in-process engine (for training)
and the balatrobot bridge (for validation against real Balatro).  Agents
interact exclusively through :class:`GameAdapter`-compatible adapters —
they never touch JSON-RPC directly.

Architecture::

                    ┌─ DirectAdapter(Rust engine, jackdaw.core.Game)   ← training
    [Agent/Model] → ┤
                    └─ LiveAdapter(backend), Rust shadow run           ← the live game,
                                                                         or BalatroBot's API
                                                                         on the simulator

Both expose their run as ``game``: a live :class:`jackdaw.core.Game` (for a
:class:`~jackdaw.bridge.live.LiveAdapter`, its shadow).  ``BridgeAdapter(backend)``
is a ``LiveAdapter``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from jackdaw.engine.actions import Action, GamePhase

if TYPE_CHECKING:
    from jackdaw.bridge.live import LiveAdapter
    from jackdaw.core import Game

# ---------------------------------------------------------------------------
# GameState — lightweight frozen snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameState:
    """Frozen snapshot of the game state for the interface contract.

    This is a lightweight summary — the full observation encoding is
    handled by the Gymnasium env wrapper.  This dataclass exposes the
    minimum typed accessors needed for agent decision-making and
    reward computation.

    ``won`` is whether the run was won.  ``G.GAME.won`` is set by end_round for
    the boss of exactly ``win_ante`` before its game-over test, so losing that
    boss sets it too (G260); such a loss is not a win.  A loss later in endless
    mode (a higher ante) keeps the win.
    """

    phase: GamePhase
    ante: int
    round: int
    dollars: int
    hands_left: int
    discards_left: int
    hand_size: int
    joker_slots: int
    consumable_slots: int
    blind_on_deck: str | None
    blind_chips: int
    chips: int
    won: bool
    done: bool


_PHASES: dict[str, GamePhase] = {p.value: p for p in GamePhase}

StateParts = tuple[str, str | None, bool, bool, list[Any], float]


def game_state_from_parts(parts: StateParts) -> GameState:
    """A :class:`GameState` from ``jackdaw._core``'s summary of a Rust run.

    Whole numbers come back as ``int``.
    """
    phase, blind_on_deck, won, done, nums, chips = parts
    ante, rnd, dollars, hands, discards, hand_size, jokers, consumables, blind_chips = nums
    return GameState(
        phase=_PHASES[phase],
        ante=ante,
        round=rnd,
        dollars=dollars,
        hands_left=hands,
        discards_left=discards,
        hand_size=hand_size,
        joker_slots=jokers,
        consumable_slots=consumables,
        blind_on_deck=blind_on_deck,
        blind_chips=blind_chips,
        chips=chips,
        won=won,
        done=done,
    )


def game_state(game: Game) -> GameState:
    """The :class:`GameState` of a Rust run."""
    return game_state_from_parts(game._g.env_state())


# ---------------------------------------------------------------------------
# GameAdapter protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class GameAdapter(Protocol):
    """Backend-agnostic game adapter.

    :class:`DirectAdapter` and :class:`~jackdaw.bridge.live.LiveAdapter` satisfy
    this protocol.  Agents and the Gymnasium env program against this interface
    exclusively.  An adapter also exposes its run as ``game``
    (a :class:`jackdaw.core.Game`), which the environments read their
    observations and masks from.
    """

    def reset(
        self,
        back_key: str,
        stake: int,
        seed: str,
        *,
        challenge: dict[str, Any] | None = None,
    ) -> GameState:
        """Start a new run.  Returns initial game state snapshot."""
        ...

    def step(self, action: Action) -> GameState:
        """Apply *action*, return new state snapshot."""
        ...

    def get_legal_actions(self) -> list[Action]:
        """Return legal actions in current state."""
        ...

    @property
    def done(self) -> bool:
        """True if game is over (won or lost)."""
        ...

    @property
    def won(self) -> bool:
        """True if the run was won."""
        ...


# ---------------------------------------------------------------------------
# DirectAdapter — the Rust engine in process (fast path for training)
# ---------------------------------------------------------------------------


class DirectAdapter:
    """Adapter over the Rust engine (:class:`jackdaw.core.Game`), in process.

    ``game`` is the live run: the next step moves it on, so read what you
    need or ``game.freeze()`` a copy.
    """

    def __init__(self) -> None:
        self._game: Game | None = None

    def reset(
        self,
        back_key: str,
        stake: int,
        seed: str,
        *,
        challenge: dict[str, Any] | str | None = None,
    ) -> GameState:
        from jackdaw.core import Game

        challenge_id = challenge.get("id") if isinstance(challenge, dict) else challenge
        self._game = Game.new(back_key, stake, seed, challenge_id)
        return game_state(self._game)

    def step(self, action: Action) -> GameState:
        self.game.step(action)
        return game_state(self.game)

    def get_legal_actions(self) -> list[Action]:
        # Every legal action has a factored form (RerollBoss and BuyAndUse are
        # action types 21 and 22).
        return self.game.legal_actions()

    @property
    def game(self) -> Game:
        """The live run (:class:`jackdaw.core.Game`)."""
        if self._game is None:
            raise RuntimeError("DirectAdapter: call reset() first")
        return self._game

    @property
    def done(self) -> bool:
        return self.game.phase == GamePhase.GAME_OVER

    @property
    def won(self) -> bool:
        return game_state(self.game).won


# ---------------------------------------------------------------------------
# BridgeAdapter — a game behind BalatroBot's JSON-RPC API
# ---------------------------------------------------------------------------


class BridgeAdapter:
    """Adapter that plays through BalatroBot's JSON-RPC API: ``BridgeAdapter(backend)``
    is a :class:`~jackdaw.bridge.live.LiveAdapter` over ``backend``.

    With :class:`~jackdaw.bridge.backend.LiveBackend` that is the live game; with
    :class:`~jackdaw.bridge.backend.SimBackend` it is BalatroBot's API on the Rust
    simulator, in process.  Each action goes out as the request the bot layer
    translates it to (``SortHand`` and the swaps as ``rearrange`` permutations); the
    adapter keeps a shadow run in the Rust engine, checks every response against it
    (:class:`~jackdaw.bridge.live.DivergenceError` on a difference) and takes
    observations, masks and legal actions from the shadow under BALATROBOT legality:
    only what the bot executes (``docs/live.md``).
    """

    def __new__(cls, backend: Any, *args: Any, **kwargs: Any) -> LiveAdapter:  # type: ignore[misc]
        from jackdaw.bridge.live import LiveAdapter

        return LiveAdapter(backend, *args, **kwargs)

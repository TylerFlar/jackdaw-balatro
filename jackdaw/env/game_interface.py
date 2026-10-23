"""Unified game interface layer for agents and Gymnasium environments.

Provides a single abstraction over the in-process engines (for training)
and the balatrobot bridge (for validation against real Balatro).  Agents
interact exclusively through :class:`GameAdapter`-compatible adapters —
they never touch raw ``game_state`` dicts or JSON-RPC directly.

Architecture::

                    ┌─ DirectAdapter(Rust engine, jackdaw.core.Game)   ← training
    [Agent/Model] → ├─ OracleAdapter(Python engine, raw_state dict)    ← parity checks
                    └─ BridgeAdapter(SimBackend or LiveBackend)        ← validation

:class:`DirectAdapter` exposes its run as ``game`` (a live
:class:`jackdaw.core.Game`); the Python-engine adapters expose the engine's
dict as ``raw_state``.  The Python engine is the reference the Rust one is
checked against until it is retired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

from jackdaw.engine.actions import Action, GamePhase
from jackdaw.env.action_space import ActionMask, ActionType

if TYPE_CHECKING:
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


def run_won(gs: dict[str, Any]) -> bool:
    """Whether the run was won, for agents and rewards.

    ``G.GAME.won`` is set by end_round for the boss of exactly ``win_ante``
    before its game-over test, so losing that boss sets it too (G260); the
    ante only goes up on a win, so such a loss ends the run at ``win_ante``.
    A loss later in endless mode (a higher ante) keeps the win.
    """
    if not gs.get("won", False):
        return False
    phase = gs.get("phase")
    lost = phase == GamePhase.GAME_OVER or phase == "game_over"
    return not (lost and gs.get("round_resets", {}).get("ante") == gs.get("win_ante", 8))


def _snapshot(gs: dict[str, Any]) -> GameState:
    """Build a :class:`GameState` snapshot from an engine game_state dict."""
    phase = gs.get("phase", GamePhase.GAME_OVER)
    if isinstance(phase, str):
        phase = GamePhase(phase)
    cr = gs.get("current_round", {})
    rr = gs.get("round_resets", {})
    blind = gs.get("blind")
    blind_chips = getattr(blind, "chips", 0) if blind else 0
    return GameState(
        phase=phase,
        ante=rr.get("ante", 1),
        round=gs.get("round", 0),
        dollars=gs.get("dollars", 0),
        hands_left=cr.get("hands_left", 0),
        discards_left=cr.get("discards_left", 0),
        hand_size=gs.get("hand_size", 8),
        joker_slots=gs.get("joker_slots", 5),
        consumable_slots=gs.get("consumable_slots", 2),
        blind_on_deck=gs.get("blind_on_deck"),
        blind_chips=blind_chips,
        chips=gs.get("chips", 0),
        won=run_won(gs),
        done=phase == GamePhase.GAME_OVER,
    )


_PHASES: dict[str, GamePhase] = {p.value: p for p in GamePhase}

StateParts = tuple[str, str | None, bool, bool, list[Any], float]


def game_state_from_parts(parts: StateParts) -> GameState:
    """A :class:`GameState` from ``jackdaw._core``'s summary of a Rust run.

    Whole numbers come back as ``int`` and ``won`` follows :func:`run_won`, as
    in :func:`_snapshot`.
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

    :class:`DirectAdapter`, :class:`OracleAdapter` and :class:`BridgeAdapter`
    satisfy this protocol.  Agents and the Gymnasium env program against
    this interface exclusively.  An adapter also exposes its run, as
    ``game`` (the Rust engine) or ``raw_state`` (the Python engine's dict).
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
    need or ``game.freeze()`` a copy.  There is no ``raw_state`` dict; the
    Python engine's adapter (:class:`OracleAdapter`) keeps one.
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
    def raw_state(self) -> dict[str, Any]:
        raise AttributeError(
            "DirectAdapter runs the Rust engine: read adapter.game (a jackdaw.core.Game); "
            "OracleAdapter keeps the Python engine's raw_state dict"
        )

    @property
    def done(self) -> bool:
        return self.game.phase == GamePhase.GAME_OVER

    @property
    def won(self) -> bool:
        return game_state(self.game).won


# ---------------------------------------------------------------------------
# OracleAdapter — the Python engine, the reference the Rust one is checked against
# ---------------------------------------------------------------------------


class OracleAdapter:
    """Adapter over the Python engine (the oracle), in process.

    ``raw_state`` returns the live engine dict directly.  It plays exactly
    as :class:`DirectAdapter` wherever the Rust engine matches the oracle;
    the parity tests run both.
    """

    def __init__(self) -> None:
        self._gs: dict[str, Any] = {}

    def reset(
        self,
        back_key: str,
        stake: int,
        seed: str,
        *,
        challenge: dict[str, Any] | None = None,
    ) -> GameState:
        from jackdaw.engine.run_init import initialize_run

        self._gs = initialize_run(back_key, stake, seed, challenge=challenge)
        self._gs["phase"] = GamePhase.BLIND_SELECT
        self._gs["blind_on_deck"] = "Small"
        return _snapshot(self._gs)

    def step(self, action: Action) -> GameState:
        from jackdaw.engine.game import step as engine_step

        engine_step(self._gs, action)
        return _snapshot(self._gs)

    def get_legal_actions(self) -> list[Action]:
        from jackdaw.engine.actions import get_legal_actions as engine_legal

        # Every legal action has a factored form (RerollBoss and BuyAndUse are
        # action types 21 and 22).
        return engine_legal(self._gs)

    @property
    def raw_state(self) -> dict[str, Any]:
        return self._gs

    @property
    def done(self) -> bool:
        phase = self._gs.get("phase")
        if isinstance(phase, str):
            return phase == GamePhase.GAME_OVER
        return phase == GamePhase.GAME_OVER

    @property
    def won(self) -> bool:
        return run_won(self._gs)


# ---------------------------------------------------------------------------
# BridgeAdapter — wraps any Backend (SimBackend or LiveBackend)
# ---------------------------------------------------------------------------

# The balatrobot endpoint each factored action type is sent to; None for the
# sim-only actions, which have no endpoint.
_BOT_METHOD_FOR_ACTION: dict[ActionType, str | None] = {
    ActionType.PlayHand: "play",
    ActionType.Discard: "discard",
    ActionType.SelectBlind: "select",
    ActionType.SkipBlind: "skip",
    ActionType.CashOut: "cash_out",
    ActionType.Reroll: "reroll",
    ActionType.NextRound: "next_round",
    ActionType.SkipPack: "pack",
    ActionType.BuyCard: "buy",
    ActionType.SellJoker: "sell",
    ActionType.SellConsumable: "sell",
    ActionType.UseConsumable: "use",
    ActionType.RedeemVoucher: "buy",
    ActionType.OpenBooster: "buy",
    ActionType.PickPackCard: "pack",
    ActionType.SwapJokersLeft: "rearrange",
    ActionType.SwapJokersRight: "rearrange",
    ActionType.SwapHandLeft: "rearrange",
    ActionType.SwapHandRight: "rearrange",
    ActionType.SortHandRank: "rearrange",
    ActionType.SortHandSuit: "rearrange",
    ActionType.RerollBoss: None,
    ActionType.BuyAndUse: None,
}


class BridgeAdapter:
    """Adapter that wraps a :class:`~jackdaw.bridge.backend.Backend`.

    Works with both :class:`SimBackend` (in-process engine behind
    JSON-RPC serialization) and :class:`LiveBackend` (HTTP proxy to
    real Balatro).

    For :class:`SimBackend`, ``raw_state`` returns the engine dict
    directly (zero-copy shortcut).  For :class:`LiveBackend``,
    ``raw_state`` returns the last JSON response converted to a
    dict via :func:`bot_state_to_game_state`.
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self._last_response: dict[str, Any] = {}
        self._last_gs: dict[str, Any] = {}

    def reset(
        self,
        back_key: str,
        stake: int,
        seed: str,
        *,
        challenge: dict[str, Any] | None = None,
    ) -> GameState:
        from jackdaw.bridge.backend import DECK_FROM_BOT, STAKE_FROM_BOT

        # Reverse-map engine keys to balatrobot enum strings
        deck_to_bot = {v: k for k, v in DECK_FROM_BOT.items()}
        stake_to_bot = {v: k for k, v in STAKE_FROM_BOT.items()}

        params: dict[str, Any] = {
            "deck": deck_to_bot.get(back_key, "RED"),
            "stake": stake_to_bot.get(stake, "WHITE"),
            "seed": seed,
        }
        self._last_response = self._backend.handle("start", params)
        self._last_gs = self._build_gs()
        return _snapshot(self._last_gs)

    def step(self, action: Action) -> GameState:
        from jackdaw.bridge.balatrobot_adapter import action_to_rpc

        rpc = action_to_rpc(action, self._last_gs)
        self._last_response = self._backend.handle(rpc["method"], rpc["params"])
        self._last_gs = self._build_gs()
        return _snapshot(self._last_gs)

    def get_legal_actions(self) -> list[Action]:
        """Legal actions the balatrobot protocol will execute.

        The engine follows vanilla, which allows more than balatrobot's
        endpoints accept (e.g. using a consumable while a pack is open).
        """
        from jackdaw.bridge.balatrobot_adapter import bot_accepts
        from jackdaw.engine.actions import get_legal_actions as engine_legal

        gs = self.raw_state
        return [a for a in engine_legal(gs) if bot_accepts(a, gs)]

    def restrict_mask(self, mask: ActionMask) -> ActionMask:
        """Narrow an action mask to what the balatrobot protocol will
        execute in the current state.

        :class:`~jackdaw.env.balatro_env.BalatroEnvironment` calls this on
        adapters that define it, so agents playing over the bridge are never
        offered an action the bot refuses.
        """
        from jackdaw.bridge.balatrobot_adapter import bot_refusal

        gs = self.raw_state
        for type_idx in np.nonzero(mask.type_mask)[0]:
            at = ActionType(int(type_idx))
            method = _BOT_METHOD_FOR_ACTION[at]
            if method is None:
                usable = False
            elif at == ActionType.UseConsumable:
                # Judged per consumable: use.lua refuses card-selecting
                # ones outside SELECTING_HAND.
                entity_mask = mask.entity_masks[at]
                for i in np.nonzero(entity_mask)[0]:
                    if bot_refusal(method, {"consumable": int(i)}, gs) is not None:
                        entity_mask[i] = False
                usable = bool(entity_mask.any())
            else:
                usable = bot_refusal(method, {}, gs) is None
            if not usable:
                mask.type_mask[at] = False
                mask.entity_masks.pop(at, None)
        return mask

    @property
    def raw_state(self) -> dict[str, Any]:
        from jackdaw.bridge.backend import SimBackend

        # SimBackend: return the live engine dict (zero-copy)
        if isinstance(self._backend, SimBackend) and self._backend._gs is not None:
            return self._backend._gs
        # LiveBackend / fallback: return converted state
        return self._last_gs

    @property
    def done(self) -> bool:
        state = self._last_response.get("state", "")
        return state == "GAME_OVER"

    @property
    def won(self) -> bool:
        return run_won(self._last_gs)

    # -- internal -----------------------------------------------------------

    def _build_gs(self) -> dict[str, Any]:
        """Build a game_state dict from the last bridge response.

        For SimBackend, returns the live engine dict directly.
        For LiveBackend, converts the JSON response.
        """
        from jackdaw.bridge.backend import SimBackend

        if isinstance(self._backend, SimBackend) and self._backend._gs is not None:
            return self._backend._gs

        from jackdaw.bridge.balatrobot_adapter import bot_state_to_game_state

        return bot_state_to_game_state(self._last_response)

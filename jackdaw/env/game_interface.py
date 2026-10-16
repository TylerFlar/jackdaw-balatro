"""Unified game interface layer for agents and Gymnasium environments.

Provides a single abstraction over both the direct in-process engine
(for training) and the balatrobot bridge (for validation against real
Balatro).  Agents interact exclusively through :class:`GameInterface`-
compatible adapters — they never touch raw ``game_state`` dicts or
JSON-RPC directly.

Architecture::

                    ┌─ DirectAdapter(engine step/get_legal_actions)  ← training
    [Agent/Model] → │
                    └─ BridgeAdapter(SimBackend or LiveBackend)      ← validation
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np

from jackdaw.engine.actions import Action, GamePhase
from jackdaw.env.action_space import ActionMask, ActionType

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
        won=gs.get("won", False),
        done=phase == GamePhase.GAME_OVER,
    )


# ---------------------------------------------------------------------------
# GameAdapter protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class GameAdapter(Protocol):
    """Backend-agnostic game adapter.

    Both :class:`DirectAdapter` and :class:`BridgeAdapter` satisfy this
    protocol.  Agents and the Gymnasium env program against this
    interface exclusively.
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
    def raw_state(self) -> dict[str, Any]:
        """Access the underlying engine dict (for observation encoding)."""
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
# DirectAdapter — wraps the engine directly (fast path for training)
# ---------------------------------------------------------------------------


class DirectAdapter:
    """Zero-overhead adapter that wraps the in-process engine.

    No serialization, no IPC, no copies.  ``raw_state`` returns the
    live engine dict directly.
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
        return bool(self._gs.get("won", False))


# ---------------------------------------------------------------------------
# BridgeAdapter — wraps any Backend (SimBackend or LiveBackend)
# ---------------------------------------------------------------------------

# The balatrobot endpoint each factored action type is sent to.
_BOT_METHOD_FOR_ACTION: dict[ActionType, str] = {
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
            if at == ActionType.UseConsumable:
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
        return bool(self._last_response.get("won", False))

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

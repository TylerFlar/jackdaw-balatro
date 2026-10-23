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

import math
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
        from jackdaw.engine.actions import SIM_ONLY_ACTIONS
        from jackdaw.engine.actions import get_legal_actions as engine_legal

        # The factored action space has no type for the sim-only actions
        # (RerollBoss, ...) yet; offer only what an env action can express.
        return [a for a in engine_legal(self._gs) if not isinstance(a, SIM_ONLY_ACTIONS)]

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

# The bot view's ``state`` for each engine phase.
_BOT_STATE: dict[GamePhase, str] = {
    GamePhase.BLIND_SELECT: "BLIND_SELECT",
    GamePhase.SELECTING_HAND: "SELECTING_HAND",
    GamePhase.ROUND_EVAL: "ROUND_EVAL",
    GamePhase.SHOP: "SHOP",
    GamePhase.PACK_OPENING: "SMODS_BOOSTER_OPENED",
    GamePhase.GAME_OVER: "GAME_OVER",
}


class DivergenceError(RuntimeError):
    """The simulator behind the bot protocol and the adapter's shadow run no
    longer agree on the game state."""


class BridgeAdapter:
    """Adapter that plays through BalatroBot's JSON-RPC API.

    Works with both :class:`~jackdaw.bridge.backend.SimBackend` (BalatroBot's
    API on the Rust simulator, in-process) and
    :class:`~jackdaw.bridge.backend.LiveBackend` (HTTP proxy to real Balatro).

    **SimBackend.**  Each action goes out as the request the bot layer
    translates it to (``SortHand`` and the swaps as ``rearrange``
    permutations).  The adapter also keeps a *shadow* run in the Python engine,
    stepped with the same moves; ``raw_state`` is the shadow's state dict (what
    the observation and mask code reads), and after every step the shadow is
    checked against the bot's view (:class:`DivergenceError` on a difference).
    Legal actions and the mask follow BALATROBOT legality, computed by the
    simulator: only what the live bot executes.

    **LiveBackend.**  ``raw_state`` is the last response converted by
    :func:`~jackdaw.bridge.balatrobot_adapter.bot_state_to_game_state`, and
    legality is the Python approximation in
    :mod:`jackdaw.bridge.balatrobot_adapter`.
    """

    def __init__(self, backend: Any) -> None:
        self._backend = backend
        self._last_response: dict[str, Any] = {}
        self._last_gs: dict[str, Any] = {}
        self._shadow: dict[str, Any] | None = None

    def _sim(self) -> Any:
        """The SimBackend, or None for any other backend."""
        from jackdaw.bridge.backend import SimBackend

        return self._backend if isinstance(self._backend, SimBackend) else None

    def reset(
        self,
        back_key: str,
        stake: int,
        seed: str,
        *,
        challenge: dict[str, Any] | None = None,
    ) -> GameState:
        """Start a run. BalatroBot's ``start`` has no challenge argument, so
        ``challenge`` is ignored, and a back the bot can't name plays as Red."""
        from jackdaw.bridge.backend import DECK_FROM_BOT, STAKE_FROM_BOT

        # Reverse-map engine keys to balatrobot enum strings
        deck_to_bot = {v: k for k, v in DECK_FROM_BOT.items()}
        stake_to_bot = {v: k for k, v in STAKE_FROM_BOT.items()}

        params: dict[str, Any] = {
            "deck": deck_to_bot.get(back_key, "RED"),
            "stake": stake_to_bot.get(stake, "WHITE"),
            "seed": seed,
        }
        # ``start`` needs the main menu.
        self._backend.handle("menu", {})
        self._last_response = self._backend.handle("start", params)
        if self._sim() is not None:
            from jackdaw.engine.run_init import initialize_run

            gs = initialize_run(
                DECK_FROM_BOT[params["deck"]], STAKE_FROM_BOT[params["stake"]], seed
            )
            gs["phase"] = GamePhase.BLIND_SELECT
            gs["blind_on_deck"] = "Small"
            self._shadow = gs
            self._check_shadow("start")
        else:
            self._shadow = None
        self._last_gs = self._build_gs()
        return _snapshot(self._last_gs)

    def step(self, action: Action) -> GameState:
        sim = self._sim()
        if sim is not None and self._shadow is not None:
            request = sim.request_for(action)
            if request is None:
                raise ValueError(f"{type(action).__name__} has no BalatroBot request")
            method, params = request
            refusal = sim.check(method, params)
            if refusal is not None:
                raise refusal
            self._step_shadow(action, method, params)
            self._last_response = sim.handle(method, params)
            self._check_shadow(method)
        else:
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
        sim = self._sim()
        if sim is not None and self._shadow is not None:
            return sim.legal_actions("balatrobot")

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
        if self._sim() is not None and self._shadow is not None:
            return self._restrict_to_legal(mask)

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
        # SimBackend: the shadow run's live engine dict
        if self._shadow is not None:
            return self._shadow
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

        For SimBackend, the shadow run's engine dict.
        For LiveBackend, converts the JSON response.
        """
        if self._shadow is not None:
            return self._shadow

        from jackdaw.bridge.balatrobot_adapter import bot_state_to_game_state

        return bot_state_to_game_state(self._last_response)

    def _restrict_to_legal(self, mask: ActionMask) -> ActionMask:
        """Keep the action types, and the entities of entity-targeted types,
        that some BALATROBOT-legal action uses."""
        from jackdaw.env.action_space import engine_action_to_factored

        gs = self.raw_state
        allowed: dict[ActionType, set[int]] = {}
        for action in self.get_legal_actions():
            try:
                fa = engine_action_to_factored(action, gs)
            except ValueError:  # an action with no factored type
                continue
            targets = allowed.setdefault(ActionType(fa.action_type), set())
            if fa.entity_target is not None:
                targets.add(int(fa.entity_target))
        for type_idx in np.nonzero(mask.type_mask)[0]:
            at = ActionType(int(type_idx))
            targets = allowed.get(at)
            entity_mask = mask.entity_masks.get(at)
            if targets is not None and entity_mask is not None and targets:
                for i in np.nonzero(entity_mask)[0]:
                    if int(i) not in targets:
                        entity_mask[i] = False
                if not entity_mask.any():
                    targets = None
            if targets is None:
                mask.type_mask[at] = False
                mask.entity_masks.pop(at, None)
        return mask

    def _step_shadow(self, action: Action, method: str, params: dict[str, Any]) -> None:
        """Take on the shadow run the move the bot makes: a ``rearrange`` is a
        permutation of the area (the hand's sort mode stays), anything else
        the action itself."""
        gs = self._shadow
        assert gs is not None
        if method == "rearrange":
            for area in ("hand", "jokers", "consumables"):
                if area in params:
                    cards = gs[area]
                    gs[area] = [cards[i] for i in params[area]]
            return
        from jackdaw.engine.game import step as engine_step

        engine_step(gs, action)

    def _check_shadow(self, method: str) -> None:
        """Compare the shadow run with the bot's view; raise on a difference."""
        gs = self._shadow
        assert gs is not None
        view = self._last_response
        cr = gs.get("current_round", {})
        rnd = view.get("round", {})
        expected = {
            "state": _BOT_STATE.get(gs.get("phase"), "?"),
            "money": gs.get("dollars", 0),
            "ante": gs.get("round_resets", {}).get("ante", 1),
            "round": gs.get("round", 0),
            "chips": gs.get("chips", 0),
            "hands_left": cr.get("hands_left", 0),
            "discards_left": cr.get("discards_left", 0),
            "won": bool(gs.get("won", False)),
            "hand": [c.card_key for c in gs.get("hand", [])],
            "jokers": [c.center_key for c in gs.get("jokers", [])],
            "consumables": [c.center_key for c in gs.get("consumables", [])],
        }
        got = {
            "state": view.get("state"),
            "money": view.get("money", 0),
            "ante": view.get("ante_num", 1),
            "round": view.get("round_num", 0),
            "chips": rnd.get("chips", 0),
            "hands_left": rnd.get("hands_left", 0),
            "discards_left": rnd.get("discards_left", 0),
            "won": bool(view.get("won", False)),
            "hand": [c["key"] for c in view.get("hand", {}).get("cards", [])],
            "jokers": [c["key"] for c in view.get("jokers", {}).get("cards", [])],
            "consumables": [c["key"] for c in view.get("consumables", {}).get("cards", [])],
        }
        diffs = [
            f"{k}: shadow {expected[k]!r}, bot {got[k]!r}"
            for k in expected
            if not _same(expected[k], got[k])
        ]
        if diffs:
            raise DivergenceError(f"after {method!r}: " + "; ".join(diffs))


def _same(a: Any, b: Any) -> bool:
    """Equal, with numbers compared to the 14 significant digits the bot's
    JSON encoder writes (``%.14g``)."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b or math.isclose(a, b, rel_tol=1e-13)
    return a == b

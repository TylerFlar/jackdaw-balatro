"""Tests for the unified game interface layer.

Covers:
- DirectAdapter reset/step/legal_actions cycle
- DirectAdapter runs to completion with a random agent
- BridgeAdapter(SimBackend) (BalatroBot's API on the simulator) plays as DirectAdapter
- done/won flags set correctly on game over and win
- GameAdapter protocol compliance
"""

from __future__ import annotations

import random

import pytest

from jackdaw.bridge.backend import SimBackend
from jackdaw.bridge.live import DivergenceError, LiveAdapter
from jackdaw.core import Game, Legality
from jackdaw.engine.actions import (
    CashOut,
    Discard,
    GamePhase,
    NextRound,
    OpenBooster,
    PlayHand,
    SelectBlind,
    SellCard,
    SkipPack,
    SortHand,
    SwapHandLeft,
    SwapHandRight,
    SwapJokersLeft,
    SwapJokersRight,
    UseConsumable,
)
from jackdaw.env.action_space import ActionType, get_action_mask
from jackdaw.env.game_interface import (
    BridgeAdapter,
    DirectAdapter,
    GameAdapter,
    GameState,
    game_state,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEED = "TEST_INTERFACE_42"
BACK = "b_red"
STAKE = 1


def _random_agent_step(adapter: GameAdapter) -> None:
    """Pick a random legal action, resolving marker actions."""
    legal = adapter.get_legal_actions()
    assert legal, "No legal actions available"

    hand = adapter.game.hand  # type: ignore[attr-defined]

    # Separate progress-making from utility
    progress = [
        a
        for a in legal
        if not isinstance(
            a, (SortHand, SwapHandLeft, SwapHandRight, SwapJokersLeft, SwapJokersRight)
        )
    ]
    pool = progress if progress else legal

    action = random.choice(pool)

    # Resolve marker PlayHand/Discard
    if isinstance(action, PlayHand) and not action.card_indices and hand:
        n = min(5, len(hand))
        count = random.randint(1, n)
        indices = tuple(sorted(random.sample(range(len(hand)), count)))
        action = PlayHand(card_indices=indices)

    if isinstance(action, Discard) and not action.card_indices and hand:
        n = min(5, len(hand))
        count = random.randint(1, n)
        indices = tuple(sorted(random.sample(range(len(hand)), count)))
        action = Discard(card_indices=indices)

    adapter.step(action)


def _run_to_completion(adapter: GameAdapter, max_actions: int = 5000) -> int:
    """Run the adapter to completion using random actions. Returns action count."""
    actions = 0
    while actions < max_actions:
        if adapter.done:
            break
        if adapter.won and adapter.game.phase == GamePhase.SHOP:  # type: ignore[attr-defined]
            break
        legal = adapter.get_legal_actions()
        if not legal:
            break
        _random_agent_step(adapter)
        actions += 1
    return actions


def _bridge() -> LiveAdapter:
    return BridgeAdapter(SimBackend())


# ---------------------------------------------------------------------------
# GameState dataclass tests
# ---------------------------------------------------------------------------


class TestGameState:
    def test_frozen(self):
        gs = GameState(
            phase=GamePhase.BLIND_SELECT,
            ante=1,
            round=0,
            dollars=4,
            hands_left=4,
            discards_left=3,
            hand_size=8,
            joker_slots=5,
            consumable_slots=2,
            blind_on_deck="Small",
            blind_chips=0,
            chips=0,
            won=False,
            done=False,
        )
        with pytest.raises(AttributeError):
            gs.dollars = 10  # type: ignore[misc]

    def test_fields(self):
        gs = GameState(
            phase=GamePhase.SELECTING_HAND,
            ante=2,
            round=3,
            dollars=15,
            hands_left=3,
            discards_left=2,
            hand_size=8,
            joker_slots=5,
            consumable_slots=2,
            blind_on_deck="Boss",
            blind_chips=450,
            chips=100,
            won=False,
            done=False,
        )
        assert gs.phase == GamePhase.SELECTING_HAND
        assert gs.ante == 2
        assert gs.dollars == 15
        assert gs.blind_chips == 450
        assert not gs.done


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_direct_adapter_satisfies_protocol(self):
        assert isinstance(DirectAdapter(), GameAdapter)

    def test_bridge_adapter_satisfies_protocol(self):
        assert isinstance(_bridge(), GameAdapter)

    def test_bridge_adapter_is_a_live_adapter(self):
        adapter = _bridge()
        assert isinstance(adapter, LiveAdapter)
        assert adapter.legality == Legality.BALATROBOT


# ---------------------------------------------------------------------------
# DirectAdapter tests
# ---------------------------------------------------------------------------


class TestDirectAdapter:
    def test_reset_returns_game_state(self):
        adapter = DirectAdapter()
        state = adapter.reset(BACK, STAKE, SEED)
        assert isinstance(state, GameState)
        assert state.phase == GamePhase.BLIND_SELECT
        assert state.blind_on_deck == "Small"
        assert state.dollars >= 0
        assert not state.done
        assert not state.won

    def test_reset_starts_the_run(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        game = adapter.game
        assert isinstance(game, Game)
        assert game.seed == SEED
        assert game.phase == GamePhase.BLIND_SELECT
        assert game_state(game).blind_on_deck == "Small"

    def test_step_select_blind(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        state = adapter.step(SelectBlind())
        assert state.phase == GamePhase.SELECTING_HAND
        assert state.hands_left > 0

    def test_legal_actions_blind_select(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        legal = adapter.get_legal_actions()
        assert any(isinstance(a, SelectBlind) for a in legal)

    def test_legal_actions_selecting_hand(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        adapter.step(SelectBlind())
        legal = adapter.get_legal_actions()
        assert any(isinstance(a, PlayHand) for a in legal)

    def test_done_false_initially(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        assert not adapter.done
        assert not adapter.won

    def test_game_is_the_live_run(self):
        """``game`` is the run itself, not a copy: a step moves it on."""
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        game = adapter.game
        assert adapter.game is game
        adapter.step(SelectBlind())
        assert game.phase == GamePhase.SELECTING_HAND

    def test_game_before_reset_raises(self):
        with pytest.raises(RuntimeError, match="reset"):
            DirectAdapter().game  # noqa: B018

    def test_run_to_completion_random(self):
        """DirectAdapter completes a full run with random agent."""
        random.seed(12345)
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        actions = _run_to_completion(adapter, max_actions=5000)
        assert actions > 0
        # Game should have ended
        assert adapter.done or adapter.won

    def test_reset_clears_previous_run(self):
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        adapter.step(SelectBlind())
        # Reset should start fresh
        state = adapter.reset(BACK, STAKE, "DIFFERENT_SEED")
        assert state.phase == GamePhase.BLIND_SELECT

    def test_reset_starts_a_challenge(self):
        adapter = DirectAdapter()
        adapter.reset("b_challenge", 1, SEED, challenge="c_omelette_1")
        assert adapter.game.jokers  # Omelette starts with five Egg jokers


# ---------------------------------------------------------------------------
# BridgeAdapter(SimBackend) tests
# ---------------------------------------------------------------------------


class TestBridgeAdapterSim:
    def test_reset_returns_game_state(self):
        adapter = _bridge()
        state = adapter.reset(BACK, STAKE, SEED)
        assert isinstance(state, GameState)
        assert state.phase == GamePhase.BLIND_SELECT
        assert not state.done

    def test_step_select_blind(self):
        adapter = _bridge()
        adapter.reset(BACK, STAKE, SEED)
        state = adapter.step(SelectBlind())
        assert state.phase == GamePhase.SELECTING_HAND

    def test_legal_actions(self):
        adapter = _bridge()
        adapter.reset(BACK, STAKE, SEED)
        legal = adapter.get_legal_actions()
        assert any(isinstance(a, SelectBlind) for a in legal)

    def test_game_is_the_shadow_run(self):
        """The adapter's ``game`` is its shadow run, kept in step with the simulator
        behind the bot protocol."""
        backend = SimBackend()
        adapter = BridgeAdapter(backend)
        adapter.reset(BACK, STAKE, SEED)
        adapter.step(SelectBlind())
        game = adapter.game
        assert game.phase == GamePhase.SELECTING_HAND
        view = backend.view()
        assert view["money"] == game.dollars
        assert len(view["hand"]["cards"]) == len(game.hand)

    def test_a_sort_is_a_permutation(self):
        """SortHand goes out as a ``rearrange`` permutation: the hand is sorted
        by suit on both sides."""
        adapter = _bridge()
        adapter.reset(BACK, STAKE, SEED)
        adapter.step(SelectBlind())
        adapter.step(SortHand(mode="suit"))
        suits = [c.suit for c in adapter.game.hand]
        assert suits == sorted(suits, key=suits.index)  # grouped by suit
        live = [c["key"] for c in adapter.backend.view()["hand"]["cards"]]
        assert live == [c.front for c in adapter.game.hand]

    def test_a_divergence_is_reported(self):
        adapter = _bridge()
        adapter.reset(BACK, STAKE, SEED)
        adapter.game.debug_set(money=adapter.game.dollars + 1)  # the shadow drifts
        with pytest.raises(DivergenceError, match="money"):
            adapter.step(SelectBlind())

    def test_done_won_flags(self):
        adapter = _bridge()
        adapter.reset(BACK, STAKE, SEED)
        assert not adapter.done
        assert not adapter.won

    def test_run_to_completion(self):
        """BridgeAdapter(SimBackend) completes a run with random agent."""
        random.seed(12345)
        adapter = _bridge()
        adapter.reset(BACK, STAKE, SEED)
        actions = _run_to_completion(adapter, max_actions=5000)
        assert actions > 0
        assert adapter.done or adapter.won


def _debug(adapter: GameAdapter, method: str, params: dict) -> None:
    """BalatroBot's ``add`` / ``set`` on the adapter's run (for the bridge, on both the
    shadow and the simulator behind it)."""
    if isinstance(adapter, LiveAdapter):
        adapter.request(method, params)
    elif method == "add":
        adapter.game.debug_add(**params)  # type: ignore[attr-defined]
    else:
        adapter.game.debug_set(**params)  # type: ignore[attr-defined]


def _to_shop(adapter: GameAdapter) -> None:
    """Beat the Small Blind (``set`` its chips) and cash out."""
    adapter.step(SelectBlind())
    _debug(adapter, "set", {"chips": adapter.game.blind.chips})  # type: ignore[attr-defined]
    adapter.step(PlayHand(card_indices=(0,)))
    adapter.step(CashOut())
    assert adapter.game.phase == GamePhase.SHOP  # type: ignore[attr-defined]


def _open_pack_with_tray_planet(adapter: GameAdapter) -> None:
    """In a pack (with picks left), holding a Pluto."""
    _to_shop(adapter)
    _debug(adapter, "add", {"key": "c_pluto"})
    _debug(adapter, "set", {"money": 50})
    adapter.step(OpenBooster(card_index=0))
    assert adapter.game.phase == GamePhase.PACK_OPENING  # type: ignore[attr-defined]


class TestBridgeFollowsBalatrobot:
    """Vanilla lets a consumable be used or sold while a pack is open;
    balatrobot's use/sell endpoints only run in SELECTING_HAND and SHOP.
    The bridge offers only what the bot will execute."""

    def test_legal_actions(self):
        for adapter, offered in ((DirectAdapter(), True), (_bridge(), False)):
            adapter.reset(BACK, STAKE, SEED)
            _open_pack_with_tray_planet(adapter)
            legal = adapter.get_legal_actions()
            assert any(isinstance(a, UseConsumable) for a in legal) is offered
            assert any(isinstance(a, SellCard) for a in legal) is offered

    def test_env_mask(self):
        from jackdaw.env.balatro_env import BalatroEnvironment
        from jackdaw.env.game_spec import FactoredAction

        for factory, offered in ((DirectAdapter, True), (_bridge, False)):
            env = BalatroEnvironment(adapter_factory=factory, back_keys=[BACK], stakes=[STAKE])
            env.reset(seed=SEED)
            adapter = env._adapter
            _to_shop(adapter)
            _debug(adapter, "add", {"key": "c_pluto"})
            _debug(adapter, "set", {"money": 50})
            *_, info = env.step(FactoredAction(ActionType.OpenBooster, entity_target=0))
            mask = info["action_mask"]
            assert bool(mask.type_mask[ActionType.UseConsumable]) is offered
            assert bool(mask.type_mask[ActionType.SellConsumable]) is offered
            assert mask.type_mask[ActionType.PickPackCard]  # the bot can pick

    def test_card_selecting_consumable_masked_in_the_shop(self):
        # use.lua refuses a consumable that needs card selection outside
        # SELECTING_HAND, even where the endpoint itself is allowed.
        adapter = _bridge()
        adapter.reset(BACK, STAKE, SEED)
        _to_shop(adapter)
        _debug(adapter, "add", {"key": "c_pluto"})
        _debug(adapter, "add", {"key": "c_magician"})
        mask = get_action_mask(adapter.game, Legality.BALATROBOT)
        assert list(mask.entity_masks[ActionType.UseConsumable]) == [True, False]


# ---------------------------------------------------------------------------
# DirectAdapter vs BridgeAdapter(SimBackend) equivalence
# ---------------------------------------------------------------------------


class TestEquivalence:
    """DirectAdapter and BridgeAdapter(SimBackend) must produce identical
    behavior for the same seed when driven by the same deterministic agent.
    """

    def test_identical_sequence(self):
        """Both adapters produce the same state sequence with greedy agent."""
        direct = DirectAdapter()
        bridge = _bridge()

        ds = direct.reset(BACK, STAKE, SEED)
        bs = bridge.reset(BACK, STAKE, SEED)

        assert ds.phase == bs.phase
        assert ds.dollars == bs.dollars
        assert ds.ante == bs.ante

        max_steps = 200
        for i in range(max_steps):
            if direct.done or bridge.done:
                break
            if direct.won and direct.game.phase == GamePhase.SHOP:
                break

            # Take same greedy action on both
            d_legal = direct.get_legal_actions()
            b_legal = bridge.get_legal_actions()

            if not d_legal or not b_legal:
                break

            # Use direct's legal actions to pick action, apply to both
            hand = direct.game.hand

            action = None
            for a in d_legal:
                if isinstance(a, (SelectBlind, CashOut, NextRound, SkipPack)):
                    action = a
                    break

            if action is None:
                for a in d_legal:
                    if isinstance(a, PlayHand) and hand:
                        n = min(5, len(hand))
                        action = PlayHand(card_indices=tuple(range(n)))
                        break

            if action is None:
                action = d_legal[0]

            ds = direct.step(action)
            bs = bridge.step(action)

            # Core state fields must match
            assert ds == bs, f"Step {i}: {ds} != {bs}"
            assert direct.game.state_hash() == bridge.game.state_hash(), f"Step {i}"

        # Both must end in the same terminal state
        assert direct.done == bridge.done
        assert direct.won == bridge.won


# ---------------------------------------------------------------------------
# Game over / win flag tests
# ---------------------------------------------------------------------------


class TestTerminalFlags:
    def test_game_over_sets_done(self):
        """When the game reaches GAME_OVER, done must be True."""
        random.seed(99999)
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        _run_to_completion(adapter, max_actions=5000)

        if adapter.game.phase == GamePhase.GAME_OVER:
            assert adapter.done is True

    def test_won_flag_on_win(self):
        """If the run is won, adapter.won must be True."""
        # A random agent that may or may not win: the flag must agree with the run.
        random.seed(77777)
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        _run_to_completion(adapter, max_actions=5000)

        assert adapter.won == game_state(adapter.game).won

    def test_done_consistent_with_phase(self):
        """adapter.done must match phase == GAME_OVER."""
        adapter = DirectAdapter()
        adapter.reset(BACK, STAKE, SEED)
        # Not done at start
        assert not adapter.done
        assert adapter.game.phase != GamePhase.GAME_OVER

    def test_bridge_done_flag(self):
        """BridgeAdapter done/won flags match its shadow run."""
        random.seed(88888)
        adapter = _bridge()
        adapter.reset(BACK, STAKE, SEED)
        _run_to_completion(adapter, max_actions=5000)

        assert adapter.won == game_state(adapter.game).won
        assert adapter.done == (adapter.game.phase == GamePhase.GAME_OVER)

"""Tests for live_env module.

Covers:
- SimBridgeBalatroEnv produces same trajectory as DirectAdapter (key correctness test)
- LiveBalatroEnv connection error handling (mocked HTTP)
- validate_episode detects divergences and reports clean runs
- LiveBackend tests marked @pytest.mark.live
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from jackdaw.engine.actions import (
    Action,
    CashOut,
    Discard,
    GamePhase,
    NextRound,
    PlayHand,
    ReorderJokers,
    SelectBlind,
    SkipPack,
    SortHand,
)
from jackdaw.env.game_interface import DirectAdapter, GameAdapter, GameState
from jackdaw.env.live_env import (
    BalatrobotConnectionError,
    LiveBalatroEnv,
    SimBridgeBalatroEnv,
    StepDivergence,
    ValidationResult,
    validate_episode,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = "TEST_LIVE_ENV_42"
BACK = "b_red"
STAKE = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _greedy_agent(adapter: GameAdapter) -> Action | None:
    """Deterministic greedy agent for validation tests.

    Picks actions in a fixed priority order so both envs see identical choices.
    """
    legal = adapter.get_legal_actions()
    if not legal:
        return None

    hand = adapter.raw_state.get("hand", [])

    # Filter out non-progress actions
    progress = [a for a in legal if not isinstance(a, (SortHand, ReorderJokers))]
    pool = progress if progress else legal

    # Priority: SelectBlind > CashOut > NextRound > SkipPack > PlayHand > first
    for a in pool:
        if isinstance(a, SelectBlind):
            return a
    for a in pool:
        if isinstance(a, CashOut):
            return a
    for a in pool:
        if isinstance(a, NextRound):
            return a
    for a in pool:
        if isinstance(a, SkipPack):
            return a

    # Play hand with first N cards
    for a in pool:
        if isinstance(a, PlayHand) and hand:
            n = min(5, len(hand))
            return PlayHand(card_indices=tuple(range(n)))

    # Discard first N cards
    for a in pool:
        if isinstance(a, Discard) and hand:
            n = min(5, len(hand))
            return Discard(card_indices=tuple(range(n)))

    return pool[0]


def _run_to_completion(adapter: GameAdapter, max_actions: int = 5000) -> int:
    """Run adapter to completion with greedy agent. Returns action count."""
    actions = 0
    while actions < max_actions:
        if adapter.done:
            break
        phase = adapter.raw_state.get("phase")
        if adapter.won and phase == GamePhase.SHOP:
            break
        action = _greedy_agent(adapter)
        if action is None:
            break
        adapter.step(action)
        actions += 1
    return actions


# ---------------------------------------------------------------------------
# SimBridgeBalatroEnv tests
# ---------------------------------------------------------------------------


class TestSimBridgeBalatroEnv:
    """Core correctness: SimBridgeBalatroEnv must match DirectAdapter."""

    def test_reset_returns_game_state(self):
        env = SimBridgeBalatroEnv()
        state = env.reset(BACK, STAKE, SEED)
        assert isinstance(state, GameState)
        assert state.phase == GamePhase.BLIND_SELECT
        assert not state.done

    def test_step_select_blind(self):
        env = SimBridgeBalatroEnv()
        env.reset(BACK, STAKE, SEED)
        state = env.step(SelectBlind())
        assert state.phase == GamePhase.SELECTING_HAND

    def test_legal_actions(self):
        env = SimBridgeBalatroEnv()
        env.reset(BACK, STAKE, SEED)
        legal = env.get_legal_actions()
        assert any(isinstance(a, SelectBlind) for a in legal)

    def test_run_to_completion(self):
        env = SimBridgeBalatroEnv()
        env.reset(BACK, STAKE, SEED)
        actions = _run_to_completion(env, max_actions=5000)
        assert actions > 0
        assert env.done or env.won

    def test_same_trajectory_as_direct(self):
        """Key correctness test: SimBridgeBalatroEnv must produce the same
        state sequence as DirectAdapter for the same seed and agent."""
        direct = DirectAdapter()
        bridge = SimBridgeBalatroEnv()

        ds = direct.reset(BACK, STAKE, SEED)
        bs = bridge.reset(BACK, STAKE, SEED)

        assert ds.phase == bs.phase
        assert ds.dollars == bs.dollars
        assert ds.ante == bs.ante

        for i in range(300):
            if direct.done or bridge.done:
                break
            d_phase = direct.raw_state.get("phase")
            if direct.won and d_phase == GamePhase.SHOP:
                break

            action = _greedy_agent(direct)
            if action is None:
                break

            ds = direct.step(action)
            bs = bridge.step(action)

            assert ds.phase == bs.phase, f"Step {i}: phase {ds.phase} != {bs.phase}"
            assert ds.dollars == bs.dollars, f"Step {i}: dollars {ds.dollars} != {bs.dollars}"
            assert ds.ante == bs.ante, f"Step {i}: ante mismatch"
            assert ds.round == bs.round, f"Step {i}: round mismatch"
            assert ds.chips == bs.chips, f"Step {i}: chips mismatch"
            assert ds.hands_left == bs.hands_left, f"Step {i}: hands_left mismatch"
            assert ds.discards_left == bs.discards_left, f"Step {i}: discards_left mismatch"

        assert direct.done == bridge.done
        assert direct.won == bridge.won


# ---------------------------------------------------------------------------
# LiveBalatroEnv connection error handling tests
# ---------------------------------------------------------------------------


class TestLiveBalatroEnvConnection:
    """Test connection management with mocked HTTP calls."""

    def test_health_check_success(self):
        env = LiveBalatroEnv(host="127.0.0.1", port=99999)
        with patch.object(env._backend, "handle", return_value={"status": "ok"}):
            assert env.health_check() is True

    def test_health_check_failure(self):
        env = LiveBalatroEnv(host="127.0.0.1", port=99999)
        with patch.object(env._backend, "handle", side_effect=ConnectionError("refused")):
            assert env.health_check() is False

    def test_reset_raises_on_no_connection(self):
        env = LiveBalatroEnv(host="127.0.0.1", port=99999, retries=1)
        with patch.object(env._backend, "handle", side_effect=ConnectionError("refused")):
            with pytest.raises(BalatrobotConnectionError, match="Cannot reach balatrobot"):
                env.reset(BACK, STAKE, SEED)

    def test_reset_retries_on_transient_error(self):
        env = LiveBalatroEnv(host="127.0.0.1", port=99999, retries=3)

        # Health check succeeds, but first two reset attempts fail
        call_count = 0
        health_response = {"status": "ok"}
        reset_response = {
            "state": "BLIND_SELECT",
            "round_num": 0,
            "ante_num": 1,
            "money": 4,
            "deck": "RED",
            "stake": "WHITE",
            "seed": SEED,
            "won": False,
            "used_vouchers": {},
            "hands": {},
            "round": {
                "hands_left": 4,
                "hands_played": 0,
                "discards_left": 3,
                "discards_used": 0,
                "reroll_cost": 5,
                "chips": 0,
            },
            "blinds": {
                "small": {
                    "type": "SMALL",
                    "status": "SELECT",
                    "name": "Small Blind",
                    "effect": "No special effect",
                    "score": 100,
                    "tag_name": "",
                    "tag_effect": "",
                },
                "big": {
                    "type": "BIG",
                    "status": "UPCOMING",
                    "name": "Big Blind",
                    "effect": "No special effect",
                    "score": 200,
                    "tag_name": "",
                    "tag_effect": "",
                },
                "boss": {
                    "type": "BOSS",
                    "status": "UPCOMING",
                    "name": "The Hook",
                    "effect": "The Hook",
                    "score": 300,
                    "tag_name": "",
                    "tag_effect": "",
                },
            },
            "jokers": {"count": 0, "limit": 5, "highlighted_limit": 0, "cards": []},
            "consumables": {"count": 0, "limit": 2, "highlighted_limit": 0, "cards": []},
            "cards": {"count": 52, "limit": 52, "highlighted_limit": 0, "cards": []},
            "hand": {"count": 0, "limit": 8, "highlighted_limit": 5, "cards": []},
            "shop": {"count": 0, "limit": 0, "highlighted_limit": 0, "cards": []},
            "vouchers": {"count": 0, "limit": 0, "highlighted_limit": 0, "cards": []},
            "packs": {"count": 0, "limit": 0, "highlighted_limit": 0, "cards": []},
            "pack": {"count": 0, "limit": 0, "highlighted_limit": 0, "cards": []},
        }

        def mock_handle(method, params):
            nonlocal call_count
            if method == "health":
                return health_response
            call_count += 1
            if call_count <= 2:
                raise TimeoutError("connection timed out")
            return reset_response

        with patch.object(env._backend, "handle", side_effect=mock_handle):
            state = env.reset(BACK, STAKE, SEED)
            assert state.phase == GamePhase.BLIND_SELECT

    def test_reset_fails_after_all_retries(self):
        env = LiveBalatroEnv(host="127.0.0.1", port=99999, retries=2)

        def mock_handle(method, params):
            if method == "health":
                return {"status": "ok"}
            raise TimeoutError("timed out")

        with patch.object(env._backend, "handle", side_effect=mock_handle):
            with pytest.raises(BalatrobotConnectionError, match="All 2 attempts failed"):
                env.reset(BACK, STAKE, SEED)

    def test_step_retries_on_error(self):
        env = LiveBalatroEnv(host="127.0.0.1", port=99999, retries=2)

        call_count = 0
        step_response = {
            "state": "SELECTING_HAND",
            "round_num": 0,
            "ante_num": 1,
            "money": 4,
            "won": False,
            "round": {
                "hands_left": 4,
                "discards_left": 3,
                "chips": 0,
                "hands_played": 0,
                "discards_used": 0,
                "reroll_cost": 5,
            },
        }

        def mock_handle(method, params):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise TimeoutError("timed out")
            return step_response

        # Pre-set adapter state so step doesn't fail on missing _last_response
        env._adapter._last_response = {"state": "BLIND_SELECT"}
        env._adapter._last_gs = {"phase": GamePhase.BLIND_SELECT}

        with patch.object(env._backend, "handle", side_effect=mock_handle):
            state = env.step(SelectBlind())
            assert state.phase == GamePhase.SELECTING_HAND

    def test_error_message_includes_instructions(self):
        env = LiveBalatroEnv(host="10.0.0.1", port=12346)
        with patch.object(env._backend, "handle", side_effect=ConnectionError):
            with pytest.raises(BalatrobotConnectionError) as exc_info:
                env.reset(BACK, STAKE, SEED)
            msg = str(exc_info.value)
            assert "balatrobot" in msg.lower()
            assert "10.0.0.1:12346" in msg

    def test_done_and_won_delegate(self):
        env = LiveBalatroEnv(host="127.0.0.1", port=99999)
        env._adapter._last_response = {"state": "GAME_OVER", "won": True}
        assert env.done is True
        assert env.won is True


# ---------------------------------------------------------------------------
# validate_episode tests
# ---------------------------------------------------------------------------


class TestValidateEpisode:
    def test_clean_validation(self):
        """validate_episode reports no divergences for identical envs."""
        direct = DirectAdapter()
        bridge = SimBridgeBalatroEnv()

        result = validate_episode(
            direct,
            bridge,
            SEED,
            _greedy_agent,
            max_steps=300,
        )
        assert result.ok, f"Divergences: {result.divergences}"
        assert result.steps > 0
        assert result.ref_done == result.test_done
        assert result.ref_won == result.test_won

    def test_detects_state_divergence(self):
        """validate_episode catches divergences when test env differs."""

        class DriftingAdapter:
            """Adapter that adds $1 every step to create divergence."""

            def __init__(self):
                self._inner = DirectAdapter()
                self._step_count = 0

            def reset(self, back_key, stake, seed, *, challenge=None):
                self._step_count = 0
                return self._inner.reset(back_key, stake, seed, challenge=challenge)

            def step(self, action):
                self._step_count += 1
                self._inner.step(action)
                # Inject divergence: add $1 to raw state
                self._inner._gs["dollars"] += 1
                # Return snapshot with altered dollars
                from jackdaw.env.game_interface import _snapshot

                return _snapshot(self._inner._gs)

            def get_legal_actions(self):
                return self._inner.get_legal_actions()

            @property
            def raw_state(self):
                return self._inner.raw_state

            @property
            def done(self):
                return self._inner.done

            @property
            def won(self):
                return self._inner.won

        ref = DirectAdapter()
        drifting = DriftingAdapter()

        result = validate_episode(
            ref,
            drifting,
            SEED,
            _greedy_agent,
            max_steps=50,
        )
        assert not result.ok
        # Should have at least one dollars divergence
        dollar_divs = [d for d in result.divergences if d.field == "dollars"]
        assert len(dollar_divs) > 0

    def test_validation_result_fields(self):
        result = ValidationResult(seed="TEST", steps=10)
        assert result.ok
        result.divergences.append(StepDivergence(step=1, field="dollars", expected=4, actual=5))
        assert not result.ok


# ---------------------------------------------------------------------------
# Live backend tests (require running balatrobot)
# ---------------------------------------------------------------------------


@pytest.mark.live
class TestLiveBackend:
    """Tests that require a running balatrobot instance.

    Run with: pytest -m live
    """

    def test_live_health_check(self):
        env = LiveBalatroEnv()
        assert env.health_check(), "balatrobot is not running"

    def test_live_reset_and_step(self):
        env = LiveBalatroEnv()
        if not env.health_check():
            pytest.skip("balatrobot not running")
        state = env.reset(BACK, STAKE, SEED)
        assert state.phase == GamePhase.BLIND_SELECT
        state = env.step(SelectBlind())
        assert state.phase == GamePhase.SELECTING_HAND

    def test_live_run_to_completion(self):
        env = LiveBalatroEnv()
        if not env.health_check():
            pytest.skip("balatrobot not running")
        env.reset(BACK, STAKE, SEED)
        actions = _run_to_completion(env, max_actions=5000)
        assert actions > 0

    def test_live_vs_direct_validation(self):
        """Validate that live backend matches direct engine."""
        live = LiveBalatroEnv()
        if not live.health_check():
            pytest.skip("balatrobot not running")
        direct = DirectAdapter()
        result = validate_episode(
            direct,
            live,
            SEED,
            _greedy_agent,
            max_steps=300,
        )
        assert result.ok, f"Divergences: {result.divergences[:5]}"

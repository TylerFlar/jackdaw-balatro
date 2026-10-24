"""Tests for entity-based observation encoding.

Covers:
- Encode initial game state, verify shapes and dtypes
- Encode after playing a hand, verify hand cards reduced
- Encode with jokers, verify joker entity count matches
- Encode with face-down cards: the flag is set and the identity hidden
- Encode empty areas produce shape (0, D) arrays
- Center key mapping covers all keys in centers.json
- No NaN/Inf in encoded observations across 100 random-agent steps

The encoder is the Rust engine's (``test_rust_observation.py`` checks it against the
Python encoder's frozen outputs); states the policy doesn't reach on its own are built
by editing the run's snapshot.
"""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np
import pytest

from jackdaw.core import Game
from jackdaw.engine.actions import (
    CashOut,
    Discard,
    GamePhase,
    PlayHand,
    SelectBlind,
    SortHand,
    SwapHandLeft,
    SwapHandRight,
    SwapJokersLeft,
    SwapJokersRight,
)
from jackdaw.engine.data import load
from jackdaw.env.observation import (
    _CENTER_KEY_TO_ID,
    D_CONSUMABLE,
    D_GLOBAL,
    D_JOKER,
    D_PLAYING_CARD,
    D_SHOP,
    NUM_CENTER_KEYS,
    Observation,
    center_key_id,
    encode_observation,
)
from jackdaw.env.rust_observation import encode_snapshot

SEED = "OBS_TEST_42"
BACK = "b_red"
STAKE = 1


def _new_game(seed: str = SEED) -> Game:
    return Game.new(BACK, STAKE, seed)


def _dealt(seed: str = SEED) -> Game:
    game = _new_game(seed)
    game.step(SelectBlind())
    assert game.phase == GamePhase.SELECTING_HAND
    return game


def _shop(seed: str = SEED) -> Game:
    game = _dealt(seed)
    game.debug_set(chips=game.blind.chips)
    game.step(PlayHand(card_indices=(0,)))
    game.step(CashOut())
    assert game.phase == GamePhase.SHOP
    return game


# ---------------------------------------------------------------------------
# Shape and dtype tests
# ---------------------------------------------------------------------------


class TestShapesAndDtypes:
    def test_initial_state_shapes(self):
        obs = encode_observation(_new_game())

        assert isinstance(obs, Observation)
        assert obs.global_context.shape == (D_GLOBAL,)
        assert obs.global_context.dtype == np.float32

        # At BLIND_SELECT, hand is empty (cards not dealt yet)
        for name, width in (
            ("hand_cards", D_PLAYING_CARD),
            ("jokers", D_JOKER),
            ("consumables", D_CONSUMABLE),
            ("shop_cards", D_SHOP),
            ("pack_cards", D_PLAYING_CARD),
        ):
            arr = getattr(obs, name)
            assert arr.ndim == 2 and arr.shape[1] == width, name
            assert arr.dtype == np.float32, name

    def test_selecting_hand_has_cards(self):
        game = _dealt()
        obs = encode_observation(game)
        assert len(game.hand) > 0
        assert obs.hand_cards.shape == (len(game.hand), D_PLAYING_CARD)

    def test_global_context_dimension(self):
        obs = encode_observation(_new_game())
        assert obs.global_context.shape == (D_GLOBAL,)
        # 211 base features + 24 strategic ones
        assert D_GLOBAL == 235


# ---------------------------------------------------------------------------
# Hand card count after playing
# ---------------------------------------------------------------------------


class TestPlayHand:
    def test_hand_reduced_after_play(self):
        game = _dealt()
        hand_before = len(game.hand)
        assert encode_observation(game).hand_cards.shape[0] == hand_before

        # Play first 3 cards; the hand refills, or the round ends
        game.step(PlayHand(card_indices=tuple(range(min(3, hand_before)))))
        obs_after = encode_observation(game)
        assert obs_after.hand_cards.ndim == 2
        assert obs_after.hand_cards.shape[1] == D_PLAYING_CARD
        assert obs_after.hand_cards.shape[0] == len(game.hand)


# ---------------------------------------------------------------------------
# Joker encoding
# ---------------------------------------------------------------------------


class TestJokerEncoding:
    def test_joker_count_matches(self):
        game = _new_game()
        assert encode_observation(game).jokers.shape[0] == len(game.jokers) == 0
        game.debug_add("j_joker")
        game.debug_add("j_egg")
        assert encode_observation(game).jokers.shape == (2, D_JOKER)

    def test_single_joker_features(self):
        game = _new_game()
        game.debug_add("j_joker")
        vec = encode_observation(game).jokers[0]
        # center id, normalized
        assert vec[0] == np.float32(center_key_id("j_joker") / NUM_CENTER_KEYS)
        # sell_value should be encoded
        assert vec[3] > 0.0


# ---------------------------------------------------------------------------
# Face-down cards (boss blind)
# ---------------------------------------------------------------------------


def _hand_card(snap: dict[str, Any], i: int) -> dict[str, Any]:
    return snap["areas"]["hand"][i]


class TestFaceDownCards:
    def test_face_down_flag(self):
        snap = _dealt().to_dict()
        _hand_card(snap, 0)["facing"] = "back"
        vec = encode_snapshot(snap).hand_cards[0]
        assert vec.shape == (D_PLAYING_CARD,)

        # face_down flag at index 7
        assert vec[7] == 1.0
        # rank/suit/enhancement should be zeroed when face-down
        assert vec[0] == 0.0  # rank_id
        assert vec[1] == 0.0  # suit
        assert vec[3] == 0.0  # enhancement

    def test_face_up_card(self):
        game = _dealt()
        game.debug_add("H_A")
        vec = encode_observation(game).hand_cards[-1]
        assert vec[7] == 0.0  # not face down
        assert vec[0] == pytest.approx(14.0 / 14.0)  # Ace


_SUITS = ("Spades", "Hearts", "Clubs", "Diamonds")
_RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace")
_KEY_RANK = {"10": "T", "Jack": "J", "Queen": "Q", "King": "K", "Ace": "A"}


def _identities(game: Game) -> dict[str, tuple[str, dict[str, Any]]]:
    """Every playing card identity's ``front`` and ``base``, made by BalatroBot's ``add``
    in a copy of the run."""
    out = {}
    for suit in _SUITS:
        for rank in _RANKS:
            key = f"{suit[0]}_{_KEY_RANK.get(rank, rank)}"
            scratch = game.clone()
            scratch.debug_add(key)
            card = scratch.to_dict()["areas"]["hand"][-1]
            out[key] = (card["front"], card["base"])
    return out


def _encodings(snap: dict[str, Any]) -> list[np.ndarray]:
    """The parts of the observation that see the hand."""
    obs = encode_snapshot(snap)
    return [obs.global_context, obs.hand_cards, obs.jokers]


def _same(ref: list[np.ndarray], snap: dict[str, Any], where: str) -> None:
    for a, b in zip(ref, _encodings(snap), strict=True):
        np.testing.assert_array_equal(a, b, err_msg=where)


class TestFaceDownHidesIdentity:
    """A face-down card's identity must not reach any part of the observation
    (issue #25): two states a player cannot tell apart encode identically."""

    def test_hidden_hand_card_identity_not_observable(self):
        game = _dealt("LEAK1")
        identities = _identities(game)
        base = game.to_dict()
        for i in (0, 3, 7):
            snap = base
            _hand_card(snap, i)["facing"] = "back"
            ref = _encodings(snap)
            card = _hand_card(snap, i)
            saved = dict(card)
            for key, (front, card_base) in identities.items():
                card["front"], card["base"] = front, card_base
                _same(ref, snap, f"card {i} as {key}")
            card.update(saved)
            # Enhancements (Stone, Wild change hand evaluation), editions,
            # seals and debuff are hidden too.
            for enh in ("m_stone", "m_wild", "m_glass", "m_lucky"):
                card["center"] = enh
                _same(ref, snap, f"card {i} {enh}")
            card.update(saved)
            card["edition"] = {"polychrome": True, "x_mult": 1.5, "type": "polychrome"}
            card["seal"] = "Red"
            card["debuff"] = True
            _same(ref, snap, f"card {i} edition/seal/debuff")
            card.update(saved)
            card["facing"] = "front"

    def test_hidden_card_excluded_from_best_hand(self):
        snap = _dealt("LEAK1").to_dict()
        for card in snap["areas"]["hand"][1:]:
            card["facing"] = "back"
        obs = encode_snapshot(snap)
        # A lone visible card can only form High Card, and it is that card.
        assert obs.hand_cards[0, 13] == 1.0
        assert not obs.hand_cards[1:, 13].any()
        assert (obs.hand_cards[1:, [7]] == 1.0).all()

    def test_hidden_debuffed_card_not_counted(self):
        snap = _dealt("LEAK1").to_dict()
        n = len(snap["areas"]["hand"])
        _hand_card(snap, 0)["debuff"] = True
        assert encode_snapshot(snap).global_context[232] == pytest.approx(1.0 / n)
        _hand_card(snap, 0)["facing"] = "back"
        assert encode_snapshot(snap).global_context[232] == 0.0

    def test_face_down_joker_encodes_position_only(self):
        game = _dealt("LEAK1")
        for key in ("j_joker", "j_merry_andy", "j_blueprint"):
            game.debug_add(key)
        snap = game.to_dict()
        face_up = encode_snapshot(snap).jokers
        assert face_up[:, 0].all()

        snap["areas"]["jokers"][1]["facing"] = "back"
        rows = encode_snapshot(snap).jokers
        expected = np.zeros(D_JOKER, dtype=np.float32)
        expected[9] = 1 / 20.0
        np.testing.assert_array_equal(rows[1], expected)
        np.testing.assert_array_equal(rows[[0, 2]], face_up[[0, 2]])

        # Which joker is face down in that slot changes nothing.
        scratch = game.clone()
        made = scratch.debug_add("j_cavendish")
        other = next(c for c in scratch.to_dict()["areas"]["jokers"] if c["sort_id"] == made)
        other["sort_id"] = snap["areas"]["jokers"][1]["sort_id"]
        other["facing"] = "back"
        snap["areas"]["jokers"][1] = other
        np.testing.assert_array_equal(encode_snapshot(snap).jokers, rows)

    def test_amber_acorn_jokers_hidden(self):
        game = _new_game("ACORN")
        for key in ("j_joker", "j_greedy_joker", "j_blueprint"):
            game.debug_add(key)
        game.apply_setup({"op": "force_boss", "key": "bl_final_acorn"})
        game.step(SelectBlind())
        assert all(j.face_down for j in game.jokers)

        rows = encode_observation(game).jokers
        np.testing.assert_array_equal(rows[:, 9], np.arange(3, dtype=np.float32) / 20.0)
        assert not np.delete(rows, 9, axis=1).any()


# ---------------------------------------------------------------------------
# Empty areas
# ---------------------------------------------------------------------------


class TestEmptyAreas:
    def test_empty_jokers(self):
        assert encode_observation(_new_game()).jokers.shape == (0, D_JOKER)

    def test_empty_consumables(self):
        assert encode_observation(_new_game()).consumables.shape == (0, D_CONSUMABLE)

    def test_empty_shop(self):
        assert encode_observation(_new_game()).shop_cards.shape == (0, D_SHOP)

    def test_empty_pack(self):
        assert encode_observation(_new_game()).pack_cards.shape == (0, D_PLAYING_CARD)

    def test_empty_hand(self):
        assert encode_observation(_new_game()).hand_cards.shape == (0, D_PLAYING_CARD)


# ---------------------------------------------------------------------------
# Center key mapping coverage
# ---------------------------------------------------------------------------


class TestCenterKeyMapping:
    def test_covers_all_centers_json_keys(self):
        missing = [key for key in load("centers") if key not in _CENTER_KEY_TO_ID]
        assert not missing, f"Center keys missing from mapping: {missing}"

    def test_ids_are_contiguous(self):
        ids = sorted(_CENTER_KEY_TO_ID.values())
        assert ids[0] == 1  # 0 is reserved for unknown
        assert ids[-1] == len(ids)  # contiguous from 1..N

    def test_unknown_key_returns_zero(self):
        assert center_key_id("nonexistent_key") == 0

    def test_num_center_keys_matches(self):
        assert NUM_CENTER_KEYS == len(_CENTER_KEY_TO_ID)
        assert NUM_CENTER_KEYS > 0

    def test_the_encoder_uses_the_same_ids(self):
        game = _shop()
        game.debug_add("j_blueprint")
        game.debug_add("c_pluto")
        obs = encode_observation(game)
        assert obs.jokers[0, 0] == np.float32(center_key_id("j_blueprint") / NUM_CENTER_KEYS)
        assert obs.consumables[0, 0] == np.float32(center_key_id("c_pluto") / NUM_CENTER_KEYS)
        shop = [*game.shop_jokers, *game.shop_vouchers, *game.shop_boosters]
        expected = [np.float32(center_key_id(c.key) / NUM_CENTER_KEYS) for c in shop]
        assert obs.shop_cards[:, 0].tolist() == expected


# ---------------------------------------------------------------------------
# No NaN/Inf across random-agent steps
# ---------------------------------------------------------------------------


class TestNoNaNInf:
    def test_100_random_steps(self):
        """Encode observations across 100 random-agent steps.

        Verify no NaN or Inf values appear in any array.
        """
        rng = random.Random(54321)
        game = _new_game()

        def _check_obs(obs: Observation, step_num: int) -> None:
            for name in (
                "global_context",
                "hand_cards",
                "jokers",
                "consumables",
                "shop_cards",
                "pack_cards",
            ):
                arr = getattr(obs, name)
                assert not np.any(np.isnan(arr)), f"NaN in {name} at step {step_num}"
                assert not np.any(np.isinf(arr)), f"Inf in {name} at step {step_num}"

        _check_obs(encode_observation(game), 0)

        for step_num in range(1, 101):
            if game.phase == GamePhase.GAME_OVER:
                break
            legal = game.legal_actions()
            if not legal:
                break

            # Pick a random non-utility action
            progress = [
                a
                for a in legal
                if not isinstance(
                    a, (SortHand, SwapHandLeft, SwapHandRight, SwapJokersLeft, SwapJokersRight)
                )
            ]
            action = rng.choice(progress or legal)

            # Resolve marker actions
            hand = game.hand
            if isinstance(action, (PlayHand, Discard)) and not action.card_indices and hand:
                count = rng.randint(1, min(5, len(hand)))
                action = type(action)(tuple(sorted(rng.sample(range(len(hand)), count))))

            game.step(action)
            _check_obs(encode_observation(game), step_num)


# ---------------------------------------------------------------------------
# Global context specific tests
# ---------------------------------------------------------------------------


class TestGlobalContext:
    def test_phase_one_hot(self):
        phase_vec = encode_observation(_new_game()).global_context[0:6]
        # BLIND_SELECT = index 0
        assert phase_vec[0] == 1.0
        assert np.sum(phase_vec) == 1.0

    def test_phase_changes_after_select(self):
        phase_vec = encode_observation(_dealt()).global_context[0:6]
        # SELECTING_HAND = index 1
        assert phase_vec[1] == 1.0
        assert np.sum(phase_vec) == 1.0

    def test_blind_on_deck_one_hot(self):
        bod_vec = encode_observation(_new_game()).global_context[6:10]
        # "Small" = index 1
        assert bod_vec[1] == 1.0
        assert np.sum(bod_vec) == 1.0

    def test_hand_levels_encoded(self):
        # Hand levels start at index 30, 12 * 5 = 60 values
        hl_vec = encode_observation(_new_game()).global_context[30:90]
        # All should be non-negative (levels, chips, mult are positive)
        assert np.all(hl_vec >= 0.0)
        # At least some visible hands should have non-zero chips/mult
        assert np.sum(hl_vec) > 0.0


# ---------------------------------------------------------------------------
# Encoding individual entities
# ---------------------------------------------------------------------------


class TestIndividualEncoders:
    def test_encode_playing_card_all_suits(self):
        game = _dealt()
        for letter in "HDCS":
            game.debug_add(f"{letter}_A")
        rows = encode_observation(game).hand_cards[-4:]
        assert rows.shape == (4, D_PLAYING_CARD)
        assert (rows[:, 0] > 0.0).all()  # rank_id
        assert sorted(rows[:, 1].tolist()) == pytest.approx([0.0, 1 / 3, 2 / 3, 1.0])

    def test_encode_consumable(self):
        game = _new_game()
        game.debug_add("c_strength")
        vec = encode_observation(game).consumables[0]
        assert vec.shape == (D_CONSUMABLE,)
        assert vec[0] > 0.0  # center_key_id

    def test_encode_shop_item(self):
        game = _shop()
        game.debug_set(money=50)
        rows = encode_observation(game).shop_cards
        assert rows.shape[1] == D_SHOP
        assert (rows[:, 3] == 1.0).all()  # affordable
        assert (rows[: len(game.shop_jokers), 4] == 1.0).all()  # a free slot (no jokers)

    def test_shop_item_affordable_on_credit(self):
        game = _shop()
        game.debug_set(money=0)
        assert not encode_observation(game).shop_cards[:, 3].any()
        game.debug_add("j_credit_card")  # bankrupt_at -20
        assert encode_observation(game).shop_cards[:, 3].all()

    def test_consumable_can_use_matches_the_mask(self):
        """A targeting tarot reads usable once the hand can satisfy it."""
        game = _new_game()
        game.debug_add("c_magician")
        assert encode_observation(game).consumables[0, 3] == 0.0  # blind select: no hand
        game.step(SelectBlind())
        assert encode_observation(game).consumables[0, 3] == 1.0


class TestEndlessOverflowEncoding:
    """Endless mode overflows blind targets (ante 39+) and big scores to
    inf; the encoding must stay finite for the network."""

    def test_overflowed_blind_and_score_encode_finite(self):
        game = _new_game()
        game.debug_set(ante=39)
        game.step(SelectBlind())
        assert math.isnan(game.blind.chips)
        snap = game.to_dict()
        for chips in (0, 10**300, math.inf, math.nan):
            snap["game"]["chips"] = chips
            obs = encode_snapshot(snap)
            assert np.isfinite(obs.global_context).all(), f"chips={chips}"

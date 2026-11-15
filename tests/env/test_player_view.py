"""The observation is the player's view: no card identity the player hasn't seen reaches it.

Each test builds states a player can't tell apart (they differ only in hidden information)
and checks that the observation encodes them all the same, while the engine tells them apart
(their state hashes differ). States are built by editing a run's snapshot, as in
``test_observation.py``; every comparison covers the whole observation, not just the features
the player's view decides.
"""

from __future__ import annotations

import copy
import random
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from typing import Any

import numpy as np
import pytest

from jackdaw.core import Game
from jackdaw.engine.actions import (
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
from jackdaw.env import (
    BalatroEnvironment,
    BalatroGymnasiumEnv,
    BalatroVectorEnv,
    DirectAdapter,
)
from jackdaw.env.action_space import ActionType
from jackdaw.env.observation import Observation, encode_observation
from jackdaw.env.rust_gym import factored_from_parts
from jackdaw.env.rust_observation import encode_snapshot

Snapshot = dict[str, Any]

#: The bosses that deal cards face down (``Blind:stay_flipped``, blind.lua:605-622).
FACE_DOWN_BOSSES = ("bl_wheel", "bl_house", "bl_mark", "bl_fish")
#: A challenge with the ``flipped_cards`` rule (X-ray Vision).
FLIPPED_CHALLENGE = "c_xray_1"
#: What tells one playing card from another, as a snapshot records it.
IDENTITY = ("front", "base")

HISTOGRAM = slice(159, 211)
#: ``global[28]``: the discard pile's size.
DISCARD_SIZE = 28


def _dealt(seed: str) -> Game:
    game = Game.new("b_red", 1, seed)
    game.step(SelectBlind())
    assert game.phase == GamePhase.SELECTING_HAND
    return game


def _arrays(obs: Observation) -> list[np.ndarray]:
    return [
        obs.global_context,
        obs.hand_cards,
        obs.jokers,
        obs.consumables,
        obs.shop_cards,
        obs.pack_cards,
    ]


def _encode(snap: Snapshot) -> list[np.ndarray]:
    return _arrays(encode_snapshot(snap))


def _state_hash(snap: Snapshot) -> str:
    return Game.from_dict(snap).state_hash()


def _equal(a: list[np.ndarray], b: list[np.ndarray]) -> bool:
    return all(x.shape == y.shape and x.tobytes() == y.tobytes() for x, y in zip(a, b, strict=True))


def _check_hidden(base: Snapshot, variants: Iterable[tuple[str, Snapshot]]) -> None:
    """Every variant is a state a player can't tell from ``base``: the observation encodes
    each one exactly as ``base``, though the engine tells them apart."""
    ref, ref_hash = _encode(base), _state_hash(base)
    told_apart = 0
    for label, snap in variants:
        assert _equal(ref, _encode(snap)), f"the observation sees {label}"
        told_apart += _state_hash(snap) != ref_hash
    assert told_apart, "the variants should be other states"


def _swap_identity(a: dict[str, Any], b: dict[str, Any]) -> None:
    for key in IDENTITY:
        a[key], b[key] = b[key], a[key]


def _swaps(snap: Snapshot, area: str, index: int) -> Iterator[tuple[str, Snapshot]]:
    """``snap`` with card ``index`` of ``area`` exchanged for each undrawn card in turn."""
    for j in range(len(snap["areas"]["deck"])):
        s = copy.deepcopy(snap)
        _swap_identity(s["areas"][area][index], s["areas"]["deck"][j])
        yield f"{area}[{index}] <-> deck[{j}]", s


def _boss(key: str, disabled: bool = False) -> Callable[[Snapshot], None]:
    def apply(snap: Snapshot) -> None:
        snap["blind"]["key"] = key
        snap["blind"]["boss"] = True
        snap["blind"]["disabled"] = disabled

    return apply


def _flipped_cards_rule(snap: Snapshot) -> None:
    snap["modifiers"]["flipped_cards"] = 4


#: The rounds that can deal face down: each boss (also once disabled, since it may already have
#: dealt face down) and the challenge rule.
FACE_DOWN_ROUNDS = {
    **{key: _boss(key) for key in FACE_DOWN_BOSSES},
    "bl_wheel disabled": _boss("bl_wheel", disabled=True),
    "flipped_cards": _flipped_cards_rule,
}


# ---------------------------------------------------------------------------
# 1. Discard pile: cards discarded face down
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("round_kind", sorted(FACE_DOWN_ROUNDS))
def test_card_discarded_unseen_is_hidden(round_kind: str) -> None:
    """A card dealt face down and discarded unseen: which undrawn card it was can't show."""
    snap = _dealt("PV_DISCARD").to_dict()
    FACE_DOWN_ROUNDS[round_kind](snap)
    snap["areas"]["discard"].append(snap["areas"]["deck"].pop())
    _check_hidden(snap, _swaps(snap, "discard", -1))
    assert not encode_snapshot(snap).global_context[HISTOGRAM].any()


def test_discards_show_in_other_rounds() -> None:
    """Without face-down dealing the player saw every discard: the histogram shows each one."""
    game = _dealt("PV_DISCARD")
    game.step(Discard(card_indices=(0, 1, 2)))
    assert np.count_nonzero(encode_observation(game).global_context[HISTOGRAM]) == 3


def test_wheel_round_from_the_engine() -> None:
    """The Wheel as the engine plays it: a card it dealt face down, then discarded."""
    for n in range(50):
        game = Game.new("b_red", 1, f"PV_WHEEL_{n}")
        game.apply_setup({"op": "force_boss", "key": "bl_wheel"})
        game.step(SelectBlind())
        hand = game.to_dict()["areas"]["hand"]
        down = [i for i, c in enumerate(hand) if c["facing"] == "back"]
        if down:
            break
    else:
        pytest.fail("no face-down card in 50 Wheel deals")
    game.step(Discard(card_indices=(down[0],)))
    snap = game.to_dict()
    assert snap["blind"]["key"] == "bl_wheel" and len(snap["areas"]["discard"]) == 1
    assert not encode_observation(game).global_context[HISTOGRAM].any()
    _check_hidden(snap, _swaps(snap, "discard", -1))


# ---------------------------------------------------------------------------
# 2. Draw outs: the exact draw pile after a face-down deal
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("round_kind", sorted(FACE_DOWN_ROUNDS))
@pytest.mark.parametrize("index", [0, 4, 7])
def test_face_down_hand_card_against_the_deck(round_kind: str, index: int) -> None:
    """A face-down hand card exchanged with an undrawn card: the outs can't tell."""
    snap = _dealt("PV_OUTS").to_dict()
    FACE_DOWN_ROUNDS[round_kind](snap)
    snap["areas"]["hand"][index]["facing"] = "back"
    _check_hidden(snap, _swaps(snap, "hand", index))


# ---------------------------------------------------------------------------
# 3. Stone cards: the hidden base rank and suit
# ---------------------------------------------------------------------------


def _bases(game: Game) -> dict[str, dict[str, Any]]:
    """Each of the 52 identities' ``front`` and ``base``, made by BalatroBot's ``add`` in a
    copy of the run."""
    out = {}
    for suit in "SHCD":
        for rank in "23456789TJQKA":
            scratch = game.clone()
            scratch.debug_add(f"{suit}_{rank}")
            card = scratch.to_dict()["areas"]["hand"][-1]
            out[f"{suit}_{rank}"] = {key: card[key] for key in IDENTITY}
    return out


def _stone_variants(
    snap: Snapshot, area: str, index: int, bases: dict[str, dict[str, Any]]
) -> Iterator[tuple[str, Snapshot]]:
    for key, identity in bases.items():
        s = copy.deepcopy(snap)
        s["areas"][area][index].update(copy.deepcopy(identity))
        yield f"Stone in {area}[{index}] on {key}", s


@pytest.mark.parametrize("index", [0, 3, 7])
def test_stone_in_hand_hides_its_base(index: int) -> None:
    game = _dealt("PV_STONE")
    bases = _bases(game)
    snap = game.to_dict()
    snap["areas"]["hand"][index]["center"] = "m_stone"
    _check_hidden(snap, _stone_variants(snap, "hand", index, bases))
    row = encode_snapshot(snap).hand_cards[index]
    assert row[3] == pytest.approx(6 / 8)  # still a Stone (enhancement)
    assert not row[[0, 1, 2, 8]].any()


def test_stone_in_deck_hides_its_base() -> None:
    game = _dealt("PV_STONE")
    snap = game.to_dict()
    snap["areas"]["deck"][0]["center"] = "m_stone"
    _check_hidden(snap, _stone_variants(snap, "deck", 0, _bases(game)))


def test_stone_in_discard_hides_its_base() -> None:
    game = _dealt("PV_STONE")
    snap = game.to_dict()
    snap["areas"]["discard"].append(snap["areas"]["deck"].pop())
    snap["areas"]["discard"][-1]["center"] = "m_stone"
    _check_hidden(snap, _stone_variants(snap, "discard", -1, _bases(game)))


@pytest.mark.parametrize("area", ["hand", "pack_cards"])
def test_stones_alone_hide_which_is_highest(area: str) -> None:
    """Only Stones face up: High Card's pick (column 13) can't follow their hidden bases."""
    game = _dealt("PV_STONE")
    bases = _bases(game)
    snap = game.to_dict()
    hand = snap["areas"]["hand"]
    if area == "pack_cards":
        snap["areas"]["pack_cards"] = [hand.pop() for _ in range(3)]
    for card in snap["areas"][area]:
        card["center"] = "m_stone"
    _check_hidden(snap, _stone_variants(snap, area, 0, bases))
    obs = encode_snapshot(snap)
    rows = obs.hand_cards if area == "hand" else obs.pack_cards
    np.testing.assert_array_equal(rows[:, 13], np.eye(len(rows), dtype=np.float32)[0])


# ---------------------------------------------------------------------------
# 4. times_played: copies that look the same
# ---------------------------------------------------------------------------


def test_play_count_hidden() -> None:
    snap = _dealt("PV_PLAYED").to_dict()
    hand = snap["areas"]["hand"]

    def played(i: int, n: int) -> tuple[str, Snapshot]:
        s = copy.deepcopy(snap)
        s["areas"]["hand"][i]["base"]["times_played"] = n
        return f"hand[{i}] played {n} times", s

    _check_hidden(snap, (played(i, n) for i in range(len(hand)) for n in (1, 5)))
    assert not encode_snapshot(snap).hand_cards[:, 11].any()


# ---------------------------------------------------------------------------
# States the engine reaches
# ---------------------------------------------------------------------------

#: What the player can't see of a hidden card, as a snapshot records it.
HIDDEN_FIELDS = ("center", "front", "base", "edition", "seal")
#: The snapshot areas that hold playing cards.
CARD_AREAS = ("deck", "hand", "discard", "pack_cards")
#: The 52 cards, as ``debug_add`` names them.
CARD_KEYS = [f"{suit}_{rank}" for suit in "SHCD" for rank in "23456789TJQKA"]


def _engine_states(seed: str, runs: int, steps: int) -> Iterator[Snapshot]:
    """Snapshots of random play in rounds that deal face down, with Stone cards. Each run meets
    one of the four bosses every round (or plays X-ray Vision) with Marble Joker, Stones join the
    hand, and some blinds are beaten outright so that runs reach the shop and its packs."""
    rng = random.Random(seed)
    idle = (SortHand, SwapHandLeft, SwapHandRight, SwapJokersLeft, SwapJokersRight)
    for run in range(runs):
        boss = (*FACE_DOWN_BOSSES, None)[run % 5]
        game = Game.new("b_red", 1, f"{seed}_{run}", None if boss else FLIPPED_CHALLENGE)
        game.debug_add("j_marble")
        game.debug_set(money=200)
        for _ in range(steps):
            if game.phase == GamePhase.GAME_OVER:
                break
            if game.phase == GamePhase.BLIND_SELECT and boss:
                game.apply_setup({"op": "force_boss", "key": boss})
            if game.phase == GamePhase.SELECTING_HAND:
                if rng.random() < 0.3:
                    game.debug_add(rng.choice(CARD_KEYS), enhancement="m_stone")
                if rng.random() < 0.15:
                    game.debug_set(chips=1e12)  # the next hand beats the blind
            yield game.to_dict()
            legal = game.legal_actions()
            if not legal:
                break
            action = rng.choice([a for a in legal if not isinstance(a, idle)] or legal)
            hand = game.hand
            if isinstance(action, (PlayHand, Discard)) and not action.card_indices and hand:
                count = rng.randint(1, min(5, len(hand)))
                action = type(action)(tuple(sorted(rng.sample(range(len(hand)), count))))
            game.step(action)


def _hidden_cards(snap: Snapshot) -> list[dict[str, Any]]:
    """The cards the player can't see: the deck, the face-down hand cards and, in a round that
    can deal face down, the discard pile."""
    areas = snap["areas"]
    cards = areas["deck"] + [c for c in areas["hand"] if c["facing"] == "back"]
    if (snap["blind"] or {}).get("key") in FACE_DOWN_BOSSES or "flipped_cards" in snap["modifiers"]:
        cards += areas["discard"]
    return cards


def _shuffle_hidden(snap: Snapshot, rng: random.Random) -> Snapshot:
    s = copy.deepcopy(snap)
    cards = _hidden_cards(s)
    identities = [{key: card[key] for key in HIDDEN_FIELDS} for card in cards]
    rng.shuffle(identities)
    for card, identity in zip(cards, identities, strict=True):
        card.update(identity)
    return s


def _rebase_stones(snap: Snapshot, rng: random.Random, bases: list[dict[str, Any]]) -> Snapshot:
    s = copy.deepcopy(snap)
    for area in CARD_AREAS:
        for card in s["areas"][area]:
            if card["center"] == "m_stone":
                card.update(copy.deepcopy(rng.choice(bases)))
    return s


def _replay_counts(snap: Snapshot, rng: random.Random) -> Snapshot:
    s = copy.deepcopy(snap)
    for area in CARD_AREAS:
        for card in s["areas"][area]:
            if card["base"]:
                card["base"]["times_played"] = rng.randint(0, 30)
    return s


def test_engine_states_hide_what_the_player_cant_see() -> None:
    """On states the engine reaches in face-down rounds with Stone cards, every state a player
    can't tell apart has the same observation: the hidden cards' identities shuffled, each Stone
    on another base, other play counts, and all three at once. A feature that starts reading a
    hidden card field fails this."""
    rng = random.Random(55)
    deck = Game.new("b_red", 1, "PV_BASES").to_dict()["areas"]["deck"]
    bases = [{key: card[key] for key in IDENTITY} for card in deck]
    seen: Counter[str] = Counter()
    for snap in _engine_states("PV_ENGINE", runs=10, steps=25):
        observation, state = _encode(snap), _state_hash(snap)
        hand = snap["areas"]["hand"]
        seen["face down"] += any(card["facing"] == "back" for card in hand)
        seen["stones"] += any(
            card["center"] == "m_stone" for area in CARD_AREAS for card in snap["areas"][area]
        )
        variants = {
            "hidden cards shuffled": _shuffle_hidden(snap, rng),
            "stones rebased": _rebase_stones(snap, rng, bases),
            "play counts": _replay_counts(snap, rng),
            "all three": _replay_counts(
                _rebase_stones(_shuffle_hidden(snap, rng), rng, bases), rng
            ),
        }
        for label, variant in variants.items():
            assert _equal(observation, _encode(variant)), f"the observation sees {label}"
            seen["told apart"] += _state_hash(variant) != state
    assert min(seen["face down"], seen["stones"], seen["told apart"]) >= 20, seen


# ---------------------------------------------------------------------------
# The envs serve it
# ---------------------------------------------------------------------------


def _index(table: list[Any], action_type: ActionType) -> int:
    return next(i for i, p in enumerate(table) if factored_from_parts(p).action_type == action_type)


def test_envs_hide_face_down_discards() -> None:
    """After a discard in an X-ray Vision run (cards dealt face down), the discard histogram
    from ``BalatroGymnasiumEnv`` and ``BalatroVectorEnv`` is blank, the pile not empty."""
    gym = BalatroGymnasiumEnv(DirectAdapter)
    gym.reset(seed=1)
    gym._core.reset("b_red", 1, "PV_XRAY", FLIPPED_CHALLENGE, 7, mask=True)
    for action_type in (ActionType.SelectBlind, ActionType.Discard):
        obs, *_ = gym.step(_index(gym._core.action_table(), action_type))

    vec = BalatroVectorEnv(1)
    vec.reset(seed=1)
    out, masks = vec._outputs()
    vec._core.reset([("b_red", 1, "PV_XRAY", FLIPPED_CHALLENGE, 7)], out, masks.view(np.uint8))
    for action_type in (ActionType.SelectBlind, ActionType.Discard):
        vobs, *_ = vec.step(np.array([_index(vec._core.action_table(0), action_type)]))

    for context in (obs["global"], vobs["global"][0]):
        assert context[DISCARD_SIZE] > 0
        assert not context[HISTOGRAM].any()


def test_balatro_environment_hides_a_stones_base() -> None:
    """A Stone added to the hand: its row shows the Stone, not its base."""
    env = BalatroEnvironment(adapter_factory=DirectAdapter)
    env.reset()
    *_, info = env.step(factored_from_parts((int(ActionType.SelectBlind), None, None)))
    info["game"].debug_add("H_A", enhancement="m_stone")
    *_, info = env.step(factored_from_parts((int(ActionType.SortHandRank), None, None)))
    hand = info["observation"].hand_cards
    rows = hand[hand[:, 3] == np.float32(6 / 8)]
    assert len(rows) == 1
    assert not rows[:, [0, 1, 2, 8]].any()

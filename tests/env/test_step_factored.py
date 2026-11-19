"""Factored steps (``BalatroGymnasiumEnv.offers`` / ``step_factored``, ``_core.GymEnv``'s).

The action table samples: past 200 card selections of a kind (a hand of 8 cards or more) or 500
actions in all, it holds a random subset. A factored step reaches any action the table would
hold if it held them all, and steps exactly as the index naming it would.
"""

from __future__ import annotations

import itertools
import json
import random
from collections import Counter
from collections.abc import Iterator

import numpy as np
import pytest

from jackdaw.core import Game, IllegalActionError
from jackdaw.engine.actions import GamePhase
from jackdaw.env import BalatroGymnasiumEnv, DirectAdapter, get_action_mask
from jackdaw.env.action_space import ActionType
from jackdaw.env.game_spec import FactoredAction
from jackdaw.env.gymnasium_wrapper import _offers
from jackdaw.env.sampler import Xoshiro256
from tests.env.generic import GenericAdapter

N_TYPES = len(ActionType)


_CARD_TYPES = (ActionType.PlayHand, ActionType.Discard)
_ENTITY_CARD_TYPES = (ActionType.UseConsumable, ActionType.PickPackCard)


def _candidates(n_hand: int, n_entities: int) -> list[FactoredAction]:
    """The factored actions worth asking about over a hand of ``n_hand`` cards and up to
    ``n_entities`` entities: every target shape for every type, every selection of up to 5
    cards (one past the hand too) for the types that take cards, and a few out of order."""
    sels: list[tuple[int, ...]] = [()]
    for k in range(1, 6):
        sels.extend(itertools.combinations(range(n_hand + 1), k))
    sels.extend([(1, 0), (2, 0, 1), (0, 0)])
    out = []
    for t in range(N_TYPES):
        out.append(FactoredAction(action_type=t))
        out.append(FactoredAction(action_type=t, card_target=(0,)))
        for e in range(n_entities + 1):
            out.append(FactoredAction(action_type=t, entity_target=e))
            out.append(FactoredAction(action_type=t, entity_target=e, card_target=(0,)))
        if t in _CARD_TYPES:
            out.extend(FactoredAction(action_type=t, card_target=sel) for sel in sels)
        if t in _ENTITY_CARD_TYPES:
            for e in range(n_entities + 1):
                out.extend(
                    FactoredAction(action_type=t, entity_target=e, card_target=sel) for sel in sels
                )
    return out


def _states(seed: int, steps: int = 160) -> Iterator[Game]:
    """A run through every phase: random actions of the action table, money to spend, and
    each blind won once a hand is played (half the time) so the run goes on to the shops,
    packs and consumables."""
    game = Game.new("b_red", 1, f"OFFERS_{seed}")
    rng = random.Random(seed)
    state = Xoshiro256(seed).state
    for _ in range(steps):
        if game.phase == GamePhase.GAME_OVER:
            return
        if game.phase == GamePhase.SHOP and game.dollars < 20:
            game.debug_set(money=40)
        if game.phase == GamePhase.SELECTING_HAND and rng.random() < 0.5:
            game.debug_set(chips=game.blind.chips)
        yield game
        actions, state = game._g.action_table(state)
        if not actions:
            return
        t, e, cards = rng.choice(actions)
        try:
            game.step(json.loads(game._g.factored_to_engine(t, e, cards)))
        except IllegalActionError:
            continue


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_offers_is_the_unsampled_table(seed: int) -> None:
    """Every table entry is offered; when the table didn't sample, nothing else is; and the
    generic path's check agrees with the Rust one, in every phase of a run."""
    phases: Counter[str] = Counter()
    unsampled = 0
    for game in _states(seed):
        table, _ = game._g.action_table(Xoshiro256(0).state)
        listed = {(t, e, None if c is None else tuple(c)) for t, e, c in table}
        per_type = Counter((t, e) for t, e, c in table if c is not None)
        sampled = len(table) == 500 or any(n == 200 for n in per_type.values())
        mask, info = get_action_mask(game), {"game": game}
        areas = (game.consumables, game.pack_cards, game.jokers, game.shop_jokers, game.hand)
        for fa in _candidates(len(game.hand), max(len(a) for a in areas)):
            parts = (fa.action_type, fa.entity_target, fa.card_target)
            offered = game._g._offers(*parts)
            assert offered == _offers(mask, info, fa), fa
            if parts in listed:
                assert offered, fa
            elif not sampled:
                assert not offered, fa
        phases[game.phase.value] += 1
        unsampled += not sampled
    assert {"blind_select", "selecting_hand", "shop"} <= set(phases), phases
    assert unsampled > 10


def _pair(seed: int) -> tuple[BalatroGymnasiumEnv, BalatroGymnasiumEnv]:
    a, b = BalatroGymnasiumEnv(DirectAdapter), BalatroGymnasiumEnv(DirectAdapter)
    a.reset(seed=seed)
    b.reset(seed=seed)
    return a, b


@pytest.mark.parametrize("seed", [4, 5])
def test_a_factored_step_is_the_index_step(seed: int) -> None:
    """Stepping a table entry by index or as a factored action: the same episode, the same
    tables after it (the sampler draws the same)."""
    by_index, by_action = _pair(seed)
    rng = random.Random(seed)
    for _ in range(400):
        n = int(by_index.action_masks().sum())
        if n == 0:
            break
        i = rng.randrange(n)
        fa = by_index._action_table[i]
        o1, r1, t1, u1, _ = by_index.step(i)
        o2, r2, t2, u2, _ = by_action.step_factored(fa)
        assert (r1, t1, u1) == (r2, t2, u2)
        assert all(np.array_equal(o1[k], o2[k]) for k in o1)
        assert by_index._core.rng_state == by_action._core.rng_state
        assert list(by_index._action_table) == list(by_action._action_table)
        if t1 or u1:
            break
    assert by_index.game.state_hash() == by_action.game.state_hash()


def test_a_sampled_out_play_is_reachable() -> None:
    """The issue's repro (jackdaw-arena#18): with 8 cards some plays aren't in the table."""
    env = BalatroGymnasiumEnv(DirectAdapter)
    env.reset(seed=1)
    for _ in range(3):
        t = list(env._action_table)
        env.step(next((i for i, a in enumerate(t) if a.action_type == ActionType.SelectBlind), 0))
    plays = {a.card_target for a in env._action_table if a.action_type == ActionType.PlayHand}
    every = [c for k in range(1, 6) for c in itertools.combinations(range(8), k)]
    missing = [c for c in every if c not in plays]
    assert len(env.game.hand) == 8 and missing
    for sel in missing:
        assert env.offers(FactoredAction(action_type=ActionType.PlayHand, card_target=sel))
    before = env.game
    env.step_factored(FactoredAction(action_type=ActionType.PlayHand, card_target=missing[0]))
    assert env.game.hands_left == before.hands_left - 1


def test_replays_mix_indices_and_actions() -> None:
    """An episode given as a mix of indices and factored actions replays exactly."""
    env = BalatroGymnasiumEnv(DirectAdapter)
    env.reset(seed=9)
    rng = random.Random(9)
    steps: list[int | FactoredAction] = []
    for _ in range(300):
        n = int(env.action_masks().sum())
        if n == 0:
            break
        game = env.game
        if game.phase.value == "selecting_hand" and rng.random() < 0.5 and len(game.hand) >= 5:
            sel = tuple(sorted(rng.sample(range(len(game.hand)), rng.randint(1, 5))))
            fa = FactoredAction(action_type=ActionType.Discard, card_target=sel)
            if not env.offers(fa):
                fa = FactoredAction(action_type=ActionType.PlayHand, card_target=sel)
            steps.append(fa)
            _, _, t, u, _ = env.step_factored(fa)
        else:
            i = rng.randrange(n)
            steps.append(i)
            _, _, t, u, _ = env.step(i)
        if t or u:
            break
    replay = BalatroGymnasiumEnv(DirectAdapter)
    replay.reset(seed=9)
    for s in steps:
        replay.step(s) if isinstance(s, int) else replay.step_factored(s)
    assert replay.game.state_hash() == env.game.state_hash()
    assert replay._core.rng_state == env._core.rng_state


def test_what_isnt_offered_is_refused() -> None:
    env = BalatroGymnasiumEnv(DirectAdapter)
    env.reset(seed=3)
    env.step_factored(FactoredAction(action_type=ActionType.SelectBlind))
    before = env.game.state_hash()
    for fa in [
        FactoredAction(action_type=ActionType.PlayHand, card_target=(1, 0)),  # not ascending
        FactoredAction(action_type=ActionType.PlayHand, card_target=(0, 1, 2, 3, 4, 5)),
        FactoredAction(action_type=ActionType.PlayHand, card_target=(9,)),
        FactoredAction(action_type=ActionType.PlayHand),
        FactoredAction(action_type=ActionType.CashOut),
        FactoredAction(action_type=ActionType.BuyCard, entity_target=0),
    ]:
        assert not env.offers(fa), fa
        with pytest.raises(IllegalActionError):
            env.step_factored(fa)
    assert env.game.state_hash() == before
    # Tuples work as well as FactoredAction.
    assert env.offers((ActionType.PlayHand, None, (0, 1)))


def test_an_ended_episode_offers_nothing() -> None:
    env = BalatroGymnasiumEnv(DirectAdapter, max_steps=1)
    env.reset(seed=3)
    _, _, _, truncated, _ = env.step_factored(FactoredAction(action_type=ActionType.SelectBlind))
    assert truncated
    assert not env.offers(FactoredAction(action_type=ActionType.PlayHand, card_target=(0,)))


@pytest.mark.parametrize("copy", [True, False])
def test_the_generic_path_steps_factored_actions(copy: bool) -> None:
    """The generic path (any adapter but ``DirectAdapter``) plays the same episode."""
    rust = BalatroGymnasiumEnv(DirectAdapter, copy=copy)
    generic = BalatroGymnasiumEnv(GenericAdapter, copy=copy)  # type: ignore[arg-type]
    rust.reset(seed=11)
    generic.reset(seed=11)
    rng = random.Random(11)
    for _ in range(200):
        game = rust.game
        if game.phase.value == "selecting_hand":
            sel = tuple(sorted(rng.sample(range(len(game.hand)), min(len(game.hand), 3))))
            fa = FactoredAction(action_type=ActionType.PlayHand, card_target=sel)
        else:
            fa = rust._action_table[rng.randrange(len(rust._action_table))]
        assert rust.offers(fa) and generic.offers(fa)
        o1, r1, t1, u1, _ = rust.step_factored(fa)
        o2, r2, t2, u2, _ = generic.step_factored(fa)
        assert (r1, t1, u1) == (r2, t2, u2)
        assert all(np.array_equal(o1[k], o2[k]) for k in o1)
        if t1 or u1:
            break
    assert rust.game.state_hash() == generic.game.state_hash()


def test_the_env_player_view() -> None:
    """The env's view: the run's, with the last hand's events when recording, under a key that
    the sampler seed sets (the same seed, the same ids), on both paths."""
    views = []
    for adapter in (DirectAdapter, GenericAdapter):
        env = BalatroGymnasiumEnv(adapter)  # type: ignore[arg-type]
        env.record_scoring = True
        env.reset(seed=21)
        env.step_factored(FactoredAction(action_type=ActionType.SelectBlind))
        env.step_factored(FactoredAction(action_type=ActionType.PlayHand, card_target=(0, 1)))
        v = env.player_view()
        assert v["last_play"] is not None and v["phase"] == "selecting_hand"
        views.append(v)
    assert views[0] == views[1]
    other = BalatroGymnasiumEnv(DirectAdapter)
    other.reset(seed=22)
    assert other.player_view()["last_play"] is None

"""A game built from the player view (``Game.from_player_view``, ``ViewTracker``).

At every decision of whole runs the run's view is built into a game, on the last one stepped
with the action since. That game must show what the view shows (its own view is the run's, card
ids aside), offer the same actions and give the same observation; and compared with the run
field by field, it may differ only where the view hides something (#66).
"""

from __future__ import annotations

import collections
import json
import random
from collections.abc import Iterator
from typing import Any

import pytest

from jackdaw import _core
from jackdaw.cli.smart_agent import smart_agent
from jackdaw.core import Game, ViewTracker
from jackdaw.engine import actions as A
from jackdaw.engine.actions import GamePhase

KEY = 0x5EED

BACKS = ("b_red", "b_blue", "b_yellow", "b_green", "b_black", "b_magic", "b_nebula", "b_ghost",
         "b_abandoned", "b_checkered", "b_zodiac", "b_painted", "b_anaglyph", "b_plasma",
         "b_erratic")  # fmt: skip


def normal(view: dict[str, Any]) -> Any:
    """A view with its card ids replaced by the order they appear in."""
    return json.loads(_core.Game._player_view_without_ids(json.dumps(view)))


def _forced(action: Any, game: Game) -> Any:
    """Cerulean Bell: a play or discard holds the forced card."""
    if not isinstance(action, (A.PlayHand, A.Discard)):
        return action
    forced = [i for i, c in enumerate(game.hand) if c.ability.get("forced_selection")]
    if not forced or forced[0] in action.card_indices:
        return action
    return type(action)(card_indices=tuple(sorted({*list(action.card_indices)[:4], forced[0]})))


def agent(game: Game, rng: random.Random) -> Any:
    """``smart_agent``, with a random move one time in five so the runs go everywhere."""
    legal = game.legal_actions()
    if rng.random() < 0.8:
        return _forced(smart_agent(game, legal), game)
    action = rng.choice(legal)
    if isinstance(action, (A.PlayHand, A.Discard)):
        n = len(game.hand)
        cards = tuple(sorted(rng.sample(range(n), rng.randint(1, min(5, n)))))
        action = _forced(type(action)(card_indices=cards), game)
    return action


def decisions(runs: int, seed: int, steps: int = 400) -> Iterator[tuple[Game, Game, str]]:
    """``(run, rebuilt, where)`` at every decision of ``runs`` runs, the rebuilt game kept by a
    :class:`ViewTracker` across each run."""
    rng = random.Random(seed)
    for n in range(runs):
        game = Game.new(rng.choice(BACKS), rng.randint(1, 8), f"REBUILD{seed}_{n}")
        tracker = ViewTracker()
        for step in range(steps):
            if game.phase == GamePhase.GAME_OVER:
                break
            rebuilt = tracker.update(game.player_view(KEY))
            yield game, rebuilt, f"run {n} step {step} ({game.phase.value})"
            action = agent(game, rng)
            game.step(action)
            tracker.step(action)


# ---------------------------------------------------------------------------
# What shows
# ---------------------------------------------------------------------------


def test_every_view_builds_into_a_game_that_shows_it() -> None:
    """The rebuilt game's own view is the run's, it offers the same actions, and its
    observation (the player's view too) is the run's, value for value."""
    n = 0
    for game, rebuilt, where in decisions(runs=25, seed=1):
        assert normal(rebuilt.player_view(KEY + 1)) == normal(game.player_view(KEY)), where
        assert rebuilt.legal_actions() == game.legal_actions(), where
        real, built = game.observation_buffer(), rebuilt.observation_buffer()
        assert built[1] == real[1] and bytes(built[0]) == bytes(real[0]), where
        n += 1
    assert n > 500


def test_a_view_alone_builds_a_game_that_shows_it() -> None:
    """Without a shadow too (a new run's hidden counters, with the view's vouchers and
    jokers), the game shows what the view shows."""
    n = 0
    for game, _rebuilt, where in decisions(runs=10, seed=2):
        view = game.player_view(KEY)
        alone = Game.from_player_view(view)
        assert normal(alone.player_view(KEY + 1)) == normal(view), where
        assert alone.legal_actions() == game.legal_actions(), where
        n += 1
    assert n > 200


# ---------------------------------------------------------------------------
# What doesn't: the run, field by field, but where the view hides it
# ---------------------------------------------------------------------------


def _identity(c: dict[str, Any]) -> tuple[Any, ...]:
    """What the deck view shows of a playing card."""
    edition = (c["edition"] or {}).get("type")
    front = None if c["center"] == "m_stone" else c["front"]
    return c["center"], front, edition, c["seal"], c["ability"]["perma_bonus"]


def _shown(c: dict[str, Any]) -> dict[str, Any]:
    """A card as the view shows it face up: no sort id, no Stone Card base, nor what no
    player can tell of a playing card (its play count, original rank and suit, whether it came
    in through ``add_to_deck``)."""
    c = dict(c)
    c.pop("sort_id")
    if c["base"] is not None:
        c.pop("added_to_deck")
        if c["center"] == "m_stone":
            c["front"] = c["base"] = None
        else:
            base = dict(c["base"])
            for k in ("times_played", "original_value", "suit_nominal_original"):
                base.pop(k)
            c["base"] = base
    return c


def _cards(game: Game, snap: dict[str, Any]) -> dict[str, Any]:
    """The cards as a player can know them: each area's face-up cards in order (a face-down
    joker by its size and sell button), and the playing cards they can't see as the deck view
    counts them."""
    areas = snap["areas"]
    flipped = [w for _s, _e, w, _j in game._g._card_runtime("discard")]
    hand = areas["hand"]
    remaining = areas["deck"] + [c for c in hand if c["facing"] == "back"]
    remaining += [c for c, w in zip(areas["discard"], flipped, strict=True) if w]
    seen = [c for c, w in zip(areas["discard"], flipped, strict=True) if not w]
    out: dict[str, Any] = {
        "remaining": sorted(map(_identity, remaining), key=repr),
        "discard": sorted(map(_identity, seen), key=repr),
        "hand": [_shown(c) if c["facing"] == "front" else None for c in hand],
    }
    for area in ("jokers", "consumables", "shop_jokers", "shop_vouchers", "shop_booster",
                 "pack_cards", "play"):  # fmt: skip
        out[area] = [
            _shown(c)
            if c["facing"] == "front"
            else (
                c["center"] in ("j_half", "j_photograph", "j_square", "j_wee") and c["center"],
                c["stickers"]["eternal"],
            )  # fmt: skip
            for c in areas[area]
        ]
    # The Pillar reads which cards were played this ante: in its ante that is the run's.
    if snap["round_resets"]["blind_choices"].get("Boss") == "bl_pillar":
        played = [c for c in remaining + seen if c["ability"].get("played_this_ante")]
        out["played_this_ante"] = sorted(map(_identity, played), key=repr)
    return out


#: Hidden from the view, so a stand-in in the rebuild: the RNG and seed, the next sort id, an
#: Arcana or Spectral pack's dealt cards (no rule reads them), the Orbital Tag hands no tag
#: shows, and the cards of the round no joker shows.
HIDDEN = ("rng", "run.seed", "sim.next_sort_id", "pack.dealt", "game.orbital_choices")
ROUND_CARDS = {
    "idol_card": "j_idol",
    "mail_card": "j_mail",
    "ancient_card": "j_ancient",
    "castle_card": "j_castle",
}


def _drop(snap: dict[str, Any], path: str) -> None:
    *parents, last = path.split(".")
    for p in parents:
        snap = snap.get(p) or {}
    snap.pop(last, None)


def _comparable(game: Game, view: dict[str, Any]) -> dict[str, Any]:
    snap = game.to_dict()
    cards = _cards(game, snap)
    snap.pop("areas")
    for path in HIDDEN:
        _drop(snap, path)
    shown = {c["key"] for k in ("jokers", "consumables") for c in view[k] if not c["face_down"]}
    shown |= {c["key"] for c in (view["shop"] or {}).get("jokers", []) if not c["face_down"]}
    shown |= {c["key"] for c in (view["pack"] or {}).get("cards", []) if not c["face_down"]}
    for card, joker in ROUND_CARDS.items():
        if joker not in shown:
            snap["current_round"].pop(card)
    # The ante's voucher shows in its shop.
    if view["shop"] is None:
        snap["current_round"].pop("voucher")
    # The observation reads the run's awarded tags as a set.
    snap["sim"]["tag_history"] = sorted(set(snap["sim"]["tag_history"]))
    snap["cards"] = cards
    return snap


def _diff(a: Any, b: Any, path: str = "$") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        return [
            d for k in sorted(set(a) | set(b)) for d in _diff(a.get(k), b.get(k), f"{path}.{k}")
        ]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in _diff(x, y, f"{path}[{i}]")]
    return [] if a == b else [path]


def test_the_rebuild_is_the_run_where_the_view_shows_it() -> None:
    """Field by field, the rebuilt game is the run except where the view hides something: the
    RNG, the draw pile's order, face-down cards, a Stone Card's rank, what can't tell two
    look-alike cards apart, the Orbital Tag hands and round cards no tag or joker shows, the
    ante's voucher before its shop. Its shadow carries the rest: the consumables used, the cards
    played this ante (read in The Pillar's ante), the sort, the counters its purchases moved."""
    n = 0
    differences: collections.Counter[str] = collections.Counter()
    first: dict[str, str] = {}
    for game, rebuilt, where in decisions(runs=40, seed=3):
        view = game.player_view(KEY)
        for d in _diff(_comparable(game, view), _comparable(rebuilt, view)):
            differences[d] += 1
            first.setdefault(d, where)
        n += 1
    assert n > 800
    assert not differences, {d: (c, first[d]) for d, c in differences.most_common(10)}


# ---------------------------------------------------------------------------
# The tracker
# ---------------------------------------------------------------------------


def test_sort_ids_follow_the_cards_a_player_follows() -> None:
    game = Game.new("b_red", 1, "FOLLOW")
    tracker = ViewTracker()
    tracker.update(game.player_view(KEY))
    game.step(A.SelectBlind())
    tracker.step(A.SelectBlind())
    rebuilt = tracker.update(game.player_view(KEY))
    view = game.player_view(KEY)
    ids = rebuilt.view_ids
    assert [ids[c["id"]] for c in view["hand"]] == [c.sort_id for c in rebuilt.hand]
    # A sort moves the cards; each keeps its sort id.
    game.step(A.SortHand(mode="suit"))
    tracker.step(A.SortHand(mode="suit"))
    after = tracker.update(game.player_view(KEY))
    assert {c["id"]: after.view_ids[c["id"]] for c in game.player_view(KEY)["hand"]} == {
        c["id"]: ids[c["id"]] for c in view["hand"]
    }
    # Discarding: the kept cards keep their sort ids.
    kept = [c["id"] for c in game.player_view(KEY)["hand"][2:]]
    game.step(A.Discard(card_indices=(0, 1)))
    tracker.step(A.Discard(card_indices=(0, 1)))
    later = tracker.update(game.player_view(KEY))
    assert all(later.view_ids[i] == ids[i] for i in kept)


def test_the_tracker_takes_factored_actions() -> None:
    """A factored action (as ``step_factored`` takes it) steps the shadow as its engine action."""
    from jackdaw.env.action_space import ActionType
    from jackdaw.env.game_spec import FactoredAction

    game = Game.new("b_red", 1, "FACTORED")
    tracker = ViewTracker()
    tracker.update(game.player_view(KEY))
    game.step(A.SelectBlind())
    tracker.step(FactoredAction(action_type=ActionType.SelectBlind))
    rebuilt = tracker.update(game.player_view(KEY))
    assert rebuilt.phase == GamePhase.SELECTING_HAND
    game.step(A.PlayHand(card_indices=(0, 1)))
    tracker.step((int(ActionType.PlayHand), None, (0, 1)))
    rebuilt = tracker.update(game.player_view(KEY))
    assert rebuilt.to_dict()["hands"] == game.to_dict()["hands"]


def test_the_tracker_starts_over_with_a_new_run() -> None:
    tracker = ViewTracker()
    game = Game.new("b_red", 1, "FIRST")
    tracker.update(game.player_view(KEY))
    game.step(A.SelectBlind())
    tracker.step(A.SelectBlind())
    tracker.update(game.player_view(KEY))
    # A new run, of another deck, without reset().
    other = Game.new("b_blue", 2, "SECOND")
    rebuilt = tracker.update(other.player_view(KEY))
    assert normal(rebuilt.player_view(1)) == normal(other.player_view(KEY))
    assert rebuilt.to_dict()["run"]["back"] == "b_blue"
    with pytest.raises(RuntimeError):
        ViewTracker().step(A.SelectBlind())


def test_an_action_the_shadow_refuses_is_dropped() -> None:
    game = Game.new("b_red", 1, "REFUSED")
    tracker = ViewTracker()
    tracker.update(game.player_view(KEY))
    tracker.step(A.PlayHand(card_indices=(0,)))  # no hand at blind select
    assert normal(tracker.update(game.player_view(KEY)).player_view(1)) == normal(
        game.player_view(KEY)
    )


def test_the_pillar_reads_what_the_shadow_kept() -> None:
    """The cards played in the Small Blind are back in the draw pile at The Pillar, where the
    view doesn't show them; the shadow knows them, so the rebuilt game has the same cards played
    this ante, and The Pillar debuffs the same cards when they're drawn."""
    snap = Game.new("b_red", 1, "PILLAR").to_dict()
    snap["round_resets"]["blind_choices"]["Boss"] = "bl_pillar"
    game = Game.from_dict(snap)
    tracker = ViewTracker()

    def step(action: Any) -> Game:
        rebuilt = tracker.update(game.player_view(KEY))
        game.step(action)
        tracker.step(action)
        return rebuilt

    step(A.SelectBlind())
    game.debug_set(chips=game.blind.chips)
    step(A.PlayHand(card_indices=(0, 1, 2, 3, 4)))
    step(A.CashOut())
    step(A.NextRound())
    step(A.SkipBlind())
    step(A.SelectBlind())
    rebuilt = tracker.update(game.player_view(KEY))
    assert rebuilt.blind is not None and rebuilt.blind.key == "bl_pillar"

    def played(g: Game) -> list[tuple[Any, ...]]:
        snap = g.to_dict()
        return sorted(
            (_identity(c), c["debuff"])
            for area in ("deck", "discard", "hand")
            for c in snap["areas"][area]
            if c["ability"].get("played_this_ante")
        )

    assert len(played(game)) == 5
    assert played(rebuilt) == played(game)
    # Most of them are in the draw pile, where only the shadow knew them.
    assert sum(c["ability"].get("played_this_ante", False)
               for c in game.to_dict()["areas"]["deck"]) >= 2  # fmt: skip


def test_a_card_the_shadow_destroyed_comes_back() -> None:
    """A Glass Card shatters at random (1 in 4): where the shadow's rolls shattered it and the
    run's didn't, the next game has it back, under its sort id."""
    for n in range(80):
        game = Game.new("b_red", 1, f"GLASS{n}")
        game.step(A.SelectBlind())
        sid = game.debug_add("H_A", enhancement="m_glass")
        game.debug_set_hand([sid] + [c.sort_id for c in game.hand[:4]])
        tracker = ViewTracker(seed=f"OTHER{n}")
        rebuilt = tracker.update(game.player_view(KEY))
        glass = rebuilt.hand[0].sort_id
        play = A.PlayHand(card_indices=(0,))
        shadow = rebuilt.clone()
        shadow.step(play)
        game.step(play)
        survived = any(c.key == "m_glass" for c in game.discard_pile)
        shattered = all(c.sort_id != glass for c in shadow.discard_pile)
        if not (survived and shattered):
            continue
        tracker.step(play)
        again = tracker.update(game.player_view(KEY))
        assert any(c.sort_id == glass and c.key == "m_glass" for c in again.discard_pile)
        return
    pytest.fail("no Glass Card shattered for the shadow only in 80 runs")


def test_seeds_give_other_guesses_at_whats_hidden() -> None:
    game = Game.new("b_red", 1, "GUESSES")
    game.step(A.SelectBlind())
    view = game.player_view(KEY)
    a, b = (Game.from_player_view(view, seed=s) for s in ("AAAA", "BBBB"))
    assert normal(a.player_view(1)) == normal(b.player_view(2)) == normal(view)
    assert [c.front for c in a.deck] != [c.front for c in b.deck]
    assert sorted(c.front for c in a.deck) == sorted(c.front for c in game.deck)
    assert a.seed == "AAAA"


def test_the_cash_out_shows() -> None:
    """The round evaluation screen shows what Cash Out pays; so does the view, and the rebuilt
    game pays it."""
    game = Game.new("b_red", 1, "CASHOUT")
    game.step(A.SelectBlind())
    game.debug_set(chips=game.blind.chips)
    game.step(A.PlayHand(card_indices=(0,)))
    view = game.player_view(KEY)
    assert view["current_round"]["cash_out"] == game.to_dict()["sim"]["pending_cashout"] > 0
    rebuilt = Game.from_player_view(view)
    game.step(A.CashOut())
    rebuilt.step(A.CashOut())
    assert rebuilt.dollars == game.dollars


def test_a_view_that_doesnt_add_up_is_refused() -> None:
    game = Game.new("b_red", 1, "BAD")
    game.step(A.SelectBlind())
    view = game.player_view(KEY)
    with pytest.raises(ValueError, match="remaining cards"):
        Game.from_player_view({**view, "deck_size": 3})
    with pytest.raises(ValueError, match="format"):
        Game.from_player_view({**view, "format": "something/1"})


# ---------------------------------------------------------------------------
# The environment
# ---------------------------------------------------------------------------


def test_the_env_keeps_the_view_game() -> None:
    """``BalatroGymnasiumEnv.view_game()`` is the game the tracker builds from the env's views
    and steps, through index and factored steps and resets."""
    import numpy as np

    from jackdaw.env import BalatroGymnasiumEnv, DirectAdapter

    env = BalatroGymnasiumEnv(adapter_factory=DirectAdapter)
    env.reset(seed=11)
    rng = np.random.default_rng(0)
    for t in range(120):
        if t % 4 != 3:  # skip some decisions: the env keeps the tracker in step anyway
            game = env.view_game()
            assert normal(game.player_view(1)) == normal(env.player_view()), t
            assert game.legal_actions() == env.game.legal_actions(), t
        a = int(rng.choice(np.nonzero(env.action_masks())[0]))
        if t % 2:
            _, _, terminated, truncated, _ = env.step(a)
        else:
            _, _, terminated, truncated, _ = env.step_factored(env._action_table[a])
        if terminated or truncated:
            env.reset()

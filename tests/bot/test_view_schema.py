"""The gamestate view fits BalatroBot's published schema (openrpc.json ``GameState``,
patched where the schema is wrong: see :mod:`tests.bot.schema`)."""

from __future__ import annotations

import json
import random
from collections import Counter

import pytest

from jackdaw import _core
from tests.bot.driver import BotCallError, call, random_run
from tests.bot.schema import SchemaError, schemas, validate, validate_gamestate

_DECKS = ["RED", "BLUE", "YELLOW", "GREEN", "BLACK", "MAGIC", "NEBULA", "GHOST"]
_DECKS += ["ABANDONED", "CHECKERED", "ZODIAC", "PAINTED", "ANAGLYPH", "PLASMA", "ERRATIC"]
_STAKES = ["WHITE", "RED", "GREEN", "BLACK", "BLUE", "PURPLE", "ORANGE", "GOLD"]


def test_views_of_random_runs_fit_the_schema() -> None:
    states: Counter[str] = Counter()
    methods: Counter[str] = Counter()

    def check(method: str, view: dict) -> None:
        states[view["state"]] += 1
        methods[method] += 1
        try:
            validate_gamestate(view)
        except SchemaError as exc:
            raise AssertionError(f"after {method}: {exc}") from exc

    for index in range(12):
        random_run(
            f"SCHEMA{index}",
            random.Random(index),
            deck=_DECKS[index % len(_DECKS)],
            stake=_STAKES[index % len(_STAKES)],
            max_steps=150,
            on_view=check,
        )
    expected = ["BLIND_SELECT", "SELECTING_HAND", "ROUND_EVAL", "SHOP", "SMODS_BOOSTER_OPENED"]
    assert all(states[s] for s in expected + ["GAME_OVER"]), states
    assert all(methods[m] for m in ["play", "discard", "buy", "sell", "use", "pack"]), methods


def test_menu_and_unknown_views_fit_the_schema(tmp_path) -> None:
    s = _core.BotSession()
    validate_gamestate(call(s, "gamestate"))
    call(s, "start", {"deck": "RED", "stake": "WHITE", "seed": "SCHEMA_MENU"})
    bad = tmp_path / "bad.jkr"
    bad.write_text("not a snapshot", encoding="utf-8")
    with pytest.raises(BotCallError):
        call(s, "load", {"path": str(bad)})
    view = call(s, "gamestate")
    assert view["state"] == "UNKNOWN"
    validate_gamestate(view)


def test_modifiers_and_stickers_fit_the_schema() -> None:
    s = _core.BotSession()
    call(s, "start", {"deck": "RED", "stake": "GOLD", "seed": "SCHEMA_MODS"})
    call(s, "select")
    call(s, "add", {"key": "j_joker", "edition": "NEGATIVE", "eternal": True})
    call(s, "add", {"key": "j_egg", "perishable": 3, "rental": True})
    call(s, "add", {"key": "H_K", "seal": "PURPLE", "enhancement": "LUCKY", "edition": "HOLO"})
    call(s, "add", {"key": "c_fool", "edition": "NEGATIVE"})
    view = call(s, "gamestate")
    validate_gamestate(view)
    joker = view["jokers"]["cards"][0]
    assert joker["modifier"]["edition"] == "NEGATIVE" and joker["modifier"]["eternal"] is True
    card = view["hand"]["cards"][-1]
    assert card["modifier"] == {"seal": "PURPLE", "edition": "HOLO", "enhancement": "LUCKY"}


def test_path_result_fits_the_schema(tmp_path) -> None:
    s = _core.BotSession()
    call(s, "start", {"deck": "RED", "stake": "WHITE", "seed": "SCHEMA_SAVE"})
    out = call(s, "save", {"path": str(tmp_path / "run.jkr")})
    validate(out, schemas()["PathResult"])
    validate(call(s, "load", {"path": str(tmp_path / "run.jkr")}), schemas()["PathResult"])


def test_rpc_discover_serves_the_vendored_document() -> None:
    from tests.bot.schema import openrpc

    s = _core.BotSession()
    ok, text = s.call("rpc.discover", "{}")
    assert ok
    assert json.loads(text) == openrpc()


def test_the_deck_limit_follows_cryptid() -> None:
    """Cryptid raises ``G.deck.config.card_limit`` by one per copy (card.lua:1210), and the
    frame's clamp to ``#G.playing_cards`` keeps it there."""
    s = _core.BotSession()
    call(s, "start", {"deck": "RED", "stake": "WHITE", "seed": "SCHEMA_CRYPTID"})
    view = call(s, "select")
    assert (view["cards"]["count"], view["cards"]["limit"]) == (44, 52)
    call(s, "add", {"key": "c_cryptid"})
    view = call(s, "use", {"consumable": 0, "cards": [0]})
    assert (view["cards"]["count"], view["cards"]["limit"]) == (44, 54)
    assert view["hand"]["count"] == 10

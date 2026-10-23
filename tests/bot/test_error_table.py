"""Error-table coverage gate: every error message in BalatroBot's endpoints, triggered.

:mod:`tests.bot.lua_messages` lists every ``message =`` of the vendored
endpoint Lua (one site per assignment for a message held in a variable).  Each
row below drives a fresh session (``jackdaw._core.BotSession``) into a state,
sends one request and checks the error's name (and so its code) and its exact
message; the row names the site it covers, and the message must fit that
site's pattern.  The gate fails when a site has no row and no exemption, so a
new upstream message (or a renamed one) can't go unnoticed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from jackdaw import _core
from tests.bot.lua_messages import sites

CODES = {"INTERNAL_ERROR": -32000, "BAD_REQUEST": -32001, "INVALID_STATE": -32002}
CODES["NOT_ALLOWED"] = -32003


# ---------------------------------------------------------------------------
# States
# ---------------------------------------------------------------------------


def _call(s: Any, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    ok, text = s.call(method, json.dumps(params or {}))
    data = json.loads(text)
    assert ok, f"setup {method} {params}: {data}"
    return data


def menu() -> Any:
    return _core.BotSession()


def blind(seed: str = "TEST123") -> Any:
    s = menu()
    _call(s, "start", {"deck": "RED", "stake": "WHITE", "seed": seed})
    return s


def hand(seed: str = "TEST123") -> Any:
    s = blind(seed)
    _call(s, "select")
    return s


def shop(seed: str = "TEST123") -> Any:
    s = hand(seed)
    _call(s, "set", {"chips": 1000})
    _call(s, "play", {"cards": [0]})
    _call(s, "cash_out")
    return s


def shop_with_free_pack_slot(seed: str = "TEST123") -> Any:
    """In the shop, one pack opened and skipped: one booster slot free."""
    s = shop(seed)
    _call(s, "set", {"money": 100})
    _call(s, "buy", {"pack": 0})
    _call(s, "pack", {"skip": True})
    return s


def added_pack(seed: str, key: str) -> Callable[[], Any]:
    """In a pack of kind ``key`` that ``add`` put in the shop (contents fixed by ``seed``)."""

    def state() -> Any:
        s = shop_with_free_pack_slot(seed)
        _call(s, "add", {"key": key})
        _call(s, "buy", {"pack": 1})
        return s

    return state


def buffoon(jokers: int = 0) -> Any:
    """TEST123's first shop pack, a Buffoon Pack (Red Card, Joker), opened while holding
    ``jokers`` Jokers."""
    s = shop()
    _call(s, "set", {"money": 100})
    for _ in range(jokers):
        _call(s, "add", {"key": "j_joker"})
    _call(s, "buy", {"pack": 0})
    return s


# ---------------------------------------------------------------------------
# Rows
# ---------------------------------------------------------------------------


@dataclass
class Row:
    site: str  # Site.id this row covers
    state: Callable[[], Any]
    method: str
    params: dict[str, Any] | Callable[[Path], dict[str, Any]]
    name: str
    message: str | Callable[[Path], str]
    prep: list[tuple[str, dict[str, Any]]] = field(default_factory=list)


def _jokers(n: int) -> list[tuple[str, dict[str, Any]]]:
    return [("add", {"key": "j_joker"})] * n


def _missing(tmp: Path) -> Path:
    return tmp / "missing.jkr"


def _garbage(tmp: Path) -> Path:
    path = tmp / "garbage.jkr"
    path.write_text("CORRUPTED_SAVE_FILE_FOR_TESTING", encoding="utf-8")
    return path


def _nowhere(tmp: Path) -> Path:
    return tmp / "no" / "such" / "dir" / "run.jkr"


ARCANA_DEATH = added_pack("BOTERR1", "p_arcana_normal_1")  # Death, The World, ...
ARCANA_MAGICIAN = added_pack("BOTERR9", "p_arcana_normal_1")  # The Magician, ...
SPECTRAL_ANKH = added_pack("BOTERR1", "p_spectral_normal_1")  # Ankh, Trance

BAD, STATE, NOT, INTERNAL = "BAD_REQUEST", "INVALID_STATE", "NOT_ALLOWED", "INTERNAL_ERROR"

ROWS: list[Row] = [
    # add.lua
    Row(
        "add.lua:176",
        hand,
        "add",
        {"key": "x_bad"},
        BAD,
        "Invalid card key format. Expected: joker (j_*), consumable (c_*), voucher (v_*), "
        "or playing card (SUIT_RANK)",
    ),
    Row(
        "add.lua:185",
        shop,
        "add",
        {"key": "H_A"},
        STATE,
        "Playing cards can only be added in SELECTING_HAND state",
    ),
    Row(
        "add.lua:194",
        hand,
        "add",
        {"key": "v_overstock_norm"},
        STATE,
        "Vouchers can only be added in SHOP state",
    ),
    Row(
        "add.lua:203",
        hand,
        "add",
        {"key": "p_arcana_normal_1"},
        STATE,
        "Packs can only be added in SHOP state",
    ),
    Row(
        "add.lua:224",
        shop,
        "add",
        {"key": "p_arcana_normal_1"},
        NOT,
        "Cannot add pack, shop booster slots are full",
    ),
    Row(
        "add.lua:235",
        shop_with_free_pack_slot,
        "add",
        {"key": "p_nope"},
        BAD,
        "Pack key not found: p_nope",
    ),
    Row(
        "add.lua:245",
        hand,
        "add",
        {"key": "j_joker", "seal": "RED"},
        BAD,
        "Seal can only be applied to playing cards",
    ),
    Row(
        "add.lua:257",
        hand,
        "add",
        {"key": "H_A", "seal": "SHINY"},
        BAD,
        "Invalid seal value. Expected: RED, BLUE, GOLD, or PURPLE",
    ),
    Row(
        "add.lua:267",
        shop,
        "add",
        {"key": "v_overstock_norm", "edition": "FOIL"},
        BAD,
        "Edition cannot be applied to vouchers",
    ),
    Row(
        "add.lua:276",
        hand,
        "add",
        {"key": "c_fool", "edition": "FOIL"},
        BAD,
        "Consumables can only have NEGATIVE edition",
    ),
    Row(
        "add.lua:288",
        hand,
        "add",
        {"key": "j_joker", "edition": "SHINY"},
        BAD,
        "Invalid edition value. Expected: HOLO, FOIL, POLYCHROME, or NEGATIVE",
    ),
    Row(
        "add.lua:298",
        hand,
        "add",
        {"key": "j_joker", "enhancement": "GLASS"},
        BAD,
        "Enhancement can only be applied to playing cards",
    ),
    Row(
        "add.lua:310",
        hand,
        "add",
        {"key": "H_A", "enhancement": "SHINY"},
        BAD,
        "Invalid enhancement value. Expected: BONUS, MULT, WILD, GLASS, STEEL, STONE, GOLD, "
        "or LUCKY",
    ),
    Row(
        "add.lua:320",
        hand,
        "add",
        {"key": "c_fool", "eternal": True},
        BAD,
        "Eternal can only be applied to jokers",
    ),
    Row(
        "add.lua:329",
        hand,
        "add",
        {"key": "c_fool", "perishable": 2},
        BAD,
        "Perishable can only be applied to jokers",
    ),
    Row(
        "add.lua:339",
        hand,
        "add",
        {"key": "j_joker", "perishable": 0},
        BAD,
        "Perishable must be a positive integer (>= 1)",
    ),
    Row(
        "add.lua:349",
        hand,
        "add",
        {"key": "c_fool", "rental": True},
        BAD,
        "Rental can only be applied to jokers",
    ),
    Row("add.lua:439", hand, "add", {"key": "j_nope"}, BAD, "Failed to add card: j_nope"),
    # buy.lua
    Row(
        "buy.lua:70",
        shop,
        "buy",
        {},
        BAD,
        "Invalid arguments. You must provide one of: card, voucher, pack",
    ),
    Row(
        "buy.lua:79",
        shop,
        "buy",
        {"card": 0, "pack": 0},
        BAD,
        "Invalid arguments. Cannot provide more than one of: card, voucher, or pack",
    ),
    Row(
        "buy.lua:96#1",
        shop,
        "buy",
        {"card": 0},
        BAD,
        "No jokers/consumables/cards in the shop. Reroll to restock the shop",
        prep=[("set", {"money": 100}), ("buy", {"card": 0}), ("buy", {"card": 0})],
    ),
    Row(
        "buy.lua:96#2",
        shop,
        "buy",
        {"voucher": 0},
        BAD,
        "No vouchers to redeem. Defeat boss blind to restock",
        prep=[("set", {"money": 100}), ("buy", {"voucher": 0})],
    ),
    Row(
        "buy.lua:96#3",
        shop,
        "buy",
        {"pack": 0},
        BAD,
        "No packs to open",
        prep=[("set", {"money": 100})] + [("buy", {"pack": 0}), ("pack", {"skip": True})] * 2,
    ),
    Row(
        "buy.lua:113#1",
        shop,
        "buy",
        {"card": 9},
        BAD,
        "Card index out of range. Index: 9, Available cards: 2",
    ),
    Row(
        "buy.lua:113#2",
        shop,
        "buy",
        {"voucher": 3},
        BAD,
        "Voucher index out of range. Index: 3, Available: 1",
    ),
    Row(
        "buy.lua:113#3",
        shop,
        "buy",
        {"pack": 5},
        BAD,
        "Pack index out of range. Index: 5, Available: 2",
    ),
    Row(
        "buy.lua:126",
        shop,
        "buy",
        {"card": 0},
        BAD,
        "Card is not affordable. Cost: 5, Available money: 0",
        prep=[("set", {"money": 0})],
    ),
    Row(
        "buy.lua:136",
        shop,
        "buy",
        {"card": 0},
        BAD,
        "Cannot purchase joker card, joker slots are full. Current: 5, Limit: 5",
        prep=[("set", {"money": 100})] + _jokers(5),
    ),
    Row(
        "buy.lua:150",
        shop,
        "buy",
        {"card": 1},
        BAD,
        "Cannot purchase consumable card, consumable slots are full. Current: 2, Limit: 2",
        prep=[("set", {"money": 100}), ("add", {"key": "c_fool"}), ("add", {"key": "c_fool"})],
    ),
    # discard.lua
    Row(
        "discard.lua:41",
        hand,
        "discard",
        {"cards": []},
        BAD,
        "Must provide at least one card to discard",
    ),
    Row(
        "discard.lua:49",
        hand,
        "discard",
        {"cards": [0]},
        BAD,
        "No discards left",
        prep=[("set", {"discards": 0})],
    ),
    Row(
        "discard.lua:57",
        hand,
        "discard",
        {"cards": [0, 1, 2, 3, 4, 5]},
        BAD,
        "You can only discard 5 cards",
    ),
    Row("discard.lua:66", hand, "discard", {"cards": [99]}, BAD, "Invalid card index: 99"),
    # load.lua
    Row(
        "load.lua:52",
        shop,
        "load",
        lambda tmp: {"path": str(_missing(tmp))},
        INTERNAL,
        lambda tmp: f"Failed to read save file: '{_missing(tmp)}'",
    ),
    Row(
        "load.lua:78",
        shop,
        "load",
        lambda tmp: {"path": str(_garbage(tmp))},
        INTERNAL,
        "Invalid save file format",
    ),
    # pack.lua
    Row(
        "pack.lua:99",
        buffoon,
        "pack",
        {},
        BAD,
        "Invalid arguments. You must provide one of: card, skip",
    ),
    Row(
        "pack.lua:107",
        buffoon,
        "pack",
        {"card": 0, "skip": True},
        BAD,
        "Invalid arguments. Cannot provide both card and skip",
    ),
    Row(
        "pack.lua:130",
        buffoon,
        "pack",
        {"card": 9},
        BAD,
        "Card index out of range. Index: 9, Available cards: 2",
    ),
    Row(
        "pack.lua:144",
        lambda: buffoon(jokers=5),
        "pack",
        {"card": 0},
        NOT,
        "Cannot select joker, joker slots are full. Current: 5, Limit: 5",
    ),
    Row(
        "pack.lua:163",
        SPECTRAL_ANKH,
        "pack",
        {"card": 0},
        NOT,
        "Card 'c_ankh' requires at least 1 joker. Current: 0",
    ),
    Row(
        "pack.lua:191#1",
        ARCANA_DEATH,
        "pack",
        {"card": 0},
        BAD,
        "Card 'c_death' requires exactly 2 target card(s). Provided: 0",
    ),
    Row(
        "pack.lua:191#2",
        ARCANA_MAGICIAN,
        "pack",
        {"card": 0},
        BAD,
        "Card 'c_magician' requires 1-2 target card(s). Provided: 0",
    ),
    Row(
        "pack.lua:368",
        ARCANA_MAGICIAN,
        "pack",
        {"card": 0, "targets": [99]},
        BAD,
        "Target card index out of range. Index: 99, Hand size: 8",
    ),
    # play.lua
    Row("play.lua:41", hand, "play", {"cards": []}, BAD, "Must provide at least one card to play"),
    Row(
        "play.lua:49",
        hand,
        "play",
        {"cards": [0, 1, 2, 3, 4, 5]},
        BAD,
        "You can only play 5 cards",
    ),
    Row("play.lua:58", hand, "play", {"cards": [99]}, BAD, "Invalid card index: 99"),
    # rearrange.lua
    Row(
        "rearrange.lua:54",
        hand,
        "rearrange",
        {},
        BAD,
        "Must provide exactly one of: hand, jokers, or consumables",
    ),
    Row(
        "rearrange.lua:60",
        hand,
        "rearrange",
        {"hand": [0], "jokers": [0]},
        BAD,
        "Can only rearrange one type at a time",
    ),
    Row(
        "rearrange.lua:73",
        shop,
        "rearrange",
        {"hand": [0]},
        STATE,
        "Can only rearrange hand during hand selection",
    ),
    Row(
        "rearrange.lua:91",
        buffoon,
        "rearrange",
        {"hand": []},
        NOT,
        "No cards to rearrange. You can only rearrange hand in Arcana and Spectral packs.",
    ),
    Row(
        "rearrange.lua:144",
        hand,
        "rearrange",
        {"hand": [0]},
        BAD,
        "Must provide exactly 8 indices for hand",
    ),
    Row(
        "rearrange.lua:156",
        hand,
        "rearrange",
        {"hand": [99, 1, 2, 3, 4, 5, 6, 7]},
        BAD,
        "Index out of range for hand: 99",
    ),
    Row(
        "rearrange.lua:165",
        hand,
        "rearrange",
        {"hand": [0, 1, 1, 3, 4, 5, 6, 7]},
        BAD,
        "Duplicate index in hand: 1",
    ),
    # reroll.lua
    Row(
        "reroll.lua:33",
        shop,
        "reroll",
        {},
        NOT,
        "Not enough dollars to reroll. Available: 0, Required: 5",
        prep=[("set", {"money": 0})],
    ),
    # save.lua
    Row(
        "save.lua:91",
        shop,
        "save",
        lambda tmp: {"path": str(_nowhere(tmp))},
        INTERNAL,
        lambda tmp: f"Failed to write save file to '{_nowhere(tmp)}'",
    ),
    # screenshot.lua
    Row(
        "screenshot.lua:49",
        hand,
        "screenshot",
        lambda tmp: {"path": str(tmp / "shot.png")},
        INTERNAL,
        "Failed to encode screenshot",
    ),
    # sell.lua
    Row(
        "sell.lua:46",
        hand,
        "sell",
        {},
        BAD,
        "Must provide exactly one of: joker or consumable",
    ),
    Row(
        "sell.lua:52",
        hand,
        "sell",
        {"joker": 0, "consumable": 0},
        BAD,
        "Can only sell one item at a time",
    ),
    Row("sell.lua:65", hand, "sell", {"joker": 0}, NOT, "No jokers available to sell"),
    Row(
        "sell.lua:77",
        hand,
        "sell",
        {"consumable": 0},
        NOT,
        "No consumables available to sell",
    ),
    Row(
        "sell.lua:90",
        hand,
        "sell",
        {"joker": 3},
        BAD,
        "Index out of range for joker: 3",
        prep=_jokers(1),
    ),
    # set.lua
    Row("set.lua:75", menu, "set", {"money": 1}, STATE, "Can only set during an active run"),
    Row("set.lua:92", hand, "set", {}, BAD, "Must provide at least one field to set"),
    Row("set.lua:102", hand, "set", {"money": -1}, BAD, "Money must be a positive integer"),
    Row("set.lua:114", hand, "set", {"chips": -1}, BAD, "Chips must be a positive integer"),
    Row("set.lua:126", hand, "set", {"ante": -1}, BAD, "Ante must be a positive integer"),
    Row("set.lua:138", hand, "set", {"round": -1}, BAD, "Round must be a positive integer"),
    Row("set.lua:150", hand, "set", {"hands": -1}, BAD, "Hands must be a positive integer"),
    Row(
        "set.lua:162",
        hand,
        "set",
        {"discards": -1},
        BAD,
        "Discards must be a positive integer",
    ),
    Row(
        "set.lua:173",
        hand,
        "set",
        {"shop": True},
        NOT,
        "Can re-stock shop only in SHOP state",
    ),
    # skip.lua
    Row(
        "skip.lua:39",
        blind,
        "skip",
        {},
        NOT,
        "Cannot skip Boss blind",
        prep=[("skip", {}), ("skip", {})],
    ),
    # start.lua
    Row(
        "start.lua:86",
        menu,
        "start",
        {"deck": "RED", "stake": "PINK", "seed": "X"},
        BAD,
        "Invalid stake enum. Must be one of: WHITE, RED, GREEN, BLACK, BLUE, PURPLE, ORANGE, "
        "GOLD. Got: PINK",
    ),
    Row(
        "start.lua:98",
        menu,
        "start",
        {"deck": "PINK", "stake": "WHITE", "seed": "X"},
        BAD,
        "Invalid deck enum. Must be one of: RED, BLUE, YELLOW, GREEN, BLACK, MAGIC, NEBULA, "
        "GHOST, ABANDONED, CHECKERED, ZODIAC, PAINTED, ANAGLYPH, PLASMA, ERRATIC. Got: PINK",
    ),
    # use.lua
    Row("use.lua:49", hand, "use", {"consumable": 0}, BAD, "Consumable index out of range: 0"),
    Row(
        "use.lua:63",
        shop,
        "use",
        {"consumable": 0},
        STATE,
        "Consumable 'The Magician' requires card selection and can only be used in "
        "SELECTING_HAND state",
        prep=[("add", {"key": "c_magician"})],
    ),
    Row(
        "use.lua:75",
        hand,
        "use",
        {"consumable": 0},
        BAD,
        "Consumable 'The Magician' requires card selection",
        prep=[("add", {"key": "c_magician"})],
    ),
    Row(
        "use.lua:85",
        hand,
        "use",
        {"consumable": 0, "cards": [99]},
        BAD,
        "Card index out of range: 99",
        prep=[("add", {"key": "c_magician"})],
    ),
    Row(
        "use.lua:102",
        hand,
        "use",
        {"consumable": 0, "cards": [0]},
        BAD,
        "Consumable 'Death' requires exactly 2 cards (provided: 1)",
        prep=[("add", {"key": "c_death"})],
    ),
    Row(
        "use.lua:131",
        hand,
        "use",
        {"consumable": 0, "cards": [0, 1, 2]},
        BAD,
        "Consumable 'The Magician' requires at most 2 cards (provided: 3)",
        prep=[("add", {"key": "c_magician"})],
    ),
    Row(
        "use.lua:170",
        hand,
        "use",
        {"consumable": 0},
        NOT,
        "Consumable 'Ankh' cannot be used at this time",
        prep=[("add", {"key": "c_ankh"})],
    ),
    Row(
        "use.lua:179",
        hand,
        "use",
        {"consumable": 0},
        NOT,
        "Cannot use consumable 'Ankh': insufficient space",
        prep=_jokers(5) + [("add", {"key": "c_ankh"})],
    ),
]

# Messages no request can produce, in the sim or (as far as the Lua shows) the live game.
EXEMPT: dict[str, str] = {
    "add.lua:213": "add requires SHOP for packs, and the shop's booster area always exists there",
    "buy.lua:188": (
        "the line before reads children.buy_button.definition: a missing button raises there "
        "(INTERNAL_ERROR, the voucher-from-add row), and a button always has a definition"
    ),
    "load.lua:66": (
        "the sim reads the snapshot file in place; there is no copy into the profile "
        "directory to fail"
    ),
    "pack.lua:116": (
        "G.pack_cards exists for as long as the state is SMODS_BOOSTER_OPENED in the sim; "
        "live it can be missing only between frames"
    ),
    "pack.lua:209": (
        "every card with a target requirement comes in an Arcana or Spectral pack, whose "
        "targets pack.lua:368 checks first"
    ),
    "rearrange.lua:82": "G.hand exists in every state rearrange accepts",
    "rearrange.lua:105": (
        "G.jokers exists in every state rearrange accepts (an empty one takes the empty "
        "permutation)"
    ),
    "rearrange.lua:119": "G.consumeables exists in every state rearrange accepts",
    "save.lua:62": (
        "every state save accepts is inside a run (outside one, the dispatcher's state check "
        "answers first)"
    ),
    "save.lua:82": "the sim always serializes the run (no compressed temp file to read back)",
    "screenshot.lua:62": "the sim has no renderer: screenshot fails at encoding (line 49)",
    "start.lua:126": "every deck enum maps to a back in the game data",
    "use.lua:117": (
        "no vanilla consumable has 1 < min_highlighted < max_highlighted (Death's 2 of 2 is "
        "the 'exactly' message)"
    ),
}

# Rows for messages that come from the Lua runtime, not from a ``message =``.
EXTRA_ROWS: list[Row] = [
    Row(
        "",
        shop_with_free_pack_slot,
        "buy",
        {"voucher": 1},
        INTERNAL,
        '[SMODS balatrobot "src/lua/endpoints/buy.lua"]:182: attempt to index field '
        "'buy_button' (a nil value)",
        prep=[("add", {"key": "v_overstock_norm"})],
    ),
]


def _run(row: Row, tmp: Path) -> tuple[bool, dict[str, Any]]:
    s = row.state()
    for method, params in row.prep:
        _call(s, method, params)
    params = row.params(tmp) if callable(row.params) else row.params
    ok, text = s.call(row.method, json.dumps(params))
    return ok, json.loads(text)


@pytest.mark.parametrize("row", ROWS + EXTRA_ROWS, ids=lambda r: r.site or r.message[:40])
def test_error_row(row: Row, tmp_path: Path) -> None:
    ok, error = _run(row, tmp_path)
    assert not ok, f"{row.method} succeeded"
    message = row.message(tmp_path) if callable(row.message) else row.message
    assert (error["name"], error["message"]) == (row.name, message)
    assert error["code"] == CODES[row.name]
    if row.site:
        site = {s.id: s for s in sites()}[row.site]
        assert site.matches(message), f"{message!r} doesn't fit {site.pattern.pattern}"
        assert site.name == row.name, f"{row.site} sends {site.name}"


def test_every_message_has_a_row_or_an_exemption() -> None:
    ids = {s.id for s in sites()}
    covered = {r.site for r in ROWS}
    assert not covered - ids, f"rows name unknown sites: {sorted(covered - ids)}"
    assert not set(EXEMPT) - ids, f"exemptions name unknown sites: {sorted(set(EXEMPT) - ids)}"
    assert not covered & set(EXEMPT), f"covered and exempt: {sorted(covered & set(EXEMPT))}"
    missing = sorted(ids - covered - set(EXEMPT))
    assert not missing, f"error messages with no row: {missing}"

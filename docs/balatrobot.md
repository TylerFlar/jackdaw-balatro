---
layout: default
title: BalatroBot on the simulator
---

# BalatroBot on the simulator

Jackdaw speaks [BalatroBot](https://github.com/coder/balatrobot) v1.5.2's JSON-RPC API on top of
its own engine. The command and view layer is a port of BalatroBot's Lua
(`crates/jackdaw-bot`, pinned to upstream commit `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`;
the vendored source is in `third_party/balatrobot/v1.5.2/`), so a client written for the live
game (an agent, a script, BalatroBot's own test suite) drives the simulator unchanged, and an
agent trained on the simulator can be pointed at the live game.

What "the same API" covers:

- **Commands.** Every v1.5.2 endpoint, with the same parameters, the same argument validation
  (Lua semantics: JSON `null` is an absent field, `{}` and `[]` are both an empty table,
  "integer" means `floor(v) == v`), the same state gates and the same checks in the same order,
  with the exact error names, codes and messages.
- **The view.** `gamestate` and every endpoint that returns the game state send the structure
  `gamestate.lua` builds: the same fields, present or left out under the same conditions, empty
  tables as `[]`, numbers printed with `%.14g`.
- **Text fields carry names.** Where the live game sends a localized description (a card's
  `value.effect`, a boss blind's `effect`, a tag's `tag_effect`), the simulator sends the name.

Game rules follow vanilla Balatro 1.0.1o, as the engine does everywhere. Where BalatroBot pushes
the game past a gate a player can't (selling an eternal joker, a greyed-out pack pick), the
simulator refuses; where Steamodded changes vanilla, the difference is documented in
[smods-diffs.md](parity/smods-diffs) rather than imitated.

## Entry points

**`jackdaw serve`** runs BalatroBot's HTTP server on the simulator:

```bash
jackdaw serve                                   # 127.0.0.1:12346, like BalatroBot
jackdaw serve --port 0                          # any free port (printed on start)
jackdaw serve --host 0.0.0.0 --fs-root runs/    # off loopback a filesystem root is required
jackdaw serve --debug                           # also BalatroBot's test endpoints
```

**`SimBackend`** is the same session in-process (`jackdaw.bridge`). `handle(method, params)`
returns the result dict or raises `RPCError` with BalatroBot's `code`, `message` and
`data = {"name": ...}`. `add` also accepts the engine's names (`m_glass`, `foil`, `Gold`), as
`LiveBackend` does.

```python
from jackdaw.bridge import SimBackend
from jackdaw.engine.actions import PlayHand

sim = SimBackend()
sim.handle("start", {"deck": "RED", "stake": "WHITE", "seed": "TEST123"})
view = sim.handle("select", {})
sim.legal_actions()                             # BALATROBOT legality, as engine actions
sim.request_for(PlayHand(card_indices=(0, 2)))  # ("play", {"cards": [0, 2]})
```

**`jackdaw.core.Game`** (the Rust engine) has the bot layer too: `bot_view()` (the view as a
dict), `bot_request(action)` (the request that performs an action, or `None`) and
`legal_actions(Legality.BALATROBOT)`. `jackdaw._core.BotSession` is the raw session (JSON text
in and out) behind all of these.

**`BridgeAdapter(SimBackend())`** plays the environment over the protocol: actions go out as
requests, the mask and legal actions follow BALATROBOT legality, and a shadow run in the Python
engine supplies `raw_state` and is checked against every response (`DivergenceError`).

**`LiveAdapter`** (`jackdaw.bridge`; also what `BridgeAdapter(LiveBackend(...))` gives) plays the
live game the same way, with a shadow run in the Rust engine: each request is taken by the
shadow through this layer first, then sent to the game, and the game's view is compared with the
shadow's `bot_view()`. Known Steamodded and BalatroBot differences are handled by a deviation
catalog; `jackdaw lockstep` runs it at scale. See [Lockstep against the live game](live).

## Command-space policy

The engine's action set and BalatroBot's endpoints don't line up one to one.

**Translated.** Each engine action has one request (`Game.bot_request`, Rust
`jackdaw_bot::translate`). `SortHand` and the `Swap*` actions go out as `rearrange` with the
whole permutation (a sort is the permutation that sorts the hand). The sort *mode*
(`G.hand.config.sort`) has no BalatroBot counterpart: after a translated sort by suit, cards
drawn later are still placed by rank, in the live game and in the simulator alike.

**Sim-only, masked under BALATROBOT legality.** Moves vanilla allows that the live bot can't make:

- `RerollBoss` (Director's Cut, Retcon) and `BuyAndUse` have no endpoint;
- `use` and `sell` outside `SELECTING_HAND` and `SHOP`: vanilla also lets a consumable be used
  at blind select, on the cash-out screen and in an open pack, and a card be sold at blind
  select and in an open pack;
- a consumable that takes cards (`max_highlighted`), used in the shop;
- Aura through `use`: its config has no `max_highlighted`, so `use.lua` never highlights a target
  and the card can't be used (`pack` special-cases it, so an Aura pick works);
- buying, or picking from a pack, a Negative joker at full joker slots (the bot compares the count
  with the limit and has no Negative exception).

**BalatroBot allows, vanilla forbids.** The live endpoints call `G.FUNCS` directly and skip two
UI gates. The simulator keeps vanilla's rules and refuses, with `NOT_ALLOWED`:

- selling an eternal joker: `Cannot sell '<name>': the card is eternal (vanilla's sell button
  refuses it)`;
- a pack pick the game greys out (Judgement or The Soul at full joker slots, The Emperor at full
  consumable slots, ...): `Cannot select '<name>' from the pack: vanilla greys this pick out (it
  cannot be used now)` ([smods-diffs.md B01](parity/smods-diffs#b01)).

`play` and `discard` click the requested cards as BalatroBot does: under Cerulean Bell the forced
card stays highlighted and the selection stops at five, so a request that leaves the forced card
out plays what the live game plays ([B02](parity/smods-diffs#b02)). A request whose clicks end
with no card highlighted (one index given twice) is refused: vanilla never plays an empty hand.

**Debug only.** `add`, `set`, `save`, `load` and `menu` change the run in ways no move does;
they are never in a legal-action list.

**Legality.** `Legality.VANILLA` is the engine's legal list (Balatro 1.0.1o).
`Legality.BALATROBOT` keeps the vanilla actions whose request passes BalatroBot's checks: the
dispatcher's argument and state checks and the endpoint's own validation, run as a dry run that
changes nothing. Every action it allows is one the live bot accepts. From Python:
`Game.legal_actions(Legality.BALATROBOT)`, `SimBackend.legal_actions()`; from Rust:
`jackdaw_bot::legal_actions(&game, Legality::Balatrobot)` and, per action,
`jackdaw_bot::bot_accepts(&game, &action)`. A `PlayHand` / `Discard` with no cards (the legal
list's marker for "any selection") is checked with one card, the forced one if any.

## Endpoints

| method | states | notes |
|---|---|---|
| `health` | any | |
| `gamestate` | any | the view (`MENU` before a run) |
| `rpc.discover` | any | the vendored `openrpc.json` |
| `menu` | any | ends the run |
| `start` | `MENU` | random vanilla seed when none is given |
| `select`, `skip` | `BLIND_SELECT` | the Boss can't be skipped |
| `play`, `discard` | `SELECTING_HAND` | clicks as above; at most 5 cards |
| `cash_out` | `ROUND_EVAL` | |
| `buy`, `reroll`, `next_round` | `SHOP` | `buy` takes `card`, `voucher` or `pack` |
| `sell`, `use` | `SELECTING_HAND`, `SHOP` | eternal jokers refused |
| `pack` | `SMODS_BOOSTER_OPENED` | greyed picks refused |
| `rearrange` | `SELECTING_HAND`, `SHOP`, `SMODS_BOOSTER_OPENED` | the whole permutation |
| `add`, `set` | see upstream | debug |
| `save`, `load` | see upstream | snapshot files (below) |
| `screenshot` | see upstream | always `INTERNAL_ERROR` |
| `test_*` | any | only with `--debug` / `debug=True` |

`use` highlights its targets the vanilla way: Cerulean Bell's forced card first, then the given
cards, a repeat skipped. BalatroBot clears the highlight with `force` first, dropping the forced
card; for the requests the legal list makes (the forced card among the targets) the two agree.

## The view

- **States.** `G.STATE`'s name, with Steamodded's `SMODS_BOOSTER_OPENED` for every open pack;
  `MENU` before the first `start` and after `menu`; `UNKNOWN` after a `load` that deleted the run
  and then failed.
- **Areas.** `jokers`, `consumables`, `cards` (the deck), `hand`, and in the shop `shop`,
  `vouchers`, `packs`; `pack` while a pack is open. Each has `count`, `limit`,
  `highlighted_limit` (5 for the hand and the deck, 1 elsewhere) and `cards`. Limits are the
  game's `card_limit`s: joker and consumable slots, the hand size, the shop's joker slots, one
  voucher slot per shop (plus one per Voucher Tag), two booster slots, the pack's size, and the
  deck's limit (52, raised by an overfilling emplace, lowered to the playing-card count every
  frame, plus one for Marble Joker, Cryptid and DNA).
- **Leftover shop areas.** After the shop closes, `G.shop_jokers`, `G.shop_vouchers` and
  `G.shop_booster` still point at the removed areas: the view keeps reporting them with no cards
  and the last frame's count, until the run ends (read from the code, not yet a live capture:
  [gaps.md](parity/gaps) B21).
- **Cards.** `label` is the center's label (`Glass Card`, `Joker`, `The Fool`); playing cards
  cost 1 to buy and to sell; `modifier.enhancement` is `ability.effect` upper-cased for every
  card, so jokers and consumables report theirs (`"MULT"`, `"ENHANCE"`); flags that are false
  (`eternal`, `rental`, `debuff`, `hidden`, `highlight`) are left out, and a card with none has
  `modifier: []` / `state: []`.
- **Hands.** All twelve, secret ones included, with `example` lists.
- **Blinds.** `score` is always computed from the blind's multiplier and the ante's base amount;
  `effect`, `tag_name` and `tag_effect` carry names.
- **`used_vouchers`** maps each redeemed voucher's key to `""`.
- **JSON.** Encoded as Steamodded's rxi `json.lua` does: an empty table is `[]`, numbers are
  printed with `%.14g` (14 significant digits, exact ties to even, as the game's LuaJIT 2.0.5
  does on Windows). Object keys are not in Lua's `pairs` order (JSON objects are unordered).
- **NaN and infinity** can't be encoded. The live server then sends nothing and the client times
  out; the simulator answers `INTERNAL_ERROR` `Failed to encode response: unexpected number
  value 'inf'`. The request itself has taken effect in both.

## Save and load

`save` writes the run's canonical snapshot (`jackdaw.snapshot/1` JSON,
[snapshot-schema.md](snapshot-schema)) and `load` reads one; neither touches the game's `.jkr`
saves, so files don't move between the simulator and the live game. As in the live game, `load`
deletes the run before reading the file: a bad file leaves no run (`UNKNOWN`) until `menu` or a
good `load`. With `--fs-root` (required off loopback) every path must stay inside that
directory. The runtime-only parts of the view restart on `load`: the deck limit becomes the
playing-card count and the leftover shop areas are gone (B20).

## The server

`jackdaw serve` follows `server.lua`: one connection at a time, one request per connection
(`Connection: close`), `POST /` only (other methods: 405, other paths: 404), bodies up to 64 KiB.
The JSON-RPC envelope: `jsonrpc` must be `"2.0"`, `id` is required and must be an integer or a
string and is echoed back, errors carry `{code, message, data: {name}}`. Endpoint errors use
BalatroBot's names and codes: `INTERNAL_ERROR` -32000, `BAD_REQUEST` -32001, `INVALID_STATE`
-32002, `NOT_ALLOWED` -32003.

## Differences from the live game

Besides the policy above:

- **Timing.** The live endpoints answer once the game's events settle; the simulator settles
  every request at once. A live request can hang where the simulator answers: `skip` with a tag
  that opens a pack waits for `BLIND_SELECT` live; the simulator returns the open pack.
- **`screenshot`** always fails with `Failed to encode screenshot` (nothing is drawn).
- **`add`** (debug) follows the Python oracle, so parity setups stay exact: a joker's
  `'edi'..ante` poll runs even when an edition is given (live skips it, gaps.md B15); a playing
  card draws no `'front'..ante` card (live draws one, B18); an enhanced playing card isn't checked
  against the boss (live debuffs it at once, B19). Stickers are forced as `force_stickers` does:
  eternal and perishable together, on any joker. A voucher `add` puts in the shop has no buy
  button live, so buying it raises in `buy.lua`; the simulator reports the same Lua error
  (`INTERNAL_ERROR`).
- **`rearrange`** also writes the new positions into `ability.order` and the shared
  `config.center.order` live; no rule reads them during a run, and the simulator doesn't.
- **Test endpoints** (`--debug`) are BalatroBot's `test_endpoint`, `test_state_endpoint`,
  `test_error_endpoint` and `test_validation`, with their exact responses.

## How it is checked

- `tests/bot/test_error_table.py`: every error message in the vendored endpoints (94 sites, one
  per `message =` or per value of a message variable) is produced by a test row with its exact
  name, code and message, or listed as unreachable with the reason (13).
- `tests/bot/test_upstream_suite.py`: BalatroBot's own endpoint tests run against
  `jackdaw serve` (455 pass; 6 strict xfails in `tests/bot/upstream_xfail.json`: described text
  fields and screenshots).
- `tests/bot/test_view_schema.py`: views from random runs fit `openrpc.json`'s `GameState`
  schema, patched where the schema is wrong (`tests/bot/schema.py`).
- `tests/bot/test_bot_native.py`: every engine tape is replayed through the requests; each step's
  snapshot equals the native step's (the sort mode aside, as above).
- `tests/parity/test_rust_setup.py`: the parity setups (`add`, `set`, `force_boss`) give the
  oracle's snapshot in Rust.

Steamodded behaviour (`SMODS.add_card`, `SMODS.add_booster_to_shop`, `SMODS_BOOSTER_OPENED`) was
read from Steamodded 1.0.0~BETA-1503a as installed with BalatroBot and from the Lovely dump of the
patched game; none of it is vendored.

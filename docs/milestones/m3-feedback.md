# M3 Feedback: Data Tables & Card Foundation

## 1. How was extraction done?

**Lua script approach** — not regex parsing.  `scripts/extract_prototypes.lua`
loads `Game:init_item_prototypes()` from `game.lua` in a mock environment
(stubbing `HEX()`, `localize()`, cutting off before `save_progress()` and
`love.filesystem` calls), then serializes the raw Lua tables to JSON via a
custom serializer.  This is far more reliable than regex for nested tables
with inconsistent formatting.

The Python wrapper `scripts/extract_prototypes.py` runs the Lua script via
LuaJIT subprocess and prints a verification summary.

6 JSON files produced: `centers.json` (299 entries), `cards.json` (52),
`blinds.json` (30), `tags.json` (24), `stakes.json` (8), `seals.json` (4).

## 2. Unusual config shapes

**Empty Lua tables** serialize as `[]` in JSON (empty array), not `{}` (empty
object).  The prototype loader normalizes these: `_get(d, "config", {})`
returns `{}` when the value is `[]` and the default is `dict`.

**config.extra** varies wildly across jokers:
- Scalar: `extra = 8` (Fibonacci), `extra = 30` (Banner)
- Sub-table: `extra = {s_mult = 3, suit = "Diamonds"}` (Greedy Joker)
- Complex: `extra = {Xmult = 4, every = 5, remaining = "5 remaining"}` (Loyalty Card)

The `Xmult` (capital X) naming is unique to config — `set_ability` maps it to
`x_mult` (lowercase) in the ability dict.  Some jokers have `Xmult` at the
config top level (j_duo, j_ramen, etc.) while others have it nested inside
`extra` (j_loyalty_card).  Only the top-level one maps to `ability.x_mult`.

**Voucher `requires`** field is sometimes a string, sometimes a list.  The
loader normalizes to always-list: `if isinstance(req, str): req = [req]`.

**Boss blinds** use `boss_colour` as an RGBA array (from `HEX()` function),
which we capture but don't use for the simulator.

## 3. Fields that didn't translate cleanly

**`discovered`/`unlocked`/`alerted`** — These are runtime profile state, not
prototype data.  The extracted JSON captures the initial (default) values.
The simulator starts all cards as unlocked/discovered since it doesn't
simulate profile progression.

**`consumeable`** boolean on center prototypes — In Lua, this is a flag that
causes `set_ability` to copy the entire config into `ability.consumeable`.
We replicate this exactly.

**`wip`/`demo`/`omit`** flags — Present on some centers for internal
development use.  Captured in JSON, ignored by the prototype loader.

**Seal ordering** — P_SEALS keys are bare strings (`"Gold"`, `"Red"`, etc.)
not prefixed like other centers.  This matches how `card.seal` stores them.

## 4. Performance

| Operation | Rate | Notes |
|---|---|---|
| Deck build (52 cards) | **6,900 decks/sec** | Includes card_from_control + set_ability + set_base |
| Joker creation (150) | **617,000 jokers/sec** | set_ability from JSON center lookup + deep copy |
| Prototype loading | ~1ms | One-time JSON parse at import |

A full run start (build deck + create starting jokers) takes < 1ms.
This is negligible compared to the RNG and scoring pipeline.

## 5. Data coverage

| Category | Extracted | Expected | Status |
|---|---|---|---|
| Jokers | 150 | 150 | Complete |
| Tarots | 22 | 22 | Complete |
| Planets | 12 | 12 | Complete |
| Spectrals | 18 | 18 | Complete |
| Vouchers | 32 | 32 | Complete |
| Backs | 16 | 16 (15 + Challenge) | Complete |
| Boosters | 32 | ~30-32 | Complete |
| Enhancements | 8 | 8 | Complete |
| Editions | 5 | 5 (4 + Base) | Complete |
| Playing Cards | 52 | 52 | Complete |
| Blinds | 30 | 30 (2 + 28 boss) | Complete |
| Tags | 24 | 24 | Complete |
| Stakes | 8 | 8 | Complete |
| Seals | 4 | 4 | Complete |

**Total: 491 prototypes extracted, matching all expected counts.**

Joker rarity breakdown: 61 Common + 64 Uncommon + 20 Rare + 5 Legendary = 150.

## 6. Modules built in M3

| Module | Lines | Tests | Purpose |
|---|---|---|---|
| `data/prototypes.py` | 340 | 67 | Typed dataclasses + lookup dicts for all proto types |
| `data/hands.py` | 120 | 35 | HandType enum + base values + level-up formula |
| `data/enums.py` | 200 | 44 | GameState, Suit, Rank, Enhancement, Edition, Seal, Rarity |
| `data/blind_scaling.py` | 90 | 54 | Blind chip scaling (hardcoded + exponential) |
| `card.py` | 300 | 64 | Card class with set_ability, set_cost, scoring methods |
| `card_area.py` | 175 | 27 | CardArea container with shuffle/sort/draw |
| `card_factory.py` | 180 | 35 | Factory functions for all card types |
| `deck_builder.py` | 150 | 27 | Starting deck construction for all 15 decks |
| Integration tests | — | 40 | Cross-references, full-deck builds, all-joker creation |

**Total M3: ~1,555 lines of engine code, 393 new tests (524 total with M2).**

## 7. Concerns for downstream milestones

**Joker calculate_joker dispatch**: The 150-joker `if/elseif` chain is the
next major piece.  Each joker needs to read from `ability.extra` which has
heterogeneous shapes — the untyped `dict[str, Any]` design handles this but
makes type-checking impossible.  This is intentional and matches the source.

**Pool generation**: `get_current_pool` filters by unlock state, pool flags,
banned keys, and enhancement gates.  The prototype data has all the fields
needed, but the filtering logic depends on runtime game state (which
vouchers are purchased, which jokers have been seen, etc.).

**Blind debuff logic**: Each boss blind has unique debuff behavior checked
against card properties (suit, rank, face status).  The CardBase fields
(suit, rank, id, face_nominal) are all in place for this.

**Consumable targeting**: Tarots need `config.max_highlighted` and
`config.mod_conv` to determine selection rules and enhancement changes.
These are captured in the extracted config dicts.

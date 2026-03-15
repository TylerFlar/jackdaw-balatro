# RNG System

Balatro's pseudorandom number generation: algorithm, seed lifecycle, per-system
streams, determinism analysis, and every random call site.

---

## Architecture Overview

The system is a **hybrid design** with three layers:

```
Layer 1: pseudohash(str)          — custom float hash, string → [0,1)
Layer 2: pseudoseed(key)          — stateful float LCG per named stream
Layer 3: math.randomseed + random — Lua's built-in RNG, re-seeded per call
```

Every gameplay random event flows through all three layers:
1. A **named key string** (e.g. `'boss'`, `'lucky_mult'`) identifies the RNG stream
2. `pseudoseed(key)` advances that stream's state and returns a float
3. `pseudorandom()` re-seeds `math.randomseed()` with that float, then draws from `math.random()`

Lua's global `math.random` is re-seeded on **virtually every random operation**.
The actual deterministic sequence comes from `pseudoseed`'s float LCG, not from
Lua's RNG sequence.

---

## Algorithm Details

### `pseudohash(str)` — String Hash (`misc_functions.lua:279`)

```lua
function pseudohash(str)
    local num = 1
    for i = #str, 1, -1 do
        num = ((1.1239285023 / num) * string.byte(str, i) * math.pi + math.pi * i) % 1
    end
    return num
end
```

- Iterates string bytes in **reverse**
- Accumulator starts at 1, each step: `(1.1239... / num * byte * π + π * i) % 1`
- Returns a float in `[0, 1)` — pure deterministic hash, no state
- The division by `num`, multiplication by byte and π, and mod 1 create good mixing

### `pseudoseed(key)` — Stateful LCG (`misc_functions.lua:298`)

```lua
function pseudoseed(key, predict_seed)
    if key == 'seed' then return math.random() end   -- non-deterministic escape hatch

    if predict_seed then                               -- stateless preview mode
        local _pseed = pseudohash(key .. (predict_seed or ''))
        _pseed = math.abs(tonumber(string.format("%.13f",
            (2.134453429141 + _pseed * 1.72431234) % 1)))
        return (_pseed + (pseudohash(predict_seed) or 0)) / 2
    end

    -- Normal mode: advance stored state
    if not G.GAME.pseudorandom[key] then
        G.GAME.pseudorandom[key] = pseudohash(key .. (G.GAME.pseudorandom.seed or ''))
    end

    G.GAME.pseudorandom[key] = math.abs(tonumber(string.format("%.13f",
        (2.134453429141 + G.GAME.pseudorandom[key] * 1.72431234) % 1)))
    return (G.GAME.pseudorandom[key] + (G.GAME.pseudorandom.hashed_seed or 0)) / 2
end
```

Three modes:
1. **`key == 'seed'`**: Returns raw `math.random()` — intentionally non-deterministic
2. **`predict_seed` provided**: Stateless preview (used by `get_first_legendary` to
   test seeds without mutating game state)
3. **Normal (no predict_seed)**: Stateful — advances `G.GAME.pseudorandom[key]`

The core transformation is a **float-domain linear congruential generator**:
```
next = (2.134453429141 + current × 1.72431234) % 1
```
Effectively: additive constant ≈ 0.1345 (after mod 1), multiplier ≈ 1.7243.

The `string.format("%.13f", ...)` truncation is critical — it prevents
floating-point drift and ensures cross-platform reproducibility at 13 decimal
digits.

The return value averages the advanced state with `hashed_seed`, coupling every
output to the original seed: `(advanced + hashed_seed) / 2`.

### `pseudorandom(seed, min, max)` — Bridge to math.random (`misc_functions.lua:315`)

```lua
function pseudorandom(seed, min, max)
    if type(seed) == 'string' then seed = pseudoseed(seed) end
    math.randomseed(seed)
    if min and max then return math.random(min, max)
    else return math.random() end
end
```

Re-seeds Lua's global RNG with the `pseudoseed` output, then draws one value.
Each call temporarily hijacks `math.random` state.

### `pseudorandom_element(_t, seed)` — Weighted Selection (`misc_functions.lua:253`)

```lua
function pseudorandom_element(_t, seed)
    if seed then math.randomseed(seed) end
    local keys = {}
    for k, v in pairs(_t) do
        keys[#keys+1] = {k = k, v = v}
    end
    -- Deterministic sort (critical: pairs() order is non-deterministic)
    if keys[1] and keys[1].v and type(keys[1].v) == 'table' and keys[1].v.sort_id then
        table.sort(keys, function(a, b) return a.v.sort_id < b.v.sort_id end)
    else
        table.sort(keys, function(a, b) return a.k < b.k end)
    end
    local key = keys[math.random(#keys)].k
    return _t[key], key
end
```

**Sorts the table deterministically** before picking — by `sort_id` if available,
otherwise by key (lexicographic). This is essential because Lua's `pairs()`
iteration order is non-deterministic for hash tables.

Returns: `(selected_value, selected_key)`.

### `pseudoshuffle(list, seed)` — Fisher-Yates (`misc_functions.lua:206`)

```lua
function pseudoshuffle(list, seed)
    if seed then math.randomseed(seed) end
    -- Pre-sort by sort_id if available
    if list[1] and list[1].sort_id then
        table.sort(list, function(a, b) return (a.sort_id or 1) < (b.sort_id or 2) end)
    end
    -- Standard Fisher-Yates (backward variant)
    for i = #list, 2, -1 do
        local j = math.random(i)
        list[i], list[j] = list[j], list[i]
    end
end
```

- **Pre-sorts** by `sort_id` to ensure deterministic starting order
- Standard unbiased **Fisher-Yates shuffle** (backward variant)
- Mutates the list in-place

### `generate_starting_seed()` — Entropy Source (`misc_functions.lua:219`)

```lua
function generate_starting_seed()
    -- For Gold Stake (8): ensure first legendary is one player hasn't gold-stickered
    if G.GAME.stake >= 8 then
        -- ... loop generating seeds, rejecting those whose first legendary is already beaten ...
    end
    -- Normal: use cursor position + time as entropy
    return random_string(8,
        G.CONTROLLER.cursor_hover.T.x * 0.33411983 +
        G.CONTROLLER.cursor_hover.T.y * 0.874146 +
        0.412311010 * G.CONTROLLER.cursor_hover.time)
end
```

Entropy comes from **mouse cursor position and hover time** at the moment the
player starts a run. The irrational-looking multipliers spread the entropy.

### `random_string(length, seed)` — Seed String Generation (`misc_functions.lua:270`)

Generates an 8-character alphanumeric string (digits 1-9, letters A-N, P-Z).
Deliberately excludes `0` and `O` to avoid visual ambiguity.

---

## Seed Lifecycle

### 1. Engine Boot

```lua
-- globals.lua:125
self.SEED = os.time()

-- main.lua:31
math.randomseed(G.SEED)
```

`G.SEED` seeds Lua's RNG for non-gameplay uses (visual effects, audio pitch).
It has **no connection** to the gameplay seed.

### 2. Run Start — Seed Selection

```lua
-- game.lua:2162-2164
self.GAME.pseudorandom.seed =
    args.seed                              -- player-entered seed (sets seeded=true)
    or ("TUTORIAL" if tutorial incomplete)  -- fixed string for tutorial
    or generate_starting_seed()             -- random from cursor entropy
```

The seed is always a **string** (e.g. `"TUTORIAL"`, `"A3K9NZ2B"`).

### 3. Hash Computation

```lua
-- game.lua:2167-2168
for k, v in pairs(self.GAME.pseudorandom) do
    if v == 0 then self.GAME.pseudorandom[k] = pseudohash(k .. seed) end
end
self.GAME.pseudorandom.hashed_seed = pseudohash(seed)
```

`hashed_seed` is computed once as a float and mixed into every subsequent output.
The loop re-hashes any keys loaded from a save file (which are stored as 0).

### 4. Consumption

Each gameplay system uses a named key:
```lua
pseudoseed('boss')           -- advances G.GAME.pseudorandom['boss']
pseudorandom('lucky_mult')   -- advances ['lucky_mult'], returns via math.random
pseudorandom_element(pool, pseudoseed('Joker1'..ante))  -- pool selection
pseudoshuffle(deck.cards, pseudoseed('shuffle'))         -- deck shuffle
```

### 5. Save/Load

The entire `G.GAME.pseudorandom` table is serialized in save files. Loading a save
restores every stream's exact position, so the RNG continues deterministically.

---

## Independent RNG Streams

Each seed key is an independent stream. Streams don't interfere with each other —
calling `pseudoseed('boss')` doesn't affect the state of `pseudoseed('lucky_mult')`.

### Complete Seed Key Inventory

#### Deck & Hand Management
| Key | Purpose |
|-----|---------|
| `'shuffle'` | Deck shuffle (Fisher-Yates) |
| `'flipped_card'` | Whether dealt card is face-down (challenge modifier) |
| `'erratic'` | Erratic Deck suit/rank randomization |
| `'edition_deck'` | Which cards get editions in edition-starting decks |
| `'front'` (+ante, +append) | Random card face (suit+rank) for new playing cards |

#### Blind System
| Key | Purpose |
|-----|---------|
| `'boss'` | Boss blind selection for the ante |
| `'hook'` | The Hook — select cards to discard |
| `'cerulean_bell'` | Cerulean Bell — force card selection |
| `'crimson_heart'` | Crimson Heart — select joker to debuff |
| `'wheel'` | The Wheel — chance to flip cards (prob/7) |

#### Shop & Card Creation
| Key | Purpose |
|-----|---------|
| `'cdt'`+ante | Card type distribution (Joker vs Tarot vs Planet) |
| `'rarity'`+ante+append | Joker rarity roll (common/uncommon/rare/legendary) |
| `'Joker1'`..`'Joker3'`+append+ante | Joker pool selection by rarity |
| `'Joker4'` | Legendary joker selection (with predict_seed) |
| `_type`+append+ante | Non-joker pool keys (Tarot, Planet, Spectral, Voucher, Tag, Enhanced) |
| `_pool_key+'_resample'+it` | Resampling when card unavailable |
| `'soul_'+_type+ante` | Soul/Black Hole forced key chance (>0.997) |
| `'pack_generic'`+ante | Booster pack type selection |
| `'Voucher_fromtag'` | Voucher from Voucher Tag |

#### Edition System
| Key | Purpose |
|-----|---------|
| `_key` or `'edition_generic'` | Edition poll (foil/holo/polychrome/negative) |
| `'packetper'`+ante / `'etperpoll'`+ante | Eternal/Perishable modifier |
| `'packssjr'`+ante / `'ssjr'`+ante | Rental modifier |

#### Scoring
| Key | Purpose |
|-----|---------|
| `'lucky_mult'` | Lucky Card +mult trigger (prob/5) |
| `'lucky_money'` | Lucky Card +$20 trigger (prob/15) |
| `'glass'` | Glass Card destruction chance |

#### Joker Effects
| Key | Joker |
|-----|-------|
| `'wheel_of_fortune'` | Wheel of Fortune — add edition |
| `'ectoplasm'` | Ectoplasm — select joker for edition |
| `'hex'` | Hex — select joker for Polychrome |
| `'halu'`+ante | Hallucination — create Tarot |
| `'cavendish'` / `'gros_michel'` | Self-destruction chance |
| `'8ball'` | 8 Ball — create Tarot |
| `'business'` | Business Card — earn $2 |
| `'bloodstone'` | Bloodstone — xMult on Hearts |
| `'parking'` | Reserved Parking — $ on face cards |
| `'space'` | Space Joker — level up hand |
| `'misprint'` | Misprint — random mult range |
| `'invisible'` | Invisible Joker — duplicate selection |
| `'perkeo'` | Perkeo — consumable copy selection |
| `'madness'` | Madness — joker destruction selection |
| `'to_do'` / `'false_to_do'` | To Do List — target hand selection |
| `'omen_globe'` | Omen Globe voucher — Spectral in Standard Pack |

#### Round-Start Targeting
| Key | Purpose |
|-----|---------|
| `'idol'`+ante | Idol joker target (rank + suit) |
| `'mail'`+ante | Mail-In Rebate target rank |
| `'anc'`+ante | Ancient Joker target suit |
| `'cas'`+ante | Castle joker target suit |

#### Spectral Card Effects
| Key | Card |
|-----|------|
| `'sigil'` | Sigil — suit conversion |
| `'ouija'` | Ouija — rank conversion |
| `'random_destroy'` | Card destruction selection |
| `'familiar_create'` | Familiar — face card creation |
| `'grim_create'` | Grim — Ace creation |
| `'incantation_create'` | Incantation — number card creation |
| `'spe_card'` | Enhancement for created card |
| `'immolate'` | Immolate — destruction selection |
| `'ankh_choice'` | Ankh — joker duplication |

#### Standard Pack
| Key | Purpose |
|-----|---------|
| `'stdset'`+ante | Enhanced vs Base card type |
| `'stdseal'`+ante | Seal chance |
| `'stdsealtype'`+ante | Seal type selection |

#### Illusion Voucher
| Key | Purpose |
|-----|---------|
| `'illusion'` | Enhanced vs Base, seal chance, edition poll (3 consecutive calls) |

#### Joker Creation Effects
| Key | Joker |
|-----|-------|
| `'cert_fr'` | Certificate — playing card front |
| `'certsl'` | Certificate — seal type |
| `'marb_fr'` | Marble Joker — Stone Card front |

#### Tags
| Key | Purpose |
|-----|---------|
| `'orbital'` | Orbital Tag — which hand to level |

#### Special
| Key | Purpose |
|-----|---------|
| `'seed'` | **Non-deterministic** — returns raw `math.random()` |

---

## Determinism Analysis

### Is the seed fully deterministic?

**Almost.** Given the same seed string and the same sequence of player actions,
the run will produce identical outcomes — with **three exceptions**:

### Known Sources of Non-Determinism

| # | Location | Code | Impact |
|---|----------|------|--------|
| 1 | `tag.lua:211` | `math.random(1,2)` | Charm Tag: picks Mega Arcana pack variant 1 or 2 |
| 2 | `tag.lua:226` | `math.random(1,2)` | Meteor Tag: picks Mega Celestial pack variant 1 or 2 |
| 3 | `common_events.lua:1947` | `math.random(1,2)` | First Buffoon pack: picks variant 1 or 2 |
| 4 | `card.lua:959` | `math.random(100,1000000)` | Stone Card `get_id()` return — inconsequential (overridden by scoring) |

These three `math.random` calls are **not** routed through `pseudoseed` and use
whatever state Lua's global RNG happens to be in. In practice, the global RNG
state is heavily influenced by `pseudorandom` (which re-seeds it on every call),
so these may be "accidentally deterministic" in many cases — but they are not
**guaranteed** to be deterministic.

### Seed generation itself is non-deterministic

`generate_starting_seed()` uses cursor position and hover time as entropy.
This is intentional — each new run should have a unique seed. Only the tutorial
(`"TUTORIAL"`) and player-entered seeds are fixed.

### Frame timing does NOT affect RNG

The RNG system has no dependency on `dt`, frame rate, or timing. The event system
processes events in a fixed order regardless of frame timing. All random decisions
use the named seed streams, not wall-clock time.

### Player input ORDER affects outcomes

While the RNG is deterministic for a given sequence of actions, the **order** of
player decisions matters because each action advances different seed streams.
Playing hand A then buying joker B produces different outcomes than buying B first,
because the shop pool stream will be at a different position.

---

## Pool Selection Algorithm

`get_current_pool` (`common_events.lua:1963`) builds weighted pools, but the
selection itself is **unweighted** (uniform random from the filtered pool).

The weighting happens at pool **construction** time:
- Cards are filtered by: set membership, unlock status, `banned_keys`, pool flags,
  enhancement gates, stake requirements
- Rarity is rolled separately via `pseudorandom('rarity'..ante)` before the pool
  is queried — the rarity roll determines WHICH pool to draw from, then a uniform
  random pick selects from that rarity's pool

For non-joker cards (Tarot, Planet, etc.), the type is rolled first via weighted
rates (`joker_rate=20, tarot_rate=4, planet_rate=4`, etc.), then a uniform pick
from that type's pool.

---

## Seed Key Naming Conventions

| Pattern | Meaning |
|---------|---------|
| `key` | Global stream, not ante-specific |
| `key + ante` | Per-ante stream (resets when ante changes) |
| `key + ante + append` | Per-ante + context (append is typically `'shop'`, `'pack'`, etc.) |
| `_pool_key + '_resample' + iteration` | Retry stream for unavailable cards |

The ante-suffix pattern means that the shop on ante 3 uses different random
values than the shop on ante 5, even if the player makes identical choices.
This prevents "seed manipulation" where players could predict future shops
by observing early ones.

---

## Source File Reference

| Function | File | Line |
|----------|------|------|
| `pseudohash` | `misc_functions.lua` | 279 |
| `pseudoseed` | `misc_functions.lua` | 298 |
| `pseudorandom` | `misc_functions.lua` | 315 |
| `pseudorandom_element` | `misc_functions.lua` | 253 |
| `pseudoshuffle` | `misc_functions.lua` | 206 |
| `generate_starting_seed` | `misc_functions.lua` | 219 |
| `random_string` | `misc_functions.lua` | 270 |
| `get_starting_params` | `misc_functions.lua` | 1868 |
| `create_card` | `common_events.lua` | 2082 |
| `get_current_pool` | `common_events.lua` | 1963 |
| `poll_edition` | `common_events.lua` | 2055 |
| `get_next_voucher_key` | `common_events.lua` | 1901 |
| `get_pack` | `common_events.lua` | 1944 |
| Seed init in `start_run` | `game.lua` | 2162 |
| `init_game_object` | `game.lua` | 1862 |

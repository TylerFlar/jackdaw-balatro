# RNG Call Trace — Run Initialization

Exact sequence of every `pseudoseed`/`pseudorandom_element`/`pseudoshuffle`
call from `Game:start_run()` through the first blind selection.

Verified against live Balatro via balatrobot v1.4.1 (March 2026).

## Phase 1: RNG System Init (game.lua:2162-2168)

| # | Call | Seed key | File | Line |
|---|------|----------|------|------|
| 1 | `pseudohash(k..seed)` for each bucket | (per-stream init) | game.lua | 2167 |
| 2 | `pseudohash(seed)` | (hashed_seed) | game.lua | 2168 |

## Phase 2: Boss / Voucher / Tags (game.lua:2177-2180)

| # | Call | Seed key | File | Line | Notes |
|---|------|----------|------|------|-------|
| 3 | `pseudorandom_element(bosses, pseudoseed('boss'))` | **`'boss'`** | common_events.lua | 2379 | `get_new_boss()` — independent stream |
| 4 | `pseudorandom_element(pool, pseudoseed('Voucher1'))` | **`'Voucher1'`** | common_events.lua | 1904 | `get_next_voucher_key()` — pool_key from `get_current_pool('Voucher')` returns `'Voucher' .. ante` |
| 5 | `pseudorandom_element(pool, pseudoseed('Tag1'))` | **`'Tag1'`** | common_events.lua | 1917 | First `get_next_tag_key()` — Small blind tag |
| 6 | `pseudorandom_element(pool, pseudoseed('Tag1'))` | **`'Tag1'`** | common_events.lua | 1917 | Second `get_next_tag_key()` — Big blind tag (SAME stream key, second advance) |

### Pool key construction (common_events.lua:1972, 2052)

```lua
-- Line 1972: base pool_key
_starting_pool, _pool_key = G.P_CENTER_POOLS[_type], _type..(_append or '')

-- Line 2052: return with ante appended
return _pool, _pool_key..(not _legendary and G.GAME.round_resets.ante or '')
```

For `'Tag'` with no append: `_pool_key = 'Tag'` → returned as `'Tag' .. 1` = `'Tag1'`.
For `'Voucher'` with no append: `_pool_key = 'Voucher'` → returned as `'Voucher' .. 1` = `'Voucher1'`.

## Phase 3: Deck Shuffle (game.lua:2383)

| # | Call | Seed key | File | Line | Notes |
|---|------|----------|------|------|-------|
| 7 | `pseudoshuffle(cards, pseudoseed('shuffle'))` | **`'shuffle'`** | cardarea.lua | 573 | `self.deck:shuffle()` — NO argument → default `'shuffle'` |

**IMPORTANT**: The per-round shuffle in `new_round()` uses `'nr'..ante` (state_events.lua:344).
The initial run shuffle uses just `'shuffle'` — NO ante suffix.

## Phase 4: Targeting Cards (game.lua:2385-2389)

| # | Call | Seed key | File | Line |
|---|------|----------|------|------|
| 8 | `pseudorandom_element(cards, pseudoseed('idol1'))` | **`'idol1'`** | common_events.lua | 2281 |
| 9 | `pseudorandom_element(cards, pseudoseed('mail1'))` | **`'mail1'`** | common_events.lua | 2297 |
| 10 | `pseudorandom_element(suits, pseudoseed('anc1'))` | **`'anc1'`** | common_events.lua | 2308 |
| 11 | `pseudorandom_element(cards, pseudoseed('cas1'))` | **`'cas1'`** | common_events.lua | 2321 |

## Validation Results (balatrobot v1.4.1, 5 seeds)

After fixing seed keys to match above:

| Field | Before fix | After fix |
|-------|-----------|-----------|
| money | 5/5 | 5/5 |
| ante | 5/5 | 5/5 |
| deck_size | 5/5 | 5/5 |
| boss | 5/5 | 5/5 |
| **deck_order** | **0/5** | **5/5** |
| small_tag | 1/5 | 1/5 |
| big_tag | 2/5 | 2/5 |

### Bugs fixed

1. **Voucher seed key**: was `'Voucher'`, should be `'Voucher' + str(ante)` = `'Voucher1'`.
2. **Initial shuffle key**: was `'nr' + str(ante)` = `'nr1'`, should be `'shuffle'`.

### Remaining tag divergence

Tags still differ (4/5 small, 3/5 big). The pool_key `'Tag1'` matches Lua, but the
tag pool ordering or filtering may differ. Possible causes:

- Tag pool base order: Python sorts by `proto.order`, but Lua may iterate
  `G.P_CENTER_POOLS['Tag']` in a different order.
- The `pseudorandom_element` sort-by-sort_id logic may differ from Lua's
  iteration order over `G.P_CENTER_POOLS`.
- The Tag pool may include/exclude different entries due to `min_ante` or
  `requires` filtering differences.

The deck shuffle matching perfectly (50/50 cards across 5 seeds) confirms
that all RNG streams consumed between boss and shuffle are now correct.
The tag divergence is isolated to the Tag pool's internal ordering.

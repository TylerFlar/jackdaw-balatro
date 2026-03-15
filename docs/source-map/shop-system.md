# Shop System

How shop offerings are generated, priced, bought, sold, rerolled, and how
vouchers and booster packs work.

---

## Shop Structure

The shop has three areas, created in `G.UIDEF.shop()` (`UI_definitions.lua:637`):

| Area | CardArea global | card_limit | Contents |
|------|----------------|-----------|----------|
| Joker slots | `G.shop_jokers` | `G.GAME.shop.joker_max` (default **2**) | Jokers, Tarots, Planets, Playing Cards, Spectrals |
| Voucher slot | `G.shop_vouchers` | 1 | Single voucher (determined at round start) |
| Booster slots | `G.shop_booster` | 2 | Booster packs |

Plus a reroll button and a "Next Round" button.

---

## Shop Entry Flow

When `G.STATE` becomes `SHOP` (after cash_out):

1. `Game:update_shop` (`game.lua:3072`) creates `G.shop` UIBox if not exists
2. After slide-in animation (0.2s), populates:

### Joker Slot Population (`game.lua:3111`)

```lua
for i = 1, G.GAME.shop.joker_max - #G.shop_jokers.cards do
    G.shop_jokers:emplace(create_card_for_shop(G.shop_jokers))
end
```

### Voucher Population (`game.lua:3125`)

Creates a card from `G.GAME.current_round.voucher` (set by `get_next_voucher_key()`
at ante start). Voucher Tag can add a second voucher.

### Booster Population (`game.lua:3145`)

Creates 2 packs via `get_pack('shop_pack')`, stored in `G.GAME.current_round.used_packs`.

### Post-Population Hooks

- Tags fire `'voucher_add'` (Voucher Tag adds free voucher)
- Tags fire `'shop_final_pass'` (Coupon Tag makes everything free)

---

## Card Type Selection for Joker Slots

`create_card_for_shop(area)` (`UI_definitions.lua:742`) determines **what type
of card** fills each joker slot:

### Priority Checks

1. **Tutorial forced cards** — hardcoded for tutorial progression
2. **Tag-forced cards** — Rare Tag, Uncommon Tag fire `'store_joker_create'`
3. **Weighted random** — normal path (see below)

### Weighted Random Selection

Default rates (from `G.GAME` in `init_game_object`):

| Type | Rate | Base Probability |
|------|------|-----------------|
| Joker | `joker_rate` = 20 | ~71.4% |
| Tarot | `tarot_rate` = 4 | ~14.3% |
| Planet | `planet_rate` = 4 | ~14.3% |
| Spectral | `spectral_rate` = 0 | 0% |
| Playing Card | `playing_card_rate` = 0 | 0% |

**Modified by vouchers:**
- Tarot Merchant: `tarot_rate *= 1.5` → 6
- Planet Merchant: `planet_rate *= 1.5` → 6
- Magic Trick: `playing_card_rate = 4`
- Ghost Deck back: `spectral_rate` set at run start

The roll uses seed `'cdt'..ante`:
```lua
local poll = pseudorandom(pseudoseed('cdt'..ante)) * total_rate
```

### Illusion Voucher Enhancement

If player has the Illusion voucher and a Playing Card is rolled:
- 60% chance: Enhanced card (random enhancement)
- 40% chance: Base card
- 20% chance: random edition (Foil 50%, Holo 35%, Polychrome 15%)
- Separate seal chance roll

### Post-Creation Hooks

After each card is created, tags fire `'store_joker_modify'` — this is where
edition tags (Foil/Holo/Polychrome/Negative Tag) apply their edition and
set cost to 0.

---

## Pool Generation

### `get_current_pool(_type, _rarity, _legendary, _append)` (`common_events.lua:1963`)

Builds a filtered list of card center keys for selection.

### Rarity Roll (Joker type only)

If no `_rarity` override:
```lua
local roll = pseudorandom('rarity'..ante..append)
```

| Roll | Rarity | Pool |
|------|--------|------|
| > 0.95 | 3 (Rare) | `G.P_JOKER_RARITY_POOLS[3]` |
| > 0.7 | 2 (Uncommon) | `G.P_JOKER_RARITY_POOLS[2]` |
| ≤ 0.7 | 1 (Common) | `G.P_JOKER_RARITY_POOLS[1]` |
| (forced) | 4 (Legendary) | `G.P_JOKER_RARITY_POOLS[4]` |

Non-joker types use `G.P_CENTER_POOLS[_type]` directly.

### Filter Conditions

Every card in the starting pool is checked. **Must pass ALL** to be included:

| Filter | Condition |
|--------|-----------|
| Duplicate check | Not in `G.GAME.used_jokers` (unless Showman joker owned) |
| Unlock check | `v.unlocked ~= false` (legendaries bypass this) |
| Banned check | Not in `G.GAME.banned_keys` |
| Pool flag (no) | If `v.no_pool_flag` set, flag must NOT be true |
| Pool flag (yes) | If `v.yes_pool_flag` set, flag must be true |
| Enhancement gate | If `v.enhancement_gate` set, a card with that enhancement must exist in deck |
| Soul/Black Hole | Always excluded (only appear via soul chance in `create_card`) |

**Type-specific filters:**

| Type | Extra condition |
|------|----------------|
| Voucher | Not already used, all prerequisites used, not already in shop |
| Planet | If `softlock`, hand type must have been played at least once |
| Tag | Required center must be discovered, `min_ante` must be met |

**Failed cards** are added as `"UNAVAILABLE"` strings (not removed) to preserve
deterministic index alignment for seeded selection.

**Empty pool fallbacks:** Joker→`j_joker`, Tarot→`c_strength`, Planet→`c_pluto`,
Spectral→`c_incantation`, Voucher→`v_blank`, Tag→`tag_handy`.

---

## Card Creation

### `create_card()` (`common_events.lua:2082`)

The unified card factory. Full flow:

**1. Soul/Black Hole chance** (if `soulable=true`):
- Roll `pseudorandom('soul_'.._type..ante)`
- If > **0.997** (0.3% chance): force Soul or Black Hole
- Spectral type gets TWO rolls (one for each)

**2. Pool selection** (if no forced key):
- Call `get_current_pool()` for filtered pool
- Select with `pseudorandom_element(pool, pseudoseed(pool_key))`
- If `"UNAVAILABLE"`, resample with `pool_key..'_resample'..iteration`

**3. Joker modifiers** (shop/pack jokers only):

| Modifier | Roll seed | Threshold | Condition |
|----------|-----------|-----------|-----------|
| Eternal | `'etperpoll'`/`'packetper'`+ante | > 0.7 | `enable_eternals_in_shop` (stake ≥ 4) |
| Perishable | same roll | 0.4–0.7 | `enable_perishables_in_shop` (stake ≥ 7) |
| Rental | `'ssjr'`/`'packssjr'`+ante | > 0.7 | `enable_rentals_in_shop` (stake ≥ 8) |

Note: Eternal and Perishable share the same roll (mutually exclusive). Rental is a separate roll.

**4. Edition** (jokers only):
- `poll_edition('edi'..append..ante)`

---

## Edition Polling

### `poll_edition(_key, _mod, _no_neg, _guaranteed)` (`common_events.lua:2055`)

Single roll checked top-down (first match wins):

| Edition | Normal threshold (rate=1, mod=1) | Effective chance |
|---------|----------------------------------|-----------------|
| Negative | > 1 − 0.003 × mod | 0.3% |
| Polychrome | > 1 − 0.006 × rate × mod | 0.3% |
| Holo | > 1 − 0.02 × rate × mod | 1.4% |
| Foil | > 1 − 0.04 × rate × mod | 2.0% |
| None | everything else | 96.0% |

`rate` = `G.GAME.edition_rate` (doubled by Hone voucher, doubled again by Glow Up).
Negative chance ignores `rate` — only `mod` affects it.

**Guaranteed mode** (used by Wheel of Fortune, etc.): uses `×25` multiplier instead
of `rate×mod`, giving: Foil 50%, Holo 35%, Polychrome 7.5%, Negative 7.5%.

---

## Booster Pack Selection

### `get_pack(_key, _type)` (`common_events.lua:1944`)

**First-shop guarantee:** The very first booster pack offered is always a Buffoon
pack (variant 1 or 2, selected via **non-deterministic** `math.random(1,2)`).

**Normal selection:** Weighted random from `G.P_CENTER_POOLS['Booster']`:
- Each pack has a `weight` field (defaults to 1)
- Banned packs excluded
- Roll: `pseudorandom(pseudoseed(key..ante)) * total_weight`
- First pack whose cumulative weight crosses the roll is selected

---

## Pricing

### `Card:set_cost()` (`card.lua:369`)

```
extra_cost = G.GAME.inflation + edition_surcharge
cost = max(1, floor((base_cost + extra_cost + 0.5) × (100 - discount_percent) / 100))
```

| Factor | Source | Value |
|--------|--------|-------|
| `base_cost` | Center prototype `cost` field | Jokers: 2-8 by rarity. Tarots: 3. Planets: 3. Spectrals: 4. Vouchers: 10. |
| `inflation` | `G.GAME.inflation` | +1 per purchase in Inflation challenge |
| Edition surcharge | Card's edition | Foil: +2, Holo: +3, Poly: +5, Negative: +5 |
| `discount_percent` | `G.GAME.discount_percent` | Clearance Sale: 25%, Liquidation: 50% |
| Booster ante scaling | `modifiers.booster_ante_scaling` | Adds `ante - 1` to booster cost |

**Special overrides:**
- Astronomer joker: Planet cards and Celestial packs → cost = 0
- Rental cards → cost = 1
- Couponed (from tags) → cost = 0

### Sell Price

```lua
sell_cost = max(1, floor(cost / 2)) + (extra_value or 0)
```

- `extra_value` grows from: Egg joker (+3/round), Gift Card joker (+1/round to all)
- Sell price is always at least $1
- Eternal cards cannot be sold

---

## Reroll Mechanics

### `G.FUNCS.reroll_shop` (`button_callbacks.lua:2855`)

1. **Deduct cost** if `reroll_cost > 0` (free rerolls skip this)
2. **Decrement free rerolls** (`max(free_rerolls - 1, 0)`)
3. **Recalculate cost** via `calculate_reroll_cost()`
4. **Remove all cards** from `G.shop_jokers`
5. **Repopulate** with fresh `create_card_for_shop()` calls (same flow as initial population)
6. **Notify jokers** with `{reroll_shop = true}` (Flash Card gains +2 mult)
7. **Save run**

**Only joker slots are rerolled.** Vouchers and boosters are never rerolled.

### `calculate_reroll_cost()` (`common_events.lua:2263`)

```
if free_rerolls > 0: cost = 0
else: cost = base_reroll_cost + reroll_cost_increase
```

- `base_reroll_cost` defaults to **$5** (from `starting_params.reroll_cost`)
- Each non-free reroll: `reroll_cost_increase += 1` → cost goes $5, $6, $7, $8...
- `temp_reroll_cost` overrides base (D6 Tag sets to 0)
- Free reroll sources: Chaos the Clown (1/shop), some tags

### RNG for Rerolls

Rerolls use the **same seed streams** as initial population — `'cdt'..ante`,
`'rarity'..ante..append`, pool keys with ante suffix. Since each `pseudoseed()`
call advances the stream's state, rerolled cards are different from initial cards
but still deterministic for a given action sequence.

---

## Buy Transaction

### `G.FUNCS.buy_from_shop` (`button_callbacks.lua:2404`)

1. **Space check**: `check_for_buy_space(card)` — verifies joker or consumable
   slot available. Negative edition cards get +1 bonus (don't consume a slot).
2. **Remove from shop**: `area:remove_card(card)`
3. **Add to deck**: `card:add_to_deck()` — triggers passive effects (hand size,
   negative slot expansion, discovery)
4. **Place by type**:
   - Playing cards → `G.deck` (added to `G.playing_cards`, triggers `playing_card_joker_effects`)
   - Consumables → `G.consumeables`
   - Jokers → `G.jokers`
5. **Notify jokers**: `calculate_joker({buying_card = true, card = c1})` on all owned jokers
6. **Inflation**: if active, increment `G.GAME.inflation`, recalculate all costs
7. **Deduct money**: `ease_dollars(-card.cost)`
8. **Stats**: increment cards_purchased, type-specific buy counters

### Buy-and-Use (consumables)

If the "Buy and Use" button is clicked, the buy transaction runs first, then
immediately calls `G.FUNCS.use_card()` to use the consumable.

---

## Voucher Redemption

### `Card:redeem()` (`card.lua:1813`)

1. Mark as discovered
2. Clear `G.GAME.current_round.voucher`
3. Record in `G.GAME.used_vouchers[key] = true`
4. Deduct cost
5. **Apply effect**: `self:apply_to_run()` — this is where the voucher's game-rule
   change takes effect

### Key Voucher Effects (`Card:apply_to_run`, `card.lua:1880`)

| Voucher | Effect |
|---------|--------|
| Overstock / Overstock Plus | `change_shop_size(1)` → +1 shop joker slot |
| Clearance Sale | `discount_percent = 25` |
| Liquidation | `discount_percent = 50` |
| Hone | `edition_rate *= 2` |
| Glow Up | `edition_rate *= 2` |
| Grabber | `round_resets.hands += 1` |
| Nacho Tong | `round_resets.hands += 1` |
| Wasteful | `round_resets.discards += 1` |
| Recyclomancy | `round_resets.discards += 1` |
| Seed Money | `interest_cap = 50` |
| Money Tree | `interest_cap = 50` |
| Tarot Merchant | `tarot_rate *= 1.5` |
| Planet Merchant | `planet_rate *= 1.5` |
| Tarot Tycoon | `tarot_rate *= 1.5` |
| Planet Tycoon | `planet_rate *= 1.5` |
| Magic Trick | `playing_card_rate = 4` |
| Illusion | `playing_card_rate = 4` + enhancement/seal/edition on playing cards |
| Crystal Ball | `consumable_slots += 1` |
| Omen Globe | Spectral chance in Standard Packs |
| Observatory | Planet cards in consumable slots give xMult |
| Antimatter | `joker_slots += 1` |
| Hieroglyph | `ante -= 1` each round |
| Petroglyph | `hand/discard -= 1` each round |
| Directors Cut | `base_reroll_cost = max(0, base_reroll_cost - 1)` |
| Retcon | `base_reroll_cost = 0` |
| Paint Brush | `hand_size += 1` |
| Palette | `hand_size -= 1` |

---

## Sell Transaction

### `Card:sell_card()` (`card.lua:1590`)

1. **Self-notification**: `calculate_joker({selling_self = true})` — Luchador disables
   boss, Diet Cola adds Double Tag, Invisible duplicates a joker
2. **Award money**: `ease_dollars(sell_cost)`
3. **Dissolve animation**: card removed from game
4. **Stats**: increment cards_sold

### `Card:can_sell_card()` (`card.lua:1640`)

Cannot sell if:
- Cards are in play area
- Controller is locked
- `G.GAME.STOP_USE > 0`
- Card is Eternal
- Tutorial restrictions

---

## Slot Limits

### Player Joker Slots (`G.jokers.config.card_limit`)

- Default: 5 (from `starting_params.joker_slots`)
- Modified by: deck back, challenge, Antimatter voucher
- Negative edition cards don't count against the limit (+1 bonus in space check)
- Ectoplasm reduces: `card_limit -= 1` per activation

### Player Consumable Slots (`G.consumeables.config.card_limit`)

- Default: 2 (from `starting_params.consumable_slots`)
- Modified by: deck back, challenge, Crystal Ball voucher

### Shop Joker Slots (`G.GAME.shop.joker_max`)

- Default: 2
- Modified by: Overstock voucher (+1 each, max 2 upgrades)
- `change_shop_size(mod)` handles dynamic expansion/contraction

### Buffer System (joker_buffer / consumeable_buffer)

During gameplay (not shop), asynchronous card creation uses buffers to prevent
over-filling slots:

```lua
-- Check: count + buffer < limit
if #G.jokers.cards + G.GAME.joker_buffer < G.jokers.config.card_limit then
    G.GAME.joker_buffer = G.GAME.joker_buffer + 1
    -- Queue event to create card...
    -- In event callback: G.GAME.joker_buffer = 0
end
```

Used by: Riff-raff, Cartomancer (joker_buffer); 8 Ball, Vagabond, Superposition,
Seance, Hallucination, Purple Seal, Cerulean Bell effects (consumeable_buffer).

---

## Source File Reference

| Function | File | Line |
|----------|------|------|
| `Game:update_shop` | `game.lua` | 3072 |
| `G.UIDEF.shop` | `UI_definitions.lua` | 637 |
| `create_card_for_shop` | `UI_definitions.lua` | 742 |
| `create_shop_card_ui` | `UI_definitions.lua` | 802 |
| `get_current_pool` | `common_events.lua` | 1963 |
| `create_card` | `common_events.lua` | 2082 |
| `poll_edition` | `common_events.lua` | 2055 |
| `get_pack` | `common_events.lua` | 1944 |
| `get_next_voucher_key` | `common_events.lua` | 1901 |
| `change_shop_size` | `common_events.lua` | 1097 |
| `calculate_reroll_cost` | `common_events.lua` | 2263 |
| `Card:set_cost` | `card.lua` | 369 |
| `Card:sell_card` | `card.lua` | 1590 |
| `Card:can_sell_card` | `card.lua` | 1640 |
| `Card:redeem` | `card.lua` | 1813 |
| `Card:apply_to_run` | `card.lua` | 1880 |
| `G.FUNCS.buy_from_shop` | `button_callbacks.lua` | 2404 |
| `G.FUNCS.reroll_shop` | `button_callbacks.lua` | 2855 |
| `G.FUNCS.can_buy` | `button_callbacks.lua` | 55 |
| `G.FUNCS.can_redeem` | `button_callbacks.lua` | 96 |
| `G.FUNCS.check_for_buy_space` | `button_callbacks.lua` | 2392 |

# Decks & Stakes

> **Reference documentation** — produced during initial Balatro v1.0.1o source analysis.
> The jackdaw engine is the authoritative implementation; see `jackdaw/engine/` for current behavior.

Starting conditions, modifiers, and card pool mutations for all 15 decks and
all 8 stake levels.

---

## The 15 Decks

Each deck is a `Back` prototype in `G.P_CENTERS` with a `config` table.
Effects are applied via `Back:apply_to_run()` (`back.lua:174`) at run start,
and optionally via `Back:trigger_effect()` (`back.lua:108`) during gameplay.

### Deck Summary Table

| # | Deck | Key | Starting Cards | Rule Changes | Starting Items |
|---|------|-----|----------------|-------------|----------------|
| 1 | Red Deck | `b_red` | Standard 52 | +1 discard | — |
| 2 | Blue Deck | `b_blue` | Standard 52 | +1 hand | — |
| 3 | Yellow Deck | `b_yellow` | Standard 52 | +$10 starting money | — |
| 4 | Green Deck | `b_green` | Standard 52 | $2/unused hand, $1/unused discard, no interest | — |
| 5 | Black Deck | `b_black` | Standard 52 | -1 hand, +1 joker slot | — |
| 6 | Magic Deck | `b_magic` | Standard 52 | — | Crystal Ball voucher, 2× The Fool |
| 7 | Nebula Deck | `b_nebula` | Standard 52 | -1 consumable slot | Telescope voucher |
| 8 | Ghost Deck | `b_ghost` | Standard 52 | 2× spectral rate in shop | 1× Hex spectral |
| 9 | Abandoned Deck | `b_abandoned` | **40 cards** (no J/Q/K) | — | — |
| 10 | Checkered Deck | `b_checkered` | **52 cards, 2 suits** (Spades+Hearts only) | — | — |
| 11 | Zodiac Deck | `b_zodiac` | Standard 52 | — | Tarot Merchant + Planet Merchant + Overstock vouchers |
| 12 | Painted Deck | `b_painted` | Standard 52 | +2 hand size, -1 joker slot | — |
| 13 | Anaglyph Deck | `b_anaglyph` | Standard 52 | Double Tag after each boss blind | — |
| 14 | Plasma Deck | `b_plasma` | Standard 52 | 2× ante scaling, chips/mult balanced at scoring | — |
| 15 | Erratic Deck | `b_erratic` | **52 cards, random suits+ranks** | — | — |

---

## Detailed Deck Configs

### Red Deck (`b_red`)
```lua
config = {discards = 1}
```
- `starting_params.discards += 1` → 4 discards (normally 3)
- Unlocked by default

### Blue Deck (`b_blue`)
```lua
config = {hands = 1}
```
- `starting_params.hands += 1` → 5 hands (normally 4)
- Unlock: discover 20 items

### Yellow Deck (`b_yellow`)
```lua
config = {dollars = 10}
```
- `starting_params.dollars += 10` → $14 (normally $4)
- Unlock: discover 50 items

### Green Deck (`b_green`)
```lua
config = {extra_hand_bonus = 2, extra_discard_bonus = 1, no_interest = true}
```
- `G.GAME.modifiers.money_per_hand = 2` — earn $2 per unused hand at round end
- `G.GAME.modifiers.money_per_discard = 1` — earn $1 per unused discard
- `G.GAME.modifiers.no_interest = true` — no interest on savings
- Unlock: discover 75 items

### Black Deck (`b_black`)
```lua
config = {hands = -1, joker_slot = 1}
```
- `starting_params.hands -= 1` → 3 hands
- `starting_params.joker_slots += 1` → 6 joker slots
- Unlock: discover 100 items

### Magic Deck (`b_magic`)
```lua
config = {voucher = 'v_crystal_ball', consumables = {'c_fool', 'c_fool'}}
```
- Activates Crystal Ball voucher (+1 consumable slot, enables spectral shop pool)
- Starts with 2× The Fool tarot in consumable slots
- Unlock: win with Red Deck

### Nebula Deck (`b_nebula`)
```lua
config = {voucher = 'v_telescope', consumable_slot = -1}
```
- Activates Telescope voucher (first Celestial pack card matches most-played hand)
- `starting_params.consumable_slots -= 1` → 1 consumable slot
- Unlock: win with Blue Deck

### Ghost Deck (`b_ghost`)
```lua
config = {spectral_rate = 2, consumables = {'c_hex'}}
```
- `G.GAME.spectral_rate = 2` — spectral cards appear in shop at 2× rate
- Starts with 1× Hex spectral in consumable slots
- Unlock: win with Yellow Deck

### Abandoned Deck (`b_abandoned`)
```lua
config = {remove_faces = true}
```
- `starting_params.no_faces = true`
- During deck building, cards with rank J, Q, or K are **removed**
- Result: **40 cards** (2-10 + A in each of 4 suits)
- Unlock: win with Green Deck

### Checkered Deck (`b_checkered`)
```lua
config = {}
```
- Handled by a hardcoded name check in `Back:apply_to_run()` (`back.lua:239`)
- After the standard 52 cards are created, all Clubs → Spades, all Diamonds → Hearts
- Result: **52 cards in only 2 suits** (26 Spades + 26 Hearts, two of each rank per suit)
- Unlock: win with Black Deck

### Zodiac Deck (`b_zodiac`)
```lua
config = {vouchers = {'v_tarot_merchant', 'v_planet_merchant', 'v_overstock_norm'}}
```
- Activates 3 vouchers at run start:
  - Tarot Merchant (tarot rate ×2.4 in shop)
  - Planet Merchant (planet rate ×2.4 in shop)
  - Overstock (+1 shop joker slot)
- Unlock: win at stake 2

### Painted Deck (`b_painted`)
```lua
config = {hand_size = 2, joker_slot = -1}
```
- `starting_params.hand_size += 2` → 10 hand size
- `starting_params.joker_slots -= 1` → 4 joker slots
- Standard 52-card deck, no card pool changes
- Unlock: win at stake 3

### Anaglyph Deck (`b_anaglyph`)
```lua
config = {}
```
- No run-start changes to starting params
- **In-game effect** (`trigger_effect`, context `'eval'`): after defeating a boss
  blind, creates a Double Tag
- Unlock: win at stake 4

### Plasma Deck (`b_plasma`)
```lua
config = {ante_scaling = 2}
```
- `starting_params.ante_scaling = 2` — blind chip requirements scale 2× faster
- **In-game effect** (`trigger_effect`, context `'final_scoring_step'`): during
  scoring, chips and mult are **averaged**:
  ```lua
  total = chips + mult
  chips = floor(total / 2)
  mult  = floor(total / 2)
  ```
  This fires as the last step before `floor(chips × mult)`.
- Also responds to `'blind_amount'` context (returns early, preventing the display
  from showing the ante_scaling adjustment — cosmetic only)
- Unlock: win at stake 5

### Erratic Deck (`b_erratic`)
```lua
config = {randomize_rank_suit = true}
```
- `starting_params.erratic_suits_and_ranks = true`
- During deck building, each card's key is replaced with a random `G.P_CARDS` key
  via `pseudorandom_element(G.P_CARDS, pseudoseed('erratic'))`
- Result: **52 cards with random suits and ranks** (duplicates possible — you might
  get three 7 of Hearts and no Ace of Spades)
- Unlock: win at stake 7

---

## Deck Building Flow

How the starting deck is constructed in `Game:start_run()` (`game.lua:2328-2375`):

```
1. Check for challenge deck with explicit card list
   → If challenge provides deck.cards: use that list directly
   → Otherwise: iterate G.P_CARDS (52 entries)

2. For each card in G.P_CARDS:
   a. Erratic Deck: replace key with random P_CARDS key
   b. Parse suit letter and rank letter
   c. Challenge filtering: yes_ranks, no_ranks, yes_suits, no_suits
   d. Challenge enhancements: apply global enhancement/edition/seal
   e. Abandoned Deck: skip if rank is J, Q, or K (no_faces)
   f. If keeping: add to card_protos list

3. Append any extra_cards from starting params

4. Sort card_protos deterministically (by s+r+e+d+g concatenation)

5. For each proto: card_from_control(proto)
   → Creates Card with P_CARDS[suit_rank] base + P_CENTERS[enhancement] center
   → Applies edition and seal if specified
   → Emplaces into G.deck, adds to G.playing_cards

6. Post-creation: Back:apply_to_run() fires via event queue
   → Checkered Deck: change Clubs→Spades, Diamonds→Hearts
   → Edition decks: apply editions to random cards
   → Starting vouchers, consumables, etc.
```

### `card_from_control(control)` (`misc_functions.lua:1625`)

| Control field | Type | Meaning |
|---|---|---|
| `s` | string | Suit letter: H, C, D, S |
| `r` | string | Rank letter: 2-9, T, J, Q, K, A |
| `e` | string/nil | Enhancement center key (e.g. `'m_glass'`). Defaults to `'c_base'`. |
| `d` | string/nil | Edition key (e.g. `'foil'`, `'holo'`, `'polychrome'`) |
| `g` | string/nil | Seal (e.g. `'Gold'`, `'Red'`, `'Blue'`, `'Purple'`) |

---

## The 8 Stakes

Stakes are cumulative — each includes all effects from lower stakes.

### Stake Modifier Table

| Stake | Name | New Effect | Category |
|-------|------|------------|----------|
| 1 | White Stake | (base game, no modifiers) | — |
| 2 | Red Stake | Small Blind gives **no reward money** | Economy |
| 3 | Green Stake | Blind chips use **scaling level 2** (harder) | Engine |
| 4 | Black Stake | **Eternal** jokers can appear in shop/packs (30%) | Shop |
| 5 | Blue Stake | **-1 discard** per round | Engine |
| 6 | Purple Stake | Blind chips use **scaling level 3** (hardest) | Engine |
| 7 | Orange Stake | **Perishable** jokers can appear in shop/packs (30%) | Shop |
| 8 | Gold Stake | **Rental** jokers can appear in shop/packs (30%) | Shop |

### Detailed Effects

**Stake 2 (Red) — No Small Blind reward:**
```lua
G.GAME.modifiers.no_blind_reward = {Small = true}
```
In `Blind:set_blind()` (`blind.lua:84`): if `no_blind_reward[self:get_type()]`,
`self.dollars = 0`. The player gets $0 for beating the Small Blind.

**Stake 3 (Green) — Scaling level 2:**
```lua
G.GAME.modifiers.scaling = 2
```
Selects a harder chip requirement table in `get_blind_amount()`.

**Stake 4 (Black) — Eternal jokers:**
```lua
G.GAME.modifiers.enable_eternals_in_shop = true
```
In `create_card()` (`common_events.lua:2137`), when creating jokers for
shop/packs: single roll, if > 0.7 → Eternal (30% chance).
Eternal jokers **cannot be sold or destroyed**.

**Stake 5 (Blue) — Fewer discards:**
```lua
G.GAME.starting_params.discards = G.GAME.starting_params.discards - 1
```
Normally 3, becomes 2. Stacks with deck modifiers (Red Deck at stake 5 = 3 discards).

**Stake 6 (Purple) — Scaling level 3:**
```lua
G.GAME.modifiers.scaling = 3
```
Overrides stake 3's scaling = 2. Hardest blind chip scaling.

**Stake 7 (Orange) — Perishable jokers:**
```lua
G.GAME.modifiers.enable_perishables_in_shop = true
```
Same roll as Eternal: if 0.4 < roll ≤ 0.7 → Perishable (30% chance).
Eternal and Perishable are **mutually exclusive** (same roll).
Perishable jokers become debuffed after 5 rounds.

**Stake 8 (Gold) — Rental jokers:**
```lua
G.GAME.modifiers.enable_rentals_in_shop = true
```
**Separate roll** from Eternal/Perishable: if > 0.7 → Rental (30% chance).
A joker **can be both** Eternal+Rental or Perishable+Rental.
Rental jokers cost $3 per round (deducted at end of round).

### Eternal/Perishable/Rental Roll Logic

```
Roll 1 (eternal_perishable_poll):
  > 0.7 AND enable_eternals   → ETERNAL  (30%)
  > 0.4 AND enable_perishables → PERISHABLE (30%, mutually exclusive with eternal)
  ≤ 0.4                        → neither  (40%)

Roll 2 (rental_poll, independent):
  > 0.7 AND enable_rentals    → RENTAL   (30%)
  ≤ 0.7                       → not rental (70%)
```

---

## Blind Chip Scaling

### `get_blind_amount(ante)` (`misc_functions.lua:919`)

Returns the **base** chip requirement. Actual requirement = base × blind_mult × ante_scaling.

| Blind | Multiplier |
|-------|-----------|
| Small Blind | ×1 |
| Big Blind | ×1.5 |
| Boss Blind | ×2 |

### Base Amounts by Ante and Scaling Level

| Ante | Level 1 (White/Red) | Level 2 (Green–Blue) | Level 3 (Purple–Gold) |
|------|---------------------|----------------------|-----------------------|
| 1 | 300 | 300 | 300 |
| 2 | 800 | 900 | 1,000 |
| 3 | 2,000 | 2,600 | 3,200 |
| 4 | 5,000 | 8,000 | 9,000 |
| 5 | 11,000 | 20,000 | 25,000 |
| 6 | 20,000 | 36,000 | 60,000 |
| 7 | 35,000 | 60,000 | 110,000 |
| 8 | 50,000 | 100,000 | 200,000 |

**Antes 9+:** Exponential formula: `floor(base_8 × (1.6 + (0.75 × (ante-8))^(1 + 0.2×(ante-8)))^(ante-8))`
with significant-figure rounding. All three scaling levels use the same formula
but different `base_8` values.

**Plasma Deck modifier:** `ante_scaling = 2` means the result is multiplied by 2.
So ante 1 boss = 300 × 2 × 2 = 1,200 chips (vs 600 normally).

---

## Deck Unlock Progression

| Deck | Unlock Condition |
|------|-----------------|
| Red Deck | Unlocked by default |
| Blue Deck | Discover 20 items |
| Yellow Deck | Discover 50 items |
| Green Deck | Discover 75 items |
| Black Deck | Discover 100 items |
| Magic Deck | Win with Red Deck |
| Nebula Deck | Win with Blue Deck |
| Ghost Deck | Win with Yellow Deck |
| Abandoned Deck | Win with Green Deck |
| Checkered Deck | Win with Black Deck |
| Zodiac Deck | Win at stake ≥ 2 |
| Painted Deck | Win at stake ≥ 3 |
| Anaglyph Deck | Win at stake ≥ 4 |
| Plasma Deck | Win at stake ≥ 5 |
| Erratic Deck | Win at stake ≥ 7 |

---

## Source File Reference

| Topic | File | Lines |
|-------|------|-------|
| Deck prototypes | `game.lua` | 628-642 (in `init_item_prototypes`) |
| Stake prototypes | `game.lua` | 253-260 |
| Stake modifier application | `game.lua` | 2049-2059 (in `start_run`) |
| Deck building loop | `game.lua` | 2328-2375 (in `start_run`) |
| `Back:apply_to_run` | `back.lua` | 174-278 |
| `Back:trigger_effect` | `back.lua` | 108-172 |
| `card_from_control` | `misc_functions.lua` | 1625-1632 |
| `get_blind_amount` | `misc_functions.lua` | 919-954 |
| `create_card` modifiers | `common_events.lua` | 2133-2152 |
| Challenge definitions | `challenges.lua` | 1-738 |

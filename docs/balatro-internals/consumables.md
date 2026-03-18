# Consumables & Vouchers

> **Reference documentation** — produced during initial Balatro v1.0.1o source analysis.
> The jackdaw engine is the authoritative implementation; see `jackdaw/engine/` for current behavior.

Every Tarot, Planet, Spectral, and Voucher effect. Pack opening mechanics.
Selection rules. RNG involvement.

---

## Tarot Cards

All tarots are used via `Card:use_consumeable()` (`card.lua:1091`).
Selection rules are in `Card:can_use_consumeable()` (`card.lua:1523`).

### Enhancement Tarots

These require the player to highlight cards in hand. Usable during
SELECTING_HAND or inside Tarot/Spectral/Planet pack states.

| Card | Key | Select | Effect | RNG |
|------|-----|--------|--------|-----|
| The Magician | `c_magician` | 1-2 cards | Enhance → **Lucky Card** | No |
| The Empress | `c_empress` | 1-2 cards | Enhance → **Mult Card** | No |
| The Hierophant | `c_heirophant` | 1-2 cards | Enhance → **Bonus Card** | No |
| The Lovers | `c_lovers` | 1 card | Enhance → **Wild Card** | No |
| The Chariot | `c_chariot` | 1 card | Enhance → **Steel Card** | No |
| Justice | `c_justice` | 1 card | Enhance → **Glass Card** | No |
| The Devil | `c_devil` | 1 card | Enhance → **Gold Card** | No |
| The Tower | `c_tower` | 1 card | Enhance → **Stone Card** | No |

### Suit-Change Tarots

| Card | Key | Select | Effect | RNG |
|------|-----|--------|--------|-----|
| The Star | `c_star` | 1-3 cards | Change suit → **Diamonds** | No |
| The Moon | `c_moon` | 1-3 cards | Change suit → **Clubs** | No |
| The Sun | `c_sun` | 1-3 cards | Change suit → **Hearts** | No |
| The World | `c_world` | 1-3 cards | Change suit → **Spades** | No |

### Transformation Tarots

| Card | Key | Select | Effect | RNG |
|------|-----|--------|--------|-----|
| Strength | `c_strength` | 1-2 cards | **Increase rank by 1** (K→A→2→3...) | No |
| Death | `c_death` | Exactly 2 | **Copy rightmost onto leftmost** (suit, rank, enhancement, edition, seal) | No |
| The Hanged Man | `c_hanged_man` | 1-2 cards | **Destroy** highlighted cards | No |

### Generation Tarots

| Card | Key | Select | Effect | RNG |
|------|-----|--------|--------|-----|
| The Fool | `c_fool` | 0 cards | Create copy of **last Tarot/Planet used** | No (deterministic copy) |
| The High Priestess | `c_high_priestess` | 0 cards | Create **2 random Planet cards** | Yes — seed `'pri'` |
| The Emperor | `c_emperor` | 0 cards | Create **2 random Tarot cards** | Yes — seed `'emp'` |
| Judgement | `c_judgement` | 0 cards | Create **1 random Joker** | Yes — seed `'jud'` |

### Economy Tarots

| Card | Key | Select | Effect | RNG |
|------|-----|--------|--------|-----|
| The Hermit | `c_hermit` | 0 cards | Gain `min(current_dollars, $20)` | No |
| Temperance | `c_temperance` | 0 cards | Gain $ = total sell value of all jokers (cap $50) | No |

### Special Tarots

| Card | Key | Select | Effect | RNG |
|------|-----|--------|--------|-----|
| The Wheel of Fortune | `c_wheel_of_fortune` | 0 cards | **1 in 4** chance: add random edition to a random editionless joker. Fails silently otherwise. | Yes — seed `'wheel_of_fortune'`, edition via `poll_edition` |

---

## Planet Cards

All planets level up a specific poker hand by 1. No card selection needed.
No RNG. Usable in any valid state.

Leveling formula (`level_up_hand`, `common_events.lua:464`):
```
level += 1   (or +amount if specified)
chips = s_chips + l_chips × (level - 1)
mult  = s_mult  + l_mult  × (level - 1)
```

| Card | Key | Hand Type | Base Chips | Base Mult | +Chips/lvl | +Mult/lvl |
|------|-----|-----------|-----------|----------|-----------|----------|
| Pluto | `c_pluto` | High Card | 5 | 1 | +10 | +1 |
| Mercury | `c_mercury` | Pair | 10 | 2 | +15 | +1 |
| Uranus | `c_uranus` | Two Pair | 20 | 2 | +20 | +1 |
| Venus | `c_venus` | Three of a Kind | 30 | 3 | +20 | +2 |
| Saturn | `c_saturn` | Straight | 30 | 4 | +30 | +3 |
| Jupiter | `c_jupiter` | Flush | 35 | 4 | +15 | +2 |
| Earth | `c_earth` | Full House | 40 | 4 | +25 | +2 |
| Mars | `c_mars` | Four of a Kind | 60 | 7 | +30 | +3 |
| Neptune | `c_neptune` | Straight Flush | 100 | 8 | +40 | +4 |
| Planet X | `c_planet_x` | Five of a Kind | 120 | 12 | +35 | +3 |
| Ceres | `c_ceres` | Flush House | 140 | 14 | +40 | +4 |
| Eris | `c_eris` | Flush Five | 160 | 16 | +50 | +3 |

**Example**: Pair at level 5 = 10 + 15×4 = **70 chips**, 2 + 1×4 = **6 mult**.

### Hidden Planet

| Card | Key | Effect | RNG |
|------|-----|--------|-----|
| Black Hole | `c_black_hole` | Level up **ALL 12 hands** by 1 | No |

Black Hole only appears via the 0.3% soul chance roll in `create_card`.

---

## Spectral Cards

Spectrals have powerful but often destructive effects. Most require cards in
hand but not highlighted. Usable during SELECTING_HAND or inside pack states.

### Seal Spectrals

| Card | Key | Select | Effect | RNG |
|------|-----|--------|--------|-----|
| Talisman | `c_talisman` | 1 card | Add **Gold Seal** | No |
| Deja Vu | `c_deja_vu` | 1 card | Add **Red Seal** | No |
| Trance | `c_trance` | 1 card | Add **Blue Seal** | No |
| Medium | `c_medium` | 1 card | Add **Purple Seal** | No |

### Destroy-and-Create Spectrals

| Card | Key | Select | Destroys | Creates | RNG |
|------|-----|--------|----------|---------|-----|
| Familiar | `c_familiar` | 0 (need >1 in hand) | 1 random card | 3 enhanced face cards (J/Q/K) | Yes — seeds `'random_destroy'`, `'familiar_create'`, `'spe_card'` |
| Grim | `c_grim` | 0 (need >1 in hand) | 1 random card | 2 enhanced Aces | Yes — seeds `'random_destroy'`, `'grim_create'`, `'spe_card'` |
| Incantation | `c_incantation` | 0 (need >1 in hand) | 1 random card | 4 enhanced number cards (2-10) | Yes — seeds `'random_destroy'`, `'incantation_create'`, `'spe_card'` |
| Immolate | `c_immolate` | 0 (need >1 in hand) | 5 random cards | — (gives **$20**) | Yes — seed `'immolate'` |

Created cards get random non-Stone enhancements and random suits.

### Edition Spectrals

| Card | Key | Select | Effect | RNG |
|------|-----|--------|--------|-----|
| Aura | `c_aura` | 1 card (no existing edition) | Add random edition (Foil/Holo/Poly) to playing card | Yes — `poll_edition('aura', nil, true, true)` guaranteed mode, no Negative |
| Ectoplasm | `c_ectoplasm` | 0 cards | Add **Negative** to random editionless joker. **-1 hand size** (escalating) | Yes — seed `'ectoplasm'` |
| Hex | `c_hex` | 0 cards | Add **Polychrome** to random editionless joker. **Destroy all other non-eternal jokers** | Yes — seed `'hex'` |

### Joker Spectrals

| Card | Key | Select | Effect | RNG |
|------|-----|--------|--------|-----|
| Wraith | `c_wraith` | 0 cards | Create **1 Rare Joker**. **Set money to $0** | Yes — rarity forced to 0.99 (Rare), seed `'wra'` |
| Ankh | `c_ankh` | 0 cards | Pick random joker, **destroy all others** (non-eternal), **duplicate** the chosen one | Yes — seed `'ankh_choice'` |
| The Soul | `c_soul` | 0 cards | Create **1 Legendary Joker** | Yes — seed `'sou'` |

### Deck Mutation Spectrals

| Card | Key | Select | Effect | RNG |
|------|-----|--------|--------|-----|
| Sigil | `c_sigil` | 0 (need >1 in hand) | Change **all hand cards** to 1 random suit | Yes — seed `'sigil'` |
| Ouija | `c_ouija` | 0 (need >1 in hand) | Change **all hand cards** to 1 random rank. **-1 hand size** | Yes — seed `'ouija'` |
| Cryptid | `c_cryptid` | 1 card | Create **2 exact copies** of highlighted card, add to deck | No (deterministic copy) |

---

## Vouchers

All 32 vouchers (16 base + 16 upgrades). Each costs $10. Upgrades require
their base voucher to be purchased first. Effects applied via `Card:apply_to_run()`
(`card.lua:1880`) or checked inline by other systems.

### Shop Modifiers

| Base | Effect | Upgrade | Additional Effect |
|------|--------|---------|-------------------|
| Overstock | +1 shop joker slot | Overstock Plus | +1 more shop joker slot |
| Clearance Sale | 25% discount on all shop cards | Liquidation | 50% discount |
| Tarot Merchant | Tarot rate ×2.4 in shop | Tarot Tycoon | Tarot rate ×8 |
| Planet Merchant | Planet rate ×2.4 in shop | Planet Tycoon | Planet rate ×8 |
| Magic Trick | Playing cards appear in shop | Illusion | Playing cards get editions/enhancements/seals |

### Economy

| Base | Effect | Upgrade | Additional Effect |
|------|--------|---------|-------------------|
| Seed Money | Interest cap raised to $50 | Money Tree | Interest cap raised to $100 |

### Reroll

| Base | Effect | Upgrade | Additional Effect |
|------|--------|---------|-------------------|
| Reroll Surplus | Reroll cost -$2 | Reroll Glut | Reroll cost -$2 more |
| Director's Cut | Can reroll boss blind (once, $10) | Retcon | Can reroll boss unlimited ($10 each) |

### Hands & Discards

| Base | Effect | Upgrade | Additional Effect |
|------|--------|---------|-------------------|
| Grabber | +1 hand per round | Nacho Tong | +1 more hand per round |
| Wasteful | +1 discard per round | Recyclomancy | +1 more discard per round |
| Paint Brush | +1 hand size | Palette | +1 more hand size |
| Hieroglyph | -1 ante, -1 hand per round | Petroglyph | -1 ante, -1 discard per round |

### Slots

| Base | Effect | Upgrade | Additional Effect |
|------|--------|---------|-------------------|
| Crystal Ball | +1 consumable slot | Omen Globe | Spectral cards can appear in Arcana packs (20% per card) |
| Blank | (no gameplay effect, tracks for unlock) | Antimatter | +1 joker slot |

### Editions

| Base | Effect | Upgrade | Additional Effect |
|------|--------|---------|-------------------|
| Hone | Edition rate ×2 | Glow Up | Edition rate ×4 (total) |

### Pack Modifiers

| Base | Effect | Upgrade | Additional Effect |
|------|--------|---------|-------------------|
| Telescope | First Celestial pack card = most-played hand's planet | Observatory | Planet cards in consumable slots give x1.5 mult when their hand type scores |

---

## Booster Packs

### Pack Types and Sizes

Packs are defined in `G.P_CENTER_POOLS['Booster']` with `kind`, `extra` (cards
shown), and `config.choose` (cards the player can take).

| Pack Type | Variants | Cards Shown | Cards to Take | Contents |
|-----------|----------|-------------|---------------|----------|
| **Arcana** (Tarot) | Normal (2), Jumbo, Mega | 3 / 5 / 5 | 1 / 1 / 2 | Tarot cards (20% Spectral with Omen Globe) |
| **Celestial** (Planet) | Normal (2), Jumbo, Mega | 3 / 5 / 5 | 1 / 1 / 2 | Planet cards (first = most-played hand's planet with Telescope) |
| **Spectral** | Normal only | 2 | 1 | Spectral cards |
| **Standard** | Normal (4), Jumbo (2), Mega (2) | 3 / 5 / 5 | 1 / 1 / 2 | Playing cards with enhancements/editions/seals |
| **Buffoon** (Joker) | Normal (2), Jumbo, Mega | 2 / 4 / 4 | 1 / 1 / 2 | Joker cards |

### Pack Opening Flow (`Card:open`, `card.lua:1681`)

1. Set `G.GAME.pack_size = self.ability.extra` (cards to show)
2. Set `G.GAME.pack_choices = self.config.center.config.choose or 1`
3. Set `G.STATE` to pack-specific state, `G.GAME.PACK_INTERRUPT = SHOP`
4. Card generation happens in the `update_*_pack` functions
5. Jokers notified: `calculate_joker({open_booster = true})`

### Pack Card Generation

**Arcana packs** (`game.lua:3341`, `update_arcana_pack`):
- Hand is dealt (cards drawn from deck) so player can target them with tarots
- Cards: `create_card('Tarot', G.pack_cards, nil, nil, true, true, nil, 'ar'..i)`
- With Omen Globe: each card has 20% chance of being Spectral instead
  (`pseudorandom('omen_globe') > 0.8`)

**Celestial packs** (`game.lua:3528`, `update_celestial_pack`):
- Hand is NOT dealt
- Cards: `create_card('Planet', G.pack_cards, nil, nil, true, true, nil, 'pl'..i)`
- With Telescope: first card (i==1) is forced to the planet for the player's
  most-played hand type

**Spectral packs** (`game.lua:3392`, `update_spectral_pack`):
- Hand is dealt
- Cards: `create_card('Spectral', G.pack_cards, nil, nil, true, true, nil, 'spe'..i)`

**Standard packs** (`game.lua:3443`, `update_standard_pack`):
- Hand is NOT dealt
- Each card: 40% Enhanced, 60% Base (`pseudorandom('stdset'..ante) > 0.6`)
- Edition: `poll_edition('standard_edition'..i, edition_rate*2)` (doubled rate)
- Seal: 20% chance (`pseudorandom('stdseal'..ante) > 0.8`), uniform among
  Red/Blue/Gold/Purple
- Playing card front: random from `G.P_CARDS`

**Buffoon packs** (`game.lua:3492`, `update_buffoon_pack`):
- Hand is NOT dealt
- Cards: `create_card('Joker', G.pack_cards, nil, nil, true, true, nil, 'buf'..i)`

### Pack Closing (`G.FUNCS.end_consumeable`, `button_callbacks.lua:2565`)

When the player has taken their allowed choices (or clicks "Skip"):
1. Remaining unchosen cards are removed
2. `G.pack_cards` CardArea is cleaned up
3. Hand cards returned to deck via `draw_from_hand_to_deck()`
4. State restored to `G.GAME.PACK_INTERRUPT` (typically SHOP)
5. Shop UI slides back in

### Which Packs Deal the Hand?

| Pack Type | Hand Dealt? | Reason |
|-----------|-------------|--------|
| Arcana | Yes | Tarots target hand cards |
| Spectral | Yes | Spectrals target hand cards |
| Celestial | No | Planets don't need card selection |
| Standard | No | Cards are added to deck, not selected from hand |
| Buffoon | No | Jokers don't need card selection |

---

## Selection Rule Summary

### Global Blockers (all consumables)

- Cards in play area (`#G.play.cards > 0`) → blocked
- Controller locked → blocked
- `G.GAME.STOP_USE > 0` → blocked
- State is HAND_PLAYED, DRAW_TO_HAND, or PLAY_TAROT → blocked

### Per-Card Rules

| Requirement | Cards |
|-------------|-------|
| **0 cards, always usable** | Planets, Black Hole, The Hermit, Temperance |
| **0 cards, slot available** | The Fool (consumable slot + last tarot exists), Emperor/High Priestess (consumable slot), Judgement/Wraith/Soul (joker slot) |
| **0 cards, eligible joker** | Wheel of Fortune (editionless joker), Ectoplasm (editionless joker), Hex (editionless joker), Ankh (any joker + limit > 1) |
| **0 cards, >1 in hand** | Familiar, Grim, Incantation, Immolate, Sigil, Ouija |
| **1 card highlighted** | Lovers, Chariot, Justice, Devil, Tower, Talisman, Deja Vu, Trance, Medium, Cryptid |
| **1 card, no edition** | Aura |
| **1-2 cards highlighted** | Magician, Empress, Hierophant, Strength, Hanged Man |
| **Exactly 2 cards** | Death |
| **1-3 cards highlighted** | Star, Moon, Sun, World |

### State Requirements for Card-Targeting Consumables

Tarots/Spectrals that modify hand cards require one of:
- `SELECTING_HAND`
- `TAROT_PACK`
- `SPECTRAL_PACK`
- `PLANET_PACK`

This means they work during normal gameplay AND inside pack openings (e.g.,
using a Tarot received from an Arcana pack on your dealt hand).

---

## RNG Summary

### Deterministic (no RNG)

All enhancement tarots, suit-change tarots, Strength, Death, Hanged Man,
The Fool (copies known card), The Hermit, Temperance, all Planets, Black Hole,
all seal spectrals, Cryptid.

### RNG-Dependent

| Card | Seed Key(s) | What's Random |
|------|-------------|---------------|
| High Priestess | `'pri'` | Which 2 planets |
| Emperor | `'emp'` | Which 2 tarots |
| Judgement | `'jud'` | Which joker |
| Wheel of Fortune | `'wheel_of_fortune'` | Success (1/4), which joker, which edition |
| Familiar | `'random_destroy'`, `'familiar_create'`, `'spe_card'` | Which card destroyed, rank/suit/enhancement of created cards |
| Grim | `'random_destroy'`, `'grim_create'`, `'spe_card'` | Same pattern |
| Incantation | `'random_destroy'`, `'incantation_create'`, `'spe_card'` | Same pattern |
| Immolate | `'immolate'` | Which 5 cards destroyed |
| Sigil | `'sigil'` | Which suit |
| Ouija | `'ouija'` | Which rank |
| Aura | `'aura'` via `poll_edition` | Which edition (Foil/Holo/Poly) |
| Ectoplasm | `'ectoplasm'` | Which joker |
| Hex | `'hex'` | Which joker |
| Wraith | `'wra'` | Which rare joker |
| Ankh | `'ankh_choice'` | Which joker survives |
| The Soul | `'sou'` | Which legendary |

---

## Source File Reference

| Function | File | Line |
|----------|------|------|
| `Card:use_consumeable` | `card.lua` | 1091 |
| `Card:can_use_consumeable` | `card.lua` | 1523 |
| `Card:open` | `card.lua` | 1681 |
| `Card:redeem` | `card.lua` | 1813 |
| `Card:apply_to_run` | `card.lua` | 1880 |
| `level_up_hand` | `common_events.lua` | 464 |
| `G.FUNCS.end_consumeable` | `button_callbacks.lua` | 2565 |
| `update_arcana_pack` | `game.lua` | 3341 |
| `update_spectral_pack` | `game.lua` | 3392 |
| `update_standard_pack` | `game.lua` | 3443 |
| `update_buffoon_pack` | `game.lua` | 3492 |
| `update_celestial_pack` | `game.lua` | 3528 |

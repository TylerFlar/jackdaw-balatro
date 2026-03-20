# Observation Encoding Audit

Audit of `jackdaw/env/observation.py` comparing what the agent sees vs what a
human player sees. Conducted against the full `game_state` dict from the engine
and the [balatrobot GameState schema](https://coder.github.io/balatrobot/api/#gamestate-schema).

## Encoded Features (211 global + variable entities)

### Global Context (D_GLOBAL = 211)

| Offset | Width | Feature | Source |
|--------|-------|---------|--------|
| 0-5 | 6 | Phase one-hot | `gs["phase"]` |
| 6-9 | 4 | blind_on_deck one-hot (None/Small/Big/Boss) | `gs["blind_on_deck"]` |
| 10 | 1 | Ante (normalized /8) | `gs["round_resets"]["ante"]` |
| 11 | 1 | Round (normalized /30) | `gs["round"]` |
| 12 | 1 | Dollars (log-scaled) | `gs["dollars"]` |
| 13 | 1 | Hands left (normalized /10) | `gs["current_round"]["hands_left"]` |
| 14 | 1 | Discards left (normalized /10) | `gs["current_round"]["discards_left"]` |
| 15 | 1 | Hand size (normalized /15) | `gs["hand_size"]` |
| 16 | 1 | Joker slots (normalized /10) | `gs["joker_slots"]` |
| 17 | 1 | Consumable slots (normalized /5) | `gs["consumable_slots"]` |
| 18 | 1 | Blind chips required (log-scaled) | `gs["blind"].chips` |
| 19 | 1 | Chips scored (log-scaled) | `gs["chips"]` |
| 20 | 1 | Score fraction (clamped /10) | `chips / blind_chips` |
| 21 | 1 | Reroll cost (normalized /10) | `gs["current_round"]["reroll_cost"]` |
| 22 | 1 | Free rerolls (normalized /5) | `gs["current_round"]["free_rerolls"]` |
| 23 | 1 | Interest cap (normalized /100) | `gs["interest_cap"]` |
| 24 | 1 | Discount percent (normalized /50) | `gs["discount_percent"]` |
| 25 | 1 | Skips count (normalized /10) | `gs["skips"]` |
| 26 | 1 | Boss blind key ID (normalized) | `gs["blind"].key` |
| 27 | 1 | Deck cards remaining (log-scaled) | `len(gs["deck"])` |
| 28 | 1 | Discard pile size (log-scaled) | `len(gs["discard_pile"])` |
| 29 | 1 | Meta flags packed | `four_fingers/shortcut/smeared/splash` |
| 30-89 | 60 | Hand levels: 12 types × 5 | `gs["hand_levels"]` |
| 90-121 | 32 | Vouchers owned (binary) | `gs["used_vouchers"]` |
| 122-129 | 8 | Blind effect features | `gs["blind"]` object |
| 130-132 | 3 | Round position one-hot (small/big/boss) | `gs["round_resets"]["blind_states"]` |
| 133-134 | 2 | Round progress (hands_played, discards_used) | `gs["current_round"]` |
| 135-158 | 24 | Awarded tags (binary) | `gs["awarded_tags"]` |
| 159-210 | 52 | Discard pile suit×rank histogram | `gs["discard_pile"]` cards |

### Blind Effect Features (offset 122, 8 dims)

| Dim | Feature | Normalization |
|-----|---------|---------------|
| 0 | Is boss blind | 0/1 |
| 1 | Effect disabled (Chicot/Luchador) | 0/1 |
| 2 | Blind multiplier | /4.0 (small=1, big=1.5, boss=2+) |
| 3-6 | Debuff suit one-hot (Hearts/Diamonds/Clubs/Spades) | 0/1 |
| 7 | Debuff targets face cards | 0/1 |

### Entity Types

| Type | Dims | Count | Key features |
|------|------|-------|--------------|
| Playing card | 14 | len(hand) | rank, suit, enhancement, edition, seal, debuff, scoring, position |
| Joker | 15 | len(jokers) | center_key, ability stats, edition, eternal/perishable/rental |
| Consumable | 7 | len(consumables) | center_key, can_use, targeting info |
| Shop item | 9 | len(shop_*) | center_key, cost, affordable, has_slot |
| Pack card | 14 | len(pack_cards) | Same encoding as playing cards |

## Remaining Gaps (LOW priority)

These features are not encoded but are unlikely to significantly impact
learning. They can be added incrementally if needed.

| Gap | Importance | Notes |
|-----|-----------|-------|
| Joker trigger state | LOW | Per-joker counters partially captured by `ability_extra` |
| Targeting cards (idol/mail/ancient/castle) | LOW | Only matters with specific jokers |
| Shop/pool rates | LOW | Agent sees actual shop contents |
| Run statistics | LOW | Mostly retrospective |
| Economy edge cases (money_per_hand, inflation) | LOW | Derivable from vouchers |

## Comparison with balatrobot GameState

The [balatrobot API](https://coder.github.io/balatrobot/api/#gamestate-schema)
exposes these fields, all of which are now covered:

| balatrobot field | Our encoding |
|------------------|-------------|
| `state` | Phase one-hot [0:6] |
| `round_num`, `ante_num` | Scalars [10:11] |
| `money` | Scalar [12] |
| `round.hands_left/played` | Scalars [13], [133] |
| `round.discards_left/used` | Scalars [14], [134] |
| `round.reroll_cost` | Scalar [21] |
| `round.chips` | Scalar [19] |
| `blinds[].type/status/score/effect` | blind_on_deck [6:9], blind_effect [122:129], round_pos [130:132] |
| `blinds[].tag_name/tag_effect` | Tags binary [135:158] |
| `hands{}` (levels) | Hand levels [30:89] |
| `jokers.cards[]` | Joker entities (15 dims each) |
| `consumables.cards[]` | Consumable entities (7 dims each) |
| `hand.cards[]` | Playing card entities (14 dims each) |
| `shop.cards[]` | Shop entities (9 dims each) |
| `cards` (deck) | Deck size [27], discard histogram [159:210] |
| `vouchers` / `used_vouchers` | Vouchers binary [90:121] |
| `pack.cards[]` | Pack card entities (14 dims each) |

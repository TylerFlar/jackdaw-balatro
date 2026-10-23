# Canonical snapshot schema (`jackdaw.snapshot/1`)

A snapshot is the complete, engine-agnostic state of a run. One format serves
three purposes:

- **Parity.** The Python oracle and the Rust engine both export snapshots, and
  equal game states must give byte-identical canonical encodings (see
  [Canonical encoding](#canonical-encoding)).
- **Golden v2.** `tests/fixtures/golden_v2.json` stores digests of snapshots
  along the recorded tapes, so it doesn't depend on either engine's internals.
- **`Game.to_dict()` / `Game.from_dict()`**, which are also what BalatroBot's
  `save`/`load` write and read in the sim.

A snapshot holds the **semantic** state: everything that can change what
happens next, and nothing else. It leaves out:

- per-step scratch values;
- write-only statistics;
- values the engine derives from other fields, such as a card's name from its
  center;
- aliases (one object reachable from two places).

`tests/parity/test_snapshot_roundtrip.py` proves completeness against the
Python oracle. At every step of every tape, and along seeded fuzz runs, it
rebuilds a state from the snapshot, applies the same action to both states, and
requires identical snapshots afterwards.

Field names follow Lua `G.GAME` where one exists. Values are JSON-like: `null`,
booleans, numbers, strings, arrays and string-keyed objects.

## Top level

| Key | Type | Meaning |
|---|---|---|
| `schema` | string | `"jackdaw.snapshot/1"` |
| `phase` | string | `blind_select`, `selecting_hand`, `round_eval`, `shop`, `pack_opening`, `game_over` |
| `run` | object | Run identity, below |
| `game` | object | Scalar `G.GAME` fields, below |
| `starting_params` | object | `G.GAME.starting_params`, as set at run start (dollars, hand_size, discards, hands, reroll_cost, joker_slots, consumable_slots, ante_scaling, no_faces, erratic_suits_and_ranks) |
| `modifiers` | object | `G.GAME.modifiers`: stake, challenge and back rules (e.g. `scaling`, `enable_eternals_in_shop`, `no_blind_reward`, `discard_cost`) |
| `round_resets` | object | `G.GAME.round_resets` |
| `current_round` | object | `G.GAME.current_round` |
| `round_bonus` | object | `{next_hands, discards}` |
| `blind` | object \| null | The current blind, below |
| `hands` | object | Poker hand name → hand state (all 12 hands) |
| `tags` | array | **Pending** tags in order: `{key}` (tags that have not triggered yet; `G.GAME.tags`) |
| `sets` | object | `used_jokers`, `used_vouchers`, `banned_keys` (sorted key arrays), `pool_flags` (object), `bosses_used` (boss key → count) |
| `consumeable_usage` | object | `by_key` (center key → times used) and `totals` (`tarot`, `planet`, `spectral`, `tarot_planet`, `all`) |
| `areas` | object | Area name → array of cards, in area order |
| `pack` | object \| null | The open booster pack (only in `pack_opening`) |
| `rng` | object | `{seed, hashed_seed, streams}` |
| `sim` | object | Simulator state that has no direct `G.GAME` field |

### `run`

`seed` (string), `back` (center key, e.g. `"b_red"` or `"b_challenge"`), `stake`
(1–8), `challenge` (challenge id or null), `win_ante` (normally 8), `won`
(boolean).

### `game`

**Economy:** `dollars`, `chips` (chips scored this round), `interest_cap`,
`interest_amount`, `discount_percent`, `inflation`, `rental_rate`, `bankrupt_at`,
`base_reroll_cost`.

**Rates:** `edition_rate`, `joker_rate`, `tarot_rate`, `planet_rate`,
`spectral_rate`, `playing_card_rate`, `probabilities_normal`
(`G.GAME.probabilities.normal`).

**Area limits:** `hand_size`, `joker_slots`, `consumable_slots`, plus
`shop_joker_max` (`G.GAME.shop.joker_max`).

**Progress:** `round`, `skips`, `hands_played` (whole run), `unused_discards`,
`starting_deck_size`, `blind_on_deck` (`Small`/`Big`/`Boss`/null).

**Memory:** `last_tarot_planet` (center key or null), `last_hand_played`
(hand name or null).

**Misc:** `pack_size`, `ecto_minus`, `perishable_rounds`, `first_shop_buffoon`
(whether the first-shop Buffoon pack guarantee has been used).

### `round_resets`

`hands`, `discards`, `reroll_cost`, `temp_reroll_cost` (null or number),
`temp_handsize` (null or number), `ante`, `blind_ante`, `boss_rerolled`.

Three maps, each keyed by `Small`/`Big`/`Boss`:
- `blind_states`: `Select`, `Current`, `Upcoming`, `Skipped` or `Defeated`.
- `blind_choices`: the blind key for each slot.
- `blind_tags`: the tag key for `Small`/`Big` (absent before the first ante is
  rolled).

### `current_round`

- **Counters:** `hands_left`, `hands_played`, `discards_left`, `discards_used`,
  `reroll_cost`, `reroll_cost_increase`, `free_rerolls`, `jokers_purchased`.
- `voucher`: the shop voucher key this ante.
- `most_played_poker_hand`.
- **Round targets:** `idol_card` `{suit, rank, id}`, `mail_card` `{rank, id}`,
  `ancient_card` `{suit}`, `castle_card` `{suit}`.

### `blind`

Shared fields:
- `key` (blind key; `""` for the empty blind)
- `chips` (target score)
- `mult`
- `dollars` (reward)
- `boss`, `disabled`, `triggered`

Boss-specific state:
- `prepped`: The Fish / Crimson Heart one-shot flag.
- `eye_hands`: The Eye's hands already played this round, sorted.
- `mouth_hand`: The Mouth's locked hand, or null.
- `discards_sub` / `hands_sub`: what The Water / The Needle removed.

### `hands`

Keyed by the poker hand name (`"Flush Five"` … `"High Card"`):
`{level, chips, mult, played, played_this_round, visible}`.

### `areas`

Keys: `deck` (index 0 is the bottom; cards are drawn from the end), `hand`
(display order), `discard`, `play` (empty between steps), `jokers`,
`consumables`, `shop_jokers`, `shop_vouchers`, `shop_booster`, `pack_cards`.

### Card

| Key | Meaning |
|---|---|
| `sort_id` | Per-run creation counter (`Card.sort_id`); the card's identity. Sorting and RNG picks use it |
| `center` | `P_CENTERS` key (`c_base`, `m_glass`, `j_joker`, `c_fool`, `v_hone`, `p_arcana_normal_1`, …) |
| `front` | `P_CARDS` key for playing cards (`H_A`), else null |
| `base` | Playing cards only: `{suit, value, id, nominal, suit_nominal, suit_nominal_original, face_nominal, original_value, times_played}`, else null. `suit_nominal_original` is the tiebreak of the suit the card was created with (kept across suit changes); `original_value` is the rank at creation and null once a suit or rank change has rebuilt the base (`Card:set_base`, card.lua:97-145) |
| `ability` | `self.ability`, **minus** the keys derived from the center: `name`, `effect`, `set`, `order`, `consumeable`, `type`. `extra` keeps its Lua shape: null, a number, a string, or an object |
| `edition` | The Lua edition table, e.g. `{"foil": true, "chips": 50, "type": "foil"}`, or null |
| `seal` | `Red`, `Blue`, `Gold`, `Purple`, or null |
| `debuff`, `facing` | Boolean; `front`/`back` |
| `playing_card` | `self.playing_card` index, or null |
| `costs` | `{base_cost, extra_cost, cost, sell_cost}` |
| `stickers` | `{eternal, perishable, perish_tally (null unless perishable), rental, pinned}` |

### `pack`

- `kind`: `Arcana`, `Celestial`, `Spectral`, `Standard`, `Buffoon`.
- `choices_remaining`.
- `dealt`: `sort_id`s of the cards an Arcana/Spectral pack dealt into the hand
  that are still held. They return to the deck when the pack closes.
- `return_phase`: the phase to return to when the pack closes.

### `rng`

- `seed`: the seed string.
- `hashed_seed`: `pseudohash(seed)`.
- `streams`: stream key → current `G.GAME.pseudorandom[key]` value.

Stream insertion order is not part of the state.

### `sim`

| Key | Meaning |
|---|---|
| `next_sort_id` | `sort_id` the next created card receives |
| `pending_tag_packs` | Booster keys queued by tags (Double Tag), opened when the current pack closes |
| `pending_deck_bottom` | Cards (Marble Joker stones) that join the bottom of the deck after the next round-start shuffle |
| `shop_d6ed` | D6 Tag already applied this shop |
| `last_blind_was_boss` | The last defeated blind was a boss (eval-time tags such as Investment) |
| `pending_cashout` | Dollars `cash_out` will pay; 0 outside `round_eval` |
| `last_score` | The last played hand: `{hand_type, chips, mult, total, dollars, debuffed}`, or null |
| `tag_history` | Keys of every tag awarded this run in order, minus consumed Double Tags. Observation v1 encodes this set |

## Canonical encoding

Defined in `tests/parity/canon.py`; the Rust implementation must match byte
for byte.

| Value | Encoding |
|---|---|
| `null` | `N` |
| `true` / `false` | `T` / `F` |
| Number | `D` + IEEE-754 double, big-endian |
| String | `S` + u32 big-endian byte length + UTF-8 bytes |
| Array | `[` + u32 count + items |
| Object | `{` + u32 count + (key, value) pairs, keys sorted by UTF-8 bytes; keys encoded as strings |

- **All numbers are doubles.** Integers and floats are the same Lua number.
- `-0.0` is encoded as `0.0`.
- Every NaN is encoded as `0x7FF8000000000000`.

The digest is the SHA-256 of the encoding. Trajectory digests chain records:
`chain(prev, item) = sha256(prev + ":" + item)`.

In JSON text, a non-finite number is written as `{"$f": "inf"|"-inf"|"nan"}`.

## Versioning

Adding, removing or renaming a field, or changing its meaning, bumps the schema
version. The exporter, the Rust snapshot code and this document change in the
same PR, and golden v2 is regenerated.

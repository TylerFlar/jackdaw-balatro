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
| `sets` | object | `used_jokers`, `used_vouchers`, `banned_keys` (sorted key arrays), `pool_flags` (object), `bosses_used` (boss key → count). `used_jokers` holds only Joker, Tarot, Planet, Spectral and Voucher keys: Lua also marks playing cards' `c_base` and enhancements, boosters and editions, but no rule reads those marks (only these pools and the Soul / Black Hole rolls read the set, common_events.lua:1987, 2090, 2096) |
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

**Hand sort:** `hand_sort` is `G.hand.config.sort`, the sort mode every
automatic hand sort uses: `"desc"` (by rank, the default) or `"suit desc"`.
The sort buttons set it (button_callbacks.lua:36-48) and it lasts for the
rest of the run.

**Progress:** `round`, `skips`, `hands_played` (whole run), `unused_discards`,
`starting_deck_size`, `blind_on_deck` (`Small`/`Big`/`Boss`/null).

**Memory:** `last_tarot_planet` (center key or null), `last_hand_played`
(hand name or null).

**Misc:** `pack_size`, `ecto_minus`, `perishable_rounds`, `first_shop_buffoon`
(whether the first-shop Buffoon pack guarantee has been used).

**Orbital Tag:** `orbital_choices` (`G.GAME.orbital_choices`): ante (as a
string key) → `{Small, Big, Boss}` → the poker hand an Orbital Tag for that
blind levels. Each ante's three hands are rolled when its first blind-select
screen is built (UI_definitions.lua:1506-1516); every ante seen so far keeps
its entry.

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
- `prepped`: The Fish / Crimson Heart one-shot flag. `set_blind` sets it (The
  Fish clears it), `press_play` sets it and every `drawn_to_hand` clears it.
- `eye_hands`: The Eye's hands already played this round, sorted (empty for
  any other blind).
- `mouth_hand`: The Mouth's locked hand, or null.
- `discards_sub` / `hands_sub`: what The Water / The Needle removed.

Before the first blind of a run and from each defeat on (`Blind:defeat` runs
`set_blind(nil)`, blind.lua:336), the blind is the **empty blind**: `key` `""`,
`chips`, `mult` and `dollars` 0, not a boss, not disabled, `prepped` true.

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
| `ability` | `self.ability`, **minus** the keys derived from the center (`name`, `effect`, `set`, `order`, `consumeable`, `type`) and the sticker fields, which live under `stickers`. `extra` keeps its Lua shape: null, a number, a string, or an object |
| `edition` | The Lua edition table, e.g. `{"foil": true, "chips": 50, "type": "foil"}`, or null |
| `seal` | `Red`, `Blue`, `Gold`, `Purple`, or null |
| `debuff`, `facing` | Boolean; `front`/`back`. Cards in `deck` and `discard` are always written `front`: Lua turns them face down every frame (`CardArea:align_cards`, cardarea.lua:410-430), so their facing carries nothing, and a card drawn from them faces whichever way the draw decides |
| `added_to_deck` | `Card.added_to_deck`: whether the card's `add_to_deck` effects (passives, a Negative's slot) are applied. True for owned jokers and consumables; false while a joker is debuffed (its passives are off), for shop and pack cards, and for playing cards (the sim never runs their no-op `add_to_deck`) |
| `costs` | `{base_cost, extra_cost, cost, sell_cost}` |

A card has no `playing_card` field. Lua numbers playing cards from the session-wide
`G.playing_card` counter, which is never reset (it keeps counting across runs), and renumbers
them on every removal (card.lua:4752-4760). No game rule reads the numbers, so the snapshot
leaves them out; whether a card is in `G.playing_cards` follows from its area.
| `stickers` | `{eternal, perishable, perish_tally (null unless perishable), rental, pinned}`. Lua keeps the first four in `self.ability` |

### `pack`

- `kind`: `Arcana`, `Celestial`, `Spectral`, `Standard`, `Buffoon`.
- `choices_remaining`.
- `dealt`: `sort_id`s of the cards an Arcana/Spectral pack dealt into the hand
  that are still held, ascending. They return to the deck when the pack closes.
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
| `pending_tag_packs` | Booster keys queued by tags, opened when the current pack closes. Always empty: a Double Tag's pack-tag copies stay in `tags` until a pack closes, since G106 |
| `pending_deck_bottom` | Cards that join the bottom of the deck after the next round-start shuffle. Always empty: Marble Joker's Stones, which used to wait here, are shuffled in since G55 |
| `shop_d6ed` | D6 Tag already applied this shop |
| `shop_free` | A Coupon Tag already applied this shop (`G.GAME.shop_free`, tag.lua:448); a second one waits for the next shop |
| `hand_last_poll_size` | Luxury Tax: `floor(dollars / minus_hand_size_per_X_dollar)` when the hand size was last adjusted (`G.hand.config.last_poll_size`, cardarea.lua:245); 0 in every other run |
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

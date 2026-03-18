# The G Table — Balatro's Global State

> **Reference documentation** — produced during initial Balatro v1.0.1o source analysis.
> The jackdaw engine is the authoritative implementation; see `jackdaw/engine/` for current behavior.

`G` is the single global `Game` instance. It holds **everything**: rendering state,
UI elements, prototype data, and the per-run game state (`G.GAME`). Understanding
the split between `G.*` (persistent across runs) and `G.GAME.*` (reset each run)
is critical for the simulator.
---

## Table of Contents

1. [G — Top-Level Fields](#g--top-level-fields)
2. [G.GAME — Per-Run State](#ggame--per-run-state)
3. [G.GAME.starting_params](#ggamestarting_params)
4. [G.GAME.round_resets](#ggameround_resets)
5. [G.GAME.current_round](#ggamecurrent_round)
6. [G.GAME.hands — Poker Hand Levels](#ggamehands--poker-hand-levels)
7. [G.GAME.modifiers](#ggamemodifiers)
8. [G.GAME.round_scores](#ggameround_scores)
9. [G.GAME.probabilities](#ggameprobabilities)
10. [G.GAME.pseudorandom](#ggamepseudorandom)
11. [G.GAME.pool_flags](#ggamepool_flags)
12. [Initialization Order](#initialization-order)

---

## G — Top-Level Fields

### Card Areas (created per-run in `start_run`)

| Field | Class | card_limit | type | Description |
|-------|-------|-----------|------|-------------|
| `G.hand` | CardArea | `starting_params.hand_size` (8) | `'hand'` | Player's current hand of cards |
| `G.deck` | CardArea | 52 | `'deck'` | Draw pile |
| `G.play` | CardArea | 5 | `'play'` | Cards currently played on the table |
| `G.discard` | CardArea | 500 | `'discard'` | Discard pile |
| `G.jokers` | CardArea | `starting_params.joker_slots` (5) | `'joker'` | Active joker slots |
| `G.consumeables` | CardArea | `starting_params.consumable_slots` (2) | `'joker'` | Tarot/Planet/Spectral slots |
| `G.shop_jokers` | CardArea | varies | `'shop'` | Shop joker offerings (created in UI_definitions) |
| `G.shop_vouchers` | CardArea | varies | `'voucher'` | Shop voucher offerings |
| `G.shop_booster` | CardArea | varies | `'shop'` | Booster pack offerings |
| `G.pack_cards` | CardArea | varies | `'title'` | Cards inside an opened booster pack |
| `G.title_top` | CardArea | — | `'title'` | Splash screen card display |

### Master Card List

| Field | Type | Description |
|-------|------|-------------|
| `G.playing_cards` | `Card[]` | Master array of all playing cards in the run. Created during deck building in `start_run`. Cards are added/removed as they are created/destroyed. Distinct from `G.deck.cards` — a card in `G.hand` is still in `G.playing_cards`. |

### Prototype / Data Tables (loaded once at startup, persist forever)

| Field | Type | Description |
|-------|------|-------------|
| `G.P_CARDS` | table | Playing card prototypes keyed by suit+rank (e.g. `"S_A"`, `"H_K"`). 52 entries. |
| `G.P_CENTERS` | table | Center prototypes for every card type — jokers (`j_*`), tarots (`c_*`), planets (`p_*`), spectrals, vouchers (`v_*`), backs (`b_*`), enhancements (`m_*`), editions, boosters, seals. Keyed by string ID. |
| `G.P_CENTER_POOLS` | table | Centers grouped by set (Joker, Tarot, Planet, Spectral, Voucher, Back, Stake, etc.). Each is an array of center prototypes. |
| `G.P_BLINDS` | table | Blind prototypes keyed by blind key (e.g. `"bl_small"`, `"bl_big"`, `"bl_hook"`). Contains name, mult, dollars, debuff config, boss flag, pos. |
| `G.P_TAGS` | table | Tag prototypes keyed by tag key (e.g. `"tag_double"`, `"tag_rare"`). Contains name, config, pos. |
| `G.P_JOKER_RARITY_POOLS` | table | Jokers grouped by rarity: `[1]`=Common, `[2]`=Uncommon, `[3]`=Rare, `[4]`=Legendary. |
| `G.P_SEALS` | table | Seal definitions. |
| `G.P_LOCKED` | table | Locked card display info. |

### Game State Machine

| Field | Type | Set in | Description |
|-------|------|--------|-------------|
| `G.STATES` | table | `globals.lua` | Enum: `SELECTING_HAND=1, HAND_PLAYED=2, DRAW_TO_HAND=3, GAME_OVER=4, SHOP=5, PLAY_TAROT=6, BLIND_SELECT=7, ROUND_EVAL=8, TAROT_PACK=9, PLANET_PACK=10, MENU=11, TUTORIAL=12, SPLASH=13, SANDBOX=14, SPECTRAL_PACK=15, DEMO_CTA=16, STANDARD_PACK=17, BUFFOON_PACK=18, NEW_ROUND=19` |
| `G.STAGES` | table | `globals.lua` | Enum: `MAIN_MENU=1, RUN=2, SANDBOX=3` |
| `G.STATE` | number | `globals.lua` | Current state. Changes constantly during gameplay. Starts as `STATES.SPLASH`. |
| `G.STAGE` | number | `globals.lua` | Current stage. `MAIN_MENU` until run starts, then `RUN`. |
| `G.STATE_COMPLETE` | boolean | `globals.lua` | Set `true` when current state's work is done, triggers transition to next state. |
| `G.TAROT_INTERRUPT` | number/nil | `card.lua` | Saves the interrupted `STATE` when a consumable is used mid-state. |

### Managers & Controllers (engine layer, not in extracted source)

| Field | Type | Description |
|-------|------|-------------|
| `G.E_MANAGER` | EventManager | Event queue. `add_event(Event({...}))` is the primary async mechanism. |
| `G.CONTROLLER` | Controller | Input handling — cursor position, HID flags, drag/focus/hover tracking, locks. |
| `G.SOUND_MANAGER` | SoundManager | Audio thread communication via `channel:push()`. |
| `G.STEAM` | luasteam/nil | Steam API integration. `nil` on non-PC platforms. |

### UI Elements

| Field | Type | Description |
|-------|------|-------------|
| `G.HUD` | UIBox | Main HUD — chips, mult, hand name, dollars, ante, round. |
| `G.HUD_blind` | UIBox | Blind info panel — name, chip target, reward. |
| `G.HUD_tags` | UIBox[] | Array of tag indicator UIBoxes attached to `G.ROOM_ATTACH`. |
| `G.buttons` | UIBox | Play/Discard/Reroll action buttons. |
| `G.OVERLAY_MENU` | UIBox/nil | Modal overlay (settings, run info, collection, etc.). |
| `G.OVERLAY_TUTORIAL` | table/nil | Tutorial state with Jimbo reference and step tracking. |
| `G.ROOM` | UIBox | Root scene container. `G.ROOM.T` is the room transform. |
| `G.ROOM_ATTACH` | UIBox | Attachment point for positioned UI elements. |
| `G.ROOM_ORIG` | table | Original room position `{x, y, r}`. |

### Rendering & Dimensions

| Field | Type | Value | Description |
|-------|------|-------|-------------|
| `G.TILESIZE` | number | 20 | Base tile size in pixels |
| `G.TILESCALE` | number | 3.65 | Scale multiplier |
| `G.TILE_W` | number | 20 | Room width in tile units |
| `G.TILE_H` | number | 11.5 | Room height in tile units |
| `G.CARD_W` | number | ~2.049 | Card width (`2.4*35/41`) |
| `G.CARD_H` | number | ~2.751 | Card height (`2.4*47/41`) |
| `G.HIGHLIGHT_H` | number | ~0.550 | Highlight offset (`0.2*CARD_H`) |
| `G.CANVAS` | Canvas | — | Main render target |
| `G.CANV_SCALE` | number | 1 | Canvas DPI scale |
| `G.WINDOWTRANS` | table | — | `{x, y, w, h, real_window_w, real_window_h}` |

### Timing

| Field | Type | Description |
|-------|------|-------------|
| `G.TIMERS` | table | `{TOTAL, REAL, REAL_SHADER, UPTIME, BACKGROUND}` — frame timers |
| `G.FRAMES` | table | `{DRAW, MOVE}` — frame counters |
| `G.SEED` | number | `os.time()` at startup — initial math.randomseed value |
| `G.FPS_CAP` | number | 500 (default) |

### Colors (`G.C`)

Comprehensive color palette. Key simulator-relevant entries:

| Field | Value | Used for |
|-------|-------|----------|
| `G.C.MULT` | `#FE5F55` | Multiplier display |
| `G.C.CHIPS` | `#009DFF` | Chips display |
| `G.C.MONEY` | `#F3B958` | Dollar display |
| `G.C.SUITS.Hearts` | `#FE5F55` | Suit color |
| `G.C.SUITS.Diamonds` | `#FE5F55` | Suit color |
| `G.C.SUITS.Spades` | `#374649` | Suit color |
| `G.C.SUITS.Clubs` | `#424E54` | Suit color |
| `G.C.RARITY[1..4]` | blue/green/red/purple | Common/Uncommon/Rare/Legendary |

### Settings (`G.SETTINGS`)

Persistent across runs. Key fields:

| Field | Type | Description |
|-------|------|-------------|
| `G.SETTINGS.GAMESPEED` | number | 1 (normal). Affects animation delays only, not game logic. |
| `G.SETTINGS.language` | string | `'en-us'` |
| `G.SETTINGS.profile` | number | Active profile slot (1-3) |
| `G.SETTINGS.paused` | boolean | Pause state |
| `G.SETTINGS.screenshake` | boolean | Screenshake toggle |
| `G.SETTINGS.SOUND` | table | `{volume, music_volume, game_sounds_volume}` |
| `G.SETTINGS.WINDOW` | table | Display/resolution settings |
| `G.SETTINGS.GRAPHICS` | table | `{texture_scaling, shadows, crt, bloom}` |
| `G.SETTINGS.ACHIEVEMENTS_EARNED` | table | Keyed by achievement name |
| `G.SETTINGS.run_stake_stickers` | boolean | Show stake stickers on cards |

### Other Top-Level Fields

| Field | Type | Description |
|-------|------|-------------|
| `G.VERSION` | string | `'1.0.1o-FULL'` |
| `G.FUNCS` | table | Global function registry. Button callbacks, UI update functions. |
| `G.ARGS` | table | Shared argument scratch space between functions. |
| `G.CHALLENGES` | table | Challenge definitions (from `challenges.lua`). |
| `G.PROFILES` | table | Array of 3 profile data slots. |
| `G.METRICS` | table | Career statistics `{cards, decks, bosses}`. |
| `G.I` | table | Instance tracking: `{NODE, MOVEABLE, SPRITE, UIBOX, POPUP, CARD, CARDAREA, ALERT}` |
| `G.ANIMATION_ATLAS` | table | Spritesheet atlas references |
| `G.ASSET_ATLAS` | table | Asset atlas references |
| `G.handlist` | string[] | Ordered hand names: `{"Flush Five", ..., "High Card"}` |
| `G.sort_id` | number | Auto-incrementing card sort ID |
| `G.tagid` | number | Auto-incrementing tag ID |
| `G.VIEWING_DECK` | bool/nil | True when deck-view overlay is open |
| `G.boss_throw_hand` | bool/nil | Set when boss blind would debuff the highlighted hand |
| `G.PITCH_MOD` | number | Audio pitch modifier (1 normal, 0.5 game over) |

### Feature Flags (`G.F_*`)

Set per-platform in `globals.lua`. Simulator-relevant ones:

| Flag | Default | Description |
|------|---------|-------------|
| `G.F_NO_SAVING` | false | Disables run save/load |
| `G.F_SAVE_TIMER` | 5 (PC) | Auto-save interval in seconds |

---

## G.GAME — Per-Run State

Created by `Game:init_game_object()` at the start of each run. This is the
**entire mutable run state** — everything needed to save/load a run.

### Core Scalars

| Field | Type | Init | Description |
|-------|------|------|-------------|
| `chips` | number | 0 | Chips scored so far this blind. Reset each blind. |
| `chips_text` | string | `'0'` | Display string for chips. |
| `dollars` | number | `starting_params.dollars` (4) | Player money. Modified by `ease_dollars()`. |
| `bankrupt_at` | number | 0 | Minimum dollar threshold (negative = debt allowed). |
| `round` | number | 0 | Current round number (increments each round regardless of ante). |
| `stake` | number | 1 | Difficulty level (1–8). Set at run start. |
| `win_ante` | number | 8 | Ante required to win. |
| `won` | boolean | false | Set true by `win_game()`. |
| `challenge` | string/nil | nil | Challenge ID if running a challenge. |
| `seeded` | boolean | false | True if run uses a fixed seed. |
| `sort` | string | `'desc'` | Current hand sort mode. |
| `inflation` | number | 0 | Cumulative price inflation from Inflation challenge. |
| `discount_percent` | number | 0 | Shop discount from Clearance Sale voucher. |
| `interest_cap` | number | 25 | Max interest per round (raised by Money Tree voucher to 50). |
| `interest_amount` | number | 1 | Dollars of interest per $5 held. |
| `max_jokers` | number | 0 | Highest joker count achieved (for unlocks). |
| `hands_played` | number | 0 | Total hands played in run (for Handy tag). |
| `unused_discards` | number | 0 | Total unused discards in run (for Garbage tag). |
| `skips` | number | 0 | Total blinds skipped (for Skip tag). |
| `current_boss_streak` | number | 0 | Consecutive boss defeats (for unlocks). |
| `pack_size` | number | 2 | Default booster pack card count. |
| `starting_deck_size` | number | 52 | Cards in deck at run start. |
| `ecto_minus` | number | 1 | Ectoplasm joker slot penalty accumulator. |
| `perishable_rounds` | number | 5 | Rounds until perishable cards expire. |
| `base_reroll_cost` | number | 5 | Base shop reroll cost. |
| `STOP_USE` | number | 0 | Consumable use lock counter (prevents concurrent use). |
| `blind_on_deck` | string/nil | nil | `'Small'`, `'Big'`, or `'Boss'` — which blind is next. |
| `facing_blind` | boolean/nil | nil | True during blind scoring phase. |
| `starting_voucher_count` | number | 0 | Vouchers from deck/challenge at run start. |
| `tag_tally` | number | 0 | Auto-incrementing tag counter. |
| `last_tarot_planet` | string/nil | nil | Key of last Tarot/Planet used (for Perkeo). |

### Rate Fields (pool generation weights)

| Field | Type | Init | Description |
|-------|------|------|-------------|
| `edition_rate` | number | 1 | Edition appearance rate. Modified by Hone/Glow Up vouchers. |
| `joker_rate` | number | 20 | Relative weight for jokers in shop. 0 = no shop jokers. |
| `tarot_rate` | number | 4 | Tarot weight. Modified by Tarot Merchant voucher. |
| `planet_rate` | number | 4 | Planet weight. Modified by Planet Merchant voucher. |
| `spectral_rate` | number | 0 | Spectral rate. Set by Ghost Deck. |
| `playing_card_rate` | number | 0 | Playing card weight in shop. Modified by Magic Trick voucher. |
| `rental_rate` | number | 3 | Rental card appearance rate in shop (stake 8+). |

### Buffer Fields (prevent duplicate creation)

| Field | Type | Description |
|-------|------|-------------|
| `joker_buffer` | number | Pending joker additions. Incremented before create, decremented after. Prevents exceeding slot limit during async events. |
| `consumeable_buffer` | number | Same for consumables. |
| `dollar_buffer` | number | Pending dollar additions (for display timing). |

### Tables

| Field | Type | Description |
|-------|------|-------------|
| `tags` | Tag[] | Active tags for this ante. |
| `used_vouchers` | table | `{[voucher_key] = true}` for purchased vouchers. Persists across rounds. |
| `used_jokers` | table | `{[joker_key] = true}` for jokers that have appeared in the run. |
| `banned_keys` | table | `{[card_key] = true}` for cards excluded by challenge restrictions. |
| `bosses_used` | table | `{[blind_key] = count}` tracking boss blind appearances. |
| `selected_back` | Back | The deck (Back object) for this run. |
| `blind` | Blind | The active Blind object (or empty blind between rounds). |
| `shop` | table | `{joker_max = 2}` — max joker slots in shop. |
| `previous_round` | table | `{dollars = N}` — money at end of previous round. |
| `cards_played` | table | Per-rank play tracking. `{['Ace'] = {suits={}, total=0}, ...}` |
| `joker_usage` | table | Per-joker usage stats. |
| `consumeable_usage` | table | Per-consumable usage stats. |
| `consumeable_usage_total` | table | `{tarot, planet, spectral, tarot_planet, all}` — cumulative counts. |
| `hand_usage` | table | Per-hand-type usage stats. |
| `round_bonus` | table | `{next_hands=0, discards=0}` — bonuses to add next round. |
| `last_blind` | table | `{boss=bool, name=string}` — info about previous blind. |
| `orbital_choices` | table | `{[ante] = {[blind_type] = hand_name}}` — Orbital Tag hand picks. |
| `challenge_tab` | table/nil | Full challenge definition if in a challenge run. |

---

## G.GAME.starting_params

Returned by `get_starting_params()` in `misc_functions.lua:1868`. Modified by
stake effects, `Back:apply_to_run()`, and challenge modifiers **before** being
copied into `round_resets` and CardArea limits.

| Field | Type | Default | Modified by | Feeds into |
|-------|------|---------|-------------|------------|
| `dollars` | number | 4 | Back (Yellow +10), challenge | `G.GAME.dollars` |
| `hand_size` | number | 8 | Back (Painted +2, Black −1), challenge | `G.hand.config.card_limit` |
| `hands` | number | 4 | Back (Blue +1, Black −1), challenge | `round_resets.hands` |
| `discards` | number | 3 | Back (Red +1), stake 5+ (−1), challenge | `round_resets.discards` |
| `reroll_cost` | number | 5 | Back (reroll_discount), challenge | `round_resets.reroll_cost`, `base_reroll_cost` |
| `joker_slots` | number | 5 | Back (Black +1, Painted −1), challenge | `G.jokers.config.card_limit` |
| `consumable_slots` | number | 2 | Back (consumable_slot), challenge | `G.consumeables.config.card_limit` |
| `ante_scaling` | number | 1 | Back (Plasma = 3), challenge | `G.GAME.starting_params.ante_scaling` (read by blind chip calc) |
| `no_faces` | boolean | false | Back (Abandoned Deck) | Filters J/Q/K during deck building |
| `erratic_suits_and_ranks` | boolean | false | Back (Erratic Deck) | Randomizes suits/ranks during deck building |

---

## G.GAME.round_resets

Reset values for each ante. Copied from `starting_params` at run start, then
modified by vouchers/cards during play. At the start of each round,
`current_round.hands_left` and `current_round.discards_left` are set from these.

| Field | Type | Init | Description |
|-------|------|------|-------------|
| `ante` | number | 1 | Current ante number. Incremented by `ease_ante()`. |
| `blind_ante` | number | 1 | Ante for voucher unlock tracking. |
| `hands` | number | `starting_params.hands` | Hands per round. Modified by Grabber/Nacho Tong vouchers. |
| `discards` | number | `starting_params.discards` | Discards per round. Modified by vouchers. |
| `reroll_cost` | number | `base_reroll_cost` | Shop reroll base cost. |
| `temp_reroll_cost` | number/nil | nil | Temporary reroll cost override (D6 tag sets to 0). Cleared each round. |
| `temp_handsize` | number/nil | nil | Temporary hand size bonus (Juggle tag). Cleared each round. |
| `blind_states` | table | `{Small='Select', Big='Upcoming', Boss='Upcoming'}` | Current state of each blind slot: `'Select'`, `'Current'`, `'Defeated'`, `'Upcoming'`, `'Skipped'`. |
| `loc_blind_states` | table | `{Small='', Big='', Boss=''}` | Localized display strings for blind states. |
| `blind_choices` | table | `{Small='bl_small', Big='bl_big', Boss=<key>}` | Blind prototype keys for this ante. Boss is from `get_new_boss()`. |
| `blind_tags` | table | `{Small=<tag_key>, Big=<tag_key>}` | Tag keys attached to Small and Big blinds. |
| `boss_rerolled` | boolean | false | Whether boss was rerolled this ante (Boss Tag). |

---

## G.GAME.current_round

Per-round state. Reset at the start of each round from `round_resets`.

| Field | Type | Init | Description |
|-------|------|------|-------------|
| `hands_left` | number | `round_resets.hands` | Remaining hand plays. Decremented by `ease_hands_played()`. |
| `hands_played` | number | 0 | Hands played this round. Read by The House, DNA joker, Acrobat, etc. |
| `discards_left` | number | `round_resets.discards` | Remaining discards. Decremented by `ease_discard()`. |
| `discards_used` | number | 0 | Discards used this round. Read by Burnt Joker, Castle, Faceless Joker. |
| `dollars` | number | 0 | Dollars earned this round (for round eval display). |
| `round_dollars` | number | 0 | Running dollar total for round eval. |
| `reroll_cost` | number | `base_reroll_cost` | Current reroll cost (increases each reroll). |
| `reroll_cost_increase` | number | 0 | Cumulative reroll cost increase this shop visit. |
| `free_rerolls` | number | 0 | Free rerolls remaining (Chaos tag). |
| `jokers_purchased` | number | 0 | Jokers bought this round. |
| `dollars_to_be_earned` | string | `'!!!'` | Display string for blind reward (e.g. `'$$$'`). |
| `most_played_poker_hand` | string | `'High Card'` | Most-played hand type this round. Used by The Ox boss blind. |
| `cards_flipped` | number | 0 | Cards flipped face-down (for challenge modifier tracking). |
| `round_text` | string | `'Round '` | Display prefix (becomes `'Endless Round '` after winning). |
| `used_packs` | table | `{}` | Track packs opened this round. |
| `voucher` | string/nil | from `get_next_voucher_key()` | Voucher available in shop after defeating boss. |

### current_round.current_hand (live hand display)

| Field | Type | Description |
|-------|------|-------------|
| `chips` | number | Base chips from hand type + level. Updated by `update_hand_text()`. |
| `mult` | number | Base mult from hand type + level. Updated by `update_hand_text()`. |
| `chip_total` | number | `chips × mult` after all joker effects. |
| `handname` | string | Display name of current hand (e.g. `"Full House"`). |
| `hand_level` | string/number | Level indicator for display. |

### current_round — Joker targeting cards

These are set by `reset_idol_card()`, `reset_mail_rank()`, `reset_ancient_card()`,
`reset_castle_card()` at the start of each round.

| Field | Type | Description |
|-------|------|-------------|
| `idol_card` | table | `{suit='Spades', rank='Ace'}` — target for Idol joker |
| `mail_card` | table | `{rank='Ace'}` — target for Mail-In Rebate joker |
| `ancient_card` | table | `{suit='Spades'}` — target for Ancient Joker |
| `castle_card` | table | `{suit='Spades'}` — target for Castle joker |

---

## G.GAME.hands — Poker Hand Levels

Table keyed by hand name string. Initialized in `init_game_object()`.
Modified by Planet cards (`level_up_hand()`) and The Arm boss blind.

| Hand Name | Order | Base Chips | Base Mult | +Chips/lvl | +Mult/lvl | Visible |
|-----------|-------|-----------|----------|-----------|----------|---------|
| Flush Five | 1 | 160 | 16 | 50 | 3 | false (secret) |
| Flush House | 2 | 140 | 14 | 40 | 4 | false (secret) |
| Five of a Kind | 3 | 120 | 12 | 35 | 3 | false (secret) |
| Straight Flush | 4 | 100 | 8 | 40 | 4 | true |
| Four of a Kind | 5 | 60 | 7 | 30 | 3 | true |
| Full House | 6 | 40 | 4 | 25 | 2 | true |
| Flush | 7 | 35 | 4 | 15 | 2 | true |
| Straight | 8 | 30 | 4 | 30 | 3 | true |
| Three of a Kind | 9 | 30 | 3 | 20 | 2 | true |
| Two Pair | 10 | 20 | 2 | 20 | 1 | true |
| Pair | 11 | 10 | 2 | 15 | 1 | true |
| High Card | 12 | 5 | 1 | 10 | 1 | true |

### Per-hand fields:

| Field | Type | Description |
|-------|------|-------------|
| `chips` | number | Current chip value (`s_chips + (level-1) * l_chips`) |
| `mult` | number | Current mult value (`s_mult + (level-1) * l_mult`) |
| `s_chips` | number | Starting chips (never changes) |
| `s_mult` | number | Starting mult (never changes) |
| `l_chips` | number | Chips gained per level-up |
| `l_mult` | number | Mult gained per level-up |
| `level` | number | Current level (starts 1) |
| `order` | number | Display sort order |
| `visible` | boolean | Whether shown in hand info UI |
| `played` | number | Total times played in run |
| `played_this_round` | number | Times played this round (reset each round) |
| `example` | table | Example cards for UI display |

---

## G.GAME.modifiers

Run-wide modifiers set by challenges and stake. Empty table `{}` by default.

### From stake level:

| Field | Set when | Value | Effect |
|-------|----------|-------|--------|
| `no_blind_reward` | stake >= 2 | `{Small=true}` | No money reward for Small Blind |
| `scaling` | stake >= 3 | 2 (stake 3-5), 3 (stake 6+) | Blind chip scaling factor |
| `enable_eternals_in_shop` | stake >= 4 | true | Eternal jokers appear in shop |
| `enable_perishables_in_shop` | stake >= 7 | true | Perishable jokers appear in shop |
| `enable_rentals_in_shop` | stake >= 8 | true | Rental jokers appear in shop |

### From challenges:

| Field | Type | Challenge | Effect |
|-------|------|-----------|--------|
| `no_blind_reward` | table | The Omelette, Cruelty | `{Small=true, Big=true, Boss=true}` |
| `no_interest` | boolean | The Omelette, Mad World | Disables end-of-round interest |
| `no_extra_hand_money` | boolean | The Omelette, Mad World | No bonus for unused hands |
| `chips_dollar_cap` | boolean | Rich get Richer | Chip earnings capped by dollars |
| `no_shop_jokers` | boolean | Bram Poker, Jokerless | `joker_rate` set to 0 |
| `flipped_cards` | number | X-ray Vision | 1 in N cards drawn face-down |
| `minus_hand_size_per_X_dollar` | number | Luxury Tax | Lose 1 hand size per X dollars held |
| `all_eternal` | boolean | Non-Perishable | All jokers are eternal |
| `debuff_played_cards` | boolean | Double or Nothing | Played cards get debuffed |
| `inflation` | boolean | Inflation | Shop prices increase each round |
| `set_eternal_ante` | number | Typecast | Ante where all jokers become eternal |
| `set_joker_slots_ante` | number | Typecast | Ante where joker slots lock |
| `discard_cost` | number | Golden Needle | Dollar cost per discard |
| `no_reward_specific` | — | Cruelty | Blocks reward for specific blind types |
| `money_per_hand` | number | Green Deck | Dollars per unused hand |
| `money_per_discard` | number | Green Deck | Dollars per unused discard |

---

## G.GAME.round_scores

Cumulative run statistics displayed in the run info screen. Each entry is
`{label=string, amt=number}`.

| Key | Label | Tracks |
|-----|-------|--------|
| `furthest_ante` | "Ante" | Highest ante reached |
| `furthest_round` | "Round" | Highest round reached |
| `hand` | "Best Hand" | Highest single-hand chip score |
| `poker_hand` | "Most Played Hand" | Most-played hand type |
| `new_collection` | "New Discoveries" | Cards discovered this run |
| `cards_played` | "Cards Played" | Total cards played |
| `cards_discarded` | "Cards Discarded" | Total cards discarded |
| `times_rerolled` | "Times Rerolled" | Total shop rerolls |
| `cards_purchased` | "Cards Purchased" | Total cards bought |

---

## G.GAME.probabilities

| Field | Type | Init | Modified by | Description |
|-------|------|------|-------------|-------------|
| `normal` | number | 1 | Observatory voucher, Oops! All 6s joker | Base probability numerator. Most chance effects use `probabilities.normal/N`. Oops! doubles it. |

---

## G.GAME.pseudorandom

The deterministic RNG system. Seeds are per-context to ensure reproducibility.

| Field | Type | Description |
|-------|------|-------------|
| `seed` | string | Run seed string (8 chars, or `'TUTORIAL'`). |
| `hashed_seed` | number | `pseudohash(seed)` — numeric hash for RNG. |
| `<context_key>` | number | Per-context seeds. Keys like `'soul'`, `'stdset'`, `'edition'`, `'joker'`, `'shop'`, `'wheel'`, etc. Each is `pseudohash(key..seed)`. Consumed and regenerated by `pseudoseed()`. |

---

## G.GAME.pool_flags

Dynamic flags that affect card pool composition.

| Field | Type | Description |
|-------|------|-------------|
| `gros_michel_extinct` | boolean | Set when Gros Michel is destroyed. Removes it from joker pool and enables Cavendish. |

---

## Initialization Order

Complete sequence when `G.FUNCS.start_run` / `Game:start_run` is called:

1. **Prep stage** → `G.STAGE = STAGES.RUN`, `G.STATE = STATES.BLIND_SELECT`
2. **Resolve selected_back** from args, challenge, or viewed deck
3. **`G.GAME = init_game_object()`** — creates all fields with defaults
4. **Set stake** → `G.GAME.stake = args.stake`
5. **Apply stake modifiers** → fills `G.GAME.modifiers` based on stake level
6. **Create Back object** → `G.GAME.selected_back = Back(selected_back)`
7. **`Back:apply_to_run()`** → modifies `starting_params`, creates starting vouchers/consumables
8. **Apply challenge** (if any):
   - a. Add starting jokers (with eternal/pinned flags)
   - b. Add starting consumables
   - c. Add starting vouchers → `used_vouchers[id] = true`, `Card.apply_to_run()`
   - d. Apply rule modifiers → override `starting_params` fields
   - e. Apply custom rules → set `modifiers.*` and `joker_rate`
   - f. Apply restrictions → set `banned_keys[id] = true`
9. **Feed starting_params → round_resets**:
   - `round_resets.hands = starting_params.hands`
   - `round_resets.discards = starting_params.discards`
   - `round_resets.reroll_cost = starting_params.reroll_cost`
   - `G.GAME.dollars = starting_params.dollars`
10. **Set seed** → seeded flag, seed string, hash all pseudorandom contexts
11. **Generate first ante** → boss blind, voucher, blind tags
12. **Create CardAreas** → `G.hand`, `G.deck`, `G.jokers`, `G.consumeables`, `G.play`, `G.discard`
13. **Build deck** → iterate `G.P_CARDS`, filter by challenge/back, create Cards
14. **Set initial round** → copy `round_resets` into `current_round`, shuffle deck
15. **Reset targeting cards** → idol, mail, ancient, castle
16. **Create Blind object** → `G.GAME.blind = Blind(0,0,2,1)`
17. **Create HUD** → `G.HUD`, `G.HUD_blind`, `G.HUD_tags`
18. **`reset_blinds()`** → finalize blind selection UI state

---

## Key Architectural Notes

### G vs G.GAME

- **`G.*`** fields persist across runs: prototype data (`P_*`), settings, profiles,
  engine objects (E_MANAGER, CONTROLLER), rendering state, UI framework.
- **`G.GAME.*`** is the run state: everything that would be saved/loaded for a run
  in progress. Created fresh by `init_game_object()`.
- **CardAreas** (`G.hand`, `G.deck`, etc.) live on `G` directly, not inside `G.GAME`.
  They are recreated each run in `start_run`. The cards *inside* them are the
  mutable state — `G.GAME` references the cards indirectly through these areas.
- **`G.playing_cards`** is the master card list on `G`, not `G.GAME`.

### Save/Load Boundary

When saving, the system serializes:
- `G.GAME` (the entire run state table)
- Each CardArea's cards via `CardArea:save()` → `Card:save()`
- `G.GAME.blind:save()`, `G.GAME.selected_back:save()`
- Tags via `Tag:save()`

When loading, `start_run` receives a `saveTable` argument that reconstructs
all state. Without a `saveTable`, a fresh run is initialized.

### The Scoring Pipeline

The scoring path reads from multiple G locations:

1. `G.hand.highlighted` → cards selected to play
2. `evaluate_poker_hand()` → determines hand type from card ranks/suits
3. `G.GAME.hands[handname]` → base chips/mult for that hand at current level
4. `G.GAME.blind:modify_hand()` → boss blind may halve chips/mult (The Flint)
5. `G.GAME.blind:debuff_hand()` → boss blind may void the hand entirely
6. `eval_card()` per card → individual card chip bonuses, enhancements, seals, editions
7. `Card:calculate_joker()` per joker → the 2,800-line scoring engine
8. `Back:trigger_effect('final_scoring_step')` → Plasma Deck averages chips+mult
9. Result compared to `G.GAME.blind.chips` to determine if blind is beaten

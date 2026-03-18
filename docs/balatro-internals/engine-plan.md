# Engine Plan — Simulator Blueprint

> **Reference documentation** — produced during initial Balatro v1.0.1o source analysis.
> The jackdaw engine is the authoritative implementation; see `jackdaw/engine/` for current behavior.

> **Note:** The engine is now complete. This was the original planning document
> used to guide the build sequence. Kept for historical context.

Synthesized from all source map documents. This is the blueprint for building
a headless Balatro simulator.

---

## 1. Minimal GameState

Every field needed to fully describe a game at any decision point.

### Card State

```
playing_cards[]              — master list of all playing cards in the run
hand.cards[]                 — cards in player's hand
deck.cards[]                 — draw pile
discard.cards[]              — discard pile
play.cards[]                 — cards in play area (during scoring)
jokers.cards[]               — active joker cards
consumeables.cards[]         — tarot/planet/spectral slots
pack_cards.cards[]           — cards inside opened booster (if any)
shop_jokers.cards[]          — current shop offerings
shop_vouchers.cards[]        — current shop voucher
shop_booster.cards[]         — current shop booster packs

Per-card fields:
  card.ability               — entire mutable state table (name, effect, set,
                                mult, h_mult, h_x_mult, x_mult, t_mult, t_chips,
                                bonus, perma_bonus, extra, extra_value, type,
                                h_size, d_size, h_dollars, p_dollars, order,
                                perma_debuff, played_this_ante, eternal,
                                perishable, perish_tally, rental, forced_selection,
                                hands_played_at_create, and joker-specific fields:
                                caino_xmult, invis_rounds, yorick_discards,
                                loyalty_remaining, to_do_poker_hand, ...)
  card.base                  — suit, rank, id, nominal, suit_nominal,
                                face_nominal, times_played
  card.edition               — nil | {foil=true} | {holo=true} |
                                {polychrome=true} | {negative=true}
  card.seal                  — nil | 'Red' | 'Blue' | 'Gold' | 'Purple'
  card.debuff                — boolean
  card.config.center         — prototype reference (P_CENTERS key)
  card.config.card           — base card reference (P_CARDS key)
  card.sort_id               — unique ordering value
  card.playing_card          — integer index (nil for non-playing-cards)
  card.cost / sell_cost      — current prices
```

### Run Progression

```
GAME.round                          — current round number
GAME.round_resets.ante              — current ante
GAME.win_ante                       — target ante to win (8)
GAME.won                            — game won flag
GAME.stake                          — difficulty (1-8)
GAME.challenge                      — challenge ID or nil
GAME.seeded                         — fixed seed flag
GAME.blind_on_deck                  — 'Small' | 'Big' | 'Boss'
GAME.round_resets.blind_states      — {Small=, Big=, Boss=} state strings
GAME.round_resets.blind_choices     — {Small='bl_small', Big='bl_big', Boss=<key>}
GAME.round_resets.blind_tags        — {Small=<tag_key>, Big=<tag_key>}
GAME.round_resets.boss_rerolled     — boss rerolled this ante
GAME.tags[]                         — active Tag objects
GAME.blind                          — active Blind (chips, mult, disabled, triggered,
                                       boss-specific: hands{}, only_hand, etc.)
GAME.last_blind                     — {boss=bool, name=string}
GAME.bosses_used                    — {[blind_key]=count}
GAME.hands_played                   — total hands in run
GAME.unused_discards                — total unused discards in run
GAME.skips                          — total blinds skipped
GAME.last_hand_played               — hand type name
GAME.starting_deck_size             — cards at run start
STATE                               — current game state enum
```

### Per-Round State

```
GAME.current_round.hands_left       — remaining hand plays
GAME.current_round.hands_played     — hands played this round
GAME.current_round.discards_left    — remaining discards
GAME.current_round.discards_used    — discards used this round
GAME.current_round.most_played_poker_hand — for The Ox
GAME.current_round.idol_card        — {suit, rank} targeting
GAME.current_round.mail_card        — {rank} targeting
GAME.current_round.ancient_card     — {suit} targeting
GAME.current_round.castle_card      — {suit} targeting
GAME.current_round.voucher          — voucher key for shop
GAME.current_round.used_packs       — packs opened
GAME.current_round.jokers_purchased — jokers bought
GAME.current_round.reroll_cost      — current reroll price
GAME.current_round.reroll_cost_increase — cumulative reroll increase
GAME.current_round.free_rerolls     — free rerolls remaining
GAME.current_round.dollars          — dollars earned this round
GAME.round_resets.temp_reroll_cost  — temporary override (D6 Tag)
GAME.round_resets.temp_handsize     — temporary hand size bonus
GAME.round_bonus                    — {next_hands=0, discards=0}
```

### Economy

```
GAME.dollars                 — current money
GAME.bankrupt_at             — minimum dollar threshold (0 or -20)
GAME.interest_cap            — max interest per round (25/50/100)
GAME.interest_amount         — dollars per $5 held
GAME.discount_percent        — shop discount (0/25/50)
GAME.inflation               — cumulative price increase
GAME.base_reroll_cost        — base reroll cost
GAME.previous_round.dollars  — money at end of prior round
```

### Hand Levels

```
GAME.hands[<hand_name>]      — for all 12 types:
  .chips                     — current chip value
  .mult                      — current mult value
  .s_chips / .s_mult         — base values (never change)
  .l_chips / .l_mult         — per-level increments
  .level                     — current level
  .played                    — total times played in run
  .played_this_round         — times played this round
  .visible                   — shown in UI
```

### Pool / Rate State

```
GAME.edition_rate            — edition appearance rate
GAME.joker_rate              — joker weight in shop (20)
GAME.tarot_rate              — tarot weight (4)
GAME.planet_rate             — planet weight (4)
GAME.spectral_rate           — spectral weight (0)
GAME.playing_card_rate       — playing card weight (0)
GAME.rental_rate             — rental appearance rate (3)
GAME.used_vouchers           — {[key]=true}
GAME.used_jokers             — {[key]=true}
GAME.banned_keys             — {[key]=true}
GAME.pool_flags              — {gros_michel_extinct=bool, ...}
GAME.probabilities.normal    — base probability numerator (1, or 2 with Oops)
GAME.shop.joker_max          — shop joker slot count
```

### RNG State

```
GAME.pseudorandom.seed       — seed string (e.g. "A3K9NZ2B")
GAME.pseudorandom.hashed_seed — numeric hash of seed
GAME.pseudorandom[<key>]     — per-stream float state (~65 keys)
```

### Modifiers & Params

```
GAME.starting_params         — frozen: dollars, hand_size, hands, discards,
                                reroll_cost, joker_slots, consumable_slots,
                                ante_scaling, no_faces, erratic_suits_and_ranks
GAME.round_resets.hands      — hands per round (from starting_params + vouchers)
GAME.round_resets.discards   — discards per round
GAME.round_resets.reroll_cost
GAME.modifiers               — stake + challenge modifiers (scaling, no_interest,
                                no_blind_reward, enable_eternals/perishables/rentals,
                                all_eternal, inflation, chips_dollar_cap,
                                debuff_played_cards, flipped_cards,
                                minus_hand_size_per_X_dollar, discard_cost, etc.)
```

### Miscellaneous

```
GAME.chips                   — chips scored so far this blind
GAME.joker_buffer / consumeable_buffer — slot reservation counters
GAME.STOP_USE                — consumable use lock
GAME.ecto_minus              — Ectoplasm penalty accumulator
GAME.last_tarot_planet       — key of last tarot/planet used
GAME.consumeable_usage_total — {tarot, planet, spectral, tarot_planet, all}
GAME.pack_size / pack_choices — active pack state
GAME.orbital_choices         — {[ante]={[blind_type]=hand_name}}
GAME.cards_played            — per-rank play tracking
GAME.selected_back           — Back object (name, effect.config)
GAME.max_jokers              — highest joker count achieved

Slot limits:
  jokers.config.card_limit       — joker slot count
  consumeables.config.card_limit — consumable slot count
  hand.config.card_limit         — hand size
```

---

## 2. Complete Action Space

### BLIND_SELECT

| Action | Parameters | Validation |
|--------|-----------|------------|
| `select_blind` | — | Blind must be in 'Select' state |
| `skip_blind` | — | Can only skip Small or Big, not Boss |
| `use_consumable` | `card_idx`, `target_indices[]?` | `can_use_consumeable()` rules |

### SELECTING_HAND

| Action | Parameters | Validation |
|--------|-----------|------------|
| `play_hand` | `card_indices[1..5]` | 1-5 cards from hand, `hands_left > 0` |
| `discard` | `card_indices[1..5]` | 1-5 cards from hand, `discards_left > 0` |
| `use_consumable` | `card_idx`, `target_indices[]?` | Per-card selection rules |
| `sort_hand` | `mode: 'rank'\|'suit'` | Always valid (no state mutation) |

### ROUND_EVAL

| Action | Parameters | Validation |
|--------|-----------|------------|
| `cash_out` | — | Always valid |
| `use_consumable` | `card_idx`, `target_indices[]?` | Limited selection rules |

### SHOP

| Action | Parameters | Validation |
|--------|-----------|------------|
| `buy_card` | `card_idx` | Affordable, slot available |
| `buy_and_use` | `card_idx`, `target_indices[]?` | Affordable, usable |
| `sell_card` | `card_idx`, `area` | Not eternal, not in play |
| `redeem_voucher` | `card_idx` | Affordable |
| `open_booster` | `card_idx` | Affordable |
| `reroll` | — | Affordable or free rerolls |
| `use_consumable` | `card_idx`, `target_indices[]?` | Standard rules |
| `next_round` | — | Always valid |

### Pack States

| Action | Parameters | Validation |
|--------|-----------|------------|
| `pick_card` | `pack_card_idx` | `pack_choices > 0` |
| `skip_pack` | — | Always valid |

### Automated (no player input)

| Trigger | Condition | Result |
|---------|-----------|--------|
| Draw to hand | After discard/new round | Cards drawn up to hand limit |
| Scoring | After play hand | Full 14-phase pipeline |
| Round end | After NEW_ROUND state | `end_round()` evaluation |
| Game over | Blind not met + no saves | Terminal |

---

## 3. Dependency Order

### Build Phases

```
PHASE 1 — Foundation (independent, no game-logic deps)
├── RNG System (pseudohash, pseudoseed, pseudorandom, pseudoshuffle)
├── Data Tables (P_CENTERS, P_CARDS, P_BLINDS, P_TAGS prototypes)
├── Card Struct (ability, base, edition, seal, debuff)
├── Poker Hand Evaluator (flush/straight/n-of-a-kind detection)
├── Hand Level System (12 types, level-up formula)
└── Economy Helpers (interest calc, sell price formula)

PHASE 2 — Card Mechanics (depends on Phase 1)
├── Card Enhancement/Edition/Seal scoring methods
│   (get_chip_bonus, get_chip_mult, get_chip_x_mult, get_edition, etc.)
├── Blind System (debuff_hand, modify_hand, press_play, debuff_card)
├── Deck Building (standard 52, Abandoned/Erratic/Checkered mutations)
└── Base Scoring Pipeline (phases 1-4 and 12 only: hand detect →
    base chips/mult → blind modify → floor(chips × mult))

PHASE 3 — The Hard Part (depends on Phase 2)
├── Joker System (150 jokers in calculate_joker)
│   Build in sub-phases:
│   ├── Simple jokers (flat bonuses, ~3 jokers)
│   ├── Conditional jokers (suit/rank/hand-type checks, ~70 jokers)
│   ├── Scaling jokers (mutable state, ~30 jokers)
│   ├── Retrigger jokers (6 jokers)
│   ├── Copy jokers (Blueprint/Brainstorm/Perkeo)
│   └── Side-effect jokers (card creation/destruction, ~25 jokers)
└── Full Scoring Pipeline (all 14 phases with joker hooks, retriggers,
    card destruction, side effects)

PHASE 4 — Game Systems (depends on Phase 3)
├── Pool Generation (get_current_pool, filtering, rarity roll)
├── Card Creation Factory (create_card, soul chance, edition poll,
│   eternal/perishable/rental rolls)
├── Consumable System (22 tarots + 12 planets + 18 spectrals)
├── Voucher System (32 vouchers with prerequisite chains)
├── Shop System (population, buy/sell/reroll transactions)
├── Pack System (5 pack types, card generation per type)
├── Tag System (~20 tags, hooks at blind select/shop/round eval)
└── Back System (15 decks, apply_to_run + trigger_effect)

PHASE 5 — Orchestration (depends on Phase 4)
└── Game Loop / State Machine
    (state transitions, round lifecycle, ante progression,
     action dispatch, win/lose detection)
```

### Critical Path

```
RNG → Card Creation → Shop → State Machine
       ↓
Hand Eval → Base Scoring → Joker System → Full Scoring → Consumables
                ↑
         Card Methods + Blind System
```

The **longest pole** is the Joker System — it's both the most complex single
piece and a prerequisite for the full scoring pipeline, which in turn is needed
by consumables and the state machine.

---

## 4. Tricky Lua Patterns

### Event Queue → Synchronous Calls

The source runs everything through `G.E_MANAGER:add_event(Event({...}))` with
blocking, delays, conditional triggers, and nested chains. The scoring pipeline
alone is ~30 queued events.

**Recommendation:** Flatten to synchronous function calls. Replace every
`add_event` with a direct call. The event system exists for animation timing,
not for game logic ordering. The logical order is already determined by the
code structure (events are queued sequentially within `evaluate_play`).

**Caveat:** Event ordering DOES matter for side effects. Joker state mutations
during scoring happen in a specific sequence. Preserve the call order.

### `calculate_joker` Monolith

1,770-line `if/elseif` chain matching on `self.ability.name` strings.

**Options:**
1. **Dispatch table** keyed by joker key (e.g., `j_joker`) → handler function.
   Cleanest architecture but requires extracting ~150 inline blocks.
2. **One big match/switch** — faithful port, easiest to verify correctness.
3. **Trait/interface per joker** — most idiomatic but highest divergence risk.

**Recommendation:** Start with option 2 (big match) for correctness, refactor
to option 1 (dispatch table) once all jokers pass tests. The context-flag-bag
pattern maps to an enum + struct in typed languages.

### Floating-Point Determinism

The RNG uses `string.format("%.13f", ...)` to truncate floats to 13 decimal
places. The constants `2.134453429141` and `1.72431234` and the
`(advanced + hashed_seed) / 2` averaging must be reproduced **exactly**.

**Risk:** Different languages/platforms may produce different floating-point
results for the same operations. Test against known seed→output pairs early.

### `pairs()` Non-Determinism

Lua's `pairs()` iterates hash tables in undefined order. The source compensates
with explicit sorting (by `sort_id` or key) before random selection.

**Port:** Use ordered iteration or sort before any random access to collections.

### Deep Table Copy for Ability State

`copy_table(center.config.extra)` ensures each card instance owns its mutable
state. Languages with value semantics (Rust structs) get this for free.
Reference-type languages need explicit clone.

### Blueprint Recursion Prevention

`context.blueprint` counter incremented on each delegation, checked before
recursing. Must be replicated to prevent infinite Blueprint↔Brainstorm loops.

### `STOP_USE` and Buffer Counters

Concurrency guards for the async event system. In a synchronous simulator,
`STOP_USE` can likely be ignored. `joker_buffer` and `consumeable_buffer` can
be replaced with direct "will this fit?" checks, but verify no edge cases.

---

## 5. What to Ignore Entirely

### Zero Impact on Game Logic

| Category | Examples | Why safe |
|----------|---------|---------|
| **Rendering** | Sprites, atlases, shaders, particles, canvas, transforms (T/VT), parallax, dissolve/materialize animations | Pure visual |
| **UI Framework** | UIBox, UIT nodes, layout, HUD, overlays, buttons (as display objects) | Pure presentation |
| **Audio** | Sound manager, `play_sound()`, `G.PITCH_MOD`, music tracks | Pure presentation |
| **Input System** | Controller, cursor, drag/drop, focus, HID detection | Replace with action dispatch |
| **Text Rendering** | DynaText, text animations, `card_eval_status_text` display | Scoring popups are visual only |
| **Animation Timing** | `delay()`, ease functions, `G.TIMERS`, `G.FRAMES`, `GAMESPEED` | Replace with immediate execution |
| **Platform** | Steam integration, crash reports, HTTP, window management | Not gameplay |
| **Localization** | All `localization/*.lua`, `localize()` calls | Display strings only |
| **Jimbo** | Card_Character, tutorial overlay, speech bubbles | Mascot character |
| **Debug** | test_functions.lua, perf overlay, DT_* functions | Dev tools |
| **Save I/O Threading** | save_manager thread, profile.lua file I/O | Threaded save is an implementation detail |

### Replace With Simpler Equivalents

| Source Pattern | Replacement |
|----------------|-------------|
| `ease_dollars(n)` | `GAME.dollars += n` |
| `ease_chips(n)` | `GAME.chips = n` |
| `ease_hands_played(n)` | `current_round.hands_left += n` |
| `ease_discard(n)` | `current_round.discards_left += n` |
| `draw_card(from, to, ...)` | `card = from.remove(); to.add(card)` |
| `card:start_dissolve()` | `remove_card(card)` |
| `card:start_materialize()` | (no-op, card already exists) |
| `card:flip()` | `card.facing = toggle` |
| `card_eval_status_text(...)` | (no-op, pure display) |
| `attention_text(...)` | (no-op, pure display) |
| `update_hand_text(...)` | (no-op, or update display state if needed) |
| `G.E_MANAGER:add_event(Event({func=f}))` | `f()` (call directly) |
| `G.ROOM.jiggle += n` | (no-op) |

### Keep but Simplify

| System | What to keep | What to cut |
|--------|-------------|-------------|
| **Card Areas** | Card list management (add/remove/sort/shuffle) | Layout, alignment, draw, highlight visuals |
| **Blind Object** | Debuff logic, chip target, boss state | Animated sprite, tilt, particles, hover |
| **Tag Object** | `apply_to_run` effects | UI generation, hover, alert, HUD positioning |
| **Back Object** | `apply_to_run` + `trigger_effect` | Sprite position, locale display |
| **Achievement/Unlock** | Skip entirely for simulation | Or keep as optional stat tracking |

---

## 6. Complexity Ranking

| Priority | System | Difficulty | Est. Size | Dependencies | Notes |
|----------|--------|-----------|-----------|-------------|-------|
| 1 | RNG System | 5/10 | ~100 loc | None | Float precision is the risk. Validate early with known seed pairs. |
| 2 | Data Tables | 2/10 | ~500 loc | None | Tedious transcription. ~300 prototypes. Consider auto-extracting from Lua. |
| 3 | Card Representation | 3/10 | ~200 loc | Data Tables | Struct with ability, base, edition, seal. Deep-clone for `extra`. |
| 4 | Poker Hand Evaluator | 3/10 | ~200 loc | Card repr | Well-defined algorithm. Must handle Four Fingers + Shortcut. |
| 5 | Hand Level System | 2/10 | ~50 loc | None | 12 types, linear formula. |
| 6 | Economy Helpers | 2/10 | ~50 loc | None | Interest, sell price, cost formula. |
| 7 | Deck Building | 4/10 | ~150 loc | RNG, Cards, Data | Standard 52 + Abandoned/Erratic/Checkered/challenge mutations. |
| 8 | Card Scoring Methods | 3/10 | ~100 loc | Card repr | `get_chip_bonus`, `get_chip_mult`, `get_edition`, etc. 8 methods. |
| 9 | Blind System | 6/10 | ~400 loc | Cards, Scoring | ~26 bosses. Each has unique debuff/modify/press_play logic. |
| 10 | Base Scoring Pipeline | 5/10 | ~200 loc | Hand Eval, Hands, Blinds, Cards | Phases 1-4 + 12. No jokers yet. Good validation checkpoint. |
| **11** | **Joker System** | **10/10** | **~2000 loc** | Everything above | **The hardest piece.** 150 jokers × multiple phases. Build incrementally by category. |
| 12 | Full Scoring Pipeline | 9/10 | ~500 loc | Joker System | Phases 5-14: retriggers, per-card joker effects, destruction, side effects. |
| 13 | Consumable System | 8/10 | ~600 loc | Cards, Hands, RNG, Jokers | ~50 effects. Tarots are simple; spectrals are complex (destruction/creation). |
| 14 | Pool Generation | 7/10 | ~300 loc | RNG, Data, Vouchers | 7+ filter conditions, rarity roll, UNAVAILABLE preservation, resampling. |
| 15 | Shop System | 7/10 | ~400 loc | Pools, Economy, Cards | Buy/sell/reroll, weighted type selection, slot limits, pricing. |
| 16 | Pack System | 5/10 | ~200 loc | Shop, Consumables | 5 pack types, card generation, pick/skip mechanics. |
| 17 | Tag System | 5/10 | ~300 loc | Shop, Economy, Cards | ~20 tags. Most simple, a few complex (Double, D6, edition tags). |
| 18 | Back (Deck) System | 4/10 | ~150 loc | Starting Params, Scoring | 15 decks. Plasma scoring hook is the only complex one. |
| 19 | Voucher System | 3/10 | ~150 loc | Economy, Params | 32 vouchers. Simple param mutations + prerequisite chain. |
| 20 | State Machine | 6/10 | ~400 loc | All systems | Synchronous game loop. Dramatically simpler than the source's async event-driven model. |

### Estimated Totals

| Category | Lines | % of effort |
|----------|-------|-------------|
| Joker System | ~2,000 | 30% |
| Scoring Pipeline | ~700 | 10% |
| Consumables | ~600 | 9% |
| Shop + Pools | ~700 | 10% |
| Blind System | ~400 | 6% |
| State Machine | ~400 | 6% |
| Data Tables | ~500 | 7% |
| Everything Else | ~1,500 | 22% |
| **Total** | **~6,800** | **100%** |

Compared to ~15,000 lines of game logic in the source (excluding UI/engine),
the simulator should be roughly **45% the size** — the savings come from
eliminating animation, UI, event queue overhead, and display formatting.

---

## Summary: Recommended Build Sequence

```
Week 1: Foundation
  ✓ RNG System (with validation against known seeds)
  ✓ Data Tables (auto-extract from Lua if possible)
  ✓ Card struct + Poker Hand Evaluator
  ✓ Hand Level System + Economy Helpers

Week 2: Core Scoring
  ✓ Card scoring methods (enhancement/edition/seal)
  ✓ Blind system (all 26 bosses)
  ✓ Base scoring pipeline (phases 1-4, 12)
  ✓ Deck building (all 15 decks)
  → CHECKPOINT: can compute scores for any hand against any blind

Week 3-4: Joker System
  ✓ Simple + conditional jokers (~73)
  ✓ Scaling jokers (~31)
  ✓ Retrigger jokers (6) + retrigger loop
  ✓ Side-effect jokers (~25)
  ✓ Copy jokers (Blueprint/Brainstorm)
  ✓ Full scoring pipeline (all 14 phases)
  → CHECKPOINT: full scoring matches source for any hand+joker combo

Week 5: Game Systems
  ✓ Consumable system (tarots, planets, spectrals)
  ✓ Pool generation + card creation factory
  ✓ Shop system (buy/sell/reroll/vouchers)
  ✓ Pack system
  ✓ Tag system
  → CHECKPOINT: can simulate a complete shop visit

Week 6: Orchestration
  ✓ State machine / game loop
  ✓ Stake modifiers (all 8)
  ✓ Challenge system
  ✓ Full run simulation
  → CHECKPOINT: can play a complete run from seed to win/loss
```

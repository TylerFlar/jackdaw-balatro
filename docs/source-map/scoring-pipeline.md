# Scoring Pipeline

Exact trace from "play hand" button press to final score computation.
Every function call, every phase, exact order of operations.

---

## Entry Point

**Player clicks "Play"** → `G.FUNCS.play_cards_from_highlighted`
(`state_events.lua:450`)

---

## Phase 0: Setup & Card Movement

`play_cards_from_highlighted` (`state_events.lua:450-537`)

```
1. Guard: if G.play already has cards, return
2. stop_use() — lock further input
3. Clear forced_selection on all playing_cards
4. Sort G.hand.highlighted by x-position (left to right)
5. Queue event: G.STATE = HAND_PLAYED, STATE_COMPLETE = true
6. Increment career stats (c_cards_played, c_hands_played)
7. ease_hands_played(-1) — decrement hands_left
8. For each highlighted card (left to right):
   a. Track face cards played stat
   b. card.base.times_played += 1
   c. card.ability.played_this_ante = true
   d. round_scores.cards_played.amt += 1
   e. draw_card(G.hand, G.play, ...) — animate card to play area
9. G.GAME.blind:press_play() — boss blind reaction
   (The Hook discards 2 random cards, The Tooth costs $1/card, etc.)
10. Queue event chain:
    a. check_for_unlock({type = 'hand_contents'})
    b. G.FUNCS.evaluate_play() — THE SCORING ENGINE (see below)
    c. G.FUNCS.draw_from_play_to_discard() — move surviving cards to discard
    d. G.STATE_COMPLETE = false — triggers state machine advance
```

---

## Phase 1: Hand Type Detection

`G.FUNCS.evaluate_play` (`state_events.lua:571`) calls:

```lua
text, disp_text, poker_hands, scoring_hand, non_loc_disp_text =
    G.FUNCS.get_poker_hand_info(G.play.cards)
```

### `G.FUNCS.get_poker_hand_info` (`state_events.lua:540`)

1. Calls `evaluate_poker_hand(cards)` (`misc_functions.lua:376`)
2. Returns the **highest-priority** matching hand

### `evaluate_poker_hand(hand)` (`misc_functions.lua:376`)

Computes component parts:
```
_5       = get_X_same(5, hand)    — groups of 5 same rank
_4       = get_X_same(4, hand)    — groups of 4
_3       = get_X_same(3, hand)    — groups of 3
_2       = get_X_same(2, hand)    — pairs
_flush   = get_flush(hand)        — flush suit group
_straight = get_straight(hand)    — straight sequence
_highest = get_highest(hand)      — single highest card
```

Priority check (first match wins):

| Priority | Hand | Condition |
|----------|------|-----------|
| 1 | Flush Five | `_5` AND `_flush` |
| 2 | Flush House | `_3` AND `_2` AND `_flush` |
| 3 | Five of a Kind | `_5` |
| 4 | Straight Flush | `_flush` AND `_straight` |
| 5 | Four of a Kind | `_4` |
| 6 | Full House | `_3` AND `_2` |
| 7 | Flush | `_flush` |
| 8 | Straight | `_straight` |
| 9 | Three of a Kind | `_3` |
| 10 | Two Pair | two entries in `_2` |
| 11 | Pair | `_2` |
| 12 | High Card | `_highest` |

**Downward propagation**: Five of a Kind → also populates Four/Three/Pair.
Four of a Kind → also populates Three/Pair. Three of a Kind → also populates Pair.

**Joker modifiers**:
- `Four Fingers` — flush/straight need only 4 cards instead of 5
- `Shortcut` — straights allow 1-rank gaps
- `Pareidolia` — all cards count as face cards (affects `is_face`, not hand detection)
- `Splash` — all played cards become scoring cards (applied in Phase 2)

### Return value

`scoring_hand` = the specific cards forming the detected hand (e.g., the 3 Kings
for Three of a Kind). Other played cards are **not** in `scoring_hand` unless
augmented in Phase 2.

---

## Phase 2: Scoring Hand Augmentation

Back in `evaluate_play` (`state_events.lua:580-600`):

```
1. If Splash joker is present: ALL played cards added to scoring_hand
2. Stone Cards not already in scoring_hand: added to scoring_hand
3. scoring_hand sorted by x-position
```

Then hand stats updated:
```
G.GAME.hands[text].played += 1
G.GAME.hands[text].played_this_round += 1
G.GAME.last_hand_played = text
```

---

## Phase 3: Boss Blind Debuff Check

```lua
if G.GAME.blind:debuff_hand(scoring_hand, poker_hands, text) then
    -- DEBUFFED PATH (Phase 3a)
else
    -- NORMAL SCORING PATH (Phase 4+)
end
```

### `Blind:debuff_hand` (`blind.lua:519`)

Returns `true` (hand scores zero) if:
- Blind debuffs this hand type (e.g., The Head debuffs Hearts hands)
- Card count too small (`h_size_ge`) or too large (`h_size_le`)
- **The Eye**: blocks repeated hand types (each type usable once)
- **The Mouth**: blocks all hand types except the first one played
- **The Arm**: doesn't block, but levels down the hand type
- **The Ox**: doesn't block, but drains all money if most-played hand is used

### Phase 3a: Debuffed Path (`state_events.lua:997-1027`)

```
mult = 0, hand_chips = 0
Display "Not Allowed!" text
For each joker: eval_card(joker, {debuffed_hand = true})
    → jokers can still react (e.g., Matador gains $8 vs boss blinds)
Skip to Phase 10 (final score = 0)
```

---

## Phase 4: Base Chips & Mult Lookup

```lua
mult = mod_mult(G.GAME.hands[text].mult)         -- base mult from hand level
hand_chips = mod_chips(G.GAME.hands[text].chips)  -- base chips from hand level
```

`mod_mult` and `mod_chips` (`misc_functions.lua:684,691`) are simple wrappers
that floor the values and ensure minimum of 0 (chips) or 1 (mult).

The hand levels table (`G.GAME.hands`):
```
chips = s_chips + (level - 1) * l_chips
mult  = s_mult  + (level - 1) * l_mult
```

Display update: `update_hand_text({...}, {handname, level, mult, chips})`

---

## Phase 5: Joker "Before" Phase

(`state_events.lua:628-638`)

```
For each joker in G.jokers.cards:
    result = eval_card(joker, {cardarea=G.jokers, before=true,
                               scoring_name=text, scoring_hand=scoring_hand,
                               poker_hands=poker_hands, full_hand=G.play.cards})
    If result.level_up:
        level_up_hand(joker, text)  — permanently upgrades the poker hand
```

Jokers using this: **Space Joker** (1-in-4 chance to level up played hand).

---

## Phase 6: Blind Hand Modification

(`state_events.lua:645`)

```lua
mult, hand_chips, _ = G.GAME.blind:modify_hand(G.play.cards, poker_hands, text, mult, hand_chips)
```

### `Blind:modify_hand` (`blind.lua:510`)

- **The Flint**: `mult = max(floor(mult*0.5 + 0.5), 1)` and
  `hand_chips = max(floor(hand_chips*0.5 + 0.5), 0)`

---

## Phase 7: Per-Scored-Card Loop

(`state_events.lua:648-780`)

For each card `i` in `scoring_hand` (left to right):

### 7a. Debuff check
If `scoring_hand[i].debuff` → show debuff text, **skip this card entirely**.

### 7b. Collect retriggers

```
reps = {1}   -- base: 1 evaluation (the card itself)

-- Red Seal retrigger
seal_result = eval_card(card, {repetition_only=true, cardarea=G.play, ...})
if seal_result.seals and seal_result.seals.repetitions:
    append seal_result.seals.repetitions to reps

-- Joker retriggers
for each joker:
    joker_result = eval_card(joker, {repetition=true, other_card=card,
                                     cardarea=G.play, ...})
    if joker_result.jokers and joker_result.jokers.repetitions:
        append joker_result.jokers.repetitions to reps
```

`reps` is an array like `{1, 1, 1}` meaning 3 total evaluations (base + 2 retriggers).

**Retrigger sources**:
- **Red Seal**: `Card:calculate_seal({repetition=true})` → `repetitions = 1`
- **Sock and Buskin**: retrigger face cards
- **Hanging Chad**: retrigger first scored card
- **Dusk**: retrigger all on final hand
- **Seltzer**: retrigger all played cards
- **Hack**: retrigger 2/3/4/5 rank cards

### 7c. For each repetition (including base)

```
-- Card's own scoring:
eval = eval_card(scoring_hand[i], {cardarea=G.play, full_hand=G.play.cards,
                                   scoring_hand=scoring_hand, poker_hand=text})

-- Per-card joker effects:
for each joker:
    joker_eval = joker:calculate_joker({individual=true, other_card=scoring_hand[i],
                                        full_hand=G.play.cards, ...})
```

### `eval_card(card, {cardarea=G.play})` (`common_events.lua:580`)

Calls these Card methods and returns:

| Return field | Method | What it computes |
|---|---|---|
| `ret.chips` | `card:get_chip_bonus()` | Base rank chips + bonus + perma_bonus. Stone Card: just bonus. |
| `ret.mult` | `card:get_chip_mult()` | Enhancement mult. Lucky Card: probabilistic (normal/5 chance). |
| `ret.x_mult` | `card:get_chip_x_mult()` | Enhancement x_mult (if > 1). Glass Card = 2x. |
| `ret.p_dollars` | `card:get_p_dollars()` | Per-play dollars. Gold Card, Lucky Card ($, 1-in-15 chance). |
| `ret.edition` | `card:get_edition()` | `{chip_mod, mult_mod, x_mult_mod}` from Foil/Holo/Polychrome. |

### 7d. Apply effects (exact order within each rep)

```
1. chips → hand_chips += eval.chips               [ADDITIVE]
2. mult  → mult += eval.mult                      [ADDITIVE]
3. p_dollars → earned, display                     [SIDE EFFECT]
4. extra effects from jokers:
   a. extra.mult_mod → mult += value              [ADDITIVE]
   b. extra.chip_mod → hand_chips += value        [ADDITIVE]
   c. extra.swap → swap hand_chips and mult        [SWAP]
   d. extra.func → call arbitrary function         [SIDE EFFECT]
5. x_mult → mult *= eval.x_mult                   [MULTIPLICATIVE]
6. joker individual x_mult → mult *= value         [MULTIPLICATIVE]
7. edition:
   a. chip_mod → hand_chips += edition.chip_mod   [ADDITIVE]
   b. mult_mod → mult += edition.mult_mod         [ADDITIVE]
   c. x_mult_mod → mult *= edition.x_mult_mod    [MULTIPLICATIVE]
```

**Critical**: within a single card evaluation, additive effects come first,
then multiplicative. Edition x_mult is last.

---

## Phase 8: Per-Held-Card Loop

(`state_events.lua:784-872`)

For each card in `G.hand.cards` (cards **not** played, still in hand):

### 8a. Collect retriggers (same mechanism as Phase 7b)
- Red Seal
- **Mime** joker: retriggers held cards that have effects

### 8b. For each repetition

```
eval = eval_card(card, {cardarea=G.hand, ...})
for each joker:
    joker_eval = joker:calculate_joker({individual=true, other_card=card,
                                        cardarea=G.hand, ...})
```

### `eval_card(card, {cardarea=G.hand})` (`common_events.lua:580`)

| Return field | Method | What it computes |
|---|---|---|
| `ret.h_mult` | `card:get_chip_h_mult()` | Held-in-hand mult. Steel Card = +h_mult. |
| `ret.x_mult` | `card:get_chip_h_x_mult()` | Held-in-hand x_mult. |

### 8c. Apply effects (per rep)

```
1. dollars from joker → earned                     [SIDE EFFECT]
2. h_mult → mult += eval.h_mult                   [ADDITIVE]
3. x_mult → mult *= eval.x_mult                   [MULTIPLICATIVE]
4. joker x_mult → mult *= value                   [MULTIPLICATIVE]
```

**Held-in-hand joker examples**: Baron (+1.5x per King held), Shoot the Moon
(+13 mult per Queen held), Reserved Parking ($ per face card held).

---

## Phase 9: Joker Main Scoring

(`state_events.lua:877-943`)

For each card in `G.jokers.cards` **AND** `G.consumeables.cards` (left to right):

### 9a. Edition pre-pass (additive only)

```lua
edition_eval = eval_card(card, {cardarea=G.jokers, edition=true, ...})
```

Applies `chip_mod` and `mult_mod` from the joker's edition:
```
hand_chips += edition.chip_mod    -- Foil: +50 chips
mult += edition.mult_mod          -- Holo: +10 mult
```

### 9b. Main joker effect

```lua
joker_eval = eval_card(card, {cardarea=G.jokers, joker_main=true, ...})
```

Returns `{mult_mod, chip_mod, Xmult_mod, message, ...}`. Applied in order:
```
1. mult += joker_eval.mult_mod                     [ADDITIVE]
2. hand_chips += joker_eval.chip_mod               [ADDITIVE]
3. Xmult_mod → mult *= joker_eval.Xmult_mod       [MULTIPLICATIVE]
```

**Note the naming difference**: in joker_main context, the return fields are
`mult_mod`/`chip_mod`/`Xmult_mod`, not `mult`/`chips`/`x_mult`.

### 9c. Joker-on-joker effects

```lua
for each other_joker v:
    v:calculate_joker({other_joker=card, full_hand=G.play.cards, ...})
```

**Baseball Card** uses this: gives x1.5 for each Uncommon joker.

### 9d. Edition post-pass (multiplicative only)

```
mult *= edition.x_mult_mod        -- Polychrome: x1.5
```

**Critical ordering**: Edition additive (Foil/Holo) is applied BEFORE the
joker's own effect. Edition multiplicative (Polychrome) is applied AFTER
the joker's own effect AND after joker-on-joker effects.

---

## Phase 10: Deck Back Final Step

(`state_events.lua:946`)

```lua
hand_chips, mult = G.GAME.selected_back:trigger_effect{
    context = 'final_scoring_step', chips = hand_chips, mult = mult
}
```

### `Back:trigger_effect` (`back.lua:125`)

- **Plasma Deck**: `tot = chips + mult; chips = floor(tot/2); mult = floor(tot/2)`
  (averages chips and mult — shows "Balanced!" text)

---

## Phase 11: Card Destruction

(`state_events.lua:950-996`)

```
For each joker:
    result = joker:calculate_joker({destroying_card=scoring_hand[i], full_hand=...})
    If result.remove: mark card for destruction

For each scoring card:
    If Glass Card: 1/G.GAME.probabilities.normal in 4 chance to shatter
    If marked for destruction: card:shatter() or card:start_dissolve()

If any cards destroyed:
    For each joker: eval_card(joker, {removing_playing_cards=true, removed=destroyed_list})
```

Side effects: destroyed cards are removed from `G.playing_cards`.

---

## Phase 12: Final Score Computation

(`state_events.lua:1029-1065`)

```lua
chip_total = math.floor(hand_chips * mult)
```

This is the **one line** where the final score is computed.

Then:
```
check_and_set_high_score('hand', chip_total)
ease_chips(G.GAME.chips + chip_total)  — adds to running blind total
```

`ease_chips` (`common_events.lua:41`) smoothly animates `G.GAME.chips` to the new
value. This is where the score **actually mutates `G.GAME.chips`**.

---

## Phase 13: Joker "After" Phase

(`state_events.lua:1068-1075`)

```
For each joker:
    eval_card(joker, {cardarea=G.jokers, after=true, scoring_name=text,
                      scoring_hand=scoring_hand, full_hand=G.play.cards})
```

Jokers using this:
- **Ice Cream**: loses chips each hand
- **Seltzer**: decrements uses, destroyed when 0
- **Green Joker**: loses mult on discard (after context check)
- **Mr. Bones**: if chips < blind and would die, prevents game over

---

## Phase 14: Post-Play Modifiers

(`state_events.lua:1077-1084`)

```
If G.GAME.modifiers.debuff_played_cards:
    For each scoring card: card.ability.perma_debuff = true
```

This is the "Double or Nothing" challenge — played cards become permanently debuffed.

---

## Complete Scoring Order Summary

```
┌─────────────────────────────────────────────┐
│  1. Detect hand type (evaluate_poker_hand)  │
│  2. Augment scoring hand (Splash, Stone)    │
│  3. Boss blind debuff check                 │
│  4. Base chips/mult from hand level         │
│  5. Joker "before" (level-up chance)        │
│  6. Blind:modify_hand (The Flint halves)    │
├─────────────────────────────────────────────┤
│  7. FOR EACH scored card (with retriggers): │
│     a. +chips (rank + bonus)                │
│     b. +mult (enhancement)                  │
│     c. +$ (Gold Card, Lucky Card)           │
│     d. Joker per-card: +chips, +mult        │
│     e. Joker per-card: xMult                │
│     f. Card edition: +chips, +mult, xMult   │
├─────────────────────────────────────────────┤
│  8. FOR EACH held card (with retriggers):   │
│     a. +mult (Steel Card)                   │
│     b. Joker per-held: +mult, xMult         │
├─────────────────────────────────────────────┤
│  9. FOR EACH joker (left to right):         │
│     a. Edition +chips, +mult (Foil, Holo)   │
│     b. Joker main: +chips, +mult, xMult     │
│     c. Joker-on-joker: +chips, +mult, xMult │
│     d. Edition xMult (Polychrome)           │
├─────────────────────────────────────────────┤
│ 10. Deck back final step (Plasma averages)  │
│ 11. Card destruction (Glass shatter, etc.)  │
│ 12. FINAL = floor(hand_chips × mult)        │
│ 13. Joker "after" effects                   │
│ 14. Debuff played cards (challenge mod)     │
└─────────────────────────────────────────────┘
```

---

## Side Effects During Scoring

Scoring is **not** a pure computation. These mutations happen:

| Side Effect | Where | What |
|---|---|---|
| `G.GAME.chips` | Phase 12 | Score added to blind progress |
| `G.GAME.dollars` | Phase 7d, 8c | Lucky Card $, Gold Card $, joker $ |
| `G.GAME.hands[text].played` | Phase 2 | Hand play count incremented |
| `G.GAME.hands[text].level` | Phase 5 | Space Joker can level up hand |
| `card.base.times_played` | Phase 0 | Per-card play counter |
| `card.ability.played_this_ante` | Phase 0 | Pillar blind debuff tracking |
| `card.ability.perma_debuff` | Phase 14 | Double or Nothing challenge |
| `G.playing_cards` | Phase 11 | Destroyed cards removed |
| Joker internal state | Phase 9b | Many jokers mutate self.ability (e.g., Ice Cream loses chips, Ride the Bus resets, etc.) |
| `G.GAME.current_round.hands_played` | Phase 0 (post-eval) | Incremented after scoring |
| `G.GAME.round_scores.cards_played` | Phase 0 | Career stat |
| `G.GAME.blind.triggered` | Phase 6, 7 | Boss blind flags its activation |
| High scores | Phase 12 | `check_and_set_high_score('hand', ...)` |
| `G.GAME.blind.hands[handname]` | Phase 3 | The Eye marks hand type as used |
| `G.GAME.blind.only_hand` | Phase 3 | The Mouth locks to first hand type |

---

## Retrigger Mechanics

Retriggers are **not** recursive calls. They are a **loop counter**.

```lua
-- Collect retriggers into array
reps = {1}  -- base evaluation
if Red_Seal: table.insert(reps, 1)
for each joker with repetition effect:
    table.insert(reps, joker_repetitions)

-- Execute N times
for j = 1, #reps do
    -- full card evaluation (chips + mult + x_mult + edition + joker individual)
    -- if j > 1: show "Again!" text
end
```

Each retrigger is a complete re-evaluation of that card — all chip bonuses,
mult bonuses, x_mult bonuses, edition effects, and per-card joker effects
fire again. The retrigger count from different sources is **additive**
(Red Seal + Hack = 2 extra evaluations).

Retriggers apply independently to scored cards (Phase 7) and held cards (Phase 8).

---

## +Chips / +Mult / xMult Distinction in Code

The code uses different field names depending on context:

### In `eval_card` returns (playing card context):
| Field | Type | Applied as |
|---|---|---|
| `chips` | number | `hand_chips += value` |
| `mult` | number | `mult += value` |
| `x_mult` | number | `mult *= value` |
| `h_mult` | number | `mult += value` (held cards) |

### In `calculate_joker` returns (individual context):
| Field | Type | Applied as |
|---|---|---|
| `chips` | number | `hand_chips += value` |
| `mult` | number | `mult += value` |
| `x_mult` | number | `mult *= value` |

### In `calculate_joker` returns (joker_main context):
| Field | Type | Applied as |
|---|---|---|
| `chip_mod` | number | `hand_chips += value` |
| `mult_mod` | number | `mult += value` |
| `Xmult_mod` | number | `mult *= value` |

### In `get_edition` returns:
| Field | Type | Applied as |
|---|---|---|
| `chip_mod` | number | `hand_chips += value` (Phase 9a, pre) |
| `mult_mod` | number | `mult += value` (Phase 9a, pre) |
| `x_mult_mod` | number | `mult *= value` (Phase 9d, post) |

**Key insight**: all additive effects (`+chips`, `+mult`) are applied before
multiplicative effects (`xMult`) within each phase. But xMult from Phase 7
(per-card) compounds with xMult from Phase 9 (per-joker). Joker order
(left to right) matters for xMult stacking.

---

## Source File Reference

| Function | File | Line |
|----------|------|------|
| `play_cards_from_highlighted` | `state_events.lua` | 450 |
| `evaluate_play` | `state_events.lua` | 571 |
| `get_poker_hand_info` | `state_events.lua` | 540 |
| `draw_from_play_to_discard` | `state_events.lua` | 1088 |
| `evaluate_poker_hand` | `misc_functions.lua` | 376 |
| `get_flush` | `misc_functions.lua` | 522 |
| `get_straight` | `misc_functions.lua` | 548 |
| `get_X_same` | `misc_functions.lua` | 592 |
| `mod_chips` | `misc_functions.lua` | 684 |
| `mod_mult` | `misc_functions.lua` | 691 |
| `eval_card` | `common_events.lua` | 580 |
| `card_eval_status_text` | `common_events.lua` | 779 |
| `Card:get_chip_bonus` | `card.lua` | 976 |
| `Card:get_chip_mult` | `card.lua` | 984 |
| `Card:get_chip_x_mult` | `card.lua` | 999 |
| `Card:get_chip_h_mult` | `card.lua` | 1006 |
| `Card:get_chip_h_x_mult` | `card.lua` | 1011 |
| `Card:get_edition` | `card.lua` | 1016 |
| `Card:get_p_dollars` | `card.lua` | 1068 |
| `Card:calculate_joker` | `card.lua` | 2291 |
| `Card:calculate_seal` | `card.lua` | 2242 |
| `Blind:debuff_hand` | `blind.lua` | 519 |
| `Blind:modify_hand` | `blind.lua` | 510 |
| `Blind:press_play` | `blind.lua` | 464 |
| `Back:trigger_effect` | `back.lua` | 108 |

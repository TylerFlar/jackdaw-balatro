# Joker System Architecture

How joker abilities are defined, stored, evaluated, and categorized.

---

## How Joker Abilities Are Defined

There are **no external data files**. All 150 joker prototypes are defined inline
in `Game:init_item_prototypes()` (`game.lua:216`). Each entry goes into `G.P_CENTERS`
keyed as `j_<name>`.

### Prototype Structure (the "center")

```lua
G.P_CENTERS.j_greedy_joker = {
    name = "Greedy Joker",
    set = "Joker",
    pos = {x = 6, y = 0},
    order = 7,
    rarity = 1,             -- 1=Common, 2=Uncommon, 3=Rare, 4=Legendary
    cost = 5,
    effect = "Suit Mult",   -- category hint for calculate_joker dispatch
    config = {extra = {s_mult = 3, suit = 'Diamonds'}},
    blueprint_compat = true,
    perishable_compat = true,
    eternal_compat = true,
}
```

| Prototype field | Type | Description |
|---|---|---|
| `name` | string | Display name |
| `set` | string | Always `"Joker"` |
| `pos` / `soul_pos` | `{x,y}` | Spritesheet coords (soul_pos for legendaries) |
| `order` | int | Sort position (1-150) |
| `rarity` | int | 1=Common, 2=Uncommon, 3=Rare, 4=Legendary |
| `cost` | int | Base shop price |
| `effect` | string | Dispatch hint: `"Mult"`, `"Suit Mult"`, `"Type Mult"`, etc. Many are `""` |
| `config` | table | **Numeric ability values** — the payload (see below) |
| `blueprint_compat` | bool | Can be copied by Blueprint/Brainstorm |
| `perishable_compat` | bool | Can receive Perishable sticker |
| `eternal_compat` | bool | Can receive Eternal sticker |
| `unlock_condition` | table/nil | e.g., `{type = 'c_losses', extra = 5}` |
| `enhancement_gate` | string/nil | Only appears in pool if enhancement discovered |
| `no_pool_flag` / `yes_pool_flag` | string/nil | Pool inclusion gated by `G.GAME.pool_flags` |

### Config Patterns

The `config` table shape varies per joker:

**1. Flat fields** mapping directly to `self.ability`:
```lua
config = {mult = 4}                    -- Joker (flat mult)
config = {t_mult = 8, type = 'Pair'}   -- Jolly Joker (hand-conditional mult)
config = {Xmult = 2}                   -- Ramen (note: uppercase X)
config = {h_size = 1}                  -- Juggler (hand size)
config = {d_size = 1}                  -- Drunkard (discard count)
```

**2. `extra` as a single value** (interpreted per-joker in `calculate_joker`):
```lua
config = {extra = 8}                   -- Fibonacci (mult per card)
config = {extra = 30}                  -- Banner (chips per discard remaining)
```

**3. `extra` as a sub-table** (complex jokers):
```lua
config = {extra = {s_mult = 3, suit = 'Diamonds'}}    -- Greedy Joker
config = {extra = {chips = 0, chip_mod = 15}}          -- Runner (accumulator)
config = {extra = {Xmult = 4, every = 5}}              -- Loyalty Card
config = {extra = {odds = 6, mult = 15}}               -- Gros Michel
```

**4. Empty config** (purely behavioral, no numeric tuning):
```lua
config = {}                            -- Four Fingers, DNA, Pareidolia
```

---

## How Config Becomes self.ability

`Card:set_ability(center)` (`card.lua:223`) destructures the prototype config
into a flat `self.ability` table:

```lua
self.ability = {
    name    = center.name,
    effect  = center.effect,          -- dispatch category string
    set     = center.set,             -- "Joker"
    mult    = center.config.mult or 0,
    h_mult  = center.config.h_mult or 0,
    h_x_mult= center.config.h_x_mult or 0,
    h_dollars= center.config.h_dollars or 0,
    p_dollars= center.config.p_dollars or 0,
    t_mult  = center.config.t_mult or 0,
    t_chips = center.config.t_chips or 0,
    x_mult  = center.config.Xmult or 1,  -- NOTE: Xmult → x_mult
    h_size  = center.config.h_size or 0,
    d_size  = center.config.d_size or 0,
    bonus   = (old_bonus or 0) + (center.config.bonus or 0),
    extra   = copy_table(center.config.extra),  -- deep copy
    extra_value = 0,
    type    = center.config.type or '',
    order   = center.order,
    perma_bonus = <preserved>,          -- survives center changes
}
```

### self.ability Field Glossary

| Field | Type | Meaning |
|---|---|---|
| `mult` | number | Flat additive mult (Joker=4, Green Joker accumulates here) |
| `t_mult` / `t_chips` | number | Hand-type conditional mult/chips (Jolly=8, Sly=50) |
| `x_mult` | number | Multiplicative mult. Config key `Xmult` maps to ability key `x_mult`. |
| `h_size` / `d_size` | number | Hand size / discard count modifiers |
| `h_mult` / `h_x_mult` | number | Held-in-hand mult / x_mult |
| `bonus` / `perma_bonus` | number | Additive chip bonuses (perma survives center changes) |
| `extra` | any | Deep-copied grab bag — single number or sub-table, joker-specific |
| `extra_value` | number | Runtime accumulator (Egg sell value growth) |
| `effect` | string | Category hint for `calculate_joker` branching |
| `type` | string | Poker hand type condition (`'Pair'`, `'Flush'`, etc.) |

### Special post-init fields (card.lua:308-337)

Some jokers get additional fields after the main assignment:
- **Invisible Joker**: `self.ability.invis_rounds = 0`
- **To Do List**: `self.ability.to_do_poker_hand = <random hand>`
- **Caino**: `self.ability.caino_xmult = 1`
- **Yorick**: `self.ability.yorick_discards = self.ability.extra.discards`
- **Loyalty Card**: `self.ability.loyalty_remaining = self.ability.extra.every`

All cards get: `self.ability.hands_played_at_create = G.GAME.hands_played`

---

## How Joker State Is Persisted

Joker state lives on `self.ability` and is saved/loaded via `Card:save()` / `Card:load()`:

- `Card:save()` (`card.lua:4625`) serializes `self.ability` as-is into the save table
- `Card:load()` (`card.lua:4659`) restores `self.ability` from the save table
- Mutable fields (`mult`, `x_mult`, `extra.chips`, `caino_xmult`, `invis_rounds`,
  `yorick_discards`, etc.) survive save/load because they're part of `self.ability`
- The deep-copied `extra` sub-table is saved with its current runtime values

**No separate state system** — joker runtime state is just mutated fields on
`self.ability`. There's no event log, no accumulator pattern, just direct mutation.

---

## Joker Evaluation Structure

### The Dispatch Function

`Card:calculate_joker(context)` (`card.lua:2291-4063`) is a single ~1,770-line
method on the Card class. It's a massive `if/elseif` chain, not a dispatch table.

```lua
function Card:calculate_joker(context)
    if self.debuff then return nil end    -- debuffed jokers do nothing

    if self.ability.set == "Planet" then
        -- Observatory voucher effect
    elseif self.ability.set == "Joker" then
        -- Blueprint/Brainstorm copy logic first
        -- Then context-based dispatch
    end
end
```

**There is no registry, no lookup table, no plugin system.** Each joker's logic
is an inline `if self.ability.name == "Joker Name"` block within the appropriate
context branch. Adding a new joker means adding a new `if` block.

### Context Object

Every call to `calculate_joker` receives a `context` table. The context keys
determine which phase is being evaluated:

```lua
context = {
    -- Phase flags (exactly one is typically set):
    before = true,              -- pre-scoring
    after = true,               -- post-scoring
    joker_main = true,          -- main scoring pass (implicit: none of the above)
    individual = true,          -- per-card evaluation
    repetition = true,          -- retrigger check
    end_of_round = true,        -- end of round
    setting_blind = true,       -- blind being set
    first_hand_drawn = true,    -- first cards drawn
    discard = true,             -- cards being discarded
    destroying_card = true,     -- card being destroyed during scoring
    selling_self = true,        -- this joker being sold
    selling_card = true,        -- any card being sold
    open_booster = true,        -- booster pack opened
    skip_blind = true,          -- blind skipped
    reroll_shop = true,         -- shop rerolled
    ending_shop = true,         -- leaving shop
    debuffed_hand = true,       -- hand was debuffed by blind
    other_joker = <Card>,       -- joker-on-joker interaction
    playing_card_added = true,  -- new card added to deck

    -- Context data:
    cardarea = G.play/G.hand/G.jokers,  -- where the card is
    other_card = <Card>,        -- the specific card being evaluated
    full_hand = G.play.cards,   -- all played cards
    scoring_hand = {...},       -- scoring subset
    scoring_name = "Flush",     -- hand type name
    poker_hands = {...},        -- all detected hand types
    blueprint = <int>,          -- copy depth (prevents infinite Blueprint loops)
    blueprint_card = <Card>,    -- the copying joker
}
```

### Return Value Format

`calculate_joker` returns a table (or `nil` for no effect). The recognized fields
depend on the phase:

**Individual context (per-card scoring):**
| Field | Type | Applied as |
|---|---|---|
| `chips` | number | `hand_chips += value` |
| `mult` | number | `mult += value` |
| `x_mult` | number | `mult *= value` |
| `dollars` | number | Money earned |
| `card` | Card | Source card (for display) |
| `extra` | table | `{message, colour, func, mult_mod, chip_mod, swap}` |

**Joker main context:**
| Field | Type | Applied as |
|---|---|---|
| `chip_mod` | number | `hand_chips += value` |
| `mult_mod` | number | `mult += value` |
| `Xmult_mod` | number | `mult *= value` |
| `message` | string | Display text |
| `colour` | color | Message color |
| `card` | Card | Source card |
| `dollars` | number | Money earned |

**Repetition context:**
| Field | Type | Meaning |
|---|---|---|
| `repetitions` | number | How many extra times to evaluate the card |
| `message` | string | Usually "Again!" |
| `card` | Card | Source joker |

**Other contexts:**
| Field | Type | Meaning |
|---|---|---|
| `level_up` | bool | Level up the poker hand (before phase) |
| `saved` | bool | Prevent game over (Mr. Bones) |
| `remove` | bool | Destroy the card (destroying_card phase) |

**Note the naming inconsistency**: individual context uses `chips`/`mult`/`x_mult`,
joker_main uses `chip_mod`/`mult_mod`/`Xmult_mod`. This matters for the
application logic in `evaluate_play`.

---

## All Phases a Joker Can Hook Into

| Phase | When it fires | Example jokers |
|---|---|---|
| `before` | Before per-card scoring | Space Joker (level up), DNA (copy card), Green Joker (+mult), Vampire (strip enhancements) |
| `individual` + `cardarea=G.play` | Once per scored card per retrigger | Greedy Joker (+mult per Diamond), Fibonacci (+mult per 2/3/5/8/A), Hiker (+perma chips) |
| `individual` + `cardarea=G.hand` | Once per held card per retrigger | Baron (xMult per King held), Shoot the Moon (+mult per Queen) |
| `repetition` + `cardarea=G.play` | Retrigger check per scored card | Sock and Buskin, Hanging Chad, Dusk, Hack, Seltzer |
| `repetition` + `cardarea=G.hand` | Retrigger check per held card | Mime |
| `joker_main` (default) | Main scoring pass, left to right | Joker (+4 mult), Steel Joker (xMult), Abstract (+3 per joker) |
| `other_joker` | Per other joker in main pass | Baseball Card (xMult for Uncommons) |
| `after` | After all scoring | Ice Cream (decay), Seltzer (decay), Mr. Bones (save) |
| `setting_blind` | When blind is set | Chicot (disable boss), Madness (xMult + destroy), Burglar (swap hands/discards) |
| `first_hand_drawn` | First cards drawn each round | Certificate (add card), DNA (setup), Trading Card (setup) |
| `end_of_round` | End of round | Campfire (reset), Rocket (grow), Egg (sell value), Gift Card (buff all) |
| `discard` | Per discarded card | Castle (+chips), Mail-In Rebate ($), Green Joker (-mult), Ramen (decay) |
| `pre_discard` | Before discard resolves | Burnt Joker (level up discarded hand) |
| `destroying_card` | Card being destroyed in scoring | Sixth Sense (create Spectral) |
| `cards_destroyed` / `remove_playing_cards` | After cards destroyed | Caino (+xMult per face), Glass Joker (+xMult per glass) |
| `selling_self` | This joker sold | Luchador (disable boss), Diet Cola (Double Tag), Invisible (dupe joker) |
| `selling_card` | Any card sold | Campfire (+xMult) |
| `open_booster` | Booster pack opened | Hallucination (create Tarot) |
| `skip_blind` | Blind skipped | Throwback (+xMult) |
| `reroll_shop` | Shop rerolled | Flash Card (+mult) |
| `ending_shop` | Leaving shop | Perkeo (copy consumable) |
| `debuffed_hand` | Hand debuffed by boss | Matador (earn $8) |
| `using_consumeable` | Consumable used | Constellation (+xMult), Fortune Teller (display) |
| `playing_card_added` | Card added to deck | Hologram (+xMult) |

---

## Complete Joker Categorization

### SIMPLE — Flat bonus, no conditions

| Joker | Phase | Effect |
|---|---|---|
| Joker | main | +4 mult |
| Misprint | main | +random(0-23) mult |
| Stuntman | main | +250 chips |

### CONDITIONAL — Bonus when condition met

#### By hand type:
| Joker | Condition | Effect |
|---|---|---|
| Jolly Joker | Pair | +8 mult |
| Zany Joker | Three of a Kind | +12 mult |
| Mad Joker | Two Pair | +10 mult |
| Crazy Joker | Straight | +12 mult |
| Droll Joker | Flush | +10 mult |
| Sly Joker | Pair | +50 chips |
| Wily Joker | Three of a Kind | +100 chips |
| Clever Joker | Two Pair | +80 chips |
| Devious Joker | Straight | +100 chips |
| Crafty Joker | Flush | +80 chips |
| The Duo | Pair in hand | xMult |
| The Trio | Three of a Kind in hand | xMult |
| The Family | Four of a Kind in hand | xMult |
| The Order | Straight in hand | xMult |
| The Tribe | Flush in hand | xMult |
| Supernova | (any) | +mult = times played |
| Card Sharp | Same hand twice this round | xMult |

#### By card property (per scored card):
| Joker | Condition | Effect |
|---|---|---|
| Greedy Joker | Diamond | +3 mult |
| Lusty Joker | Heart | +3 mult |
| Wrathful Joker | Spade | +3 mult |
| Gluttonous Joker | Club | +3 mult |
| Scary Face | Face card | +30 chips |
| Smiley Face | Face card | +5 mult |
| Scholar | Ace | +20 chips, +4 mult |
| Walkie Talkie | 10 or 4 | +10 chips, +4 mult |
| Fibonacci | 2/3/5/8/Ace | +8 mult |
| Even Steven | Even (2-10) | +4 mult |
| Odd Todd | Odd + Ace | +31 chips |
| Photograph | First face card | xMult |
| The Idol | Matching suit+rank | xMult |
| Bloodstone | Heart (prob) | xMult |
| Ancient Joker | Round suit | xMult |
| Triboulet | King or Queen | x2 mult |
| Arrowhead | Spade | +50 chips |
| Onyx Agate | Club | +7 mult |

#### By card property (per held card):
| Joker | Condition | Effect |
|---|---|---|
| Shoot the Moon | Queen | +13 mult |
| Baron | King | x1.5 mult |
| Raised Fist | Lowest rank | +2×rank mult |
| Reserved Parking | Face (prob) | +$1 |

#### By game state:
| Joker | Condition | Effect |
|---|---|---|
| Half Joker | ≤3 cards played | +20 mult |
| Abstract Joker | (always) | +3 mult per joker |
| Acrobat | Last hand of round | x3 mult |
| Mystic Summit | 0 discards left | +15 mult |
| Banner | (always) | +30 chips per discard left |
| Blue Joker | (always) | +2 chips per deck card |
| Erosion | (always) | +4 mult per card below starting deck |
| Stone Joker | (always) | +25 chips per Stone Card in deck |
| Steel Joker | (always) | x(1+0.2×steel count) mult |
| Bull | (always) | +2 chips per $1 held |
| Driver's License | ≥16 enhanced cards | x3 mult |
| Blackboard | All held cards black | x3 mult |
| Joker Stencil | (always) | x1 per empty joker slot |
| Flower Pot | All 4 suits in hand | xMult |
| Seeing Double | Club + other suit | xMult |
| Bootstraps | (always) | +2 mult per $5 held |
| Fortune Teller | (always) | +mult = tarot usage total |
| Loyalty Card | Every N+1 hands | xMult |
| Matador | Blind triggered | +$8 |

#### Conditional side-effect:
| Joker | Condition | Side Effect |
|---|---|---|
| 8 Ball | 8 played (prob) | Create Tarot |
| Sixth Sense | Single 6 on first hand | Create Spectral, destroy 6 |
| Vagabond | $4 or less | Create Tarot |
| Superposition | Ace + Straight | Create Tarot |
| Seance | Matching hand type | Create Spectral |
| DNA | First hand, single card | Copy card to deck |
| Burnt Joker | First discard | Level up discarded hand type |

#### Conditional economy:
| Joker | Condition | Money |
|---|---|---|
| Golden Ticket | Gold Card played | +$1 per |
| Business Card | Face (prob) | +$2 |
| Rough Gem | Diamond played | +$1 |
| Faceless Joker | ≥3 face discarded | +$5 |
| Mail-In Rebate | Matching rank discard | +$3 |
| To Do List | Matching hand type | +$4 |
| Trading Card | First discard, 1 card | +$3, destroy card |

### SCALING — Mutable state across hands/rounds

| Joker | State field | How it grows | How it shrinks/resets |
|---|---|---|---|
| Green Joker | `ability.mult` | +1 per hand played | -1 per discard |
| Ride the Bus | `ability.mult` | +1 per hand with no face cards | Resets to 0 when face card scored |
| Spare Trousers | `ability.mult` | +2 per Two Pair/Full House | Never |
| Runner | `ability.extra.chips` | +15 per Straight | Never |
| Square Joker | `ability.extra.chips` | +4 per hand with exactly 4 cards | Never |
| Ice Cream | `ability.extra.chips` | Never (starts at 100) | -5 per hand; self-destructs at 0 |
| Popcorn | `ability.mult` | Never (starts at 20) | -4 per round; self-destructs at 0 |
| Flash Card | `ability.mult` | +2 per shop reroll | Never |
| Red Card | `ability.mult` | +3 per booster skipped | Never |
| Castle | `ability.extra.chips` | +3 per matching-suit discard | Resets when suit changes |
| Wee Joker | `ability.extra.chips` | +8 per 2 scored | Never |
| Lucky Cat | `ability.x_mult` | +0.25 per Lucky Card trigger | Never |
| Campfire | `ability.x_mult` | +0.25 per card sold | Resets to 1 after boss blind |
| Hologram | `ability.x_mult` | +0.25 per card added to deck | Never |
| Constellation | `ability.x_mult` | +0.1 per Planet used | Never |
| Glass Joker | `ability.x_mult` | +0.75 per Glass Card destroyed | Never |
| Caino | `ability.caino_xmult` | +1 per face card destroyed | Never |
| Vampire | `ability.x_mult` | +0.1 per enhancement stripped | Never |
| Obelisk | `ability.x_mult` | +0.2 per hand that isn't most played | Resets to 1 when most played |
| Madness | `ability.x_mult` | +0.5 per non-boss blind set (destroys a random joker) | Never |
| Hit the Road | `ability.x_mult` | +0.5 per Jack discarded | Resets to 1 end of round |
| Ceremonial Dagger | `ability.mult` | +2× sell cost when next joker destroyed | Never |
| Ramen | `ability.x_mult` | Never (starts at 2) | -0.01 per discard; self-destructs at ≤1 |
| Yorick | `ability.x_mult` | +1 every 23 discards | Never (counter: yorick_discards) |
| Swashbuckler | `ability.mult` | Sum of other jokers' sell values | Recalculated |
| Throwback | `ability.x_mult` | +0.25 per blind skipped (total) | Never (reads G.GAME.skips) |
| Invisible Joker | `ability.invis_rounds` | +1 per round | Triggers at threshold |
| Rocket | `ability.extra.dollars` | +2 per boss beaten | Never |
| Turtle Bean | `ability.extra.h_size` | Never (starts at +5) | -1 per round; self-destructs at 0 |
| Seltzer | `ability.extra` | Never (starts at 10) | -1 per hand; self-destructs at 0 |
| Egg | `ability.extra_value` | +3 per round | Never (affects sell price) |

### RETRIGGER — Provides extra card evaluations

| Joker | Target | Condition | Reps |
|---|---|---|---|
| Sock and Buskin | Scored face cards | Always | +1 |
| Hanging Chad | First scored card | Always | +2 |
| Dusk | All scored cards | Last hand only | +1 |
| Seltzer | All scored cards | Has uses remaining | +1 |
| Hack | Scored 2/3/4/5 | Always | +1 |
| Mime | All held cards | Card had an effect | +1 |

(Red Seal provides +1 retrigger via `Card:calculate_seal`, not a joker)

### COPY — Delegates to another joker

| Joker | Target | Mechanism |
|---|---|---|
| Blueprint | Joker to the right | Calls `other_joker:calculate_joker(context)` with `context.blueprint` counter |
| Brainstorm | Leftmost joker | Same mechanism |
| Perkeo | Random consumable | Creates Negative copy when leaving shop |

Blueprint/Brainstorm use `context.blueprint` (incremented each delegation) to
prevent infinite loops when copying each other.

### SIDE_EFFECT — Mutates game state beyond scoring

**Create cards:**
| Joker | Creates |
|---|---|
| Certificate | Playing card with random seal (first hand drawn) |
| Marble Joker | Stone Card added to deck (each blind) |
| DNA | Copy of played card into deck (first hand) |
| Riff-raff | Up to 2 Common jokers (non-boss blind) |
| Cartomancer | Tarot card (each blind) |
| 8 Ball | Tarot card (probabilistic) |
| Vagabond | Tarot card (when poor) |
| Superposition | Tarot card (Ace + Straight) |
| Seance | Spectral card (matching hand type) |
| Sixth Sense | Spectral card (6 played) |
| Hallucination | Tarot card (booster opened, probabilistic) |
| Perkeo | Copy of random consumable (Negative edition) |

**Destroy cards/jokers:**
| Joker | Destroys |
|---|---|
| Madness | Random joker (setting blind) |
| Ceremonial Dagger | Adjacent joker (setting blind) |
| Trading Card | Discarded card (first discard) |

**Modify game rules:**
| Joker | Effect |
|---|---|
| Chicot | Disables boss blind |
| Luchador | Disables boss blind (when sold) |
| Burglar | Swaps hands and discards for the round |
| Midas Mask | Turns face cards to Gold Cards (before scoring) |
| Vampire | Strips enhancements from scored cards |
| Hiker | Adds permanent +5 chip bonus to scored cards |

### META — Passive rule modifications (not in calculate_joker)

These jokers have no `calculate_joker` logic. Their effects are checked inline
by other systems:

| Joker | System | Effect |
|---|---|---|
| Four Fingers | `get_flush`, `get_straight` | Flush/Straight need 4 cards |
| Shortcut | `get_straight` | Straights allow 1-rank gaps |
| Pareidolia | `Card:is_face` | All cards are face cards |
| Smeared Joker | `Card:is_suit` | Hearts=Diamonds, Spades=Clubs |
| Splash | `evaluate_play` | All played cards score |
| Showman | `get_current_pool` | Allows duplicate jokers in shop |
| Juggler | `Card:add_to_deck` | +1 hand size (via `h_size` field) |
| Drunkard | `Card:add_to_deck` | +1 discard (via `d_size` field) |
| Troubadour | `Card:add_to_deck` | +2 hand size, -1 hand per round |
| Merry Andy | `Card:add_to_deck` | +3 discards, -1 hand size |
| Oops! All 6s | `G.GAME.probabilities.normal` | Doubles all probabilities |
| Credit Card | `G.GAME.bankrupt_at` | Allows -$20 debt |
| Chaos the Clown | `update_shop` | 1 free reroll per shop |
| Astronomer | `Card:set_cost` | Planet cards cost $0 |

### ECONOMY — Primarily about money

| Joker | Phase | Income |
|---|---|---|
| Golden Joker | end_of_round | +$4 |
| Cloud 9 | end_of_round | +$1 per 9 in full deck |
| Satellite | end_of_round | +$1 per unique Planet used |
| To the Moon | end_of_round | +$1 extra interest per $5 |
| Delayed Gratification | end_of_round | +$2 per remaining discard |
| Rocket | end_of_round (boss) | +$1, grows +$2 per boss |
| Egg | end_of_round | +$3 sell value |
| Gift Card | end_of_round | +$1 sell value to all jokers/consumables |

---

## Category Counts

| Category | Count | % |
|---|---|---|
| Conditional | ~70 | 47% |
| Scaling | ~31 | 21% |
| Side Effect | ~25 | 17% |
| Economy | ~18 | 12% |
| Meta (passive) | ~17 | 11% |
| Retrigger | 6 | 4% |
| Simple | 3 | 2% |
| Copy | 3 | 2% |

(Jokers can be in multiple categories; percentages don't sum to 100%)

---

## Architectural Summary

1. **No dispatch table** — `calculate_joker` is one monolithic `if/elseif` chain
   branching on `context` flags and `self.ability.name`/`self.ability.effect`.

2. **No ability classes** — every joker is an instance of `Card`. The only
   distinction is the values on `self.ability` and the matching branch in
   `calculate_joker`.

3. **State lives on self.ability** — mutable joker state (accumulated mult,
   xMult, chip counters) is stored as direct mutations to `self.ability` fields.
   These persist through save/load because `Card:save()` serializes the entire
   `ability` table.

4. **Deep copy isolates instances** — `config.extra` is `copy_table`'d in
   `set_ability`, so each joker card has its own copy of mutable state.

5. **Order matters** — jokers are evaluated left-to-right in Phase 9 of scoring.
   xMult compounds multiplicatively, so position affects total score.

6. **Blueprint loop prevention** — `context.blueprint` is a counter incremented
   each delegation. The copy jokers check this to avoid infinite recursion.

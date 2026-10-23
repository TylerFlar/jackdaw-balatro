# Steamodded and BalatroBot deviation catalog

The Rust port and the parity ledger follow vanilla Balatro 1.0.1o (`balatro_source/`). The live
game that jackdaw is checked against is not vanilla: it runs Lovely, Steamodded (SMODS) and
BalatroBot. Lockstep runs and the `LS…`/`ES…` seeds cited in the Python engine were captured on that
modded game. This catalog lists every place where the mods change what happens in a run. When Rust,
Python and a live capture disagree, look here first. An entry means the live game is not vanilla at
that point, so a live capture cannot settle the question there.

A ledger row whose unit is affected gets the status `smods-diff`, with the entry id (for example
`S01`) in its `notes`. See [README.md](README.md).

## Scope and method

What was reviewed (installed under `%APPDATA%/Balatro/Mods/`; only these three mods are present):

- **Lovely 0.9.0.**
- **Steamodded `1.0.0~BETA-1503a-STEAMODDED`** (`smods/version.lua`):
  - Every lovely patch in `smods/lovely/*.toml`: 67 files and 940 patches (774 pattern, 152 regex,
    12 module, 2 copy).
  - The Lua the game actually runs: Lovely's dump (`Mods/lovely/dump/`, with its `*.json`
    patch-region maps), diffed file by file against `balatro_source/`.
  - Steamodded's own code: `src/overrides.lua`, `src/utils.lua` (the calculation pipeline, card
    limits, polls) and `src/game_object.lua` (scoring parameters, stickers, boosters, suits and
    ranks, and every `take_ownership` of a vanilla object).
- **BalatroBot.** The installed manifest says 1.4.0, but its endpoint Lua matches the vendored
  [`third_party/balatrobot/v1.5.2`](../../third_party/balatrobot/v1.5.2/VENDORED.md). Only
  `core/server.lua`, `endpoints/load.lua` and `utils/openrpc.json` differ, plus a debug shuffle
  hook in the installed `balatrobot.lua` (see B04). Reviewed: the entry file, `settings.lua`, every
  endpoint and the gamestate builder.
- **The Python oracle** at `port/rust-core` `c494415`, cross-referenced for every entry.

Patch numbers (`#n`) are the 1-based position of a `[[patches]]` block in its toml file. Appendix A
classifies all 940 patches, and Appendix B lists the ones that did not apply.

## How to read an entry

Each entry gives:

- **SMODS**: the toml file, patch number, `target` and a short excerpt of the `pattern`, plus any
  `src/` lines involved.
- **Vanilla**: `file:line` in `balatro_source/`. The source is proprietary, so it is quoted only as
  identifiers or short conditions. Steamodded and BalatroBot code is quoted from their own files.
- **Difference**, **scenario**, **severity** and **Python** status.

Severities:

| severity | meaning |
|---|---|
| RNG-desync | a seeded stream is drawn a different number of times, so later draws on that stream differ |
| state-difference | same RNG use, but a different game state or outcome (cards, money, score, slots, which card is picked) |
| cosmetic-in-practice | differs in principle, but not observably with vanilla content under bot play |

"Python follows" is one of `vanilla`, `SMODS`, `neither` or `n/a` (not modelled).

## Summary

| id | area | what differs in the live game | severity | Python follows |
|---|---|---|---|---|
| [S01](#s01) | blinds, deck | Marble Joker's Stone card joins the deck after the round's `'nr'` shuffle, at the bottom | state-difference | vanilla (G55) |
| [S02](#s02) | round start | `first_hand_drawn` (Certificate) runs after `Blind:drawn_to_hand` (Cerulean Bell), once per round | state-difference | vanilla |
| [S03](#s03) | jokers | Certificate's seal roll also draws once from `'stdseal'..ante` | RNG-desync | vanilla |
| [S04](#s04) | hand size | hand-size changes apply at once and the hand refills to the limit; The Manacle's disable draws 1 card, not 2 | state-difference | vanilla (engine); SMODS (SimBackend `add`) |
| [S05](#s05) | copies | `copy_card` prices the copy before copying its abilities; a copy stripped of Negative keeps the +1 slot | state-difference | vanilla |
| [S06](#s06) | jokers | Seeing Double counts one Wild Card as every suit at once | state-difference | vanilla |
| [S07](#s07) | hand eval | `get_flush` tries suits as Diamonds, Clubs, Hearts, Spades | state-difference | vanilla |
| [S08](#s08) | spectrals | Sigil/Grim/Familiar/Incantation suit order (Clubs and Diamonds swapped); Familiar and Incantation roll suit before rank | state-difference | vanilla (Grim: neither) |
| [S09](#s09) | round end | perishable/rental stickers run before the joker's end-of-round effect; debuffed jokers do not tick perishable | state-difference | vanilla |
| [S10](#s10) | scoring | ×Mult is applied as `m + m*(X-1)` instead of `m*X` (last-bit differences) | state-difference | vanilla |
| [S11](#s11) | RNG | `Card:init` no longer draws 3 unseeded `math.random()` | cosmetic-in-practice | n/a |
| [S12](#s12) | RNG | `pseudoseed` returns unseeded `math.random()` while the game is paused (except `'to_do'`) | cosmetic-in-practice | n/a |
| [S13](#s13) | blinds | Crimson Heart with no joker left to pick: SMODS re-picks, vanilla errors | cosmetic-in-practice | neither |
| [S14](#s14) | deck | `G.deck.config.card_limit` follows `#G.playing_cards` every frame, up as well as down | cosmetic-in-practice | n/a |
| [B01](#b01) | BalatroBot `pack` | pack picks skip `can_use_consumeable`/`can_select_card` | state-difference | vanilla |
| [B02](#b02) | BalatroBot `play`/`discard` | with Cerulean Bell, the forced card is added and the last requested card is dropped | state-difference | vanilla |
| [B03](#b03) | BalatroBot `add` (debug) | no edition poll when an edition is given; the hand refills to the new limit | RNG-desync | neither / SMODS |
| [B04](#b04) | BalatroBot runtime | fixed `dt`, game speed, debug hooks | no gameplay effect | n/a |

Changes that were examined and turned out to keep vanilla outcomes are listed under
[Examined and found equivalent](#examined-and-found-equivalent).

## Steamodded entries

<a id="s01"></a>
### S01: Marble Joker's Stone card joins the deck after the round shuffle

- **SMODS:** `lovely/better_calc.toml` #75, "Fix Marble Joker to send the created card to
  playing_card_added correctly". Target `card.lua`; the pattern is the whole Marble block, which
  begins `G.E_MANAGER:add_event(Event({` and contains
  `local front = pseudorandom_element(G.P_CARDS, pseudoseed('marb_fr'))`.
- **Vanilla:**
  - `card.lua:2580-2601`: at `setting_blind`, Marble queues four things:
    1. an event that creates the Stone in `G.play`;
    2. the status text;
    3. an event that raises the deck limit;
    4. `draw_card(G.play, G.deck, ...)`, itself an event.

    It also calls `playing_card_joker_effects({true})` straight away.
  - `functions/state_events.lua:335-348`: `new_round` runs the jokers' `setting_blind` loop, then
    `delay(0.4)`, then queues the event that sets `DRAW_TO_HAND` and calls
    `G.deck:shuffle('nr'..ante)` (line 344).
  - `cardarea.lua:32-34`: a deck emplace inserts at index 1, the bottom of the pile.
- **Difference:** in vanilla all four Marble events sit in the queue ahead of the shuffle event, so
  the Stone is already in `G.deck` when it is shuffled: 53 cards, sorted by `sort_id` and then
  Fisher-Yates. Steamodded makes one event that creates the Stone and only then queues the move to
  the deck:

  ```lua
  G.E_MANAGER:add_event(Event({
      func = function()
          local card = create_playing_card({
              front = pseudorandom_element(G.P_CARDS, pseudoseed('marb_fr')),
              center = G.P_CENTERS.m_stone}, G.play, nil, nil, {G.C.SECONDARY_SET.Enhanced})
          ...
          G.E_MANAGER:add_event(Event({
          func = function()
              draw_card(G.play,G.deck, 90,'up', nil)
              return true
          end}))
          playing_card_joker_effects({card})
  ```

  The inner event lands behind the shuffle event that is already queued. The shuffle sees 52 cards,
  and the Stone then enters at the bottom, so it is the last card drawn that round.
  `playing_card_joker_effects` also gets the real card inside the event, where vanilla passes
  `{true}` at once. Hologram's gain is the same, only later.
- **Scenario:** any blind selected with Marble Joker, or a Blueprint or Brainstorm copying it. The
  `'nr'..ante` seed is the same, but it permutes 52 cards instead of 53, so the whole deal order of
  that round differs. The Stone is never drawn before the last card. From the next round on, the
  Stone is shuffled normally.
- **Severity:** state-difference: the deck order of every round in which Marble fires.
- **Python: vanilla** since gap G55 was fixed.
  - It used to follow Steamodded: the Stone waited in `pending_deck_bottom` ("live-verified:
    LS86UJ9R") and joined the deck at index 0 after the `'nr'` shuffle.
  - `jackdaw/engine/game.py` (`_apply_setting_blind_mutations`) now puts the Stone in the deck
    before the shuffle, as Rust does. The snapshot keeps `sim.pending_deck_bottom`, always empty.
  - A live capture of a round in which Marble fires therefore differs from both engines.

<a id="s02"></a>
### S02: `first_hand_drawn` fires after the blind's `drawn_to_hand`

- **SMODS:** all in `lovely/better_calc.toml`:
  - #36 (target `game.lua`): the pattern starting
    `if G.GAME.current_round.hands_played == 0 and` / `...calculate_joker({first_hand_drawn = true})`
    is replaced by a comment.
  - #40 and #41 (target `functions/state_events.lua`, around
    `for i=1, hand_space do --draw cards from deckL`): add `SMODS.cards_to_draw` bookkeeping, plus a
    trailing event with `trigger = 'before'` and `delay = 0.4` that calls
    `SMODS.calculate_context({first_hand_drawn = not G.GAME.current_round.any_hand_drawn and G.GAME.facing_blind, ...})`.
  - #43 (pattern `G.GAME.current_round.discards_used = 0`): resets `any_hand_drawn` in `new_round`.
  - #37-#39: record the drawn cards.
  - #72: Certificate.
- **Vanilla:**
  - `game.lua:3208-3245` (`Game:update_draw_to_hand`): inside one event it calls
    `draw_from_deck_to_hand(nil)`, which queues the draws. Then, synchronously, it runs every
    joker's `first_hand_drawn` (lines 3226-3231, when `hands_played == 0`, `discards_used == 0` and
    `facing_blind`). Then it queues the event that sets `SELECTING_HAND` and calls
    `G.GAME.blind:drawn_to_hand()` (lines 3233-3241).
  - Certificate (`card.lua:2462-2482`) queues its card-creating event during that joker loop.
- **Difference:** the order of queued work changes.

  | step | vanilla | SMODS |
  |---|---|---|
  | 1 | the draws | the draws |
  | 2 | Certificate's card is made | `first_hand_drawn` event (Certificate queues its card only now) |
  | 3 | `SELECTING_HAND` + `drawn_to_hand` | `SELECTING_HAND` + `drawn_to_hand` |
  | 4 | — | Certificate's card is made |

  The trigger also changes. Steamodded fires once per round, via `any_hand_drawn`. Vanilla fires on
  every pass through `DRAW_TO_HAND` while no hand has been played or discarded.
- **Scenarios:**
  - Certificate plus Cerulean Bell. `blind.lua:574-587` picks the forced card with
    `pseudorandom_element(G.hand.cards, pseudoseed('cerulean_bell'))`. In vanilla that is over 9
    cards, and the Certificate card can be forced. In Steamodded it is over 8 cards. The same draw
    gives a different forced card.
  - Consumables destroy every card in hand before the first play. `game.lua:2544-2548` sends the
    game back to `DRAW_TO_HAND`, and vanilla's Certificate fires again (a second sealed card).
    Steamodded's does not.
  - DNA and Trading Card also use this context, but only for a juice animation.
- **Severity:** state-difference.
- **Python: vanilla.** `game.py:2073-2086` (`_draw_to_hand`) draws, then calls
  `_fire_first_hand_drawn` when `hands_played == 0 and discards_used == 0`, then sets
  `SELECTING_HAND` and runs `_blind_drawn_to_hand`. Gap G14 records the fix. Note that LSI5EB7T,
  "dealt 8 + a red-seal H_K as the 9th card", holds in both orders.

<a id="s03"></a>
### S03: Certificate's seal roll also draws from `'stdseal'..ante`

- **SMODS:**
  - `lovely/better_calc.toml` #72, "Fix Certificate to use SMODS.poll_seal...". Target `card.lua`;
    the pattern begins `G.E_MANAGER:add_event(Event({` / `local _card = create_playing_card({` /
    `front = pseudorandom_element(G.P_CARDS, pseudoseed('cert_fr'))`.
  - `src/utils.lua:673-719` (`SMODS.poll_seal`).
- **Vanilla:** `card.lua:2463-2479` makes one draw, `pseudorandom(pseudoseed('certsl'))`: above 0.75
  Red, above 0.5 Blue, above 0.25 Gold, otherwise Purple.
- **Difference:** Steamodded calls
  `_card:set_seal(SMODS.poll_seal({type_key = 'certsl', guaranteed = true}), nil, true)`.
  - `poll_seal` always draws `pseudorandom(pseudoseed(key))` first, with `key = 'stdseal'..ante`
    (`utils.lua:708`). `guaranteed` only skips the threshold.
  - Only then does it draw `'certsl'` for the type.
  - The type mapping is unchanged: `lovely/seal.toml` #11 reorders `G.P_SEALS`, so the pool runs
    Red, Blue, Gold, Purple with weight 5 each, which gives the thresholds 0.75, 0.5 and 0.25.
- **Scenario:** Certificate fires, then a Standard Pack is opened in the same ante. The pack's seal
  rolls use the same `'stdseal'..ante` stream: vanilla `card.lua:1763-1771`, and in Steamodded the
  Standard pack's `SMODS.poll_seal({mod = 10})`. That stream is one draw further along per
  Certificate trigger, so whether a pack card gets a seal, and which one, differs.
- **Severity:** RNG-desync on `'stdseal'..ante`.
- **Python: vanilla.** `game.py:2834-2845` draws only `'certsl'`.

<a id="s04"></a>
### S04: hand-size changes apply at once and the hand refills to the limit

- **SMODS:**
  - `lovely/card_limit.toml` #4, "Bypass CardArea:change_size()". Target `cardarea.lua`, pattern
    `function CardArea:change_size(delta)`; the inserted body is
    `if true then self.config.card_limits.mod = (self.config.card_limits.mod or 0) + delta return end`.
  - #8 (`CardArea:update`, pattern
    `self.config.temp_limit = math.max(#self.cards, self.config.card_limit)`): now calls
    `self:handle_card_limit()` every frame. See `src/utils.lua:3169-3206`.
  - #16 (`G.FUNCS.draw_from_deck_to_hand`, pattern `local hand_space = e or*`).
  - #17 and #18: The Serpent.
  - `lovely/fixes.toml` #25, "Fix Manacle's unnecessary card draw". Regex
    `G\.hand:change_size\(1\)(\s+G\.FUNCS\.draw_from_deck_to_hand\(1\))`.
- **Vanilla:**
  - `cardarea.lua:94-110`: `change_size` queues an event. When `delta > 0`, the limit is above 1,
    the hand is not empty and the state is `DRAW_TO_HAND` or `SELECTING_HAND`, it draws exactly
    `delta` cards.
  - `functions/state_events.lua:355-377`: `draw_from_deck_to_hand(e)` draws
    `e or min(#deck, limit - #hand)`. The Serpent's cap of 3 applies whenever it is called after the
    first play or discard.
  - `blind.lua:386-390`: disabling The Manacle calls `G.hand:change_size(1)` and then
    `G.FUNCS.draw_from_deck_to_hand(1)`, which is two draws mid-round.
- **Difference:**
  - In Steamodded, `change_size` only moves `card_limits.mod`: no event, no draw.
  - Every frame, `handle_card_limit` acts on `G.hand`:
    - In `SELECTING_HAND`, if the limit grew and the hand is below it, it calls
      `draw_from_deck_to_hand`, which now ignores its argument and draws up to the limit
      (`limit - #hand - SMODS.cards_to_draw`).
    - In `DRAW_TO_HAND`, it queues a top-up if the hand is still short (skipped for The Serpent).
  - Steamodded's `draw_from_deck_to_hand` picks the exact cards when it is called and passes them to
    `draw_card`.
  - The Serpent's cap applies only in `DRAW_TO_HAND`.
- **Scenarios:**
  - **A short hand gets a bigger limit.** The Hanged Man destroys 2 cards (6 left, limit 8), then
    Judgement creates Juggler (+1 hand size). Vanilla draws 1 card: 7 cards, limit 9. Steamodded
    fills the hand to 9.
  - **The Manacle disabled mid-round.** Luchador is sold, or The Soul creates Chicot (whose
    `add_to_deck` disables the boss), while the hand holds 7 of 7. Vanilla draws 2 cards: 9 cards in
    hand, limit 8. Steamodded draws 1 (8 cards).
  - In both cases the round draws the same total, but the cards available for the next play or
    discard differ.
  - No difference when Chicot disables The Manacle at `setting_blind`, before the deal: both deal to
    a full hand.
  - **Setting the blind shrinks the hand.** Madness destroys Turtle Bean (or a Juggler) at
    `setting_blind`. The card leaves after its dissolve, about a second later, and only then
    queues `change_size`: vanilla has dealt to the old limit by then (12 cards, limit 8).
    Steamodded's draw checks the limit while it deals and stops at the new one (8 cards). The
    lockstep's S04 also covers `select`.
- **No observable effect with vanilla content:** the card preselection, the `SMODS.cards_to_draw`
  bookkeeping (nothing reorders the deck between the call and the draw events), the `DRAW_TO_HAND`
  top-up, and The Serpent's restriction. Outside `DRAW_TO_HAND`, vanilla calls
  `draw_from_deck_to_hand` only for packs (no Serpent there) and for The Manacle.
- **A lost round.** The end of a lost round queues its hand-size changes (Turtle Bean's decay, a
  perishing Juggler's `remove_from_deck`) and then sets `GAME_OVER`; the game pauses in the same
  frame and never runs them (gaps.md G265). Steamodded has already applied them: the live view
  at the game over shows the smaller hand. The lockstep drops `hand.limit` there (S04-GAME-OVER).
- **Severity:** state-difference.
- **Python:**
  - **Engine: vanilla.** `game.py:2206-2226` (`change_hand_size`) draws exactly `delta`.
    `game.py:1491-1499` gives The Manacle's disable both draws.
  - **SimBackend: SMODS.** `jackdaw/bridge/backend.py:273-280` (`add`) tops the hand up to
    `hand_size` after `add_to_deck`: "matches balatrobot which immediately fills on add". See also
    B03.

<a id="s05"></a>
### S05: `copy_card` prices the copy before copying its abilities; a stripped Negative keeps its slot

- **SMODS:**
  - `lovely/fixes.toml` #54 and #55, "copy_card: Reorder timings". Target
    `functions/common_events.lua`; the `for k, v in pairs(other.ability) do` copy loop moves to just
    before `return new_card`.
  - #56, "copy_card: Subtract from .ability.card_limit".
  - `src/overrides.lua:2056-2205` (`Card:set_edition`). It returns early, without `set_cost`, when
    the card had no edition, and line 2198 adds the edition's `card_limit` to
    `ability.card_limit`.
- **Vanilla:**
  - `functions/common_events.lua:2156-2181`: `set_ability`, `set_base`, then a copy of every
    `other.ability` entry (lines 2161-2167), then `set_edition` (2169-2171) and `set_seal` (2173).
  - Both of those end in `set_cost` (`card.lua:461`, `:498`), so the price sees the copied
    `extra_value`, `rental` and `couponed` (`card.lua:369-386`).
  - A Negative's extra slot comes from `add_to_deck` (`card.lua:630-640`).
- **Difference:**
  - **Price.** Steamodded runs `set_edition` (no `set_cost` when there is no edition), then
    `set_seal`, whose `set_cost` sees a fresh ability, and copies the abilities only after that. The
    copy's `cost` and `sell_cost` ignore the source's `extra_value`, `rental` and `couponed` until
    something next calls `set_cost` on it.
  - **Slot.** A Negative's +1 slot lives in `ability.card_limit` under Steamodded. With
    `strip_edition`, the subtraction from #56 runs before the ability copy, and the copy restores
    `card_limit = 1`. The copy is not Negative, but `handle_card_limit` still counts its slot.
- **Scenarios:**
  - **Price.** Invisible Joker is sold while an Egg has grown to +$9 sell value. In vanilla the copy
    sells for the grown value. In Steamodded it shows the base value until a `set_cost` trigger:
    - Gift Card or Egg at round end,
    - Clearance Sale or Liquidation,
    - Astronomer joining or leaving.

    This affects Swashbuckler's mult and the money from selling. A rental source gives a $1-cost
    copy in vanilla and a normally priced one in Steamodded.
  - **Slot.** Ankh (`card.lua:1445`) or Invisible Joker (`card.lua:2384`) copies a Negative joker.
    In vanilla the copy is plain and takes a slot. In Steamodded it is plain but brings its own slot,
    so the joker limit is one higher. Every room check reads that limit (Riff-raff, Judgement, shop
    buys), so later pool draws can differ.
  - Perkeo is not affected: it calls `set_edition({negative = true})` after `copy_card`, so
    `set_cost` reruns on the full ability.
- **Severity:** state-difference.
- **Python: vanilla.** `jackdaw/engine/card_factory.py:801-880` copies the ability first and calls
  `set_cost` last. Copies made with `strip_edition` get no slot (gap G28).

<a id="s06"></a>
### S06: Seeing Double counts one Wild Card as every suit

- **SMODS:**
  - `lovely/fixes.toml` #59, "fix Seeing Double compat issues with modded suits". Target
    `card.lua`; the suit-tally block becomes
    `if SMODS.seeing_double_check(context.scoring_hand, 'Clubs') then`.
  - `src/utils.lua:2433-2458`.
- **Vanilla:** `card.lua:3840-3870`.
  - Non-Wild scoring cards add 1 to every suit they match.
  - Each Wild Card then fills only the first empty suit, in the order Clubs, Diamonds, Spades,
    Hearts (an `elseif` chain).
  - It triggers when Clubs > 0 and Hearts, Diamonds or Spades > 0.
- **Difference:** for each Wild Card, Steamodded loops over all suits with no break:

  ```lua
  for k, v in pairs(suit_tally) do
      if hand[i]:is_suit(k) and suit_tally[k] == 0  then suit_tally[k] = suit_tally[k] + 1 end
  end
  ```

  A single Wild Card fills every empty suit.
- **Scenario:** a scoring hand whose only suited, non-debuffed card is one Wild Card. Examples: a
  High Card of a Wild Card; a Wild Card with Stone cards; a pair where the other card is debuffed.
  Steamodded gives X2, vanilla nothing. Any hand with another suited scoring card behaves the same in
  both.
- **Severity:** state-difference (score).
- **Python: vanilla.** `jackdaw/engine/jokers.py:1268-1295`. Flower Pot (`card.lua:3807`) only had
  its Wild Card check swapped for `SMODS.has_any_suit` (`lovely/enhancement.toml` #4-#5) and is
  unchanged.

<a id="s07"></a>
### S07: `get_flush` tries the suits in a different order

- **SMODS:**
  - `lovely/playing_card.toml` #10. Target `functions/misc_functions.lua`, pattern
    `local suits = { "Spades", "Hearts", "Clubs", "Diamonds" }`, replaced by
    `local suits = SMODS.Suit.obj_buffer`.
  - #11-#13: Four Fingers goes through `SMODS.four_fingers('flush')`, and the unused `#hand > 5`
    guard is dropped.
  - `SMODS.Suit.obj_buffer` is in registration order, Diamonds, Clubs, Hearts, Spades
    (`src/game_object.lua:2004-2031`).
- **Vanilla:** `functions/misc_functions.lua:522-546` returns the matching cards of the first suit,
  in the order Spades, Hearts, Clubs, Diamonds, that reaches 5 cards (4 with Four Fingers).
- **Difference:** when two suits both reach the threshold with different card sets, the flush part,
  and so the scored cards, differs. With 5 played cards that needs Four Fingers and at least 3 Wild
  Cards:
  - 3 Wild Cards + a Heart + a Spade: vanilla's flush is the Wild Cards + the Spade; Steamodded's is
    the Wild Cards + the Heart.
  - 4 Wild Cards + a Diamond: vanilla scores the 4 Wild Cards; Steamodded scores all 5. With a Spade
    instead, it is the other way round.
  - With Smeared Joker, the red and black groups behave the same way: vanilla prefers black,
    Steamodded red.
- **Effect:** which cards score in Flush and Straight Flush (chips and per-card effects).
- **Severity:** state-difference.
- **Python: vanilla.** `jackdaw/engine/hand_eval.py:62`.

<a id="s08"></a>
### S08: suit and rank rolls of Sigil, Grim, Familiar and Incantation

- **SMODS:**
  - `src/game_object.lua`: `take_ownership` of `'sigil'` (2265-2291), `'grim'` (2349-2380),
    `'familiar'` (2381-2418) and `'incantation'` (2419-2458).
  - `lovely/pool.toml` #1. Target `functions/misc_functions.lua`, pattern
    `function pseudorandom_element(_t, seed)`; it adds
    `if _t == SMODS.Suits then _t = SMODS.Suit:obj_list(true) end`, which gives Spades, Hearts,
    Clubs, Diamonds.
- **Vanilla:**
  - `card.lua:1233` (Sigil): suit from `{'S','H','D','C'}` on `'sigil'`.
  - `card.lua:1319-1327`:
    - Familiar: rank from `{'J','Q','K'}`, then suit, both on `'familiar_create'`.
    - Grim: suit only, on `'grim_create'`.
    - Incantation: rank from `{'2'..'9','T'}`, then suit, on `'incantation_create'`.
  - The suit list is always S, H, D, C.
- **Difference:**
  - The suit list becomes S, H, C, D: the 3rd and 4th entries are swapped.
  - Familiar and Incantation draw the suit first and the rank second, from the same stream.
  - Ouija is unchanged: `SMODS.Ranks` iterates 2…10, J, Q, K, A, the same as vanilla's
    `{'2',...,'A'}`.
  - The number of draws per stream is unchanged.
- **Scenario:**
  - Sigil and Grim give Clubs where vanilla gives Diamonds, and the reverse.
  - Familiar and Incantation make different suit/rank combinations: the first `familiar_create`
    draw picks among 4 suits instead of 3 ranks.
- **Severity:** state-difference.
- **Python:**
  - Sigil (`jackdaw/engine/consumables.py:1069-1081`), Familiar and Incantation
    (`consumables.py:958-978`, rank then suit): **vanilla**.
  - **Grim: neither.** `consumables.py:1016` calls
    `_roll_card_spec(["A"], "grim_create", "grim_create", ...)`, which advances `'grim_create'` once
    for the one-element rank list before the suit draw. Each Ace's suit therefore uses the 2nd, 4th,
    … seed instead of the 1st, 2nd, …. This is a vanilla-parity bug of its own, unrelated to
    Steamodded. `jackdaw/cli/scenarios/spectrals.py` already expects these spectrals to fail against
    the live game.

<a id="s09"></a>
### S09: end-of-round stickers run before the joker, and not on debuffed jokers

- **SMODS:**
  - `lovely/better_calc.toml` #28. Target `functions/state_events.lua`; the pattern is end_round's
    joker loop, `...calculate_joker({end_of_round = true, ...})` plus the rental and perishable
    calls. It is replaced by
    `SMODS.calculate_context({end_of_round = true, game_over = game_over, beat_boss = ...})`.
  - #2 (`eval_card`): adds
    `if not card:can_calculate(...) then if card.ability.rental then ... calculate_sticker(context, 'rental') ... end return {}, {} end`.
  - `lovely/can_calculate.toml` #4 removes `calculate_joker`'s own debuff check.
  - `src/game_object.lua:3097-3147`: the perishable and rental Stickers' `calculate`. Stickers run in
    the order eternal, perishable, rental, pinned, before `calculate_joker` inside `eval_card`.
- **Vanilla:** `functions/state_events.lua:99-110`. For each joker in turn:
  1. `calculate_joker({end_of_round = ...})`, which returns nil for a debuffed joker
     (`card.lua:2292`);
  2. `calculate_rental`;
  3. `calculate_perishable` (`card.lua:2271-2289`).

  Debuffed jokers get steps 2 and 3 as well.
- **Difference:** in Steamodded each joker gets its perishable tick, then its rental charge, then
  its own end-of-round effect. A joker that is debuffed at that moment still pays rent, but its
  perishable tally does not tick.
- **Scenarios:**
  - **Crimson Heart's victim.** Crimson Heart debuffed a Perishable joker for the last hand. Vanilla
    ticks it (for example 3 rounds left → 2); Steamodded does not.
  - **A Perishable Turtle Bean expiring this round.** Steamodded debuffs it first:
    `remove_from_deck` takes away `h_size` hand size. Its decay then still runs (Steamodded's
    `calculate_joker` has no debuff check) and takes one more: `-(h_size + 1)` in total. Vanilla
    decays first (−1, leaving `h_size - 1`) and then removes `h_size - 1`: `-h_size` in total. The
    live hand ends up permanently one card smaller.
  - Other jokers that expire this round fire their effect in both, but in Steamodded only after their
    passives are gone.
- **Severity:** state-difference.
- **Python: vanilla.** `game.py:2352-2378` runs, per joker: the effect if it is not debuffed, then
  `calculate_rental`, then `calculate_perishable`.

<a id="s10"></a>
### S10: ×Mult arithmetic

- **SMODS:**
  - `src/game_object.lua:3760-3809`, the `'mult'` `SMODS.Scoring_Parameter`.
    - Its ×mult branch (line 3788) calls `self:modify(mult * (amount - 1))`.
    - `modify` (line 3806) sets `mult = mod_mult(self.current + amount)`.
  - All scoring is routed through it by `lovely/better_calc.toml` #1 and #11-#15: the per-card,
    edition and joker effect blocks of `G.FUNCS.evaluate_play`.
- **Vanilla:** `functions/state_events.lua:753, 762, 859, 912, 925, 934`:
  `mult = mod_mult(mult*X)`.
- **Difference:** Steamodded computes `m + m*(X-1)`, rounding twice, where vanilla computes `m*X`,
  rounding once.
  - **Exact:** whenever `X-1` is 0.5, 1, 2, … and `m` is exact. That covers Polychrome, Steel, Glass
    and ×2/×3/×1.5 jokers on whole mults.
  - **Can differ:** for ×1.1, ×1.2, ×1.3, ×1.99 and similar (Constellation, Vampire, Obelisk, Steel
    Joker, Ramen), and after the mult has become fractional.
  - **Measured:** random chains of +mult and ×X gave a different final mult in about 15% of chains,
    and a different `floor(chips*mult)` in about 0.45%. Single steps on whole bases 1-5000: ×1.1
    differs for 254 bases, ×1.3 for 1351, ×1.5 and ×1.75 for none.
- **Severity:** state-difference: a last-bit difference in mult, sometimes ±1 chip in the score.
- **Python: vanilla.** `jackdaw/engine/scoring.py` uses `mult *= ...`, for example at lines 321,
  351, 397, 917, 955 and 975.

<a id="s11"></a>
### S11: `Card:init` no longer draws unseeded random numbers

- **SMODS:** `lovely/better_calc.toml` #67 (the patch is commented "wtf"). Target `card.lua`, pattern
  `self.discard_pos = { r = 3.6*(math.random()-0.5), x = math.random(), y = math.random() }`,
  replaced by zeros.
- **Vanilla:** `card.lua:46-50` makes three `math.random()` calls for every Card created.
- **Difference:** the state of LuaJIT's global, unseeded generator differs. Every `pseudorandom` call
  reseeds it, so seeded streams are unaffected. Only vanilla's unseeded picks see a difference:
  - the first-shop Buffoon pack variant (`common_events.lua:1947`, `get_pack`);
  - the Charm and Meteor Tag mega-pack variants (`tag.lua:211`, `:226`);
  - the throwaway Stone card ids (`Card:get_id`).

  Variants `_1` and `_2` of a pack have the same contents; only the key and art differ.
- **Severity:** cosmetic-in-practice. It only touches pack-variant keys that were never
  deterministic (see "What the source cannot tell" in [README.md](README.md)).
- **Python: n/a.** The variant is always `_1` (`rng.py:59-74`, `tags.py:262-271`,
  `shop.py:203-209`).

<a id="s12"></a>
### S12: `pseudoseed` while the game is paused

- **SMODS:** `lovely/fixes.toml` #48, "Fix prng calls on collection advancing seeds / Keep vanilla
  behaviour for to-do list". Target `functions/misc_functions.lua`, after the pattern
  `if key == 'seed' then return math.random() end`; it adds
  `if G.SETTINGS.paused and key ~= 'to_do' then return math.random() end`.
- **Vanilla:** `functions/misc_functions.lua:298-313` always advances
  `G.GAME.pseudorandom[key]`.
- **Difference:** while an overlay menu has paused the game (collection, options, run info), any
  seeded draw other than `'to_do'` returns an unseeded number and does not advance its stream.
  - BalatroBot never pauses.
  - No vanilla gameplay draw happens while paused.
  - Card UIs built by menus can draw (To Do List's `'to_do'`), and Steamodded keeps exactly that one.
- **Severity:** cosmetic-in-practice.
- **Python: n/a.**

<a id="s13"></a>
### S13: Crimson Heart with no joker left to pick

- **SMODS:** `lovely/fixes.toml`:
  - #19 (regex on `if self.name == 'Crimson Heart' and self.prepped and G.jokers.cards\[1\] then`);
  - #20, "Fix bad logic if not enough choices for debuff";
  - #21-#22: the `crimson_heart_chosen` flag and `Blind:debuff_card`;
  - #23-#24: clear the flag on disable and defeat.

  `lovely/pool.toml` #3 also makes `pseudorandom_element` return nil for an empty list.
- **Vanilla:** `blind.lua:588-602`.
  - Candidates are the jokers that are not debuffed (all of them if there are fewer than 2).
  - Every joker is `set_debuff(false)`.
  - Then it calls `pseudorandom_element(candidates, pseudoseed('crimson_heart'))`.
- **Difference:** Steamodded remembers the previous pick, excludes it and debuffed jokers, and falls
  back to all non-debuffed jokers. The candidate set is the same in every normal case. It differs
  only when nothing is left with 2 or more jokers. Example: two jokers, one picked last hand and the
  other an expired Perishable. Vanilla calls `math.random(0)` on an empty list, which is a Lua error.
  Steamodded re-picks the previous joker.
- **Severity:** cosmetic-in-practice: vanilla would crash.
- **Python: neither, in that edge.** `jackdaw/engine/blind.py:424-441` skips the roll and makes no
  `'crimson_heart'` draw when no candidate is left. Normal cases match both.

## BalatroBot

BalatroBot drives the game through `G.FUNCS` callbacks. The entries below cover the places where it
does more than press a button a player could press. File references are to
`third_party/balatrobot/v1.5.2/src/lua/`; the installed copies of these files are identical.

<a id="s14"></a>
### S14: the deck's card limit follows the playing cards

- **SMODS:** `lovely/card_limit.toml` #5, "CardArea:update()". Target `cardarea.lua`, pattern
  `if self == G.deck and self.config.card_limit > #G.playing_cards then self.config.card_limit = #G.playing_cards end`;
  the payload sets `card_limit` (and `card_limits.total_slots`) to `#G.playing_cards` whenever they
  differ.
- **Vanilla:** `cardarea.lua:265` only lowers the limit to `#G.playing_cards`. It goes up when an
  `emplace` overfills the deck (`cardarea.lua:47-51`) and by one for Cryptid, Marble Joker and DNA
  (`card.lua:1210, 2595, 3506`).
- **Difference:** a playing card that joins the run outside the deck (Certificate's card, a
  Standard pack pick, Familiar, Grim, Incantation) raises the live deck limit and leaves vanilla's.
  Only BalatroBot's view reads it (`cards.limit`): no rule does.
- **Severity:** cosmetic-in-practice.
- **Lockstep:** S14 drops `cards.limit` when the live value is the shadow's playing card count.

<a id="b01"></a>
### B01: the `pack` endpoint skips the UI's pick gates

- **BalatroBot:** `endpoints/pack.lua`:
  - 138-151: its own joker-slot check refuses when
    `G.jokers.config.card_count >= G.jokers.config.card_limit`;
  - 158-166: Ankh needs at least one joker;
  - per-card target counts;
  - 233-242: then `G.FUNCS.use_card(btn)`, called directly.

  It never calls `can_use_consumeable` or `can_select_card`.
- **Vanilla, and the Steamodded UI:** a pack card is taken through its button's gate.
  - Jokers: `can_select_card` needs a Negative edition or a free slot (`button_callbacks.lua`;
    `lovely/booster.toml` #25 keeps the Negative exception through `card_limit`).
  - Consumables: `can_use_consumeable` (`card.lua:1523` onwards). Judgement, The Soul and Wraith
    need a free joker slot (`card.lua:1557-1563`); The Emperor, The High Priestess and The Fool need
    a consumable slot (`card.lua:1550-1556`).
  - Steamodded keeps both gates on the pack buttons (the dump's `UI_definitions.lua:263-291` and
    `button_callbacks.lua:2105-2124`).
- **Difference:**
  - Judgement, Wraith and The Soul picked at full joker slots create a 6th joker. LSBVJSQL in
    `game.py:1741-1743`; that comment blames "the smods booster UI", but it is this endpoint.
  - The Emperor, The High Priestess and The Fool picked at full consumable slots use up the pick and
    create nothing.
  - The bot's own joker check has no Negative exception, so a pick vanilla allows (a Negative joker
    at full slots) is refused.
- **Severity:** state-difference: picks that are illegal in vanilla are accepted, and one legal pick
  is refused.
- **Python: vanilla.** `jackdaw/engine/consumables.py:189-214` (`pack_pick_block_reason`) and
  `game.py:1735-1745` raise `IllegalActionError`, and the lockstep policy vetoes these picks.

<a id="b02"></a>
### B02: `play` and `discard` under Cerulean Bell

- **BalatroBot:** `endpoints/play.lua:65-81` and `endpoints/discard.lua:73-93`:
  `G.hand:unhighlight_all()`, then `click()` on each requested card, then
  `G.FUNCS.play_cards_from_highlighted` or `discard_cards_from_highlighted`.
- **Vanilla:** `unhighlight_all` and `remove_from_highlighted` both keep a card with
  `forced_selection` (`cardarea.lua:201-208`), and the hand's highlight limit is 5. A player
  therefore always plays or discards the forced card.
- **Difference:** a request that leaves the forced card out is not rejected. The forced card stays
  highlighted, the requested cards are added until the limit, and the last requested card is
  silently dropped. ES7L222Z in `game.py:897-907`: a full house became three of a kind.
- **Severity:** state-difference: a different hand is played.
- **Python: vanilla.** `game.py:897-907` (`_require_forced_card`) raises unless the forced card is
  selected.
- **Bot layer: BalatroBot.** `jackdaw-bot`'s `play` and `discard` (`crates/jackdaw-bot/src/
  endpoints/play.rs`) click the requested cards the same way, so `jackdaw serve` and `SimBackend`
  play the hand the live game plays; the engine then takes that (vanilla) selection.

<a id="b03"></a>
### B03: the debug `add` endpoint

- **BalatroBot:** `endpoints/add.lua:355-434` calls `SMODS.add_card` with the key, the edition,
  `stickers` and `force_stickers = true`. A `perishable` value overrides `perish_tally`.
- **Steamodded:**
  - `SMODS.create_card` (`src/utils.lua:374-416`) sets `SMODS.bypass_create_card_edition` when an
    edition is passed, so `create_card` skips the `'edi'..ante` edition poll. In the dump's
    `common_events.lua:2394-2398` the poll runs only when no edition was given.
  - `add_to_deck` → `change_size` → `handle_card_limit` fills the hand to the new limit (S04).
  - A playing card goes through `create_card('Base', ...)`, which draws a random front from
    `'front'..ante` (dump `common_events.lua:2363`) before `SMODS.set_create_card_front` replaces
    it; an enhancement is applied with a second `set_ability`, which lets the boss debuff the
    card (`card.lua:365`).
- `add` is not a vanilla action. It is listed because the simulator imitates it (the oracle's
  `jackdaw/engine/debug.py::add_card`, which the Rust `Game::debug_add` and so `SimBackend` and
  `jackdaw serve` follow):
  - it polls `'edi'..ante` even when an edition was requested, which the live game does not
    (**neither**, gaps.md B15);
  - it draws no `'front'..ante` card for a playing card and doesn't check the boss for an
    enhanced one (**neither**, gaps.md B18, B19);
  - it fills the hand (**SMODS**, see S04).
- **Severity:** RNG-desync on `'edi'..ante` (an edition passed) and `'front'..ante` (a playing
  card added) between the simulator and the live game.

<a id="b04"></a>
### B04: runtime settings and hooks (no gameplay effect)

- **`src/lua/settings.lua`:** fixes `love.update`'s `dt` (1/60, or 4.99/60 headless), sets
  `G.SETTINGS.GAMESPEED` to 4 (10 in fast mode), turns on reduced motion and turns off audio and
  shaders.
  - This changes only timing: the order and contents of the event queue do not depend on `dt`.
  - Per-frame work (`Card:update` values such as Swashbuckler's, Steamodded's `handle_card_limit`)
    runs between events either way.
- **The installed `balatrobot.lua` only** (not in v1.5.2): a debug `CardArea:shuffle` override.
  - It prints, then re-implements `pseudoshuffle` step for step: seed `pseudoseed(_seed or 'shuffle')`,
    sort by `sort_id`, the same Fisher-Yates loop, then `set_ranks`.
  - The results are identical to vanilla.
- **`dump_pool.lua`** (installed only) is never loaded: the manifest's `main_file` is
  `balatrobot.lua`. It would only print anyway.
- **Other hooks only read state:**
  - the `love.update` hook (`check_game_over`, the server loop);
  - the gamestate builder, which uses `generate_UIBox_ability_table` to avoid hover side effects.

  The installed `server.lua` and `load.lua` differ from v1.5.2 only in connection handling and a
  file-existence check.
- **Debug endpoints:** `set` and `add` write state directly by design.

## Examined and found equivalent

These are Steamodded changes to gameplay code that keep vanilla outcomes. They are listed so that
nobody chases them.

| change | SMODS location | why it changes nothing |
|---|---|---|
| Calculation pipeline (`SMODS.calculate_context`, `eval_card`, `trigger_effects`) | `better_calc.toml` #1-#35, #44-#53 | Contexts fire in the same order, with the same values and the same effect order (chips, then mult, then ×mult). The only differences are S09 and S10. |
| Debuff checks moved into `eval_card`/`can_calculate` | `can_calculate.toml` #1-#3, #5-#6 | Every vanilla caller goes through `eval_card`. The per-frame joker tallies update while debuffed, which no score reads. |
| To Do List and Matador no longer return `dollars` | `joker_retriggers.toml` #27-#28 | Both already call `ease_dollars` (`card.lua:3491-3497`, `:2736-2743`, `:3719-3726`). Removing the return keeps it to one payment. |
| Other joker returns rewritten for retrigger support | `joker_retriggers.toml` #1-#26 | Return shapes only. |
| Blueprint and Brainstorm via `SMODS.blueprint_effect` | `better_calc.toml` #97-#98 | Same copy targets. |
| Sixth Sense flags its card | `better_calc.toml` #96 | Reproduces vanilla's `break` after the first destroying joker (`state_events.lua:956-959`). |
| Probabilities via `SMODS.pseudorandom_probability` | `listed_probabilities.toml` #1-#2, #13; `src/utils.lua:2700-2715` | Same streams and thresholds. The numerator is `base * normal / 2^#oops`, and each non-debuffed Oops! All 6s doubles it again. |
| Glass Card shatters through its `calculate` | `enhancement.toml` #21-#24; `src/game_object.lua:3250-3257`; `src/utils.lua:2010-2049` | One `'glass'` roll per non-debuffed scoring Glass Card, even after a joker destroyed it (G34). Unscored cards get `cardarea = 'unscored'` and do not roll. |
| Purple Seal on discard | `better_calc.toml` #55 | Fires only through `eval_card`, so debuffed seals do nothing, as `card.lua:2242` does. |
| Scoring hand rebuilt in `G.play` order | `better_calc.toml` #70 | Vanilla sorts by `T.x` (`state_events.lua:580-600`). The play area is laid out in card order, so the result is the same. |
| Poker hands (`PokerHand` parts with `or_more`; `get_straight`, `evaluate_poker_hand` overrides) | `poker_hand.toml` #1-#9; `src/overrides.lua` | Same hands and parts, including vanilla's fix-ups. |
| Pool and poll functions | `pool.toml` #2-#11; `rarity.toml` #4; `edition.toml` #2, #7, #14; `src/overrides.lua:2215-2283` | `get_current_pool` and its `UNAVAILABLE` placeholders are unchanged. Rarity thresholds stay at 0.7/0.95. `poll_edition`'s weights reproduce vanilla's thresholds: 0.003/0.006/0.02/0.04 scaled by `edition_rate` and `_mod`, and Illusion's guaranteed poll gives 0.85/0.5. |
| Shop slot type roll | `center.toml` #27-#28 | Same `'cdt'` rates and order. |
| Booster contents through `SMODS.Booster` | `booster.toml` #1-#3; `src/game_object.lua` | Same key per draw (draws on different keys are independent). The Standard pack still uses the `'Enhancedsta'..ante` pool. The extra `'soul_smods_'..type..ante` draw is on its own stream and only matters for modded cards. |
| Shop and pack sticker rolls | `sticker.toml` #4-#7 | Vanilla's stickers keep `should_apply = false`, so `'etperpoll'`/`'packetper'` and `'ssjr'`/`'packssjr'` run as in vanilla. |
| Seal pool order | `seal.toml` #1, #11 | Makes `SMODS.poll_seal` map to the same seals as vanilla (Standard pack: threshold 0.8, then 0.75/0.5/0.25). |
| Next voucher | `shop.toml` #4-#8, #10 | `SMODS.get_next_vouchers` makes the same `'Voucher'..ante` / `'Voucher_fromtag'` draws. |
| Chaos the Clown | `shop.toml` #11-#14 | `round_resets.free_rerolls` holds the same count as `find_joker('Chaos the Clown')`. |
| Starting deck | `playing_card.toml` #1-#8 | Same `card_protos` and the same sort. Erratic picks over the same sorted `P_CARDS` keys. |
| Deck vouchers applied inside an event | `better_calc.toml` #66 | The vouchers' own effects (`Card:apply_to_run`) were already events. The end state is the same. |
| `used_jokers` by key | `pool.toml` #12-#14 | Vanilla matches by name in jokers and consumables (`misc_functions.lua:903-915`); Steamodded by key in jokers, consumables and vouchers. Names are unique per key for every pool that checks it. |
| `set_ability` keeps unknown ability fields | `fixes.toml` #14-#16 | Vanilla re-sets only playing cards. `played_this_ante` is reset as in vanilla, and `perma_bonus` and `forced_selection` are kept as in vanilla. |
| Blind refresh only while in a blind | `fixes.toml` #29-#30 | Outside a blind, vanilla's refresh only clears debuffs that cannot exist there. |
| `create_playing_card` calls `add_to_deck` | `fixes.toml` #69; `better_calc.toml` #73 | `add_to_deck` returns at once for playing cards. |
| Food jokers destroyed through `SMODS.destroy_cards` | `better_calc.toml` #84 | `getting_sliced` is set at once, so the eaten joker stops reacting to later contexts in the same action (vanilla: Ramen prints "Eaten!" again per further discarded card). No value changes. |
| Scaling jokers through `SMODS.scale_card` | `scaling.toml` #1-#29 | Same increments. Ceremonial Dagger and Glass Joker now apply outside an event, with the same values. |
| `playing_card_added` gets the real card later | `better_calc.toml` #72, #75, #76; `src/game_object.lua` spectrals | Hologram gains the same. |
| Eternal and `getting_sliced` checks via `SMODS.is_eternal` | `better_calc.toml` #87-#91 | Same exclusions for Ceremonial Dagger, Madness, Ankh, Hex and selling. |
| Campfire is not upgraded by its own sale | `better_calc.toml` #94 | Vanilla never sends `selling_card` to the card being sold. |
| Win check `not G.GAME.won and ante >= win_ante` | `fixes.toml` #31 | The ante cannot pass `win_ante` without its boss being beaten, and vanilla's win screen runs once (`win_notified`). |
| Orbital Tag fallback hand | `tag.toml` #3 | Draws `'orbital'` only for a tag with no hand, which vanilla flows never create. |
| Event manager front-insert fix | `event.toml` #2-#4 (#1 does not apply) | Vanilla never adds events to the front of a queue. |
| `get_nominal` ×10 ranks; `suit_nominal_original` 10× larger | `playing_card.toml` #17, #19 | Same order, except for two cards of the same current rank and suit but different original suits whose node ids differ by more than about 160,000 (where `suit_nominal_original` and `unique_val` cross over). |
| `Card.unique_val` saved and loaded | `fixes.toml` #8-#9 | Vanilla regenerates it on load. This only affects the tie-break above after a save and load. |
| `Blind:debuff_hand` calls `get_poker_hand_info` again | `blind.toml` #14 | Only uses the unseeded Stone ids. |
| The Eye and The Wheel take-overs | `src/game_object.lua:1784-1811` | Same hand table and the same 1-in-7 `'wheel'` roll. |
| Strength take-over | `src/game_object.lua:2209-2264` | `SMODS.modify_rank` wraps the same way (K→A, A→2). |
| `get_new_boss` with `SMODS.add_to_pool` | `blind.toml` #33 | No vanilla blind defines `in_pool`. |
| Blind hooks and `if` chains turned into `elseif` | `blind.toml` #1-#18, #35-#40 | No vanilla blind defines the hooks, and blind names are exclusive. |
| Card limits through the `card_limits` metatable | `card_limit.toml` #1-#3, #5-#7, #23; `edition.toml` #9-#10; `booster.toml` #25 | Same slot counts and buy checks, except the stripped copy in S05. |
| Booster and voucher states (`SMODS_BOOSTER_OPENED`, `SMODS_REDEEM_VOUCHER`) | `booster.toml` | Same flow; the bridge maps the names. |
| Level up | `scoring_calculation.toml` #14, #21-#22 | Same level, chips and mult arithmetic. |
| `generate_starting_seed` skips cards that are not `in_pool` | `fixes.toml` #28 | Only used to make the seed of an unseeded Gold Stake run. |

## Python behaviours that follow SMODS instead of vanilla

The live-tuned Python code was found by searching for "live", "lockstep", "verified", "observed",
"balatrobot", "smods" and seed-like tokens (`[A-Z0-9]{8}`). That turns up 74 places that cite 41
seeds. Checked against the current code, only these follow Steamodded rather than vanilla:

1. **Marble Joker's Stone placement (S01).** Fixed by gap G55: the oracle now shuffles the Stone
   in, as vanilla and the Rust port do (it used `pending_deck_bottom`, LS86UJ9R).
2. **SimBackend `add` refills the hand to the new limit (S04, B03).**
   `jackdaw/bridge/backend.py:273-280`. This is the bridge only: the engine draws exactly the
   increase, as vanilla does.

Every other live-verified behaviour is shared by vanilla and Steamodded, or Python deliberately
follows vanilla:

- LSI5EB7T holds in both orders (S02).
- LSBVJSQL and ES7L222Z are BalatroBot endpoint behaviours that Python does not copy (B01, B02).
- The "dump card.lua" line numbers cited in `game.py` and `shop.py` are lines of the Lovely dump,
  but the logic they point at is vanilla:

  | cited | vanilla |
  |---|---|
  | dump `:786-793` | `card.lua:616-622` |
  | dump `:850` | `card.lua:673-679` |
  | dump `:511` (`:520` in the current dump) | `card.lua:383` |

## Other Python findings from this review

These are not Steamodded behaviours, but the review turned them up:

- **Grim draws `'grim_create'` twice per Ace** (`consumables.py:1016`), so it matches neither
  vanilla nor the live game (S08).
- **SimBackend `add` polls `'edi'..ante` even when an edition is requested**
  (`bridge/backend.py:253-266`). The live game skips that poll (B03).
- **Crimson Heart with no candidate skips its roll** (`blind.py:436`) (S13).
- **Two stale comments:**
  - `packs.py:231-233` says Steamodded rolls Standard-pack enhancements on a `'std_enhance'`
    stream. `SMODS.poll_enhancement` (`src/utils.lua:824`), whose default key that is, has no
    caller. Steamodded's Standard pack (`src/game_object.lua:1698-1702`) creates set
    `'Enhanced'`/`'Base'` with `key_append = 'sta'`, which is the same `'Enhancedsta'..ante` pool
    as vanilla, so the live game can check this field after all.
  - `game.py:1741-1743` blames "the smods booster UI" for the 6th joker; it is BalatroBot's
    `pack` endpoint (B01).

## Appendix A: patch inventory by file

Every patch in `smods/lovely/*.toml` falls into one of three classes:

| class | meaning | patches |
|---|---|---:|
| S | changes behaviour (the entry ids are given) | 40 |
| eq | touches gameplay code but keeps vanilla outcomes | 365 |
| none | no gameplay effect: UI, text, sprites, sound, saving, profile/unlock state, or mod-API plumbing that vanilla content never reaches | 535 |

The patch descriptions below are condensed from each patch's own comment.

| file | n | S | eq | none |
|---|---:|---|---|---|
| achievements.toml | 8 | | | #1-#8 achievements, unlocks and discovery in seeded runs |
| atlas.toml | 5 | | | #1-#5 sprite atlases, card scale |
| back.toml | 15 | | #4, #6 deck `apply`/`calculate` hooks; #14 `config.jokers` (no vanilla deck uses it) | #1-#3, #7-#13 deck UI, loc and unlocks; #5 `save_run` event; #15 load selected back |
| better_calc.toml | 99 | #1, #11-#15 S10; #2, #28 S09; #36-#43 S02; #67 S11; #72 S02/S03; #75 S01 | #3-#7 `eval_card` values; #10 deck/discard areas (off by default); #16-#26, #29, #31-#35, #44, #46-#53 contexts through `SMODS.calculate_context`; #55 Purple Seal; #64-#66 `apply_to_run`, Observatory, deck vouchers in events; #70 scoring hand order; #73-#74 `create_playing_card`; #76 DNA; #77 area check; #84 food jokers; #85-#86 destroyed contexts; #87-#91 eternal/sliced checks; #94 Campfire; #95 ante contexts; #96 Sixth Sense; #97-#98 Blueprint/Brainstorm; #99 reload `last_hand` | #8-#9, #56-#62 status text and sound; #27 cash-out rows; #30 store blind tag; #45 remember used card's area; #54 reroll cost for context; #63 voucher card area; #68-#69 x-chips; #71 `ending_booster` context; #78-#83 debuff text, saved message, dissolve colours, hand display; #92-#93 `tag_added` context |
| better_calc_fixes.toml | 3 | | #1-#3 keep `context.blueprint_card` in a local | |
| blind.toml | 42 | | #1-#18 blind hooks (debuffed-by-blind flag, set/disable/defeat/debuff/flip/draw/debuff_hand/modify_hand/press_play); #33 `get_new_boss`; #35-#40 `elseif` chains, `blind.effect` | #19-#32, #34 blind loc, UI, sprites, collection, challenge UI; #41-#42 blind save/load |
| blind_ui.toml | 13 | | | #1-#13 blind HUD (#13 did not apply) |
| booster.toml | 27 | | #1-#6, #8, #11-#13, #16-#17, #19, #22-#27 booster states, sizes, open/use/skip/redeem flow | #7, #9-#10 card area draw and align; #14-#15 career stats; #18 alignment; #20 free-voucher animation; #21 select-button UI |
| calc_returns.toml | 3 | | | #1-#3 attention text |
| calculate_card_added.toml | 1 | | #1 `card_added` context (no vanilla listener) | |
| can_calculate.toml | 6 | #4 S09 | #1-#3, #5-#6 debuff checks moved to `eval_card` | |
| card_limit.toml | 23 | #4, #8, #16-#18 S04 | #1-#3 card-limit metatable; #5 deck limit; #6-#7 Negative slot moved to `card_limit` (the mechanism behind S05); #23 deck limit during setup | #9-#15 counts and info queue; #19 unlock; #20-#22 challenge UI |
| cardarea.toml | 2 | | | #1-#2 card area drawing |
| cardareas.toml | 3 | | | #1-#3 custom area setup, button alignment |
| center.toml | 49 | | #1-#2, #4-#5, #17-#18, #26, #37-#38, #44 center hooks; #27-#28 shop rates; #29-#35 `create_card`; #39-#42 pack use flows; #48 consumable usage tally | #3, #6-#16, #19-#25, #36, #43, #45-#47, #49 sprites, UI, loc, collection, tallies, card back |
| challenge.toml | 7 | | #4-#7 challenge bans and apply | #1-#3 challenge UI |
| compact_cashout.toml | 3 | | | #1-#3 cash-out rows |
| compat_0_9_8.toml | 3 | | | #1-#3 old-API fallbacks |
| core.toml | 4 | | | #1-#4 core loader, version modules |
| crash_handler.toml | 3 | | | #1-#3 crash handler |
| create_sprite.toml | 8 | | | #1-#8 sprites |
| deck_skins.toml | 4 | | | #1-#4 deck skins |
| dollar_row.toml | 5 | | | #1-#5 money rows |
| easings.toml | 3 | | | #1-#3 easing curves |
| edition.toml | 14 | | #2 sort edition pool; #7 edition cost; #9-#10 buy-space with `card_limit`; #14 edition poll bypass hook | #1, #3-#6, #8, #11-#13 debug, tooltips, shaders, save, badges |
| enhancement.toml | 32 | | #1-#8 no-rank/no-suit/any-suit checks (incl. Raised Fist, Flower Pot, Seeing Double's Wild Card check); #10-#14 target cards, played-cards tally; #18-#25 extra enhancements (off), Blue Seal, enhancement gate, Glass; #27-#32 enhancement cache, `replace_base_card`, debuff loop guard | #9 unlock; #15 delayed sprites (did not apply); #16-#17 UI; #26 cache reset |
| event.toml | 4 | | #1-#4 front-insert fix (#1 does not apply) | |
| fixes.toml | 71 | #19-#22 S13; #25 S04; #48 S12; #54-#56 S05; #59 S06 | #8-#9 `unique_val` save/load; #14-#16 `set_ability`; #18 Matador (removed upstream); #23-#24 Crimson Heart flags; #28 starting seed; #29-#31 in-blind refresh, win check; #38-#45 extra-enhancement counts, reroll vouchers without a card; #57 edition table copy; #58 Smeared Joker; #60 Blue Seal fallback; #63-#64, #66-#69 `Card:remove`, `getting_sliced`, `pseudoshuffle`, starting cards, `create_playing_card` | #1-#7, #10-#13, #17, #26-#27, #32-#37, #46-#47, #49-#53, #61-#62, #65, #70-#71 start-up, Steam, saves, UI, loc, unlocks, animation, booster-skip button |
| gradient.toml | 4 | | | #1-#4 colour gradients |
| hand_limit.toml | 6 | | #1, #3-#4 play and discard limits (5) | #2, #5-#6 indicators |
| jimboquip.toml | 12 | | | #1-#12 Jimbo quips |
| joker_retriggers.toml | 28 | | #1-#28 return values for retriggers (#27-#28: no double payment) | |
| joker_size.toml | 3 | | | #1-#3 joker sprite size |
| keybind.toml | 3 | | | #1-#3 keybinds |
| language.toml | 6 | | | #1-#6 languages |
| libs.toml | 5 | | | #1-#5 libraries |
| listed_probabilities.toml | 13 | | #1-#2 probability rolls; #13 Oops! All 6s `mod_probability` | #3-#12 probability text |
| loader.toml | 1 | | | #1 discovery save |
| menu.toml | 6 | | | #1-#6 menus |
| mod.toml | 3 | | | #1-#3 per-mod hooks (none installed) |
| multi_box_descriptions.toml | 6 | | | #1-#6 descriptions |
| number_formatting.toml | 15 | | | #1-#15 number formatting |
| override_notice.toml | 34 | | | #1-#34 comments marking functions that `src/` overrides (those overrides are reviewed in the entries and the table above) |
| perma_bonus.toml | 19 | | #10-#19 ability defaults, permanent bonuses, Lucky Card roll, dollar buffer | #1-#9 UI |
| playing_card.toml | 31 | #10-#13 S07 | #1-#8 suit/rank tables, starting deck; #14-#21 `get_X_same`, `get_nominal`, `set_base`, `change_suit`, `is_face`; #30-#31 `set_base` | #9, #22-#29 colours, load, UI |
| poker_hand.toml | 9 | | #1-#9 hand parts, visible hands, play counts | |
| poker_hand_screen.toml | 4 | | | #1-#4 hand screen |
| poker_hand_text.toml | 2 | | | #1-#2 hand text |
| pool.toml | 14 | #1 S08 | #2-#14 `pseudorandom_element` filters, `get_current_pool`, `used_jokers` by key | |
| preflight.toml | 7 | | | #1-#7 preflight modules |
| rarity.toml | 5 | | #4 rarity poll; #5 Baseball Card rarity check | #1-#3 badges, colours |
| scaling.toml | 29 | | #1-#29 `SMODS.scale_card` | |
| scoring_calculation.toml | 22 | | #1 final score; #14 level up; #21-#22 `mod_mult`/`mod_chips` | #2-#13, #15-#20 hand text, score display, save |
| screenshader_rendering.toml | 2 | | | #1-#2 rendering |
| seal.toml | 16 | | #1 Standard pack seal; #3 seal hook; #5 Gold Seal; #11 seal order; #14-#16 seal ability, `card_limit`, copy | #2, #4, #6-#10, #12-#13 badges, tooltips, alerts, sounds |
| shaders.toml | 1 | | | #1 shaders |
| shop.toml | 20 | | #1-#8, #10 defaults, extra boosters and vouchers, voucher roll and redeem; #11-#14 free rerolls; #17-#18 shop voucher save/load; #19 Illusion edition; #20 `modify_shop_card` context | #9 restock message; #15-#16 shop width |
| sound.toml | 11 | | | #1-#11 sound |
| stake.toml | 26 | | #4-#5 stake modifiers; #14 `get_blind_amount` | #1-#3, #6-#13, #15-#26 stake UI, stickers, profile usage |
| sticker.toml | 9 | | #4-#7 sticker rolls in `create_card` | #1-#3, #8-#9 UI |
| tag.toml | 11 | | #1-#2 tag hooks; #3 Orbital fallback; #11 Boss Tag guard | #4-#10 tag UI |
| text_effect.toml | 9 | | | #1-#9 text effects |
| threads.toml | 3 | | | #1-#3 threads |
| ui.toml | 36 | | | #1-#36 UI |
| ui_additional_text_props.toml | 28 | | | #1-#28 text properties |
| ui_elements.toml | 16 | | | #1-#16 UI elements |
| ui_overflow.toml | 5 | | | #1-#5 UI overflow |

Steamodded's `src/` files are not lovely patches but run the same way: `overrides.lua`, `utils.lua`
and `game_object.lua` (with the `take_ownership` calls at lines 1362, 1411-1415, 1784, 1794, 1889,
2209, 2265, 2292, 2349, 2381, 2419, 2930, 3250, 3259 and 3436-3535). They are covered by the entries
(S03-S06, S08-S10) and by the table above.

## Appendix B: patches that did not apply

From the newest Lovely log (`Mods/lovely/log/lovely-2026.09.29-22.43.58.log`):

- `blind_ui.toml` #13 (regex `\(k ==1 and blind\.name == 'The Wheel' and '1' or ''\)\.\.` on
  `functions/UI_definitions.lua`): no match. This is collection text.
- `enhancement.toml` #15 (pattern `if delay_sprites then` … on `card.lua`): no match. This is sprite
  timing.
- `event.toml` #1 targets `event.lua` rather than `engine/event.lua`, so it never applies (Lovely
  does not warn about that). Patches #2-#4 still apply and set the counter themselves.

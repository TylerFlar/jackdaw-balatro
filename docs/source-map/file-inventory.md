# Balatro Source File Inventory

Source directory: `balatro_source/`

Total: **44 Lua files** (~37,500 lines of game/engine code + localization)

---

## Category: Core Game Logic

These files define the rules, scoring, progression, and card mechanics.
**All are relevant to the simulator.**

| File | Lines | Description |
|------|------:|-------------|
| `game.lua` | 3,629 | **The orchestrator.** Game class — `init_game_object` (creates entire G.GAME run state), `start_run` (run initialization: stake modifiers, deck building, CardArea creation, seed setup, challenge application), `update` (main game loop state machine dispatching to `update_selecting_hand`, `update_shop`, `update_hand_played`, `update_draw_to_hand`, `update_new_round`, `update_blind_select`, `update_round_eval`, `update_game_over`, and all pack states), `init_item_prototypes` (loads P_CENTERS, P_CARDS, P_BLINDS, P_TAGS — all card/blind/tag data), save/load, profile management. |
| `card.lua` | 4,771 | **The big one.** Card class — abilities, scoring (`calculate_joker` is 2,800+ lines covering every joker effect), editions, seals, enhancements, suit/rank checks, consumable use (tarots/planets/spectrals), cost, sell, perishable/rental/eternal, debuff, save/load. |
| `blind.lua` | 751 | Blind class — boss blind debuff logic (`debuff_card`, `debuff_hand`, `modify_hand`, `press_play`, `stay_flipped`, `drawn_to_hand`), chip target calculation, defeat/disable. Every boss blind's mechanical effect is here. |
| `back.lua` | 288 | Back (deck) class — `apply_to_run` sets starting params per deck (hands, discards, dollars, joker slots, hand size, vouchers, consumables, ante scaling, editions, suit changes). `trigger_effect` handles Plasma/Anaglyph deck in-round effects. |
| `tag.lua` | 595 | Tag class — `apply_to_run` handles all tag trigger logic (economy tags, pack tags, edition tags, orbital, boss reroll, voucher, double tag, etc.). |
| `challenges.lua` | 738 | Static data table defining all 20 challenge configurations — custom rules, modifiers, starting jokers/consumables/vouchers, deck composition, banned cards/blinds/tags. |
| `globals.lua` | 522 | `Game:set_globals()` — feature flags, game states enum, stage enum, settings defaults, constants (card dimensions, hand list, colours), render scale. Seeds the RNG. Instantiates `G = Game()`. |
| `functions/state_events.lua` | 1,642 | Round lifecycle — `win_game()`, `end_round()` (scoring evaluation, game-over check, joker end-of-round effects, blind rewards), `new_round()` (ante progression, blind selection, deck reset). |
| `functions/common_events.lua` | 2,745 | Core game operations — `eval_card` (card scoring pipeline), `create_card` / `create_playing_card`, `get_current_pool` (shop/pack pool generation), `poll_edition`, `check_for_unlock` (all unlock conditions), `calculate_reroll_cost`, `level_up_hand`, `update_hand_text`, `ease_dollars`/`ease_chips`, `draw_card`, blind amount scaling, card copy, etc. |
| `functions/misc_functions.lua` | 2,022 | Utility + poker hand evaluation — `evaluate_poker_hand` (flush/straight/n-of-a-kind detection), `get_flush`, `get_straight`, `get_X_same`, pseudorandom system (`pseudoseed`, `pseudorandom`, `pseudoshuffle`), number formatting, save/load serialization, localization engine, career stats, high scores, sound helpers, colour utilities. |

---

## Category: Card Areas / Containers

| File | Lines | Description |
|------|------:|-------------|
| `cardarea.lua` | 668 | CardArea class — manages card containers (hand, deck, discard, jokers, consumables, shop, play area). Handles emplacement, removal, highlighting, sorting, hand-size changes, `draw_card_from` (deck→hand draw with blind flip logic), `parse_highlighted` (live hand evaluation while selecting). |

---

## Category: Engine / Framework

LÖVE2D engine layer. Provides the OOP system, scene graph, rendering pipeline,
input handling, and async event system that the game logic builds on.

| File | Lines | Description |
|------|------:|-------------|
| `engine/object.lua` | 37 | Base OOP class — `Object:extend()` for inheritance, `:is()` for type checking. Foundation for every class. |
| `engine/event.lua` | 195 | Event/timer system — `Event` class and `EventManager`. Supports delayed execution, easing animations, conditional triggers (`'after'`, `'ease'`, `'immediate'`), blocking/non-blocking queues. **Simulator-relevant**: the entire game runs through `G.E_MANAGER:add_event()`. |
| `engine/node.lua` | 389 | Base scene node — transforms (`T` table: x,y,w,h,r), visibility/collision/hover/drag states, parent-child hierarchy, alignment system. |
| `engine/moveable.lua` | 517 | Moveable node — adds visible transform (`VT`) that eases toward target transform (`T`), role-based positioning (Major/Minor bonds), parallax shadow calculation. |
| `engine/controller.lua` | 1,382 | Input controller — mouse/keyboard/gamepad/touch handling, cursor collision detection, drag/drop, focus management, button/pip registry, HID device switching. |
| `engine/ui.lua` | 1,054 | UIBox framework — builds hierarchical UI from definition tables (`{n=UIT.ROOT, config={...}, nodes={...}}`), handles layout (rows/columns/padding/alignment), text/object embedding, recalculation. |
| `engine/sprite.lua` | 216 | Sprite rendering — draws quads from atlas spritesheets with scaling, shader support (dissolve, etc.), draw step customization. |
| `engine/animatedsprite.lua` | 107 | Animated sprite — frame-based animation at configurable FPS, sprite position management within atlases. |
| `engine/particles.lua` | 177 | Particle system — generates particles with velocity, rotation, lifespan, color, and optional attachment to parent moveables. |
| `engine/text.lua` | 315 | Text/DynaText — multi-string display with pop-in/pop-out animations, letter-by-letter alignment, colour cycling, shadow rendering. |
| `engine/string_packer.lua` | 72 | Serialization — `STR_PACK` / `STR_UNPACK` for Lua table↔string conversion with `love.data.compress`/`decompress`. Used by save system. |
| `engine/sound_manager.lua` | 207 | Audio manager — threaded sound playback, pitch/volume control, music track state machine (splash→main→shop→boss transitions), per-state mixing. |
| `engine/save_manager.lua` | 84 | Save I/O thread — writes compressed `.jkr` files for progress, settings, metrics, run state, and unlock notifications. |
| `engine/profile.lua` | 188 | Profile management — loads/saves per-profile data (progress, unlocks, career stats), handles profile deletion and migration. |
| `engine/http_manager.lua` | 23 | HTTP stub — thread template for HTTPS requests (crash reports, high scores). Minimal implementation. |

---

## Category: UI / Rendering — IGNORE for simulator

These files are pure presentation, input handling, or UI structure.
They contain no game-rule logic needed for simulation.

| File | Lines | Description |
|------|------:|-------------|
| `functions/UI_definitions.lua` | 6,436 | All UI layout definitions (`G.UIDEF.*`) — HUD, menus, shop UI, collection screens, settings panels, popups, card info boxes, run setup, credits. Purely declarative UI trees. |
| `functions/button_callbacks.lua` | 3,203 | `G.FUNCS.*` callbacks — button handlers, option cycles, text input, shop buy/sell/redeem wiring, `cash_out`, menu navigation, profile management, window settings. |
| `card_character.lua` | 164 | Card_Character class — Jimbo (tutorial mascot) rendering, speech bubbles, button overlays. Pure presentation. |
| `main.lua` | 388 | LÖVE2D entry point — `love.run()` game loop, `love.load()` Steam init, input handlers (keyboard/mouse/gamepad/touch), `love.resize`, `love.errhand` crash reporting. Framework glue. |
| `conf.lua` | 11 | LÖVE2D config — window title, dimensions, console toggle. |
| `functions/test_functions.lua` | 237 | Debug/dev tools — stress test UI, video playback controller, `aprint` debug overlay. |

---

## Category: Localization — IGNORE for simulator

| File | Lines | Description |
|------|------:|-------------|
| `localization/en-us.lua` | — | English strings (base language) |
| `localization/de.lua` | — | German |
| `localization/es_419.lua` | — | Latin American Spanish |
| `localization/es_ES.lua` | — | European Spanish |
| `localization/fr.lua` | — | French |
| `localization/id.lua` | — | Indonesian |
| `localization/it.lua` | — | Italian |
| `localization/ja.lua` | — | Japanese |
| `localization/ko.lua` | — | Korean |
| `localization/nl.lua` | — | Dutch |
| `localization/pl.lua` | — | Polish |
| `localization/pt_BR.lua` | — | Brazilian Portuguese |
| `localization/ru.lua` | — | Russian |
| `localization/zh_CN.lua` | — | Simplified Chinese |
| `localization/zh_TW.lua` | — | Traditional Chinese |

---

## External Dependencies (not Lua source)

| Item | Description |
|------|-------------|
| `bit` | LuaJIT bitwise operations library (built-in) |
| `luasteam` | Steam API bindings (platform .so/.dll) |
| `https` | HTTPS request library (for crash reports) |
| `resources/` | Textures, fonts, sounds, shaders, gamepad mappings |
| `version.jkr` | Compressed version metadata |

---

## Simulator Relevance Summary

### Must extract for simulator (core game rules):
- **`game.lua`** — `init_game_object` (G.GAME structure), `start_run` (run setup, deck building, stake/challenge application), `init_item_prototypes` (all card/blind/tag data definitions), state machine update functions
- **`card.lua`** — all card abilities + `calculate_joker` (the scoring engine)
- **`blind.lua`** — boss blind effects on scoring/hands
- **`back.lua`** — deck starting parameters + in-round effects
- **`tag.lua`** — tag trigger effects
- **`globals.lua`** — game state constants, hand list, starting config
- **`functions/state_events.lua`** — round/ante lifecycle
- **`functions/common_events.lua`** — `eval_card`, card creation, pool generation, unlock checks, scoring helpers
- **`functions/misc_functions.lua`** — poker hand evaluation, pseudorandom system, blind amount scaling
- **`challenges.lua`** — challenge definitions (static data)
- **`cardarea.lua`** — card container logic (hand size, highlighting, draw mechanics)

### Partially relevant (specific functions needed):
- **`functions/button_callbacks.lua`** — `G.FUNCS.buy_from_shop` (shop transaction logic embedded in UI callback code)
- **`engine/event.lua`** — event queue semantics (blocking, ordering) affect game logic execution order
- **`engine/string_packer.lua`** — save format if we need to load/inspect save files

### Safe to ignore entirely:
- `functions/UI_definitions.lua` (UI layouts)
- `card_character.lua` (Jimbo mascot)
- `main.lua` (LÖVE2D bootstrap)
- `conf.lua` (window config)
- `functions/test_functions.lua` (debug tools)
- All `localization/*.lua` files
- `engine/controller.lua` (input handling)
- `engine/node.lua`, `engine/moveable.lua`, `engine/sprite.lua`, `engine/animatedsprite.lua` (scene graph / rendering)
- `engine/ui.lua` (UI framework)
- `engine/particles.lua`, `engine/text.lua` (visual effects)
- `engine/sound_manager.lua` (audio)
- `engine/save_manager.lua`, `engine/profile.lua` (save I/O)
- `engine/http_manager.lua` (networking stub)

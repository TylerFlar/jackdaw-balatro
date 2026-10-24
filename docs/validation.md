---
layout: default
title: Validation Scenarios
---

# Validation Scenarios

Jackdaw validates its engine by playing scenarios on live Balatro in **lockstep** with a shadow
run in the simulator: every request goes to the shadow first and then to the game, and every
answer from the game is compared with the shadow's view on every structural field. When the
two diverge, the scenario fails, reports exactly which fields differ, and writes a repro bundle.

This is a primary tool for finding and fixing bugs in the engine. For organic games at scale,
see [Lockstep against the live game](live), which also explains bundles, replay and the
deviation catalog.

## Prerequisites

- Balatro 1.0.1o with Steamodded and [BalatroBot](https://github.com/coder/balatrobot) v1.5.2,
  listening (default: `127.0.0.1:12346`), on a fully unlocked profile: see
  [live.md](live#what-you-need)

## Running Scenarios

```bash
jackdaw validate                          # all 254 scenarios (275 results)
jackdaw validate --category jokers        # all 150 joker scenarios
jackdaw validate --category tarots        # all 22 tarot scenarios
jackdaw validate --category planets       # all 13 planet scenarios
jackdaw validate --category spectrals     # all 16 spectral scenarios
jackdaw validate --category boss_blinds   # 7 seed groups, all 28 bosses
jackdaw validate --category modifiers     # all 22 modifier scenarios
jackdaw validate --category tags          # all 24 tag scenarios
jackdaw validate --scenario joker_jolly   # single scenario by name
jackdaw validate --host 127.0.0.1 --port 12346  # custom connection
jackdaw validate --url http://127.0.0.1:12347   # or a URL
jackdaw validate --bundles validate-bundles/    # write repro bundles for failures
jackdaw validate --delay 0                # no pauses between actions (they are for watching)
jackdaw validate --sim                    # self-check against the in-process simulator
```

Each scenario reports:

- **PASS**: the game and the shadow agreed after every request. Differences the deviation
  catalog explains (for example `S08` for Familiar's suits under Steamodded) are listed as
  `explained`.
- **SKIP**: a request the catalog vetoes (for example `BB-SKIP-PACK`: skipping for a tag that
  opens a pack, which BalatroBot never answers), or the run ended before the scenario could test
  its target.
- **FAIL**: a divergence, with the differing fields and the bundle; or a request the simulator
  refused.

`jackdaw validate --sim` must pass everything: the simulator against itself.

## How It Works

Each scenario:

1. Gets a fresh `LiveAdapter` on the BalatroBot connection
2. Starts a run on the game (`menu` + `start`) and a shadow run from the seed, deck and stake the
   game reports
3. Uses `add`/`set` debug commands to inject specific game state
4. Executes actions: each request is taken by the shadow, sent to the game, and its answer
   compared with the shadow's view (a difference raises a `DivergenceError`)
5. Calls `compare_state()` for a last comparison of the game's `gamestate` with the shadow

## Scenario Files

All scenarios live in `jackdaw/cli/scenarios/`:

| File | Category | What it tests |
|------|----------|---------------|
| `jokers.py` | jokers | 150+ joker effects |
| `boss_blinds.py` | boss_blinds | Boss blind effects |
| `tarots.py` | tarots | Tarot consumables |
| `modifiers.py` | modifiers | Enhancements, editions, seals |
| `spectrals.py` | spectrals | Spectral cards |
| `planets.py` | planets | Planet cards |
| `tags.py` | tags | Tag effects |
| `helpers.py` | — | Shared utilities and runner patterns |
| `__init__.py` | — | Registry, `Scenario` and `ScenarioResult` types |

## Writing a Scenario

A scenario is a function `fn(ls, *, delay) -> ScenarioResult`, where `ls` is the lockstep (a
`LiveAdapter`). The helpers take it as their first argument; "both" in a helper's name means
the shadow and the game.

### Minimal Example

```python
from jackdaw.cli.scenarios import register, ScenarioResult
from jackdaw.cli.scenarios.helpers import (
    Lockstep,
    start_both,
    select_blind,
    add_both,
    play_hand,
    compare_state,
)

@register(
    name="my_scenario",
    category="jokers",
    description="Test that my joker does the right thing",
)
def _my_scenario(ls: Lockstep, *, delay: float = 0.3) -> ScenarioResult:
    # 1. Start a run on the game and its shadow
    start_both(ls, seed="MY_SEED", delay=delay)

    # 2. Get to the right game phase
    select_blind(ls, delay=delay)

    # 3. Inject the state you want to test
    add_both(ls, key="j_jolly")

    # 4. Execute an action (compared with the shadow as it happens)
    play_hand(ls, [0, 1, 2, 3, 4], delay=delay)

    # 5. One last comparison
    diffs = compare_state(ls, label="after play with j_jolly")
    return ScenarioResult(
        passed=len(diffs) == 0,
        diffs=diffs,
        details=f"My scenario: {'PASS' if not diffs else 'FAIL'}",
    )
```

If you're adding a scenario to a new file, register the import in `__init__.py`'s `get_all_scenarios()`.

### Run it

```bash
jackdaw validate --scenario my_scenario
jackdaw validate --sim --scenario my_scenario   # first, against the simulator itself
```

## Using Runner Patterns

Most scenarios follow a standard pattern. Instead of writing the full flow yourself, use a runner:

### Joker (passive, triggers on any hand)

```python
from jackdaw.cli.scenarios.helpers import run_joker_scenario

@register(name="joker_joker", category="jokers", description="+4 Mult")
def _joker_joker(ls, *, delay=0.3):
    return run_joker_scenario(ls, joker_key="j_joker", seed="J_JOKER", delay=delay)
```

This starts a game, selects blind, adds the joker, plays cards `[0,1,2,3,4]`, and compares.

### Joker (needs a specific hand)

```python
from jackdaw.cli.scenarios.helpers import run_joker_with_setup

@register(name="joker_jolly", category="jokers", description="+8 Mult if Pair")
def _joker_jolly(ls, *, delay=0.3):
    return run_joker_with_setup(
        ls,
        joker_key="j_jolly",
        hand_preset="PAIR",       # injects cards that form a pair
        delay=delay,
    )
```

Available hand presets (defined in `helpers.py`):

| Preset | Hand triggered |
|--------|---------------|
| `PAIR` | Pair |
| `TWO_PAIR` | Two Pair |
| `THREE_KIND` | Three of a Kind |
| `FOUR_KIND` | Four of a Kind |
| `FULL_HOUSE` | Full House |
| `STRAIGHT` | Straight |
| `FLUSH_HEARTS` | Flush (Hearts) |
| `FLUSH_DIAMONDS` | Flush (Diamonds) |
| `FLUSH_CLUBS` | Flush (Clubs) |
| `FLUSH_SPADES` | Flush (Spades) |
| `STRAIGHT_FLUSH` | Straight Flush |
| `FACE_CARDS` | Face cards (J, Q, K) |
| `EVEN_RANKS` | Even-ranked cards |
| `ODD_RANKS` | Odd-ranked cards |
| `FIBONACCI` | Fibonacci ranks (A, 2, 3, 5, 8) |
| `HIGH_CARD` | High Card (no pairs/straights/flushes) |

You can also pass a list of card keys directly instead of a preset name:

```python
run_joker_with_setup(ls, joker_key="j_hack", hand_preset=["H_2", "D_3", "S_4", "C_5", "H_2"])
```

### Consumable

```python
from jackdaw.cli.scenarios.helpers import run_consumable_scenario

@register(name="tarot_fool", category="tarots", description="The Fool")
def _tarot_fool(ls, *, delay=0.3):
    return run_consumable_scenario(
        ls,
        consumable_key="c_fool",
        targets=[0, 1],              # card indices to target (or None)
        delay=delay,
    )
```

### Card Modifier

```python
from jackdaw.cli.scenarios.helpers import run_modifier_scenario

@register(name="mod_glass_foil", category="modifiers", description="Glass + Foil Ace")
def _mod_glass_foil(ls, *, delay=0.3):
    return run_modifier_scenario(
        ls,
        card_key="H_A",
        enhancement="GLASS",
        edition="FOIL",
        seal="RED",
        delay=delay,
    )
```

## Debug API Reference

The parameters are BalatroBot's, as the game takes them.

### `add` — inject cards

```python
add_both(ls,
    key="j_jolly",              # required: card key
    edition="FOIL",             # optional: FOIL, HOLO, POLYCHROME, NEGATIVE
    enhancement="GLASS",        # optional: BONUS, MULT, WILD, GLASS, STEEL, STONE, GOLD, LUCKY
    seal="RED",                 # optional: RED, BLUE, GOLD, PURPLE
    eternal=True,               # optional: cannot be sold/destroyed
    perishable=5,               # optional: rounds until it is debuffed
    rental=True,                # optional: costs $1 per round
)
```

Card key naming:
- **Jokers**: `j_joker`, `j_jolly`, `j_zany`, ...
- **Tarots**: `c_fool`, `c_magician`, `c_high_priestess`, ...
- **Planets**: `c_mercury`, `c_venus`, `c_earth`, ...
- **Spectrals**: `c_familiar`, `c_grim`, `c_incantation`, ...
- **Vouchers and packs**: `v_overstock_norm`, `p_arcana_normal_1`, ... (in the shop)
- **Playing cards**: `{suit}_{rank}` — `H_A`, `S_T`, `D_K`, `C_2`, ...
  - Suits: `H` (Hearts), `D` (Diamonds), `C` (Clubs), `S` (Spades)
  - Ranks: `2`-`9`, `T` (10), `J`, `Q`, `K`, `A`

`add` is a debug command: the game builds the card through Steamodded, which differs from the
simulator in a few documented ways ([B03](live#b03); catalog entries `B03`, `B03-EDITION`).

### `set` — modify game state

```python
set_both(ls,
    money=100,        # dollars
    hands=10,         # hands remaining in round
    discards=7,       # discards remaining
    ante=5,           # current ante
    round=3,          # current round
    chips=500,        # chips scored this round
    shop=True,        # repopulate shop
)
```

## Helper Functions

All helpers take the lockstep `ls`:

### Setup

| Function | What it does |
|----------|-------------|
| `start_both(ls, seed, deck, stake, delay)` | Start a run on the game and its shadow |
| `select_blind(ls, delay)` | Select the current blind |
| `skip_blind(ls, delay)` | Skip the current blind |
| `add_both(ls, **params)` | Inject a card (`add`) |
| `set_both(ls, **params)` | Modify game state (`set`) |

### Actions

| Function | What it does |
|----------|-------------|
| `play_hand(ls, cards, delay)` | Play cards by index |
| `discard(ls, cards, delay)` | Discard cards by index |
| `use_consumable(ls, consumable, cards, delay)` | Use a consumable |
| `buy_card(ls, index, delay)` | Buy a shop card |
| `sell_joker(ls, index, delay)` | Sell a joker |
| `sell_consumable(ls, index, delay)` | Sell a consumable |
| `cash_out(ls, delay)` | Cash out after beating a blind |
| `next_round(ls, delay)` | Leave shop, advance to next round |
| `reroll_shop(ls, delay)` | Reroll shop offerings |

Any other request: `ls.request(method, params)`.

### Flow Control

| Function | What it does |
|----------|-------------|
| `play_through_blind(ls, max_hands)` | Play hands until blind is beaten or game ends |
| `advance_past_blind(ls, delay)` | Select blind, play through, cash out, next round |
| `get_state(ls)` | The game state (`SELECTING_HAND`, ...) |
| `get_hand_count(ls)` | Number of cards in hand |
| `current_blind_score(ls)` | The score the current blind needs |
| `gamestate(ls)` | The game's last view |

### Comparison

```python
diffs = compare_state(ls, label="after play")
```

Compares the game's `gamestate` with the shadow on every structural field once more and returns
the differences the deviation catalog doesn't drop, as strings. Every request has already been
compared as it was made, so this is normally empty; a difference during the scenario fails it
with a divergence before it gets here.

## Multi-Step Scenarios

For scenarios that need more than a single action, just chain helpers:

```python
@register(name="complex_example", category="jokers", description="Multi-step test")
def _complex_example(ls, *, delay=0.3):
    start_both(ls, seed="COMPLEX", delay=delay)

    # Play through the small blind first
    select_blind(ls, delay=delay)
    play_through_blind(ls, delay=delay)
    cash_out(ls, delay=delay)
    next_round(ls, delay=delay)

    # Now test our joker on the big blind
    select_blind(ls, delay=delay)
    add_both(ls, key="j_blueprint")
    add_both(ls, key="j_jolly")
    set_both(ls, money=50)

    play_hand(ls, [0, 1, 2, 3, 4], delay=delay)

    diffs = compare_state(ls, label="after big blind play")
    return ScenarioResult(passed=not diffs, diffs=diffs, details="Complex example")
```

## Grouped Sub-Results

Use `sub_results` to report multiple checks within one scenario:

```python
@register(name="multi_check", category="jokers", description="Multiple plays")
def _multi_check(ls, *, delay=0.3):
    start_both(ls, seed="MULTI", delay=delay)
    select_blind(ls, delay=delay)
    add_both(ls, key="j_joker")

    all_subs = []

    play_hand(ls, [0, 1, 2, 3, 4], delay=delay)
    diffs1 = compare_state(ls, label="hand 1")
    all_subs.append(("hand 1", ScenarioResult(passed=not diffs1, diffs=diffs1)))

    play_hand(ls, [0, 1, 2, 3, 4], delay=delay)
    diffs2 = compare_state(ls, label="hand 2")
    all_subs.append(("hand 2", ScenarioResult(passed=not diffs2, diffs=diffs2)))

    all_passed = all(r.passed for _, r in all_subs)
    return ScenarioResult(passed=all_passed, sub_results=all_subs, details="Multi check")
```

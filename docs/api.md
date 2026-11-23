---
layout: default
title: Python API
---

# The Python API

The game engine is written in Rust (`crates/jackdaw-core`) and runs in-process through the
`jackdaw._core` extension. `jackdaw.core` is its Python face. It reproduces Balatro 1.0.1o
exactly, down to the RNG: it was ported function by function from the game's Lua source, checked
step by step against a Python reference engine (whose results are frozen in the test fixtures),
against LuaJIT itself for the RNG and the hand evaluation, and in lockstep against the live game
([Lockstep against the live game](live)).

For reinforcement learning, start with the [RL Project Guide](rl-guide): the environments wrap
this API. Use `jackdaw.core` directly for scripting, analysis, search, or your own environment.

## A run

```python
from jackdaw.core import Game
from jackdaw.engine.actions import SelectBlind, PlayHand, Discard, CashOut, BuyCard, NextRound

game = Game.new("b_red", stake=1, seed="TUTORIAL")   # back key, stake 1-8, seed
game.step(SelectBlind())                              # the round starts, the hand is dealt
game.step(PlayHand(card_indices=(0, 2, 3)))           # hand indices
print(game.phase, game.chips, game.hands_left)
```

- **Creating a run.** `Game.new(back, stake, seed=None, challenge=None)`. Without a seed it
  draws one the way the game does. For a challenge, pass its id (`"c_omelette_1"`); the back is
  then the challenge deck.
- **Stepping.** `step(action)` takes an action dataclass from `jackdaw.engine.actions`, or its
  tape form (`["PlayHand", [0, 2, 3]]`), and plays until the game waits for the next decision.
- **Errors.** An action the rules refuse raises `IllegalActionError`, and the run is unchanged.
  `NotPortedError` would mean a missing rule; nothing raises it now that every rule is ported.

The actions are `SelectBlind`, `SkipBlind`, `RerollBoss`, `PlayHand`, `Discard`, `SortHand`,
`SwapHandLeft`/`Right`, `SwapJokersLeft`/`Right`, `CashOut`, `BuyCard`, `BuyAndUse`,
`SellCard`, `UseConsumable`, `RedeemVoucher`, `OpenBooster`, `PickPackCard`, `SkipPack`,
`Reroll` and `NextRound`.

## What is legal

```python
for action in game.legal_actions():
    print(action)
```

- **Order.** `legal_actions()` lists the actions in a fixed order.
- **Plays and discards.** `PlayHand` and `Discard` appear once each, with no cards, standing
  for any selection of 1-5 hand cards (Cerulean Bell's forced card must be included).
- **Consumables.** A consumable that targets cards carries a default selection.
- **Checking one action.** `game.check(action)` returns why the rules refuse an action, or
  `None`, without changing the run.

`legal_actions(Legality.BALATROBOT)` keeps only what the live
[BalatroBot](balatrobot) also executes. Use it for agents that will play the real game. It
drops sim-only actions such as `RerollBoss` and `BuyAndUse`, and moves the bot refuses.

## Reading the run

Scalars are properties: `phase`, `seed`, `ante`, `round`, `dollars`, `chips`, `hands_left`,
`discards_left`, `won`. Areas and the rest are read-only views:

| Property | Type |
|---|---|
| `hand`, `jokers`, `consumables`, `deck`, `discard_pile` | `tuple[CardView, ...]` |
| `shop_jokers`, `shop_vouchers`, `shop_boosters`, `pack_cards` | `tuple[CardView, ...]` |
| `blind` | `BlindView` (`key`, `chips` to beat, `dollars`, `boss`, `disabled`) or `None` |
| `hand_levels` | mapping hand name → `HandLevel` (`level`, `chips`, `mult`, `played`, `visible`) |
| `tags` | held tag keys |
| `last_score` | `Score` (`hand_type`, `chips`, `mult`, `total`, `dollars`, `debuffed`) or `None` |

A `CardView` has:
- `key` (the center: `"j_joker"`, `"c_fool"`, `"m_glass"`, `"c_base"`...);
- for playing cards, `front` (`"H_A"`), `suit` and `rank`;
- `edition`, `seal`, `debuff` and `face_down`;
- `cost` and `sell_cost`;
- the stickers;
- a read-only `ability`.

Views are frozen: they describe the run when they were read.

## Exact previews and search

```python
score = game.preview_play((0, 1, 2, 3, 4))   # the real play on a copy
print(score.hand_type, score.total)

branch = game.clone()                        # an independent copy, well under a microsecond
branch.step(Discard(card_indices=(5, 6, 7)))
```

`preview_play` runs the actual step on a copy, so it includes every joker, enhancement, edition,
seal, the boss and the RNG. The run itself is unchanged. `clone()` is cheap enough for tree
search. `scoring_cards(positions)` gives the hand those cards make and which of them would score
(`("Pair", [1, 3])`), without playing them.

### The range of a play that rolls

```python
low = game.preview_play((0, 1, 2), rolls="low")    # every chance check fails unless certain
high = game.preview_play((0, 1, 2), rolls="high")  # every chance check succeeds
```

A play with a Lucky Card, a Glass Card or a chance joker can score differently from one roll to
the next. `rolls="low"` scores it with every chance check failing unless it is certain, and
`rolls="high"` with every chance check succeeding, so
`low.total <= preview_play(cards).total <= high.total`. The modes bound the play's score, not
the board it leaves: under "high" a Glass Card also shatters and Gros Michel goes extinct.
A chance roll here is a check `roll < G.GAME.probabilities.normal / odds` (Lucky Cards' mult and
money, Glass Cards, Space Joker, 8 Ball, Business Card, Bloodstone, Reserved Parking,
Hallucination, Gros Michel, Cavendish, The Wheel of Fortune, The Wheel's face-down draws), plus
Misprint's mult (its minimum or maximum).
"Low" fires an effect only when it is certain (Oops! All 6s can make it so); "high" fires every
possible one. These rolls draw nothing from the RNG; the rolls that choose a card, a pack or an
edition (8 Ball's Tarot, The Soul) still draw as in play. The Wheel of Fortune picks its joker
and its edition from the stream its chance roll shares, so under a bound they can differ from
the ones the real use would pick.

The mode behind it is `Game.roll_bound` (`None`, `"low"` or `"high"`). It is not part of the
game: `clone()` carries it and `replace_state` keeps it, while `to_dict()` and `state_hash()`
leave it out (a run from `from_dict` starts without one). With it unset, every roll is the
game's own.

### Analysis boards

To score plays from a state the player has not reached (another hand, The Hook switched off),
edit a clone:

| Setter | The edit | Same board as `replace_state` of |
|---|---|---|
| `debug_set_hand(sort_ids)` | the hand becomes these cards, in order, from the hand or the deck; the deck keeps its other cards in order, then the hand cards not taken (so the deck's length changes only when the new hand's size differs from the old). In SELECTING_HAND only. | the snapshot with `areas.hand` and `areas.deck` so edited |
| `debug_set_blind_disabled(True)` | `blind.disabled`, written directly (no `Blind:disable` effects) | `blind.disabled` set |
| `debug_clear_forced_selection()` | Cerulean Bell's forced card is no longer forced or highlighted | `forced_selection` cleared on the hand cards |

Each gives the same `state_hash`, the same highlights and the same result for every play as
`replace_state` of the edited `to_dict()`, without the snapshot's round trip. Like `debug_set`,
they write state no player action reaches: use them on a copy.

## How a hand scored

```python
game.record_scoring = True
game.step(PlayHand(card_indices=(0, 1, 2, 3, 4)))
play = game.last_play                  # a Play; None after a step that played no hand
print(play.hand, play.level, play.score.total)
for e in play.events:
    print(e.kind, e.card, e.target, e.chips, e.mult, e.x_mult, e.hand_chips, e.hand_mult)
```

With `record_scoring` on, each play records how it scored, in the order `G.FUNCS.evaluate_play`
shows it. A `Play` has:
- `hand`, and its `level` when played (before any level-up);
- `played` and `scoring`, the cards' sort ids in play order;
- `events`;
- `score`, a `Score`.

Each `ScoreEvent` has a `kind`, the `card` it belongs to (where the game shows it) and its
`target` when that's another card. It records what changed: `chips` and `mult` added, `x_mult`
multiplied, `dollars` paid. It also carries the hand's chips and mult after it (`hand_chips`,
`hand_mult`), so the last event's pair is the score's.

| `kind` | What happened |
|---|---|
| `level_up` | a joker levelled the hand up before it scored (Space Joker): the new `level` and base |
| `hand` | scoring starts: the hand's base chips and mult at `level` |
| `blind` | the boss changed them (The Flint) |
| `card` | a scoring card's own chips, mult, x mult or money (enhancements, Gold Seal) |
| `held` | a held card's bonus (Steel Card) |
| `edition` | an edition's bonus, on a scoring card, a joker or a consumable |
| `joker` | a joker's effect, on `target` when it acts on a card |
| `retrigger` | `target` scores again, or counts again in hand: `card` is the Red Seal's card or the joker |
| `debuffed` | a scoring card the boss debuffs, which adds nothing |
| `not_allowed` | the boss doesn't allow the hand: it scores nothing |
| `plasma` | Plasma Deck balanced the chips and mult |
| `destroyed` | a scoring card destroyed after the hand: a Glass Card that `shattered`, or Sixth Sense's 6 |

Recording is off by default and changes nothing: the run, its snapshot and hash, the observations
and the masks are the same either way.

## What a player sees

```python
view = game.player_view(key=0x2F6A91C3)        # keep the key from whoever reads the view
for card in view["hand"]:
    print(card["id"], card["front"] if not card["face_down"] else "face down")
print(view["deck_size"], [e["front"] for e in view["deck"] if e["remaining"]])
```

`player_view(key)` returns the state a player sees, as plain JSON-compatible data
(`jackdaw.player_view/1`), with what they can't see left out. A search or rule-based agent that
reads only this can't use hidden information, and the environments return the same document
(`BalatroGymnasiumEnv.player_view()`, and Jackdaw Arena's agent protocol), so it runs the same
code in both places.

**Card ids.** A card's sort id gives its identity away (a run makes its deck in a fixed order),
so the view names each card by an id derived from its sort id, `key` and the times the player
has lost track of it. An id stays the same while the player can follow the card, a face-down
card's included: through sorts, swaps, a play, a shop purchase. A card comes back under a new id
after going into the deck, and a joker after Amber Acorn turns it face down and shuffles it.
Ids are integers below 2^53. Keep `key` from whoever reads the view and the same for a whole
run; the environments derive theirs from the episode's sampler seed.

**Cards** (in `hand`, `play`, `jokers`, `consumables`, the shop's lists and the pack's `cards`,
in their order) have `id`, `face_down`, `key`, `front`, `suit`, `rank`, `edition`, `seal`,
`debuff`, `cost`, `sell_cost`, `stickers` (`eternal`, `perishable`, `perish_tally`, `rental`,
`pinned`), `ability` (as the snapshot writes it), `shape`, `sellable`, `shaking` and `forced`.
A face-down card shows only its `id`, its place, `shape`, `sellable`, `shaking` and `forced`:
every other field is `null`, as the game shows its back and a `?` for its sell value. A Stone
Card has no `front`, `suit` or `rank`.

- **`shape`**: the card's size, where it isn't the standard one: `"half"` (Half Joker, 1.7
  times shorter), `"photograph"` (Photograph, 1.2 times shorter), `"square"` (Square Joker) or
  `"wee"` (Wee Joker, 0.7 times the size); `null` for every other card (card.lua:237-257). The
  back is drawn at the card's size, so it shows face down too, which places these four jokers
  under Amber Acorn.
- **`sellable`**: the card's sell button is active (`Card:can_sell_card`, card.lua:1649): a
  joker that isn't eternal, or a consumable; `false` anywhere else (the hand, the shop, a pack).
  The sell button shows whichever way the card faces (card.lua:4582), so a face-down eternal
  joker shows as one, though its sticker doesn't. Face up, it repeats `stickers.eternal`.
- **`shaking`**: the joker shakes because its effect is ready (`juice_card_until`): Trading
  Card until the round's first discard, DNA until its first hand, Loyalty Card before the hand
  that gives the x mult, Invisible Joker once it can be sold for a copy. It shows face down
  too, which places Amber Acorn's jokers before any hand.
- **`forced`**: Cerulean Bell's forced card.

| Field | What it holds |
|---|---|
| `phase` | as `Game.phase` |
| `run` | `back`, `stake`, `challenge`, `win_ante`, `won` |
| `game` | `ante`, `round`, `dollars`, `chips` (the round's score), `skips`, `hands_played` (in the run), `unused_discards`, `hand_size`, `joker_slots`, `consumable_slots`, the interest, discount, inflation, rental and bankruptcy rules, `probabilities_normal`, `last_tarot_planet` (The Fool's card), `last_hand_played`, `most_played_hand` (The Ox's), `blind_on_deck`, `boss_rerolled` |
| `current_round` | `hands_left`, `discards_left`, `hands_played`, `discards_used`, `reroll_cost`, `free_rerolls`, `cash_out` (what the Cash Out button pays at round evaluation; 0 elsewhere); and, while one of those jokers shows face up (owned, in the shop or in a pack), `idol_card`, `mail_card`, `ancient_card`, `castle_card` |
| `blind` | the blind being played: `key`, `name`, `chips`, `mult`, `dollars`, `boss`, `disabled`, `eye_hands` (The Eye), `mouth_hand` (The Mouth); `null` between rounds |
| `blinds` | `Small`, `Big`, `Boss` as blind select shows them: `key`, `state`, `chips` to beat, skip `tag` and the Orbital Tag's `orbital_hand` |
| `hands` | each poker hand's `level`, `chips`, `mult`, `played`, `played_this_round`, `visible` |
| `vouchers`, `tags` | redeemed vouchers and held tags, by key |
| `consumable_usage` | Tarots, Planets and Spectrals used in the run |
| `hand`, `play`, `jokers`, `consumables` | cards, as above, every one of them (no cap at 8 or 5) |
| `shop` | `jokers`, `vouchers`, `boosters` while the shop is open (in the shop, or in a pack opened there); else `null` |
| `pack` | `kind`, `choices_remaining`, `cards` while a pack is open; else `null` |
| `deck` | the deck view: every playing card, sorted by suit then rank, each with `key`, `front`, `suit`, `rank`, `edition`, `seal`, `perma_bonus` and `remaining`. No ids and no order |
| `deck_size`, `discard_size` | the draw pile's and the discard pile's sizes |
| `last_score` | as `Game.last_score` |
| `last_play` | with `record_scoring` on, the last hand's scoring as `last_play_json` gives it, its cards named by their view ids; else `null` |

**The deck** is what the game's deck view shows (UI_definitions.lua:3233): every playing card,
with `remaining` set for the cards still to draw and, as in the game, for the cards dealt face
down this round, wherever they are now, until the boss that dealt them is disabled. So the cards
a player can't place (in the draw pile, face down in the hand, discarded face down) count
together, and the deck's order never shows.

**Not in the view**: the seed, the RNG state, the cards of shops, packs and vouchers not shown
yet, sort ids, a card's play count. Non-finite numbers come as in snapshots (`{"$f": "inf"}`
in the JSON; `float("inf")` from `player_view`).

Two things the view reads aren't in snapshots, as in the game's own save: which cards were
dealt face down, and which jokers shake. `Game.from_dict` makes the first up so that the view
never shows more than the game would (the hand's face-down cards, and in a round that deals face
down, the whole discard pile, count as remaining), and starts with no joker shaking. The ids
start afresh there too.

## A game from the view

```python
from jackdaw.core import Game, ViewTracker

game = Game.from_player_view(view)                 # a Game: plan, preview, roll out on it

tracker = ViewTracker()                            # one per run: what the run's actions taught
game = tracker.update(env.player_view())
action = plan(game)                                # a factored action or an engine action
env.step_factored(action)
tracker.step(action)
game = tracker.update(env.player_view())           # built on the last game, stepped with it
```

An agent that plans with the engine (a search, rollouts, `preview_play`) needs a `Game`. In a
local run it can take the run itself; where it has only the player's view (Jackdaw Arena) it
has `Game.from_player_view(view, shadow=None, *, seed=None)`: a game whose every field a player
sees is the view's, and whose hidden parts are stand-ins. Its own `player_view()` shows what
`view` shows, card ids aside (and `last_play`, the scoring of a hand it never played), and it
offers the same legal actions and gives the same observation as the run. So the same agent
code runs on the same kind of state in both places.

- **The shadow** is a game built from an earlier view of the run, then stepped with the action
  taken since. It carries what the agent knows from its own actions though no view lists it:
  the consumables it used (Satellite counts the Planets), the cards it played this ante (The
  Pillar debuffs them when they come back from the draw pile), the hand sort it chose, the
  counters its purchases moved (Troubadour's hands, the vouchers' rates). The view then
  overwrites every field it shows. Without a shadow those come from a new run of the view's
  back, stake and challenge, with what the view's vouchers and jokers do to them.
- **Stand-ins**: the RNG and the seed (the shadow's, or a new run's of `seed`), the draw pile's
  order (the view gives it as a multiset), the cards dealt face down (taken from the cards the
  deck view counts as remaining, and left face down), a Stone Card's rank and suit, play
  counts, the Orbital Tag hands and the round's Idol, Mail-In Rebate, Ancient Joker and Castle
  cards while nothing shows them, and the ante's voucher before its shop. An agent that reads
  only what a player sees (no face-down card's identity, no deck order) reads no more than the
  view.
- **Amber Acorn** turns the jokers face down and shuffles them. Each face-down joker gets one
  of the shadow's that looks the same from the back: its size (`shape`), its sell button
  (`sellable`), whether it shakes. Those that look alike may trade places, as for a player.
- **Where the shadow rolled.** The shadow's step draws from its own RNG, so where the run drew
  other cards, rolled another boss or voucher, or shattered another Glass Card, the view's
  outcome replaces the shadow's. The sets that remember such outcomes follow the view's cards:
  the jokers seen (`used_jokers`), the bosses met, Gros Michel's extinction, the per-round hands
  and discards. A card the shadow's roll destroyed that the run kept comes back as it was. And
  where the shadow's last hand didn't end the round the run's did, the round's end happens in
  the rebuild (after a boss: the next ante's voucher, the end of the ante's plays).
- **Sort ids** follow the view's card ids. `game.view_ids` maps each id the view named to the
  card's sort id, and a card a player follows from one view to the next keeps its sort id.

`seed` gives the game a new RNG, other face-down stand-ins and a new draw pile order: rebuilds
of one view under several seeds are several guesses at what's hidden (a determinization). A
rebuild takes about 0.2 ms from Python.

`ViewTracker` keeps the shadow across a run: `update(view)` returns the game for a view (a
copy, to step as you like), `step(action)` gives it the action taken (a `FactoredAction`, its
`(action_type, entity_target, card_target)`, or an engine action). An action the last game
refuses (what it can't see differed) is dropped, and the next game builds on the last one as
it was. `reset()` starts another run; a view of a new run starts one too.
`BalatroGymnasiumEnv.view_game()` and Jackdaw Arena's `ArenaEnv.view_game()` keep one for you,
fed with the env's own steps.

## Building a state

```python
from jackdaw.core import GameBuilder

game = (
    GameBuilder("b_red", stake=2, seed="DEMO0001")
    .boss("bl_hook")
    .select_blind()
    .joker("j_blueprint")
    .joker("j_joker", edition="polychrome")
    .card("H_A", enhancement="m_glass", seal="Red")   # into the hand during a round
    .money(50)
    .build()
)
```

The additions go through the same operations as BalatroBot's debug commands `add` and `set`
(`Game.debug_add`, `Game.debug_set`, `Game.debug_restock_shop`).

## Saving and comparing runs

- `to_dict()` returns the canonical snapshot (`jackdaw.snapshot/1`,
  [schema](snapshot-schema)). It is plain JSON-compatible data and is also the save format;
  `Game.from_dict(snapshot)` restores it exactly.
- `state_hash()` is the SHA-256 digest of the canonical snapshot. Equal states have equal hashes.
- `replace_state(snapshot)` loads an edited snapshot into the run at the same decision point,
  keeping what a snapshot doesn't carry (the card-area settings BalatroBot's view reports, the
  hand's highlights, and the modes `record_scoring` and `roll_bound`). The live lockstep uses
  it to copy live values into its shadow run.

## BalatroBot

- `bot_view()` returns BalatroBot v1.5.2's `gamestate` view of the run, as the live bot would
  send it, except that text fields carry names.
- `bot_request(action)` returns the `(method, params)` request that performs an action live, or
  `None` for a sim-only action.
- `jackdaw serve` runs a BalatroBot-compatible JSON-RPC server on the simulator.
- `jackdaw.bridge.LiveAdapter` plays the live game in lockstep with a `Game` as its shadow, and
  `jackdaw lockstep` does it at scale: see [Lockstep against the live game](live).

See [BalatroBot on the simulator](balatrobot).

## Observations

`observation_buffer()` returns the RL observation (encoder v1) as raw float32 data, and
`jackdaw.env.encode_observation(game)` wraps it as an `Observation`; `jackdaw.env.get_action_mask(game)`
gives the action mask. The observation is the player's view: it leaves out the card identities
the player hasn't seen ([Player view](rl-guide#player-view)). The environments in `jackdaw.env`
(`BalatroEnvironment`, `BalatroGymnasiumEnv`, `BalatroVectorEnv`) build on these; see the
[RL Project Guide](rl-guide).

## Where the source doesn't decide

A few things the Lua source leaves open, and the engine's choices:

- **`pairs()` over the poker hands.** To Do List's hand, the Orbital Tag's hand and The Ox's
  tie-break follow the order `pairs(G.GAME.hands)` visits the hands in, which Lua doesn't
  define. The engine uses the order the game's own LuaJIT 2.0.5 gives (probed in its
  `lua51.dll`), confirmed against the live game.
- **Unseeded draws.** The first shop's Buffoon pack and the Charm and Meteor Tags' packs pick
  their `_1` or `_2` variant with an unseeded `math.random`. The variants hold the same cards; the
  engine always picks `_1`.
- **`unique_val`**, the last tie-break between two identical cards, is the creation order.
- **The profile.** Shop and pack pools leave out locked cards; the engine assumes a fully
  unlocked profile.
- **Where vanilla stops.** With two or more jokers, all debuffed, Crimson Heart's pick draws from
  an empty list and the game errors; the engine skips the pick. When the hand, the deck and the
  play area are all empty, vanilla queues `end_round` again every frame; the engine runs it once.

## Speed

On one core of a Ryzen 9 9950X3D (Windows, MSVC release build):

| Operation | Time |
|---|---|
| A new run | 4 µs |
| `clone()` | 0.3 µs |
| A play, no jokers | 1.0 µs |
| A play with five jokers | 2.1 µs |
| Cash out and fill the shop | 2.2 µs |
| Opening a booster pack | 1.7 µs |
| A reroll | 1.1 µs |
| An RNG draw | 0.05 µs |

`cargo test --release -p jackdaw-core --test perf -- --ignored --nocapture` prints these.

The environments step about 155,000 times a second (`BalatroEnvironment`) and 180,000 times a
second (`BalatroGymnasiumEnv`) on one core, and `BalatroVectorEnv` about 1.1 million times a
second on 8 threads (256 envs), counting the time inside the environment, not the policy's
(`uv run pytest -m benchmark`).

# Changelog

All notable changes to jackdaw. The version follows [semantic versioning](https://semver.org);
while it is 0.x, a minor release (0.2 → 0.3) can break things, and a patch release only fixes them.

## Unreleased

## [0.2.1] - 2026-10-24

Documentation only: the engine, the environments and the API are unchanged, so a seed plays the
same run as in 0.2.0, with the same observations and actions.

### Documentation

- [Jackdaw Arena](https://arena.tylerflar.com), a hosted, verifiable benchmark for jackdaw
  agents: a README section and a docs page on evaluating an agent and verifying its episodes.
- Install from `main`, which only moves at a release, or from a release tag. `dev` is now the
  default branch, where pull requests go.

## [0.2.0] - 2026-10-23

The game engine is now written in Rust and runs in process through the `jackdaw._core`
extension. It follows Balatro 1.0.1o's Lua source function by function, down to the RNG, and
was checked against the game in lockstep through BalatroBot. The engine runs about 100 times
faster than 0.1's Python engine and the environments 15 to 30 times faster, and many of its
rules changed (below), so a seed doesn't play the same run as in 0.1.

### Breaking changes

- **The Python engine is gone.** `jackdaw.engine` keeps only the action types
  (`jackdaw.engine.actions`) and the data tables. The game is `jackdaw.core.Game`, with read-only
  views instead of the mutable state dict.
- **`info["raw_state"]` is now `info["game"]`**, a live `jackdaw.core.Game` (read it at once or
  keep `.freeze()`); likewise `Transition.game` and `adapter.game`.
- **The action space (version 2).** Two action types are added, 21 `RerollBoss` and 22
  `BuyAndUse` (ids 0-20 keep their meaning; a policy with one output per type needs two more).
  The jokers can be reordered at every decision point, as the game's drag and drop allows. At
  most 10 free moves (swaps and sorts) in a row by default (`max_free_moves`). Policies trained
  on 0.1 need retraining: `jackdaw.env.ACTION_VERSION` (2) numbers the action space from now on.
- **Refused actions end the episode.** An action the masks allow but the engine refuses (an env
  bug) truncates the episode, with the reason in `info["refused"]` / `info["balatro/refused"]`.
  `BalatroGymnasiumEnv` used to raise; `on_refusal="raise"` keeps that.
- **The Gym action table's sampler** (when a state has more than 500 actions, or more than 200
  card selections for one action) is a seeded xoshiro256** instead of numpy's generator.
- **Installing from source needs a Rust toolchain** (installing from git builds the extension).
- **`BridgeAdapter` over a live backend is `LiveAdapter`**: a lockstep with a shadow run, whose
  observations and masks come from the shadow.

### Rules

115 differences from the game are fixed (a few only in state no rule reads). They were found by
auditing every gameplay function, branch and RNG call of the Lua source, and by lockstep runs
against the live game. Among them:

- Decks: the Green Deck's economy, the Magic and Ghost Decks' starting consumables, the Anaglyph
  Deck's Double Tag.
- Challenges: starting jokers and consumables, pinned jokers, every rule (`chips_dollar_cap`,
  `flipped_cards`, `inflation`, ...), and the bans in the boss, tag and voucher pools.
- Blinds: `Blind:disable` in full (Chicot, Luchador, Verdant Leaf), The Ox, Amber Acorn, Crimson
  Heart, The Manacle, the boss reset at defeat, and the debuff checks after cards change.
- Jokers: Blueprint and Brainstorm's copy rules, Satellite, Bull and Bootstraps, Mime, Hologram,
  8 Ball, Hallucination, Ceremonial Dagger, Madness, Riff-raff, Mr. Bones, Joker Stencil and the
  other per-frame values.
- Consumables and cards: editions through `set_edition` and `set_cost`, Ectoplasm, Ankh, copies,
  sticker eligibility, playing-card prices, hand visibility, card order in the deck and the
  hand, `SortHand`.
- Tags, the shop and vouchers: Orbital, Boss, Coupon and pack tags, Overstock, the shop's
  purchase counts, Negative cards at full slots.
- The RNG: LuaJIT's NaN handling, exact `%.13f` and `%.14g` ties, and the order the game's own
  LuaJIT visits the poker hands in.

### Added

- `jackdaw.core.Game`: `legal_actions(Legality.VANILLA | BALATROBOT)`, `check`, `preview_play`,
  `scoring_cards`, `clone`, `to_dict` / `from_dict` (the `jackdaw.snapshot/1` schema),
  `state_hash`, `replace_state`, `bot_view`, `bot_request` and the debug operations; `GameBuilder`
  builds a run in a chosen state.
- `BalatroVectorEnv`: many Gym envs stepped together in Rust, with the built-in rewards computed
  in Rust, `AutoresetMode.NEXT_STEP` or `SAME_STEP`, and partial resets
  (`options={"reset_mask": ...}`).
- `max_free_moves`, `on_refusal` and `action_version` on every env.
- BalatroBot v1.5.2's API on the simulator: `jackdaw serve`, `SimBackend`, and the BALATROBOT
  legality that keeps only what the live bot executes.
- `LiveAdapter`, `jackdaw lockstep` and `jackdaw lockstep replay`, with the deviation catalog for
  the live game's Steamodded and BalatroBot differences; `jackdaw validate` runs on the lockstep.
- A MaskablePPO training demo in `examples/`, with its results.
- Wheels for Linux, macOS and Windows (CPython 3.12+, abi3), built on release tags.

### Performance

| | 0.1 (Python) | 0.2 (Rust) |
|---|---:|---:|
| A new run | 327 µs | 4 µs |
| A play | 124 µs | 1.0 µs |
| Cash out and shop | 326 µs | 2.2 µs |
| A reroll | 153 µs | 1.1 µs |
| Copying a run | 563 µs | 0.3 µs |
| `BalatroEnvironment` | 10-12K steps/s | 155K steps/s |
| `BalatroGymnasiumEnv` | 5.4K steps/s | 180K steps/s |
| `BalatroVectorEnv`, 8 threads | | 1.1M steps/s |

## [0.1.0]

The first version: the engine in Python.

---
layout: default
title: Lockstep against the live game
---

# Lockstep against the live game

`jackdaw lockstep` plays organic games on the real Balatro through
[BalatroBot](https://github.com/coder/balatrobot) and checks every step against the simulator.
Each live run has a **shadow** run in the Rust engine, started from the live run's seed, deck and
stake. For every request:

1. the shadow takes it first, through the simulator's port of the same BalatroBot endpoint
   ([BalatroBot on the simulator](balatrobot)): the bot's checks, then vanilla's rules. A
   request the shadow refuses never reaches the game;
2. the [deviation catalog](#the-deviation-catalog) may veto it;
3. it goes to the live game;
4. the live game's answer (its `gamestate` view) is compared with the shadow's `bot_view()`, field
   by field.

Differences the catalog explains are dropped or *reconciled* (the shadow takes the live value).
Anything else stops the game with a **divergence**: a repro bundle is written, and
`jackdaw lockstep replay` reruns it on the simulator alone. This is the scaled-up form of the
lockstep runs that found the engine's bugs #15–#74.

The same machinery is a Python adapter, `LiveAdapter`, and `jackdaw validate` runs its scenarios
on it (see [From Python](#from-python) and [validation.md](validation)).

## What you need

- **Balatro 1.0.1o** (the version the simulator follows), with
  [Lovely](https://github.com/ethangreen-dev/lovely-injector),
  [Steamodded](https://github.com/Steamodded/smods) and **BalatroBot v1.5.2** installed. No
  other mods: the catalog covers Steamodded `1.0.0~BETA-1503a` and BalatroBot 1.5.2
  (`docs/parity/smods-diffs.md`); anything else changes the game in ways nothing accounts for.
- **A fully unlocked profile.** Shop and pack pools leave out locked cards, and the simulator
  assumes everything is unlocked. A partly locked profile diverges at its first shop.
- This repository installed (`uv sync --dev`), so `jackdaw` is on the path.

Before pointing it at the game, check the setup against the simulator, which needs nothing
running (`sim` is an in-process BalatroBot on the simulator):

```bash
jackdaw lockstep --url sim --seeds 10
```

or against `jackdaw serve`, which is BalatroBot's HTTP API on the simulator:

```bash
jackdaw serve --port 12346 &
jackdaw lockstep --seeds 10
```

Both must finish with `unclassified divergences: 0`.

## Starting Balatro with BalatroBot

BalatroBot listens on `http://127.0.0.1:12346` once the game is running. Start the game the way
BalatroBot's documentation describes (its `balatrobot serve` launcher, for example
`uvx balatrobot serve --fast --no-audio --love-path <path to Balatro>`), or start Balatro with the
mod installed and its environment variables set (`src/lua/settings.lua` in the vendored
BalatroBot):

| variable | default | meaning |
|---|---|---|
| `BALATROBOT_PORT` | `12346` | the port the API listens on |
| `BALATROBOT_HOST` | `127.0.0.1` | the address it binds |
| `BALATROBOT_FAST` | `0` | `1`: fast mode (shorter animations) |
| `BALATROBOT_HEADLESS` | `0` | `1`: no window |
| `BALATROBOT_GAMESPEED` | `4` | the game speed |

Fast mode only changes timing (smods-diffs.md B04). Whatever the speed, an endpoint answers once
the game reaches the state it waits for, which can come before the game's queued events have
all run; the lockstep allows for that (see [timing](#timing)).

Check that it answers:

```bash
curl -s -X POST http://127.0.0.1:12346 -d '{"jsonrpc":"2.0","method":"health","id":1}'
```

## Ports and instances

One BalatroBot instance plays one game at a time, so lockstep scales by running several game
processes, each with its own port (`BALATROBOT_PORT=12346`, `12347`, ...), and giving each
instance a worker:

```bash
jackdaw lockstep --instances 4 --seeds 200            # http://127.0.0.1:12346 ... :12349
jackdaw lockstep --url http://127.0.0.1:12346 --url http://127.0.0.1:12350 --seeds 50
jackdaw lockstep --instances 2 --host 192.168.1.20    # :12346 and :12347 on that host
```

Without `--url`, instance *i* is `http://HOST:(12346 + i)`. With `--url` (repeatable) there is
one instance per URL. The games are shared out as the instances free up.

## Running `jackdaw lockstep`

```bash
jackdaw lockstep --instances 4 --seeds 200 --decks RED,BLUE --stakes WHITE,GOLD --steps 1000
```

| option | default | |
|---|---|---|
| `--seeds` | `10` | a count of random seeds (vanilla's alphabet, from `--rng-seed`), a comma-separated list, or `@file` with one seed per line |
| `--decks`, `--stakes` | `RED`, `WHITE` | BalatroBot's names; the games cycle through every deck/stake combination |
| `--steps` | `1000` | actions per game at most |
| `--policy` | `smart` | `smart` plays to get deep (it tries every play on a copy of the shadow and keeps the best, buys jokers, planets, vouchers and packs, and takes a random move now and then); `random` picks a random kind of move, then a random instance of it (half its plays are the best one, so runs still reach shops and packs) |
| `--instances`, `--url`, `--host` | one instance on `:12346` | see above |
| `--out` | `lockstep-out/<time>` | where the bundles and `summary.json` go |
| `--timeout` | `30` | seconds to wait for each live answer |
| `--settle` | `0` | seconds to wait after each live answer, then compare a fresh `gamestate` instead of the answer (see [timing](#timing)) |
| `--recheck` | `2` live, `0` on the simulator | while a live answer differs from the shadow, read `gamestate` again for up to this many seconds and compare the last reading (see [timing](#timing)) |
| `--endless` | off | keep playing after a win |
| `--fail-fast` | off | stop at the first unclassified divergence |
| `--json` | off | print the summary as JSON |

A game ends when it is won or lost, at `--steps`, or at a divergence. Each finished game prints a
line; the end prints the totals: games, steps, outcomes, divergences, the requests and the kinds
of action taken, and how often each catalog entry explained a difference, reconciled the shadow
or vetoed an action. The exit code is 0 when no game diverged unclassified, 1 otherwise, and 2
when an instance can't be reached.

Outcomes: `won`, `lost`, `max-steps`; `divergence` (unclassified: a new difference to triage);
`classified` (a catalog entry explains the difference but the shadow can't take the live value,
so the run stops); `stuck` (every move the policy tried was refused or vetoed); `not-ported` and
`error` (the simulator or the client failed).

`summary.json` holds the totals and every game's details (seed, deck, stake, instance, outcome,
steps, ante, the counts above, and the divergence with its bundle path).

## What a bundle contains

A divergence writes a directory under `<out>/bundles/`, named after the time, the instance, the
seed and the step:

| file | |
|---|---|
| `bundle.json` | the start parameters (the `start` request and the seed, deck, stake and back read back from the game), the **action log** (every request the shadow took, with the engine action it came from, and every reconcile as snapshot edits), the divergent step, and the **field diff**: each `{path, sim, live}` that differs and isn't explained, the differences classified under a catalog entry, and the catalog entries whose trigger fired at that step. Also the card id pairing so far, the catalog's run memory and the counts of explained, reconciled and vetoed differences. |
| `sim_view.json`, `live_view.json` | both views after the divergent step |
| `sim_view_before.json`, `live_view_before.json` | both views before it |
| `shadow_before.json` | the shadow's canonical snapshot before the step (`Game.from_dict` loads it) |
| `summary.txt` | the above in a few lines, with the replay command |

Paths name fields as the view nests them: `money`, `round.chips`, `hands.Pair.level`,
`blinds.boss.name`, `hand.cards[3].modifier.seal`. Card ids differ between the games (live ids
are a session-wide counter): cards are paired by position within their area and compared on
content, and a path ending in `.id` means a pairing contradicts an earlier one (the same faces in
the same places, but two cards that changed places). Text fields are never compared: the live
game sends localized descriptions where the simulator sends names (`value.effect`, a blind's
`effect` and `tag_effect`, `used_vouchers`' descriptions).

The divergence kinds: `view` (the views differ), `reconcile` (a reconcile couldn't make them
agree), `live-refused` (the live game refused a request the shadow took) and `live-error` (it
didn't answer: a timeout or a lost connection; `live_view.json` is whatever `gamestate` said
afterwards, if anything).

## Replaying and triaging a bundle

```bash
jackdaw lockstep replay lockstep-out/20261023-101500/bundles/20261023-101734-i0-7HE6IQX3-step58
```

Replay needs no game. It starts a run from the bundle's start parameters, takes the log's
requests through the simulator's BalatroBot layer (reapplying the recorded reconciles), checks
that the result is the recorded shadow, takes the divergent request, and compares the
simulator's view now with the recorded live view. It reports:

- whether the log still replays and gives the recorded shadow (if not, the simulator has changed
  since the bundle was written, before the divergent step);
- how the simulator's view of the divergent step differs from the recorded one;
- the differences from the live view now, and what the catalog makes of them.

Exit code 1: the divergence reproduces; 0: the simulator now agrees with the live game; 2: the
bundle can't be replayed. `--json` prints the report as JSON.

To triage:

1. Read `summary.txt`: the step, the request and the fields. The catalog entries listed as
   active at that step are the known differences that could be involved.
2. Find where the difference starts. `sim_view_before.json` / `live_view_before.json` agree (the
   previous step compared clean), so the cause is this request. `Game.from_dict(shadow_before)`
   gives the shadow at that point to experiment with.
3. Decide what it is:
   - **A simulator bug** (the live game is vanilla here): fix the engine, add a regression test,
     and check that `jackdaw lockstep replay` now says *not reproduced*.
   - **A view bug** (the game state agrees but the simulator reports it differently): fix
     `crates/jackdaw-bot/src/view.rs`.
   - **A Steamodded or BalatroBot difference** (the live game is not vanilla here): describe it
     in `docs/parity/smods-diffs.md` or `docs/balatrobot.md` and add a catalog entry (below).
     The simulator keeps following vanilla.
4. Run the seed again: `jackdaw lockstep --seeds <SEED> --decks <DECK> --stakes <STAKE>`.

### Timing

BalatroBot answers a request once the game reaches the state it waits for: `play` answers when
the game is back in `SELECTING_HAND` (or at the round's evaluation, or at game over), `select`
once the hand is dealt, `cash_out` once the shop has its cards. Work the game queues behind that
point can still be pending when the answer is built: Steamodded runs some contexts in trailing
events (smods-diffs.md S02), the boss's debuffs outlast the `play` that beats it
(BB-BOSS-DEFEAT), and a tag's shop effect (the D6 Tag's free reroll, the Voucher Tag's voucher,
the edition tags) runs in its own animated events after `cash_out` answers.

So when a live answer differs from the shadow in a way the catalog doesn't explain, the lockstep
reads `gamestate` again, every 0.1 s for up to `--recheck` seconds (2 by default against a live
game), and compares the last reading: what the game settles to without another request is what
the simulator models. A difference that lasts is reported as usual. The summary counts the
rechecked answers by request (`rechecked: cash_out 12`).

One race changes the game rather than the answer: a `cash_out` sent as soon as `play` answers can
pay the previous round's total (BB-CASHOUT-RACE). `LiveBackend` leaves a second before a
`cash_out` so it doesn't happen, and the shadow takes the live money if it does.

A refusal can be the same race: right after a pack pick, `use` is refused while the pick's
`G.GAME.STOP_USE` holds. The shadow took the request and a refused request changes nothing, so a
refusal for the game's state (`INVALID_STATE`, `NOT_ALLOWED`) is asked again within the same
`--recheck` window before it counts as a divergence (`rechecked: use (refused) 3`).

`--settle 0.5` is the blunt form: the lockstep waits half a second after every answer and
compares a fresh `gamestate`.

## The deviation catalog

`jackdaw/bridge/deviations.py` lists every known difference between the live game and the
simulator, each under the id of the document that describes it, with what the lockstep does
about it:

- **ignore**: a difference on its paths is dropped (and counted);
- **reconcile**: the live value is copied into the shadow, when the snapshot can carry it, and
  the step must then compare clean;
- **veto**: the action isn't taken in lockstep: it is left out of the legal actions and refused
  (`VetoedActionError`) before it reaches the game.

An entry explains a difference only when it is *active* at that step: its trigger fired (the
step is one the deviation can affect, e.g. selecting a blind while holding Marble Joker), or,
for a sticky entry, fired earlier in the run. A difference an active entry explains but can't
repair is a *classified* divergence.

| id | mode | what |
|---|---|---|
| [S01](parity/smods-diffs#s01) | reconcile | Marble Joker's Stone joins the deck after the round's shuffle: the shadow takes the live deal order |
| [S02](parity/smods-diffs#s02) | reconcile | Certificate's card comes after Cerulean Bell's pick: the shadow takes the live hand and forced card |
| [S03](parity/smods-diffs#s03) | reconcile | Certificate draws from `'stdseal'..ante`: a Standard pack opened in that ante takes the live seals |
| [S04](parity/smods-diffs#s04) | reconcile | hand-size changes refill the hand; The Manacle's disable draws 1: the shadow takes the live hand/deck split |
| [S04-GAME-OVER](parity/smods-diffs#s04) | ignore | a lost round's hand-size changes: Steamodded applies them, vanilla's never run (the game pauses at GAME_OVER) |
| [S05](parity/smods-diffs#s05) | reconcile | Invisible Joker's and Ankh's copies: their prices and a stripped Negative's slot take the live values (for the rest of the run) |
| [S06](parity/smods-diffs#s06) | reconcile | Seeing Double with one Wild Card: the live chips |
| [S07](parity/smods-diffs#s07) | veto | Four Fingers with three or more Wild Cards: plays not taken |
| [S08](parity/smods-diffs#s08) | reconcile | Sigil, Grim, Familiar, Incantation: the live fronts |
| [S09](parity/smods-diffs#s09) | reconcile | perishable stickers of debuffed jokers don't tick: the live tally |
| [S10](parity/smods-diffs#s10) | reconcile | xMult rounding: a play's chips within 1 of the live value |
| [S11](parity/smods-diffs#s11) | ignore | the unseeded `_1`/`_2` variant of the first shop's Buffoon pack and the Charm/Meteor packs |
| [S12](parity/smods-diffs#s12) | ignore | `pseudoseed` while paused (BalatroBot never pauses) |
| [S13](parity/smods-diffs#s13) | veto | Crimson Heart with every joker debuffed: plays and discards not taken |
| [S14](parity/smods-diffs#s14) | ignore | the deck's card limit (`cards.limit`) follows the playing cards live |
| [B01](parity/smods-diffs#b01) | veto | pack picks vanilla greys out (refused by the simulator's `pack` endpoint) |
| [B02](parity/smods-diffs#b02) | veto | play/discard/use leaving out Cerulean Bell's forced card |
| [B03](parity/smods-diffs#b03) | reconcile | debug `add` of an enhanced card: the live debuff |
| B03-EDITION | reconcile | debug `add` with an edition skips the edition poll (gaps.md B15): later shop/pack editions and prices (sticky) |
| [B04](parity/smods-diffs#b04) | ignore | BalatroBot's runtime settings (no gameplay effect) |
| BB-SIM-ONLY | veto | sim-only moves: RerollBoss, BuyAndUse, and what vanilla allows but the bot refuses ([command-space policy](balatrobot#command-space-policy)) |
| BB-ETERNAL | veto | selling an eternal joker |
| BB-SKIP-PACK | veto | skipping for a tag that opens a pack: the live `skip` never answers |
| BB-NONFINITE | veto | a step whose view would hold NaN or infinity: the live server sends nothing |
| BB-TEXT | ignore | text fields (never compared) |
| BB-SHOP-AREAS | ignore | the shop's areas after leaving the shop (gaps.md B21, read from the code; the count settles it) |
| BB-BOSS-DEFEAT | ignore | the `play` that beats a boss answers before `Blind:defeat` has undone the boss ([timing](balatrobot#differences-from-the-live-game)): the cards it debuffed, Crimson Heart's joker and its passive, The Manacle's hand size, Amber Acorn's face-down jokers; at the win screen, which pauses the game, a later `gamestate` check too |
| BB-CASHOUT-RACE | reconcile | a `cash_out` sent as soon as `play` answers can pay the previous round's total ([timing](balatrobot#differences-from-the-live-game)); `LiveBackend` waits before `cash_out` so it doesn't, and the shadow takes the live money if it does |
| BB-REARRANGE-ORDER | ignore | `rearrange` writes `ability.order` (not in the view) |
| BB-SORT-MODE | ignore | `SortHand` goes out as a permutation; the shadow takes the same move |

A reconcile's edits are recorded in the action log, so a later divergence still replays.

## Adding a deviation entry

When triage shows a new Steamodded or BalatroBot difference:

1. **Document it** in `docs/parity/smods-diffs.md` (a Steamodded patch: the next `S` id, with an
   `<a id="sNN">` anchor) or `docs/balatrobot.md` (BalatroBot's command space or view). The entry
   says what differs, when, and why the simulator keeps vanilla.
2. **Add the entry** to `CATALOG` in `jackdaw/bridge/deviations.py`:

   ```python
   Deviation(
       "S14",
       "What differs, in one line",
       "docs/parity/smods-diffs.md#s14",
       Mode.RECONCILE,
       paths=("hand.cards[*].modifier.seal",),   # view paths it can make differ
       trigger=_s14_trigger,                     # StepContext -> bool: may this step be affected?
       reconcile=_s14_reconcile,                 # the snapshot edits that copy the live values
       note="How the lockstep handles it.",
   ),
   ```

   - The **trigger** sees the request (`ctx.method`, `ctx.params`, `ctx.action`), the shadow
     before and after it (`ctx.pre`, `ctx.post`, their views and snapshots) and the run's memory.
     Keep it as narrow as the deviation: an entry that is always active hides real bugs on its
     paths. `sticky=True` keeps it active for the rest of the run once it fired.
   - **Paths** are globs over the diff paths: `*` is one key, `[*]` one index, `**` anything,
     and a path also covers everything below it (`hand` is the whole hand area). `accepts` can
     look at the values too (S10 accepts a difference of at most one chip).
   - A **reconcile** returns edits to the shadow's canonical snapshot
     (`jackdaw/bridge/reconcile.py`: set a field, set a card's field by sort id, or rearrange the
     hand and deck). The helpers there pair live cards with shadow cards (by id, then by face) and
     build a playing card's `base` for a new front. Raise `ReconcileError` when the live value
     can't be carried: the divergence is then classified under the entry.
   - A **veto** returns the reason the step isn't taken (`veto=`), or recognises the shadow's
     refusal (`refusal=`). Vetoed actions are also left out of `get_legal_actions()`.
   - An **ignore** entry only needs paths (and a trigger when it applies to some steps only).
3. **Test it**: `tests/live/test_live_adapter.py` makes `FakeLive` (the simulator's BalatroBot
   session) show the deviation after a request, and checks that the step is reconciled (or
   vetoed, or ignored) and the run goes on in step. `tests/live/test_deviations.py` checks that
   every entry's source and anchor exist.

## From Python

`LiveAdapter` follows the env's adapter protocol (`reset`, `step`, `get_legal_actions`, `done`,
`won`, `game`): observations, masks and legal actions come from the shadow, under BALATROBOT
legality, so an agent sees exactly what it saw in training.

```python
from jackdaw.bridge import LiveAdapter, LiveBackend
from jackdaw.env import BalatroEnvironment

env = BalatroEnvironment(
    adapter_factory=lambda: LiveAdapter(LiveBackend(url="http://127.0.0.1:12346"),
                                        bundle_dir="bundles"),
    back_keys=["b_red"], stakes=[1],
)
obs, mask, info = env.reset(seed="ABCD1234")   # info["game"] is the shadow run
```

`BridgeAdapter(LiveBackend(...))` is the same adapter. A step raises `DivergenceError` on an
unexplained difference (with `.diffs`, `.classified`, `.bundle_path`); a vetoed action ends the
episode as truncated, with the reason in `info["vetoed"]`.

In BalatroBot's terms:

```python
adapter = LiveAdapter(LiveBackend())
adapter.start("RED", "WHITE", "ABCD1234")       # menu + start; returns the live view
adapter.request("add", {"key": "j_joker"})       # any request, in lockstep (debug ones too)
adapter.step(SelectBlind())                      # an engine action, translated
adapter.check()                                  # compare the live gamestate once more
```

BalatroBot's `start` can't begin a challenge: start it in the game, then
`adapter.attach(challenge="c_omelette_1")` shadows the run from its first blind select.

## How it is checked

With no game available, the tests use the simulator as the live game (`jackdaw serve`, or the
in-process `FakeLive` with live-style card ids), where nothing may diverge:

- `tests/live/test_live_adapter.py`, `test_lockstep_cli.py`, `test_viewdiff.py`,
  `test_deviations.py` (default suite): the adapter, injected differences with their bundles and
  replays, every reconcile and veto on the deviation it exists for, the env on the adapter, and
  `jackdaw lockstep` / `jackdaw validate` over HTTP.
- `tests/live/test_lockstep_long.py` (`pytest -m lockstep`): 240 organic games in process and 200
  seeds across four deck/stake combinations on two `jackdaw serve` instances, every kind of move,
  no divergence; every `jackdaw validate` scenario over HTTP.

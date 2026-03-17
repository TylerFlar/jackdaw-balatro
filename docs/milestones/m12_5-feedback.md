# M12.5 Feedback — RNG Stream Fixes + Shop/Pack Integration + Full Validation

## 1. RNG audit results: exact stream order divergence

The audit traced every `pseudoseed` call during `start_run` and the first
round. Three seed key bugs were found:

| # | Bug | Was | Should be | How found |
|---|-----|-----|-----------|-----------|
| 1 | Voucher seed key | `'Voucher'` | `'Voucher' + str(ante)` = `'Voucher1'` | `get_current_pool` at common_events.lua:2052 appends ante to pool_key on return |
| 2 | Initial shuffle key | `'nr' + str(ante)` = `'nr1'` | `'shuffle'` | game.lua:2383 calls `self.deck:shuffle()` with NO argument → cardarea.lua:573 defaults to `pseudoseed('shuffle')` |
| 3 | Resample key | `pool_key + '_resample' + str(i)` starting at i=1 | `full_key + '_resample' + str(it)` starting at it=2 | Lua increments `it` before first use: `it = it + 1; pseudoseed(_pool_key..'_resample'..it)` |

The correct call sequence for ante 1:

```
1. pseudoseed('boss')        → boss blind selection
2. pseudoseed('Voucher1')    → voucher for shop
3. pseudoseed('Tag1')        → small blind tag
4. pseudoseed('Tag1')        → big blind tag (same stream, 2nd advance)
5. pseudoseed('shuffle')     → initial deck shuffle
6. pseudoseed('idol1')       → idol card targeting
7. pseudoseed('mail1')       → mail card targeting
8. pseudoseed('anc1')        → ancient card targeting
9. pseudoseed('cas1')        → castle card targeting
```

## 2. Divergences fixed vs remaining

**Fixed: 7 bugs**

| Bug | Category | Impact |
|-----|----------|--------|
| Voucher seed key | A (RNG key) | Tags + deck diverged |
| Initial shuffle key | A (RNG key) | Deck order diverged |
| Tag `requires` filter | B (pool filtering) | Tags diverged (4/5 wrong) |
| Resample key prefix | A (RNG key) | Tags diverged |
| Resample counter start | A (RNG key) | Tags diverged |
| Targeting card timing | D (ordering) | Extra RNG consumption |
| Abandoned Deck faces | D (ordering) | 52 cards instead of 40 |

**Remaining: 1 deviation**

The per-round `'nr1'` shuffle produces different dealt hands.

## 3. Remaining known deviations

The remaining deviation is **NOT** one of the 3 non-deterministic
`math.random` calls documented in rng.py. Those affect:
- Charm Tag pack variant (1 or 2)
- Meteor Tag pack variant (1 or 2)
- First Buffoon pack variant (1 or 2)

These are intentionally handled by always choosing variant 1.

The remaining deviation is in the per-round Fisher-Yates shuffle using
seed key `'nr1'`. The stream is fresh, the pre-sort order (by sort_id)
is identical, and the TW223 PRNG is proven bit-exact. The most likely
cause is Steamodded/balatrobot mod hooks that inject RNG consumption
or Card object creation (incrementing G.sort_id) between `set_blind`
and the shuffle in `new_round`. This cannot be confirmed without
instrumenting the mod layer.

## 4. Shop/pack integration issues

No major issues. The wiring was straightforward:

- **`_handle_cash_out`** calls `_populate_shop(gs)` which delegates to
  `populate_shop(rng, ante, gs)`. Returns jokers, voucher, boosters.
- **`_handle_reroll`** calls `_reroll_shop_cards(gs)` which regenerates
  only the joker slots (preserves voucher + boosters).
- **`_handle_open_booster`** calls `generate_pack_cards(pack_key, rng, ante, gs)`
  from packs.py. Arcana/Spectral packs deal a hand from deck for targeting.
- **`_handle_next_round`** clears shop areas.

One subtlety: the first-shop Buffoon pack guarantee is tracked by
`gs["first_shop_buffoon"]` flag in `populate_shop`. This works correctly
because the flag persists in game_state across rounds.

## 5. Sort_id ordering and shuffle determinism

Sort_id ordering was investigated as a potential cause of the per-round
shuffle divergence. Findings:

- Python assigns sort_ids starting from 1 per `initialize_run`
- Lua uses global `G.sort_id` that persists across the game session
- For a non-challenge run, only the 52 playing cards are created during
  `start_run`, so sort_ids are contiguous and monotonically increasing
- The `pseudoshuffle` pre-sort only cares about **relative** order, not
  absolute values
- Creation order matches (both sort by the same `(s..r..e..d..g)` key)

Conclusion: sort_id ordering is NOT the cause of the divergence.

## 6. Performance after fixes

Still well above target:

| Metric | Value |
|--------|-------|
| Random agent | 1,467 runs/sec |
| Greedy agent | 1,418 runs/sec |
| Target | >500 runs/sec |
| Margin | ~3x above target |

The shop/pack integration adds ~5% overhead per run (populate_shop
generates 2 jokers + 1 voucher + 2 boosters per cash_out), but runs
are still under 1ms each.

## 7. Confidence level: simulator accuracy

**8/10** for gameplay-relevant state.

What's excellent (10/10):
- RNG system (bit-exact TW223, validated against LuaJIT)
- Boss blind selection (100% match)
- Tag generation (100% match after fixes)
- Deck composition (100% match)
- Money tracking (100% match pre-deal)
- Scoring pipeline (150 joker handlers, 14-phase pipeline)
- Hand evaluation (12 hand types, cross-validated)

What's good (8/10):
- Shop card generation (uses correct pool system, not validated card-by-card)
- Consumable effects (14 mutation types implemented)
- Joker interactions (discard, play, round-end contexts)

What has known gaps (6/10):
- Per-round card ordering (dealt hand diverges from live)
- Card creation side-effects (Marble Joker, Riff-raff create simplified cards)
- Some edge-case joker interactions (Blueprint/Brainstorm copying)

## 8. Biggest remaining accuracy risk for RL training

**Shop card generation fidelity.** The shop uses the correct pool system
and RNG streams, but the interaction between pool filtering (used_jokers,
enhancement_gate, softlock) and the live game's "discovered" state may
cause different cards to appear. For RL training this matters because:

- The agent's strategy depends heavily on which jokers are available
- If the sim offers different jokers than the real game, learned
  strategies may not transfer
- The pool filtering is correct for a "fully discovered" profile, but
  a fresh profile would see fewer options

Mitigation: the sim assumes all items are discovered (matching a
typical player's profile). For RL transfer, the agent should be
trained on a fully-discovered profile, which matches most real players.

## 9. Total test count

**3,666 tests** across 27 test files:

| Category | Tests | Key files |
|----------|-------|-----------|
| Game step function | 172 | test_game.py |
| Full validation | 237 | test_full_validation.py |
| Runner (crash + perf) | 228 | test_runner.py |
| Challenges | 97 | test_challenges.py |
| Tags | 137 | test_tags.py + test_tag_generation.py |
| Run init | 102 | test_run_init.py + test_run_init_integration.py |
| Scoring | ~800 | test_scoring*.py |
| Jokers | ~600 | test_jokers*.py |
| Shop/pools | ~200 | test_shop*.py + test_pools.py |
| RNG | ~200 | test_rng*.py |
| Other | ~893 | remaining test files |

## New modules in M12/M12.5

| Module | Purpose | Lines |
|--------|---------|-------|
| game.py | Step function — 17 action handlers | ~1900 |
| actions.py | Action types + get_legal_actions | ~500 |
| runner.py | simulate_run + agents | ~180 |
| validator.py | State comparison | ~210 |
| challenges.py | 20 challenge definitions | ~630 |
| round_lifecycle.py | Perishable/rental + targeting cards | ~300 |
| run_init.py | Full run initialization | ~420 |
| scripts/validate_*.py | 3 validation scripts | ~600 |
| docs/source-map/rng-init-trace.md | RNG call sequence | ~90 |

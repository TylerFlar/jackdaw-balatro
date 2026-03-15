# M2 Feedback: RNG System

## 1. Does Python's float arithmetic match Lua's at 13-decimal truncation?

**Yes, exactly.** Both Python and LuaJIT use IEEE 754 double-precision floats.
The critical `truncate_13` operation — Lua's `tonumber(string.format("%.13f", x))`
— is replicated in Python as `float(f"{x:.13f}")`. Both use C's `printf`-family
rounding rules under the hood, producing identical bit patterns.

This was validated across 80 pseudoseed values (8 streams × 10 calls) for two
different seeds, plus 40 full-pipeline values. Every stored state value matches
LuaJIT to all 13 decimal places. The pseudohash function (which has no
truncation) matches to 15 decimal places (the limit of IEEE 754 representation).

## 2. Which secondary PRNG approach?

**Option A: replicate LuaJIT's exact PRNG (TW223).**

Option C (Python's `random.seed()` + `random.random()`) was tested first and
definitively ruled out — Python's Mersenne Twister produces completely different
output sequences from LuaJIT's TW223 for the same seed. Not even the integer
ranges overlap reliably.

Option B (use pseudoseed output directly as the random value) was considered
but rejected because the game explicitly calls `math.randomseed(sv)` then
`math.random()` as two separate operations. The `math.random()` output is what
downstream game logic actually consumes, and it differs from the seed value.

The TW223 implementation was ported from LuaJIT v2.1's `lj_prng.c`:
- `_tw223_step()`: the 4-generator combined Tausworthe step (unrolled)
- `_luajit_seed()`: `d = d*π + e` transformation, IEEE 754 bit reinterpretation
  via `struct.pack/unpack`, min-bit conditioning, 10 warm-up steps
- `_luajit_random()`: 52-bit mantissa masking → double in [1,2) → subtract 1

A key finding: LuaJIT's `math.randomseed(float)` uses the IEEE 754 bit pattern
of the double as seed material, NOT a truncated integer. Even 1 ULP (unit in
the last place, ~2.2e-16) difference in the seed float produces completely
different output. This means printed/rounded seed values cannot be used for
testing — only the exact float from `pseudoseed` works.

## 3. Oracle test results

**131 tests, all passing.** Zero failures.

| Test Category | Count | Status |
|---|---|---|
| pseudohash (Lua ground truth) | 10 parametrized + 5 property | All pass |
| pseudoseed (Lua ground truth, 2 seeds × 4 streams × 5 calls) | 8 parametrized + 7 property | All pass |
| predict_seed (Lua ground truth) | 3 parametrized + 2 property | All pass |
| PseudoRandom.random (TW223 pipeline) | 6 | All pass |
| PseudoRandom.element (LuaJIT ground truth, 5 test types) | 7 | All pass |
| PseudoRandom.shuffle (LuaJIT ground truth, 52-card + sort_id) | 10 | All pass |
| State management (save/load/rehash) | 3 | All pass |
| generate_starting_seed (LuaJIT ground truth, 5 entropy values) | 10 | All pass |
| Functional API wrappers | 10 | All pass |
| Fixture oracle (8 streams × 10 calls from JSON) | 13 | All pass |
| Live LuaJIT oracle (subprocess, 3 seeds) | 12 | All pass |
| Full pipeline oracle (float + int + shuffle + element) | 10 | All pass |
| Game sequence simulation (deck + boss + tags + shop + rerolls) | 7 | All pass |
| Performance benchmarks | 4 | All pass |

The LuaJIT oracle tests run against both:
- A pre-generated JSON fixture (`tests/fixtures/rng_oracle_TESTSEED.json`) — always available
- Live LuaJIT 2.1 subprocess (`scripts/lua_rng_oracle.lua`) — requires LuaJIT installed

## 4. Performance

| Operation | Rate | Notes |
|---|---|---|
| `pseudoseed` (Layer 2 only) | **2.6M calls/sec** | Float LCG + truncation |
| `pseudorandom` (full pipeline) | **120K calls/sec** | Layer 2 + TW223 seed + TW223 step |
| `pseudoshuffle(52)` | **19K shuffles/sec** | 51 TW223 draws per shuffle |
| `pseudorandom_element` | ~100K calls/sec | 1 TW223 draw + sort |

The bottleneck in `pseudorandom` is TW223 seeding: each call does 4 `struct`
operations + 10 warm-up steps. This is inherent to the LuaJIT design (re-seeds
on every call). Optimizations applied:
- Pre-compiled `struct.Struct` objects
- Unrolled TW223 step (4 generators inline, no loop)
- Unrolled seeding (4 state words inline, no loop)
- Pre-computed high masks and constants

At 120K full-pipeline calls/sec, a game action requiring ~200 random draws takes
~1.7ms. This is adequate for RL training — the scoring pipeline, not the RNG,
will be the bottleneck.

If performance becomes critical, options include:
- C extension for the TW223 hot path (~10x speedup expected)
- Caching TW223 state when the same seed is used consecutively
- Batch mode for drawing multiple values from one seeding

## 5. Edge Cases and Concerns

### Float precision at scale
After 1M pseudoseed advances on a single stream, the output still produces
unique values with no observable drift. The 13-decimal truncation acts as a
"float normalizer" that prevents accumulation of rounding errors.

### Empty string hash
`pseudohash("")` returns `1.0` (the loop doesn't execute, accumulator stays
at initial value). This is technically outside [0, 1) but matches Lua. No
game code calls pseudohash with an empty string.

### The 3 non-deterministic calls
The three `math.random()` calls in `tag.lua:211`, `tag.lua:226`, and
`common_events.lua:1947` that bypass pseudoseed are a known deviation. These
only affect pack variant selection (1 or 2) for Charm Tag, Meteor Tag, and
first Buffoon pack. The simulator should handle these with a deterministic
fallback (always pick variant 1, or use the pseudoseed system).

### `pseudoseed('seed')` escape hatch
The Lua source has a special case: `pseudoseed('seed')` returns raw
`math.random()` (non-deterministic). This is used only for seed generation,
never during gameplay. The Python implementation does not replicate this
(callers should use `generate_starting_seed` instead).

### Sort stability in element/shuffle
Python's `list.sort()` is stable (preserves insertion order for equal keys).
Lua's `table.sort` is NOT stable (uses introsort). For unique sort_ids and
unique string keys this doesn't matter. If duplicate sort_ids appear, the
results may differ between Python and Lua. No game code produces duplicate
sort_ids in the contexts where these functions are called.

### Downstream milestone dependencies
The RNG system is a dependency for:
- **Pool generation** (M3): uses `pseudoseed('rarity'..ante)`, `pseudorandom_element(pool, ...)`
- **Deck building** (M3): uses `pseudoshuffle(deck, pseudoseed('shuffle'))`
- **Scoring** (M4): Lucky Card uses `pseudorandom('lucky_mult')`, Glass Card uses `pseudorandom('glass')`
- **Shop** (M5): uses `pseudoseed('cdt'..ante)` for card type distribution
- **Consumables** (M5): spectral effects use various seeds

All of these are covered by the stream-independence and full-pipeline tests.

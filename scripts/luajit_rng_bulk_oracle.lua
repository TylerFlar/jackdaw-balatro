--- LuaJIT RNG bulk oracle: digests of about 10^6 draws from the game's own RNG functions.
---
--- Usage (from the repo root, needs the local balatro_source/ and LuaJIT 2.1):
---   luajit scripts/luajit_rng_bulk_oracle.lua > tests/fixtures/rng_luajit_bulk.json
---   luajit scripts/luajit_rng_bulk_oracle.lua --trace N     (one line per value of run N)
--- or `cargo xtask lua-oracles` for the fixture.
---
--- pseudohash, pseudoseed, pseudorandom, pseudorandom_element, pseudoshuffle and random_string
--- are loaded verbatim from balatro_source/functions/misc_functions.lua, as in
--- scripts/luajit_rng_oracle.lua. Each run starts G.GAME.pseudorandom from a run seed (some runs
--- then inject NaN or other values into it) and plays the schedule below, folding every result
--- into a 64-bit FNV-1a over the 8 little-endian bytes of its IEEE-754 bit pattern. Integer
--- results are Lua numbers (doubles) too and fold the same way. The fixture keeps the schedule's
--- parameters and one digest per run, not the draws; crates/jackdaw-core/tests/rng_luajit_bulk.rs
--- replays the schedule with jackdaw_core::rng and compares the digests run by run. When one
--- differs, `--trace N` prints run N's values for a diff against the test's trace.
---
--- Schedule of a run: fold hashed_seed; for each key (ki = 1..#KEYS, key = KEYS[ki]) fold
--- pseudohash(key .. seed), the stream's state before its first draw, and the stateless
--- pseudoseed(key, seed); then for round = 1..ROUNDS and each key, run
--- OPS[(3 * round + ki) % #OPS + 1]:
---   random           fold pseudorandom(key)
---   seed             fold pseudoseed(key)
---   range            fold pseudorandom(key, lo, hi), {lo, hi} = RANGES[(round + ki) % #RANGES + 1]
---   element_array    n = ELEMENT_SIZES[(round + 2 * ki) % #ELEMENT_SIZES + 1];
---                    pseudorandom_element({10, 20, ..., 10 * n}, pseudoseed(key)): fold the key
---   element_sort_id  t[i] = {sort_id = sort_id(i)} for i = 1..n (same n); the game sorts by
---                    sort_id: fold the chosen sort_id, then its key i
---   element_keyed    t[NAMES[i]] = {id = i}; no sort_id, so the game sorts by key: fold the id
---   shuffle_sort_id  m = SHUFFLE_SIZES[(round + 3 * ki) % #SHUFFLE_SIZES + 1];
---                    pseudoshuffle({{sort_id = sort_id(1)}, ..., {sort_id = sort_id(m)}},
---                    pseudoseed(key)), which sorts by sort_id first: fold the m sort_ids in order
---   shuffle_plain    the same over {{id = 1}, ..., {id = m}} (no sort_id, no sort): fold the ids
--- with sort_id(i) = i * 7919 % 10007 (distinct for i < 10007, and not monotonic in i).
---
--- "seed" and "hashed_seed" are never stream keys: pseudoseed('seed') returns math.random()
--- without touching a stream, and a stream named hashed_seed would overwrite the run's hash.
--- The game uses neither.

local ffi = require("ffi")
local bit = require("bit")
ffi.cdef("typedef union { double d; uint64_t u; } jd_bits;")

local function hex(x)
  local b = ffi.new("jd_bits")
  b.d = x
  return string.format("%08x%08x", tonumber(b.u / 4294967296ULL), tonumber(b.u % 4294967296ULL))
end

---------------------------------------------------------------------------
-- Load the game's functions by name from misc_functions.lua
---------------------------------------------------------------------------

local root = arg and arg[0] and arg[0]:match("^(.*)scripts[/\\]") or "./"
local path = root .. "balatro_source/functions/misc_functions.lua"
local f = assert(io.open(path, "r"), "cannot open " .. path)
local src = f:read("*a")
f:close()

local wanted = {
  pseudoshuffle = true, pseudorandom_element = true, random_string = true,
  pseudohash = true, pseudoseed = true, pseudorandom = true,
}
local chunks, current = {}, nil
for line in (src .. "\n"):gmatch("([^\n]*)\n") do
  local name = line:match("^function ([%w_]+)%(")
  if name and wanted[name] then current = { name = name, lines = {} } end
  if current then
    current.lines[#current.lines + 1] = line
    if line:match("^end") then
      chunks[#chunks + 1] = table.concat(current.lines, "\n")
      current = nil
    end
  end
end
assert(#chunks == 6, "expected 6 RNG functions, found " .. #chunks)
for _, code in ipairs(chunks) do assert(loadstring(code))() end

---------------------------------------------------------------------------
-- Parameters
---------------------------------------------------------------------------

local NAN = 0 / 0          -- the default NaN, 0xfff8000000000000 (sign bit set)
local POS_NAN = -(0 / 0)   -- 0x7ff8000000000000
local INF = 1 / 0

local ROUNDS = 120

local OPS = { "random", "range", "seed", "element_array", "random", "element_sort_id", "range",
  "shuffle_sort_id", "element_keyed", "shuffle_plain" }

local ASCII = {}
for b = 0, 127 do ASCII[#ASCII + 1] = string.char(b) end

local KEYS = {
  -- Keys the game builds (create_card, shops, packs, bosses, jokers, consumables, decks).
  "rarity1sho", "rarity3", "cdt1", "cdt8", "shuffle", "nr2", "boss", "Joker1sho", "Joker4",
  "Tarotar11", "soul_Spectral5", "edisho1", "stdset2", "front1", "Tag1", "Voucher4", "erratic",
  "lucky_mult", "lucky_money", "wheel_of_fortune", "misprint", "bloodstone", "idol1", "anc3",
  "cashout1", "hook", "cert_fr", "flipped_card", "random_destroy", "omen_globe",
  -- Odd ones: empty, one byte, long, punctuation, control bytes, non-ASCII (UTF-8) bytes.
  "", "a", string.rep("long", 40), "x y-z.!", "tab\tkey", "\xc3\xa9t\xc3\xa9", "\xe2\x82\xac",
  "\xe6\x97\xa5\xe6\x9c\xac", "0", "1e5",
  table.concat(ASCII),     -- every ASCII byte, NUL to DEL
  -- U+0080, U+07FF, U+0800, U+FFFF, U+10000, U+10FFFF: UTF-8's lead and continuation byte ranges
  "\xc2\x80\xdf\xbf\xe0\xa0\x80\xef\xbf\xbf\xf0\x90\x80\x80\xf4\x8f\xbf\xbf",
}

local RANGES = { { 1, 2 }, { 1, 4 }, { 1, 6 }, { 0, 23 }, { 1, 52 }, { -10, 10 }, { 1, 100 },
  { 0, 1 }, { 5, 5 }, { -52, -1 }, { 1, 1000000 }, { 0, 2 ^ 30 }, { -1000000000, 1000000000 },
  { 1, 2 ^ 53 } }

local ELEMENT_SIZES = { 1, 2, 3, 4, 5, 7, 8, 13, 22, 32, 52, 60, 150 }
local SHUFFLE_SIZES = { 0, 1, 2, 3, 5, 8, 13, 21, 44, 52, 60 }

-- Keys of the element_keyed table; Lua orders them by bytes.
local NAMES = { "j_joker", "j_greedy_joker", "c_fool", "c_magician", "v_overstock_norm",
  "p_arcana_normal_1", "m_bonus", "e_foil", "b_red", "Red", "red", "_", "10", "9", "z",
  "\xc3\xa9", "\xe2\x82\xac", "\xe6\x97\xa5\xe6\x9c\xac", " ", "a b" }

local function sort_id(i) return i * 7919 % 10007 end

-- Run seeds: fixed edge cases, then seeds made by the game's random_string.
local runs = {}
for _, s in ipairs({
  "",                      -- pseudohash("") = 1: the loop never runs
  "A", "1", "Z", "AB",
  "TESTSEED", "TUTORIAL", "A3K9NZ2B", "GOLD_D0",
  "7LB2WVPK",              -- its 'erratic' stream hashes to NaN
  "abcdefgh", "lowercase", "0123456789", "00000000", "99999999", "ZZZZZZZZ", "HELLO WORLD",
  "S\xc3\x89\xc3\x89D\xe2\x82\xac",
  string.rep("LONGSEED", 8),
  string.rep("0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ", 8),
}) do
  runs[#runs + 1] = { seed = s }
end
local LENGTHS = { 8, 8, 8, 8, 8, 8, 1, 2, 5, 16, 32 }
for i = 1, 180 do
  -- Mixed like generate_starting_seed's cursor position and time.
  local x = i * 0.33411983 + (i % 17) * 0.874146 + 0.412311010 * i / 3
  local len = LENGTHS[i % #LENGTHS + 1]
  runs[#runs + 1] = { seed = random_string(len, x), made_by = { len, x } }
end

-- Values the game never stores but the arithmetic must still match: NaN of either sign (the
-- run's hash, or a stream's state), infinities, zeros, huge, negative and non-integral values.
local injections = {
  { seed = "TESTSEED", hashed_seed = NAN },
  { seed = "TESTSEED", hashed_seed = POS_NAN },
  { seed = "TESTSEED", hashed_seed = INF },
  { seed = "TESTSEED", hashed_seed = -INF },
  { seed = "TESTSEED", hashed_seed = 0 },
  { seed = "TESTSEED", hashed_seed = -1 / INF },     -- -0
  { seed = "TESTSEED", hashed_seed = 1e300 },
  { seed = "TESTSEED", hashed_seed = -0.75 },
  { seed = "A3K9NZ2B", streams = { shuffle = NAN } },
  { seed = "A3K9NZ2B", streams = { boss = POS_NAN } },
  { seed = "A3K9NZ2B", streams = { cdt1 = INF, rarity1sho = -INF } },
  { seed = "A3K9NZ2B", streams = { Joker4 = 1e300, erratic = -0.5, idol1 = 2 ^ 52 - 0.5 } },
  { seed = "A3K9NZ2B", streams = { misprint = 0, hook = -1 / INF, anc3 = 2 ^ -1074 } },
  { seed = "7LB2WVPK", hashed_seed = POS_NAN, streams = { erratic = NAN } },
  { seed = "", streams = { [""] = NAN } },
}
for _, inj in ipairs(injections) do runs[#runs + 1] = inj end

---------------------------------------------------------------------------
-- The schedule
---------------------------------------------------------------------------

local FNV_OFFSET, FNV_PRIME = 0xcbf29ce484222325ULL, 0x100000001b3ULL
local cell = ffi.new("jd_bits")
local digest, folded, trace

local function fold(x, round, ki, op)
  cell.d = x
  local u = cell.u
  local h = digest
  for shift = 0, 56, 8 do
    h = bit.bxor(h, bit.band(bit.rshift(u, shift), 0xFF)) * FNV_PRIME
  end
  digest = h
  folded = folded + 1
  if trace then io.write(string.format("%d %d %s %s\n", round, ki, op, hex(x))) end
end

local function play(run)
  G = { GAME = { pseudorandom = { seed = run.seed, hashed_seed = pseudohash(run.seed) } } }
  local pr = G.GAME.pseudorandom
  if run.hashed_seed then pr.hashed_seed = run.hashed_seed end
  for key, value in pairs(run.streams or {}) do pr[key] = value end
  digest, folded = FNV_OFFSET, 0
  fold(pr.hashed_seed, 0, 0, "hashed_seed")
  for ki, key in ipairs(KEYS) do
    -- The raw hash keeps the sign of a NaN that %.13f and math.abs erase in the stream state.
    fold(pseudohash(key .. run.seed), 0, ki, "hash")
    fold(pseudoseed(key, run.seed), 0, ki, "predict")
  end
  for round = 1, ROUNDS do
    for ki, key in ipairs(KEYS) do
      local op = OPS[(3 * round + ki) % #OPS + 1]
      if op == "random" then
        fold(pseudorandom(key), round, ki, op)
      elseif op == "seed" then
        fold(pseudoseed(key), round, ki, op)
      elseif op == "range" then
        local r = RANGES[(round + ki) % #RANGES + 1]
        fold(pseudorandom(key, r[1], r[2]), round, ki, op)
      elseif op == "element_array" or op == "element_sort_id" then
        local n = ELEMENT_SIZES[(round + 2 * ki) % #ELEMENT_SIZES + 1]
        local t = {}
        for i = 1, n do
          if op == "element_array" then t[i] = 10 * i else t[i] = { sort_id = sort_id(i) } end
        end
        local v, k = pseudorandom_element(t, pseudoseed(key))
        if op == "element_sort_id" then fold(v.sort_id, round, ki, op) end
        fold(k, round, ki, op)
      elseif op == "element_keyed" then
        local t = {}
        for i, name in ipairs(NAMES) do t[name] = { id = i } end
        local v = pseudorandom_element(t, pseudoseed(key))
        fold(v.id, round, ki, op)
      else
        local m = SHUFFLE_SIZES[(round + 3 * ki) % #SHUFFLE_SIZES + 1]
        local list = {}
        for i = 1, m do
          if op == "shuffle_sort_id" then
            list[i] = { sort_id = sort_id(i) }
          else
            list[i] = { id = i }
          end
        end
        pseudoshuffle(list, pseudoseed(key))
        for i = 1, m do fold(list[i].sort_id or list[i].id, round, ki, op) end
      end
    end
  end
  return digest, folded
end

local function hex64(u)
  return string.format("%08x%08x", tonumber(u / 4294967296ULL), tonumber(u % 4294967296ULL))
end

if arg[1] == "--trace" then
  local run = runs[assert(tonumber(arg[2]), "--trace N: a run index (0-based)") + 1]
  assert(run, "no such run")
  trace = true
  play(run)
  return
end

---------------------------------------------------------------------------
-- JSON output
---------------------------------------------------------------------------

local function q(s)
  return '"' .. s:gsub('[%c"\\]', function(c)
    if c == '"' then return '\\"' elseif c == "\\" then return "\\\\" end
    return string.format("\\u%04x", c:byte())
  end) .. '"'
end
local function list(t, f)
  local parts = {}
  for i, v in ipairs(t) do parts[i] = f and f(v) or v end
  return "[" .. table.concat(parts, ",") .. "]"
end
local function int(x) return string.format("%d", x) end

local lines = {}
local total_values = 0
for i, run in ipairs(runs) do
  local h, n = play(run)
  total_values = total_values + n
  local parts = { q("seed") .. ":" .. q(run.seed) }
  if run.made_by then
    parts[#parts + 1] = q("random_string") .. ":" .. list({ int(run.made_by[1]), q(hex(run.made_by[2])) })
  end
  if run.hashed_seed then parts[#parts + 1] = q("hashed_seed") .. ":" .. q(hex(run.hashed_seed)) end
  if run.streams then
    local keys = {}
    for k in pairs(run.streams) do keys[#keys + 1] = k end
    table.sort(keys)
    local s = {}
    for j, k in ipairs(keys) do s[j] = q(k) .. ":" .. q(hex(run.streams[k])) end
    parts[#parts + 1] = q("streams") .. ":{" .. table.concat(s, ",") .. "}"
  end
  parts[#parts + 1] = q("values") .. ":" .. int(n)
  parts[#parts + 1] = q("digest") .. ":" .. q(hex64(h))
  lines[i] = "{" .. table.concat(parts, ",") .. "}"
end

local out = {
  "{",
  q("luajit") .. ":" .. q(jit and jit.version or _VERSION) .. ",",
  q("source") .. ":" .. q("balatro_source/functions/misc_functions.lua (Balatro 1.0.1o)") .. ",",
  q("generator") .. ":" .. q("scripts/luajit_rng_bulk_oracle.lua") .. ",",
  q("digest") .. ":" .. q("FNV-1a 64 over the little-endian bytes of each value's IEEE-754 bits") .. ",",
  q("draws_per_run") .. ":" .. int(ROUNDS * #KEYS) .. ",",
  q("total_draws") .. ":" .. int(ROUNDS * #KEYS * #runs) .. ",",
  q("total_values") .. ":" .. int(total_values) .. ",",
  q("rounds") .. ":" .. int(ROUNDS) .. ",",
  q("ops") .. ":" .. list(OPS, q) .. ",",
  q("keys") .. ":" .. list(KEYS, q) .. ",",
  q("ranges") .. ":" .. list(RANGES, function(r) return list(r, int) end) .. ",",
  q("element_sizes") .. ":" .. list(ELEMENT_SIZES, int) .. ",",
  q("shuffle_sizes") .. ":" .. list(SHUFFLE_SIZES, int) .. ",",
  q("names") .. ":" .. list(NAMES, q) .. ",",
  q("runs") .. ":[",
  table.concat(lines, ",\n"),
  "]",
  "}",
}
io.write(table.concat(out, "\n"), "\n")

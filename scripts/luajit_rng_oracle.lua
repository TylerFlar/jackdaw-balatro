--- LuaJIT RNG oracle: exact bit patterns from the game's own RNG functions.
---
--- Usage (from the repo root, needs the local balatro_source/ and LuaJIT 2.1):
---   luajit scripts/luajit_rng_oracle.lua > tests/fixtures/rng_luajit.json
--- or `cargo xtask lua-oracles`.
---
--- The functions under test (pseudohash, pseudoseed, pseudorandom, pseudorandom_element,
--- pseudoshuffle, random_string) are loaded verbatim from
--- balatro_source/functions/misc_functions.lua, so the fixture records what the game computes on
--- LuaJIT (x86-64), including NaN bit patterns. Every double is written as 16 hex digits.

local ffi = require("ffi")
ffi.cdef("typedef union { double d; uint64_t u; } jd_bits;")

local function hex(x)
  local b = ffi.new("jd_bits")
  b.d = x
  local hi = tonumber(b.u / 4294967296ULL)
  local lo = tonumber(b.u % 4294967296ULL)
  return string.format("%08x%08x", hi, lo)
end

local function from_hex(h)
  local b = ffi.new("jd_bits")
  b.u = tonumber(h:sub(1, 8), 16) * 4294967296ULL + tonumber(h:sub(9, 16), 16)
  return b.d
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

G = { GAME = { pseudorandom = {} } }
local function start(seed)
  G.GAME.pseudorandom = { seed = seed, hashed_seed = pseudohash(seed) }
end

---------------------------------------------------------------------------
-- JSON output
---------------------------------------------------------------------------

local out = {}
local function emit(s) out[#out + 1] = s end
local function q(s) return '"' .. s:gsub('\\', '\\\\'):gsub('"', '\\"') .. '"' end
local function list(t) return "[" .. table.concat(t, ",") .. "]" end

---------------------------------------------------------------------------
-- Cases
---------------------------------------------------------------------------

-- A deterministic stream of doubles for inputs (independent of the code under test).
local lcg = 12345
local function next_unit()
  lcg = (lcg * 1103515245 + 12345) % 2147483648
  local a = lcg
  lcg = (lcg * 1103515245 + 12345) % 2147483648
  return (a * 2147483648 + lcg) / 2^62
end

local ALPHABET = "123456789ABCDEFGHIJKLMNPQRSTUVWXYZ"
local function seed_string(n)
  local t = {}
  for i = 1, 8 do
    local k = math.floor(n % #ALPHABET) + 1
    n = math.floor(n / #ALPHABET) + i * 7919
    t[i] = ALPHABET:sub(k, k)
  end
  return table.concat(t)
end

emit("{")
emit(q("luajit") .. ":" .. q(jit and jit.version or _VERSION) .. ",")
emit(q("source") .. ":" .. q("balatro_source/functions/misc_functions.lua (Balatro 1.0.1o)") .. ",")

-- pseudohash
do
  local strs = { "", "A", "TESTSEED", "TUTORIAL", "A3K9NZ2B", "7LB2WVPK", "erratic7LB2WVPK",
    "shuffleTESTSEED", "Joker1sho1TESTSEED", "\xe2\x82\xac", "boss" }
  for i = 1, 60 do strs[#strs + 1] = seed_string(i * 104729) end
  local items = {}
  for _, s in ipairs(strs) do
    local bytes = {}
    for i = 1, #s do bytes[i] = tostring(s:byte(i)) end
    items[#items + 1] = "{" .. q("bytes") .. ":" .. list(bytes) .. "," .. q("hash") .. ":" .. q(hex(pseudohash(s))) .. "}"
  end
  emit(q("pseudohash") .. ":" .. list(items) .. ",")
end

-- tonumber(string.format("%.13f", x)), including exact ties
do
  local xs = { 0, 1 / 16384, 3 / 16384, 5 / 16384, 1 + 1 / 16384, 0.99999999999996, 0.5, 1e-14, 5e-14,
    2.5, -(1 + 1 / 16384), 1e300, 0 / 0, -(0 / 0), 1 / 0 }
  for i = 1, 3000 do xs[#xs + 1] = next_unit() end
  for i = 1, 200 do xs[#xs + 1] = math.floor(next_unit() * 2^20) / 2^14 / 64 end -- dyadic, many ties
  local items = {}
  for _, x in ipairs(xs) do
    items[#items + 1] = list({ q(hex(x)), q(hex(tonumber(string.format("%.13f", x)))) })
  end
  emit(q("truncate13") .. ":" .. list(items) .. ",")
end

-- tostring(x)
do
  local xs = { 1, -3, 0.5, 1e15, 123456789012345, 99999999999999, 1 / 3, 1e-5, 0.1, 2^53, 1e14,
    12345678901234.5, 0 / 0, 1 / 0, -1 / 0, 1e-300, 123.456, 1e100 }
  for i = 1, 300 do xs[#xs + 1] = (next_unit() - 0.5) * 10 ^ math.floor(next_unit() * 40 - 20) end
  local items = {}
  for _, x in ipairs(xs) do items[#items + 1] = list({ q(hex(x)), q(tostring(x)) }) end
  emit(q("tostring") .. ":" .. list(items) .. ",")
end

-- math.randomseed(d) then draws: 4 floats, then math.random(1, 52), math.random(7), math.random(0, 2^30)
do
  local ds = { 0, 1, -1, 0.5, 1e-300, 1e300, 1 / 0, -1 / 0, 0 / 0, -(0 / 0), 2^-1074 }
  for i = 1, 400 do ds[#ds + 1] = next_unit() end
  local items = {}
  for _, d in ipairs(ds) do
    math.randomseed(d)
    local draws = { q(hex(math.random())), q(hex(math.random())), q(hex(math.random())), q(hex(math.random())) }
    local ints = { tostring(math.random(1, 52)), tostring(math.random(7)), tostring(math.random(0, 2^30)) }
    items[#items + 1] = "{" .. q("d") .. ":" .. q(hex(d)) .. "," .. q("floats") .. ":" .. list(draws) .. ","
      .. q("ints") .. ":" .. list(ints) .. "}"
  end
  emit(q("seeded") .. ":" .. list(items) .. ",")
end

-- Stream sequences: interleaved pseudoseed / pseudorandom calls per run seed
do
  local seeds = { "TESTSEED", "TUTORIAL", "A3K9NZ2B", "7LB2WVPK", "GOLD_D0" }
  for i = 1, 20 do seeds[#seeds + 1] = seed_string(i * 7777) end
  local keys = { "shuffle", "boss", "Tag1", "Joker1sho1", "rarity1sho", "cdt1", "edisho1", "erratic",
    "lucky_mult", "nr1", "cashout1", "Voucher1", "idol1", "anc1" }
  -- Runs with NaN injected where no natural seed reaches it: the run's hashed_seed (either
  -- sign), or a stream's stored state. `inject` records what was set before the first call.
  local injections = {
    { seed = "TESTSEED", hashed_seed = 0 / 0 },
    { seed = "TESTSEED", hashed_seed = -(0 / 0) },
    { seed = "A3K9NZ2B", stream = "shuffle", value = 0 / 0 },
    { seed = "A3K9NZ2B", stream = "boss", value = -(0 / 0) },
  }
  local runs = {}
  local plan = {}
  for _, s in ipairs(seeds) do plan[#plan + 1] = { seed = s } end
  for _, inj in ipairs(injections) do plan[#plan + 1] = inj end
  for _, p in ipairs(plan) do
    local s = p.seed
    start(s)
    local inject = {}
    if p.hashed_seed then
      G.GAME.pseudorandom.hashed_seed = p.hashed_seed
      inject[#inject + 1] = list({ q("hashed_seed"), q(hex(p.hashed_seed)) })
    end
    if p.stream then
      G.GAME.pseudorandom[p.stream] = p.value
      inject[#inject + 1] = list({ q(p.stream), q(hex(p.value)) })
    end
    local ops = {}
    for i = 1, 60 do
      local key = keys[math.floor(next_unit() * #keys) + 1]
      local kind = math.floor(next_unit() * 3)
      if kind == 0 then
        ops[#ops + 1] = list({ q("seed"), q(key), q(hex(pseudoseed(key))) })
      elseif kind == 1 then
        ops[#ops + 1] = list({ q("random"), q(key), q(hex(pseudorandom(key))) })
      else
        ops[#ops + 1] = list({ q("range"), q(key), "1", "52", tostring(pseudorandom(key, 1, 52)) })
      end
    end
    local hs = G.GAME.pseudorandom.hashed_seed
    runs[#runs + 1] = "{" .. q("seed") .. ":" .. q(s) .. "," .. q("hashed_seed") .. ":" .. q(hex(hs)) .. ","
      .. q("inject") .. ":" .. list(inject) .. "," .. q("ops") .. ":" .. list(ops) .. "}"
  end
  emit(q("streams") .. ":" .. list(runs) .. ",")
end

-- pseudoseed(key, predict_seed)
do
  local items = {}
  local keys = { "Joker4", "erratic", "boss", "shuffle" }
  for i = 1, 40 do
    local key, p = keys[(i % #keys) + 1], seed_string(i * 31337)
    items[#items + 1] = list({ q(key), q(p), q(hex(pseudoseed(key, p))) })
  end
  emit(q("predict") .. ":" .. list(items) .. ",")
end

-- pseudorandom_element over an array 1..n (keys sort numerically)
do
  local items = {}
  for i = 1, 200 do
    local n = math.floor(next_unit() * 60) + 1
    local t = {}
    for k = 1, n do t[k] = k * 10 end
    local seed = next_unit()
    local _, key = pseudorandom_element(t, seed)
    items[#items + 1] = list({ q(hex(seed)), tostring(n), tostring(key) })
  end
  emit(q("element") .. ":" .. list(items) .. ",")
end

-- pseudoshuffle over cards with sort_ids
do
  local items = {}
  for i = 1, 100 do
    local n = math.floor(next_unit() * 60) + 1
    local list_t = {}
    for k = 1, n do list_t[k] = { sort_id = k } end
    local seed = next_unit()
    pseudoshuffle(list_t, seed)
    local perm = {}
    for k = 1, n do perm[k] = tostring(list_t[k].sort_id) end
    items[#items + 1] = list({ q(hex(seed)), list(perm) })
  end
  emit(q("shuffle") .. ":" .. list(items) .. ",")
end

-- random_string(8, seed)
do
  local items = {}
  for i = 1, 100 do
    local seed = next_unit() * 1000
    items[#items + 1] = list({ q(hex(seed)), q(random_string(8, seed)) })
  end
  emit(q("random_string") .. ":" .. list(items))
end

emit("}")
io.write(table.concat(out, "\n"), "\n")

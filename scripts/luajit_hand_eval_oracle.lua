--- LuaJIT hand-evaluation oracle: the game's own poker-hand code on generated hands.
---
--- Usage (from the repo root, needs the local balatro_source/ and LuaJIT 2.1):
---   luajit scripts/luajit_hand_eval_oracle.lua > tests/fixtures/hand_eval_luajit.json
--- or `cargo xtask lua-oracles`.
---
--- evaluate_poker_hand, get_flush, get_straight, get_X_same and get_highest are loaded from
--- balatro_source/functions/misc_functions.lua, G.FUNCS.get_poker_hand_info from
--- state_events.lua, and Card:get_nominal / get_id / is_suit from card.lua, verbatim. Cards are
--- minimal tables with those methods; jokers are stubbed through find_joker. Every case records
--- its inputs and, for each poker hand, the groups as 1-based positions in the hand.

local root = arg and arg[0] and arg[0]:match("^(.*)scripts[/\\]") or "./"
local function read(rel)
  local f = assert(io.open(root .. rel, "r"), "cannot open " .. rel)
  local s = f:read("*a")
  f:close()
  return s
end

local function load_functions(src, names, pattern_prefix)
  local found = 0
  for name in pairs(names) do
    local code = src:match("\n(" .. pattern_prefix .. name:gsub("%.", "%%.") .. "%(.-\nend)\n")
    assert(code, "function " .. name .. " not found")
    assert(loadstring(code))()
    found = found + 1
  end
  return found
end

G = { FUNCS = {} }
localize = function(text) return text end

-- find_joker(name) is stubbed by the active flags of the current case.
local active = {}
function find_joker(name, non_debuff)
  if active[name] then return { true } end
  return {}
end

local misc = read("balatro_source/functions/misc_functions.lua")
load_functions(misc, { evaluate_poker_hand = true, get_flush = true, get_straight = true,
  get_X_same = true, get_highest = true }, "function ")
local se = read("balatro_source/functions/state_events.lua")
do
  local code = se:match("\n(G%.FUNCS%.get_poker_hand_info = function%(_cards%).-\nend)\n")
  assert(code, "get_poker_hand_info not found")
  assert(loadstring(code))()
end

-- Card methods from card.lua, attached to a minimal card class.
Card = {}
Card.__index = Card
local card_src = read("balatro_source/card.lua")
for _, name in ipairs({ "get_nominal", "get_id", "is_suit" }) do
  local code = card_src:match("\n(function Card:" .. name .. "%(.-\nend)\n")
  assert(code, "Card:" .. name .. " not found")
  assert(loadstring(code))()
end

---------------------------------------------------------------------------
-- Case generation
---------------------------------------------------------------------------

local SUITS = { "Hearts", "Diamonds", "Clubs", "Spades" }
local VALUES = { "2", "3", "4", "5", "6", "7", "8", "9", "10", "Jack", "Queen", "King", "Ace" }
local NOMINAL = { 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11 }
local FACE = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.2, 0.3, 0.4 }
local SUIT_NOMINAL = { Hearts = 0.03, Diamonds = 0.01, Clubs = 0.02, Spades = 0.04 }
local SUIT_ORIGINAL = { Hearts = 0.003, Diamonds = 0.001, Clubs = 0.002, Spades = 0.004 }

local lcg = 987654321
local function rnd(n) -- integer in [1, n]
  lcg = (lcg * 1103515245 + 12345) % 2147483648
  return math.floor(lcg / 2147483648 * n) + 1
end

local function make_card(spec)
  local c = setmetatable({}, Card)
  c.base = {
    suit = SUITS[spec.suit], value = VALUES[spec.rank], id = spec.rank + 1,
    nominal = NOMINAL[spec.rank], face_nominal = FACE[spec.rank],
    suit_nominal = SUIT_NOMINAL[SUITS[spec.suit]],
    suit_nominal_original = SUIT_ORIGINAL[SUITS[spec.original_suit]],
  }
  c.ability = { effect = "", name = "Default Base" }
  if spec.enh == "stone" then c.ability.effect = "Stone Card"; c.ability.name = "Stone Card" end
  if spec.enh == "wild" then c.ability.effect = "Wild Card"; c.ability.name = "Wild Card" end
  c.debuff = spec.debuff
  c.vampired = nil
  c.unique_val = spec.unique_val
  return c
end

local cases = {}
local function add_case(specs, flags)
  cases[#cases + 1] = { specs = specs, flags = flags }
end

local function random_specs(n)
  local specs = {}
  for i = 1, n do
    local rank = rnd(13)
    local suit = rnd(4)
    local roll = rnd(20)
    specs[i] = {
      rank = rank, suit = suit,
      original_suit = (rnd(6) == 1) and rnd(4) or suit,
      enh = roll == 1 and "stone" or (roll <= 3 and "wild" or "none"),
      debuff = rnd(10) == 1,
      unique_val = 1 - rnd(5000) / 1603301,
    }
  end
  return specs
end

local flag_sets = {
  {}, { ["Four Fingers"] = true }, { ["Shortcut"] = true }, { ["Smeared Joker"] = true },
  { ["Four Fingers"] = true, ["Shortcut"] = true },
  { ["Four Fingers"] = true, ["Smeared Joker"] = true, ["Shortcut"] = true },
}

-- Structured hands that hit every poker hand, then random ones.
for _, fs in ipairs(flag_sets) do
  for i = 1, 200 do
    local n = rnd(5)
    local specs = random_specs(n)
    -- bias towards made hands: copy ranks/suits around
    if n >= 2 and rnd(2) == 1 then specs[2].rank = specs[1].rank end
    if n >= 4 and rnd(3) == 1 then specs[4].rank = specs[3].rank end
    if n >= 3 and rnd(3) == 1 then specs[3].rank = specs[1].rank end
    if rnd(3) == 1 then for k = 2, n do specs[k].suit = specs[1].suit end end
    if n == 5 and rnd(3) == 1 then
      local start = rnd(10) -- 1 = Ace-low
      local seq = {}
      for k = 0, 4 do
        local r = start + k - 1
        if r == 0 then r = 13 end
        if rnd(5) == 1 and fs["Shortcut"] then r = math.min(r + 1, 13) end
        seq[k + 1] = r
      end
      for k = 1, 5 do specs[k].rank = seq[k] end
    end
    add_case(specs, fs)
  end
  for i = 1, 100 do add_case(random_specs(rnd(8)), fs) end
end
-- Rare hands: five of a kind, flush five, flush house, full house, four of a kind (with the
-- odd wild, stone or debuffed card mixed in by random_specs).
for _, fs in ipairs(flag_sets) do
  for i = 1, 60 do
    local specs = random_specs(5)
    local shape = rnd(5)
    local r1, r2, suit = specs[1].rank, specs[2].rank, specs[1].suit
    if r2 == r1 then r2 = (r1 % 13) + 1 end
    for k = 1, 5 do
      if shape == 1 then specs[k].rank = r1                              -- five of a kind
      elseif shape == 2 then specs[k].rank = r1; specs[k].suit = suit    -- flush five
      elseif shape == 3 then specs[k].rank = k <= 3 and r1 or r2; specs[k].suit = suit -- flush house
      elseif shape == 4 then specs[k].rank = k <= 3 and r1 or r2         -- full house
      else specs[k].rank = k <= 4 and r1 or r2 end                       -- four of a kind
    end
    -- shuffle positions so groups are not always leading
    for k = 5, 2, -1 do
      local j = rnd(k)
      specs[k], specs[j] = specs[j], specs[k]
    end
    add_case(specs, fs)
  end
end
-- Exact duplicates (same rank and suit) with different unique_vals: High Card tie-breaks.
for i = 1, 100 do
  local specs = random_specs(rnd(3))
  local dup = {}
  for k, v in pairs(specs[1]) do dup[k] = v end
  dup.unique_val = 1 - rnd(5000) / 1603301
  specs[#specs + 1] = dup
  add_case(specs, flag_sets[rnd(#flag_sets)])
end

---------------------------------------------------------------------------
-- Evaluate and emit JSON
---------------------------------------------------------------------------

local ORDER = { "Flush Five", "Flush House", "Five of a Kind", "Straight Flush", "Four of a Kind",
  "Full House", "Flush", "Straight", "Three of a Kind", "Two Pair", "Pair", "High Card" }

local function q(s) return '"' .. s .. '"' end
local function list(t) return "[" .. table.concat(t, ",") .. "]" end

local out = {}
for _, case in ipairs(cases) do
  active = case.flags
  local hand = {}
  for i, spec in ipairs(case.specs) do hand[i] = make_card(spec) end
  local pos = {}
  for i, c in ipairs(hand) do pos[c] = i end
  local function idx(group)
    local t = {}
    for i, c in ipairs(group) do t[i] = tostring(pos[c]) end
    return list(t)
  end
  local text, _, poker_hands, scoring_hand = G.FUNCS.get_poker_hand_info(hand)
  local results = {}
  for _, name in ipairs(ORDER) do
    local groups = {}
    for gi, g in ipairs(poker_hands[name]) do groups[gi] = idx(g) end
    results[#results + 1] = list(groups)
  end
  local specs = {}
  for i, s in ipairs(case.specs) do
    specs[i] = string.format('{"rank":%d,"suit":%q,"original_suit":%q,"enh":%q,"debuff":%s,"unique_val":%.17g,"nominal":%.17g}',
      s.rank + 1, SUITS[s.suit], SUITS[s.original_suit], s.enh, tostring(s.debuff), s.unique_val, hand[i]:get_nominal())
  end
  local flags = {}
  for k in pairs(case.flags) do flags[#flags + 1] = q(k) end
  table.sort(flags)
  out[#out + 1] = "{" .. q("cards") .. ":" .. list(specs) .. "," .. q("flags") .. ":" .. list(flags) .. ","
    .. q("text") .. ":" .. q(text) .. "," .. q("scoring") .. ":" .. idx(scoring_hand) .. ","
    .. q("hands") .. ":" .. list(results) .. "}"
end

io.write('{"luajit":', q(jit and jit.version or _VERSION), ',"hand_order":[')
for i, n in ipairs(ORDER) do io.write(i > 1 and "," or "", q(n)) end
io.write('],"cases":[\n', table.concat(out, ",\n"), "\n]}\n")

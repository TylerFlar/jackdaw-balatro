--- Extract game tables that live outside P_CENTERS from the Balatro 1.0.1o source.
---
--- Usage (from the repo root, needs the local balatro_source/ and LuaJIT 2.1):
---   luajit scripts/extract_game_tables.lua [OUT_DIR]      (default jackdaw/engine/data)
---
--- Writes, from the game's own code:
---   hands.json          G.GAME.hands as Game:init_game_object builds it (game.lua), incl. the
---                       example cards BalatroBot's gamestate reports
---   challenges.json     G.CHALLENGES (challenges.lua), in game order
---   blind_amounts.json  get_blind_amount(ante) for scaling 1-3 and antes -1..80, evaluated by
---                       LuaJIT (the endless formula uses C pow; the table pins its results)
---
--- Numbers are written with %.17g (integral values as integers) so they round-trip; the blind
--- amounts also carry each double's 16-hex-digit bit pattern (they reach inf and NaN).

local root = arg and arg[0] and arg[0]:match("^(.*)scripts[/\\]") or "./"
local out_dir = arg[1] or (root .. "jackdaw/engine/data")

local function read(rel)
  local f = assert(io.open(root .. rel, "r"), "cannot open " .. rel)
  local s = f:read("*a")
  f:close()
  return s
end

---------------------------------------------------------------------------
-- JSON encoder: sorted object keys, sequences as arrays, %.17g numbers
---------------------------------------------------------------------------

local function is_array(t)
  local n = 0
  for _ in pairs(t) do n = n + 1 end
  for i = 1, n do if t[i] == nil then return false end end
  return true
end

local function encode(v, indent)
  indent = indent or ""
  local t = type(v)
  if t == "nil" then return "null" end
  if t == "boolean" then return tostring(v) end
  if t == "number" then
    assert(v == v and v ~= math.huge and v ~= -math.huge, "non-finite number in table")
    if v == math.floor(v) and math.abs(v) < 2^53 then return string.format("%d", v) end
    return string.format("%.17g", v)
  end
  if t == "string" then
    return '"' .. v:gsub('[%c"\\]', function(c)
      if c == '"' then return '\\"' elseif c == "\\" then return "\\\\" end
      return string.format("\\u%04x", c:byte())
    end) .. '"'
  end
  assert(t == "table", "cannot encode " .. t)
  local inner = indent .. "  "
  local parts = {}
  if is_array(v) then
    if #v == 0 then return "[]" end
    for i = 1, #v do parts[i] = inner .. encode(v[i], inner) end
    return "[\n" .. table.concat(parts, ",\n") .. "\n" .. indent .. "]"
  end
  local keys = {}
  for k in pairs(v) do keys[#keys + 1] = k end
  table.sort(keys, function(a, b) return tostring(a) < tostring(b) end)
  for i, k in ipairs(keys) do
    parts[i] = inner .. encode(tostring(k)) .. ": " .. encode(v[k], inner)
  end
  return "{\n" .. table.concat(parts, ",\n") .. "\n" .. indent .. "}"
end

local function write(name, value)
  local f = assert(io.open(out_dir .. "/" .. name, "wb"))
  f:write(encode(value), "\n")
  f:close()
  io.stderr:write("wrote ", out_dir, "/", name, "\n")
end

---------------------------------------------------------------------------
-- hands.json: the `hands = {...}` literal in Game:init_game_object
---------------------------------------------------------------------------

do
  local src = read("balatro_source/game.lua")
  local start = src:find("\n        hands = {\n", 1, true)
  assert(start, "hands table not found in game.lua")
  local body_start = start + #"\n        hands = "
  -- The literal closes at the first line that is exactly 8 spaces + "}".
  local stop = src:find("\n        }", body_start, true)
  assert(stop, "end of hands table not found")
  local literal = src:sub(body_start, stop + #"\n        }" - 1)
  local hands = assert(loadstring("return " .. literal))()
  local count = 0
  for _ in pairs(hands) do count = count + 1 end
  assert(count == 12, "expected 12 poker hands, found " .. count)
  write("hands.json", hands)
end

---------------------------------------------------------------------------
-- challenges.json
---------------------------------------------------------------------------

do
  G = {}
  assert(loadstring(read("balatro_source/challenges.lua")))()
  assert(#G.CHALLENGES == 20, "expected 20 challenges")
  write("challenges.json", G.CHALLENGES)
end

---------------------------------------------------------------------------
-- blind_amounts.json
---------------------------------------------------------------------------

do
  local src = read("balatro_source/functions/misc_functions.lua")
  local code = src:match("\n(function get_blind_amount%(ante%).-\nend)\n")
  assert(code, "get_blind_amount not found")
  assert(loadstring(code))()

  local ffi = require("ffi")
  ffi.cdef("typedef union { double d; uint64_t u; } jd_bits;")
  local function hex(x)
    local b = ffi.new("jd_bits")
    b.d = x
    return string.format("%08x%08x", tonumber(b.u / 4294967296ULL), tonumber(b.u % 4294967296ULL))
  end

  local tables = {}
  for scaling = 1, 3 do
    G = { GAME = { modifiers = { scaling = scaling } } }
    local rows = {}
    for ante = -1, 80 do
      local v = get_blind_amount(ante)
      rows[#rows + 1] = { ante = ante, bits = hex(v), value = tostring(v) }
    end
    tables[tostring(scaling)] = rows
  end
  write("blind_amounts.json", tables)
end

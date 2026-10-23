--[[
  Error definitions for BalatroBot API.
  Type aliases defined in types.lua.
]]

---@type ErrorNames
BB_ERROR_NAMES = {
  INTERNAL_ERROR = "INTERNAL_ERROR",
  BAD_REQUEST = "BAD_REQUEST",
  INVALID_STATE = "INVALID_STATE",
  NOT_ALLOWED = "NOT_ALLOWED",
}

---@type ErrorCodes
BB_ERROR_CODES = {
  INTERNAL_ERROR = -32000,
  BAD_REQUEST = -32001,
  INVALID_STATE = -32002,
  NOT_ALLOWED = -32003,
}

return BB_ERROR_NAMES, BB_ERROR_CODES

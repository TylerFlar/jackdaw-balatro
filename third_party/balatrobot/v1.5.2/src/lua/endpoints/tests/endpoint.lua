-- src/lua/endpoints/tests/endpoint.lua

-- ==========================================================================
-- Test Endpoint Endpoint Params
-- ==========================================================================

---@class Request.Endpoint.Test.Endpoint.Params
---@field required_string string A required string field
---@field optional_string? string Optional string field
---@field required_integer integer Required integer field
---@field optional_integer? integer Optional integer field
---@field optional_array_integers? integer[] Optional array of integers

-- ==========================================================================
-- Test Endpoint Endpoint
-- ==========================================================================

---@type Endpoint
return {

  name = "test_endpoint",

  description = "Test endpoint with schema for dispatcher testing",

  schema = {
    required_string = {
      type = "string",
      required = true,
      description = "A required string field",
    },

    optional_string = {
      type = "string",
      required = false,
      description = "Optional string field",
    },

    required_integer = {
      type = "integer",
      required = true,
      description = "Required integer field",
    },

    optional_integer = {
      type = "integer",
      required = false,
      description = "Optional integer field",
    },

    optional_array_integers = {
      type = "array",
      required = false,
      items = "integer",
      description = "Optional array of integers",
    },
  },

  requires_state = nil,

  ---@param args Request.Endpoint.Test.Endpoint.Params
  ---@param send_response fun(response: Response.Endpoint)
  execute = function(args, send_response)
    send_response({
      success = true,
      received_args = args,
    })
  end,
}

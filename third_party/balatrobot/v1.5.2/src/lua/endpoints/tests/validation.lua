-- src/lua/endpoints/tests/validation.lua

-- ==========================================================================
-- Test Validation Endpoint Params
-- ==========================================================================

---@class Request.Endpoint.Test.Validation.Params
---@field required_field string Required string field for basic validation testing
---@field string_field? string Optional string field for type validation
---@field integer_field? integer Optional integer field for type validation
---@field boolean_field? boolean Optional boolean field for type validation
---@field array_field? table Optional array field for type validation
---@field table_field? table Optional table field for type validation
---@field array_of_integers? integer[] Optional array that must contain only integers

-- ==========================================================================
-- Test Validation Endpoint
-- ==========================================================================

---@type Endpoint
return {

  name = "test_validation",

  description = "Comprehensive validation test endpoint for validator module testing",

  schema = {
    required_field = {
      type = "string",
      required = true,
      description = "Required string field for basic validation testing",
    },

    string_field = {
      type = "string",
      required = false,
      description = "Optional string field for type validation",
    },

    integer_field = {
      type = "integer",
      required = false,
      description = "Optional integer field for type validation",
    },

    boolean_field = {
      type = "boolean",
      required = false,
      description = "Optional boolean field for type validation",
    },

    array_field = {
      type = "array",
      required = false,
      description = "Optional array field for type validation",
    },

    table_field = {
      type = "table",
      required = false,
      description = "Optional table field for type validation",
    },

    array_of_integers = {
      type = "array",
      required = false,
      items = "integer",
      description = "Optional array that must contain only integers",
    },
  },

  requires_state = nil,

  ---@param args Request.Endpoint.Test.Validation.Params
  ---@param send_response fun(response: Response.Endpoint)
  execute = function(args, send_response)
    send_response({
      success = true,
      received_args = args,
    })
  end,
}

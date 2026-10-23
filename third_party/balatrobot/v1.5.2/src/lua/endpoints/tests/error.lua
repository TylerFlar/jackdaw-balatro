-- src/lua/endpoints/tests/error.lua

-- ==========================================================================
-- Test Error Endpoint Params
-- ==========================================================================

---@class Request.Endpoint.Test.Error.Params
---@field error_type "throw_error"|"success" Whether to throw an error or succeed

-- ==========================================================================
-- Test Error Endpoint
-- ==========================================================================

---@type Endpoint
return {

  name = "test_error_endpoint",

  description = "Test endpoint that throws runtime errors",

  schema = {
    error_type = {
      type = "string",
      required = true,
      enum = { "throw_error", "success" },
      description = "Whether to throw an error or succeed",
    },
  },

  requires_state = nil,

  ---@param args Request.Endpoint.Test.Error.Params
  ---@param send_response fun(response: Response.Endpoint)
  execute = function(args, send_response)
    if args.error_type == "throw_error" then
      error("Intentional test error from endpoint execution")
    else
      send_response({
        success = true,
      })
    end
  end,
}

-- src/lua/endpoints/tests/state.lua

-- ==========================================================================
-- Test State Endpoint Params
-- ==========================================================================

---@class Request.Endpoint.Test.State.Params

-- ==========================================================================
-- TestState Endpoint
-- ==========================================================================

---@type Endpoint
return {

  name = "test_state_endpoint",

  description = "Test endpoint that requires specific game states",

  schema = {},

  requires_state = { G.STATES.SPLASH, G.STATES.MENU },

  ---@param _ Request.Endpoint.Test.State.Params
  ---@param send_response fun(response: Response.Endpoint)
  execute = function(_, send_response)
    send_response({
      success = true,
      state_validated = true,
    })
  end,
}

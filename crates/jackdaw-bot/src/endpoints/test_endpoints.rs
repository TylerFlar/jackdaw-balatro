//! The debug-mode test endpoints BalatroBot's own core tests call.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/endpoints/tests/{echo,state,error,validation}.lua`, which `utils/debugger.lua`
//! registers when `BB_SETTINGS.debug` is on (`echo.lua` registers as `test_endpoint`). They
//! touch no game state.

use super::{Ctx, Endpoint};
use crate::errors::BotError;
use crate::json::{self, Json};
use crate::lua::Value;
use crate::states::GState;
use crate::validator::{Field, FieldType};

fn echo(ctx: &mut Ctx) -> Result<Json, BotError> {
    if ctx.dry_run {
        return Ok(Json::Null);
    }
    let args = json::from_value(&Value::Table(ctx.args.clone()))
        .map_err(|e| BotError::internal(format!("Failed to encode response: {e}")))?;
    Ok(Json::obj([
        ("success", Json::Bool(true)),
        ("received_args", args),
    ]))
}

/// `tests/echo.lua` (registered as `test_endpoint`).
pub const ECHO: Endpoint = Endpoint {
    name: "test_endpoint",
    schema: &[
        Field::new("optional_integer", FieldType::Integer, false),
        Field::new("required_string", FieldType::String, true),
        Field::new("optional_string", FieldType::String, false),
        Field::array_of("optional_array_integers", FieldType::Integer, false),
        Field::new("required_integer", FieldType::Integer, true),
    ],
    requires_state: None,
    run: echo,
};

/// `tests/state.lua`: only at the splash screen or the menu.
pub const STATE: Endpoint = Endpoint {
    name: "test_state_endpoint",
    schema: &[],
    requires_state: Some(&[GState::Splash, GState::Menu]),
    run: |ctx| {
        if ctx.dry_run {
            return Ok(Json::Null);
        }
        Ok(Json::obj([
            ("success", Json::Bool(true)),
            ("state_validated", Json::Bool(true)),
        ]))
    },
};

/// `tests/error.lua`: `error()` inside `execute`, which the dispatcher reports as
/// `INTERNAL_ERROR` with Lua's `chunkname:line: message` text.
pub const ERROR: Endpoint = Endpoint {
    name: "test_error_endpoint",
    schema: &[Field::new("error_type", FieldType::String, true)],
    requires_state: None,
    run: |ctx| {
        if ctx.dry_run {
            return Ok(Json::Null);
        }
        if ctx.str("error_type") == Some("throw_error") {
            return Err(BotError::internal(
                "[SMODS balatrobot \"src/lua/endpoints/tests/error.lua\"]:36: Intentional test error from endpoint execution",
            ));
        }
        Ok(Json::obj([("success", Json::Bool(true))]))
    },
};

/// `tests/validation.lua`: every field type.
pub const VALIDATION: Endpoint = Endpoint {
    name: "test_validation",
    schema: &[
        Field::new("array_field", FieldType::Array, false),
        Field::array_of("array_of_integers", FieldType::Integer, false),
        Field::new("table_field", FieldType::Table, false),
        Field::new("required_field", FieldType::String, true),
        Field::new("integer_field", FieldType::Integer, false),
        Field::new("string_field", FieldType::String, false),
        Field::new("boolean_field", FieldType::Boolean, false),
    ],
    requires_state: None,
    run: echo,
};

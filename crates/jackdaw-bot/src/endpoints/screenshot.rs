//! `screenshot`: capture the screen to a PNG file.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/endpoints/screenshot.lua`. The sim draws nothing, so the capture never yields an
//! image: the endpoint always fails as the live one does when encoding fails
//! (`INTERNAL_ERROR`, "Failed to encode screenshot"), and writes no file.

use super::{Ctx, Endpoint};
use crate::errors::BotError;
use crate::json::Json;
use crate::validator::{Field, FieldType};

pub const ENDPOINT: Endpoint = Endpoint {
    name: "screenshot",
    schema: &[Field::new("path", FieldType::String, true)],
    requires_state: None,
    run,
};

fn run(ctx: &mut Ctx) -> Result<Json, BotError> {
    if ctx.dry_run {
        return Ok(Json::Null);
    }
    Err(BotError::internal("Failed to encode screenshot"))
}

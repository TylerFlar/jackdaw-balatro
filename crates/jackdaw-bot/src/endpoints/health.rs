//! `health`: connection check.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/endpoints/health.lua`.

use super::{Ctx, Endpoint};
use crate::errors::BotError;
use crate::json::Json;

pub const ENDPOINT: Endpoint = Endpoint {
    name: "health",
    schema: &[],
    requires_state: None,
    run,
};

fn run(ctx: &mut Ctx) -> Result<Json, BotError> {
    if ctx.dry_run {
        return Ok(Json::Null);
    }
    Ok(Json::obj([("status", Json::str("ok"))]))
}

//! `select`: play the blind on deck.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/endpoints/select.lua`: `G.FUNCS.select_blind` on the blind on deck's button.

use jackdaw_core::action::Action;

use super::{Ctx, Endpoint, states};
use crate::errors::BotError;
use crate::json::Json;

pub const ENDPOINT: Endpoint = Endpoint {
    name: "select",
    schema: &[],
    requires_state: Some(states::BLIND_SELECT),
    run,
};

fn run(ctx: &mut Ctx) -> Result<Json, BotError> {
    if ctx.dry_run {
        return Ok(Json::Null);
    }
    ctx.step(&Action::SelectBlind, "Cannot select the blind")
}

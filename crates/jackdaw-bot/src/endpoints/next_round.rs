//! `next_round`: leave the shop for the blind-select screen.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/endpoints/next_round.lua`: `G.FUNCS.toggle_shop`.

use jackdaw_core::action::Action;

use super::{Ctx, Endpoint, states};
use crate::errors::BotError;
use crate::json::Json;

pub const ENDPOINT: Endpoint = Endpoint {
    name: "next_round",
    schema: &[],
    requires_state: Some(states::SHOP),
    run,
};

fn run(ctx: &mut Ctx) -> Result<Json, BotError> {
    if ctx.dry_run {
        return Ok(Json::Null);
    }
    ctx.step(&Action::NextRound, "Cannot leave the shop")
}

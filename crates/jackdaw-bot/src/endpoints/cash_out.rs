//! `cash_out`: collect the round's money and go to the shop.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/endpoints/cash_out.lua`: `G.FUNCS.cash_out`.

use jackdaw_core::action::Action;

use super::{Ctx, Endpoint, states};
use crate::errors::BotError;
use crate::json::Json;

pub const ENDPOINT: Endpoint = Endpoint {
    name: "cash_out",
    schema: &[],
    requires_state: Some(states::ROUND_EVAL),
    run,
};

fn run(ctx: &mut Ctx) -> Result<Json, BotError> {
    if ctx.dry_run {
        return Ok(Json::Null);
    }
    ctx.step(&Action::CashOut, "Cannot cash out")
}

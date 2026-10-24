//! `reroll`: reroll the shop's cards.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/endpoints/reroll.lua`: refused when a paid reroll costs more than the money above
//! `bankrupt_at`; then `G.FUNCS.reroll_shop`.

use jackdaw_core::action::Action;

use super::{Ctx, Endpoint, states};
use crate::errors::BotError;
use crate::json::Json;
use crate::lua::tostring;

pub const ENDPOINT: Endpoint = Endpoint {
    name: "reroll",
    schema: &[],
    requires_state: Some(states::SHOP),
    run,
};

fn run(ctx: &mut Ctx) -> Result<Json, BotError> {
    let gs = &ctx.game().gs;
    let reroll_cost = gs.current_round.reroll_cost;
    let available = gs.dollars - gs.bankrupt_at;
    if reroll_cost > 0.0 && available < reroll_cost {
        return Err(BotError::not_allowed(format!(
            "Not enough dollars to reroll. Available: {}, Required: {}",
            tostring(available),
            tostring(reroll_cost)
        )));
    }
    if ctx.dry_run {
        return Ok(Json::Null);
    }
    ctx.step(&Action::Reroll, "Cannot reroll")
}

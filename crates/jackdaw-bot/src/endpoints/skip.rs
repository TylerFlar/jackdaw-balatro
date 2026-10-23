//! `skip`: skip the Small or Big blind for its tag.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/endpoints/skip.lua`: the Boss blind can't be skipped; otherwise
//! `G.FUNCS.skip_blind` on the tag's skip button. (When the tag opens a pack, the live endpoint
//! keeps waiting for `BLIND_SELECT`; the sim answers with the open pack.)

use jackdaw_core::action::Action;
use jackdaw_core::state::BlindSlot;

use super::{Ctx, Endpoint, states};
use crate::errors::BotError;
use crate::json::Json;

pub const ENDPOINT: Endpoint = Endpoint {
    name: "skip",
    schema: &[],
    requires_state: Some(states::BLIND_SELECT),
    run,
};

fn run(ctx: &mut Ctx) -> Result<Json, BotError> {
    // `get_blinds_info()[string.lower(G.GAME.blind_on_deck)].type == "BOSS"`.
    if ctx.game().gs.blind_on_deck == Some(BlindSlot::Boss) {
        return Err(BotError::not_allowed("Cannot skip Boss blind"));
    }
    if ctx.dry_run {
        return Ok(Json::Null);
    }
    ctx.step(&Action::SkipBlind, "Cannot skip the blind")
}

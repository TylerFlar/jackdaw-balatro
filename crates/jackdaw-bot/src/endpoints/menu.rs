//! `menu`: back to the main menu from any state.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/endpoints/menu.lua`: `G.FUNCS.go_to_menu` deletes the run (unless the game already
//! is at the menu), and the response is the menu's view.

use super::{Ctx, Endpoint};
use crate::errors::BotError;
use crate::json::Json;
use crate::session::Run;
use crate::states::GState;

pub const ENDPOINT: Endpoint = Endpoint {
    name: "menu",
    schema: &[],
    requires_state: None,
    run,
};

fn run(ctx: &mut Ctx) -> Result<Json, BotError> {
    if ctx.dry_run {
        return Ok(Json::Null);
    }
    if ctx.session.state() != Some(GState::Menu) {
        ctx.session.run = Run::Menu;
    }
    Ok(ctx.session.view())
}

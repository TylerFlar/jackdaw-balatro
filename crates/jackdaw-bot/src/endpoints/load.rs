//! `load`: replace the run with one read from a file.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/endpoints/load.lua`. The file is a snapshot written by `save` (`jackdaw.snapshot/1`
//! JSON), not a `.jkr` save. As in the live game, the current run is deleted
//! (`G:delete_run()`) before the file is decoded: a file that isn't a valid snapshot leaves no
//! run, in state `UNKNOWN`, until `menu` or another `load`. (Copying the file to the game's
//! save directory, "Failed to prepare save file for loading", has no counterpart here.)

use jackdaw_core::game::Game;

use super::{Ctx, Endpoint};
use crate::errors::BotError;
use crate::json::Json;
use crate::session::Run;
use crate::validator::{Field, FieldType};

pub const ENDPOINT: Endpoint = Endpoint {
    name: "load",
    schema: &[Field::new("path", FieldType::String, true)],
    requires_state: None,
    run,
};

fn run(ctx: &mut Ctx) -> Result<Json, BotError> {
    if ctx.dry_run {
        return Ok(Json::Null);
    }
    let path = ctx.str("path").unwrap_or_default().to_owned();
    let unreadable = || BotError::internal(format!("Failed to read save file: '{path}'"));
    let source = ctx.session.resolve_path(&path).ok_or_else(unreadable)?;
    let bytes = std::fs::read(&source).map_err(|_| unreadable())?;
    // `G:delete_run()`: the run is gone whatever the file holds.
    let deleted = match std::mem::take(&mut ctx.session.run) {
        Run::Active(g) => Some(g),
        Run::Deleted(g) => g,
        Run::Menu => None,
    };
    ctx.session.run = Run::Deleted(deleted);
    let game = serde_json::from_slice::<serde_json::Value>(&bytes)
        .ok()
        .and_then(|v| Game::from_snapshot(&v).ok())
        .ok_or_else(|| BotError::internal("Invalid save file format"))?;
    ctx.session.run = Run::Active(Box::new(game));
    Ok(Json::obj([
        ("success", Json::Bool(true)),
        ("path", Json::Str(path)),
    ]))
}

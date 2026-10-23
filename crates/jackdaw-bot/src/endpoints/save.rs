//! `save`: write the run to a file.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/endpoints/save.lua`. The live game writes its compressed `.jkr` save; the sim writes
//! its canonical snapshot (`jackdaw.snapshot/1` JSON, `docs/snapshot-schema.md`), which `load`
//! reads back. A path outside the session's filesystem root fails like an unwritable one.
//! (The "Can only save during an active run" and "Failed to save game state" errors can't
//! happen here: every state the endpoint accepts is a run, and the snapshot always exists.)

use super::{Ctx, Endpoint};
use crate::errors::BotError;
use crate::json::Json;
use crate::states::GState;
use crate::validator::{Field, FieldType};

pub const ENDPOINT: Endpoint = Endpoint {
    name: "save",
    schema: &[Field::new("path", FieldType::String, true)],
    requires_state: Some(&[
        GState::SelectingHand,
        GState::HandPlayed,
        GState::DrawToHand,
        GState::GameOver,
        GState::Shop,
        GState::PlayTarot,
        GState::BlindSelect,
        GState::RoundEval,
        GState::TarotPack,
        GState::PlanetPack,
        GState::SpectralPack,
        GState::StandardPack,
        GState::BuffoonPack,
        GState::NewRound,
        GState::SmodsBoosterOpened,
    ]),
    run,
};

fn run(ctx: &mut Ctx) -> Result<Json, BotError> {
    if ctx.dry_run {
        return Ok(Json::Null);
    }
    let path = ctx.str("path").unwrap_or_default().to_owned();
    let fail = || BotError::internal(format!("Failed to write save file to '{path}'"));
    let target = ctx.session.resolve_path(&path).ok_or_else(fail)?;
    let text = ctx.game().snapshot().to_string();
    std::fs::write(&target, text).map_err(|_| fail())?;
    Ok(Json::obj([
        ("success", Json::Bool(true)),
        ("path", Json::Str(path)),
    ]))
}

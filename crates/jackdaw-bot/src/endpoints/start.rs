//! `start`: a new run with a deck, a stake and an optional seed.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/endpoints/start.lua`. The stake is checked before the deck. Without a seed the run
//! gets one of vanilla's random 8-character seeds. The live game's "Deck not found in game
//! data" error cannot happen: all fifteen decks exist.

use jackdaw_core::game::Game;
use jackdaw_core::run::RunSpec;

use super::{Ctx, Endpoint, states};
use crate::errors::BotError;
use crate::json::Json;
use crate::session::Run;
use crate::validator::{Field, FieldType};

pub const ENDPOINT: Endpoint = Endpoint {
    name: "start",
    schema: &[
        Field::new("deck", FieldType::String, true),
        Field::new("stake", FieldType::String, true),
        Field::new("seed", FieldType::String, false),
    ],
    requires_state: Some(states::MENU),
    run,
};

/// `STAKE_ENUM_TO_NUMBER`.
fn stake_number(name: &str) -> Option<u8> {
    Some(match name {
        "WHITE" => 1,
        "RED" => 2,
        "GREEN" => 3,
        "BLACK" => 4,
        "BLUE" => 5,
        "PURPLE" => 6,
        "ORANGE" => 7,
        "GOLD" => 8,
        _ => return None,
    })
}

/// `DECK_ENUM_TO_NAME`, as the back's key.
pub fn deck_key(name: &str) -> Option<&'static str> {
    Some(match name {
        "RED" => "b_red",
        "BLUE" => "b_blue",
        "YELLOW" => "b_yellow",
        "GREEN" => "b_green",
        "BLACK" => "b_black",
        "MAGIC" => "b_magic",
        "NEBULA" => "b_nebula",
        "GHOST" => "b_ghost",
        "ABANDONED" => "b_abandoned",
        "CHECKERED" => "b_checkered",
        "ZODIAC" => "b_zodiac",
        "PAINTED" => "b_painted",
        "ANAGLYPH" => "b_anaglyph",
        "PLASMA" => "b_plasma",
        "ERRATIC" => "b_erratic",
        _ => return None,
    })
}

fn run(ctx: &mut Ctx) -> Result<Json, BotError> {
    let stake_arg = ctx.str("stake").unwrap_or_default();
    let Some(stake) = stake_number(stake_arg) else {
        return Err(BotError::bad_request(format!(
            "Invalid stake enum. Must be one of: WHITE, RED, GREEN, BLACK, BLUE, PURPLE, ORANGE, GOLD. Got: {stake_arg}"
        )));
    };
    let deck_arg = ctx.str("deck").unwrap_or_default();
    let Some(back) = deck_key(deck_arg) else {
        return Err(BotError::bad_request(format!(
            "Invalid deck enum. Must be one of: RED, BLUE, YELLOW, GREEN, BLACK, MAGIC, NEBULA, GHOST, ABANDONED, CHECKERED, ZODIAC, PAINTED, ANAGLYPH, PLASMA, ERRATIC. Got: {deck_arg}"
        )));
    };
    if ctx.dry_run {
        return Ok(Json::Null);
    }
    let seed = match ctx.str("seed") {
        Some(s) => s.to_owned(),
        None => ctx.session.random_seed(),
    };
    let spec = RunSpec {
        back: jackdaw_core::card::center_by_key(back).expect("every deck has a back"),
        stake,
        seed,
        challenge: None,
    };
    let game = Game::new_run(&spec).map_err(BotError::internal)?;
    ctx.session.run = Run::Active(Box::new(game));
    Ok(ctx.session.view())
}

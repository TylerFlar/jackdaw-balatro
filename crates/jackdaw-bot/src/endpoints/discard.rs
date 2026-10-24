//! `discard`: discard cards from the hand.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/endpoints/discard.lua`. The cards are selected as for `play` (see
//! [`super::play::click_select`]), then `G.FUNCS.discard_cards_from_highlighted` runs.

use jackdaw_core::action::Action;

use super::play::{HIGHLIGHTED_LIMIT, click_select, hand_indices};
use super::{Ctx, Endpoint, states};
use crate::errors::BotError;
use crate::json::Json;
use crate::lua::tostring;
use crate::validator::{Field, FieldType};

pub const ENDPOINT: Endpoint = Endpoint {
    name: "discard",
    schema: &[Field::array_of("cards", FieldType::Integer, true)],
    requires_state: Some(states::SELECTING_HAND),
    run,
};

fn run(ctx: &mut Ctx) -> Result<Json, BotError> {
    let cards = ctx.table("cards").cloned().unwrap_or_default();
    if cards.is_empty() {
        return Err(BotError::bad_request(
            "Must provide at least one card to discard",
        ));
    }
    if ctx.game().gs.current_round.discards_left <= 0.0 {
        return Err(BotError::bad_request("No discards left"));
    }
    if cards.len() > HIGHLIGHTED_LIMIT {
        return Err(BotError::bad_request(format!(
            "You can only discard {HIGHLIGHTED_LIMIT} cards"
        )));
    }
    let indices = hand_indices(ctx.game(), &cards)
        .map_err(|bad| BotError::bad_request(format!("Invalid card index: {}", tostring(bad))))?;
    let selected = click_select(ctx.game(), &indices);
    if selected.is_empty() {
        return Err(BotError::not_allowed(
            "Cannot discard: the selection leaves no card highlighted",
        ));
    }
    if ctx.dry_run {
        return Ok(Json::Null);
    }
    ctx.step(&Action::Discard(selected), "Cannot discard these cards")
}

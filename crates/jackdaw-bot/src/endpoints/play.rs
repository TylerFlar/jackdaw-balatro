//! `play`: play cards from the hand.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/endpoints/play.lua`. After the checks, BalatroBot selects the cards the way a
//! player does: `G.hand:unhighlight_all()` (Cerulean Bell's forced card stays highlighted),
//! then `card:click()` on each index in order. A click toggles its card, and adds it only while
//! fewer than five are highlighted (cardarea.lua:131-212). The highlighted cards are then
//! played (`G.FUNCS.play_cards_from_highlighted`), in hand order. A request that ends with no
//! card highlighted (an index given twice toggles its card off) is refused: vanilla never plays
//! an empty hand.

use jackdaw_core::action::Action;
use jackdaw_core::game::Game;

use super::{Ctx, Endpoint, states};
use crate::errors::BotError;
use crate::json::Json;
use crate::lua::{Table, tostring};
use crate::validator::{Field, FieldType};

/// `G.hand.config.highlighted_limit`.
pub const HIGHLIGHTED_LIMIT: usize = 5;

pub const ENDPOINT: Endpoint = Endpoint {
    name: "play",
    schema: &[Field::array_of("cards", FieldType::Integer, true)],
    requires_state: Some(states::SELECTING_HAND),
    run,
};

/// The indices of `cards` (an integer array) that name hand cards, or the first that doesn't.
pub fn hand_indices(game: &Game, cards: &Table) -> Result<Vec<usize>, f64> {
    let n = game.gs.areas.hand.len();
    cards
        .ipairs()
        .map(|v| {
            let i = v.as_num().unwrap_or(f64::NAN);
            if i >= 0.0 && i < n as f64 {
                Ok(i as usize)
            } else {
                Err(i)
            }
        })
        .collect()
}

/// `G.hand:unhighlight_all()`, then `card:click()` on each index: the hand positions left
/// highlighted, in hand order.
pub fn click_select(game: &Game, indices: &[usize]) -> Vec<usize> {
    let hand = &game.gs.areas.hand;
    let forced = |id: u32| {
        hand.iter()
            .any(|c| c.sort_id == id && c.ability.forced_selection)
    };
    let mut highlighted: Vec<u32> = game
        .rt
        .highlighted
        .iter()
        .copied()
        .filter(|&id| forced(id))
        .collect();
    for &i in indices {
        let id = hand[i].sort_id;
        if let Some(pos) = highlighted.iter().position(|&h| h == id) {
            // `remove_from_highlighted` keeps a forced card.
            if !forced(id) {
                highlighted.remove(pos);
            }
        } else if highlighted.len() < HIGHLIGHTED_LIMIT {
            highlighted.push(id);
        }
    }
    let mut positions: Vec<usize> = highlighted
        .iter()
        .filter_map(|id| hand.iter().position(|c| c.sort_id == *id))
        .collect();
    positions.sort_unstable();
    positions
}

fn run(ctx: &mut Ctx) -> Result<Json, BotError> {
    let cards = ctx.table("cards").cloned().unwrap_or_default();
    if cards.is_empty() {
        return Err(BotError::bad_request(
            "Must provide at least one card to play",
        ));
    }
    if cards.len() > HIGHLIGHTED_LIMIT {
        return Err(BotError::bad_request(format!(
            "You can only play {HIGHLIGHTED_LIMIT} cards"
        )));
    }
    let indices = hand_indices(ctx.game(), &cards)
        .map_err(|bad| BotError::bad_request(format!("Invalid card index: {}", tostring(bad))))?;
    let selected = click_select(ctx.game(), &indices);
    if selected.is_empty() {
        return Err(BotError::not_allowed(
            "Cannot play: the selection leaves no card highlighted",
        ));
    }
    if ctx.dry_run {
        return Ok(Json::Null);
    }
    ctx.step(&Action::PlayHand(selected), "Cannot play these cards")
}

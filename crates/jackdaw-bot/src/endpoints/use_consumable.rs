//! `use`: use a consumable, with target cards from the hand when it takes them.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/endpoints/use.lua`, step for step. A consumable "requires cards" when its config has
//! `max_highlighted`; those can only be used in `SELECTING_HAND`. Aura has no
//! `max_highlighted`, so BalatroBot never highlights a target for it and its use is refused
//! unless Cerulean Bell keeps a card highlighted.
//!
//! The targets are highlighted the vanilla way (the engine's `UseConsumable`): Cerulean Bell's
//! forced card first, then the given cards, a repeat skipped. BalatroBot instead clears the
//! highlight with `force` (dropping the forced card) and adds each index, repeats included
//! (`docs/balatrobot.md`).

use jackdaw_core::action::Action;

use super::play::hand_indices;
use super::{Ctx, Endpoint};
use crate::errors::BotError;
use crate::json::Json;
use crate::lua::tostring;
use crate::states::GState;
use crate::validator::{Field, FieldType};

pub const ENDPOINT: Endpoint = Endpoint {
    name: "use",
    schema: &[
        Field::array_of("cards", FieldType::Integer, false),
        Field::new("consumable", FieldType::Integer, true),
    ],
    requires_state: Some(&[GState::SelectingHand, GState::Shop]),
    run,
};

fn plural(n: f64) -> &'static str {
    if n == 1.0 { "" } else { "s" }
}

fn run(ctx: &mut Ctx) -> Result<Json, BotError> {
    let game = ctx.game();
    let index = ctx.num("consumable").expect("required");
    // Step 1: the consumable's index.
    let n = game.gs.areas.consumables.len();
    if index < 0.0 || index >= n as f64 {
        return Err(BotError::bad_request(format!(
            "Consumable index out of range: {}",
            tostring(index)
        )));
    }
    let i = index as usize;
    let card = &game.gs.areas.consumables[i];
    let id = card.sort_id;
    let name = card.name();
    let center = card.center();
    // Step 2: `ability.consumeable.max_highlighted ~= nil`.
    let max = center.config_num("max_highlighted");
    let requires_cards = max.is_some();
    // Step 3.
    if requires_cards && ctx.session.state() != Some(GState::SelectingHand) {
        return Err(BotError::invalid_state(format!(
            "Consumable '{name}' requires card selection and can only be used in SELECTING_HAND state"
        )));
    }
    let mut targets = None;
    if let Some(max) = max {
        // Step 4.
        let cards = ctx.table("cards").cloned().unwrap_or_default();
        if cards.is_empty() {
            return Err(BotError::bad_request(format!(
                "Consumable '{name}' requires card selection"
            )));
        }
        let picked = hand_indices(game, &cards).map_err(|bad| {
            BotError::bad_request(format!("Card index out of range: {}", tostring(bad)))
        })?;
        // Step 5.
        let min = center.config_num("min_highlighted").unwrap_or(1.0);
        let count = picked.len() as f64;
        if min == max && count != min {
            return Err(BotError::bad_request(format!(
                "Consumable '{name}' requires exactly {} card{} (provided: {})",
                tostring(min),
                plural(min),
                tostring(count)
            )));
        }
        if count < min {
            return Err(BotError::bad_request(format!(
                "Consumable '{name}' requires at least {} card{} (provided: {})",
                tostring(min),
                plural(min),
                tostring(count)
            )));
        }
        if count > max {
            return Err(BotError::bad_request(format!(
                "Consumable '{name}' requires at most {} card{} (provided: {})",
                tostring(max),
                plural(max),
                tostring(count)
            )));
        }
        targets = Some(picked);
    }
    // Steps 6-8: highlight, then the game's own gates.
    let (can_use, check_use) = game
        .consumable_gates(id, targets.as_deref().unwrap_or(&[]))
        .map_err(|e| BotError::not_allowed(format!("Cannot use consumable '{name}': {e}")))?;
    if !can_use {
        return Err(BotError::not_allowed(format!(
            "Consumable '{name}' cannot be used at this time"
        )));
    }
    if check_use {
        return Err(BotError::not_allowed(format!(
            "Cannot use consumable '{name}': insufficient space"
        )));
    }
    if ctx.dry_run {
        return Ok(Json::Null);
    }
    ctx.step(
        &Action::UseConsumable { index: i, targets },
        "Cannot use the consumable",
    )
}

//! `rearrange`: put the hand, the jokers or the consumables in a new order.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/endpoints/rearrange.lua`. The whole permutation is applied
//! (`G.hand.cards = new_array`), as dragging the cards would; pinned jokers go back to the left
//! when the area next aligns. The hand can be rearranged while choosing a hand and inside a
//! pack that dealt one. (BalatroBot also writes the new positions into `ability.order` and
//! the shared `center.order`; no rule reads them.)

use jackdaw_core::area::AreaId;

use super::{Ctx, Endpoint};
use crate::errors::BotError;
use crate::json::Json;
use crate::lua::tostring;
use crate::states::GState;
use crate::validator::{Field, FieldType};

pub const ENDPOINT: Endpoint = Endpoint {
    name: "rearrange",
    schema: &[
        Field::array_of("jokers", FieldType::Integer, false),
        Field::array_of("hand", FieldType::Integer, false),
        Field::array_of("consumables", FieldType::Integer, false),
    ],
    requires_state: Some(&[
        GState::SelectingHand,
        GState::Shop,
        GState::SmodsBoosterOpened,
    ]),
    run,
};

fn run(ctx: &mut Ctx) -> Result<Json, BotError> {
    let given = ["hand", "jokers", "consumables"]
        .iter()
        .filter(|k| ctx.arg(k).is_some())
        .count();
    if given == 0 {
        return Err(BotError::bad_request(
            "Must provide exactly one of: hand, jokers, or consumables",
        ));
    }
    if given > 1 {
        return Err(BotError::bad_request(
            "Can only rearrange one type at a time",
        ));
    }
    let state = ctx.session.state();
    let (area, name) = if ctx.arg("hand").is_some() {
        if !matches!(
            state,
            Some(GState::SelectingHand | GState::SmodsBoosterOpened)
        ) {
            return Err(BotError::invalid_state(
                "Can only rearrange hand during hand selection",
            ));
        }
        if state == Some(GState::SmodsBoosterOpened) && ctx.game().gs.areas.hand.is_empty() {
            return Err(BotError::not_allowed(
                "No cards to rearrange. You can only rearrange hand in Arcana and Spectral packs.",
            ));
        }
        (AreaId::Hand, "hand")
    } else if ctx.arg("jokers").is_some() {
        (AreaId::Jokers, "jokers")
    } else {
        (AreaId::Consumables, "consumables")
    };
    let indices: Vec<f64> = ctx
        .table(name)
        .map(|t| t.ipairs().filter_map(|v| v.as_num()).collect())
        .unwrap_or_default();
    let n = ctx.game().gs.areas.get(area).len();
    if indices.len() != n {
        return Err(BotError::bad_request(format!(
            "Must provide exactly {n} indices for {name}"
        )));
    }
    let mut seen = vec![false; n];
    let mut order = Vec::with_capacity(n);
    for &idx in &indices {
        if idx < 0.0 || idx >= n as f64 {
            return Err(BotError::bad_request(format!(
                "Index out of range for {name}: {}",
                tostring(idx)
            )));
        }
        let i = idx as usize;
        if seen[i] {
            return Err(BotError::bad_request(format!(
                "Duplicate index in {name}: {}",
                tostring(idx)
            )));
        }
        seen[i] = true;
        order.push(i);
    }
    if ctx.dry_run {
        return Ok(Json::Null);
    }
    ctx.game_mut()
        .rearrange(area, &order)
        .map_err(|e| super::step_error("Cannot rearrange", e))?;
    Ok(ctx.session.view())
}

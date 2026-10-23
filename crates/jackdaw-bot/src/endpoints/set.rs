//! `set` (debug): write game values, or restock the shop.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/endpoints/set.lua`, branch for branch. The fields are written one at a time in the
//! Lua's order (money, chips, ante, round, hands, discards), so a negative value is refused only
//! after the fields before it were set. `shop = true` restocks the shop
//! ([`jackdaw_core::game::Game::debug_restock_shop`]).

use jackdaw_core::debug::SetField;

use super::{Ctx, Endpoint};
use crate::errors::BotError;
use crate::json::Json;
use crate::states::GState;
use crate::validator::{Field, FieldType};

pub const ENDPOINT: Endpoint = Endpoint {
    name: "set",
    schema: &[
        Field::new("round", FieldType::Integer, false),
        Field::new("hands", FieldType::Integer, false),
        Field::new("ante", FieldType::Integer, false),
        Field::new("discards", FieldType::Integer, false),
        Field::new("chips", FieldType::Integer, false),
        Field::new("shop", FieldType::Boolean, false),
        Field::new("money", FieldType::Integer, false),
    ],
    requires_state: None,
    run,
};

/// The numeric fields in the order set.lua writes them, with their error text.
type Setter = fn(f64) -> SetField;

const FIELDS: [(&str, &str, Setter); 6] = [
    ("money", "Money", SetField::Money),
    ("chips", "Chips", SetField::Chips),
    ("ante", "Ante", SetField::Ante),
    ("round", "Round", SetField::Round),
    ("hands", "Hands", SetField::Hands),
    ("discards", "Discards", SetField::Discards),
];

fn run(ctx: &mut Ctx) -> Result<Json, BotError> {
    if !ctx.session.in_run_stage() {
        return Err(BotError::invalid_state("Can only set during an active run"));
    }
    let any = FIELDS.iter().any(|(k, _, _)| ctx.arg(k).is_some()) || ctx.arg("shop").is_some();
    if !any {
        return Err(BotError::bad_request(
            "Must provide at least one field to set",
        ));
    }
    for (key, label, field) in FIELDS {
        let Some(v) = ctx.num(key) else { continue };
        if v < 0.0 {
            return Err(BotError::bad_request(format!(
                "{label} must be a positive integer"
            )));
        }
        if !ctx.dry_run
            && let Some(g) = ctx.session.g_game_mut()
        {
            g.debug_set(field(v));
        }
    }
    if ctx.truthy("shop") {
        if ctx.session.state() != Some(GState::Shop) {
            return Err(BotError::not_allowed(
                "Can re-stock shop only in SHOP state",
            ));
        }
        if !ctx.dry_run {
            ctx.game_mut()
                .debug_restock_shop()
                .map_err(|e| super::step_error("Cannot restock the shop", e))?;
        }
    }
    if ctx.dry_run {
        return Ok(Json::Null);
    }
    Ok(ctx.session.view())
}

//! `sell`: sell a joker or a consumable.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/endpoints/sell.lua`. BalatroBot calls `G.FUNCS.sell_card` directly, skipping the
//! sell button's `Card:can_sell_card` gate, so the live game sells an eternal joker; vanilla
//! never does, and the sim refuses it (`NOT_ALLOWED`).

use jackdaw_core::action::Action;
use jackdaw_core::area::AreaId;
use jackdaw_core::state::Phase;

use super::{Ctx, Endpoint};
use crate::errors::BotError;
use crate::json::Json;
use crate::lua::tostring;
use crate::states::GState;
use crate::validator::{Field, FieldType};

pub const ENDPOINT: Endpoint = Endpoint {
    name: "sell",
    schema: &[
        Field::new("joker", FieldType::Integer, false),
        Field::new("consumable", FieldType::Integer, false),
    ],
    requires_state: Some(&[GState::SelectingHand, GState::Shop]),
    run,
};

fn run(ctx: &mut Ctx) -> Result<Json, BotError> {
    let joker = ctx.num("joker");
    let consumable = ctx.num("consumable");
    let count = usize::from(joker.is_some()) + usize::from(consumable.is_some());
    if count == 0 {
        return Err(BotError::bad_request(
            "Must provide exactly one of: joker or consumable",
        ));
    }
    if count > 1 {
        return Err(BotError::bad_request("Can only sell one item at a time"));
    }
    let gs = &ctx.game().gs;
    let (area, index, kind) = match joker {
        Some(i) => {
            if gs.areas.jokers.is_empty() {
                return Err(BotError::not_allowed("No jokers available to sell"));
            }
            (AreaId::Jokers, i, "joker")
        }
        None => {
            if gs.areas.consumables.is_empty() {
                return Err(BotError::not_allowed("No consumables available to sell"));
            }
            (
                AreaId::Consumables,
                consumable.expect("one is set"),
                "consumable",
            )
        }
    };
    let cards = gs.areas.get(area);
    // `source_array[args.x + 1]`.
    let pos = index + 1.0;
    if !(pos >= 1.0 && pos <= cards.len() as f64) {
        return Err(BotError::bad_request(format!(
            "Index out of range for {kind}: {}",
            tostring(pos - 1.0)
        )));
    }
    let i = index as usize;
    // Vanilla's gate, which BalatroBot skips (Card:can_sell_card, card.lua:1653-1665).
    if cards[i].ability.eternal {
        return Err(BotError::not_allowed(format!(
            "Cannot sell '{}': the card is eternal (vanilla's sell button refuses it)",
            cards[i].name()
        )));
    }
    debug_assert!(matches!(gs.phase, Phase::SelectingHand | Phase::Shop));
    if ctx.dry_run {
        return Ok(Json::Null);
    }
    ctx.step(&Action::SellCard { area, index: i }, "Cannot sell the card")
}

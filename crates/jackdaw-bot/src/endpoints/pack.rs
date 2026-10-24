//! `pack`: take a card from the open booster pack, or skip it.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/endpoints/pack.lua`, branch for branch:
//!
//! - an Arcana or Spectral pack (judged by its first card's set) first checks every target
//!   index against the dealt hand;
//! - `select_card` then checks the card index, a joker against the slots (no Negative
//!   exception), Ankh's need for a joker, and the target count of a card that takes targets
//!   (Aura: exactly 1; else `min_highlighted`..`max_highlighted`);
//! - BalatroBot then calls `G.FUNCS.use_card` directly, skipping the button's
//!   `can_use_consumeable` / `can_select_card` gate. The sim keeps vanilla's gate: a pick the
//!   game greys out (Judgement at full joker slots, The Emperor at full consumable slots, ...)
//!   is refused with `NOT_ALLOWED` (`docs/balatrobot.md`; `docs/live.md`, B01).
//!
//! Targets are highlighted the vanilla way, as for `use`. `skip` runs `G.FUNCS.skip_booster`.

use jackdaw_core::action::Action;
use jackdaw_data::CenterSet;

use super::{Ctx, Endpoint, states};
use crate::errors::BotError;
use crate::json::Json;
use crate::lua::tostring;
use crate::validator::{Field, FieldType};

pub const ENDPOINT: Endpoint = Endpoint {
    name: "pack",
    schema: &[
        Field::array_of("targets", FieldType::Integer, false),
        Field::new("skip", FieldType::Boolean, false),
        Field::new("card", FieldType::Integer, false),
    ],
    requires_state: Some(states::BOOSTER),
    run,
};

/// `get_consumable_target_requirements(card_key)`.
enum Requirement {
    Targets { min: f64, max: f64 },
    Joker,
}

fn requirement(key: &str) -> Option<Requirement> {
    if key == "c_aura" {
        return Some(Requirement::Targets { min: 1.0, max: 1.0 });
    }
    if key == "c_ankh" {
        return Some(Requirement::Joker);
    }
    let center = jackdaw_data::center_by_key(key)?.get();
    let max = center.config_num("max_highlighted")?;
    Some(Requirement::Targets {
        min: center.config_num("min_highlighted").unwrap_or(1.0),
        max,
    })
}

fn run(ctx: &mut Ctx) -> Result<Json, BotError> {
    let card_arg = ctx.num("card");
    let skip = ctx.truthy("skip");
    let set = usize::from(card_arg.is_some()) + usize::from(skip);
    if set == 0 {
        return Err(BotError::bad_request(
            "Invalid arguments. You must provide one of: card, skip",
        ));
    }
    if set > 1 {
        return Err(BotError::bad_request(
            "Invalid arguments. Cannot provide both card and skip",
        ));
    }
    if skip {
        if ctx.dry_run {
            return Ok(Json::Null);
        }
        return ctx.step(&Action::SkipPack, "Cannot skip the pack");
    }
    let index = card_arg.expect("card is set");
    let targets: Vec<f64> = ctx
        .table("targets")
        .map(|t| t.ipairs().filter_map(|v| v.as_num()).collect())
        .unwrap_or_default();
    let game = ctx.game();
    let gs = &game.gs;
    let hand_size = gs.areas.hand.len();
    let in_hand = |t: f64| t + 1.0 >= 1.0 && t + 1.0 <= hand_size as f64;
    let target_error = |t: f64| {
        BotError::bad_request(format!(
            "Target card index out of range. Index: {}, Hand size: {hand_size}",
            tostring(t)
        ))
    };
    // Arcana and Spectral packs wait for the dealt hand, then check every target first.
    let needs_hand = gs
        .areas
        .pack_cards
        .first()
        .is_some_and(|c| matches!(c.set(), CenterSet::Tarot | CenterSet::Spectral));
    if needs_hand && let Some(&bad) = targets.iter().find(|&&t| !in_hand(t)) {
        return Err(target_error(bad));
    }
    // select_card().
    let pack = &gs.areas.pack_cards;
    let pos = index + 1.0;
    if !(pos >= 1.0 && pos <= pack.len() as f64) {
        return Err(BotError::bad_request(format!(
            "Card index out of range. Index: {}, Available cards: {}",
            tostring(index),
            pack.len()
        )));
    }
    let i = index as usize;
    let card = &pack[i];
    let jokers = gs.areas.jokers.len() as f64;
    if card.set() == CenterSet::Joker && jokers >= gs.joker_slots {
        return Err(BotError::not_allowed(format!(
            "Cannot select joker, joker slots are full. Current: {}, Limit: {}",
            tostring(jokers),
            tostring(gs.joker_slots)
        )));
    }
    let key = card.center.key();
    let mut use_targets = None;
    match requirement(key) {
        Some(Requirement::Joker) => {
            if jokers == 0.0 {
                return Err(BotError::not_allowed(format!(
                    "Card '{key}' requires at least 1 joker. Current: {}",
                    tostring(jokers)
                )));
            }
        }
        Some(Requirement::Targets { min, max }) => {
            let count = targets.len() as f64;
            if count < min || count > max {
                let msg = if min == max {
                    format!(
                        "Card '{key}' requires exactly {} target card(s). Provided: {}",
                        tostring(min),
                        tostring(count)
                    )
                } else {
                    format!(
                        "Card '{key}' requires {}-{} target card(s). Provided: {}",
                        tostring(min),
                        tostring(max),
                        tostring(count)
                    )
                };
                return Err(BotError::bad_request(msg));
            }
            if let Some(&bad) = targets.iter().find(|&&t| !in_hand(t)) {
                return Err(target_error(bad));
            }
            if !targets.is_empty() {
                use_targets = Some(targets.iter().map(|&t| t as usize).collect::<Vec<_>>());
            }
        }
        None => {}
    }
    // Vanilla's button gate, which BalatroBot skips.
    if card.center().consumeable {
        let (can_use, check_use) = game
            .consumable_gates(card.sort_id, use_targets.as_deref().unwrap_or(&[]))
            .map_err(|e| BotError::not_allowed(format!("Cannot select '{}': {e}", card.name())))?;
        if !can_use || check_use {
            return Err(BotError::not_allowed(format!(
                "Cannot select '{}' from the pack: vanilla greys this pick out (it cannot be used now)",
                card.name()
            )));
        }
    }
    if ctx.dry_run {
        return Ok(Json::Null);
    }
    let name = card.name();
    ctx.step(
        &Action::PickPackCard {
            index: i,
            targets: use_targets,
        },
        &format!("Cannot select '{name}' from the pack"),
    )
}

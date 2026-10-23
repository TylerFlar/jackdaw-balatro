//! `add` (debug): create a joker, consumable, voucher, pack or playing card.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/endpoints/add.lua`, check for check. BalatroBot then calls Steamodded's
//! `SMODS.add_card` (or `SMODS.add_booster_to_shop` for a pack); the sim runs
//! [`jackdaw_core::game::Game::debug_add`], which follows the Python oracle where it has an
//! `add` and Steamodded otherwise. The differences from the live game are listed in
//! `docs/balatrobot.md` (and `docs/parity/gaps.md` B15, B18, B19). A key that names no card of
//! its kind fails as `SMODS.add_card` does: `Failed to add card: <key>`.

use jackdaw_core::card::{EditionKind, Seal};
use jackdaw_core::debug::{AddKey, AddSpec};

use super::{Ctx, Endpoint};
use crate::errors::BotError;
use crate::json::Json;
use crate::states::GState;
use crate::validator::{Field, FieldType};

pub const ENDPOINT: Endpoint = Endpoint {
    name: "add",
    schema: &[
        Field::new("rental", FieldType::Boolean, false),
        Field::new("edition", FieldType::String, false),
        Field::new("perishable", FieldType::Integer, false),
        Field::new("seal", FieldType::String, false),
        Field::new("enhancement", FieldType::String, false),
        Field::new("key", FieldType::String, true),
        Field::new("eternal", FieldType::Boolean, false),
    ],
    requires_state: Some(&[GState::SelectingHand, GState::Shop, GState::RoundEval]),
    run,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum CardType {
    Joker,
    Consumable,
    Voucher,
    Pack,
    PlayingCard,
}

impl CardType {
    fn name(self) -> &'static str {
        match self {
            CardType::Joker => "joker",
            CardType::Consumable => "consumable",
            CardType::Voucher => "voucher",
            CardType::Pack => "pack",
            CardType::PlayingCard => "playing_card",
        }
    }
}

/// `detect_card_type(key)`.
fn detect_card_type(key: &str) -> Option<CardType> {
    let prefix = key.get(..2).unwrap_or(key);
    Some(match prefix {
        "j_" => CardType::Joker,
        "c_" => CardType::Consumable,
        "v_" => CardType::Voucher,
        "p_" => CardType::Pack,
        _ => {
            // `^[HDCS]_[2-9TJQKA]$`
            let b = key.as_bytes();
            if b.len() == 3
                && b"HDCS".contains(&b[0])
                && b[1] == b'_'
                && b"23456789TJQKA".contains(&b[2])
            {
                CardType::PlayingCard
            } else {
                return None;
            }
        }
    })
}

fn run(ctx: &mut Ctx) -> Result<Json, BotError> {
    let key = ctx.str("key").unwrap_or_default().to_owned();
    let Some(card_type) = detect_card_type(&key) else {
        return Err(BotError::bad_request(
            "Invalid card key format. Expected: joker (j_*), consumable (c_*), voucher (v_*), or playing card (SUIT_RANK)",
        ));
    };
    let state = ctx.session.state();
    if card_type == CardType::PlayingCard && state != Some(GState::SelectingHand) {
        return Err(BotError::invalid_state(
            "Playing cards can only be added in SELECTING_HAND state",
        ));
    }
    if card_type == CardType::Voucher && state != Some(GState::Shop) {
        return Err(BotError::invalid_state(
            "Vouchers can only be added in SHOP state",
        ));
    }
    if card_type == CardType::Pack && state != Some(GState::Shop) {
        return Err(BotError::invalid_state(
            "Packs can only be added in SHOP state",
        ));
    }
    if card_type == CardType::Pack {
        // `G.shop_booster` always exists in SHOP ("Shop booster area not available" can't
        // happen); its `card_limit` is 2.
        if ctx.game().gs.areas.shop_booster.len() >= 2 {
            return Err(BotError::not_allowed(
                "Cannot add pack, shop booster slots are full",
            ));
        }
        if jackdaw_data::center_by_key(&key).is_none() {
            return Err(BotError::bad_request(format!("Pack key not found: {key}")));
        }
    }
    let seal_arg = ctx.str("seal").map(str::to_owned);
    if seal_arg.is_some() && card_type != CardType::PlayingCard {
        return Err(BotError::bad_request(
            "Seal can only be applied to playing cards",
        ));
    }
    let mut seal = None;
    if let Some(s) = &seal_arg {
        seal = Some(match s.as_str() {
            "RED" => Seal::Red,
            "BLUE" => Seal::Blue,
            "GOLD" => Seal::Gold,
            "PURPLE" => Seal::Purple,
            _ => {
                return Err(BotError::bad_request(
                    "Invalid seal value. Expected: RED, BLUE, GOLD, or PURPLE",
                ));
            }
        });
    }
    let edition_arg = ctx.str("edition").map(str::to_owned);
    if edition_arg.is_some() && matches!(card_type, CardType::Voucher | CardType::Pack) {
        return Err(BotError::bad_request(format!(
            "Edition cannot be applied to {}s",
            card_type.name()
        )));
    }
    if card_type == CardType::Consumable && edition_arg.as_deref().is_some_and(|e| e != "NEGATIVE")
    {
        return Err(BotError::bad_request(
            "Consumables can only have NEGATIVE edition",
        ));
    }
    let mut edition = None;
    if let Some(e) = &edition_arg {
        edition = Some(match e.as_str() {
            "HOLO" => EditionKind::Holo,
            "FOIL" => EditionKind::Foil,
            "POLYCHROME" => EditionKind::Polychrome,
            "NEGATIVE" => EditionKind::Negative,
            _ => {
                return Err(BotError::bad_request(
                    "Invalid edition value. Expected: HOLO, FOIL, POLYCHROME, or NEGATIVE",
                ));
            }
        });
    }
    let enhancement_arg = ctx.str("enhancement").map(str::to_owned);
    if enhancement_arg.is_some() && card_type != CardType::PlayingCard {
        return Err(BotError::bad_request(
            "Enhancement can only be applied to playing cards",
        ));
    }
    let mut enhancement = None;
    if let Some(e) = &enhancement_arg {
        let key = match e.as_str() {
            "BONUS" => "m_bonus",
            "MULT" => "m_mult",
            "WILD" => "m_wild",
            "GLASS" => "m_glass",
            "STEEL" => "m_steel",
            "STONE" => "m_stone",
            "GOLD" => "m_gold",
            "LUCKY" => "m_lucky",
            _ => {
                return Err(BotError::bad_request(
                    "Invalid enhancement value. Expected: BONUS, MULT, WILD, GLASS, STEEL, STONE, GOLD, or LUCKY",
                ));
            }
        };
        enhancement = jackdaw_data::center_by_key(key);
    }
    let eternal = ctx.truthy("eternal");
    if eternal && card_type != CardType::Joker {
        return Err(BotError::bad_request(
            "Eternal can only be applied to jokers",
        ));
    }
    let perishable = ctx.num("perishable");
    if perishable.is_some() && card_type != CardType::Joker {
        return Err(BotError::bad_request(
            "Perishable can only be applied to jokers",
        ));
    }
    if let Some(p) = perishable
        && (p.floor() != p || p < 1.0)
    {
        return Err(BotError::bad_request(
            "Perishable must be a positive integer (>= 1)",
        ));
    }
    let rental = ctx.truthy("rental");
    if rental && card_type != CardType::Joker {
        return Err(BotError::bad_request(
            "Rental can only be applied to jokers",
        ));
    }
    // `SMODS.add_card` raises on a key that names no card of its kind.
    let failed = || BotError::bad_request(format!("Failed to add card: {key}"));
    let parsed = AddKey::parse(&key).ok_or_else(failed)?;
    let kind_ok = matches!(
        (card_type, parsed),
        (CardType::Joker, AddKey::Joker(_))
            | (CardType::Consumable, AddKey::Consumable(_))
            | (CardType::Voucher, AddKey::Voucher(_))
            | (CardType::Pack, AddKey::Booster(_))
            | (CardType::PlayingCard, AddKey::Playing(_))
    );
    if !kind_ok {
        return Err(failed());
    }
    if ctx.dry_run {
        return Ok(Json::Null);
    }
    let spec = AddSpec {
        key: parsed,
        seal,
        edition,
        enhancement,
        eternal,
        perishable,
        rental,
    };
    let game = ctx.game_mut();
    let id = game
        .debug_add(&spec)
        .map_err(|e| super::step_error("Cannot add the card", e))?;
    if matches!(parsed, AddKey::Voucher(_)) {
        // `SMODS.add_card` never calls `create_shop_card_ui`: the voucher has no buy button.
        game.rt.areas.no_buy_button.push(id);
    }
    Ok(ctx.session.view())
}

//! `buy`: buy a shop card, redeem a voucher or open a pack.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/endpoints/buy.lua`. The checks read the gamestate view, as the Lua does: the area's
//! cards, the card's `cost.buy`, and for a joker or a consumable the destination area's
//! `count` against its `limit`. That last check has no Negative exception, so a Negative
//! joker at full slots, which vanilla sells, is refused. Then `G.FUNCS.buy_from_shop` (cards)
//! or `G.FUNCS.use_card` (vouchers, packs) runs.
//!
//! The buy button is read from the card's UI (`children.buy_button.definition`). Every shop
//! card has one except a voucher that `add` placed (`SMODS.add_card` skips
//! `create_shop_card_ui`): indexing the missing button raises, and the dispatcher reports the
//! Lua error as INTERNAL_ERROR. The `if not btn` branch after it ("No buy button found for
//! card") is unreachable.

use jackdaw_core::action::Action;
use jackdaw_core::area::AreaId;

use super::{Ctx, Endpoint, states};
use crate::errors::BotError;
use crate::json::Json;
use crate::lua::tostring;
use crate::validator::{Field, FieldType};

pub const ENDPOINT: Endpoint = Endpoint {
    name: "buy",
    schema: &[
        Field::new("pack", FieldType::Integer, false),
        Field::new("voucher", FieldType::Integer, false),
        Field::new("card", FieldType::Integer, false),
    ],
    requires_state: Some(states::SHOP),
    run,
};

#[derive(Clone, Copy, PartialEq, Eq)]
enum What {
    Card,
    Voucher,
    Pack,
}

fn run(ctx: &mut Ctx) -> Result<Json, BotError> {
    // The last of card, voucher, pack that is set picks the area; `set` counts them.
    let mut what = None;
    let mut index = 0.0;
    let mut set = 0;
    for (name, w) in [
        ("card", What::Card),
        ("voucher", What::Voucher),
        ("pack", What::Pack),
    ] {
        if let Some(i) = ctx.num(name) {
            what = Some(w);
            index = i;
            set += 1;
        }
    }
    let Some(what) = what else {
        return Err(BotError::bad_request(
            "Invalid arguments. You must provide one of: card, voucher, pack",
        ));
    };
    if set > 1 {
        return Err(BotError::bad_request(
            "Invalid arguments. Cannot provide more than one of: card, voucher, or pack",
        ));
    }
    let game = ctx.game();
    let gs = &game.gs;
    let area = match what {
        What::Card => AreaId::ShopJokers,
        What::Voucher => AreaId::ShopVouchers,
        What::Pack => AreaId::ShopBooster,
    };
    let cards = gs.areas.get(area);
    if cards.is_empty() {
        return Err(BotError::bad_request(match what {
            What::Card => "No jokers/consumables/cards in the shop. Reroll to restock the shop",
            What::Voucher => "No vouchers to redeem. Defeat boss blind to restock",
            What::Pack => "No packs to open",
        }));
    }
    // `area.cards[args.x + 1]`.
    let pos = index + 1.0;
    if !(pos >= 1.0 && pos <= cards.len() as f64) {
        let (i, n) = (tostring(index), cards.len());
        return Err(BotError::bad_request(match what {
            What::Card => format!("Card index out of range. Index: {i}, Available cards: {n}"),
            What::Voucher => format!("Voucher index out of range. Index: {i}, Available: {n}"),
            What::Pack => format!("Pack index out of range. Index: {i}, Available: {n}"),
        }));
    }
    let i = index as usize;
    let card = &cards[i];
    let available = gs.dollars - gs.bankrupt_at;
    if card.cost > 0.0 && card.cost > available {
        return Err(BotError::bad_request(format!(
            "Card is not affordable. Cost: {}, Available money: {}",
            tostring(card.cost),
            tostring(available)
        )));
    }
    let set_name = crate::view::card_set_name(card);
    if set_name == "JOKER" {
        let (count, limit) = (gs.areas.jokers.len() as f64, gs.joker_slots);
        if count >= limit {
            return Err(BotError::bad_request(format!(
                "Cannot purchase joker card, joker slots are full. Current: {}, Limit: {}",
                tostring(count),
                tostring(limit)
            )));
        }
    }
    if matches!(set_name, "PLANET" | "SPECTRAL" | "TAROT") {
        let (count, limit) = (gs.areas.consumables.len() as f64, gs.consumable_slots);
        if count >= limit {
            return Err(BotError::bad_request(format!(
                "Cannot purchase consumable card, consumable slots are full. Current: {}, Limit: {}",
                tostring(count),
                tostring(limit)
            )));
        }
    }
    // `G.shop_vouchers.cards[pos].children.buy_button.definition` (buy.lua:182): a voucher
    // that `add` put in the shop has no buy button, and indexing it raises.
    if what == What::Voucher && game.rt.areas.no_buy_button.contains(&card.sort_id) {
        return Err(BotError::internal(
            "[SMODS balatrobot \"src/lua/endpoints/buy.lua\"]:182: attempt to index field \
             'buy_button' (a nil value)",
        ));
    }
    if ctx.dry_run {
        return Ok(Json::Null);
    }
    match what {
        What::Card => ctx.step(&Action::BuyCard(i), "Cannot buy the card"),
        What::Voucher => ctx.step(&Action::RedeemVoucher(i), "Cannot redeem the voucher"),
        What::Pack => ctx.step(&Action::OpenBooster(i), "Cannot open the pack"),
    }
}

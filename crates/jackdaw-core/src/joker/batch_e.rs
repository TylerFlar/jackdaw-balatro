//! Joker batch E: one function per joker, all of its `calculate_joker` branches. These jokers
//! act in the shop and consumable contexts: `open_booster`, `selling_self`, `selling_card`,
//! `reroll_shop`, `ending_shop`, `skip_blind`, `skipping_booster`, `playing_card_added` and
//! `using_consumeable`, plus Campfire's and Invisible Joker's end of round and the main step.
//!
//! The main step's shared rules (mod.rs) score Constellation, Hologram, Campfire and Throwback:
//! their XMult lives in `ability.x_mult` and the shared `x_mult > 1` rule returns it, so their
//! functions only grow or reset it. Many branches also queue a `card_eval_status_text` event;
//! that is UI only and not queued here.

use jackdaw_data::{CenterSet, tags};

use super::api::{message, mult_mod};
use super::{CalcFn, Ctx, Kind};
use crate::area::AreaId;
use crate::card::EditionKind;
use crate::event::Event;
use crate::game::Game;
use crate::pools::PoolType;
use crate::scoring::Effect;
use crate::state::BlindSlot;

/// The function of a ported joker in this batch.
pub(super) fn lookup(key: &str) -> Option<CalcFn> {
    Some(match key {
        "j_constellation" => constellation,
        "j_hologram" => hologram,
        "j_red_card" => red_card,
        "j_luchador" => luchador,
        "j_hallucination" => hallucination,
        "j_fortune_teller" => fortune_teller,
        "j_diet_cola" => diet_cola,
        "j_flash" => flash_card,
        "j_campfire" => campfire,
        "j_throwback" => throwback,
        "j_perkeo" => perkeo,
        "j_invisible" => invisible_joker,
        _ => return None,
    })
}

/// The main step's `+Mult` of a joker that keeps it in `ability.mult` (Red Card, Flash Card).
fn stored_mult(g: &Game, id: u32) -> Option<Effect> {
    let m = g.card(id).ability.mult;
    (m > 0.0).then(|| mult_mod(m))
}

// ---------------------------------------------------------------------------
// Consumable contexts
// ---------------------------------------------------------------------------

/// Hallucination: when a pack opens, with a free consumable slot (counting the buffer), a 1 in
/// `extra` chance of a Tarot (card.lua:2335-2351). No Blueprint guard: a copy rolls too, and
/// each hit reserves its slot until the creation event runs (G22).
fn hallucination(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::OpenBooster || !g.consumable_room() {
        return None;
    }
    let ante = g.gs.round_resets.ante;
    let odds = g.extra(id);
    if g.roll(&crate::rng::Key::new("halu").n(ante), odds) {
        g.create_consumable_later(PoolType::Tarot, "hal", None);
    }
    None
}

/// Constellation: +`extra` XMult for each Planet used (card.lua:2727-2733).
fn constellation(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::UsingConsumeable || ctx.blueprint > 0 {
        return None;
    }
    let used = ctx.consumeable?;
    if g.card(used).set() == CenterSet::Planet {
        let extra = g.extra(id);
        g.card_mut(id).ability.x_mult += extra;
    }
    None
}

/// Fortune Teller: +Mult equal to the Tarots used this run (card.lua:4016-4021). Its
/// `using_consumeable` branch only shows the count (card.lua:2722-2726).
fn fortune_teller(g: &mut Game, _id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::JokerMain {
        return None;
    }
    // `G.GAME.consumeable_usage_total` is nil until the first consumable is used.
    let tarot = g.gs.consumeable_usage.totals.get("tarot").copied()?;
    (tarot > 0.0).then(|| mult_mod(tarot))
}

/// Hologram: +`extra` XMult per playing card added to the deck (card.lua:2456-2461). The
/// dispatch refuses a Hologram that is `getting_sliced`.
fn hologram(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind == Kind::PlayingCardAdded && ctx.blueprint == 0 && !ctx.cards.is_empty() {
        let added = ctx.cards.len() as f64 * g.extra(id);
        g.card_mut(id).ability.x_mult += added;
    }
    None
}

// ---------------------------------------------------------------------------
// Selling
// ---------------------------------------------------------------------------

/// Luchador: sold during an active boss blind, it disables the boss (card.lua:2355-2360). No
/// Blueprint guard: a sold Blueprint copying it disables the boss too.
fn luchador(g: &mut Game, _id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let active = g.gs.blind.as_ref().is_some_and(|b| !b.disabled);
    if ctx.kind == Kind::SellingSelf && active && g.blind_type() == Some(BlindSlot::Boss) {
        g.blind_disable();
    }
    None
}

/// Diet Cola: sold, it queues a Double Tag (card.lua:2361-2370). No Blueprint guard.
fn diet_cola(g: &mut Game, _id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind == Kind::SellingSelf {
        g.queue(Event::AddTag(tags::TAG_DOUBLE, None));
    }
    None
}

/// Invisible Joker: `invis_rounds` counts the rounds it ends (card.lua:2934-2944); sold after
/// `extra` of them, it copies a random other joker at once, while the jokers (itself included)
/// fit the slots (card.lua:2371-2394). A Negative joker's copy loses the edition (G28), and a
/// copied Invisible Joker starts counting again.
fn invisible_joker(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    match ctx.kind {
        Kind::SellingSelf if ctx.blueprint == 0 => {
            let rounds = g.card(id).ability.invis_rounds.unwrap_or(0.0);
            if rounds < g.extra(id) {
                return None;
            }
            let mut others: Vec<u32> = g.joker_ids().into_iter().filter(|&j| j != id).collect();
            if others.is_empty() || (g.gs.areas.jokers.len() as f64) > g.gs.joker_slots {
                return None;
            }
            // `pseudorandom_element` sorts cards by sort id.
            others.sort_unstable();
            let seed = g.gs.rng.seed("invisible");
            let chosen = others[crate::rng::element_index(others.len(), seed)];
            let negative = g
                .card(chosen)
                .edition
                .is_some_and(|e| e.kind == EditionKind::Negative);
            let mut card = g.copy_card(chosen, negative, AreaId::Jokers);
            if card.ability.invis_rounds.is_some() {
                card.ability.invis_rounds = Some(0.0);
            }
            // Lua runs `add_to_deck` before the emplace; its effects don't depend on that.
            let copy = card.sort_id;
            g.gs.areas.emplace(AreaId::Jokers, card, false, false);
            g.add_to_deck(copy, false);
            None
        }
        // The dispatch already refused Blueprint copies of this branch.
        Kind::EndOfRound if !ctx.individual && !ctx.repetition => {
            let rounds = g.card(id).ability.invis_rounds.unwrap_or(0.0) + 1.0;
            g.card_mut(id).ability.invis_rounds = Some(rounds);
            Some(message())
        }
        _ => None,
    }
}

/// Campfire: +`extra` XMult for each other card sold (card.lua:2396-2402); back to X1 at the
/// end of a boss round (card.lua:2889-2895).
fn campfire(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    match ctx.kind {
        Kind::SellingCard if ctx.blueprint == 0 => {
            let extra = g.extra(id);
            g.card_mut(id).ability.x_mult += extra;
            None
        }
        // The dispatch already refused Blueprint copies of this branch.
        Kind::EndOfRound if !ctx.individual && !ctx.repetition => {
            let boss = g.gs.blind.as_ref().is_some_and(|b| b.boss);
            if boss && g.card(id).ability.x_mult > 1.0 {
                g.card_mut(id).ability.x_mult = 1.0;
                return Some(message());
            }
            None
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// The shop, packs and skips
// ---------------------------------------------------------------------------

/// Flash Card: +`extra` Mult per shop reroll (card.lua:2403-2411); +Mult in the main step
/// (card.lua:3998-4003).
fn flash_card(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    match ctx.kind {
        Kind::RerollShop if ctx.blueprint == 0 => {
            let extra = g.extra(id);
            g.card_mut(id).ability.mult += extra;
            None
        }
        Kind::JokerMain => stored_mult(g, id),
        _ => None,
    }
}

/// Red Card: +`extra` Mult per booster pack skipped (card.lua:2441-2455); +Mult in the main
/// step (card.lua:4034-4039).
fn red_card(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    match ctx.kind {
        Kind::SkippingBooster if ctx.blueprint == 0 => {
            let extra = g.extra(id);
            g.card_mut(id).ability.mult += extra;
            None
        }
        Kind::JokerMain => stored_mult(g, id),
        _ => None,
    }
}

/// Perkeo: leaving the shop with a consumable held, an event copies a random consumable as a
/// Negative (card.lua:2412-2426). No Blueprint guard (G16): every copy queues its own event,
/// and a later event can copy an earlier event's card.
fn perkeo(g: &mut Game, _id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind == Kind::EndingShop && !g.gs.areas.consumables.is_empty() {
        g.queue(Event::Call(perkeo_copy, 0, 0.0));
    }
    None
}

/// Perkeo's event (card.lua:2415-2422): `copy_card` of a consumable picked from `'perkeo'`,
/// `set_edition({negative = true})`, `add_to_deck` (the Negative's extra slot), `emplace`.
fn perkeo_copy(g: &mut Game, _: u32, _: f64) {
    let mut ids = g.gs.areas.ids(AreaId::Consumables);
    // `pseudorandom_element` sorts cards by sort id.
    ids.sort_unstable();
    let seed = g.gs.rng.seed("perkeo");
    let picked = ids[crate::rng::element_index(ids.len(), seed)];
    let mut card = g.copy_card(picked, false, AreaId::Consumables);
    let cost = g.cost_context(AreaId::Consumables);
    // The copy is not in the deck yet, so the edition adds no slot; `add_to_deck` does.
    card.set_edition(Some(EditionKind::Negative), &cost);
    let copy = card.sort_id;
    g.gs.areas.emplace(AreaId::Consumables, card, false, false);
    g.add_to_deck(copy, false);
}

/// Throwback: `Card:update` keeps `x_mult` at 1 + `extra` per blind skipped this run
/// (card.lua:4176-4178, cards.rs `update_cards`), and the shared `x_mult > 1` rule scores it.
/// Its `skip_blind` branch only shows the new value (card.lua:2428-2440).
fn throwback(_g: &mut Game, _id: u32, _ctx: &mut Ctx) -> Option<Effect> {
    None
}

//! Joker batch A: one function per joker, all of its `calculate_joker` branches.

use super::api::{mult, mult_mod, repetitions, xmult_mod};
use super::{CalcFn, Ctx, Kind};
use crate::area::AreaId;
use crate::card::Suit;
use crate::game::Game;
use crate::scoring::Effect;

/// The function of a ported joker in this batch.
pub(super) fn lookup(key: &str) -> Option<CalcFn> {
    Some(match key {
        "j_joker" => joker,
        "j_greedy_joker" | "j_lusty_joker" | "j_wrathful_joker" | "j_gluttenous_joker" => suit_mult,
        "j_hack" => hack,
        "j_gros_michel" | "j_cavendish" => banana,
        _ => return None,
    })
}

/// Joker: +4 Mult (card.lua:3980-3985).
fn joker(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    match ctx.kind {
        Kind::JokerMain => Some(mult_mod(g.card(id).ability.mult)),
        _ => None,
    }
}

/// Greedy, Lusty, Wrathful and Gluttonous Joker (`effect == 'Suit Mult'`): +Mult for each
/// scoring card of their suit (card.lua:3217-3223).
fn suit_mult(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::Individual || ctx.cardarea != Some(AreaId::Play) {
        return None;
    }
    let suit = Suit::from_name(g.extra_str(id, "suit")).expect("Suit Mult suit");
    let other = ctx.other_card?;
    let smeared = g.has_joker("Smeared Joker");
    if g.card(other).is_suit(suit, false, false, smeared) == Some(true) {
        return Some(mult(g.extra_num(id, "s_mult"), id));
    }
    None
}

/// Hack: retriggers each played 2, 3, 4 and 5 (card.lua:3374-3384).
fn hack(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::Repetition || ctx.cardarea != Some(AreaId::Play) {
        return None;
    }
    let other = ctx.other_card?;
    if matches!(g.card(other).get_id(), Some(2..=5)) {
        return Some(repetitions(g.extra(id), id));
    }
    None
}

/// Gros Michel and Cavendish: +Mult / XMult, and a 1 in `odds` chance at the end of each round
/// of going extinct (card.lua:3019-3045, 4022-4033).
fn banana(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let gros = g.card(id).name() == "Gros Michel";
    match ctx.kind {
        Kind::EndOfRound if !ctx.individual && !ctx.repetition => {
            let odds = g.extra_num(id, "odds");
            if g.roll(if gros { "gros_michel" } else { "cavendish" }, odds) {
                g.dissolve_later(id);
                if gros {
                    g.gs.sets.pool_flags.insert(
                        "gros_michel_extinct".to_owned(),
                        crate::state::LuaValue::Bool(true),
                    );
                }
            }
            Some(super::api::message())
        }
        Kind::JokerMain if gros => Some(mult_mod(g.extra_num(id, "mult"))),
        Kind::JokerMain => Some(xmult_mod(g.extra_num(id, "Xmult"))),
        _ => None,
    }
}

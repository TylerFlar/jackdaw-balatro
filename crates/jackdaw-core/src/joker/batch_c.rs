//! Joker batch C: one function per joker, all of its `calculate_joker` branches.
//!
//! Most of these jokers act in `evaluate_play`'s per-card contexts: `individual` for a scored
//! card (card.lua:3065-3270) or a held one (3271-3341), and `repetition` for a scored card
//! (3342-3385) or a held one (3386-3395). Their other branches live here too: Mime's at the end
//! of the round (2877-2887), Seltzer's countdown after each hand (3601-3630), Wee Joker's main
//! step (3873-3879) and Baseball Card's joker-on-joker step (3396-3408). Lucky Cat scores through
//! the main step's shared `x_mult > 1` rule (`joker_main_shared`).
//!
//! A context table holds one flag, so each function returns `None` (Lua's nil) for every
//! context but its own. The checks keep Lua's order: a roll is only drawn once the conditions
//! before it in the `and` chain hold.

use super::api::{chip_mod, chips, message, mult, repetitions, x_mult, xmult_mod};
use super::{CalcFn, Ctx, Kind};
use crate::area::AreaId;
use crate::card::{Extra, Suit};
use crate::event::Event;
use crate::game::Game;
use crate::lua;
use crate::pools::PoolType;
use crate::scoring::{Effect, ExtraMods};

/// The function of a ported joker in this batch.
pub(super) fn lookup(key: &str) -> Option<CalcFn> {
    Some(match key {
        "j_mime" => mime,
        "j_8_ball" => eight_ball,
        "j_dusk" => dusk,
        "j_raised_fist" => raised_fist,
        "j_fibonacci" => fibonacci,
        "j_scary_face" => scary_face,
        "j_even_steven" => even_steven,
        "j_odd_todd" => odd_todd,
        "j_scholar" => scholar,
        "j_business" => business_card,
        "j_hiker" => hiker,
        "j_baron" => baron,
        "j_photograph" => photograph,
        "j_reserved_parking" => reserved_parking,
        "j_lucky_cat" => lucky_cat,
        "j_ancient" => ancient_joker,
        "j_walkie_talkie" => walkie_talkie,
        "j_selzer" => seltzer,
        "j_smiley" => smiley_face,
        "j_ticket" => golden_ticket,
        "j_sock_and_buskin" => sock_and_buskin,
        "j_hanging_chad" => hanging_chad,
        "j_rough_gem" => rough_gem,
        "j_bloodstone" => bloodstone,
        "j_arrowhead" => arrowhead,
        "j_onyx_agate" => onyx_agate,
        "j_idol" => the_idol,
        "j_shoot_the_moon" => shoot_the_moon,
        "j_triboulet" => triboulet,
        "j_baseball" => baseball_card,
        "j_wee" => wee_joker,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Shared checks
// ---------------------------------------------------------------------------

/// `context.other_card` when the context is `kind` (individual or repetition) for a card of
/// `area` (`context.cardarea == G.play` or `G.hand`).
fn other_card(ctx: &Ctx, kind: Kind, area: AreaId) -> Option<u32> {
    if ctx.kind == kind && ctx.cardarea == Some(area) {
        ctx.other_card
    } else {
        None
    }
}

/// `card:is_face()`: Pareidolia makes every card a face; a debuffed card is never one.
fn is_face(g: &Game, card: u32) -> bool {
    g.card(card).is_face(false, g.has_joker("Pareidolia"))
}

/// `card:is_suit(suit)`, with Smeared Joker. A debuffed card returns nil, which is falsy.
fn is_suit(g: &Game, card: u32, suit: Suit) -> bool {
    g.card(card)
        .is_suit(suit, false, false, g.has_joker("Smeared Joker"))
        == Some(true)
}

/// `{dollars = x, card = self}`, after adding `x` to `G.GAME.dollar_buffer` and queueing the
/// event that clears it (Golden Ticket, Business Card, Rough Gem, Reserved Parking).
fn dollars(g: &mut Game, x: f64, id: u32) -> Effect {
    g.buffer_dollars(x);
    Effect {
        dollars: Some(x),
        card: Some(id),
        ..Effect::default()
    }
}

/// `{extra = {message = ...}, card = self}`: a scaling joker's "Upgrade!" (Hiker, Lucky Cat,
/// Wee Joker). The `extra` table has no `mult_mod`, `chip_mod`, `swap` or `func`, so applying
/// it changes nothing.
fn upgrade(id: u32) -> Effect {
    Effect {
        extra: Some(ExtraMods::default()),
        card: Some(id),
        ..Effect::default()
    }
}

/// `{message = localize('k_debuffed'), card = self}`: a held-card joker meeting a debuffed
/// card. Nothing scores, but the table counts as an effect of the card (Mime, Red Seal).
fn debuffed(id: u32) -> Effect {
    Effect {
        message: true,
        card: Some(id),
        ..Effect::default()
    }
}

// ---------------------------------------------------------------------------
// Individual: scored cards (card.lua:3066-3270)
// ---------------------------------------------------------------------------

/// Hiker: each scored card keeps +5 chips for good (card.lua:3067-3075). No Blueprint guard
/// (G16): a copy adds its own bonus.
fn hiker(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Play)?;
    let extra = g.extra(id);
    let a = &mut g.card_mut(other).ability;
    a.perma_bonus = lua::add(a.perma_bonus, extra);
    Some(upgrade(id))
}

/// Lucky Cat: gains `extra` XMult each time a Lucky Card triggers (card.lua:3076-3082). Its
/// XMult scores through the main step's shared `x_mult > 1` rule.
fn lucky_cat(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Play)?;
    if !g.card(other).lucky_trigger || ctx.blueprint > 0 {
        return None;
    }
    let extra = g.extra(id);
    let a = &mut g.card_mut(id).ability;
    a.x_mult = lua::add(a.x_mult, extra);
    Some(upgrade(id))
}

/// Wee Joker: gains `chip_mod` chips for each scored 2 (card.lua:3083-3092) and gives its
/// chips in the main step, even while they are 0 (card.lua:3873-3879).
fn wee_joker(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    match ctx.kind {
        Kind::Individual if ctx.cardarea == Some(AreaId::Play) => {
            let other = ctx.other_card?;
            if g.card(other).get_id() != Some(2) || ctx.blueprint > 0 {
                return None;
            }
            let chips = lua::add(g.extra_num(id, "chips"), g.extra_num(id, "chip_mod"));
            g.set_extra_num(id, "chips", chips);
            Some(upgrade(id))
        }
        Kind::JokerMain => Some(chip_mod(g.extra_num(id, "chips"))),
        _ => None,
    }
}

/// Photograph: XMult on the scoring hand's first face card (card.lua:3093-3105).
fn photograph(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Play)?;
    let first_face = ctx.scoring_hand.iter().copied().find(|&c| is_face(g, c));
    (first_face == Some(other)).then(|| x_mult(g.extra(id), id))
}

/// 8 Ball: a scored 8 makes a Tarot, 1 in `extra` (card.lua:3106-3126). With no room (cards plus
/// `consumeable_buffer`) there is no roll (G22); a hit reserves the slot, and the Tarot is made
/// by the event the effect's `extra.func` queues when scoring applies it.
fn eight_ball(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Play)?;
    if !g.consumable_room() || g.card(other).get_id() != Some(8) {
        return None;
    }
    let odds = g.extra(id);
    if !g.roll("8ball", odds) {
        return None;
    }
    g.rt.consumeable_buffer += 1.0;
    Some(Effect {
        extra: Some(ExtraMods {
            event: Some(Event::CreateConsumable {
                ty: PoolType::Tarot,
                append: "8ba",
                forced: None,
            }),
            ..ExtraMods::default()
        }),
        card: Some(id),
        ..Effect::default()
    })
}

/// The Idol: XMult for each scored card of the round's rank and suit (card.lua:3127-3135,
/// picked by `reset_idol_card`).
fn the_idol(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Play)?;
    let (suit, rank) = g.gs.current_round.idol_card;
    let hit = g.card(other).get_id() == Some(rank.id()) && is_suit(g, other, suit);
    hit.then(|| x_mult(g.extra(id), id))
}

/// Scary Face: +chips for each scored face card (card.lua:3136-3142).
fn scary_face(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Play)?;
    is_face(g, other).then(|| chips(g.extra(id), id))
}

/// Smiley Face: +Mult for each scored face card (card.lua:3143-3149).
fn smiley_face(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Play)?;
    is_face(g, other).then(|| mult(g.extra(id), id))
}

/// Golden Ticket: $ for each scored Gold Card, through the dollar buffer (card.lua:3150-3158).
fn golden_ticket(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Play)?;
    if g.card(other).name() != "Gold Card" {
        return None;
    }
    let x = g.extra(id);
    Some(dollars(g, x, id))
}

/// Scholar: +chips and +Mult for each scored Ace (card.lua:3159-3166).
fn scholar(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Play)?;
    (g.card(other).get_id() == Some(14)).then(|| chips_and_mult(g, id))
}

/// Walkie Talkie: +chips and +Mult for each scored 10 or 4 (card.lua:3167-3174).
fn walkie_talkie(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Play)?;
    matches!(g.card(other).get_id(), Some(10 | 4)).then(|| chips_and_mult(g, id))
}

/// `{chips = extra.chips, mult = extra.mult, card = self}` (Scholar, Walkie Talkie).
fn chips_and_mult(g: &Game, id: u32) -> Effect {
    Effect {
        chips: Some(g.extra_num(id, "chips")),
        mult: Some(g.extra_num(id, "mult")),
        card: Some(id),
        ..Effect::default()
    }
}

/// Business Card: a scored face card pays $2, 1 in `extra` (card.lua:3175-3184).
fn business_card(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Play)?;
    let odds = g.extra(id);
    if !is_face(g, other) || !g.roll("business", odds) {
        return None;
    }
    Some(dollars(g, 2.0, id))
}

/// Fibonacci: +Mult for each scored Ace, 2, 3, 5 or 8 (card.lua:3185-3195).
fn fibonacci(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Play)?;
    matches!(g.card(other).get_id(), Some(2 | 3 | 5 | 8 | 14)).then(|| mult(g.extra(id), id))
}

/// Even Steven: +Mult for each scored 10, 8, 6, 4 or 2 (card.lua:3196-3205). A Stone Card's id
/// is negative in Lua (`None` here), so it never counts.
fn even_steven(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Play)?;
    let even = matches!(g.card(other).get_id(), Some(n) if n <= 10 && n % 2 == 0);
    even.then(|| mult(g.extra(id), id))
}

/// Odd Todd: +chips for each scored Ace, 9, 7, 5 or 3 (card.lua:3206-3216).
fn odd_todd(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Play)?;
    let odd = matches!(g.card(other).get_id(), Some(n) if (n <= 10 && n % 2 == 1) || n == 14);
    odd.then(|| chips(g.extra(id), id))
}

/// Rough Gem: $ for each scored Diamond, through the dollar buffer (card.lua:3224-3232).
fn rough_gem(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Play)?;
    if !is_suit(g, other, Suit::Diamonds) {
        return None;
    }
    let x = g.extra(id);
    Some(dollars(g, x, id))
}

/// Onyx Agate: +Mult for each scored Club (card.lua:3233-3239).
fn onyx_agate(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Play)?;
    is_suit(g, other, Suit::Clubs).then(|| mult(g.extra(id), id))
}

/// Arrowhead: +chips for each scored Spade (card.lua:3240-3246).
fn arrowhead(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Play)?;
    is_suit(g, other, Suit::Spades).then(|| chips(g.extra(id), id))
}

/// Bloodstone: XMult for a scored Heart, 1 in `odds` (card.lua:3247-3254). Only Hearts roll.
fn bloodstone(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Play)?;
    let odds = g.extra_num(id, "odds");
    if !is_suit(g, other, Suit::Hearts) || !g.roll("bloodstone", odds) {
        return None;
    }
    Some(x_mult(g.extra_num(id, "Xmult"), id))
}

/// Ancient Joker: XMult for each scored card of the round's suit (card.lua:3255-3261, picked by
/// `reset_ancient_card`). The suit is set from the run's start (game.lua:2387-2388), so an
/// unset suit never reaches a hand.
fn ancient_joker(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Play)?;
    let suit = g.gs.current_round.ancient_card?;
    is_suit(g, other, suit).then(|| x_mult(g.extra(id), id))
}

/// Triboulet: XMult for each scored King or Queen (card.lua:3262-3269).
fn triboulet(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Play)?;
    matches!(g.card(other).get_id(), Some(12 | 13)).then(|| x_mult(g.extra(id), id))
}

// ---------------------------------------------------------------------------
// Individual: held cards (card.lua:3271-3341)
// ---------------------------------------------------------------------------

/// Shoot the Moon: +13 Mult for each Queen held (card.lua:3272-3286).
fn shoot_the_moon(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Hand)?;
    let c = g.card(other);
    if c.get_id() != Some(12) {
        return None;
    }
    if c.debuff {
        return Some(debuffed(id));
    }
    Some(Effect {
        h_mult: Some(13.0),
        card: Some(id),
        ..Effect::default()
    })
}

/// Baron: XMult for each King held (card.lua:3287-3301).
fn baron(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Hand)?;
    let c = g.card(other);
    if c.get_id() != Some(13) {
        return None;
    }
    if c.debuff {
        return Some(debuffed(id));
    }
    Some(x_mult(g.extra(id), id))
}

/// Reserved Parking: a held face card pays $, 1 in `odds` (card.lua:3302-3319). Its debuff
/// arm is dead: `is_face()` is false for a debuffed card, so a debuffed card never rolls.
fn reserved_parking(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Hand)?;
    let odds = g.extra_num(id, "odds");
    if !is_face(g, other) || !g.roll("parking", odds) {
        return None;
    }
    if g.card(other).debuff {
        return Some(debuffed(id));
    }
    let x = g.extra_num(id, "dollars");
    Some(dollars(g, x, id))
}

/// Raised Fist: twice the rank of the lowest held card as Mult (card.lua:3320-3340). The scan
/// reads `base.id` (debuffed cards count, Stone Cards don't) and keeps the last of equal ranks.
fn raised_fist(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Individual, AreaId::Hand)?;
    let (mut temp_mult, mut temp_id) = (15.0, 15);
    let mut raised = None;
    for c in &g.gs.areas.hand {
        let Some(base) = c.base else { continue };
        if temp_id >= base.id() && !c.is_stone() {
            temp_mult = base.nominal();
            temp_id = base.id();
            raised = Some(c.sort_id);
        }
    }
    if raised != Some(other) {
        return None;
    }
    if g.card(other).debuff {
        return Some(debuffed(id));
    }
    Some(Effect {
        h_mult: Some(lua::mul(2.0, temp_mult)),
        card: Some(id),
        ..Effect::default()
    })
}

// ---------------------------------------------------------------------------
// Repetition (card.lua:3342-3395, 2877-2887)
// ---------------------------------------------------------------------------

/// Sock and Buskin: retriggers each scored face card (card.lua:3344-3351).
fn sock_and_buskin(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Repetition, AreaId::Play)?;
    is_face(g, other).then(|| repetitions(g.extra(id), id))
}

/// Hanging Chad: retriggers the first scoring card (card.lua:3352-3359).
fn hanging_chad(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let other = other_card(ctx, Kind::Repetition, AreaId::Play)?;
    (ctx.scoring_hand.first() == Some(&other)).then(|| repetitions(g.extra(id), id))
}

/// Dusk: retriggers every scored card on the round's last hand (card.lua:3360-3366).
/// `hands_left` has already dropped for the hand being played.
fn dusk(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    other_card(ctx, Kind::Repetition, AreaId::Play)?;
    (g.gs.current_round.hands_left == 0.0).then(|| repetitions(g.extra(id), id))
}

/// Seltzer: retriggers every scored card (card.lua:3367-3373); after each hand, scored or not,
/// one fewer hand is left, and after the last it is drunk and destroyed (card.lua:3601-3630).
fn seltzer(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    match ctx.kind {
        Kind::Repetition if ctx.cardarea == Some(AreaId::Play) => Some(repetitions(1.0, id)),
        Kind::After if ctx.blueprint == 0 => {
            let left = g.extra(id);
            if lua::sub(left, 1.0) <= 0.0 {
                g.dissolve_later(id);
            } else {
                g.card_mut(id).ability.extra = Extra::Num(lua::sub(left, 1.0));
            }
            Some(message())
        }
        _ => None,
    }
}

/// Mime: retriggers each held card that had an effect, while scoring (card.lua:3386-3394) and
/// at the end of the round (card.lua:2877-2887): `next(context.card_effects[1]) or
/// #context.card_effects > 1` (G20).
fn mime(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let repetition = match ctx.kind {
        Kind::Repetition => true,
        Kind::EndOfRound => !ctx.individual && ctx.repetition,
        _ => false,
    };
    if !repetition || ctx.cardarea != Some(AreaId::Hand) {
        return None;
    }
    let has_effect = ctx.card_effects.first() == Some(&true) || ctx.card_effects.len() > 1;
    has_effect.then(|| repetitions(g.extra(id), id))
}

// ---------------------------------------------------------------------------
// Joker on joker (card.lua:3396-3408)
// ---------------------------------------------------------------------------

/// Baseball Card: XMult for each Uncommon joker, debuffed or not, as the main step reaches it
/// (card.lua:3396-3408). The juice event it queues is UI only.
fn baseball_card(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::OtherJoker {
        return None;
    }
    let other = ctx.other_joker?;
    let uncommon = g.card(other).center().rarity == Some(2);
    (uncommon && other != id).then(|| xmult_mod(g.extra(id)))
}

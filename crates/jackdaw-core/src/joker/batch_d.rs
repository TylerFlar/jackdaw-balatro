//! Joker batch D: one function per joker, all of its `calculate_joker` branches.
//!
//! The round lifecycle: `setting_blind`, `first_hand_drawn`, the discard contexts, card
//! destruction, `end_of_round` (with `game_over`) and `debuffed_hand`.
//!
//! Lua queues most of these effects with `G.E_MANAGER:add_event`; each event body is an
//! [`Event::Call`] function here, and an `add_event` inside a body is a `g.queue` from that
//! function, so the queued work keeps Lua's order (Marble Joker's Stone joins the deck before
//! the round's `'nr'` shuffle, a sliced joker leaves after the round's creations, ...).

use jackdaw_data::centers;

use super::api::{chip_mod, message, mult_mod, xmult_mod};
use super::{CalcFn, Ctx, Kind};
use crate::area::AreaId;
use crate::card::Seal;
use crate::event::Event;
use crate::game::Game;
use crate::hand_eval::{PAIRS_ORDER, get_poker_hand_info};
use crate::pools::{CardSpec, PoolType};
use crate::scoring::Effect;

/// The function of a ported joker in this batch.
pub(super) fn lookup(key: &str) -> Option<CalcFn> {
    Some(match key {
        "j_ceremonial" => ceremonial_dagger,
        "j_marble" => marble_joker,
        "j_egg" => egg,
        "j_burglar" => burglar,
        "j_dna" => dna,
        "j_sixth_sense" => sixth_sense,
        "j_faceless" => faceless_joker,
        "j_todo_list" => to_do_list,
        "j_madness" => madness,
        "j_riff_raff" => riff_raff,
        "j_rocket" => rocket,
        "j_gift" => gift_card,
        "j_turtle_bean" => turtle_bean,
        "j_mail" => mail_in_rebate,
        "j_trading" => trading_card,
        "j_ramen" => ramen,
        "j_castle" => castle,
        "j_mr_bones" => mr_bones,
        "j_certificate" => certificate,
        "j_glass" => glass_joker,
        "j_hit_the_road" => hit_the_road,
        "j_cartomancer" => cartomancer,
        "j_burnt" => burnt_joker,
        "j_caino" => caino,
        "j_yorick" => yorick,
        "j_chicot" => chicot,
        "j_matador" => matador,
        _ => return None,
    })
}

/// An entry of `playing_card_joker_effects({true})`: Certificate and Marble Joker report a
/// playing card that a later event makes (card.lua:2481, 2600). The listeners only count the
/// entries.
const PENDING_CARD: u32 = u32::MAX;

/// `context.blind.boss` at `setting_blind` (the blind prototype of `round_resets.blind`).
fn boss_blind(ctx: &Ctx) -> bool {
    ctx.blind.is_some_and(|b| b.get().boss.is_some())
}

/// The plain `end_of_round` branch: neither `individual` nor `repetition`. The dispatch has
/// already turned away Blueprint and Brainstorm there (`elseif not context.blueprint`).
fn round_end(ctx: &Ctx) -> bool {
    ctx.kind == Kind::EndOfRound && !ctx.individual && !ctx.repetition
}

impl Game {
    /// `(context.blueprint_card or self).getting_sliced`.
    fn source_sliced(&self, id: u32, ctx: &Ctx) -> bool {
        self.gs
            .areas
            .card(ctx.source(id))
            .is_some_and(|c| c.getting_sliced)
    }

    /// `G.GAME.blind.triggered`.
    fn blind_triggered(&self) -> bool {
        self.gs.blind.as_ref().is_some_and(|b| b.triggered)
    }

    /// `ease_dollars(x)` and `G.GAME.dollar_buffer` for money paid during a hand (To Do List,
    /// Matador): the money, then the event that clears the buffer.
    fn pay_now(&mut self, x: f64) {
        self.ease_dollars(x);
        self.buffer_dollars(x);
    }
}

// ---------------------------------------------------------------------------
// setting_blind
// ---------------------------------------------------------------------------

/// Ceremonial Dagger (card.lua:2561-2579, 3737-3742): at `setting_blind` it slices the joker to
/// its right (not Eternal, not already sliced) and, in an event, gains twice its sell value as
/// Mult; +Mult in the main step.
fn ceremonial_dagger(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    match ctx.kind {
        Kind::SettingBlind if ctx.blueprint == 0 => {
            let jokers = g.joker_ids();
            let pos = jokers.iter().position(|&j| j == id)?;
            let &next = jokers.get(pos + 1)?;
            let n = g.card(next);
            if !g.card(id).getting_sliced && !n.ability.eternal && !n.getting_sliced {
                g.card_mut(next).getting_sliced = true;
                g.rt.joker_buffer -= 1.0;
                g.queue(Event::Call(dagger_slice, id, f64::from(next)));
            }
            None
        }
        Kind::JokerMain => {
            let m = g.card(id).ability.mult;
            (m > 0.0).then(|| mult_mod(m))
        }
        _ => None,
    }
}

/// Ceremonial Dagger's event: `joker_buffer = 0`, `mult += 2 * sell_cost` of the sliced joker,
/// then its `start_dissolve` (which queues its removal).
fn dagger_slice(g: &mut Game, id: u32, sliced: f64) {
    let sliced = sliced as u32;
    g.rt.joker_buffer = 0.0;
    let sell = g.gs.areas.card(sliced).map_or(0.0, |c| c.sell_cost);
    if let Some(c) = g.gs.areas.card_mut(id) {
        c.ability.mult += sell * 2.0;
    }
    g.queue(Event::RemoveCard(sliced));
}

/// Madness (card.lua:2503-2521): outside boss blinds it gains `extra` XMult and slices a random
/// other joker (not Eternal, not already sliced; `'madness'`). The XMult scores through the
/// main step's shared rule.
fn madness(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::SettingBlind || ctx.blueprint > 0 || boss_blind(ctx) {
        return None;
    }
    let extra = g.extra(id);
    g.card_mut(id).ability.x_mult += extra;
    let mut pool: Vec<u32> = g
        .joker_ids()
        .into_iter()
        .filter(|&j| {
            let c = g.card(j);
            j != id && !c.ability.eternal && !c.getting_sliced
        })
        .collect();
    if pool.is_empty() {
        return None;
    }
    // pseudorandom_element sorts cards by sort_id.
    pool.sort_unstable();
    let seed = g.gs.rng.seed("madness");
    let victim = pool[crate::rng::element_index(pool.len(), seed)];
    if !g.source_sliced(id, ctx) {
        g.card_mut(victim).getting_sliced = true;
        // An event that calls start_dissolve, which queues the removal.
        g.dissolve_later(victim);
    }
    None
}

/// Burglar (card.lua:2522-2528): an event takes every discard and adds `extra` hands, both
/// through queued eases.
fn burglar(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind == Kind::SettingBlind && !g.source_sliced(id, ctx) {
        g.queue(Event::Call(burglar_event, id, 0.0));
    }
    None
}

/// Burglar's event: `ease_discard(-discards_left)` and `ease_hands_played(extra)`, each an
/// event of its own.
fn burglar_event(g: &mut Game, id: u32, _: f64) {
    let left = g.gs.current_round.discards_left;
    g.queue(Event::EaseDiscard(-left));
    let hands =
        g.gs.areas
            .card(id)
            .and_then(|c| c.ability.extra.num())
            .unwrap_or(0.0);
    g.queue(Event::EaseHandsPlayed(hands));
}

/// Riff-raff (card.lua:2529-2544): with room (counting `G.GAME.joker_buffer`), reserves up to
/// two slots and creates that many Common jokers (`'rif'`) in an event.
fn riff_raff(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind == Kind::SettingBlind && !g.source_sliced(id, ctx) && g.joker_room() {
        let held = g.gs.areas.jokers.len() as f64 + g.rt.joker_buffer;
        let n = 2.0f64.min(g.gs.joker_slots - held);
        g.rt.joker_buffer += n;
        g.queue(Event::Call(riff_raff_event, id, n));
    }
    None
}

/// Riff-raff's event: each joker is created (rarity roll fixed at 0: Common), added to the deck
/// and emplaced, and the joker buffer is cleared.
fn riff_raff_event(g: &mut Game, _: u32, n: f64) {
    // `for i = 1, jokers_to_create do`
    let mut i = 1.0;
    while i <= n {
        let card = g.create_card(CardSpec::new(PoolType::Joker, AreaId::Jokers, "rif").rarity(0.0));
        let cid = card.sort_id;
        g.gs.areas.emplace(AreaId::Jokers, card, false, false);
        g.add_to_deck(cid, false);
        g.rt.joker_buffer = 0.0;
        i += 1.0;
    }
}

/// Cartomancer (card.lua:2545-2559): with consumable room it reserves a slot; an event queues
/// the event that creates the Tarot (`'car'`).
fn cartomancer(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind == Kind::SettingBlind && !g.source_sliced(id, ctx) && g.consumable_room() {
        g.rt.consumeable_buffer += 1.0;
        g.queue(Event::Call(cartomancer_event, id, 0.0));
    }
    None
}

/// Cartomancer's outer event.
fn cartomancer_event(g: &mut Game, _: u32, _: f64) {
    g.queue(Event::CreateConsumable {
        ty: PoolType::Tarot,
        append: "car",
        forced: None,
    });
}

/// Marble Joker (card.lua:2580-2601): an event makes a Stone Card (`'marb_fr'` front) in
/// `G.play`, `draw_card` moves it to the deck, and the playing-card listeners hear of it at
/// once. All of it is queued ahead of `new_round`'s shuffle event, so the Stone is shuffled
/// into the deck.
fn marble_joker(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind == Kind::SettingBlind && !g.source_sliced(id, ctx) {
        g.queue(Event::Call(marble_create, id, 0.0));
        // `G.deck.config.card_limit + 1` (card.lua:2593-2597); only BalatroBot's view reads it.
        g.queue(Event::Call(marble_deck_limit, id, 0.0));
        g.queue(Event::DrawCard {
            from: AreaId::Play,
            to: AreaId::Deck,
            card: None,
            sort: false,
        });
        g.playing_card_joker_effects(&[PENDING_CARD]);
    }
    None
}

/// Marble Joker's second event: the deck's card limit grows by one.
fn marble_deck_limit(g: &mut Game, _: u32, _: f64) {
    g.rt.areas.deck_limit += 1.0;
}

/// Marble Joker's event: the Stone Card joins `G.play` and `G.playing_cards`.
fn marble_create(g: &mut Game, _: u32, _: f64) {
    let front = g.random_front("marb_fr");
    g.create_playing_card(Some(front), Some(centers::M_STONE), AreaId::Play);
}

/// Chicot (card.lua:596-600 in `add_to_deck`, 2492-2502): at a boss's `setting_blind` an event
/// queues the event that disables the blind.
fn chicot(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind == Kind::SettingBlind
        && ctx.blueprint == 0
        && boss_blind(ctx)
        && !g.card(id).getting_sliced
    {
        g.queue(Event::Call(chicot_event, id, 0.0));
    }
    None
}

/// Chicot's outer event.
fn chicot_event(g: &mut Game, id: u32, _: f64) {
    g.queue(Event::Call(chicot_disable, id, 0.0));
}

/// Chicot's inner event: `G.GAME.blind:disable()`.
fn chicot_disable(g: &mut Game, _: u32, _: f64) {
    g.blind_disable();
}

// ---------------------------------------------------------------------------
// first_hand_drawn
// ---------------------------------------------------------------------------

/// Certificate (card.lua:2462-2482): an event deals a random card (`'cert_fr'`) with a random
/// seal (`'certsl'`) into the hand; the playing-card listeners hear of it at once.
fn certificate(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind == Kind::FirstHandDrawn {
        g.queue(Event::Call(certificate_create, id, 0.0));
        g.playing_card_joker_effects(&[PENDING_CARD]);
    }
    None
}

/// Certificate's event: `create_playing_card` into `G.hand`, the seal, the boss's
/// `debuff_card`, then `G.hand:sort()`.
fn certificate_create(g: &mut Game, _: u32, _: f64) {
    let front = g.random_front("cert_fr");
    let cid = g.create_playing_card(Some(front), Some(centers::C_BASE), AreaId::Hand);
    let roll = g.gs.rng.random("certsl");
    let seal = if roll > 0.75 {
        Seal::Red
    } else if roll > 0.5 {
        Seal::Blue
    } else if roll > 0.25 {
        Seal::Gold
    } else {
        Seal::Purple
    };
    let cost = g.cost_context(AreaId::Hand);
    g.card_mut(cid).set_seal(Some(seal), &cost);
    g.debuff_card(cid, false);
    g.sort_hand();
}

// ---------------------------------------------------------------------------
// The played hand
// ---------------------------------------------------------------------------

/// DNA (card.lua:2483-2486, 3501-3523): if the round's first hand is a single card, a copy of
/// it joins the hand and the deck (`playing_cards_created`, which `card_eval_status_text`
/// reports to the playing-card listeners, common_events.lua:922-924). Its `first_hand_drawn`
/// branch only animates.
fn dna(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::Before
        || g.gs.current_round.hands_played != 0.0
        || ctx.full_hand.len() != 1
    {
        return None;
    }
    let copy = g.copy_card(ctx.full_hand[0], false, AreaId::Hand);
    let cid = copy.sort_id;
    g.gs.areas.emplace(AreaId::Hand, copy, false, false);
    // `_card:add_to_deck()` does nothing more for a playing card.
    g.add_to_deck(cid, false);
    // `G.deck.config.card_limit + 1` (card.lua:3506); only BalatroBot's view reads it.
    g.rt.areas.deck_limit += 1.0;
    g.playing_card_joker_effects(&[cid]);
    Some(Effect {
        card: Some(id),
        message: true,
        ..Effect::default()
    })
}

/// To Do List (card.lua:2975-2984, 3491-3500): pays when the played hand is its hand, in
/// `before` (with the dollar buffer); a new hand at the end of the round (`'to_do'`, any
/// visible hand but the current one, in `pairs(G.GAME.hands)` order).
fn to_do_list(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    match ctx.kind {
        Kind::Before => {
            if ctx.scoring_name != g.card(id).ability.to_do_poker_hand {
                return None;
            }
            let dollars = g.extra_num(id, "dollars");
            g.pay_now(dollars);
            Some(Effect {
                dollars: Some(dollars),
                message: true,
                ..Effect::default()
            })
        }
        _ if round_end(ctx) => {
            let current = g.card(id).ability.to_do_poker_hand;
            let hands: Vec<_> = PAIRS_ORDER
                .into_iter()
                .filter(|&h| g.gs.hand(h).visible && Some(h) != current)
                .collect();
            let seed = g.gs.rng.seed("to_do");
            let pick = hands[crate::rng::element_index(hands.len(), seed)];
            g.card_mut(id).ability.to_do_poker_hand = Some(pick);
            Some(message())
        }
        _ => None,
    }
}

/// Sixth Sense (card.lua:2603-2621): the round's first hand, a single 6, is destroyed; with
/// consumable room a Spectral (`'sixth'`) is made by an event.
fn sixth_sense(g: &mut Game, _: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::DestroyingCard || ctx.blueprint > 0 {
        return None;
    }
    let six = ctx.full_hand.len() == 1
        && g.card(ctx.full_hand[0]).get_id() == Some(6)
        && g.gs.current_round.hands_played == 0.0;
    if !six {
        return None;
    }
    if g.consumable_room() {
        g.create_consumable_later(PoolType::Spectral, "sixth", None);
    }
    // `return true`
    Some(Effect::default())
}

/// Matador (card.lua:2735-2747, 3719-3730): pays when the boss blind triggered this hand,
/// whether the hand is debuffed or scored.
fn matador(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if !matches!(ctx.kind, Kind::DebuffedHand | Kind::JokerMain) || !g.blind_triggered() {
        return None;
    }
    let dollars = g.extra(id);
    g.pay_now(dollars);
    Some(Effect {
        dollars: Some(dollars),
        message: true,
        ..Effect::default()
    })
}

/// Glass Joker (card.lua:2647-2670, 2687-2721): +`extra` XMult per shattered Glass Card, added
/// by a nested event; scores through the main step's shared rule.
fn glass_joker(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.blueprint > 0 {
        return None;
    }
    match ctx.kind {
        // `cards_destroyed` (`glass_shattered`) has no caller in 1.0.1o; `remove_playing_cards`
        // is the live branch. Both count the shattered cards.
        Kind::CardsDestroyed | Kind::RemovePlayingCards => {
            let glasses = ctx
                .removed
                .iter()
                .filter(|&&c| g.gs.areas.card(c).is_some_and(|c| c.shattered))
                .count();
            if glasses > 0 {
                g.queue(Event::Call(glass_event, id, glasses as f64));
            }
            None
        }
        Kind::UsingConsumeable => {
            // The Hanged Man: the highlighted Glass Cards it is about to destroy; its own events
            // run after the jokers see the use (card.lua:2709-2720). The used card waits in the
            // play area.
            let hanged = ctx
                .consumeable
                .and_then(|c| g.gs.areas.card(c))
                .is_some_and(|c| c.center == jackdaw_data::centers::C_HANGED_MAN);
            if hanged {
                let glasses =
                    g.rt.highlighted
                        .iter()
                        .filter(|&&h| {
                            g.gs.areas
                                .card(h)
                                .is_some_and(|c| c.center == jackdaw_data::centers::M_GLASS)
                        })
                        .count();
                if glasses > 0
                    && let Some(c) = g.gs.areas.card_mut(id)
                {
                    let extra = c.ability.extra.num().unwrap_or(0.0);
                    c.ability.x_mult += extra * glasses as f64;
                }
            }
            None
        }
        _ => None,
    }
}

/// Glass Joker's outer event.
fn glass_event(g: &mut Game, id: u32, glasses: f64) {
    g.queue(Event::Call(glass_grow, id, glasses));
}

/// Glass Joker's inner event: `x_mult += extra * glasses`.
fn glass_grow(g: &mut Game, id: u32, glasses: f64) {
    if let Some(c) = g.gs.areas.card_mut(id) {
        let extra = c.ability.extra.num().unwrap_or(0.0);
        c.ability.x_mult += extra * glasses;
    }
}

/// Caino (card.lua:2623-2646, 2673-2685, 4052-4057): +`extra` XMult (`caino_xmult`) per face
/// card destroyed; XMult in the main step.
fn caino(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let faces = |g: &Game, cards: &[u32]| {
        let pareidolia = g.has_joker("Pareidolia");
        cards
            .iter()
            .filter(|&&c| {
                g.gs.areas
                    .card(c)
                    .is_some_and(|c| c.is_face(false, pareidolia))
            })
            .count() as f64
    };
    match ctx.kind {
        // No caller in 1.0.1o: the increase waits for a nested event.
        Kind::CardsDestroyed if ctx.blueprint == 0 => {
            let n = faces(g, ctx.removed);
            if n > 0.0 {
                g.queue(Event::Call(caino_event, id, n));
            }
            None
        }
        Kind::RemovePlayingCards if ctx.blueprint == 0 => {
            let n = faces(g, ctx.removed);
            if n > 0.0 {
                let extra = g.extra(id);
                let c = g.card_mut(id);
                c.ability.caino_xmult = Some(c.ability.caino_xmult.unwrap_or(1.0) + n * extra);
            }
            None
        }
        Kind::JokerMain => {
            let x = g.card(id).ability.caino_xmult.unwrap_or(1.0);
            (x > 1.0).then(|| xmult_mod(x))
        }
        _ => None,
    }
}

/// Caino's outer `cards_destroyed` event.
fn caino_event(g: &mut Game, id: u32, faces: f64) {
    g.queue(Event::Call(caino_grow, id, faces));
}

/// Caino's inner `cards_destroyed` event.
fn caino_grow(g: &mut Game, id: u32, faces: f64) {
    if let Some(c) = g.gs.areas.card_mut(id) {
        let extra = c.ability.extra.num().unwrap_or(0.0);
        c.ability.caino_xmult = Some(c.ability.caino_xmult.unwrap_or(1.0) + faces * extra);
    }
}

// ---------------------------------------------------------------------------
// Discards
// ---------------------------------------------------------------------------

/// Burnt Joker (card.lua:2748-2755): the round's first discard (not The Hook's) levels up its
/// poker hand.
fn burnt_joker(g: &mut Game, _: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::PreDiscard || ctx.hook || g.gs.current_round.discards_used > 0.0 {
        return None;
    }
    let cards: Vec<_> = ctx
        .full_hand
        .iter()
        .filter_map(|&c| g.gs.areas.card(c).cloned())
        .collect();
    let info = get_poker_hand_info(&g.eval_cards(&cards), g.eval_flags());
    if let Some(hand) = info.hand {
        g.level_up_hand(hand, 1.0);
    }
    None
}

/// Faceless Joker (card.lua:2858-2872): on the last card of a discard with at least `faces`
/// face cards, an event pays `dollars`.
fn faceless_joker(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::Discard || ctx.other_card != ctx.full_hand.last().copied() {
        return None;
    }
    let pareidolia = g.has_joker("Pareidolia");
    let faces = ctx
        .full_hand
        .iter()
        .filter(|&&c| {
            g.gs.areas
                .card(c)
                .is_some_and(|c| c.is_face(false, pareidolia))
        })
        .count() as f64;
    if faces >= g.extra_num(id, "faces") {
        let dollars = g.extra_num(id, "dollars");
        g.queue(Event::Call(faceless_pay, id, dollars));
    }
    None
}

/// Faceless Joker's event: `ease_dollars`, itself an event.
fn faceless_pay(g: &mut Game, _: u32, dollars: f64) {
    g.ease_dollars(dollars);
}

/// Mail-In Rebate (card.lua:2825-2834): pays for each discarded card of the round's rank.
fn mail_in_rebate(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::Discard {
        return None;
    }
    let other = g.card(ctx.other_card?);
    if other.debuff || other.get_id() != Some(g.gs.current_round.mail_card.id()) {
        return None;
    }
    g.ease_dollars(g.extra(id));
    Some(Effect {
        card: Some(id),
        message: true,
        ..Effect::default()
    })
}

/// Trading Card (card.lua:2487-2490, 2802-2812): a first discard of a single card pays and
/// destroys the card. Its `first_hand_drawn` branch only animates.
fn trading_card(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::Discard
        || ctx.blueprint > 0
        || g.gs.current_round.discards_used > 0.0
        || ctx.full_hand.len() != 1
    {
        return None;
    }
    g.ease_dollars(g.extra(id));
    Some(Effect {
        remove: true,
        card: Some(id),
        message: true,
        ..Effect::default()
    })
}

/// Castle (card.lua:2814-2824, 3880-3886): +`chip_mod` Chips per discarded card of the round's
/// suit; +Chips in the main step.
fn castle(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    match ctx.kind {
        Kind::Discard if ctx.blueprint == 0 => {
            let suit = g.gs.current_round.castle_card;
            let smeared = g.has_joker("Smeared Joker");
            let other = g.card(ctx.other_card?);
            if other.debuff || other.is_suit(suit, false, false, smeared) != Some(true) {
                return None;
            }
            let chips = g.extra_num(id, "chips") + g.extra_num(id, "chip_mod");
            g.set_extra_num(id, "chips", chips);
            Some(Effect {
                card: Some(id),
                message: true,
                ..Effect::default()
            })
        }
        Kind::JokerMain => {
            let chips = g.extra_num(id, "chips");
            (chips > 0.0).then(|| chip_mod(chips))
        }
        _ => None,
    }
}

/// Hit the Road (card.lua:2835-2845, 3011-3017): +`extra` XMult per discarded Jack, back to 1
/// at the end of the round; scores through the main step's shared rule.
fn hit_the_road(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    match ctx.kind {
        Kind::Discard if ctx.blueprint == 0 => {
            let other = g.card(ctx.other_card?);
            if other.debuff || other.get_id() != Some(11) {
                return None;
            }
            let extra = g.extra(id);
            g.card_mut(id).ability.x_mult += extra;
            Some(Effect {
                card: Some(id),
                message: true,
                ..Effect::default()
            })
        }
        _ if round_end(ctx) && g.card(id).ability.x_mult > 1.0 => {
            g.card_mut(id).ability.x_mult = 1.0;
            Some(message())
        }
        _ => None,
    }
}

/// Ramen (card.lua:2757-2787): loses `extra` XMult per discarded card; eaten (an event, then
/// its removal event) once that would bring it to 1 or less.
fn ramen(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::Discard || ctx.blueprint > 0 {
        return None;
    }
    let extra = g.extra(id);
    let x = g.card(id).ability.x_mult;
    if x - extra <= 1.0 {
        g.dissolve_later(id);
    } else {
        g.card_mut(id).ability.x_mult = x - extra;
    }
    Some(message())
}

/// Yorick (card.lua:2788-2801): every `discards` discarded cards, +`xmult` XMult; scores through
/// the main step's shared rule.
fn yorick(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::Discard || ctx.blueprint > 0 {
        return None;
    }
    let left = g.card(id).ability.yorick_discards.unwrap_or(0.0);
    if left <= 1.0 {
        let discards = g.extra_num(id, "discards");
        let xmult = g.extra_num(id, "xmult");
        let c = g.card_mut(id);
        c.ability.yorick_discards = Some(discards);
        c.ability.x_mult += xmult;
        return Some(message());
    }
    g.card_mut(id).ability.yorick_discards = Some(left - 1.0);
    None
}

// ---------------------------------------------------------------------------
// end_of_round
// ---------------------------------------------------------------------------

/// Egg (card.lua:2985-2992): +`extra` sell value each round.
fn egg(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if !round_end(ctx) {
        return None;
    }
    let extra = g.extra(id);
    let area = g.gs.areas.locate(id).map_or(AreaId::Jokers, |l| l.0);
    let cost = g.cost_context(area);
    let c = g.card_mut(id);
    c.ability.extra_value += extra;
    c.set_cost(&cost);
    Some(message())
}

/// Gift Card (card.lua:2993-3010): +`extra` sell value to every joker and consumable each
/// round.
fn gift_card(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if !round_end(ctx) {
        return None;
    }
    let extra = g.extra(id);
    for area in [AreaId::Jokers, AreaId::Consumables] {
        let cost = g.cost_context(area);
        for c in g.gs.areas.get_mut(area) {
            c.ability.extra_value += extra;
            c.set_cost(&cost);
        }
    }
    Some(message())
}

/// Rocket (card.lua:1664-1666, 2896-2902): pays `dollars` at cash-out (`calculate_dollar_bonus`,
/// round.rs), `increase` more after each boss.
fn rocket(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if !round_end(ctx) || !g.gs.blind.as_ref().is_some_and(|b| b.boss) {
        return None;
    }
    let dollars = g.extra_num(id, "dollars") + g.extra_num(id, "increase");
    g.set_extra_num(id, "dollars", dollars);
    Some(message())
}

/// Turtle Bean (card.lua:605-607, 662-664, 2903-2933): +`h_size` hand size while held, one less
/// each round; eaten (an event, then its removal event) when it would reach 0.
fn turtle_bean(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if !round_end(ctx) {
        return None;
    }
    let h_size = g.extra_num(id, "h_size");
    let h_mod = g.extra_num(id, "h_mod");
    if h_size - h_mod <= 0.0 {
        g.dissolve_later(id);
    } else {
        g.set_extra_num(id, "h_size", h_size - h_mod);
        g.change_size(AreaId::Hand, -h_mod);
    }
    Some(message())
}

/// Mr. Bones (card.lua:3047-3063): a lost round with at least a quarter of the blind's chips is
/// saved, and Mr. Bones dissolves (an event that calls `start_dissolve`).
fn mr_bones(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if !round_end(ctx) || !ctx.game_over {
        return None;
    }
    let blind_chips = g.gs.blind.as_ref().map_or(0.0, |b| b.chips);
    // `G.GAME.chips/G.GAME.blind.chips >= 0.25`: false for a NaN ratio.
    let ratio = crate::lua::div(g.gs.chips, blind_chips);
    if ratio >= 0.25 {
        g.dissolve_later(id);
        return Some(Effect {
            saved: true,
            message: true,
            ..Effect::default()
        });
    }
    None
}

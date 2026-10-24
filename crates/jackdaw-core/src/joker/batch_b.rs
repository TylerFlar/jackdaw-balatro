//! Joker batch B: one function per joker, all of its `calculate_joker` branches. These jokers
//! act in the main step and the `before` and `after` passes of `evaluate_play`, plus Green
//! Joker's discard and Popcorn's end of round.
//!
//! The main step's shared rules (mod.rs) run before these functions: a joker whose Xmult lives
//! in `ability.x_mult` (Vampire, Obelisk, Joker Stencil) scores through them while it is above
//! 1, so its function only has its own branches.

use jackdaw_data::centers;

use super::api::{chip_mod, message, mult_mod, xmult_mod};
use super::{CalcFn, Ctx, Kind};
use crate::card::Suit;
use crate::event::Event;
use crate::game::Game;
use crate::hand_eval::HandType;
use crate::lua;
use crate::pools::PoolType;
use crate::scoring::Effect;

/// The function of a ported joker in this batch.
pub(super) fn lookup(key: &str) -> Option<CalcFn> {
    Some(match key {
        "j_half" => half,
        "j_stencil" => stencil,
        "j_banner" => banner,
        "j_mystic_summit" => mystic_summit,
        "j_misprint" => misprint,
        "j_steel_joker" => steel_joker,
        "j_abstract" => abstract_joker,
        "j_supernova" => supernova,
        "j_ride_the_bus" => ride_the_bus,
        "j_space" => space,
        "j_runner" => runner,
        "j_ice_cream" => ice_cream,
        "j_blue_joker" => blue_joker,
        "j_card_sharp" => card_sharp,
        "j_square" => square,
        "j_erosion" => erosion,
        "j_stone" => stone_joker,
        "j_bull" => bull,
        "j_popcorn" => popcorn,
        "j_trousers" => spare_trousers,
        "j_acrobat" => acrobat,
        "j_swashbuckler" => swashbuckler,
        "j_flower_pot" => flower_pot,
        "j_seeing_double" => seeing_double,
        "j_stuntman" => stuntman,
        "j_drivers_license" => drivers_license,
        "j_bootstraps" => bootstraps,
        "j_blackboard" => blackboard,
        "j_green_joker" => green_joker,
        "j_superposition" => superposition,
        "j_seance" => seance,
        "j_vagabond" => vagabond,
        "j_obelisk" => obelisk,
        "j_midas_mask" => midas_mask,
        "j_vampire" => vampire,
        _ => return None,
    })
}

/// `{message = ..., card = self}`: a non-empty table crediting the joker.
fn noted(id: u32) -> Effect {
    Effect {
        card: Some(id),
        message: true,
        ..Effect::default()
    }
}

/// `G.GAME.dollars + (G.GAME.dollar_buffer or 0)`: money held plus money earned earlier in the
/// hand being scored.
fn dollars_with_buffer(g: &Game) -> f64 {
    lua::add(g.gs.dollars, g.rt.dollar_buffer)
}

// ---------------------------------------------------------------------------
// Main step only
// ---------------------------------------------------------------------------

/// Half Joker: +Mult when the played hand has at most `size` cards (card.lua:3672-3677).
fn half(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    (ctx.kind == Kind::JokerMain && (ctx.full_hand.len() as f64) <= g.extra_num(id, "size"))
        .then(|| mult_mod(g.extra_num(id, "mult")))
}

/// Joker Stencil: `Card:update` keeps `x_mult` at the empty joker slots plus one per Stencil
/// (card.lua:4203-4208), and the shared `x_mult > 1` rule scores it. Its own branch returns
/// the same `x_mult` while a slot is empty (card.lua:3966-3972); it can only run when the
/// shared rule did not, so it never fires with a fresh `x_mult`.
fn stencil(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let empty = g.gs.joker_slots - g.gs.areas.jokers.len() as f64;
    (ctx.kind == Kind::JokerMain && empty > 0.0).then(|| xmult_mod(g.card(id).ability.x_mult))
}

/// Banner: +Chips per discard left (card.lua:3707-3712).
fn banner(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let left = g.gs.current_round.discards_left;
    (ctx.kind == Kind::JokerMain && left > 0.0).then(|| chip_mod(left * g.extra(id)))
}

/// Mystic Summit: +Mult with `d_remaining` discards left (card.lua:3694-3699).
fn mystic_summit(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    (ctx.kind == Kind::JokerMain
        && g.gs.current_round.discards_left == g.extra_num(id, "d_remaining"))
    .then(|| mult_mod(g.extra_num(id, "mult")))
}

/// Misprint: +Mult drawn from `pseudorandom('misprint', min, max)` (card.lua:3700-3706).
fn misprint(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::JokerMain {
        return None;
    }
    let (min, max) = (g.extra_num(id, "min"), g.extra_num(id, "max"));
    let roll = g.gs.rng.random_range("misprint", min as i64, max as i64);
    Some(mult_mod(roll as f64))
}

/// Steel Joker: X(1 + `extra` per Steel Card in the deck), from the per-frame `steel_tally`
/// (card.lua:3929-3935, 4185-4190).
fn steel_joker(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let tally = g.card(id).ability.tally.unwrap_or(0.0);
    (ctx.kind == Kind::JokerMain && tally > 0.0).then(|| xmult_mod(1.0 + g.extra(id) * tally))
}

/// Abstract Joker: +Mult per card of the Joker set in `G.jokers` (card.lua:3678-3687).
fn abstract_joker(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    (ctx.kind == Kind::JokerMain).then(|| mult_mod(g.joker_count() * g.extra(id)))
}

/// Supernova: +Mult equal to the times the scored hand was played this run, this play
/// included (card.lua:3731-3736; `evaluate_play` counts it first, state_events.lua:574).
fn supernova(g: &mut Game, _id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let hand = ctx.scoring_name?;
    (ctx.kind == Kind::JokerMain).then(|| mult_mod(g.gs.hand(hand).played))
}

/// Blue Joker: +Chips per card left in the deck (card.lua:3887-3893).
fn blue_joker(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let deck = g.gs.areas.deck.len() as f64;
    (ctx.kind == Kind::JokerMain && deck > 0.0).then(|| chip_mod(g.extra(id) * deck))
}

/// Card Sharp: XMult when the scored hand was already played this round (card.lua:4040-4045).
fn card_sharp(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let hand = ctx.scoring_name?;
    (ctx.kind == Kind::JokerMain && g.gs.hand(hand).played_this_round > 1.0)
        .then(|| xmult_mod(g.extra_num(id, "Xmult")))
}

/// Erosion: +Mult per card below the starting deck size (card.lua:3894-3900).
fn erosion(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let missing = g.gs.starting_deck_size - g.playing_card_count() as f64;
    (ctx.kind == Kind::JokerMain && missing > 0.0).then(|| mult_mod(g.extra(id) * missing))
}

/// Stone Joker: +Chips per Stone Card in the deck, from the per-frame `stone_tally`
/// (card.lua:3922-3928, 4197-4202).
fn stone_joker(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let tally = g.card(id).ability.tally.unwrap_or(0.0);
    (ctx.kind == Kind::JokerMain && tally > 0.0).then(|| chip_mod(g.extra(id) * tally))
}

/// Bull: +Chips per dollar held, counting money earned earlier in the hand
/// (card.lua:3936-3942).
fn bull(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let money = dollars_with_buffer(g);
    (ctx.kind == Kind::JokerMain && money > 0.0).then(|| chip_mod(g.extra(id) * money.max(0.0)))
}

/// Acrobat: XMult on the round's last hand (card.lua:3688-3693).
fn acrobat(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    (ctx.kind == Kind::JokerMain && g.gs.current_round.hands_left == 0.0)
        .then(|| xmult_mod(g.extra(id)))
}

/// Swashbuckler: +Mult from the per-frame `mult`, the other jokers' sell value
/// (card.lua:3974-3979, 4240-4248).
fn swashbuckler(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let m = g.card(id).ability.mult;
    (ctx.kind == Kind::JokerMain && m > 0.0).then(|| mult_mod(m))
}

/// Stuntman: +Chips (card.lua:3713-3718); its hand size is in `add_to_deck`.
fn stuntman(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    (ctx.kind == Kind::JokerMain).then(|| chip_mod(g.extra_num(id, "chip_mod")))
}

/// Driver's License: XMult with at least 16 enhanced cards in the deck, from the per-frame
/// `driver_tally` (card.lua:3943-3950, 4179-4184).
fn drivers_license(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    let tally = g.card(id).ability.tally.unwrap_or(0.0);
    (ctx.kind == Kind::JokerMain && tally >= 16.0).then(|| xmult_mod(g.extra(id)))
}

/// Bootstraps: +Mult per `dollars` held, counting money earned earlier in the hand
/// (card.lua:4046-4051).
fn bootstraps(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::JokerMain {
        return None;
    }
    let steps = lua::floor(lua::div(dollars_with_buffer(g), g.extra_num(id, "dollars")));
    (steps >= 1.0).then(|| mult_mod(g.extra_num(id, "mult") * steps))
}

/// Blackboard: XMult when every held card is a Spade or a Club, with `is_suit`'s flush rules
/// (card.lua:3951-3965). An empty hand qualifies.
fn blackboard(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::JokerMain {
        return None;
    }
    let smeared = g.has_joker("Smeared Joker");
    let all_black = g.gs.areas.hand.iter().all(|c| {
        c.is_suit(Suit::Clubs, false, true, smeared) == Some(true)
            || c.is_suit(Suit::Spades, false, true, smeared) == Some(true)
    });
    all_black.then(|| xmult_mod(g.extra(id)))
}

/// Flower Pot: XMult when the scoring hand has all four suits (card.lua:3807-3839). Each
/// non-Wild card fills the first of Hearts, Diamonds, Spades, Clubs it belongs to (debuffed
/// cards too) that is still empty; then each Wild Card fills one more, unless debuffed.
fn flower_pot(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::JokerMain {
        return None;
    }
    const ORDER: [Suit; 4] = [Suit::Hearts, Suit::Diamonds, Suit::Spades, Suit::Clubs];
    let smeared = g.has_joker("Smeared Joker");
    let mut found = [false; 4];
    for wild_pass in [false, true] {
        for &c in ctx.scoring_hand {
            let card = g.card(c);
            if (card.name() == "Wild Card") != wild_pass {
                continue;
            }
            // The non-Wild pass bypasses the debuff (`is_suit(suit, true)`).
            let hit = (0..4).find(|&i| {
                card.is_suit(ORDER[i], !wild_pass, false, smeared) == Some(true) && !found[i]
            });
            if let Some(i) = hit {
                found[i] = true;
            }
        }
    }
    found.iter().all(|&f| f).then(|| xmult_mod(g.extra(id)))
}

/// Seeing Double: XMult when the scoring hand has a Club and a card of another suit
/// (card.lua:3840-3872). Non-Wild cards count every suit they belong to; then each Wild Card
/// fills one empty suit, Clubs first. Debuffed cards belong to no suit.
fn seeing_double(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::JokerMain {
        return None;
    }
    let smeared = g.has_joker("Smeared Joker");
    // Hearts, Diamonds, Spades, Clubs.
    let mut n = [0u32; 4];
    let order = [Suit::Hearts, Suit::Diamonds, Suit::Spades, Suit::Clubs];
    for &c in ctx.scoring_hand {
        let card = g.card(c);
        if card.name() != "Wild Card" {
            for (i, &s) in order.iter().enumerate() {
                if card.is_suit(s, false, false, smeared) == Some(true) {
                    n[i] += 1;
                }
            }
        }
    }
    for &c in ctx.scoring_hand {
        let card = g.card(c);
        if card.name() == "Wild Card" {
            // Clubs, Diamonds, Spades, Hearts.
            for i in [3, 1, 2, 0] {
                if card.is_suit(order[i], false, false, smeared) == Some(true) && n[i] == 0 {
                    n[i] += 1;
                    break;
                }
            }
        }
    }
    ((n[0] > 0 || n[1] > 0 || n[2] > 0) && n[3] > 0).then(|| xmult_mod(g.extra(id)))
}

// ---------------------------------------------------------------------------
// Consumable creators (main step)
// ---------------------------------------------------------------------------

/// Superposition: a Tarot when the scoring hand has an Ace and the hand contains a Straight,
/// with room (card.lua:3762-3786).
fn superposition(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::JokerMain || !g.consumable_room() {
        return None;
    }
    let aces = ctx
        .scoring_hand
        .iter()
        .filter(|&&c| g.card(c).get_id() == Some(14))
        .count();
    if aces >= 1 && ctx.has_hand(HandType::Straight) {
        g.create_consumable_later(PoolType::Tarot, "sup", None);
        return Some(noted(id));
    }
    None
}

/// Seance: a Spectral when the hand contains `poker_hand` (a Straight Flush), with room
/// (card.lua:3787-3806).
fn seance(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::JokerMain || !g.consumable_room() {
        return None;
    }
    let hand = HandType::from_name(g.extra_str(id, "poker_hand")).expect("Seance's hand");
    if ctx.has_hand(hand) {
        g.create_consumable_later(PoolType::Spectral, "sea", None);
        return Some(noted(id));
    }
    None
}

/// Vagabond: a Tarot when the hand is played with `extra` dollars or less (no dollar buffer),
/// with room (card.lua:3743-3761).
fn vagabond(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::JokerMain || !g.consumable_room() {
        return None;
    }
    if g.gs.dollars <= g.extra(id) {
        g.create_consumable_later(PoolType::Tarot, "vag", None);
        return Some(noted(id));
    }
    None
}

// ---------------------------------------------------------------------------
// Scaling: `before` (and discard / end of round), then the main step
// ---------------------------------------------------------------------------

/// Ride the Bus: before each hand, +`extra` Mult without a scoring face card, back to 0 with
/// one (card.lua:3525-3542); +Mult in the main step (3992-3997).
fn ride_the_bus(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    match ctx.kind {
        Kind::Before if ctx.blueprint == 0 => {
            let pareidolia = g.has_joker("Pareidolia");
            let faces = ctx
                .scoring_hand
                .iter()
                .any(|&c| g.card(c).is_face(false, pareidolia));
            if faces {
                let last = g.card(id).ability.mult;
                g.card_mut(id).ability.mult = 0.0;
                if last > 0.0 {
                    return Some(noted(id));
                }
            } else {
                let extra = g.extra(id);
                g.card_mut(id).ability.mult += extra;
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

/// Space Joker: before each hand, a 1 in `extra` chance to level up the played hand
/// (card.lua:3420-3426). No Blueprint guard: a copy rolls its own chance.
fn space(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::Before {
        return None;
    }
    let odds = g.extra(id);
    g.roll("space", odds).then(|| Effect {
        level_up: true,
        ..noted(id)
    })
}

/// Square Joker: before each hand of exactly four cards, +`chip_mod` Chips
/// (card.lua:3427-3434); +Chips in the main step, whatever the value (3901-3907).
fn square(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    match ctx.kind {
        Kind::Before if ctx.full_hand.len() == 4 && ctx.blueprint == 0 => {
            grow_chips(g, id);
            Some(noted(id))
        }
        Kind::JokerMain => Some(chip_mod(g.extra_num(id, "chips"))),
        _ => None,
    }
}

/// Runner: before each hand containing a Straight, +`chip_mod` Chips (card.lua:3435-3442);
/// +Chips in the main step, whatever the value (3908-3914).
fn runner(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    match ctx.kind {
        Kind::Before if ctx.has_hand(HandType::Straight) && ctx.blueprint == 0 => {
            grow_chips(g, id);
            Some(noted(id))
        }
        Kind::JokerMain => Some(chip_mod(g.extra_num(id, "chips"))),
        _ => None,
    }
}

/// `self.ability.extra.chips = self.ability.extra.chips + self.ability.extra.chip_mod`.
fn grow_chips(g: &mut Game, id: u32) {
    let chips = g.extra_num(id, "chips") + g.extra_num(id, "chip_mod");
    g.set_extra_num(id, "chips", chips);
}

/// Spare Trousers: before each hand containing a Two Pair or a Full House, +`extra` Mult
/// (card.lua:3412-3419); +Mult in the main step (3986-3991).
fn spare_trousers(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    match ctx.kind {
        Kind::Before
            if (ctx.has_hand(HandType::TwoPair) || ctx.has_hand(HandType::FullHouse))
                && ctx.blueprint == 0 =>
        {
            let extra = g.extra(id);
            g.card_mut(id).ability.mult += extra;
            Some(noted(id))
        }
        Kind::JokerMain => {
            let m = g.card(id).ability.mult;
            (m > 0.0).then(|| mult_mod(m))
        }
        _ => None,
    }
}

/// Green Joker: +`hand_add` Mult before each hand (card.lua:3563-3569), -`discard_sub` (not
/// below 0) on the last card of each discard (2846-2856); +Mult in the main step (4010-4015).
fn green_joker(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    match ctx.kind {
        Kind::Discard if ctx.blueprint == 0 && ctx.other_card == ctx.full_hand.last().copied() => {
            let prev = g.card(id).ability.mult;
            let next = (prev - g.extra_num(id, "discard_sub")).max(0.0);
            g.card_mut(id).ability.mult = next;
            (next != prev).then(|| noted(id))
        }
        Kind::Before if ctx.blueprint == 0 => {
            let add = g.extra_num(id, "hand_add");
            g.card_mut(id).ability.mult += add;
            Some(noted(id))
        }
        Kind::JokerMain => {
            let m = g.card(id).ability.mult;
            (m > 0.0).then(|| mult_mod(m))
        }
        _ => None,
    }
}

/// Obelisk: before each hand, back to X1 when the played hand is the most played (no other
/// visible hand played as often), else +`extra` XMult (card.lua:3543-3562). The shared
/// `x_mult > 1` rule scores it.
fn obelisk(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::Before || ctx.blueprint > 0 {
        return None;
    }
    let name = ctx.scoring_name?;
    let play_more_than = g.gs.hand(name).played;
    let reset = !HandType::ALL.into_iter().any(|h| {
        let s = g.gs.hand(h);
        h != name && s.played >= play_more_than && s.visible
    });
    if reset {
        if g.card(id).ability.x_mult > 1.0 {
            g.card_mut(id).ability.x_mult = 1.0;
            return Some(noted(id));
        }
    } else {
        let extra = g.extra(id);
        g.card_mut(id).ability.x_mult += extra;
    }
    None
}

/// Ice Cream: after each hand, -`chip_mod` Chips, melting instead when that would reach 0
/// (card.lua:3571-3600); +Chips in the main step, whatever the value (3915-3921).
fn ice_cream(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    match ctx.kind {
        Kind::After if ctx.blueprint == 0 => {
            let (chips, step) = (g.extra_num(id, "chips"), g.extra_num(id, "chip_mod"));
            if chips - step <= 0.0 {
                g.dissolve_later(id);
            } else {
                g.set_extra_num(id, "chips", chips - step);
            }
            Some(message())
        }
        Kind::JokerMain => Some(chip_mod(g.extra_num(id, "chips"))),
        _ => None,
    }
}

/// Popcorn: at the end of each round, -`extra` Mult, eaten instead when that would reach 0
/// (card.lua:2945-2974); +Mult in the main step (4004-4009).
fn popcorn(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    match ctx.kind {
        // The dispatch already refused Blueprint copies of this branch.
        Kind::EndOfRound if !ctx.individual && !ctx.repetition => {
            let (m, step) = (g.card(id).ability.mult, g.extra(id));
            if m - step <= 0.0 {
                g.dissolve_later(id);
            } else {
                g.card_mut(id).ability.mult = m - step;
            }
            Some(message())
        }
        Kind::JokerMain => {
            let m = g.card(id).ability.mult;
            (m > 0.0).then(|| mult_mod(m))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Enhancement changes (`before`)
// ---------------------------------------------------------------------------

/// Midas Mask: before each hand, every scoring face card becomes a Gold Card
/// (card.lua:3443-3464), through a non-initial `set_ability` (the boss re-checks it).
fn midas_mask(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::Before || ctx.blueprint > 0 {
        return None;
    }
    let mut faces = 0;
    for &c in ctx.scoring_hand {
        let pareidolia = g.has_joker("Pareidolia");
        if g.card(c).is_face(false, pareidolia) {
            faces += 1;
            g.set_card_ability(c, centers::M_GOLD);
        }
    }
    (faces > 0).then(|| noted(id))
}

/// Vampire: before each hand, every enhanced, non-debuffed scoring card loses its enhancement
/// and Vampire gains `extra` XMult per card (card.lua:3465-3490). The card stays `vampired`
/// until a queued event clears it. The shared `x_mult > 1` rule scores it.
fn vampire(g: &mut Game, id: u32, ctx: &mut Ctx) -> Option<Effect> {
    if ctx.kind != Kind::Before || ctx.blueprint > 0 {
        return None;
    }
    let mut enhanced = 0.0;
    for &c in ctx.scoring_hand {
        let card = g.card(c);
        if card.center != centers::C_BASE && !card.debuff && !card.vampired {
            enhanced += 1.0;
            g.card_mut(c).vampired = true;
            g.set_card_ability(c, centers::C_BASE);
            g.queue(Event::Call(clear_vampired, c, 0.0));
        }
    }
    if enhanced > 0.0 {
        let extra = g.extra(id);
        g.card_mut(id).ability.x_mult += extra * enhanced;
        return Some(noted(id));
    }
    None
}

/// Vampire's event: `v.vampired = nil` (card.lua:3472-3478).
fn clear_vampired(g: &mut Game, card: u32, _: f64) {
    if let Some(c) = g.gs.areas.card_mut(card) {
        c.vampired = false;
    }
}

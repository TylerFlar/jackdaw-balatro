//! The `Card:use_consumeable` blocks that work on the hand: the enhancing, suit and rank Tarots
//! and Death (`mod_conv` / `suit_conv`, card.lua:1101-1151), the seal Spectrals (1178-1190),
//! Aura (1192-1199), Cryptid (1201-1220), Sigil and Ouija (1222-1262), and the `remove_card`
//! block: The Hanged Man, Familiar, Grim, Incantation and Immolate (1269-1371).

use jackdaw_data::{CardId, POOL_ENHANCED, centers};

use super::{arg_id, id_arg, unhighlight_hand};
use crate::area::AreaId;
use crate::card::{Base, Extra, Rank, Seal, Suit};
use crate::event::Event;
use crate::game::Game;
use crate::joker::{Ctx, Kind};

/// `{'S','H','D','C'}`: the suits Sigil, Familiar, Grim and Incantation draw from.
const SUITS: [Suit; 4] = [Suit::Spades, Suit::Hearts, Suit::Diamonds, Suit::Clubs];

impl Game {
    /// The `mod_conv` / `suit_conv` block (card.lua:1101-1151). Each highlighted card changes in
    /// its own event; a last event unhighlights the hand.
    pub(super) fn use_conversion(&mut self, id: u32) {
        let n = self.rt.highlighted.len();
        match self.card(id).name() {
            "Death" => {
                // The rightmost highlighted card (largest `T.x`, the hand position), chosen now.
                let hand = self.gs.areas.ids(AreaId::Hand);
                let pos = |c: u32| hand.iter().position(|&h| h == c);
                let mut rightmost = self.rt.highlighted[0];
                for &c in &self.rt.highlighted {
                    if pos(c) > pos(rightmost) {
                        rightmost = c;
                    }
                }
                for i in 0..n {
                    self.queue(Event::Call(death_copy, rightmost, i as f64));
                }
            }
            "Strength" => {
                for i in 0..n {
                    self.queue(Event::Call(strength_rank_up, id, i as f64));
                }
            }
            _ if self.card(id).center().config("suit_conv").is_some() => {
                for i in 0..n {
                    self.queue(Event::Call(convert_suit, id, i as f64));
                }
            }
            _ => {
                for i in 0..n {
                    self.queue(Event::Call(enhance, id, i as f64));
                }
            }
        }
        self.queue(Event::Call(unhighlight_hand, id, 0.0));
    }

    /// Talisman, Deja Vu, Trance and Medium (card.lua:1178-1190): the first highlighted card,
    /// taken now, gets the seal in `ability.extra`; then the hand is unhighlighted.
    pub(super) fn use_seal(&mut self, id: u32) {
        let conv_card = self.rt.highlighted[0];
        self.queue(Event::Call(set_seal, id, id_arg(conv_card)));
        self.queue(Event::Call(unhighlight_hand, id, 0.0));
    }

    /// Sigil and Ouija (card.lua:1222-1262): one suit or rank drawn now, then every card of the
    /// hand rebuilt in its own event. Ouija also takes a hand size.
    pub(super) fn use_sigil_ouija(&mut self, id: u32) {
        let n = self.gs.areas.hand.len();
        if self.card(id).name() == "Sigil" {
            let seed = self.gs.rng.seed("sigil");
            let suit = crate::rng::element_index(SUITS.len(), seed);
            for i in 0..n {
                self.queue(Event::Call(sigil_card, i as u32, suit as f64));
            }
        } else {
            // `{'2', ..., '9', 'T', 'J', 'Q', 'K', 'A'}`.
            let seed = self.gs.rng.seed("ouija");
            let rank = crate::rng::element_index(Rank::ALL.len(), seed);
            for i in 0..n {
                self.queue(Event::Call(ouija_card, i as u32, rank as f64));
            }
            self.change_size(AreaId::Hand, -1.0);
        }
    }

    /// The `remove_card` block (card.lua:1269-1371): the cards to destroy are chosen now and
    /// destroyed in an event, and the jokers see them as removed at once, before any of it
    /// happens (Glass Cards are not `shattered` yet).
    pub(super) fn use_remove_card(&mut self, id: u32) {
        let mut destroyed: Vec<u32> = Vec::new();
        match self.card(id).name() {
            "The Hanged Man" => {
                destroyed.extend(self.rt.highlighted.iter().rev());
                self.queue(Event::Call(hanged_man_destroy, id, 0.0));
            }
            "Familiar" | "Grim" | "Incantation" => {
                let mut hand = self.gs.areas.ids(AreaId::Hand);
                hand.sort_unstable();
                let seed = self.gs.rng.seed("random_destroy");
                destroyed.push(hand[crate::rng::element_index(hand.len(), seed)]);
                self.queue(Event::Call(destroy_one, destroyed[0], 0.0));
                self.queue(Event::Call(create_spectral_cards, id, 0.0));
            }
            "Immolate" => {
                // `temp_hand` is sorted by `playing_card`, then `pseudoshuffle` sorts it again by
                // sort id before shuffling.
                let mut temp_hand = self.gs.areas.ids(AreaId::Hand);
                temp_hand.sort_unstable();
                let seed = self.gs.rng.seed("immolate");
                crate::rng::shuffle(&mut temp_hand, seed);
                let count = self.extra_num(id, "destroy");
                let mut i = 1.0;
                while i <= count {
                    // `destroyed_cards[#destroyed_cards+1] = temp_hand[i]`: nil past the hand.
                    if let Some(&c) = temp_hand.get(i as usize - 1) {
                        destroyed.push(c);
                    }
                    i += 1.0;
                }
                // One event destroys them, last first; an event per card runs the same.
                for &c in destroyed.iter().rev() {
                    self.queue(Event::Call(destroy_one, c, 0.0));
                }
                let dollars = self.extra_num(id, "dollars");
                self.ease_dollars(dollars);
            }
            _ => {}
        }
        for j in self.joker_ids() {
            let mut ctx = Ctx::new(Kind::RemovePlayingCards);
            ctx.removed = destroyed.clone();
            self.calculate_joker(j, &mut ctx);
        }
    }
}

/// `G.hand.highlighted[i]`, read when the event runs.
fn highlighted(g: &Game, i: f64) -> u32 {
    *g.rt
        .highlighted
        .get(i as usize)
        .expect("the highlighted card an event changes")
}

/// Death: `copy_card(rightmost, G.hand.highlighted[i])` unless it is the rightmost
/// (card.lua:1115-1119).
fn death_copy(g: &mut Game, rightmost: u32, i: f64) {
    let target = highlighted(g, i);
    if target != rightmost {
        g.copy_card_onto(rightmost, target);
    }
}

/// Strength: the next rank, an Ace going to 2 (card.lua:1121-1135).
fn strength_rank_up(g: &mut Game, _: u32, i: f64) {
    let card = highlighted(g, i);
    let base = g.card(card).base.expect("a playing card");
    let id = base.id();
    let next = if id == 14 { 2 } else { (id + 1).min(14) };
    let rank = Rank::from_id(next).expect("a rank id");
    g.set_card_base(card, Base::front_for(base.suit, rank));
}

/// The Star, The Moon, The Sun and The World: `change_suit(suit_conv)` (card.lua:1137-1140).
fn convert_suit(g: &mut Game, consumable: u32, i: f64) {
    let suit = g
        .card(consumable)
        .center()
        .config("suit_conv")
        .and_then(|v| v.as_str())
        .and_then(Suit::from_name)
        .expect("a suit Tarot's suit_conv");
    let card = highlighted(g, i);
    g.change_card_suit(card, suit);
}

/// The enhancing Tarots: `set_ability(G.P_CENTERS[mod_conv])` (card.lua:1141-1144), not
/// initial, so it stamps `hands_played_at_create` and the boss re-checks the card.
fn enhance(g: &mut Game, consumable: u32, i: f64) {
    let center = g.mod_conv(consumable);
    let card = highlighted(g, i);
    g.set_card_ability(card, center);
}

/// `conv_card:set_seal(self.ability.extra, nil, true)` (card.lua:1185-1187).
fn set_seal(g: &mut Game, consumable: u32, conv_card: f64) {
    let seal = match g.card(consumable).ability.extra {
        Extra::Str(s) => Seal::from_name(s).expect("a seal Spectral's seal"),
        ref other => panic!("a seal Spectral's extra is its seal, not {other:?}"),
    };
    let card = arg_id(conv_card);
    let (area, _) = g.gs.areas.locate(card).expect("the sealed card");
    let cost = g.cost_context(area);
    g.card_mut(card).set_seal(Some(seal), &cost);
}

/// Aura: a guaranteed edition, never Negative, on the highlighted card (card.lua:1192-1199).
pub(super) fn aura(g: &mut Game, _: u32, _: f64) {
    let edition = g.poll_edition("aura", 1.0, true, true);
    let card = highlighted(g, 0.0);
    g.set_card_edition(card, edition);
}

/// Cryptid: `extra` copies of the highlighted card join the hand, then the jokers see them
/// (card.lua:1201-1220). Each copy is added to the deck (nothing for a playing card) and
/// `G.deck`'s card limit grows, which nothing reads.
pub(super) fn cryptid(g: &mut Game, id: u32, _: f64) {
    let extra = g.extra(id);
    let source = highlighted(g, 0.0);
    let mut new_cards = Vec::new();
    let mut i = 1.0;
    while i <= extra {
        let card = g.copy_card(source, false, AreaId::Hand);
        new_cards.push(card.sort_id);
        g.gs.areas.emplace(AreaId::Hand, card, false, false);
        i += 1.0;
    }
    g.playing_card_joker_effects(&new_cards);
}

/// Sigil: `G.hand.cards[i]` takes the drawn suit, keeping its rank (card.lua:1232-1244).
fn sigil_card(g: &mut Game, i: u32, suit: f64) {
    let card = g.gs.areas.hand[i as usize].sort_id;
    let base = g.card(card).base.expect("a playing card");
    g.set_card_base(card, Base::front_for(SUITS[suit as usize], base.value));
}

/// Ouija: `G.hand.cards[i]` takes the drawn rank, keeping its suit (card.lua:1246-1256).
fn ouija_card(g: &mut Game, i: u32, rank: f64) {
    let card = g.gs.areas.hand[i as usize].sort_id;
    let base = g.card(card).base.expect("a playing card");
    g.set_card_base(card, Base::front_for(base.suit, Rank::ALL[rank as usize]));
}

/// The Hanged Man: every highlighted card, last first, shatters or dissolves
/// (card.lua:1279-1291).
fn hanged_man_destroy(g: &mut Game, _: u32, _: f64) {
    for card in g.rt.highlighted.clone().into_iter().rev() {
        g.shatter_or_dissolve(card);
    }
}

/// One chosen card shatters or dissolves (Familiar, Grim, Incantation and Immolate's events).
fn destroy_one(g: &mut Game, card: u32, _: f64) {
    g.shatter_or_dissolve(card);
}

/// Familiar, Grim and Incantation: `extra` new enhanced cards join the hand
/// (card.lua:1311-1339). Each draws its rank and suit, then a non-Stone enhancement from
/// `'spe_card'`. Lua hands the jokers a list of `true`s, one per card; the port hands them the
/// new cards, which no joker reads beyond their number.
fn create_spectral_cards(g: &mut Game, id: u32, _: f64) {
    let name = g.card(id).name();
    let extra = g.extra(id);
    // `G.P_CENTER_POOLS["Enhanced"]` without `m_stone`, in pool order.
    let pool: Vec<_> = POOL_ENHANCED
        .iter()
        .copied()
        .filter(|&c| c != centers::M_STONE)
        .collect();
    let mut cards = Vec::new();
    let mut i = 1.0;
    while i <= extra {
        let (rank, suit) = match name {
            "Familiar" => {
                const FACES: [Rank; 3] = [Rank::Jack, Rank::Queen, Rank::King];
                let r = draw(g, "familiar_create", FACES.len());
                let s = draw(g, "familiar_create", SUITS.len());
                (FACES[r], SUITS[s])
            }
            "Grim" => (Rank::Ace, SUITS[draw(g, "grim_create", SUITS.len())]),
            _ => {
                // `{'2', ..., '9', 'T'}`.
                let r = draw(g, "incantation_create", 9);
                let s = draw(g, "incantation_create", SUITS.len());
                (Rank::ALL[r], SUITS[s])
            }
        };
        let front: CardId = Base::front_for(suit, rank);
        let center = pool[draw(g, "spe_card", pool.len())];
        cards.push(g.create_playing_card(Some(front), Some(center), AreaId::Hand));
        i += 1.0;
    }
    g.playing_card_joker_effects(&cards);
}

/// `pseudorandom_element(list, pseudoseed(key))` over a list of `n` strings: the 0-based index.
fn draw(g: &mut Game, key: &str, n: usize) -> usize {
    let seed = g.gs.rng.seed(key);
    crate::rng::element_index(n, seed)
}

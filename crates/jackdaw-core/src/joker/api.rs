//! Helpers joker implementations share: reading and writing a joker's ability, queueing the
//! common effects, and the per-frame values `Card:update` keeps (card.lua:4126-4254).

use jackdaw_data::{CenterId, CenterSet};

use crate::area::AreaId;
use crate::card::{Card, ExtraVal};
use crate::event::Event;
use crate::game::Game;
use crate::pools::PoolType;
use crate::scoring::Effect;

/// `{mult = x}`: an individual effect's +Mult.
pub fn mult(x: f64, card: u32) -> Effect {
    Effect {
        mult: Some(x),
        card: Some(card),
        ..Effect::default()
    }
}

/// `{chips = x}`.
pub fn chips(x: f64, card: u32) -> Effect {
    Effect {
        chips: Some(x),
        card: Some(card),
        ..Effect::default()
    }
}

/// `{x_mult = x}`.
pub fn x_mult(x: f64, card: u32) -> Effect {
    Effect {
        x_mult: Some(x),
        card: Some(card),
        ..Effect::default()
    }
}

/// `{mult_mod = x}`: a main-step +Mult.
pub fn mult_mod(x: f64) -> Effect {
    Effect {
        mult_mod: Some(x),
        message: true,
        ..Effect::default()
    }
}

/// `{chip_mod = x}`.
pub fn chip_mod(x: f64) -> Effect {
    Effect {
        chip_mod: Some(x),
        message: true,
        ..Effect::default()
    }
}

/// `{Xmult_mod = x}`.
pub fn xmult_mod(x: f64) -> Effect {
    Effect {
        xmult_mod: Some(x),
        message: true,
        ..Effect::default()
    }
}

/// `{message = ...}`: an effect with nothing but its message (still a non-nil table).
pub fn message() -> Effect {
    Effect {
        message: true,
        ..Effect::default()
    }
}

/// `{repetitions = n, card = self}`.
pub fn repetitions(n: f64, card: u32) -> Effect {
    Effect {
        repetitions: Some(n),
        card: Some(card),
        message: true,
        ..Effect::default()
    }
}

impl Game {
    /// The card with this sort id (it must exist).
    pub fn card(&self, id: u32) -> &Card {
        self.gs
            .areas
            .card(id)
            .unwrap_or_else(|| panic!("no card {id}"))
    }

    /// The card with this sort id, mutably.
    pub fn card_mut(&mut self, id: u32) -> &mut Card {
        self.gs
            .areas
            .card_mut(id)
            .unwrap_or_else(|| panic!("no card {id}"))
    }

    /// `self.ability.extra` as a number.
    pub fn extra(&self, id: u32) -> f64 {
        self.card(id)
            .ability
            .extra
            .num()
            .unwrap_or_else(|| panic!("card {id}: ability.extra is not a number"))
    }

    /// `self.ability.extra.<key>` as a number.
    pub fn extra_num(&self, id: u32, key: &str) -> f64 {
        self.card(id)
            .ability
            .extra
            .table()
            .and_then(|t| t.num(key))
            .unwrap_or_else(|| panic!("card {id}: ability.extra.{key} is not a number"))
    }

    /// `self.ability.extra.<key>` as a string.
    pub fn extra_str(&self, id: u32, key: &str) -> &str {
        match self.card(id).ability.extra.table().and_then(|t| t.get(key)) {
            Some(ExtraVal::Str(s)) => s,
            other => panic!("card {id}: ability.extra.{key} is not a string ({other:?})"),
        }
    }

    /// `self.ability.extra.<key> = value`.
    pub fn set_extra_num(&mut self, id: u32, key: &str, value: f64) {
        self.card_mut(id)
            .ability
            .extra
            .table_mut()
            .unwrap_or_else(|| panic!("card {id}: ability.extra is not a table"))
            .set(key, ExtraVal::Num(value));
    }

    /// `G.GAME.probabilities.normal`.
    pub fn probability(&self) -> f64 {
        self.gs.probabilities_normal
    }

    /// `pseudorandom(key) < G.GAME.probabilities.normal / odds`.
    pub fn roll(&mut self, key: &str, odds: f64) -> bool {
        let p = self.probability();
        self.gs.rng.random(key) < p / odds
    }

    /// `ease_dollars(x)`: queued.
    pub fn ease_dollars(&mut self, x: f64) {
        self.queue(Event::EaseDollars(x));
    }

    /// Money a joker earns during scoring: `G.GAME.dollar_buffer` covers it until its
    /// `ease_dollars` runs (Bull, Bootstraps and friends read it).
    pub fn buffer_dollars(&mut self, x: f64) {
        self.rt.dollar_buffer += x;
        self.queue(Event::ClearDollarBuffer);
    }

    /// `#G.consumeables.cards + G.GAME.consumeable_buffer < G.consumeables.config.card_limit`.
    pub fn consumable_room(&self) -> bool {
        (self.gs.areas.consumables.len() as f64) + self.rt.consumeable_buffer
            < self.gs.consumable_slots
    }

    /// `#G.jokers.cards + G.GAME.joker_buffer < G.jokers.config.card_limit`.
    pub fn joker_room(&self) -> bool {
        (self.gs.areas.jokers.len() as f64) + self.rt.joker_buffer < self.gs.joker_slots
    }

    /// Reserve a consumable slot and queue the card's creation (`consumeable_buffer += 1`,
    /// then an event that runs `create_card`, `add_to_deck`, `emplace` and clears the buffer).
    pub fn create_consumable_later(
        &mut self,
        ty: PoolType,
        append: &'static str,
        forced: Option<CenterId>,
    ) {
        self.rt.consumeable_buffer += 1.0;
        self.queue(Event::CreateConsumable { ty, append, forced });
    }

    /// The card shows its destruction and leaves the run in a later event (Gros Michel's
    /// extinction, `start_dissolve` from an event, ...).
    pub fn dissolve_later(&mut self, id: u32) {
        self.queue(Event::Dissolve(id));
    }

    /// `#G.playing_cards`.
    pub fn playing_card_count(&self) -> usize {
        self.playing_cards().count() + self.gs.sim.pending_deck_bottom.len()
    }

    /// Cards of `G.playing_cards` matching `f`.
    pub fn count_playing_cards(&self, f: impl Fn(&Card) -> bool) -> f64 {
        self.playing_cards().filter(|c| f(c)).count() as f64
    }

    /// Steel Joker's `steel_tally` (card.lua:4185-4190).
    pub fn steel_tally(&self) -> f64 {
        self.count_playing_cards(|c| c.center == jackdaw_data::centers::M_STEEL)
    }

    /// Stone Joker's `stone_tally` (card.lua:4197-4202).
    pub fn stone_tally(&self) -> f64 {
        self.count_playing_cards(|c| c.center == jackdaw_data::centers::M_STONE)
    }

    /// Driver's License's `driver_tally` (card.lua:4179-4184): enhanced playing cards.
    pub fn driver_tally(&self) -> f64 {
        self.count_playing_cards(|c| c.center != jackdaw_data::centers::C_BASE)
    }

    /// `#G.jokers.cards` counting only cards of the Joker set (Abstract Joker).
    pub fn joker_count(&self) -> f64 {
        self.gs
            .areas
            .jokers
            .iter()
            .filter(|c| c.set() == CenterSet::Joker)
            .count() as f64
    }

    /// The jokers' sort ids, in order.
    pub fn joker_ids(&self) -> crate::area::Ids {
        self.gs.areas.ids(AreaId::Jokers)
    }
}

//! Consumables: using one (`G.FUNCS.use_card`, button_callbacks.lua:2155-2316, and
//! `Card:use_consumeable`, card.lua:1091-1519), when it can be used
//! (`Card:can_use_consumeable`, 1521-1572, and `Card:check_use`, 1574-1581), and the usage
//! counts (`set_consumeable_usage`, misc_functions.lua:1184-1229).
//!
//! The Planets and Black Hole are ported; any other consumable refuses the step as unported.

use jackdaw_data::{CenterId, CenterSet};

use crate::area::AreaId;
use crate::event::Event;
use crate::game::{Game, LuaState};
use crate::hand_eval::{HandType, PAIRS_ORDER};
use crate::joker::{Ctx, Kind};

/// The pack states, where a used card stays over the hand instead of moving to `G.play`.
fn is_pack_state(state: LuaState) -> bool {
    matches!(
        state,
        LuaState::TarotPack
            | LuaState::PlanetPack
            | LuaState::SpectralPack
            | LuaState::StandardPack
            | LuaState::BuffoonPack
    )
}

impl Game {
    /// `G.FUNCS.use_card` for a consumable in `from` (the consumables or a pack): the card is
    /// used, the jokers see it, and it dissolves; then the state comes back. The card waits in
    /// the play area until it dissolves.
    pub(crate) fn use_consumable_card(&mut self, id: u32, from: AreaId) {
        let prev = self.rt.state;
        if !is_pack_state(prev) {
            // `G.STATE = G.STATES.PLAY_TAROT`; `G.STATE_COMPLETE` is left alone.
            self.rt.state = LuaState::PlayTarot;
        }
        let card = self
            .gs
            .areas
            .remove_card(from, Some(id))
            .expect("the used consumable");
        self.gs.areas.emplace(AreaId::Play, card, false, false);
        self.use_consumeable(id);
        for j in self.joker_ids() {
            let mut ctx = Ctx::new(Kind::UsingConsumeable);
            ctx.consumeable = Some(id);
            self.calculate_joker(j, &mut ctx);
        }
        self.queue(Event::UseCardAfter {
            card: id,
            dissolve: true,
            prev,
            from,
        });
    }

    /// `Card:use_consumeable(area, copier)` for the card's own use.
    pub(crate) fn use_consumeable(&mut self, id: u32) {
        self.set_consumeable_usage(id);
        let card = self.card(id);
        if card.debuff {
            return;
        }
        let center = card.center();
        let hand = center
            .config("hand_type")
            .and_then(|v| v.as_str())
            .and_then(HandType::from_name);
        if center.name == "Black Hole" {
            // Every hand, in `pairs(G.GAME.hands)` order (card.lua:1153-1174).
            for h in PAIRS_ORDER {
                self.level_up_hand(h, 1.0);
            }
        } else if let Some(h) = hand {
            // card.lua:1264-1268.
            self.level_up_hand(h, 1.0);
        } else {
            self.unported(&format!("consumable {}", center.key));
        }
    }

    /// `set_consumeable_usage(card)`: the run's usage counts, and for a Tarot or Planet the
    /// Fool's memory, two events later.
    pub(crate) fn set_consumeable_usage(&mut self, id: u32) {
        let card = self.card(id);
        let center = card.center;
        let set = card.set();
        if !card.center().consumeable {
            return;
        }
        *self
            .gs
            .consumeable_usage
            .by_key
            .entry(center.key())
            .or_insert(0.0) += 1.0;
        let totals = &mut self.gs.consumeable_usage.totals;
        if totals.is_empty() {
            for k in ["tarot", "planet", "spectral", "tarot_planet", "all"] {
                totals.insert(k.to_owned(), 0.0);
            }
        }
        let mut bump = |k: &str| *totals.get_mut(k).expect("usage total") += 1.0;
        match set {
            CenterSet::Tarot => {
                bump("tarot");
                bump("tarot_planet");
            }
            CenterSet::Planet => {
                bump("planet");
                bump("tarot_planet");
            }
            CenterSet::Spectral => bump("spectral"),
            _ => {}
        }
        bump("all");
        if matches!(set, CenterSet::Tarot | CenterSet::Planet) {
            self.queue(Event::Call(
                last_tarot_planet_outer,
                u32::from(center.0),
                0.0,
            ));
        }
    }

    /// `card:can_use_consumeable()` at rest (no card in play, nothing locked).
    pub(crate) fn can_use_consumeable(&self, id: u32) -> bool {
        if !self.gs.areas.play.is_empty() {
            return false;
        }
        let state = self.rt.state;
        if matches!(
            state,
            LuaState::HandPlayed | LuaState::DrawToHand | LuaState::PlayTarot
        ) {
            return false;
        }
        let card = self.card(id);
        let center = card.center();
        let name = center.name;
        let in_consumables = self.gs.areas.locate(id).map(|(a, _)| a) == Some(AreaId::Consumables);
        if name == "The Hermit"
            || center.config("hand_type").is_some()
            || name == "Temperance"
            || name == "Black Hole"
        {
            return true;
        }
        let editionless_jokers = || {
            self.gs
                .areas
                .jokers
                .iter()
                .any(|j| j.set() == CenterSet::Joker && j.edition.is_none())
        };
        if name == "The Wheel of Fortune" && editionless_jokers() {
            return true;
        }
        if name == "Ankh"
            && self
                .gs
                .areas
                .jokers
                .iter()
                .any(|j| j.set() == CenterSet::Joker && self.gs.joker_slots > 1.0)
        {
            return true;
        }
        let highlighted: Vec<u32> = self.rt.highlighted.clone();
        if name == "Aura"
            && highlighted.len() == 1
            && self
                .gs
                .areas
                .card(highlighted[0])
                .is_some_and(|c| c.edition.is_none())
        {
            return true;
        }
        if (name == "Ectoplasm" || name == "Hex") && editionless_jokers() {
            return true;
        }
        let consumable_room =
            (self.gs.areas.consumables.len() as f64) < self.gs.consumable_slots || in_consumables;
        if (name == "The Emperor" || name == "The High Priestess") && consumable_room {
            return true;
        }
        if name == "The Fool"
            && consumable_room
            && self
                .gs
                .last_tarot_planet
                .is_some_and(|c| c.key() != "c_fool")
        {
            return true;
        }
        if matches!(name, "Judgement" | "The Soul" | "Wraith") {
            return (self.gs.areas.jokers.len() as f64) < self.gs.joker_slots;
        }
        if matches!(
            state,
            LuaState::SelectingHand
                | LuaState::TarotPack
                | LuaState::SpectralPack
                | LuaState::PlanetPack
        ) {
            if let Some(max) = center.config_num("max_highlighted") {
                // `consumeable.mod_num = math.min(5, max_highlighted)`, set by Card:update
                // (card.lua:4154-4156).
                let mod_num = max.min(5.0);
                let min = center.config_num("min_highlighted").unwrap_or(1.0);
                let n = highlighted.len() as f64;
                if mod_num >= n && n >= min {
                    return true;
                }
            }
            if matches!(
                name,
                "Familiar" | "Grim" | "Incantation" | "Immolate" | "Sigil" | "Ouija"
            ) && self.gs.areas.hand.len() > 1
            {
                return true;
            }
        }
        false
    }

    /// `card:check_use()`: Ankh refuses to be used with the joker slots full (card.lua:1574).
    pub(crate) fn check_use(&self, id: u32) -> bool {
        self.card(id).name() == "Ankh" && (self.gs.areas.jokers.len() as f64) >= self.gs.joker_slots
    }
}

/// `set_consumeable_usage`'s outer event (misc_functions.lua:1213-1225).
fn last_tarot_planet_outer(g: &mut Game, center: u32, _: f64) {
    g.queue(Event::Call(last_tarot_planet_inner, center, 0.0));
}

/// `G.GAME.last_tarot_planet = card.config.center_key`.
fn last_tarot_planet_inner(g: &mut Game, center: u32, _: f64) {
    g.gs.last_tarot_planet = Some(CenterId(center as u16));
}

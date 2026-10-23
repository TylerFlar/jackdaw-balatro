//! Consumables: using one (`G.FUNCS.use_card`, button_callbacks.lua:2155-2316, and
//! `Card:use_consumeable`, card.lua:1091-1521), when it can be used
//! (`Card:can_use_consumeable`, 1523-1579, and `Card:check_use`, 1581-1588), and the usage
//! counts (`set_consumeable_usage`, misc_functions.lua:1184-1229).
//!
//! [`Game::use_consumeable`] is the Lua function's body: a run of independent `if` blocks, kept
//! in Lua order because their events and RNG draws are observable. The blocks that work on the
//! hand live in [`hand`], the ones that make cards or move money and jokers in [`run`].
//!
//! Lua's event closures become [`Event::Call`]s: the `u32` is the used consumable (or the card
//! the closure captured) and the `f64` a value the closure captured (an index into
//! `G.hand.highlighted` or `G.hand.cards`, a drawn suit or rank, a sort id). A closure that
//! reads `G.hand.highlighted[i]` reads it when it runs, as Lua does. Events that only play
//! sounds, juice a card or show text are not queued; neither are the two `flip` passes around a
//! card change, which turn each card over and back with nothing in between reading its facing.

mod hand;
mod run;

use jackdaw_data::{CardId, CenterId, CenterSet};

use crate::area::AreaId;
use crate::card::{Base, EditionKind, Suit};
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

/// A sort id carried in an event's `f64` argument.
fn id_arg(id: u32) -> f64 {
    f64::from(id)
}

/// The sort id an event's `f64` argument carries.
fn arg_id(v: f64) -> u32 {
    v as u32
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

    /// `Card:use_consumeable(area, copier)` for the card's own use (card.lua:1091-1521). The used
    /// card is `used_tarot`; it stays in the play area until `use_card` dissolves it, after
    /// every event queued here.
    pub(crate) fn use_consumeable(&mut self, id: u32) {
        // `stop_use()` only locks the controls.
        self.set_consumeable_usage(id);
        let card = self.card(id);
        if card.debuff {
            return;
        }
        let center = card.center();
        let name = center.name;
        // `update_hand_text` for `max_highlighted` cards is display only.
        if center.config("mod_conv").is_some() || center.config("suit_conv").is_some() {
            self.use_conversion(id);
        }
        if name == "Black Hole" {
            // Every hand, in `pairs(G.GAME.hands)` order (card.lua:1153-1176).
            for h in PAIRS_ORDER {
                self.level_up_hand(h, 1.0);
            }
        }
        if matches!(name, "Talisman" | "Deja Vu" | "Trance" | "Medium") {
            self.use_seal(id);
        }
        if name == "Aura" {
            self.queue(Event::Call(hand::aura, id, 0.0));
        }
        if name == "Cryptid" {
            self.queue(Event::Call(hand::cryptid, id, 0.0));
        }
        if name == "Sigil" || name == "Ouija" {
            self.use_sigil_ouija(id);
        }
        if let Some(h) = center
            .config("hand_type")
            .and_then(|v| v.as_str())
            .and_then(HandType::from_name)
        {
            // card.lua:1264-1267.
            self.level_up_hand(h, 1.0);
        }
        if center
            .config("remove_card")
            .and_then(|v| v.as_bool())
            .unwrap_or(false)
        {
            self.use_remove_card(id);
        }
        if name == "The Fool" {
            self.queue(Event::Call(run::fool, id, 0.0));
        }
        if name == "The Hermit" {
            self.queue(Event::Call(run::hermit, id, 0.0));
        }
        if name == "Temperance" {
            self.queue(Event::Call(run::temperance, id, 0.0));
        }
        if name == "The Emperor" || name == "The High Priestess" {
            self.use_emperor_priestess(id);
        }
        if name == "Judgement" || name == "The Soul" {
            self.queue(Event::Call(run::judgement_soul, id, 0.0));
        }
        if name == "Ankh" {
            self.use_ankh();
        }
        if name == "Wraith" {
            self.queue(Event::Call(run::wraith, id, 0.0));
        }
        if matches!(name, "The Wheel of Fortune" | "Ectoplasm" | "Hex") {
            self.use_edition_joker(id);
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
        if name == "The Wheel of Fortune" && !self.editionless_jokers().is_empty() {
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
        if (name == "Ectoplasm" || name == "Hex") && !self.editionless_jokers().is_empty() {
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

    /// `card:check_use()`: Ankh refuses to be used with the joker slots full (card.lua:1581-1588).
    pub(crate) fn check_use(&self, id: u32) -> bool {
        self.card(id).name() == "Ankh" && (self.gs.areas.jokers.len() as f64) >= self.gs.joker_slots
    }

    /// `eligible_strength_jokers` (The Wheel of Fortune) and `eligible_editionless_jokers`
    /// (Ectoplasm, Hex), which `Card:update` rebuilds every frame (card.lua:4209-4224): every
    /// card of the Joker set in `G.jokers` without an edition, debuffed or not, in
    /// `pseudorandom_element`'s order (by sort id).
    fn editionless_jokers(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self
            .gs
            .areas
            .jokers
            .iter()
            .filter(|j| j.set() == CenterSet::Joker && j.edition.is_none())
            .map(|j| j.sort_id)
            .collect();
        ids.sort_unstable();
        ids
    }

    // -----------------------------------------------------------------------
    // Card changes the consumables share
    // -----------------------------------------------------------------------

    /// `card:set_base(front)` on a card already in play (not initial): the base is rebuilt
    /// (`times_played` back to 0, `original_value` unset, the first suit's tiebreak kept), then
    /// the boss re-checks the card (card.lua:97-145).
    fn set_card_base(&mut self, id: u32, front: CardId) {
        let card = self.card_mut(id);
        card.base = Some(crate::card::set_base(card.base.as_ref(), front, false));
        self.debuff_card(id, false);
    }

    /// `card:change_suit(suit)` (card.lua:547-562): `set_base` with the new suit's front, then
    /// the boss check once more.
    fn change_card_suit(&mut self, id: u32, suit: Suit) {
        let value = self.card(id).base.expect("a playing card").value;
        self.set_card_base(id, Base::front_for(suit, value));
        self.debuff_card(id, false);
    }

    /// `card:set_edition(edition, true)` on a card in an area (card.lua:387-462), with the card
    /// limit a Negative edition adds to a card already added to the deck.
    fn set_card_edition(&mut self, id: u32, kind: Option<EditionKind>) {
        let (area, _) = self
            .gs
            .areas
            .locate(id)
            .expect("the card gaining an edition");
        let cost = self.cost_context(area);
        let card = self.card_mut(id);
        let consumeable = card.center().consumeable;
        if card.set_edition(kind, &cost) {
            if consumeable {
                self.gs.consumable_slots += 1.0;
            } else {
                self.gs.joker_slots += 1.0;
            }
        }
    }

    /// `card:shatter()` for a Glass Card, else `card:start_dissolve()`: the card is marked and
    /// leaves the run when its animation ends (card.lua:2079-2182).
    fn shatter_or_dissolve(&mut self, id: u32) {
        if let Some(c) = self.gs.areas.card_mut(id)
            && c.name() == "Glass Card"
        {
            c.shattered = true;
        }
        self.queue(Event::RemoveCard(id));
    }

    /// `copy_card(other, new_card)` onto a card that is already in play (Death, card.lua:1117;
    /// common_events.lua:2156-2181): `set_ability` with the other's center and `set_base` with
    /// its front (neither initial, so the boss checks the card twice), every ability entry the
    /// other has, its edition, seal, debuff and pin.
    ///
    /// The ability is the fresh table `set_ability` built, overwritten by every entry of the
    /// other's; the only entry the other can lack while the target has it is
    /// `forced_selection` (Cerulean Bell), which `set_ability` carries over.
    fn copy_card_onto(&mut self, other: u32, target: u32) {
        let src = self.card(other).clone();
        self.set_card_ability(target, src.center);
        if let Some(b) = src.base {
            self.set_card_base(target, b.front);
        }
        let (area, _) = self.gs.areas.locate(target).expect("the copy's target");
        let cost = self.cost_context(area);
        let card = self.card_mut(target);
        let forced = card.ability.forced_selection;
        card.ability = src.ability.clone();
        card.ability.forced_selection |= forced;
        // `set_edition(other.edition or {}, nil, true)`: a playing card is never added to the
        // deck in the port, so a Negative edition takes no slot (none exists on playing cards).
        card.set_edition(src.edition.map(|e| e.kind), &cost);
        card.set_seal(src.seal, &cost);
        card.debuff = src.debuff;
        card.pinned = src.pinned;
    }

    /// `create_card(...)`, `card:add_to_deck()` and `area:emplace(card)` for a card made into
    /// the jokers or the consumables. Lua adds the card to the deck before emplacing it; the
    /// port emplaces first because `add_to_deck` finds the card by its area, and none of its
    /// effects depend on the card's area.
    fn create_and_emplace(&mut self, spec: crate::pools::CardSpec, area: AreaId) -> u32 {
        let card = self.create_card(spec);
        let id = card.sort_id;
        self.gs.areas.emplace(area, card, false, false);
        self.add_to_deck(id, false);
        id
    }

    /// The center a consumable's `mod_conv` names (`m_lucky`, ...).
    fn mod_conv(&self, id: u32) -> CenterId {
        let key = self
            .card(id)
            .center()
            .config("mod_conv")
            .and_then(|v| v.as_str())
            .expect("an enhancing Tarot's mod_conv");
        jackdaw_data::center_by_key(key).expect("mod_conv names a center")
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

/// `G.hand:unhighlight_all()` in an event (the state is `PLAY_TAROT` or a pack's, so the hand
/// is not parsed).
fn unhighlight_hand(g: &mut Game, _: u32, _: f64) {
    g.unhighlight_all_hand();
}

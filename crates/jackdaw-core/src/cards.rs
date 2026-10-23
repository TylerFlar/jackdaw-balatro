//! `Card` methods that act on the game: debuffs, deck membership (passive joker effects) and
//! removal (card.lua), plus `find_joker` (misc_functions.lua:903).

use jackdaw_data::CenterSet;

use crate::area::AreaId;
use crate::event::Event;
use crate::game::Game;

impl Game {
    /// `find_joker(name, non_debuff)`: the jokers and consumables named `name`, in area order.
    /// Lua's `non_debuff` flag *includes* debuffed cards; here that's `include_debuffed`.
    pub fn find_joker(&self, name: &str, include_debuffed: bool) -> Vec<u32> {
        [AreaId::Jokers, AreaId::Consumables]
            .into_iter()
            .flat_map(|a| self.gs.areas.get(a).iter())
            .filter(|c| c.name() == name && (include_debuffed || !c.debuff))
            .map(|c| c.sort_id)
            .collect()
    }

    /// `next(find_joker(name))`: a non-debuffed card named `name` is held.
    pub fn has_joker(&self, name: &str) -> bool {
        [AreaId::Jokers, AreaId::Consumables]
            .into_iter()
            .flat_map(|a| self.gs.areas.get(a).iter())
            .any(|c| c.name() == name && !c.debuff)
    }

    /// `Card:set_debuff(should_debuff)` (card.lua:526-538). A spent perishable stays debuffed;
    /// a joker that changes state loses or regains its passive effects.
    pub(crate) fn set_debuff(&mut self, id: u32, should_debuff: bool) {
        let Some((area, i)) = self.gs.areas.locate(id) else {
            return;
        };
        let card = &self.gs.areas.get(area)[i];
        if card.ability.perishable && card.ability.perish_tally.unwrap_or(0.0) <= 0.0 {
            if !card.debuff {
                self.gs.areas.get_mut(area)[i].debuff = true;
                if area == AreaId::Jokers {
                    self.remove_from_deck(id, true);
                }
            }
            return;
        }
        // Card:update re-debuffs a perma_debuff card every frame (card.lua:4158).
        let should_debuff = should_debuff || card.ability.perma_debuff;
        if should_debuff != card.debuff {
            if area == AreaId::Jokers {
                if should_debuff {
                    self.remove_from_deck(id, true);
                } else {
                    self.add_to_deck(id, true);
                }
            }
            if let Some(c) = self.gs.areas.card_mut(id) {
                c.debuff = should_debuff;
            }
        }
    }

    /// `Card:add_to_deck(from_debuff)` (card.lua:564-643): the card's passive effects start.
    pub(crate) fn add_to_deck(&mut self, id: u32, from_debuff: bool) {
        let Some(card) = self.gs.areas.card_mut(id) else {
            return;
        };
        if card.added_to_deck {
            return;
        }
        card.added_to_deck = true;
        if matches!(card.set(), CenterSet::Enhanced | CenterSet::Default) {
            // A playing card only gets the flag (card.lua:568-575); its removal reads it and
            // queues the boss re-debuff (G205).
            return;
        }
        let card = card.clone();
        let a = &card.ability;
        if a.h_size != 0.0 {
            self.change_size(AreaId::Hand, a.h_size);
        }
        if a.d_size > 0.0 {
            self.gs.round_resets.discards += a.d_size;
            self.queue(Event::EaseDiscard(a.d_size));
        }
        let table_num = |k: &str| a.extra.table().and_then(|t| t.num(k)).unwrap_or(0.0);
        match card.name() {
            "Credit Card" => self.gs.bankrupt_at -= a.extra.num().unwrap_or(0.0),
            "Chicot" => {
                if self
                    .gs
                    .blind
                    .as_ref()
                    .is_some_and(|b| b.boss && !b.disabled)
                {
                    self.blind_disable();
                }
            }
            "Chaos the Clown" => {
                self.gs.current_round.free_rerolls += 1.0;
                self.calculate_reroll_cost(true);
            }
            "Turtle Bean" => self.change_size(AreaId::Hand, table_num("h_size")),
            // `for k, v in pairs(G.GAME.probabilities)`: `normal` is the only key.
            "Oops! All 6s" => self.gs.probabilities_normal *= 2.0,
            "To the Moon" => self.gs.interest_amount += a.extra.num().unwrap_or(0.0),
            "Astronomer" => self.queue(Event::SetCostAll),
            "Troubadour" => {
                self.change_size(AreaId::Hand, table_num("h_size"));
                self.gs.round_resets.hands += table_num("h_plays");
            }
            "Stuntman" => self.change_size(AreaId::Hand, -table_num("h_size")),
            _ => {}
        }
        if card
            .edition
            .is_some_and(|e| e.kind == crate::card::EditionKind::Negative)
        {
            if from_debuff {
                if let Some(c) = self.gs.areas.card_mut(id) {
                    c.ability.queue_negative_removal = false;
                }
            } else if card.center().consumeable {
                self.gs.consumable_slots += 1.0;
            } else {
                self.gs.joker_slots += 1.0;
            }
        }
        if self.gs.blind.is_some() {
            self.queue(Event::BlindRedebuff);
        }
    }

    /// `Card:remove_from_deck(from_debuff)` (card.lua:645-700): the card's passive effects stop.
    pub(crate) fn remove_from_deck(&mut self, id: u32, from_debuff: bool) {
        let Some(card) = self.gs.areas.card(id).cloned() else {
            return;
        };
        self.remove_from_deck_card(id, card, from_debuff);
    }

    /// [`Game::remove_from_deck`] for a card that may already be out of every area.
    pub(crate) fn remove_from_deck_card(
        &mut self,
        id: u32,
        card: crate::card::Card,
        from_debuff: bool,
    ) {
        if !card.added_to_deck {
            return;
        }
        if let Some(c) = self.gs.areas.card_mut(id) {
            c.added_to_deck = false;
        }
        let a = &card.ability;
        if a.h_size != 0.0 {
            self.change_size(AreaId::Hand, -a.h_size);
        }
        if a.d_size > 0.0 {
            self.gs.round_resets.discards -= a.d_size;
            self.queue(Event::EaseDiscard(-a.d_size));
        }
        let table_num = |k: &str| a.extra.table().and_then(|t| t.num(k)).unwrap_or(0.0);
        match card.name() {
            "Credit Card" => self.gs.bankrupt_at += a.extra.num().unwrap_or(0.0),
            "Chaos the Clown" => {
                self.gs.current_round.free_rerolls -= 1.0;
                self.calculate_reroll_cost(true);
            }
            "Turtle Bean" => self.change_size(AreaId::Hand, -table_num("h_size")),
            "Oops! All 6s" => self.gs.probabilities_normal /= 2.0,
            "To the Moon" => self.gs.interest_amount -= a.extra.num().unwrap_or(0.0),
            "Astronomer" => self.queue(Event::SetCostAll),
            "Troubadour" => {
                self.change_size(AreaId::Hand, -table_num("h_size"));
                self.gs.round_resets.hands -= table_num("h_plays");
            }
            "Stuntman" => self.change_size(AreaId::Hand, table_num("h_size")),
            _ => {}
        }
        if card
            .edition
            .is_some_and(|e| e.kind == crate::card::EditionKind::Negative)
        {
            if from_debuff {
                if let Some(c) = self.gs.areas.card_mut(id) {
                    c.ability.queue_negative_removal = true;
                }
            } else if card.center().consumeable {
                self.gs.consumable_slots -= 1.0;
            } else {
                self.gs.joker_slots -= 1.0;
            }
        }
        if self.gs.blind.is_some() {
            self.queue(Event::BlindRedebuff);
        }
    }

    /// `Card:remove()` (card.lua:4727-4769): the card leaves its area and the run.
    pub(crate) fn remove_card(&mut self, id: u32) {
        let Some((area, i)) = self.gs.areas.locate(id) else {
            return;
        };
        let card = self.gs.areas.get_mut(area).remove(i);
        self.rt.highlighted.retain(|&h| h != id);
        self.remove_from_deck_card(id, card.clone(), false);
        if card.ability.queue_negative_removal {
            if card.center().consumeable {
                self.gs.consumable_slots -= 1.0;
            } else {
                self.gs.joker_slots -= 1.0;
            }
        }
        // Every center with this name leaves used_jokers unless a joker or consumable of that
        // name is still held, debuffed or not.
        let name = card.name();
        if self.find_joker(name, true).is_empty() {
            for c in jackdaw_data::CENTERS.iter().filter(|c| c.name == name) {
                self.gs.sets.used_jokers.remove(c.key);
            }
        }
        // Lua also drops the card from G.playing_cards and renumbers every card's
        // `playing_card` (card.lua:4752-4760). Those numbers follow the session-wide
        // `G.playing_card` counter, which no run state can reproduce, and nothing in the game
        // reads them, so the port doesn't keep them.
    }

    /// One frame of `Card:update` for every card (card.lua:4157-4248): the values some cards keep
    /// fresh every frame. Lua runs it after the frame's event and state handler, so a card's
    /// code sees the values as of the end of the previous event.
    pub(crate) fn update_cards(&mut self) {
        use jackdaw_data::centers::{
            C_BASE, C_TEMPERANCE, J_CLOUD_9, J_DRIVERS_LICENSE, J_STEEL_JOKER, J_STENCIL, J_STONE,
            J_SWASHBUCKLER, J_THROWBACK, M_STEEL, M_STONE,
        };
        // The cards with per-frame values, one bit each (each name has a single center).
        let bit = |c: &crate::card::Card| -> u8 {
            match c.center {
                J_STEEL_JOKER => 1,
                J_STONE => 2,
                J_CLOUD_9 => 4,
                J_DRIVERS_LICENSE => 8,
                J_STENCIL => 16,
                J_SWASHBUCKLER => 32,
                J_THROWBACK => 64,
                C_TEMPERANCE => 128,
                _ => 0,
            }
        };
        let mut present = 0u8;
        for a in AreaId::ALL {
            for c in self.gs.areas.get_mut(a) {
                if c.ability.perma_debuff {
                    c.debuff = true;
                }
                present |= bit(c);
            }
        }
        if present == 0 {
            return;
        }
        let count = |g: &Game, f: &dyn Fn(&crate::card::Card) -> bool| {
            g.playing_card_ids()
                .into_iter()
                .filter(|&id| g.gs.areas.card(id).is_some_and(f))
                .count() as f64
        };
        let steel = (present & 1 != 0).then(|| count(self, &|c| c.center == M_STEEL));
        let stone = (present & 2 != 0).then(|| count(self, &|c| c.center == M_STONE));
        let nine = (present & 4 != 0).then(|| count(self, &|c| c.get_id() == Some(9)));
        let driver = (present & 8 != 0).then(|| count(self, &|c| c.center != C_BASE));
        let joker_limit = self.gs.joker_slots;
        let jokers = &self.gs.areas.jokers;
        let n_jokers = jokers.len() as f64;
        let stencils = jokers.iter().filter(|c| c.center == J_STENCIL).count() as f64;
        let joker_sell: f64 = jokers
            .iter()
            .filter(|c| c.set() == CenterSet::Joker)
            .map(|c| c.sell_cost)
            .sum();
        let sells: Vec<(u32, f64)> = if present & 32 != 0 {
            jokers.iter().map(|c| (c.sort_id, c.sell_cost)).collect()
        } else {
            Vec::new()
        };
        let skips = self.gs.skips;
        for a in AreaId::ALL {
            for c in self.gs.areas.get_mut(a) {
                match c.center {
                    C_TEMPERANCE => {
                        let cap = c.ability.extra.num().unwrap_or(0.0);
                        c.ability.money = Some(joker_sell.min(cap));
                    }
                    J_THROWBACK => {
                        c.ability.x_mult = 1.0 + skips * c.ability.extra.num().unwrap_or(0.0);
                    }
                    J_DRIVERS_LICENSE => c.ability.driver_tally = driver,
                    J_STEEL_JOKER => c.ability.steel_tally = steel,
                    J_CLOUD_9 => c.ability.nine_tally = nine,
                    J_STONE => c.ability.stone_tally = stone,
                    J_STENCIL => c.ability.x_mult = (joker_limit - n_jokers) + stencils,
                    J_SWASHBUCKLER => {
                        // Every other card in G.jokers, debuffed or not.
                        let own = c.sort_id;
                        c.ability.mult = sells
                            .iter()
                            .filter(|(id, _)| *id != own)
                            .map(|(_, s)| s)
                            .sum();
                    }
                    _ => {}
                }
            }
        }
    }

    /// `CardArea:change_size(delta)` (cardarea.lua:94-112): queued.
    pub(crate) fn change_size(&mut self, area: AreaId, delta: f64) {
        if delta != 0.0 {
            self.queue(Event::ChangeSize { area, delta });
        }
    }

    /// The body of `CardArea:change_size`'s event.
    pub(crate) fn change_size_event(&mut self, area: AreaId, delta: f64) {
        let real = match area {
            AreaId::Hand => &mut self.gs.hand_size,
            AreaId::Jokers => &mut self.gs.joker_slots,
            AreaId::Consumables => &mut self.gs.consumable_slots,
            other => unreachable!("change_size on {other:?}"),
        };
        *real += delta;
        let real = *real;
        if delta > 0.0
            && real > 1.0
            && area == AreaId::Hand
            && !self.gs.areas.hand.is_empty()
            && matches!(
                self.rt.state,
                crate::game::LuaState::DrawToHand | crate::game::LuaState::SelectingHand
            )
        {
            for _ in 0..(delta.abs() as u32) {
                self.queue(Event::DrawCard {
                    from: AreaId::Deck,
                    to: AreaId::Hand,
                    card: None,
                    sort: false,
                });
                self.queue(Event::SortHand);
            }
        }
    }

    /// `ease_discard(mod)`'s body (common_events.lua:112-131): no change from 0 discards, and
    /// never below 0.
    pub(crate) fn ease_discard_now(&mut self, m: f64) {
        let left = self.gs.current_round.discards_left;
        if left.max(m).abs() == 0.0 {
            return;
        }
        self.gs.current_round.discards_left = left + m.max(-left);
    }

    /// `G.hand.config.card_limit`: the hand size, never below 0.
    pub fn hand_limit(&self) -> f64 {
        self.gs.hand_size.max(0.0)
    }

    /// `calculate_reroll_cost(skip_increment)` (common_events.lua:2263-2269).
    pub(crate) fn calculate_reroll_cost(&mut self, skip_increment: bool) {
        let cr = &mut self.gs.current_round;
        if cr.free_rerolls < 0.0 {
            cr.free_rerolls = 0.0;
        }
        if cr.free_rerolls > 0.0 {
            cr.reroll_cost = 0.0;
            return;
        }
        if !skip_increment {
            cr.reroll_cost_increase += 1.0;
        }
        let base = self
            .gs
            .round_resets
            .temp_reroll_cost
            .unwrap_or(self.gs.round_resets.reroll_cost);
        self.gs.current_round.reroll_cost = base + self.gs.current_round.reroll_cost_increase;
    }
}

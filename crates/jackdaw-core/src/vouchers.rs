//! Vouchers: redeeming one (`G.FUNCS.use_card` and `Card:redeem`, card.lua:1808-1868), its
//! effect (`Card:apply_to_run`, card.lua:1870-1956) and `change_shop_size`
//! (common_events.lua:1097-1118).

use crate::area::AreaId;
use crate::event::Event;
use crate::game::{Game, LuaState};

impl Game {
    /// `G.FUNCS.use_card` for the shop's voucher, then `Card:redeem`.
    pub(crate) fn redeem_voucher(&mut self, id: u32) {
        let prev = self.rt.state;
        self.rt.state = LuaState::PlayTarot;
        let card = self
            .gs
            .areas
            .remove_card(AreaId::ShopVouchers, Some(id))
            .expect("the redeemed voucher");
        let center = card.center;
        let cost = card.cost;
        let extra = card.ability.extra.num();
        self.gs.areas.emplace(AreaId::Play, card, false, false);
        self.gs.sets.used_vouchers.insert(center.key());
        self.ease_dollars(-cost);
        self.gs.current_round.voucher = None;
        self.apply_voucher(center.get().name, extra);
        // The `buying_card` contexts have no effect in any joker.
        if self.gs.modifier("inflation").is_some() {
            self.gs.inflation += 1.0;
            self.queue(Event::SetCostAll);
        }
        self.queue(Event::UseCardAfter {
            card: id,
            dissolve: true,
            prev,
            from: AreaId::ShopVouchers,
        });
    }

    /// `Card:apply_to_run()` for a voucher named `name` with `ability.extra` `extra`.
    pub(crate) fn apply_voucher(&mut self, name: &str, extra: Option<f64>) {
        let x = || extra.unwrap_or_else(|| panic!("{name}: ability.extra"));
        match name {
            "Overstock" | "Overstock Plus" => {
                self.queue(Event::Call(|g, _, _| g.change_shop_size(1.0), 0, 0.0));
            }
            "Tarot Merchant" | "Tarot Tycoon" => {
                self.queue(Event::Call(|g, _, v| g.gs.tarot_rate = 4.0 * v, 0, x()));
            }
            "Planet Merchant" | "Planet Tycoon" => {
                self.queue(Event::Call(|g, _, v| g.gs.planet_rate = 4.0 * v, 0, x()));
            }
            "Hone" | "Glow Up" => {
                self.queue(Event::Call(|g, _, v| g.gs.edition_rate = v, 0, x()));
            }
            "Magic Trick" | "Illusion" => {
                self.queue(Event::Call(|g, _, v| g.gs.playing_card_rate = v, 0, x()));
            }
            "Crystal Ball" => {
                self.queue(Event::Call(|g, _, _| g.gs.consumable_slots += 1.0, 0, 0.0));
            }
            "Clearance Sale" | "Liquidation" => {
                self.queue(Event::Call(
                    |g, _, v| {
                        g.gs.discount_percent = v;
                        g.run_event(Event::SetCostAll);
                    },
                    0,
                    x(),
                ));
            }
            "Reroll Surplus" | "Reroll Glut" => {
                self.queue(Event::Call(
                    |g, _, v| {
                        g.gs.round_resets.reroll_cost -= v;
                        let cr = &mut g.gs.current_round;
                        cr.reroll_cost = (cr.reroll_cost - v).max(0.0);
                    },
                    0,
                    x(),
                ));
            }
            "Seed Money" | "Money Tree" => {
                self.queue(Event::Call(|g, _, v| g.gs.interest_cap = v, 0, x()));
            }
            "Grabber" | "Nacho Tong" => {
                self.gs.round_resets.hands += x();
                self.queue(Event::EaseHandsPlayed(x()));
            }
            "Paint Brush" | "Palette" => self.change_size(AreaId::Hand, 1.0),
            "Wasteful" | "Recyclomancy" => {
                self.gs.round_resets.discards += x();
                self.queue(Event::EaseDiscard(x()));
            }
            "Antimatter" => {
                self.queue(Event::Call(|g, _, _| g.gs.joker_slots += 1.0, 0, 0.0));
            }
            "Hieroglyph" | "Petroglyph" => {
                self.queue(Event::EaseAnte(-x()));
                self.gs.round_resets.blind_ante -= x();
                if name == "Hieroglyph" {
                    self.gs.round_resets.hands -= x();
                    self.queue(Event::EaseHandsPlayed(-x()));
                } else {
                    self.gs.round_resets.discards -= x();
                    self.queue(Event::EaseDiscard(-x()));
                }
            }
            // Telescope, Observatory, Omen Globe, Blank, Director's Cut and Retcon act elsewhere.
            _ => {}
        }
    }

    /// `change_shop_size(mod)`: the shop's joker slots, filled at once while the shop is open.
    pub(crate) fn change_shop_size(&mut self, delta: f64) {
        self.gs.shop_joker_max += delta;
        if !self.rt.shop_open {
            return;
        }
        if delta < 0.0 {
            let max = self.gs.shop_joker_max as usize;
            for id in self
                .gs
                .areas
                .ids(AreaId::ShopJokers)
                .into_iter()
                .skip(max)
                .rev()
            {
                self.remove_card(id);
            }
        }
        if delta > 0.0 {
            let n = self.gs.shop_joker_max - self.gs.areas.shop_jokers.len() as f64;
            let mut i = 1.0;
            while i <= n {
                let card = self.create_card_for_shop(AreaId::ShopJokers);
                self.gs
                    .areas
                    .emplace(AreaId::ShopJokers, card, false, false);
                i += 1.0;
            }
        }
    }
}

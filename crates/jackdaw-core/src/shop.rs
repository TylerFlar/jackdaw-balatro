//! The shop: `G.FUNCS.cash_out` (button_callbacks.lua:2913-2954), `Game:update_shop`
//! (game.lua:3072-3179), `create_card_for_shop` (UI_definitions.lua:742-799), `get_pack`
//! (common_events.lua:1944-1961), rerolls (`G.FUNCS.reroll_shop`, button_callbacks.lua:2855-2911),
//! buying (`G.FUNCS.buy_from_shop`, 2404-2479), selling (`Card:sell_card`, card.lua:1593-1651, and
//! `G.FUNCS.sell_card`, button_callbacks.lua:2317-2325) and leaving (`G.FUNCS.toggle_shop`,
//! 2481-2511).

use jackdaw_data::{CenterId, CenterSet, POOL_BOOSTER, centers};

use crate::area::AreaId;
use crate::card::{Card, EditionKind};
use crate::event::Event;
use crate::game::{Game, LuaState};
use crate::joker::{Ctx, Kind};
use crate::pools::{CardSpec, PoolType};
use crate::state::BlindSlot;

impl Game {
    /// `G.FUNCS.cash_out`: the round's money, then the shop.
    pub(crate) fn cash_out(&mut self) {
        let ante = self.gs.round_resets.ante;
        self.gs.areas.shuffle(
            AreaId::Deck,
            &mut self.gs.rng,
            &crate::rng::Key::new("cashout").n(ante),
        );
        self.queue(Event::CashOutBody);
        // `ease_dollars(G.GAME.current_round.dollars)`. Lua keeps the amount in
        // `current_round.dollars` until the next round's evaluation; the snapshot's
        // `pending_cashout` is 0 outside round_eval.
        let dollars = std::mem::take(&mut self.gs.sim.pending_cashout);
        self.ease_dollars(dollars);
        self.queue(Event::EaseChips(0.0));
        if self.gs.round_resets.blind_state(BlindSlot::Boss) == "Defeated" {
            self.gs.round_resets.blind_ante = self.gs.round_resets.ante;
            let small = self.get_next_tag_key("");
            let big = self.get_next_tag_key("");
            self.gs.round_resets.blind_tags = Some([Some(small), Some(big)]);
        }
        self.reset_blinds();
    }

    /// `cash_out`'s event (button_callbacks.lua:2921-2936).
    pub(crate) fn cash_out_body(&mut self) {
        let rr = &self.gs.round_resets;
        let bonus = &self.gs.round_bonus;
        let discards = (rr.discards + bonus.discards).max(0.0);
        let hands = (rr.hands + bonus.next_hands).max(1.0);
        let cr = &mut self.gs.current_round;
        cr.jokers_purchased = 0.0;
        cr.discards_left = discards;
        cr.hands_left = hands;
        self.set_state(LuaState::Shop, false);
        self.gs.sim.shop_free = false;
        self.gs.sim.shop_d6ed = false;
    }

    /// `Game:update_shop`: once per arrival, an event that queues the shop's construction.
    /// Returning from a pack the shop already exists, so only the `shop_start` tags run.
    pub(crate) fn update_shop(&mut self) -> bool {
        if self.rt.state_complete {
            return false;
        }
        let exists = self.rt.shop_open;
        self.rt.shop_open = true;
        self.queue(Event::ShopOpen(exists));
        self.rt.state_complete = true;
        true
    }

    /// The shop's construction (game.lua:3090-3166).
    pub(crate) fn shop_build(&mut self, exists: bool) {
        self.apply_tags_shop_start();
        if exists {
            return;
        }
        // The new shop's `G.shop_vouchers` holds one card (UI_definitions.lua:646-651).
        self.rt.areas.voucher_limit = 1.0;
        let n = self.gs.shop_joker_max - self.gs.areas.shop_jokers.len() as f64;
        let mut i = 1.0;
        while i <= n {
            let card = self.create_card_for_shop(AreaId::ShopJokers);
            self.gs
                .areas
                .emplace(AreaId::ShopJokers, card, false, false);
            i += 1.0;
        }
        if let Some(v) = self.gs.current_round.voucher {
            let card = self.new_card(v, None, AreaId::ShopVouchers);
            self.gs
                .areas
                .emplace(AreaId::ShopVouchers, card, false, false);
        }
        for _ in 0..2 {
            // `current_round.used_packs[i]` is empty for a new shop (new_round clears it).
            let key = self.get_pack("shop_pack", None);
            let card = self.new_card(key, None, AreaId::ShopBooster);
            self.gs
                .areas
                .emplace(AreaId::ShopBooster, card, false, false);
        }
        self.apply_tags_voucher_add();
        self.apply_tags_shop_final_pass();
    }

    /// `create_card_for_shop(area)`: a tag's joker, or a card of a type drawn by the shop rates.
    /// The card is not in an area yet.
    pub(crate) fn create_card_for_shop(&mut self, area: AreaId) -> Card {
        if let Some(card) = self.apply_tags_store_joker_create(area) {
            self.apply_tags_store_joker_modify(&card);
            return card;
        }
        let ante = self.gs.round_resets.ante;
        let g = &self.gs;
        let total =
            g.joker_rate + g.tarot_rate + g.planet_rate + g.playing_card_rate + g.spectral_rate;
        let polled = self.gs.rng.random(&crate::rng::Key::new("cdt").n(ante)) * total;
        let illusion = self
            .gs
            .sets
            .used_vouchers
            .contains(jackdaw_data::centers::V_ILLUSION);
        // The table's entries are built before the loop, so Illusion's roll always happens.
        let playing = if illusion && self.gs.rng.random("illusion") > 0.6 {
            PoolType::Enhanced
        } else {
            PoolType::Base
        };
        let g = &self.gs;
        let rates = [
            (PoolType::Joker, g.joker_rate),
            (PoolType::Tarot, g.tarot_rate),
            (PoolType::Planet, g.planet_rate),
            (playing, g.playing_card_rate),
            (PoolType::Spectral, g.spectral_rate),
        ];
        let mut check = 0.0;
        for (ty, val) in rates {
            if polled > check && polled <= check + val {
                let mut card = self.create_card(CardSpec::new(ty, area, "sho"));
                self.queue(Event::StoreJokerModify(card.sort_id));
                if matches!(ty, PoolType::Base | PoolType::Enhanced)
                    && illusion
                    && self.gs.rng.random("illusion") > 0.8
                {
                    let poll = self.gs.rng.random("illusion");
                    let edition = if poll > 1.0 - 0.15 {
                        EditionKind::Polychrome
                    } else if poll > 0.5 {
                        EditionKind::Holo
                    } else {
                        EditionKind::Foil
                    };
                    let cost = self.cost_context(area);
                    card.set_edition(Some(edition), &cost);
                }
                return card;
            }
            check += val;
        }
        unreachable!("the shop rates cover the roll ({polled} of {total})")
    }

    /// `get_pack(_key, _type)`: the first shop's Buffoon pack, then packs by weight.
    pub(crate) fn get_pack(&mut self, key: &str, kind: Option<&str>) -> CenterId {
        if !self.gs.first_shop_buffoon && !self.gs.sets.banned_keys.contains("p_buffoon_normal_1") {
            self.gs.first_shop_buffoon = true;
            // `math.random(1, 2)` is unseeded: the sim always takes the first variant, as the
            // Python engine did.
            return centers::P_BUFFOON_NORMAL_1;
        }
        let banned = |c: CenterId| self.gs.sets.banned_keys.contains(c.key());
        let fits = |c: CenterId| kind.is_none_or(|k| c.get().kind == Some(k));
        let weight = |c: CenterId| c.get().weight.unwrap_or(1.0);
        let cume: f64 = POOL_BOOSTER
            .iter()
            .filter(|&&c| fits(c) && !banned(c))
            .map(|&c| weight(c))
            .sum();
        let ante = self.gs.round_resets.ante;
        let poll = self.gs.rng.random(&crate::rng::Key::new(key).n(ante)) * cume;
        let mut it = 0.0;
        for &c in POOL_BOOSTER.iter() {
            if banned(c) {
                continue;
            }
            if fits(c) {
                it += weight(c);
            }
            if it >= poll && it - weight(c) <= poll {
                return c;
            }
        }
        unreachable!("get_pack's walk ends on a pack")
    }

    /// `G.FUNCS.reroll_shop`: pay, then the reroll event.
    pub(crate) fn reroll_shop(&mut self) {
        let cost = self.gs.current_round.reroll_cost;
        if cost > 0.0 {
            self.ease_dollars(-cost);
        }
        self.queue(Event::RerollBody);
    }

    /// `reroll_shop`'s event (button_callbacks.lua:2865-2890): the free reroll count, the next
    /// cost, and new shop cards.
    pub(crate) fn reroll_body(&mut self) {
        let cr = &mut self.gs.current_round;
        let final_free = cr.free_rerolls > 0.0;
        cr.free_rerolls = (cr.free_rerolls - 1.0).max(0.0);
        self.calculate_reroll_cost(final_free);
        for id in self.gs.areas.ids(AreaId::ShopJokers).into_iter().rev() {
            self.remove_card(id);
        }
        let n = self.gs.shop_joker_max - self.gs.areas.shop_jokers.len() as f64;
        let mut i = 1.0;
        while i <= n {
            let card = self.create_card_for_shop(AreaId::ShopJokers);
            self.gs
                .areas
                .emplace(AreaId::ShopJokers, card, false, false);
            i += 1.0;
        }
        self.queue(Event::RerollAfter);
    }

    /// The jokers see the reroll (button_callbacks.lua:2895-2901).
    pub(crate) fn reroll_jokers(&mut self) {
        for j in self.joker_ids() {
            let mut ctx = Ctx::new(Kind::RerollShop);
            self.calculate_joker(j, &mut ctx);
        }
    }

    /// `G.FUNCS.check_for_buy_space(card)` (button_callbacks.lua:2392-2402).
    pub(crate) fn check_for_buy_space(&self, card: &Card) -> bool {
        let negative = card
            .edition
            .is_some_and(|e| e.kind == EditionKind::Negative);
        let extra = if negative { 1.0 } else { 0.0 };
        let set = card.set();
        matches!(
            set,
            CenterSet::Voucher | CenterSet::Enhanced | CenterSet::Default
        ) || (set == CenterSet::Joker
            && (self.gs.areas.jokers.len() as f64) < self.gs.joker_slots + extra)
            || (card.center().consumeable
                && (self.gs.areas.consumables.len() as f64) < self.gs.consumable_slots + extra)
    }

    /// `buy_from_shop`'s event (button_callbacks.lua:2414-2477): the card joins the run, then
    /// the purchase is paid.
    pub(crate) fn buy_from_shop(&mut self, id: u32) {
        let Some((area, _)) = self.gs.areas.locate(id) else {
            return;
        };
        let card = self
            .gs
            .areas
            .remove_card(area, Some(id))
            .expect("the bought card");
        let set = card.set();
        let consumeable = card.center().consumeable;
        let cost = card.cost;
        // `c1:add_to_deck()` runs before the card lands; its effects don't depend on that.
        match set {
            CenterSet::Default | CenterSet::Enhanced => {
                self.gs.areas.emplace(AreaId::Deck, card, false, false);
                self.add_to_deck(id, false);
                self.playing_card_joker_effects(&[id]);
            }
            _ => {
                let to = if consumeable {
                    AreaId::Consumables
                } else {
                    AreaId::Jokers
                };
                self.gs.areas.emplace(to, card, false, false);
                self.add_to_deck(id, false);
                // The `buying_card` contexts (here and below) have no effect in any joker.
            }
        }
        if set == CenterSet::Joker {
            self.gs.current_round.jokers_purchased += 1.0;
        }
        if self.gs.modifier("inflation").is_some() {
            self.gs.inflation += 1.0;
            self.queue(Event::SetCostAll);
        }
        if cost != 0.0 {
            self.ease_dollars(-cost);
        }
    }

    /// `buy_from_shop`'s event for the "buy and use" button (button_callbacks.lua:2414-2475,
    /// G46): the consumable is paid for and used at once, never landing in the consumables.
    pub(crate) fn buy_and_use(&mut self, id: u32) {
        let card = self
            .gs
            .areas
            .remove_card(AreaId::ShopJokers, Some(id))
            .expect("the bought card");
        let cost = card.cost;
        self.gs.areas.emplace(AreaId::Play, card, false, false);
        self.add_to_deck(id, false);
        if self.gs.modifier("inflation").is_some() {
            self.gs.inflation += 1.0;
            self.queue(Event::SetCostAll);
        }
        if cost != 0.0 {
            self.ease_dollars(-cost);
        }
        // `G.FUNCS.use_card(e, true)`. Ankh's check_use stops it after the purchase: the card
        // is in no area and never removed, so it is simply gone.
        if self.check_use(id) {
            self.gs.areas.remove_card(AreaId::Play, Some(id));
            return;
        }
        self.use_consumable_card(id, AreaId::Play);
    }

    /// `G.FUNCS.sell_card`: `card:sell_card()`, then the other jokers see the sale.
    pub(crate) fn sell_card(&mut self, id: u32) {
        let mut ctx = Ctx::new(Kind::SellingSelf);
        self.calculate_joker(id, &mut ctx);
        self.queue(Event::SellPayout(id));
        for j in self.joker_ids() {
            if j != id {
                let mut ctx = Ctx::new(Kind::SellingCard);
                ctx.card = Some(id);
                self.calculate_joker(j, &mut ctx);
            }
        }
    }

    /// `Card:sell_card`'s event (card.lua:1612-1626): the sale price, the card dissolves, and
    /// selling a joker disables Verdant Leaf.
    pub(crate) fn sell_payout(&mut self, id: u32) {
        let Some(card) = self.gs.areas.card(id) else {
            return;
        };
        let joker = card.set() == CenterSet::Joker;
        let price = card.sell_cost;
        self.ease_dollars(price);
        self.queue(Event::RemoveCard(id));
        if joker && self.blind_name() == "Verdant Leaf" {
            self.queue(Event::Call(|g, _, _| g.blind_disable(), 0, 0.0));
        }
    }

    /// `G.FUNCS.toggle_shop`: the jokers see the shop end, then it closes.
    pub(crate) fn toggle_shop(&mut self) {
        if !self.rt.shop_open {
            return;
        }
        for j in self.joker_ids() {
            let mut ctx = Ctx::new(Kind::EndingShop);
            self.calculate_joker(j, &mut ctx);
        }
        self.queue(Event::LeaveShop);
    }

    /// `toggle_shop`'s second event (button_callbacks.lua:2496-2509): `G.shop:remove()` removes
    /// every card left in the shop, then the blind-select screen comes up.
    pub(crate) fn leave_shop(&mut self) {
        // The removed areas stay reachable (and in BalatroBot's view) with these counts.
        let count = |a: AreaId| self.gs.areas.get(a).len() as f64;
        self.rt.areas.stale_shop = Some(crate::game::StaleShop {
            jokers: (count(AreaId::ShopJokers), self.gs.shop_joker_max),
            vouchers: (count(AreaId::ShopVouchers), self.rt.areas.voucher_limit),
            booster: (count(AreaId::ShopBooster), 2.0),
        });
        for area in [
            AreaId::ShopJokers,
            AreaId::ShopVouchers,
            AreaId::ShopBooster,
        ] {
            for id in self.gs.areas.ids(area) {
                self.remove_card(id);
            }
        }
        self.rt.shop_open = false;
        self.set_state(LuaState::BlindSelect, false);
    }
}

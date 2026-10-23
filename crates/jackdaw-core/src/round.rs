//! The end of a round: `end_round` (state_events.lua:87-288), round evaluation
//! (`G.FUNCS.evaluate_round`, state_events.lua:1135-1208) and the per-round resets
//! (common_events.lua:2271-2324).

use crate::area::AreaId;
use crate::event::Event;
use crate::game::{Game, LuaState};
use crate::hand_eval::HandType;
use crate::joker::{Ctx, Kind};
use crate::state::BlindSlot;

impl Game {
    /// `end_round()`: queued.
    pub(crate) fn end_round(&mut self) {
        self.queue(Event::EndRound);
    }

    /// `end_round`'s event (state_events.lua:91-286).
    pub(crate) fn end_round_body(&mut self) {
        let blind_chips = self.gs.blind.as_ref().map_or(0.0, |b| b.chips);
        let mut game_over = crate::lua::sub(self.gs.chips, blind_chips) < 0.0;
        for id in self.gs.areas.ids(AreaId::Jokers) {
            let mut ctx = Ctx::new(Kind::EndOfRound);
            ctx.game_over = game_over;
            if let Some(ret) = self.calculate_joker(id, &mut ctx)
                && ret.saved
            {
                game_over = false;
            }
            self.calculate_rental(id);
            self.calculate_perishable(id);
        }
        let boss = self.blind_type() == Some(BlindSlot::Boss);
        if self.gs.round_resets.ante == self.gs.win_ante && boss {
            self.gs.won = true;
        }
        if game_over {
            // The rent and the self-destructions queued above never run (G265).
            self.game_over();
            return;
        }
        self.gs.unused_discards += self.gs.current_round.discards_left;
        if boss {
            // The most played hand. The loop never updates `_order` (it stays 100), so every
            // tie replaces the pick: the most played hand that `pairs` visits last wins.
            let mut best = (HandType::HighCard, -1.0);
            for h in crate::hand_eval::PAIRS_ORDER {
                let played = self.gs.hand(h).played;
                if played >= best.1 {
                    best = (h, played);
                }
            }
            self.gs.current_round.most_played_poker_hand = best.0;
        }
        for id in self.gs.areas.ids(AreaId::Hand) {
            self.end_of_round_hand_card(id);
        }
        self.draw_from_hand_to_discard();
        if boss {
            if let Some(ante) = self.modifier_num("set_eternal_ante")
                && self.gs.round_resets.ante == ante
            {
                for c in &mut self.gs.areas.jokers {
                    c.set_eternal(true);
                }
            }
            if let Some(ante) = self.modifier_num("set_joker_slots_ante")
                && self.gs.round_resets.ante == ante
            {
                self.gs.joker_slots = 0.0;
            }
            self.queue(Event::EaseAnte(1.0));
        }
        self.queue(Event::DiscardToDeck);
        self.queue(Event::EndRoundFinal);
    }

    /// A numeric `G.GAME.modifiers[key]`.
    pub(crate) fn modifier_num(&self, key: &str) -> Option<f64> {
        self.gs.modifiers.get(key).and_then(|v| v.as_num())
    }

    /// The end-of-round effects of one held card (state_events.lua:171-233): Gold Cards pay,
    /// Blue Seals make the last hand's Planet, Red Seals and Mime repeat them.
    fn end_of_round_hand_card(&mut self, id: u32) {
        // reps = {1}; repetitions are appended during the first pass only.
        let mut reps = 1usize;
        let mut j = 0;
        while j < reps {
            let mut effects = vec![self.get_end_of_round_effect(id)];
            for k in self.gs.areas.ids(AreaId::Jokers) {
                let mut ctx = Ctx::new(Kind::EndOfRound);
                ctx.cardarea = Some(AreaId::Hand);
                ctx.other_card = Some(id);
                ctx.individual = true;
                if let Some(ret) = self.calculate_joker(k, &mut ctx) {
                    effects.push(ret);
                }
            }
            if j == 0 {
                // Red Seal: only when the card has an effect.
                let has_effect = !effects[0].is_empty() || effects.len() > 1;
                if has_effect && let Some(n) = self.red_seal_repetitions(id) {
                    reps += n;
                }
                for k in self.gs.areas.ids(AreaId::Jokers) {
                    let mut ctx = Ctx::new(Kind::EndOfRound);
                    ctx.cardarea = Some(AreaId::Hand);
                    ctx.other_card = Some(id);
                    ctx.repetition = true;
                    ctx.card_effects = effects.iter().map(|e| !e.is_empty()).collect();
                    if let Some(ret) = self.eval_joker(k, &mut ctx) {
                        reps += ret.repetitions.unwrap_or(0.0) as usize;
                    }
                }
            }
            for e in &effects {
                if let Some(d) = e.h_dollars {
                    self.queue(Event::EaseDollars(d));
                }
            }
            j += 1;
        }
    }

    /// `Card:get_end_of_round_effect()` (card.lua:1033-1066).
    pub(crate) fn get_end_of_round_effect(&mut self, id: u32) -> crate::scoring::Effect {
        let mut ret = crate::scoring::Effect::default();
        let Some(c) = self.gs.areas.card(id) else {
            return ret;
        };
        if c.debuff {
            return ret;
        }
        if c.ability.h_dollars > 0.0 {
            ret.h_dollars = Some(c.ability.h_dollars);
            ret.card = Some(id);
        }
        if c.seal == Some(crate::card::Seal::Blue)
            && (self.gs.areas.consumables.len() as f64) + self.rt.consumeable_buffer
                < self.gs.consumable_slots
        {
            self.rt.consumeable_buffer += 1.0;
            self.queue(Event::BlueSealPlanet);
            ret.effect = true;
        }
        ret
    }

    /// `eval_card(card, {repetition_only = true, ...}).seals.repetitions`: a Red Seal repeats
    /// the card once.
    pub(crate) fn red_seal_repetitions(&self, id: u32) -> Option<usize> {
        let c = self.gs.areas.card(id)?;
        (!c.debuff && c.seal == Some(crate::card::Seal::Red)).then_some(1)
    }

    /// `Card:calculate_rental()` (card.lua:2271-2276).
    fn calculate_rental(&mut self, id: u32) {
        if self.gs.areas.card(id).is_some_and(|c| c.ability.rental) {
            self.queue(Event::EaseDollars(-self.gs.rental_rate));
        }
    }

    /// `Card:calculate_perishable()` (card.lua:2278-2289).
    fn calculate_perishable(&mut self, id: u32) {
        let Some(c) = self.gs.areas.card_mut(id) else {
            return;
        };
        let tally = c.ability.perish_tally.unwrap_or(0.0);
        if c.ability.perishable && tally > 0.0 {
            if tally == 1.0 {
                c.ability.perish_tally = Some(0.0);
                // `self:set_debuff()` with no argument: the spent-perishable branch debuffs.
                self.set_debuff(id, false);
            } else {
                c.ability.perish_tally = Some(tally - 1.0);
            }
        }
    }

    /// `end_round`'s last event (state_events.lua:251-283).
    pub(crate) fn end_round_final(&mut self) {
        self.set_state(LuaState::RoundEval, false);
        match self.rt.round_blind {
            Some(b) if Game::is_small(b) => {
                self.gs
                    .round_resets
                    .set_blind_state(BlindSlot::Small, "Defeated");
            }
            Some(b) if Game::is_big(b) => {
                self.gs
                    .round_resets
                    .set_blind_state(BlindSlot::Big, "Defeated");
            }
            _ => {
                self.gs.current_round.voucher = self.get_next_voucher_key(false);
                self.gs
                    .round_resets
                    .set_blind_state(BlindSlot::Boss, "Defeated");
                for a in AreaId::PLAYING {
                    for c in self.gs.areas.get_mut(a) {
                        c.ability.played_this_ante = false;
                    }
                }
            }
        }
        if let Some(t) = self.gs.round_resets.temp_handsize.take() {
            self.change_size(AreaId::Hand, -t);
        }
        if self.gs.round_resets.temp_reroll_cost.take().is_some() {
            self.calculate_reroll_cost(true);
        }
        self.reset_idol_card();
        self.reset_mail_rank();
        self.reset_ancient_card();
        self.reset_castle_card();
        for a in AreaId::PLAYING {
            for c in self.gs.areas.get_mut(a) {
                c.ability.discarded = false;
                c.ability.forced_selection = false;
            }
        }
    }

    /// The playing cards a per-round pick draws from: everything but Stone Cards, by sort id.
    fn non_stone_cards(&self) -> Vec<u32> {
        // Marble's pending stones are Stone Cards: the areas hold every other playing card.
        let mut ids: Vec<u32> = AreaId::PLAYING
            .into_iter()
            .flat_map(|a| self.gs.areas.get(a).iter())
            .filter(|c| !c.is_stone())
            .map(|c| c.sort_id)
            .collect();
        ids.sort_unstable();
        ids
    }

    /// `pseudoseed(key .. ante)`.
    fn ante_seed(&mut self, key: &str) -> f64 {
        let k = crate::rng::Key::new(key).n(self.gs.round_resets.ante);
        self.gs.rng.seed(&k)
    }

    /// `reset_idol_card()` (common_events.lua:2271-2286).
    pub(crate) fn reset_idol_card(&mut self) {
        use crate::card::{Rank, Suit};
        let cr = &mut self.gs.current_round;
        cr.idol_card = (Suit::Spades, Rank::Ace);
        let valid = self.non_stone_cards();
        if !valid.is_empty() {
            let seed = self.ante_seed("idol");
            let pick = valid[crate::rng::element_index(valid.len(), seed)];
            let base = self
                .gs
                .areas
                .card(pick)
                .and_then(|c| c.base)
                .expect("playing card");
            self.gs.current_round.idol_card = (base.suit, base.value);
        }
    }

    /// `reset_mail_rank()` (common_events.lua:2288-2301).
    pub(crate) fn reset_mail_rank(&mut self) {
        self.gs.current_round.mail_card = crate::card::Rank::Ace;
        let valid = self.non_stone_cards();
        if !valid.is_empty() {
            let seed = self.ante_seed("mail");
            let pick = valid[crate::rng::element_index(valid.len(), seed)];
            let base = self
                .gs
                .areas
                .card(pick)
                .and_then(|c| c.base)
                .expect("playing card");
            self.gs.current_round.mail_card = base.value;
        }
    }

    /// `reset_ancient_card()` (common_events.lua:2303-2310).
    pub(crate) fn reset_ancient_card(&mut self) {
        use crate::card::Suit;
        let current = self.gs.current_round.ancient_card;
        let suits: Vec<Suit> = [Suit::Spades, Suit::Hearts, Suit::Clubs, Suit::Diamonds]
            .into_iter()
            .filter(|&s| Some(s) != current)
            .collect();
        let seed = self.ante_seed("anc");
        self.gs.current_round.ancient_card =
            Some(suits[crate::rng::element_index(suits.len(), seed)]);
    }

    /// `reset_castle_card()` (common_events.lua:2312-2324).
    pub(crate) fn reset_castle_card(&mut self) {
        self.gs.current_round.castle_card = crate::card::Suit::Spades;
        let valid = self.non_stone_cards();
        if !valid.is_empty() {
            let seed = self.ante_seed("cas");
            let pick = valid[crate::rng::element_index(valid.len(), seed)];
            let base = self
                .gs
                .areas
                .card(pick)
                .and_then(|c| c.base)
                .expect("playing card");
            self.gs.current_round.castle_card = base.suit;
        }
    }

    /// `G.FUNCS.evaluate_round()` (state_events.lua:1135-1208): the cash-out total.
    pub(crate) fn evaluate_round(&mut self) {
        let mut dollars = 0.0;
        let blind = self.gs.blind.clone();
        let beaten = blind
            .as_ref()
            .is_some_and(|b| crate::lua::sub(self.gs.chips, b.chips) >= 0.0);
        if beaten {
            dollars += blind.as_ref().map_or(0.0, |b| b.dollars);
        }
        self.queue(Event::BlindDefeat);
        self.back_trigger_eval();
        let cr = &self.gs.current_round;
        if cr.hands_left > 0.0 && self.gs.modifier("no_extra_hand_money").is_none() {
            let per = self.modifier_num("money_per_hand").unwrap_or(1.0);
            dollars += cr.hands_left * per;
        }
        if cr.discards_left > 0.0
            && let Some(per) = self.modifier_num("money_per_discard")
        {
            dollars += cr.discards_left * per;
        }
        for id in self.gs.areas.ids(AreaId::Jokers) {
            if let Some(d) = self.calculate_dollar_bonus(id) {
                dollars += d;
            }
        }
        dollars += self.apply_tags_eval();
        if self.gs.dollars >= 5.0 && self.gs.modifier("no_interest").is_none() {
            let interest = self.gs.interest_amount
                * crate::lua::floor(self.gs.dollars / 5.0).min(self.gs.interest_cap / 5.0);
            dollars += interest;
        }
        self.queue(Event::SetCashout(dollars));
    }

    /// `Card:calculate_dollar_bonus()` (card.lua:1655-1679).
    pub(crate) fn calculate_dollar_bonus(&mut self, id: u32) -> Option<f64> {
        let c = self.gs.areas.card(id)?;
        if c.debuff || !c.is_joker() {
            return None;
        }
        let a = &c.ability;
        match c.name() {
            "Golden Joker" => a.extra.num(),
            "Cloud 9" => {
                let tally = self.nine_tally();
                (tally > 0.0).then(|| a.extra.num().unwrap_or(0.0) * tally)
            }
            "Rocket" => a.extra.table().and_then(|t| t.num("dollars")),
            "Satellite" => {
                let planets = self
                    .gs
                    .consumeable_usage
                    .by_key
                    .keys()
                    .filter(|k| {
                        jackdaw_data::center_by_key(k)
                            .is_some_and(|c| c.get().set == jackdaw_data::CenterSet::Planet)
                    })
                    .count() as f64;
                (planets > 0.0).then(|| a.extra.num().unwrap_or(0.0) * planets)
            }
            "Delayed Gratification" => {
                let cr = &self.gs.current_round;
                (cr.discards_used == 0.0 && cr.discards_left > 0.0)
                    .then(|| cr.discards_left * a.extra.num().unwrap_or(0.0))
            }
            _ => None,
        }
    }

    /// Cloud 9's `nine_tally` (per frame, card.lua:4191-4196): playing cards with rank 9.
    pub(crate) fn nine_tally(&self) -> f64 {
        self.playing_cards()
            .filter(|c| c.get_id() == Some(9))
            .count() as f64
    }
}

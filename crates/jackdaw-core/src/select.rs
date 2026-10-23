//! The blind-select screen: `Game:update_blind_select` (game.lua:3248-3298) and the gameplay
//! part of `create_UIBox_blind_select` / `create_UIBox_blind_choice` (UI_definitions.lua:1415-1530),
//! plus the ante's blinds and tags: `reset_blinds`, `get_new_boss` and `get_next_tag_key`
//! (common_events.lua:1914-1925, 2326-2383).

use jackdaw_data::{BlindId, TagId, tags};

use crate::event::Event;
use crate::game::Game;
use crate::hand_eval::PAIRS_ORDER;
use crate::state::BlindSlot;

impl Game {
    /// `Game:update_blind_select` (game.lua:3248-3298): once per arrival, an event that queues
    /// the screen's construction.
    pub(crate) fn update_blind_select(&mut self) -> bool {
        self.rt.shop_open = false;
        if self.rt.state_complete {
            return false;
        }
        self.rt.state_complete = true;
        self.queue(Event::BlindSelectOpen);
        true
    }

    /// The screen's construction (game.lua:3260-3295): `create_UIBox_blind_select()`, then the
    /// held tags' `immediate` pass and their `new_blind_choice` pass.
    pub(crate) fn blind_select_build(&mut self) {
        let rr = &self.gs.round_resets;
        let done =
            |slot: BlindSlot| matches!(rr.blind_state(slot), "Defeated" | "Skipped" | "Hide");
        let on_deck = if !done(BlindSlot::Small) {
            BlindSlot::Small
        } else if !done(BlindSlot::Big) {
            BlindSlot::Big
        } else {
            BlindSlot::Boss
        };
        self.gs.blind_on_deck = Some(on_deck);
        // `create_UIBox_blind_choice` for each blind: the blind on deck is the one to select,
        // and this ante's Orbital Tag hands are rolled once (UI_definitions.lua:1486-1516).
        self.gs.round_resets.set_blind_state(on_deck, "Select");
        let ante = crate::rng::Key::new("").n(self.gs.round_resets.ante);
        if !self.gs.orbital_choices.contains_key(&*ante) {
            let visible: Vec<_> = PAIRS_ORDER
                .into_iter()
                .filter(|&h| self.gs.hand(h).visible)
                .collect();
            let mut roll = || {
                let seed = self.gs.rng.seed("orbital");
                visible[crate::rng::element_index(visible.len(), seed)]
            };
            let hands = [roll(), roll(), roll()];
            self.gs.orbital_choices.insert(ante.to_string(), hands);
        }
        self.apply_tags_immediate();
        self.apply_tags_new_blind_choice();
    }

    /// `G.FUNCS.skip_blind` (button_callbacks.lua:2727-2775): the blind on deck's tag joins the
    /// run and the next blind comes up; an event then runs the jokers' `skip_blind` context and
    /// the tags' `immediate` and `new_blind_choice` passes.
    pub(crate) fn skip_blind(&mut self, slot: BlindSlot) {
        self.gs.skips += 1.0;
        let tag = self
            .gs
            .round_resets
            .blind_tags
            .and_then(|t| t[slot.index()])
            .expect("the skipped blind has a tag");
        let tag = self.new_tag(tag, self.orbital_choice(slot));
        self.add_tag(tag);
        let skip_to = match slot {
            BlindSlot::Small => BlindSlot::Big,
            _ => BlindSlot::Boss,
        };
        self.gs.round_resets.set_blind_state(slot, "Skipped");
        self.gs.round_resets.set_blind_state(skip_to, "Select");
        self.gs.blind_on_deck = Some(skip_to);
        self.queue(Event::SkipBlindBody);
    }

    /// `skip_blind`'s event (button_callbacks.lua:2762-2773).
    pub(crate) fn skip_blind_body(&mut self) {
        for j in self.joker_ids() {
            let mut ctx = crate::joker::Ctx::new(crate::joker::Kind::SkipBlind);
            self.calculate_joker(j, &mut ctx);
        }
        self.apply_tags_immediate();
        self.apply_tags_new_blind_choice();
    }

    /// `G.FUNCS.reroll_boss` (button_callbacks.lua:2718-2765): Director's Cut or Retcon for $10,
    /// or a Boss Tag for free; an event then picks the new boss.
    pub(crate) fn reroll_boss(&mut self, from_tag: bool) {
        self.gs.round_resets.boss_rerolled = true;
        if !from_tag {
            self.ease_dollars(-10.0);
        }
        self.queue(Event::RerollBossBody);
    }

    /// `reroll_boss`'s timed event: the new boss, its blind choice rebuilt (which marks the
    /// blind on deck again), then the tags' `new_blind_choice` pass.
    pub(crate) fn reroll_boss_body(&mut self) {
        let boss = self.get_new_boss();
        self.gs.round_resets.blind_choices[BlindSlot::Boss.index()] = Some(boss);
        if let Some(on_deck) = self.gs.blind_on_deck {
            self.gs.round_resets.set_blind_state(on_deck, "Select");
        }
        self.apply_tags_new_blind_choice();
    }

    /// `G.FUNCS.reroll_boss_button` (button_callbacks.lua:2784-2797).
    pub(crate) fn can_reroll_boss(&self) -> bool {
        let used = &self.gs.sets.used_vouchers;
        self.gs.dollars - self.gs.bankrupt_at - 10.0 >= 0.0
            && (used.contains("v_retcon")
                || (used.contains("v_directors_cut") && !self.gs.round_resets.boss_rerolled))
    }

    /// The Orbital Tag hand rolled for `slot` this ante, if any.
    pub(crate) fn orbital_choice(&self, slot: BlindSlot) -> Option<crate::hand_eval::HandType> {
        let ante = crate::rng::Key::new("").n(self.gs.round_resets.ante);
        self.gs.orbital_choices.get(&*ante).map(|h| h[slot.index()])
    }

    /// `reset_blinds()` (common_events.lua:2326-2336): after a boss, the next ante's blinds.
    pub(crate) fn reset_blinds(&mut self) {
        if self.gs.round_resets.blind_state(BlindSlot::Boss) == "Defeated" {
            for slot in BlindSlot::ALL {
                self.gs.round_resets.set_blind_state(slot, "Upcoming");
            }
            self.gs.blind_on_deck = Some(BlindSlot::Small);
            let boss = self.get_new_boss();
            self.gs.round_resets.blind_choices[BlindSlot::Boss.index()] = Some(boss);
            self.gs.round_resets.boss_rerolled = false;
        }
    }

    /// `get_new_boss()` (common_events.lua:2338-2383): a boss of this ante's kind, among the
    /// least used, picked from `'boss'`.
    pub(crate) fn get_new_boss(&mut self) -> BlindId {
        let ante = self.gs.round_resets.ante;
        let a = ante.max(1.0);
        let win = self.gs.win_ante;
        let mut eligible: Vec<(&'static str, f64)> = Vec::new();
        for b in jackdaw_data::BLINDS.iter() {
            let Some(boss) = &b.boss else { continue };
            let ok = if boss.showdown {
                crate::lua::modulo(ante, win) == 0.0 && ante >= 2.0
            } else {
                f64::from(boss.min) <= a && (crate::lua::modulo(a, win) != 0.0 || ante < 2.0)
            };
            if ok && !self.gs.sets.banned_keys.contains(b.key) {
                let used = self
                    .gs
                    .sets
                    .bosses_used
                    .get(b.key)
                    .copied()
                    .unwrap_or_else(|| panic!("bosses_used has no {}", b.key));
                eligible.push((b.key, used));
            }
        }
        let min_use = eligible
            .iter()
            .map(|&(_, u)| u)
            .fold(100.0, |m: f64, u| if u <= m { u } else { m });
        eligible.retain(|&(_, u)| u <= min_use);
        // `pseudorandom_element` over a table keyed by boss key: sorted by key.
        eligible.sort_unstable_by_key(|&(k, _)| k);
        let seed = self.gs.rng.seed("boss");
        let key = eligible[crate::rng::element_index(eligible.len(), seed)].0;
        *self.gs.sets.bosses_used.get_mut(key).expect("boss counted") += 1.0;
        jackdaw_data::blind_by_key(key).expect("a boss blind")
    }

    /// `get_next_tag_key(append)` (common_events.lua:1914-1925) over the Tag pool
    /// (`get_current_pool('Tag')`, common_events.lua:1982-1986). The sim's profile has
    /// discovered every center, so a tag's `requires` is always met.
    pub(crate) fn get_next_tag_key(&mut self, append: &str) -> TagId {
        let ante = self.gs.round_resets.ante;
        let mut entries: Vec<Option<TagId>> = Vec::with_capacity(jackdaw_data::TAGS.len());
        let mut size = 0;
        for (i, t) in jackdaw_data::TAGS.iter().enumerate() {
            let add = t.min_ante.is_none_or(|m| f64::from(m) <= ante);
            if add && !self.gs.sets.banned_keys.contains(t.key) {
                entries.push(Some(TagId(i as u8)));
                size += 1;
            } else {
                entries.push(None);
            }
        }
        if size == 0 {
            entries = vec![Some(tags::TAG_HANDY)];
        }
        let key = crate::rng::Key::new("Tag").s(append).n(ante);
        let n = entries.len();
        let mut pick = entries[crate::rng::element_index(n, self.gs.rng.seed(&key))];
        let mut it = 1;
        while pick.is_none() {
            it += 1;
            let seed = self
                .gs
                .rng
                .seed(&crate::rng::Key::new(&key).s("_resample").i(it));
            pick = entries[crate::rng::element_index(n, seed)];
        }
        pick.expect("resampled")
    }
}

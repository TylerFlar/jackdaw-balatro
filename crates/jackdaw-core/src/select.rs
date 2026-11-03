//! The blind-select screen: `Game:update_blind_select` (game.lua:3248-3298) and the gameplay
//! part of `create_UIBox_blind_select` / `create_UIBox_blind_choice` (UI_definitions.lua:1415-1530),
//! plus the ante's blinds and tags: `reset_blinds`, `get_new_boss` and `get_next_tag_key`
//! (common_events.lua:1914-1925, 2326-2383).

use jackdaw_data::{BlindId, TagId, tags};

use crate::event::Event;
use crate::game::Game;
use crate::hand_eval::PAIRS_ORDER;
use crate::state::BlindSlot;

/// The bosses (`G.P_BLINDS` entries with a `boss` table), sorted by key.
fn bosses_by_key() -> &'static [BlindId] {
    static BOSSES: std::sync::OnceLock<Vec<BlindId>> = std::sync::OnceLock::new();
    BOSSES.get_or_init(|| {
        let mut ids: Vec<BlindId> = (0..jackdaw_data::BLINDS.len())
            .filter(|&i| jackdaw_data::BLINDS[i].boss.is_some())
            .map(|i| BlindId(i as u8))
            .collect();
        ids.sort_unstable_by_key(|id| jackdaw_data::BLINDS[usize::from(id.0)].key);
        ids
    })
}

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
            let mut visible = [PAIRS_ORDER[0]; PAIRS_ORDER.len()];
            let mut n = 0;
            for h in PAIRS_ORDER {
                if self.gs.hand(h).visible {
                    visible[n] = h;
                    n += 1;
                }
            }
            let mut roll = || {
                let seed = self.gs.rng.seed("orbital");
                visible[crate::rng::element_index(n, seed)]
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
            && (used.contains(jackdaw_data::centers::V_RETCON)
                || (used.contains(jackdaw_data::centers::V_DIRECTORS_CUT)
                    && !self.gs.round_resets.boss_rerolled))
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
        let sets = &self.gs.sets;
        // `pseudorandom_element` over a table keyed by boss key: the bosses in key order.
        let mut eligible = [(BlindId(0), 0.0); jackdaw_data::BLINDS.len()];
        let mut n = 0;
        let mut min_use: f64 = 100.0;
        for &id in bosses_by_key() {
            let b = &jackdaw_data::BLINDS[usize::from(id.0)];
            let boss = b.boss.as_ref().expect("a boss");
            let ok = if boss.showdown {
                crate::lua::modulo(ante, win) == 0.0 && ante >= 2.0
            } else {
                f64::from(boss.min) <= a && (crate::lua::modulo(a, win) != 0.0 || ante < 2.0)
            };
            if ok && !sets.banned_keys.contains(b.key) {
                let used = sets
                    .bosses_used
                    .get(id)
                    .unwrap_or_else(|| panic!("bosses_used has no {}", b.key));
                if used <= min_use {
                    min_use = used;
                }
                eligible[n] = (id, used);
                n += 1;
            }
        }
        // The least used, still in key order.
        let mut m = 0;
        for i in 0..n {
            if eligible[i].1 <= min_use {
                eligible[m] = eligible[i];
                m += 1;
            }
        }
        let seed = self.gs.rng.seed("boss");
        let (id, used) = eligible[..m][crate::rng::element_index(m, seed)];
        self.gs.sets.bosses_used.set(id, used + 1.0);
        id
    }

    /// `get_next_tag_key(append)` (common_events.lua:1914-1925) over the Tag pool
    /// (`get_current_pool('Tag')`, common_events.lua:1982-1986). The sim's profile has
    /// discovered every center, so a tag's `requires` is always met.
    pub(crate) fn get_next_tag_key(&mut self, append: &str) -> TagId {
        // As with the center pools (`Game::pick_from_current_pool`), the pool isn't built: a
        // drawn slot is checked alone, and an empty pool falls back to the Handy Tag.
        let ante = self.gs.round_resets.ante;
        let in_pool = |g: &Game, i: usize| {
            let t = &jackdaw_data::TAGS[i];
            t.min_ante.is_none_or(|m| f64::from(m) <= ante)
                && !g.gs.sets.banned_keys.contains(t.key)
        };
        let key = crate::rng::Key::new("Tag").s(append).n(ante);
        let n = jackdaw_data::TAGS.len();
        let first = crate::rng::element_index(n, self.gs.rng.seed(&key));
        if in_pool(self, first) {
            return TagId(first as u8);
        }
        if !(0..n).any(|i| in_pool(self, i)) {
            return tags::TAG_HANDY;
        }
        let mut it = 1;
        loop {
            it += 1;
            let seed = self
                .gs
                .rng
                .seed(&crate::rng::Key::new(&key).s("_resample").i(it));
            let i = crate::rng::element_index(n, seed);
            if in_pool(self, i) {
                return TagId(i as u8);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run::RunSpec;

    /// The `get_new_boss` this replaced: the eligible bosses collected in blind order, the
    /// least used kept, sorted by key.
    fn get_new_boss_sorted(g: &mut Game) -> BlindId {
        let ante = g.gs.round_resets.ante;
        let a = ante.max(1.0);
        let win = g.gs.win_ante;
        let mut eligible: Vec<(&'static str, BlindId, f64)> = Vec::new();
        for (i, b) in jackdaw_data::BLINDS.iter().enumerate() {
            let Some(boss) = &b.boss else { continue };
            let ok = if boss.showdown {
                crate::lua::modulo(ante, win) == 0.0 && ante >= 2.0
            } else {
                f64::from(boss.min) <= a && (crate::lua::modulo(a, win) != 0.0 || ante < 2.0)
            };
            if ok && !g.gs.sets.banned_keys.contains(b.key) {
                let id = BlindId(i as u8);
                let used = g.gs.sets.bosses_used.get(id).expect("a count");
                eligible.push((b.key, id, used));
            }
        }
        let min_use = eligible
            .iter()
            .map(|&(_, _, u)| u)
            .fold(100.0, |m: f64, u| if u <= m { u } else { m });
        eligible.retain(|&(_, _, u)| u <= min_use);
        eligible.sort_unstable_by_key(|&(k, _, _)| k);
        let seed = g.gs.rng.seed("boss");
        let (_, id, used) = eligible[crate::rng::element_index(eligible.len(), seed)];
        g.gs.sets.bosses_used.set(id, used + 1.0);
        id
    }

    /// The same boss, the same draw and the same counts as the sorting version, over antes,
    /// win antes, use counts and bans.
    #[test]
    fn get_new_boss_matches_the_sorted_pick() {
        let base = Game::new_run(&RunSpec {
            back: jackdaw_data::centers::B_RED,
            stake: 1,
            seed: "BOSSPICK".to_owned(),
            challenge: None,
        })
        .unwrap();
        let bosses: Vec<BlindId> = base.gs.sets.bosses_used.iter().map(|(b, _)| b).collect();
        // A small deterministic generator for the states.
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        };
        let mut g = base.clone();
        let mut fresh = 0;
        for case in 0..20_000 {
            if case % 50 == 0 {
                g = base.clone();
                g.gs.rng = crate::rng::Rng::new(&format!("BOSS{case}"));
                g.gs.win_ante = [8.0, 8.0, 4.0, 6.0, 1.0][(next() % 5) as usize];
                g.gs.sets.banned_keys.clear();
                if next() % 3 == 0 {
                    for _ in 0..(next() % 6) {
                        let b = bosses[(next() % bosses.len() as u64) as usize];
                        let key = jackdaw_data::BLINDS[usize::from(b.0)].key;
                        g.gs.sets.banned_keys.insert(key.to_owned());
                    }
                }
                if next() % 4 == 0 {
                    for &b in &bosses {
                        g.gs.sets.bosses_used.set(b, (next() % 3) as f64);
                    }
                }
            }
            g.gs.round_resets.ante = [
                -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 12.0, 16.0, 39.0,
            ][(next() % 12) as usize];
            let ante = g.gs.round_resets.ante;
            let win = g.gs.win_ante;
            // Skip states with no eligible boss (both versions panic there).
            let any = bosses.iter().any(|&b| {
                let bl = &jackdaw_data::BLINDS[usize::from(b.0)];
                let boss = bl.boss.as_ref().unwrap();
                let ok = if boss.showdown {
                    crate::lua::modulo(ante, win) == 0.0 && ante >= 2.0
                } else {
                    f64::from(boss.min) <= ante.max(1.0)
                        && (crate::lua::modulo(ante.max(1.0), win) != 0.0 || ante < 2.0)
                };
                ok && !g.gs.sets.banned_keys.contains(bl.key)
            });
            if !any {
                continue;
            }
            let mut old = g.clone();
            let picked = g.get_new_boss();
            assert_eq!(picked, get_new_boss_sorted(&mut old), "case {case}");
            assert_eq!(g.gs, old.gs, "case {case}");
            fresh += 1;
        }
        assert!(fresh > 15_000);
    }
}

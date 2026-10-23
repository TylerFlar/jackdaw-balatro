//! Tags (`Tag`, tag.lua) and deck effects (`Back:trigger_effect`, back.lua:104-140) used by the
//! round flow. A tag that triggers is marked at once (`self.triggered = true`) and leaves
//! `G.GAME.tags` through `Tag:yep`'s last event (tag.lua:62-100).

use jackdaw_data::{TagId, tags};

use crate::area::AreaId;
use crate::event::Event;
use crate::game::Game;

impl Game {
    /// `add_tag(tag)` (UI_definitions.lua:1252-1272): the held tags see it (`tag_add`), then it
    /// joins `G.GAME.tags`.
    pub(crate) fn add_tag(&mut self, tag: TagId) {
        let held: Vec<TagId> = self.gs.tags.clone();
        for t in held {
            if t == tags::TAG_DOUBLE && tag != tags::TAG_DOUBLE {
                // Double Tag copies the new tag (tag.lua:284-299).
                return self.unported("Double Tag copying a tag");
            }
        }
        self.gs.tags.push(tag);
        self.gs.sim.tag_history.push(tag);
    }

    /// `Tag:yep`'s removal event.
    pub(crate) fn remove_tag(&mut self, tag: TagId) {
        if let Some(i) = self.gs.tags.iter().position(|&t| t == tag) {
            self.gs.tags.remove(i);
        }
    }

    /// `tag:apply_to_run({type = 'round_start_bonus'})` for every held tag
    /// (game.lua:3214-3216): the Juggle Tag (tag.lua:334-343).
    pub(crate) fn apply_tags_round_start_bonus(&mut self) {
        for tag in self.gs.tags.clone() {
            if tag == tags::TAG_JUGGLE {
                let h = tag
                    .get()
                    .config("h_size")
                    .and_then(|v| v.as_num())
                    .expect("Juggle Tag's hand size");
                self.queue(Event::RemoveTag(tag));
                self.change_size(AreaId::Hand, h);
                let t = self.gs.round_resets.temp_handsize.unwrap_or(0.0);
                self.gs.round_resets.temp_handsize = Some(t + h);
            }
        }
    }

    /// `tag:apply_to_run({type = 'eval'})` for every held tag (state_events.lua:1180-1188): the
    /// Investment Tag pays after a boss (tag.lua:115-129). Returns the dollars.
    pub(crate) fn apply_tags_eval(&mut self) -> f64 {
        let mut dollars = 0.0;
        for tag in self.gs.tags.clone() {
            if tag == tags::TAG_INVESTMENT && self.gs.sim.last_blind_was_boss {
                self.queue(Event::RemoveTag(tag));
                dollars += tag
                    .get()
                    .config("dollars")
                    .and_then(|v| v.as_num())
                    .expect("Investment Tag's dollars");
            }
        }
        dollars
    }

    /// `G.GAME.selected_back:trigger_effect({context = 'eval'})`: the Anaglyph Deck makes a
    /// Double Tag after a boss (back.lua:111-120).
    pub(crate) fn back_trigger_eval(&mut self) {
        if self.gs.back.get().name == "Anaglyph Deck" && self.gs.sim.last_blind_was_boss {
            self.queue(Event::AddTag(tags::TAG_DOUBLE));
        }
    }

    /// `trigger_effect({context = 'final_scoring_step'})`: the Plasma Deck balances chips and
    /// mult (back.lua:124-140).
    pub(crate) fn back_final_scoring_step(&self, chips: f64, mult: f64) -> Option<(f64, f64)> {
        if self.gs.back.get().name != "Plasma Deck" {
            return None;
        }
        let tot = crate::lua::add(chips, mult);
        let half = crate::lua::floor(tot / 2.0);
        Some((half, half))
    }
}

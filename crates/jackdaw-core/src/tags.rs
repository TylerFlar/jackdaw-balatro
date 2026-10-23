//! Tags (`Tag`, tag.lua) and deck effects (`Back:trigger_effect`, back.lua:104-140).
//!
//! `Tag:apply_to_run(context)` (tag.lua:115-450) is one pass per context type here; each pass
//! visits `G.GAME.tags` in order like the Lua callers' loops. A tag that fires is marked
//! `triggered` at once and leaves `G.GAME.tags` through the last event `Tag:yep` / `Tag:nope`
//! queue (tag.lua:24-100); its own effect, when it has one, is the event `yep` queues before
//! that. The UI events of `yep` and `nope` are not queued.

use jackdaw_data::{TagId, tags};

use crate::area::AreaId;
use crate::card::{Card, EditionKind};
use crate::event::Event;
use crate::game::Game;
use crate::hand_eval::HandType;
use crate::pools::{CardSpec, PoolType};
use crate::state::HeldTag;

/// The context type a tag fires in (`tag.config.type`).
fn tag_type(tag: TagId) -> &'static str {
    tag.get()
        .config("type")
        .and_then(|v| v.as_str())
        .unwrap_or("")
}

fn tag_num(tag: TagId, key: &str) -> f64 {
    tag.get()
        .config(key)
        .and_then(|v| v.as_num())
        .unwrap_or_else(|| panic!("{}: config.{key}", tag.key()))
}

impl Game {
    /// `add_tag(tag)` (UI_definitions.lua:1252-1272): the held tags see it (`tag_add`), then it
    /// joins `G.GAME.tags`.
    pub(crate) fn add_tag(&mut self, tag: HeldTag) {
        let n = self.gs.tags.len();
        for i in 0..n {
            let held = self.gs.tags[i];
            // The Double Tag copies any other tag (tag.lua:318-333).
            if !held.triggered && held.id == tags::TAG_DOUBLE && tag.id != tags::TAG_DOUBLE {
                self.gs.tags[i].triggered = true;
                if let Some(j) = self
                    .gs
                    .sim
                    .tag_history
                    .iter()
                    .position(|&t| t == tags::TAG_DOUBLE)
                {
                    // `tag_history` keeps awarded tags minus consumed Double Tags.
                    self.gs.sim.tag_history.remove(j);
                }
                self.queue(Event::AddTag(tag.id, tag.orbital_hand));
                self.queue(Event::RemoveTag(held.id));
            }
        }
        self.gs.tags.push(HeldTag {
            triggered: false,
            ..tag
        });
        self.gs.sim.tag_history.push(tag.id);
    }

    /// `Tag(key)` with `G.orbital_hand` set to `orbital_hand` (a Double Tag's copy) or, for an
    /// Orbital Tag made on the blind-select screen, this ante's hand for `blind`
    /// (`Tag:set_ability`, tag.lua:101-112).
    pub(crate) fn new_tag(&self, id: TagId, orbital_hand: Option<HandType>) -> HeldTag {
        let mut tag = HeldTag::new(id);
        if id == tags::TAG_ORBITAL {
            tag.orbital_hand = orbital_hand;
        }
        tag
    }

    /// `Tag:remove()` at the end of `yep` / `nope`: the tag leaves `G.GAME.tags`. Triggered tags
    /// of one key are interchangeable, so the first is removed.
    pub(crate) fn remove_tag(&mut self, tag: TagId) {
        if let Some(i) = self.gs.tags.iter().position(|t| t.id == tag && t.triggered) {
            self.gs.tags.remove(i);
        }
    }

    /// Mark the tag at `i` triggered and queue `Tag:yep`'s effect (if any) and removal.
    fn tag_yep(&mut self, i: usize, effect: Option<Event>) {
        let tag = self.gs.tags[i].id;
        self.gs.tags[i].triggered = true;
        if let Some(e) = effect {
            self.queue(e);
        }
        self.queue(Event::RemoveTag(tag));
    }

    /// The untriggered tag at `i` fires in `ty`.
    fn tag_ready(&self, i: usize, ty: &str) -> bool {
        let t = self.gs.tags[i];
        !t.triggered && tag_type(t.id) == ty
    }

    /// `apply_to_run({type = 'round_start_bonus'})` for every held tag (game.lua:3214-3216): the
    /// Juggle Tag (tag.lua:334-343).
    pub(crate) fn apply_tags_round_start_bonus(&mut self) {
        for i in 0..self.gs.tags.len() {
            if !self.tag_ready(i, "round_start_bonus") {
                continue;
            }
            let tag = self.gs.tags[i].id;
            if tag == tags::TAG_JUGGLE {
                let h = tag_num(tag, "h_size");
                self.tag_yep(i, None);
                self.change_size(AreaId::Hand, h);
                let t = self.gs.round_resets.temp_handsize.unwrap_or(0.0);
                self.gs.round_resets.temp_handsize = Some(t + h);
            }
        }
    }

    /// `apply_to_run({type = 'eval'})` for every held tag (state_events.lua:1180-1188): the
    /// Investment Tag pays after a boss (tag.lua:117-129). Returns the dollars.
    pub(crate) fn apply_tags_eval(&mut self) -> f64 {
        let mut dollars = 0.0;
        for i in 0..self.gs.tags.len() {
            if !self.tag_ready(i, "eval") {
                continue;
            }
            let tag = self.gs.tags[i].id;
            if tag == tags::TAG_INVESTMENT && self.gs.sim.last_blind_was_boss {
                self.tag_yep(i, None);
                dollars += tag_num(tag, "dollars");
            }
        }
        dollars
    }

    /// `apply_to_run({type = 'immediate'})` for every held tag (tag.lua:130-206).
    pub(crate) fn apply_tags_immediate(&mut self) {
        for i in 0..self.gs.tags.len() {
            if !self.tag_ready(i, "immediate") {
                continue;
            }
            let held = self.gs.tags[i];
            let tag = held.id;
            match tag.get().name {
                "Top-up Tag" => {
                    let n = tag_num(tag, "spawn_jokers");
                    self.tag_yep(i, Some(Event::Call(top_up, 0, n)));
                }
                "Skip Tag" => {
                    self.tag_yep(i, None);
                    let x = self.gs.skips * tag_num(tag, "skip_bonus");
                    self.ease_dollars(x);
                }
                "Garbage Tag" => {
                    self.tag_yep(i, None);
                    let x = self.gs.unused_discards * tag_num(tag, "dollars_per_discard");
                    self.ease_dollars(x);
                }
                "Handy Tag" => {
                    self.tag_yep(i, None);
                    let x = self.gs.hands_played * tag_num(tag, "dollars_per_hand");
                    self.ease_dollars(x);
                }
                "Economy Tag" => {
                    self.tag_yep(i, None);
                    self.queue(Event::Call(economy, 0, tag_num(tag, "max")));
                }
                "Orbital Tag" => {
                    let hand = held.orbital_hand.expect("an Orbital Tag carries its hand");
                    self.level_up_hand(hand, tag_num(tag, "levels"));
                    self.tag_yep(i, None);
                }
                other => self.unported(&format!("tag {other} (immediate)")),
            }
        }
    }

    /// `apply_to_run({type = 'new_blind_choice'})` for the held tags until one fires
    /// (tag.lua:207-300): the pack tags open their pack, the Boss Tag rerolls the boss.
    pub(crate) fn apply_tags_new_blind_choice(&mut self) {
        for i in 0..self.gs.tags.len() {
            if self.tag_ready(i, "new_blind_choice") {
                let name = self.gs.tags[i].id.get().name;
                return self.unported(&format!("tag {name} (new_blind_choice)"));
            }
        }
    }

    /// `apply_to_run({type = 'shop_start'})` for every held tag (game.lua:3095-3097): the D6
    /// Tag makes rerolls start at $0 this shop (tag.lua:375-386).
    pub(crate) fn apply_tags_shop_start(&mut self) {
        for i in 0..self.gs.tags.len() {
            if !self.tag_ready(i, "shop_start") {
                continue;
            }
            if self.gs.tags[i].id == tags::TAG_D_SIX && !self.gs.sim.shop_d6ed {
                self.gs.sim.shop_d6ed = true;
                self.tag_yep(i, Some(Event::Call(d6, 0, 0.0)));
            }
        }
    }

    /// `apply_to_run({type = 'store_joker_create', area = area})` for the held tags until one
    /// returns a card (UI_definitions.lua:753-762): the Rare and Uncommon Tags make the shop
    /// joker, free (tag.lua:344-374).
    pub(crate) fn apply_tags_store_joker_create(&mut self, area: AreaId) -> Option<Card> {
        for i in 0..self.gs.tags.len() {
            if !self.tag_ready(i, "store_joker_create") {
                continue;
            }
            let tag = self.gs.tags[i].id;
            let card = match tag.get().name {
                "Rare Tag" => {
                    let mut rares: Vec<&str> = Vec::new();
                    for c in &self.gs.areas.jokers {
                        if c.center().rarity == Some(3) && !rares.contains(&c.center.key()) {
                            rares.push(c.center.key());
                        }
                    }
                    if jackdaw_data::JOKER_RARITY_POOLS[2].len() > rares.len() {
                        let card = self
                            .create_card(CardSpec::new(PoolType::Joker, area, "rta").rarity(1.0));
                        self.tag_yep(i, Some(Event::Call(couponed, card.sort_id, 0.0)));
                        Some(card)
                    } else {
                        // `Tag:nope()`.
                        self.gs.tags[i].triggered = true;
                        self.queue(Event::RemoveTag(tag));
                        None
                    }
                }
                "Uncommon Tag" => {
                    let card =
                        self.create_card(CardSpec::new(PoolType::Joker, area, "uta").rarity(0.9));
                    self.tag_yep(i, Some(Event::Call(couponed, card.sort_id, 0.0)));
                    Some(card)
                }
                other => {
                    self.unported(&format!("tag {other} (store_joker_create)"));
                    None
                }
            };
            if card.is_some() {
                return card;
            }
        }
        None
    }

    /// `apply_to_run({type = 'store_joker_modify', card = card})` for the held tags until one
    /// applies: an editionless shop joker gets the tag's edition, free (tag.lua:387-436).
    /// `card` may still be outside the areas (a tag-made joker), so it is passed in.
    pub(crate) fn apply_tags_store_joker_modify(&mut self, card: &Card) {
        if card.edition.is_some() || card.set() != jackdaw_data::CenterSet::Joker {
            return;
        }
        for i in 0..self.gs.tags.len() {
            if !self.tag_ready(i, "store_joker_modify") {
                continue;
            }
            let edition = match self.gs.tags[i].id.get().name {
                "Foil Tag" => EditionKind::Foil,
                "Holographic Tag" => EditionKind::Holo,
                "Polychrome Tag" => EditionKind::Polychrome,
                "Negative Tag" => EditionKind::Negative,
                other => {
                    self.unported(&format!("tag {other} (store_joker_modify)"));
                    return;
                }
            };
            let code = edition_code(edition);
            self.tag_yep(i, Some(Event::Call(tag_edition, card.sort_id, code)));
            return;
        }
    }

    /// `apply_to_run({type = 'voucher_add'})` for every held tag (game.lua:3160-3162): the
    /// Voucher Tag adds a second voucher to the shop (tag.lua:301-317).
    pub(crate) fn apply_tags_voucher_add(&mut self) {
        for i in 0..self.gs.tags.len() {
            if !self.tag_ready(i, "voucher_add") {
                continue;
            }
            if self.gs.tags[i].id == tags::TAG_VOUCHER {
                self.tag_yep(i, Some(Event::Call(voucher_tag, 0, 0.0)));
            }
        }
    }

    /// `apply_to_run({type = 'shop_final_pass'})` for every held tag (game.lua:3163-3165): the
    /// Coupon Tag makes the shop's jokers and packs free once per shop (tag.lua:437-450).
    pub(crate) fn apply_tags_shop_final_pass(&mut self) {
        for i in 0..self.gs.tags.len() {
            if !self.tag_ready(i, "shop_final_pass") {
                continue;
            }
            if self.gs.tags[i].id == tags::TAG_COUPON && !self.gs.sim.shop_free {
                self.gs.sim.shop_free = true;
                self.tag_yep(i, Some(Event::Call(coupon, 0, 0.0)));
            }
        }
    }

    /// `G.GAME.selected_back:trigger_effect({context = 'eval'})`: the Anaglyph Deck makes a
    /// Double Tag after a boss (back.lua:111-120).
    pub(crate) fn back_trigger_eval(&mut self) {
        if self.gs.back.get().name == "Anaglyph Deck" && self.gs.sim.last_blind_was_boss {
            self.queue(Event::AddTag(tags::TAG_DOUBLE, None));
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

fn edition_code(e: EditionKind) -> f64 {
    match e {
        EditionKind::Foil => 0.0,
        EditionKind::Holo => 1.0,
        EditionKind::Polychrome => 2.0,
        EditionKind::Negative => 3.0,
    }
}

/// The Top-up Tag's event: common jokers while there is room (tag.lua:135-145).
fn top_up(g: &mut Game, _: u32, n: f64) {
    let mut i = 0.0;
    while i < n {
        if (g.gs.areas.jokers.len() as f64) < g.gs.joker_slots {
            let card =
                g.create_card(CardSpec::new(PoolType::Joker, AreaId::Jokers, "top").rarity(0.0));
            let id = card.sort_id;
            g.gs.areas.emplace(AreaId::Jokers, card, false, false);
            g.add_to_deck(id, false);
        }
        i += 1.0;
    }
}

/// The Economy Tag's event: `ease_dollars(min(max, max(0, dollars)), true)` (tag.lua:180-186).
fn economy(g: &mut Game, _: u32, max: f64) {
    let x = max.min(g.gs.dollars.max(0.0));
    g.gs.dollars += x;
}

/// The D6 Tag's event: `temp_reroll_cost = 0; calculate_reroll_cost(true)` (tag.lua:379-383).
fn d6(g: &mut Game, _: u32, _: f64) {
    g.gs.round_resets.temp_reroll_cost = Some(0.0);
    g.calculate_reroll_cost(true);
}

/// A tag-made shop joker's event: `couponed = true; set_cost()` (tag.lua:357-362, 367-371).
fn couponed(g: &mut Game, card: u32, _: f64) {
    if let Some((area, _)) = g.gs.areas.locate(card) {
        let cost = g.cost_context(area);
        let c = g.card_mut(card);
        c.ability.couponed = true;
        c.set_cost(&cost);
    }
}

/// A store_joker_modify tag's event: the edition, then `couponed` and `set_cost()`
/// (tag.lua:394-430).
fn tag_edition(g: &mut Game, card: u32, code: f64) {
    let edition = match code as u8 {
        0 => EditionKind::Foil,
        1 => EditionKind::Holo,
        2 => EditionKind::Polychrome,
        _ => EditionKind::Negative,
    };
    if let Some((area, _)) = g.gs.areas.locate(card) {
        let cost = g.cost_context(area);
        let c = g.card_mut(card);
        c.set_edition(Some(edition), &cost);
        c.ability.couponed = true;
        c.set_cost(&cost);
    }
}

/// The Voucher Tag's event: a second voucher in the shop (tag.lua:303-315).
fn voucher_tag(g: &mut Game, _: u32, _: f64) {
    let Some(key) = g.get_next_voucher_key(true) else {
        return;
    };
    let card = g.new_card(key, None, AreaId::ShopVouchers);
    g.gs.areas.emplace(AreaId::ShopVouchers, card, false, false);
}

/// The Coupon Tag's event: the shop's jokers and packs become free (tag.lua:441-447).
fn coupon(g: &mut Game, _: u32, _: f64) {
    for area in [AreaId::ShopJokers, AreaId::ShopBooster] {
        let cost = g.cost_context(area);
        for c in g.gs.areas.get_mut(area) {
            c.ability.couponed = true;
            c.set_cost(&cost);
        }
    }
}

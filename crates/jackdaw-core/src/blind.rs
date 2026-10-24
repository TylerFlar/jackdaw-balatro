//! The blind (`Blind`, blind.lua): setting it, its debuffs and the boss effects.

use jackdaw_data::{BlindId, Value, blinds};

use crate::area::AreaId;
use crate::event::Event;
use crate::game::Game;
use crate::hand_eval::{HandType, PokerHands};
use crate::state::{BlindSlot, BlindState};

/// What `Blind:debuff_card` reads besides the card (blind.lua:624-652): the blind's debuff
/// table and state, and whether Pareidolia and Smeared Joker are held.
struct DebuffRule {
    disabled: bool,
    name: &'static str,
    suit: Option<crate::card::Suit>,
    face: bool,
    value: Option<&'static str>,
    nominal: Option<f64>,
    pareidolia: bool,
    smeared: bool,
}

/// The empty blind between rounds (`set_blind(nil)`).
fn empty_blind() -> BlindState {
    BlindState {
        key: None,
        chips: 0.0,
        mult: 0.0,
        dollars: 0.0,
        boss: false,
        disabled: false,
        triggered: false,
        prepped: true,
        eye_hands: Vec::new(),
        mouth_hand: None,
        discards_sub: None,
        hands_sub: None,
    }
}

impl Game {
    /// `G.GAME.blind.name` (`''` for the empty blind).
    pub fn blind_name(&self) -> &'static str {
        self.gs
            .blind
            .as_ref()
            .and_then(|b| b.key)
            .map_or("", |k| k.get().name)
    }

    /// The active (not disabled) boss's name, or `""`.
    fn active_boss(&self) -> &'static str {
        match &self.gs.blind {
            Some(b) if !b.disabled => self.blind_name(),
            _ => "",
        }
    }

    /// `G.GAME.blind.debuff` (`config.blind.debuff`, `{}` for none).
    fn blind_debuff(&self) -> &'static [(&'static str, Value)] {
        self.gs
            .blind
            .as_ref()
            .and_then(|b| b.key)
            .map_or(&[], |k| k.get().debuff)
    }

    /// `Blind:get_type()` (blind.lua:346-354): `None` for the empty blind.
    pub fn blind_type(&self) -> Option<BlindSlot> {
        match self.blind_name() {
            "" => None,
            "Small Blind" => Some(BlindSlot::Small),
            "Big Blind" => Some(BlindSlot::Big),
            _ => Some(BlindSlot::Boss),
        }
    }

    /// `Blind:set_blind(blind, reset)` (blind.lua:78-216). `reset` re-applies the debuffs of
    /// the current blind.
    pub(crate) fn set_blind(&mut self, blind: Option<BlindId>, reset: bool) {
        if !reset {
            let mut b = empty_blind();
            b.key = blind;
            if let Some(k) = blind {
                let proto = k.get();
                b.dollars = proto.dollars;
                b.mult = proto.mult;
                b.boss = proto.boss.is_some();
            }
            self.gs.blind = Some(b);
            if let Some(slot) = self.blind_type()
                && self.no_blind_reward(slot)
            {
                self.blind_mut().dollars = 0.0;
            }
            self.gs.sim.last_blind_was_boss = self.blind_mut().boss;
            let amount = crate::rules::get_blind_amount(self, self.gs.round_resets.ante);
            let chips = crate::lua::mul(
                crate::lua::mul(amount, self.blind_mut().mult),
                self.gs.starting_params.ante_scaling,
            );
            self.blind_mut().chips = if blind.is_some() { chips } else { 0.0 };
        }
        if !reset {
            match self.blind_name() {
                "The Eye" => self.blind_mut().eye_hands.clear(),
                "The Mouth" => self.blind_mut().mouth_hand = None,
                "The Fish" => self.blind_mut().prepped = false,
                "The Water" => {
                    let sub = self.gs.current_round.discards_left;
                    self.blind_mut().discards_sub = Some(sub);
                    self.queue(Event::EaseDiscard(-sub));
                }
                "The Needle" => {
                    let sub = self.gs.round_resets.hands - 1.0;
                    self.blind_mut().hands_sub = Some(sub);
                    self.queue(Event::EaseHandsPlayed(-sub));
                }
                "The Manacle" => self.change_size(AreaId::Hand, -1.0),
                "Amber Acorn" if !self.gs.areas.jokers.is_empty() => {
                    for c in &mut self.gs.areas.jokers {
                        c.flip();
                    }
                    if self.gs.areas.jokers.len() > 1 {
                        self.queue(Event::AmberAcorn);
                    }
                }
                _ => {}
            }
        }
        self.debuff_playing_cards();
        if !reset {
            for id in self.gs.areas.ids(AreaId::Jokers) {
                self.debuff_card(id, true);
            }
        }
    }

    fn blind_mut(&mut self) -> &mut BlindState {
        self.gs.blind.as_mut().expect("a blind is set")
    }

    /// `G.GAME.modifiers.no_blind_reward[type]`.
    fn no_blind_reward(&self, slot: BlindSlot) -> bool {
        use crate::state::LuaValue;
        match self.gs.modifiers.get("no_blind_reward") {
            Some(LuaValue::Table(t)) => t.get(slot.name()).is_some_and(LuaValue::truthy),
            _ => false,
        }
    }

    /// `Blind:debuff_card(card, from_blind)` (blind.lua:624-652).
    pub(crate) fn debuff_card(&mut self, id: u32, _from_blind: bool) {
        let Some((area, i)) = self.gs.areas.locate(id) else {
            // Cards outside the areas (Marble's pending stones) are debuffed when they land.
            return;
        };
        let rule = self.debuff_rule();
        self.debuff_card_at(area, i, &rule);
    }

    /// `for _, v in ipairs(G.playing_cards) do self:debuff_card(v) end`. A playing card's
    /// debuff changes its own flag and nothing else (no joker, so not the rule either): the
    /// order doesn't matter, and the pass walks the areas. Marble's pending stones are
    /// debuffed when they land.
    pub(crate) fn debuff_playing_cards(&mut self) {
        let rule = self.debuff_rule();
        for a in AreaId::PLAYING {
            for i in 0..self.gs.areas.get(a).len() {
                self.debuff_card_at(a, i, &rule);
            }
        }
    }

    /// What `Blind:debuff_card` reads besides the card.
    fn debuff_rule(&self) -> DebuffRule {
        let debuff = self.blind_debuff();
        let get = |k: &str| jackdaw_data::lookup(debuff, k);
        DebuffRule {
            disabled: self.gs.blind.as_ref().is_none_or(|b| b.disabled),
            name: self.blind_name(),
            suit: match get("suit") {
                Some(Value::Str(s)) => Some(crate::card::Suit::from_name(s).expect("boss suit")),
                _ => None,
            },
            face: get("is_face").and_then(Value::as_str) == Some("face"),
            value: match get("value") {
                Some(Value::Str(v)) => Some(*v),
                _ => None,
            },
            nominal: get("nominal").and_then(Value::as_num),
            pareidolia: self.has_joker("Pareidolia"),
            smeared: self.has_joker("Smeared Joker"),
        }
    }

    /// `Blind:debuff_card` (blind.lua:624-652) on the card at `area[i]`.
    fn debuff_card_at(&mut self, area: AreaId, i: usize, rule: &DebuffRule) {
        let card = &self.gs.areas.get(area)[i];
        if !rule.disabled && area != AreaId::Jokers {
            let mut hit = false;
            if let Some(suit) = rule.suit {
                hit = card.is_suit(suit, true, false, rule.smeared) == Some(true);
            }
            if !hit && rule.face {
                hit = card.is_face(true, rule.pareidolia);
            }
            if !hit && rule.name == "The Pillar" && card.ability.played_this_ante {
                hit = true;
            }
            if !hit && let Some(v) = rule.value {
                hit = card.base.is_some_and(|b| b.value.name() == v);
            }
            if !hit && let Some(n) = rule.nominal {
                hit = card.base.is_some_and(|b| b.nominal() == n);
            }
            if hit {
                self.set_debuff_at(area, i, true);
                return;
            }
        }
        if rule.name == "Crimson Heart" && !rule.disabled && area == AreaId::Jokers {
            return;
        }
        if rule.name == "Verdant Leaf" && !rule.disabled && area != AreaId::Jokers {
            self.set_debuff_at(area, i, true);
            return;
        }
        self.set_debuff_at(area, i, false);
    }

    /// `Blind:debuff_hand(cards, hand, handname, check)` (blind.lua:519-570). `cards` is the
    /// number of cards in the hand.
    pub(crate) fn blind_debuff_hand(
        &mut self,
        cards: usize,
        hands: &PokerHands,
        handname: HandType,
        check: bool,
    ) -> bool {
        let Some(b) = &self.gs.blind else {
            return false;
        };
        if b.disabled {
            return false;
        }
        let name = self.blind_name();
        let debuff = self.blind_debuff();
        let get = |k: &str| jackdaw_data::lookup(debuff, k);
        // `self.debuff` is always a table, so this block always runs.
        self.blind_mut().triggered = false;
        if let Some(Value::Str(h)) = get("hand")
            && hands.contains(HandType::from_name(h).expect("boss hand"))
        {
            self.blind_mut().triggered = true;
            return true;
        }
        if let Some(n) = get("h_size_ge").and_then(Value::as_num)
            && (cards as f64) < n
        {
            self.blind_mut().triggered = true;
            return true;
        }
        if let Some(n) = get("h_size_le").and_then(Value::as_num)
            && (cards as f64) > n
        {
            self.blind_mut().triggered = true;
            return true;
        }
        if name == "The Eye" {
            if self.blind_mut().eye_hands.contains(&handname) {
                self.blind_mut().triggered = true;
                return true;
            }
            if !check {
                self.blind_mut().eye_hands.push(handname);
            }
        }
        if name == "The Mouth" {
            if let Some(only) = self.blind_mut().mouth_hand
                && only != handname
            {
                self.blind_mut().triggered = true;
                return true;
            }
            if !check {
                self.blind_mut().mouth_hand = Some(handname);
            }
        }
        if name == "The Arm" {
            self.blind_mut().triggered = false;
            if self.gs.hand(handname).level > 1.0 {
                self.blind_mut().triggered = true;
                if !check {
                    self.level_up_hand(handname, -1.0);
                }
            }
        }
        if name == "The Ox" {
            self.blind_mut().triggered = false;
            if Some(handname) == Some(self.gs.current_round.most_played_poker_hand) {
                self.blind_mut().triggered = true;
                if !check {
                    // ease_dollars(-G.GAME.dollars, true): instant.
                    self.gs.dollars -= self.gs.dollars;
                }
            }
        }
        false
    }

    /// `Blind:press_play()` (blind.lua:464-508).
    pub(crate) fn blind_press_play(&mut self) {
        match self.active_boss() {
            "The Hook" => {
                self.queue(Event::HookDiscard);
                self.blind_mut().triggered = true;
            }
            "Crimson Heart" => {
                if !self.gs.areas.jokers.is_empty() {
                    self.blind_mut().triggered = true;
                    self.blind_mut().prepped = true;
                }
            }
            "The Fish" => self.blind_mut().prepped = true,
            "The Tooth" => {
                self.queue(Event::Tooth);
                self.blind_mut().triggered = true;
            }
            _ => {}
        }
    }

    /// The Hook's event (blind.lua:467-483): two random held cards are discarded.
    pub(crate) fn hook_discard(&mut self) {
        let mut cards: Vec<u32> = self.gs.areas.ids(AreaId::Hand).to_vec();
        let mut any = false;
        for i in 0..2 {
            if i < self.gs.areas.hand.len() {
                cards.sort_unstable();
                let seed = self.gs.rng.seed("hook");
                let k = crate::rng::element_index(cards.len(), seed);
                let picked = cards.remove(k);
                self.add_to_highlighted(picked);
                any = true;
            }
        }
        if any {
            self.discard_cards_from_highlighted(true);
        }
    }

    /// `Blind:modify_hand(cards, poker_hands, text, mult, hand_chips)` (blind.lua:510-517).
    pub(crate) fn blind_modify_hand(&mut self, mult: f64, chips: f64) -> (f64, f64, bool) {
        if self.active_boss() == "The Flint" {
            self.blind_mut().triggered = true;
            let m = crate::lua::floor(mult * 0.5 + 0.5).max(1.0);
            let c = crate::lua::floor(chips * 0.5 + 0.5).max(0.0);
            return (m, c, true);
        }
        (mult, chips, false)
    }

    /// `Blind:drawn_to_hand()` (blind.lua:572-603).
    pub(crate) fn blind_drawn_to_hand(&mut self) {
        match self.active_boss() {
            "Cerulean Bell" => {
                let any_forced = self
                    .gs
                    .areas
                    .hand
                    .iter()
                    .any(|c| c.ability.forced_selection);
                if !any_forced && !self.gs.areas.hand.is_empty() {
                    self.unhighlight_all_hand();
                    let mut ids = self.gs.areas.ids(AreaId::Hand);
                    ids.sort_unstable();
                    let seed = self.gs.rng.seed("cerulean_bell");
                    let forced = ids[crate::rng::element_index(ids.len(), seed)];
                    if let Some(c) = self.gs.areas.card_mut(forced) {
                        c.ability.forced_selection = true;
                    }
                    self.add_to_highlighted(forced);
                }
            }
            "Crimson Heart"
                if self.gs.blind.as_ref().is_some_and(|b| b.prepped)
                    && !self.gs.areas.jokers.is_empty() =>
            {
                let all = self.gs.areas.ids(AreaId::Jokers);
                let single = all.len() < 2;
                let mut jokers = Vec::new();
                for id in all {
                    let debuffed = self.gs.areas.card(id).is_some_and(|c| c.debuff);
                    if !debuffed || single {
                        jokers.push(id);
                    }
                    self.set_debuff(id, false);
                }
                if !jokers.is_empty() {
                    jokers.sort_unstable();
                    let seed = self.gs.rng.seed("crimson_heart");
                    let pick = jokers[crate::rng::element_index(jokers.len(), seed)];
                    self.set_debuff(pick, true);
                }
            }
            _ => {}
        }
        if let Some(b) = self.gs.blind.as_mut() {
            b.prepped = false;
        }
    }

    /// `Blind:stay_flipped(area, card)` (blind.lua:605-622), then the `flipped_cards` challenge
    /// rule (common_events.lua:403-407).
    pub(crate) fn stay_flipped(&mut self, to: AreaId, card: &crate::card::Card) -> bool {
        let mut flipped = false;
        if to == AreaId::Hand {
            let pareidolia = self.has_joker("Pareidolia");
            match self.active_boss() {
                "The Wheel" => {
                    let roll = self.gs.rng.random("wheel");
                    flipped = roll < self.gs.probabilities_normal / 7.0;
                }
                "The House" => {
                    flipped = self.gs.current_round.hands_played == 0.0
                        && self.gs.current_round.discards_used == 0.0;
                }
                "The Mark" => flipped = card.is_face(true, pareidolia),
                "The Fish" => flipped = self.gs.blind.as_ref().is_some_and(|b| b.prepped),
                _ => {}
            }
            if let Some(n) = self
                .gs
                .modifiers
                .get("flipped_cards")
                .and_then(|v| v.as_num())
                && self.gs.rng.random("flipped_card") < 1.0 / n
            {
                flipped = true;
            }
        }
        flipped
    }

    /// `Blind:disable()` (blind.lua:356-415).
    pub(crate) fn blind_disable(&mut self) {
        self.blind_mut().disabled = true;
        for c in &mut self.gs.areas.jokers {
            if c.facing == crate::card::Facing::Back {
                c.flip();
            }
        }
        match self.blind_name() {
            "The Water" => {
                let sub = self.blind_mut().discards_sub.unwrap_or(0.0);
                self.queue(Event::EaseDiscard(sub));
            }
            "The Wheel" | "The House" | "The Mark" | "The Fish" => {
                for c in &mut self.gs.areas.hand {
                    if c.facing == crate::card::Facing::Back {
                        c.flip();
                    }
                }
            }
            "The Needle" => {
                let sub = self.blind_mut().hands_sub.unwrap_or(0.0);
                self.queue(Event::EaseHandsPlayed(sub));
            }
            "The Wall" => self.blind_mut().chips /= 2.0,
            "Cerulean Bell" => {
                for id in self.playing_card_ids() {
                    if let Some(c) = self.gs.areas.card_mut(id) {
                        c.ability.forced_selection = false;
                    }
                }
            }
            "The Manacle" => {
                self.change_size(AreaId::Hand, 1.0);
                self.draw_from_deck_to_hand(Some(1));
            }
            "Violet Vessel" => self.blind_mut().chips /= 3.0,
            _ => {}
        }
        self.queue(Event::DisabledBossCheck);
        self.debuff_playing_cards();
        for id in self.gs.areas.ids(AreaId::Jokers) {
            self.debuff_card(id, false);
        }
    }

    /// `Blind:defeat()` (blind.lua:276-344).
    pub(crate) fn blind_defeat(&mut self) {
        self.queue(Event::BlindReset);
        for c in &mut self.gs.areas.jokers {
            if c.facing == crate::card::Facing::Back {
                c.flip();
            }
        }
        if self.blind_name() == "The Manacle" && !self.blind_mut().disabled {
            self.change_size(AreaId::Hand, 1.0);
        }
    }

    /// Small and Big blind keys.
    pub(crate) fn is_small(b: BlindId) -> bool {
        b == blinds::BL_SMALL
    }

    pub(crate) fn is_big(b: BlindId) -> bool {
        b == blinds::BL_BIG
    }
}

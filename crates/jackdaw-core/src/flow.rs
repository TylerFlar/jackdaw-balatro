//! The hand flow: selecting a blind, drawing, playing and discarding (state_events.lua,
//! button_callbacks.lua) and the per-frame state handlers (`Game:update_*`, game.lua).

use crate::area::AreaId;
use crate::card::Card;
use crate::event::Event;
use crate::game::{Game, LuaState};
use crate::hand_eval::get_poker_hand_info;
use crate::state::{BlindSlot, HandSort};

/// `G.hand.config.highlighted_limit`.
pub const HAND_HIGHLIGHT_LIMIT: usize = 5;

impl Game {
    // -----------------------------------------------------------------------
    // Highlighting (cardarea.lua:131-212)
    // -----------------------------------------------------------------------

    /// `G.hand:add_to_highlighted(card)` (cardarea.lua:131-160): at most five cards.
    pub(crate) fn add_to_highlighted(&mut self, id: u32) {
        if self.rt.highlighted.len() < HAND_HIGHLIGHT_LIMIT {
            self.rt.highlighted.push(id);
        }
        if self.rt.state == LuaState::SelectingHand {
            self.parse_highlighted();
        }
    }

    /// `G.hand:unhighlight_all()` (cardarea.lua:201-212): Cerulean Bell's card stays.
    pub(crate) fn unhighlight_all_hand(&mut self) {
        let areas = &self.gs.areas;
        self.rt.highlighted.retain(|&id| {
            areas.card(id).is_some_and(|c| {
                c.ability.forced_selection && areas.locate(id).map(|l| l.0) == Some(AreaId::Hand)
            })
        });
        if self.rt.state == LuaState::SelectingHand {
            self.parse_highlighted();
        }
    }

    /// `CardArea:parse_highlighted()` (cardarea.lua:162-185): only the blind's `triggered`
    /// flag is gameplay state here (`debuff_hand` in check mode).
    pub(crate) fn parse_highlighted(&mut self) {
        let cards: Vec<Card> = self
            .rt
            .highlighted
            .iter()
            .filter_map(|&id| self.gs.areas.card(id).cloned())
            .collect();
        let info = get_poker_hand_info(&self.eval_cards(&cards), self.eval_flags());
        if let Some(text) = info.hand {
            self.blind_debuff_hand(cards.len(), &info.poker_hands, text, true);
        }
    }

    /// The hand's cards in highlight order sorted by position (`table.sort(G.hand.highlighted,
    /// function(a,b) return a.T.x < b.T.x end)`).
    fn sorted_highlighted(&mut self) -> crate::area::Ids {
        let hand = self.gs.areas.ids(AreaId::Hand);
        self.rt
            .highlighted
            .sort_by_key(|id| hand.iter().position(|h| h == id).unwrap_or(usize::MAX));
        crate::area::Ids::from_slice(&self.rt.highlighted)
    }

    // -----------------------------------------------------------------------
    // Blind selection and the round start
    // -----------------------------------------------------------------------

    /// `G.FUNCS.select_blind(e)` (button_callbacks.lua:2513-2553).
    pub(crate) fn select_blind(&mut self) {
        self.rt.facing_blind = true;
        self.queue(Event::SelectBlind);
        self.queue(Event::NewRound);
    }

    /// `select_blind`'s second event (button_callbacks.lua:2530-2545).
    pub(crate) fn select_blind_event(&mut self) {
        self.queue(Event::EaseRound(1.0));
        let slot = self.gs.blind_on_deck.unwrap_or(BlindSlot::Small);
        self.rt.round_blind = self.gs.round_resets.blind_choice(slot);
        self.gs.round_resets.set_blind_state(slot, "Current");
    }

    /// `new_round()` (state_events.lua:290-353).
    pub(crate) fn new_round(&mut self) {
        self.queue(Event::NewRoundBody);
    }

    /// `new_round`'s event (state_events.lua:295-351).
    pub(crate) fn new_round_body(&mut self) {
        let rr = &self.gs.round_resets;
        let rb = &self.gs.round_bonus;
        let cr = &mut self.gs.current_round;
        cr.discards_left = (rr.discards + rb.discards).max(0.0);
        cr.hands_left = (rr.hands + rb.next_hands).max(1.0);
        cr.hands_played = 0.0;
        cr.discards_used = 0.0;
        cr.reroll_cost_increase = 0.0;
        for h in &mut self.gs.hands {
            h.played_this_round = 0.0;
        }
        self.gs.current_round.free_rerolls = self.find_joker("Chaos the Clown", false).len() as f64;
        self.calculate_reroll_cost(true);
        self.gs.round_bonus.next_hands = 0.0;
        self.gs.round_bonus.discards = 0.0;
        let blind = self.rt.round_blind;
        let slot = match blind {
            Some(b) if Game::is_small(b) => BlindSlot::Small,
            Some(b) if Game::is_big(b) => BlindSlot::Big,
            _ => BlindSlot::Boss,
        };
        self.gs.round_resets.set_blind_state(slot, "Current");
        self.set_blind(blind, false);
        for id in self.gs.areas.ids(AreaId::Jokers) {
            let mut ctx = crate::joker::Ctx::new(crate::joker::Kind::SettingBlind);
            ctx.blind = blind;
            self.calculate_joker(id, &mut ctx);
        }
        self.queue(Event::NewRoundShuffle);
    }

    /// `new_round`'s nested event (state_events.lua:340-349).
    pub(crate) fn new_round_shuffle(&mut self) {
        self.set_state(LuaState::DrawToHand, self.rt.state_complete);
        let key = crate::rng::Key::new("nr").n(self.gs.round_resets.ante);
        self.gs.areas.shuffle(AreaId::Deck, &mut self.gs.rng, &key);
        self.rt.state_complete = false;
    }

    // -----------------------------------------------------------------------
    // Drawing
    // -----------------------------------------------------------------------

    /// Luxury Tax (`modifiers.minus_hand_size_per_X_dollar`): every frame, `G.hand`'s update
    /// compares `floor(dollars / X)` with `config.last_poll_size` (nil = 0) and changes the
    /// hand size by the difference (cardarea.lua:245-251). There are no frames here: as in the
    /// Python engine, the poll runs after every action and before every draw, where vanilla's
    /// frames would have caught up. True when the hand size changed.
    pub(crate) fn poll_hand_size(&mut self) -> bool {
        let Some(x) = self.modifier_num("minus_hand_size_per_X_dollar") else {
            return false;
        };
        let poll = (self.gs.dollars / x).floor();
        let last = self.gs.sim.hand_last_poll_size;
        if poll == last {
            return false;
        }
        self.gs.hand_size += last - poll;
        self.gs.sim.hand_last_poll_size = poll;
        true
    }

    /// `G.FUNCS.draw_from_deck_to_hand(e)` (state_events.lua:355-377). True when the game is
    /// lost because the hand can hold no cards.
    pub(crate) fn draw_from_deck_to_hand(&mut self, e: Option<i64>) -> bool {
        self.poll_hand_size();
        let pack = matches!(
            self.rt.state,
            LuaState::TarotPack | LuaState::SpectralPack | LuaState::PlanetPack
        );
        if !pack && self.hand_limit() <= 0.0 && self.gs.areas.hand.is_empty() {
            self.set_state(LuaState::GameOver, false);
            return true;
        }
        let deck = self.gs.areas.deck.len() as i64;
        let mut space = e.unwrap_or_else(|| {
            deck.min(self.hand_limit() as i64 - self.gs.areas.hand.len() as i64)
        });
        if self.active_boss_name() == "The Serpent"
            && (self.gs.current_round.hands_played > 0.0
                || self.gs.current_round.discards_used > 0.0)
        {
            space = deck.min(3);
        }
        for _ in 0..space.max(0) {
            self.queue(Event::DrawCard {
                from: AreaId::Deck,
                to: AreaId::Hand,
                card: None,
                sort: true,
            });
        }
        false
    }

    /// The name of the blind if it's an active boss, else `""`.
    pub(crate) fn active_boss_name(&self) -> &'static str {
        match &self.gs.blind {
            Some(b) if !b.disabled => self.blind_name(),
            _ => "",
        }
    }

    /// `draw_card`'s event (common_events.lua:395-423).
    pub(crate) fn draw_card_event(
        &mut self,
        from: AreaId,
        to: AreaId,
        card: Option<u32>,
        sort: bool,
    ) {
        let moved = match card {
            Some(id) => self.gs.areas.remove_card(from, Some(id)),
            // CardArea:draw_card_from (cardarea.lua:592-611): every area accepts here except a
            // full one that is neither the deck nor the hand.
            None => {
                let room = to == AreaId::Deck
                    || to == AreaId::Hand
                    || (self.gs.areas.get(to).len() as f64) < self.area_limit(to);
                if room {
                    self.gs.areas.remove_card(from, None)
                } else {
                    None
                }
            }
        };
        if let Some(c) = moved {
            if from == AreaId::Hand {
                self.rt.highlighted.retain(|&h| h != c.sort_id);
            }
            let flipped = self.stay_flipped(to, &c);
            self.gs.areas.emplace(to, c, false, flipped);
        }
        if sort {
            debug_assert_eq!(to, AreaId::Hand, "only hand draws sort");
            // When the next event is another sorting draw into the hand, it runs right after
            // this one (the queue runs in order) and sorts: no draw reads the hand's order
            // (`stay_flipped` looks at the card alone), and every correct sort gives the same
            // order, so a run of draws sorts once, after its last card.
            let next_sorts = matches!(
                self.rt.queue.front(),
                Some(Event::DrawCard {
                    to: AreaId::Hand,
                    sort: true,
                    ..
                })
            );
            if !next_sorts {
                self.sort_hand();
            }
        }
    }

    /// `config.card_limit` of an area that `draw_card_from` checks.
    fn area_limit(&self, area: AreaId) -> f64 {
        match area {
            AreaId::Discard => 500.0,
            AreaId::Play => 5.0,
            AreaId::Jokers => self.gs.joker_slots,
            AreaId::Consumables => self.gs.consumable_slots,
            _ => f64::INFINITY,
        }
    }

    /// `G.hand:sort()` with `G.hand.config.sort` (cardarea.lua:577-590).
    pub(crate) fn sort_hand(&mut self) {
        let suit = self.gs.hand_sort == HandSort::SuitDesc;
        let hand = &mut self.gs.areas.hand;
        // get_nominal is unique per card (unique_val), so the order is total and every correct
        // sort gives the same hand. After a draw only the new last card is out of place: move it
        // into position.
        let n = hand.len();
        let mut keys = [0.0; 64];
        if (2..=keys.len()).contains(&n) {
            for (k, c) in keys.iter_mut().zip(hand.iter()) {
                *k = c.get_nominal(suit);
            }
            let (rest, last) = (&keys[..n - 1], keys[n - 1]);
            if rest.windows(2).all(|w| w[0] > w[1]) {
                let at = rest.partition_point(|&k| k > last);
                hand[at..].rotate_right(1);
                return;
            }
            // Otherwise sort the positions by their keys, then move each card once.
            let mut order = [0usize; 64];
            for (i, slot) in order.iter_mut().enumerate().take(n) {
                *slot = i;
            }
            order[..n]
                .sort_by(|&a, &b| keys[b].partial_cmp(&keys[a]).expect("nominals are finite"));
            crate::area::permute(hand, &order[..n]);
            return;
        }
        hand.sort_by(|a, b| {
            b.get_nominal(suit)
                .partial_cmp(&a.get_nominal(suit))
                .expect("nominals are finite")
        });
    }

    /// `update_draw_to_hand`'s event (game.lua:3219-3244).
    pub(crate) fn draw_to_hand_body(&mut self) {
        if self.draw_from_deck_to_hand(None) {
            return;
        }
        if self.gs.current_round.hands_played == 0.0
            && self.gs.current_round.discards_used == 0.0
            && self.rt.facing_blind
        {
            for id in self.gs.areas.ids(AreaId::Jokers) {
                let mut ctx = crate::joker::Ctx::new(crate::joker::Kind::FirstHandDrawn);
                self.calculate_joker(id, &mut ctx);
            }
        }
        self.queue(Event::DrawnToHand);
    }

    // -----------------------------------------------------------------------
    // Playing and discarding
    // -----------------------------------------------------------------------

    /// `G.FUNCS.play_cards_from_highlighted(e)` (state_events.lua:450-538).
    pub(crate) fn play_cards_from_highlighted(&mut self) {
        if !self.gs.areas.play.is_empty() {
            return;
        }
        if let Some(b) = self.gs.blind.as_mut() {
            b.triggered = false;
        }
        self.clear_forced_selection();
        let highlighted = self.sorted_highlighted();
        self.queue(Event::SetState(LuaState::HandPlayed, true));
        self.queue(Event::EaseHandsPlayed(-1.0));
        for &id in &highlighted {
            if let Some(c) = self.gs.areas.card_mut(id) {
                if let Some(b) = c.base.as_mut() {
                    b.times_played += 1.0;
                }
                c.ability.played_this_ante = true;
            }
            self.queue(Event::DrawCard {
                from: AreaId::Hand,
                to: AreaId::Play,
                card: Some(id),
                sort: false,
            });
        }
        self.blind_press_play();
        self.queue(Event::PlayHandQueued);
    }

    /// `ability.forced_selection = nil` on every playing card (state_events.lua:384, 459).
    fn clear_forced_selection(&mut self) {
        for a in AreaId::PLAYING {
            for c in self.gs.areas.get_mut(a) {
                if c.ability.forced_selection {
                    c.ability.forced_selection = false;
                }
            }
        }
    }

    /// `G.FUNCS.discard_cards_from_highlighted(e, hook)` (state_events.lua:379-448).
    pub(crate) fn discard_cards_from_highlighted(&mut self, hook: bool) {
        self.clear_forced_selection();
        let count = self
            .rt
            .highlighted
            .len()
            .min(500usize.saturating_sub(self.gs.areas.play.len()));
        if count == 0 {
            return;
        }
        let highlighted = self.sorted_highlighted();
        for id in self.gs.areas.ids(AreaId::Jokers) {
            let mut ctx = crate::joker::Ctx::new(crate::joker::Kind::PreDiscard);
            ctx.full_hand = &highlighted;
            ctx.hook = hook;
            self.calculate_joker(id, &mut ctx);
        }
        let mut destroyed = Vec::new();
        for &id in &highlighted[..count] {
            self.calculate_seal_discard(id);
            let mut removed = false;
            for j in self.gs.areas.ids(AreaId::Jokers) {
                let mut ctx = crate::joker::Ctx::new(crate::joker::Kind::Discard);
                ctx.other_card = Some(id);
                ctx.full_hand = &highlighted;
                if let Some(ret) = self.calculate_joker(j, &mut ctx)
                    && ret.remove
                {
                    removed = true;
                }
            }
            if removed {
                destroyed.push(id);
                self.destroy_card(id);
            } else {
                if let Some(c) = self.gs.areas.card_mut(id) {
                    c.ability.discarded = true;
                }
                self.queue(Event::DrawCard {
                    from: AreaId::Hand,
                    to: AreaId::Discard,
                    card: Some(id),
                    sort: false,
                });
            }
        }
        if !destroyed.is_empty() {
            for j in self.gs.areas.ids(AreaId::Jokers) {
                let mut ctx = crate::joker::Ctx::new(crate::joker::Kind::RemovePlayingCards);
                ctx.removed = &destroyed;
                self.eval_joker(j, &mut ctx);
            }
        }
        if !hook {
            if let Some(cost) = self
                .gs
                .modifiers
                .get("discard_cost")
                .and_then(|v| v.as_num())
            {
                self.queue(Event::EaseDollars(-cost));
            }
            self.queue(Event::EaseDiscard(-1.0));
            self.gs.current_round.discards_used += 1.0;
            self.set_state(LuaState::DrawToHand, self.rt.state_complete);
            self.queue(Event::StateIncomplete);
        }
    }

    /// `Card:calculate_seal({discard = true})` (card.lua:2253-2268): a Purple Seal makes a Tarot.
    fn calculate_seal_discard(&mut self, id: u32) {
        let Some(c) = self.gs.areas.card(id) else {
            return;
        };
        if c.debuff || c.seal != Some(crate::card::Seal::Purple) {
            return;
        }
        if (self.gs.areas.consumables.len() as f64) + self.rt.consumeable_buffer
            < self.gs.consumable_slots
        {
            self.rt.consumeable_buffer += 1.0;
            self.queue(Event::CreateConsumable {
                ty: crate::pools::PoolType::Tarot,
                append: "8ba",
                forced: None,
            });
        }
    }

    /// A card shatters (Glass) or dissolves; it leaves the run when the animation ends.
    pub(crate) fn destroy_card(&mut self, id: u32) {
        let glass = self
            .gs
            .areas
            .card(id)
            .is_some_and(|c| c.name() == "Glass Card");
        if let Some(c) = self.gs.areas.card_mut(id) {
            if glass {
                c.shattered = true;
            } else {
                c.destroyed = true;
            }
        }
        self.queue(Event::RemoveCard(id));
    }

    /// `G.FUNCS.draw_from_play_to_discard()` (state_events.lua:1088-1097).
    pub(crate) fn draw_from_play_to_discard(&mut self) {
        for c in self.gs.areas.ids(AreaId::Play) {
            let gone = self
                .gs
                .areas
                .card(c)
                .is_some_and(|c| c.shattered || c.destroyed);
            if !gone {
                self.queue(Event::DrawCard {
                    from: AreaId::Play,
                    to: AreaId::Discard,
                    card: Some(c),
                    sort: false,
                });
            }
        }
    }

    /// `G.FUNCS.draw_from_hand_to_discard()` (state_events.lua:1128-1133).
    pub(crate) fn draw_from_hand_to_discard(&mut self) {
        for _ in 0..self.gs.areas.hand.len() {
            self.queue(Event::DrawCard {
                from: AreaId::Hand,
                to: AreaId::Discard,
                card: None,
                sort: false,
            });
        }
    }

    // -----------------------------------------------------------------------
    // State handlers (Game:update and Game:update_*, game.lua:2544-2607, 3010-3339)
    // -----------------------------------------------------------------------

    /// One frame of the current state's handler. True when it did anything.
    pub(crate) fn update_state(&mut self) -> bool {
        match self.rt.state {
            LuaState::SelectingHand => {
                if self.gs.areas.hand.is_empty() && !self.gs.areas.deck.is_empty() {
                    self.set_state(LuaState::DrawToHand, false);
                    return true;
                }
                self.update_selecting_hand()
            }
            LuaState::HandPlayed => {
                if self.rt.state_complete {
                    return false;
                }
                self.rt.state_complete = true;
                self.queue(Event::HandPlayedCheck);
                true
            }
            LuaState::DrawToHand => {
                if self.rt.state_complete {
                    return false;
                }
                self.rt.state_complete = true;
                self.apply_tags_round_start_bonus();
                self.queue(Event::DrawToHandBody);
                true
            }
            LuaState::NewRound => {
                if self.rt.state_complete {
                    return false;
                }
                self.rt.state_complete = true;
                self.end_round();
                true
            }
            LuaState::RoundEval => {
                if self.rt.state_complete {
                    return false;
                }
                self.rt.state_complete = true;
                self.queue(Event::RoundEvalOpen);
                true
            }
            LuaState::GameOver => false,
            LuaState::Shop => self.update_shop(),
            LuaState::TarotPack
            | LuaState::PlanetPack
            | LuaState::SpectralPack
            | LuaState::StandardPack
            | LuaState::BuffoonPack => self.update_pack(),
            LuaState::BlindSelect => self.update_blind_select(),
            _ => {
                if self.rt.state_complete {
                    return false;
                }
                self.rt.state_complete = true;
                true
            }
        }
    }

    /// `Game:update_selecting_hand` (game.lua:3010-3070): with hand, deck and play empty the
    /// round ends.
    fn update_selecting_hand(&mut self) -> bool {
        let empty = self.gs.areas.hand.is_empty() && self.gs.areas.deck.is_empty();
        let mut acted = false;
        if empty && self.gs.areas.play.is_empty() && !self.rt.end_round_called {
            self.rt.end_round_called = true;
            self.end_round();
            acted = true;
        }
        if !self.rt.state_complete {
            self.rt.state_complete = true;
            if empty && !self.rt.end_round_called {
                self.rt.end_round_called = true;
                self.end_round();
            }
            acted = true;
        }
        acted
    }
}

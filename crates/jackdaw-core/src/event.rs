//! Queued events (`G.E_MANAGER:add_event`): one variant per gameplay-relevant Lua event closure,
//! carrying what the closure captured. UI-only events (sounds, juice, text) are not queued.
//!
//! [`Game::run_event`] is the closure body; the comment on each variant cites the Lua lines.

use jackdaw_data::{CenterId, TagId};

use crate::area::AreaId;
use crate::game::{Game, LuaState};
use crate::hand_eval::HandType;
use crate::pools::{CardSpec, PoolType};

/// A queued event.
#[derive(Clone, Debug)]
pub enum Event {
    /// `G.STATE = state; G.STATE_COMPLETE = complete`.
    SetState(LuaState, bool),
    /// `G.STATE_COMPLETE = false` (the state is already set).
    StateIncomplete,
    /// `draw_card(from, to, ..., sort, card)`'s event (common_events.lua:395-423): move `card`,
    /// or the next card of `from` (`CardArea:draw_card_from`).
    DrawCard {
        from: AreaId,
        to: AreaId,
        card: Option<u32>,
        sort: bool,
    },
    /// `ease_dollars(mod)` (common_events.lua:68-109).
    EaseDollars(f64),
    /// `ease_discard(mod)` (common_events.lua:111-150).
    EaseDiscard(f64),
    /// `ease_hands_played(mod)` (common_events.lua:152-189).
    EaseHandsPlayed(f64),
    /// `ease_ante(mod)` (common_events.lua:191-222).
    EaseAnte(f64),
    /// `ease_round(mod)` (common_events.lua:224-256).
    EaseRound(f64),
    /// `ease_chips(mod)` (common_events.lua:41-66): `G.GAME.chips` eases to `mod`.
    SetChips(f64),
    /// `G.GAME.dollar_buffer = 0` (card.lua:1086 and the joker economy effects).
    ClearDollarBuffer,
    /// `G.GAME.consumeable_buffer = 0`.
    ClearConsumeableBuffer,
    /// `G.GAME.joker_buffer = 0`.
    ClearJokerBuffer,
    /// `G.FUNCS.select_blind`'s second event (button_callbacks.lua:2530-2545).
    SelectBlind,
    /// `G.FUNCS.select_blind`'s third event: `new_round()` (button_callbacks.lua:2546-2552).
    NewRound,
    /// `new_round`'s event (state_events.lua:293-351).
    NewRoundBody,
    /// `new_round`'s nested event: draw state and the round shuffle (state_events.lua:340-349).
    NewRoundShuffle,
    /// `update_draw_to_hand`'s event (game.lua:3219-3244).
    DrawToHandBody,
    /// `update_draw_to_hand`'s nested event (game.lua:3233-3241).
    DrawnToHand,
    /// `play_cards_from_highlighted`'s last event (state_events.lua:504-537).
    PlayHandQueued,
    /// `G.FUNCS.evaluate_play()` (state_events.lua:509-515).
    EvaluatePlay,
    /// After scoring: discard the played cards, count the hand (state_events.lua:517-527).
    AfterPlay,
    /// `update_hand_played`'s event (game.lua:3193-3204).
    HandPlayedCheck,
    /// `end_round`'s event (state_events.lua:88-287).
    EndRound,
    /// `end_round`'s last event: round evaluation starts (state_events.lua:251-283).
    EndRoundFinal,
    /// `draw_from_discard_to_deck`'s event (state_events.lua:1108-1119).
    DiscardToDeck,
    /// `update_round_eval`'s event and the `evaluate_round` call it leads to
    /// (game.lua:3309-3337).
    RoundEvalOpen,
    /// `G.GAME.blind:defeat()` (state_events.lua:1148-1155).
    BlindDefeat,
    /// `Blind:defeat`'s timed event: `self:set_blind(nil, nil, true)` (blind.lua:330-337).
    BlindReset,
    /// The cash-out row: `G.GAME.current_round.dollars = dollars` (common_events.lua:1087).
    SetCashout(f64),
    /// The end of `Card:shatter` / `Card:start_dissolve`: `self:remove()` (card.lua:2079-2182).
    /// Lua runs it on a timer off the blocking chain; the port runs it in queue order.
    RemoveCard(u32),
    /// The Hook (blind.lua:466-487).
    HookDiscard,
    /// The Tooth's outer event (blind.lua:497-507).
    Tooth,
    /// Amber Acorn: `G.jokers:shuffle('aajk')` (blind.lua:196-204).
    ShuffleJokers,
    /// `CardArea:change_size(delta)`'s event (cardarea.lua:94-112).
    ChangeSize { area: AreaId, delta: f64 },
    /// `G.hand:sort()` (cardarea.lua:104).
    SortHand,
    /// `Blind:disable`'s event (blind.lua:397-406).
    DisabledBossCheck,
    /// The `debuff_played_cards` challenge rule (state_events.lua:1077-1084).
    PermaDebuff(Vec<u32>),
    /// Amber Acorn's outer event: three joker shuffles (blind.lua:196-203).
    AmberAcorn,
    /// `G.GAME.blind:set_blind(nil, true, nil)`: the boss re-checks every playing card
    /// (`add_to_deck` / `remove_from_deck`, card.lua:641, 699).
    BlindRedebuff,
    /// `v:set_cost()` for every card (Astronomer, card.lua:612-617).
    SetCostAll,
    /// A consumable made into `G.consumeables` (`create_card` + `add_to_deck` + `emplace`),
    /// then `G.GAME.consumeable_buffer = 0` (Purple Seal, Hallucination, 8 Ball, ...).
    CreateConsumable {
        ty: PoolType,
        append: &'static str,
        forced: Option<CenterId>,
    },
    /// Blue Seal: the last played hand's Planet (card.lua:1043-1060).
    BlueSealPlanet,
    /// `Tag:yep`'s last event: `self:remove()` (tag.lua:92-99).
    RemoveTag(TagId),
    /// `add_tag(Tag(key))`, with `G.orbital_hand` set to the hand (a Double Tag's copy).
    AddTag(TagId, Option<HandType>),
    /// An event that ends in the card's removal by a nested event: a joker's extinction
    /// (Gros Michel, card.lua:3021-3034) or `start_dissolve` called from an event.
    Dissolve(u32),
    /// `ease_chips(mod)`'s outer event: it queues the ease to `mod` (common_events.lua:41-66).
    EaseChips(f64),
    /// `cash_out`'s event: the next round's hands and discards, then the shop
    /// (button_callbacks.lua:2921-2936).
    CashOutBody,
    /// `update_shop`'s event: it queues the shop's construction (game.lua:3081-3091). The flag
    /// says whether the shop already existed (`shop_exists`).
    ShopOpen(bool),
    /// The shop's construction (game.lua:3088-3173).
    ShopBuild(bool),
    /// `create_card_for_shop`'s event: the `store_joker_modify` tags see the card
    /// (UI_definitions.lua:777-783).
    StoreJokerModify(u32),
    /// `reroll_shop`'s event (button_callbacks.lua:2865-2890).
    RerollBody,
    /// `reroll_shop`'s timed event, which queues [`Event::RerollJokers`]
    /// (button_callbacks.lua:2891-2904).
    RerollAfter,
    /// The jokers see the reroll (`reroll_shop` context).
    RerollJokers,
    /// `buy_from_shop`'s event (button_callbacks.lua:2414-2477).
    BuyFromShop(u32),
    /// `Card:sell_card`'s event: the sale price and the dissolve (card.lua:1612-1626).
    SellPayout(u32),
    /// `toggle_shop`'s second event: the shop closes (button_callbacks.lua:2496-2509).
    LeaveShop,
    /// `update_blind_select`'s event: it queues the screen's construction (game.lua:3260).
    BlindSelectOpen,
    /// The blind-select screen's construction (game.lua:3261-3294).
    BlindSelectBuild,
    /// `skip_blind`'s event: the jokers and tags see the skip (button_callbacks.lua:2762-2773).
    SkipBlindBody,
    /// A card's own queued work: `f(game, card, value)`. Jokers and consumables use it for
    /// event closures no other card shares, so their code stays in one place.
    Call(fn(&mut Game, u32, f64), u32, f64),
}

impl Game {
    /// Run one event's body.
    pub(crate) fn run_event(&mut self, event: Event) {
        match event {
            Event::SetState(state, complete) => self.set_state(state, complete),
            Event::StateIncomplete => self.rt.state_complete = false,
            Event::DrawCard {
                from,
                to,
                card,
                sort,
            } => self.draw_card_event(from, to, card, sort),
            Event::EaseDollars(m) => self.gs.dollars += m,
            Event::EaseDiscard(m) => self.ease_discard_now(m),
            Event::EaseHandsPlayed(m) => self.gs.current_round.hands_left += m,
            Event::EaseAnte(m) => self.gs.round_resets.ante += m,
            Event::EaseRound(m) => self.gs.round += m,
            Event::SetChips(v) => self.gs.chips = v,
            Event::ClearDollarBuffer => self.rt.dollar_buffer = 0.0,
            Event::ClearConsumeableBuffer => self.rt.consumeable_buffer = 0.0,
            Event::ClearJokerBuffer => self.rt.joker_buffer = 0.0,
            Event::SelectBlind => self.select_blind_event(),
            Event::NewRound => self.new_round(),
            Event::NewRoundBody => self.new_round_body(),
            Event::NewRoundShuffle => self.new_round_shuffle(),
            Event::DrawToHandBody => self.draw_to_hand_body(),
            Event::DrawnToHand => {
                self.set_state(LuaState::SelectingHand, false);
                self.blind_drawn_to_hand();
            }
            Event::PlayHandQueued => {
                self.queue(Event::EvaluatePlay);
                self.queue(Event::AfterPlay);
                self.queue(Event::StateIncomplete);
            }
            Event::EvaluatePlay => self.evaluate_play(),
            Event::AfterPlay => {
                self.draw_from_play_to_discard();
                self.gs.hands_played += 1.0;
                self.gs.current_round.hands_played += 1.0;
            }
            Event::HandPlayedCheck => {
                let blind_chips = self.gs.blind.as_ref().map_or(0.0, |b| b.chips);
                let next = if crate::lua::sub(self.gs.chips, blind_chips) >= 0.0
                    || self.gs.current_round.hands_left < 1.0
                {
                    LuaState::NewRound
                } else {
                    LuaState::DrawToHand
                };
                self.set_state(next, false);
            }
            Event::EndRound => self.end_round_body(),
            Event::EndRoundFinal => self.end_round_final(),
            Event::DiscardToDeck => {
                let n = self.gs.areas.discard.len();
                for _ in 0..n {
                    self.queue(Event::DrawCard {
                        from: AreaId::Discard,
                        to: AreaId::Deck,
                        card: None,
                        sort: false,
                    });
                }
            }
            Event::RoundEvalOpen => {
                self.rt.facing_blind = false;
                self.evaluate_round();
            }
            Event::BlindDefeat => self.blind_defeat(),
            Event::BlindReset => self.set_blind(None, false),
            Event::SetCashout(d) => self.gs.sim.pending_cashout = d,
            Event::RemoveCard(id) => self.remove_card(id),
            Event::HookDiscard => self.hook_discard(),
            Event::Tooth => {
                for _ in 0..self.gs.areas.play.len() {
                    self.queue(Event::EaseDollars(-1.0));
                }
            }
            Event::ShuffleJokers => {
                self.gs
                    .areas
                    .shuffle(AreaId::Jokers, &mut self.gs.rng, "aajk");
            }
            Event::ChangeSize { area, delta } => self.change_size_event(area, delta),
            Event::SortHand => self.sort_hand(),
            Event::DisabledBossCheck => {
                if let Some(b) = &self.gs.blind
                    && b.boss
                    && crate::lua::sub(self.gs.chips, b.chips) >= 0.0
                {
                    self.set_state(LuaState::NewRound, false);
                }
            }
            Event::AmberAcorn => {
                for _ in 0..3 {
                    self.queue(Event::ShuffleJokers);
                }
            }
            Event::BlindRedebuff => {
                if self.gs.blind.is_some() {
                    self.set_blind(None, true);
                }
            }
            Event::SetCostAll => {
                for a in AreaId::ALL {
                    let ctx = self.cost_context(a);
                    for c in self.gs.areas.get_mut(a) {
                        c.set_cost(&ctx);
                    }
                }
            }
            Event::CreateConsumable { ty, append, forced } => {
                let card =
                    self.create_card(CardSpec::new(ty, AreaId::Consumables, append).forced(forced));
                let id = card.sort_id;
                self.gs
                    .areas
                    .emplace(AreaId::Consumables, card, false, false);
                self.add_to_deck(id, false);
                self.rt.consumeable_buffer = 0.0;
            }
            Event::BlueSealPlanet => {
                if let Some(hand) = self.gs.last_hand_played {
                    let planet = jackdaw_data::POOL_PLANET.iter().copied().rfind(|p| {
                        p.get().config("hand_type").and_then(|v| v.as_str()) == Some(hand.name())
                    });
                    let card = self.create_card(
                        CardSpec::new(PoolType::Planet, AreaId::Consumables, "blusl")
                            .forced(planet),
                    );
                    let id = card.sort_id;
                    self.gs
                        .areas
                        .emplace(AreaId::Consumables, card, false, false);
                    self.add_to_deck(id, false);
                    self.rt.consumeable_buffer = 0.0;
                }
            }
            Event::RemoveTag(tag) => self.remove_tag(tag),
            Event::AddTag(tag, orbital_hand) => {
                let tag = self.new_tag(tag, orbital_hand);
                self.add_tag(tag);
            }
            Event::EaseChips(v) => self.queue(Event::SetChips(v)),
            Event::CashOutBody => self.cash_out_body(),
            Event::ShopOpen(exists) => self.queue(Event::ShopBuild(exists)),
            Event::ShopBuild(exists) => self.shop_build(exists),
            Event::StoreJokerModify(id) => {
                if let Some(card) = self.gs.areas.card(id).cloned() {
                    self.apply_tags_store_joker_modify(&card);
                }
            }
            Event::RerollBody => self.reroll_body(),
            Event::RerollAfter => self.queue(Event::RerollJokers),
            Event::RerollJokers => self.reroll_jokers(),
            Event::BuyFromShop(id) => self.buy_from_shop(id),
            Event::SellPayout(id) => self.sell_payout(id),
            Event::LeaveShop => self.leave_shop(),
            Event::BlindSelectOpen => self.queue(Event::BlindSelectBuild),
            Event::BlindSelectBuild => self.blind_select_build(),
            Event::SkipBlindBody => self.skip_blind_body(),
            Event::Dissolve(id) => self.queue(Event::RemoveCard(id)),
            Event::Call(f, card, value) => f(self, card, value),
            Event::PermaDebuff(ids) => {
                for id in ids {
                    if let Some(c) = self.gs.areas.card_mut(id) {
                        c.ability.perma_debuff = true;
                        c.debuff = true;
                    }
                }
            }
        }
    }
}

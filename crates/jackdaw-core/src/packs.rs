//! Booster packs: opening (`Card:open`, card.lua:1681-1806), the pack screens
//! (`Game:update_arcana_pack` and friends, game.lua:3341-3570), picking a card
//! (`G.FUNCS.use_card` in a pack, button_callbacks.lua:2155-2316), skipping
//! (`G.FUNCS.skip_booster`, 2558-2563) and closing (`G.FUNCS.end_consumeable`, 2565-2637).
//!
//! `Card:open` creates the contents in a timed event that doesn't block the queue, so the jokers'
//! `open_booster` effects (Hallucination's Tarot) and the Inflation pass, queued after it, run
//! first in the real game (G44). The port queues the contents after them.

use jackdaw_data::{CenterId, CenterSet, POOL_PLANET};

use crate::area::AreaId;
use crate::card::Seal;
use crate::event::Event;
use crate::game::{Game, LuaState};
use crate::hand_eval::HandType;
use crate::joker::{Ctx, Kind};
use crate::pools::{CardSpec, PoolType};
use crate::state::{PackState, Phase};

impl Game {
    /// `G.FUNCS.use_card` for a booster, then `Card:open`: the pack's state, its price, and
    /// the event that fills it. The booster leaves its area for the play area.
    pub(crate) fn open_booster(&mut self, id: u32) {
        let Some((area, _)) = self.gs.areas.locate(id) else {
            return;
        };
        let card = self
            .gs
            .areas
            .remove_card(area, Some(id))
            .expect("the opened booster");
        let center = card.center();
        let kind = center.kind.expect("a booster has a kind");
        let state = LuaState::for_pack(kind).expect("a known booster kind");
        // `G.GAME.PACK_INTERRUPT = G.STATE`: the state the pack returns to.
        let return_phase = self.gs.phase;
        let size = card.ability.extra.num().expect("a booster's size");
        let choose = center.config_num("choose").unwrap_or(1.0);
        let cost = card.cost;
        self.gs.areas.emplace(AreaId::Play, card, false, false);
        self.set_state(state, false);
        self.gs.pack_size = size;
        self.gs.pack = Some(PackState {
            kind: kind.to_owned(),
            choices_remaining: choose,
            dealt: Vec::new(),
            return_phase,
        });
        if cost > 0.0 {
            self.ease_dollars(-cost);
        }
        self.queue(Event::OpenBooster(id));
    }

    /// `Card:open`'s event (card.lua:1724-1803): the jokers see the pack, Inflation raises
    /// prices, then the contents arrive and the booster is gone.
    pub(crate) fn open_booster_body(&mut self, id: u32) {
        for j in self.joker_ids() {
            let mut ctx = Ctx::new(Kind::OpenBooster);
            ctx.card = Some(id);
            self.calculate_joker(j, &mut ctx);
        }
        if self.gs.modifier("inflation").is_some() {
            self.gs.inflation += 1.0;
            self.queue(Event::SetCostAll);
        }
        self.queue(Event::PackContents(id));
        self.queue(Event::RemoveCard(id));
    }

    /// The pack's cards (card.lua:1731-1781), emplaced into `G.pack_cards`.
    pub(crate) fn pack_contents(&mut self, id: u32) {
        let Some(booster) = self.gs.areas.card(id) else {
            return;
        };
        let kind = booster.center().kind.expect("a booster kind");
        let size = booster.ability.extra.num().expect("a booster's size");
        let ante = crate::lua::tostring_number(self.gs.round_resets.ante);
        let area = AreaId::PackCards;
        let mut cards = Vec::new();
        let mut i = 1.0;
        while i <= size {
            let card = match kind {
                "Arcana" => {
                    if self.gs.sets.used_vouchers.contains("v_omen_globe")
                        && self.gs.rng.random("omen_globe") > 0.8
                    {
                        self.create_card(CardSpec::new(PoolType::Spectral, area, "ar2").soulable())
                    } else {
                        self.create_card(CardSpec::new(PoolType::Tarot, area, "ar1").soulable())
                    }
                }
                "Celestial" => {
                    let forced = if self.gs.sets.used_vouchers.contains("v_telescope") && i == 1.0 {
                        self.telescope_planet()
                    } else {
                        None
                    };
                    self.create_card(
                        CardSpec::new(PoolType::Planet, area, "pl1")
                            .soulable()
                            .forced(forced),
                    )
                }
                "Spectral" => {
                    self.create_card(CardSpec::new(PoolType::Spectral, area, "spe").soulable())
                }
                "Standard" => {
                    let ty = if self.gs.rng.random(&format!("stdset{ante}")) > 0.6 {
                        PoolType::Enhanced
                    } else {
                        PoolType::Base
                    };
                    let mut card = self.create_card(CardSpec::new(ty, area, "sta").soulable());
                    let cost = self.cost_context(area);
                    let edition =
                        self.poll_edition(&format!("standard_edition{ante}"), 2.0, true, false);
                    card.set_edition(edition, &cost);
                    if self.gs.rng.random(&format!("stdseal{ante}")) > 1.0 - 0.02 * 10.0 {
                        let t = self.gs.rng.random(&format!("stdsealtype{ante}"));
                        let seal = if t > 0.75 {
                            Seal::Red
                        } else if t > 0.5 {
                            Seal::Blue
                        } else if t > 0.25 {
                            Seal::Gold
                        } else {
                            Seal::Purple
                        };
                        card.set_seal(Some(seal), &cost);
                    }
                    card
                }
                "Buffoon" => {
                    self.create_card(CardSpec::new(PoolType::Joker, area, "buf").soulable())
                }
                other => unreachable!("booster kind {other}"),
            };
            cards.push(card);
            i += 1.0;
        }
        for card in cards {
            self.gs.areas.emplace(area, card, false, false);
        }
    }

    /// Telescope: the first Celestial card is the Planet of the most played visible hand,
    /// earliest in `G.handlist` on a tie (card.lua:1745-1758).
    fn telescope_planet(&self) -> Option<CenterId> {
        let mut best = None;
        let mut tally = 0.0;
        // `G.handlist` is the display order (`HandType::ALL`), not `pairs`.
        for h in HandType::ALL {
            let hs = self.gs.hand(h);
            if hs.visible && hs.played > tally {
                best = Some(h);
                tally = hs.played;
            }
        }
        let hand = best?;
        POOL_PLANET
            .iter()
            .copied()
            .rfind(|p| p.get().config("hand_type").and_then(|v| v.as_str()) == Some(hand.name()))
    }

    /// The pack screen's state handler: an Arcana or Spectral pack deals a hand
    /// (game.lua:3345-3388, 3396-3439); the others only show their cards.
    pub(crate) fn update_pack(&mut self) -> bool {
        if self.rt.state_complete {
            return false;
        }
        self.rt.state_complete = true;
        if matches!(self.rt.state, LuaState::TarotPack | LuaState::SpectralPack) {
            self.queue(Event::PackScreen);
        }
        true
    }

    /// `G.FUNCS.use_card` for a pack card (button_callbacks.lua:2155-2316): a consumable is
    /// used, a joker joins the jokers, a playing card the deck. Then the pack counts the pick.
    pub(crate) fn pick_pack_card(&mut self, id: u32) {
        let card = self.card(id);
        let set = card.set();
        if card.center().consumeable {
            self.use_consumable_card(id, AreaId::PackCards);
            return;
        }
        let card = self
            .gs
            .areas
            .remove_card(AreaId::PackCards, Some(id))
            .expect("the picked card");
        match set {
            CenterSet::Default | CenterSet::Enhanced => {
                self.gs.areas.emplace(AreaId::Deck, card, false, false);
                self.playing_card_joker_effects(&[id]);
            }
            _ => {
                self.gs.areas.emplace(AreaId::Jokers, card, false, false);
                self.add_to_deck(id, false);
            }
        }
        self.queue(Event::UseCardAfter {
            card: id,
            dissolve: false,
            prev: self.rt.state,
            from: AreaId::PackCards,
        });
    }

    /// `use_card`'s timed event (button_callbacks.lua:2252-2254): the used card dissolves,
    /// then the nested event restores the state.
    pub(crate) fn use_card_after(
        &mut self,
        card: u32,
        dissolve: bool,
        prev: LuaState,
        from: AreaId,
    ) {
        if dissolve {
            self.queue(Event::RemoveCard(card));
        }
        self.queue(Event::UseCardDone { prev, from });
    }

    /// `use_card`'s nested event (button_callbacks.lua:2253-2310): the state comes back, and in a
    /// pack a pick is counted; the last pick closes the pack.
    pub(crate) fn use_card_done(&mut self, prev: LuaState, from: AreaId) {
        self.rt.state = prev;
        if let Some(phase) = prev.phase() {
            self.gs.phase = phase;
        }
        let in_pack = matches!(prev.phase(), Some(Phase::PackOpening));
        if in_pack && self.gs.pack.is_some() && from != AreaId::Consumables {
            let pack = self.gs.pack.as_mut().expect("the open pack");
            if pack.choices_remaining > 1.0 {
                pack.choices_remaining -= 1.0;
            } else {
                self.end_consumeable();
            }
        }
    }

    /// `G.FUNCS.skip_booster`: the jokers see the skip, then the pack closes.
    pub(crate) fn skip_booster(&mut self) {
        for j in self.joker_ids() {
            let mut ctx = Ctx::new(Kind::SkippingBooster);
            self.calculate_joker(j, &mut ctx);
        }
        self.end_consumeable();
    }

    /// `G.FUNCS.end_consumeable`: the pack's screen and leftover cards go, the hand returns to
    /// the deck, then the game goes back to where the pack was opened.
    pub(crate) fn end_consumeable(&mut self) {
        self.queue(Event::PackClose);
        self.queue(Event::PackHandToDeck);
    }

    /// `end_consumeable`'s second event (button_callbacks.lua:2589-2632).
    pub(crate) fn pack_hand_to_deck(&mut self) {
        // `G.FUNCS.draw_from_hand_to_deck()` (state_events.lua:1121-1126).
        for _ in 0..self.gs.areas.hand.len() {
            self.queue(Event::DrawCard {
                from: AreaId::Hand,
                to: AreaId::Deck,
                card: None,
                sort: false,
            });
        }
        self.queue(Event::PackReturn);
        self.apply_tags_new_blind_choice();
    }

    /// `end_consumeable`'s nested event: `G.STATE = G.GAME.PACK_INTERRUPT`.
    pub(crate) fn pack_return(&mut self) {
        let Some(pack) = self.gs.pack.take() else {
            return;
        };
        let state = match pack.return_phase {
            Phase::Shop => LuaState::Shop,
            Phase::BlindSelect => LuaState::BlindSelect,
            other => panic!("a pack returns to the shop or blind select, not {other:?}"),
        };
        self.rt.state = state;
        self.gs.phase = pack.return_phase;
    }

    /// The pack screen's nested event: `G.FUNCS.draw_from_deck_to_hand()`, then the dealt
    /// cards are recorded for `pack.dealt`.
    pub(crate) fn pack_deal(&mut self) {
        self.draw_from_deck_to_hand(None);
        self.queue(Event::PackDealt);
    }

    /// `pack.dealt`: the hand's cards after the deal (the hand is empty when a pack opens).
    pub(crate) fn pack_dealt(&mut self) {
        let ids = self.gs.areas.ids(AreaId::Hand);
        if let Some(pack) = self.gs.pack.as_mut() {
            pack.dealt = ids;
        }
    }
}

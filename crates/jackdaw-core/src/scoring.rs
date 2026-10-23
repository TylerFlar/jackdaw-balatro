//! Scoring a played hand: `G.FUNCS.evaluate_play` (state_events.lua:571-1086), `eval_card`
//! (common_events.lua:580-656) and `level_up_hand` (common_events.lua:464-493).

use jackdaw_data::HANDS;

use crate::area::AreaId;
use crate::card::Card;
use crate::event::Event;
use crate::game::Game;
use crate::hand_eval::{EvalCard, EvalFlags, HandType, PokerHands, get_poker_hand_info};
use crate::joker::{Ctx, Kind};
use crate::lua;
use crate::state::LastScore;

/// `card:get_edition()` (card.lua:1016-1031): an edition's scoring bonuses.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EditionMods {
    pub chip_mod: Option<f64>,
    pub mult_mod: Option<f64>,
    pub x_mult_mod: Option<f64>,
}

/// `extra = {mult_mod, chip_mod, swap, func}` in a joker's individual effect.
#[derive(Clone, Debug, Default)]
pub struct ExtraMods {
    pub mult_mod: Option<f64>,
    pub chip_mod: Option<f64>,
    pub swap: bool,
    /// `extra.func`, run when the effect is applied: 8 Ball queues its Tarot (card.lua:3110).
    pub event: Option<Event>,
}

/// An effect table: an `eval_card` result or a `calculate_joker` return. Lua tables with
/// optional keys; `None` is nil.
#[derive(Clone, Debug, Default)]
pub struct Effect {
    pub chips: Option<f64>,
    pub mult: Option<f64>,
    pub x_mult: Option<f64>,
    pub p_dollars: Option<f64>,
    pub dollars: Option<f64>,
    pub h_mult: Option<f64>,
    pub h_dollars: Option<f64>,
    pub edition: Option<EditionMods>,
    pub extra: Option<ExtraMods>,
    /// joker_main / other_joker effects.
    pub chip_mod: Option<f64>,
    pub mult_mod: Option<f64>,
    pub xmult_mod: Option<f64>,
    pub repetitions: Option<f64>,
    /// The card credited (`card = self`; Blueprint credits itself).
    pub card: Option<u32>,
    pub message: bool,
    pub saved: bool,
    pub remove: bool,
    pub level_up: bool,
    /// Blue Seal's `ret.effect`.
    pub effect: bool,
}

impl Effect {
    /// `next(t) == nil`.
    pub fn is_empty(&self) -> bool {
        let Effect {
            chips,
            mult,
            x_mult,
            p_dollars,
            dollars,
            h_mult,
            h_dollars,
            edition,
            extra,
            chip_mod,
            mult_mod,
            xmult_mod,
            repetitions,
            card,
            message,
            saved,
            remove,
            level_up,
            effect,
        } = self;
        chips.is_none()
            && mult.is_none()
            && x_mult.is_none()
            && p_dollars.is_none()
            && dollars.is_none()
            && h_mult.is_none()
            && h_dollars.is_none()
            && edition.is_none()
            && extra.is_none()
            && chip_mod.is_none()
            && mult_mod.is_none()
            && xmult_mod.is_none()
            && repetitions.is_none()
            && card.is_none()
            && !message
            && !saved
            && !remove
            && !level_up
            && !effect
    }
}

/// `mod_mult(_mult)` (misc_functions.lua:691).
pub fn mod_mult(m: f64) -> f64 {
    m
}

impl Game {
    /// `mod_chips(_chips)` (misc_functions.lua:684-689): the `chips_dollar_cap` challenge rule.
    pub fn mod_chips(&self, chips: f64) -> f64 {
        if self.gs.modifier("chips_dollar_cap").is_some() {
            return chips.min(self.gs.dollars.max(0.0));
        }
        chips
    }

    /// The joker flags hand evaluation reads (non-debuffed Four Fingers, Shortcut).
    pub fn eval_flags(&self) -> EvalFlags {
        EvalFlags {
            four_fingers: self.has_joker("Four Fingers"),
            shortcut: self.has_joker("Shortcut"),
        }
    }

    /// The evaluation view of cards (`get_id`, flush suits with Smeared Joker, `get_nominal`).
    pub fn eval_cards(&self, cards: &[Card]) -> Vec<EvalCard> {
        let smeared = self.has_joker("Smeared Joker");
        cards
            .iter()
            .map(|c| EvalCard {
                id: c.get_id(),
                flush_suits: c.flush_suits(smeared),
                nominal: c.get_nominal(false),
            })
            .collect()
    }

    /// `level_up_hand(card, hand, instant, amount)` (common_events.lua:464-493).
    pub(crate) fn level_up_hand(&mut self, hand: HandType, amount: f64) {
        let base = &HANDS[hand.index()];
        let h = self.gs.hand_mut(hand);
        h.level = (h.level + amount).max(0.0);
        h.mult = (base.s_mult + base.l_mult * (h.level - 1.0)).max(1.0);
        h.chips = (base.s_chips + base.l_chips * (h.level - 1.0)).max(0.0);
    }

    /// `eval_card(card, context)` for a joker or consumable in `G.jokers` context:
    /// `card:calculate_joker(context)`, or the edition with `context.edition`.
    pub(crate) fn eval_joker(&mut self, id: u32, ctx: &mut Ctx) -> Option<Effect> {
        self.calculate_joker(id, ctx)
    }

    /// `card:get_edition()`.
    fn edition_mods(&self, id: u32) -> Option<EditionMods> {
        let c = self.gs.areas.card(id)?;
        let e = c.get_edition()?;
        Some(EditionMods {
            chip_mod: e.chips,
            mult_mod: e.mult,
            x_mult_mod: e.x_mult,
        })
    }

    /// `eval_card(card, {cardarea = G.play, ...})` for a scoring playing card.
    fn eval_played_card(&mut self, id: u32) -> Effect {
        let mut ret = Effect::default();
        let normal = self.gs.probabilities_normal;
        let Some((area, i)) = self.gs.areas.locate(id) else {
            return ret;
        };
        let rng = &mut self.gs.rng;
        let card = &mut self.gs.areas.get_mut(area)[i];
        let chips = card.get_chip_bonus();
        if chips > 0.0 {
            ret.chips = Some(chips);
        }
        let mult = card.get_chip_mult(rng, normal);
        if mult > 0.0 {
            ret.mult = Some(mult);
        }
        let x_mult = card.get_chip_x_mult();
        if x_mult > 0.0 {
            ret.x_mult = Some(x_mult);
        }
        let p_dollars = card.get_p_dollars(rng, normal);
        if p_dollars > 0.0 {
            ret.p_dollars = Some(p_dollars);
            // Card:get_p_dollars feeds the dollar buffer (card.lua:1084-1087).
            self.rt.dollar_buffer += p_dollars;
            self.queue(Event::ClearDollarBuffer);
        }
        ret.edition = self.edition_mods(id);
        ret
    }

    /// `eval_card(card, {cardarea = G.hand, ...})` for a held playing card.
    fn eval_held_card(&mut self, id: u32) -> Effect {
        let mut ret = Effect::default();
        let Some(c) = self.gs.areas.card(id) else {
            return ret;
        };
        let h_mult = c.get_chip_h_mult();
        if h_mult > 0.0 {
            ret.h_mult = Some(h_mult);
        }
        let h_x_mult = c.get_chip_h_x_mult();
        if h_x_mult > 0.0 {
            ret.x_mult = Some(h_x_mult);
        }
        ret
    }

    /// The scoring context for a card of the played hand.
    fn scoring_ctx(&self, kind: Kind, s: &Scoring) -> Ctx {
        let mut ctx = Ctx::new(kind);
        ctx.full_hand = s.full_hand.clone();
        ctx.scoring_hand = s.scoring_hand.clone();
        ctx.scoring_name = Some(s.text);
        ctx.poker_hands = Some(s.poker_hands);
        ctx
    }

    /// `G.FUNCS.evaluate_play()` (state_events.lua:571-1086).
    pub(crate) fn evaluate_play(&mut self) {
        let full_hand = self.gs.areas.ids(AreaId::Play);
        let cards: Vec<Card> = self.gs.areas.play.clone();
        let info = get_poker_hand_info(&self.eval_cards(&cards), self.eval_flags());
        let Some(text) = info.hand else {
            return;
        };
        let h = self.gs.hand_mut(text);
        h.played += 1.0;
        h.played_this_round += 1.0;
        h.visible = true;
        self.gs.last_hand_played = Some(text);

        // The scoring hand: the best hand's cards, plus every played Stone Card (or every
        // played card with Splash), in play order.
        let mut scoring: Vec<u32> = info
            .scoring
            .as_slice()
            .iter()
            .map(|&i| full_hand[i as usize])
            .collect();
        if self.has_joker("Splash") {
            scoring = full_hand.clone();
        } else {
            for c in &cards {
                if c.is_stone() && !scoring.contains(&c.sort_id) {
                    scoring.push(c.sort_id);
                }
            }
        }
        scoring.sort_by_key(|id| full_hand.iter().position(|p| p == id));
        let s = Scoring {
            full_hand: full_hand.clone(),
            scoring_hand: scoring.clone(),
            text,
            poker_hands: info.poker_hands,
        };

        let mut mult;
        let mut hand_chips;
        let debuffed = self.blind_debuff_hand(full_hand.len(), &info.poker_hands, text, false);
        if !debuffed {
            for j in self.gs.areas.ids(AreaId::Jokers) {
                let mut ctx = self.scoring_ctx(Kind::Before, &s);
                ctx.cardarea = Some(AreaId::Jokers);
                if let Some(e) = self.eval_joker(j, &mut ctx)
                    && e.level_up
                {
                    self.level_up_hand(text, 1.0);
                }
            }
            mult = mod_mult(self.gs.hand(text).mult);
            hand_chips = self.mod_chips(self.gs.hand(text).chips);
            let (m, c, _) = self.blind_modify_hand(mult, hand_chips);
            mult = mod_mult(m);
            hand_chips = self.mod_chips(c);

            for &id in &scoring {
                if self.gs.areas.card(id).is_some_and(|c| c.debuff) {
                    if let Some(b) = self.gs.blind.as_mut() {
                        b.triggered = true;
                    }
                    continue;
                }
                let mut reps = 1usize;
                if let Some(n) = self.red_seal_repetitions(id) {
                    reps += n;
                }
                for j in self.gs.areas.ids(AreaId::Jokers) {
                    let mut ctx = self.scoring_ctx(Kind::Repetition, &s);
                    ctx.cardarea = Some(AreaId::Play);
                    ctx.other_card = Some(id);
                    if let Some(e) = self.eval_joker(j, &mut ctx) {
                        reps += e.repetitions.unwrap_or(0.0) as usize;
                    }
                }
                for _ in 0..reps {
                    let mut effects = vec![self.eval_played_card(id)];
                    for k in self.gs.areas.ids(AreaId::Jokers) {
                        let mut ctx = self.scoring_ctx(Kind::Individual, &s);
                        ctx.cardarea = Some(AreaId::Play);
                        ctx.other_card = Some(id);
                        if let Some(e) = self.calculate_joker(k, &mut ctx) {
                            effects.push(e);
                        }
                    }
                    if let Some(c) = self.gs.areas.card_mut(id) {
                        c.lucky_trigger = false;
                    }
                    for e in &effects {
                        if let Some(x) = e.chips {
                            hand_chips = self.mod_chips(lua::add(hand_chips, x));
                        }
                        if let Some(x) = e.mult {
                            mult = mod_mult(lua::add(mult, x));
                        }
                        if let Some(x) = e.p_dollars {
                            self.queue(Event::EaseDollars(x));
                        }
                        if let Some(x) = e.dollars {
                            self.queue(Event::EaseDollars(x));
                        }
                        if let Some(x) = &e.extra {
                            if let Some(m) = x.mult_mod {
                                mult = mod_mult(lua::add(mult, m));
                            }
                            if let Some(c) = x.chip_mod {
                                hand_chips = self.mod_chips(lua::add(hand_chips, c));
                            }
                            if x.swap {
                                let old = mult;
                                mult = mod_mult(hand_chips);
                                hand_chips = self.mod_chips(old);
                            }
                            if let Some(ev) = &x.event {
                                self.queue(ev.clone());
                            }
                        }
                        if let Some(x) = e.x_mult {
                            mult = mod_mult(lua::mul(mult, x));
                        }
                        if let Some(ed) = e.edition {
                            hand_chips =
                                self.mod_chips(lua::add(hand_chips, ed.chip_mod.unwrap_or(0.0)));
                            mult = lua::add(mult, ed.mult_mod.unwrap_or(0.0));
                            mult = mod_mult(lua::mul(mult, ed.x_mult_mod.unwrap_or(1.0)));
                        }
                    }
                }
            }

            for id in self.gs.areas.ids(AreaId::Hand) {
                let mut reps = 1usize;
                let mut j = 0;
                while j < reps {
                    let mut effects = vec![self.eval_held_card(id)];
                    for k in self.gs.areas.ids(AreaId::Jokers) {
                        let mut ctx = self.scoring_ctx(Kind::Individual, &s);
                        ctx.cardarea = Some(AreaId::Hand);
                        ctx.other_card = Some(id);
                        if let Some(e) = self.calculate_joker(k, &mut ctx) {
                            effects.push(e);
                        }
                    }
                    if j == 0 {
                        let has_effect = !effects[0].is_empty() || effects.len() > 1;
                        if has_effect && let Some(n) = self.red_seal_repetitions(id) {
                            reps += n;
                        }
                        for k in self.gs.areas.ids(AreaId::Jokers) {
                            let mut ctx = self.scoring_ctx(Kind::Repetition, &s);
                            ctx.cardarea = Some(AreaId::Hand);
                            ctx.other_card = Some(id);
                            ctx.card_effects = effects.iter().map(|e| !e.is_empty()).collect();
                            if let Some(e) = self.eval_joker(k, &mut ctx) {
                                reps += e.repetitions.unwrap_or(0.0) as usize;
                            }
                        }
                    }
                    for e in &effects {
                        if let Some(x) = e.dollars {
                            self.queue(Event::EaseDollars(x));
                        }
                        if let Some(x) = e.h_mult {
                            mult = mod_mult(lua::add(mult, x));
                        }
                        if let Some(x) = e.x_mult {
                            mult = mod_mult(lua::mul(mult, x));
                        }
                    }
                    j += 1;
                }
            }

            let jokers = self.gs.areas.ids(AreaId::Jokers);
            let consumables = self.gs.areas.ids(AreaId::Consumables);
            for id in jokers.iter().chain(consumables.iter()).copied() {
                let edition = self.edition_mods(id);
                if let Some(ed) = edition {
                    if let Some(c) = ed.chip_mod {
                        hand_chips = self.mod_chips(lua::add(hand_chips, c));
                    }
                    if let Some(m) = ed.mult_mod {
                        mult = mod_mult(lua::add(mult, m));
                    }
                }
                let mut ctx = self.scoring_ctx(Kind::JokerMain, &s);
                ctx.cardarea = Some(AreaId::Jokers);
                if let Some(e) = self.eval_joker(id, &mut ctx) {
                    if let Some(m) = e.mult_mod {
                        mult = mod_mult(lua::add(mult, m));
                    }
                    if let Some(c) = e.chip_mod {
                        hand_chips = self.mod_chips(lua::add(hand_chips, c));
                    }
                    if let Some(x) = e.xmult_mod {
                        mult = mod_mult(lua::mul(mult, x));
                    }
                }
                for v in self.gs.areas.ids(AreaId::Jokers) {
                    let mut ctx = self.scoring_ctx(Kind::OtherJoker, &s);
                    ctx.other_joker = Some(id);
                    if let Some(e) = self.calculate_joker(v, &mut ctx) {
                        if let Some(m) = e.mult_mod {
                            mult = mod_mult(lua::add(mult, m));
                        }
                        if let Some(c) = e.chip_mod {
                            hand_chips = self.mod_chips(lua::add(hand_chips, c));
                        }
                        if let Some(x) = e.xmult_mod {
                            mult = mod_mult(lua::mul(mult, x));
                        }
                    }
                }
                if let Some(x) = edition.and_then(|e| e.x_mult_mod) {
                    mult = mod_mult(lua::mul(mult, x));
                }
            }

            if let Some((c, m)) = self.back_final_scoring_step(hand_chips, mult) {
                mult = mod_mult(m);
                hand_chips = self.mod_chips(c);
            }

            let mut destroyed = Vec::new();
            for &id in &scoring {
                let mut hit = false;
                for j in self.gs.areas.ids(AreaId::Jokers) {
                    let mut ctx = Ctx::new(Kind::DestroyingCard);
                    ctx.destroying_card = Some(id);
                    ctx.full_hand = full_hand.clone();
                    if self.calculate_joker(j, &mut ctx).is_some() {
                        hit = true;
                        break;
                    }
                }
                let glass = self.gs.areas.card(id).and_then(|c| {
                    (c.name() == "Glass Card" && !c.debuff).then(|| c.ability.extra.num())
                });
                if let Some(extra) = glass {
                    let roll = self.gs.rng.random("glass");
                    if roll < self.gs.probabilities_normal / extra.unwrap_or(4.0) {
                        hit = true;
                    }
                }
                if hit {
                    let is_glass = self
                        .gs
                        .areas
                        .card(id)
                        .is_some_and(|c| c.name() == "Glass Card");
                    if let Some(c) = self.gs.areas.card_mut(id) {
                        if is_glass {
                            c.shattered = true;
                        } else {
                            c.destroyed = true;
                        }
                    }
                    destroyed.push(id);
                }
            }
            for j in self.gs.areas.ids(AreaId::Jokers) {
                let mut ctx = Ctx::new(Kind::RemovePlayingCards);
                ctx.cardarea = Some(AreaId::Jokers);
                ctx.removed = destroyed.clone();
                self.eval_joker(j, &mut ctx);
            }
            for id in destroyed {
                self.queue(Event::RemoveCard(id));
            }
        } else {
            mult = mod_mult(0.0);
            hand_chips = self.mod_chips(0.0);
            for j in self.gs.areas.ids(AreaId::Jokers) {
                let mut ctx = self.scoring_ctx(Kind::DebuffedHand, &s);
                ctx.cardarea = Some(AreaId::Jokers);
                self.eval_joker(j, &mut ctx);
            }
        }

        let total = lua::floor(lua::mul(hand_chips, mult));
        self.queue(Event::SetChips(lua::add(self.gs.chips, total)));
        for j in self.gs.areas.ids(AreaId::Jokers) {
            let mut ctx = self.scoring_ctx(Kind::After, &s);
            ctx.cardarea = Some(AreaId::Jokers);
            self.eval_joker(j, &mut ctx);
        }
        if self.gs.modifier("debuff_played_cards").is_some() {
            self.queue(Event::PermaDebuff(scoring));
        }
        self.gs.sim.last_score = Some(LastScore {
            hand_type: text,
            chips: hand_chips,
            mult,
            total,
            dollars: 0.0,
            debuffed,
        });
    }
}

/// The played hand as the scoring contexts see it.
struct Scoring {
    full_hand: Vec<u32>,
    scoring_hand: Vec<u32>,
    text: HandType,
    poker_hands: PokerHands,
}

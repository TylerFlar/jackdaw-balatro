//! The legal actions at a decision point, in the form the Python engine's `get_legal_actions`
//! listed them: one marker `PlayHand` and `Discard` (empty selections) for the hand, and each
//! usable consumable with the default hand selection it would be used on.
//!
//! A consumable's default selection is the first `min_highlighted` cards of the hand, or else
//! the first such combination it can be used on; Aura needs one editionless card.

use jackdaw_data::CenterSet;

use crate::action::Action;
use crate::area::AreaId;
use crate::card::EditionKind;
use crate::game::{Game, LuaState};
use crate::state::{BlindSlot, HandSort};

/// `n` choose `k`, in lexicographic order.
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if k > n {
        return out;
    }
    let mut idx: Vec<usize> = (0..k).collect();
    loop {
        out.push(idx.clone());
        // The rightmost position that can still move right.
        let mut i = k;
        while i > 0 && idx[i - 1] == i - 1 + n - k {
            i -= 1;
        }
        if i == 0 {
            return out;
        }
        idx[i - 1] += 1;
        for j in i..k {
            idx[j] = idx[j - 1] + 1;
        }
    }
}

impl Game {
    /// Every legal action here, in the Python engine's order.
    pub fn legal_actions(&self) -> Vec<Action> {
        let mut out = Vec::new();
        match self.rt.state {
            LuaState::BlindSelect => {
                out.push(Action::SelectBlind);
                if matches!(
                    self.gs.blind_on_deck.unwrap_or(BlindSlot::Small),
                    BlindSlot::Small | BlindSlot::Big
                ) {
                    out.push(Action::SkipBlind);
                }
                if self.can_reroll_boss() {
                    out.push(Action::RerollBoss);
                }
                self.sell_actions(&mut out);
                self.usable_consumables(&mut out);
                // A drag reorders the jokers in every state (G206).
                self.joker_swaps(&mut out);
            }
            LuaState::SelectingHand => {
                let n = self.gs.areas.hand.len();
                let cr = &self.gs.current_round;
                if n > 0 && cr.hands_left > 0.0 {
                    out.push(Action::PlayHand(Vec::new()));
                }
                if n > 0 && cr.discards_left > 0.0 {
                    out.push(Action::Discard(Vec::new()));
                }
                if n > 1 {
                    out.push(Action::SortHand(HandSort::Desc));
                    out.push(Action::SortHand(HandSort::SuitDesc));
                }
                self.hand_swaps(&mut out);
                self.sell_actions(&mut out);
                self.usable_consumables(&mut out);
                self.joker_swaps(&mut out);
            }
            LuaState::RoundEval => {
                out.push(Action::CashOut);
                // Card:can_sell_card has no state gate (card.lua:1640-1653; G203).
                self.sell_actions(&mut out);
                self.usable_consumables(&mut out);
                self.joker_swaps(&mut out);
            }
            LuaState::Shop => self.shop_actions(&mut out),
            LuaState::TarotPack
            | LuaState::PlanetPack
            | LuaState::SpectralPack
            | LuaState::StandardPack
            | LuaState::BuffoonPack => {
                let remaining = self.gs.pack.as_ref().map_or(0.0, |p| p.choices_remaining);
                if remaining > 0.0 {
                    for (i, card) in self.gs.areas.pack_cards.iter().enumerate() {
                        let targets = if card.center().consumeable {
                            match self.consumable_use(card.sort_id, true) {
                                Some(t) => t,
                                None => continue,
                            }
                        } else {
                            if card.set() == CenterSet::Joker && !self.joker_fits(card) {
                                continue;
                            }
                            None
                        };
                        out.push(Action::PickPackCard { index: i, targets });
                    }
                }
                if self.can_skip_booster() {
                    out.push(Action::SkipPack);
                }
                self.sell_actions(&mut out);
                self.usable_consumables(&mut out);
                // An Arcana or Spectral pack's dealt hand is dragged like a round's (G206): its
                // order picks Death's rightmost card (card.lua:1112-1117) and the order it goes
                // back to the deck (state_events.lua:1121-1126).
                self.hand_swaps(&mut out);
                self.joker_swaps(&mut out);
            }
            _ => {}
        }
        out
    }

    /// Neighbour swaps of the hand: each card but the first moving left, then each but the last
    /// moving right. A drag reorders the hand whenever it holds cards (`CardArea:align_cards`
    /// sorts it by position every frame, cardarea.lua:436-465): while choosing a hand, and in an
    /// Arcana or Spectral pack once its hand is dealt.
    fn hand_swaps(&self, out: &mut Vec<Action>) {
        let n = self.gs.areas.hand.len();
        if n > 1 {
            out.extend((1..n).map(Action::SwapHandLeft));
            out.extend((0..n - 1).map(Action::SwapHandRight));
        }
    }

    /// A pack's joker can be taken (`G.FUNCS.can_select_card`, button_callbacks.lua:2112):
    /// a Negative one always, any other with a free slot.
    fn joker_fits(&self, card: &crate::card::Card) -> bool {
        let negative = card
            .edition
            .is_some_and(|e| e.kind == EditionKind::Negative);
        negative || (self.gs.areas.jokers.len() as f64) < self.gs.joker_slots
    }

    fn spend_ok(&self, cost: f64) -> bool {
        cost <= 0.0 || cost <= self.gs.dollars - self.gs.bankrupt_at
    }

    fn shop_actions(&self, out: &mut Vec<Action>) {
        let shop = &self.gs.areas.shop_jokers;
        for (i, card) in shop.iter().enumerate() {
            if !self.spend_ok(card.cost) {
                continue;
            }
            // The same check the purchase makes (G.FUNCS.check_for_buy_space): a Negative card
            // raises the limit by one (G201, G264).
            if !self.check_for_buy_space(card) {
                continue;
            }
            out.push(Action::BuyCard(i));
        }
        for (i, card) in shop.iter().enumerate() {
            if !card.center().consumeable || !self.spend_ok(card.cost) {
                continue;
            }
            if let Some(targets) = self.consumable_use(card.sort_id, false) {
                out.push(Action::BuyAndUse { index: i, targets });
            }
        }
        self.sell_actions(out);
        self.usable_consumables(out);
        let spendable = self.gs.dollars - self.gs.bankrupt_at;
        for (i, v) in self.gs.areas.shop_vouchers.iter().enumerate() {
            if v.cost <= spendable {
                out.push(Action::RedeemVoucher(i));
            }
        }
        for (i, b) in self.gs.areas.shop_booster.iter().enumerate() {
            if self.spend_ok(b.cost) {
                out.push(Action::OpenBooster(i));
            }
        }
        let cr = &self.gs.current_round;
        if cr.free_rerolls > 0.0 || self.spend_ok(cr.reroll_cost) {
            out.push(Action::Reroll);
        }
        out.push(Action::NextRound);
        self.joker_swaps(out);
    }

    fn sell_actions(&self, out: &mut Vec<Action>) {
        for (i, j) in self.gs.areas.jokers.iter().enumerate() {
            if !j.ability.eternal {
                out.push(Action::SellCard {
                    area: AreaId::Jokers,
                    index: i,
                });
            }
        }
        for i in 0..self.gs.areas.consumables.len() {
            out.push(Action::SellCard {
                area: AreaId::Consumables,
                index: i,
            });
        }
    }

    /// Neighbour swaps of the jokers that move no pinned joker. Their area is a `'joker'` one, so
    /// its cards can be dragged (cardarea.lua:214-227) and a drag reorders them at every decision
    /// point (cardarea.lua:509-529); a pinned joker sorts back to the front (cardarea.lua:528).
    fn joker_swaps(&self, out: &mut Vec<Action>) {
        let jokers = &self.gs.areas.jokers;
        let free = |i: usize, o: usize| !jokers[i].pinned && !jokers[o].pinned;
        for i in 1..jokers.len() {
            if free(i, i - 1) {
                out.push(Action::SwapJokersLeft(i));
            }
        }
        for i in 0..jokers.len().saturating_sub(1) {
            if free(i, i + 1) {
                out.push(Action::SwapJokersRight(i));
            }
        }
    }

    fn usable_consumables(&self, out: &mut Vec<Action>) {
        for (i, c) in self.gs.areas.consumables.iter().enumerate() {
            if let Some(targets) = self.consumable_use(c.sort_id, true) {
                out.push(Action::UseConsumable { index: i, targets });
            }
        }
    }

    /// How many hand cards the consumable (by sort id) is used against (`min_highlighted`), if
    /// any.
    pub fn selection_size(&self, id: u32) -> Option<usize> {
        let center = self.card(id).center();
        if center.name == "Aura" {
            return Some(1);
        }
        center.config_num("max_highlighted")?;
        Some(center.config_num("min_highlighted").unwrap_or(1.0) as usize)
    }

    /// Whether the consumable (by sort id) can be used on `selection` (hand indices), judged
    /// with that selection highlighted (after Cerulean Bell's card, as a use highlights them).
    /// With `check_use`, `Card:check_use` must pass too: Buy and Use asks only
    /// `can_use_consumeable` (button_callbacks.lua:77-89, G138).
    pub fn usable_with(&self, id: u32, selection: &[usize], check_use: bool) -> bool {
        let Ok(highlighted) = self.consumable_highlight(selection) else {
            return false;
        };
        self.can_use_consumeable_with(id, &highlighted) && !(check_use && self.check_use(id))
    }

    /// Whether the boss can be rerolled here (Director's Cut / Retcon, $10).
    pub fn boss_reroll_allowed(&self) -> bool {
        self.can_reroll_boss()
    }

    /// Whether the open pack can be skipped now (`G.FUNCS.can_skip_booster`, G202).
    pub fn pack_skip_allowed(&self) -> bool {
        self.can_skip_booster()
    }

    /// `Some(targets)` when the consumable (by sort id) can be used now: `Some(None)` for one
    /// that takes no selection, `Some(Some(selection))` with its default selection.
    pub fn consumable_use(&self, id: u32, check_use: bool) -> Option<Option<Vec<usize>>> {
        let Some(size) = self.selection_size(id) else {
            return self.usable_with(id, &[], check_use).then_some(None);
        };
        let n = self.gs.areas.hand.len();
        if n < size {
            return None;
        }
        let first: Vec<usize> = (0..size).collect();
        if self.usable_with(id, &first, check_use) {
            return Some(Some(first));
        }
        combinations(n, size)
            .into_iter()
            .find(|sel| *sel != first && self.usable_with(id, sel, check_use))
            .map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::combinations;
    use crate::action::Action;
    use crate::area::AreaId;
    use crate::game::Game;
    use crate::run::RunSpec;

    #[test]
    fn a_drag_reorders_the_jokers_at_blind_select() {
        // G206: the joker area re-sorts by position in any state (cardarea.lua:509-529).
        let mut g = Game::new_run(&RunSpec {
            back: jackdaw_data::center_by_key("b_red").unwrap(),
            stake: 1,
            seed: "REORDER".into(),
            challenge: None,
        })
        .unwrap();
        for key in ["j_joker", "j_greedy_joker"] {
            g.apply_setup_op(&serde_json::json!({"op": "add", "key": key}))
                .unwrap();
        }
        let legal = g.legal_actions();
        assert!(legal.contains(&Action::SwapJokersRight(0)));
        assert!(legal.contains(&Action::SwapJokersLeft(1)));
        let before = g.gs.areas.ids(AreaId::Jokers);
        g.step(&Action::SwapJokersRight(0)).unwrap();
        assert_eq!(*g.gs.areas.ids(AreaId::Jokers), [before[1], before[0]]);
        // The hand holds no cards here, so it has nothing to reorder.
        assert!(!legal.iter().any(|a| matches!(a, Action::SwapHandRight(_))));
        assert!(g.step(&Action::SwapHandRight(0)).is_err());
    }

    #[test]
    fn combinations_are_lexicographic() {
        assert_eq!(
            combinations(4, 2),
            vec![
                vec![0, 1],
                vec![0, 2],
                vec![0, 3],
                vec![1, 2],
                vec![1, 3],
                vec![2, 3]
            ]
        );
        assert_eq!(combinations(3, 0), vec![Vec::<usize>::new()]);
        assert!(combinations(2, 3).is_empty());
    }
}

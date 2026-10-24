//! Debug operations: BalatroBot's `add` and `set` endpoints and the shop restock behind
//! `set {shop: true}` (third_party/balatrobot/v1.5.2/src/lua/endpoints/add.lua, set.lua), plus
//! the parity harness's setup operations (`tests/parity/harness.py`).
//!
//! These write state no player action can reach, so they follow what the retired Python
//! engine's versions did (its `debug.py`): a parity spec's setup gives the canonical snapshot
//! that engine gave, which `tests/parity/test_rust_setup.py` checks against the frozen digests.
//! Where the Python engine had no behaviour (vouchers, packs, consumable editions, sticker
//! counts) they follow BalatroBot and Steamodded's `SMODS.add_card`. The known differences
//! from the live game are rows in `docs/parity/gaps.md`:
//!
//! - a joker's `'edi'..ante` edition poll runs even when an edition is given (the live game
//!   skips it, B15);
//! - an added playing card draws no `'front'..ante` card (Steamodded's `create_card('Base')`
//!   draws one, B18) and is not checked against the boss (an enhancement passed to `SMODS.add_card`
//!   goes through a non-initial `set_ability`, which calls `Blind:debuff_card`, B19).
//!
//! Stickers are forced, as `force_stickers` does in the live game: eternal and perishable
//! together, on any joker, even ones vanilla never makes eternal (`docs/balatrobot.md`).

use jackdaw_data::{BlindId, CardId, CenterId, CenterSet, centers};
use serde_json::Value;

use crate::action::Action;
use crate::area::AreaId;
use crate::card::{EditionKind, Seal};
use crate::game::{Game, LuaState, StepError};
use crate::state::BlindSlot;

/// What `add` creates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AddKey {
    /// A joker center (`j_*`): into the jokers.
    Joker(CenterId),
    /// A Tarot, Planet or Spectral (`c_*`): into the consumables.
    Consumable(CenterId),
    /// A voucher (`v_*`): into the shop's voucher area (`SMODS.add_card{area = G.shop_vouchers}`).
    Voucher(CenterId),
    /// A booster (`p_*`): into the shop (`SMODS.add_booster_to_shop`).
    Booster(CenterId),
    /// A playing card (`H_A`): into the hand during a round, else onto the top of the deck.
    Playing(CardId),
}

impl AddKey {
    /// The kind of card a BalatroBot key names, if the key exists and has the kind its prefix
    /// claims (`c_base` is a center, but not a consumable).
    pub fn parse(key: &str) -> Option<AddKey> {
        if let Some(front) = jackdaw_data::card_by_key(key) {
            return Some(AddKey::Playing(front));
        }
        let center = jackdaw_data::center_by_key(key)?;
        let c = center.get();
        Some(match &key[..2] {
            "j_" if c.set == CenterSet::Joker => AddKey::Joker(center),
            "c_" if c.consumeable => AddKey::Consumable(center),
            "v_" if c.set == CenterSet::Voucher => AddKey::Voucher(center),
            "p_" if c.set == CenterSet::Booster => AddKey::Booster(center),
            _ => return None,
        })
    }
}

/// The arguments of `add` (after BalatroBot's validation).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AddSpec {
    pub key: AddKey,
    pub seal: Option<Seal>,
    pub edition: Option<EditionKind>,
    /// A playing card's enhancement center (`m_*`); `c_base` when absent.
    pub enhancement: Option<CenterId>,
    pub eternal: bool,
    /// Perishable for this many rounds (`perish_tally`).
    pub perishable: Option<f64>,
    pub rental: bool,
}

impl AddSpec {
    pub fn new(key: AddKey) -> AddSpec {
        AddSpec {
            key,
            seal: None,
            edition: None,
            enhancement: None,
            eternal: false,
            perishable: None,
            rental: false,
        }
    }
}

/// One `set` field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SetField {
    /// `G.GAME.dollars`.
    Money(f64),
    /// `G.GAME.chips`.
    Chips(f64),
    /// `G.GAME.round_resets.ante`.
    Ante(f64),
    /// `G.GAME.round`.
    Round(f64),
    /// `G.GAME.current_round.hands_left`.
    Hands(f64),
    /// `G.GAME.current_round.discards_left`.
    Discards(f64),
}

impl Game {
    /// BalatroBot's `add`: create the card and let the game settle. Returns its sort id. On an
    /// error the game is unchanged.
    pub fn debug_add(&mut self, spec: &AddSpec) -> Result<u32, StepError> {
        let mut next = self.clone();
        let id = next.add_card(spec)?;
        if let Some(what) = next.rt.unported.take() {
            return Err(StepError::Unported(what));
        }
        *self = next;
        Ok(id)
    }

    fn add_card(&mut self, spec: &AddSpec) -> Result<u32, StepError> {
        let in_round = self.rt.state == LuaState::SelectingHand;
        let id = match spec.key {
            AddKey::Joker(center) => {
                let mut card = self.new_card(center, None, AreaId::Jokers);
                let cost = self.cost_context(AreaId::Jokers);
                if spec.edition.is_some() {
                    card.set_edition(spec.edition, &cost);
                }
                // `force_stickers` (Steamodded's `Card:add_sticker(v, true)`): no compatibility
                // check, and BalatroBot then writes the perishable count (add.lua:446-448).
                if spec.eternal {
                    card.ability.eternal = true;
                }
                if let Some(rounds) = spec.perishable {
                    card.ability.perishable = true;
                    card.ability.perish_tally = crate::card::OptF64::some(rounds);
                }
                if spec.rental {
                    card.set_rental(true, &cost);
                }
                // create_card's poll, which the Python engine ran even when an edition was given
                // (B15).
                let ante = crate::lua::tostring_number(self.gs.round_resets.ante);
                let polled = self.poll_edition(&format!("edi{ante}"), 1.0, false, false);
                if spec.edition.is_none() {
                    card.set_edition(polled, &cost);
                }
                card.set_cost(&cost);
                let id = card.sort_id;
                let old_hand_size = self.gs.hand_size;
                self.gs.areas.emplace(AreaId::Jokers, card, false, false);
                self.add_to_deck(id, false);
                self.settle();
                // A bigger hand mid-round fills up to the new size at once, as Steamodded's
                // `handle_card_limit` does (smods-diffs.md S04).
                if in_round && self.gs.hand_size > old_hand_size {
                    self.fill_hand();
                }
                id
            }
            AddKey::Consumable(center) => {
                let mut card = self.new_card(center, None, AreaId::Consumables);
                let cost = self.cost_context(AreaId::Consumables);
                if spec.edition.is_some() {
                    card.set_edition(spec.edition, &cost);
                }
                card.set_cost(&cost);
                let id = card.sort_id;
                self.gs
                    .areas
                    .emplace(AreaId::Consumables, card, false, false);
                self.add_to_deck(id, false);
                self.settle();
                id
            }
            AddKey::Voucher(center) => {
                // `SMODS.add_card{key, area = G.shop_vouchers}`: `add_to_deck`, then the area.
                let card = self.new_card(center, None, AreaId::ShopVouchers);
                let id = card.sort_id;
                self.gs
                    .areas
                    .emplace(AreaId::ShopVouchers, card, false, false);
                self.add_to_deck(id, false);
                self.settle();
                id
            }
            AddKey::Booster(center) => {
                // `SMODS.add_booster_to_shop(key)`: a plain shop booster.
                let card = self.new_card(center, None, AreaId::ShopBooster);
                let id = card.sort_id;
                self.gs
                    .areas
                    .emplace(AreaId::ShopBooster, card, false, false);
                self.settle();
                id
            }
            AddKey::Playing(front) => {
                let center = spec.enhancement.unwrap_or(centers::C_BASE);
                let area = if in_round { AreaId::Hand } else { AreaId::Deck };
                let mut card = self.new_card(center, Some(front), area);
                let cost = self.cost_context(area);
                if spec.edition.is_some() {
                    card.set_edition(spec.edition, &cost);
                }
                if spec.seal.is_some() {
                    card.set_seal(spec.seal, &cost);
                }
                let id = card.sort_id;
                if in_round {
                    self.gs.areas.emplace(AreaId::Hand, card, false, false);
                } else {
                    // The Python engine appended to its deck list: the top of the draw pile.
                    let mut card = card;
                    card.facing = crate::card::Facing::Back;
                    self.gs.areas.deck.push(card);
                }
                self.settle();
                id
            }
        };
        Ok(id)
    }

    /// Draw up to the hand size (the Python engine's `_draw_hand`): `min(#deck, size - #hand)`
    /// cards, each sorted in.
    fn fill_hand(&mut self) {
        self.poll_hand_size();
        let room = self.gs.hand_size - self.gs.areas.hand.len() as f64;
        let n = (self.gs.areas.deck.len() as f64).min(room);
        let mut i = 0.0;
        while i < n {
            self.queue(crate::event::Event::DrawCard {
                from: AreaId::Deck,
                to: AreaId::Hand,
                card: None,
                sort: true,
            });
            i += 1.0;
        }
        self.settle();
    }

    /// One field of BalatroBot's `set`: a direct write, as set.lua:98-168 does.
    pub fn debug_set(&mut self, field: SetField) {
        match field {
            SetField::Money(v) => self.gs.dollars = v,
            SetField::Chips(v) => self.gs.chips = v,
            SetField::Ante(v) => self.gs.round_resets.ante = v,
            SetField::Round(v) => self.gs.round = v,
            SetField::Hands(v) => self.gs.current_round.hands_left = v,
            SetField::Discards(v) => self.gs.current_round.discards_left = v,
        }
    }

    /// `set {shop: true}` (set.lua:170-189): `G.shop:remove()` takes the shop's cards (their
    /// `used_jokers` marks go with them), `current_round.used_packs` is cleared, and
    /// `G:update_shop()` builds a new shop as on arrival: new jokers, the ante's voucher if it
    /// is still unbought, new packs, and the held tags' shop passes. SHOP only.
    pub fn debug_restock_shop(&mut self) -> Result<(), StepError> {
        if self.rt.state != LuaState::Shop {
            return Err(StepError::Illegal("the shop restocks only in SHOP".into()));
        }
        let mut next = self.clone();
        for area in [
            AreaId::ShopJokers,
            AreaId::ShopVouchers,
            AreaId::ShopBooster,
        ] {
            for id in next.gs.areas.ids(area) {
                next.remove_card(id);
            }
        }
        next.rt.shop_open = false;
        next.rt.state_complete = false;
        next.settle();
        if let Some(what) = next.rt.unported.take() {
            return Err(StepError::Unported(what));
        }
        *self = next;
        Ok(())
    }

    /// BalatroBot's `rearrange` (rearrange.lua:173-203): the area's cards in the order of
    /// `order`, a permutation of their positions (`G.hand.cards = new_array`). The next
    /// `CardArea:align_cards` puts pinned jokers back on the left (cardarea.lua:528).
    pub fn rearrange(&mut self, area: AreaId, order: &[usize]) -> Result<(), StepError> {
        let cards = self.gs.areas.get(area);
        let mut seen = vec![false; cards.len()];
        if order.len() != cards.len()
            || order
                .iter()
                .any(|&i| i >= seen.len() || std::mem::replace(&mut seen[i], true))
        {
            return Err(StepError::Illegal(format!(
                "{order:?} is not a permutation of {} cards",
                cards.len()
            )));
        }
        let new: Vec<_> = order.iter().map(|&i| cards[i]).collect();
        *self.gs.areas.get_mut(area) = new;
        self.gs.areas.align(area);
        Ok(())
    }

    /// BalatroBot's gates for using a consumable (use.lua:167-183, pack.lua): with `targets`
    /// highlighted as a use highlights them, `(card:can_use_consumeable(), card:check_use())`
    /// (the second is true when the use is refused for lack of room).
    pub fn consumable_gates(&self, id: u32, targets: &[usize]) -> Result<(bool, bool), StepError> {
        let mut g = self.clone();
        g.rt.highlighted = g.consumable_highlight(targets)?;
        Ok((g.can_use_consumeable(id), g.check_use(id)))
    }

    /// Make `blind` the ante's boss and put it on deck (the parity harness's `force_boss`).
    pub fn debug_force_boss(&mut self, blind: BlindId) {
        self.gs.round_resets.blind_choices[BlindSlot::Boss.index()] = Some(blind);
        self.gs.blind_on_deck = Some(BlindSlot::Boss);
    }

    /// Apply one setup operation of a parity spec (`tests/parity/harness.py`): `force_boss`,
    /// `action`, `add` (the Python engine's names: `{"key": "H_A", "seal": "Red",
    /// "enhancement": "m_steel", "edition": "foil"}`) or `set` (`dollars`, `chips`,
    /// `hands_left`, `ante`).
    pub fn apply_setup_op(&mut self, op: &Value) -> Result<(), String> {
        let kind = op
            .get("op")
            .and_then(Value::as_str)
            .ok_or("a setup op needs an \"op\"")?;
        let str_field = |k: &str| op.get(k).and_then(Value::as_str);
        match kind {
            "force_boss" => {
                let key = str_field("key").ok_or("force_boss needs a key")?;
                let blind =
                    jackdaw_data::blind_by_key(key).ok_or(format!("unknown blind {key:?}"))?;
                self.debug_force_boss(blind);
            }
            "action" => {
                let action = Action::from_json(op.get("action").ok_or("action needs an action")?)?;
                self.step(&action).map_err(|e| e.to_string())?;
            }
            "add" => {
                let key = str_field("key").ok_or("add needs a key")?;
                let parsed = AddKey::parse(key).ok_or(format!("unknown card key {key:?}"))?;
                let mut spec = AddSpec::new(parsed);
                if let Some(s) = str_field("seal") {
                    spec.seal = Some(Seal::from_name(s).ok_or(format!("unknown seal {s:?}"))?);
                }
                if let Some(e) = str_field("edition") {
                    spec.edition =
                        Some(EditionKind::from_name(e).ok_or(format!("unknown edition {e:?}"))?);
                }
                if let Some(m) = str_field("enhancement")
                    && m != "c_base"
                {
                    spec.enhancement = Some(
                        jackdaw_data::center_by_key(m)
                            .ok_or(format!("unknown enhancement {m:?}"))?,
                    );
                }
                self.debug_add(&spec).map_err(|e| e.to_string())?;
            }
            "set" => {
                let obj = op.as_object().ok_or("set must be an object")?;
                for (field, value) in obj {
                    let v = || {
                        value
                            .as_f64()
                            .ok_or(format!("set {field} needs a number, got {value}"))
                    };
                    match field.as_str() {
                        "op" => {}
                        "dollars" => self.debug_set(SetField::Money(v()?)),
                        "chips" => self.debug_set(SetField::Chips(v()?)),
                        "hands_left" => self.debug_set(SetField::Hands(v()?)),
                        "ante" => self.debug_set(SetField::Ante(v()?)),
                        other => return Err(format!("unknown set field {other:?}")),
                    }
                }
            }
            other => return Err(format!("unknown setup op {other:?}")),
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::run::RunSpec;

    fn run(seed: &str) -> Game {
        Game::new_run(&RunSpec {
            back: jackdaw_data::center_by_key("b_red").unwrap(),
            stake: 1,
            seed: seed.into(),
            challenge: None,
        })
        .unwrap()
    }

    #[test]
    fn keys_parse_by_kind() {
        assert!(matches!(AddKey::parse("j_joker"), Some(AddKey::Joker(_))));
        assert!(matches!(
            AddKey::parse("c_fool"),
            Some(AddKey::Consumable(_))
        ));
        assert!(matches!(AddKey::parse("v_hone"), Some(AddKey::Voucher(_))));
        assert!(matches!(
            AddKey::parse("p_arcana_normal_1"),
            Some(AddKey::Booster(_))
        ));
        assert!(matches!(AddKey::parse("H_A"), Some(AddKey::Playing(_))));
        assert_eq!(
            AddKey::parse("c_base"),
            None,
            "a center, but not a consumable"
        );
        assert_eq!(AddKey::parse("j_nope"), None);
    }

    #[test]
    fn add_joker_in_a_round_fills_a_bigger_hand() {
        let mut g = run("ADD_TEST");
        g.step(&Action::SelectBlind).unwrap();
        let hand = g.gs.areas.hand.len();
        let id = g
            .debug_add(&AddSpec::new(AddKey::parse("j_juggler").unwrap()))
            .unwrap();
        assert_eq!(g.gs.areas.jokers.last().map(|c| c.sort_id), Some(id));
        assert!(g.gs.areas.jokers[0].added_to_deck);
        assert_eq!(
            g.gs.areas.hand.len(),
            hand + 1,
            "Juggler's +1 hand size draws"
        );
    }

    #[test]
    fn add_playing_card_and_stickers() {
        let mut g = run("ADD_TEST2");
        g.step(&Action::SelectBlind).unwrap();
        let mut spec = AddSpec::new(AddKey::parse("H_A").unwrap());
        spec.enhancement = jackdaw_data::center_by_key("m_glass");
        spec.seal = Some(Seal::Red);
        let id = g.debug_add(&spec).unwrap();
        let c = g.gs.areas.hand.last().unwrap();
        assert_eq!(
            (c.sort_id, c.name(), c.seal),
            (id, "Glass Card", Some(Seal::Red))
        );
        let mut spec = AddSpec::new(AddKey::parse("j_joker").unwrap());
        spec.eternal = true;
        spec.perishable = Some(3.0);
        g.debug_add(&spec).unwrap();
        let j = g.gs.areas.jokers.last().unwrap();
        assert!(j.ability.eternal && j.ability.perishable);
        assert_eq!(j.ability.perish_tally.get(), Some(3.0));
    }

    #[test]
    fn restock_needs_the_shop() {
        let mut g = run("ADD_TEST3");
        assert!(g.debug_restock_shop().is_err());
    }
}

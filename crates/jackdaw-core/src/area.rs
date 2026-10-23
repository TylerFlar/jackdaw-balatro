//! Card areas (`CardArea`, cardarea.lua): the ten areas of [`Areas`] addressed by id, with the
//! gameplay part of the Lua methods.
//!
//! Card identity is the sort id: Lua code holds card references across events, and the port
//! holds sort ids and looks the card up again ([`Areas::locate`]).
//!
//! Facing follows Lua: `CardArea:align_cards` turns every card in the deck and the discard pile
//! face down (cardarea.lua:415, 427), and `emplace` turns a face-down card face up when it lands
//! anywhere else unless it stays flipped (cardarea.lua:38). The snapshot writes deck and
//! discard cards as face up, since their facing carries no information there.

use crate::card::{Card, Facing};
use crate::rng::{Rng, shuffle};
use crate::state::Areas;

/// One of the run's card areas.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AreaId {
    Deck,
    Hand,
    Discard,
    Play,
    Jokers,
    Consumables,
    ShopJokers,
    ShopVouchers,
    ShopBooster,
    PackCards,
}

/// `CardArea.config.type`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AreaType {
    Deck,
    Discard,
    Hand,
    Play,
    /// `G.jokers` and `G.consumeables` (game.lua:2235-2245).
    Joker,
    /// The shop's three areas (UI_definitions.lua:638-655).
    Shop,
    /// `G.pack_cards` (UI_definitions.lua:1631).
    Consumeable,
}

impl AreaId {
    pub const ALL: [AreaId; 10] = [
        AreaId::Deck,
        AreaId::Hand,
        AreaId::Discard,
        AreaId::Play,
        AreaId::Jokers,
        AreaId::Consumables,
        AreaId::ShopJokers,
        AreaId::ShopVouchers,
        AreaId::ShopBooster,
        AreaId::PackCards,
    ];

    /// The snapshot name (`Areas::NAMES`).
    pub fn name(self) -> &'static str {
        Areas::NAMES[self as usize]
    }

    pub fn kind(self) -> AreaType {
        match self {
            AreaId::Deck => AreaType::Deck,
            AreaId::Discard => AreaType::Discard,
            AreaId::Hand => AreaType::Hand,
            AreaId::Play => AreaType::Play,
            AreaId::Jokers | AreaId::Consumables => AreaType::Joker,
            AreaId::ShopJokers | AreaId::ShopVouchers | AreaId::ShopBooster => AreaType::Shop,
            AreaId::PackCards => AreaType::Consumeable,
        }
    }

    /// Deck and discard keep their cards face down.
    pub fn face_down(self) -> bool {
        matches!(self.kind(), AreaType::Deck | AreaType::Discard)
    }

    /// Areas that hold playing cards between steps (`G.playing_cards` lives in these).
    pub const PLAYING: [AreaId; 4] = [AreaId::Deck, AreaId::Hand, AreaId::Discard, AreaId::Play];
}

impl Areas {
    pub fn get(&self, id: AreaId) -> &Vec<Card> {
        match id {
            AreaId::Deck => &self.deck,
            AreaId::Hand => &self.hand,
            AreaId::Discard => &self.discard,
            AreaId::Play => &self.play,
            AreaId::Jokers => &self.jokers,
            AreaId::Consumables => &self.consumables,
            AreaId::ShopJokers => &self.shop_jokers,
            AreaId::ShopVouchers => &self.shop_vouchers,
            AreaId::ShopBooster => &self.shop_booster,
            AreaId::PackCards => &self.pack_cards,
        }
    }

    pub fn get_mut(&mut self, id: AreaId) -> &mut Vec<Card> {
        match id {
            AreaId::Deck => &mut self.deck,
            AreaId::Hand => &mut self.hand,
            AreaId::Discard => &mut self.discard,
            AreaId::Play => &mut self.play,
            AreaId::Jokers => &mut self.jokers,
            AreaId::Consumables => &mut self.consumables,
            AreaId::ShopJokers => &mut self.shop_jokers,
            AreaId::ShopVouchers => &mut self.shop_vouchers,
            AreaId::ShopBooster => &mut self.shop_booster,
            AreaId::PackCards => &mut self.pack_cards,
        }
    }

    /// The area and index of the card with this sort id.
    pub fn locate(&self, sort_id: u32) -> Option<(AreaId, usize)> {
        AreaId::ALL.into_iter().find_map(|a| {
            self.get(a)
                .iter()
                .position(|c| c.sort_id == sort_id)
                .map(|i| (a, i))
        })
    }

    pub fn card(&self, sort_id: u32) -> Option<&Card> {
        self.locate(sort_id).map(|(a, i)| &self.get(a)[i])
    }

    pub fn card_mut(&mut self, sort_id: u32) -> Option<&mut Card> {
        let (a, i) = self.locate(sort_id)?;
        Some(&mut self.get_mut(a)[i])
    }

    /// The sort ids of an area's cards, in order.
    pub fn ids(&self, id: AreaId) -> Vec<u32> {
        self.get(id).iter().map(|c| c.sort_id).collect()
    }

    /// `CardArea:emplace(card, location, stay_flipped)` (cardarea.lua:32-64), minus UI.
    pub fn emplace(&mut self, id: AreaId, mut card: Card, front: bool, stay_flipped: bool) {
        if id.face_down() {
            card.facing = Facing::Back;
        } else if card.facing == Facing::Back && !stay_flipped {
            card.facing = Facing::Front;
        }
        let cards = self.get_mut(id);
        if front || id.kind() == AreaType::Deck {
            cards.insert(0, card);
        } else {
            cards.push(card);
        }
        self.align(id);
    }

    /// `CardArea:remove_card(card)` (cardarea.lua:66-92): the given card, or the area's next
    /// card (the last one of a deck or discard pile, else the first).
    pub fn remove_card(&mut self, id: AreaId, sort_id: Option<u32>) -> Option<Card> {
        let cards = self.get_mut(id);
        let index = match sort_id {
            Some(s) => cards.iter().position(|c| c.sort_id == s)?,
            None if matches!(id.kind(), AreaType::Deck | AreaType::Discard) => {
                cards.len().checked_sub(1)?
            }
            None if cards.is_empty() => return None,
            None => 0,
        };
        Some(cards.remove(index))
    }

    /// `CardArea:align_cards()` (cardarea.lua:410-552) where it changes order: positions follow
    /// the index everywhere except that pinned jokers sit leftmost, newest first (line 528).
    pub fn align(&mut self, id: AreaId) {
        if id.kind() == AreaType::Joker && self.get(id).iter().any(|c| c.pinned) {
            let cards = self.get_mut(id);
            let key = |(k, c): &(usize, &Card)| -> (bool, i64, usize) {
                if c.pinned {
                    (false, -i64::from(c.sort_id), *k)
                } else {
                    (true, 0, *k)
                }
            };
            let mut order: Vec<(usize, &Card)> = cards.iter().enumerate().collect();
            order.sort_by_key(key);
            let perm: Vec<usize> = order.into_iter().map(|(k, _)| k).collect();
            let old = std::mem::take(cards);
            let mut slots: Vec<Option<Card>> = old.into_iter().map(Some).collect();
            *cards = perm
                .into_iter()
                .map(|k| slots[k].take().expect("permutation"))
                .collect();
        }
    }

    /// `CardArea:shuffle(seed)` (cardarea.lua:572): `pseudoshuffle(self.cards, pseudoseed(seed))`.
    /// The next frame's `align_cards` puts pinned jokers back in front (cardarea.lua:528; G200).
    pub fn shuffle(&mut self, id: AreaId, rng: &mut Rng, key: &str) {
        let seed = rng.seed(key);
        let cards = self.get_mut(id);
        // pseudoshuffle sorts by sort_id first when the entries carry one (misc_functions.lua).
        cards.sort_by_key(|c| c.sort_id);
        shuffle(cards, seed);
        self.align(id);
    }
}

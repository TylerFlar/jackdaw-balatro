//! Pools and card creation: `get_current_pool`, `create_card`, `get_next_voucher_key`
//! (common_events.lua:1901-2154).

use jackdaw_data::{
    CenterId, CenterSet, JOKER_RARITY_POOLS, P_CARDS, POOL_BOOSTER, POOL_ENHANCED, POOL_PLANET,
    POOL_SPECTRAL, POOL_TAROT, POOL_TAROT_PLANET, POOL_VOUCHER, centers,
};

use crate::area::AreaId;
use crate::card::{Card, CostContext};
use crate::game::Game;

/// `get_current_pool`'s `_type` for center pools.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PoolType {
    Joker,
    Tarot,
    Planet,
    Spectral,
    TarotPlanet,
    Voucher,
    Booster,
    Enhanced,
}

impl PoolType {
    pub fn name(self) -> &'static str {
        match self {
            PoolType::Joker => "Joker",
            PoolType::Tarot => "Tarot",
            PoolType::Planet => "Planet",
            PoolType::Spectral => "Spectral",
            PoolType::TarotPlanet => "Tarot_Planet",
            PoolType::Voucher => "Voucher",
            PoolType::Booster => "Booster",
            PoolType::Enhanced => "Enhanced",
        }
    }

    fn starting_pool(self) -> &'static [CenterId] {
        match self {
            PoolType::Tarot => &POOL_TAROT,
            PoolType::Planet => &POOL_PLANET,
            PoolType::Spectral => &POOL_SPECTRAL,
            PoolType::TarotPlanet => &POOL_TAROT_PLANET,
            PoolType::Voucher => &POOL_VOUCHER,
            PoolType::Booster => &POOL_BOOSTER,
            PoolType::Enhanced => &POOL_ENHANCED,
            PoolType::Joker => unreachable!("joker pools go by rarity"),
        }
    }

    /// The fallback key of an empty pool.
    fn fallback(self) -> CenterId {
        match self {
            PoolType::Tarot | PoolType::TarotPlanet => centers::C_STRENGTH,
            PoolType::Planet => centers::C_PLUTO,
            PoolType::Spectral => centers::C_INCANTATION,
            PoolType::Voucher => centers::V_BLANK,
            _ => centers::J_JOKER,
        }
    }
}

/// A culled pool: `None` marks an `'UNAVAILABLE'` slot.
pub struct Pool {
    pub entries: Vec<Option<CenterId>>,
    pub key: String,
}

impl Game {
    /// `get_current_pool(_type, _rarity, _legendary, _append)` (common_events.lua:1963-2053).
    pub fn get_current_pool(
        &mut self,
        ty: PoolType,
        rarity: Option<f64>,
        legendary: bool,
        append: &str,
    ) -> Pool {
        let ante = crate::lua::tostring_number(self.gs.round_resets.ante);
        let (start, key): (&'static [CenterId], String) = if ty == PoolType::Joker {
            let roll = match rarity {
                Some(r) => r,
                None => self.gs.rng.random(&format!("rarity{ante}{append}")),
            };
            let r = if legendary {
                4
            } else if roll > 0.95 {
                3
            } else if roll > 0.7 {
                2
            } else {
                1
            };
            let key = format!("Joker{r}{}", if legendary { "" } else { append });
            (JOKER_RARITY_POOLS[r - 1], key)
        } else {
            (ty.starting_pool(), format!("{}{append}", ty.name()))
        };
        let showman = self.has_joker("Showman");
        let mut entries = Vec::with_capacity(start.len());
        let mut size = 0;
        for &id in start {
            let v = id.get();
            let mut add = false;
            if ty == PoolType::Enhanced {
                add = true;
            } else if !(self.gs.sets.used_jokers.contains(v.key) && !showman)
                && (v.unlocked != Some(false) || v.rarity == Some(4))
            {
                if v.set == CenterSet::Voucher {
                    if !self.gs.sets.used_vouchers.contains(v.key) {
                        let requires_met = v
                            .requires
                            .iter()
                            .all(|r| self.gs.sets.used_vouchers.contains(r));
                        let in_shop = self.gs.areas.shop_vouchers.iter().any(|c| c.center == id);
                        add = requires_met && !in_shop;
                    }
                } else if v.set == CenterSet::Planet {
                    let softlock = v.config("softlock").and_then(|x| x.as_bool()) == Some(true);
                    let hand = v
                        .config("hand_type")
                        .and_then(|x| x.as_str())
                        .and_then(crate::hand_eval::HandType::from_name);
                    add = !softlock || hand.is_some_and(|h| self.gs.hand(h).played > 0.0);
                } else if let Some(gate) = v.enhancement_gate {
                    add = self.playing_card_ids().into_iter().any(|pc| {
                        self.gs
                            .areas
                            .card(pc)
                            .is_some_and(|c| c.center.key() == gate)
                    });
                } else {
                    add = true;
                }
                if v.name == "Black Hole" || v.name == "The Soul" {
                    add = false;
                }
            }
            if let Some(f) = v.no_pool_flag
                && self.pool_flag(f)
            {
                add = false;
            }
            if let Some(f) = v.yes_pool_flag
                && !self.pool_flag(f)
            {
                add = false;
            }
            if add && !self.gs.sets.banned_keys.contains(v.key) {
                entries.push(Some(id));
                size += 1;
            } else {
                entries.push(None);
            }
        }
        if size == 0 {
            entries = vec![Some(ty.fallback())];
        }
        let key = if legendary {
            key
        } else {
            format!("{key}{ante}")
        };
        Pool { entries, key }
    }

    /// `G.GAME.pool_flags[flag]` is truthy.
    fn pool_flag(&self, flag: &str) -> bool {
        self.gs
            .sets
            .pool_flags
            .get(flag)
            .is_some_and(crate::state::LuaValue::truthy)
    }

    /// `pseudorandom_element(_pool, pseudoseed(key))`, resampling `'UNAVAILABLE'` slots.
    pub fn pick_from_pool(&mut self, pool: &Pool) -> CenterId {
        let n = pool.entries.len();
        let seed = self.gs.rng.seed(&pool.key);
        let mut pick = pool.entries[crate::rng::element_index(n, seed)];
        let mut it = 1;
        while pick.is_none() {
            it += 1;
            let seed = self.gs.rng.seed(&format!("{}_resample{it}", pool.key));
            pick = pool.entries[crate::rng::element_index(n, seed)];
        }
        pick.expect("resampled")
    }

    /// `get_next_voucher_key(_from_tag)` (common_events.lua:1901-1912).
    pub fn get_next_voucher_key(&mut self, from_tag: bool) -> Option<CenterId> {
        let mut pool = self.get_current_pool(PoolType::Voucher, None, false, "");
        if from_tag {
            pool.key = "Voucher_fromtag".to_owned();
        }
        Some(self.pick_from_pool(&pool))
    }

    /// The values `Card:set_cost` reads for a card in `area`.
    pub fn cost_context(&self, area: AreaId) -> CostContext {
        CostContext {
            inflation: self.gs.inflation,
            discount_percent: self.gs.discount_percent,
            booster_ante_scaling: self.gs.modifier("booster_ante_scaling").is_some(),
            ante: self.gs.round_resets.ante,
            astronomer: self.has_joker("Astronomer"),
            in_shop: matches!(area, AreaId::ShopJokers | AreaId::ShopBooster),
        }
    }

    /// `Card(..., front, center)`: a new card with the next sort id; `set_ability` marks every
    /// center of its name in `G.GAME.used_jokers` (card.lua:349-354).
    pub fn new_card(
        &mut self,
        center: CenterId,
        front: Option<jackdaw_data::CardId>,
        area: AreaId,
    ) -> Card {
        let sort_id = self.gs.next_sort_id();
        let cost = self.cost_context(area);
        let mut card = Card::new(sort_id, center, front, self.gs.hands_played, &cost);
        let name = center.get().name;
        for c in jackdaw_data::CENTERS.iter().filter(|c| c.name == name) {
            self.gs.sets.used_jokers.insert(c.key);
        }
        if name == "To Do List" {
            // `pairs(G.GAME.hands)` order (card.lua:312-315).
            let visible: Vec<_> = crate::hand_eval::PAIRS_ORDER
                .into_iter()
                .filter(|&h| self.gs.hand(h).visible)
                .collect();
            card.roll_to_do_hand(&visible, &mut self.gs.rng);
        }
        card
    }

    /// `create_card(_type, area, legendary, _rarity, skip_materialize, soulable, forced_key,
    /// key_append)` (common_events.lua:2082-2154) for consumables, vouchers and boosters.
    pub fn create_card(
        &mut self,
        ty: PoolType,
        area: AreaId,
        soulable: bool,
        forced: Option<CenterId>,
        append: &str,
    ) -> Card {
        let mut forced = forced;
        if forced.is_none() && soulable && !self.gs.sets.banned_keys.contains("c_soul") {
            let ante = crate::lua::tostring_number(self.gs.round_resets.ante);
            let showman = self.has_joker("Showman");
            if matches!(
                ty,
                PoolType::Tarot | PoolType::Spectral | PoolType::TarotPlanet
            ) && !(self.gs.sets.used_jokers.contains("c_soul") && !showman)
                && self.gs.rng.random(&format!("soul_{}{ante}", ty.name())) > 0.997
            {
                forced = Some(centers::C_SOUL);
            }
            if matches!(ty, PoolType::Planet | PoolType::Spectral)
                && !(self.gs.sets.used_jokers.contains("c_black_hole") && !showman)
                && self.gs.rng.random(&format!("soul_{}{ante}", ty.name())) > 0.997
            {
                forced = Some(centers::C_BLACK_HOLE);
            }
        }
        let center = match forced {
            Some(c) if !self.gs.sets.banned_keys.contains(c.key()) => c,
            _ => {
                let pool = self.get_current_pool(ty, None, false, append);
                self.pick_from_pool(&pool)
            }
        };
        let front = if ty == PoolType::Enhanced {
            let ante = crate::lua::tostring_number(self.gs.round_resets.ante);
            let seed = self.gs.rng.seed(&format!("front{append}{ante}"));
            Some(jackdaw_data::CardId(
                crate::rng::element_index(P_CARDS.len(), seed) as u8,
            ))
        } else {
            None
        };
        self.new_card(center, front, area)
    }
}

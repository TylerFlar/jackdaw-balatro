//! Pools and card creation: `get_current_pool`, `create_card`, `poll_edition`,
//! `create_playing_card`, `copy_card` and `get_next_voucher_key` (common_events.lua:1901-2181).

use jackdaw_data::{
    CardId, CenterId, CenterSet, JOKER_RARITY_POOLS, P_CARDS, POOL_BOOSTER, POOL_ENHANCED,
    POOL_PLANET, POOL_SPECTRAL, POOL_TAROT, POOL_TAROT_PLANET, POOL_VOUCHER, centers,
};

use crate::area::AreaId;
use crate::card::{Card, CostContext, EditionKind};
use crate::game::Game;
use crate::rng::Key;

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
    /// A plain playing card (`'Base'`): always `c_base`.
    Base,
}

impl PoolType {
    /// The pool type a center's set stands for (`Default` keeps the requested type).
    pub fn for_set(set: CenterSet) -> Option<PoolType> {
        Some(match set {
            CenterSet::Joker => PoolType::Joker,
            CenterSet::Tarot => PoolType::Tarot,
            CenterSet::Planet => PoolType::Planet,
            CenterSet::Spectral => PoolType::Spectral,
            CenterSet::Voucher => PoolType::Voucher,
            CenterSet::Booster => PoolType::Booster,
            CenterSet::Enhanced => PoolType::Enhanced,
            _ => return None,
        })
    }

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
            PoolType::Base => "Base",
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
            PoolType::Base => unreachable!("base cards are always c_base"),
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

/// The arguments of `create_card` (`skip_materialize` is UI only).
#[derive(Clone, Copy, Debug)]
pub struct CardSpec<'a> {
    pub ty: PoolType,
    pub area: AreaId,
    pub legendary: bool,
    /// `_rarity`: the rarity roll to use instead of drawing one.
    pub rarity: Option<f64>,
    pub soulable: bool,
    pub forced: Option<CenterId>,
    pub append: &'a str,
}

impl<'a> CardSpec<'a> {
    /// `create_card(ty, area, nil, nil, nil, nil, nil, append)`.
    pub fn new(ty: PoolType, area: AreaId, append: &'a str) -> CardSpec<'a> {
        CardSpec {
            ty,
            area,
            legendary: false,
            rarity: None,
            soulable: false,
            forced: None,
            append,
        }
    }

    pub fn forced(self, forced: Option<CenterId>) -> CardSpec<'a> {
        CardSpec { forced, ..self }
    }

    pub fn soulable(self) -> CardSpec<'a> {
        CardSpec {
            soulable: true,
            ..self
        }
    }

    pub fn legendary(self) -> CardSpec<'a> {
        CardSpec {
            legendary: true,
            ..self
        }
    }

    pub fn rarity(self, rarity: f64) -> CardSpec<'a> {
        CardSpec {
            rarity: Some(rarity),
            ..self
        }
    }
}

impl Game {
    /// `get_current_pool(_type, _rarity, _legendary, _append)` (common_events.lua:1963-2053) and
    /// the draw from it, `pseudorandom_element(_pool, pseudoseed(_pool_key))` with its
    /// `'UNAVAILABLE'` resamples (2097-2104). Building the pool draws nothing, so it isn't built:
    /// a drawn slot is checked on its own, and the other slots only after a miss, to tell an empty
    /// pool (whose one-entry fallback Lua draws with that same first seed). `key` replaces the
    /// pool key (`get_next_voucher_key`'s `'Voucher_fromtag'`).
    pub fn pick_from_current_pool(
        &mut self,
        ty: PoolType,
        rarity: Option<f64>,
        legendary: bool,
        append: &str,
        key: Option<&str>,
    ) -> CenterId {
        let ante = self.gs.round_resets.ante;
        let (start, pool_key) = if ty == PoolType::Joker {
            let roll = match rarity {
                Some(r) => r,
                None => self.gs.rng.random(&Key::new("rarity").n(ante).s(append)),
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
            let key = if legendary {
                Key::new("Joker4")
            } else {
                Key::new("Joker").i(r as u64).s(append).n(ante)
            };
            (JOKER_RARITY_POOLS[r - 1], key)
        } else {
            (ty.starting_pool(), Key::new(ty.name()).s(append).n(ante))
        };
        let pool_key = key.map_or(pool_key, Key::new);
        let showman = self.has_joker("Showman");
        let n = start.len();
        let first = start[crate::rng::element_index(n, self.gs.rng.seed(&pool_key))];
        if self.in_pool(ty, first, showman) {
            return first;
        }
        if !start.iter().any(|&c| self.in_pool(ty, c, showman)) {
            return ty.fallback();
        }
        let mut it = 1;
        loop {
            it += 1;
            let seed = self.gs.rng.seed(&Key::new(&pool_key).s("_resample").i(it));
            let c = start[crate::rng::element_index(n, seed)];
            if self.in_pool(ty, c, showman) {
                return c;
            }
        }
    }

    /// Whether `id`, a center of the starting pool, has its key in a `ty` pool rather than
    /// `'UNAVAILABLE'` (common_events.lua:1987-2036). Lua also needs `v.unlocked ~= false or
    /// v.rarity == 4`, where `unlocked` comes from the profile (game.lua:741-751). The engine
    /// assumes a fully unlocked profile, so every center passes; the prototypes' flags are a
    /// fresh profile's.
    fn in_pool(&self, ty: PoolType, id: CenterId, showman: bool) -> bool {
        let v = id.get();
        let sets = &self.gs.sets;
        let mut add = false;
        if ty == PoolType::Enhanced {
            add = true;
        } else if !(sets.used_jokers.contains(id) && !showman) {
            if v.set == CenterSet::Voucher {
                if !sets.used_vouchers.contains(id) {
                    let requires_met = v.requires.iter().all(|r| {
                        jackdaw_data::center_by_key(r)
                            .is_some_and(|c| sets.used_vouchers.contains(c))
                    });
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
                let gate = jackdaw_data::center_by_key(gate);
                add = self.playing_cards().any(|c| Some(c.center) == gate);
            } else {
                add = true;
            }
            if id == centers::C_BLACK_HOLE || id == centers::C_SOUL {
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
        add && !sets.banned_keys.contains(v.key)
    }

    /// `G.GAME.pool_flags[flag]` is truthy.
    fn pool_flag(&self, flag: &str) -> bool {
        self.gs
            .sets
            .pool_flags
            .get(flag)
            .is_some_and(crate::state::LuaValue::truthy)
    }

    /// `get_next_voucher_key(_from_tag)` (common_events.lua:1901-1912).
    pub fn get_next_voucher_key(&mut self, from_tag: bool) -> Option<CenterId> {
        let key = from_tag.then_some("Voucher_fromtag");
        Some(self.pick_from_current_pool(PoolType::Voucher, None, false, "", key))
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
    pub fn new_card(&mut self, center: CenterId, front: Option<CardId>, area: AreaId) -> Card {
        let cost = self.cost_context(area);
        self.new_card_with_cost(center, front, &cost)
    }

    /// [`Game::new_card`] with the cost values already read.
    pub fn new_card_with_cost(
        &mut self,
        center: CenterId,
        front: Option<CardId>,
        cost: &CostContext,
    ) -> Card {
        let sort_id = self.gs.next_sort_id();
        let mut card = Card::new(sort_id, center, front, self.gs.hands_played, cost);
        self.set_ability_effects(&mut card);
        card
    }

    /// The game-level part of `Card:set_ability` for the card's new center: every center of its
    /// name joins `G.GAME.used_jokers` (card.lua:349-354), and To Do List draws its hand
    /// (card.lua:311-322). The boss re-debuff of a non-initial call is the caller's.
    fn set_ability_effects(&mut self, card: &mut Card) {
        self.mark_used(card.center);
        if card.name() == "To Do List" {
            // `pairs(G.GAME.hands)` order (card.lua:312-315).
            let visible: Vec<_> = crate::hand_eval::PAIRS_ORDER
                .into_iter()
                .filter(|&h| self.gs.hand(h).visible)
                .collect();
            card.roll_to_do_hand(&visible, &mut self.gs.rng);
        }
    }

    /// `set_ability`'s `G.GAME.used_jokers` marks: every center of this one's name.
    pub(crate) fn mark_used(&mut self, center: CenterId) {
        for &c in jackdaw_data::same_name_centers(center) {
            self.gs.sets.used_jokers.insert(c);
        }
    }

    /// `card:set_ability(center)` on a card already in play (an enhancement change): the new
    /// ability, its game-level effects, then the boss re-debuff of a non-initial call
    /// (card.lua:223-366).
    pub fn set_card_ability(&mut self, id: u32, center: CenterId) {
        let hands_played = self.gs.hands_played;
        let mut card = *self.card(id);
        card.set_ability(center, hands_played);
        self.set_ability_effects(&mut card);
        *self.card_mut(id) = card;
        self.debuff_card(id, false);
    }

    /// `pseudorandom_element(G.P_CARDS, pseudoseed(key))`: a random front.
    pub fn random_front(&mut self, key: &str) -> CardId {
        let seed = self.gs.rng.seed(key);
        CardId(crate::rng::element_index(P_CARDS.len(), seed) as u8)
    }

    /// `create_playing_card(card_init, area)` (common_events.lua:1927-1942): a new playing card
    /// emplaced in `area`. `front` defaults to a draw from `'front'`, `center` to `c_base`.
    /// Returns its sort id; the card is part of `G.playing_cards` from now on.
    pub fn create_playing_card(
        &mut self,
        front: Option<CardId>,
        center: Option<CenterId>,
        area: AreaId,
    ) -> u32 {
        let front = match front {
            Some(f) => f,
            None => self.random_front("front"),
        };
        let card = self.new_card(center.unwrap_or(centers::C_BASE), Some(front), area);
        let id = card.sort_id;
        self.gs.areas.emplace(area, card, false, false);
        id
    }

    /// `copy_card(other, nil, nil, playing_card, strip_edition)` (common_events.lua:2156-2181):
    /// a new card, with the next sort id, carrying `other`'s center, front, every ability entry,
    /// edition (unless stripped), seal, debuff and pin. It is in no area yet; `area` is where it
    /// goes, for its price.
    ///
    /// Lua builds it as a blank `c_base` card, then `set_ability(other's center)` (which marks
    /// `used_jokers` and draws a To Do List hand that the ability copy then overwrites) and a
    /// non-initial `set_base(other's front)`: the copy's `times_played` restarts at 0, its
    /// `original_value` stays unset, and its suit tiebreak is its current suit's.
    pub fn copy_card(&mut self, other: u32, strip_edition: bool, area: AreaId) -> Card {
        let src = *self.card(other);
        let cost = self.cost_context(area);
        let mut card = self.new_card(centers::C_BASE, None, area);
        card.set_ability(src.center, self.gs.hands_played);
        self.set_ability_effects(&mut card);
        card.base = src
            .base
            .map(|b| crate::card::set_base(None, b.front, false));
        card.ability = src.ability;
        if !strip_edition {
            card.set_edition(src.edition.map(|e| e.kind), &cost);
        }
        card.set_seal(src.seal, &cost);
        card.debuff = src.debuff;
        card.pinned = src.pinned;
        card
    }

    /// `create_card(_type, area, legendary, _rarity, skip_materialize, soulable, forced_key,
    /// key_append)` (common_events.lua:2082-2154). The card is not yet in an area.
    pub fn create_card(&mut self, spec: CardSpec) -> Card {
        let CardSpec {
            mut ty,
            area,
            legendary,
            rarity,
            soulable,
            mut forced,
            append,
        } = spec;
        let ante = self.gs.round_resets.ante;
        if forced.is_none() && soulable && !self.gs.sets.banned_keys.contains("c_soul") {
            let showman = self.has_joker("Showman");
            if matches!(
                ty,
                PoolType::Tarot | PoolType::Spectral | PoolType::TarotPlanet
            ) && !(self
                .gs
                .sets
                .used_jokers
                .contains(jackdaw_data::centers::C_SOUL)
                && !showman)
                && self
                    .gs
                    .rng
                    .random(&crate::rng::Key::new("soul_").s(ty.name()).n(ante))
                    > 0.997
            {
                forced = Some(centers::C_SOUL);
            }
            if matches!(ty, PoolType::Planet | PoolType::Spectral)
                && !(self
                    .gs
                    .sets
                    .used_jokers
                    .contains(jackdaw_data::centers::C_BLACK_HOLE)
                    && !showman)
                && self
                    .gs
                    .rng
                    .random(&crate::rng::Key::new("soul_").s(ty.name()).n(ante))
                    > 0.997
            {
                forced = Some(centers::C_BLACK_HOLE);
            }
        }
        if ty == PoolType::Base {
            forced = Some(centers::C_BASE);
        }
        let center = match forced {
            Some(c) if !self.gs.sets.banned_keys.contains(c.key()) => {
                // `_type = (center.set ~= 'Default' and center.set or _type)`.
                if let Some(t) = PoolType::for_set(c.get().set) {
                    ty = t;
                }
                c
            }
            _ => self.pick_from_current_pool(ty, rarity, legendary, append, None),
        };
        let front = if matches!(ty, PoolType::Base | PoolType::Enhanced) {
            Some(self.random_front(&crate::rng::Key::new("front").s(append).n(ante)))
        } else {
            None
        };
        let mut card = self.new_card(center, front, area);
        if ty == PoolType::Joker {
            let cost = self.cost_context(area);
            if self.gs.modifier("all_eternal").is_some() {
                card.set_eternal(true);
            }
            if matches!(area, AreaId::ShopJokers | AreaId::PackCards) {
                let pack = area == AreaId::PackCards;
                let key = if pack { "packetper" } else { "etperpoll" };
                let poll = self.gs.rng.random(&crate::rng::Key::new(key).n(ante));
                if self.gs.modifier("enable_eternals_in_shop").is_some() && poll > 0.7 {
                    card.set_eternal(true);
                } else if self.gs.modifier("enable_perishables_in_shop").is_some()
                    && poll > 0.4
                    && poll <= 0.7
                {
                    card.set_perishable(self.gs.perishable_rounds);
                }
                let key = if pack { "packssjr" } else { "ssjr" };
                if self.gs.modifier("enable_rentals_in_shop").is_some()
                    && self.gs.rng.random(&crate::rng::Key::new(key).n(ante)) > 0.7
                {
                    card.set_rental(true, &cost);
                }
            }
            let edition = self.poll_edition(
                &crate::rng::Key::new("edi").s(append).n(ante),
                1.0,
                false,
                false,
            );
            card.set_edition(edition, &cost);
        }
        card
    }

    /// `poll_edition(_key, _mod, _no_neg, _guaranteed)` (common_events.lua:2055-2080).
    pub fn poll_edition(
        &mut self,
        key: &str,
        modifier: f64,
        no_neg: bool,
        guaranteed: bool,
    ) -> Option<EditionKind> {
        let poll = self.gs.rng.random(key);
        let (rate, modifier) = if guaranteed {
            (1.0, 25.0)
        } else {
            (self.gs.edition_rate, modifier)
        };
        if poll > 1.0 - 0.003 * modifier && !no_neg {
            Some(EditionKind::Negative)
        } else if poll > 1.0 - 0.006 * rate * modifier {
            Some(EditionKind::Polychrome)
        } else if poll > 1.0 - 0.02 * rate * modifier {
            Some(EditionKind::Holo)
        } else if poll > 1.0 - 0.04 * rate * modifier {
            Some(EditionKind::Foil)
        } else {
            None
        }
    }
}

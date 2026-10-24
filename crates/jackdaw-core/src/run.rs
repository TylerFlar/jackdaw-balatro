//! A new run: `Game:start_run` (game.lua:2018-2449) with `Game:init_game_object`
//! (game.lua:1862-2017), `get_starting_params` (misc_functions.lua:1868-1881), the stakes,
//! `Back:apply_to_run` (back.lua:166-290) and a challenge's rules, cards and bans.
//!
//! `start_run` changes the state synchronously and queues the back's and the challenge's cards
//! and vouchers as events; the blind-select screen is then built by its state handler. The run
//! is returned settled at the blind-select screen.
//!
//! The sim's profile has every center unlocked and discovered, and `G.sort_id` starts at 0 for
//! each run.

use std::collections::BTreeMap;

use jackdaw_data::{BLINDS, CenterId, CenterSet, HANDS, P_CARDS, Value, centers};

use crate::area::AreaId;
use crate::card::{Card, CostContext, EditionKind, Rank, Seal, Suit};
use crate::event::Event;
use crate::game::{Game, LuaState, Runtime};
use crate::hand_eval::HandType;
use crate::pools::{CardSpec, PoolType};
use crate::rng::Rng;
use crate::state::{
    Areas, BlindSlot, ConsumeableUsage, CurrentRound, GameState, HandSort, HandState, LuaValue,
    Phase, RoundBonus, RoundResets, Sets, SimState, StartingParams,
};

/// The `'static` id of a `G.CHALLENGES` entry.
pub fn challenge_id(id: &str) -> Option<&'static str> {
    jackdaw_data::challenge_by_id(id).map(|c| c.id)
}

/// A run's setup.
#[derive(Clone, Debug, PartialEq)]
pub struct RunSpec {
    /// The back's center key (`b_red`, ...); a challenge always uses `b_challenge`.
    pub back: CenterId,
    /// 1 (White) to 8 (Gold).
    pub stake: u8,
    pub seed: String,
    /// A `G.CHALLENGES` id (`c_omelette_1`, ...).
    pub challenge: Option<&'static str>,
}

/// `Game:init_game_object()` with `get_starting_params()`.
fn init_game_object(seed: &str) -> GameState {
    let hands = std::array::from_fn(|i| {
        let h = &HANDS[i];
        HandState {
            level: 1.0,
            chips: h.s_chips,
            mult: h.s_mult,
            played: 0.0,
            played_this_round: 0.0,
            visible: h.visible,
        }
    });
    let mut bosses_used = crate::state::BossCounts::empty();
    for (i, b) in BLINDS.iter().enumerate() {
        if b.boss.is_some() {
            bosses_used.set(jackdaw_data::BlindId(i as u8), 0.0);
        }
    }
    GameState {
        phase: Phase::BlindSelect,
        back: centers::B_RED,
        stake: 1.0,
        challenge: None,
        win_ante: 8.0,
        won: false,
        dollars: 0.0,
        chips: 0.0,
        round: 0.0,
        skips: 0.0,
        hands_played: 0.0,
        unused_discards: 0.0,
        interest_cap: 25.0,
        interest_amount: 1.0,
        discount_percent: 0.0,
        inflation: 0.0,
        rental_rate: 3.0,
        bankrupt_at: 0.0,
        base_reroll_cost: 5.0,
        edition_rate: 1.0,
        joker_rate: 20.0,
        tarot_rate: 4.0,
        planet_rate: 4.0,
        spectral_rate: 0.0,
        playing_card_rate: 0.0,
        probabilities_normal: 1.0,
        hand_size: 8.0,
        hand_sort: HandSort::Desc,
        joker_slots: 5.0,
        consumable_slots: 2.0,
        starting_deck_size: 52.0,
        last_tarot_planet: None,
        last_hand_played: None,
        pack_size: 2.0,
        ecto_minus: 1.0,
        perishable_rounds: 5.0,
        blind_on_deck: None,
        shop_joker_max: 2.0,
        first_shop_buffoon: false,
        orbital_choices: BTreeMap::new(),
        starting_params: StartingParams {
            dollars: 4.0,
            hand_size: 8.0,
            discards: 3.0,
            hands: 4.0,
            reroll_cost: 5.0,
            joker_slots: 5.0,
            consumable_slots: 2.0,
            ante_scaling: 1.0,
            no_faces: false,
            erratic_suits_and_ranks: false,
        },
        modifiers: BTreeMap::new(),
        round_resets: RoundResets {
            hands: 1.0,
            discards: 1.0,
            reroll_cost: 1.0,
            temp_reroll_cost: None,
            temp_handsize: None,
            ante: 1.0,
            blind_ante: 1.0,
            blind_states: ["Select", "Upcoming", "Upcoming"],
            blind_choices: [
                jackdaw_data::blind_by_key("bl_small"),
                jackdaw_data::blind_by_key("bl_big"),
                None,
            ],
            blind_tags: None,
            boss_rerolled: false,
        },
        current_round: CurrentRound {
            hands_left: 0.0,
            hands_played: 0.0,
            discards_left: 0.0,
            discards_used: 0.0,
            reroll_cost: 5.0,
            reroll_cost_increase: 0.0,
            free_rerolls: 0.0,
            jokers_purchased: 0.0,
            voucher: None,
            most_played_poker_hand: HandType::HighCard,
            idol_card: (Suit::Spades, Rank::Ace),
            mail_card: Rank::Ace,
            ancient_card: Some(Suit::Spades),
            castle_card: Suit::Spades,
        },
        round_bonus: RoundBonus::default(),
        blind: None,
        hands,
        tags: Vec::new(),
        sets: Sets {
            bosses_used,
            ..Sets::default()
        },
        consumeable_usage: ConsumeableUsage::default(),
        areas: Areas::default(),
        pack: None,
        // The streams `start_run` draws from before the first decision: the boss, the voucher,
        // the blind tags, the deck's shuffle, the round's cards and the Orbital Tag hands, and
        // the voucher's and the tags' first resamples (half the vouchers and a third of the tags
        // are out of the first ante's pools).
        rng: Rng::new_primed(
            seed,
            &[
                "boss",
                "Voucher1",
                "Tag1",
                "shuffle",
                "idol1",
                "mail1",
                "anc1",
                "cas1",
                "orbital",
                "Voucher1_resample2",
                "Voucher1_resample3",
                "Tag1_resample2",
                "Tag1_resample3",
            ],
        ),
        sim: SimState {
            next_sort_id: 1,
            ..SimState::default()
        },
    }
}

/// A table entry of a `jackdaw_data::Value` table.
fn field<'a>(v: &'a Value, key: &str) -> Option<&'a Value> {
    match v {
        Value::Table(entries) => entries.iter().find(|(k, _)| *k == key).map(|(_, v)| v),
        _ => None,
    }
}

fn list(v: Option<&Value>) -> &'static [Value] {
    match v {
        Some(Value::List(items)) => items,
        _ => &[],
    }
}

fn string(v: Option<&Value>) -> Option<&'static str> {
    match v {
        Some(Value::Str(s)) => Some(s),
        _ => None,
    }
}

/// A `jackdaw_data::Value` as a `G.GAME.modifiers` value.
fn lua_value(v: &Value) -> LuaValue {
    match v {
        Value::Bool(b) => LuaValue::Bool(*b),
        Value::Num(x) => LuaValue::Num(*x),
        Value::Str(s) => LuaValue::Str((*s).to_owned()),
        Value::List(items) => LuaValue::Table(
            items
                .iter()
                .enumerate()
                .map(|(i, x)| ((i + 1).to_string(), lua_value(x)))
                .collect(),
        ),
        Value::Table(entries) => LuaValue::Table(
            entries
                .iter()
                .map(|(k, x)| ((*k).to_owned(), lua_value(x)))
                .collect(),
        ),
    }
}

/// A playing card prototype of the starting deck (`card_protos` entries).
#[derive(Clone, Debug)]
struct Proto {
    /// The `P_CARDS` front of `s` and `r`.
    front: jackdaw_data::CardId,
    s: &'static str,
    r: &'static str,
    e: Option<&'static str>,
    d: Option<&'static str>,
    g: Option<&'static str>,
}

impl Proto {
    /// The bytes of the sort key `s..r..e..d..g` (game.lua:2357-2359), which Lua compares as
    /// strings.
    fn key(&self) -> impl Iterator<Item = u8> + '_ {
        [
            self.s,
            self.r,
            self.e.unwrap_or(""),
            self.d.unwrap_or(""),
            self.g.unwrap_or(""),
        ]
        .into_iter()
        .flat_map(str::bytes)
    }

    /// The order of the sort keys (Lua string order). Suits and ranks are one letter each, so
    /// they compare on their own before the rest.
    fn cmp(a: &Proto, b: &Proto) -> std::cmp::Ordering {
        let short = |p: &Proto| p.s.len() == 1 && p.r.len() == 1;
        if short(a) && short(b) {
            let head = (a.s.as_bytes()[0], a.r.as_bytes()[0]);
            let first = head.cmp(&(b.s.as_bytes()[0], b.r.as_bytes()[0]));
            if first.is_ne() {
                return first;
            }
        }
        a.key().cmp(b.key())
    }
}

impl Game {
    /// `G.FUNCS.start_run` for a new run, settled at the blind-select screen.
    pub fn new_run(spec: &RunSpec) -> Result<Game, String> {
        let challenge = match spec.challenge {
            Some(id) => Some(
                jackdaw_data::challenge_by_id(id)
                    .ok_or_else(|| format!("unknown challenge {id}"))?,
            ),
            None => None,
        };
        let back = match challenge {
            Some(_) => centers::B_CHALLENGE,
            None => spec.back,
        };
        if back.get().set != CenterSet::Back {
            return Err(format!("{} is not a back", back.key()));
        }
        if !(1..=8).contains(&spec.stake) {
            return Err(format!("stake {} is not 1 to 8", spec.stake));
        }
        let mut gs = init_game_object(&spec.seed);
        gs.back = back;
        gs.stake = f64::from(spec.stake);
        gs.challenge = spec.challenge.map(str::to_owned);
        let mut g = Game {
            gs,
            rt: Runtime {
                state: LuaState::BlindSelect,
                state_complete: false,
                ..Runtime::default()
            },
        };
        g.apply_stake();
        g.back_apply_to_run();
        if let Some(ch) = challenge {
            g.apply_challenge(&ch.data);
        }
        let sp = g.gs.starting_params.clone();
        g.gs.round_resets.hands = sp.hands;
        g.gs.round_resets.discards = sp.discards;
        g.gs.dollars = sp.dollars;
        g.gs.base_reroll_cost = sp.reroll_cost;
        g.gs.round_resets.reroll_cost = sp.reroll_cost;
        g.gs.current_round.reroll_cost = sp.reroll_cost;
        // The first ante's boss, voucher and blind tags (game.lua:2152-2159).
        let boss = g.get_new_boss();
        g.gs.round_resets.blind_choices[BlindSlot::Boss.index()] = Some(boss);
        g.gs.current_round.voucher = g.get_next_voucher_key(false);
        let small = g.get_next_tag_key("");
        let big = g.get_next_tag_key("");
        g.gs.round_resets.blind_tags = Some([Some(small), Some(big)]);
        // The card areas take their limits from the starting params (game.lua:2196-2224).
        g.gs.consumable_slots = sp.consumable_slots;
        g.gs.joker_slots = sp.joker_slots;
        g.gs.hand_size = sp.hand_size;
        // The deck comes shuffled (`G.deck:shuffle()`, game.lua:2375).
        g.build_deck(challenge.map(|c| &c.data));
        let rr = &g.gs.round_resets;
        g.gs.current_round.discards_left = rr.discards;
        g.gs.current_round.hands_left = rr.hands;
        g.reset_idol_card();
        g.reset_mail_rank();
        g.gs.current_round.ancient_card = None;
        g.reset_ancient_card();
        g.reset_castle_card();
        // `G.GAME.blind = Blind(...)`, `set_blind(nil, nil, true)`, `reset_blinds()`.
        g.set_blind(None, false);
        g.reset_blinds();
        g.settle();
        g.gs.rng.unprime();
        if let Some(what) = g.rt.unported.take() {
            return Err(format!("not ported yet: {what}"));
        }
        Ok(g)
    }

    /// The stake's modifiers (game.lua:2043-2051).
    fn apply_stake(&mut self) {
        let stake = self.gs.stake;
        let m = &mut self.gs.modifiers;
        if stake >= 2.0 {
            let mut t = BTreeMap::new();
            t.insert("Small".to_owned(), LuaValue::Bool(true));
            m.insert("no_blind_reward".to_owned(), LuaValue::Table(t));
        }
        if stake >= 3.0 {
            m.insert("scaling".to_owned(), LuaValue::Num(2.0));
        }
        if stake >= 4.0 {
            m.insert("enable_eternals_in_shop".to_owned(), LuaValue::Bool(true));
        }
        if stake >= 5.0 {
            self.gs.starting_params.discards -= 1.0;
        }
        if stake >= 6.0 {
            self.gs
                .modifiers
                .insert("scaling".to_owned(), LuaValue::Num(3.0));
        }
        if stake >= 7.0 {
            self.gs.modifiers.insert(
                "enable_perishables_in_shop".to_owned(),
                LuaValue::Bool(true),
            );
        }
        if stake >= 8.0 {
            self.gs
                .modifiers
                .insert("enable_rentals_in_shop".to_owned(), LuaValue::Bool(true));
        }
    }

    /// `Back:apply_to_run()` (back.lua:166-290).
    fn back_apply_to_run(&mut self) {
        let back = self.gs.back.get();
        let cfg = |k: &str| back.config(k).filter(|v| v.as_bool() != Some(false));
        let num = |k: &str| back.config_num(k);
        if let Some(v) = cfg("voucher").and_then(|v| v.as_str()) {
            self.start_voucher(v);
        }
        let sp = &mut self.gs.starting_params;
        if let Some(x) = num("hands") {
            sp.hands += x;
        }
        if let Some(list) = cfg("consumables") {
            let keys: Vec<CenterId> = match list {
                Value::List(items) => items
                    .iter()
                    .filter_map(|v| v.as_str())
                    .map(|k| jackdaw_data::center_by_key(k).expect("a back's consumable"))
                    .collect(),
                _ => Vec::new(),
            };
            for k in keys {
                // One event creates them all in order (back.lua:181-191); each is its own
                // `Call` here, queued together.
                self.queue(Event::Call(back_consumable, u32::from(k.0), 0.0));
            }
        }
        let sp = &mut self.gs.starting_params;
        if let Some(x) = num("dollars") {
            sp.dollars += x;
        }
        if cfg("remove_faces").is_some() {
            sp.no_faces = true;
        }
        if let Some(x) = num("spectral_rate") {
            self.gs.spectral_rate = x;
        }
        let sp = &mut self.gs.starting_params;
        if let Some(x) = num("discards") {
            sp.discards += x;
        }
        if let Some(x) = num("reroll_discount") {
            sp.reroll_cost -= x;
        }
        if let Some(Value::List(items)) = cfg("vouchers") {
            for v in items.iter().filter_map(|v| v.as_str()) {
                self.start_voucher(v);
            }
        }
        if back.name == "Checkered Deck" {
            self.queue(Event::Call(checkered, 0, 0.0));
        }
        let sp = &mut self.gs.starting_params;
        if cfg("randomize_rank_suit").is_some() {
            sp.erratic_suits_and_ranks = true;
        }
        if let Some(x) = num("joker_slot") {
            sp.joker_slots += x;
        }
        if let Some(x) = num("hand_size") {
            sp.hand_size += x;
        }
        if let Some(x) = num("ante_scaling") {
            sp.ante_scaling = x;
        }
        if let Some(x) = num("consumable_slot") {
            sp.consumable_slots += x;
        }
        let m = &mut self.gs.modifiers;
        if cfg("no_interest").is_some() {
            m.insert("no_interest".to_owned(), LuaValue::Bool(true));
        }
        if let Some(x) = num("extra_hand_bonus") {
            m.insert("money_per_hand".to_owned(), LuaValue::Num(x));
        }
        if let Some(x) = num("extra_discard_bonus") {
            m.insert("money_per_discard".to_owned(), LuaValue::Num(x));
        }
    }

    /// A starting voucher: used at once, its effect through `Card.apply_to_run(nil, center)`.
    fn start_voucher(&mut self, key: &str) {
        let center = jackdaw_data::center_by_key(key).expect("a starting voucher");
        self.gs.sets.used_vouchers.insert(center);
        let extra = center.get().config_num("extra");
        self.apply_voucher(center.get().name, extra);
    }

    /// The challenge's cards, vouchers, rules and bans (game.lua:2053-2131).
    fn apply_challenge(&mut self, ch: &'static Value) {
        for j in list(field(ch, "jokers")) {
            let key = string(field(j, "id")).expect("a challenge joker's id");
            let center = jackdaw_data::center_by_key(key).expect("a challenge joker");
            let edition = string(field(j, "edition"))
                .map(|e| EditionKind::from_name(e).unwrap_or_else(|| panic!("edition {e}")));
            let mut flags = 0u32;
            if field(j, "eternal").is_some_and(|v| matches!(v, Value::Bool(true))) {
                flags |= 1;
            }
            if field(j, "pinned").is_some_and(|v| matches!(v, Value::Bool(true))) {
                flags |= 2;
            }
            let code = edition.map_or(0.0, |e| f64::from(edition_code(e) + 1));
            self.queue(Event::Call(
                challenge_joker,
                u32::from(center.0) | (flags << 16),
                code,
            ));
        }
        for c in list(field(ch, "consumeables")) {
            let key = string(field(c, "id")).expect("a challenge consumable's id");
            let center = jackdaw_data::center_by_key(key).expect("a challenge consumable");
            self.queue(Event::Call(challenge_joker, u32::from(center.0), 0.0));
        }
        for v in list(field(ch, "vouchers")) {
            let key = string(field(v, "id")).expect("a challenge voucher's id");
            let center = jackdaw_data::center_by_key(key).expect("a challenge voucher");
            self.gs.sets.used_vouchers.insert(center);
            self.queue(Event::Call(
                |g, c, _| {
                    let center = CenterId(c as u16);
                    let extra = center.get().config_num("extra");
                    g.apply_voucher(center.get().name, extra);
                },
                u32::from(center.0),
                0.0,
            ));
        }
        if let Some(rules) = field(ch, "rules") {
            for m in list(field(rules, "modifiers")) {
                let id = string(field(m, "id")).expect("a modifier id");
                let value = field(m, "value").and_then(|v| v.as_num());
                self.set_starting_param(id, value);
            }
            for c in list(field(rules, "custom")) {
                let id = string(field(c, "id")).expect("a rule id");
                let value = field(c, "value");
                let m = &mut self.gs.modifiers;
                match (id, value) {
                    ("no_reward", _) => {
                        let mut t = BTreeMap::new();
                        for slot in ["Small", "Big", "Boss"] {
                            t.insert(slot.to_owned(), LuaValue::Bool(true));
                        }
                        m.insert("no_blind_reward".to_owned(), LuaValue::Table(t));
                    }
                    ("no_reward_specific", Some(v)) => {
                        let slot = v.as_str().expect("a blind slot").to_owned();
                        let entry = m
                            .entry("no_blind_reward".to_owned())
                            .or_insert_with(|| LuaValue::Table(BTreeMap::new()));
                        if let LuaValue::Table(t) = entry {
                            t.insert(slot, LuaValue::Bool(true));
                        }
                    }
                    (_, Some(v)) => {
                        m.insert(id.to_owned(), lua_value(v));
                    }
                    ("no_shop_jokers", None) => self.gs.joker_rate = 0.0,
                    (_, None) => {
                        m.insert(id.to_owned(), LuaValue::Bool(true));
                    }
                }
            }
        }
        if let Some(r) = field(ch, "restrictions") {
            for b in list(field(r, "banned_cards")) {
                let id = string(field(b, "id")).expect("a banned card");
                self.gs.sets.banned_keys.insert(id.to_owned());
                for k in list(field(b, "ids")) {
                    if let Some(k) = k.as_str() {
                        self.gs.sets.banned_keys.insert(k.to_owned());
                    }
                }
            }
            for key in ["banned_tags", "banned_other"] {
                for b in list(field(r, key)) {
                    let id = string(field(b, "id")).expect("a banned key");
                    self.gs.sets.banned_keys.insert(id.to_owned());
                }
            }
        }
    }

    /// `G.GAME.starting_params[id] = value` for a challenge modifier.
    fn set_starting_param(&mut self, id: &str, value: Option<f64>) {
        let v = value.unwrap_or_else(|| panic!("challenge modifier {id} has no value"));
        let sp = &mut self.gs.starting_params;
        match id {
            "dollars" => sp.dollars = v,
            "hand_size" => sp.hand_size = v,
            "discards" => sp.discards = v,
            "hands" => sp.hands = v,
            "reroll_cost" => sp.reroll_cost = v,
            "joker_slots" => sp.joker_slots = v,
            "consumable_slots" => sp.consumable_slots = v,
            "ante_scaling" => sp.ante_scaling = v,
            other => panic!("unknown starting param {other}"),
        }
    }

    /// The starting deck (game.lua:2311-2367): the challenge's cards, or every `P_CARDS` front
    /// the deck's rules keep, sorted, made by `card_from_control`.
    fn build_deck(&mut self, ch: Option<&'static Value>) {
        let de = ch.and_then(|c| field(c, "deck"));
        let mut protos: Vec<Proto> = Vec::with_capacity(P_CARDS.len());
        let custom = de.and_then(|d| field(d, "cards"));
        if let Some(Value::List(cards)) = custom {
            for c in cards.iter() {
                let s = string(field(c, "s")).expect("a deck card's suit");
                let r = string(field(c, "r")).expect("a deck card's rank");
                protos.push(Proto {
                    front: jackdaw_data::card_by_parts(s, r).expect("a front"),
                    s,
                    r,
                    e: string(field(c, "e")),
                    d: string(field(c, "d")),
                    g: string(field(c, "g")),
                });
            }
        } else {
            let has = |v: Option<&Value>, x: &str| match v {
                Some(Value::Table(t)) => {
                    t.iter().any(|(k, b)| *k == x && b.as_bool() != Some(false))
                }
                Some(Value::List(items)) => items.iter().any(|i| i.as_str() == Some(x)),
                _ => false,
            };
            let erratic = self.gs.starting_params.erratic_suits_and_ranks;
            let no_faces = self.gs.starting_params.no_faces;
            for i in 0..P_CARDS.len() {
                let mut front = i;
                if erratic {
                    let seed = self.gs.rng.seed("erratic");
                    front = crate::rng::element_index(P_CARDS.len(), seed);
                }
                let key = P_CARDS[front].key;
                let front = jackdaw_data::CardId(front as u8);
                let (s, r) = (&key[0..1], &key[2..3]);
                let mut keep = true;
                let (mut e, mut d, mut g) = (None, None, None);
                if let Some(de) = de {
                    if field(de, "yes_ranks").is_some() && !has(field(de, "yes_ranks"), r) {
                        keep = false;
                    }
                    if has(field(de, "no_ranks"), r) {
                        keep = false;
                    }
                    if field(de, "yes_suits").is_some() && !has(field(de, "yes_suits"), s) {
                        keep = false;
                    }
                    if has(field(de, "no_suits"), s) {
                        keep = false;
                    }
                    e = string(field(de, "enhancement"));
                    d = string(field(de, "edition"));
                    g = string(field(de, "gold_seal"));
                }
                if no_faces && matches!(r, "K" | "Q" | "J") {
                    keep = false;
                }
                if keep {
                    protos.push(Proto {
                        front,
                        s,
                        r,
                        e,
                        d,
                        g,
                    });
                }
            }
        }
        protos.sort_by(Proto::cmp);
        // The cards are made in sort order (sort ids `first..`), each emplaced in front of the
        // deck, and `start_run` then shuffles the deck (`G.deck:shuffle()`, game.lua:2375):
        // pseudoshuffle sorts by sort id, then swaps. Making a deck card draws nothing and no
        // card reads another, so the swaps come first and each card is made in its place.
        let n = protos.len();
        let mut order: Vec<usize> = (0..n).collect();
        crate::rng::shuffle(&mut order, self.gs.rng.seed("shuffle"));
        let first = self.gs.sim.next_sort_id;
        self.gs.sim.next_sort_id += n as u32;
        // Nothing a deck card's creation does changes the costs the next one sees. A new card's
        // ability and costs come from its center, so cards of one center start as one template.
        let cost = self.cost_context(AreaId::Deck);
        let hands_played = self.gs.hands_played;
        let mut templates: Vec<Card> = Vec::new();
        let deck: Vec<Card> = order
            .iter()
            .map(|&k| {
                let p = &protos[k];
                let center =
                    p.e.map(|e| jackdaw_data::center_by_key(e).expect("an enhancement"))
                        .unwrap_or(centers::C_BASE);
                let template = match templates.iter().find(|t| t.center == center) {
                    Some(t) => *t,
                    None => {
                        let t = Card::new(0, center, None, hands_played, &cost);
                        templates.push(t);
                        t
                    }
                };
                card_from_control(p, template, first + k as u32, &cost)
            })
            .collect();
        // set_ability's `used_jokers` marks, once per center (no deck card is a To Do List).
        let mut marked = crate::state::CenterBits::default();
        for c in &deck {
            if !marked.contains(c.center) {
                marked.insert(c.center);
                self.mark_used(c.center);
            }
        }
        debug_assert!(
            self.gs.areas.deck.is_empty(),
            "a new run's deck starts empty"
        );
        self.gs.areas.deck = deck;
        self.gs.starting_deck_size = self.playing_card_count() as f64;
    }
}

/// `card_from_control(control)` (misc_functions.lua:1625-1633) for the card with sort id
/// `sort_id`, face down in the deck, from `template`: `Card(...)` for its center with no front
/// (`Card:init`'s `set_base` touches neither the ability nor the costs). The caller marks its
/// center used.
fn card_from_control(p: &Proto, template: Card, sort_id: u32, cost: &CostContext) -> Card {
    {
        let mut card = template;
        card.sort_id = sort_id;
        card.base = Some(crate::card::set_base(None, p.front, true));
        card.facing = crate::card::Facing::Back;
        if let Some(d) = p.d {
            card.set_edition(Some(EditionKind::from_name(d).expect("an edition")), cost);
        }
        if let Some(gs) = p.g {
            card.set_seal(Some(Seal::from_name(gs).expect("a seal")), cost);
        }
        card
    }
}

fn edition_code(e: EditionKind) -> u32 {
    match e {
        EditionKind::Foil => 0,
        EditionKind::Holo => 1,
        EditionKind::Polychrome => 2,
        EditionKind::Negative => 3,
    }
}

/// A back's starting consumable (back.lua:181-191):
/// `create_card('Tarot', G.consumeables, nil, nil, nil, nil, key, 'deck')`.
fn back_consumable(g: &mut Game, center: u32, _: f64) {
    let key = CenterId(center as u16);
    let card = g
        .create_card(CardSpec::new(PoolType::Tarot, AreaId::Consumables, "deck").forced(Some(key)));
    let id = card.sort_id;
    g.gs.areas.emplace(AreaId::Consumables, card, false, false);
    g.add_to_deck(id, false);
}

/// A challenge card's event: `add_joker(id, edition, silent)` (common_events.lua:372-384), then
/// the joker's Eternal and pinned stickers (game.lua:2058-2064).
fn challenge_joker(g: &mut Game, packed: u32, code: f64) {
    let center = CenterId((packed & 0xFFFF) as u16);
    let eternal = packed & (1 << 16) != 0;
    let pinned = packed & (2 << 16) != 0;
    let consumeable = center.get().consumeable;
    let area = if consumeable {
        AreaId::Consumables
    } else {
        AreaId::Jokers
    };
    let card = g.new_card(center, None, area);
    let id = card.sort_id;
    g.gs.areas.emplace(area, card, false, false);
    g.add_to_deck(id, false);
    if code > 0.0 {
        let edition = match code as u32 - 1 {
            0 => EditionKind::Foil,
            1 => EditionKind::Holo,
            2 => EditionKind::Polychrome,
            _ => EditionKind::Negative,
        };
        let cost = g.cost_context(area);
        let slot = g.card_mut(id).set_edition(Some(edition), &cost);
        if slot {
            if consumeable {
                g.gs.consumable_slots += 1.0;
            } else {
                g.gs.joker_slots += 1.0;
            }
        }
    }
    if eternal {
        g.card_mut(id).set_eternal(true);
    }
    if pinned {
        g.card_mut(id).pinned = true;
    }
}

/// The Checkered Deck's event: Clubs become Spades and Diamonds Hearts (back.lua:233-247).
fn checkered(g: &mut Game, _: u32, _: f64) {
    for id in g.playing_card_ids() {
        let suit = g.card(id).base.map(|b| b.suit);
        let to = match suit {
            Some(Suit::Clubs) => Suit::Spades,
            Some(Suit::Diamonds) => Suit::Hearts,
            _ => continue,
        };
        g.change_suit(id, to);
    }
}

impl Game {
    /// `Card:change_suit(new_suit)` (card.lua:153-161): a non-initial `set_base`, then the blind
    /// re-checks the card.
    pub(crate) fn change_suit(&mut self, id: u32, suit: Suit) {
        let base = self.card(id).base.expect("a playing card");
        let front = crate::card::Base::front_for(suit, base.value);
        let new = crate::card::set_base(Some(&base), front, false);
        self.card_mut(id).base = Some(new);
        self.debuff_card(id, false);
    }
}

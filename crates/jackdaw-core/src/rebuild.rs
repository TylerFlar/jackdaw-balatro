//! A game built from the player's view (`jackdaw.player_view/1`, [`crate::player_view`]).
//!
//! An agent that plans with the engine (a search, rollouts, score previews) needs a [`Game`]. In
//! a local run it has the real one; where it has only the player's view (Jackdaw Arena) it has
//! this one: a game whose every field the player sees is the view's, and whose hidden parts are
//! stand-ins. The same agent code then runs on the same kind of state in both places.
//!
//! - **What the view shows** is the view's: the phase, the run, the money and counters, the
//!   blinds and hand levels, the vouchers and tags, every card of the hand, jokers,
//!   consumables, shop and pack in its place, and the playing cards the deck view counts.
//! - **What it doesn't show** comes from a base game. With a *shadow* (a game built from an
//!   earlier view of the run, then stepped with the action the agent took since), that is what
//!   the agent knows from its own actions though no view lists it: the consumables it used
//!   (Satellite counts the Planets), the cards it played this ante (The Pillar), the sort it
//!   chose, the hidden counters its purchases moved. Without one, it is a new run of the view's
//!   back, stake and challenge, with what the view's vouchers and jokers do to the hidden
//!   counters.
//! - **Stand-ins** fill what nobody sees: the RNG and seed, the draw pile's order, the identity
//!   of face-down cards (taken from the cards the deck view counts as remaining, and left face
//!   down), Amber Acorn's shuffle (each face-down joker matched on its size and sell button),
//!   a Stone Card's rank and suit, a card's play count.
//! - **Sort ids** follow the view's card ids ([`ViewIds`]): a card a player follows from one
//!   view to the next keeps its sort id.
//!
//! The shadow's own RNG differs from the run's, so where its step rolled (a shop, a pack, a
//! draw, the next boss), the view's outcome replaces the shadow's, and the sets that remember
//! such outcomes follow: the jokers seen (`used_jokers`), the bosses met, Gros Michel's
//! extinction, the ante's voucher.
//!
//! The game's own player view is `view` again, card ids aside: [`same_view`] compares them.

use std::collections::{HashMap, HashSet};

use jackdaw_data::{BlindId, CardId, CenterId, CenterSet, TagId};
use serde_json::{Map, Value};

use crate::area::{AreaId, Areas};
use crate::card::{Card, CostContext, Edition, EditionKind, Facing, OptF64, Rank, Seal, Suit};
use crate::game::Game;
use crate::hand_eval::HandType;
use crate::legal::can_sell;
use crate::player_view::{FORMAT, shape};
use crate::rng::{Rng, element_index, shuffle};
use crate::run::RunSpec;
use crate::state::{
    BlindSlot, BlindState, CenterBits, GameState, HandState, HeldTag, LastScore, LuaValue,
    PackState, Phase, RoundResets,
};

/// Why a view can't be built into a game.
#[derive(Clone, Debug, PartialEq)]
pub struct RebuildError(pub String);

impl std::fmt::Display for RebuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "player view: {}", self.0)
    }
}

impl std::error::Error for RebuildError {}

type Result<T> = std::result::Result<T, RebuildError>;

fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(RebuildError(msg.into()))
}

/// The sort id of each card a view names, by its id in the view.
pub type ViewIds = HashMap<u64, u32>;

/// The seed of a game built with neither a shadow nor a seed.
pub const DEFAULT_SEED: &str = "PLAYVIEW";

/// The RNG stream the stand-ins draw from (the draw pile's order, face-down cards).
const STREAM: &str = "player_view";

/// What a game built from a view builds on: a game built from an earlier view of the run, then
/// stepped with the actions the agent took since.
#[derive(Clone, Copy, Debug)]
pub struct Shadow<'a> {
    pub game: &'a Game,
    /// The view ids its own build gave out.
    pub ids: &'a ViewIds,
    /// The playing cards its steps destroyed. Where its rolls chose them (Familiar, Glass) and
    /// the run's chose others, the view shows them still there, and they come back as they were
    /// (with what doesn't show, such as having been played this ante).
    pub gone: &'a [Card],
}

/// A game built from a view, and the sort id each card of the view got.
#[derive(Debug)]
pub struct Rebuilt {
    pub game: Game,
    pub ids: ViewIds,
}

// ---------------------------------------------------------------------------
// Reading the view
// ---------------------------------------------------------------------------

fn field<'a>(v: &'a Value, key: &str, path: &str) -> Result<&'a Value> {
    v.get(key)
        .ok_or_else(|| RebuildError(format!("{path}: missing {key:?}")))
}

fn num(v: &Value, key: &str, path: &str) -> Result<f64> {
    crate::snapshot::as_num(field(v, key, path)?, &format!("{path}.{key}"))
        .map_err(|e| RebuildError(e.0))
}

fn opt_num(v: &Value, key: &str, path: &str) -> Result<Option<f64>> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(x) => crate::snapshot::as_num(x, &format!("{path}.{key}"))
            .map(Some)
            .map_err(|e| RebuildError(e.0)),
    }
}

fn boolean(v: &Value, key: &str, path: &str) -> Result<bool> {
    field(v, key, path)?
        .as_bool()
        .ok_or_else(|| RebuildError(format!("{path}.{key}: expected a boolean")))
}

fn string<'a>(v: &'a Value, key: &str, path: &str) -> Result<&'a str> {
    field(v, key, path)?
        .as_str()
        .ok_or_else(|| RebuildError(format!("{path}.{key}: expected a string")))
}

/// A string, or `None` for `null` or a missing key.
fn opt_str<'a>(v: &'a Value, key: &str, path: &str) -> Result<Option<&'a str>> {
    match v.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => err(format!("{path}.{key}: expected a string")),
    }
}

fn array<'a>(v: &'a Value, key: &str, path: &str) -> Result<&'a [Value]> {
    field(v, key, path)?
        .as_array()
        .map(Vec::as_slice)
        .ok_or_else(|| RebuildError(format!("{path}.{key}: expected an array")))
}

fn lookup<T>(found: Option<T>, what: &str, key: &str, path: &str) -> Result<T> {
    found.ok_or_else(|| RebuildError(format!("{path}: unknown {what} {key:?}")))
}

fn center(key: &str, path: &str) -> Result<CenterId> {
    lookup(jackdaw_data::center_by_key(key), "center", key, path)
}

fn front(key: &str, path: &str) -> Result<CardId> {
    lookup(jackdaw_data::card_by_key(key), "front", key, path)
}

fn hand_type(name: &str, path: &str) -> Result<HandType> {
    lookup(HandType::from_name(name), "hand", name, path)
}

fn edition(name: &str, path: &str) -> Result<EditionKind> {
    lookup(EditionKind::from_name(name), "edition", name, path)
}

fn seal(name: &str, path: &str) -> Result<Seal> {
    lookup(Seal::from_name(name), "seal", name, path)
}

fn blind_id(key: &str, path: &str) -> Result<BlindId> {
    lookup(jackdaw_data::blind_by_key(key), "blind", key, path)
}

fn tag_id(key: &str, path: &str) -> Result<TagId> {
    lookup(jackdaw_data::tag_by_key(key), "tag", key, path)
}

fn suit(name: &str, path: &str) -> Result<Suit> {
    lookup(Suit::from_name(name), "suit", name, path)
}

fn rank(name: &str, path: &str) -> Result<Rank> {
    lookup(Rank::from_name(name), "rank", name, path)
}

fn slot_name(name: &str, path: &str) -> Result<BlindSlot> {
    lookup(BlindSlot::from_name(name), "blind slot", name, path)
}

/// A playing card's center (the base card or an enhancement).
fn is_playing(c: CenterId) -> bool {
    matches!(c.get().set, CenterSet::Default | CenterSet::Enhanced)
}

/// What a face-up card shows.
struct Face<'a> {
    center: CenterId,
    /// `None` off playing cards, and on a Stone Card, whose rank and suit don't show.
    front: Option<CardId>,
    edition: Option<EditionKind>,
    seal: Option<Seal>,
    debuff: bool,
    cost: f64,
    sell_cost: f64,
    eternal: bool,
    perishable: bool,
    perish_tally: Option<f64>,
    rental: bool,
    pinned: bool,
    ability: &'a Value,
}

/// A card as the view shows it.
struct ViewCard<'a> {
    path: String,
    id: u64,
    /// `None` face down.
    face: Option<Face<'a>>,
    /// The card's size (`Some(None)`: the standard one), when the view says.
    shape: Option<Option<&'a str>>,
    /// Whether its sell button is active, when the view says.
    sellable: Option<bool>,
    shaking: bool,
    forced: bool,
}

fn view_card(v: &Value, path: String) -> Result<ViewCard<'_>> {
    let id = field(v, "id", &path)?
        .as_u64()
        .ok_or_else(|| RebuildError(format!("{path}.id: expected an integer")))?;
    let face = if boolean(v, "face_down", &path)? {
        None
    } else {
        let st = field(v, "stickers", &path)?;
        let sp = format!("{path}.stickers");
        Some(Face {
            center: center(string(v, "key", &path)?, &path)?,
            front: opt_str(v, "front", &path)?
                .map(|f| front(f, &path))
                .transpose()?,
            edition: opt_str(v, "edition", &path)?
                .map(|e| edition(e, &path))
                .transpose()?,
            seal: opt_str(v, "seal", &path)?
                .map(|s| seal(s, &path))
                .transpose()?,
            debuff: boolean(v, "debuff", &path)?,
            cost: num(v, "cost", &path)?,
            sell_cost: num(v, "sell_cost", &path)?,
            eternal: boolean(st, "eternal", &sp)?,
            perishable: boolean(st, "perishable", &sp)?,
            perish_tally: opt_num(st, "perish_tally", &sp)?,
            rental: boolean(st, "rental", &sp)?,
            pinned: boolean(st, "pinned", &sp)?,
            ability: field(v, "ability", &path)?,
        })
    };
    Ok(ViewCard {
        id,
        face,
        shape: v.get("shape").map(Value::as_str),
        sellable: v.get("sellable").and_then(Value::as_bool),
        shaking: v.get("shaking").and_then(Value::as_bool).unwrap_or(false),
        forced: v.get("forced").and_then(Value::as_bool).unwrap_or(false),
        path,
    })
}

/// The cards of one of the view's lists (`null` for none).
fn view_cards<'a>(list: Option<&'a Value>, path: &str) -> Result<Vec<ViewCard<'a>>> {
    match list {
        None | Some(Value::Null) => Ok(Vec::new()),
        Some(Value::Array(cards)) => cards
            .iter()
            .enumerate()
            .map(|(i, c)| view_card(c, format!("{path}[{i}]")))
            .collect(),
        Some(_) => err(format!("{path}: expected an array")),
    }
}

// ---------------------------------------------------------------------------
// Playing cards as the deck view counts them
// ---------------------------------------------------------------------------

/// A number's bits with one zero, for a key.
fn bits(x: f64) -> u64 {
    if x == 0.0 { 0 } else { x.to_bits() }
}

/// What the deck view shows of a playing card: the cards it can't tell apart are the same.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Identity {
    center: CenterId,
    /// `None` for a Stone Card.
    front: Option<CardId>,
    edition: Option<EditionKind>,
    seal: Option<Seal>,
    perma_bonus: u64,
}

impl Identity {
    fn of(c: &Card) -> Identity {
        Identity {
            center: c.center,
            front: if c.is_stone() {
                None
            } else {
                c.base.map(|b| b.front)
            },
            edition: c.edition.map(|e| e.kind),
            seal: c.seal,
            perma_bonus: bits(c.ability.perma_bonus),
        }
    }

    fn of_face(face: &Face<'_>, path: &str) -> Result<Identity> {
        Ok(Identity {
            center: face.center,
            front: face.front,
            edition: face.edition,
            seal: face.seal,
            perma_bonus: bits(num(
                face.ability,
                "perma_bonus",
                &format!("{path}.ability"),
            )?),
        })
    }

    fn of_entry(e: &Value, path: &str) -> Result<Identity> {
        Ok(Identity {
            center: center(string(e, "key", path)?, path)?,
            front: opt_str(e, "front", path)?
                .map(|f| front(f, path))
                .transpose()?,
            edition: opt_str(e, "edition", path)?
                .map(|x| edition(x, path))
                .transpose()?,
            seal: opt_str(e, "seal", path)?
                .map(|s| seal(s, path))
                .transpose()?,
            perma_bonus: bits(num(e, "perma_bonus", path)?),
        })
    }
}

/// A multiset of identities that keeps the order they first came in, so that walking it is
/// deterministic.
#[derive(Default)]
struct Bag {
    counts: HashMap<Identity, usize>,
    order: Vec<Identity>,
    len: usize,
}

impl Bag {
    fn add(&mut self, id: Identity) {
        let n = self.counts.entry(id).or_insert(0);
        if *n == 0 && !self.order.contains(&id) {
            self.order.push(id);
        }
        *n += 1;
        self.len += 1;
    }

    fn contains(&self, id: &Identity) -> bool {
        self.counts.get(id).is_some_and(|&n| n > 0)
    }

    fn take(&mut self, id: &Identity) -> bool {
        match self.counts.get_mut(id) {
            Some(n) if *n > 0 => {
                *n -= 1;
                self.len -= 1;
                true
            }
            _ => false,
        }
    }

    /// Take one of the cards, each as likely as the others.
    fn take_random(&mut self, rng: &mut Rng) -> Identity {
        let mut k = element_index(self.len, rng.seed(STREAM));
        for id in &self.order {
            let n = self.counts[id];
            if k < n {
                let id = *id;
                self.take(&id);
                return id;
            }
            k -= n;
        }
        unreachable!("an index within the bag")
    }

    /// Every card left, in the order they came in.
    fn drain(&mut self) -> Vec<Identity> {
        let mut out = Vec::with_capacity(self.len);
        for id in &self.order {
            let n = self.counts.get_mut(id).expect("a counted identity");
            out.extend(std::iter::repeat_n(*id, *n));
            *n = 0;
        }
        self.len = 0;
        out
    }
}

// ---------------------------------------------------------------------------
// The base game's cards
// ---------------------------------------------------------------------------

/// The base's cards, each taken at most once into the new game; the ones never taken are the
/// cards the view says don't exist (the shadow's own draws and rolls). Behind them, the playing
/// cards the shadow's steps destroyed ([`Shadow::gone`]): where the view says one still exists,
/// the shadow's rolls picked it and the run's picked another, and it comes back as it was.
struct Pool {
    cards: Vec<(AreaId, Card)>,
    /// Whether the card is one the shadow's steps destroyed (its area is meaningless).
    gone: Vec<bool>,
    taken: Vec<bool>,
    by_sort: HashMap<u32, usize>,
    /// The playing cards, by what the deck view shows of them.
    by_identity: HashMap<Identity, Vec<usize>>,
}

impl Pool {
    fn new(gs: &GameState, gone: &[Card]) -> Pool {
        let mut cards = Vec::new();
        for a in AreaId::ALL {
            cards.extend(gs.areas.get(a).iter().map(|c| (a, *c)));
        }
        let live = cards.len();
        let mut by_sort: HashMap<u32, usize> = cards
            .iter()
            .enumerate()
            .map(|(i, (_, c))| (c.sort_id, i))
            .collect();
        for c in gone {
            if !by_sort.contains_key(&c.sort_id) && c.base.is_some() {
                by_sort.insert(c.sort_id, cards.len());
                cards.push((AreaId::Deck, *c));
            }
        }
        let n = cards.len();
        let mut by_identity: HashMap<Identity, Vec<usize>> = HashMap::new();
        for (i, (_, c)) in cards.iter().enumerate() {
            if c.base.is_some() {
                by_identity.entry(Identity::of(c)).or_default().push(i);
            }
        }
        Pool {
            gone: (0..n).map(|i| i >= live).collect(),
            taken: vec![false; n],
            cards,
            by_sort,
            by_identity,
        }
    }

    fn take_at(&mut self, i: usize) -> (usize, AreaId, Card) {
        self.taken[i] = true;
        let (a, c) = self.cards[i];
        (i, a, c)
    }

    /// The card with this sort id, if the base has it (or its steps destroyed it) and it isn't
    /// taken.
    fn take_sort(&mut self, sort_id: u32) -> Option<(usize, AreaId, Card)> {
        let &i = self.by_sort.get(&sort_id)?;
        (!self.taken[i]).then(|| self.take_at(i))
    }

    /// The first card of the base not taken for which `ok` holds, looking through the areas
    /// `prefer` in turn.
    fn take_where(
        &mut self,
        prefer: &[AreaId],
        ok: impl Fn(&Card) -> bool,
    ) -> Option<(usize, AreaId, Card)> {
        for &area in prefer {
            let found = (0..self.cards.len()).find(|&i| {
                !self.taken[i] && !self.gone[i] && self.cards[i].0 == area && ok(&self.cards[i].1)
            });
            if let Some(i) = found {
                return Some(self.take_at(i));
            }
        }
        None
    }

    /// A playing card with this identity for which `ok` holds: the base's, from the areas
    /// `prefer` in turn, else one the shadow's steps destroyed.
    fn take_identity_where(
        &mut self,
        id: &Identity,
        prefer: &[AreaId],
        ok: impl Fn(&Card) -> bool,
    ) -> Option<(usize, AreaId, Card)> {
        let list = self.by_identity.get(id)?;
        let free = |i: usize| !self.taken[i] && ok(&self.cards[i].1);
        let found = prefer
            .iter()
            .find_map(|&area| {
                list.iter()
                    .copied()
                    .find(|&i| !self.gone[i] && self.cards[i].0 == area && free(i))
            })
            .or_else(|| list.iter().copied().find(|&i| self.gone[i] && free(i)))?;
        Some(self.take_at(found))
    }

    fn take_identity(&mut self, id: &Identity, prefer: &[AreaId]) -> Option<(usize, AreaId, Card)> {
        self.take_identity_where(id, prefer, |_| true)
    }

    /// The base's own cards (not the ones its steps destroyed), taken or not.
    fn base(&self) -> impl Iterator<Item = (AreaId, &Card)> {
        self.cards
            .iter()
            .zip(&self.gone)
            .filter(|(_, g)| !**g)
            .map(|((a, c), _)| (*a, c))
    }
}

// ---------------------------------------------------------------------------
// Building
// ---------------------------------------------------------------------------

/// The areas of the cards that aren't playing cards, and where the view lists them.
const NON_PLAYING: [(AreaId, &str); 6] = [
    (AreaId::Jokers, "jokers"),
    (AreaId::Consumables, "consumables"),
    (AreaId::ShopJokers, "jokers"),
    (AreaId::ShopVouchers, "vouchers"),
    (AreaId::ShopBooster, "boosters"),
    (AreaId::PackCards, "cards"),
];

/// A face-down card of the view that the base can't name, and what shows of it.
struct Waiting<'a> {
    list: usize,
    index: usize,
    shape: Option<Option<&'a str>>,
    sellable: Option<bool>,
    shaking: bool,
    forced: bool,
}

/// What the base held that the view overwrites, read before it does.
struct Before {
    boss: Option<BlindId>,
    voucher: Option<CenterId>,
    blind: Option<BlindState>,
    tags: Vec<HeldTag>,
    vouchers: CenterBits,
    jokers: Vec<Card>,
    ante: f64,
    phase: Phase,
}

struct Builder<'a> {
    view: &'a Value,
    gs: GameState,
    pool: Pool,
    /// The base's view ids.
    known: &'a ViewIds,
    ids: ViewIds,
    /// Whether the base is a shadow (else a new run).
    shadow: bool,
    /// Whether the draw pile gets a new order (no shadow, or a new seed).
    reshuffle: bool,
    /// The playing cards made here rather than taken from the base: the boss debuffs them as
    /// it would have.
    made: Vec<u32>,
}

impl Builder<'_> {
    fn rng(&mut self) -> &mut Rng {
        &mut self.gs.rng
    }

    fn cost_context(&self, area: AreaId) -> CostContext {
        CostContext {
            inflation: self.gs.inflation,
            discount_percent: self.gs.discount_percent,
            booster_ante_scaling: self.gs.modifier("booster_ante_scaling").is_some(),
            ante: self.gs.round_resets.ante,
            astronomer: false,
            in_shop: matches!(area, AreaId::ShopJokers | AreaId::ShopBooster),
        }
    }

    /// A front for a playing card whose rank and suit nobody sees (a Stone Card's).
    fn stand_in_front(&mut self) -> CardId {
        let n = jackdaw_data::P_CARDS.len();
        CardId(element_index(n, self.rng().seed(STREAM)) as u8)
    }

    /// A playing card with this identity and sort id, as the game makes one.
    fn playing_card(&mut self, id: &Identity, sort_id: u32) -> Card {
        let front = match id.front {
            Some(f) => f,
            None => self.stand_in_front(),
        };
        let ctx = self.cost_context(AreaId::Deck);
        let mut c = Card::new(sort_id, id.center, Some(front), self.gs.hands_played, &ctx);
        c.edition = id.edition.map(Edition::new);
        c.seal = id.seal;
        c.ability.perma_bonus = f64::from_bits(id.perma_bonus);
        c.set_cost(&ctx);
        c
    }

    fn new_playing_card(&mut self, id: &Identity) -> Card {
        let sort_id = self.gs.next_sort_id();
        self.made.push(sort_id);
        self.playing_card(id, sort_id)
    }

    /// A face-up card from what the view shows, with `sort_id`. `like` is the base's card when
    /// it's the same one (same center, and same front for a playing card): what doesn't show
    /// (a playing card's play count and original rank and suit) is kept from it.
    fn face_up(
        &mut self,
        vc: &ViewCard<'_>,
        face: &Face<'_>,
        area: AreaId,
        sort_id: u32,
        like: Option<&Card>,
    ) -> Result<Card> {
        let playing = is_playing(face.center);
        let front = match (face.front, playing) {
            (Some(f), _) => Some(f),
            (None, true) => Some(match like.and_then(|c| c.base) {
                Some(b) => b.front,
                None => self.stand_in_front(),
            }),
            (None, false) => None,
        };
        let ctx = self.cost_context(area);
        let mut c = Card::new(sort_id, face.center, front, self.gs.hands_played, &ctx);
        c.ability = crate::snapshot::ability_from(
            face.ability,
            face.center,
            &format!("{}.ability", vc.path),
        )
        .map_err(|e| RebuildError(e.0))?;
        c.ability.eternal = face.eternal;
        c.ability.perishable = face.perishable;
        c.ability.perish_tally = OptF64::from(face.perish_tally);
        c.ability.rental = face.rental;
        c.edition = face.edition.map(Edition::new);
        c.seal = face.seal;
        c.set_cost(&ctx);
        c.cost = face.cost;
        c.sell_cost = face.sell_cost;
        c.debuff = face.debuff;
        c.pinned = face.pinned;
        c.juice = vc.shaking;
        c.ability.forced_selection = vc.forced;
        c.added_to_deck = match area {
            AreaId::Jokers => !face.debuff,
            AreaId::Consumables => true,
            AreaId::Hand | AreaId::Play | AreaId::Deck | AreaId::Discard => {
                like.is_some_and(|l| l.added_to_deck)
            }
            _ => false,
        };
        if let Some(l) = like {
            c.view_epoch = l.view_epoch;
            if let (Some(b), Some(lb)) = (c.base.as_mut(), l.base) {
                b.times_played = lb.times_played;
                if lb.front == b.front {
                    b.original_value = lb.original_value;
                    b.suit_nominal_original = lb.suit_nominal_original;
                }
            }
        }
        Ok(c)
    }

    /// The base's card `c` where the view shows a face-down card.
    fn face_down(mut c: Card, vc: &ViewCard<'_>) -> Card {
        c.facing = Facing::Back;
        c.juice = vc.shaking;
        c.ability.forced_selection = vc.forced;
        c
    }

    // -- scalars ---------------------------------------------------------------------

    fn scalars(&mut self, before: &Before) -> Result<()> {
        let v = self.view;
        let gs = &mut self.gs;
        let ph = string(v, "phase", "$")?;
        gs.phase = lookup(Phase::from_name(ph), "phase", ph, "$.phase")?;
        let run = field(v, "run", "$")?;
        gs.win_ante = num(run, "win_ante", "$.run")?;
        gs.won = boolean(run, "won", "$.run")?;

        let g = field(v, "game", "$")?;
        let p = "$.game";
        gs.round_resets.ante = num(g, "ante", p)?;
        for (dst, key) in [
            (&mut gs.round, "round"),
            (&mut gs.dollars, "dollars"),
            (&mut gs.chips, "chips"),
            (&mut gs.skips, "skips"),
            (&mut gs.hands_played, "hands_played"),
            (&mut gs.unused_discards, "unused_discards"),
            (&mut gs.hand_size, "hand_size"),
            (&mut gs.joker_slots, "joker_slots"),
            (&mut gs.consumable_slots, "consumable_slots"),
            (&mut gs.interest_cap, "interest_cap"),
            (&mut gs.interest_amount, "interest_amount"),
            (&mut gs.discount_percent, "discount_percent"),
            (&mut gs.inflation, "inflation"),
            (&mut gs.rental_rate, "rental_rate"),
            (&mut gs.bankrupt_at, "bankrupt_at"),
            (&mut gs.probabilities_normal, "probabilities_normal"),
        ] {
            *dst = num(g, key, p)?;
        }
        gs.last_tarot_planet = opt_str(g, "last_tarot_planet", p)?
            .map(|k| center(k, p))
            .transpose()?;
        gs.last_hand_played = opt_str(g, "last_hand_played", p)?
            .map(|h| hand_type(h, p))
            .transpose()?;
        gs.current_round.most_played_poker_hand = hand_type(string(g, "most_played_hand", p)?, p)?;
        gs.blind_on_deck = opt_str(g, "blind_on_deck", p)?
            .map(|b| slot_name(b, p))
            .transpose()?;
        gs.round_resets.boss_rerolled = boolean(g, "boss_rerolled", p)?;

        let cr = field(v, "current_round", "$")?;
        let p = "$.current_round";
        let c = &mut gs.current_round;
        c.hands_left = num(cr, "hands_left", p)?;
        c.discards_left = num(cr, "discards_left", p)?;
        c.hands_played = num(cr, "hands_played", p)?;
        c.discards_used = num(cr, "discards_used", p)?;
        c.reroll_cost = num(cr, "reroll_cost", p)?;
        c.free_rerolls = num(cr, "free_rerolls", p)?;
        if let Some(x) = cr.get("idol_card") {
            let q = "$.current_round.idol_card";
            c.idol_card = (
                suit(string(x, "suit", q)?, q)?,
                rank(string(x, "rank", q)?, q)?,
            );
        }
        if let Some(x) = cr.get("mail_card") {
            let q = "$.current_round.mail_card";
            c.mail_card = rank(string(x, "rank", q)?, q)?;
        }
        if let Some(x) = cr.get("ancient_card") {
            let q = "$.current_round.ancient_card";
            c.ancient_card = opt_str(x, "suit", q)?.map(|s| suit(s, q)).transpose()?;
        }
        if let Some(x) = cr.get("castle_card") {
            let q = "$.current_round.castle_card";
            c.castle_card = suit(string(x, "suit", q)?, q)?;
        }
        if let Some(x) = opt_num(cr, "cash_out", p)? {
            gs.sim.pending_cashout = x;
        }
        // The reroll cost is the base cost plus the round's increase (`calculate_reroll_cost`,
        // common_events.lua:2263-2275); the increase doesn't show, the cost does.
        if c.free_rerolls <= 0.0 {
            let base = gs
                .round_resets
                .temp_reroll_cost
                .unwrap_or(gs.round_resets.reroll_cost);
            c.reroll_cost_increase = (c.reroll_cost - base).max(0.0);
        }

        self.blinds()?;
        self.blind(before)?;
        // `set_blind` sets it to the blind's own flag and a defeat sets the empty blind
        // (blind.lua:336), so at rest it is the current blind's.
        self.gs.sim.last_blind_was_boss = self.gs.blind.as_ref().is_some_and(|b| b.boss);

        let gs = &mut self.gs;
        let hands = field(v, "hands", "$")?;
        for h in HandType::ALL {
            let p = format!("$.hands.{}", h.name());
            let e = field(hands, h.name(), "$.hands")?;
            *gs.hand_mut(h) = HandState {
                level: num(e, "level", &p)?,
                chips: num(e, "chips", &p)?,
                mult: num(e, "mult", &p)?,
                played: num(e, "played", &p)?,
                played_this_round: num(e, "played_this_round", &p)?,
                visible: boolean(e, "visible", &p)?,
            };
        }
        gs.sets.used_vouchers = array(v, "vouchers", "$")?
            .iter()
            .map(|k| {
                let k = k
                    .as_str()
                    .ok_or_else(|| RebuildError("$.vouchers: expected strings".into()))?;
                center(k, "$.vouchers")
            })
            .collect::<Result<CenterBits>>()?;
        let tags = array(v, "tags", "$")?
            .iter()
            .map(|k| {
                let k = k
                    .as_str()
                    .ok_or_else(|| RebuildError("$.tags: expected strings".into()))?;
                tag_id(k, "$.tags")
            })
            .collect::<Result<Vec<_>>>()?;
        gs.tags = if before.tags.iter().map(|t| t.id).eq(tags.iter().copied()) {
            before.tags.clone()
        } else {
            tags.iter().map(|&t| HeldTag::new(t)).collect()
        };
        // A held tag was awarded: the run's awards (a set, as the observation reads them) hold it.
        for &t in &tags {
            if !gs.sim.tag_history.contains(&t) {
                gs.sim.tag_history.push(t);
            }
        }
        if let Value::Object(totals) = field(v, "consumable_usage", "$")? {
            gs.consumeable_usage.totals = totals
                .iter()
                .map(|(k, x)| {
                    crate::snapshot::as_num(x, "$.consumable_usage")
                        .map(|n| (k.clone(), n))
                        .map_err(|e| RebuildError(e.0))
                })
                .collect::<Result<_>>()?;
        }
        gs.sim.last_score = match field(v, "last_score", "$")? {
            Value::Null => None,
            s => {
                let p = "$.last_score";
                Some(LastScore {
                    hand_type: hand_type(string(s, "hand_type", p)?, p)?,
                    chips: num(s, "chips", p)?,
                    mult: num(s, "mult", p)?,
                    total: num(s, "total", p)?,
                    dollars: num(s, "dollars", p)?,
                    debuffed: boolean(s, "debuffed", p)?,
                })
            }
        };
        gs.pack = match v.get("pack") {
            None | Some(Value::Null) => None,
            Some(pk) => Some(PackState {
                kind: string(pk, "kind", "$.pack")?.to_owned(),
                choices_remaining: num(pk, "choices_remaining", "$.pack")?,
                dealt: Vec::new(),
                // A pack opens in the shop, or from a tag at blind select.
                return_phase: match v.get("shop") {
                    None | Some(Value::Null) => Phase::BlindSelect,
                    Some(_) => Phase::Shop,
                },
            }),
        };
        if gs.phase == Phase::PackOpening
            && gs
                .pack
                .as_ref()
                .is_none_or(|p| crate::game::LuaState::for_pack(&p.kind).is_none())
        {
            return err("$.pack: an open pack of a known kind in pack_opening");
        }
        Ok(())
    }

    /// The ante's three blinds: their states, keys, skip tags and Orbital Tag hands.
    fn blinds(&mut self) -> Result<()> {
        let v = field(self.view, "blinds", "$")?;
        let rr = &mut self.gs.round_resets;
        let mut tags = rr.blind_tags.unwrap_or([None, None]);
        let mut any_tag = rr.blind_tags.is_some();
        let ante_key = rr.ante.to_string();
        for slot in BlindSlot::ALL {
            let p = format!("$.blinds.{}", slot.name());
            let b = field(v, slot.name(), "$.blinds")?;
            let state = string(b, "state", &p)?;
            let state = RoundResets::STATES
                .into_iter()
                .find(|&s| s == state)
                .ok_or_else(|| RebuildError(format!("{p}.state: unknown state {state:?}")))?;
            rr.set_blind_state(slot, state);
            if let Some(k) = opt_str(b, "key", &p)? {
                rr.blind_choices[slot.index()] = Some(blind_id(k, &p)?);
            }
            if slot != BlindSlot::Boss {
                let tag = opt_str(b, "tag", &p)?.map(|t| tag_id(t, &p)).transpose()?;
                any_tag |= tag.is_some();
                tags[slot.index()] = tag;
            }
            if let Some(h) = opt_str(b, "orbital_hand", &p)? {
                let h = hand_type(h, &p)?;
                self.gs
                    .orbital_choices
                    .entry(ante_key.clone())
                    .or_insert([HandType::Pair; 3])[slot.index()] = h;
            }
        }
        let rr = &mut self.gs.round_resets;
        rr.blind_tags = any_tag.then_some(tags);
        Ok(())
    }

    /// The blind being played: what shows from the view, the rest from the base's when it's
    /// the same blind.
    fn blind(&mut self, before: &Before) -> Result<()> {
        let v = field(self.view, "blind", "$")?;
        if v.is_null() {
            if self.gs.blind.as_ref().is_some_and(|b| b.key.is_some()) {
                self.gs.blind = Some(crate::blind::empty_blind());
            }
            return Ok(());
        }
        let p = "$.blind";
        let key = blind_id(string(v, "key", p)?, p)?;
        let same = before.blind.as_ref().filter(|b| b.key == Some(key));
        let eye_hands = array(v, "eye_hands", p)?
            .iter()
            .map(|h| hand_type(h.as_str().unwrap_or_default(), p))
            .collect::<Result<Vec<_>>>()?;
        let rr = &self.gs.round_resets;
        self.gs.blind = Some(BlindState {
            key: Some(key),
            chips: num(v, "chips", p)?,
            mult: num(v, "mult", p)?,
            dollars: num(v, "dollars", p)?,
            boss: boolean(v, "boss", p)?,
            disabled: boolean(v, "disabled", p)?,
            triggered: same.is_some_and(|b| b.triggered),
            // Each draw to the hand clears it (blind.lua:440), so a round at rest has it clear.
            prepped: same.is_some_and(|b| b.prepped),
            eye_hands,
            mouth_hand: opt_str(v, "mouth_hand", p)?
                .map(|h| hand_type(h, p))
                .transpose()?,
            // What The Water and The Needle took (blind.lua:93-104), given back if disabled.
            discards_sub: match same {
                Some(b) => b.discards_sub,
                None => (key.get().name == "The Water").then_some(rr.discards),
            },
            hands_sub: match same {
                Some(b) => b.hands_sub,
                None => (key.get().name == "The Needle").then_some(rr.hands - 1.0),
            },
        });
        Ok(())
    }

    // -- jokers, consumables, shop, pack ------------------------------------------------

    fn non_playing(&mut self) -> Result<Vec<(AreaId, Vec<Card>)>> {
        let v = self.view;
        let shop = v.get("shop").filter(|s| !s.is_null());
        let pack = v.get("pack").filter(|s| !s.is_null());
        let mut out = Vec::new();
        let mut waiting = Vec::new();
        for (area, key) in NON_PLAYING {
            let (list, path) = match area {
                AreaId::Jokers | AreaId::Consumables => (v.get(key), format!("$.{key}")),
                AreaId::PackCards => (pack.and_then(|p| p.get(key)), format!("$.pack.{key}")),
                _ => (shop.and_then(|s| s.get(key)), format!("$.shop.{key}")),
            };
            let cards = view_cards(list, &path)?;
            let mut built = Vec::with_capacity(cards.len());
            for (i, vc) in cards.into_iter().enumerate() {
                let known = self.known.get(&vc.id).copied();
                let claimed = known.and_then(|s| self.pool.take_sort(s));
                let card = match (&vc.face, claimed) {
                    (Some(face), claimed) => {
                        let sort_id = match claimed {
                            Some((_, _, c)) => c.sort_id,
                            None => self.gs.next_sort_id(),
                        };
                        let like = claimed
                            .map(|(_, _, c)| c)
                            .filter(|c| c.center == face.center);
                        Some(self.face_up(&vc, face, area, sort_id, like.as_ref())?)
                    }
                    (None, Some((_, _, c))) => Some(Builder::face_down(c, &vc)),
                    (None, None) => {
                        if area != AreaId::Jokers {
                            return err(format!(
                                "{}: a face-down card the game never deals there",
                                vc.path
                            ));
                        }
                        waiting.push(Waiting {
                            list: out.len(),
                            index: i,
                            shape: vc.shape,
                            sellable: vc.sellable,
                            shaking: vc.shaking,
                            forced: vc.forced,
                        });
                        None
                    }
                };
                if let Some(c) = &card {
                    self.ids.insert(vc.id, c.sort_id);
                }
                built.push((vc.id, card));
            }
            out.push((area, built));
        }
        // Face-down jokers the base can't name: Amber Acorn turned them down and shuffled them,
        // and the player lost track of which is which. Each gets one of the base's jokers left
        // that looks the same from the back: its size, whether it can be sold, whether it
        // shakes (the base's own shake follows the same rules). The closest matches go first, so
        // a loose one never takes what a close one needs.
        let fits = |c: &Card, w: &Waiting<'_>, tier: u8| {
            c.is_joker()
                && w.shape.is_none_or(|s| shape(c) == s)
                && (tier > 1 || w.sellable.is_none_or(|s| can_sell(c, AreaId::Jokers) == s))
                && (tier > 0 || c.juice == w.shaking)
        };
        let mut got: Vec<Option<Card>> = vec![None; waiting.len()];
        for tier in 0..3 {
            for (k, w) in waiting.iter().enumerate() {
                if got[k].is_none() {
                    got[k] = self
                        .pool
                        .take_where(&[AreaId::Jokers], |c| fits(c, w, tier))
                        .map(|(_, _, c)| c);
                }
            }
        }
        for (w, c) in waiting.iter().zip(got) {
            let mut c = match c {
                Some(c) => c,
                None => self.stand_in_joker(w.shape.flatten(), w.sellable),
            };
            c.facing = Facing::Back;
            c.juice = w.shaking;
            c.ability.forced_selection = w.forced;
            let (id, slot) = &mut out[w.list].1[w.index];
            self.ids.insert(*id, c.sort_id);
            *slot = Some(c);
        }
        Ok(out
            .into_iter()
            .map(|(a, cards)| {
                (
                    a,
                    cards.into_iter().map(|(_, c)| c.expect("built")).collect(),
                )
            })
            .collect())
    }

    /// A joker for a face-down one nothing is known of but its size and sell button.
    fn stand_in_joker(&mut self, sh: Option<&str>, sellable: Option<bool>) -> Card {
        use jackdaw_data::centers::{J_HALF, J_JOKER, J_PHOTOGRAPH, J_SQUARE, J_WEE};
        let center = match sh {
            Some("half") => J_HALF,
            Some("photograph") => J_PHOTOGRAPH,
            Some("square") => J_SQUARE,
            Some("wee") => J_WEE,
            _ => J_JOKER,
        };
        let sort_id = self.gs.next_sort_id();
        let ctx = self.cost_context(AreaId::Jokers);
        let mut c = Card::new(sort_id, center, None, self.gs.hands_played, &ctx);
        c.set_eternal(sellable == Some(false));
        c.added_to_deck = true;
        c
    }

    // -- playing cards --------------------------------------------------------------------

    /// The hand, the play area, the discard pile and the draw pile.
    fn playing(&mut self) -> Result<[Vec<Card>; 4]> {
        let v = self.view;
        let mut remaining = Bag::default();
        let mut seen = Bag::default();
        for (i, e) in array(v, "deck", "$")?.iter().enumerate() {
            let p = format!("$.deck[{i}]");
            let id = Identity::of_entry(e, &p)?;
            if boolean(e, "remaining", &p)? {
                remaining.add(id);
            } else {
                seen.add(id);
            }
        }
        let deck_size = num(v, "deck_size", "$")? as usize;
        let discard_size = num(v, "discard_size", "$")? as usize;
        let lists = [
            view_cards(v.get("hand"), "$.hand")?,
            view_cards(v.get("play"), "$.play")?,
        ];
        let mut held: [Vec<Option<Card>>; 2] =
            [vec![None; lists[0].len()], vec![None; lists[1].len()]];

        // Face up in the hand: the deck view counts these as seen.
        for (k, area) in [AreaId::Hand, AreaId::Play].into_iter().enumerate() {
            for (i, vc) in lists[k].iter().enumerate() {
                let Some(face) = &vc.face else { continue };
                let id = Identity::of_face(face, &vc.path)?;
                if !seen.take(&id) && !remaining.take(&id) {
                    return err(format!("{}: a card the deck view doesn't count", vc.path));
                }
                // Among cards that look alike, the one that was played this ante as this one
                // shows it was (The Pillar reads it).
                let played = face
                    .ability
                    .get("played_this_ante")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                let from = [AreaId::Deck, AreaId::Hand, AreaId::Discard, AreaId::Play];
                let claimed = self
                    .known
                    .get(&vc.id)
                    .and_then(|&s| self.pool.take_sort(s))
                    .or_else(|| {
                        self.pool.take_identity_where(&id, &from, |c| {
                            c.ability.played_this_ante == played
                        })
                    })
                    .or_else(|| self.pool.take_identity(&id, &from));
                let sort_id = match claimed {
                    Some((_, _, c)) => c.sort_id,
                    None => self.gs.next_sort_id(),
                };
                let like = claimed
                    .map(|(_, _, c)| c)
                    .filter(|c| c.base.is_some() && Identity::of(c) == id);
                held[k][i] = Some(self.face_up(vc, face, area, sort_id, like.as_ref())?);
            }
        }

        // The discard pile's cards the player saw: the seen ones left.
        let mut discard = Vec::new();
        for id in seen.drain() {
            let c = match self.pool.take_identity(
                &id,
                &[AreaId::Discard, AreaId::Hand, AreaId::Play, AreaId::Deck],
            ) {
                Some((_, _, c)) => c,
                None => self.new_playing_card(&id),
            };
            discard.push(Card {
                wheel_flipped: false,
                ..c
            });
        }
        let Some(flipped) = discard_size.checked_sub(discard.len()) else {
            return err(format!(
                "$.discard_size: {discard_size}, below the {} cards the deck view shows there",
                discard.len()
            ));
        };
        let down: Vec<(usize, usize)> = (0..2)
            .flat_map(|k| {
                lists[k]
                    .iter()
                    .enumerate()
                    .filter(|(_, vc)| vc.face.is_none())
                    .map(move |(i, _)| (k, i))
            })
            .collect();
        if remaining.len != down.len() + flipped + deck_size {
            return err(format!(
                "$.deck: {} remaining cards for {} face down, {flipped} discarded face down and \
                 a draw pile of {deck_size}",
                remaining.len,
                down.len()
            ));
        }

        // Face down in the hand: a card the base knows keeps its stand-in while that is still
        // among the remaining cards; the others get one of those at random.
        let mut pick = Vec::new();
        for &(k, i) in &down {
            let vc = &lists[k][i];
            match self.known.get(&vc.id).and_then(|&s| self.pool.take_sort(s)) {
                Some((_, _, c)) if c.base.is_some() && remaining.take(&Identity::of(&c)) => {
                    held[k][i] = Some(c);
                }
                Some((_, _, c)) => pick.push((k, i, Some(c.sort_id))),
                None => pick.push((k, i, None)),
            }
        }
        for (k, i, sort_id) in pick {
            let id = remaining.take_random(&mut self.gs.rng);
            held[k][i] = Some(match sort_id {
                Some(s) => {
                    self.made.push(s);
                    self.playing_card(&id, s)
                }
                None => match self
                    .pool
                    .take_identity(&id, &[AreaId::Deck, AreaId::Discard, AreaId::Hand])
                {
                    Some((_, _, c)) => c,
                    None => self.new_playing_card(&id),
                },
            });
        }
        let mut cards: [Vec<Card>; 2] = [Vec::new(), Vec::new()];
        for k in 0..2 {
            for (i, c) in held[k].iter().enumerate() {
                let vc = &lists[k][i];
                let mut c = c.expect("every hand card built");
                if vc.face.is_none() {
                    c = Builder::face_down(c, vc);
                    c.wheel_flipped = true;
                }
                self.ids.insert(vc.id, c.sort_id);
                cards[k].push(c);
            }
        }

        // Discarded face down: the base's own such cards while they're still remaining, then
        // stand-ins.
        let mut down_discards = Vec::new();
        while down_discards.len() < flipped {
            let Some((_, _, c)) = self.pool.take_where(&[AreaId::Discard], |c| {
                c.wheel_flipped && c.base.is_some() && remaining.contains(&Identity::of(c))
            }) else {
                break;
            };
            remaining.take(&Identity::of(&c));
            down_discards.push(c);
        }
        while down_discards.len() < flipped {
            let id = remaining.take_random(&mut self.gs.rng);
            let c = match self
                .pool
                .take_identity(&id, &[AreaId::Deck, AreaId::Discard, AreaId::Hand])
            {
                Some((_, _, c)) => c,
                None => self.new_playing_card(&id),
            };
            down_discards.push(c);
        }
        discard.extend(down_discards.into_iter().map(|c| Card {
            wheel_flipped: true,
            ..c
        }));

        // The draw pile: the remaining cards left. The base's draw pile keeps its order; the
        // cards that join it go in at random places (or, for a new order, everything is
        // shuffled).
        let mut kept = Vec::new();
        let mut joining = Vec::new();
        for id in remaining.drain() {
            match self.pool.take_identity(&id, &[AreaId::Deck]) {
                Some((i, _, c)) => kept.push((i, c)),
                None => joining.push(
                    match self
                        .pool
                        .take_identity(&id, &[AreaId::Discard, AreaId::Hand, AreaId::Play])
                    {
                        Some((_, _, c)) => c,
                        None => self.new_playing_card(&id),
                    },
                ),
            }
        }
        kept.sort_by_key(|&(i, _)| i);
        let mut deck: Vec<Card> = kept.into_iter().map(|(_, c)| c).collect();
        if self.reshuffle {
            deck.extend(joining);
            shuffle(&mut deck, self.gs.rng.seed(STREAM));
        } else {
            for c in joining {
                let at = element_index(deck.len() + 1, self.gs.rng.seed(STREAM));
                deck.insert(at, c);
            }
        }
        for c in deck.iter_mut().chain(discard.iter_mut()) {
            c.facing = Facing::Back;
            c.juice = false;
            c.ability.forced_selection = false;
        }
        for c in &mut deck {
            c.wheel_flipped = false;
        }
        let [hand, play] = cards;
        Ok([hand, play, discard, deck])
    }

    // -- what the base's rolls left behind -------------------------------------------------

    /// The sets that remember the shadow's rolls, set to what the view's outcomes leave: the
    /// cards the base had that the view doesn't (its own draws and rolls) never existed, and
    /// the view's cards did.
    fn reconcile(&mut self, before: &Before) {
        use jackdaw_data::centers::{J_CAVENDISH, J_GROS_MICHEL};
        let gs = &mut self.gs;
        let present: Vec<(AreaId, CenterId)> = AreaId::NOT_PLAYING
            .iter()
            .flat_map(|&a| gs.areas.get(a).iter().map(move |c| (a, c.center)))
            .collect();
        let held: HashSet<&str> = present
            .iter()
            .filter(|(a, _)| matches!(a, AreaId::Jokers | AreaId::Consumables))
            .map(|(_, c)| c.get().name)
            .collect();
        // The base's cards the view has none of (by center: a face-down joker's sort id may
        // have gone to another of the same look) never existed, or left as the view shows.
        let mut left: HashMap<CenterId, usize> = HashMap::new();
        for &(_, c) in &present {
            *left.entry(c).or_insert(0) += 1;
        }
        let mut dropped: Vec<(AreaId, CenterId)> = Vec::new();
        for (a, c) in self.pool.base() {
            if is_playing(c.center) {
                continue;
            }
            match left.get_mut(&c.center) {
                Some(n) if *n > 0 => *n -= 1,
                _ => dropped.push((a, c.center)),
            }
        }
        // `used_jokers` (card.lua:349-354, 4733-4750): a card leaves it when removed unless one
        // of its name is still held; every card that exists was marked when made.
        for &(_, c) in &dropped {
            if !held.contains(c.get().name) {
                for &x in jackdaw_data::same_name_centers(c) {
                    gs.sets.used_jokers.remove(x);
                }
            }
        }
        let dropped_gros = dropped
            .iter()
            .any(|&(a, c)| a == AreaId::Jokers && c == J_GROS_MICHEL);
        for &(_, c) in &present {
            for &x in jackdaw_data::same_name_centers(c) {
                gs.sets.used_jokers.insert(x);
            }
        }
        // The jokers in the deck are the view's: the per-round hands and discards follow them
        // (where the base's rolls debuffed or destroyed another joker than the run's).
        let (h0, d0) = per_round(&before.jokers);
        let (h1, d1) = per_round(&gs.areas.jokers().to_vec());
        gs.round_resets.hands += h1 - h0;
        gs.round_resets.discards += d1 - d0;
        // Gros Michel's extinction (card.lua:2905-2925) opens the pool to Cavendish.
        let flag = "gros_michel_extinct";
        if present.iter().any(|&(_, c)| c == J_CAVENDISH)
            || (dropped_gros && !present.iter().any(|&(_, c)| c == J_GROS_MICHEL))
        {
            gs.sets
                .pool_flags
                .insert(flag.to_owned(), LuaValue::Bool(true));
        } else if present
            .iter()
            .any(|&(a, c)| a == AreaId::Jokers && c == J_GROS_MICHEL)
        {
            gs.sets.pool_flags.remove(flag);
        }
        // The boss the view shows is the one the run rolled (`get_new_boss`, which counts it).
        let boss = gs.round_resets.blind_choice(BlindSlot::Boss);
        if let (Some(was), Some(is)) = (before.boss, boss)
            && was != is
        {
            // A base in this ante rolled its own boss there, which the run never met.
            if before.ante == gs.round_resets.ante
                && let Some(n) = gs.sets.bosses_used.get(was)
            {
                gs.sets.bosses_used.set(was, (n - 1.0).max(0.0));
            }
            let n = gs.sets.bosses_used.get(is).unwrap_or(0.0);
            gs.sets.bosses_used.set(is, n + 1.0);
        }
        // The ante's voucher shows in the shop, first among its vouchers (UI_definitions.lua:
        // 690-702), until a voucher is redeemed; a base in this ante knows if one was.
        if self.view.get("shop").is_some_and(|s| !s.is_null())
            && (before.voucher.is_some() || before.ante != gs.round_resets.ante)
        {
            gs.current_round.voucher = gs.areas.shop_vouchers().first().map(|c| c.center);
        }
        // A pack's dealt hand.
        if let Some(pack) = gs.pack.as_mut()
            && matches!(pack.kind.as_str(), "Arcana" | "Spectral")
        {
            let mut ids: Vec<u32> = gs.areas.hand().iter().map(|c| c.sort_id).collect();
            ids.sort_unstable();
            pack.dealt = ids;
        }
    }

    /// A base whose round didn't end where the run's did (its rolls scored the last hand
    /// otherwise): what `end_round` does that the view doesn't show (state_events.lua:251-283),
    /// and after a boss, the next ante's voucher and the end of the ante's plays. Returns
    /// whether the ante's voucher needs rolling.
    fn catch_up(&mut self, before: &Before) -> bool {
        let behind = matches!(before.phase, Phase::SelectingHand | Phase::GameOver)
            && matches!(
                self.gs.phase,
                Phase::RoundEval | Phase::Shop | Phase::BlindSelect | Phase::PackOpening
            );
        if !behind {
            return false;
        }
        let gs = &mut self.gs;
        gs.round_resets.temp_handsize = None;
        gs.round_resets.temp_reroll_cost = None;
        let boss = before
            .blind
            .as_ref()
            .is_some_and(|b| b.boss && b.key.is_some());
        if boss {
            for a in AreaId::PLAYING {
                gs.areas.get_mut(a).for_each(|c| {
                    if c.facing == Facing::Back || !matches!(a, AreaId::Hand | AreaId::Play) {
                        c.ability.played_this_ante = false;
                    }
                });
            }
        }
        boss && self.view.get("shop").is_none_or(Value::is_null)
    }

    /// Without a shadow: what the view's vouchers and jokers did to the counters the view
    /// doesn't show (`Card:apply_to_run`, card.lua:1786-1874; `Card:add_to_deck`, 564-643),
    /// past what the new run's own did, and what the visible counters say of the rest.
    fn hidden_counters(&mut self, before: &Before) {
        let gs = &mut self.gs;
        let rr = &mut gs.round_resets;
        rr.blind_ante = rr.ante;
        for v in gs.sets.used_vouchers.iter() {
            if before.vouchers.contains(v) {
                continue;
            }
            let x = v.get().config_num("extra").unwrap_or(0.0);
            match v.get().name {
                "Overstock" | "Overstock Plus" => gs.shop_joker_max += 1.0,
                "Tarot Merchant" | "Tarot Tycoon" => gs.tarot_rate = gs.tarot_rate.max(4.0 * x),
                "Planet Merchant" | "Planet Tycoon" => {
                    gs.planet_rate = gs.planet_rate.max(4.0 * x);
                }
                "Hone" | "Glow Up" => gs.edition_rate = gs.edition_rate.max(x),
                "Magic Trick" | "Illusion" => gs.playing_card_rate = gs.playing_card_rate.max(x),
                "Reroll Surplus" | "Reroll Glut" => rr.reroll_cost -= x,
                "Grabber" | "Nacho Tong" => rr.hands += x,
                "Wasteful" | "Recyclomancy" => rr.discards += x,
                "Hieroglyph" => rr.hands -= x,
                "Petroglyph" => rr.discards -= x,
                _ => {}
            }
        }
        gs.current_round.reroll_cost_increase = if gs.current_round.free_rerolls > 0.0 {
            0.0
        } else {
            (gs.current_round.reroll_cost - rr.temp_reroll_cost.unwrap_or(rr.reroll_cost)).max(0.0)
        };
        // The first shop's Buffoon pack is past once a shop has opened (`first_shop_buffoon`).
        gs.first_shop_buffoon |=
            gs.round >= 2.0 || self.view.get("shop").is_some_and(|s| !s.is_null());
    }
}

/// What the jokers in the deck add to the hands and discards each round (`Card:add_to_deck`,
/// card.lua:564-643; `remove_from_deck` takes it back): Drunkard's and Merry Andy's discards,
/// Troubadour's hands.
fn per_round(jokers: &[Card]) -> (f64, f64) {
    let (mut hands, mut discards) = (0.0, 0.0);
    for c in jokers.iter().filter(|c| c.added_to_deck) {
        if c.ability.d_size > 0.0 {
            discards += c.ability.d_size;
        }
        if c.name() == "Troubadour" {
            hands += c
                .ability
                .extra
                .table()
                .and_then(|t| t.num("h_plays"))
                .unwrap_or(0.0);
        }
    }
    (hands, discards)
}

/// Whether the card is in the hand or the play area, where a face-up card shows its debuff.
fn hand_or_play(game: &Game, id: u32) -> bool {
    matches!(
        game.gs.areas.locate(id),
        Some((AreaId::Hand | AreaId::Play, _))
    )
}

/// A new run of the view's back, stake and challenge.
fn new_run(back: CenterId, stake: f64, challenge: Option<&str>, seed: &str) -> Result<Game> {
    let challenge = match challenge {
        Some(id) => Some(
            jackdaw_data::challenge_by_id(id)
                .ok_or_else(|| RebuildError(format!("$.run.challenge: unknown {id:?}")))?
                .id,
        ),
        None => None,
    };
    if !(1.0..=8.0).contains(&stake) || stake.fract() != 0.0 {
        return err(format!("$.run.stake: {stake} is not 1 to 8"));
    }
    Game::new_run(&RunSpec {
        back,
        stake: stake as u8,
        seed: seed.to_owned(),
        challenge,
    })
    .map_err(RebuildError)
}

/// A game whose every field the player sees is `view`'s ([`crate::player_view`]), and whose
/// hidden parts come from `shadow` (with the view ids it knows) or a new run, or are stand-ins.
/// `seed` gives the game a new RNG, and the draw pile a new order (without a shadow, it is the
/// new run's seed; else the shadow's RNG goes on).
pub fn from_player_view(
    view: &Value,
    shadow: Option<Shadow<'_>>,
    seed: Option<&str>,
) -> Result<Rebuilt> {
    let format = string(view, "format", "$")?;
    if format != FORMAT {
        return err(format!("$.format: {format:?}, not {FORMAT:?}"));
    }
    let run = field(view, "run", "$")?;
    let back = center(string(run, "back", "$.run")?, "$.run.back")?;
    let stake = num(run, "stake", "$.run")?;
    let challenge = opt_str(run, "challenge", "$.run")?;
    let empty = ViewIds::new();
    let (base, known, gone) = match shadow {
        Some(Shadow { game: g, ids, gone }) => {
            let gs = &g.gs;
            if gs.back != back || gs.stake != stake || gs.challenge.as_deref() != challenge {
                return err(format!(
                    "the shadow is another run ({} at stake {}), not {} at stake {stake}",
                    gs.back.key(),
                    gs.stake,
                    back.key()
                ));
            }
            if !g.rt.queue.is_empty() {
                return err("the shadow is in the middle of a step");
            }
            (g.clone(), ids, gone)
        }
        None => (
            new_run(back, stake, challenge, seed.unwrap_or(DEFAULT_SEED))?,
            &empty,
            &[][..],
        ),
    };
    let Game {
        mut gs,
        rt: _,
        trace,
    } = base;
    if shadow.is_some()
        && let Some(s) = seed
    {
        let bound = gs.rng.bound();
        gs.rng = Rng::new(s);
        gs.rng.set_bound(bound);
    }
    let before = Before {
        boss: gs.round_resets.blind_choice(BlindSlot::Boss),
        voucher: gs.current_round.voucher,
        blind: gs.blind.clone(),
        tags: gs.tags.clone(),
        vouchers: gs.sets.used_vouchers,
        jokers: gs.areas.jokers().to_vec(),
        ante: gs.round_resets.ante,
        phase: gs.phase,
    };
    let mut b = Builder {
        view,
        pool: Pool::new(&gs, gone),
        gs,
        known,
        ids: ViewIds::new(),
        shadow: shadow.is_some(),
        reshuffle: shadow.is_none() || seed.is_some(),
        made: Vec::new(),
    };
    b.scalars(&before)?;
    let others = b.non_playing()?;
    let [hand, play, discard, deck] = b.playing()?;
    let mut areas = Areas::default();
    areas.set(AreaId::Deck, deck);
    areas.set(AreaId::Hand, hand);
    areas.set(AreaId::Discard, discard);
    areas.set(AreaId::Play, play);
    for (area, cards) in others {
        areas.set(area, cards);
    }
    b.gs.areas = areas;
    let roll_voucher = b.catch_up(&before);
    b.reconcile(&before);
    if !b.shadow {
        b.hidden_counters(&before);
    }
    let Builder { gs, ids, made, .. } = b;
    // Face-up cards show their debuffs, and the base's cards keep theirs; the boss debuffs the
    // cards made here as it would have (`Blind:debuff_card`, blind.lua:624-652).
    let mut game = Game::from_state(gs);
    if roll_voucher {
        // The boss's defeat rolled the next ante's voucher (state_events.lua:258), which
        // doesn't show before its shop.
        game.gs.current_round.voucher = game.get_next_voucher_key(false);
    }
    for id in made {
        if game
            .gs
            .areas
            .card(id)
            .is_some_and(|c| c.facing == Facing::Back || !hand_or_play(&game, id))
        {
            game.debuff_card(id, true);
        }
    }
    game.trace.enabled = trace.enabled;
    Ok(Rebuilt { game, ids })
}

// ---------------------------------------------------------------------------
// Comparing views
// ---------------------------------------------------------------------------

/// `view` with its card ids replaced by the order they first appear in, so that two views of
/// the same state under different keys (or of a game and its rebuild) compare equal.
pub fn without_ids(view: &Value) -> Value {
    fn walk(v: &Value, ids: &mut HashMap<u64, usize>, key: Option<&str>) -> Value {
        match v {
            Value::Object(m) => Value::Object(
                m.iter()
                    .map(|(k, x)| (k.clone(), walk(x, ids, Some(k))))
                    .collect::<Map<_, _>>(),
            ),
            Value::Array(items) => Value::Array(items.iter().map(|x| walk(x, ids, key)).collect()),
            Value::Number(n)
                if matches!(key, Some("id" | "card" | "target" | "played" | "scoring")) =>
            {
                let id = n.as_u64().unwrap_or(u64::MAX);
                let next = ids.len();
                Value::from(*ids.entry(id).or_insert(next))
            }
            other => other.clone(),
        }
    }
    walk(view, &mut HashMap::new(), None)
}

/// Whether two views show the same thing, card ids aside ([`without_ids`]).
pub fn same_view(a: &Value, b: &Value) -> bool {
    without_ids(a) == without_ids(b)
}

impl Game {
    /// A game built from the player's view ([`from_player_view`]).
    pub fn from_player_view(
        view: &Value,
        shadow: Option<Shadow<'_>>,
        seed: Option<&str>,
    ) -> Result<Rebuilt> {
        from_player_view(view, shadow, seed)
    }
}

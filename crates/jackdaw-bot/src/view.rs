//! The gamestate view: what BalatroBot's `gamestate` endpoint (and every gameplay endpoint's
//! response) reports.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/utils/gamestate.lua`, field for field. Where it reads a live object, the port reads
//! the engine's state:
//!
//! - **Structure exact.** A field the Lua leaves `nil` is absent (`modifier.eternal` only when
//!   true, `seed` only when set), an empty table is sent as `[]` (see [`crate::json`]), every
//!   number is a Lua number.
//! - **Names only.** The text fields BalatroBot fills from localized UI text carry names
//!   instead: a card's `value.effect` is its `ability.name`, a boss's `effect` its name, a tag's
//!   `tag_effect` the tag's name. Fields that are empty in the live game stay empty (the Small
//!   and Big blinds' `effect`, every `used_vouchers` description).
//! - **Areas.** Every area carries its `config.highlighted_limit` (the hand and the deck 5, the
//!   others 1, cardarea.lua:18) and its vanilla `config.card_limit`: joker and consumable slots,
//!   the hand size, the deck's limit ([`jackdaw_core::game::AreaMemo`]), the shop's
//!   `joker_max`, its voucher slots and 2 boosters, and the pack's size.
//! - **The shop's areas outlive the shop.** `toggle_shop` removes `G.shop`, but `G.shop_jokers`,
//!   `G.shop_vouchers` and `G.shop_booster` still point at the emptied areas, which this view
//!   (unlike `G.pack_cards`, checked for `REMOVED`) keeps reporting until the run is deleted:
//!   `cards` `[]` with the card counts of their last frame.
//! - **Blind scores** are always `floor(get_blind_amount(ante) * mult * ante_scaling)`, as the
//!   Lua computes them, whatever the current blind's actual target is.
//! - **The menu.** Before `start` (and after `menu`) the game holds a fresh
//!   `Game:init_game_object()` with the Red Deck selected and no card areas.
//! - **States.** A pack is `SMODS_BOOSTER_OPENED` (Steamodded's single booster state); after a
//!   failed `load` has deleted the run the state is `UNKNOWN` (`G.STATE = -1`).

use jackdaw_core::area::AreaId;
use jackdaw_core::card::{Card, EditionKind, Facing, Seal};
use jackdaw_core::game::Game;
use jackdaw_core::hand_eval::HandType;
use jackdaw_core::state::{BlindSlot, GameState};
use jackdaw_data::{BLINDS, BlindId, CenterSet, HANDS, TagId};

use crate::json::Json;
use crate::states::GState;

/// `DECK_KEY_TO_NAME` (gamestate.lua:37-53): the Challenge Deck has no entry.
pub fn deck_name(back_key: &str) -> Option<&'static str> {
    Some(match back_key {
        "b_red" => "RED",
        "b_blue" => "BLUE",
        "b_yellow" => "YELLOW",
        "b_green" => "GREEN",
        "b_black" => "BLACK",
        "b_magic" => "MAGIC",
        "b_nebula" => "NEBULA",
        "b_ghost" => "GHOST",
        "b_abandoned" => "ABANDONED",
        "b_checkered" => "CHECKERED",
        "b_zodiac" => "ZODIAC",
        "b_painted" => "PAINTED",
        "b_anaglyph" => "ANAGLYPH",
        "b_plasma" => "PLASMA",
        "b_erratic" => "ERRATIC",
        _ => return None,
    })
}

/// `STAKE_LEVEL_TO_NAME` (gamestate.lua:66-75).
pub fn stake_name(stake: f64) -> Option<&'static str> {
    const NAMES: [&str; 8] = [
        "WHITE", "RED", "GREEN", "BLACK", "BLUE", "PURPLE", "ORANGE", "GOLD",
    ];
    if stake.fract() != 0.0 || !(1.0..=8.0).contains(&stake) {
        return None;
    }
    Some(NAMES[stake as usize - 1])
}

fn num(x: f64) -> Json {
    Json::Num(x)
}

fn count(n: usize) -> Json {
    Json::Num(n as f64)
}

// ---------------------------------------------------------------------------
// Cards (gamestate.lua:174-370)
// ---------------------------------------------------------------------------

/// `card.label` as `Card:set_ability` sets it (card.lua:339-346): the center's label (playing
/// cards: `"Base Card"`, `"Glass Card"`, ...), else the set's name; jokers, editions,
/// consumables, vouchers and boosters use their name.
pub fn card_label(card: &Card) -> String {
    let center = card.center();
    let named = matches!(
        center.set,
        CenterSet::Joker | CenterSet::Edition | CenterSet::Voucher | CenterSet::Booster
    ) || center.consumeable;
    if named {
        return center.name.to_owned();
    }
    center
        .label
        .map_or_else(|| center.set.name().to_owned(), str::to_owned)
}

/// The set enum (gamestate.lua:327-348).
pub fn card_set_name(card: &Card) -> &'static str {
    match card.set() {
        CenterSet::Joker => "JOKER",
        CenterSet::Tarot => "TAROT",
        CenterSet::Planet => "PLANET",
        CenterSet::Spectral => "SPECTRAL",
        CenterSet::Voucher => "VOUCHER",
        CenterSet::Booster => "BOOSTER",
        CenterSet::Edition => "EDITION",
        _ => match card.effect() {
            Some(e) if e != "Base" => "ENHANCED",
            _ => "DEFAULT",
        },
    }
}

/// `string.upper(card.ability.effect:gsub(" Card", ""))` for any card whose `ability.effect`
/// is set and isn't `"Base"` (gamestate.lua:241-243): jokers and consumables report theirs too.
fn enhancement(card: &Card) -> Option<String> {
    match card.effect() {
        Some(e) if e != "Base" => Some(e.replace(" Card", "").to_ascii_uppercase()),
        _ => None,
    }
}

fn seal_name(seal: Seal) -> &'static str {
    match seal {
        Seal::Gold => "GOLD",
        Seal::Red => "RED",
        Seal::Blue => "BLUE",
        Seal::Purple => "PURPLE",
    }
}

fn edition_name(kind: EditionKind) -> &'static str {
    match kind {
        EditionKind::Foil => "FOIL",
        EditionKind::Holo => "HOLO",
        EditionKind::Polychrome => "POLYCHROME",
        EditionKind::Negative => "NEGATIVE",
    }
}

/// `convert_rank_to_enum` (gamestate.lua:193-218).
fn rank_letter(card: &Card) -> Option<String> {
    card.base.map(|b| b.value.letter().to_string())
}

/// `extract_card(card)` (gamestate.lua:326-370) for a card of `area`.
fn card(game: &Game, c: &Card, area: AreaId) -> Json {
    let key = match &c.base {
        Some(b) => b.front.get().key.to_owned(),
        None => c.center.key().to_owned(),
    };
    let mut value = Vec::new();
    if let Some(b) = &c.base {
        value.push(("suit", Json::str(b.suit.letter().to_string())));
    }
    if let Some(r) = rank_letter(c) {
        value.push(("rank", Json::str(r)));
    }
    value.push(("effect", Json::str(c.name())));
    let mut modifier = Vec::new();
    if let Some(s) = c.seal {
        modifier.push(("seal", Json::str(seal_name(s))));
    }
    if let Some(e) = c.edition {
        modifier.push(("edition", Json::str(edition_name(e.kind))));
    }
    if let Some(e) = enhancement(c) {
        modifier.push(("enhancement", Json::str(e)));
    }
    if c.ability.eternal {
        modifier.push(("eternal", Json::Bool(true)));
    }
    if let Some(t) = c.ability.perish_tally
        && t > 0.0
    {
        modifier.push(("perishable", num(t)));
    }
    if c.ability.rental {
        modifier.push(("rental", Json::Bool(true)));
    }
    let mut state = Vec::new();
    if c.debuff {
        state.push(("debuff", Json::Bool(true)));
    }
    // `CardArea:align_cards` turns every card of the deck and the discard pile face down each
    // frame (cardarea.lua:413-430); the snapshot doesn't keep their facing.
    if c.facing == Facing::Back || area.face_down() {
        state.push(("hidden", Json::Bool(true)));
    }
    if area == AreaId::Hand && game.rt.highlighted.contains(&c.sort_id) {
        state.push(("highlight", Json::Bool(true)));
    }
    Json::obj([
        ("id", num(f64::from(c.sort_id))),
        ("key", Json::str(key)),
        ("set", Json::str(card_set_name(c))),
        ("label", Json::str(card_label(c))),
        ("value", Json::obj(value)),
        ("modifier", Json::obj(modifier)),
        ("state", Json::obj(state)),
        (
            "cost",
            Json::obj([("sell", num(c.sell_cost)), ("buy", num(c.cost))]),
        ),
    ])
}

/// `extract_area(area)` (gamestate.lua:379-403) for a live area.
fn area(game: &Game, id: AreaId, limit: f64, highlighted_limit: f64) -> Json {
    let cards = game.gs.areas.get(id);
    Json::obj([
        ("count", count(cards.len())),
        ("limit", num(limit)),
        (
            "cards",
            Json::Arr(cards.iter().map(|c| card(game, c, id)).collect()),
        ),
        ("highlighted_limit", num(highlighted_limit)),
    ])
}

/// `extract_area` for a removed area: `cards` is gone, the last frame's count stays.
fn stale_area((count, limit): (f64, f64)) -> Json {
    Json::obj([
        ("count", num(count)),
        ("limit", num(limit)),
        ("cards", Json::Arr(Vec::new())),
        ("highlighted_limit", num(1.0)),
    ])
}

// ---------------------------------------------------------------------------
// Hands, round, blinds (gamestate.lua:409-712)
// ---------------------------------------------------------------------------

/// `hand.example` from `Game:init_game_object` (`{card key, scores}` pairs).
fn example(h: HandType) -> Json {
    Json::Arr(
        HANDS[h.index()]
            .example
            .iter()
            .map(|&(k, s)| Json::Arr(vec![Json::str(k), Json::Bool(s)]))
            .collect(),
    )
}

/// `extract_hand_info(G.GAME.hands)`: all twelve hands, visible or not.
fn hands(rows: impl Fn(HandType) -> (f64, f64, f64, f64, f64)) -> Json {
    Json::Obj(
        HandType::ALL
            .into_iter()
            .map(|h| {
                let (level, chips, mult, played, played_this_round) = rows(h);
                let row = Json::obj([
                    ("order", num(f64::from(HANDS[h.index()].order))),
                    ("level", num(level)),
                    ("chips", num(chips)),
                    ("mult", num(mult)),
                    ("played", num(played)),
                    ("played_this_round", num(played_this_round)),
                    ("example", example(h)),
                ]);
                (h.name().to_owned(), row)
            })
            .collect(),
    )
}

/// `convert_status_to_enum` (gamestate.lua:576-590).
fn status(s: &str) -> &'static str {
    match s {
        "Defeated" => "DEFEATED",
        "Skipped" => "SKIPPED",
        "Current" => "CURRENT",
        "Select" => "SELECT",
        _ => "UPCOMING",
    }
}

/// What `gamestate.get_blinds_info` reads from `G.GAME`.
struct BlindInputs<'a> {
    ante: f64,
    scaling: f64,
    ante_scaling: f64,
    choices: [Option<BlindId>; 3],
    states: [&'a str; 3],
    tags: Option<[Option<TagId>; 2]>,
}

/// `gamestate.get_blinds_info()` (gamestate.lua:594-712).
fn blinds(b: &BlindInputs) -> Json {
    let base_amount = jackdaw_data::blind_amount(b.scaling as u8, b.ante as i64);
    let score = |mult: f64| num((base_amount * mult * b.ante_scaling).floor());
    let mut out = Vec::new();
    for slot in BlindSlot::ALL {
        let ty = match slot {
            BlindSlot::Small => "SMALL",
            BlindSlot::Big => "BIG",
            BlindSlot::Boss => "BOSS",
        };
        let choice = match slot {
            BlindSlot::Small => b.choices[0].or(jackdaw_data::blind_by_key("bl_small")),
            BlindSlot::Big => b.choices[1].or(jackdaw_data::blind_by_key("bl_big")),
            BlindSlot::Boss => b.choices[2],
        };
        let mut effect = String::new();
        let mut st = "UPCOMING";
        let mut tag_name = String::new();
        let mut tag_effect = String::new();
        let (name, score_v) = match choice {
            Some(id) => {
                let proto = &BLINDS[id.0 as usize];
                // Names only: a boss's localized effect text becomes its name; the Small and
                // Big blinds have none (gamestate.lua:487-489).
                if proto.boss.is_some() && !matches!(proto.key, "bl_small" | "bl_big") {
                    effect = proto.name.to_owned();
                }
                if !b.states[slot.index()].is_empty() {
                    st = status(b.states[slot.index()]);
                }
                if slot != BlindSlot::Boss
                    && let Some(tags) = b.tags
                    && let Some(tag) = tags[slot.index()]
                {
                    tag_name = tag.get().name.to_owned();
                    tag_effect = tag.get().name.to_owned();
                }
                (proto.name.to_owned(), score(proto.mult))
            }
            // The boss fallback before the first boss is rolled (gamestate.lua:703-707).
            None => ("Boss Blind".to_owned(), score(2.0)),
        };
        let key = match slot {
            BlindSlot::Small => "small",
            BlindSlot::Big => "big",
            BlindSlot::Boss => "boss",
        };
        out.push((
            key,
            Json::obj([
                ("type", Json::str(ty)),
                ("status", Json::str(st)),
                ("name", Json::Str(name)),
                ("effect", Json::Str(effect)),
                ("score", score_v),
                ("tag_name", Json::Str(tag_name)),
                ("tag_effect", Json::Str(tag_effect)),
            ]),
        ));
    }
    Json::obj(out)
}

// ---------------------------------------------------------------------------
// The whole view (gamestate.lua:720-815)
// ---------------------------------------------------------------------------

/// What the session holds, as the view sees it.
#[derive(Clone, Copy)]
pub enum Source<'a> {
    /// The main menu: `G.GAME` is a fresh `init_game_object()`, no card areas.
    Menu,
    /// A run in progress.
    Run(&'a Game),
    /// A run deleted by a failed `load` (`G:delete_run()` ran, `G.STATE = -1`): `G.GAME` is the
    /// deleted run's (with `won = false`) and the card areas are gone.
    Deleted(&'a Game),
    /// A failed `load` from the main menu: `G.STATE = -1` with the menu's `G.GAME`.
    DeletedMenu,
}

/// The state name the view reports (`get_state_name(G.STATE)`, gamestate.lua:19-31).
pub fn state_name(src: Source) -> &'static str {
    match src {
        Source::Menu => GState::Menu.name(),
        Source::Run(g) => GState::of(g).name(),
        Source::Deleted(_) | Source::DeletedMenu => "UNKNOWN",
    }
}

/// `gamestate.get_gamestate()`.
pub fn gamestate(src: Source) -> Json {
    let mut out = vec![("state", Json::str(state_name(src)))];
    let (gs, game): (Option<&GameState>, Option<&Game>) = match src {
        Source::Menu | Source::DeletedMenu => (None, None),
        Source::Run(g) => (Some(&g.gs), Some(g)),
        Source::Deleted(g) => (Some(&g.gs), None),
    };
    match gs {
        Some(gs) => run_fields(gs, matches!(src, Source::Deleted(_)), &mut out),
        None => menu_fields(&mut out),
    }
    if let Some(g) = game {
        area_fields(g, &mut out);
    }
    Json::obj(out)
}

/// The `G.GAME` fields of a run.
fn run_fields(gs: &GameState, deleted: bool, out: &mut Vec<(&'static str, Json)>) {
    out.push(("round_num", num(gs.round)));
    out.push(("ante_num", num(gs.round_resets.ante)));
    out.push(("money", num(gs.dollars)));
    // `G:delete_run()` resets `G.GAME.won` (game.lua:1188).
    out.push(("won", Json::Bool(gs.won && !deleted)));
    if let Some(d) = deck_name(gs.back.key()) {
        out.push(("deck", Json::str(d)));
    }
    if let Some(s) = stake_name(gs.stake) {
        out.push(("stake", Json::str(s)));
    }
    out.push(("seed", Json::str(gs.rng.seed_str())));
    out.push((
        "used_vouchers",
        Json::Obj(
            gs.sets
                .used_vouchers
                .sorted_keys()
                .into_iter()
                .map(|k| (k.to_owned(), Json::str("")))
                .collect(),
        ),
    ));
    out.push((
        "hands",
        hands(|h| {
            let s = gs.hand(h);
            (s.level, s.chips, s.mult, s.played, s.played_this_round)
        }),
    ));
    let cr = &gs.current_round;
    out.push((
        "round",
        Json::obj([
            ("hands_left", num(cr.hands_left)),
            ("hands_played", num(cr.hands_played)),
            ("discards_left", num(cr.discards_left)),
            ("discards_used", num(cr.discards_used)),
            ("reroll_cost", num(cr.reroll_cost)),
            ("chips", num(gs.chips)),
        ]),
    ));
    let rr = &gs.round_resets;
    let scaling = match gs.modifier("scaling") {
        Some(v) => v.as_num().unwrap_or(1.0),
        None => 1.0,
    };
    out.push((
        "blinds",
        blinds(&BlindInputs {
            ante: rr.ante,
            scaling,
            ante_scaling: gs.starting_params.ante_scaling,
            choices: rr.blind_choices,
            states: rr.blind_states,
            tags: rr.blind_tags,
        }),
    ));
}

/// The `G.GAME` fields at the main menu: `init_game_object()` with `selected_back` the Red Deck
/// (game.lua:1529, 1862-2016).
fn menu_fields(out: &mut Vec<(&'static str, Json)>) {
    out.push(("round_num", num(0.0)));
    out.push(("ante_num", num(1.0)));
    out.push(("money", num(0.0)));
    out.push(("won", Json::Bool(false)));
    out.push(("deck", Json::str("RED")));
    out.push(("stake", Json::str("WHITE")));
    out.push(("used_vouchers", Json::Obj(Vec::new())));
    out.push((
        "hands",
        hands(|h| {
            let base = &HANDS[h.index()];
            (1.0, base.s_chips, base.s_mult, 0.0, 0.0)
        }),
    ));
    out.push((
        "round",
        Json::obj([
            ("hands_left", num(0.0)),
            ("hands_played", num(0.0)),
            ("discards_left", num(0.0)),
            ("discards_used", num(0.0)),
            ("reroll_cost", num(5.0)),
            ("chips", num(0.0)),
        ]),
    ));
    out.push((
        "blinds",
        blinds(&BlindInputs {
            ante: 1.0,
            scaling: 1.0,
            ante_scaling: 1.0,
            choices: [
                jackdaw_data::blind_by_key("bl_small"),
                jackdaw_data::blind_by_key("bl_big"),
                None,
            ],
            states: ["Select", "Upcoming", "Upcoming"],
            tags: None,
        }),
    ));
}

/// The card areas of a run in progress.
fn area_fields(g: &Game, out: &mut Vec<(&'static str, Json)>) {
    let gs = &g.gs;
    out.push(("jokers", area(g, AreaId::Jokers, gs.joker_slots, 1.0)));
    out.push((
        "consumables",
        area(g, AreaId::Consumables, gs.consumable_slots, 1.0),
    ));
    out.push(("cards", area(g, AreaId::Deck, g.rt.areas.deck_limit, 5.0)));
    out.push(("hand", area(g, AreaId::Hand, g.hand_limit(), 5.0)));
    if g.rt.shop_open {
        out.push(("shop", area(g, AreaId::ShopJokers, gs.shop_joker_max, 1.0)));
        out.push((
            "vouchers",
            area(g, AreaId::ShopVouchers, g.rt.areas.voucher_limit, 1.0),
        ));
        out.push(("packs", area(g, AreaId::ShopBooster, 2.0, 1.0)));
    } else if let Some(stale) = g.rt.areas.stale_shop {
        out.push(("shop", stale_area(stale.jokers)));
        out.push(("vouchers", stale_area(stale.vouchers)));
        out.push(("packs", stale_area(stale.booster)));
    }
    if gs.pack.is_some() {
        out.push(("pack", area(g, AreaId::PackCards, gs.pack_size, 1.0)));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jackdaw_core::run::RunSpec;

    fn new_game(seed: &str) -> Game {
        Game::new_run(&RunSpec {
            back: jackdaw_data::center_by_key("b_red").unwrap(),
            stake: 1,
            seed: seed.to_owned(),
            challenge: None,
        })
        .unwrap()
    }

    #[test]
    fn menu_view() {
        let v = gamestate(Source::Menu);
        assert_eq!(v.get("state"), Some(&Json::str("MENU")));
        assert_eq!(v.get("deck"), Some(&Json::str("RED")));
        assert!(v.get("seed").is_none());
        assert!(v.get("jokers").is_none());
        let boss = v.get("blinds").unwrap().get("boss").unwrap();
        assert_eq!(boss.get("name"), Some(&Json::str("Boss Blind")));
        assert_eq!(boss.get("score"), Some(&Json::Num(600.0)));
        let hands = v.get("hands").unwrap();
        let ff = hands.get("Flush Five").unwrap();
        assert_eq!(ff.get("chips"), Some(&Json::Num(160.0)));
        assert_eq!(
            crate::json::encode(v.get("used_vouchers").unwrap()).unwrap(),
            "[]"
        );
    }

    #[test]
    fn run_view_matches_test123() {
        // BalatroBot's own suite pins TEST123's first ante (test_gamestate.py).
        let g = new_game("TEST123");
        let v = gamestate(Source::Run(&g));
        assert_eq!(v.get("state"), Some(&Json::str("BLIND_SELECT")));
        let blinds = v.get("blinds").unwrap();
        let small = blinds.get("small").unwrap();
        assert_eq!(small.get("score"), Some(&Json::Num(300.0)));
        assert_eq!(small.get("tag_name"), Some(&Json::str("Polychrome Tag")));
        let big = blinds.get("big").unwrap();
        assert_eq!(big.get("tag_name"), Some(&Json::str("Investment Tag")));
        let boss = blinds.get("boss").unwrap();
        assert_eq!(boss.get("name"), Some(&Json::str("The Manacle")));
        assert_eq!(boss.get("score"), Some(&Json::Num(600.0)));
        let cards = v.get("cards").unwrap();
        assert_eq!(cards.get("count"), Some(&Json::Num(52.0)));
        assert_eq!(cards.get("limit"), Some(&Json::Num(52.0)));
        let first = match cards.get("cards") {
            Some(Json::Arr(c)) => &c[0],
            other => panic!("{other:?}"),
        };
        assert_eq!(first.get("label"), Some(&Json::str("Base Card")));
        assert_eq!(first.get("set"), Some(&Json::str("DEFAULT")));
        assert_eq!(
            first.get("state").unwrap().get("hidden"),
            Some(&Json::Bool(true))
        );
        assert!(v.get("shop").is_none());
    }
}

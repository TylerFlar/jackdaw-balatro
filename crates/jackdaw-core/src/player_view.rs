//! The player's view of a run (`jackdaw.player_view/1`): the state a player sees, as structured
//! data, with what they can't see left out. `docs/api.md` documents the format.
//!
//! - **Card ids.** A run's sort ids give the cards' identities away (`Game:start_run` makes the
//!   deck in a fixed order), so the view names each card by an id derived from its sort id, the
//!   times the player has lost track of it ([`Card::view_epoch`]: it went into the deck, or
//!   turned face down) and a key the caller keeps secret. An id stays the same while the player
//!   can follow the card, a face-down card's included, and a card the player loses track of
//!   comes back under a new one.
//! - **Face-down cards** (dealt face down by a boss, Amber Acorn's jokers) keep their id and
//!   their place, and show nothing else: every identity field is `null`. What the game still
//!   shows of them shows: the card's size ([`shape`]: four jokers have their own), whether its
//!   sell button is active ([`crate::legal::can_sell`]: not for an eternal joker), whether it
//!   shakes ([`Card::juice`]) and whether Cerulean Bell forces it.
//! - **The deck** is a multiset, as the deck view shows it (UI_definitions.lua:3233-3275): every
//!   playing card, sorted, with whether it counts as remaining (in the draw pile, or dealt face
//!   down: `wheel_flipped`). No order, no ids.
//! - **No RNG, no future**: no seed, no RNG state, no contents of shops, packs or vouchers not
//!   shown yet, no sort ids.
//! - **The last hand's scoring** ([`Game::last_play`], when recorded), by the same card ids.

use serde_json::{Map, Value, json};
use sha2::{Digest, Sha256};

use crate::area::AreaId;
use crate::card::{Card, Facing, Seal};
use crate::game::Game;
use crate::state::{BlindSlot, GameState, Phase};
use crate::trace::PlayTrace;

/// The format of [`player_view`]'s document.
pub const FORMAT: &str = "jackdaw.player_view/1";

/// The id the view gives card `sort_id` at view epoch `epoch`, under `key`: the first 53 bits
/// (an integer JavaScript keeps exactly) of SHA-256 over the format, the key, the sort id and
/// the epoch. Without the key, it says nothing of the card.
pub fn card_id(key: u64, sort_id: u32, epoch: u32) -> u64 {
    let mut h = Sha256::new();
    h.update(FORMAT.as_bytes());
    h.update(key.to_le_bytes());
    h.update(sort_id.to_le_bytes());
    h.update(epoch.to_le_bytes());
    let d = h.finalize();
    u64::from_le_bytes(d[..8].try_into().expect("8 bytes")) >> 11
}

fn num(x: f64) -> Value {
    crate::snapshot::num(x)
}

/// The size of the four jokers whose card isn't the standard one (`Card:set_ability`,
/// card.lua:237-257; `Card:load`, 4669-4680): Half Joker (height / 1.7), Photograph
/// (height / 1.2), Square Joker (height = width) and Wee Joker (both x 0.7). The back is drawn at
/// the card's size (`set_sprites`, card.lua:213), so it shows face down too. The game applies
/// the sizes once the joker is discovered; the engine's profile has discovered everything.
pub fn shape(c: &Card) -> Option<&'static str> {
    use jackdaw_data::centers::{J_HALF, J_PHOTOGRAPH, J_SQUARE, J_WEE};
    match c.center {
        J_HALF => Some("half"),
        J_PHOTOGRAPH => Some("photograph"),
        J_SQUARE => Some("square"),
        J_WEE => Some("wee"),
        _ => None,
    }
}

/// A card where the player sees it: its id and, face up, what it shows.
fn card(c: &Card, area: AreaId, key: u64) -> Value {
    let id = card_id(key, c.sort_id, c.view_epoch);
    let mut m = Map::new();
    m.insert("id".into(), id.into());
    let face_down = c.facing == Facing::Back;
    m.insert("face_down".into(), face_down.into());
    if face_down {
        for k in [
            "key",
            "front",
            "suit",
            "rank",
            "edition",
            "seal",
            "debuff",
            "cost",
            "sell_cost",
            "stickers",
            "ability",
        ] {
            m.insert(k.into(), Value::Null);
        }
    } else {
        // A Stone Card shows no rank or suit (card.lua:4419).
        let base = c.base.as_ref().filter(|_| !c.is_stone());
        let a = &c.ability;
        m.insert("key".into(), c.center.key().into());
        m.insert("front".into(), json!(base.map(|b| b.front.get().key)));
        m.insert("suit".into(), json!(base.map(|b| b.suit.name())));
        m.insert("rank".into(), json!(base.map(|b| b.value.name())));
        m.insert("edition".into(), json!(c.edition.map(|e| e.kind.name())));
        m.insert("seal".into(), json!(c.seal.map(Seal::name)));
        m.insert("debuff".into(), c.debuff.into());
        m.insert("cost".into(), num(c.cost));
        m.insert("sell_cost".into(), num(c.sell_cost));
        m.insert(
            "stickers".into(),
            json!({
                "eternal": a.eternal,
                "perishable": a.perishable,
                "perish_tally": if a.perishable { crate::snapshot::opt_num(a.perish_tally.get()) } else { Value::Null },
                "rental": a.rental,
                "pinned": c.pinned,
            }),
        );
        m.insert("ability".into(), crate::snapshot::ability_json(c));
    }
    // What shows whichever way the card faces: its size, whether its sell button is active,
    // whether it shakes, and Cerulean Bell's forcing.
    m.insert("shape".into(), json!(shape(c)));
    m.insert("sellable".into(), crate::legal::can_sell(c, area).into());
    m.insert("shaking".into(), c.juice.into());
    m.insert("forced".into(), c.ability.forced_selection.into());
    Value::Object(m)
}

fn cards(gs: &GameState, a: AreaId, key: u64) -> Value {
    Value::Array(gs.areas.get(a).iter().map(|c| card(c, a, key)).collect())
}

/// A playing card as the deck view shows it (face up, wherever it is), and whether it counts as
/// remaining there.
fn deck_entry(c: &Card, remaining: bool) -> (DeckKey, Value) {
    let base = c.base.as_ref().filter(|_| !c.is_stone());
    let entry = json!({
        "key": c.center.key(),
        "front": base.map(|b| b.front.get().key),
        "suit": base.map(|b| b.suit.name()),
        "rank": base.map(|b| b.value.name()),
        "edition": c.edition.map(|e| e.kind.name()),
        "seal": c.seal.map(Seal::name),
        "perma_bonus": num(c.ability.perma_bonus),
        "remaining": remaining,
    });
    let order = DeckKey {
        // The deck view's suit order (UI_definitions.lua:3238-3244).
        suit: base.map_or(4, |b| match b.suit.name() {
            "Spades" => 0,
            "Hearts" => 1,
            "Clubs" => 2,
            _ => 3,
        }),
        rank: std::cmp::Reverse(base.map_or(0, |b| b.value.id())),
        rest: entry.to_string(),
    };
    (order, entry)
}

/// The deck view's order: by suit, then rank from the Ace down, Stones last; ties by the entry.
#[derive(PartialEq, Eq, PartialOrd, Ord)]
struct DeckKey {
    suit: u8,
    rank: std::cmp::Reverse<u8>,
    rest: String,
}

/// The deck view: every playing card, with whether it counts as remaining
/// (UI_definitions.lua:3264: in the draw pile, or dealt face down).
fn deck(gs: &GameState) -> Value {
    let mut entries: Vec<(DeckKey, Value)> = Vec::new();
    for a in AreaId::PLAYING {
        for c in gs.areas.get(a).iter() {
            entries.push(deck_entry(c, a == AreaId::Deck || c.wheel_flipped));
        }
    }
    entries.sort_by(|x, y| x.0.cmp(&y.0));
    Value::Array(entries.into_iter().map(|(_, v)| v).collect())
}

/// The blind being played: what its chip and money badges and its description show.
fn blind(gs: &GameState) -> Value {
    let Some(b) = gs.blind.as_ref() else {
        return Value::Null;
    };
    let Some(k) = b.key else {
        return Value::Null;
    };
    let mut eye: Vec<&str> = b.eye_hands.iter().map(|h| h.name()).collect();
    eye.sort_unstable();
    json!({
        "key": k.key(),
        "name": k.get().name,
        "chips": num(b.chips),
        "mult": num(b.mult),
        "dollars": num(b.dollars),
        "boss": b.boss,
        "disabled": b.disabled,
        "eye_hands": eye,
        "mouth_hand": b.mouth_hand.map(|h| h.name()),
    })
}

/// The ante's three blinds, as blind select shows them: each one's state, its score to beat, and
/// the Small and Big blinds' skip tags (with the hand an Orbital Tag levels).
fn blinds(game: &Game) -> Value {
    let gs = &game.gs;
    let rr = &gs.round_resets;
    let amount = crate::rules::get_blind_amount(game, rr.ante);
    let ante_scaling = gs.starting_params.ante_scaling;
    let orbital = gs.orbital_choices.get(&rr.ante.to_string());
    let mut out = Map::new();
    for slot in BlindSlot::ALL {
        let choice = rr.blind_choice(slot).or(match slot {
            BlindSlot::Small => jackdaw_data::blind_by_key("bl_small"),
            BlindSlot::Big => jackdaw_data::blind_by_key("bl_big"),
            BlindSlot::Boss => None,
        });
        let tag = match slot {
            BlindSlot::Boss => None,
            _ => rr.blind_tags.and_then(|t| t[slot.index()]),
        };
        let orbital_hand = tag
            .filter(|t| t.key() == "tag_orbital")
            .and(orbital)
            .map(|hands| hands[slot.index()].name());
        out.insert(
            slot.name().to_owned(),
            json!({
                "key": choice.map(|b| b.key()),
                "state": rr.blind_state(slot),
                "chips": choice.map(|b| num((amount * b.get().mult * ante_scaling).floor())),
                "tag": tag.map(|t| t.key()),
                "orbital_hand": orbital_hand,
            }),
        );
    }
    Value::Object(out)
}

/// The shop is open: in the shop, or in a pack opened there.
fn shop_open(gs: &GameState) -> bool {
    gs.phase == Phase::Shop
        || (gs.phase == Phase::PackOpening
            && gs
                .pack
                .as_ref()
                .is_some_and(|p| p.return_phase == Phase::Shop))
}

/// The cards the four jokers that name one each round show, while the player can see one of
/// them face up (in the jokers, the shop or a pack): The Idol, Mail-In Rebate, Ancient Joker and
/// Castle.
fn round_cards(gs: &GameState, out: &mut Map<String, Value>) {
    let cr = &gs.current_round;
    let shown = |name: &str| {
        [AreaId::Jokers, AreaId::ShopJokers, AreaId::PackCards]
            .iter()
            .any(|&a| {
                gs.areas
                    .get(a)
                    .iter()
                    .any(|c| c.facing == Facing::Front && c.name() == name)
            })
    };
    if shown("The Idol") {
        out.insert(
            "idol_card".into(),
            json!({"suit": cr.idol_card.0.name(), "rank": cr.idol_card.1.name()}),
        );
    }
    if shown("Mail-In Rebate") {
        out.insert("mail_card".into(), json!({"rank": cr.mail_card.name()}));
    }
    if shown("Ancient Joker") {
        out.insert(
            "ancient_card".into(),
            json!({"suit": cr.ancient_card.map(|s| s.name())}),
        );
    }
    if shown("Castle") {
        out.insert("castle_card".into(), json!({"suit": cr.castle_card.name()}));
    }
}

/// The last hand's scoring, its cards named by their view ids as the play began.
fn last_play(game: &Game, p: &PlayTrace, key: u64) -> Value {
    let id = |sort_id: u32| {
        let epoch = p
            .epoch_of(sort_id)
            .or_else(|| game.gs.areas.card(sort_id).map(|c| c.view_epoch))
            .unwrap_or(0);
        Value::from(card_id(key, sort_id, epoch))
    };
    let mut v = p.to_json(game.gs.sim.last_score.as_ref());
    for (list, ids) in [("played", &p.played), ("scoring", &p.scoring)] {
        v[list] = ids.iter().map(|&sid| id(sid)).collect();
    }
    if let Some(events) = v["events"].as_array_mut() {
        for (e, ev) in events.iter_mut().zip(&p.events) {
            if let Some(c) = ev.card {
                e["card"] = id(c);
            }
            if let Some(t) = ev.target {
                e["target"] = id(t);
            }
        }
    }
    v
}

/// The player's view of `game` ([`FORMAT`]), its cards named under `key`
/// ([`card_id`]).
pub fn player_view(game: &Game, key: u64) -> Value {
    let gs = &game.gs;
    let cr = &gs.current_round;
    let rr = &gs.round_resets;
    let mut current_round = Map::new();
    for (k, v) in [
        ("hands_left", cr.hands_left),
        ("discards_left", cr.discards_left),
        ("hands_played", cr.hands_played),
        ("discards_used", cr.discards_used),
        ("reroll_cost", cr.reroll_cost),
        ("free_rerolls", cr.free_rerolls),
        // What the Cash Out button pays (round evaluation; 0 elsewhere).
        ("cash_out", gs.sim.pending_cashout),
    ] {
        current_round.insert(k.into(), num(v));
    }
    round_cards(gs, &mut current_round);
    let shop = shop_open(gs).then(|| {
        json!({
            "jokers": cards(gs, AreaId::ShopJokers, key),
            "vouchers": cards(gs, AreaId::ShopVouchers, key),
            "boosters": cards(gs, AreaId::ShopBooster, key),
        })
    });
    let pack = gs
        .pack
        .as_ref()
        .filter(|_| gs.phase == Phase::PackOpening)
        .map(|p| {
            json!({
                "kind": p.kind,
                "choices_remaining": num(p.choices_remaining),
                "cards": cards(gs, AreaId::PackCards, key),
            })
        });
    let usage: Map<String, Value> = gs
        .consumeable_usage
        .totals
        .iter()
        .map(|(k, v)| (k.clone(), num(*v)))
        .collect();
    json!({
        "format": FORMAT,
        "phase": gs.phase.name(),
        "run": {
            "back": gs.back.key(),
            "stake": num(gs.stake),
            "challenge": gs.challenge,
            "win_ante": num(gs.win_ante),
            "won": gs.won,
        },
        "game": {
            "ante": num(rr.ante),
            "round": num(gs.round),
            "dollars": num(gs.dollars),
            "chips": num(gs.chips),
            "skips": num(gs.skips),
            "hands_played": num(gs.hands_played),
            "unused_discards": num(gs.unused_discards),
            "hand_size": num(gs.hand_size),
            "joker_slots": num(gs.joker_slots),
            "consumable_slots": num(gs.consumable_slots),
            "interest_cap": num(gs.interest_cap),
            "interest_amount": num(gs.interest_amount),
            "discount_percent": num(gs.discount_percent),
            "inflation": num(gs.inflation),
            "rental_rate": num(gs.rental_rate),
            "bankrupt_at": num(gs.bankrupt_at),
            "probabilities_normal": num(gs.probabilities_normal),
            "last_tarot_planet": gs.last_tarot_planet.map(|c| c.key()),
            "last_hand_played": gs.last_hand_played.map(|h| h.name()),
            "most_played_hand": cr.most_played_poker_hand.name(),
            "blind_on_deck": gs.blind_on_deck.map(BlindSlot::name),
            "boss_rerolled": rr.boss_rerolled,
        },
        "current_round": current_round,
        "blind": blind(gs),
        "blinds": blinds(game),
        "hands": crate::snapshot::hands_json(gs),
        "vouchers": gs.sets.used_vouchers.sorted_keys(),
        "tags": gs.tags.iter().map(|t| t.id.key()).collect::<Vec<_>>(),
        "consumable_usage": usage,
        "hand": cards(gs, AreaId::Hand, key),
        "play": cards(gs, AreaId::Play, key),
        "jokers": cards(gs, AreaId::Jokers, key),
        "consumables": cards(gs, AreaId::Consumables, key),
        "shop": shop,
        "pack": pack,
        "deck": deck(gs),
        "deck_size": gs.areas.get(AreaId::Deck).len(),
        "discard_size": gs.areas.get(AreaId::Discard).len(),
        "last_score": crate::snapshot::last_score_json(gs),
        "last_play": game.last_play().map(|p| last_play(game, p, key)),
    })
}

impl Game {
    /// The player's view of the run ([`player_view`]).
    pub fn player_view(&self, key: u64) -> Value {
        player_view(self, key)
    }
}

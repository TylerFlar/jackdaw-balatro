//! Canonical snapshots (`docs/snapshot-schema.md`): `GameState` to and from JSON.
//!
//! The snapshot is the format the Python oracle exports for parity checks, the save format,
//! and `Game.to_dict()`. Import is strict: unknown keys, missing keys and malformed values are
//! errors, so a field the Rust state doesn't hold can't be dropped silently. Non-finite numbers
//! appear in JSON text as `{"$f": "inf" | "-inf" | "nan"}`.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use jackdaw_data::{
    BlindId, CenterId, TagId, blind_by_key, card_by_key, center_by_key, tag_by_key,
};
use serde_json::{Map, Value, json};

use crate::area::AreaId;
use crate::card::{
    Ability, Base, Card, Edition, EditionKind, Extra, ExtraVal, Facing, LuaStr, Rank, Seal, Suit,
};
use crate::hand_eval::HandType;
use crate::rng::Rng;
use crate::state::{
    Areas, BlindSlot, BlindState, ConsumeableUsage, CurrentRound, GameState, HandSort, HandState,
    LastScore, LuaValue, PackState, Phase, RoundBonus, RoundResets, Sets, SimState, StartingParams,
};

pub const SCHEMA: &str = "jackdaw.snapshot/1";

/// A snapshot that can't be imported.
#[derive(Debug, Clone, PartialEq)]
pub struct SnapshotError(pub String);

impl std::fmt::Display for SnapshotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "snapshot: {}", self.0)
    }
}

impl std::error::Error for SnapshotError {}

type Result<T> = std::result::Result<T, SnapshotError>;

fn err<T>(msg: impl Into<String>) -> Result<T> {
    Err(SnapshotError(msg.into()))
}

// ---------------------------------------------------------------------------
// Numbers
// ---------------------------------------------------------------------------

/// A number as JSON: finite values as numbers (integral ones as integers), the rest as
/// `{"$f": ...}`.
pub fn num(x: f64) -> Value {
    if x.is_nan() {
        json!({"$f": "nan"})
    } else if x.is_infinite() {
        json!({"$f": if x > 0.0 { "inf" } else { "-inf" }})
    } else if x == x.trunc() && x.abs() < 9_007_199_254_740_992.0 {
        json!(x as i64)
    } else {
        json!(x)
    }
}

fn opt_num(x: Option<f64>) -> Value {
    x.map_or(Value::Null, num)
}

fn as_num(v: &Value, path: &str) -> Result<f64> {
    if let Some(x) = v.as_f64() {
        return Ok(x);
    }
    if let Some(Value::String(s)) = v.as_object().and_then(|o| o.get("$f")) {
        return Ok(match s.as_str() {
            "inf" => f64::INFINITY,
            "-inf" => f64::NEG_INFINITY,
            "nan" => f64::NAN,
            other => return err(format!("{path}: bad non-finite marker {other:?}")),
        });
    }
    err(format!("{path}: expected a number, got {v}"))
}

// ---------------------------------------------------------------------------
// Reading helpers
// ---------------------------------------------------------------------------

/// A JSON object whose keys must all be consumed.
struct Obj<'a> {
    map: &'a Map<String, Value>,
    path: String,
    seen: BTreeSet<&'a str>,
}

impl<'a> Obj<'a> {
    fn new(v: &'a Value, path: &str) -> Result<Obj<'a>> {
        match v.as_object() {
            Some(map) => Ok(Obj {
                map,
                path: path.to_owned(),
                seen: BTreeSet::new(),
            }),
            None => err(format!("{path}: expected an object, got {v}")),
        }
    }

    fn field(&mut self, key: &'a str) -> Result<&'a Value> {
        self.seen.insert(key);
        self.map
            .get(key)
            .ok_or_else(|| SnapshotError(format!("{}: missing key {key:?}", self.path)))
    }

    fn opt(&mut self, key: &'a str) -> Option<&'a Value> {
        self.seen.insert(key);
        self.map.get(key).filter(|v| !v.is_null())
    }

    fn sub(&self, key: &str) -> String {
        format!("{}.{key}", self.path)
    }

    fn num(&mut self, key: &'a str) -> Result<f64> {
        let p = self.sub(key);
        as_num(self.field(key)?, &p)
    }

    fn opt_num(&mut self, key: &'a str) -> Result<Option<f64>> {
        let p = self.sub(key);
        self.opt(key).map(|v| as_num(v, &p)).transpose()
    }

    fn bool(&mut self, key: &'a str) -> Result<bool> {
        let p = self.sub(key);
        self.field(key)?
            .as_bool()
            .ok_or_else(|| SnapshotError(format!("{p}: expected a boolean")))
    }

    fn str(&mut self, key: &'a str) -> Result<&'a str> {
        let p = self.sub(key);
        self.field(key)?
            .as_str()
            .ok_or_else(|| SnapshotError(format!("{p}: expected a string")))
    }

    fn opt_str(&mut self, key: &'a str) -> Result<Option<&'a str>> {
        let p = self.sub(key);
        match self.opt(key) {
            None => Ok(None),
            Some(v) => v
                .as_str()
                .map(Some)
                .ok_or_else(|| SnapshotError(format!("{p}: expected a string"))),
        }
    }

    fn arr(&mut self, key: &'a str) -> Result<&'a Vec<Value>> {
        let p = self.sub(key);
        self.field(key)?
            .as_array()
            .ok_or_else(|| SnapshotError(format!("{p}: expected an array")))
    }

    fn done(self) -> Result<()> {
        for k in self.map.keys() {
            if !self.seen.contains(k.as_str()) {
                return err(format!("{}: unknown key {k:?}", self.path));
            }
        }
        Ok(())
    }
}

fn center(key: &str, path: &str) -> Result<CenterId> {
    center_by_key(key).ok_or_else(|| SnapshotError(format!("{path}: unknown center {key:?}")))
}

fn blind(key: &str, path: &str) -> Result<BlindId> {
    blind_by_key(key).ok_or_else(|| SnapshotError(format!("{path}: unknown blind {key:?}")))
}

fn tag(key: &str, path: &str) -> Result<TagId> {
    tag_by_key(key).ok_or_else(|| SnapshotError(format!("{path}: unknown tag {key:?}")))
}

fn hand_type(name: &str, path: &str) -> Result<HandType> {
    HandType::from_name(name).ok_or_else(|| SnapshotError(format!("{path}: unknown hand {name:?}")))
}

fn suit(name: &str, path: &str) -> Result<Suit> {
    Suit::from_name(name).ok_or_else(|| SnapshotError(format!("{path}: unknown suit {name:?}")))
}

fn rank(name: &str, path: &str) -> Result<Rank> {
    Rank::from_name(name).ok_or_else(|| SnapshotError(format!("{path}: unknown rank {name:?}")))
}

fn phase(name: &str, path: &str) -> Result<Phase> {
    Phase::from_name(name).ok_or_else(|| SnapshotError(format!("{path}: unknown phase {name:?}")))
}

fn lua_value(v: &Value, path: &str) -> Result<LuaValue> {
    Ok(match v {
        Value::Bool(b) => LuaValue::Bool(*b),
        Value::Number(_) => LuaValue::Num(as_num(v, path)?),
        Value::String(s) => LuaValue::Str(s.clone()),
        Value::Object(map) if map.contains_key("$f") => LuaValue::Num(as_num(v, path)?),
        Value::Object(map) => LuaValue::Table(
            map.iter()
                .map(|(k, v)| Ok((k.clone(), lua_value(v, &format!("{path}.{k}"))?)))
                .collect::<Result<_>>()?,
        ),
        _ => return err(format!("{path}: unsupported value {v}")),
    })
}

fn lua_value_json(v: &LuaValue) -> Value {
    match v {
        LuaValue::Bool(b) => json!(b),
        LuaValue::Num(x) => num(*x),
        LuaValue::Str(s) => json!(s),
        LuaValue::Table(t) => Value::Object(
            t.iter()
                .map(|(k, v)| (k.clone(), lua_value_json(v)))
                .collect(),
        ),
    }
}

// ---------------------------------------------------------------------------
// Cards
// ---------------------------------------------------------------------------

fn extra_json(extra: &Extra) -> Value {
    match extra {
        Extra::Nil => Value::Null,
        Extra::Num(x) => num(*x),
        Extra::Str(s) => json!(s),
        Extra::Table(t) => Value::Object(
            t.entries()
                .iter()
                .map(|(k, v)| {
                    let v = match v {
                        ExtraVal::Num(x) => num(*x),
                        ExtraVal::Str(s) => json!(s.as_str()),
                        ExtraVal::Bool(b) => json!(b),
                    };
                    ((*k).to_owned(), v)
                })
                .collect(),
        ),
    }
}

/// `ability.extra` from a snapshot, checked against the center's config shape (the game never
/// adds keys to it, so the keys come from the center).
fn extra_from(v: Option<&Value>, center: CenterId, path: &str) -> Result<Extra> {
    let base = Extra::from_config(center.get().config("extra"));
    let Some(v) = v else {
        return Ok(Extra::Nil);
    };
    match v {
        Value::String(s) => match base {
            Extra::Str(st) if st == s.as_str() => Ok(Extra::Str(st)),
            _ => err(format!(
                "{path}: string extra {s:?} differs from the center's"
            )),
        },
        Value::Object(map) if !map.contains_key("$f") => {
            let Extra::Table(mut table) = base else {
                return err(format!("{path}: table extra on a center without one"));
            };
            let mut keys: Vec<&str> = table.entries().iter().map(|(k, _)| *k).collect();
            keys.sort_unstable();
            let mut got: Vec<&str> = map.keys().map(String::as_str).collect();
            got.sort_unstable();
            if keys != got {
                return err(format!(
                    "{path}: extra keys {got:?} differ from the center's {keys:?}"
                ));
            }
            for (k, val) in map {
                let p = format!("{path}.{k}");
                let ev = match val {
                    Value::String(s) => ExtraVal::Str(LuaStr::Owned(s.clone().into_boxed_str())),
                    Value::Bool(b) => ExtraVal::Bool(*b),
                    other => ExtraVal::Num(as_num(other, &p)?),
                };
                table.set(k, ev);
            }
            Ok(Extra::Table(table))
        }
        other => Ok(Extra::Num(as_num(other, path)?)),
    }
}

fn ability_json(c: &Card) -> Value {
    let a = &c.ability;
    let mut m = Map::new();
    for (k, v) in [
        ("mult", a.mult),
        ("h_mult", a.h_mult),
        ("h_x_mult", a.h_x_mult),
        ("h_dollars", a.h_dollars),
        ("p_dollars", a.p_dollars),
        ("t_mult", a.t_mult),
        ("t_chips", a.t_chips),
        ("x_mult", a.x_mult),
        ("h_size", a.h_size),
        ("d_size", a.d_size),
        ("extra_value", a.extra_value),
        ("perma_bonus", a.perma_bonus),
        ("bonus", a.bonus),
        ("hands_played_at_create", a.hands_played_at_create),
    ] {
        m.insert(k.to_owned(), num(v));
    }
    m.insert("extra".to_owned(), extra_json(&a.extra));
    m.insert(
        "forced_selection".to_owned(),
        if a.forced_selection {
            json!(true)
        } else {
            Value::Null
        },
    );
    for (k, v) in [
        ("invis_rounds", a.invis_rounds),
        ("caino_xmult", a.caino_xmult),
        ("yorick_discards", a.yorick_discards),
        ("burnt_hand", a.burnt_hand),
        ("loyalty_remaining", a.loyalty_remaining),
    ] {
        if let Some(x) = v {
            m.insert(k.to_owned(), num(x));
        }
    }
    if let Some(h) = a.to_do_poker_hand {
        m.insert("to_do_poker_hand".to_owned(), json!(h.name()));
    }
    if a.couponed {
        m.insert("couponed".to_owned(), json!(true));
    }
    if a.played_this_ante {
        m.insert("played_this_ante".to_owned(), json!(true));
    }
    if a.perma_debuff {
        m.insert("perma_debuff".to_owned(), json!(true));
    }
    Value::Object(m)
}

fn ability_from(v: &Value, center: CenterId, path: &str) -> Result<Ability> {
    let mut o = Obj::new(v, path)?;
    let mut a = Ability {
        mult: o.num("mult")?,
        h_mult: o.num("h_mult")?,
        h_x_mult: o.num("h_x_mult")?,
        h_dollars: o.num("h_dollars")?,
        p_dollars: o.num("p_dollars")?,
        t_mult: o.num("t_mult")?,
        t_chips: o.num("t_chips")?,
        x_mult: o.num("x_mult")?,
        h_size: o.num("h_size")?,
        d_size: o.num("d_size")?,
        extra_value: o.num("extra_value")?,
        perma_bonus: o.num("perma_bonus")?,
        bonus: o.num("bonus")?,
        hands_played_at_create: o.num("hands_played_at_create")?,
        ..Ability::default()
    };
    let p = o.sub("extra");
    a.extra = extra_from(o.opt("extra"), center, &p)?;
    a.forced_selection = match o.opt("forced_selection") {
        None => false,
        Some(Value::Bool(true)) => true,
        Some(other) => return err(format!("{path}.forced_selection: unexpected {other}")),
    };
    a.invis_rounds = o.opt_num("invis_rounds")?;
    a.caino_xmult = o.opt_num("caino_xmult")?;
    a.yorick_discards = o.opt_num("yorick_discards")?;
    a.burnt_hand = o.opt_num("burnt_hand")?;
    a.loyalty_remaining = o.opt_num("loyalty_remaining")?;
    let p = o.sub("to_do_poker_hand");
    a.to_do_poker_hand = o
        .opt_str("to_do_poker_hand")?
        .map(|h| hand_type(h, &p))
        .transpose()?;
    a.couponed = o.opt("couponed").and_then(Value::as_bool).unwrap_or(false);
    a.played_this_ante = o
        .opt("played_this_ante")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    a.perma_debuff = o
        .opt("perma_debuff")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    o.done()?;
    Ok(a)
}

fn edition_json(e: &Option<Edition>) -> Value {
    let Some(e) = e else {
        return Value::Null;
    };
    let mut m = Map::new();
    m.insert(e.kind.name().to_owned(), json!(true));
    m.insert("type".to_owned(), json!(e.kind.name()));
    if let Some(x) = e.chips {
        m.insert("chips".to_owned(), num(x));
    }
    if let Some(x) = e.mult {
        m.insert("mult".to_owned(), num(x));
    }
    if let Some(x) = e.x_mult {
        m.insert("x_mult".to_owned(), num(x));
    }
    Value::Object(m)
}

fn edition_from(v: Option<&Value>, path: &str) -> Result<Option<Edition>> {
    let Some(v) = v else { return Ok(None) };
    let mut o = Obj::new(v, path)?;
    let kind_name = o.str("type")?;
    let kind = EditionKind::from_name(kind_name)
        .ok_or_else(|| SnapshotError(format!("{path}: unknown edition {kind_name:?}")))?;
    if !o.bool(kind.name())? {
        return err(format!("{path}: edition flag {} is false", kind.name()));
    }
    let e = Edition {
        kind,
        chips: o.opt_num("chips")?,
        mult: o.opt_num("mult")?,
        x_mult: o.opt_num("x_mult")?,
    };
    o.done()?;
    Ok(Some(e))
}

/// A card as a snapshot object.
pub fn card_json(c: &Card) -> Value {
    let base = c.base.as_ref().map_or(Value::Null, |b| {
        json!({
            "suit": b.suit.name(),
            "value": b.value.name(),
            "id": b.id(),
            "nominal": num(b.nominal()),
            "suit_nominal": num(b.suit_nominal()),
            "suit_nominal_original": num(b.suit_nominal_original),
            "face_nominal": num(b.face_nominal()),
            "original_value": b.original_value.map(Rank::name),
            "times_played": num(b.times_played),
        })
    });
    json!({
        "sort_id": c.sort_id,
        "center": c.center.key(),
        "front": c.base.as_ref().map(|b| b.front.get().key),
        "base": base,
        "ability": ability_json(c),
        "edition": edition_json(&c.edition),
        "seal": c.seal.map(Seal::name),
        "debuff": c.debuff,
        "added_to_deck": c.added_to_deck,
        "facing": match c.facing { Facing::Front => "front", Facing::Back => "back" },
        "playing_card": c.playing_card,
        "costs": {
            "base_cost": num(c.base_cost),
            "extra_cost": num(c.extra_cost),
            "cost": num(c.cost),
            "sell_cost": num(c.sell_cost),
        },
        "stickers": {
            "eternal": c.ability.eternal,
            "perishable": c.ability.perishable,
            "perish_tally": if c.ability.perishable { opt_num(c.ability.perish_tally) } else { Value::Null },
            "rental": c.ability.rental,
            "pinned": c.pinned,
        },
    })
}

/// A card from a snapshot object.
pub fn card_from(v: &Value, path: &str) -> Result<Card> {
    let mut o = Obj::new(v, path)?;
    let sort_id = o.num("sort_id")? as u32;
    let p = o.sub("center");
    let center = center(o.str("center")?, &p)?;
    let front = o.opt_str("front")?;
    let base = match o.opt("base") {
        None => None,
        Some(bv) => {
            let bp = o.sub("base");
            let mut b = Obj::new(bv, &bp)?;
            let s = suit(b.str("suit")?, &bp)?;
            let value = rank(b.str("value")?, &bp)?;
            let original_value = b
                .opt_str("original_value")?
                .map(|r| rank(r, &bp))
                .transpose()?;
            let front_key =
                front.ok_or_else(|| SnapshotError(format!("{path}: base without front")))?;
            let front_id = card_by_key(front_key)
                .ok_or_else(|| SnapshotError(format!("{path}: unknown front {front_key:?}")))?;
            let base = Base {
                front: front_id,
                suit: s,
                value,
                suit_nominal_original: b.num("suit_nominal_original")?,
                original_value,
                times_played: b.num("times_played")?,
            };
            // Derived fields must agree with the suit and rank.
            for (k, want) in [
                ("id", f64::from(base.id())),
                ("nominal", base.nominal()),
                ("suit_nominal", base.suit_nominal()),
                ("face_nominal", base.face_nominal()),
            ] {
                let got = b.num(k)?;
                if got != want {
                    return err(format!(
                        "{bp}.{k}: {got} disagrees with {}/{}",
                        s.name(),
                        value.name()
                    ));
                }
            }
            if base.front != Base::front_for(s, value) {
                return err(format!(
                    "{path}: front {front_key} disagrees with {}/{}",
                    s.name(),
                    value.name()
                ));
            }
            b.done()?;
            Some(base)
        }
    };
    let ap = o.sub("ability");
    let mut ability = ability_from(o.field("ability")?, center, &ap)?;
    let ep = o.sub("edition");
    let edition = edition_from(o.opt("edition"), &ep)?;
    let sp = o.sub("seal");
    let seal = o
        .opt_str("seal")?
        .map(|s| {
            Seal::from_name(s).ok_or_else(|| SnapshotError(format!("{sp}: unknown seal {s:?}")))
        })
        .transpose()?;
    let debuff = o.bool("debuff")?;
    let added_to_deck = o.bool("added_to_deck")?;
    let facing = match o.str("facing")? {
        "front" => Facing::Front,
        "back" => Facing::Back,
        other => return err(format!("{path}.facing: {other:?}")),
    };
    let playing_card = o.opt_num("playing_card")?.map(|x| x as u32);
    let cp = o.sub("costs");
    let mut costs = Obj::new(o.field("costs")?, &cp)?;
    let (base_cost, extra_cost, cost, sell_cost) = (
        costs.num("base_cost")?,
        costs.num("extra_cost")?,
        costs.num("cost")?,
        costs.num("sell_cost")?,
    );
    costs.done()?;
    let stp = o.sub("stickers");
    let mut st = Obj::new(o.field("stickers")?, &stp)?;
    ability.eternal = st.bool("eternal")?;
    ability.perishable = st.bool("perishable")?;
    ability.perish_tally = st.opt_num("perish_tally")?;
    ability.rental = st.bool("rental")?;
    let pinned = st.bool("pinned")?;
    st.done()?;
    o.done()?;
    Ok(Card {
        sort_id,
        center,
        base,
        ability,
        edition,
        seal,
        debuff,
        facing,
        playing_card,
        base_cost,
        extra_cost,
        cost,
        sell_cost,
        added_to_deck,
        pinned,
        lucky_trigger: false,
        vampired: false,
        shattered: false,
        destroyed: false,
        getting_sliced: false,
    })
}

fn cards_json(cards: &[Card]) -> Value {
    Value::Array(cards.iter().map(card_json).collect())
}

fn cards_from(v: &[Value], path: &str) -> Result<Vec<Card>> {
    v.iter()
        .enumerate()
        .map(|(i, c)| card_from(c, &format!("{path}[{i}]")))
        .collect()
}

// ---------------------------------------------------------------------------
// GameState
// ---------------------------------------------------------------------------

/// A snapshot of the whole run.
pub fn export(gs: &GameState) -> Value {
    let rr = &gs.round_resets;
    let cr = &gs.current_round;
    let slots3 = |f: &dyn Fn(BlindSlot) -> Value| -> Value {
        Value::Object(
            BlindSlot::ALL
                .iter()
                .map(|&s| (s.name().to_owned(), f(s)))
                .collect(),
        )
    };
    let blind_states = slots3(&|s| json!(rr.blind_state(s)));
    // Python stores Small/Big/Boss choices; Boss is absent until rolled.
    let mut choices = Map::new();
    for s in BlindSlot::ALL {
        if let Some(b) = rr.blind_choice(s) {
            choices.insert(s.name().to_owned(), json!(b.key()));
        }
    }
    let blind_tags = rr.blind_tags.map_or(Value::Null, |tags| {
        json!({
            "Small": tags[0].map(TagId::key),
            "Big": tags[1].map(TagId::key),
        })
    });
    let blind = gs.blind.as_ref().map_or(Value::Null, |b| {
        let mut eye: Vec<&str> = b.eye_hands.iter().map(|h| h.name()).collect();
        eye.sort_unstable();
        json!({
            "key": b.key.map_or("", BlindId::key),
            "chips": num(b.chips),
            "mult": num(b.mult),
            "dollars": num(b.dollars),
            "boss": b.boss,
            "disabled": b.disabled,
            "triggered": b.triggered,
            "prepped": b.prepped,
            "eye_hands": eye,
            "mouth_hand": b.mouth_hand.map(HandType::name),
            "discards_sub": opt_num(b.discards_sub),
            "hands_sub": opt_num(b.hands_sub),
        })
    });
    let hands = Value::Object(
        HandType::ALL
            .iter()
            .map(|&h| {
                let s = gs.hand(h);
                (
                    h.name().to_owned(),
                    json!({
                        "level": num(s.level),
                        "chips": num(s.chips),
                        "mult": num(s.mult),
                        "played": num(s.played),
                        "played_this_round": num(s.played_this_round),
                        "visible": s.visible,
                    }),
                )
            })
            .collect(),
    );
    let sp = &gs.starting_params;
    let areas = Value::Object(
        AreaId::ALL
            .iter()
            .map(|&a| {
                let mut cards = cards_json(gs.areas.get(a));
                // Deck and discard hold their cards face down; the snapshot writes them face up.
                if a.face_down() {
                    for c in cards.as_array_mut().expect("array") {
                        c["facing"] = json!("front");
                    }
                }
                (a.name().to_owned(), cards)
            })
            .collect(),
    );
    let pack = gs.pack.as_ref().map_or(Value::Null, |p| {
        json!({
            "kind": p.kind,
            "choices_remaining": num(p.choices_remaining),
            "dealt": p.dealt,
            "return_phase": p.return_phase.name(),
        })
    });
    let streams: Map<String, Value> = gs
        .rng
        .streams()
        .iter()
        .map(|(k, v)| (k.clone(), num(*v)))
        .collect();
    let last_score = gs.sim.last_score.map_or(Value::Null, |s| {
        json!({
            "hand_type": s.hand_type.name(),
            "chips": num(s.chips),
            "mult": num(s.mult),
            "total": num(s.total),
            "dollars": num(s.dollars),
            "debuffed": s.debuffed,
        })
    });
    json!({
        "schema": SCHEMA,
        "phase": gs.phase.name(),
        "run": {
            "seed": gs.rng.seed_str(),
            "back": gs.back.key(),
            "stake": num(gs.stake),
            "challenge": gs.challenge,
            "win_ante": num(gs.win_ante),
            "won": gs.won,
        },
        "game": {
            "dollars": num(gs.dollars),
            "chips": num(gs.chips),
            "round": num(gs.round),
            "skips": num(gs.skips),
            "hands_played": num(gs.hands_played),
            "unused_discards": num(gs.unused_discards),
            "interest_cap": num(gs.interest_cap),
            "interest_amount": num(gs.interest_amount),
            "discount_percent": num(gs.discount_percent),
            "inflation": num(gs.inflation),
            "rental_rate": num(gs.rental_rate),
            "bankrupt_at": num(gs.bankrupt_at),
            "base_reroll_cost": num(gs.base_reroll_cost),
            "edition_rate": num(gs.edition_rate),
            "joker_rate": num(gs.joker_rate),
            "tarot_rate": num(gs.tarot_rate),
            "planet_rate": num(gs.planet_rate),
            "spectral_rate": num(gs.spectral_rate),
            "playing_card_rate": num(gs.playing_card_rate),
            "probabilities_normal": num(gs.probabilities_normal),
            "hand_size": num(gs.hand_size),
            "hand_sort": gs.hand_sort.name(),
            "joker_slots": num(gs.joker_slots),
            "consumable_slots": num(gs.consumable_slots),
            "starting_deck_size": num(gs.starting_deck_size),
            "last_tarot_planet": gs.last_tarot_planet.map(CenterId::key),
            "last_hand_played": gs.last_hand_played.map(HandType::name),
            "pack_size": num(gs.pack_size),
            "ecto_minus": num(gs.ecto_minus),
            "perishable_rounds": num(gs.perishable_rounds),
            "blind_on_deck": gs.blind_on_deck.map(BlindSlot::name),
            "shop_joker_max": num(gs.shop_joker_max),
            "first_shop_buffoon": gs.first_shop_buffoon,
            "orbital_choices": Value::Object(
                gs.orbital_choices
                    .iter()
                    .map(|(ante, hands)| {
                        let m: Map<String, Value> = BlindSlot::ALL
                            .iter()
                            .zip(hands)
                            .map(|(s, h)| (s.name().to_owned(), json!(h.name())))
                            .collect();
                        (ante.clone(), Value::Object(m))
                    })
                    .collect(),
            ),
        },
        "starting_params": {
            "dollars": num(sp.dollars),
            "hand_size": num(sp.hand_size),
            "discards": num(sp.discards),
            "hands": num(sp.hands),
            "reroll_cost": num(sp.reroll_cost),
            "joker_slots": num(sp.joker_slots),
            "consumable_slots": num(sp.consumable_slots),
            "ante_scaling": num(sp.ante_scaling),
            "no_faces": sp.no_faces,
            "erratic_suits_and_ranks": sp.erratic_suits_and_ranks,
        },
        "modifiers": Value::Object(gs.modifiers.iter().map(|(k, v)| (k.clone(), lua_value_json(v))).collect()),
        "round_resets": {
            "hands": num(rr.hands),
            "discards": num(rr.discards),
            "reroll_cost": num(rr.reroll_cost),
            "temp_reroll_cost": opt_num(rr.temp_reroll_cost),
            "temp_handsize": opt_num(rr.temp_handsize),
            "ante": num(rr.ante),
            "blind_ante": num(rr.blind_ante),
            "blind_states": blind_states,
            "blind_choices": Value::Object(choices),
            "blind_tags": blind_tags,
            "boss_rerolled": rr.boss_rerolled,
        },
        "current_round": {
            "hands_left": num(cr.hands_left),
            "hands_played": num(cr.hands_played),
            "discards_left": num(cr.discards_left),
            "discards_used": num(cr.discards_used),
            "reroll_cost": num(cr.reroll_cost),
            "reroll_cost_increase": num(cr.reroll_cost_increase),
            "free_rerolls": num(cr.free_rerolls),
            "jokers_purchased": num(cr.jokers_purchased),
            "voucher": cr.voucher.map(CenterId::key),
            "most_played_poker_hand": cr.most_played_poker_hand.name(),
            "idol_card": {"suit": cr.idol_card.0.name(), "rank": cr.idol_card.1.name(), "id": cr.idol_card.1.id()},
            "mail_card": {"rank": cr.mail_card.name(), "id": cr.mail_card.id()},
            "ancient_card": {"suit": cr.ancient_card.map(Suit::name)},
            "castle_card": {"suit": cr.castle_card.name()},
        },
        "round_bonus": {"next_hands": num(gs.round_bonus.next_hands), "discards": num(gs.round_bonus.discards)},
        "blind": blind,
        "hands": hands,
        "tags": gs.tags.iter().map(|t| json!({"key": t.key()})).collect::<Vec<_>>(),
        "sets": {
            "used_jokers": gs.sets.used_jokers.iter().collect::<Vec<_>>(),
            "used_vouchers": gs.sets.used_vouchers.iter().collect::<Vec<_>>(),
            "banned_keys": gs.sets.banned_keys.iter().collect::<Vec<_>>(),
            "pool_flags": Value::Object(gs.sets.pool_flags.iter().map(|(k, v)| (k.clone(), lua_value_json(v))).collect()),
            "bosses_used": Value::Object(gs.sets.bosses_used.iter().map(|(k, v)| ((*k).to_owned(), num(*v))).collect()),
        },
        "consumeable_usage": {
            "by_key": Value::Object(gs.consumeable_usage.by_key.iter().map(|(k, v)| ((*k).to_owned(), num(*v))).collect()),
            "totals": Value::Object(gs.consumeable_usage.totals.iter().map(|(k, v)| (k.clone(), num(*v))).collect()),
        },
        "areas": areas,
        "pack": pack,
        "rng": {
            "seed": gs.rng.seed_str(),
            "hashed_seed": num(gs.rng.hashed_seed()),
            "streams": streams,
        },
        "sim": {
            "next_sort_id": gs.sim.next_sort_id,
            "pending_tag_packs": gs.sim.pending_tag_packs.iter().map(|c| c.key()).collect::<Vec<_>>(),
            "pending_deck_bottom": cards_json(&gs.sim.pending_deck_bottom),
            "shop_d6ed": gs.sim.shop_d6ed,
            "shop_free": gs.sim.shop_free,
            "hand_last_poll_size": num(gs.sim.hand_last_poll_size),
            "last_blind_was_boss": gs.sim.last_blind_was_boss,
            "pending_cashout": num(gs.sim.pending_cashout),
            "last_score": last_score,
            "tag_history": gs.sim.tag_history.iter().map(|t| t.key()).collect::<Vec<_>>(),
        },
    })
}

fn str_list<'a>(v: &'a [Value], path: &str) -> Result<Vec<&'a str>> {
    v.iter()
        .map(|x| {
            x.as_str()
                .ok_or_else(|| SnapshotError(format!("{path}: expected strings")))
        })
        .collect()
}

/// A run from a snapshot.
pub fn import(v: &Value) -> Result<GameState> {
    let mut o = Obj::new(v, "$")?;
    if o.str("schema")? != SCHEMA {
        return err(format!("unsupported schema {:?}", o.str("schema")?));
    }
    let ph = phase(o.str("phase")?, "$.phase")?;

    let mut run = Obj::new(o.field("run")?, "$.run")?;
    let seed = run.str("seed")?.to_owned();
    let back = center(run.str("back")?, "$.run.back")?;
    let stake = run.num("stake")?;
    let challenge = run.opt_str("challenge")?.map(str::to_owned);
    let win_ante = run.num("win_ante")?;
    let won = run.bool("won")?;
    run.done()?;

    let mut g = Obj::new(o.field("game")?, "$.game")?;
    let mut sp = Obj::new(o.field("starting_params")?, "$.starting_params")?;
    let starting_params = StartingParams {
        dollars: sp.num("dollars")?,
        hand_size: sp.num("hand_size")?,
        discards: sp.num("discards")?,
        hands: sp.num("hands")?,
        reroll_cost: sp.num("reroll_cost")?,
        joker_slots: sp.num("joker_slots")?,
        consumable_slots: sp.num("consumable_slots")?,
        ante_scaling: sp.num("ante_scaling")?,
        no_faces: sp.bool("no_faces")?,
        erratic_suits_and_ranks: sp.bool("erratic_suits_and_ranks")?,
    };
    sp.done()?;

    let modifiers = match o.field("modifiers")? {
        Value::Object(map) => map
            .iter()
            .map(|(k, v)| Ok((k.clone(), lua_value(v, &format!("$.modifiers.{k}"))?)))
            .collect::<Result<BTreeMap<_, _>>>()?,
        other => return err(format!("$.modifiers: expected an object, got {other}")),
    };

    let mut rr = Obj::new(o.field("round_resets")?, "$.round_resets")?;
    let mut states = Obj::new(rr.field("blind_states")?, "$.round_resets.blind_states")?;
    let blind_states = [
        states.str("Small")?.to_owned(),
        states.str("Big")?.to_owned(),
        states.str("Boss")?.to_owned(),
    ];
    states.done()?;
    let mut choices = Obj::new(rr.field("blind_choices")?, "$.round_resets.blind_choices")?;
    let mut blind_choices = [None; 3];
    for (i, slot) in BlindSlot::ALL.iter().enumerate() {
        if let Some(k) = choices.opt_str(slot.name())? {
            blind_choices[i] = Some(blind(k, "$.round_resets.blind_choices")?);
        }
    }
    choices.done()?;
    let blind_tags = match rr.opt("blind_tags") {
        None => None,
        Some(bt) => {
            let mut t = Obj::new(bt, "$.round_resets.blind_tags")?;
            let small = t
                .opt_str("Small")?
                .map(|k| tag(k, "$.round_resets.blind_tags"))
                .transpose()?;
            let big = t
                .opt_str("Big")?
                .map(|k| tag(k, "$.round_resets.blind_tags"))
                .transpose()?;
            t.done()?;
            Some([small, big])
        }
    };
    let round_resets = RoundResets {
        hands: rr.num("hands")?,
        discards: rr.num("discards")?,
        reroll_cost: rr.num("reroll_cost")?,
        temp_reroll_cost: rr.opt_num("temp_reroll_cost")?,
        temp_handsize: rr.opt_num("temp_handsize")?,
        ante: rr.num("ante")?,
        blind_ante: rr.num("blind_ante")?,
        blind_states,
        blind_choices,
        blind_tags,
        boss_rerolled: rr.bool("boss_rerolled")?,
    };
    rr.done()?;

    let mut cr = Obj::new(o.field("current_round")?, "$.current_round")?;
    let mut idol = Obj::new(cr.field("idol_card")?, "$.current_round.idol_card")?;
    let idol_card = (
        suit(idol.str("suit")?, "$.idol_card")?,
        rank(idol.str("rank")?, "$.idol_card")?,
    );
    idol.num("id")?;
    idol.done()?;
    let mut mail = Obj::new(cr.field("mail_card")?, "$.current_round.mail_card")?;
    let mail_card = rank(mail.str("rank")?, "$.mail_card")?;
    mail.num("id")?;
    mail.done()?;
    let mut anc = Obj::new(cr.field("ancient_card")?, "$.current_round.ancient_card")?;
    let ancient_card = anc
        .opt_str("suit")?
        .map(|s| suit(s, "$.ancient_card"))
        .transpose()?;
    anc.done()?;
    let mut castle = Obj::new(cr.field("castle_card")?, "$.current_round.castle_card")?;
    let castle_card = suit(castle.str("suit")?, "$.castle_card")?;
    castle.done()?;
    let current_round = CurrentRound {
        hands_left: cr.num("hands_left")?,
        hands_played: cr.num("hands_played")?,
        discards_left: cr.num("discards_left")?,
        discards_used: cr.num("discards_used")?,
        reroll_cost: cr.num("reroll_cost")?,
        reroll_cost_increase: cr.num("reroll_cost_increase")?,
        free_rerolls: cr.num("free_rerolls")?,
        jokers_purchased: cr.num("jokers_purchased")?,
        voucher: cr
            .opt_str("voucher")?
            .map(|k| center(k, "$.current_round.voucher"))
            .transpose()?,
        most_played_poker_hand: hand_type(cr.str("most_played_poker_hand")?, "$.current_round")?,
        idol_card,
        mail_card,
        ancient_card,
        castle_card,
    };
    cr.done()?;

    let mut rb = Obj::new(o.field("round_bonus")?, "$.round_bonus")?;
    let round_bonus = RoundBonus {
        next_hands: rb.num("next_hands")?,
        discards: rb.num("discards")?,
    };
    rb.done()?;

    let blind_state = match o.opt("blind") {
        None => None,
        Some(bv) => {
            let mut b = Obj::new(bv, "$.blind")?;
            let key = b.str("key")?;
            let eye = str_list(b.arr("eye_hands")?, "$.blind.eye_hands")?
                .into_iter()
                .map(|h| hand_type(h, "$.blind.eye_hands"))
                .collect::<Result<Vec<_>>>()?;
            let st = BlindState {
                key: if key.is_empty() {
                    None
                } else {
                    Some(blind(key, "$.blind.key")?)
                },
                chips: b.num("chips")?,
                mult: b.num("mult")?,
                dollars: b.num("dollars")?,
                boss: b.bool("boss")?,
                disabled: b.bool("disabled")?,
                triggered: b.bool("triggered")?,
                prepped: b.bool("prepped")?,
                eye_hands: eye,
                mouth_hand: b
                    .opt_str("mouth_hand")?
                    .map(|h| hand_type(h, "$.blind.mouth_hand"))
                    .transpose()?,
                discards_sub: b.opt_num("discards_sub")?,
                hands_sub: b.opt_num("hands_sub")?,
            };
            b.done()?;
            Some(st)
        }
    };

    let mut hs = Obj::new(o.field("hands")?, "$.hands")?;
    let mut hands = [HandState {
        level: 0.0,
        chips: 0.0,
        mult: 0.0,
        played: 0.0,
        played_this_round: 0.0,
        visible: false,
    }; 12];
    for h in HandType::ALL {
        let p = format!("$.hands.{}", h.name());
        let mut e = Obj::new(hs.field(h.name())?, &p)?;
        hands[h.index()] = HandState {
            level: e.num("level")?,
            chips: e.num("chips")?,
            mult: e.num("mult")?,
            played: e.num("played")?,
            played_this_round: e.num("played_this_round")?,
            visible: e.bool("visible")?,
        };
        e.done()?;
    }
    hs.done()?;

    let tags = o
        .arr("tags")?
        .iter()
        .map(|t| {
            let mut to = Obj::new(t, "$.tags[]")?;
            let k = tag(to.str("key")?, "$.tags[]")?;
            to.done()?;
            Ok(k)
        })
        .collect::<Result<Vec<_>>>()?;

    let mut sets_o = Obj::new(o.field("sets")?, "$.sets")?;
    let center_keys = |v: &[Value], path: &str| -> Result<BTreeSet<&'static str>> {
        str_list(v, path)?
            .into_iter()
            .map(|k| Ok(center(k, path)?.key()))
            .collect()
    };
    let sets = Sets {
        used_jokers: center_keys(sets_o.arr("used_jokers")?, "$.sets.used_jokers")?,
        used_vouchers: center_keys(sets_o.arr("used_vouchers")?, "$.sets.used_vouchers")?,
        banned_keys: str_list(sets_o.arr("banned_keys")?, "$.sets.banned_keys")?
            .into_iter()
            .map(str::to_owned)
            .collect(),
        pool_flags: match sets_o.field("pool_flags")? {
            Value::Object(m) => m
                .iter()
                .map(|(k, v)| Ok((k.clone(), lua_value(v, "$.sets.pool_flags")?)))
                .collect::<Result<_>>()?,
            other => return err(format!("$.sets.pool_flags: {other}")),
        },
        bosses_used: match sets_o.field("bosses_used")? {
            Value::Object(m) => m
                .iter()
                .map(|(k, v)| {
                    Ok((
                        blind(k, "$.sets.bosses_used")?.key(),
                        as_num(v, "$.sets.bosses_used")?,
                    ))
                })
                .collect::<Result<_>>()?,
            other => return err(format!("$.sets.bosses_used: {other}")),
        },
    };
    sets_o.done()?;

    let mut cu = Obj::new(o.field("consumeable_usage")?, "$.consumeable_usage")?;
    let consumeable_usage = ConsumeableUsage {
        by_key: match cu.field("by_key")? {
            Value::Object(m) => m
                .iter()
                .map(|(k, v)| {
                    Ok((
                        center(k, "$.consumeable_usage")?.key(),
                        as_num(v, "$.consumeable_usage")?,
                    ))
                })
                .collect::<Result<_>>()?,
            other => return err(format!("$.consumeable_usage.by_key: {other}")),
        },
        totals: match cu.field("totals")? {
            Value::Object(m) => m
                .iter()
                .map(|(k, v)| Ok((k.clone(), as_num(v, "$.consumeable_usage.totals")?)))
                .collect::<Result<_>>()?,
            other => return err(format!("$.consumeable_usage.totals: {other}")),
        },
    };
    cu.done()?;

    let mut ao = Obj::new(o.field("areas")?, "$.areas")?;
    let mut areas = Areas::default();
    for a in AreaId::ALL {
        let p = format!("$.areas.{}", a.name());
        let mut cards = cards_from(ao.arr(a.name())?, &p)?;
        if a.face_down() {
            for c in &mut cards {
                c.facing = Facing::Back;
            }
        }
        *areas.get_mut(a) = cards;
    }
    ao.done()?;

    let pack = match o.opt("pack") {
        None => None,
        Some(pv) => {
            let mut p = Obj::new(pv, "$.pack")?;
            let st = PackState {
                kind: p.str("kind")?.to_owned(),
                choices_remaining: p.num("choices_remaining")?,
                dealt: p
                    .arr("dealt")?
                    .iter()
                    .map(|v| as_num(v, "$.pack.dealt").map(|x| x as u32))
                    .collect::<Result<_>>()?,
                return_phase: phase(p.str("return_phase")?, "$.pack.return_phase")?,
            };
            p.done()?;
            Some(st)
        }
    };

    let mut r = Obj::new(o.field("rng")?, "$.rng")?;
    let rng_seed = r.str("seed")?.to_owned();
    let hashed_seed = r.num("hashed_seed")?;
    let streams: HashMap<String, f64> = match r.field("streams")? {
        Value::Object(m) => m
            .iter()
            .map(|(k, v)| Ok((k.clone(), as_num(v, "$.rng.streams")?)))
            .collect::<Result<_>>()?,
        other => return err(format!("$.rng.streams: {other}")),
    };
    r.done()?;
    if rng_seed != seed {
        return err("$.rng.seed differs from $.run.seed");
    }

    let mut s = Obj::new(o.field("sim")?, "$.sim")?;
    let last_score = match s.opt("last_score") {
        None => None,
        Some(lv) => {
            let mut l = Obj::new(lv, "$.sim.last_score")?;
            let ls = LastScore {
                hand_type: hand_type(l.str("hand_type")?, "$.sim.last_score")?,
                chips: l.num("chips")?,
                mult: l.num("mult")?,
                total: l.num("total")?,
                dollars: l.num("dollars")?,
                debuffed: l.bool("debuffed")?,
            };
            l.done()?;
            Some(ls)
        }
    };
    let sim = SimState {
        next_sort_id: s.num("next_sort_id")? as u32,
        pending_tag_packs: str_list(s.arr("pending_tag_packs")?, "$.sim.pending_tag_packs")?
            .into_iter()
            .map(|k| center(k, "$.sim.pending_tag_packs"))
            .collect::<Result<_>>()?,
        pending_deck_bottom: cards_from(
            s.arr("pending_deck_bottom")?,
            "$.sim.pending_deck_bottom",
        )?,
        shop_d6ed: s.bool("shop_d6ed")?,
        shop_free: s.bool("shop_free")?,
        hand_last_poll_size: s.num("hand_last_poll_size")?,
        last_blind_was_boss: s.bool("last_blind_was_boss")?,
        pending_cashout: s.num("pending_cashout")?,
        last_score,
        tag_history: str_list(s.arr("tag_history")?, "$.sim.tag_history")?
            .into_iter()
            .map(|k| tag(k, "$.sim.tag_history"))
            .collect::<Result<_>>()?,
    };
    s.done()?;

    let gs = GameState {
        phase: ph,
        back,
        stake,
        challenge,
        win_ante,
        won,
        dollars: g.num("dollars")?,
        chips: g.num("chips")?,
        round: g.num("round")?,
        skips: g.num("skips")?,
        hands_played: g.num("hands_played")?,
        unused_discards: g.num("unused_discards")?,
        interest_cap: g.num("interest_cap")?,
        interest_amount: g.num("interest_amount")?,
        discount_percent: g.num("discount_percent")?,
        inflation: g.num("inflation")?,
        rental_rate: g.num("rental_rate")?,
        bankrupt_at: g.num("bankrupt_at")?,
        base_reroll_cost: g.num("base_reroll_cost")?,
        edition_rate: g.num("edition_rate")?,
        joker_rate: g.num("joker_rate")?,
        tarot_rate: g.num("tarot_rate")?,
        planet_rate: g.num("planet_rate")?,
        spectral_rate: g.num("spectral_rate")?,
        playing_card_rate: g.num("playing_card_rate")?,
        probabilities_normal: g.num("probabilities_normal")?,
        hand_size: g.num("hand_size")?,
        hand_sort: {
            let m = g.str("hand_sort")?;
            HandSort::from_name(m)
                .ok_or_else(|| SnapshotError(format!("$.game.hand_sort: {m:?}")))?
        },
        joker_slots: g.num("joker_slots")?,
        consumable_slots: g.num("consumable_slots")?,
        starting_deck_size: g.num("starting_deck_size")?,
        last_tarot_planet: g
            .opt_str("last_tarot_planet")?
            .map(|k| center(k, "$.game.last_tarot_planet"))
            .transpose()?,
        last_hand_played: g
            .opt_str("last_hand_played")?
            .map(|h| hand_type(h, "$.game.last_hand_played"))
            .transpose()?,
        pack_size: g.num("pack_size")?,
        ecto_minus: g.num("ecto_minus")?,
        perishable_rounds: g.num("perishable_rounds")?,
        blind_on_deck: g
            .opt_str("blind_on_deck")?
            .map(|b| {
                BlindSlot::from_name(b)
                    .ok_or_else(|| SnapshotError(format!("$.game.blind_on_deck: {b:?}")))
            })
            .transpose()?,
        shop_joker_max: g.num("shop_joker_max")?,
        first_shop_buffoon: g.bool("first_shop_buffoon")?,
        orbital_choices: match g.field("orbital_choices")? {
            Value::Object(m) => m
                .iter()
                .map(|(ante, v)| {
                    let p = format!("$.game.orbital_choices.{ante}");
                    let mut o = Obj::new(v, &p)?;
                    let mut hands = [HandType::HighCard; 3];
                    for (i, slot) in BlindSlot::ALL.iter().enumerate() {
                        hands[i] = hand_type(o.str(slot.name())?, &p)?;
                    }
                    o.done()?;
                    Ok((ante.clone(), hands))
                })
                .collect::<Result<_>>()?,
            other => return err(format!("$.game.orbital_choices: {other}")),
        },
        starting_params,
        modifiers,
        round_resets,
        current_round,
        round_bonus,
        blind: blind_state,
        hands,
        tags,
        sets,
        consumeable_usage,
        areas,
        pack,
        rng: Rng::from_parts(seed, hashed_seed, streams),
        sim,
    };
    g.done()?;
    o.done()?;
    Ok(gs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numbers_round_trip() {
        for x in [0.0, 1.0, 1.5, -2.0, 1e300, f64::INFINITY, f64::NEG_INFINITY] {
            let back = as_num(&num(x), "$").unwrap();
            assert_eq!(back.to_bits(), x.to_bits(), "{x}");
        }
        assert!(as_num(&num(f64::NAN), "$").unwrap().is_nan());
        // The canonical form has one zero.
        assert_eq!(num(-0.0), json!(0));
        assert_eq!(num(3.0), json!(3));
        assert_eq!(num(0.25), json!(0.25));
    }
}

//! `cargo xtask codegen [--check]`: generate `crates/jackdaw-data/src/generated.rs` from the
//! game-data JSON in `jackdaw/engine/data/`.
//!
//! The JSON comes from the game's own Lua (`scripts/extract_prototypes.lua` for the prototype
//! tables, `scripts/extract_game_tables.lua` for hands, challenges and blind amounts). The output
//! is formatted with rustfmt and checked in; `--check` regenerates it in memory and fails if the
//! committed file differs.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;
use std::fs;
use std::io::Write as _;
use std::path::Path;
use std::process::{Command, Stdio};

use serde_json::{Map, Value as Json};

const OUTPUT: &str = "crates/jackdaw-data/src/generated.rs";
const DATA_DIR: &str = "jackdaw/engine/data";

/// Center sets in id order.
const SET_ORDER: &[(&str, &str)] = &[
    ("Joker", "Joker"),
    ("Tarot", "Tarot"),
    ("Planet", "Planet"),
    ("Spectral", "Spectral"),
    ("Voucher", "Voucher"),
    ("Booster", "Booster"),
    ("Enhanced", "Enhanced"),
    ("Edition", "Edition"),
    ("Back", "Back"),
    ("Default", "Default"),
];

pub fn run(root: &Path, check: bool) -> Result<(), String> {
    let source = generate(root)?;
    let formatted = rustfmt(&source)?;
    let path = root.join(OUTPUT);
    if check {
        let current = fs::read_to_string(&path)
            .unwrap_or_default()
            .replace("\r\n", "\n");
        if current != formatted {
            return Err(format!(
                "{OUTPUT} is stale; run `cargo xtask codegen` and commit the result"
            ));
        }
        println!("{OUTPUT} is up to date");
    } else {
        fs::write(&path, formatted).map_err(|e| format!("cannot write {OUTPUT}: {e}"))?;
        println!("wrote {OUTPUT}");
    }
    Ok(())
}

fn rustfmt(source: &str) -> Result<String, String> {
    let mut child = Command::new("rustfmt")
        .args(["--edition", "2024", "--emit", "stdout"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("cannot run rustfmt: {e}"))?;
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(source.as_bytes())
        .map_err(|e| format!("rustfmt stdin: {e}"))?;
    let output = child
        .wait_with_output()
        .map_err(|e| format!("rustfmt: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "rustfmt failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(String::from_utf8(output.stdout)
        .map_err(|e| e.to_string())?
        .replace("\r\n", "\n"))
}

fn load(root: &Path, name: &str) -> Result<Json, String> {
    let path = root.join(DATA_DIR).join(name);
    let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

fn obj<'a>(v: &'a Json, what: &str) -> Result<&'a Map<String, Json>, String> {
    v.as_object()
        .ok_or_else(|| format!("{what}: expected an object"))
}

/// A Rust identifier for a game key: `j_8_ball` -> `J_8_BALL`.
fn const_name(key: &str) -> String {
    key.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn lit_str(s: &str) -> String {
    format!("{s:?}")
}

fn lit_f64(x: f64) -> String {
    format!("{x:?}")
}

fn opt<T>(v: Option<T>, f: impl FnOnce(T) -> String) -> String {
    match v {
        Some(x) => format!("Some({})", f(x)),
        None => "None".to_owned(),
    }
}

/// A `Value` literal for a JSON value (objects become key-sorted tables).
fn value(v: &Json) -> Result<String, String> {
    Ok(match v {
        Json::Bool(b) => format!("Value::Bool({b})"),
        Json::Number(n) => format!("Value::Num({})", lit_f64(n.as_f64().ok_or("bad number")?)),
        Json::String(s) => format!("Value::Str({})", lit_str(s)),
        Json::Array(items) => {
            let parts: Result<Vec<_>, _> = items.iter().map(value).collect();
            format!("Value::List(&[{}])", parts?.join(", "))
        }
        Json::Object(map) => format!("Value::Table({})", table(map)?),
        Json::Null => return Err("null in game data".to_owned()),
    })
}

/// A `&[(&str, Value)]` literal, sorted by key bytes.
fn table(map: &Map<String, Json>) -> Result<String, String> {
    let sorted: BTreeMap<&str, &Json> = map.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let mut parts = Vec::with_capacity(sorted.len());
    for (k, v) in sorted {
        parts.push(format!("({}, {})", lit_str(k), value(v)?));
    }
    Ok(format!("&[{}]", parts.join(", ")))
}

/// `config` that may be an object or an empty Lua table written as `[]`.
fn config_table(v: Option<&Json>) -> Result<String, String> {
    match v {
        None => Ok("&[]".to_owned()),
        Some(Json::Object(map)) => table(map),
        Some(Json::Array(items)) if items.is_empty() => Ok("&[]".to_owned()),
        Some(other) => Err(format!("unexpected config {other}")),
    }
}

fn str_field<'a>(o: &'a Map<String, Json>, k: &str) -> Option<&'a str> {
    o.get(k).and_then(Json::as_str)
}

fn num_field(o: &Map<String, Json>, k: &str) -> Option<f64> {
    o.get(k).and_then(Json::as_f64)
}

fn bool_field(o: &Map<String, Json>, k: &str) -> Option<bool> {
    o.get(k).and_then(Json::as_bool)
}

/// Sort `items` by Lua `order` and require the orders to be distinct: Lua sorts pools with the
/// unstable `table.sort`, so only distinct orders give one deterministic pool order.
fn sort_by_unique_order(pool: &str, items: &mut [(u16, String)]) -> Result<(), String> {
    items.sort();
    for w in items.windows(2) {
        if w[0].0 == w[1].0 {
            return Err(format!(
                "pool {pool}: {} and {} share order {}; Lua's unstable sort would make the pool order undefined",
                w[0].1, w[1].1, w[0].0
            ));
        }
    }
    Ok(())
}

fn generate(root: &Path) -> Result<String, String> {
    let mut out = String::new();
    out.push_str(
        "// @generated by `cargo xtask codegen` from jackdaw/engine/data/*.json. Do not edit.\n\
         //! Generated game tables (see `cargo xtask codegen`).\n\
         #![allow(clippy::all)]\n\
         use crate::types::*;\n\n",
    );
    gen_centers(root, &mut out)?;
    gen_blinds(root, &mut out)?;
    gen_tags(root, &mut out)?;
    gen_stakes_and_seals(root, &mut out)?;
    gen_cards(root, &mut out)?;
    gen_hands(root, &mut out)?;
    gen_challenges(root, &mut out)?;
    gen_blind_amounts(root, &mut out)?;
    Ok(out)
}

fn gen_centers(root: &Path, out: &mut String) -> Result<(), String> {
    let data = load(root, "centers.json")?;
    let centers = obj(&data, "centers.json")?;
    let set_rank = |set: Option<&str>| -> usize {
        set.and_then(|s| SET_ORDER.iter().position(|(n, _)| *n == s))
            .unwrap_or(SET_ORDER.len())
    };
    // Id order: set rank, then Lua order, then key.
    let mut ordered: Vec<(&String, &Map<String, Json>)> = Vec::new();
    for (k, v) in centers {
        ordered.push((k, obj(v, k)?));
    }
    ordered.sort_by(|(ka, a), (kb, b)| {
        let ra = (
            set_rank(str_field(a, "set")),
            num_field(a, "order").unwrap_or(f64::MAX),
            ka.as_str(),
        );
        let rb = (
            set_rank(str_field(b, "set")),
            num_field(b, "order").unwrap_or(f64::MAX),
            kb.as_str(),
        );
        ra.partial_cmp(&rb).expect("orders are numbers")
    });
    let id_of: BTreeMap<&str, usize> = ordered
        .iter()
        .enumerate()
        .map(|(i, (k, _))| (k.as_str(), i))
        .collect();

    out.push_str("/// `CenterId` constants named after the center keys.\npub mod centers {\n    use crate::types::CenterId;\n");
    for (i, (k, _)) in ordered.iter().enumerate() {
        writeln!(
            out,
            "    pub const {}: CenterId = CenterId({i});",
            const_name(k)
        )
        .unwrap();
    }
    out.push_str("}\n\n");

    writeln!(
        out,
        "/// `G.P_CENTERS`, indexed by `CenterId`.\npub static CENTERS: [Center; {}] = [",
        ordered.len()
    )
    .unwrap();
    for (k, c) in &ordered {
        let set = match str_field(c, "set") {
            Some(s) => SET_ORDER
                .iter()
                .find(|(n, _)| *n == s)
                .map(|(_, v)| format!("CenterSet::{v}"))
                .ok_or_else(|| format!("{k}: unknown set {s}"))?,
            None => "CenterSet::None".to_owned(),
        };
        let requires: Vec<String> = c
            .get("requires")
            .and_then(Json::as_array)
            .map(|a| a.iter().filter_map(Json::as_str).map(lit_str).collect())
            .unwrap_or_default();
        writeln!(
            out,
            "    Center {{ key: {key}, name: {name}, set: {set}, order: {order}, cost: {cost}, cost_mult: {cost_mult}, \
             rarity: {rarity}, effect: {effect}, label: {label}, consumeable: {consumeable}, \
             blueprint_compat: {bp}, eternal_compat: {ec}, perishable_compat: {pc}, unlocked: {unlocked}, \
             discovered: {discovered}, hidden: {hidden}, omit: {omit}, available: {available}, kind: {kind}, \
             weight: {weight}, freq: {freq}, max: {max}, stake: {stake}, requires: &[{requires}], \
             enhancement_gate: {gate}, yes_pool_flag: {yes}, no_pool_flag: {no}, soul_pos: {soul}, config: {config} }},",
            key = lit_str(k),
            name = lit_str(str_field(c, "name").unwrap_or("")),
            order = opt(num_field(c, "order"), |x| format!("{}", x as u16)),
            cost = opt(num_field(c, "cost"), lit_f64),
            cost_mult = opt(num_field(c, "cost_mult"), lit_f64),
            rarity = opt(num_field(c, "rarity"), |x| format!("{}", x as u8)),
            effect = opt(str_field(c, "effect"), lit_str),
            label = opt(str_field(c, "label"), lit_str),
            consumeable = bool_field(c, "consumeable").unwrap_or(false),
            bp = bool_field(c, "blueprint_compat").unwrap_or(false),
            ec = bool_field(c, "eternal_compat").unwrap_or(false),
            pc = bool_field(c, "perishable_compat").unwrap_or(false),
            unlocked = opt(bool_field(c, "unlocked"), |b| b.to_string()),
            discovered = opt(bool_field(c, "discovered"), |b| b.to_string()),
            hidden = bool_field(c, "hidden").unwrap_or(false),
            omit = bool_field(c, "omit").unwrap_or(false),
            available = opt(bool_field(c, "available"), |b| b.to_string()),
            kind = opt(str_field(c, "kind"), lit_str),
            weight = opt(num_field(c, "weight"), lit_f64),
            freq = opt(num_field(c, "freq"), lit_f64),
            max = opt(num_field(c, "max"), lit_f64),
            stake = opt(num_field(c, "stake"), |x| format!("{}", x as u8)),
            requires = requires.join(", "),
            gate = opt(str_field(c, "enhancement_gate"), lit_str),
            yes = opt(str_field(c, "yes_pool_flag"), lit_str),
            no = opt(str_field(c, "no_pool_flag"), lit_str),
            soul = c.contains_key("soul_pos"),
            config = config_table(c.get("config"))?,
        )
        .unwrap();
    }
    out.push_str("];\n\n");

    // Key lookup, sorted by key bytes.
    let keys: BTreeSet<&str> = id_of.keys().copied().collect();
    writeln!(out, "/// `(key, id)` sorted by key, for lookup by key.\npub static CENTER_KEYS: [(&str, CenterId); {}] = [", keys.len()).unwrap();
    for k in &keys {
        writeln!(out, "    ({}, CenterId({})),", lit_str(k), id_of[k]).unwrap();
    }
    out.push_str("];\n\n");

    // Pools (game.lua:815-842).
    let member = |c: &Map<String, Json>, set: &str| str_field(c, "set") == Some(set);
    let not_omitted = |c: &Map<String, Json>| {
        !bool_field(c, "omit").unwrap_or(false) && !bool_field(c, "skip_pool").unwrap_or(false)
    };
    let pool_items = |pred: &dyn Fn(&Map<String, Json>) -> bool| -> Vec<(u16, String)> {
        ordered
            .iter()
            .filter(|(_, c)| pred(c))
            .map(|(k, c)| {
                (
                    num_field(c, "order").map_or(u16::MAX, |o| o as u16),
                    (*k).clone(),
                )
            })
            .collect()
    };
    let mut emit_pool = |name: &str, doc: &str, items: Vec<(u16, String)>| {
        writeln!(
            out,
            "/// {doc}\npub static {name}: [CenterId; {}] = [{}];\n",
            items.len(),
            items
                .iter()
                .map(|(_, k)| format!("centers::{}", const_name(k)))
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap();
    };
    for (set, _) in SET_ORDER {
        let mut items = if *set == "Joker" {
            pool_items(&|c| member(c, "Joker"))
        } else {
            pool_items(&|c| member(c, set) && not_omitted(c))
        };
        if *set == "Back" || *set == "Edition" || *set == "Default" {
            // Lua sorts Back by order adjusted for unlocks (all unlocked: by order) and never
            // sorts Edition or Default; none of the three feeds an RNG draw.
            items.sort();
        } else {
            sort_by_unique_order(set, &mut items)?;
        }
        emit_pool(
            &format!("POOL_{}", const_name(set)),
            &format!("`G.P_CENTER_POOLS.{set}` in Lua order."),
            items,
        );
    }
    // Mixed pools: tarot and planet orders overlap, so Lua's order between equal orders is
    // undefined; ties are broken by set then key here. The game only draws from these with a
    // forced key.
    let mut tp = pool_items(&|c| member(c, "Tarot") || member(c, "Planet"));
    tp.sort_by(|a, b| {
        (
            a.0,
            set_rank(str_field(centers[&a.1].as_object().unwrap(), "set")),
            &a.1,
        )
            .cmp(&(
                b.0,
                set_rank(str_field(centers[&b.1].as_object().unwrap(), "set")),
                &b.1,
            ))
    });
    emit_pool(
        "POOL_TAROT_PLANET",
        "`G.P_CENTER_POOLS.Tarot_Planet` (equal orders: by set, then key).",
        tp,
    );
    let mut cons = pool_items(&|c| bool_field(c, "consumeable").unwrap_or(false));
    cons.sort_by(|a, b| {
        (
            a.0,
            set_rank(str_field(centers[&a.1].as_object().unwrap(), "set")),
            &a.1,
        )
            .cmp(&(
                b.0,
                set_rank(str_field(centers[&b.1].as_object().unwrap(), "set")),
                &b.1,
            ))
    });
    emit_pool(
        "POOL_CONSUMEABLES",
        "`G.P_CENTER_POOLS.Consumeables` (equal orders: by set, then key).",
        cons,
    );

    let mut rarity_names = Vec::new();
    for r in 1..=4u8 {
        let mut items =
            pool_items(&|c| member(c, "Joker") && num_field(c, "rarity") == Some(f64::from(r)));
        sort_by_unique_order(&format!("rarity {r}"), &mut items)?;
        let name = format!("POOL_JOKER_RARITY_{r}");
        emit_pool(
            &name,
            &format!("`G.P_JOKER_RARITY_POOLS[{r}]` in Lua order."),
            items,
        );
        rarity_names.push(name);
    }
    writeln!(out, "/// `G.P_JOKER_RARITY_POOLS` (index 0 is rarity 1).\npub static JOKER_RARITY_POOLS: [&[CenterId]; 4] = [{}];\n",
        rarity_names.iter().map(|n| format!("&{n}")).collect::<Vec<_>>().join(", ")).unwrap();
    Ok(())
}

fn gen_blinds(root: &Path, out: &mut String) -> Result<(), String> {
    let data = load(root, "blinds.json")?;
    let blinds = obj(&data, "blinds.json")?;
    let mut items: Vec<(u16, String)> = blinds
        .iter()
        .map(|(k, v)| {
            (
                num_field(v.as_object().unwrap(), "order").unwrap() as u16,
                k.clone(),
            )
        })
        .collect();
    sort_by_unique_order("blinds", &mut items)?;
    out.push_str("/// `BlindId` constants named after the blind keys.\npub mod blinds {\n    use crate::types::BlindId;\n");
    for (i, (_, k)) in items.iter().enumerate() {
        writeln!(
            out,
            "    pub const {}: BlindId = BlindId({i});",
            const_name(k)
        )
        .unwrap();
    }
    out.push_str("}\n\n");
    writeln!(
        out,
        "/// `G.P_BLINDS` in Lua order, indexed by `BlindId`.\npub static BLINDS: [Blind; {}] = [",
        items.len()
    )
    .unwrap();
    for (_, k) in &items {
        let b = obj(&blinds[k], k)?;
        let boss = match b.get("boss").and_then(Json::as_object) {
            Some(boss) => format!(
                "Some(BossRange {{ min: {}, max: {}, showdown: {} }})",
                num_field(boss, "min").unwrap_or(0.0) as u8,
                num_field(boss, "max").unwrap_or(0.0) as u8,
                bool_field(boss, "showdown").unwrap_or(false)
            ),
            None => "None".to_owned(),
        };
        writeln!(
            out,
            "    Blind {{ key: {}, name: {}, order: {}, mult: {}, dollars: {}, boss: {boss}, debuff: {} }},",
            lit_str(k),
            lit_str(str_field(b, "name").unwrap_or("")),
            num_field(b, "order").unwrap() as u8,
            lit_f64(num_field(b, "mult").unwrap()),
            lit_f64(num_field(b, "dollars").unwrap()),
            config_table(b.get("debuff"))?,
        )
        .unwrap();
    }
    out.push_str("];\n\n");
    Ok(())
}

fn gen_tags(root: &Path, out: &mut String) -> Result<(), String> {
    let data = load(root, "tags.json")?;
    let tags = obj(&data, "tags.json")?;
    let mut items: Vec<(u16, String)> = tags
        .iter()
        .map(|(k, v)| {
            (
                num_field(v.as_object().unwrap(), "order").unwrap() as u16,
                k.clone(),
            )
        })
        .collect();
    sort_by_unique_order("tags", &mut items)?;
    out.push_str("/// `TagId` constants named after the tag keys.\npub mod tags {\n    use crate::types::TagId;\n");
    for (i, (_, k)) in items.iter().enumerate() {
        writeln!(out, "    pub const {}: TagId = TagId({i});", const_name(k)).unwrap();
    }
    out.push_str("}\n\n");
    writeln!(out, "/// `G.P_TAGS` in Lua order (also `G.P_CENTER_POOLS.Tag`), indexed by `TagId`.\npub static TAGS: [Tag; {}] = [", items.len()).unwrap();
    for (_, k) in &items {
        let t = obj(&tags[k], k)?;
        writeln!(
            out,
            "    Tag {{ key: {}, name: {}, order: {}, min_ante: {}, requires: {}, config: {} }},",
            lit_str(k),
            lit_str(str_field(t, "name").unwrap_or("")),
            num_field(t, "order").unwrap() as u8,
            opt(num_field(t, "min_ante"), |x| format!("{}", x as u8)),
            opt(str_field(t, "requires"), lit_str),
            config_table(t.get("config"))?,
        )
        .unwrap();
    }
    out.push_str("];\n\n");
    Ok(())
}

fn gen_stakes_and_seals(root: &Path, out: &mut String) -> Result<(), String> {
    let data = load(root, "stakes.json")?;
    let stakes = obj(&data, "stakes.json")?;
    let mut items: Vec<(u16, String)> = stakes
        .iter()
        .map(|(k, v)| {
            (
                num_field(v.as_object().unwrap(), "order").unwrap() as u16,
                k.clone(),
            )
        })
        .collect();
    sort_by_unique_order("stakes", &mut items)?;
    writeln!(
        out,
        "/// `G.P_STAKES` in Lua order (index `stake - 1`).\npub static STAKES: [Stake; {}] = [",
        items.len()
    )
    .unwrap();
    for (_, k) in &items {
        let s = obj(&stakes[k], k)?;
        writeln!(
            out,
            "    Stake {{ key: {}, name: {}, order: {}, stake_level: {} }},",
            lit_str(k),
            lit_str(str_field(s, "name").unwrap_or("")),
            num_field(s, "order").unwrap() as u8,
            num_field(s, "stake_level").unwrap() as u8,
        )
        .unwrap();
    }
    out.push_str("];\n\n");

    let data = load(root, "seals.json")?;
    let seals = obj(&data, "seals.json")?;
    let mut items: Vec<(u16, String)> = seals
        .iter()
        .map(|(k, v)| {
            (
                num_field(v.as_object().unwrap(), "order").unwrap() as u16,
                k.clone(),
            )
        })
        .collect();
    sort_by_unique_order("seals", &mut items)?;
    writeln!(out, "/// `G.P_SEALS` in Lua order (`G.P_CENTER_POOLS.Seal`).\npub static SEALS: [SealProto; {}] = [", items.len()).unwrap();
    for (o, k) in &items {
        writeln!(out, "    SealProto {{ key: {}, order: {o} }},", lit_str(k)).unwrap();
    }
    out.push_str("];\n\n");
    Ok(())
}

fn gen_cards(root: &Path, out: &mut String) -> Result<(), String> {
    let data = load(root, "cards.json")?;
    let cards = obj(&data, "cards.json")?;
    let keys: BTreeSet<&String> = cards.keys().collect(); // bytewise key order
    writeln!(out, "/// `G.P_CARDS` in bytewise key order, indexed by `CardId`.\npub static P_CARDS: [PlayingCard; {}] = [", keys.len()).unwrap();
    for k in &keys {
        let c = obj(&cards[*k], k)?;
        writeln!(
            out,
            "    PlayingCard {{ key: {}, name: {}, suit: {}, value: {} }},",
            lit_str(k),
            lit_str(str_field(c, "name").unwrap_or("")),
            lit_str(str_field(c, "suit").ok_or("card suit")?),
            lit_str(str_field(c, "value").ok_or("card value")?),
        )
        .unwrap();
    }
    out.push_str("];\n\n");
    Ok(())
}

fn gen_hands(root: &Path, out: &mut String) -> Result<(), String> {
    let data = load(root, "hands.json")?;
    let hands = obj(&data, "hands.json")?;
    let mut items: Vec<(u16, String)> = hands
        .iter()
        .map(|(k, v)| {
            (
                num_field(v.as_object().unwrap(), "order").unwrap() as u16,
                k.clone(),
            )
        })
        .collect();
    sort_by_unique_order("hands", &mut items)?;
    writeln!(out, "/// `G.GAME.hands` base rows in Lua `order` (Flush Five .. High Card).\npub static HANDS: [HandBase; {}] = [", items.len()).unwrap();
    for (o, k) in &items {
        let h = obj(&hands[k], k)?;
        let example: Vec<String> = h
            .get("example")
            .and_then(Json::as_array)
            .ok_or("hand example")?
            .iter()
            .map(|e| {
                let pair = e.as_array().expect("example pair");
                format!(
                    "({}, {})",
                    lit_str(pair[0].as_str().unwrap()),
                    pair[1].as_bool().unwrap()
                )
            })
            .collect();
        writeln!(
            out,
            "    HandBase {{ name: {}, order: {o}, visible: {}, s_chips: {}, s_mult: {}, l_chips: {}, l_mult: {}, example: &[{}] }},",
            lit_str(k),
            bool_field(h, "visible").unwrap(),
            lit_f64(num_field(h, "s_chips").unwrap()),
            lit_f64(num_field(h, "s_mult").unwrap()),
            lit_f64(num_field(h, "l_chips").unwrap()),
            lit_f64(num_field(h, "l_mult").unwrap()),
            example.join(", "),
        )
        .unwrap();
    }
    out.push_str("];\n\n");
    Ok(())
}

fn gen_challenges(root: &Path, out: &mut String) -> Result<(), String> {
    let data = load(root, "challenges.json")?;
    let list = data
        .as_array()
        .ok_or("challenges.json: expected an array")?;
    writeln!(
        out,
        "/// `G.CHALLENGES` in game order.\npub static CHALLENGES: [Challenge; {}] = [",
        list.len()
    )
    .unwrap();
    for ch in list {
        let c = obj(ch, "challenge")?;
        writeln!(
            out,
            "    Challenge {{ id: {}, name: {}, data: Value::Table({}) }},",
            lit_str(str_field(c, "id").ok_or("challenge id")?),
            lit_str(str_field(c, "name").unwrap_or("")),
            table(c)?,
        )
        .unwrap();
    }
    out.push_str("];\n\n");
    Ok(())
}

fn gen_blind_amounts(root: &Path, out: &mut String) -> Result<(), String> {
    let data = load(root, "blind_amounts.json")?;
    let tables = obj(&data, "blind_amounts.json")?;
    out.push_str(
        "/// `get_blind_amount(ante)` as LuaJIT computes it, as `f64` bits: index `[scaling - 1][ante + 1]`\n\
         /// for antes -1..=80 (endless antes overflow to NaN from ante 39 at scaling 1).\n",
    );
    let rows: Vec<&Vec<Json>> = (1..=3)
        .map(|s| tables[&s.to_string()].as_array().ok_or("blind amount rows"))
        .collect::<Result<_, _>>()?;
    let n = rows[0].len();
    writeln!(out, "pub static BLIND_AMOUNT_BITS: [[u64; {n}]; 3] = [").unwrap();
    for (s, row) in rows.iter().enumerate() {
        if row.len() != n {
            return Err(format!(
                "scaling {} has {} antes, expected {n}",
                s + 1,
                row.len()
            ));
        }
        let parts: Vec<String> = row
            .iter()
            .map(|r| format!("0x{}", r["bits"].as_str().unwrap()))
            .collect();
        writeln!(out, "    [{}],", parts.join(", ")).unwrap();
    }
    out.push_str("];\n");
    Ok(())
}

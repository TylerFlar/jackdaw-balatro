//! End-to-end: a session driven the way BalatroBot's own fixtures drive the live game
//! (third_party/balatrobot/v1.5.2/tests/fixtures/fixtures.json).

use jackdaw_bot::json::{self, Json};
use jackdaw_bot::{Session, SessionConfig};

fn call(s: &mut Session, method: &str, params: &str) -> Result<Json, (String, String)> {
    let v = json::decode(params.as_bytes()).expect("params decode");
    s.dispatch(method, v.as_ref())
        .map_err(|e| (e.name.name().to_owned(), e.message))
}

fn ok(s: &mut Session, method: &str, params: &str) -> Json {
    call(s, method, params).unwrap_or_else(|e| panic!("{method} {params}: {e:?}"))
}

fn err(s: &mut Session, method: &str, params: &str) -> (String, String) {
    call(s, method, params).expect_err("an error")
}

fn path<'a>(v: &'a Json, keys: &[&str]) -> &'a Json {
    keys.iter()
        .fold(v, |v, k| v.get(k).unwrap_or_else(|| panic!("no {k}")))
}

fn cards(v: &Json) -> &Vec<Json> {
    match v.get("cards") {
        Some(Json::Arr(c)) => c,
        other => panic!("cards: {other:?}"),
    }
}

/// `menu | start TEST123 | select | set chips 1000 | play [0] | cash_out`.
fn shop_test123() -> Session {
    let mut s = Session::new(SessionConfig::default());
    ok(&mut s, "menu", "{}");
    ok(
        &mut s,
        "start",
        r#"{"deck": "RED", "stake": "WHITE", "seed": "TEST123"}"#,
    );
    ok(&mut s, "select", "{}");
    ok(&mut s, "set", r#"{"chips": 1000}"#);
    let v = ok(&mut s, "play", r#"{"cards": [0]}"#);
    assert_eq!(path(&v, &["state"]), &Json::str("ROUND_EVAL"));
    let v = ok(&mut s, "cash_out", "{}");
    assert_eq!(path(&v, &["state"]), &Json::str("SHOP"));
    s
}

#[test]
fn the_first_shop_of_test123() {
    let mut s = shop_test123();
    let v = s.view();
    // test_buy.py: shop card 0 is a joker, card 1 a planet; one voucher; Buffoon + Standard.
    let shop = path(&v, &["shop"]);
    assert_eq!(path(&cards(shop)[0], &["set"]), &Json::str("JOKER"));
    assert_eq!(path(&cards(shop)[1], &["set"]), &Json::str("PLANET"));
    assert_eq!(path(&v, &["vouchers", "count"]), &Json::Num(1.0));
    let packs = cards(path(&v, &["packs"]));
    assert_eq!(path(&packs[0], &["label"]), &Json::str("Buffoon Pack"));
    assert_eq!(path(&packs[1], &["label"]), &Json::str("Standard Pack"));
    // test_gamestate.py: the first reroll costs 5, the next 6.
    assert_eq!(path(&v, &["round", "reroll_cost"]), &Json::Num(5.0));
    let after = ok(&mut s, "reroll", "{}");
    assert_eq!(path(&after, &["round", "reroll_cost"]), &Json::Num(6.0));
}

#[test]
fn buy_errors_match_balatrobot() {
    let mut s = shop_test123();
    assert_eq!(
        err(&mut s, "buy", "{}"),
        (
            "BAD_REQUEST".into(),
            "Invalid arguments. You must provide one of: card, voucher, pack".into()
        )
    );
    assert_eq!(
        err(&mut s, "buy", r#"{"card": 999}"#).1,
        "Card index out of range. Index: 999, Available cards: 2"
    );
    ok(&mut s, "set", r#"{"money": 0}"#);
    assert_eq!(
        err(&mut s, "buy", r#"{"card": 0}"#).1,
        "Card is not affordable. Cost: 5, Available money: 0"
    );
    assert_eq!(
        err(&mut s, "buy", r#"{"card": "x"}"#).1,
        "Field 'card' must be an integer"
    );
}

#[test]
fn leaving_the_shop_keeps_its_areas() {
    let mut s = shop_test123();
    let v = ok(&mut s, "next_round", "{}");
    assert_eq!(path(&v, &["state"]), &Json::str("BLIND_SELECT"));
    let shop = path(&v, &["shop"]);
    assert_eq!(
        path(shop, &["count"]),
        &Json::Num(2.0),
        "the last frame's count"
    );
    assert!(cards(shop).is_empty(), "the removed area has no cards");
    assert_eq!(path(&v, &["vouchers", "count"]), &Json::Num(1.0));
}

#[test]
fn save_and_load_round_trip() {
    let mut s = shop_test123();
    let dir = std::env::temp_dir().join(format!("jackdaw_bot_save_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("shop.jkr");
    let p = json::encode(&Json::str(file.to_string_lossy().into_owned())).unwrap();
    let before = s.view();
    ok(&mut s, "save", &format!(r#"{{"path": {p}}}"#));
    ok(&mut s, "menu", "{}");
    let r = ok(&mut s, "load", &format!(r#"{{"path": {p}}}"#));
    assert_eq!(path(&r, &["success"]), &Json::Bool(true));
    let after = s.view();
    assert_eq!(path(&after, &["shop"]), path(&before, &["shop"]));
    assert_eq!(path(&after, &["money"]), path(&before, &["money"]));
    // A file that isn't a snapshot deletes the run first.
    std::fs::write(&file, b"CORRUPTED_SAVE_FILE_FOR_TESTING").unwrap();
    assert_eq!(
        err(&mut s, "load", &format!(r#"{{"path": {p}}}"#)),
        ("INTERNAL_ERROR".into(), "Invalid save file format".into())
    );
    assert_eq!(path(&s.view(), &["state"]), &Json::str("UNKNOWN"));
    assert_eq!(
        err(&mut s, "save", &format!(r#"{{"path": {p}}}"#)).0,
        "INVALID_STATE"
    );
    ok(&mut s, "menu", "{}");
    assert_eq!(path(&s.view(), &["state"]), &Json::str("MENU"));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn play_and_discard_checks() {
    let mut s = Session::new(SessionConfig::default());
    ok(
        &mut s,
        "start",
        r#"{"deck": "RED", "stake": "WHITE", "seed": "TEST123"}"#,
    );
    ok(&mut s, "select", "{}");
    assert_eq!(
        err(&mut s, "play", r#"{"cards": []}"#).1,
        "Must provide at least one card to play"
    );
    assert_eq!(
        err(&mut s, "play", r#"{"cards": [0, 1, 2, 3, 4, 5]}"#).1,
        "You can only play 5 cards"
    );
    assert_eq!(
        err(&mut s, "play", r#"{"cards": [99]}"#).1,
        "Invalid card index: 99"
    );
    assert_eq!(err(&mut s, "play", r#"{"cards": [0, 0]}"#).0, "NOT_ALLOWED");
    // test_gamestate.py: TEST123's first card scores 16, then 31 after a discard.
    let v = ok(&mut s, "play", r#"{"cards": [0]}"#);
    assert_eq!(path(&v, &["round", "chips"]), &Json::Num(16.0));
    let v = ok(&mut s, "discard", r#"{"cards": [0]}"#);
    assert_eq!(path(&v, &["round", "discards_left"]), &Json::Num(3.0));
    let v = ok(&mut s, "play", r#"{"cards": [0]}"#);
    assert_eq!(path(&v, &["round", "chips"]), &Json::Num(31.0));
    ok(&mut s, "set", r#"{"discards": 0}"#);
    assert_eq!(
        err(&mut s, "discard", r#"{"cards": [0]}"#).1,
        "No discards left"
    );
}

#[test]
fn rearrange_applies_the_whole_permutation() {
    let mut s = Session::new(SessionConfig::default());
    ok(
        &mut s,
        "start",
        r#"{"deck": "RED", "stake": "WHITE", "seed": "TEST123"}"#,
    );
    let v = ok(&mut s, "select", "{}");
    let ids: Vec<Json> = cards(path(&v, &["hand"]))
        .iter()
        .map(|c| c.get("id").unwrap().clone())
        .collect();
    let v = ok(&mut s, "rearrange", r#"{"hand": [7, 6, 5, 4, 3, 2, 1, 0]}"#);
    let after: Vec<Json> = cards(path(&v, &["hand"]))
        .iter()
        .map(|c| c.get("id").unwrap().clone())
        .collect();
    let reversed: Vec<Json> = ids.into_iter().rev().collect();
    assert_eq!(after, reversed);
    assert_eq!(
        err(&mut s, "rearrange", r#"{"hand": [0, 1, 1, 3, 4, 5, 6, 7]}"#).1,
        "Duplicate index in hand: 1"
    );
}

#[test]
fn add_validation_and_cards() {
    let mut s = shop_test123();
    assert_eq!(
        err(&mut s, "add", r#"{"key": "x_bad"}"#).1,
        "Invalid card key format. Expected: joker (j_*), consumable (c_*), voucher (v_*), or playing card (SUIT_RANK)"
    );
    assert_eq!(
        err(&mut s, "add", r#"{"key": "H_A"}"#),
        (
            "INVALID_STATE".into(),
            "Playing cards can only be added in SELECTING_HAND state".into()
        )
    );
    assert_eq!(
        err(&mut s, "add", r#"{"key": "j_nope"}"#).1,
        "Failed to add card: j_nope"
    );
    let v = ok(
        &mut s,
        "add",
        r#"{"key": "j_joker", "eternal": true, "perishable": 5}"#,
    );
    let j = cards(path(&v, &["jokers"])).last().unwrap().clone();
    assert_eq!(path(&j, &["modifier", "eternal"]), &Json::Bool(true));
    assert_eq!(path(&j, &["modifier", "perishable"]), &Json::Num(5.0));
    assert_eq!(path(&j, &["label"]), &Json::str("Joker"));
    let v = ok(&mut s, "add", r#"{"key": "v_overstock_norm"}"#);
    assert_eq!(path(&v, &["vouchers", "count"]), &Json::Num(2.0));
    let v = ok(
        &mut s,
        "add",
        r#"{"key": "c_pluto", "edition": "NEGATIVE"}"#,
    );
    let c = cards(path(&v, &["consumables"])).last().unwrap().clone();
    assert_eq!(path(&c, &["modifier", "edition"]), &Json::str("NEGATIVE"));
    assert_eq!(
        err(&mut s, "add", r#"{"key": "p_arcana_normal_1"}"#),
        (
            "NOT_ALLOWED".into(),
            "Cannot add pack, shop booster slots are full".into()
        )
    );
}

#[test]
fn the_won_flag() {
    // test_gamestate.py::test_won_true_extraction.
    let mut s = Session::new(SessionConfig::default());
    ok(
        &mut s,
        "start",
        r#"{"deck": "RED", "stake": "WHITE", "seed": "TEST123"}"#,
    );
    ok(&mut s, "skip", "{}");
    let v = ok(&mut s, "skip", "{}");
    assert_eq!(
        path(&v, &["blinds", "small", "status"]),
        &Json::str("SKIPPED")
    );
    assert_eq!(
        err(&mut s, "skip", "{}"),
        ("NOT_ALLOWED".into(), "Cannot skip Boss blind".into())
    );
    ok(&mut s, "select", "{}");
    ok(&mut s, "set", r#"{"ante": 8, "chips": 10000000}"#);
    let v = ok(&mut s, "play", r#"{"cards": [0]}"#);
    assert_eq!(path(&v, &["won"]), &Json::Bool(true));
}

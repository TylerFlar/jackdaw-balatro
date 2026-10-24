//! Hand evaluation against the game's own code: `tests/fixtures/hand_eval_luajit.json` holds
//! `evaluate_poker_hand` / `get_poker_hand_info` results from LuaJIT on generated hands
//! (`scripts/luajit_hand_eval_oracle.lua`). Every poker hand's groups must match exactly.

use std::path::PathBuf;

use jackdaw_core::hand_eval::{EvalCard, EvalFlags, FLUSH_SUITS, HandType, get_poker_hand_info};
use serde_json::Value;

fn fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/hand_eval_luajit.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).expect("valid JSON")
}

fn suit_bit(suit: &str) -> u8 {
    1 << FLUSH_SUITS.iter().position(|s| *s == suit).expect("suit")
}

fn is_red(suit: &str) -> bool {
    suit == "Hearts" || suit == "Diamonds"
}

/// `Card:is_suit(suit, nil, true)` for each flush suit (card.lua).
fn flush_suits(card: &Value, smeared: bool) -> u8 {
    let suit = card["suit"].as_str().unwrap();
    match card["enh"].as_str().unwrap() {
        "stone" => 0,
        "wild" if !card["debuff"].as_bool().unwrap() => 0b1111,
        _ if smeared => FLUSH_SUITS
            .iter()
            .filter(|s| is_red(s) == is_red(suit))
            .fold(0, |acc, s| acc | suit_bit(s)),
        _ => suit_bit(suit),
    }
}

fn groups_1based(groups: &[jackdaw_core::hand_eval::Group]) -> Vec<Vec<u64>> {
    groups
        .iter()
        .map(|g| g.as_slice().iter().map(|&i| u64::from(i) + 1).collect())
        .collect()
}

#[test]
fn matches_the_games_hand_evaluation() {
    let data = fixture();
    let order: Vec<&str> = data["hand_order"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let names: Vec<&str> = HandType::ALL.iter().map(|h| h.name()).collect();
    assert_eq!(order, names, "hand order");
    let cases = data["cases"].as_array().unwrap();
    for (n, case) in cases.iter().enumerate() {
        let flags: Vec<&str> = case["flags"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap())
            .collect();
        let smeared = flags.contains(&"Smeared Joker");
        let eval_flags = EvalFlags {
            four_fingers: flags.contains(&"Four Fingers"),
            shortcut: flags.contains(&"Shortcut"),
        };
        let hand: Vec<EvalCard> = case["cards"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| EvalCard {
                id: (c["enh"] != "stone").then(|| c["rank"].as_u64().unwrap() as u8),
                flush_suits: flush_suits(c, smeared),
                nominal: c["nominal"].as_f64().unwrap(),
            })
            .collect();
        let info = get_poker_hand_info(&hand, eval_flags);
        let text = info.hand.map_or("NULL", HandType::name);
        assert_eq!(text, case["text"].as_str().unwrap(), "case {n}: {case}");
        let scoring: Vec<u64> = info
            .scoring
            .as_slice()
            .iter()
            .map(|&i| u64::from(i) + 1)
            .collect();
        let want_scoring: Vec<u64> = case["scoring"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(scoring, want_scoring, "case {n} scoring: {case}");
        for (t, hand_type) in HandType::ALL.iter().enumerate() {
            let want: Vec<Vec<u64>> = case["hands"][t]
                .as_array()
                .unwrap()
                .iter()
                .map(|g| {
                    g.as_array()
                        .unwrap()
                        .iter()
                        .map(|v| v.as_u64().unwrap())
                        .collect()
                })
                .collect();
            let got = groups_1based(info.poker_hands.get(*hand_type).as_slice());
            assert_eq!(got, want, "case {n} {}: {case}", hand_type.name());
        }
    }
}

#[test]
fn get_nominal_matches_the_game() {
    use jackdaw_core::card::{Base, Rank, Suit, nominal};
    for case in fixture()["cases"].as_array().unwrap() {
        for c in case["cards"].as_array().unwrap() {
            let suit = Suit::from_name(c["suit"].as_str().unwrap()).unwrap();
            let rank = Rank::from_id(c["rank"].as_u64().unwrap() as u8).unwrap();
            let original = Suit::from_name(c["original_suit"].as_str().unwrap()).unwrap();
            let base = Base {
                front: Base::front_for(suit, rank),
                suit,
                value: rank,
                suit_nominal_original: original.nominal_original(),
                original_value: Some(rank),
                times_played: 0.0,
            };
            let stone = c["enh"] == "stone";
            let got = nominal(&base, stone, c["unique_val"].as_f64().unwrap(), false);
            assert_eq!(
                got.to_bits(),
                c["nominal"].as_f64().unwrap().to_bits(),
                "{c}"
            );
        }
    }
}

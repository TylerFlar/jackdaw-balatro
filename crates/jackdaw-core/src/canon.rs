//! The canonical byte encoding of a snapshot and its SHA-256 digest (`docs/snapshot-schema.md`,
//! "Canonical encoding"). It matches `tests/parity/canon.py` byte for byte, so a state's digest
//! is the same whichever engine produced it.
//!
//! Every number is a big-endian IEEE double tagged `D` (integers are converted, `-0` is `0`, and
//! every NaN is `0x7FF8000000000000`); a `{"$f": ...}` marker is the number it stands for.
//! Strings are `S` + length + UTF-8, lists `[` + length + items, objects `{` + length + entries
//! sorted by their keys' UTF-8 bytes, and `N` / `T` / `F` are null, true and false.

use serde_json::Value;
use sha2::{Digest, Sha256};

const CANONICAL_NAN: u64 = 0x7FF8_0000_0000_0000;

/// The canonical encoding of a snapshot tree.
pub fn encode(v: &Value) -> Vec<u8> {
    let mut out = Vec::with_capacity(1 << 14);
    encode_into(v, &mut out);
    out
}

/// The SHA-256 hex digest of [`encode`].
pub fn digest(v: &Value) -> String {
    let hash = Sha256::digest(encode(v));
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

fn number(x: f64, out: &mut Vec<u8>) {
    out.push(b'D');
    let bits = if x.is_nan() {
        CANONICAL_NAN
    } else if x == 0.0 {
        0
    } else {
        x.to_bits()
    };
    out.extend_from_slice(&bits.to_be_bytes());
}

fn length(n: usize, out: &mut Vec<u8>) {
    out.extend_from_slice(&(n as u32).to_be_bytes());
}

fn string(s: &str, out: &mut Vec<u8>) {
    out.push(b'S');
    length(s.len(), out);
    out.extend_from_slice(s.as_bytes());
}

/// The number a `{"$f": "inf" | "-inf" | "nan"}` marker stands for.
fn marker(map: &serde_json::Map<String, Value>) -> Option<f64> {
    if map.len() != 1 {
        return None;
    }
    match map.get("$f")?.as_str()? {
        "inf" => Some(f64::INFINITY),
        "-inf" => Some(f64::NEG_INFINITY),
        "nan" => Some(f64::NAN),
        _ => None,
    }
}

fn encode_into(v: &Value, out: &mut Vec<u8>) {
    match v {
        Value::Null => out.push(b'N'),
        Value::Bool(true) => out.push(b'T'),
        Value::Bool(false) => out.push(b'F'),
        Value::Number(n) => number(n.as_f64().expect("a JSON number"), out),
        Value::String(s) => string(s, out),
        Value::Array(items) => {
            out.push(b'[');
            length(items.len(), out);
            for item in items {
                encode_into(item, out);
            }
        }
        Value::Object(map) => {
            if let Some(x) = marker(map) {
                return number(x, out);
            }
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
            out.push(b'{');
            length(entries.len(), out);
            for (k, item) in entries {
                string(k, out);
                encode_into(item, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn numbers_are_doubles_with_one_zero_and_one_nan() {
        assert_eq!(encode(&json!(1)), encode(&json!(1.0)));
        assert_eq!(encode(&json!(-0.0)), encode(&json!(0)));
        assert_eq!(encode(&json!({"$f": "nan"})), {
            let mut v = vec![b'D'];
            v.extend_from_slice(&CANONICAL_NAN.to_be_bytes());
            v
        });
    }

    #[test]
    fn objects_sort_their_keys() {
        assert_eq!(
            encode(&json!({"b": 1, "a": 2})),
            encode(&json!({"a": 2, "b": 1}))
        );
    }
}

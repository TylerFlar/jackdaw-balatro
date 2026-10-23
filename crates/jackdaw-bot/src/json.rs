//! JSON as BalatroBot reads and writes it: Steamodded's bundled rxi `json.lua` (v0.1.2,
//! `smods/libs/json/json.lua`), ported function for function.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/core/server.lua`, which `require`s that module for every request and response.
//!
//! Encoding ([`encode`]):
//! - a table with no keys is written as `[]`, so an empty object and an empty array look the
//!   same on the wire ([`Json::Obj`] with no entries encodes as `[]`);
//! - numbers are written with `string.format("%.14g")` ([`crate::lua::tostring`]);
//! - NaN and ±inf make the encoder raise `unexpected number value '...'`. BalatroBot then sends
//!   nothing and the client times out; the sim answers with an `INTERNAL_ERROR` carrying that
//!   message instead (see `docs/balatrobot.md`);
//! - strings escape `"`, `\` and control characters only (`\b \f \n \r \t`, else `\u00xx`).
//!
//! Decoding ([`decode`]) keeps rxi's leniency: a trailing comma closes an array or object, a
//! number token is whatever Lua's `tonumber` accepts (`0x10`, `01`, `1.`), `null` is `nil`.

use crate::lua::{self, Table, Value};

/// A response value, built the way the Lua endpoints build their tables. A Lua field set to
/// `nil` is simply left out of an [`Json::Obj`].
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    pub fn str(s: impl Into<String>) -> Json {
        Json::Str(s.into())
    }

    /// An object from `(key, value)` pairs.
    pub fn obj<K: Into<String>>(entries: impl IntoIterator<Item = (K, Json)>) -> Json {
        Json::Obj(entries.into_iter().map(|(k, v)| (k.into(), v)).collect())
    }

    /// The value at `key` of an object.
    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(entries) => entries.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    /// Insert or replace `key` in an object.
    pub fn set(&mut self, key: &str, value: Json) {
        if let Json::Obj(entries) = self {
            match entries.iter_mut().find(|(k, _)| k == key) {
                Some(slot) => slot.1 = value,
                None => entries.push((key.to_owned(), value)),
            }
        }
    }
}

/// Why the encoder raised (rxi's `error(...)` text).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodeError(pub String);

impl std::fmt::Display for EncodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for EncodeError {}

/// `json.encode(v)`.
pub fn encode(v: &Json) -> Result<String, EncodeError> {
    let mut out = String::with_capacity(4096);
    encode_into(v, &mut out)?;
    Ok(out)
}

fn encode_into(v: &Json, out: &mut String) -> Result<(), EncodeError> {
    match v {
        Json::Null => out.push_str("null"),
        Json::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Json::Num(x) => out.push_str(&encode_number(*x)?),
        Json::Str(s) => encode_string(s, out),
        Json::Arr(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                encode_into(item, out)?;
            }
            out.push(']');
        }
        // `rawget(val, 1) ~= nil or next(val) == nil`: an empty table takes the array branch.
        Json::Obj(entries) if entries.is_empty() => out.push_str("[]"),
        Json::Obj(entries) => {
            out.push('{');
            for (i, (k, item)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                encode_string(k, out);
                out.push(':');
                encode_into(item, out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

/// rxi's `encode_number`: `%.14g`, raising on NaN and infinities.
fn encode_number(x: f64) -> Result<String, EncodeError> {
    if !x.is_finite() {
        return Err(EncodeError(format!(
            "unexpected number value '{}'",
            lua::tostring(x)
        )));
    }
    Ok(lua::tostring(x))
}

/// rxi's `encode_string`: escapes `[%z\1-\31\\"]`.
fn encode_string(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\u{8}' => out.push_str("\\b"),
            '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 32 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

/// A decoded Lua value as a response value: what `json.encode` makes of a table the endpoint
/// got from `json.decode` (the test endpoints echo their arguments, `rpc.discover` re-encodes
/// the decoded spec). Raises as rxi does on a mixed or sparse table.
pub fn from_value(v: &Value) -> Result<Json, EncodeError> {
    Ok(match v {
        Value::Bool(b) => Json::Bool(*b),
        Value::Num(x) => Json::Num(*x),
        Value::Str(s) => Json::Str(s.clone()),
        Value::Table(t) => from_table(t)?,
    })
}

fn from_table(t: &Table) -> Result<Json, EncodeError> {
    let (seq, map) = t.parts();
    let has_first = seq.first().is_some_and(Option::is_some);
    if has_first || t.is_empty() {
        if !map.is_empty() {
            return Err(EncodeError(
                "invalid table: mixed or invalid key types".to_owned(),
            ));
        }
        if seq.iter().any(Option::is_none) {
            return Err(EncodeError("invalid table: sparse array".to_owned()));
        }
        return seq
            .iter()
            .map(|v| from_value(v.as_ref().expect("no holes")))
            .collect::<Result<_, _>>()
            .map(Json::Arr);
    }
    if seq.iter().any(Option::is_some) {
        return Err(EncodeError(
            "invalid table: mixed or invalid key types".to_owned(),
        ));
    }
    map.iter()
        .map(|(k, v)| Ok((k.clone(), from_value(v)?)))
        .collect::<Result<_, _>>()
        .map(Json::Obj)
}

/// Why the decoder raised (rxi's `decode_error` text, without position).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeError(pub String);

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for DecodeError {}

fn err<T>(msg: impl Into<String>) -> Result<T, DecodeError> {
    Err(DecodeError(msg.into()))
}

fn is_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\r' | b'\n')
}

fn is_delim(b: u8) -> bool {
    is_space(b) || matches!(b, b']' | b'}' | b',')
}

/// The first index at or after `i` whose byte is not a space.
fn skip_space(s: &[u8], mut i: usize) -> usize {
    while i < s.len() && is_space(s[i]) {
        i += 1;
    }
    i
}

/// The first index at or after `i` whose byte is a delimiter.
fn to_delim(s: &[u8], mut i: usize) -> usize {
    while i < s.len() && !is_delim(s[i]) {
        i += 1;
    }
    i
}

/// `json.decode(str)`: `Ok(None)` is `nil` (the text `null`).
pub fn decode(s: &[u8]) -> Result<Option<Value>, DecodeError> {
    let (v, i) = parse(s, skip_space(s, 0))?;
    if skip_space(s, i) < s.len() {
        return err("trailing garbage");
    }
    Ok(v)
}

fn parse(s: &[u8], i: usize) -> Result<(Option<Value>, usize), DecodeError> {
    match s.get(i) {
        Some(b'"') => parse_string(s, i).map(|(v, i)| (Some(Value::Str(v)), i)),
        Some(b'0'..=b'9' | b'-') => parse_number(s, i).map(|(x, i)| (Some(Value::Num(x)), i)),
        Some(b't' | b'f' | b'n') => parse_literal(s, i),
        Some(b'[') => parse_array(s, i).map(|(t, i)| (Some(Value::Table(t)), i)),
        Some(b'{') => parse_object(s, i).map(|(t, i)| (Some(Value::Table(t)), i)),
        Some(&c) => err(format!("unexpected character '{}'", char::from(c))),
        None => err("unexpected character ''"),
    }
}

fn parse_string(s: &[u8], i: usize) -> Result<(String, usize), DecodeError> {
    let mut res: Vec<u8> = Vec::new();
    let mut j = i + 1;
    while j < s.len() {
        let x = s[j];
        if x < 32 {
            return err("control character in string");
        } else if x == b'\\' {
            j += 1;
            let c = s.get(j).copied();
            if c == Some(b'u') {
                let (bytes, used) = parse_unicode_escape(&s[j + 1..])?;
                res.extend_from_slice(&bytes);
                j += used;
            } else {
                let mapped = match c {
                    Some(b'\\') => b'\\',
                    Some(b'/') => b'/',
                    Some(b'"') => b'"',
                    Some(b'b') => 8,
                    Some(b'f') => 12,
                    Some(b'n') => b'\n',
                    Some(b'r') => b'\r',
                    Some(b't') => b'\t',
                    Some(c) => {
                        return err(format!("invalid escape char '{}' in string", char::from(c)));
                    }
                    None => return err("invalid escape char '' in string"),
                };
                res.push(mapped);
            }
        } else if x == b'"' {
            return Ok((String::from_utf8_lossy(&res).into_owned(), j + 1));
        } else {
            res.push(x);
        }
        j += 1;
    }
    err("expected closing quote for string")
}

/// `\uXXXX` (or a `\uD8xx\uDCxx` surrogate pair) after the `u`: the UTF-8 bytes and how many
/// bytes of `rest` the hex digits took.
fn parse_unicode_escape(rest: &[u8]) -> Result<(Vec<u8>, usize), DecodeError> {
    let hex4 = |b: &[u8]| -> Option<u32> {
        if b.len() < 4 || !b[..4].iter().all(u8::is_ascii_hexdigit) {
            return None;
        }
        u32::from_str_radix(std::str::from_utf8(&b[..4]).ok()?, 16).ok()
    };
    // `^[dD][89aAbB]%x%x\\u%x%x%x%x`: a high surrogate followed by another escape.
    let pair = rest.len() >= 10
        && matches!(rest[0], b'd' | b'D')
        && matches!(rest[1], b'8' | b'9' | b'a' | b'A' | b'b' | b'B')
        && rest[2].is_ascii_hexdigit()
        && rest[3].is_ascii_hexdigit()
        && rest[4] == b'\\'
        && rest[5] == b'u'
        && rest[6..10].iter().all(u8::is_ascii_hexdigit);
    let (n, used) = if pair {
        let n1 = hex4(rest).expect("checked");
        let n2 = hex4(&rest[6..]).expect("checked");
        (
            (n1 - 0xd800) * 0x400 + (n2.wrapping_sub(0xdc00)) + 0x10000,
            10,
        )
    } else {
        match hex4(rest) {
            Some(n) => (n, 4),
            None => return err("invalid unicode escape in string"),
        }
    };
    codepoint_to_utf8(n).map(|b| (b, used))
}

/// rxi's `codepoint_to_utf8`: the plain UTF-8 bit layout, surrogates included.
fn codepoint_to_utf8(n: u32) -> Result<Vec<u8>, DecodeError> {
    Ok(if n <= 0x7f {
        vec![n as u8]
    } else if n <= 0x7ff {
        vec![(n / 64 + 192) as u8, (n % 64 + 128) as u8]
    } else if n <= 0xffff {
        vec![
            (n / 4096 + 224) as u8,
            (n % 4096 / 64 + 128) as u8,
            (n % 64 + 128) as u8,
        ]
    } else if n <= 0x10ffff {
        vec![
            (n / 262_144 + 240) as u8,
            (n % 262_144 / 4096 + 128) as u8,
            (n % 4096 / 64 + 128) as u8,
            (n % 64 + 128) as u8,
        ]
    } else {
        return err(format!("invalid unicode codepoint '{n:x}'"));
    })
}

fn parse_number(s: &[u8], i: usize) -> Result<(f64, usize), DecodeError> {
    let x = to_delim(s, i);
    let token = String::from_utf8_lossy(&s[i..x]);
    match lua::tonumber(&token) {
        Some(n) => Ok((n, x)),
        None => err(format!("invalid number '{token}'")),
    }
}

fn parse_literal(s: &[u8], i: usize) -> Result<(Option<Value>, usize), DecodeError> {
    let x = to_delim(s, i);
    match &s[i..x] {
        b"true" => Ok((Some(Value::Bool(true)), x)),
        b"false" => Ok((Some(Value::Bool(false)), x)),
        b"null" => Ok((None, x)),
        other => err(format!(
            "invalid literal '{}'",
            String::from_utf8_lossy(other)
        )),
    }
}

fn parse_array(s: &[u8], i: usize) -> Result<(Table, usize), DecodeError> {
    let mut res = Table::new();
    let mut i = i + 1;
    loop {
        i = skip_space(s, i);
        if s.get(i) == Some(&b']') {
            i += 1;
            break;
        }
        let (x, next) = parse(s, i)?;
        res.push(x);
        i = skip_space(s, next);
        let chr = s.get(i).copied();
        i += 1;
        if chr == Some(b']') {
            break;
        }
        if chr != Some(b',') {
            return err("expected ']' or ','");
        }
    }
    res.trim();
    Ok((res, i))
}

fn parse_object(s: &[u8], i: usize) -> Result<(Table, usize), DecodeError> {
    let mut res = Table::new();
    let mut i = i + 1;
    loop {
        i = skip_space(s, i);
        if s.get(i) == Some(&b'}') {
            i += 1;
            break;
        }
        if s.get(i) != Some(&b'"') {
            return err("expected string for key");
        }
        let (key, next) = parse_string(s, i)?;
        i = skip_space(s, next);
        if s.get(i) != Some(&b':') {
            return err("expected ':' after key");
        }
        i = skip_space(s, i + 1);
        let (val, next) = parse(s, i)?;
        res.set(&key, val);
        i = skip_space(s, next);
        let chr = s.get(i).copied();
        i += 1;
        if chr == Some(b'}') {
            break;
        }
        if chr != Some(b',') {
            return err("expected '}' or ','");
        }
    }
    Ok((res, i))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dec(s: &str) -> Option<Value> {
        decode(s.as_bytes()).expect("decodes")
    }

    #[test]
    fn empty_tables_encode_as_arrays() {
        assert_eq!(encode(&Json::Obj(vec![])).unwrap(), "[]");
        assert_eq!(encode(&Json::Arr(vec![])).unwrap(), "[]");
        let v = Json::obj([("a", Json::Obj(vec![])), ("b", Json::Num(1.5))]);
        assert_eq!(encode(&v).unwrap(), r#"{"a":[],"b":1.5}"#);
    }

    #[test]
    fn numbers_use_g14_and_refuse_non_finite() {
        assert_eq!(encode(&Json::Num(1e15)).unwrap(), "1e+15");
        assert_eq!(encode(&Json::Num(4.0)).unwrap(), "4");
        assert_eq!(
            encode(&Json::Num(f64::NAN)).unwrap_err().0,
            "unexpected number value 'nan'"
        );
        assert_eq!(
            encode(&Json::Arr(vec![Json::Num(f64::NEG_INFINITY)]))
                .unwrap_err()
                .0,
            "unexpected number value '-inf'"
        );
    }

    #[test]
    fn strings_escape_like_rxi() {
        let s = Json::str("a\"b\\c/d\n\u{1}é");
        assert_eq!(encode(&s).unwrap(), "\"a\\\"b\\\\c/d\\n\\u0001é\"");
    }

    #[test]
    fn decoding_is_lenient_like_rxi() {
        assert_eq!(
            dec("[1, 2,]"),
            Some(Value::Table({
                let mut t = Table::new();
                t.push(Some(Value::Num(1.0)));
                t.push(Some(Value::Num(2.0)));
                t
            }))
        );
        assert_eq!(dec("0x10"), Some(Value::Num(16.0)));
        assert_eq!(dec("null"), None);
        let obj = dec(r#"{"a": null, "b": [null], "c": "é😀"}"#).unwrap();
        let t = obj.as_table().unwrap();
        assert_eq!(t.get("a"), None);
        assert!(t.get("b").unwrap().as_table().unwrap().is_empty());
        assert_eq!(t.get("c").unwrap().as_str(), Some("é😀"));
        for bad in [
            "",
            "{invalid json}",
            "[1 2]",
            "{\"a\" 1}",
            "\"\n\"",
            "tru",
            "1 2",
        ] {
            assert!(decode(bad.as_bytes()).is_err(), "{bad:?} should not decode");
        }
    }

    #[test]
    fn decoded_tables_re_encode_like_rxi() {
        let v = dec(r#"{"x": [1, 2.5, "s"], "y": {}, "z": true}"#).unwrap();
        let j = from_value(&v).unwrap();
        assert_eq!(encode(&j).unwrap(), r#"{"x":[1,2.5,"s"],"y":[],"z":true}"#);
        let holes = dec("[1, null, 3]").unwrap();
        assert_eq!(
            from_value(&holes).unwrap_err().0,
            "invalid table: sparse array"
        );
    }
}

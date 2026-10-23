//! Lua values as BalatroBot sees a request, and Lua's number-to-text conversion.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! which runs on Balatro's LuaJIT 2.0.5 (the game ships it as `lua51.dll`).
//!
//! A decoded request is a tree of Lua values: JSON `null` is Lua `nil`, so it never appears as a
//! value (an object member set to `null` is absent, an array slot set to `null` is a hole), and
//! `{}` and `[]` both decode to an empty table. [`Table`] keeps the two kinds of key rxi's
//! decoder produces: integer keys from JSON arrays and string keys from JSON objects.
//!
//! [`tostring`] is LuaJIT 2.0.5's `tostring(number)` and `string.format("%.14g", x)`: both go to
//! the C runtime's `sprintf("%.14g")`, which prints the exact binary value rounded to 14
//! significant digits, ties to even (checked against the game's `lua51.dll`).

/// A Lua value other than `nil`.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Bool(bool),
    Num(f64),
    Str(String),
    Table(Table),
}

impl Value {
    /// Lua's `type(v)`.
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::Bool(_) => "boolean",
            Value::Num(_) => "number",
            Value::Str(_) => "string",
            Value::Table(_) => "table",
        }
    }

    pub fn as_num(&self) -> Option<f64> {
        match self {
            Value::Num(x) => Some(*x),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_table(&self) -> Option<&Table> {
        match self {
            Value::Table(t) => Some(t),
            _ => None,
        }
    }

    /// Lua truthiness: everything but `false` (and `nil`, which is no value).
    pub fn truthy(&self) -> bool {
        !matches!(self, Value::Bool(false))
    }
}

/// A Lua table built by rxi's JSON decoder.
#[derive(Clone, Debug, PartialEq, Default)]
pub struct Table {
    /// Integer keys: `seq[i]` is `t[i + 1]`; `None` is a hole left by a JSON `null`. Never ends
    /// in a hole (a trailing `null` creates no key).
    seq: Vec<Option<Value>>,
    /// String keys, in insertion order.
    map: Vec<(String, Value)>,
}

impl Table {
    pub fn new() -> Table {
        Table::default()
    }

    /// A table with string keys (a JSON object), in order.
    pub fn object<K: Into<String>>(entries: impl IntoIterator<Item = (K, Value)>) -> Table {
        let mut t = Table::new();
        for (k, v) in entries {
            t.set(&k.into(), Some(v));
        }
        t
    }

    /// A sequence `{v1, v2, ...}` (a JSON array).
    pub fn sequence(items: impl IntoIterator<Item = Value>) -> Table {
        let mut t = Table::new();
        for v in items {
            t.push(Some(v));
        }
        t
    }

    /// `t[n] = v` for the next array slot `n` (rxi's `parse_array`); `None` is `nil`.
    pub fn push(&mut self, v: Option<Value>) {
        self.seq.push(v);
    }

    /// Drop trailing holes: assigning `nil` to a missing key creates nothing.
    pub fn trim(&mut self) {
        while matches!(self.seq.last(), Some(None)) {
            self.seq.pop();
        }
    }

    /// `t[key] = v` for a string key (rxi's `parse_object`); `None` deletes the key.
    pub fn set(&mut self, key: &str, v: Option<Value>) {
        let pos = self.map.iter().position(|(k, _)| k == key);
        match (pos, v) {
            (Some(i), Some(v)) => self.map[i].1 = v,
            (Some(i), None) => {
                self.map.remove(i);
            }
            (None, Some(v)) => self.map.push((key.to_owned(), v)),
            (None, None) => {}
        }
    }

    /// `t[key]` for a string key.
    pub fn get(&self, key: &str) -> Option<&Value> {
        self.map.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// `t[i]` for an integer key (1-based).
    pub fn get_index(&self, i: i64) -> Option<&Value> {
        if i < 1 {
            return None;
        }
        self.seq.get((i - 1) as usize).and_then(Option::as_ref)
    }

    /// `next(t) == nil`.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty() && self.seq.iter().all(Option::is_none)
    }

    /// The number of keys.
    pub fn key_count(&self) -> usize {
        self.map.len() + self.seq.iter().filter(|v| v.is_some()).count()
    }

    /// BalatroBot's `is_array` (validator.lua:17-29): every key is a number and the keys are
    /// exactly `1..n`. The answer does not depend on `pairs` order.
    pub fn is_array(&self) -> bool {
        self.map.is_empty() && self.seq.iter().all(Option::is_some)
    }

    /// `ipairs(t)`: the values at `1, 2, ...` up to the first hole.
    pub fn ipairs(&self) -> impl Iterator<Item = &Value> {
        self.seq.iter().map_while(Option::as_ref)
    }

    /// `#t` for a table without holes (the only tables BalatroBot measures).
    pub fn len(&self) -> usize {
        self.ipairs().count()
    }

    /// The integer-keyed slots (holes included) and the string-keyed entries.
    pub fn parts(&self) -> (&[Option<Value>], &[(String, Value)]) {
        (&self.seq, &self.map)
    }
}

/// `type(value) == "number" and math.floor(value) == value` (validator.lua:11-13): true for
/// infinities, as `math.floor(inf) == inf`.
pub fn is_integer(v: Option<&Value>) -> bool {
    matches!(v, Some(Value::Num(x)) if x.floor() == *x)
}

/// LuaJIT 2.0.5's `tostring(x)` / `string.format("%.14g", x)` for a number.
pub fn tostring(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_owned();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_owned();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0" } else { "0" }.to_owned();
    }
    // Rust prints the exact value rounded to 14 significant digits, ties to even, like the
    // game's C runtime; the exponent is of the rounded value.
    let sci = format!("{:.13e}", x.abs());
    let (mantissa, exp) = sci.split_once('e').expect("exponent form");
    let exp: i32 = exp.parse().expect("exponent");
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    let sign = if x < 0.0 { "-" } else { "" };
    let body = if (-4..14).contains(&exp) {
        // `%f` with `13 - exp` decimals: the same 14 digits, placed around the point.
        let fixed = if exp >= 0 {
            let int_len = (exp + 1) as usize;
            format!("{}.{}", &digits[..int_len], &digits[int_len..])
        } else {
            format!("0.{}{}", "0".repeat((-exp - 1) as usize), digits)
        };
        trim_fraction(&fixed)
    } else {
        let m = trim_fraction(&format!("{}.{}", &digits[..1], &digits[1..]));
        format!("{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
    };
    format!("{sign}{body}")
}

fn trim_fraction(s: &str) -> String {
    if s.contains('.') {
        s.trim_end_matches('0').trim_end_matches('.').to_owned()
    } else {
        s.to_owned()
    }
}

/// LuaJIT 2.0.5's `tonumber(s)` for the token rxi's decoder hands it: an optional `-`, then a
/// decimal or `0x` hexadecimal numeral (with optional fraction and exponent), or `inf` / `nan`.
pub fn tonumber(s: &str) -> Option<f64> {
    let (neg, body) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s),
    };
    let value = match body {
        "inf" => f64::INFINITY,
        "nan" => f64::NAN,
        _ => {
            if let Some(hex) = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")) {
                parse_hex(hex)?
            } else {
                parse_decimal(body)?
            }
        }
    };
    // A numeral that reads as an integer is converted as one, so `-0` is `0`.
    if neg && value == 0.0 && !body.contains(['.', 'e', 'E']) {
        return Some(0.0);
    }
    Some(if neg { -value } else { value })
}

fn parse_decimal(s: &str) -> Option<f64> {
    let bytes = s.as_bytes();
    let mut i = 0;
    let mut digits = 0;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
        digits += 1;
    }
    if i < bytes.len() && bytes[i] == b'.' {
        i += 1;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
            digits += 1;
        }
    }
    if digits == 0 {
        return None;
    }
    if i < bytes.len() && (bytes[i] == b'e' || bytes[i] == b'E') {
        i += 1;
        if i < bytes.len() && (bytes[i] == b'+' || bytes[i] == b'-') {
            i += 1;
        }
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == start {
            return None;
        }
    }
    if i != bytes.len() {
        return None;
    }
    // Rust reads a decimal numeral as the nearest double, as LuaJIT's scanner does; a leading or
    // trailing `.` is padded so the standard parser takes it.
    let mut text = s.to_owned();
    if text.starts_with('.') {
        text.insert(0, '0');
    }
    text.parse().ok()
}

fn parse_hex(s: &str) -> Option<f64> {
    let (mant, exp) = match s.find(['p', 'P']) {
        Some(i) => (&s[..i], Some(&s[i + 1..])),
        None => (s, None),
    };
    let (int_part, frac_part) = match mant.split_once('.') {
        Some((a, b)) => (a, b),
        None => (mant, ""),
    };
    if int_part.is_empty() && frac_part.is_empty() {
        return None;
    }
    let mut value = 0.0f64;
    for c in int_part.chars() {
        value = value * 16.0 + f64::from(c.to_digit(16)?);
    }
    let mut scale = 1.0 / 16.0;
    for c in frac_part.chars() {
        value += f64::from(c.to_digit(16)?) * scale;
        scale /= 16.0;
    }
    if let Some(e) = exp {
        let e: i32 = e.parse().ok()?;
        value *= 2f64.powi(e);
    }
    Some(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tostring_matches_balatros_luajit() {
        // Printed by the game's own lua51.dll (LuaJIT 2.0.5): `%.14g`, ties to even.
        let cases: &[(f64, &str)] = &[
            (123_456_789_012_345.0, "1.2345678901234e+14"),
            (123_456_789_012_355.0, "1.2345678901236e+14"),
            (123_456_789_012_365.0, "1.2345678901236e+14"),
            (123_456_789_012_375.0, "1.2345678901238e+14"),
            (100_000_000_000_005.0, "1e+14"),
            (100_000_000_000_015.0, "1.0000000000002e+14"),
            (100_000_000_000_025.0, "1.0000000000002e+14"),
            (999_999_999_999_995.0, "1e+15"),
            (1_234_567_890_123_450.0, "1.2345678901234e+15"),
            (1_234_567_890_123_550.0, "1.2345678901236e+15"),
            (0.5, "0.5"),
            (2.5, "2.5"),
            (1.0 / 16384.0, "6.103515625e-05"),
            (3.0 / 16384.0, "0.00018310546875"),
            (1.000_061_035_156_25, "1.0000610351562"),
            (0.000_030_517_578_125, "3.0517578125e-05"),
            (1e300, "1e+300"),
            (1e-300, "1e-300"),
            (9_007_199_254_740_992.0, "9.007199254741e+15"),
            (9_223_372_036_854_775_808.0, "9.2233720368548e+18"),
            (5e-324, "4.9406564584125e-324"),
            (f64::MAX, "1.7976931348623e+308"),
            (0.1, "0.1"),
            (1.0 / 3.0, "0.33333333333333"),
            (2.0 / 3.0, "0.66666666666667"),
            (100.0, "100"),
            (-100.5, "-100.5"),
            (1e14, "1e+14"),
            (99_999_999_999_999.5, "1e+14"),
            (0.1 + 0.2, "0.3"),
            (1e15, "1e+15"),
            (-0.0, "-0"),
            (f64::NAN, "nan"),
            (f64::INFINITY, "inf"),
            (f64::NEG_INFINITY, "-inf"),
            (999.0, "999"),
            (-1.0, "-1"),
        ];
        for &(x, want) in cases {
            assert_eq!(tostring(x), want, "tostring({x:e})");
        }
    }

    #[test]
    fn tonumber_matches_balatros_luajit() {
        let cases: &[(&str, Option<f64>)] = &[
            ("0x10", Some(16.0)),
            ("1e5", Some(100_000.0)),
            ("01", Some(1.0)),
            ("1.", Some(1.0)),
            ("-.5", Some(-0.5)),
            ("1e", None),
            ("-inf", Some(f64::NEG_INFINITY)),
            ("0x1p4", Some(16.0)),
            ("1_000", None),
            ("1.5e+10", Some(1.5e10)),
            ("-0x10", Some(-16.0)),
            ("0X1A", Some(26.0)),
            ("1e400", Some(f64::INFINITY)),
            ("0x", None),
            ("1.2.3", None),
            ("--1", None),
            ("-", None),
            ("0.1e-2", Some(0.001)),
            ("00", Some(0.0)),
            ("1E2", Some(100.0)),
            ("4.9e-325", Some(0.0)),
            ("0x1.8", Some(1.5)),
            ("1f", None),
            ("2.5e3x", None),
        ];
        for &(s, want) in cases {
            assert_eq!(tonumber(s), want, "tonumber({s:?})");
        }
        assert_eq!(tonumber("-0").map(f64::to_bits), Some(0.0f64.to_bits()));
        assert!(tonumber("nan").is_some_and(f64::is_nan));
    }

    #[test]
    fn tables_follow_rxi_decoding() {
        let mut t = Table::new();
        t.push(Some(Value::Num(1.0)));
        t.push(None);
        t.push(Some(Value::Num(3.0)));
        assert!(!t.is_array(), "a hole breaks the sequence");
        assert_eq!(t.ipairs().count(), 1);
        let mut u = Table::new();
        u.push(Some(Value::Num(1.0)));
        u.push(None);
        u.trim();
        assert!(u.is_array());
        assert_eq!(u.len(), 1);
        let mut o = Table::new();
        o.set("a", Some(Value::Num(1.0)));
        o.set("a", None);
        assert!(
            o.is_empty() && o.is_array(),
            "an emptied object is an empty table"
        );
        assert!(is_integer(Some(&Value::Num(f64::INFINITY))));
        assert!(!is_integer(Some(&Value::Num(1.5))));
        assert!(!is_integer(Some(&Value::Str("1".into()))));
    }
}

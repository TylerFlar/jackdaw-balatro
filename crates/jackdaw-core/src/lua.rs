//! Lua number semantics, as the game's LuaJIT computes them on x86-64 (2.0.5 in the shipped
//! build; its arithmetic and RNG match 2.1, its number formatting goes through the C runtime).
//!
//! Balatro runs on LuaJIT, where every number is an IEEE-754 double. Ordinary arithmetic maps
//! onto Rust's `f64` directly; this module covers the places where an exact match needs more:
//!
//! * **NaN bits.** The RNG seeds LuaJIT's TW223 generator from a double's bit pattern, so the
//!   sign of a NaN changes every draw that follows. LuaJIT on x86-64 inherits the SSE rules: an
//!   invalid operation (`inf - inf`, `0/0`, `inf % 1`) yields the "default NaN"
//!   `0xFFF8_0000_0000_0000` (sign bit set), and an operation with NaN operands returns the first
//!   NaN operand. [`add`], [`sub`], [`mul`] and [`div`] reproduce those rules on every target, so
//!   ARM builds draw the same numbers as the live game on x86-64.
//! * **`%`** is `a - floor(a/b)*b` ([`modulo`]), not the C `fmod`.
//! * **`tonumber(string.format("%.13f", x))`**, the RNG's rounding step ([`truncate13`]).
//!   The game's LuaJIT 2.0.5 formats through the C runtime, which rounds the exact binary value
//!   and breaks exact ties to even (G181; LuaJIT 2.1's own formatter breaks them away from
//!   zero).
//! * **`tostring(x)`** for numbers, i.e. `%.14g` ([`tostring_number`]), used where the game
//!   builds RNG keys out of numbers.

/// `math.pi` (`3.14159265358979323846`).
pub const PI: f64 = std::f64::consts::PI;
/// The `e` constant LuaJIT's `math.randomseed` uses (`2.7182818284590452354`).
pub const E: f64 = std::f64::consts::E;

/// The NaN an invalid SSE operation produces (`inf - inf`, `0/0`): sign bit set, quiet.
pub const DEFAULT_NAN_BITS: u64 = 0xFFF8_0000_0000_0000;
/// The NaN `math.abs` leaves of [`DEFAULT_NAN_BITS`]: the snapshot's canonical NaN.
pub const POSITIVE_NAN_BITS: u64 = 0x7FF8_0000_0000_0000;

const QUIET_BIT: u64 = 0x0008_0000_0000_0000;

#[inline]
fn default_nan() -> f64 {
    f64::from_bits(DEFAULT_NAN_BITS)
}

/// The result of a binary SSE operation `a op b` whose native result is `r`.
#[inline]
fn sse(a: f64, b: f64, r: f64) -> f64 {
    if a.is_nan() {
        f64::from_bits(a.to_bits() | QUIET_BIT)
    } else if b.is_nan() {
        f64::from_bits(b.to_bits() | QUIET_BIT)
    } else if r.is_nan() {
        default_nan()
    } else {
        r
    }
}

/// `a + b` with x86-64 NaN propagation.
#[inline]
pub fn add(a: f64, b: f64) -> f64 {
    sse(a, b, a + b)
}

/// `a - b` with x86-64 NaN propagation.
#[inline]
pub fn sub(a: f64, b: f64) -> f64 {
    sse(a, b, a - b)
}

/// `a * b` with x86-64 NaN propagation.
#[inline]
pub fn mul(a: f64, b: f64) -> f64 {
    sse(a, b, a * b)
}

/// `a / b` with x86-64 NaN propagation.
#[inline]
pub fn div(a: f64, b: f64) -> f64 {
    sse(a, b, a / b)
}

/// LuaJIT's `math.floor`. Values of magnitude 2^52 and up (and NaN) pass through unchanged.
///
/// Below 2^52 a truncation to `i64` is exact, so the floor is computed inline: baseline x86-64
/// has no rounding instruction (SSE4.1), and `f64::floor` would call the C runtime on every RNG
/// draw. Zero keeps its sign, as `floor` does.
#[inline]
pub fn floor(x: f64) -> f64 {
    if x.abs() < 4_503_599_627_370_496.0 {
        if x == 0.0 {
            return x;
        }
        let t = x as i64 as f64;
        if t > x { t - 1.0 } else { t }
    } else {
        x
    }
}

/// Lua's `a % b`: `a - floor(a/b)*b` (LuaJIT `vm_mod`).
#[inline]
pub fn modulo(a: f64, b: f64) -> f64 {
    sub(a, mul(floor(div(a, b)), b))
}

/// `tonumber(string.format("%.13f", x))`.
///
/// The exact binary value is rounded to 13 decimals with ties to even, as the game's LuaJIT
/// 2.0.5 formats through the C runtime (G181); the decimal string then reads back as the
/// nearest double. `"nan"` reads back
/// as LuaJIT's default NaN whatever the input NaN's sign; infinities pass through.
pub fn truncate13(x: f64) -> f64 {
    if x.is_nan() {
        return default_nan();
    }
    if x.is_infinite() || x == 0.0 {
        return x;
    }
    let a = x.abs();
    if a >= 1.0 {
        return truncate13_slow(x);
    }
    // a = mant * 2^exp exactly, with exp < 0 because a < 1.
    let bits = a.to_bits();
    let biased = ((bits >> 52) & 0x7FF) as i32;
    let frac = bits & ((1u64 << 52) - 1);
    let (mant, exp) = if biased == 0 {
        (frac, -1074)
    } else {
        (frac | (1u64 << 52), biased - 1075)
    };
    let shift = (-exp) as u32;
    let scaled = u128::from(mant) * 10_000_000_000_000u128; // < 2^97
    let q = if shift >= 128 {
        0 // scaled < 2^97 <= 2^(shift-1): rounds down to zero
    } else {
        let q = scaled >> shift;
        let rem = scaled - (q << shift);
        let half = 1u128 << (shift - 1);
        if rem > half || (rem == half && q & 1 == 1) {
            q + 1
        } else {
            q
        }
    };
    // q <= 10^13 < 2^53, so both operands are exact and the quotient is correctly rounded,
    // which is exactly what reading the decimal string back yields.
    let r = (q as f64) / 1e13;
    if x < 0.0 { -r } else { r }
}

/// [`truncate13`] for `|x| >= 1` (never on the RNG path), from the exact decimal expansion.
fn truncate13_slow(x: f64) -> f64 {
    let (int_digits, frac_digits) = exact_fixed(x.abs());
    let mut digits: Vec<u8> = int_digits
        .into_iter()
        .chain(frac_digits[..13].iter().copied())
        .collect();
    if rounds_up(&digits, &frac_digits[13..]) {
        round_up_digits(&mut digits);
    }
    let int_len = digits.len() - 13;
    let text = format!(
        "{}{}.{}",
        if x < 0.0 { "-" } else { "" },
        std::str::from_utf8(&digits[..int_len]).expect("ascii digits"),
        std::str::from_utf8(&digits[int_len..]).expect("ascii digits"),
    );
    text.parse().expect("decimal parses")
}

/// The exact decimal expansion of a finite `x >= 0`: integer digits, then fraction digits
/// (at least 14). Every double has a finite expansion, so this is exact.
fn exact_fixed(x: f64) -> (Vec<u8>, Vec<u8>) {
    let text = format!("{x:.1100}");
    let (int_part, frac_part) = text.split_once('.').expect("fixed form");
    (int_part.bytes().collect(), frac_part.bytes().collect())
}

/// Whether `kept`, followed by the dropped digits `rest`, rounds up to the nearest with ties
/// to even.
fn rounds_up(kept: &[u8], rest: &[u8]) -> bool {
    match rest.first() {
        Some(&d) if d > b'5' => true,
        Some(&b'5') => {
            rest[1..].iter().any(|&d| d != b'0')
                || kept.last().is_some_and(|&d| (d - b'0') % 2 == 1)
        }
        _ => false,
    }
}

/// Add one unit in the last place of a decimal digit string, growing it on carry-out.
fn round_up_digits(digits: &mut Vec<u8>) {
    for d in digits.iter_mut().rev() {
        if *d == b'9' {
            *d = b'0';
        } else {
            *d += 1;
            return;
        }
    }
    digits.insert(0, b'1');
}

/// `tostring(x)` for a number: `%.14g` as the C runtime rounds it (exact expansion, ties to
/// even, G181). Every NaN prints as `nan`.
pub fn tostring_number(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_owned();
    }
    if x.is_infinite() {
        return if x > 0.0 {
            "inf".to_owned()
        } else {
            "-inf".to_owned()
        };
    }
    if x == 0.0 {
        return if x.is_sign_negative() {
            "-0".to_owned()
        } else {
            "0".to_owned()
        };
    }
    if x == x.trunc() && x.abs() < 1e14 {
        return format!("{}", x as i64);
    }
    // Exact scientific digits, then round to 14 significant digits.
    let text = format!("{:.800e}", x.abs());
    let (mantissa, exp) = text.split_once('e').expect("exponent form");
    let mut exp: i32 = exp.parse().expect("exponent");
    let all: Vec<u8> = mantissa.bytes().filter(u8::is_ascii_digit).collect();
    let mut digits = all[..14].to_vec();
    if rounds_up(&digits, &all[14..]) {
        round_up_digits(&mut digits);
        if digits.len() > 14 {
            digits.truncate(14);
            exp += 1;
        }
    }
    let sign = if x < 0.0 { "-" } else { "" };
    let body = if (-4..14).contains(&exp) {
        // Fixed form with 14 significant digits.
        let mut s = String::new();
        if exp < 0 {
            s.push_str("0.");
            for _ in 0..(-exp - 1) {
                s.push('0');
            }
            s.push_str(std::str::from_utf8(&digits).expect("ascii digits"));
        } else {
            let int_len = (exp + 1) as usize;
            s.push_str(std::str::from_utf8(&digits[..int_len]).expect("ascii digits"));
            s.push('.');
            s.push_str(std::str::from_utf8(&digits[int_len..]).expect("ascii digits"));
        }
        trim_fraction(&s)
    } else {
        let mut m = String::from(char::from(digits[0]));
        m.push('.');
        m.push_str(std::str::from_utf8(&digits[1..]).expect("ascii digits"));
        let m = trim_fraction(&m);
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_inline_floor_is_the_c_floor() {
        let big = 4_503_599_627_370_496.0; // 2^52: from here on, every double is an integer
        let mut cases = vec![
            0.0,
            -0.0,
            0.3,
            -0.3,
            1.0,
            -1.0,
            2.5,
            -2.5,
            f64::MIN_POSITIVE,
            -f64::MIN_POSITIVE,
            1.0 - f64::EPSILON / 2.0,
            -(1.0 - f64::EPSILON / 2.0),
            big - 0.5,
            -(big - 0.5),
            big,
            -big,
            1e300,
            -1e300,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        let mut x = 0x2545_F491_4F6C_DD1Du64;
        for _ in 0..100_000 {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            let v = f64::from_bits(x);
            if v.is_finite() {
                cases.push(v);
                cases.push(v.fract() * 1e6);
            }
        }
        for v in cases {
            assert_eq!(floor(v).to_bits(), v.floor().to_bits(), "floor({v:e})");
        }
        assert!(floor(f64::NAN).is_nan());
    }

    #[test]
    fn invalid_operations_give_the_negative_default_nan() {
        assert_eq!(modulo(f64::INFINITY, 1.0).to_bits(), DEFAULT_NAN_BITS);
        assert_eq!(div(0.0, 0.0).to_bits(), DEFAULT_NAN_BITS);
        assert_eq!(
            sub(f64::INFINITY, f64::INFINITY).to_bits(),
            DEFAULT_NAN_BITS
        );
    }

    #[test]
    fn the_first_nan_operand_wins() {
        let pos = f64::from_bits(POSITIVE_NAN_BITS);
        let neg = f64::from_bits(DEFAULT_NAN_BITS);
        assert_eq!(add(pos, neg).to_bits(), POSITIVE_NAN_BITS);
        assert_eq!(add(neg, pos).to_bits(), DEFAULT_NAN_BITS);
        assert_eq!(add(1.0, neg).to_bits(), DEFAULT_NAN_BITS);
        assert_eq!(div(neg, 2.0).to_bits(), DEFAULT_NAN_BITS);
        assert_eq!(add(mul(neg, PI), E).to_bits(), DEFAULT_NAN_BITS);
    }

    #[test]
    fn modulo_is_floored() {
        assert_eq!(modulo(5.5, 1.0), 0.5);
        assert_eq!(modulo(-0.25, 1.0), 0.75);
        assert_eq!(modulo(-7.0, 3.0), 2.0);
        assert_eq!(modulo(7.0, -3.0), -2.0);
    }

    #[test]
    fn truncate13_rounds_ties_to_even() {
        // 2^-14 = 0.00006103515625 exactly: a tie at the 13th decimal; the game's LuaJIT 2.0.5
        // prints `0.0000610351562` (G181).
        assert_eq!(truncate13(1.0 / 16384.0), 0.000_061_035_156_2);
        assert_eq!(truncate13(3.0 / 16384.0), 0.000_183_105_468_8);
        assert_eq!(truncate13(0.123_456_789_012_34), 0.123_456_789_012_3);
        assert_eq!(truncate13(0.999_999_999_999_96), 1.0);
        assert_eq!(
            truncate13(f64::from_bits(POSITIVE_NAN_BITS)).to_bits(),
            DEFAULT_NAN_BITS
        );
    }

    #[test]
    fn truncate13_matches_decimal_round_trip_off_ties() {
        // Off exact ties, Rust's exact formatter and the integer path must agree.
        let mut x = 0.123_456_789_f64;
        for _ in 0..100_000 {
            x = modulo(add(2.134_453_429_141, mul(x, 1.724_312_34)), 1.0);
            let via_string: f64 = format!("{x:.13}").parse().unwrap();
            assert_eq!(truncate13(x).to_bits(), via_string.to_bits(), "x = {x:e}");
        }
    }

    #[test]
    fn tostring_rounds_ties_to_even() {
        // 2^46 + 0.5 and 2^46 + 1.5 are exact ties at 14 significant digits (G181).
        assert_eq!(tostring_number(70_368_744_177_664.5), "70368744177664");
        assert_eq!(tostring_number(70_368_744_177_665.5), "70368744177666");
    }

    #[test]
    fn tostring_matches_luajit() {
        // Expected strings printed by LuaJIT `tostring`; on exact ties, the game's 2.0.5.
        let cases: &[(f64, &str)] = &[
            (1.0, "1"),
            (-3.0, "-3"),
            (0.5, "0.5"),
            (1e15, "1e+15"),
            // An exact tie: LuaJIT 2.1 prints ...235, the game's 2.0.5 (C runtime) ...234.
            (123_456_789_012_345.0, "1.2345678901234e+14"),
            (99_999_999_999_999.0, "99999999999999"),
            (1.0 / 3.0, "0.33333333333333"),
            (1e-5, "1e-05"),
            (0.1, "0.1"),
            (9_007_199_254_740_992.0, "9.007199254741e+15"),
            (1e14, "1e+14"),
            // An exact tie again: LuaJIT 2.1 ...235, the game's 2.0.5 ...234.
            (12_345_678_901_234.5, "12345678901234"),
            (-0.0, "-0"),
            (f64::NAN, "nan"),
            (-f64::NAN, "nan"),
            (f64::INFINITY, "inf"),
        ];
        for &(x, want) in cases {
            assert_eq!(tostring_number(x), want, "tostring({x:e})");
        }
    }

    #[test]
    fn truncate13_large_values() {
        // Expected values printed by `string.format("%.13f", x)`.
        assert_eq!(truncate13(2.5), 2.5);
        // The double nearest 1.00000000000005 lies just below the tie.
        assert_eq!(truncate13(1.000_000_000_000_05), 1.0);
        // An exact tie: LuaJIT 2.1 prints ...563, the game's 2.0.5 (C runtime) ...562 (G181).
        assert_eq!(truncate13(1.0 + 1.0 / 16384.0), 1.000_061_035_156_2);
        assert_eq!(truncate13(-(1.0 + 1.0 / 16384.0)), -1.000_061_035_156_2);
    }
}

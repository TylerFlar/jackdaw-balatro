//! Balatro's RNG, bit-exact with LuaJIT 2.1 on x86-64.
//!
//! Three layers (`functions/misc_functions.lua`):
//!
//! 1. [`pseudohash`] maps a string to a double in `[0, 1)` (or NaN).
//! 2. [`Rng::seed`] (`pseudoseed`) advances one named stream of `G.GAME.pseudorandom` and mixes
//!    in the hashed run seed.
//! 3. [`Tw223`] is LuaJIT's `math.random`: the game reseeds it from that double on every draw
//!    (`math.randomseed`), then draws once (`pseudorandom`, `pseudorandom_element`) or several
//!    times (`pseudoshuffle`, `random_string`).
//!
//! All floating-point steps go through [`crate::lua`] so NaNs carry the bits LuaJIT produces:
//! a string whose hash reaches exactly zero ("bugged seeds") seeds TW223 from a NaN pattern.

use std::borrow::Borrow;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash, Hasher};

use crate::lua;

// ---------------------------------------------------------------------------
// TW223 (lj_prng.c)
// ---------------------------------------------------------------------------

/// LuaJIT's combined Tausworthe generator (`PRNGState`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tw223 {
    u: [u64; 4],
}

/// One `TW223_GEN(z, r, i, k, q, s)` word update.
#[inline]
const fn gen_word(z: u64, k: u32, q: u32, s: u32) -> u64 {
    (((z << q) ^ z) >> (k - s)) ^ ((z & (u64::MAX << (64 - k))) << s)
}

#[inline]
const fn step_word(z: u64, word: usize) -> u64 {
    match word {
        0 => gen_word(z, 63, 31, 18),
        1 => gen_word(z, 58, 19, 28),
        2 => gen_word(z, 55, 24, 7),
        _ => gen_word(z, 47, 21, 8),
    }
}

const WARMUP_STEPS: usize = 10;

/// `WARMUP[w][k][v]` is word `w` after the 10 warm-up steps when it starts as `v << 8k`. Each
/// word's update is linear over GF(2), so the warm-up of any value is the xor of its bytes'
/// entries: 32 lookups instead of 40 word updates.
static WARMUP: [[[u64; 256]; 8]; 4] = build_warmup();

const fn build_warmup() -> [[[u64; 256]; 8]; 4] {
    let mut tables = [[[0u64; 256]; 8]; 4];
    let mut word = 0;
    while word < 4 {
        // The warm-up of each single bit.
        let mut basis = [0u64; 64];
        let mut bit = 0;
        while bit < 64 {
            let mut z = 1u64 << bit;
            let mut i = 0;
            while i < WARMUP_STEPS {
                z = step_word(z, word);
                i += 1;
            }
            basis[bit] = z;
            bit += 1;
        }
        let mut k = 0;
        while k < 8 {
            let mut v = 1usize;
            while v < 256 {
                let low = v.isolate_lowest_one();
                tables[word][k][v] =
                    tables[word][k][v ^ low] ^ basis[8 * k + low.trailing_zeros() as usize];
                v += 1;
            }
            k += 1;
        }
        word += 1;
    }
    tables
}

#[inline]
fn warm(word: usize, u: u64) -> u64 {
    let t = &WARMUP[word];
    let mut r = 0;
    for (k, row) in t.iter().enumerate() {
        r ^= row[((u >> (8 * k)) & 0xFF) as usize];
    }
    r
}

impl Tw223 {
    /// `math.randomseed(d)` (`random_seed` in `lj_prng.c`).
    pub fn seeded(d: f64) -> Self {
        let mut d = d;
        let mut u = [0u64; 4];
        let mut r: u32 = 0x1109_0601; // 64 - k[i] as four bytes
        for (i, slot) in u.iter_mut().enumerate() {
            let m = 1u64 << (r & 0xFF);
            r >>= 8;
            d = lua::add(lua::mul(d, lua::PI), lua::E);
            let mut bits = d.to_bits();
            if bits < m {
                bits += m; // make sure the top k[i] bits are not all zero
            }
            *slot = warm(i, bits);
        }
        Self { u }
    }

    /// The raw state words.
    pub fn words(&self) -> [u64; 4] {
        self.u
    }

    /// `lj_prng_u64d`: advance and return the double in `[1, 2)` minus one.
    #[inline]
    pub fn random(&mut self) -> f64 {
        let mut r = 0u64;
        for (i, z) in self.u.iter_mut().enumerate() {
            *z = step_word(*z, i);
            r ^= *z;
        }
        f64::from_bits((r & 0x000F_FFFF_FFFF_FFFF) | 0x3FF0_0000_0000_0000) - 1.0
    }

    /// `math.random(m)`: an integer in `[1, m]`.
    #[inline]
    pub fn random_n(&mut self, m: i64) -> i64 {
        crate::lua::floor(self.random() * m as f64) as i64 + 1
    }

    /// `math.random(m, n)`: an integer in `[m, n]`.
    #[inline]
    pub fn random_range(&mut self, m: i64, n: i64) -> i64 {
        crate::lua::floor(self.random() * (n - m + 1) as f64) as i64 + m
    }
}

// ---------------------------------------------------------------------------
// pseudohash / pseudoseed
// ---------------------------------------------------------------------------

/// A stream key built on the stack, as the game concatenates them for every draw
/// (`'rarity'..ante..append`, `pool_key..'_resample'..it`): `Key::new("rarity").n(ante)`.
/// A key longer than the buffer, which the game never builds, moves to the heap.
#[derive(Clone)]
pub struct Key {
    len: usize,
    buf: [u8; 48],
    long: Option<String>,
}

impl Key {
    pub fn new(s: &str) -> Key {
        let mut key = Key {
            len: 0,
            buf: [0; 48],
            long: None,
        };
        key.push(s);
        key
    }

    /// `.. s`.
    #[must_use]
    pub fn s(mut self, s: &str) -> Key {
        self.push(s);
        self
    }

    /// `.. tostring(x)`, as [`lua::tostring_number`] writes it.
    #[must_use]
    pub fn n(mut self, x: f64) -> Key {
        if x == 0.0 {
            self.push(if x.is_sign_negative() { "-0" } else { "0" });
        } else if x == x.trunc() && x.abs() < 1e14 {
            if x < 0.0 {
                self.push("-");
            }
            self.digits(x.abs() as u64);
        } else {
            self.push(&lua::tostring_number(x));
        }
        self
    }

    /// `.. tostring(i)` for an integer.
    #[must_use]
    pub fn i(mut self, i: u64) -> Key {
        self.digits(i);
        self
    }

    fn digits(&mut self, mut v: u64) {
        let mut out = [0u8; 20];
        let mut at = out.len();
        loop {
            at -= 1;
            out[at] = b'0' + (v % 10) as u8;
            v /= 10;
            if v == 0 {
                break;
            }
        }
        self.push(std::str::from_utf8(&out[at..]).expect("ascii digits"));
    }

    fn push(&mut self, s: &str) {
        if let Some(long) = &mut self.long {
            long.push_str(s);
            return;
        }
        let end = self.len + s.len();
        if end <= self.buf.len() {
            self.buf[self.len..end].copy_from_slice(s.as_bytes());
            self.len = end;
        } else {
            let mut long = String::with_capacity(end);
            long.push_str(self);
            long.push_str(s);
            self.long = Some(long);
        }
    }
}

impl std::ops::Deref for Key {
    type Target = str;

    fn deref(&self) -> &str {
        match &self.long {
            Some(long) => long,
            // Only whole `&str`s are copied in: always UTF-8.
            None => std::str::from_utf8(&self.buf[..self.len]).expect("utf-8"),
        }
    }
}

/// `pseudohash(str)` over the bytes of `str` (`misc_functions.lua:279`).
pub fn pseudohash(s: &[u8]) -> f64 {
    pseudohash_parts(s, b"")
}

/// `pseudohash(a .. b)` without building the concatenation.
pub fn pseudohash_parts(a: &[u8], b: &[u8]) -> f64 {
    let byte_at = |i: usize| {
        if i > a.len() {
            b[i - a.len() - 1]
        } else {
            a[i - 1]
        }
    };
    let mut num = 1.0;
    for i in (1..=a.len() + b.len()).rev() {
        // While every value is finite no operation can make a NaN, and plain arithmetic (in
        // Lua's order) gives the NaN-aware helpers' bits; `x % 1` is then `x - floor(x)`.
        let s = 1.123_928_502_3 / num * f64::from(byte_at(i)) * lua::PI + lua::PI * i as f64;
        if !s.is_finite() {
            return pseudohash_rest(&byte_at, i, num);
        }
        num = s - lua::floor(s);
    }
    num
}

/// The rest of [`pseudohash_parts`] from byte `from` down, with Lua's NaN semantics.
fn pseudohash_rest(byte_at: &dyn Fn(usize) -> u8, from: usize, mut num: f64) -> f64 {
    for i in (1..=from).rev() {
        let t = lua::mul(
            lua::mul(lua::div(1.123_928_502_3, num), f64::from(byte_at(i))),
            lua::PI,
        );
        num = lua::modulo(lua::add(t, lua::mul(lua::PI, i as f64)), 1.0);
    }
    num
}

/// One step of a stream: `math.abs(tonumber(string.format("%.13f", (2.134453429141 + s*1.72431234) % 1)))`.
#[inline]
fn advance(state: f64) -> f64 {
    let raw = lua::modulo(
        lua::add(2.134_453_429_141, lua::mul(state, 1.724_312_34)),
        1.0,
    );
    lua::truncate13(raw).abs()
}

/// Keys up to this long are stored inline. The longest the game builds is about 27 bytes
/// (`'Spectral'..'spe'..ante..'_resample'..n` at a four-digit ante).
const KEY_CAP: usize = 30;

/// A stream key. Game keys are stored inline, so copying the stream table allocates nothing
/// per key (a `Game` is cloned on every step); longer keys go to the heap.
#[derive(Clone)]
enum StreamKey {
    Inline { len: u8, bytes: [u8; KEY_CAP] },
    Heap(Box<str>),
}

impl StreamKey {
    fn new(key: &str) -> Self {
        if key.len() > KEY_CAP {
            return StreamKey::Heap(key.into());
        }
        let mut bytes = [0; KEY_CAP];
        bytes[..key.len()].copy_from_slice(key.as_bytes());
        StreamKey::Inline {
            len: key.len() as u8,
            bytes,
        }
    }

    fn as_str(&self) -> &str {
        match self {
            StreamKey::Inline { len, bytes } => {
                std::str::from_utf8(&bytes[..usize::from(*len)]).expect("a str's bytes")
            }
            StreamKey::Heap(s) => s,
        }
    }
}

impl PartialEq for StreamKey {
    fn eq(&self, other: &Self) -> bool {
        self.as_str() == other.as_str()
    }
}

impl Eq for StreamKey {}

/// Hashes as its `str`, so the table can be searched with a `&str`.
impl Hash for StreamKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_str().hash(state);
    }
}

impl Borrow<str> for StreamKey {
    fn borrow(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Debug for StreamKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_str().fmt(f)
    }
}

/// FxHash (rustc's hasher): the keys are short and not attacker-controlled.
#[derive(Default, Clone, Copy)]
struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let (words, rest) = bytes.as_chunks::<8>();
        for &w in words {
            self.add(u64::from_le_bytes(w));
        }
        for &b in rest {
            self.add(u64::from(b));
        }
    }

    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }

    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

type StreamMap = HashMap<StreamKey, f64, BuildHasherDefault<FxHasher>>;

/// `G.GAME.pseudorandom`: the run seed and one state per named stream.
#[derive(Clone, Debug, PartialEq)]
pub struct Rng {
    seed: String,
    hashed_seed: f64,
    streams: StreamMap,
}

impl Rng {
    /// A fresh RNG for a run seeded with `seed` (`G.GAME.pseudorandom = {seed, hashed_seed}`).
    pub fn new(seed: &str) -> Self {
        Self {
            seed: seed.to_owned(),
            hashed_seed: pseudohash(seed.as_bytes()),
            streams: StreamMap::default(),
        }
    }

    /// Rebuild an RNG from saved state (`seed`, `hashed_seed`, stream values).
    pub fn from_parts(seed: String, hashed_seed: f64, streams: HashMap<String, f64>) -> Self {
        Self {
            seed,
            hashed_seed,
            streams: streams
                .iter()
                .map(|(k, &v)| (StreamKey::new(k), v))
                .collect(),
        }
    }

    pub fn seed_str(&self) -> &str {
        &self.seed
    }

    pub fn hashed_seed(&self) -> f64 {
        self.hashed_seed
    }

    /// Each stream's key and current state, in no particular order.
    pub fn streams(&self) -> impl Iterator<Item = (&str, f64)> {
        self.streams.iter().map(|(k, &v)| (k.as_str(), v))
    }

    /// The current state of stream `key`, if it has been drawn from.
    pub fn stream(&self, key: &str) -> Option<f64> {
        self.streams.get(key).copied()
    }

    /// How many streams have been drawn from.
    pub fn stream_count(&self) -> usize {
        self.streams.len()
    }

    /// `pseudoseed(key)`: advance stream `key` and return the double that seeds TW223.
    pub fn seed(&mut self, key: &str) -> f64 {
        let hashed = self.hashed_seed;
        if let Some(state) = self.streams.get_mut(key) {
            *state = advance(*state);
            return lua::div(lua::add(*state, hashed), 2.0);
        }
        // First use: G.GAME.pseudorandom[key] = pseudohash(key .. seed), then advance.
        let state = advance(pseudohash_parts(key.as_bytes(), self.seed.as_bytes()));
        self.streams.insert(StreamKey::new(key), state);
        lua::div(lua::add(state, hashed), 2.0)
    }

    /// `pseudorandom(key)`: a double in `[0, 1)` drawn from stream `key`.
    pub fn random(&mut self, key: &str) -> f64 {
        Tw223::seeded(self.seed(key)).random()
    }

    /// `pseudorandom(key, min, max)`: an integer in `[min, max]` drawn from stream `key`.
    pub fn random_range(&mut self, key: &str, min: i64, max: i64) -> i64 {
        Tw223::seeded(self.seed(key)).random_range(min, max)
    }
}

/// `pseudoseed(key, predict_seed)`: the stateless preview the game uses to look ahead.
pub fn predict_seed(key: &str, predict: &str) -> f64 {
    let pseed = advance(pseudohash_parts(key.as_bytes(), predict.as_bytes()));
    lua::div(lua::add(pseed, pseudohash(predict.as_bytes())), 2.0)
}

/// `pseudorandom(seed)` for an already computed seed double.
pub fn random_from(seed: f64) -> f64 {
    Tw223::seeded(seed).random()
}

/// `pseudorandom(seed, min, max)` for an already computed seed double.
pub fn random_range_from(seed: f64, min: i64, max: i64) -> i64 {
    Tw223::seeded(seed).random_range(min, max)
}

/// `pseudorandom_element(t, seed)` over `n` entries whose sort order is already the Lua order
/// (keys `1..n` of an array, or entries pre-sorted by `sort_id` / key): the 0-based position.
pub fn element_index(n: usize, seed: f64) -> usize {
    assert!(n > 0, "pseudorandom_element over an empty table");
    (Tw223::seeded(seed).random_n(n as i64) - 1) as usize
}

/// `pseudoshuffle(list, seed)` over a list already sorted by `sort_id` (the game sorts first
/// when the entries carry one).
pub fn shuffle<T>(items: &mut [T], seed: f64) {
    let mut tw = Tw223::seeded(seed);
    for i in (2..=items.len()).rev() {
        let j = tw.random_n(i as i64) as usize;
        items.swap(i - 1, j - 1);
    }
}

/// `random_string(length, seed)`: the run-seed generator (`misc_functions.lua:270`).
pub fn random_string(length: usize, seed: f64) -> String {
    let mut tw = Tw223::seeded(seed);
    let mut out = String::with_capacity(length);
    for _ in 0..length {
        let c = if tw.random() > 0.7 {
            tw.random_range(i64::from(b'1'), i64::from(b'9'))
        } else if tw.random() > 0.45 {
            tw.random_range(i64::from(b'A'), i64::from(b'N'))
        } else {
            tw.random_range(i64::from(b'P'), i64::from(b'Z'))
        };
        out.push(char::from(c as u8));
    }
    out.to_ascii_uppercase()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warmup_tables_match_ten_steps() {
        let mut x = 0x0123_4567_89AB_CDEFu64;
        for _ in 0..200 {
            x = x
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            for word in 0..4 {
                let mut z = x;
                for _ in 0..WARMUP_STEPS {
                    z = step_word(z, word);
                }
                assert_eq!(warm(word, x), z);
            }
        }
    }

    #[test]
    fn known_hashes() {
        // Values printed by LuaJIT 2.1 (bit patterns).
        assert_eq!(pseudohash(b"7LB2WVPK").to_bits(), 0x3FC6_A501_398D_4000);
        assert_eq!(
            pseudohash(b"erratic7LB2WVPK").to_bits(),
            lua::DEFAULT_NAN_BITS
        );
        assert_eq!(pseudohash(b"").to_bits(), 1.0f64.to_bits());
        assert_eq!(
            pseudohash_parts(b"erratic", b"7LB2WVPK").to_bits(),
            pseudohash(b"erratic7LB2WVPK").to_bits()
        );
    }

    #[test]
    fn nan_stream_state_turns_positive() {
        // "erratic" + "7LB2WVPK" hashes to NaN; the %.13f round trip and math.abs make the
        // stream state the positive NaN, which then wins the (state + hashed_seed) addition.
        let mut rng = Rng::new("7LB2WVPK");
        let s = rng.seed("erratic");
        assert_eq!(
            rng.stream("erratic").unwrap().to_bits(),
            lua::POSITIVE_NAN_BITS
        );
        assert_eq!(s.to_bits(), lua::POSITIVE_NAN_BITS);
        let tw = Tw223::seeded(s);
        assert_eq!(
            tw.words(),
            [
                0x002F_FD00_0030_0300,
                0xF818_0180_0000_2FFD,
                0x0000_7FF8_007F_07F0,
                0xFC01_F01F_0003_FFC3
            ]
        );
        assert_eq!(Tw223::seeded(s).random_range(1, 52), 52);
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut v: Vec<u32> = (1..=52).collect();
        shuffle(&mut v, 0.123_456);
        let mut sorted = v.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (1..=52).collect::<Vec<_>>());
        assert_ne!(v, sorted);
    }
}

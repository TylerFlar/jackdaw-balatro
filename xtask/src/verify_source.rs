//! `cargo xtask verify-source`: check the committed game data against the local game source.
//!
//! Runs the extraction scripts (`scripts/extract_prototypes.lua`, `scripts/extract_game_tables.lua`)
//! on `balatro_source/` with LuaJIT, writing into a temporary directory, and compares each file
//! with its committed copy in `jackdaw/engine/data/`. The comparison is semantic: formatting and
//! object key order are ignored, array order is not, and numbers must be the same double (`-0`
//! equals `0`). Local only: the game source is proprietary and gitignored, so CI has no copy.

use std::collections::BTreeSet;
use std::env;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{self, Command};

use serde_json::{Number, Value as Json};

const DATA_DIR: &str = "jackdaw/engine/data";
const SOURCE_DIR: &str = "balatro_source";
/// Where `BALATRO_VERSION` is defined. xtask reads the constant from the source rather than
/// depending on jackdaw-data, whose generated tables it writes.
const VERSION_SOURCE: &str = "crates/jackdaw-data/src/lib.rs";

/// Each extraction script and the files it writes into its output directory.
const EXTRACTS: &[(&str, &[&str])] = &[
    (
        "scripts/extract_prototypes.lua",
        &[
            "centers.json",
            "cards.json",
            "blinds.json",
            "tags.json",
            "stakes.json",
            "seals.json",
        ],
    ),
    (
        "scripts/extract_game_tables.lua",
        &["hands.json", "challenges.json", "blind_amounts.json"],
    ),
];

/// Differences listed per file; the rest are only counted.
const MAX_LISTED: usize = 25;

pub fn run(root: &Path) -> Result<(), String> {
    let version = check_source(root)?;
    let scratch = ScratchDir::new()?;
    println!(
        "Extracting from {SOURCE_DIR}/ (Balatro {version}) into {}",
        scratch.0.display()
    );
    for (script, _) in EXTRACTS {
        extract(root, script, &scratch.0)?;
    }

    let names: Vec<&str> = EXTRACTS
        .iter()
        .flat_map(|(_, files)| files.iter().copied())
        .collect();
    let mut differing = 0;
    for name in &names {
        let committed = load(&root.join(DATA_DIR).join(name))?;
        let extracted = load(&scratch.0.join(name))?;
        let mut diffs = Vec::new();
        compare(&mut String::new(), &committed, &extracted, &mut diffs);
        if diffs.is_empty() {
            println!(
                "ok    {name}: {} entries, {} values",
                entries(&committed),
                leaves(&committed)
            );
            continue;
        }
        differing += 1;
        println!(
            "DIFF  {name}: {} difference(s), committed vs extracted:",
            diffs.len()
        );
        for diff in diffs.iter().take(MAX_LISTED) {
            println!("        {diff}");
        }
        if diffs.len() > MAX_LISTED {
            println!("        ... and {} more", diffs.len() - MAX_LISTED);
        }
    }
    if differing > 0 {
        return Err(format!(
            "{differing} of {} files differ from what {SOURCE_DIR}/ extracts to. To update them, run \
             `luajit scripts/extract_prototypes.lua`, `luajit scripts/extract_game_tables.lua` \
             and `cargo xtask codegen`",
            names.len()
        ));
    }
    println!(
        "All {} files match the data extracted from {SOURCE_DIR}/",
        names.len()
    );
    Ok(())
}

/// Check that `balatro_source/` exists and is the release the data targets; return its version.
fn check_source(root: &Path) -> Result<String, String> {
    let want = expected_version(root)?;
    let source = root.join(SOURCE_DIR);
    if !source.is_dir() {
        return Err(format!(
            "{SOURCE_DIR}/ is missing: copy the game's Lua source (Balatro {want}, gitignored) \
             into {}",
            source.display()
        ));
    }
    let path = source.join("version.jkr");
    let bytes = fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let text = String::from_utf8_lossy(&bytes);
    let got = text.lines().next().unwrap_or_default().trim();
    if got != want {
        return Err(format!(
            "{SOURCE_DIR}/version.jkr is `{got}`, but the data targets `{want}` \
             (BALATRO_VERSION in {VERSION_SOURCE})"
        ));
    }
    Ok(want)
}

/// `jackdaw_data::BALATRO_VERSION`, read from its definition.
fn expected_version(root: &Path) -> Result<String, String> {
    let path = root.join(VERSION_SOURCE);
    let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse_version(&text)
        .map(str::to_owned)
        .ok_or_else(|| format!("{VERSION_SOURCE}: no `pub const BALATRO_VERSION: &str = \"...\";`"))
}

fn parse_version(source: &str) -> Option<&str> {
    source.lines().find_map(|line| {
        line.trim()
            .strip_prefix("pub const BALATRO_VERSION: &str = \"")?
            .strip_suffix("\";")
    })
}

/// A fresh directory under the system temp directory, removed on drop.
struct ScratchDir(PathBuf);

impl ScratchDir {
    fn new() -> Result<Self, String> {
        let path = env::temp_dir().join(format!("jackdaw-verify-source-{}", process::id()));
        // A leftover of an interrupted run whose process id was reused.
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).map_err(|e| format!("cannot create {}: {e}", path.display()))?;
        Ok(Self(path))
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

/// Run one extraction script with `out_dir` as its output directory.
fn extract(root: &Path, script: &str, out_dir: &Path) -> Result<(), String> {
    let output = Command::new("luajit")
        .arg(root.join(script))
        .arg(out_dir)
        .current_dir(root)
        .output()
        .map_err(|e| format!("cannot run luajit: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "{script} failed:\n{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    Ok(())
}

fn load(path: &Path) -> Result<Json, String> {
    let text = fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))
}

/// Top-level entries of a document (object members or array elements).
fn entries(v: &Json) -> usize {
    match v {
        Json::Object(map) => map.len(),
        Json::Array(items) => items.len(),
        _ => 1,
    }
}

/// Scalar values in a document.
fn leaves(v: &Json) -> usize {
    match v {
        Json::Object(map) => map.values().map(leaves).sum(),
        Json::Array(items) => items.iter().map(leaves).sum(),
        _ => 1,
    }
}

/// Push one line per difference between `a` (committed) and `b` (extracted) under `path`.
fn compare(path: &mut String, a: &Json, b: &Json, diffs: &mut Vec<String>) {
    let at = |path: &str| {
        if path.is_empty() {
            ".".to_owned()
        } else {
            path.to_owned()
        }
    };
    match (a, b) {
        (Json::Object(x), Json::Object(y)) => {
            let keys: BTreeSet<&String> = x.keys().chain(y.keys()).collect();
            for key in keys {
                let len = path.len();
                push_key(path, key);
                match (x.get(key), y.get(key)) {
                    (Some(u), Some(v)) => compare(path, u, v, diffs),
                    (Some(u), None) => diffs.push(format!("{path}: only committed: {}", show(u))),
                    (None, Some(v)) => diffs.push(format!("{path}: only extracted: {}", show(v))),
                    (None, None) => unreachable!("key from one of the maps"),
                }
                path.truncate(len);
            }
        }
        (Json::Array(x), Json::Array(y)) => {
            if x.len() != y.len() {
                diffs.push(format!("{}: {} elements vs {}", at(path), x.len(), y.len()));
            }
            for (i, (u, v)) in x.iter().zip(y).enumerate() {
                let len = path.len();
                write!(path, "[{i}]").unwrap();
                compare(path, u, v, diffs);
                path.truncate(len);
            }
        }
        (Json::Number(x), Json::Number(y)) => {
            if !same_number(x, y) {
                diffs.push(format!("{}: {x} vs {y}", at(path)));
            }
        }
        _ => {
            if a != b {
                diffs.push(format!("{}: {} vs {}", at(path), show(a), show(b)));
            }
        }
    }
}

/// Whether two JSON numbers are the same double (`-0` and `0` count as equal).
fn same_number(x: &Number, y: &Number) -> bool {
    let (u, v) = (number(x), number(y));
    u.to_bits() == v.to_bits() || (u == 0.0 && v == 0.0)
}

/// A JSON number as the double it denotes; integers beyond 2^53 round like a parsed float.
fn number(n: &Number) -> f64 {
    n.as_f64().expect("serde_json numbers are finite doubles")
}

/// `.key` for identifier-like keys, `["key"]` otherwise.
fn push_key(path: &mut String, key: &str) {
    let plain = key.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_');
    if plain {
        write!(path, ".{key}").unwrap();
    } else {
        write!(path, "[{}]", Json::from(key)).unwrap();
    }
}

/// A value as compact JSON, shortened for the report.
fn show(v: &Json) -> String {
    const MAX: usize = 60;
    let text = v.to_string();
    match text.char_indices().nth(MAX) {
        Some((cut, _)) => format!("{}...", &text[..cut]),
        None => text,
    }
}

#[cfg(test)]
mod tests {
    use super::{compare, parse_version};
    use serde_json::{Value, json};

    fn diffs(a: Value, b: Value) -> Vec<String> {
        let mut out = Vec::new();
        compare(&mut String::new(), &a, &b, &mut out);
        out
    }

    #[test]
    fn ignores_key_order_and_number_spelling() {
        let a: Value =
            serde_json::from_str(r#"{"b": [1, 0.5], "a": {"x": -0, "y": 1e2}}"#).unwrap();
        let b: Value =
            serde_json::from_str(r#"{"a": {"y": 100.0, "x": 0}, "b": [1.0, 5e-1]}"#).unwrap();
        assert_eq!(diffs(a, b), Vec::<String>::new());
    }

    #[test]
    fn numbers_must_be_the_same_double() {
        // 104/255 written with %.15g reads back one ulp away from the double itself.
        let a: Value = serde_json::from_str("[0.407843137254902]").unwrap();
        let b: Value = serde_json::from_str("[0.40784313725490196]").unwrap();
        assert_eq!(
            diffs(a, b),
            ["[0]: 0.407843137254902 vs 0.40784313725490196"]
        );
        // Adjacent doubles (2 + 2^-51 and 2 + 2^-50).
        let a: Value = serde_json::from_str("[2.0000000000000004]").unwrap();
        let b: Value = serde_json::from_str("[2.0000000000000009]").unwrap();
        assert_eq!(diffs(a, b).len(), 1);
    }

    #[test]
    fn reports_paths() {
        let a = json!({"bl_ox": {"boss_colour": [0.25, 1], "name": "The Ox"}, "odd key": [1, 2]});
        let b = json!({"bl_ox": {"boss_colour": [0.5, 1], "mult": 2}, "odd key": [2, 1, 3]});
        assert_eq!(
            diffs(a, b),
            [
                ".bl_ox.boss_colour[0]: 0.25 vs 0.5",
                ".bl_ox.mult: only extracted: 2",
                ".bl_ox.name: only committed: \"The Ox\"",
                "[\"odd key\"]: 2 elements vs 3",
                "[\"odd key\"][0]: 1 vs 2",
                "[\"odd key\"][1]: 2 vs 1",
            ]
        );
        assert_eq!(diffs(json!([]), json!({})), [".: [] vs {}"]);
        assert_eq!(
            diffs(json!({"a": true}), json!({"a": "true"})),
            [".a: true vs \"true\""]
        );
    }

    #[test]
    fn reads_the_version_constant() {
        let source =
            "//! Doc.\n\n/// The release.\npub const BALATRO_VERSION: &str = \"1.0.1o-FULL\";\n";
        assert_eq!(parse_version(source), Some("1.0.1o-FULL"));
        assert_eq!(parse_version("const OTHER: &str = \"x\";"), None);
    }
}

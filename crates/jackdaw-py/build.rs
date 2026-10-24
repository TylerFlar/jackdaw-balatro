//! Bakes build metadata into the extension for `jackdaw._core._build_info()`.

use std::env;
use std::process::Command;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");

    let var = |name: &str| env::var(name).unwrap_or_else(|_| "unknown".to_owned());
    let rustc = Command::new(var("RUSTC"))
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map_or_else(|| "unknown".to_owned(), |version| version.trim().to_owned());

    // Cargo's PROFILE is "release" for release-derived profiles and "debug" for the rest.
    println!("cargo::rustc-env=JACKDAW_BUILD_PROFILE={}", var("PROFILE"));
    println!(
        "cargo::rustc-env=JACKDAW_BUILD_OPT_LEVEL={}",
        var("OPT_LEVEL")
    );
    println!("cargo::rustc-env=JACKDAW_BUILD_TARGET={}", var("TARGET"));
    println!("cargo::rustc-env=JACKDAW_BUILD_RUSTC={rustc}");
}

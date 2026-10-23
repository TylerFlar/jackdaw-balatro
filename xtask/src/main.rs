//! Repository automation for jackdaw: `cargo xtask <command>`.
//!
//! A std-only command line (no argument-parsing dependency) so it builds in seconds.
#![forbid(unsafe_code)]

mod codegen;
mod verify_source;

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{self, ExitCode};

const USAGE: &str = "\
Usage: cargo xtask <command> [options]

Commands:
  codegen [--check]  Generate crates/jackdaw-data from jackdaw/engine/data/*.json;
                     with --check, fail if the checked-in output is stale (CI)
  verify-source      Re-extract the game data from a local balatro_source/ with
                     LuaJIT into a temporary directory and compare it value by
                     value with the committed JSON; fails on any difference
                     (local only)
  lua-oracles        Regenerate the LuaJIT-executed fixtures (RNG, hand
                     evaluation) and the hands, challenges and blind-amount
                     JSON from a local balatro_source/ (local only)
  help               Print this message
";

#[derive(Debug, PartialEq, Eq)]
enum Command {
    Codegen { check: bool },
    VerifySource,
    LuaOracles,
    Help,
}

impl Command {
    fn name(&self) -> &'static str {
        match self {
            Self::Codegen { .. } => "codegen",
            Self::VerifySource => "verify-source",
            Self::LuaOracles => "lua-oracles",
            Self::Help => "help",
        }
    }
}

fn parse(args: &[String]) -> Result<Command, String> {
    let Some((name, rest)) = args.split_first() else {
        return Err("missing command".to_owned());
    };
    let mut command = match name.as_str() {
        "codegen" => Command::Codegen { check: false },
        "verify-source" => Command::VerifySource,
        "lua-oracles" => Command::LuaOracles,
        "help" | "-h" | "--help" => Command::Help,
        other => return Err(format!("unknown command `{other}`")),
    };
    for arg in rest {
        match (&mut command, arg.as_str()) {
            (Command::Codegen { check }, "--check") => *check = true,
            _ => return Err(format!("{name}: unexpected argument `{arg}`")),
        }
    }
    Ok(command)
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args_os()
        .skip(1)
        .map(|arg| arg.to_string_lossy().into_owned())
        .collect();
    let command = match parse(&args) {
        Ok(command) => command,
        Err(message) => {
            eprintln!("error: {message}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let result = match command {
        Command::Help => {
            print!("{USAGE}");
            Ok(())
        }
        Command::Codegen { check } => codegen::run(&repo_root(), check),
        Command::VerifySource => verify_source::run(&repo_root()),
        Command::LuaOracles => lua_oracles(),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("xtask {}: {message}", command.name());
            ExitCode::FAILURE
        }
    }
}

/// The repository root (the parent of `xtask/`).
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives in the repository root")
        .to_path_buf()
}

/// LuaJIT-executed oracles: run each script on the local Balatro source and write its fixture.
fn lua_oracles() -> Result<(), String> {
    let root = repo_root();
    if !root
        .join("balatro_source/functions/misc_functions.lua")
        .is_file()
    {
        return Err("balatro_source/ (the local Balatro 1.0.1o source) is missing".to_owned());
    }
    // The game-table extract writes its own files (jackdaw/engine/data/{hands,challenges,
    // blind_amounts}.json); `cargo xtask codegen` turns them into Rust.
    let status = process::Command::new("luajit")
        .arg(root.join("scripts/extract_game_tables.lua"))
        .current_dir(&root)
        .status()
        .map_err(|e| format!("cannot run luajit: {e}"))?;
    if !status.success() {
        return Err("scripts/extract_game_tables.lua failed".to_owned());
    }
    let oracles = [
        (
            "scripts/luajit_rng_oracle.lua",
            "tests/fixtures/rng_luajit.json",
        ),
        (
            "scripts/luajit_hand_eval_oracle.lua",
            "tests/fixtures/hand_eval_luajit.json",
        ),
    ];
    for (script, fixture) in oracles {
        let output = process::Command::new("luajit")
            .arg(root.join(script))
            .current_dir(&root)
            .output()
            .map_err(|e| format!("cannot run luajit: {e}"))?;
        if !output.status.success() {
            return Err(format!(
                "{script} failed:
{}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        fs::write(root.join(fixture), &output.stdout)
            .map_err(|e| format!("cannot write {fixture}: {e}"))?;
        println!("{script} -> {fixture} ({} bytes)", output.stdout.len());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Command, parse};

    fn args(list: &[&str]) -> Vec<String> {
        list.iter().map(|arg| (*arg).to_owned()).collect()
    }

    #[test]
    fn parses_commands_and_flags() {
        assert_eq!(
            parse(&args(&["codegen"])),
            Ok(Command::Codegen { check: false })
        );
        assert_eq!(
            parse(&args(&["codegen", "--check"])),
            Ok(Command::Codegen { check: true })
        );
        assert_eq!(parse(&args(&["verify-source"])), Ok(Command::VerifySource));
        assert_eq!(parse(&args(&["lua-oracles"])), Ok(Command::LuaOracles));
        assert_eq!(parse(&args(&["--help"])), Ok(Command::Help));
    }

    #[test]
    fn rejects_bad_input() {
        assert!(parse(&[]).is_err());
        assert!(parse(&args(&["bench"])).is_err());
        assert!(parse(&args(&["lua-oracles", "--check"])).is_err());
    }
}

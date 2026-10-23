//! Repository automation for jackdaw: `cargo xtask <command>`.
//!
//! A std-only command line (no argument-parsing dependency) so it builds in seconds.
#![forbid(unsafe_code)]

use std::env;
use std::process::ExitCode;

const USAGE: &str = "\
Usage: cargo xtask <command> [options]

Commands:
  codegen [--check]  Generate crates/jackdaw-data from jackdaw/engine/data/*.json;
                     with --check, fail if the checked-in output is stale (CI)
  verify-source      Re-extract the data from a local balatro_source/ and diff it
                     against the committed JSON (local only)
  lua-oracles        Regenerate the LuaJIT-executed RNG and blind-amount fixtures
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
    match parse(&args) {
        Ok(Command::Help) => {
            print!("{USAGE}");
            ExitCode::SUCCESS
        }
        Ok(command) => {
            eprintln!("xtask {}: not implemented yet (phase 2)", command.name());
            ExitCode::from(2)
        }
        Err(message) => {
            eprintln!("error: {message}\n\n{USAGE}");
            ExitCode::from(2)
        }
    }
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

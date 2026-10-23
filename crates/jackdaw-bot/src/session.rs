//! A BalatroBot session: what one running game with the mod loaded holds.
//!
//! The session is the sim's `G`: no run (the main menu), a run in progress, or a run that a
//! failed `load` deleted. Requests go through [`Session::dispatch`] (the dispatcher's four
//! tiers) or, as JSON-RPC text, through [`Session::handle_body`] (the server's envelope).
//!
//! `save` and `load` write and read the sim's own snapshot (`jackdaw.snapshot/1` JSON, see
//! `docs/snapshot-schema.md`), not the game's compressed `.jkr` files. With a filesystem root
//! set ([`SessionConfig::fs_root`]), their paths must lie inside it.

use std::path::{Path, PathBuf};

use jackdaw_core::game::Game;

use crate::dispatcher;
use crate::errors::BotError;
use crate::json::Json;
use crate::lua::Value;
use crate::states::GState;
use crate::view::{self, Source};

/// What the game holds.
#[derive(Clone, Debug, Default)]
pub enum Run {
    /// The main menu (`G.STATE == G.STATES.MENU`).
    #[default]
    Menu,
    /// A run in progress.
    Active(Box<Game>),
    /// A failed `load` ran `G:delete_run()` before finding the file unreadable: `G.STATE = -1`
    /// and no card areas, with the deleted run's `G.GAME` (`None`: the menu's).
    Deleted(Option<Box<Game>>),
}

/// Settings of a session.
#[derive(Clone, Debug, Default)]
pub struct SessionConfig {
    /// When set, `save` and `load` only touch files inside this directory.
    pub fs_root: Option<PathBuf>,
    /// Register BalatroBot's debug-mode test endpoints (`test_endpoint`, `test_state_endpoint`,
    /// `test_error_endpoint`, `test_validation`), as `BALATROBOT_DEBUG=1` does.
    pub debug_endpoints: bool,
}

/// One BalatroBot session.
#[derive(Clone, Debug, Default)]
pub struct Session {
    pub run: Run,
    pub config: SessionConfig,
    /// The JSON-RPC envelope's state (`current_request_id`).
    pub envelope: crate::server::Envelope,
    /// State of the generator behind seedless `start` calls.
    seed_state: u64,
}

impl Session {
    pub fn new(config: SessionConfig) -> Session {
        Session {
            run: Run::Menu,
            config,
            ..Session::default()
        }
    }

    /// A session resting in `game`.
    pub fn with_game(game: Game, config: SessionConfig) -> Session {
        Session {
            run: Run::Active(Box::new(game)),
            config,
            ..Session::default()
        }
    }

    /// The run in progress.
    pub fn game(&self) -> Option<&Game> {
        match &self.run {
            Run::Active(g) => Some(g),
            _ => None,
        }
    }

    pub fn game_mut(&mut self) -> Option<&mut Game> {
        match &mut self.run {
            Run::Active(g) => Some(g),
            _ => None,
        }
    }

    /// The run whose `G.GAME` the game holds: the active one, or one a failed `load` deleted.
    pub fn g_game_mut(&mut self) -> Option<&mut Game> {
        match &mut self.run {
            Run::Active(g) | Run::Deleted(Some(g)) => Some(g),
            _ => None,
        }
    }

    /// `G.STATE`; `None` is the `-1` a deleted run leaves.
    pub fn state(&self) -> Option<GState> {
        match &self.run {
            Run::Menu => Some(GState::Menu),
            Run::Active(g) => Some(GState::of(g)),
            Run::Deleted(_) => None,
        }
    }

    /// `G.STAGE == G.STAGES.RUN`: false only at the main menu (a deleted run keeps its stage).
    pub fn in_run_stage(&self) -> bool {
        match &self.run {
            Run::Menu => false,
            Run::Active(_) => true,
            Run::Deleted(g) => g.is_some(),
        }
    }

    /// The gamestate view.
    pub fn view(&self) -> Json {
        view::gamestate(self.source())
    }

    pub fn source(&self) -> Source<'_> {
        match &self.run {
            Run::Menu => Source::Menu,
            Run::Deleted(None) => Source::DeletedMenu,
            Run::Active(g) => Source::Run(g),
            Run::Deleted(Some(g)) => Source::Deleted(g),
        }
    }

    /// Handle one request (method and decoded params) through the dispatcher.
    pub fn dispatch(&mut self, method: &str, params: Option<&Value>) -> Result<Json, BotError> {
        dispatcher::dispatch(self, Some(method), params, false)
    }

    /// Run the dispatcher's validation of a request (schema, state, the endpoint's own checks)
    /// without executing it.
    pub fn check(&mut self, method: &str, params: Option<&Value>) -> Result<(), BotError> {
        dispatcher::dispatch(self, Some(method), params, true).map(|_| ())
    }

    /// A JSON-RPC 2.0 request body in, the response body out (`server.lua`'s
    /// `handle_jsonrpc`): see [`crate::server`].
    pub fn handle_body(&mut self, body: &[u8]) -> String {
        crate::server::handle_jsonrpc(self, body)
    }

    /// The seed of a seedless `start`: vanilla's `random_string(8, ...)` alphabet
    /// (misc_functions.lua:270-277), from an unseeded source.
    pub(crate) fn random_seed(&mut self) -> String {
        if self.seed_state == 0 {
            use std::hash::{BuildHasher, Hasher};
            let mut h = std::collections::hash_map::RandomState::new().build_hasher();
            h.write_u128(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_nanos()),
            );
            self.seed_state = h.finish() | 1;
        }
        let mut next = || {
            // xorshift64*
            let mut x = self.seed_state;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.seed_state = x;
            (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..8)
            .map(|_| {
                if next() > 0.7 {
                    char::from(b'1' + (next() * 9.0) as u8)
                } else if next() > 0.45 {
                    char::from(b'A' + (next() * 14.0) as u8)
                } else {
                    char::from(b'P' + (next() * 11.0) as u8)
                }
            })
            .collect()
    }

    /// The file a `save` or `load` path names, if the session may touch it.
    pub(crate) fn resolve_path(&self, path: &str) -> Option<PathBuf> {
        let p = Path::new(path);
        let Some(root) = &self.config.fs_root else {
            return Some(p.to_path_buf());
        };
        let joined = if p.is_absolute() {
            p.to_path_buf()
        } else {
            root.join(p)
        };
        let normal = normalize(&joined);
        let root = normalize(root);
        normal.starts_with(&root).then_some(normal)
    }
}

/// `path` with `.` and `..` resolved lexically (the file may not exist yet).
fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn random_seeds_use_vanilla_characters() {
        let mut s = Session::default();
        for _ in 0..50 {
            let seed = s.random_seed();
            assert_eq!(seed.len(), 8);
            assert!(
                seed.chars()
                    .all(|c| ('1'..='9').contains(&c) || (c.is_ascii_uppercase() && c != 'O'))
            );
        }
    }

    #[test]
    fn fs_root_confines_paths() {
        let root = std::env::temp_dir().join("jackdaw_bot_root");
        let s = Session::new(SessionConfig {
            fs_root: Some(root.clone()),
            debug_endpoints: false,
        });
        assert_eq!(
            s.resolve_path("a.json"),
            Some(normalize(&root.join("a.json")))
        );
        assert_eq!(s.resolve_path("../escape.json"), None);
        let free = Session::default();
        assert!(free.resolve_path("../anywhere.json").is_some());
    }
}

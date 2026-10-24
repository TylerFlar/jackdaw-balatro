//! The BalatroBot layer: a 1:1 port of BalatroBot's command space and gamestate view.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot, commit
//! `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`, vendored under `third_party/balatrobot/v1.5.2`),
//! so anything trained on the simulator can drive the live game through BalatroBot unchanged:
//!
//! - [`json`] and [`lua`]: Steamodded's rxi JSON (empty tables as `[]`, `%.14g` numbers) and the
//!   Lua values requests decode to;
//! - [`validator`] and [`dispatcher`]: the schema checks and the four-tier dispatch, with the
//!   exact error names, codes and messages ([`errors`]);
//! - [`view`]: `gamestate.lua`, field for field (names instead of localized text);
//! - [`endpoints`]: one module per endpoint, branch for branch;
//! - [`session`]: a game with the mod loaded (menu, run, `save`/`load` of snapshots);
//! - [`server`]: the JSON-RPC 2.0 envelope and the HTTP server behind `jackdaw serve`;
//! - [`legality`]: the engine's actions as requests, and `BALATROBOT` legality.
//!
//! `docs/balatrobot.md` lists every place the sim departs from the live game.
#![forbid(unsafe_code)]

pub mod dispatcher;
pub mod endpoints;
pub mod errors;
pub mod json;
pub mod legality;
pub mod lua;
pub mod server;
pub mod session;
pub mod states;
pub mod validator;
pub mod view;

pub use legality::{Legality, bot_accepts, legal_actions, translate};
pub use session::{Session, SessionConfig};

/// The BalatroBot release (coder/balatrobot) whose endpoints and view this crate mirrors.
pub const BALATROBOT_VERSION: &str = "1.5.2";

/// The upstream commit every ported module pins.
pub const BALATROBOT_COMMIT: &str = "9052d76f14723293f6c6b2cecaa791a5c4ae68f3";

#[cfg(test)]
mod tests {
    #[test]
    fn pins_balatrobot_and_the_vanilla_data() {
        assert_eq!(super::BALATROBOT_VERSION, "1.5.2");
        assert_eq!(jackdaw_core::data::BALATRO_VERSION, "1.0.1o-FULL");
    }

    #[test]
    fn every_module_pins_the_upstream_commit() {
        let sources: &[(&str, &str)] = &[
            ("lua.rs", include_str!("lua.rs")),
            ("json.rs", include_str!("json.rs")),
            ("errors.rs", include_str!("errors.rs")),
            ("validator.rs", include_str!("validator.rs")),
            ("dispatcher.rs", include_str!("dispatcher.rs")),
            ("states.rs", include_str!("states.rs")),
            ("view.rs", include_str!("view.rs")),
            ("server.rs", include_str!("server.rs")),
            ("endpoints/mod.rs", include_str!("endpoints/mod.rs")),
            ("endpoints/add.rs", include_str!("endpoints/add.rs")),
            ("endpoints/buy.rs", include_str!("endpoints/buy.rs")),
            (
                "endpoints/cash_out.rs",
                include_str!("endpoints/cash_out.rs"),
            ),
            ("endpoints/discard.rs", include_str!("endpoints/discard.rs")),
            (
                "endpoints/gamestate.rs",
                include_str!("endpoints/gamestate.rs"),
            ),
            ("endpoints/health.rs", include_str!("endpoints/health.rs")),
            ("endpoints/load.rs", include_str!("endpoints/load.rs")),
            ("endpoints/menu.rs", include_str!("endpoints/menu.rs")),
            (
                "endpoints/next_round.rs",
                include_str!("endpoints/next_round.rs"),
            ),
            ("endpoints/pack.rs", include_str!("endpoints/pack.rs")),
            ("endpoints/play.rs", include_str!("endpoints/play.rs")),
            (
                "endpoints/rearrange.rs",
                include_str!("endpoints/rearrange.rs"),
            ),
            ("endpoints/reroll.rs", include_str!("endpoints/reroll.rs")),
            ("endpoints/save.rs", include_str!("endpoints/save.rs")),
            (
                "endpoints/screenshot.rs",
                include_str!("endpoints/screenshot.rs"),
            ),
            ("endpoints/select.rs", include_str!("endpoints/select.rs")),
            ("endpoints/sell.rs", include_str!("endpoints/sell.rs")),
            ("endpoints/set.rs", include_str!("endpoints/set.rs")),
            ("endpoints/skip.rs", include_str!("endpoints/skip.rs")),
            ("endpoints/start.rs", include_str!("endpoints/start.rs")),
            (
                "endpoints/test_endpoints.rs",
                include_str!("endpoints/test_endpoints.rs"),
            ),
            (
                "endpoints/use_consumable.rs",
                include_str!("endpoints/use_consumable.rs"),
            ),
        ];
        for (name, text) in sources {
            assert!(
                text.contains(super::BALATROBOT_COMMIT),
                "{name} does not pin the upstream commit"
            );
        }
    }
}

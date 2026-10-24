//! `G.STATES`: the game states BalatroBot gates its endpoints on and reports in the view.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/utils/enums.lua` (the `State` alias) and the `requires_state` lists of
//! `src/lua/endpoints/*.lua`. The values are vanilla's (globals.lua:284-304) plus Steamodded's
//! `SMODS_BOOSTER_OPENED`, the one state every booster pack opens into under Steamodded (vanilla
//! has one state per pack kind).

use jackdaw_core::game::{Game, LuaState};

/// A `G.STATES` value.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GState {
    SelectingHand,
    HandPlayed,
    DrawToHand,
    GameOver,
    Shop,
    PlayTarot,
    BlindSelect,
    RoundEval,
    TarotPack,
    PlanetPack,
    Menu,
    Tutorial,
    Splash,
    Sandbox,
    SpectralPack,
    DemoCta,
    StandardPack,
    BuffoonPack,
    NewRound,
    SmodsBoosterOpened,
}

impl GState {
    /// The key in `G.STATES` (what `get_state_name` returns).
    pub fn name(self) -> &'static str {
        match self {
            GState::SelectingHand => "SELECTING_HAND",
            GState::HandPlayed => "HAND_PLAYED",
            GState::DrawToHand => "DRAW_TO_HAND",
            GState::GameOver => "GAME_OVER",
            GState::Shop => "SHOP",
            GState::PlayTarot => "PLAY_TAROT",
            GState::BlindSelect => "BLIND_SELECT",
            GState::RoundEval => "ROUND_EVAL",
            GState::TarotPack => "TAROT_PACK",
            GState::PlanetPack => "PLANET_PACK",
            GState::Menu => "MENU",
            GState::Tutorial => "TUTORIAL",
            GState::Splash => "SPLASH",
            GState::Sandbox => "SANDBOX",
            GState::SpectralPack => "SPECTRAL_PACK",
            GState::DemoCta => "DEMO_CTA",
            GState::StandardPack => "STANDARD_PACK",
            GState::BuffoonPack => "BUFFOON_PACK",
            GState::NewRound => "NEW_ROUND",
            GState::SmodsBoosterOpened => "SMODS_BOOSTER_OPENED",
        }
    }

    /// The live game's `G.STATE` for a run resting at a decision point. Every pack is
    /// Steamodded's booster state.
    pub fn of(game: &Game) -> GState {
        match game.rt.state {
            LuaState::SelectingHand => GState::SelectingHand,
            LuaState::HandPlayed => GState::HandPlayed,
            LuaState::DrawToHand => GState::DrawToHand,
            LuaState::GameOver => GState::GameOver,
            LuaState::Shop => GState::Shop,
            LuaState::PlayTarot => GState::PlayTarot,
            LuaState::BlindSelect => GState::BlindSelect,
            LuaState::RoundEval => GState::RoundEval,
            LuaState::NewRound => GState::NewRound,
            LuaState::TarotPack
            | LuaState::PlanetPack
            | LuaState::SpectralPack
            | LuaState::StandardPack
            | LuaState::BuffoonPack => GState::SmodsBoosterOpened,
        }
    }
}

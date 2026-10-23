//! Game rules that belong to no single object.

use crate::game::Game;

/// `get_blind_amount(ante)` (misc_functions.lua:919-955) at the run's
/// `G.GAME.modifiers.scaling`, from LuaJIT's own values.
pub fn get_blind_amount(g: &Game, ante: f64) -> f64 {
    let scaling = g.modifier_num("scaling").unwrap_or(1.0);
    jackdaw_data::blind_amount(scaling as u8, ante as i64)
}

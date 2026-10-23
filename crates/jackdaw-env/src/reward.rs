//! The env's view of a step for rewards (`jackdaw/env/rewards.py`): state summaries before and
//! after an action (`GameState`), the derived events (`Transition`), and the built-in rewards
//! computed the way Python computes them.

use jackdaw_core::game::Game;
use jackdaw_core::state::{BlindSlot, Phase};

/// `GameState`: the summary of a run the env hands to rewards and agents.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StateSummary {
    pub phase: Phase,
    pub ante: f64,
    pub round: f64,
    pub dollars: f64,
    pub hands_left: f64,
    pub discards_left: f64,
    pub hand_size: f64,
    pub joker_slots: f64,
    pub consumable_slots: f64,
    pub blind_on_deck: Option<BlindSlot>,
    /// The current blind's target (0 without a blind).
    pub blind_chips: f64,
    pub chips: f64,
    /// [`run_won`].
    pub won: bool,
    /// The run is over (game over).
    pub done: bool,
}

/// `run_won(gs)`: the run is won. `G.GAME.won` is set before the game-over test for the boss of
/// exactly `win_ante`, so a loss to that boss sets it too (G260); such a loss is no victory. A
/// loss later in endless mode keeps the win.
pub fn run_won(game: &Game) -> bool {
    let gs = &game.gs;
    gs.won && !(gs.phase == Phase::GameOver && gs.round_resets.ante == gs.win_ante)
}

impl StateSummary {
    pub fn of(game: &Game) -> StateSummary {
        let gs = &game.gs;
        let cr = &gs.current_round;
        StateSummary {
            phase: gs.phase,
            ante: gs.round_resets.ante,
            round: gs.round,
            dollars: gs.dollars,
            hands_left: cr.hands_left,
            discards_left: cr.discards_left,
            hand_size: gs.hand_size,
            joker_slots: gs.joker_slots,
            consumable_slots: gs.consumable_slots,
            blind_on_deck: gs.blind_on_deck,
            blind_chips: gs.blind.as_ref().map_or(0.0, |b| b.chips),
            chips: gs.chips,
            won: run_won(game),
            done: gs.phase == Phase::GameOver,
        }
    }
}

/// `Transition`: one step as a reward sees it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transition {
    pub prev: StateSummary,
    pub curr: StateSummary,
    pub terminated: bool,
    pub truncated: bool,
}

impl Transition {
    pub fn done(&self) -> bool {
        self.terminated || self.truncated
    }

    /// The hand just played ended the round.
    pub fn blind_beaten(&self) -> bool {
        self.prev.phase == Phase::SelectingHand && self.curr.phase == Phase::RoundEval
    }

    pub fn boss_beaten(&self) -> bool {
        self.blind_beaten() && self.prev.blind_on_deck == Some(BlindSlot::Boss)
    }

    /// The ante of the blind being played.
    pub fn blind_ante(&self) -> f64 {
        self.prev.ante
    }

    /// Chips this step added toward the current blind (0 outside a hand play; NaN reads as 0).
    pub fn chips_scored(&self) -> f64 {
        if self.prev.phase != Phase::SelectingHand {
            return 0.0;
        }
        let delta = self.curr.chips - self.prev.chips;
        if delta > 0.0 { delta } else { 0.0 }
    }

    pub fn run_won(&self) -> bool {
        self.curr.won && !self.prev.won
    }
}

/// `ShapedReward`'s weights (penalties as positive magnitudes).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShapedWeights {
    pub step_cost: f64,
    pub shop_step_cost: f64,
    pub blind_beaten: f64,
    pub boss_beaten: f64,
    pub hands_left: f64,
    pub score_progress: f64,
    pub win: f64,
    pub loss: f64,
}

impl Default for ShapedWeights {
    fn default() -> Self {
        ShapedWeights {
            step_cost: 0.001,
            shop_step_cost: 0.002,
            blind_beaten: 0.15,
            boss_beaten: 0.10,
            hands_left: 0.01,
            score_progress: 0.02,
            win: 0.5,
            loss: 0.2,
        }
    }
}

/// A built-in reward function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reward {
    /// `sparse_reward`: +1 for a won run, -1 otherwise, at episode end.
    Sparse,
    /// `blind_progress_reward`: +1 per blind beaten, -1 when an episode ends short of a win.
    BlindProgress,
    /// `ShapedReward(...)`.
    Shaped(ShapedWeights),
    /// No reward here (Python computes a custom one): always 0.
    None,
}

/// Python's `min(a, b)`: `a` unless `b < a`.
fn py_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

impl Reward {
    pub fn of(&self, t: &Transition) -> f64 {
        match self {
            Reward::Sparse => {
                if t.done() {
                    if t.curr.won { 1.0 } else { -1.0 }
                } else {
                    0.0
                }
            }
            Reward::BlindProgress => {
                let mut r = if t.blind_beaten() { 1.0 } else { 0.0 };
                if t.done() && !t.curr.won {
                    r -= 1.0;
                }
                r
            }
            Reward::Shaped(w) => {
                let mut r = -(if t.prev.phase == Phase::Shop {
                    w.shop_step_cost
                } else {
                    w.step_cost
                });
                if t.blind_beaten() {
                    let ante_scale = t.blind_ante() / 8.0;
                    r += w.blind_beaten * ante_scale;
                    if t.boss_beaten() {
                        r += w.boss_beaten * ante_scale;
                    }
                    r += w.hands_left * t.curr.hands_left;
                }
                let target = t.prev.blind_chips;
                let scored = t.chips_scored();
                if 0.0 < target && target < f64::INFINITY && scored != 0.0 {
                    r += w.score_progress * py_min(scored / target, 1.0);
                }
                if t.run_won() {
                    r += w.win;
                }
                if t.done() && !t.curr.won {
                    r -= w.loss;
                }
                r
            }
            Reward::None => 0.0,
        }
    }
}

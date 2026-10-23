//! The engine: a run's state plus the machinery that moves it (`G.STATE`, `G.STATE_COMPLETE`
//! and the event queue).
//!
//! Lua runs gameplay through `G.E_MANAGER` events and per-frame state handlers (`Game:update`,
//! game.lua:2449-2607): each frame processes the queue up to the first blocking event, then
//! calls the handler of the current `G.STATE`. A handler acts once `G.STATE_COMPLETE` is false
//! and appends its events to the back of the queue.
//!
//! The port keeps that structure with time removed: [`Game::settle`] runs queued events in FIFO
//! order and calls the state handler after each one, until the queue is empty and the handler
//! has nothing to do. Delays only order events in Lua, so they are dropped. Events that Lua
//! runs off the blocking chain (`blockable = false`, timed) also run in queue order; where
//! that differs from the real game, the call site says so.

use std::collections::VecDeque;

use serde_json::Value;

use crate::area::AreaId;
use crate::event::Event;
use crate::snapshot::{self, SnapshotError};
use crate::state::{GameState, Phase};

/// `G.STATE`: the game's state machine (globals.lua `G.STATES`), the states a run passes through.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum LuaState {
    SelectingHand,
    HandPlayed,
    DrawToHand,
    GameOver,
    Shop,
    PlayTarot,
    #[default]
    BlindSelect,
    RoundEval,
    TarotPack,
    PlanetPack,
    SpectralPack,
    StandardPack,
    BuffoonPack,
    NewRound,
}

impl LuaState {
    /// The decision phase this state rests in, if the game waits for input here.
    pub fn phase(self) -> Option<Phase> {
        Some(match self {
            LuaState::SelectingHand => Phase::SelectingHand,
            LuaState::BlindSelect => Phase::BlindSelect,
            LuaState::RoundEval => Phase::RoundEval,
            LuaState::Shop => Phase::Shop,
            LuaState::TarotPack
            | LuaState::PlanetPack
            | LuaState::SpectralPack
            | LuaState::StandardPack
            | LuaState::BuffoonPack => Phase::PackOpening,
            LuaState::GameOver => Phase::GameOver,
            LuaState::HandPlayed
            | LuaState::DrawToHand
            | LuaState::PlayTarot
            | LuaState::NewRound => return None,
        })
    }

    /// The pack state for a booster kind (`Card:open`, card.lua:1681).
    pub fn for_pack(kind: &str) -> Option<LuaState> {
        Some(match kind {
            "Arcana" => LuaState::TarotPack,
            "Celestial" => LuaState::PlanetPack,
            "Spectral" => LuaState::SpectralPack,
            "Standard" => LuaState::StandardPack,
            "Buffoon" => LuaState::BuffoonPack,
            _ => return None,
        })
    }
}

/// Why a step was refused.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepError {
    /// The action is not legal here; the state is unchanged.
    Illegal(String),
    /// The step needs game logic the port doesn't have yet; the state is unchanged.
    Unported(String),
}

impl std::fmt::Display for StepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StepError::Illegal(m) => write!(f, "illegal action: {m}"),
            StepError::Unported(m) => write!(f, "not ported yet: {m}"),
        }
    }
}

impl std::error::Error for StepError {}

/// Engine state that exists only while a step runs, plus `G.STATE`. Between steps the queue is
/// empty and the buffers are zero.
#[derive(Clone, Debug, Default)]
pub struct Runtime {
    /// `G.STATE`.
    pub state: LuaState,
    /// `G.STATE_COMPLETE`.
    pub state_complete: bool,
    /// `G.E_MANAGER`'s base queue.
    pub queue: VecDeque<Event>,
    /// `G.hand.highlighted`, as sort ids in highlight order.
    pub highlighted: Vec<u32>,
    /// `G.GAME.dollar_buffer`: money earned this hand whose `ease_dollars` hasn't run yet.
    pub dollar_buffer: f64,
    /// `G.GAME.joker_buffer`: jokers queued for creation.
    pub joker_buffer: f64,
    /// `G.GAME.consumeable_buffer`: consumables queued for creation.
    pub consumeable_buffer: f64,
    /// `G.GAME.facing_blind`: set by `select_blind`, cleared on reaching round evaluation.
    pub facing_blind: bool,
    /// `update_selecting_hand` calls `end_round()` every frame while hand, deck and play are
    /// empty (game.lua:3056-3058); the port calls it once per stay in `SELECTING_HAND`.
    pub end_round_called: bool,
    /// `G.GAME.round_resets.blind`: the blind chosen at blind selection, for this round.
    pub round_blind: Option<jackdaw_data::BlindId>,
    /// Set when the step reached game logic that isn't ported yet.
    pub unported: Option<String>,
}

/// A run: the persistent [`GameState`] and the engine's [`Runtime`].
#[derive(Clone, Debug)]
pub struct Game {
    pub gs: GameState,
    pub rt: Runtime,
}

impl Game {
    /// A game resting at the snapshot's decision point.
    pub fn from_state(gs: GameState) -> Game {
        let state = match gs.phase {
            Phase::BlindSelect => LuaState::BlindSelect,
            Phase::SelectingHand => LuaState::SelectingHand,
            Phase::RoundEval => LuaState::RoundEval,
            Phase::Shop => LuaState::Shop,
            Phase::GameOver => LuaState::GameOver,
            Phase::PackOpening => gs
                .pack
                .as_ref()
                .and_then(|p| LuaState::for_pack(&p.kind))
                .unwrap_or(LuaState::TarotPack),
        };
        // Cerulean Bell's forced card stays highlighted (CardArea:update, cardarea.lua:255-259).
        let highlighted = gs
            .areas
            .hand
            .iter()
            .filter(|c| c.ability.forced_selection)
            .map(|c| c.sort_id)
            .collect();
        let in_round = gs.phase == Phase::SelectingHand;
        let rt = Runtime {
            state,
            state_complete: true,
            highlighted,
            facing_blind: in_round,
            round_blind: if in_round {
                gs.blind.as_ref().and_then(|b| b.key)
            } else {
                None
            },
            ..Runtime::default()
        };
        Game { gs, rt }
    }

    pub fn from_snapshot(v: &Value) -> Result<Game, SnapshotError> {
        snapshot::import(v).map(Game::from_state)
    }

    /// The canonical snapshot of the run (the game must be at rest).
    pub fn snapshot(&self) -> Value {
        debug_assert!(
            self.rt.queue.is_empty(),
            "snapshot while events are pending"
        );
        snapshot::export(&self.gs)
    }

    /// Queue an event (`G.E_MANAGER:add_event`).
    pub fn queue(&mut self, event: Event) {
        self.rt.queue.push_back(event);
    }

    /// Set `G.STATE` and `G.STATE_COMPLETE`.
    pub fn set_state(&mut self, state: LuaState, complete: bool) {
        if state != self.rt.state {
            self.rt.end_round_called = false;
        }
        self.rt.state = state;
        self.rt.state_complete = complete;
        if let Some(phase) = state.phase() {
            self.gs.phase = phase;
        }
    }

    /// Run events and state handlers until the game waits for input.
    pub fn settle(&mut self) {
        let mut budget = 1_000_000u32;
        loop {
            budget = budget
                .checked_sub(1)
                .expect("the event loop did not settle");
            let ran = match self.rt.queue.pop_front() {
                Some(event) => {
                    self.run_event(event);
                    true
                }
                None => false,
            };
            let handled = self.update_state();
            if !ran && !handled {
                break;
            }
        }
        debug_assert!(
            self.rt.state.phase().is_some(),
            "settled in a transient state {:?}",
            self.rt.state
        );
    }

    /// `G.playing_cards`: every playing card of the run, in `playing_card` order.
    pub fn playing_card_ids(&self) -> Vec<u32> {
        let mut cards: Vec<(u32, u32)> = AreaId::PLAYING
            .into_iter()
            .flat_map(|a| self.gs.areas.get(a).iter())
            .chain(self.gs.sim.pending_deck_bottom.iter())
            .map(|c| (c.playing_card.unwrap_or(u32::MAX), c.sort_id))
            .collect();
        cards.sort_unstable();
        cards.into_iter().map(|(_, s)| s).collect()
    }
}

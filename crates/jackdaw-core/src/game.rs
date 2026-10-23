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
    /// `G.shop` exists: from the shop's first construction until `toggle_shop` closes it
    /// (it stays while a pack bought there is open).
    pub shop_open: bool,
    /// `G.GAME.round_resets.blind`: the blind chosen at blind selection, for this round.
    pub round_blind: Option<jackdaw_data::BlindId>,
    /// Set when the step reached game logic that isn't ported yet.
    pub unported: Option<String>,
    /// Card-area settings that only BalatroBot's view reads.
    pub areas: AreaMemo,
    /// The queue's length when the running event started: what it has queued since is behind.
    pub event_base: usize,
}

/// What BalatroBot's gamestate view reads from the `CardArea` objects themselves
/// (`config.card_limit`, `config.card_count`) where no rule and no snapshot field keeps it.
/// It lives with the engine's runtime: a snapshot import restarts it from the cards
/// ([`Game::from_state`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AreaMemo {
    /// `G.deck.config.card_limit`: an emplace that overfills the deck raises it
    /// (cardarea.lua:47-51), every frame lowers it to `#G.playing_cards` (cardarea.lua:263), and
    /// Cryptid, Marble Joker and DNA add one (card.lua:1210, 2595, 3506).
    pub deck_limit: f64,
    /// `G.shop_vouchers.config.card_limit`: 1 for a new shop (UI_definitions.lua:646-651), plus
    /// one per Voucher Tag that fills it (tag.lua:308).
    pub voucher_limit: f64,
    /// The shop's card areas after `toggle_shop` removed the shop: `G.shop:remove()` empties
    /// them, but `G.shop_jokers`, `G.shop_vouchers` and `G.shop_booster` keep pointing at them
    /// until the run is deleted, with the card counts of their last frame.
    pub stale_shop: Option<StaleShop>,
    /// Shop vouchers with no buy button: BalatroBot's `add` puts a voucher in the shop with
    /// `SMODS.add_card`, which never calls `create_shop_card_ui` (sort ids).
    pub no_buy_button: Vec<u32>,
}

/// Card counts and limits of the removed shop's areas (jokers, vouchers, boosters).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StaleShop {
    pub jokers: (f64, f64),
    pub vouchers: (f64, f64),
    pub booster: (f64, f64),
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
        let shop_open = gs.phase == Phase::Shop
            || (gs.phase == Phase::PackOpening
                && gs
                    .pack
                    .as_ref()
                    .is_some_and(|p| p.return_phase == Phase::Shop));
        // A snapshot keeps no CardArea settings: the deck's limit restarts at the run's playing
        // cards (its value whenever the cards are all in the deck), the shop's voucher limit at
        // its cards, and a removed shop's areas are gone (as after `load` in the game).
        let playing = AreaId::PLAYING
            .into_iter()
            .map(|a| gs.areas.get(a).len())
            .sum::<usize>()
            + gs.sim.pending_deck_bottom.len();
        let areas = AreaMemo {
            deck_limit: playing as f64,
            voucher_limit: (gs.areas.shop_vouchers.len() as f64).max(1.0),
            stale_shop: None,
            no_buy_button: Vec::new(),
        };
        let rt = Runtime {
            state,
            state_complete: true,
            highlighted,
            shop_open,
            facing_blind: in_round,
            round_blind: if in_round {
                gs.blind.as_ref().and_then(|b| b.key)
            } else {
                None
            },
            areas,
            ..Runtime::default()
        };
        let mut game = Game { gs, rt };
        game.update_cards();
        game
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

    /// A losing `end_round`'s event sets `G.STATE = G.STATES.GAME_OVER`. The same frame's
    /// `update_game_over` pauses the game (`G.SETTINGS.paused = true`, game.lua:2601, after
    /// the event pass at 2509, in which that blocking event already blocked everything behind
    /// it), and a paused game never runs an event queued before the pause (`pause_skip`,
    /// event.lua): what the event queued never happens. A losing end_round charges no rent,
    /// destroys no joker and changes no hand size (G265). Events queued before it (a joker's
    /// removal from the last hand and its boss re-debuff) keep their place: in the game their
    /// timers fire while the hand's cards are still being drawn to the discard pile.
    pub(crate) fn game_over(&mut self) {
        let base = self.rt.event_base.min(self.rt.queue.len());
        self.rt.queue.truncate(base);
        self.set_state(LuaState::GameOver, false);
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
                    self.rt.event_base = self.rt.queue.len();
                    self.run_event(event);
                    true
                }
                None => false,
            };
            let handled = self.update_state();
            if ran || handled {
                self.update_cards();
            }
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

    /// `G.playing_cards`: every playing card of the run (the cards in the deck, hand, discard
    /// pile and play area), by sort id. Lua's list is in insertion order, which no game rule
    /// depends on: its users count, test membership, or pick by sort id.
    pub fn playing_card_ids(&self) -> Vec<u32> {
        let mut cards: Vec<u32> = AreaId::PLAYING
            .into_iter()
            .flat_map(|a| self.gs.areas.get(a).iter())
            .chain(self.gs.sim.pending_deck_bottom.iter())
            .map(|c| c.sort_id)
            .collect();
        cards.sort_unstable();
        cards
    }
}

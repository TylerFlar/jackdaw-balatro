//! One Gym environment on the Rust engine: `BalatroGymnasiumEnv` over `BalatroEnvironment`, with
//! a built-in reward. The Python wrapper and the vector env both drive it, so a vector env of one
//! steps exactly as the Gym wrapper does.
//!
//! The rules follow the Python env step for step: an action is an index into the action table;
//! the run is converted, stepped and summarised; `terminated` is game over or a won run (with
//! `end_on_win`), `truncated` is the step limit; the episode's best ante and blinds beaten are
//! tracked for the episode-end info; the next table is enumerated unless the episode ended.
//!
//! A step the engine can't take yet (not ported) leaves the run unchanged and ends the episode
//! as truncated, as the step limit does.

use jackdaw_core::game::{Game, StepError as CoreStepError};
use jackdaw_core::run::RunSpec;

use crate::action::{ActionType, FactoredAction, to_engine};
use crate::gym::{ActionTable, MAX_ACTIONS, Xoshiro256};
use crate::mask::{ActionMask, Legality};
use crate::observation::{D_GLOBAL, ENTITY_WIDTHS, Encoder};
use crate::reward::{Reward, StateSummary, Transition, run_won};

/// The most rows of each entity array the Gym observation holds (`balatro_game_spec()`'s
/// `max_count`: hand cards, jokers, consumables, shop items, pack cards).
pub const MAX_COUNTS: [usize; 5] = [8, 5, 2, 10, 5];

/// The Gym observation, flattened: the global context, each entity array padded to its
/// `MAX_COUNTS` rows, then the five entity counts.
pub const GYM_OBS_LEN: usize = D_GLOBAL
    + MAX_COUNTS[0] * ENTITY_WIDTHS[0]
    + MAX_COUNTS[1] * ENTITY_WIDTHS[1]
    + MAX_COUNTS[2] * ENTITY_WIDTHS[2]
    + MAX_COUNTS[3] * ENTITY_WIDTHS[3]
    + MAX_COUNTS[4] * ENTITY_WIDTHS[4]
    + 5;

/// How many free moves (drags and sorts, [`FreeMoves`]) an env allows in a row by default: a
/// drag carries a card any distance and counts once, so ten put up to eleven cards in any order.
pub const DEFAULT_MAX_FREE_MOVES: u32 = 10;

/// The free moves taken in a row ([`ActionType::is_free`]), counted as a player makes them: a
/// sort once, and a drag once however far it carries its card.
///
/// The game moves a dragged card past one neighbour at a time: `CardArea:align_cards` sorts the
/// area by position every frame (cardarea.lua:448, 463 and 528), so a drag over three cards
/// swaps the card with each of them in turn. The envs' swaps are those steps, and a swap that
/// carries on the card the last one moved, the same way, is part of its drag (#70).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FreeMoves {
    count: u32,
    carry: Option<FactoredAction>,
}

impl FreeMoves {
    /// The free moves since the last other action, a drag counting once.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// The swap that carries the last drag on ([`carry_on`]), if the last move was a swap.
    pub fn carry(&self) -> Option<FactoredAction> {
        self.carry
    }

    /// Count `action`, the move just taken.
    pub fn record(&mut self, action: &FactoredAction) {
        if !action.action_type.is_free() {
            *self = FreeMoves::default();
            return;
        }
        if self.carry != Some(*action) {
            self.count += 1;
        }
        self.carry = carry_on(action);
    }

    /// Whether `cap` free moves are taken (`None`: no cap). The mask then offers no free move
    /// but [`FreeMoves::carry`], so the last drag can still go as far as any.
    pub fn spent(&self, cap: Option<u32>) -> bool {
        cap.is_some_and(|cap| self.count >= cap)
    }
}

/// The swap that carries on the drag `action` makes: its card one more place the same way
/// (`SwapHandLeft(i - 1)` after `SwapHandLeft(i)`). `None` for anything but a swap.
pub fn carry_on(action: &FactoredAction) -> Option<FactoredAction> {
    use ActionType::*;
    let i = action.entity_target? as usize;
    let next = match action.action_type {
        SwapHandLeft | SwapJokersLeft => i.checked_sub(1)?,
        SwapHandRight | SwapJokersRight => i + 1,
        _ => return None,
    };
    Some(FactoredAction::with_entity(action.action_type, next))
}

/// The env's settings (`BalatroEnvironment`'s `max_steps` and `end_on_win`, the reward, and the
/// cap on free moves in a row).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EnvConfig {
    pub max_steps: u64,
    pub end_on_win: bool,
    pub reward: Reward,
    /// After this many free moves in a row ([`FreeMoves`]: a drag counts once) the mask allows
    /// none but the swap carrying the last drag on, until another action; `None` for no cap.
    pub max_free_moves: Option<u32>,
}

impl Default for EnvConfig {
    fn default() -> Self {
        EnvConfig {
            max_steps: 10_000,
            end_on_win: true,
            reward: Reward::Sparse,
            max_free_moves: Some(DEFAULT_MAX_FREE_MOVES),
        }
    }
}

/// Why a step was refused; the run is unchanged.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepError {
    /// The action index is past the end of the table.
    BadIndex(usize),
    /// The factored action isn't one the env offers here ([`GymEnv::offers`]).
    NotOffered(FactoredAction),
    /// The table's action has no engine form here.
    Convert(String),
    /// The engine refused the action.
    Illegal(String),
}

impl std::fmt::Display for StepError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StepError::BadIndex(i) => write!(f, "action {i} is not in the action table"),
            StepError::NotOffered(a) => {
                write!(f, "{} ", a.action_type.name())?;
                if let Some(e) = a.entity_target {
                    write!(f, "on entity {e} ")?;
                }
                if let Some(sel) = &a.card_target {
                    write!(f, "with cards {sel:?} ")?;
                }
                f.write_str("is not an action the env offers here")
            }
            StepError::Convert(m) | StepError::Illegal(m) => f.write_str(m),
        }
    }
}

/// What a step did.
#[derive(Clone, Debug, PartialEq)]
pub struct StepResult {
    pub action: FactoredAction,
    pub transition: Transition,
    pub reward: f64,
    /// The engine can't take the step yet: the run is unchanged and the episode was cut.
    pub not_ported: Option<String>,
}

impl StepResult {
    pub fn terminated(&self) -> bool {
        self.transition.terminated
    }

    pub fn truncated(&self) -> bool {
        self.transition.truncated
    }
}

/// One Gym environment.
#[derive(Clone, Debug)]
pub struct GymEnv {
    game: Game,
    encoder: Encoder,
    mask: ActionMask,
    table: ActionTable,
    rng: Xoshiro256,
    pub config: EnvConfig,
    /// `BalatroEnvironment._step_count` (steps this episode, for the step limit).
    step_count: u64,
    /// `BalatroEnvironment.episode_length`.
    pub episode_length: u64,
    /// `BalatroEnvironment.episode_ante` (the ante after the last step).
    pub episode_ante: f64,
    /// `BalatroEnvironment.episode_won`.
    pub episode_won: bool,
    /// The best ante this episode (`balatro/ante_reached`).
    pub max_ante: f64,
    /// Blinds beaten this episode (`balatro/rounds_beaten`).
    pub blinds_beaten: u64,
    /// Free moves since the last other action.
    free_moves: FreeMoves,
    state: StateSummary,
    ended: bool,
    /// The key the player's view names cards under ([`GymEnv::player_view`]).
    view_key: u64,
    /// Every run records how its hands score ([`Game::set_record_scoring`]).
    record_scoring: bool,
}

/// The player's view key of an episode whose sampler starts at `state`: the state's words mixed
/// with SplitMix64's finalizer. It draws nothing from the sampler, so the action tables are
/// those of an env that never makes a view.
pub fn view_key(state: [u64; 4]) -> u64 {
    let mut x = 0x6a61_636b_6461_7776; // "jackdawv"
    for w in state {
        x ^= w;
        x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = x;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        x = z ^ (z >> 31);
    }
    x
}

impl GymEnv {
    /// An env at the start of `spec`'s run, its sampler seeded with `rng_seed`.
    pub fn new(spec: &RunSpec, config: EnvConfig, rng_seed: u64) -> Result<GymEnv, String> {
        let game = Game::new_run(spec)?;
        let state = StateSummary::of(&game);
        let rng = Xoshiro256::seed_from_u64(rng_seed);
        let mut env = GymEnv {
            game,
            encoder: Encoder::new(),
            mask: ActionMask::new(),
            table: ActionTable::new(),
            view_key: view_key(rng.state()),
            rng,
            config,
            step_count: 0,
            episode_length: 0,
            episode_ante: 1.0,
            episode_won: false,
            max_ante: 1.0,
            blinds_beaten: 0,
            free_moves: FreeMoves::default(),
            state,
            ended: false,
            record_scoring: false,
        };
        env.refresh();
        Ok(env)
    }

    pub fn game(&self) -> &Game {
        &self.game
    }

    pub fn state(&self) -> &StateSummary {
        &self.state
    }

    pub fn table(&self) -> &ActionTable {
        &self.table
    }

    pub fn rng_state(&self) -> [u64; 4] {
        self.rng.state()
    }

    /// The episode is over: the table is empty until the next reset.
    pub fn ended(&self) -> bool {
        self.ended
    }

    /// Start a new run (`reset`); reseed the sampler when `rng_seed` is given.
    pub fn reset(&mut self, spec: &RunSpec, rng_seed: Option<u64>) -> Result<(), String> {
        if let Some(seed) = rng_seed {
            self.rng = Xoshiro256::seed_from_u64(seed);
        }
        self.view_key = view_key(self.rng.state());
        self.game = Game::new_run(spec)?;
        self.game.set_record_scoring(self.record_scoring);
        self.step_count = 0;
        self.episode_length = 0;
        self.episode_ante = 1.0;
        self.episode_won = false;
        self.max_ante = 1.0;
        self.blinds_beaten = 0;
        self.free_moves = FreeMoves::default();
        self.state = StateSummary::of(&self.game);
        self.ended = false;
        self.refresh();
        Ok(())
    }

    /// The env's free moves in a row are at its cap: the mask allows no more, but the swap
    /// carrying the last drag on.
    pub fn free_moves_spent(&self) -> bool {
        self.free_moves.spent(self.config.max_free_moves)
    }

    /// The free moves taken in a row.
    pub fn free_moves(&self) -> FreeMoves {
        self.free_moves
    }

    /// The mask and the action table of the current state.
    fn refresh(&mut self) {
        self.mask.compute(&self.game, Legality::Vanilla);
        if self.free_moves_spent() {
            self.mask.forbid_free_moves_but(self.free_moves.carry());
        }
        self.table.enumerate(&self.mask, &self.game, &mut self.rng);
    }

    /// Take the table's action `index`.
    pub fn step(&mut self, index: usize) -> Result<StepResult, StepError> {
        let action = self.table.get(index).ok_or(StepError::BadIndex(index))?;
        self.take(action)
    }

    /// Whether the env offers `action` here: the table holds it, or would if it held every
    /// legal action rather than a sample ([`crate::gym::offers`]). An ended episode offers none.
    pub fn offers(&self, action: &FactoredAction) -> bool {
        !self.ended && crate::gym::offers(&self.mask, &self.game, action)
    }

    /// Take `action`, one the env offers ([`GymEnv::offers`]) whether or not the table's sample
    /// holds it. The step is the one an index naming it takes: the sampler draws the next table
    /// as it would then, so a run of steps given either way replays exactly.
    pub fn step_factored(&mut self, action: FactoredAction) -> Result<StepResult, StepError> {
        if !self.offers(&action) {
            return Err(StepError::NotOffered(action));
        }
        self.take(action)
    }

    fn take(&mut self, action: FactoredAction) -> Result<StepResult, StepError> {
        let engine_action = to_engine(&action, &self.game).map_err(|e| StepError::Convert(e.0))?;
        // No rollback copy: an illegal action changes nothing, and a NotPorted refusal ends the
        // episode.
        let not_ported = match self.game.step_in_place(&engine_action) {
            Ok(()) => None,
            Err(CoreStepError::Illegal(m)) => return Err(StepError::Illegal(m)),
            Err(CoreStepError::Unported(m)) => Some(m),
        };
        Ok(self.finish(action, not_ported))
    }

    /// End the episode here as truncated without a step (the run is unchanged): the envs'
    /// answer to an action the engine refused.
    pub fn cut(&mut self, action: FactoredAction) -> StepResult {
        self.finish(action, Some(String::new()))
    }

    /// [`GymEnv::step`], with a table action the engine refuses cut into a truncated episode end
    /// ([`GymEnv::cut`]): the result, and the refusal. Only an index past the table fails.
    pub fn step_or_cut(&mut self, index: usize) -> Result<(StepResult, Option<String>), StepError> {
        let action = self.table.get(index).ok_or(StepError::BadIndex(index))?;
        self.take_or_cut(action)
    }

    /// [`GymEnv::step_factored`], with an action the engine refuses cut into a truncated episode
    /// end as [`GymEnv::step_or_cut`] does. Only an action the env doesn't offer fails.
    pub fn step_factored_or_cut(
        &mut self,
        action: FactoredAction,
    ) -> Result<(StepResult, Option<String>), StepError> {
        if !self.offers(&action) {
            return Err(StepError::NotOffered(action));
        }
        self.take_or_cut(action)
    }

    fn take_or_cut(
        &mut self,
        action: FactoredAction,
    ) -> Result<(StepResult, Option<String>), StepError> {
        match self.take(action) {
            Ok(r) => Ok((r, None)),
            Err(StepError::Convert(m) | StepError::Illegal(m)) => Ok((self.cut(action), Some(m))),
            Err(e) => Err(e),
        }
    }

    /// The player's view of the run ([`jackdaw_core::player_view`]), its cards named under a
    /// key drawn from the episode's sampler seed: without the seed, an id says nothing of the
    /// card it stands for.
    pub fn player_view(&self) -> serde_json::Value {
        self.game.player_view(self.view_key)
    }

    /// Record how each hand scores, from now on and in every later run, so the player's view
    /// shows the last hand's events. Recording never changes a run.
    pub fn set_record_scoring(&mut self, on: bool) {
        self.record_scoring = on;
        self.game.set_record_scoring(on);
    }

    pub fn record_scoring(&self) -> bool {
        self.record_scoring
    }

    fn finish(&mut self, action: FactoredAction, not_ported: Option<String>) -> StepResult {
        self.episode_length += 1;
        self.step_count += 1;
        self.free_moves.record(&action);
        let prev = self.state;
        let curr = StateSummary::of(&self.game);
        let won = run_won(&self.game);
        let terminated = curr.done || (self.config.end_on_win && won);
        let truncated = self.step_count >= self.config.max_steps || not_ported.is_some();
        self.episode_ante = self.game.gs.round_resets.ante;
        self.episode_won = won;
        let transition = Transition {
            prev,
            curr,
            terminated,
            truncated,
        };
        let reward = self.config.reward.of(&transition);
        // Python's max(best, ante): the first unless the second is larger.
        if curr.ante > self.max_ante {
            self.max_ante = curr.ante;
        }
        if transition.blind_beaten() {
            self.blinds_beaten += 1;
        }
        self.state = curr;
        if terminated || truncated {
            self.ended = true;
            self.table.clear();
        } else {
            self.refresh();
        }
        StepResult {
            action,
            transition,
            reward,
            not_ported,
        }
    }

    /// The Gym observation of the current state, flattened ([`GYM_OBS_LEN`] values): the global
    /// context, each entity array's first rows padded with zeros, the entity counts.
    pub fn write_obs(&mut self, out: &mut [f32]) {
        self.encoder.encode_padded(&self.game.gs, MAX_COUNTS, out);
    }

    /// The Gym action mask: the table's actions are the first legal slots.
    pub fn write_action_mask(&self, out: &mut [u8]) {
        let n = self.table.len().min(MAX_ACTIONS);
        out[..n].fill(1);
        out[n..].fill(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action::ActionType;

    fn spec(seed: &str) -> RunSpec {
        RunSpec {
            back: jackdaw_core::card::center_by_key("b_red").unwrap(),
            stake: 1,
            seed: seed.to_owned(),
            challenge: None,
        }
    }

    #[test]
    fn layout_is_614_values() {
        assert_eq!(GYM_OBS_LEN, 614);
    }

    #[test]
    fn steps_until_the_episode_ends() {
        let mut env = GymEnv::new(&spec("ENVTEST"), EnvConfig::default(), 7).unwrap();
        let mut obs = vec![0f32; GYM_OBS_LEN];
        let mut steps = 0;
        let mut rng = Xoshiro256::seed_from_u64(1);
        while !env.ended() && steps < 2000 {
            env.write_obs(&mut obs);
            assert_eq!(obs[GYM_OBS_LEN - 5..].len(), 5);
            let n = env.table().len() as u64;
            assert!(n > 0);
            let r = env.step(rng.below(n) as usize).unwrap();
            assert_eq!(r.terminated() || r.truncated(), env.ended());
            steps += 1;
        }
        assert!(env.ended());
        assert!(env.table().is_empty());
    }

    fn find(env: &GymEnv, t: ActionType) -> Option<usize> {
        env.table()
            .actions()
            .iter()
            .position(|a| a.action_type == t)
    }

    fn has_free(env: &GymEnv) -> bool {
        env.table()
            .actions()
            .iter()
            .any(|a| a.action_type.is_free())
    }

    #[test]
    fn caps_free_moves_in_a_row() {
        let config = EnvConfig {
            max_free_moves: Some(2),
            ..EnvConfig::default()
        };
        let mut env = GymEnv::new(&spec("CAPTEST"), config, 7).unwrap();
        env.step(find(&env, ActionType::SelectBlind).unwrap())
            .unwrap();
        for _ in 0..2 {
            assert!(has_free(&env));
            env.step(find(&env, ActionType::SortHandSuit).unwrap())
                .unwrap();
        }
        assert!(env.free_moves_spent());
        assert!(!has_free(&env), "two free moves in a row: none more");
        env.step(find(&env, ActionType::Discard).unwrap()).unwrap();
        assert!(!env.free_moves_spent());
        assert!(has_free(&env), "another action starts the count again");
    }

    #[test]
    fn a_factored_step_is_the_index_step() {
        let mut by_index = GymEnv::new(&spec("FACTORED"), EnvConfig::default(), 5).unwrap();
        let mut by_action = by_index.clone();
        let mut rng = Xoshiro256::seed_from_u64(2);
        while !by_index.ended() {
            let i = rng.below(by_index.table().len() as u64) as usize;
            let action = by_index.table().get(i).unwrap();
            assert!(by_action.offers(&action));
            assert_eq!(by_index.step(i), by_action.step_factored(action));
            assert_eq!(by_index.rng_state(), by_action.rng_state());
            assert_eq!(by_index.table().actions(), by_action.table().actions());
        }
        let blind = FactoredAction::new(ActionType::SelectBlind);
        assert!(!by_action.offers(&blind));
        assert_eq!(
            by_action.step_factored(blind),
            Err(StepError::NotOffered(blind))
        );
    }

    #[test]
    fn views_draw_nothing_from_the_sampler() {
        // A dealt hand: its cards carry ids, which the key sets.
        let dealt = |env: &mut GymEnv| {
            let blind = find(env, ActionType::SelectBlind).unwrap();
            env.step(blind).unwrap();
        };
        let mut env = GymEnv::new(&spec("VIEWKEY"), EnvConfig::default(), 5).unwrap();
        dealt(&mut env);
        let before = env.rng_state();
        let a = env.player_view();
        assert_eq!(env.rng_state(), before);
        env.reset(&spec("VIEWKEY"), Some(5)).unwrap();
        dealt(&mut env);
        assert_eq!(env.player_view(), a, "the same sampler seed, the same ids");
        env.reset(&spec("VIEWKEY"), Some(6)).unwrap();
        dealt(&mut env);
        assert_ne!(env.player_view()["hand"], a["hand"]);
    }

    fn swap(t: ActionType, i: usize) -> FactoredAction {
        FactoredAction::with_entity(t, i)
    }

    /// The free moves the table holds.
    fn free_in_table(env: &GymEnv) -> Vec<FactoredAction> {
        (env.table().actions().into_iter())
            .filter(|a| a.action_type.is_free())
            .collect()
    }

    fn hand_ids(env: &GymEnv) -> Vec<u32> {
        env.game()
            .gs
            .areas
            .ids(jackdaw_core::area::AreaId::Hand)
            .to_vec()
    }

    fn in_round(seed: &str, max_free_moves: Option<u32>) -> GymEnv {
        let config = EnvConfig {
            max_free_moves,
            ..EnvConfig::default()
        };
        let mut env = GymEnv::new(&spec(seed), config, 7).unwrap();
        env.step(find(&env, ActionType::SelectBlind).unwrap())
            .unwrap();
        env
    }

    #[test]
    fn a_drag_counts_once_however_far() {
        // #70: the game drags a card past one neighbour at a time (cardarea.lua:463), so taking
        // the last of eight cards to the front is seven swaps, and one move.
        let mut env = in_round("DRAGTEST", Some(2));
        let before = hand_ids(&env);
        let n = before.len();
        assert_eq!(n, 8);
        for i in (1..n).rev() {
            env.step_factored(swap(ActionType::SwapHandLeft, i))
                .unwrap();
            assert_eq!(env.free_moves().count(), 1);
        }
        let after = hand_ids(&env);
        assert_eq!(after[0], before[n - 1]);
        assert_eq!(after[1..], before[..n - 1]);

        // The second drag reaches the cap: the table holds no free move but its next swap, which
        // carries the card as far as it goes.
        env.step_factored(swap(ActionType::SwapHandRight, 2))
            .unwrap();
        assert_eq!(env.free_moves().count(), 2);
        assert!(env.free_moves_spent());
        for i in 3..n - 1 {
            assert_eq!(free_in_table(&env), [swap(ActionType::SwapHandRight, i)]);
            let other = swap(ActionType::SwapHandLeft, i);
            assert_eq!(env.step_factored(other), Err(StepError::NotOffered(other)));
            env.step_factored(swap(ActionType::SwapHandRight, i))
                .unwrap();
        }
        assert!(free_in_table(&env).is_empty(), "the card is at the end");
        env.step(find(&env, ActionType::Discard).unwrap()).unwrap();
        assert_eq!(env.free_moves(), FreeMoves::default());
        assert!(has_free(&env));
    }

    #[test]
    fn a_swap_back_is_another_drag() {
        // Only a swap that carries the card on, the same way, continues a drag: back and forth
        // counts every time, so the cap still ends a stall.
        let mut env = in_round("DRAGBACK", Some(3));
        env.step_factored(swap(ActionType::SwapHandLeft, 4))
            .unwrap();
        env.step_factored(swap(ActionType::SwapHandRight, 3))
            .unwrap();
        assert_eq!(env.free_moves().count(), 2);
        // The card the last swap moved left is at 2 now; carrying it on is SwapHandLeft(2).
        env.step_factored(swap(ActionType::SwapHandLeft, 4))
            .unwrap();
        assert_eq!(env.free_moves().count(), 3);
        assert_eq!(free_in_table(&env), [swap(ActionType::SwapHandLeft, 3)]);
        // A sort counts once and carries nothing on.
        let mut sorted = in_round("DRAGBACK", Some(1));
        sorted
            .step(find(&sorted, ActionType::SortHandRank).unwrap())
            .unwrap();
        assert!(free_in_table(&sorted).is_empty());
    }

    #[test]
    fn a_joker_drag_counts_once() {
        let mut env = in_round("DRAGJOKER", Some(1));
        for key in [
            "j_joker",
            "j_greedy_joker",
            "j_lusty_joker",
            "j_wrathful_joker",
        ] {
            (env.game)
                .apply_setup_op(&serde_json::json!({"op": "add", "key": key}))
                .unwrap();
        }
        env.refresh();
        env.step_factored(swap(ActionType::SwapJokersLeft, 3))
            .unwrap();
        assert!(env.free_moves_spent());
        assert_eq!(free_in_table(&env), [swap(ActionType::SwapJokersLeft, 2)]);
        env.step_factored(swap(ActionType::SwapJokersLeft, 2))
            .unwrap();
        env.step_factored(swap(ActionType::SwapJokersLeft, 1))
            .unwrap();
        let jokers: Vec<&str> = (env.game.gs.areas.jokers().iter())
            .map(|j| j.center().key)
            .collect();
        assert_eq!(
            jokers,
            [
                "j_wrathful_joker",
                "j_joker",
                "j_greedy_joker",
                "j_lusty_joker"
            ]
        );
        assert!(free_in_table(&env).is_empty());
    }

    /// Put the hand in `order` (hand indices), one drag per card that isn't in its place yet:
    /// each card in turn is carried left to where it goes.
    fn drag_into(env: &mut GymEnv, order: &[usize]) {
        let want: Vec<u32> = order.iter().map(|&i| hand_ids(env)[i]).collect();
        for (k, id) in want.iter().enumerate() {
            let from = hand_ids(env).iter().position(|x| x == id).unwrap();
            for i in (k + 1..=from).rev() {
                let action = swap(ActionType::SwapHandLeft, i);
                assert!(
                    env.offers(&action),
                    "{action:?} after {:?}",
                    env.free_moves()
                );
                env.step_factored(action).unwrap();
            }
        }
        assert_eq!(hand_ids(env), want);
    }

    #[test]
    fn any_order_of_a_hand_is_within_the_default_cap() {
        // #70: reversing eight cards is 28 swaps, more than the old cap of ten in a row allowed.
        // As drags it is seven moves.
        let mut env = in_round("DRAGORDER", Some(DEFAULT_MAX_FREE_MOVES));
        drag_into(&mut env, &[7, 6, 5, 4, 3, 2, 1, 0]);
        assert_eq!(env.free_moves().count(), 7);
        // After a sort, which counts once too.
        let mut rng = Xoshiro256::seed_from_u64(3);
        for k in 0..20 {
            let mut env = in_round(&format!("DRAGORDER{k}"), Some(DEFAULT_MAX_FREE_MOVES));
            env.step(find(&env, ActionType::SortHandSuit).unwrap())
                .unwrap();
            let mut order: Vec<usize> = (0..8).collect();
            for i in (1..8).rev() {
                order.swap(i, rng.below(i as u64 + 1) as usize);
            }
            drag_into(&mut env, &order);
            assert!(env.free_moves().count() <= 8);
        }
    }

    #[test]
    fn no_cap() {
        let config = EnvConfig {
            max_free_moves: None,
            ..EnvConfig::default()
        };
        let mut env = GymEnv::new(&spec("CAPTEST"), config, 7).unwrap();
        env.step(find(&env, ActionType::SelectBlind).unwrap())
            .unwrap();
        for _ in 0..50 {
            env.step(find(&env, ActionType::SortHandRank).unwrap())
                .unwrap();
        }
        assert!(has_free(&env));
    }
}

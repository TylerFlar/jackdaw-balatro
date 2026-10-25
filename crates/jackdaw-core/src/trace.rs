//! How a played hand scored, step by step: the events `G.FUNCS.evaluate_play` shows as it goes
//! (each card's, edition's and joker's chips and mult, retriggers, the boss's halving, a Glass
//! Card shattering), with the hand's chips and mult after each.
//!
//! Recording is off unless asked for ([`Game::set_record_scoring`]). The trace lives beside
//! the run, not in it: snapshots, state hashes, observations and masks are the same with it
//! on, and a recorded step takes the same course as an unrecorded one.

use serde_json::{Map, Value, json};

use crate::game::Game;
use crate::hand_eval::HandType;
use crate::state::LastScore;

/// What a [`ScoreEvent`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScoreKind {
    /// A joker levels the hand up before it scores (Space Joker): the new level, and the
    /// hand's new base chips and mult.
    LevelUp,
    /// The hand's chips and mult as scoring starts: its base at the level it scores at.
    Hand,
    /// The boss changes them (The Flint halves both).
    Blind,
    /// A scoring card's own bonus: its chips, an enhancement's mult or x mult, a seal's money.
    Card,
    /// A card held in hand gives its bonus (a Steel Card's x mult).
    Held,
    /// An edition's bonus: a scoring card's, a joker's or a consumable's.
    Edition,
    /// A joker's effect, or a consumable's (Observatory's planets).
    Joker,
    /// A card scores, or a held card counts, again: a Red Seal or a joker repeats it.
    Retrigger,
    /// A scoring card the boss debuffs adds nothing.
    Debuffed,
    /// The boss doesn't allow the hand: it scores nothing.
    NotAllowed,
    /// Plasma Deck balances the chips and mult.
    Plasma,
    /// A scoring card is destroyed after the hand (a Glass Card shatters; Sixth Sense).
    Destroyed,
}

impl ScoreKind {
    pub fn name(self) -> &'static str {
        match self {
            ScoreKind::LevelUp => "level_up",
            ScoreKind::Hand => "hand",
            ScoreKind::Blind => "blind",
            ScoreKind::Card => "card",
            ScoreKind::Held => "held",
            ScoreKind::Edition => "edition",
            ScoreKind::Joker => "joker",
            ScoreKind::Retrigger => "retrigger",
            ScoreKind::Debuffed => "debuffed",
            ScoreKind::NotAllowed => "not_allowed",
            ScoreKind::Plasma => "plasma",
            ScoreKind::Destroyed => "destroyed",
        }
    }

    /// Whether the event sets the chips and mult outright, rather than adding to them.
    pub fn sets_score(self) -> bool {
        matches!(
            self,
            ScoreKind::LevelUp
                | ScoreKind::Hand
                | ScoreKind::Blind
                | ScoreKind::NotAllowed
                | ScoreKind::Plasma
        )
    }
}

/// One step of a hand's scoring.
#[derive(Clone, Debug, PartialEq)]
pub struct ScoreEvent {
    pub kind: ScoreKind,
    /// The card the event belongs to, where the game shows it (sort id): the scoring or held
    /// card for its own bonus, the joker for a joker's effect, the card with the Red Seal or
    /// the joker that repeats a card.
    pub card: Option<u32>,
    /// The card it happens to, when that's another one: the scoring or held card a joker acts
    /// on or a retrigger repeats, the joker Baseball Card boosts.
    pub target: Option<u32>,
    /// Chips added.
    pub chips: Option<f64>,
    /// Mult added.
    pub mult: Option<f64>,
    /// Mult multiplied.
    pub x_mult: Option<f64>,
    /// Money paid, after the hand.
    pub dollars: Option<f64>,
    /// Chips and mult traded places (`extra.swap`).
    pub swap: bool,
    /// The hand's level ([`ScoreKind::LevelUp`], [`ScoreKind::Hand`]).
    pub level: Option<f64>,
    /// The destroyed card was a Glass Card, and shattered.
    pub shattered: bool,
    /// The hand's chips after the event.
    pub hand_chips: f64,
    /// The hand's mult after the event.
    pub hand_mult: f64,
}

impl ScoreEvent {
    pub fn new(kind: ScoreKind, card: Option<u32>, hand_chips: f64, hand_mult: f64) -> Self {
        ScoreEvent {
            kind,
            card,
            target: None,
            chips: None,
            mult: None,
            x_mult: None,
            dollars: None,
            swap: false,
            level: None,
            shattered: false,
            hand_chips,
            hand_mult,
        }
    }

    pub fn target(mut self, id: u32) -> Self {
        self.target = Some(id);
        self
    }

    pub fn chips(mut self, x: f64) -> Self {
        self.chips = Some(x);
        self
    }

    pub fn mult(mut self, x: f64) -> Self {
        self.mult = Some(x);
        self
    }

    pub fn x_mult(mut self, x: f64) -> Self {
        self.x_mult = Some(x);
        self
    }

    pub fn dollars(mut self, x: f64) -> Self {
        self.dollars = Some(x);
        self
    }

    pub fn level(mut self, x: f64) -> Self {
        self.level = Some(x);
        self
    }

    pub fn swapped(mut self) -> Self {
        self.swap = true;
        self
    }

    pub fn shattered(mut self, glass: bool) -> Self {
        self.shattered = glass;
        self
    }

    /// The event as JSON: absent fields left out, non-finite numbers as `{"$f": ...}`.
    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("kind".into(), self.kind.name().into());
        let ids = [("card", self.card), ("target", self.target)];
        for (k, v) in ids {
            if let Some(id) = v {
                m.insert(k.into(), id.into());
            }
        }
        let nums = [
            ("chips", self.chips),
            ("mult", self.mult),
            ("x_mult", self.x_mult),
            ("dollars", self.dollars),
            ("level", self.level),
        ];
        for (k, v) in nums {
            if let Some(x) = v {
                m.insert(k.into(), number(x));
            }
        }
        if self.swap {
            m.insert("swap".into(), true.into());
        }
        if self.shattered {
            m.insert("shattered".into(), true.into());
        }
        m.insert("hand_chips".into(), number(self.hand_chips));
        m.insert("hand_mult".into(), number(self.hand_mult));
        Value::Object(m)
    }
}

/// A number as JSON: `{"$f": "inf" | "-inf" | "nan"}` when it isn't finite, as in snapshots.
fn number(x: f64) -> Value {
    if x.is_finite() {
        json!(x)
    } else if x.is_nan() {
        json!({"$f": "nan"})
    } else {
        json!({"$f": if x > 0.0 { "inf" } else { "-inf" }})
    }
}

/// A play's scoring, as recorded.
#[derive(Clone, Debug, PartialEq)]
pub struct PlayTrace {
    pub hand: HandType,
    /// The hand's level when it was played, before any level-up.
    pub level: f64,
    /// The played cards (sort ids), in play order.
    pub played: Vec<u32>,
    /// The cards that score, in play order: the hand's cards, every Stone Card, or all of them
    /// with Splash.
    pub scoring: Vec<u32>,
    pub events: Vec<ScoreEvent>,
}

impl PlayTrace {
    /// The play as JSON, with what it scored (the run's last score).
    pub fn to_json(&self, score: Option<&LastScore>) -> Value {
        let mut v = json!({
            "hand": self.hand.name(),
            "level": number(self.level),
            "played": self.played,
            "scoring": self.scoring,
            "events": self.events.iter().map(ScoreEvent::to_json).collect::<Vec<_>>(),
        });
        if let Some(s) = score {
            v["chips"] = number(s.chips);
            v["mult"] = number(s.mult);
            v["total"] = number(s.total);
            v["dollars"] = number(s.dollars);
            v["debuffed"] = s.debuffed.into();
        }
        v
    }
}

/// The recorder: whether it's on, and the play the last step scored.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScoreTrace {
    pub enabled: bool,
    pub play: Option<PlayTrace>,
}

impl Game {
    /// Record how each played hand scores ([`Game::last_play`]). Off by default.
    pub fn set_record_scoring(&mut self, on: bool) {
        self.trace.enabled = on;
        if !on {
            self.trace.play = None;
        }
    }

    pub fn record_scoring(&self) -> bool {
        self.trace.enabled
    }

    /// How the hand the last step played scored, when recording was on; `None` after a step
    /// that played no hand.
    pub fn last_play(&self) -> Option<&PlayTrace> {
        self.trace.play.as_ref()
    }

    /// Start recording a play, when recording is on.
    pub(crate) fn trace_play(&mut self, hand: HandType, played: &[u32], scoring: &[u32]) {
        if self.trace.enabled {
            self.trace.play = Some(PlayTrace {
                hand,
                level: self.gs.hand(hand).level,
                played: played.to_vec(),
                scoring: scoring.to_vec(),
                events: Vec::new(),
            });
        }
    }

    /// Record an event of the play, when recording is on: `make` runs only then.
    #[inline]
    pub(crate) fn record(&mut self, make: impl FnOnce() -> ScoreEvent) {
        if self.trace.enabled
            && let Some(p) = self.trace.play.as_mut()
        {
            p.events.push(make());
        }
    }
}

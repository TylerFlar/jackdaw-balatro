//! BalatroBot's error names and JSON-RPC codes.
//!
//! Ported from BalatroBot v1.5.2 (coder/balatrobot `9052d76f14723293f6c6b2cecaa791a5c4ae68f3`),
//! `src/lua/utils/errors.lua`.

/// `BB_ERROR_NAMES`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorName {
    /// Server-side failure (runtime or execution error). `-32000`.
    InternalError,
    /// The client sent invalid data (protocol or parameter error). `-32001`.
    BadRequest,
    /// The action is not allowed in the current game state. `-32002`.
    InvalidState,
    /// The game rules prevent the action. `-32003`.
    NotAllowed,
}

impl ErrorName {
    pub const ALL: [ErrorName; 4] = [
        ErrorName::InternalError,
        ErrorName::BadRequest,
        ErrorName::InvalidState,
        ErrorName::NotAllowed,
    ];

    /// The name in `error.data.name`.
    pub fn name(self) -> &'static str {
        match self {
            ErrorName::InternalError => "INTERNAL_ERROR",
            ErrorName::BadRequest => "BAD_REQUEST",
            ErrorName::InvalidState => "INVALID_STATE",
            ErrorName::NotAllowed => "NOT_ALLOWED",
        }
    }

    /// `BB_ERROR_CODES[name]`.
    pub fn code(self) -> i64 {
        match self {
            ErrorName::InternalError => -32000,
            ErrorName::BadRequest => -32001,
            ErrorName::InvalidState => -32002,
            ErrorName::NotAllowed => -32003,
        }
    }
}

/// An endpoint's error response (`{message = ..., name = ...}`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BotError {
    pub name: ErrorName,
    pub message: String,
}

impl BotError {
    pub fn new(name: ErrorName, message: impl Into<String>) -> BotError {
        BotError {
            name,
            message: message.into(),
        }
    }

    pub fn bad_request(message: impl Into<String>) -> BotError {
        BotError::new(ErrorName::BadRequest, message)
    }

    pub fn invalid_state(message: impl Into<String>) -> BotError {
        BotError::new(ErrorName::InvalidState, message)
    }

    pub fn not_allowed(message: impl Into<String>) -> BotError {
        BotError::new(ErrorName::NotAllowed, message)
    }

    pub fn internal(message: impl Into<String>) -> BotError {
        BotError::new(ErrorName::InternalError, message)
    }
}

impl std::fmt::Display for BotError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({}): {}",
            self.name.name(),
            self.name.code(),
            self.message
        )
    }
}

impl std::error::Error for BotError {}

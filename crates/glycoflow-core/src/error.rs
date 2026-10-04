use std::fmt;

/// Errors of the GlycoFlow engine.
#[derive(Debug)]
pub enum Error {
    /// GLYCAM sequence syntax.
    Parse(String),
    /// Residue / modification / template missing from the residue library.
    Library(String),
    /// Covalent-graph problems (disconnected graph, root on a distal side, ...).
    Topology(String),
    /// Invalid arguments (shapes, options).
    Invalid(String),
    Json(serde_json::Error),
    Candle(candle_core::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Parse(s) | Error::Library(s) | Error::Topology(s) | Error::Invalid(s) => {
                f.write_str(s)
            }
            Error::Json(e) => write!(f, "json: {e}"),
            Error::Candle(e) => write!(f, "candle: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Json(e)
    }
}

impl From<candle_core::Error> for Error {
    fn from(e: candle_core::Error) -> Self {
        Error::Candle(e)
    }
}

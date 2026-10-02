//! Error type of the fitting workflow.

#[derive(Debug, thiserror::Error)]
pub enum FitError {
    #[error("{0}")]
    Invalid(String),
    #[error("GlycoFlow engine: {0}")]
    Engine(#[from] glycoflow_core::Error),
    #[error(transparent)]
    Density(#[from] reglyco_density::DensityError),
    #[error(transparent)]
    Structure(#[from] glysys::BuildError),
    #[error("candle: {0}")]
    Candle(#[from] candle_core::Error),
    #[error("{path}: {source}")]
    Io {
        path: std::path::PathBuf,
        source: std::io::Error,
    },
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, FitError>;

pub(crate) fn invalid(message: impl Into<String>) -> FitError {
    FitError::Invalid(message.into())
}

pub(crate) fn read_file(path: &std::path::Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|source| FitError::Io {
        path: path.to_path_buf(),
        source,
    })
}

pub(crate) fn write_file(path: &std::path::Path, contents: impl AsRef<[u8]>) -> Result<()> {
    std::fs::write(path, contents).map_err(|source| FitError::Io {
        path: path.to_path_buf(),
        source,
    })
}

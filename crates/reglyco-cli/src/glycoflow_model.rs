//! The GlycoFlow model for `refine --objective density`: licence acceptance, local model
//! directories and the download from the gated Hugging Face repository.
//!
//! The model (network weights, metadata and residue library) is not part of ReGlyco. It is licensed
//! under the GlycoFlow Source-Available Non-Commercial License and distributed through the gated
//! Hugging Face repository `Ojas-Singh/glycoflow`, where the GlycoFlow authors approve access
//! requests. ReGlyco uses it only after the user has accepted that licence
//! (`--accept-glycoflow-license`, recorded once in the model cache, or
//! `$GLYCOFLOW_ACCEPT_LICENSE=1`), and downloads it with the user's Hugging Face token when no
//! model directory is given.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, bail};
use clap::Args;
use reglyco_glycoflow::GlycoflowModel;
use sha2::{Digest, Sha256};

/// Hugging Face repository of the model.
const REPOSITORY: &str = "Ojas-Singh/glycoflow";
/// Repository revision that is downloaded: this engine is tested against this model.
const REVISION: &str = "398b97e42caf6eeb45387c195fae07423f01c514";
/// Model files and their SHA-256 at `REVISION`.
const FILES: [(&str, &str); 3] = [
    (
        "glycoflow.safetensors",
        "4a10254347f645afdaaa032dd02243633c4acee10294cbb2ad4c72d6111ebf48",
    ),
    (
        "glycoflow.json",
        "ff6996fed93b316abbd63fbbefb9502c6a99c528725785952ed680c3febab8ec",
    ),
    (
        "residue_library.json",
        "800dbe6d0ff62f93d5e39d748c1fb036932c87f15b5e8897c6e958fcf5a356c9",
    ),
];
const LICENSE_URL: &str = "https://huggingface.co/Ojas-Singh/glycoflow/blob/main/LICENSE";
const ACCEPT_ENV: &str = "GLYCOFLOW_ACCEPT_LICENSE";
const CACHE_ENV: &str = "GLYCOFLOW_CACHE";
const ACCEPTANCE_RECORD: &str = "LICENSE-ACCEPTED";

/// `reglyco glycoflow-model`: accept the GlycoFlow licence and locate or download the model.
#[derive(Debug, Clone, Args)]
pub(crate) struct GlycoflowModelArgs {
    /// GlycoFlow model directory to check instead of the cache; default: $GLYCOFLOW_MODEL.
    #[arg(long = "glycoflow-model")]
    pub(crate) model: Option<PathBuf>,
    /// Accept the GlycoFlow Source-Available Non-Commercial License of the model (recorded once).
    #[arg(long = "accept-glycoflow-license")]
    pub(crate) accept_license: bool,
}

pub(crate) fn run(arguments: GlycoflowModelArgs) -> anyhow::Result<()> {
    let source =
        ModelSource::from_environment(arguments.model.as_deref(), arguments.accept_license)?;
    let dir = source.resolve(false)?;
    println!("{}", dir.display());
    Ok(())
}

/// Where the model comes from, and whether its licence has been accepted.
pub(crate) struct ModelSource {
    /// `--glycoflow-model` or `$GLYCOFLOW_MODEL`.
    pub(crate) local: Option<PathBuf>,
    /// Model cache: the acceptance record and downloaded revisions.
    pub(crate) cache: PathBuf,
    pub(crate) token: Option<String>,
    /// `--accept-glycoflow-license` in this run (recorded in the cache).
    pub(crate) accept_flag: bool,
    /// `$GLYCOFLOW_ACCEPT_LICENSE` (not recorded).
    pub(crate) accept_env: bool,
}

impl ModelSource {
    pub(crate) fn from_environment(
        explicit: Option<&Path>,
        accept_flag: bool,
    ) -> anyhow::Result<Self> {
        let cache = match std::env::var_os(CACHE_ENV) {
            Some(dir) => PathBuf::from(dir),
            None => cache_home()
                .map(|home| home.join("reglyco").join("glycoflow"))
                .with_context(|| {
                    format!("no cache directory for the GlycoFlow model: set ${CACHE_ENV} or $HOME")
                })?,
        };
        let accept_env = std::env::var(ACCEPT_ENV).is_ok_and(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes"
            )
        });
        Ok(Self {
            local: GlycoflowModel::resolve_dir(explicit).ok(),
            cache,
            token: hugging_face_token(),
            accept_flag,
            accept_env,
        })
    }

    pub(crate) fn license_accepted(&self) -> bool {
        self.accept_flag || self.accept_env || self.cache.join(ACCEPTANCE_RECORD).is_file()
    }

    /// Directory of the downloaded model revision.
    pub(crate) fn cached_dir(&self) -> PathBuf {
        self.cache.join(&REVISION[..12])
    }

    /// The model directory to load: the licence must be accepted; then `--glycoflow-model` /
    /// `$GLYCOFLOW_MODEL`, else the cached download, else a download with the Hugging Face token.
    pub(crate) fn resolve(&self, quiet: bool) -> anyhow::Result<PathBuf> {
        if !self.license_accepted() {
            bail!(license_message(&self.cache));
        }
        let record = self.cache.join(ACCEPTANCE_RECORD);
        if self.accept_flag && !record.is_file() {
            std::fs::create_dir_all(&self.cache)
                .with_context(|| format!("creating {}", self.cache.display()))?;
            std::fs::write(
                &record,
                format!(
                    "GlycoFlow Source-Available Non-Commercial License accepted with \
                     --accept-glycoflow-license (reglyco {}).\nTerms: {LICENSE_URL}\n",
                    env!("CARGO_PKG_VERSION")
                ),
            )
            .with_context(|| format!("writing {}", record.display()))?;
            if !quiet {
                eprintln!(
                    "GlycoFlow licence accepted; recorded in {}",
                    record.display()
                );
            }
        }
        if let Some(dir) = &self.local {
            if !dir.is_dir() {
                bail!(
                    "GlycoFlow model directory {} does not exist (--glycoflow-model / $GLYCOFLOW_MODEL)",
                    dir.display()
                );
            }
            return Ok(dir.clone());
        }
        let dir = self.cached_dir();
        if FILES.iter().all(|(name, _)| dir.join(name).is_file()) {
            return Ok(dir);
        }
        let Some(token) = &self.token else {
            bail!(
                "no GlycoFlow model: pass --glycoflow-model <dir> (or set $GLYCOFLOW_MODEL) to a \
                 directory with glycoflow.safetensors, glycoflow.json and residue_library.json, or \
                 download it from the gated Hugging Face repository: request access at \
                 https://huggingface.co/{REPOSITORY} (the GlycoFlow authors approve requests), then \
                 log in with `hf auth login` or set HF_TOKEN, and rerun"
            );
        };
        download(token, &dir, quiet)?;
        Ok(dir)
    }
}

fn license_message(cache: &Path) -> String {
    format!(
        "GlycoFlow density fitting needs the GlycoFlow model, which is not part of ReGlyco. The \
         model (network weights, metadata and residue library) is licensed under the GlycoFlow \
         Source-Available Non-Commercial License: non-commercial academic research only; \
         commercial use needs a separate written licence from the GlycoFlow authors. Terms: \
         {LICENSE_URL}\n\nTo accept the licence, rerun with --accept-glycoflow-license (recorded \
         once in {}) or set {ACCEPT_ENV}=1.",
        cache.display()
    )
}

/// `$XDG_CACHE_HOME`, `$HOME/.cache` or `%LOCALAPPDATA%`.
fn cache_home() -> Option<PathBuf> {
    let nonempty = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty());
    nonempty("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| nonempty("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .or_else(|| nonempty("LOCALAPPDATA").map(PathBuf::from))
}

/// The Hugging Face token as the Hugging Face tools find it: `$HF_TOKEN`,
/// `$HUGGING_FACE_HUB_TOKEN`, `$HF_TOKEN_PATH`, `$HF_HOME/token`, or the token file written by
/// `hf auth login`.
fn hugging_face_token() -> Option<String> {
    let clean = |token: String| {
        let token = token.trim().to_string();
        (!token.is_empty()).then_some(token)
    };
    for name in ["HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"] {
        if let Some(token) = std::env::var(name).ok().and_then(clean) {
            return Some(token);
        }
    }
    let mut files = Vec::new();
    if let Some(path) = std::env::var_os("HF_TOKEN_PATH") {
        files.push(PathBuf::from(path));
    }
    if let Some(home) = std::env::var_os("HF_HOME") {
        files.push(PathBuf::from(home).join("token"));
    }
    if let Some(home) = cache_home() {
        files.push(home.join("huggingface").join("token"));
    }
    files
        .iter()
        .find_map(|path| std::fs::read_to_string(path).ok().and_then(clean))
}

/// Downloads the model files at `REVISION` into `dir`, checking their SHA-256.
fn download(token: &str, dir: &Path, quiet: bool) -> anyhow::Result<()> {
    let parent = dir
        .parent()
        .context("model cache directory has no parent")?;
    std::fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
    let partial = parent.join(format!(".partial-{}", std::process::id()));
    std::fs::create_dir_all(&partial)?;
    let result = download_files(token, &partial, quiet);
    if let Err(error) = result {
        let _ = std::fs::remove_dir_all(&partial);
        return Err(error);
    }
    if let Err(error) = std::fs::rename(&partial, dir) {
        // another process finished the same download first
        let _ = std::fs::remove_dir_all(&partial);
        if !FILES.iter().all(|(name, _)| dir.join(name).is_file()) {
            return Err(error)
                .with_context(|| format!("moving the GlycoFlow model to {}", dir.display()));
        }
    }
    if !quiet {
        eprintln!("GlycoFlow model saved to {}", dir.display());
    }
    Ok(())
}

fn download_files(token: &str, into: &Path, quiet: bool) -> anyhow::Result<()> {
    let client = reqwest::blocking::Client::builder()
        .user_agent(concat!("reglyco/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(Duration::from_secs(30))
        .timeout(Duration::from_secs(1800))
        .build()?;
    for (name, expected) in FILES {
        let url = format!("https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/{name}");
        if !quiet {
            eprintln!("GlycoFlow: downloading {name} from huggingface.co/{REPOSITORY}...");
        }
        let mut response = client
            .get(&url)
            .bearer_auth(token)
            .send()
            .with_context(|| format!("downloading {url}"))?;
        match response.status().as_u16() {
            200 => {}
            401 => bail!(
                "Hugging Face rejected the token (HTTP 401): log in again with `hf auth login` or set \
                 HF_TOKEN to a valid token"
            ),
            403 => bail!(
                "this Hugging Face account has no access to the GlycoFlow model yet (HTTP 403): request \
                 access at https://huggingface.co/{REPOSITORY}; the GlycoFlow authors approve requests"
            ),
            404 => bail!(
                "the GlycoFlow model is not visible to this Hugging Face account (HTTP 404): request \
                 access at https://huggingface.co/{REPOSITORY}"
            ),
            status => bail!("downloading {url}: HTTP {status}"),
        }
        let path = into.join(name);
        let mut file =
            std::fs::File::create(&path).with_context(|| format!("creating {}", path.display()))?;
        let mut hasher = Sha256::new();
        let mut buffer = vec![0u8; 1 << 16];
        loop {
            let n = response
                .read(&mut buffer)
                .with_context(|| format!("downloading {url}"))?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
            file.write_all(&buffer[..n])?;
        }
        let digest = format!("{:x}", hasher.finalize());
        if digest != expected {
            bail!(
                "{name} from {REPOSITORY} has SHA-256 {digest}, expected {expected}: the download is \
                 incomplete or the file changed"
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_cache(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("reglyco-glycoflow-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn source(cache: &Path, accept_flag: bool) -> ModelSource {
        ModelSource {
            local: None,
            cache: cache.to_path_buf(),
            token: None,
            accept_flag,
            accept_env: false,
        }
    }

    #[test]
    fn model_needs_the_licence() {
        let cache = temp_cache("licence");
        let error = source(&cache, false).resolve(true).unwrap_err().to_string();
        assert!(error.contains("--accept-glycoflow-license"), "{error}");
        assert!(error.contains("Non-Commercial"), "{error}");
        assert!(!cache.exists(), "nothing is written before acceptance");
    }

    #[test]
    fn acceptance_is_recorded_once() {
        let cache = temp_cache("record");
        let error = source(&cache, true).resolve(true).unwrap_err().to_string();
        // accepted, but no model directory and no token: explains both ways to get the model
        assert!(error.contains("--glycoflow-model"), "{error}");
        assert!(error.contains("HF_TOKEN"), "{error}");
        assert!(cache.join(ACCEPTANCE_RECORD).is_file());
        assert!(source(&cache, false).license_accepted());
        std::fs::remove_dir_all(&cache).unwrap();
    }

    #[test]
    fn local_and_cached_models_need_no_download() {
        let cache = temp_cache("local");
        let mut s = source(&cache, true);
        s.local = Some(cache.join("missing"));
        assert!(
            s.resolve(true)
                .unwrap_err()
                .to_string()
                .contains("does not exist")
        );
        let dir = s.cached_dir();
        std::fs::create_dir_all(&dir).unwrap();
        s.local = Some(dir.clone());
        assert_eq!(s.resolve(true).unwrap(), dir);
        for (name, _) in FILES {
            std::fs::write(dir.join(name), b"").unwrap();
        }
        s.local = None;
        assert_eq!(s.resolve(true).unwrap(), dir);
        std::fs::remove_dir_all(&cache).unwrap();
    }
}

//! The frozen GlycoFlow model: network weights, vocabulary and residue library.

use std::path::{Path, PathBuf};

use glycoflow_core::model::Precision;
use glycoflow_core::{ModelMeta, ResidueLibrary, TorsionFlowNet};

use crate::error::{Result, invalid, read_file};

/// Where the network runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ComputeDevice {
    Cpu,
    Cuda,
}

/// A loaded GlycoFlow model directory: `glycoflow.safetensors` and `glycoflow.json` (written by
/// GlycoFlow's `scripts/export_rust_fixtures.py`) and `residue_library.json`
/// (`glycoflow/resources/residue_library.json`).
pub struct GlycoflowModel {
    pub net: TorsionFlowNet,
    pub meta: ModelMeta,
    pub library: ResidueLibrary,
    pub dir: PathBuf,
    pub device: ComputeDevice,
}

/// Environment variable naming the default model directory.
pub const MODEL_ENV: &str = "GLYCOFLOW_MODEL";

impl GlycoflowModel {
    /// The model directory from an explicit argument or `$GLYCOFLOW_MODEL`.
    pub fn resolve_dir(explicit: Option<&Path>) -> Result<PathBuf> {
        if let Some(dir) = explicit {
            return Ok(dir.to_path_buf());
        }
        std::env::var_os(MODEL_ENV).map(PathBuf::from).ok_or_else(|| {
            invalid(format!(
                "no GlycoFlow model: pass --glycoflow-model <dir> or set {MODEL_ENV} (a directory with \
                 glycoflow.safetensors, glycoflow.json and residue_library.json)"
            ))
        })
    }

    pub fn load(dir: &Path, device: ComputeDevice, precision: Precision) -> Result<Self> {
        let need = |name: &str| -> Result<PathBuf> {
            let path = dir.join(name);
            if path.is_file() {
                Ok(path)
            } else {
                Err(invalid(format!(
                    "GlycoFlow model directory {} has no {name} (expected glycoflow.safetensors, glycoflow.json, \
                     residue_library.json)",
                    dir.display()
                )))
            }
        };
        let meta_bytes = read_file(&need("glycoflow.json")?)?;
        let meta = ModelMeta::from_json_str(
            std::str::from_utf8(&meta_bytes).map_err(|e| invalid(e.to_string()))?,
        )?;
        let library = ResidueLibrary::from_json_slice(&read_file(&need("residue_library.json")?)?)?;
        let candle_device = match device {
            ComputeDevice::Cpu => candle_core::Device::Cpu,
            ComputeDevice::Cuda => candle_core::Device::new_cuda(0).map_err(|e| {
                invalid(format!(
                    "CUDA device unavailable ({e}); build reglyco with the glycoflow `cuda` feature"
                ))
            })?,
        };
        let weights = read_file(&need("glycoflow.safetensors")?)?;
        let net = TorsionFlowNet::from_safetensors(
            &weights,
            meta.config.clone(),
            &candle_device,
            precision,
        )?;
        Ok(Self {
            net,
            meta,
            library,
            dir: dir.to_path_buf(),
            device,
        })
    }
}

//! Fixture loading for the parity tests against the Python GlycoFlow reference. The fixtures, the
//! residue library and the weights belong to the licensed GlycoFlow model and are not in this
//! repository: point `GLYCOFLOW_RUST_DIR` at a GlycoFlow checkout's `rust/` directory (fixtures/
//! and weights/ from `scripts/export_rust_fixtures.py`) and run with `--ignored`.
#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::Device;
use glycoflow_core::model::Precision;
use glycoflow_core::{ModelMeta, ResidueLibrary, TorsionFlowNet};
use safetensors::{Dtype, SafeTensors};

pub const CASES: [&str; 8] = [
    "lacnac",
    "core_man3",
    "sialyl_biantennary",
    "man8",
    "araf4",
    "fruf3",
    "sulfated",
    "sulfated_3s",
];

pub const IGNORE: &str =
    "needs GLYCOFLOW_RUST_DIR (GlycoFlow rust/: fixtures, weights, residue library)";

pub fn rust_dir() -> PathBuf {
    PathBuf::from(std::env::var("GLYCOFLOW_RUST_DIR").expect(IGNORE))
}

pub struct Fixture {
    pub name: String,
    pub meta: HashMap<String, String>,
    arrays: HashMap<String, (Vec<usize>, Dtype, Vec<u8>)>,
}

impl Fixture {
    pub fn load(name: &str) -> Self {
        let path = rust_dir()
            .join("fixtures")
            .join(format!("{name}.safetensors"));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| {
            panic!(
                "{}: {e} (run scripts/export_rust_fixtures.py)",
                path.display()
            )
        });
        let (_, md) = SafeTensors::read_metadata(&bytes).unwrap();
        let meta = md.metadata().clone().unwrap_or_default();
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let arrays = st
            .tensors()
            .into_iter()
            .map(|(k, v)| (k, (v.shape().to_vec(), v.dtype(), v.data().to_vec())))
            .collect();
        Self {
            name: name.to_string(),
            meta,
            arrays,
        }
    }

    pub fn has(&self, key: &str) -> bool {
        self.arrays.contains_key(key)
    }

    pub fn shape(&self, key: &str) -> Vec<usize> {
        self.arrays
            .get(key)
            .unwrap_or_else(|| panic!("{}: no array {key}", self.name))
            .0
            .clone()
    }

    pub fn f32(&self, key: &str) -> Vec<f32> {
        let (_, dt, b) = &self.arrays[key];
        assert_eq!(*dt, Dtype::F32, "{key}");
        b.chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    pub fn i64(&self, key: &str) -> Vec<i64> {
        let (_, dt, b) = &self.arrays[key];
        assert_eq!(*dt, Dtype::I64, "{key}");
        b.chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    pub fn u8(&self, key: &str) -> Vec<u8> {
        let (_, dt, b) = &self.arrays[key];
        assert_eq!(*dt, Dtype::U8, "{key}");
        b.clone()
    }

    pub fn p3(&self, key: &str) -> Vec<[f32; 3]> {
        self.f32(key)
            .chunks_exact(3)
            .map(|c| [c[0], c[1], c[2]])
            .collect()
    }

    pub fn strings(&self, key: &str) -> Vec<String> {
        serde_json::from_str(&self.meta[key]).unwrap()
    }

    pub fn sequence(&self) -> &str {
        &self.meta["sequence"]
    }
}

pub fn library() -> ResidueLibrary {
    let path = rust_dir().join("../glycoflow/resources/residue_library.json");
    ResidueLibrary::from_json_slice(&std::fs::read(path).unwrap()).unwrap()
}

pub fn meta() -> ModelMeta {
    ModelMeta::from_json_str(
        &std::fs::read_to_string(rust_dir().join("fixtures/glycoflow.json")).unwrap(),
    )
    .unwrap()
}

pub fn model(device: &Device, precision: Precision) -> TorsionFlowNet {
    let path = rust_dir().join("weights/glycoflow.safetensors");
    let bytes = std::fs::read(&path).unwrap_or_else(|e| {
        panic!(
            "{}: {e} (run scripts/export_rust_fixtures.py)",
            path.display()
        )
    });
    TorsionFlowNet::from_safetensors(&bytes, meta().config, device, precision).unwrap()
}

/// Devices to test: CPU, plus CUDA when built with `--features cuda`.
pub fn devices() -> Vec<(&'static str, Device)> {
    let mut d = vec![("cpu", Device::Cpu)];
    if cfg!(feature = "cuda") {
        d.push(("cuda", Device::new_cuda(0).expect("cuda device")));
    }
    d
}

pub fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

/// Max |wrap(a - b)| for angles.
pub fn max_ang(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = (*x as f64 - *y as f64).rem_euclid(2.0 * std::f64::consts::PI);
            d.min(2.0 * std::f64::consts::PI - d) as f32
        })
        .fold(0.0, f32::max)
}

pub fn max_dist(a: &[[f32; 3]], b: &[[f32; 3]]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(p, q)| {
            ((p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2)).sqrt()
        })
        .fold(0.0, f32::max)
}

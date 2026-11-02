//! GlycoFlow inference engine (Rust port of the PyTorch reference in `glycoflow/`).
//!
//! * [`library`], [`sequence`], [`builder`]: GLYCAM sequence -> heavy-atom 3D template
//!   (`glycoflow/builder.py`).
//! * [`topology`]: covalent graph, rotatable torsions, distal masks, topological distances and
//!   model tokens (`glycoflow/data.py::structure_entry`, `glycoflow/dataset.py::Vocab`).
//! * [`geometry`]: dihedrals, torsion updates (`glycoflow/geometry.py`), the analytic torsion
//!   Jacobian and Cartesian -> torsion gradient propagation.
//! * [`model`]: `TorsionFlowNet.forward` on candle (CPU, optional CUDA).
//! * [`sampler`]: Euler / Heun ODE integration (`glycoflow/flow.py::sample`) with a guidance hook;
//!   [`guidance`]: the clash guidance of the Python sampler as a hook.
//! * [`kernels`]: fused CPU / CUDA (NVRTC) kernels for the memory-bound parts of the network.
//! * [`pucker`]: ring enumeration, the Cremer-Pople measure and its closed-form inverse
//!   (`glycoflow/pucker.py`, `glycoflow/data.py::ring_cycles`).
//! * [`landscape`]: rarity of a conformer within a sampled ensemble (kNN, the validated
//!   estimator) and paths between two conformers through the model's prior.
//! * [`quality`]: the validator's heavy-atom rules on a conformer, and a light repair ([`quality::tidy`]).
//! * [`pdb`]: multi-model PDB writer and ensemble alignment (`glycoflow/api.py`).
//!
//! The core API takes strings and bytes only (no filesystem access), so it builds for wasm32.

pub mod backend;
pub mod builder;
pub mod error;
pub mod geometry;
pub mod guidance;
pub mod kernels;
pub mod landscape;
pub mod library;
pub mod linalg;
pub mod model;
pub mod pdb;
pub mod pucker;
pub mod quality;
pub mod rng;
pub mod sampler;
pub mod sequence;
pub mod topology;

pub use builder::BuiltGlycan;
pub use error::{Error, Result};
pub use library::ResidueLibrary;
pub use model::{ModelConfig, ModelMeta, TorsionFlowNet};
pub use pucker::{ring_cycles, ring_pucker, ring_puckers, set_puckers, Ring, PUCKER_DIM, PUCKER_SCALE};
pub use sampler::{Method, SampleOptions};
pub use topology::{Glycan, Topology, Vocab};

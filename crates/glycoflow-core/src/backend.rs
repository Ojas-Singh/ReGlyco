//! Network evaluation outside candle: an optional backend (e.g. WebGPU in the browser) that holds
//! the same model and computes velocities for batches of conformers. Samplers register their
//! glycan with it once and send every network evaluation to it; when there is no backend, or it
//! declines (returns None), the candle path is used. The sampler, guidance and everything built on
//! it are unchanged.

use std::sync::{Arc, RwLock};

use crate::geometry::P3;
use crate::topology::{Topology, TOPO_DIST_CAP};

/// The glycan graph a backend needs (one glycan shared by every conformer of a batch).
pub struct BackendGraph<'a> {
    pub n_atoms: usize,
    pub n_torsions: usize,
    /// per atom: element, atom name, residue, link code, ring flag (vocabulary indices)
    pub tokens: &'a [[u32; 5]],
    /// topological distance [N*N], capped at the topology's cap
    pub topo_dist: &'a [u8],
    pub topo_cap: u8,
    /// torsion quads (a, b, c, d)
    pub quads: &'a [[usize; 4]],
    /// atoms moved by each torsion
    pub distal: &'a [Vec<usize>],
    /// sugar rings in canonical order (the ring head pools each ring's atoms in this order)
    pub rings: &'a [crate::pucker::Ring],
}

pub trait NetworkBackend: Send + Sync {
    /// Register a glycan; returns its handle, or None to leave this glycan to candle.
    fn register(&self, graph: &BackendGraph) -> Option<u32>;
    /// Velocities [B*T] for conformers `coords` [B*N], torsions `tau` [B*T] and times `t` [B];
    /// None to fall back to candle for this call.
    fn velocity(&self, glycan: u32, coords: &[P3], tau: &[f32], t: &[f32]) -> Option<Vec<f32>>;
    /// Torsion velocities [B*T] and ring-shape velocities [B*R*3] of a ring-head model, given the
    /// ring state `q` [B*R*3] (`TorsionFlowNet::forward_joint`); None (the default) to leave ring
    /// generation to candle.
    fn velocity_joint(
        &self,
        _glycan: u32,
        _coords: &[P3],
        _tau: &[f32],
        _t: &[f32],
        _q: &[f32],
    ) -> Option<(Vec<f32>, Vec<f32>)> {
        None
    }
    /// The glycan is no longer needed.
    fn release(&self, _glycan: u32) {}
}

static BACKEND: RwLock<Option<Arc<dyn NetworkBackend>>> = RwLock::new(None);

/// Install (or remove, with None) the process-wide network backend. Samplers created afterwards
/// use it.
pub fn set_network_backend(backend: Option<Arc<dyn NetworkBackend>>) {
    *BACKEND.write().unwrap_or_else(|e| e.into_inner()) = backend;
}

pub fn network_backend() -> Option<Arc<dyn NetworkBackend>> {
    BACKEND.read().unwrap_or_else(|e| e.into_inner()).clone()
}

/// A sampler's registration with the backend.
pub(crate) struct Registration {
    pub backend: Arc<dyn NetworkBackend>,
    pub glycan: u32,
}

impl Drop for Registration {
    fn drop(&mut self) {
        self.backend.release(self.glycan);
    }
}

pub(crate) fn register(topology: &Topology, tokens: &[[u32; 5]]) -> Option<Registration> {
    let backend = network_backend()?;
    let graph = BackendGraph {
        n_atoms: topology.n_atoms,
        n_torsions: topology.n_torsions(),
        tokens,
        topo_dist: &topology.topo_dist,
        topo_cap: TOPO_DIST_CAP,
        quads: &topology.quads,
        distal: &topology.distal,
        rings: &topology.rings,
    };
    let glycan = backend.register(&graph)?;
    Some(Registration { backend, glycan })
}

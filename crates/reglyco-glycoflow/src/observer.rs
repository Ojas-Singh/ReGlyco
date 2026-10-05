//! Watching a fit as it runs: stage changes, every step of the guided GlycoFlow generation, the
//! samples and basins the search keeps, and every iteration of the restrained Cartesian
//! refinement. For progress reporting and for showing the search (the browser fit page).
//!
//! Observing never changes a fit. Every method has a no-op default, and an unobserved fit does
//! no extra work. Methods may be called from rayon worker threads (the Cartesian stage refines
//! basins in parallel), so implementations must be `Sync` and should only record.

use std::sync::Arc;

use crate::problem::{Pose, Terms, V3};

/// One integration step of guided generation for a batch of conformers.
pub struct GuidedStep<'a> {
    /// index of the batch's first conformer in the whole sample
    pub first: usize,
    /// integration step (`steps` for the final state, t = 1)
    pub step: usize,
    pub steps: usize,
    pub t: f64,
    /// pucker template of each conformer
    pub templates: &'a [usize],
    /// torsions at t, [batch x T] (radians)
    pub tau: &'a [f32],
    /// predicted endpoint wrap(tau + (1 - t) v), [batch x T]
    pub tau_hat: &'a [f32],
    /// attachment angles in use at this step (after the guidance update)
    pub psi: &'a [f64],
    pub phi: &'a [f64],
    /// whether the objective steered this step (t >= guidance start)
    pub guided: bool,
}

/// One iteration of the restrained Cartesian refinement of a basin.
pub struct CartesianStep<'a> {
    pub basin: usize,
    /// "refine", "escalation x10", "escalation x100", ...
    pub pass: &'a str,
    pub iteration: usize,
    pub iterations: usize,
    /// coordinates at this iteration (template atom order)
    pub x: &'a [V3],
    pub terms: &'a Terms,
    pub e_restraint: f64,
}

pub trait FitObserver: Send + Sync {
    /// A stage starts ("prior", "guided sample", "select", "refine", "polish", "steric polish",
    /// "cartesian", "support+complete").
    fn stage(&self, _name: &str) {}
    /// Progress within a stage.
    fn progress(&self, _stage: &str, _done: usize, _total: usize) {}
    fn guided_step(&self, _step: &GuidedStep<'_>) {}
    /// Guided generation done: every sample's pose (best of guided and grid-searched attachment)
    /// and objective, in sample order.
    fn samples(&self, _poses: &[Pose], _energies: &[f64]) {}
    /// Basins at a stage ("select": as chosen from the samples, `sources` = their sample
    /// indices; "refine", "polish", "steric polish": after that stage; "final": after the
    /// Cartesian stage). `energies` is the stage's ranking energy.
    fn basins(&self, _stage: &str, _sources: &[usize], _poses: &[Pose], _energies: &[f64]) {}
    fn cartesian_step(&self, _step: &CartesianStep<'_>) {}
}

/// An optional observer, shareable across threads; `Default` is none.
#[derive(Clone, Default)]
pub struct Observer(pub Option<Arc<dyn FitObserver>>);

impl Observer {
    pub fn new(observer: Arc<dyn FitObserver>) -> Self {
        Self(Some(observer))
    }
    pub fn get(&self) -> Option<&dyn FitObserver> {
        self.0.as_deref()
    }
}

impl std::fmt::Debug for Observer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(if self.0.is_some() { "Observer(set)" } else { "Observer(none)" })
    }
}

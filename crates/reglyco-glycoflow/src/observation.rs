//! Observation abstraction: an experimental data term of the fitting objective.
//!
//! The fitting objective is `E(x) = observation energy + restraints + prior`. Every observation
//! is prepared once for a glycan (which atoms it scores and with what weight) and then evaluated
//! on placed conformers, returning its energy (a negative log-likelihood, lower is better) and,
//! on request, `dE/dx`. The density map is the first implementation; a SAXS curve fits the same
//! interface (prepare: form factors per atom; evaluate: chi^2 of the predicted curve and its
//! Cartesian gradient).

use rayon::prelude::*;
use reglyco_density::site_likelihood::SiteLikelihood;

use crate::error::{Result, invalid};

/// Glycan atoms an observation scores.
#[derive(Debug, Clone)]
pub struct ObservedAtoms {
    /// element symbols, template atom order
    pub elements: Vec<String>,
    /// atoms that belong to the glycan proper (false for the aglycone placeholder)
    pub scored: Vec<bool>,
}

/// Value of an observation term for one conformer.
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct ObservationValue {
    /// objective contribution (negative log-likelihood, up to a constant)
    pub energy: f64,
    pub log_likelihood: f64,
    /// observation-specific diagnostics (density: drop in SSE, partial correlation, scale)
    pub gain: f64,
    pub partial_correlation: f64,
    pub scale: f64,
}

pub trait Observation: Send + Sync {
    /// Short name for reports ("density", "saxs", ...).
    fn kind(&self) -> &'static str;

    /// Set up per-atom weights for a glycan; called once before any evaluation.
    fn prepare(&mut self, atoms: &ObservedAtoms) -> Result<()>;

    /// Energy of one placed conformer `x` (template atom order). `active` switches atoms off
    /// (support tests remove a subtree); `grad` receives `dE/dx` when given.
    fn evaluate(
        &self,
        x: &[[f64; 3]],
        active: Option<&[bool]>,
        grad: Option<&mut [[f64; 3]]>,
    ) -> ObservationValue;

    /// The density likelihood behind this observation, if it is one.
    fn density(&self) -> Option<&SiteLikelihood> {
        None
    }

    /// Batch evaluation (parallel over conformers).
    fn evaluate_batch(
        &self,
        xs: &[Vec<[f64; 3]>],
        want_grad: bool,
    ) -> Vec<(ObservationValue, Option<Vec<[f64; 3]>>)> {
        xs.par_iter()
            .map(|x| {
                if want_grad {
                    let mut g = vec![[0.0; 3]; x.len()];
                    let v = self.evaluate(x, None, Some(&mut g));
                    (v, Some(g))
                } else {
                    (self.evaluate(x, None, None), None)
                }
            })
            .collect()
    }
}

/// The profiled least-squares site likelihood of a density map
/// (`reglyco_density::site_likelihood`, Python `glycoflow/fitting/density.py`): atoms weighted
/// by their atomic number, energy = -log-likelihood.
pub struct DensityObservation {
    pub likelihood: SiteLikelihood,
    z: Vec<f64>,
}

impl DensityObservation {
    pub fn new(likelihood: SiteLikelihood) -> Self {
        Self {
            likelihood,
            z: Vec::new(),
        }
    }

    pub fn weights(&self) -> &[f64] {
        &self.z
    }
}

/// Atomic numbers of the glycan elements as used by the reference (`problem.Z_OF`).
pub fn glycan_z(element: &str) -> Option<f64> {
    match element {
        "C" => Some(6.0),
        "N" => Some(7.0),
        "O" => Some(8.0),
        "S" => Some(16.0),
        "P" => Some(15.0),
        _ => None,
    }
}

impl Observation for DensityObservation {
    fn kind(&self) -> &'static str {
        "density"
    }

    fn density(&self) -> Option<&SiteLikelihood> {
        Some(&self.likelihood)
    }

    fn prepare(&mut self, atoms: &ObservedAtoms) -> Result<()> {
        self.z = atoms
            .elements
            .iter()
            .zip(&atoms.scored)
            .map(|(e, s)| {
                glycan_z(e)
                    .map(|z| if *s { z } else { 0.0 })
                    .ok_or_else(|| invalid(format!("unsupported glycan element {e}")))
            })
            .collect::<Result<_>>()?;
        Ok(())
    }

    fn evaluate(
        &self,
        x: &[[f64; 3]],
        active: Option<&[bool]>,
        grad: Option<&mut [[f64; 3]]>,
    ) -> ObservationValue {
        let masked;
        let z: &[f64] = match active {
            Some(active) => {
                masked = self
                    .z
                    .iter()
                    .zip(active)
                    .map(|(z, a)| if *a { *z } else { 0.0 })
                    .collect::<Vec<_>>();
                &masked
            }
            None => &self.z,
        };
        let score = match grad {
            Some(g) => {
                let score = self.likelihood.evaluate(x, z, Some(&mut *g));
                negate(g);
                score
            }
            None => self.likelihood.evaluate(x, z, None),
        };
        ObservationValue {
            energy: -score.log_likelihood,
            log_likelihood: score.log_likelihood,
            gain: score.gain,
            partial_correlation: score.partial_correlation,
            scale: score.scale,
        }
    }
}

/// Negate a log-likelihood gradient in place into an energy gradient.
fn negate(g: &mut [[f64; 3]]) {
    for v in g {
        for c in v {
            *c = -*c;
        }
    }
}

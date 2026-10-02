//! GlycoFlow-guided fitting of a glycan of known sequence to a protein site in a density map.
//!
//! Rust port of the validated Python reference `glycoflow/fitting/` (GlycoFlow repository),
//! running the frozen GlycoFlow flow model through the `glycoflow-core` engine:
//!
//! * [`site`]: anchor atoms, deposited glycan tree (naming and evaluation only), GLYCAM sequence
//!   (crabWURCS `write_glycam`), environment with crystal symmetry mates ([`symmetry`]);
//! * [`observation`]: the [`observation::Observation`] trait (prepare / evaluate with `dE/dx`)
//!   and its density implementation on `reglyco_density::site_likelihood`;
//! * [`problem`]: GlycoFlow templates, attachment kinematics and the single objective
//!   `-loglik + w_env E_env + w_self E_self + E_attach + w_prior E_prior` with analytic
//!   gradients; [`prior`]: the GlycoFlow marginal prior;
//! * [`search`]: observation-guided generation through the sampler's guidance hook, attachment
//!   grid search, Adam refinement, distinct basins; [`support`]: subtree support and prior
//!   completion (pinned-torsion inpainting); [`pipeline`]: `fit_site`;
//! * [`output`], [`evaluation`], [`workflow`]: structures, reports, validation.

pub mod counter;
pub mod error;
pub mod evaluation;
pub mod infer;
pub mod model;
pub mod observation;
pub mod output;
pub mod pipeline;
pub mod prior;
pub mod problem;
pub mod search;
pub mod site;
pub mod support;
pub mod symmetry;
pub mod workflow;

pub use error::{FitError, Result};
pub use glycoflow_core;
pub use model::{ComputeDevice, GlycoflowModel};
pub use observation::{DensityObservation, Observation, ObservationValue};
pub use pipeline::{FitConfig, FitOutcome, PriorConfig, fit_site};
pub use problem::{Pose, ProblemOptions, SiteProblem, Terms};
pub use site::{Site, SiteOptions, SymmetryMode, load_site};
pub use workflow::{SiteRequest, WorkflowInput, WorkflowOptions, WorkflowResult};

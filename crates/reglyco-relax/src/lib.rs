//! OBC2/force-field coordinate minimization without format round trips.

use std::collections::BTreeSet;

use glysys::{ParameterizedSystem, ResidueId, Structure, Vec3};
use glysys_energy::{AtomSelection, EnergyComponents, EnergyEvaluator, EnergyOptions, Obc2Options};
use glysys_opt::{DifferentiableObjective, LbfgsConfig, lbfgs_minimize_with_progress};

pub type Result<T> = std::result::Result<T, RelaxError>;

#[derive(Debug, thiserror::Error)]
pub enum RelaxError {
    #[error(transparent)]
    GlySys(#[from] glysys::BuildError),
    #[error(transparent)]
    Energy(#[from] glysys_energy::EnergyError),
    #[error(transparent)]
    Optimization(#[from] glysys_opt::OptimizationError),
    #[error("the movable atom selection is empty")]
    EmptySelection,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MovableSelection {
    #[default]
    Glycans,
    All,
    Residues(Vec<ResidueId>),
    /// Explicit atom indices for high-throughput local search objectives.
    Atoms(Vec<usize>),
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct RelaxOptions {
    pub movable: MovableSelection,
    pub include_local_sidechains: bool,
    pub local_radius: f64,
    /// Optional nonbonded cutoff used during minimization.  Keeping this
    /// configurable preserves the exact all-pairs path for callers that need
    /// it, while local density fitting can use a finite cutoff to avoid
    /// rebuilding millions of distant, constant protein--glycan pairs.
    pub nonbonded_cutoff: Option<f64>,
    /// Restrict the optimization objective to terms that touch movable
    /// atoms. Fixed-only terms are constant and need not be recomputed during
    /// a local fit; the default remains the historical full objective.
    pub active_terms_only: bool,
    /// Optional OBC2 GBSA solvent. Vacuum is the responsive default; OBC2 is
    /// an explicit scientific opt-in because its global derivative evaluation
    /// is substantially more expensive for large proteins.
    pub obc2: Option<Obc2Options>,
    pub lbfgs: LbfgsConfig,
}

impl Default for RelaxOptions {
    fn default() -> Self {
        Self {
            movable: MovableSelection::Glycans,
            include_local_sidechains: true,
            local_radius: 5.0,
            nonbonded_cutoff: None,
            active_terms_only: false,
            obc2: None,
            lbfgs: LbfgsConfig::default(),
        }
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RelaxationStageDiagnostics {
    pub name: String,
    pub initial_energy: EnergyComponents,
    pub final_energy: EnergyComponents,
    pub iterations: usize,
    pub converged: bool,
    pub convergence_reason: String,
    pub movable_atoms: usize,
    pub accepted_steps: usize,
    pub final_rms_gradient: f64,
    pub final_max_gradient: f64,
    pub energy_history: Vec<f64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RelaxationDiagnostics {
    pub initial_energy: EnergyComponents,
    pub final_energy: EnergyComponents,
    pub iterations: usize,
    pub converged: bool,
    pub movable_atoms: usize,
    pub energy_history: Vec<f64>,
    pub stages: Vec<RelaxationStageDiagnostics>,
}

#[derive(Debug, Clone)]
pub struct RelaxationResult {
    pub structure: Structure,
    pub system: ParameterizedSystem,
    pub diagnostics: RelaxationDiagnostics,
}

/// Timely, structured status from the staged minimizer.
///
/// The library never writes to stdout or stderr itself. Command-line and GUI
/// clients can use these notifications to keep users informed while an OBC2
/// objective evaluation is in progress.
#[derive(Debug, Clone)]
pub enum RelaxProgress {
    InitialEnergyStarted,
    InitialEnergy {
        energy: EnergyComponents,
    },
    StageStarted {
        name: String,
        movable_atoms: usize,
    },
    Iteration {
        stage: String,
        iteration: usize,
        energy: f64,
        rms_gradient: f64,
        max_gradient: f64,
        accepted_steps: usize,
    },
    StageFinished {
        name: String,
        diagnostics: Box<RelaxationStageDiagnostics>,
    },
}

pub fn relax(
    structure: &Structure,
    system: &ParameterizedSystem,
    options: &RelaxOptions,
) -> Result<RelaxationResult> {
    relax_with_progress(structure, system, options, |_| {})
}

/// Minimize one explicit atom selection without invoking the staged relax
/// protocol. This is the lightweight path used for per-candidate search.
pub fn minimize_coordinates_once(
    system: &ParameterizedSystem,
    coordinates: &[Vec3],
    movable: &[usize],
    energy_options: EnergyOptions,
    lbfgs: &LbfgsConfig,
) -> Result<(Vec<Vec3>, RelaxationStageDiagnostics)> {
    let options = RelaxOptions {
        movable: MovableSelection::Atoms(movable.to_vec()),
        include_local_sidechains: false,
        obc2: energy_options.obc2.clone(),
        lbfgs: lbfgs.clone(),
        ..RelaxOptions::default()
    };
    minimize_stage(
        "local_search",
        system,
        coordinates,
        &options,
        energy_options,
        true,
        &mut |_| {},
    )
}

/// Run staged relaxation and send synchronous progress notifications to
/// `progress`. In particular, `StageStarted` is emitted before the expensive
/// first OBC2 evaluation, so callers can report activity immediately.
pub fn relax_with_progress<F>(
    structure: &Structure,
    system: &ParameterizedSystem,
    options: &RelaxOptions,
    mut progress: F,
) -> Result<RelaxationResult>
where
    F: FnMut(RelaxProgress),
{
    let mut coordinates = system.coordinates();
    let energy_options = EnergyOptions {
        cutoff: options.nonbonded_cutoff,
        obc2: options.obc2.clone(),
        ..EnergyOptions::default()
    };
    progress(RelaxProgress::InitialEnergyStarted);
    let initial_energy = EnergyEvaluator::new(system, energy_options.clone())?
        .energy(&coordinates)?
        .components;
    progress(RelaxProgress::InitialEnergy {
        energy: initial_energy,
    });
    let mut stages = Vec::new();

    let mut glycan_only = options.clone();
    glycan_only.include_local_sidechains = false;
    let (next, first) = minimize_stage(
        "glycans",
        system,
        &coordinates,
        &glycan_only,
        energy_options.clone(),
        options.active_terms_only,
        &mut progress,
    )?;
    coordinates = next;
    stages.push(first);

    if options.include_local_sidechains {
        let mut local = options.clone();
        local.include_local_sidechains = true;
        let (next, second) = minimize_stage(
            "glycans_and_local_sidechains",
            system,
            &coordinates,
            &local,
            energy_options,
            options.active_terms_only,
            &mut progress,
        )?;
        coordinates = next;
        stages.push(second);
    } else if !matches!(options.movable, MovableSelection::Glycans) {
        let (next, second) = minimize_stage(
            "requested_selection",
            system,
            &coordinates,
            options,
            energy_options,
            options.active_terms_only,
            &mut progress,
        )?;
        coordinates = next;
        stages.push(second);
    }
    let final_energy = stages.last().expect("at least one stage").final_energy;
    let mut relaxed_system = system.clone();
    relaxed_system.set_coordinates(&coordinates)?;
    let mut relaxed_structure = structure.clone();
    relaxed_structure.update_from_parameterized(&relaxed_system)?;
    Ok(RelaxationResult {
        structure: relaxed_structure,
        system: relaxed_system,
        diagnostics: RelaxationDiagnostics {
            initial_energy,
            final_energy,
            iterations: stages.iter().map(|stage| stage.iterations).sum(),
            converged: stages.iter().all(|stage| stage.converged),
            movable_atoms: stages
                .iter()
                .map(|stage| stage.movable_atoms)
                .max()
                .unwrap_or(0),
            energy_history: stages
                .iter()
                .flat_map(|stage| stage.energy_history.iter().copied())
                .collect(),
            stages,
        },
    })
}

fn minimize_stage<F>(
    name: &str,
    system: &ParameterizedSystem,
    coordinates: &[Vec3],
    options: &RelaxOptions,
    energy_options: EnergyOptions,
    active_terms_only: bool,
    progress: &mut F,
) -> Result<(Vec<Vec3>, RelaxationStageDiagnostics)>
where
    F: FnMut(RelaxProgress),
{
    let movable = movable_indices(system, options);
    if movable.is_empty() {
        return Err(RelaxError::EmptySelection);
    }
    progress(RelaxProgress::StageStarted {
        name: name.into(),
        movable_atoms: movable.len(),
    });
    let selection = AtomSelection::from_indices(system.atom_count(), movable.iter().copied());
    let evaluator = if active_terms_only {
        EnergyEvaluator::new(system, energy_options.clone())?.with_active_terms(selection)?
    } else {
        EnergyEvaluator::new(system, energy_options.clone())?.with_selection(selection)?
    };
    // The active evaluator is the fast objective used by local minimization;
    // retain full diagnostic energies so reports do not compare a partial
    // stage value with the complete initial energy.
    let diagnostic_evaluator = active_terms_only
        .then(|| EnergyEvaluator::new(system, energy_options.clone()))
        .transpose()?;
    let initial_energy = if let Some(evaluator) = &diagnostic_evaluator {
        evaluator.energy(coordinates)?.components
    } else {
        evaluator.energy(coordinates)?.components
    };
    let initial = flatten_selected(coordinates, &movable);
    let mut objective = CoordinateObjective {
        evaluator,
        base: coordinates.to_vec(),
        movable: movable.clone(),
    };
    let outcome = lbfgs_minimize_with_progress(&mut objective, &initial, &options.lbfgs, |item| {
        progress(RelaxProgress::Iteration {
            stage: name.into(),
            iteration: item.iteration,
            energy: item.value,
            rms_gradient: item.rms_gradient,
            max_gradient: item.max_gradient,
            accepted_steps: item.accepted_steps,
        });
    })?;
    let final_coordinates = objective.coordinates(&outcome.point);
    let evaluated = objective
        .evaluator
        .energy_and_gradient(&final_coordinates)?;
    let final_energy = if let Some(evaluator) = &diagnostic_evaluator {
        evaluator.energy(&final_coordinates)?.components
    } else {
        evaluated.components
    };
    let gradients = evaluated.gradients.expect("gradient evaluation requested");
    let gradient_values = movable
        .iter()
        .flat_map(|atom| {
            let gradient = gradients[*atom];
            [gradient.x, gradient.y, gradient.z]
        })
        .collect::<Vec<_>>();
    let final_max_gradient = gradient_values
        .iter()
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let final_rms_gradient = (gradient_values
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        / gradient_values.len().max(1) as f64)
        .sqrt();
    let diagnostics = RelaxationStageDiagnostics {
        name: name.into(),
        initial_energy,
        final_energy,
        iterations: outcome.iterations,
        converged: outcome.converged,
        convergence_reason: if outcome.converged {
            "gradient_tolerance".into()
        } else {
            "maximum_iterations".into()
        },
        movable_atoms: movable.len(),
        accepted_steps: outcome.history.len().saturating_sub(1),
        final_rms_gradient,
        final_max_gradient,
        energy_history: outcome.history,
    };
    progress(RelaxProgress::StageFinished {
        name: name.into(),
        diagnostics: Box::new(diagnostics.clone()),
    });
    Ok((final_coordinates, diagnostics))
}

fn movable_indices(system: &ParameterizedSystem, options: &RelaxOptions) -> Vec<usize> {
    if matches!(options.movable, MovableSelection::All) {
        return (0..system.atom_count()).collect();
    }
    let selected_residues = match &options.movable {
        MovableSelection::Glycans => {
            let annotated = system
                .metadata()
                .glycan_trees
                .iter()
                .flat_map(|tree| tree.residue_ids.iter().cloned())
                .collect::<BTreeSet<_>>();
            if annotated.is_empty() {
                system
                    .residues()
                    .iter()
                    .filter(|residue| is_glycan_residue_name(residue.name()))
                    .map(|residue| ResidueId {
                        chain: residue.chain().into(),
                        number: residue.number(),
                        insertion_code: residue.insertion_code(),
                    })
                    .collect()
            } else {
                annotated
            }
        }
        MovableSelection::Residues(residues) => residues.iter().cloned().collect(),
        MovableSelection::Atoms(atoms) => return atoms.clone(),
        MovableSelection::All => unreachable!(),
    };
    let mut indices = system
        .residues()
        .iter()
        .filter(|residue| {
            selected_residues.contains(&ResidueId {
                chain: residue.chain().into(),
                number: residue.number(),
                insertion_code: residue.insertion_code(),
            })
        })
        .flat_map(glysys::Residue::atom_range)
        .collect::<BTreeSet<_>>();
    if options.include_local_sidechains {
        let coordinates = system.coordinates();
        let selected_atoms = indices
            .iter()
            .copied()
            .filter(|atom| system.atoms()[*atom].element() != 1)
            .collect::<Vec<_>>();
        for residue in system.residues() {
            if is_glycan_residue_name(residue.name()) {
                continue;
            }
            let nearby = residue.atom_range().any(|atom| {
                if system.atoms()[atom].element() == 1 {
                    return false;
                }
                selected_atoms.iter().any(|selected| {
                    squared_distance(coordinates[atom], coordinates[*selected])
                        <= options.local_radius.powi(2)
                })
            });
            if nearby {
                indices.extend(residue.atom_range().filter(|atom| {
                    let name = system.atoms()[*atom].name();
                    !matches!(
                        name,
                        "N" | "CA" | "C" | "O" | "OXT" | "H" | "HA" | "HA2" | "HA3"
                    )
                }));
            }
        }
    }
    indices.into_iter().collect()
}

fn is_glycan_residue_name(name: &str) -> bool {
    !matches!(
        name,
        "ALA"
            | "ARG"
            | "ASN"
            | "ASP"
            | "ASH"
            | "CYS"
            | "CYM"
            | "CYX"
            | "GLN"
            | "GLU"
            | "GLH"
            | "GLY"
            | "HID"
            | "HIE"
            | "HIP"
            | "HIS"
            | "ILE"
            | "LEU"
            | "LYS"
            | "LYN"
            | "MET"
            | "PHE"
            | "PRO"
            | "SER"
            | "THR"
            | "TRP"
            | "TYR"
            | "VAL"
            | "ACE"
            | "NME"
            | "HOH"
            | "WAT"
            | "Na+"
            | "Cl-"
    )
}

struct CoordinateObjective<'a> {
    evaluator: EnergyEvaluator<'a>,
    base: Vec<Vec3>,
    movable: Vec<usize>,
}

impl CoordinateObjective<'_> {
    fn coordinates(&self, point: &[f64]) -> Vec<Vec3> {
        let mut coordinates = self.base.clone();
        for (offset, atom) in self.movable.iter().enumerate() {
            coordinates[*atom] = Vec3 {
                x: point[offset * 3],
                y: point[offset * 3 + 1],
                z: point[offset * 3 + 2],
            };
        }
        coordinates
    }
}

impl DifferentiableObjective for CoordinateObjective<'_> {
    fn dimension(&self) -> usize {
        self.movable.len() * 3
    }

    fn value_gradient(&mut self, point: &[f64], gradient: &mut [f64]) -> glysys_opt::Result<f64> {
        let coordinates = self.coordinates(point);
        let result = self
            .evaluator
            .energy_and_gradient(&coordinates)
            .map_err(|_| glysys_opt::OptimizationError::NonFiniteObjective)?;
        let total = result.total();
        let derivatives = result
            .gradients
            .as_ref()
            .ok_or(glysys_opt::OptimizationError::NonFiniteObjective)?;
        for (offset, atom) in self.movable.iter().enumerate() {
            gradient[offset * 3] = derivatives[*atom].x;
            gradient[offset * 3 + 1] = derivatives[*atom].y;
            gradient[offset * 3 + 2] = derivatives[*atom].z;
        }
        Ok(total)
    }
}

fn flatten_selected(coordinates: &[Vec3], selected: &[usize]) -> Vec<f64> {
    selected
        .iter()
        .flat_map(|index| {
            let point = coordinates[*index];
            [point.x, point.y, point.z]
        })
        .collect()
}

fn squared_distance(first: Vec3, second: Vec3) -> f64 {
    (first.x - second.x).powi(2) + (first.y - second.y).powi(2) + (first.z - second.z).powi(2)
}

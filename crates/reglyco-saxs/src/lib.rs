//! Re-Glyco's in-memory adapter to the reusable crabSAXS library.

pub use crabsaxs::{
    Assignment, CandidateCombination, CombinationAnalysis, ExperimentalCurve, FitOptions,
    MaximumEntropyOptions, PrOptions,
};

use crabsaxs::{
    CoordinateFeatures, DiagnosticPlot, ExperimentalAnalysis, PairDistribution, PlotCurve,
    RankWeights, SaxsFeatures, SaxsScorer, ScoreOptions, Structure, analyze_experimental,
    compute_curve, coordinate_features, features_from_fit, fit_calculated_curve, reweight_curves,
};
use glysys::{Structure as GlysysStructure, StructureAtom};
use nalgebra::Vector3;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Default)]
pub struct SaxsRefinementOptions {
    pub profile_path: PathBuf,
}

#[derive(Debug, thiserror::Error)]
pub enum SaxsError {
    #[error("SAXS input or fitting failed: {0}")]
    Crab(#[from] crabsaxs::SaxsError),
    #[error("SAXS workflow input is invalid: {0}")]
    Invalid(String),
    #[error("SAXS workflow I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("SAXS workflow serialization failed: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, SaxsError>;

/// Parse an experimental SAXS curve without filesystem access. This mirrors
/// crabSAXS' deposited-file rules so CLI and browser workflows normalize q
/// units and uncertainty columns identically.
pub fn experimental_curve_from_str(contents: &str) -> Result<ExperimentalCurve> {
    let mut curve = ExperimentalCurve {
        has_errors: true,
        ..ExperimentalCurve::default()
    };
    let mut q_scale = 1.0;
    let mut saw_numeric = false;
    for (line_index, line) in contents.lines().enumerate() {
        let trimmed = line.trim();
        let lower = trimmed.to_ascii_lowercase();
        if lower.contains("nm") && (lower.contains('q') || lower.contains("angstrom")) {
            q_scale = 0.1;
        }
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let normalized = trimmed.replace(',', " ");
        let fields = normalized.split_whitespace().collect::<Vec<_>>();
        if fields
            .first()
            .and_then(|value| value.parse::<f64>().ok())
            .is_none()
        {
            continue;
        }
        if fields.len() < 2 {
            return Err(SaxsError::Invalid(format!(
                "expected q and intensity on line {}",
                line_index + 1
            )));
        }
        let values = fields[..fields.len().min(3)]
            .iter()
            .map(|value| value.parse::<f64>())
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|_| {
                SaxsError::Invalid(format!("invalid number on line {}", line_index + 1))
            })?;
        if !saw_numeric && q_scale == 1.0 && values[0] > 1.0 {
            q_scale = 0.1;
        }
        saw_numeric = true;
        let q = values[0] * q_scale;
        let intensity = values[1];
        let sigma = values.get(2).copied().unwrap_or(1.0);
        if intensity == 0.0 && sigma == 0.0 {
            continue;
        }
        if !q.is_finite() || q < 0.0 || !intensity.is_finite() || !sigma.is_finite() || sigma <= 0.0
        {
            return Err(SaxsError::Invalid(format!(
                "q and sigma must be positive and all values finite on line {}",
                line_index + 1
            )));
        }
        if curve.q.last().is_some_and(|previous| q < *previous) {
            return Err(SaxsError::Invalid(format!(
                "q values must be non-decreasing on line {}",
                line_index + 1
            )));
        }
        curve.q.push(q);
        curve.intensity.push(intensity);
        curve.sigma.push(sigma);
        if fields.len() < 3 {
            curve.has_errors = false;
        }
    }
    if curve.q.len() < 3 {
        return Err(SaxsError::Invalid(
            "experimental SAXS data must contain at least three numeric rows".into(),
        ));
    }
    Ok(curve)
}

/// Retained compatibility entry point. It now validates the experimental
/// profile instead of returning the former milestone placeholder error.
pub fn refine_saxs(options: &SaxsRefinementOptions) -> Result<()> {
    ExperimentalCurve::from_dat_file(&options.profile_path)?;
    Ok(())
}

#[derive(Debug, Clone)]
pub struct NamedStructure {
    pub name: String,
    pub structure: Structure,
    /// Original GlySys structure when this model was generated in-memory.
    /// Keeping it allows reports to reuse glycan metadata and SNFG identity
    /// without reconstructing a molecular file from SAXS coordinates.
    pub source_structure: Option<GlysysStructure>,
    /// Re-Glyco native probability retained for provenance only. SAXS
    /// ensemble weights are initialized uniformly and never multiply this
    /// value into the likelihood.
    pub log_native_probability: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFitSummary {
    pub name: String,
    pub features: SaxsFeatures,
    pub fitted_curve: Vec<f64>,
    pub model_pr: PairDistribution,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleModelResult {
    pub best_index: usize,
    pub models: Vec<ModelFitSummary>,
    pub experimental: ExperimentalAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnbiasedEnsembleResult {
    pub names: Vec<String>,
    pub weights: Vec<f64>,
    pub params: crabsaxs::FitParams,
    pub features: SaxsFeatures,
    pub fitted_curve: Vec<f64>,
    pub mean_curve: Vec<f64>,
    pub conformer_curves: Vec<Vec<f64>>,
    pub native_log_probabilities: Vec<f64>,
    pub model_pr: PairDistribution,
    pub experimental: ExperimentalAnalysis,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReweightFitResult {
    pub names: Vec<String>,
    pub weights: Vec<f64>,
    pub prior_weights: Vec<f64>,
    pub params: crabsaxs::FitParams,
    pub features: SaxsFeatures,
    pub fitted_curve: Vec<f64>,
    pub conformer_curves: Vec<Vec<f64>>,
    pub native_log_probabilities: Vec<f64>,
    pub model_pr: PairDistribution,
    pub kl_divergence: f64,
    pub effective_sample_size: f64,
    pub iterations: usize,
    pub converged: bool,
    pub experimental: ExperimentalAnalysis,
}

pub fn adapt_structure(
    name: impl Into<String>,
    structure: &GlysysStructure,
) -> Result<NamedStructure> {
    let atoms = structure
        .atoms()
        .into_iter()
        .map(convert_atom)
        .collect::<Result<Vec<_>>>()?;
    if atoms.is_empty() {
        return Err(SaxsError::Invalid("structure contains no atoms".into()));
    }
    Ok(NamedStructure {
        name: name.into(),
        structure: Structure { atoms },
        source_structure: Some(structure.clone()),
        log_native_probability: None,
    })
}

pub fn read_models(path: impl AsRef<Path>) -> Result<Vec<NamedStructure>> {
    let path = path.as_ref();
    let models = Structure::from_pdb_file_models_with_options(
        path,
        crabsaxs::structure::ParseOptions {
            include_hetatm: true,
            include_glycans: true,
            ..Default::default()
        },
    )?;
    Ok(models
        .into_iter()
        .enumerate()
        .map(|(index, structure)| NamedStructure {
            name: format!("model-{}", index + 1),
            structure,
            source_structure: None,
            log_native_probability: None,
        })
        .collect())
}

pub fn fit_single_models(
    models: &[NamedStructure],
    experimental: &ExperimentalCurve,
    fit_options: FitOptions,
    pr_options: PrOptions,
) -> Result<SingleModelResult> {
    if models.is_empty() {
        return Err(SaxsError::Invalid(
            "single-model fitting needs at least one model".into(),
        ));
    }
    let experimental_analysis = analyze_experimental(experimental, pr_options);
    let scorer = SaxsScorer::new(
        experimental.clone(),
        ScoreOptions {
            fit: fit_options,
            quality: crabsaxs::Quality::Balanced,
        },
    )?;
    let summaries = models
        .par_iter()
        .map(|model| {
            let score = scorer.score(&model.structure)?;
            let coordinates = coordinate_features(&model.structure, pr_options)?;
            let features = features_from_fit(
                experimental,
                &score.fit.fitted_curve,
                score.fit.params,
                Some(&coordinates),
                Some(&experimental_analysis),
            )?;
            Ok(ModelFitSummary {
                name: model.name.clone(),
                features,
                fitted_curve: score.fit.fitted_curve,
                model_pr: coordinates.p_r,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let best_index = summaries
        .iter()
        .enumerate()
        .min_by(|left, right| {
            left.1
                .features
                .reduced_chi2
                .total_cmp(&right.1.features.reduced_chi2)
        })
        .map(|(index, _)| index)
        .ok_or_else(|| SaxsError::Invalid("no single-model fit succeeded".into()))?;
    Ok(SingleModelResult {
        best_index,
        models: summaries,
        experimental: experimental_analysis,
    })
}

pub fn calculate_curves(
    models: &[NamedStructure],
    experimental: &ExperimentalCurve,
    fit_options: FitOptions,
) -> Result<Vec<Vec<f64>>> {
    if models.is_empty() {
        return Err(SaxsError::Invalid(
            "curve calculation needs at least one model".into(),
        ));
    }
    let compute_options = crabsaxs::ComputeOptions {
        method: fit_options.method,
        hydrogen_mode: fit_options.hydrogen_mode,
        include_hetatm: true,
        multipole: Default::default(),
        solvent: Some(fit_options.solvent),
    };
    models
        .par_iter()
        .map(|model| {
            Ok(compute_curve(&model.structure, &experimental.q, &compute_options)?.intensity)
        })
        .collect::<Result<Vec<_>>>()
}

pub fn fit_unbiased_ensemble(
    models: &[NamedStructure],
    experimental: &ExperimentalCurve,
    fit_options: FitOptions,
    pr_options: PrOptions,
) -> Result<UnbiasedEnsembleResult> {
    let conformer_curves = calculate_curves(models, experimental, fit_options)?;
    let weights = uniform_weights(models.len());
    let mean_curve = mix_curves(&conformer_curves, &weights);
    let fit = fit_calculated_curve(&mean_curve, experimental, &fit_options)?;
    let coordinates = weighted_coordinates(models, &weights, pr_options)?;
    let experimental_analysis = analyze_experimental(experimental, pr_options);
    let features = features_from_fit(
        experimental,
        &fit.fitted_curve,
        fit.params,
        Some(&coordinates),
        Some(&experimental_analysis),
    )?;
    Ok(UnbiasedEnsembleResult {
        names: models.iter().map(|model| model.name.clone()).collect(),
        weights,
        params: fit.params,
        features,
        fitted_curve: fit.fitted_curve,
        mean_curve,
        conformer_curves,
        native_log_probabilities: models
            .iter()
            .filter_map(|model| model.log_native_probability)
            .collect(),
        model_pr: coordinates.p_r.clone(),
        experimental: experimental_analysis,
    })
}

pub fn reweight_ensemble(
    models: &[NamedStructure],
    experimental: &ExperimentalCurve,
    fit_options: FitOptions,
    mut options: MaximumEntropyOptions,
    pr_options: PrOptions,
) -> Result<ReweightFitResult> {
    let conformer_curves = calculate_curves(models, experimental, fit_options)?;
    options.fit = fit_options;
    options.prior_weights = Some(uniform_weights(models.len()));
    let result = reweight_curves(
        models.iter().map(|model| model.name.clone()).collect(),
        &conformer_curves,
        experimental,
        options,
    )?;
    let weights = result.weights.clone();
    let coordinates = weighted_coordinates(models, &weights, pr_options)?;
    let experimental_analysis = analyze_experimental(experimental, pr_options);
    let features = features_from_fit(
        experimental,
        &result.fitted_curve,
        result.params,
        Some(&coordinates),
        Some(&experimental_analysis),
    )?;
    Ok(ReweightFitResult {
        names: result.names,
        weights,
        prior_weights: result.prior_weights,
        params: result.params,
        features,
        fitted_curve: result.fitted_curve,
        conformer_curves,
        native_log_probabilities: models
            .iter()
            .filter_map(|model| model.log_native_probability)
            .collect(),
        model_pr: coordinates.p_r.clone(),
        kl_divergence: result.kl_divergence,
        effective_sample_size: result.effective_sample_size,
        iterations: result.iterations,
        converged: result.converged,
        experimental: experimental_analysis,
    })
}

pub fn plot_for_single(
    experimental: &ExperimentalCurve,
    result: &SingleModelResult,
    best_only: bool,
) -> DiagnosticPlot {
    let curves = if best_only {
        result
            .models
            .get(result.best_index)
            .into_iter()
            .map(|model| PlotCurve {
                name: model.name.clone(),
                intensity: model.fitted_curve.clone(),
            })
            .collect()
    } else {
        result
            .models
            .iter()
            .map(|model| PlotCurve {
                name: model.name.clone(),
                intensity: model.fitted_curve.clone(),
            })
            .collect()
    };
    DiagnosticPlot {
        title: "Best single glycoprotein model".into(),
        experimental: experimental.clone(),
        curves,
        experimental_pr: result.experimental.p_r.clone(),
        model_pr: result
            .models
            .get(result.best_index)
            .map(|model| model.model_pr.clone()),
        features: result
            .models
            .get(result.best_index)
            .map(|model| model.features.clone()),
        effective_sample_size: None,
    }
}

pub fn plot_for_ensemble(
    experimental: &ExperimentalCurve,
    unbiased_curve: &[f64],
    reweighted_curve: Option<&[f64]>,
    experimental_analysis: &ExperimentalAnalysis,
    features: &SaxsFeatures,
    model_pr: Option<PairDistribution>,
) -> DiagnosticPlot {
    let mut curves = vec![PlotCurve {
        name: "unbiased ensemble".into(),
        intensity: unbiased_curve.to_vec(),
    }];
    if let Some(curve) = reweighted_curve {
        curves.push(PlotCurve {
            name: "maximum-entropy reweighted".into(),
            intensity: curve.to_vec(),
        });
    }
    DiagnosticPlot {
        title: "Re-Glyco SAXS ensemble fit".into(),
        experimental: experimental.clone(),
        curves,
        experimental_pr: experimental_analysis.p_r.clone(),
        model_pr,
        features: Some(features.clone()),
        effective_sample_size: None,
    }
}

pub fn rank_candidates(
    combinations: &[CandidateCombination],
    weights: RankWeights,
) -> Result<CombinationAnalysis> {
    Ok(crabsaxs::rank_and_marginalize(combinations, weights)?)
}

fn convert_atom(atom: StructureAtom) -> Result<crabsaxs::Atom> {
    let position = atom.position;
    if ![position.x, position.y, position.z]
        .iter()
        .all(|value| value.is_finite())
    {
        return Err(SaxsError::Invalid(format!(
            "atom {} has non-finite coordinates",
            atom.name
        )));
    }
    Ok(crabsaxs::Atom {
        pos: Vector3::new(position.x, position.y, position.z),
        element: crabsaxs::Element::from_symbol(&atom.element),
        occupancy: atom.occupancy,
        is_hetatm: !is_protein_residue(&atom.residue_name),
        atom_name: atom.name,
        residue_name: atom.residue_name,
        chain_id: atom.residue.chain,
        residue_id: atom.residue.number as isize,
    })
}

fn is_protein_residue(name: &str) -> bool {
    matches!(
        name.trim().to_ascii_uppercase().as_str(),
        "ALA"
            | "ARG"
            | "ASN"
            | "ASP"
            | "CYS"
            | "GLN"
            | "GLU"
            | "GLY"
            | "HIS"
            | "ILE"
            | "LEU"
            | "LYS"
            | "MET"
            | "PHE"
            | "PRO"
            | "SER"
            | "THR"
            | "TRP"
            | "TYR"
            | "VAL"
    )
}

fn uniform_weights(count: usize) -> Vec<f64> {
    vec![1.0 / count.max(1) as f64; count]
}

fn mix_curves(curves: &[Vec<f64>], weights: &[f64]) -> Vec<f64> {
    let points = curves.first().map_or(0, Vec::len);
    (0..points)
        .map(|point| {
            curves
                .iter()
                .zip(weights)
                .map(|(curve, weight)| curve[point] * weight)
                .sum()
        })
        .collect()
}

fn weighted_coordinates(
    models: &[NamedStructure],
    weights: &[f64],
    options: PrOptions,
) -> Result<CoordinateFeatures> {
    if models.len() != weights.len() || models.is_empty() {
        return Err(SaxsError::Invalid(
            "coordinate weights do not match models".into(),
        ));
    }
    let coordinates = models
        .par_iter()
        .map(|model| coordinate_features(&model.structure, options))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let rg = coordinates
        .iter()
        .zip(weights)
        .map(|(features, weight)| features.rg * weight)
        .sum::<f64>();
    let dmax = coordinates
        .iter()
        .zip(weights)
        .map(|(features, weight)| features.dmax * weight)
        .sum::<f64>();
    let max_r = coordinates
        .iter()
        .filter_map(|features| features.p_r.r.last().copied())
        .fold(0.0, f64::max)
        .max(1.0);
    let bins = options.bins.max(16);
    let r = (0..bins)
        .map(|index| (index as f64 + 0.5) * max_r / bins as f64)
        .collect::<Vec<_>>();
    let mut p = vec![0.0; bins];
    for (features, weight) in coordinates.iter().zip(weights) {
        for (index, target) in r.iter().enumerate() {
            p[index] += weight * interpolate(&features.p_r.r, &features.p_r.p, *target);
        }
    }
    let area = r
        .windows(2)
        .zip(p.windows(2))
        .map(|(r, p)| 0.5 * (r[1] - r[0]) * (p[0] + p[1]))
        .sum::<f64>();
    if area > 0.0 {
        for value in &mut p {
            *value /= area;
        }
    }
    Ok(CoordinateFeatures {
        rg,
        dmax,
        p_r: PairDistribution {
            r,
            p,
            rg: Some(rg),
            dmax: Some(dmax),
        },
    })
}

fn interpolate(x: &[f64], y: &[f64], target: f64) -> f64 {
    if x.is_empty() || x.len() != y.len() || target < x[0] || target > *x.last().unwrap_or(&x[0]) {
        return 0.0;
    }
    let mut index = 0;
    while index + 1 < x.len() && x[index + 1] < target {
        index += 1;
    }
    if index + 1 >= x.len() {
        return y[index];
    }
    let denominator = x[index + 1] - x[index];
    if denominator <= 0.0 {
        return y[index];
    }
    let t = (target - x[index]) / denominator;
    y[index] * (1.0 - t) + y[index + 1] * t
}

pub fn write_pdb(structure: &Structure, path: impl AsRef<Path>) -> Result<()> {
    std::fs::write(path, pdb_string(structure))?;
    Ok(())
}

pub fn pdb_string(structure: &Structure) -> String {
    use std::fmt::Write;
    let mut output = String::new();
    for (index, atom) in structure.atoms.iter().enumerate() {
        let record = if atom.is_hetatm { "HETATM" } else { "ATOM  " };
        let atom_name = if atom.atom_name.len() >= 4 {
            atom.atom_name.chars().take(4).collect::<String>()
        } else {
            format!(" {:>3}", atom.atom_name)
        };
        let chain = atom.chain_id.chars().next().unwrap_or(' ');
        let residue = atom.residue_id.clamp(-999, 9999);
        let occupancy = if atom.occupancy.is_finite() {
            atom.occupancy.clamp(0.0, 1.0)
        } else {
            1.0
        };
        let _ = writeln!(
            output,
            "{record}{:>5} {atom_name:<4} {:>3} {chain}{residue:>4}    {:>8.3}{:>8.3}{:>8.3}{occupancy:>6.2}{:>6.2}          {:>2}",
            index + 1,
            atom.residue_name.chars().take(3).collect::<String>(),
            atom.pos.x,
            atom.pos.y,
            atom.pos.z,
            0.0,
            atom.element.symbol(),
        );
    }
    output.push_str("END\n");
    output
}

pub fn multi_model_pdb(models: &[NamedStructure]) -> String {
    let mut output = String::new();
    for (index, model) in models.iter().enumerate() {
        output.push_str(&format!("MODEL     {:>4}\n", index + 1));
        let model_text = pdb_string(&model.structure);
        for line in model_text.lines().filter(|line| *line != "END") {
            output.push_str(line);
            output.push('\n');
        }
        output.push_str("ENDMDL\n");
    }
    output.push_str("END\n");
    output
}

pub fn assignment(site: impl Into<String>, candidate: impl Into<String>) -> Assignment {
    Assignment {
        site: site.into(),
        candidate: candidate.into(),
    }
}

pub fn candidate_combination(
    assignments: Vec<Assignment>,
    prior: f64,
    features: SaxsFeatures,
) -> CandidateCombination {
    CandidateCombination {
        assignments,
        prior,
        features,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glysys::{BuildOptions, read_pdb_str};

    #[test]
    fn adapts_glysys_fixture_without_a_file_roundtrip() {
        let source = include_str!("../../../tests/fixtures/protein.pdb");
        let structure = read_pdb_str(source, &BuildOptions::default()).unwrap();
        let adapted = adapt_structure("fixture", &structure).unwrap();
        assert_eq!(adapted.name, "fixture");
        assert_eq!(adapted.structure.atoms.len(), structure.atoms().len());
        let first = adapted.structure.atoms.first().unwrap();
        assert!(first.pos.x.is_finite() && first.pos.y.is_finite() && first.pos.z.is_finite());
    }
}

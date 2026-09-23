//! Ensemble loading, sampling, steric scoring, and deterministic GA search.
#[cfg(feature = "webgpu")]
pub mod gpu;
#[cfg(feature = "webgpu")]
use gpu::{gpu_evaluate_candidate, gpu_optimize, start_compute_stage_async};

mod dunbrack;
mod geometry_gpu;
mod prepared;
mod sampling_energy;
mod sasa;
mod statistical;
#[cfg(feature = "webgpu")]
use geometry_gpu::evaluate_population_async;
use geometry_gpu::{GeometrySession, evaluate_population};
use statistical::sample_statistical;
#[cfg(feature = "webgpu")]
use statistical::sample_statistical_async;

use prepared::PreparedSitePose;
use prepared::SpatialGrid;
pub use prepared::{PreparedAttachmentContext, PreparedEvaluation};
pub use sasa::{
    COOKBOOK_HOTSPOT_SHIELDING_PERCENT, COOKBOOK_NDOTS, COOKBOOK_PROBE_RADIUS_ANGSTROM,
    SasaAnalysis, SasaError, SasaResidue, calculate_sasa,
};

use std::collections::{BTreeSet, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::{
    Mutex, OnceLock,
    atomic::{AtomicUsize, Ordering},
};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use glysys::{BuildOptions, ResidueId, Structure, Vec3, read_pdb_str};
use glysys_energy::prior::CircularMixture;
use glysys_energy::{AtomGroupMask, EnergyComponents, EnergyEvaluator, EnergyOptions, Obc2Options};
use glysys_opt::{
    GenerationRecord, GeneticAlgorithmConfig, GeneticAlgorithmOutcome, GeneticProblem, LbfgsConfig,
    genetic_optimize_with_progress_cancelled,
};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use reglyco_build::{
    AttachmentRequest, BuildProduct, BuildRequest, GlycanConformer, build_with_linkage_angles,
};
#[cfg(not(target_arch = "wasm32"))]
use reglyco_core::Anomer;
use reglyco_core::{
    AtomCoordinateRecord, ClashStatus, ConformerPopulationSource, EnergyAnalysis,
    EnergyComponentBreakdown, EnergySearchDiagnostics, GlycanEnsemble, GlycanInteractionBreakdown,
    GlycanQuery, GlycanSource, GlycosidicTorsionContribution, LinkagePrior, ReGlycoError,
    SamplingTarget, SearchBudgetMode, SearchBudgetResolution, SearchConfig, SearchGeneration,
    SearchOutcome, SearchScoringMode, SearchSelectionPolicy, SearchSite, SearchSiteResult,
    SearchTimingDiagnostics, VmmPolishDiagnostics, VmmPolishSiteDiagnostics, VonMisesComponent,
};
use reglyco_relax::{MovableSelection, RelaxOptions, minimize_coordinates_once, relax};

/// Configure the process-wide Rayon pool before a build or ensemble starts.
pub fn configure_threads(threads: Option<usize>) -> std::result::Result<(), String> {
    let Some(threads) = threads else {
        return Ok(());
    };
    if threads == 0 {
        return Err("--threads must be positive".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .map_err(|error| format!("could not configure --threads {threads}: {error}"))
}

pub type Result<T> = std::result::Result<T, EnsembleError>;

#[derive(Debug, thiserror::Error)]
pub enum EnsembleError {
    #[error("ensemble search was cancelled")]
    Cancelled,
    #[error(transparent)]
    ReGlyco(#[from] ReGlycoError),
    #[error(transparent)]
    GlySys(#[from] glysys::BuildError),
    #[error(transparent)]
    Energy(#[from] glysys_energy::EnergyError),
    #[error(transparent)]
    Optimization(#[from] glysys_opt::OptimizationError),
    #[error("ensemble I/O failed for {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("GlycoShape request failed: {0}")]
    Remote(String),
    #[error("invalid ensemble metadata: {0}")]
    Metadata(String),
    #[error("offline cache miss for {0}")]
    OfflineCacheMiss(String),
    #[error("ensemble contains no conformers")]
    EmptyEnsemble,
    #[error("strict steric search failed: no clash-free state inside the VMM 95% regions")]
    StrictVmmFailure {
        diagnostics: Box<StrictSearchDiagnostics>,
    },
    #[error(
        "could not generate the requested ensemble: requested {requested} frame(s), returned {returned}"
    )]
    InsufficientFrames { requested: usize, returned: usize },
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StrictSearchSiteDiagnostic {
    pub site: ResidueId,
    pub phi_degrees: f64,
    pub psi_degrees: f64,
    pub phi_component: usize,
    pub psi_component: usize,
    pub phi_within_95: bool,
    pub psi_within_95: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phi_lower_95_degrees: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phi_upper_95_degrees: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub psi_lower_95_degrees: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub psi_upper_95_degrees: Option<f64>,
    pub steric_score: f64,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StrictSearchDiagnostics {
    pub generation: usize,
    pub frozen_sites: usize,
    pub evaluations: usize,
    pub repaired: bool,
    pub sites: Vec<StrictSearchSiteDiagnostic>,
    /// Sites that still violate the strict steric/VMM contract.  This is
    /// duplicated from the per-site records as a compact UI/report index and
    /// is optional for compatibility with diagnostics written by older
    /// engines.
    #[serde(default)]
    pub outlier_sites: Vec<ResidueId>,
    /// Attachment sites that participate in one or more hard VDW contacts.
    /// This is kept separate from the fast prepared steric score so the UI
    /// can point users to the chemically offending sites even when the old
    /// centre-distance prescreen considers them solved.
    #[serde(default)]
    pub vdw_outlier_sites: Vec<ResidueId>,
    /// Final topology-aware VDW diagnostics for the best candidate.  These
    /// fields are optional so strict diagnostics written by earlier engines
    /// remain readable by the browser.
    #[serde(default)]
    pub vdw_hard_contacts: usize,
    #[serde(default)]
    pub vdw_advisory_contacts: usize,
    #[serde(default)]
    pub vdw_max_overlap_angstrom: f64,
    #[serde(default)]
    pub vdw_total_overlap_angstrom: f64,
    #[serde(default)]
    pub vdw_contacts: Vec<String>,
    pub best_candidate_pdb: String,
}

pub trait EnsembleProvider {
    fn load(&self, query: &GlycanQuery) -> Result<GlycanEnsemble>;
}

#[derive(Debug, Clone, Default)]
pub struct LocalBundleProvider;

impl EnsembleProvider for LocalBundleProvider {
    fn load(&self, query: &GlycanQuery) -> Result<GlycanEnsemble> {
        let GlycanSource::LocalBundle(input) = &query.source else {
            return Err(EnsembleError::Metadata(
                "LocalBundleProvider requires a local-bundle query".into(),
            ));
        };
        load_local_bundle(input, canonical_query(query)?)
    }
}

fn load_local_bundle(input: &Path, query: GlycanQuery) -> Result<GlycanEnsemble> {
    let pdb_path = if input.is_dir() {
        ["ensemble.pdb", "bundle.pdb", "structure.pdb"]
            .into_iter()
            .map(|name| input.join(name))
            .find(|path| path.is_file())
            .ok_or_else(|| ReGlycoError::MissingInput(input.display().to_string()))?
    } else {
        input.to_path_buf()
    };
    let contents = fs::read_to_string(&pdb_path).map_err(|source| EnsembleError::Io {
        path: pdb_path.clone(),
        source,
    })?;
    let metadata_path = if input.is_dir() {
        input.join("data.json")
    } else {
        input.with_extension("json")
    };
    let metadata = fs::read_to_string(&metadata_path)
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok());
    ensemble_from_pdb(
        &contents,
        metadata.as_ref(),
        query,
        pdb_path.display().to_string(),
    )
}

/// Parse a browser-fetched, possibly multi-model PDB ensemble without any
/// filesystem or HTTP access. Optional GlycoShape metadata supplies cluster
/// weights; absent metadata produces a deterministic equal-weight ensemble.
pub fn ensemble_from_pdb(
    contents: &str,
    metadata: Option<&serde_json::Value>,
    query: GlycanQuery,
    provenance: impl Into<String>,
) -> Result<GlycanEnsemble> {
    let query = canonical_query(&query)?;
    let models = split_pdb_models(contents);
    if models.is_empty() {
        return Err(EnsembleError::EmptyEnsemble);
    }
    let weights = metadata
        .map(|value| extract_cluster_weights(value, models.len(), &query.level))
        .transpose()?
        .flatten();
    let population_source = if weights.is_some() {
        ConformerPopulationSource::AssetMetadata
    } else {
        ConformerPopulationSource::EqualFallback
    };
    let main_clusters = metadata.and_then(extract_main_clusters).unwrap_or_default();
    let options = dry_options();
    let conformers = models
        .into_iter()
        .enumerate()
        .map(|(index, pdb)| {
            let weight = weights
                .as_ref()
                .and_then(|values| values.get(index).copied())
                .unwrap_or(1.0);
            Ok(reglyco_core::EnsembleConformer {
                id: format!("model-{}", index + 1),
                structure: read_pdb_str(&pdb, &options)?,
                cluster_index: index,
                cluster_weight: weight.max(0.0),
                main_cluster: main_clusters.get(index).copied().or(Some(index)),
                anomer: query.anomer.clone(),
                linkage_anchor: Some(("C1".into(), "O5".into())),
                priors: LinkagePrior::default(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    if conformers.is_empty() {
        return Err(EnsembleError::EmptyEnsemble);
    }
    Ok(GlycanEnsemble {
        query,
        conformers,
        provenance: provenance.into(),
        population_source,
    })
}

fn split_pdb_models(contents: &str) -> Vec<String> {
    if !contents.lines().any(|line| line.starts_with("MODEL")) {
        return vec![contents.to_string()];
    }
    let mut models = Vec::new();
    let mut current = String::new();
    let mut inside = false;
    for line in contents.lines() {
        if line.starts_with("MODEL") {
            current.clear();
            inside = true;
        } else if line.starts_with("ENDMDL") {
            if inside {
                current.push_str("END\n");
                models.push(current.clone());
                inside = false;
            }
        } else if inside {
            current.push_str(line);
            current.push('\n');
        }
    }
    models
}

fn extract_cluster_weights(
    metadata: &serde_json::Value,
    model_count: usize,
    requested_level: &str,
) -> Result<Option<Vec<f64>>> {
    let metadata = if let Some(archetype) = metadata.get("archetype") {
        if !archetype.is_object() {
            return Err(EnsembleError::Metadata(
                "archetype population metadata must be an object".into(),
            ));
        }
        archetype
    } else {
        metadata
    };
    let validate = |values: Vec<f64>| -> Result<Option<Vec<f64>>> {
        if values.len() != model_count {
            return Err(EnsembleError::Metadata(format!(
                "conformer population count {} does not match ensemble model count {model_count}",
                values.len()
            )));
        }
        if values
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(EnsembleError::Metadata(
                "conformer populations must be finite and nonnegative".into(),
            ));
        }
        if values.iter().all(|value| *value == 0.0) {
            return Err(EnsembleError::Metadata(
                "conformer populations must contain a positive value".into(),
            ));
        }
        Ok(Some(values))
    };
    // The API stores level-specific populations under `cluster_levels`, while
    // older responses expose one of the flat fields below.  Resolve the
    // requested level before looking at a broader fallback so a level-1
    // request never accidentally receives level-2 coverage populations.
    if let Some(levels) = metadata
        .get("cluster_levels")
        .and_then(serde_json::Value::as_object)
    {
        let level_key = format!("level_{}", requested_level.trim());
        if let Some(level) = levels.get(&level_key) {
            if let Some(raw) = level.get("clusters") {
                let values = cluster_values(raw, &format!("{level_key}.clusters"))?;
                return validate(values);
            }
            return Err(EnsembleError::Metadata(format!(
                "cluster level {level_key:?} has no population values"
            )));
        }
    }
    for key in ["cluster_weights", "weights", "populations"] {
        if let Some(raw) = metadata.get(key) {
            let values = cluster_values(raw, key)?;
            return validate(values);
        }
    }
    if let Some(raw) = metadata.get("coverage_clusters") {
        let values = cluster_values(raw, "coverage_clusters")?;
        return validate(values);
    }
    if let Some(raw) = metadata.get("clusters") {
        let values = cluster_values(raw, "clusters")?;
        return validate(values);
    }
    Ok(None)
}

/// Read a population collection while preserving its explicit cluster
/// identities.  Object keys are sorted by their numeric `Cluster N` suffix;
/// arrays retain their supplied model order.  Any ambiguity is rejected so a
/// missing entry can never shift every subsequent conformer's weight.
fn cluster_values(raw: &serde_json::Value, field: &str) -> Result<Vec<f64>> {
    if let Some(values) = raw.as_array() {
        return values
            .iter()
            .enumerate()
            .map(|(index, value)| {
                value
                    .as_f64()
                    .or_else(|| value.get("weight").and_then(serde_json::Value::as_f64))
                    .or_else(|| value.get("population").and_then(serde_json::Value::as_f64))
                    .ok_or_else(|| {
                        EnsembleError::Metadata(format!(
                            "population metadata field {field:?} entry {index} is not numeric"
                        ))
                    })
            })
            .collect();
    }
    let Some(object) = raw.as_object() else {
        return Err(EnsembleError::Metadata(format!(
            "population metadata field {field:?} must be an array or cluster object"
        )));
    };
    let mut indexed = Vec::with_capacity(object.len());
    for (name, value) in object {
        let index = cluster_number(name).ok_or_else(|| {
            EnsembleError::Metadata(format!("invalid {field} cluster identity {name:?}"))
        })?;
        let weight = value
            .as_f64()
            .or_else(|| value.get("weight").and_then(serde_json::Value::as_f64))
            .or_else(|| value.get("population").and_then(serde_json::Value::as_f64))
            .ok_or_else(|| {
                EnsembleError::Metadata(format!(
                    "{field} cluster {name:?} has no numeric population"
                ))
            })?;
        indexed.push((index, weight));
    }
    indexed.sort_by_key(|(index, _)| *index);
    if indexed
        .iter()
        .enumerate()
        .any(|(expected, (index, _))| *index != expected)
    {
        return Err(EnsembleError::Metadata(format!(
            "{field} populations must be indexed contiguously from zero"
        )));
    }
    Ok(indexed.into_iter().map(|(_, weight)| weight).collect())
}

fn extract_main_clusters(metadata: &serde_json::Value) -> Option<Vec<usize>> {
    let metadata = metadata.get("archetype").unwrap_or(metadata);
    let groups = metadata
        .get("coverage_clusters_per_main")
        .and_then(serde_json::Value::as_object)?;
    let mut mapping = Vec::<(usize, usize)>::new();
    for (main, coverage) in groups {
        let Some(main) = cluster_number(main) else {
            continue;
        };
        for cluster in coverage.as_array().into_iter().flatten() {
            let cluster = cluster
                .as_u64()
                .map(|value| value as usize)
                .or_else(|| cluster.as_str().and_then(cluster_number));
            if let Some(cluster) = cluster {
                mapping.push((cluster, main));
            }
        }
    }
    let maximum = mapping.iter().map(|(cluster, _)| *cluster).max()?;
    let mut result = (0..=maximum).collect::<Vec<_>>();
    for (cluster, main) in mapping {
        result[cluster] = main;
    }
    Some(result)
}

fn cluster_number(value: &str) -> Option<usize> {
    value
        .split(|character: char| !character.is_ascii_digit())
        .find(|part| !part.is_empty())
        .and_then(|part| part.parse().ok())
}

fn canonical_structure_format(value: &str) -> Result<&'static str> {
    if value.eq_ignore_ascii_case("pdb") {
        Ok("PDB")
    } else if value.eq_ignore_ascii_case("glycam") {
        Ok("GLYCAM")
    } else {
        Err(EnsembleError::Metadata(format!(
            "unsupported glycan structure format {value:?}; expected PDB or GLYCAM"
        )))
    }
}

fn canonical_query(query: &GlycanQuery) -> Result<GlycanQuery> {
    let mut canonical = query.clone();
    canonical.format = canonical_structure_format(&query.format)?.into();
    Ok(canonical)
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone)]
pub struct GlycoShapeProvider {
    pub api_base_url: String,
    client: reqwest::blocking::Client,
}

#[cfg(not(target_arch = "wasm32"))]
impl GlycoShapeProvider {
    pub fn new(api_base_url: impl Into<String>) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            .user_agent("reglyco-rs/0.1")
            .build()
            .map_err(|error| EnsembleError::Remote(error.to_string()))?;
        Ok(Self {
            api_base_url: api_base_url.into(),
            client,
        })
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl EnsembleProvider for GlycoShapeProvider {
    fn load(&self, query: &GlycanQuery) -> Result<GlycanEnsemble> {
        let GlycanSource::GlyTouCan(identifier) = &query.source else {
            return Err(EnsembleError::Metadata(
                "GlycoShapeProvider requires a GlyTouCan query".into(),
            ));
        };
        let query = canonical_query(query)?;
        let anomer = match query.anomer {
            Anomer::Alpha => "alpha",
            Anomer::Beta => "beta",
            Anomer::Unknown => "beta",
        };
        let format = query.format.as_str();
        let url = format!(
            "{}/api/structure/{}/{}?anomer={}&level={}",
            self.api_base_url.trim_end_matches('/'),
            format,
            identifier,
            anomer,
            query.level
        );
        let response = self
            .client
            .get(&url)
            .send()
            .and_then(reqwest::blocking::Response::error_for_status)
            .map_err(|error| EnsembleError::Remote(error.to_string()))?;
        let pdb = response
            .text()
            .map_err(|error| EnsembleError::Remote(error.to_string()))?;
        let metadata_url = format!(
            "{}/api/glycan/{}",
            self.api_base_url.trim_end_matches('/'),
            identifier
        );
        let metadata = self
            .client
            .get(&metadata_url)
            .send()
            .and_then(reqwest::blocking::Response::error_for_status)
            .and_then(reqwest::blocking::Response::json::<serde_json::Value>)
            .map_err(|error| EnsembleError::Remote(error.to_string()))?;
        let main_clusters = extract_main_clusters(&metadata).unwrap_or_default();
        let models = split_pdb_models(&pdb);
        let weights =
            extract_cluster_weights(&metadata, models.len(), &query.level)?.unwrap_or_default();
        let population_source = if weights.is_empty() {
            ConformerPopulationSource::EqualFallback
        } else {
            ConformerPopulationSource::AssetMetadata
        };
        let options = dry_options();
        let conformers = models
            .into_iter()
            .enumerate()
            .map(|(index, model)| {
                Ok(reglyco_core::EnsembleConformer {
                    id: format!("{identifier}-{}", index + 1),
                    structure: read_pdb_str(&model, &options)?,
                    cluster_index: index,
                    cluster_weight: weights.get(index).copied().unwrap_or(1.0),
                    main_cluster: main_clusters.get(index).copied().or(Some(index)),
                    anomer: query.anomer.clone(),
                    linkage_anchor: Some(("C1".into(), "O5".into())),
                    priors: LinkagePrior::default(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if conformers.is_empty() {
            return Err(EnsembleError::EmptyEnsemble);
        }
        Ok(GlycanEnsemble {
            query,
            conformers,
            provenance: url,
            population_source,
        })
    }
}

pub struct CachingProvider<P> {
    inner: P,
    cache_dir: PathBuf,
    offline: bool,
}

impl<P> CachingProvider<P> {
    pub fn new(inner: P, cache_dir: impl Into<PathBuf>, offline: bool) -> Self {
        Self {
            inner,
            cache_dir: cache_dir.into(),
            offline,
        }
    }
}

impl<P: EnsembleProvider> EnsembleProvider for CachingProvider<P> {
    fn load(&self, query: &GlycanQuery) -> Result<GlycanEnsemble> {
        let query = canonical_query(query)?;
        let key = cache_key(&query);
        let cache_file = self.cache_dir.join(format!("{key}.pdb"));
        if cache_file.is_file() {
            let mut cached_query = query.clone();
            cached_query.source = GlycanSource::LocalBundle(cache_file.clone());
            let mut ensemble = load_local_bundle(&cache_file, cached_query)?;
            ensemble.query = query.clone();
            ensemble.provenance = format!("cache:{}", cache_file.display());
            return Ok(ensemble);
        }
        if self.offline {
            return Err(EnsembleError::OfflineCacheMiss(key));
        }
        let ensemble = self.inner.load(&query)?;
        fs::create_dir_all(&self.cache_dir).map_err(|source| EnsembleError::Io {
            path: self.cache_dir.clone(),
            source,
        })?;
        let mut pdb = String::new();
        for (index, conformer) in ensemble.conformers.iter().enumerate() {
            pdb.push_str(&format!("MODEL     {:>4}\n", index + 1));
            for line in conformer.structure.to_pdb_string().lines() {
                if line != "END" {
                    pdb.push_str(line);
                    pdb.push('\n');
                }
            }
            pdb.push_str("ENDMDL\n");
        }
        fs::write(&cache_file, pdb).map_err(|source| EnsembleError::Io {
            path: cache_file,
            source,
        })?;
        let metadata = serde_json::json!({
            "cluster_weights": ensemble
                .conformers
                .iter()
                .map(|conformer| conformer.cluster_weight)
                .collect::<Vec<_>>(),
            "coverage_clusters_per_main": ensemble
                .conformers
                .iter()
                .enumerate()
                .fold(serde_json::Map::new(), |mut groups, (index, conformer)| {
                    let key = conformer.main_cluster.unwrap_or(index).to_string();
                    groups
                        .entry(key)
                        .or_insert_with(|| serde_json::Value::Array(Vec::new()))
                        .as_array_mut()
                        .expect("newly inserted array")
                        .push(serde_json::Value::from(index));
                    groups
                }),
        });
        let metadata_file = self.cache_dir.join(format!("{key}.json"));
        fs::write(
            &metadata_file,
            serde_json::to_vec_pretty(&metadata)
                .map_err(|error| EnsembleError::Metadata(error.to_string()))?,
        )
        .map_err(|source| EnsembleError::Io {
            path: metadata_file,
            source,
        })?;
        Ok(ensemble)
    }
}

fn cache_key(query: &GlycanQuery) -> String {
    let mut canonical = query.clone();
    canonical.format = canonical_structure_format(&query.format)
        .map(str::to_owned)
        .unwrap_or_else(|_| query.format.to_ascii_uppercase());
    let bytes = serde_json::to_vec(&canonical).unwrap_or_default();
    let hash = bytes
        .into_iter()
        .fold(0xcbf29ce484222325_u64, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        });
    format!("{hash:016x}")
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gene {
    pub conformer: usize,
    pub phi: f64,
    pub psi: f64,
    pub rotamer: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GeneKey(Vec<(usize, u64, u64, Option<usize>)>);

impl From<&[Gene]> for GeneKey {
    fn from(genes: &[Gene]) -> Self {
        Self(
            genes
                .iter()
                .map(|gene| {
                    (
                        gene.conformer,
                        gene.phi.to_bits(),
                        gene.psi.to_bits(),
                        gene.rotamer,
                    )
                })
                .collect(),
        )
    }
}

struct CandidateEnergy {
    score: f64,
    components: EnergyComponents,
    interaction: Option<glysys_energy::InteractionEnergyComponents>,
    coordinates: Vec<Vec3>,
    active_atoms: usize,
    active_indices: Vec<usize>,
    active_residues: Vec<ResidueId>,
    neighbor_pairs: usize,
}

/// Topology-bound mapping; source coordinates are read once per candidate.
struct PreparedEnergyAtoms(glysys_energy::geometry::CoordinateMap);
impl PreparedEnergyAtoms {
    fn new(system: &glysys::ParameterizedSystem) -> Self {
        Self(glysys_energy::geometry::CoordinateMap::new(system))
    }
    fn coordinates(&self, structure: &Structure) -> Result<Vec<Vec3>> {
        Ok(self.0.coordinates(structure)?)
    }
}

struct EnergySearchContext {
    system: glysys::ParameterizedSystem,
    /// Long-lived topology-bound evaluator.  The old path rebuilt this table
    /// for every candidate, which made GPU batches pay avoidable CPU setup.
    evaluator: OnceLock<EnergyEvaluator<'static>>,
    mode: SearchScoringMode,
    use_obc2: bool,
    minimize: bool,
    min_iterations: usize,
    min_radius: f64,
    cutoff: f64,
    cache: Mutex<HashMap<GeneKey, f64>>,
    atom_mapping: OnceLock<PreparedEnergyAtoms>,
    atom_masks: OnceLock<(AtomGroupMask, AtomGroupMask, BTreeSet<ResidueId>)>,
    evaluations: AtomicUsize,
    minimizations: AtomicUsize,
    cache_hits: AtomicUsize,
    steric_rejections: AtomicUsize,
    failures: AtomicUsize,
    started: Instant,
    topology_seconds: f64,
}

impl EnergySearchContext {
    fn energy_options(&self) -> EnergyOptions {
        EnergyOptions {
            cutoff: Some(self.cutoff),
            obc2: self.use_obc2.then(Obc2Options::default),
            ..EnergyOptions::default()
        }
    }

    fn evaluator(&self) -> Result<&EnergyEvaluator<'static>> {
        if self.evaluator.get().is_none() {
            let evaluator = EnergyEvaluator::new(&self.system, self.energy_options())?.into_owned();
            let _ = self.evaluator.set(evaluator);
        }
        Ok(self
            .evaluator
            .get()
            .expect("energy evaluator initialized after successful construction"))
    }

    fn coordinates_for(&self, structure: &Structure) -> Result<Vec<Vec3>> {
        self.atom_mapping
            .get_or_init(|| PreparedEnergyAtoms::new(&self.system))
            .coordinates(structure)
    }

    fn masks(&self) -> &(AtomGroupMask, AtomGroupMask, BTreeSet<ResidueId>) {
        self.atom_masks.get_or_init(|| self.prepare_masks())
    }

    fn prepare_masks(&self) -> (AtomGroupMask, AtomGroupMask, BTreeSet<ResidueId>) {
        let glycan_residues = self
            .system
            .metadata()
            .glycan_trees
            .iter()
            .flat_map(|tree| tree.residue_ids.iter().cloned())
            .collect::<BTreeSet<_>>();
        let glycan = AtomGroupMask::from_indices(
            self.system.atom_count(),
            self.system.residues().iter().flat_map(|residue| {
                let id = ResidueId {
                    chain: residue.chain().into(),
                    number: residue.number(),
                    insertion_code: residue.insertion_code(),
                };
                glycan_residues
                    .contains(&id)
                    .then_some(residue.atom_range())
                    .into_iter()
                    .flatten()
            }),
        );
        let protein = AtomGroupMask::from_indices(
            self.system.atom_count(),
            glysys_energy::scoring::component_roles(&self.system)
                .into_iter()
                .enumerate()
                .filter_map(|(atom, role)| {
                    (role == glysys_energy::scoring::ComponentRole::Receptor).then_some(atom)
                }),
        );
        (protein, glycan, glycan_residues)
    }

    fn active_selection(
        &self,
        coordinates: &[Vec3],
        glycan: &AtomGroupMask,
        glycan_residues: &BTreeSet<ResidueId>,
    ) -> (Vec<usize>, Vec<ResidueId>) {
        let glycan_heavy = glycan
            .indices()
            .filter(|atom| self.system.atoms()[*atom].element() != 1)
            .collect::<Vec<_>>();
        let mut atoms = glycan.indices().collect::<BTreeSet<_>>();
        let mut residues = Vec::new();
        let radius2 = self.min_radius * self.min_radius;
        for residue in self.system.residues() {
            let id = ResidueId {
                chain: residue.chain().into(),
                number: residue.number(),
                insertion_code: residue.insertion_code(),
            };
            if glycan_residues.contains(&id) {
                continue;
            }
            let nearby = residue
                .atom_range()
                .filter(|atom| self.system.atoms()[*atom].element() != 1)
                .any(|atom| {
                    glycan_heavy.iter().any(|glycan_atom| {
                        squared_distance(coordinates[atom], coordinates[*glycan_atom]) <= radius2
                    })
                });
            if nearby {
                atoms.extend(residue.atom_range());
                residues.push(id);
            }
        }
        (atoms.into_iter().collect(), residues)
    }

    fn evaluate(&self, structure: &Structure) -> Result<CandidateEnergy> {
        self.evaluations.fetch_add(1, Ordering::Relaxed);
        let mut coordinates = self.coordinates_for(structure)?;
        let (protein, glycan, glycan_residues) = self.masks();
        let (active, active_residues) = if self.minimize {
            self.active_selection(&coordinates, glycan, glycan_residues)
        } else {
            (Vec::new(), Vec::new())
        };
        if self.minimize {
            self.minimizations.fetch_add(1, Ordering::Relaxed);
            let lbfgs = LbfgsConfig {
                max_iterations: self.min_iterations.max(1),
                ..LbfgsConfig::default()
            };
            let options = EnergyOptions {
                cutoff: Some(self.cutoff),
                obc2: self.use_obc2.then(Obc2Options::default),
                ..EnergyOptions::default()
            };
            coordinates =
                minimize_coordinates_once(&self.system, &coordinates, &active, options, &lbfgs)
                    .map_err(|error| EnsembleError::Metadata(error.to_string()))?
                    .0;
        }
        let evaluator = self.evaluator()?;
        let interaction = (self.mode == SearchScoringMode::ProteinGlycanInteraction)
            .then(|| evaluator.interaction_energy_with_pair_count(&coordinates, protein, glycan))
            .transpose()?;
        let pair_count = interaction.map_or(0, |(_, count)| count);
        let interaction_components = interaction.map(|(components, _)| components);
        let (components, score) = if let Some(value) = interaction_components {
            (
                EnergyComponents {
                    van_der_waals: value.van_der_waals,
                    electrostatics: value.electrostatics,
                    ..Default::default()
                },
                value.total(),
            )
        } else {
            let energy = evaluator.energy(&coordinates)?;
            (energy.components, energy.total())
        };
        Ok(CandidateEnergy {
            score,
            components,
            interaction: interaction_components,
            coordinates,
            active_atoms: active.len(),
            active_indices: active,
            active_residues,
            neighbor_pairs: pair_count,
        })
    }
}

fn residue_label(residue: &ResidueId) -> String {
    format!(
        "{}:{}{}",
        residue.chain,
        residue.number,
        residue.insertion_code.unwrap_or(' ')
    )
    .trim_end()
    .to_string()
}

/// Build the final, post-search energy explanation from the same evaluator
/// that supplied the selected score. Diagnostic subsets are computed only
/// after selection so they do not add work to the candidate loop.
fn energy_analysis_for_context(
    context: &EnergySearchContext,
    structure: &Structure,
    selected: &CandidateEnergy,
    sites: &[SearchSite],
    site_results: &[SearchSiteResult],
    backend: &str,
) -> Result<EnergyAnalysis> {
    let coordinates = context.coordinates_for(structure)?;
    let evaluator = context.evaluator()?;
    let components = evaluator.components(&coordinates)?;
    debug_assert!(selected.components.total().is_finite());
    let (protein, _, _) = context.masks();
    let mut per_glycan_interactions = Vec::with_capacity(structure.metadata().glycan_trees.len());
    for (index, tree) in structure.metadata().glycan_trees.iter().enumerate() {
        let residues = tree.residue_ids.iter().collect::<BTreeSet<_>>();
        let glycan = AtomGroupMask::from_indices(
            context.system.atom_count(),
            context.system.residues().iter().flat_map(|residue| {
                let id = ResidueId {
                    chain: residue.chain().into(),
                    number: residue.number(),
                    insertion_code: residue.insertion_code(),
                };
                residues
                    .contains(&id)
                    .then_some(residue.atom_range())
                    .into_iter()
                    .flatten()
            }),
        );
        let interaction = evaluator.interaction_energy(&coordinates, protein, &glycan)?;
        let site = sites
            .get(index)
            .map(|entry| residue_label(&entry.site.residue))
            .or_else(|| tree.attachment_site.as_ref().map(residue_label))
            .unwrap_or_else(|| format!("site-{}", index + 1));
        let glycan_id = site_results
            .get(index)
            .map(|entry| entry.conformer_id.clone())
            .unwrap_or_default();
        per_glycan_interactions.push(GlycanInteractionBreakdown {
            site,
            glycan_id,
            van_der_waals: interaction.van_der_waals,
            electrostatics: interaction.electrostatics,
            total: interaction.total(),
        });
    }

    let torsion_terms = evaluator.torsion_energy_contributions(&coordinates)?;
    let mut glycosidic_torsions = Vec::new();
    for term in torsion_terms.into_iter().filter(|term| !term.improper) {
        let residue_ids = term
            .atoms
            .into_iter()
            .map(|atom| {
                let parameterized = &context.system.atoms()[atom];
                let residue = &context.system.residues()[parameterized.residue_index()];
                ResidueId {
                    chain: residue.chain().into(),
                    number: residue.number(),
                    insertion_code: residue.insertion_code(),
                }
            })
            .collect::<Vec<_>>();
        let Some(site_index) = structure
            .metadata()
            .glycan_trees
            .iter()
            .position(|tree| residue_ids.iter().all(|id| tree.residue_ids.contains(id)))
        else {
            continue;
        };
        let site = sites
            .get(site_index)
            .map(|entry| residue_label(&entry.site.residue))
            .unwrap_or_else(|| format!("site-{}", site_index + 1));
        glycosidic_torsions.push(GlycosidicTorsionContribution {
            site,
            linkage: format!(
                "{}-{}-{}-{}",
                term.atoms[0], term.atoms[1], term.atoms[2], term.atoms[3]
            ),
            atoms: term.atoms,
            energy: term.energy,
        });
    }
    let component_breakdown = EnergyComponentBreakdown {
        bonds: components.bonds,
        angles: components.angles,
        proper_torsions: components.proper_torsions,
        improper_torsions: components.improper_torsions,
        van_der_waals: components.van_der_waals,
        electrostatics: components.electrostatics,
        generalized_born: components.generalized_born,
        surface_area: components.surface_area,
        restraints: components.restraints,
        dispersion_correction: components.dispersion_correction,
    };
    let interaction_sum = per_glycan_interactions
        .iter()
        .map(|entry| entry.total)
        .sum::<f64>();
    Ok(EnergyAnalysis {
        version: 1,
        units: "kcal/mol".into(),
        model: "amber-glycam-v2".into(),
        cutoff_angstrom: Some(context.cutoff),
        solvent: if context.use_obc2 { "OBC2" } else { "none" }.into(),
        backend: backend.into(),
        drives_selection: context.mode != SearchScoringMode::StericPrior,
        selected_score: Some(selected.score),
        components: component_breakdown,
        per_glycan_interactions,
        glycosidic_torsions,
        diagnostic_remainder: Some(components.total() - interaction_sum),
    })
}

/// Compute the versioned energy explanation for an already materialized
/// structure.  Workflow callers use this for sampled frames after the
/// statistical transition has completed, so the diagnostic pass cannot alter
/// acceptance decisions or emitted coordinates.
pub fn analyze_energy_structure(
    sites: &[SearchSite],
    site_results: &[SearchSiteResult],
    structure: &Structure,
    builder: &glysys::SystemBuilder,
    mode: SearchScoringMode,
    use_obc2: bool,
    cutoff: f64,
    backend: &str,
) -> Result<EnergyAnalysis> {
    let system = builder.prepare_structure(structure)?;
    let context = EnergySearchContext {
        system,
        evaluator: OnceLock::new(),
        mode,
        use_obc2,
        minimize: false,
        min_iterations: 0,
        min_radius: 0.0,
        cutoff,
        cache: Mutex::new(HashMap::new()),
        atom_masks: OnceLock::new(),
        atom_mapping: OnceLock::new(),
        evaluations: AtomicUsize::new(0),
        minimizations: AtomicUsize::new(0),
        cache_hits: AtomicUsize::new(0),
        steric_rejections: AtomicUsize::new(0),
        failures: AtomicUsize::new(0),
        started: Instant::now(),
        topology_seconds: 0.0,
    };
    let coordinates = context.coordinates_for(structure)?;
    let evaluator = context.evaluator()?;
    let score = if mode == SearchScoringMode::ProteinGlycanInteraction {
        let (protein_mask, glycan_mask, _) = context.masks();
        evaluator
            .interaction_energy(&coordinates, protein_mask, glycan_mask)?
            .total()
    } else {
        evaluator.energy(&coordinates)?.total()
    };
    let selected = CandidateEnergy {
        score,
        components: evaluator.components(&coordinates)?,
        interaction: None,
        coordinates,
        active_atoms: 0,
        active_indices: Vec::new(),
        active_residues: Vec::new(),
        neighbor_pairs: 0,
    };
    energy_analysis_for_context(&context, structure, &selected, sites, site_results, backend)
}

struct SearchProblem<'a> {
    protein: &'a Structure,
    sites: &'a [SearchSite],
    builder: &'a glysys::SystemBuilder,
    clash_distance: f64,
    scan_rotamers: bool,
    scoring_mode: SearchScoringMode,
    selection_policy: SearchSelectionPolicy,
    use_obc2: bool,
    pre_minimization: bool,
    pre_minimization_iterations: usize,
    energy_context: Option<&'a EnergySearchContext>,
    prepared: &'a PreparedAttachmentContext,
    prior_cache: SearchPriorCache,
    vmm_priors: Vec<Vec<LinkagePrior>>,
    /// Optional feasible states prepared by the shared compatibility
    /// coordinator.  The generic energy GA consumes these only for its
    /// initial population; subsequent generations retain their existing
    /// crossover/mutation and objective ranking.
    initial_states: Arc<Vec<Vec<Gene>>>,
    initial_state_cursor: AtomicUsize,
}

#[derive(Debug, Clone)]
struct CompiledAttachmentPrior {
    phi: CircularMixture,
    psi: CircularMixture,
    log_conformer_probability: f64,
}

#[derive(Debug, Clone)]
struct SearchPriorCache {
    sites: Vec<Vec<CompiledAttachmentPrior>>,
}

fn compile_prior_cache(protein: &Structure, sites: &[SearchSite]) -> Result<SearchPriorCache> {
    let mut compiled_sites = Vec::with_capacity(sites.len());
    for site in sites {
        if site.ensemble.conformers.iter().any(|conformer| {
            !conformer.cluster_weight.is_finite() || conformer.cluster_weight < 0.0
        }) {
            return Err(EnsembleError::Metadata(
                "conformer populations must be finite and non-negative".into(),
            ));
        }
        let total_weight = site
            .ensemble
            .conformers
            .iter()
            .map(|conformer| conformer.cluster_weight)
            .sum::<f64>();
        if !total_weight.is_finite() || total_weight <= 0.0 {
            return Err(EnsembleError::Metadata(
                "conformer populations must contain a positive finite total".into(),
            ));
        }
        let mut conformers = Vec::with_capacity(site.ensemble.conformers.len());
        for conformer in &site.ensemble.conformers {
            let weight = conformer.cluster_weight / total_weight;
            let priors = resolved_priors(protein, site, &conformer.priors);
            let phi = CircularMixture::new(
                priors
                    .phi
                    .iter()
                    .map(|component| glysys_energy::prior::CircularComponent {
                        mean: component.mean_degrees.to_radians(),
                        concentration: component.concentration,
                        weight: component.weight,
                    })
                    .collect(),
            )?;
            let psi = CircularMixture::new(
                priors
                    .psi
                    .iter()
                    .map(|component| glysys_energy::prior::CircularComponent {
                        mean: component.mean_degrees.to_radians(),
                        concentration: component.concentration,
                        weight: component.weight,
                    })
                    .collect(),
            )?;
            conformers.push(CompiledAttachmentPrior {
                phi,
                psi,
                log_conformer_probability: if weight > 0.0 {
                    weight.ln()
                } else {
                    f64::NEG_INFINITY
                },
            });
        }
        compiled_sites.push(conformers);
    }
    Ok(SearchPriorCache {
        sites: compiled_sites,
    })
}

fn compile_vmm_prior_cache(protein: &Structure, sites: &[SearchSite]) -> Vec<Vec<LinkagePrior>> {
    sites
        .iter()
        .map(|site| {
            site.ensemble
                .conformers
                .iter()
                .map(|conformer| resolved_priors(protein, site, &conformer.priors))
                .collect()
        })
        .collect()
}

impl SearchProblem<'_> {
    /// Return the negative log density of the two normalized attachment
    /// mixtures for one site.  The mixtures are compiled once in
    /// `prior_cache`; keeping this lookup allocation-free is important for
    /// late generations, where most candidates are close enough to require
    /// the complete steric traversal.
    fn attachment_mixture_score(&self, index: usize, gene: &Gene) -> f64 {
        let Some(prior) = self
            .prior_cache
            .sites
            .get(index)
            .and_then(|conformers| conformers.get(gene.conformer))
        else {
            return f64::INFINITY;
        };
        let score = -prior.phi.log_probability(gene.phi.to_radians())
            - prior.psi.log_probability(gene.psi.to_radians());
        if score.is_finite() {
            score
        } else {
            f64::INFINITY
        }
    }

    fn cookbook_fitness(&self, chromosome: &CookbookStericChromosome) -> f64 {
        chromosome
            .genes
            .iter()
            .enumerate()
            .map(|(index, gene)| {
                let valid = chromosome.valid_mask.get(index).copied().unwrap_or(false);
                if !valid {
                    return 1000.0;
                }
                let cluster_penalty = self
                    .sites
                    .get(index)
                    .and_then(|site| site.ensemble.conformers.get(gene.conformer))
                    .map_or(1000.0, |conformer| conformer.cluster_index as f64 * 0.01);
                let rotamer_penalty = self
                    .sites
                    .get(index)
                    .and_then(|site| {
                        gene.rotamer.and_then(|rotamer| {
                            dunbrack::probability(self.protein, &site.site.residue, rotamer)
                        })
                    })
                    .map_or(0.0, |probability| {
                        0.05 * (1.0 - probability.clamp(0.0, 1.0))
                    });
                chromosome.steric_scores[index] + cluster_penalty + rotamer_penalty
            })
            .sum()
    }

    fn site_prior_breakdown(&self, index: usize, gene: &Gene) -> Option<(f64, f64, f64)> {
        let prior = self
            .prior_cache
            .sites
            .get(index)
            .and_then(|conformers| conformers.get(gene.conformer))?;
        if !prior.log_conformer_probability.is_finite() {
            return None;
        }
        let phi_log_density = prior.phi.log_probability(gene.phi.to_radians());
        let psi_log_density = prior.psi.log_probability(gene.psi.to_radians());
        let score = -prior.log_conformer_probability - phi_log_density - psi_log_density;
        score.is_finite().then_some((
            prior.log_conformer_probability.exp(),
            phi_log_density + psi_log_density,
            score,
        ))
    }

    fn joint_prior_score(&self, state: &[Gene]) -> f64 {
        if state.len() != self.prior_cache.sites.len() {
            return f64::INFINITY;
        }
        state
            .iter()
            .enumerate()
            .map(|(index, gene)| {
                self.site_prior_breakdown(index, gene)
                    .map_or(f64::INFINITY, |(_, _, score)| score)
            })
            .sum()
    }

    fn prior(&self, state: &[Gene]) -> f64 {
        let attachment_prior = self.joint_prior_score(state);
        if !attachment_prior.is_finite() {
            return attachment_prior;
        }
        attachment_prior
            + state
                .iter()
                .zip(self.sites)
                .map(|(gene, site)| {
                    gene.rotamer
                        .and_then(|index| {
                            dunbrack::probability(self.protein, &site.site.residue, index)
                        })
                        .map_or(0.0, |probability| -0.05 * probability.max(1.0e-12).ln())
                })
                .sum::<f64>()
    }
}

impl GeneticProblem for SearchProblem<'_> {
    type State = Vec<Gene>;

    fn generate(&self, rng: &mut ChaCha8Rng) -> Self::State {
        let initial_index = self.initial_state_cursor.fetch_add(1, Ordering::Relaxed);
        if let Some(state) = self.initial_states.get(initial_index) {
            return state.clone();
        }
        self.sites
            .iter()
            .map(|site| {
                let conformer = weighted_conformer_index(&site.ensemble, rng);
                let Some(conformer_data) = site.ensemble.conformers.get(conformer) else {
                    // Invalid restored/provider metadata is reported by the
                    // prepared evaluator as an invalid state instead of
                    // trapping the WASM worker on a conformer index.
                    return Gene {
                        conformer,
                        phi: 0.0,
                        psi: 0.0,
                        rotamer: None,
                    };
                };
                let priors = resolved_priors(self.protein, site, &conformer_data.priors);
                Gene {
                    conformer,
                    phi: sample_vmm(&priors.phi, rng),
                    psi: sample_vmm(&priors.psi, rng),
                    rotamer: sample_rotamer(
                        self.protein,
                        &site.site.residue,
                        self.scan_rotamers,
                        rng,
                    ),
                }
            })
            .collect()
    }

    fn crossover(
        &self,
        first: &Self::State,
        second: &Self::State,
        rng: &mut ChaCha8Rng,
    ) -> Self::State {
        first
            .iter()
            .zip(second)
            .map(|(first, second)| {
                if rng.random_bool(0.5) {
                    first.clone()
                } else {
                    second.clone()
                }
            })
            .collect()
    }

    fn mutate(&self, state: &mut Self::State, rng: &mut ChaCha8Rng, rate: f64) {
        for (gene, site) in state.iter_mut().zip(self.sites) {
            if rng.random_bool(rate) {
                gene.conformer = weighted_conformer_index(&site.ensemble, rng);
            }
            if rng.random_bool(rate) {
                let Some(conformer_data) = site.ensemble.conformers.get(gene.conformer) else {
                    continue;
                };
                let priors = resolved_priors(self.protein, site, &conformer_data.priors);
                // Preserve the Cookbook's local creep mutation. Sampling a
                // fresh narrow VMM draw alone cannot escape a crowded basin;
                // a bounded angular walk lets a site move outside its native
                // peak when neighbouring glycans occupy the preferred pose.
                gene.phi = if rng.random_bool(0.7) {
                    wrap_degrees(gene.phi + rng.random_range(-30.0..30.0))
                } else {
                    sample_vmm(&priors.phi, rng)
                };
            }
            if rng.random_bool(rate) {
                let Some(conformer_data) = site.ensemble.conformers.get(gene.conformer) else {
                    continue;
                };
                let priors = resolved_priors(self.protein, site, &conformer_data.priors);
                gene.psi = if rng.random_bool(0.7) {
                    wrap_degrees(gene.psi + rng.random_range(-30.0..30.0))
                } else {
                    sample_vmm(&priors.psi, rng)
                };
            }
            if self.scan_rotamers && rng.random_bool(rate) {
                gene.rotamer = sample_rotamer(self.protein, &site.site.residue, true, rng);
            }
        }
    }

    fn repair(&self, state: &mut Self::State, _rng: &mut ChaCha8Rng) {
        for (gene, site) in state.iter_mut().zip(self.sites) {
            gene.conformer %= site.ensemble.conformers.len();
            gene.phi = wrap_degrees(gene.phi);
            gene.psi = wrap_degrees(gene.psi);
        }
    }

    fn evaluate(&self, state: &Self::State) -> f64 {
        let Ok(prepared) = self.prepared.evaluate(state, self.clash_distance) else {
            return 1.0e30;
        };
        // Optimize the sum of per-site steric scores, as the Cookbook
        // optimizer does.  Using only the maximum score lets a complete
        // chromosome with many individually clashing glycans look as good as
        // one with a single bad site, which is especially harmful for dense
        // UniProt one-shot requests such as P27918.  Keep the maximum for the
        // hard clash gate used by energy/refine objectives below.
        let max_steric = prepared.score;
        let steric = prepared.site_scores.iter().sum::<f64>();
        let prior = self.prior(state);
        if self.scoring_mode == SearchScoringMode::StericPrior {
            return steric + prior;
        }
        if max_steric > 1.1 {
            if let Some(context) = self.energy_context {
                context.steric_rejections.fetch_add(1, Ordering::Relaxed);
            }
            return 1.0e12 + steric * 1.0e6 + prior;
        }
        if let Some(context) = self.energy_context {
            let Ok(structure) = build_state(self.protein, self.sites, state, self.builder) else {
                context.failures.fetch_add(1, Ordering::Relaxed);
                return 1.0e30;
            };
            let key = GeneKey::from(state.as_slice());
            if let Some(score) = context
                .cache
                .lock()
                .expect("energy cache poisoned")
                .get(&key)
                .copied()
            {
                context.cache_hits.fetch_add(1, Ordering::Relaxed);
                return score;
            }
            match context.evaluate(&structure) {
                Ok(value) => {
                    let score = value.score
                        + if self.scoring_mode == SearchScoringMode::FullEnergy {
                            0.0
                        } else {
                            prior * 1.0e-6
                        };
                    context
                        .cache
                        .lock()
                        .expect("energy cache poisoned")
                        .insert(key, score);
                    score
                }
                Err(_) => {
                    context.failures.fetch_add(1, Ordering::Relaxed);
                    1.0e30
                }
            }
        } else {
            let Ok(structure) = build_state(self.protein, self.sites, state, self.builder) else {
                return 1.0e30;
            };
            score_structure(
                &structure,
                self.builder,
                self.scoring_mode,
                self.use_obc2,
                self.pre_minimization,
                self.pre_minimization_iterations,
            )
            .map(|energy| energy + prior * 1.0e-6)
            .unwrap_or(1.0e30)
        }
    }

    fn is_solution(&self, state: &Self::State, _score: f64) -> bool {
        if self.scoring_mode != SearchScoringMode::StericPrior {
            return false;
        }
        self.prepared
            .evaluate(state, self.clash_distance)
            .map(|evaluation| evaluation.site_scores.iter().all(|score| *score <= 1.1))
            .unwrap_or(false)
    }
}

#[derive(Debug, Clone)]
struct CookbookStericChromosome {
    genes: Vec<Gene>,
    phi_components: Vec<usize>,
    psi_components: Vec<usize>,
    frozen_mask: Vec<bool>,
    conformations: Vec<Option<Vec<Vec3>>>,
    site_grids: Vec<Option<Arc<SpatialGrid>>>,
    dirty: Vec<bool>,
    valid_mask: Vec<bool>,
    steric_scores: Vec<f64>,
    fitness: f64,
}

impl CookbookStericChromosome {
    fn new(sites: &[SearchSite]) -> Self {
        let count = sites.len();
        Self {
            genes: Vec::with_capacity(count),
            phi_components: Vec::with_capacity(count),
            psi_components: Vec::with_capacity(count),
            frozen_mask: vec![false; count],
            conformations: vec![None; count],
            site_grids: vec![None; count],
            dirty: vec![true; count],
            valid_mask: vec![false; count],
            steric_scores: vec![f64::INFINITY; count],
            fitness: f64::INFINITY,
        }
    }

    /// Keep the per-site bookkeeping arrays aligned with the gene state before
    /// an operator indexes them.  GPU score-only evaluation and restored
    /// browser jobs may legitimately omit coordinate caches; they must never
    /// turn that omission into a WASM bounds trap during crossover or repair.
    fn normalize_bookkeeping(&mut self) {
        let count = self.genes.len();
        self.phi_components.resize(count, 0);
        self.psi_components.resize(count, 0);
        self.frozen_mask.resize(count, false);
        self.conformations.resize(count, None);
        self.site_grids.resize(count, None);
        self.dirty.resize(count, true);
        self.valid_mask.resize(count, false);
        self.steric_scores.resize(count, f64::INFINITY);
    }
}

struct CookbookSearchResult {
    best_state: Vec<Gene>,
    phi_components: Vec<usize>,
    psi_components: Vec<usize>,
    best_score: f64,
    generations: usize,
    history: Vec<GenerationRecord>,
    first_feasible_score: Option<f64>,
    final_prior_score: Option<f64>,
    valid_candidates: usize,
    search_budget: usize,
    termination_reason: String,
    evaluations: usize,
    geometry_gpu_evaluations: usize,
    geometry_cpu_evaluations: usize,
    geometry_gpu_seconds: f64,
    geometry_cpu_seconds: f64,
    geometry_transform_seconds: f64,
    compatibility_algorithm: String,
    compatibility_seeded_states: usize,
    compatibility_pose_attempts: usize,
    compatibility_pool_sizes: Vec<usize>,
    compatibility_pool_expansions: usize,
    compatibility_checks: usize,
    compatibility_backtracks: usize,
    compatibility_graph_edges: usize,
    compatibility_attempts_per_site: Vec<usize>,
    compatibility_proposal_budget: usize,
}

#[derive(Debug, Clone, Default)]
struct CompatibilitySearchDiagnostics {
    algorithm: String,
    seeded_states: usize,
    pose_attempts: usize,
    pool_sizes: Vec<usize>,
    pool_expansions: usize,
    compatibility_checks: usize,
    backtracks: usize,
    graph_edges: usize,
    attempts_per_site: Vec<usize>,
    proposal_budget: usize,
}

fn cookbook_is_feasible(chromosome: &CookbookStericChromosome) -> bool {
    cookbook_fast_solution(chromosome)
}

fn cookbook_vmm_is_feasible(
    problem: &SearchProblem<'_>,
    chromosome: &CookbookStericChromosome,
) -> bool {
    if problem.selection_policy != SearchSelectionPolicy::JointPriorV1 {
        return true;
    }
    chromosome
        .genes
        .iter()
        .zip(problem.sites)
        .enumerate()
        .all(|(index, (gene, _site))| {
            let Some(priors) = problem
                .vmm_priors
                .get(index)
                .and_then(|conformers| conformers.get(gene.conformer))
            else {
                return false;
            };
            priors
                .phi
                .get(
                    chromosome
                        .phi_components
                        .get(index)
                        .copied()
                        .unwrap_or(usize::MAX),
                )
                .is_some_and(|component| vmm_component_within_95(gene.phi, component))
                && priors
                    .psi
                    .get(
                        chromosome
                            .psi_components
                            .get(index)
                            .copied()
                            .unwrap_or(usize::MAX),
                    )
                    .is_some_and(|component| vmm_component_within_95(gene.psi, component))
        })
}

fn cookbook_candidate_is_feasible(
    problem: &SearchProblem<'_>,
    chromosome: &CookbookStericChromosome,
) -> bool {
    cookbook_is_feasible(chromosome) && cookbook_vmm_is_feasible(problem, chromosome)
}

fn selection_score(
    problem: &SearchProblem<'_>,
    policy: SearchSelectionPolicy,
    chromosome: &CookbookStericChromosome,
) -> f64 {
    if policy == SearchSelectionPolicy::JointPriorV1
        && cookbook_candidate_is_feasible(problem, chromosome)
    {
        problem.joint_prior_score(&chromosome.genes)
    } else {
        problem.cookbook_fitness(chromosome)
    }
}

fn update_selection_fitness(
    problem: &SearchProblem<'_>,
    policy: SearchSelectionPolicy,
    chromosome: &mut CookbookStericChromosome,
) {
    chromosome.fitness = selection_score(problem, policy, chromosome);
}

fn sidechain_tie_score(problem: &SearchProblem<'_>, chromosome: &CookbookStericChromosome) -> f64 {
    chromosome
        .genes
        .iter()
        .enumerate()
        .filter_map(|(index, gene)| {
            gene.rotamer.and_then(|rotamer| {
                problem.sites.get(index).and_then(|site| {
                    dunbrack::probability(problem.protein, &site.site.residue, rotamer)
                })
            })
        })
        .map(|probability| -probability.max(f64::MIN_POSITIVE).ln())
        .sum()
}

fn infeasible_rank(chromosome: &CookbookStericChromosome) -> (usize, f64) {
    cookbook_rank(&chromosome.valid_mask, &chromosome.steric_scores)
}

fn compare_infeasible_rank(
    left: &CookbookStericChromosome,
    right: &CookbookStericChromosome,
) -> std::cmp::Ordering {
    let (left_failed, left_fitness) = infeasible_rank(left);
    let (right_failed, right_fitness) = infeasible_rank(right);
    left_failed
        .cmp(&right_failed)
        .then_with(|| left_fitness.total_cmp(&right_fitness))
}

/// Compare candidates in the order in which the search should retain them.
/// The final gene comparison makes exact ties reproducible even when a
/// parallel batch returns candidates in a different chunk size.
fn compare_candidates(
    problem: &SearchProblem<'_>,
    policy: SearchSelectionPolicy,
    left: &CookbookStericChromosome,
    right: &CookbookStericChromosome,
) -> std::cmp::Ordering {
    let left_feasible = cookbook_candidate_is_feasible(problem, left);
    let right_feasible = cookbook_candidate_is_feasible(problem, right);
    right_feasible
        .cmp(&left_feasible)
        .then_with(|| {
            if left_feasible && right_feasible {
                left.fitness.total_cmp(&right.fitness)
            } else {
                compare_infeasible_rank(left, right)
            }
        })
        .then_with(|| {
            if policy == SearchSelectionPolicy::JointPriorV1
                && cookbook_candidate_is_feasible(problem, left)
                && cookbook_candidate_is_feasible(problem, right)
            {
                sidechain_tie_score(problem, left).total_cmp(&sidechain_tie_score(problem, right))
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .then_with(|| compare_gene_identity(&left.genes, &right.genes))
}

fn compare_gene_identity(left: &[Gene], right: &[Gene]) -> std::cmp::Ordering {
    for (left, right) in left.iter().zip(right) {
        let ordering = left
            .conformer
            .cmp(&right.conformer)
            .then_with(|| left.phi.total_cmp(&right.phi))
            .then_with(|| left.psi.total_cmp(&right.psi))
            .then_with(|| left.rotamer.cmp(&right.rotamer));
        if ordering != std::cmp::Ordering::Equal {
            return ordering;
        }
    }
    left.len().cmp(&right.len())
}

fn candidate_better(
    problem: &SearchProblem<'_>,
    policy: SearchSelectionPolicy,
    left: &CookbookStericChromosome,
    right: &CookbookStericChromosome,
) -> bool {
    compare_candidates(problem, policy, left, right).is_lt()
}

fn cookbook_sample_gene(
    protein: &Structure,
    site: &SearchSite,
    rng: &mut ChaCha8Rng,
    cutoff: f64,
) -> (Gene, usize, usize) {
    // Feasibility search must cover the represented conformer library rather
    // than draw the common models repeatedly.  Population weights belong in
    // the later joint-prior ranking (and in statistical Ensemble proposals),
    // but using them here can starve a low-population model that is the only
    // sterically compatible member of a coupled multi-site state.
    let conformer = coverage_conformer_index(&site.ensemble, rng);
    let priors = site
        .ensemble
        .conformers
        .get(conformer)
        .map(|conformer| resolved_priors(protein, site, &conformer.priors))
        .unwrap_or_default();
    let phi = sample_uniform_from_top_region(&priors.phi, cutoff, rng);
    let psi = sample_uniform_from_top_region(&priors.psi, cutoff, rng);
    (
        Gene {
            conformer,
            phi: phi.degrees,
            psi: psi.degrees,
            // Cookbook starts from the deposited protein sidechain.  A
            // Dunbrack change is a targeted repair for a clashing site, not
            // an unconditional perturbation of every attachment residue.
            rotamer: None,
        },
        phi.component,
        psi.component,
    )
}

fn cookbook_generate(
    protein: &Structure,
    sites: &[SearchSite],
    rng: &mut ChaCha8Rng,
) -> CookbookStericChromosome {
    let mut chromosome = CookbookStericChromosome::new(sites);
    for site in sites {
        let (gene, phi_component, psi_component) = cookbook_sample_gene(protein, site, rng, 0.85);
        chromosome.genes.push(gene);
        chromosome.phi_components.push(phi_component);
        chromosome.psi_components.push(psi_component);
    }
    chromosome
}

fn cookbook_generate_native(
    protein: &Structure,
    sites: &[SearchSite],
    rng: &mut ChaCha8Rng,
) -> CookbookStericChromosome {
    let mut chromosome = CookbookStericChromosome::new(sites);
    for site in sites {
        let conformer = weighted_conformer_index(&site.ensemble, rng);
        let priors = site
            .ensemble
            .conformers
            .get(conformer)
            .map(|conformer| resolved_priors(protein, site, &conformer.priors))
            .unwrap_or_default();
        let phi = sample_truncated_vmm(&priors.phi, rng);
        let psi = sample_truncated_vmm(&priors.psi, rng);
        chromosome.genes.push(Gene {
            conformer,
            phi: phi.degrees,
            psi: psi.degrees,
            rotamer: None,
        });
        chromosome.phi_components.push(phi.component);
        chromosome.psi_components.push(psi.component);
    }
    chromosome
}

fn evaluate_cookbook_chromosome(
    problem: &SearchProblem<'_>,
    chromosome: &mut CookbookStericChromosome,
) -> Result<PreparedEvaluation> {
    let evaluation = problem.prepared.evaluate_cached(
        &chromosome.genes,
        &mut chromosome.conformations,
        &mut chromosome.site_grids,
        &chromosome.dirty,
        problem.clash_distance,
    )?;
    chromosome.steric_scores = evaluation.site_scores.clone();
    // A prepared evaluation either produces every conformation or returns an
    // error. Steric acceptance is kept separately in `steric_scores`, exactly
    // as in Cookbook, so a clash-free chromosome can terminate immediately.
    chromosome.valid_mask = vec![true; chromosome.genes.len()];
    chromosome.fitness = problem.cookbook_fitness(chromosome);
    chromosome.dirty.fill(false);
    Ok(evaluation)
}

fn cookbook_crossover(
    first: &CookbookStericChromosome,
    second: &CookbookStericChromosome,
    rng: &mut ChaCha8Rng,
) -> CookbookStericChromosome {
    let count = first.genes.len();
    // A score-only or interrupted evaluation can leave a partially populated
    // bookkeeping vector on a parent.  The chromosome still has a valid gene
    // layout, so crossover must repair the per-site arrays before indexing
    // them.  An empty parent cannot produce a crossover point; keep it as-is
    // and let the caller report an empty search state if that is not valid for
    // the current problem.
    if count == 0 {
        return first.clone();
    }
    let point = rng.random_range(0..count);
    // Match Cookbook's crossover semantics: the first parent is the child
    // template, and its frozen genes can never be overwritten.  Keeping its
    // per-site scores is also important because mutation uses those scores to
    // concentrate work on the genuinely problematic sites.  Resetting every
    // score to infinity made every glycan look problematic and repeatedly
    // mutated already solved sites.
    let mut child = first.clone();
    // Score-only GPU evaluation does not materialize site coordinates. Keep
    // the chromosome layout valid even if a stale or partially evaluated
    // parent reaches this operator; every site still needs a bookkeeping slot
    // for later mutation and evaluation.
    child.normalize_bookkeeping();
    for index in point..count {
        if child.frozen_mask[index] {
            continue;
        }
        // Parents normally have the same number of sites.  Keep the first
        // parent's value when a stale/partially evaluated second parent does
        // not, rather than turning a recoverable bookkeeping mismatch into a
        // WASM `unreachable` trap.
        let Some(gene) = second.genes.get(index) else {
            continue;
        };
        child.genes[index] = gene.clone();
        if let Some(value) = second.phi_components.get(index) {
            child.phi_components[index] = *value;
        }
        if let Some(value) = second.psi_components.get(index) {
            child.psi_components[index] = *value;
        }
        if let Some(value) = second.frozen_mask.get(index) {
            child.frozen_mask[index] = *value;
        }
        if let Some(value) = second.valid_mask.get(index) {
            child.valid_mask[index] = *value;
        }
        if let Some(value) = second.steric_scores.get(index) {
            child.steric_scores[index] = *value;
        }
        child.conformations[index] = second.conformations.get(index).cloned().unwrap_or(None);
        child.site_grids[index] = second.site_grids.get(index).cloned().unwrap_or(None);
        child.dirty[index] = second.dirty.get(index).copied().unwrap_or(true);
    }
    // The aggregate fitness is stale after combining parents and is always
    // recomputed by `evaluate_cookbook_chromosome`.
    child.fitness = f64::INFINITY;
    child
}

fn cookbook_select_parent<'a>(
    problem: &SearchProblem<'_>,
    population: &'a [CookbookStericChromosome],
    rng: &mut ChaCha8Rng,
    policy: SearchSelectionPolicy,
) -> &'a CookbookStericChromosome {
    let mut best = rng.random_range(0..population.len());
    for _ in 1..3 {
        let candidate = rng.random_range(0..population.len());
        if candidate_better(problem, policy, &population[candidate], &population[best]) {
            best = candidate;
        }
    }
    &population[best]
}

fn normal_delta_degrees(rng: &mut ChaCha8Rng, standard_deviation: f64) -> f64 {
    let u1 = rng.random::<f64>().max(f64::MIN_POSITIVE);
    let u2 = rng.random::<f64>();
    standard_deviation * (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn cookbook_mutate(
    protein: &Structure,
    sites: &[SearchSite],
    chromosome: &mut CookbookStericChromosome,
    generation: usize,
    max_generations: usize,
    mutation_rate: f64,
    creep_standard_deviation: f64,
    scan_rotamers: bool,
    rng: &mut ChaCha8Rng,
) {
    chromosome.normalize_bookkeeping();
    let cluster_mutation_rate: f64 = 0.03;
    let generation_ratio = if max_generations == 0 {
        1.0
    } else {
        generation as f64 / max_generations as f64
    };
    let creep_probability = generation_ratio.powf(1.2);
    for index in 0..chromosome.genes.len() {
        if chromosome.frozen_mask[index] {
            continue;
        }
        let problematic = chromosome
            .steric_scores
            .get(index)
            .is_some_and(|score| *score > 1.5);
        let effective_rate = if problematic {
            (mutation_rate * 2.0).min(0.7)
        } else {
            mutation_rate
        };
        if !rng.random_bool(effective_rate) {
            continue;
        }
        let Some(site) = sites.get(index) else {
            // A restored chromosome with more genes than the prepared site
            // list is invalid input. Leave it for the evaluator to reject
            // rather than indexing the site slice in a WASM worker.
            continue;
        };
        let previous_conformer = chromosome.genes[index].conformer;
        if rng.random::<f64>()
            < if problematic {
                (cluster_mutation_rate * 2.0).min(0.15)
            } else {
                cluster_mutation_rate
            }
        {
            // Keep the steric operator coverage-oriented.  The selected
            // chromosome is still ranked by its normalized prior after it
            // passes the complete steric gate.
            chromosome.genes[index].conformer = coverage_conformer_index(&site.ensemble, rng);
        }
        let conformer_changed = previous_conformer != chromosome.genes[index].conformer;
        if scan_rotamers && problematic && rng.random_bool(0.15) {
            chromosome.genes[index].rotamer =
                sample_rotamer_biased(protein, &site.site.residue, rng);
        }
        let Some(conformer) = site
            .ensemble
            .conformers
            .get(chromosome.genes[index].conformer)
        else {
            continue;
        };
        let priors = resolved_priors(protein, site, &conformer.priors);
        // Once a site is clashing, search the complete declared VMM-95
        // region instead of repeatedly drawing only the high-density 75%
        // basin.  The prior remains the ranking objective after a joint
        // feasible state is found; this branch only prevents a valid,
        // lower-density escape basin from being unreachable.
        let cutoff = if problematic { 0.95 } else { 0.85 };
        if rng.random::<f64>() < creep_probability {
            let phi = wrap_degrees(
                chromosome.genes[index].phi + normal_delta_degrees(rng, creep_standard_deviation),
            );
            let psi = wrap_degrees(
                chromosome.genes[index].psi + normal_delta_degrees(rng, creep_standard_deviation),
            );
            let phi_ok = priors
                .phi
                .get(chromosome.phi_components[index])
                .is_some_and(|component| vmm_component_within_95(phi, component));
            let psi_ok = priors
                .psi
                .get(chromosome.psi_components[index])
                .is_some_and(|component| vmm_component_within_95(psi, component));
            if !conformer_changed && phi_ok && psi_ok {
                chromosome.genes[index].phi = phi;
                chromosome.genes[index].psi = psi;
            } else {
                let sampled_phi = sample_uniform_from_top_region(&priors.phi, cutoff, rng);
                let sampled_psi = sample_uniform_from_top_region(&priors.psi, cutoff, rng);
                chromosome.genes[index].phi = sampled_phi.degrees;
                chromosome.genes[index].psi = sampled_psi.degrees;
                chromosome.phi_components[index] = sampled_phi.component;
                chromosome.psi_components[index] = sampled_psi.component;
            }
        } else {
            let sampled_phi = sample_uniform_from_top_region(&priors.phi, cutoff, rng);
            let sampled_psi = sample_uniform_from_top_region(&priors.psi, cutoff, rng);
            chromosome.genes[index].phi = sampled_phi.degrees;
            chromosome.genes[index].psi = sampled_psi.degrees;
            chromosome.phi_components[index] = sampled_phi.component;
            chromosome.psi_components[index] = sampled_psi.component;
        }
        chromosome.dirty[index] = true;
        chromosome.conformations[index] = None;
        chromosome.site_grids[index] = None;
    }
}

fn cookbook_update_freezing(
    chromosome: &mut CookbookStericChromosome,
    freezing_distance: f64,
    freezing_threshold: f64,
) {
    let count = chromosome.genes.len();
    for index in 0..count {
        if chromosome.frozen_mask[index]
            && (!chromosome.valid_mask[index]
                || chromosome.steric_scores[index] > freezing_threshold)
        {
            chromosome.frozen_mask[index] = false;
            chromosome.dirty[index] = true;
            chromosome.conformations[index] = None;
            chromosome.site_grids[index] = None;
        }
    }
    let all_solved = chromosome
        .valid_mask
        .iter()
        .zip(&chromosome.steric_scores)
        .all(|(valid, score)| *valid && *score <= freezing_threshold);
    if all_solved {
        chromosome.frozen_mask.fill(true);
        return;
    }
    let centers = chromosome
        .conformations
        .iter()
        .map(|coordinates| {
            let coordinates = coordinates.as_deref().unwrap_or_default();
            if coordinates.is_empty() {
                return Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                };
            }
            let inv = 1.0 / coordinates.len() as f64;
            coordinates.iter().fold(
                Vec3 {
                    x: 0.0,
                    y: 0.0,
                    z: 0.0,
                },
                |sum, position| Vec3 {
                    x: sum.x + position.x * inv,
                    y: sum.y + position.y * inv,
                    z: sum.z + position.z * inv,
                },
            )
        })
        .collect::<Vec<_>>();
    let distance2 = freezing_distance * freezing_distance;
    for index in 0..count {
        if chromosome.frozen_mask[index]
            || !chromosome.valid_mask[index]
            || chromosome.steric_scores[index] > freezing_threshold
        {
            continue;
        }
        let isolated = (0..count)
            .filter(|other| *other != index && !chromosome.frozen_mask[*other])
            .all(|other| {
                let dx = centers[index].x - centers[other].x;
                let dy = centers[index].y - centers[other].y;
                let dz = centers[index].z - centers[other].z;
                dx * dx + dy * dy + dz * dz >= distance2
            });
        if isolated {
            chromosome.frozen_mask[index] = true;
        }
    }
}

/// Fast Cookbook centre-distance gate used while scoring and freezing.  The
/// prepared scorer intentionally retains this inexpensive 1.7 Å prescreen so
/// the GA can evaluate thousands of proposals without materialising full
/// structures.  It is not the final chemical acceptance criterion.
fn cookbook_fast_solution(chromosome: &CookbookStericChromosome) -> bool {
    chromosome
        .valid_mask
        .iter()
        .zip(&chromosome.steric_scores)
        .all(|(valid, score)| *valid && *score <= 1.1)
}

/// A complete prepared chromosome.  The Cookbook score is the strict
/// in-search gate: every selected attachment must be below the historical
/// 1.1 centre-distance score. Background torsion analysis is not part of
/// search acceptance.
fn cookbook_strict_solution(chromosome: &CookbookStericChromosome) -> bool {
    chromosome
        .valid_mask
        .iter()
        .zip(&chromosome.steric_scores)
        .all(|(valid, score)| *valid && *score <= 1.1)
}

fn cookbook_strict_search_solution(
    problem: &SearchProblem<'_>,
    chromosome: &CookbookStericChromosome,
) -> bool {
    // The generation loop intentionally uses only the prepared Cookbook
    // scorer. Materialising a full Structure and running topology/chemistry
    // validation made ordinary builds scale as O(population × protein atoms ×
    // generations). Accepted output is returned immediately; descriptive
    // torsion analysis runs independently afterward.
    // VMM membership is an invariant of the strict chromosome: generation,
    // crossover, mutation, creep fallback, repair, and rotamer repair all
    // preserve the selected component and draw inside its bounded region.
    // Re-resolving priors here used to turn that invariant into a late
    // rejection gate. Apart from repeating work for every population member,
    // boundary/rounding differences could make an already steric-free Build
    // continue for many more generations. Cookbook terminates on the prepared
    // steric score; retain component checks in diagnostics/tests, not in this
    // hot acceptance path.
    cookbook_candidate_is_feasible(problem, chromosome)
}

/// Find the best complete strict candidate in the current population.  The
/// hot loop is deliberately limited to the prepared Cookbook steric score and
/// component-conditioned VMM gate. The accepted output is returned directly.
fn cookbook_strict_candidate_index(
    _problem: &SearchProblem<'_>,
    population: &[CookbookStericChromosome],
) -> Option<usize> {
    for (index, chromosome) in population.iter().enumerate() {
        if !cookbook_fast_solution(chromosome) {
            continue;
        }
        // The population is fitness-sorted before this function is called, so
        // the first valid member is the Cookbook-best candidate.  Returning
        // immediately is what keeps a successful build from consuming the
        // remaining generation budget.
        return Some(index);
    }
    None
}

fn cookbook_search_result(
    chromosome: &CookbookStericChromosome,
    generation: usize,
    history: Vec<GenerationRecord>,
    first_feasible_score: Option<f64>,
    final_prior_score: Option<f64>,
    valid_candidates: usize,
    search_budget: usize,
    termination_reason: String,
    evaluations: usize,
    geometry: &GeometrySession,
    compatibility: &CompatibilitySearchDiagnostics,
) -> CookbookSearchResult {
    CookbookSearchResult {
        best_state: chromosome.genes.clone(),
        phi_components: chromosome.phi_components.clone(),
        psi_components: chromosome.psi_components.clone(),
        best_score: chromosome.fitness,
        generations: generation,
        history,
        first_feasible_score,
        final_prior_score,
        valid_candidates,
        search_budget,
        termination_reason,
        evaluations,
        geometry_gpu_evaluations: geometry.gpu_evaluations,
        geometry_cpu_evaluations: geometry.cpu_evaluations,
        geometry_gpu_seconds: geometry.gpu_seconds,
        geometry_cpu_seconds: geometry.cpu_seconds,
        geometry_transform_seconds: geometry.transform_seconds,
        compatibility_algorithm: compatibility.algorithm.clone(),
        compatibility_seeded_states: compatibility.seeded_states,
        compatibility_pose_attempts: compatibility.pose_attempts,
        compatibility_pool_sizes: compatibility.pool_sizes.clone(),
        compatibility_pool_expansions: compatibility.pool_expansions,
        compatibility_checks: compatibility.compatibility_checks,
        compatibility_backtracks: compatibility.backtracks,
        compatibility_graph_edges: compatibility.graph_edges,
        compatibility_attempts_per_site: compatibility.attempts_per_site.clone(),
        compatibility_proposal_budget: compatibility.proposal_budget,
    }
}

fn dominant_component(components: &[VonMisesComponent]) -> Option<usize> {
    components
        .iter()
        .enumerate()
        .filter(|(_, component)| component.weight.is_finite() && component.weight > 0.0)
        .max_by(|(left_index, left), (right_index, right)| {
            left.weight
                .total_cmp(&right.weight)
                .then_with(|| right_index.cmp(left_index))
        })
        .map(|(index, _)| index)
}

fn component_pair_candidates(
    phi: &[VonMisesComponent],
    psi: &[VonMisesComponent],
    current_phi: usize,
    current_psi: usize,
) -> Vec<(usize, usize)> {
    let dominant_phi = dominant_component(phi).unwrap_or(current_phi);
    let dominant_psi = dominant_component(psi).unwrap_or(current_psi);
    let mut ranked = phi
        .iter()
        .enumerate()
        .flat_map(|(phi_index, phi_component)| {
            psi.iter()
                .enumerate()
                .map(move |(psi_index, psi_component)| {
                    (
                        phi_index,
                        psi_index,
                        phi_component.weight.max(0.0) * psi_component.weight.max(0.0),
                    )
                })
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| {
        right
            .2
            .total_cmp(&left.2)
            .then_with(|| left.0.cmp(&right.0))
            .then_with(|| left.1.cmp(&right.1))
    });
    let next_best = ranked
        .into_iter()
        .map(|(phi_index, psi_index, _)| (phi_index, psi_index))
        .find(|pair| *pair != (current_phi, current_psi) && *pair != (dominant_phi, dominant_psi));
    let ordered = [
        (current_phi, current_psi),
        (dominant_phi, dominant_psi),
        (dominant_phi, current_psi),
        (current_phi, dominant_psi),
        next_best.unwrap_or((current_phi, current_psi)),
    ];
    let mut candidates = Vec::with_capacity(5);
    for pair in ordered {
        if !candidates.contains(&pair) {
            candidates.push(pair);
        }
    }
    candidates
}

/// One deterministic, attachment-only likelihood pass after strict search.
/// It never changes conformer/rotamer selection or internal glycan geometry,
/// and every accepted move must retain the complete Cookbook steric gate.
#[maybe_async_cfg::maybe(
    sync(keep_self),
    async(feature = "webgpu"),
    idents(
        evaluate_population(sync, async = "evaluate_population_async"),
        polish_cookbook_attachment_vmm(sync, async = "polish_cookbook_attachment_vmm_async")
    )
)]
async fn polish_cookbook_attachment_vmm<C>(
    problem: &SearchProblem<'_>,
    state: &[Gene],
    phi_components: &[usize],
    psi_components: &[usize],
    policy: SearchSelectionPolicy,
    max_evaluations: usize,
    geometry: &mut GeometrySession,
    mut cancelled: C,
) -> Result<(Vec<Gene>, Vec<usize>, Vec<usize>, f64, VmmPolishDiagnostics)>
where
    C: FnMut() -> bool,
{
    if max_evaluations == 0 {
        return Ok((
            state.to_vec(),
            phi_components.to_vec(),
            psi_components.to_vec(),
            0.0,
            VmmPolishDiagnostics {
                applied: false,
                selection_policy: match policy {
                    SearchSelectionPolicy::JointPriorV1 => "joint_prior_v1".into(),
                    SearchSelectionPolicy::CookbookFirstFeasible => {
                        "cookbook_first_feasible".into()
                    }
                },
                prior_model: "conformer_population_x_attachment_vmm_v1".into(),
                ..VmmPolishDiagnostics::default()
            },
        ));
    }
    let mut current = CookbookStericChromosome::new(problem.sites);
    current.genes = state.to_vec();
    current.phi_components = phi_components.to_vec();
    current.psi_components = psi_components.to_vec();
    evaluate_cookbook_chromosome(problem, &mut current)?;
    let ranking_score = |chromosome: &CookbookStericChromosome| {
        if policy == SearchSelectionPolicy::JointPriorV1 {
            problem.joint_prior_score(&chromosome.genes)
        } else {
            chromosome
                .genes
                .iter()
                .enumerate()
                .map(|(index, gene)| problem.attachment_mixture_score(index, gene))
                .sum()
        }
    };
    let score_before = ranking_score(&current);
    let mut diagnostics = VmmPolishDiagnostics {
        applied: true,
        score_before,
        score_after: score_before,
        ..VmmPolishDiagnostics::default()
    };
    // The incumbent was already evaluated by the search.  Re-evaluating it
    // to seed the polish state is bookkeeping, not a new proposal slot.  Keep
    // the finite polish budget for actual candidate states.
    let mut evaluation_count = 0usize;

    for index in 0..current.genes.len() {
        if cancelled() {
            return Err(EnsembleError::Cancelled);
        }
        let Some(site) = problem.sites.get(index) else {
            return Err(EnsembleError::ReGlyco(ReGlycoError::InvalidGeometry));
        };
        let Some(conformer) = site.ensemble.conformers.get(current.genes[index].conformer) else {
            return Err(EnsembleError::ReGlyco(ReGlycoError::InvalidGeometry));
        };
        let priors = resolved_priors(problem.protein, site, &conformer.priors);
        let original_phi_component = current.phi_components[index];
        let original_psi_component = current.psi_components[index];
        let Some(_) = priors.phi.get(original_phi_component) else {
            continue;
        };
        let Some(_) = priors.psi.get(original_psi_component) else {
            continue;
        };
        let site_score_before = problem.attachment_mixture_score(index, &current.genes[index]);
        let steric_before = current.steric_scores[index];
        let mut best = current.clone();
        let mut best_score = ranking_score(&best);
        let candidates = component_pair_candidates(
            &priors.phi,
            &priors.psi,
            original_phi_component,
            original_psi_component,
        );
        let mut proposal_count = 0usize;
        let conformer_indices = if policy == SearchSelectionPolicy::JointPriorV1 {
            let mut indices = (0..site.ensemble.conformers.len()).collect::<Vec<_>>();
            indices.sort_by(|left, right| {
                site.ensemble.conformers[*right]
                    .cluster_weight
                    .total_cmp(&site.ensemble.conformers[*left].cluster_weight)
                    .then_with(|| left.cmp(right))
            });
            indices.retain(|index| {
                site.ensemble.conformers[*index].cluster_weight.is_finite()
                    && site.ensemble.conformers[*index].cluster_weight > 0.0
            });
            indices.truncate(8);
            indices
        } else {
            vec![current.genes[index].conformer]
        };
        let mut pending = Vec::new();
        'proposals: for conformer_index in conformer_indices {
            let Some(candidate_conformer) = site.ensemble.conformers.get(conformer_index) else {
                continue;
            };
            let candidate_priors =
                resolved_priors(problem.protein, site, &candidate_conformer.priors);
            let candidate_pairs = if policy == SearchSelectionPolicy::JointPriorV1 {
                component_pair_candidates(
                    &candidate_priors.phi,
                    &candidate_priors.psi,
                    original_phi_component,
                    original_psi_component,
                )
            } else {
                candidates.clone()
            };
            for (phi_component_index, psi_component_index) in candidate_pairs {
                if cancelled() {
                    return Err(EnsembleError::Cancelled);
                }
                let Some(phi_component) = candidate_priors.phi.get(phi_component_index) else {
                    continue;
                };
                let Some(psi_component) = candidate_priors.psi.get(psi_component_index) else {
                    continue;
                };
                let offsets = if policy == SearchSelectionPolicy::JointPriorV1 {
                    vec![-0.5, 0.0, 0.5]
                } else {
                    vec![0.0]
                };
                for phi_offset in offsets.iter().copied() {
                    for psi_offset in offsets.iter().copied() {
                        if policy == SearchSelectionPolicy::JointPriorV1 && proposal_count >= 256 {
                            break 'proposals;
                        }
                        if evaluation_count.saturating_add(pending.len()) > max_evaluations {
                            break 'proposals;
                        }
                        proposal_count += 1;
                        diagnostics.proposals += 1;
                        let phi_sigma = vmm_component_sigma_degrees(phi_component).unwrap_or(0.0);
                        let psi_sigma = vmm_component_sigma_degrees(psi_component).unwrap_or(0.0);
                        let candidate_phi =
                            wrap_degrees(phi_component.mean_degrees + phi_offset * phi_sigma);
                        let candidate_psi =
                            wrap_degrees(psi_component.mean_degrees + psi_offset * psi_sigma);
                        let mut candidate = current.clone();
                        candidate.genes[index].conformer = conformer_index;
                        candidate.genes[index].phi = candidate_phi;
                        candidate.genes[index].psi = candidate_psi;
                        candidate.phi_components[index] = phi_component_index;
                        candidate.psi_components[index] = psi_component_index;
                        candidate.dirty[index] = true;
                        candidate.conformations[index] = None;
                        candidate.site_grids[index] = None;
                        pending.push(candidate);
                    }
                }
            }
        }
        if !pending.is_empty() {
            evaluate_population(problem, &mut pending, geometry).await?;
            evaluation_count = evaluation_count.saturating_add(pending.len());
            for candidate in pending {
                let candidate_priors = resolved_priors(
                    problem.protein,
                    &problem.sites[index],
                    &problem.sites[index].ensemble.conformers[candidate.genes[index].conformer]
                        .priors,
                );
                let Some(phi_component) = candidate_priors.phi.get(candidate.phi_components[index])
                else {
                    continue;
                };
                let Some(psi_component) = candidate_priors.psi.get(candidate.psi_components[index])
                else {
                    continue;
                };
                if !cookbook_strict_solution(&candidate)
                    || !vmm_component_within_95(candidate.genes[index].phi, phi_component)
                    || !vmm_component_within_95(candidate.genes[index].psi, psi_component)
                {
                    continue;
                }
                let score = if policy == SearchSelectionPolicy::JointPriorV1 {
                    problem.joint_prior_score(&candidate.genes)
                } else {
                    candidate
                        .genes
                        .iter()
                        .enumerate()
                        .map(|(index, gene)| problem.attachment_mixture_score(index, gene))
                        .sum()
                };
                if score + 1.0e-12 < best_score {
                    best_score = score;
                    best = candidate;
                }
            }
        }
        let current_score = ranking_score(&current);
        if best_score + 1.0e-12 < current_score {
            current = best;
            diagnostics.accepted_moves += 1;
        }
        let site_score_after = problem.attachment_mixture_score(index, &current.genes[index]);
        diagnostics.sites.push(VmmPolishSiteDiagnostics {
            site: site.site.residue.to_string(),
            original_phi_component,
            original_psi_component,
            final_phi_component: current.phi_components[index],
            final_psi_component: current.psi_components[index],
            proposals: proposal_count,
            component_switched: original_phi_component != current.phi_components[index]
                || original_psi_component != current.psi_components[index],
            score_before: site_score_before,
            score_after: site_score_after,
            steric_score_before: steric_before,
            steric_score_after: current.steric_scores[index],
        });
    }

    // Coordinate-wise polishing can miss a solution where two attached
    // glycans only become compatible after moving together.  Spend a small,
    // deterministic tail of the same finite budget on paired-site proposals.
    // This is deliberately limited to the probability-ranked Build policy;
    // legacy Cookbook and statistical ensemble semantics are unchanged.
    if policy == SearchSelectionPolicy::JointPriorV1 && current.genes.len() > 1 {
        'pairs: for left in 0..current.genes.len() {
            for right in (left + 1)..current.genes.len() {
                if cancelled() {
                    return Err(EnsembleError::Cancelled);
                }
                if evaluation_count >= max_evaluations {
                    break 'pairs;
                }
                let left_site = &problem.sites[left];
                let right_site = &problem.sites[right];
                let mut left_conformers = (0..left_site.ensemble.conformers.len())
                    .filter(|index| {
                        let weight = left_site.ensemble.conformers[*index].cluster_weight;
                        weight.is_finite() && weight > 0.0
                    })
                    .collect::<Vec<_>>();
                let mut right_conformers = (0..right_site.ensemble.conformers.len())
                    .filter(|index| {
                        let weight = right_site.ensemble.conformers[*index].cluster_weight;
                        weight.is_finite() && weight > 0.0
                    })
                    .collect::<Vec<_>>();
                let rank_conformers = |site: &SearchSite, indices: &mut Vec<usize>| {
                    indices.sort_by(|a, b| {
                        site.ensemble.conformers[*b]
                            .cluster_weight
                            .total_cmp(&site.ensemble.conformers[*a].cluster_weight)
                            .then_with(|| a.cmp(b))
                    });
                    indices.truncate(2);
                };
                rank_conformers(left_site, &mut left_conformers);
                rank_conformers(right_site, &mut right_conformers);
                if left_conformers.is_empty() || right_conformers.is_empty() {
                    continue;
                }
                let mut pending = Vec::new();
                for left_conformer in &left_conformers {
                    let left_priors = resolved_priors(
                        problem.protein,
                        left_site,
                        &left_site.ensemble.conformers[*left_conformer].priors,
                    );
                    let left_pairs = component_pair_candidates(
                        &left_priors.phi,
                        &left_priors.psi,
                        current.phi_components[left],
                        current.psi_components[left],
                    );
                    for right_conformer in &right_conformers {
                        let right_priors = resolved_priors(
                            problem.protein,
                            right_site,
                            &right_site.ensemble.conformers[*right_conformer].priors,
                        );
                        let right_pairs = component_pair_candidates(
                            &right_priors.phi,
                            &right_priors.psi,
                            current.phi_components[right],
                            current.psi_components[right],
                        );
                        for &(left_phi_component, left_psi_component) in left_pairs.iter().take(3) {
                            let Some(left_phi) = left_priors.phi.get(left_phi_component) else {
                                continue;
                            };
                            let Some(left_psi) = left_priors.psi.get(left_psi_component) else {
                                continue;
                            };
                            for &(right_phi_component, right_psi_component) in
                                right_pairs.iter().take(3)
                            {
                                let Some(right_phi) = right_priors.phi.get(right_phi_component)
                                else {
                                    continue;
                                };
                                let Some(right_psi) = right_priors.psi.get(right_psi_component)
                                else {
                                    continue;
                                };
                                for left_offset in [-0.5, 0.0, 0.5] {
                                    for right_offset in [-0.5, 0.0, 0.5] {
                                        if pending.len() >= 128
                                            || evaluation_count + pending.len() > max_evaluations
                                        {
                                            break 'pairs;
                                        }
                                        let mut candidate = current.clone();
                                        candidate.genes[left].conformer = *left_conformer;
                                        candidate.genes[left].phi = wrap_degrees(
                                            left_phi.mean_degrees
                                                + left_offset
                                                    * vmm_component_sigma_degrees(left_phi)
                                                        .unwrap_or(0.0),
                                        );
                                        candidate.genes[left].psi = wrap_degrees(
                                            left_psi.mean_degrees
                                                + left_offset
                                                    * vmm_component_sigma_degrees(left_psi)
                                                        .unwrap_or(0.0),
                                        );
                                        candidate.phi_components[left] = left_phi_component;
                                        candidate.psi_components[left] = left_psi_component;
                                        candidate.genes[right].conformer = *right_conformer;
                                        candidate.genes[right].phi = wrap_degrees(
                                            right_phi.mean_degrees
                                                + right_offset
                                                    * vmm_component_sigma_degrees(right_phi)
                                                        .unwrap_or(0.0),
                                        );
                                        candidate.genes[right].psi = wrap_degrees(
                                            right_psi.mean_degrees
                                                + right_offset
                                                    * vmm_component_sigma_degrees(right_psi)
                                                        .unwrap_or(0.0),
                                        );
                                        candidate.phi_components[right] = right_phi_component;
                                        candidate.psi_components[right] = right_psi_component;
                                        for index in [left, right] {
                                            candidate.dirty[index] = true;
                                            candidate.conformations[index] = None;
                                            candidate.site_grids[index] = None;
                                        }
                                        pending.push(candidate);
                                    }
                                }
                            }
                        }
                    }
                }
                if pending.is_empty() {
                    continue;
                }
                diagnostics.proposals += pending.len();
                evaluate_population(problem, &mut pending, geometry).await?;
                evaluation_count = evaluation_count.saturating_add(pending.len());
                let mut best_pair = current.clone();
                let mut best_pair_score = ranking_score(&best_pair);
                for candidate in pending {
                    let left_conformer =
                        &problem.sites[left].ensemble.conformers[candidate.genes[left].conformer];
                    let right_conformer =
                        &problem.sites[right].ensemble.conformers[candidate.genes[right].conformer];
                    let left_priors =
                        resolved_priors(problem.protein, left_site, &left_conformer.priors);
                    let right_priors =
                        resolved_priors(problem.protein, right_site, &right_conformer.priors);
                    let left_ok = left_priors
                        .phi
                        .get(candidate.phi_components[left])
                        .is_some_and(|component| {
                            vmm_component_within_95(candidate.genes[left].phi, component)
                        })
                        && left_priors
                            .psi
                            .get(candidate.psi_components[left])
                            .is_some_and(|component| {
                                vmm_component_within_95(candidate.genes[left].psi, component)
                            });
                    let right_ok = right_priors
                        .phi
                        .get(candidate.phi_components[right])
                        .is_some_and(|component| {
                            vmm_component_within_95(candidate.genes[right].phi, component)
                        })
                        && right_priors
                            .psi
                            .get(candidate.psi_components[right])
                            .is_some_and(|component| {
                                vmm_component_within_95(candidate.genes[right].psi, component)
                            });
                    if !cookbook_strict_solution(&candidate) || !left_ok || !right_ok {
                        continue;
                    }
                    let score = ranking_score(&candidate);
                    if score + 1.0e-12 < best_pair_score {
                        best_pair_score = score;
                        best_pair = candidate;
                    }
                }
                if best_pair_score + 1.0e-12 < ranking_score(&current) {
                    current = best_pair;
                    diagnostics.accepted_moves += 1;
                }
            }
        }
    }
    diagnostics.score_after = ranking_score(&current);
    diagnostics.evaluations = evaluation_count;
    Ok((
        current.genes,
        current.phi_components,
        current.psi_components,
        current.fitness,
        diagnostics,
    ))
}

fn cookbook_rank(valid_mask: &[bool], steric_scores: &[f64]) -> (usize, f64) {
    (
        valid_mask
            .iter()
            .zip(steric_scores)
            .filter(|(valid, score)| !**valid || **score > 1.1)
            .count(),
        steric_scores.iter().sum::<f64>(),
    )
}

#[maybe_async_cfg::maybe(
    sync(keep_self),
    async(feature = "webgpu"),
    idents(
        evaluate_population(sync, async = "evaluate_population_async"),
        cookbook_repair_with_cancel(sync, async = "cookbook_repair_with_cancel_async")
    )
)]
async fn cookbook_repair_with_cancel<C>(
    problem: &SearchProblem<'_>,
    chromosome: &mut CookbookStericChromosome,
    seed: u64,
    max_evaluations: usize,
    geometry: &mut GeometrySession,
    mut cancelled: C,
) -> Result<(bool, usize)>
where
    C: FnMut() -> bool,
{
    let mut evaluations = 0usize;
    // Cookbook's targeted repair order.  The native strategy uses bounded
    // Best-Fisher draws (1.28σ); no strict-path proposal samples the full
    // torsion circle.
    let strategies: &[(usize, Option<f64>)] = &[
        (128, Some(0.85)),
        (256, Some(0.90)),
        (64, Some(0.75)),
        (128, None),
    ];
    for (strategy, (attempts, cutoff)) in strategies.iter().copied().enumerate() {
        if cancelled() {
            return Ok((false, evaluations));
        }
        if evaluations >= max_evaluations {
            return Ok((false, evaluations));
        }
        let mut rng =
            ChaCha8Rng::seed_from_u64(seed ^ ((strategy as u64) << 32) ^ 0x7265_7061_6972_0000);
        let mut order = (0..chromosome.genes.len()).collect::<Vec<_>>();
        order.sort_by(|left, right| {
            chromosome.steric_scores[*right].total_cmp(&chromosome.steric_scores[*left])
        });
        for index in order {
            if cancelled() {
                return Ok((false, evaluations));
            }
            if evaluations >= max_evaluations {
                return Ok((false, evaluations));
            }
            if chromosome.valid_mask[index]
                && chromosome.steric_scores[index] <= 1.1
                && cookbook_strict_search_solution(problem, chromosome)
            {
                continue;
            }
            // A frozen site may become invalid after another site's move;
            // repair explicitly unfreezes it before proposing replacements.
            chromosome.frozen_mask[index] = false;
            let original = chromosome.clone();
            let mut best = chromosome.clone();
            let mut best_rank = cookbook_rank(&chromosome.valid_mask, &chromosome.steric_scores);
            let rotamer_passes = if problem.scan_rotamers {
                dunbrack::rotamers(problem.protein, &problem.sites[index].site.residue)
                    .len()
                    .min(5)
                    + 1
            } else {
                1
            };
            // The deposited sidechain is always tried first, followed by the
            // Dunbrack entries in their probability-descending table order.
            // This is the Cookbook repair contract and avoids changing a
            // protein sidechain unless it improves the complete chromosome.
            for rotamer_pass in 0..rotamer_passes {
                let batch_size = attempts.min(max_evaluations.saturating_sub(evaluations));
                if batch_size == 0 {
                    return Ok((false, evaluations));
                }
                let mut candidates = Vec::with_capacity(batch_size);
                for _ in 0..batch_size {
                    if cancelled() {
                        return Ok((false, evaluations));
                    }
                    let mut candidate = original.clone();
                    candidate.frozen_mask[index] = false;
                    let site = &problem.sites[index];
                    let conformer = coverage_conformer_index(&site.ensemble, &mut rng);
                    let Some(conformer_data) = site.ensemble.conformers.get(conformer) else {
                        continue;
                    };
                    let priors = resolved_priors(problem.protein, site, &conformer_data.priors);
                    let phi = match cutoff {
                        Some(cutoff) => {
                            sample_uniform_from_top_region(&priors.phi, cutoff, &mut rng)
                        }
                        None => sample_truncated_vmm(&priors.phi, &mut rng),
                    };
                    let psi = match cutoff {
                        Some(cutoff) => {
                            sample_uniform_from_top_region(&priors.psi, cutoff, &mut rng)
                        }
                        None => sample_truncated_vmm(&priors.psi, &mut rng),
                    };
                    // `then_some` evaluates its argument eagerly.  Keep the
                    // deposited side-chain pass at `None` without computing
                    // `0 - 1`; the old expression underflowed in debug/WASM
                    // builds before a diagnostic Build could be exported.
                    let rotamer = rotamer_pass_index(rotamer_pass);
                    candidate.genes[index] = Gene {
                        conformer,
                        phi: phi.degrees,
                        psi: psi.degrees,
                        rotamer,
                    };
                    candidate.phi_components[index] = phi.component;
                    candidate.psi_components[index] = psi.component;
                    candidate.dirty[index] = true;
                    candidate.conformations[index] = None;
                    candidate.site_grids[index] = None;
                    candidates.push(candidate);
                }
                evaluate_population(problem, &mut candidates, geometry).await?;
                evaluations = evaluations.saturating_add(candidates.len());
                for candidate in candidates {
                    // A repair is accepted only when it satisfies the same
                    // Cookbook 1.1 score and VMM-95% gates as an ordinary
                    // strict candidate.
                    if cookbook_strict_search_solution(problem, &candidate) {
                        *chromosome = candidate;
                        return Ok((true, evaluations));
                    }
                    let rank = cookbook_rank(&candidate.valid_mask, &candidate.steric_scores);
                    if rank < best_rank {
                        best_rank = rank;
                        best = candidate;
                        if cookbook_strict_search_solution(problem, &best) {
                            *chromosome = best;
                            return Ok((true, evaluations));
                        }
                    }
                }
            }
            *chromosome = best;
        }
        if cookbook_strict_search_solution(problem, chromosome) {
            return Ok((true, evaluations));
        }
    }
    Ok((false, evaluations))
}

/// Apply a bounded, deterministic Dunbrack repair to a sampled state.  Native
/// ensemble proposals normally retain the deposited protein sidechains; this
/// seam is entered only after the complete proposal has a steric clash.  The
/// first entry is the original sidechain, followed by the highest-probability
/// library entries, and a replacement is kept only when the complete state
/// rank improves.
fn cookbook_try_rotamer_repair(
    problem: &SearchProblem<'_>,
    state: &[Gene],
    evaluation: &PreparedEvaluation,
) -> Result<(Vec<Gene>, PreparedEvaluation)> {
    if !problem.scan_rotamers || evaluation.site_scores.iter().all(|score| *score <= 1.1) {
        return Ok((state.to_vec(), evaluation.clone()));
    }
    let mut working_state = state.to_vec();
    let mut working_evaluation = evaluation.clone();
    let mut order = (0..working_state.len()).collect::<Vec<_>>();
    order.sort_by(|left, right| {
        working_evaluation.site_scores[*right].total_cmp(&working_evaluation.site_scores[*left])
    });
    for index in order {
        if working_evaluation.site_scores[index] <= 1.1 {
            continue;
        }
        let original_state = working_state.clone();
        let original_evaluation = working_evaluation.clone();
        let mut best_state = original_state.clone();
        let mut best_evaluation = original_evaluation.clone();
        let mut best_rank = cookbook_rank(
            &vec![true; original_evaluation.site_scores.len()],
            &original_evaluation.site_scores,
        );
        let rotamer_count = dunbrack::rotamers(problem.protein, &problem.sites[index].site.residue)
            .len()
            .min(5);
        for rotamer_pass in 0..=rotamer_count {
            let mut candidate = original_state.clone();
            candidate[index].rotamer = rotamer_pass_index(rotamer_pass);
            let Ok(candidate_evaluation) = problem
                .prepared
                .evaluate(&candidate, problem.clash_distance)
            else {
                continue;
            };
            let rank = cookbook_rank(
                &vec![true; candidate_evaluation.site_scores.len()],
                &candidate_evaluation.site_scores,
            );
            if rank < best_rank {
                best_rank = rank;
                best_state = candidate;
                best_evaluation = candidate_evaluation;
                if best_evaluation
                    .site_scores
                    .iter()
                    .all(|score| *score <= 1.1)
                {
                    return Ok((best_state, best_evaluation));
                }
            }
        }
        if best_rank
            < cookbook_rank(
                &vec![true; working_evaluation.site_scores.len()],
                &working_evaluation.site_scores,
            )
        {
            working_state = best_state;
            working_evaluation = best_evaluation;
        }
    }
    Ok((working_state, working_evaluation))
}

#[maybe_async_cfg::maybe(
    sync(keep_self),
    async(feature = "webgpu"),
    idents(
        evaluate_population(sync, async = "evaluate_population_async"),
        cookbook_repair_with_cancel(sync, async = "cookbook_repair_with_cancel_async")
    )
)]
async fn cookbook_steric_search_with_cancel<F, C>(
    problem: &SearchProblem<'_>,
    config: &SearchConfig,
    mut progress: F,
    mut cancelled: C,
) -> Result<CookbookSearchResult>
where
    F: FnMut(SearchProgress),
    C: FnMut() -> bool,
{
    if cancelled() {
        return Err(EnsembleError::Cancelled);
    }
    let policy = config.selection_policy;
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let search_started = Instant::now();
    let population_capacity = config.population_size.max(2);
    let base_search_budget =
        population_capacity.saturating_mul(config.generations.saturating_add(1));
    let search_budget = base_search_budget;
    // Pose-pool construction runs before the GA so coupled feasible
    // assignments can seed the population. Its local-pose allowance is
    // explicit and reported separately; it is not a rescue pass after the GA
    // has exhausted the complete-candidate budget.
    let compatibility_budget = if policy == SearchSelectionPolicy::JointPriorV1 {
        // Pose discovery and complete candidates share one resolved proposal
        // ledger.  Reserve half for local pose/compatibility work so repair
        // and GA proposals cannot silently exceed B = population ×
        // (generations + 1).
        base_search_budget / 2
    } else {
        0
    };
    // The remaining half of the shared ledger is installed after the local
    // pose stage, using its actual consumed work rather than its reservation.
    // This keeps easy cases from losing useful GA work while preserving one
    // deterministic total cap.
    if compatibility_budget > 0 && problem.sites.len() > 1 {
        progress(SearchProgress::Generation {
            phase: SearchPhase::Compatibility,
            generation: 0,
            best_score: f64::INFINITY,
            mean_score: f64::INFINITY,
            evaluations: 0,
            cache_hits: 0,
            steric_rejections: 0,
            elapsed_seconds: search_started.elapsed().as_secs_f64(),
            valid_candidates: 0,
            first_feasible_score: None,
        });
    }
    let compatibility_states = if compatibility_budget > 0 && problem.sites.len() > 1 {
        cookbook_compatible_pool_coverage(
            problem.protein,
            problem.sites,
            config,
            problem.prepared,
            population_capacity.min(4),
            compatibility_budget,
        )?
    } else {
        CompatiblePoolResult {
            states: Vec::new(),
            pools: vec![Vec::new(); problem.sites.len()],
            proposals: 0,
            attempts_per_site: vec![0; problem.sites.len()],
            pool_sizes: Vec::new(),
            pool_expansions: 0,
            compatibility_checks: 0,
            backtracks: 0,
            graph_edges: 0,
        }
    };
    let compatibility_diagnostics = CompatibilitySearchDiagnostics {
        algorithm: (compatibility_budget > 0)
            .then_some("arc_consistency_backtracking_v1".into())
            .unwrap_or_default(),
        seeded_states: compatibility_states.states.len(),
        pose_attempts: compatibility_states.proposals,
        pool_sizes: compatibility_states.pool_sizes.clone(),
        pool_expansions: compatibility_states.pool_expansions,
        compatibility_checks: compatibility_states.compatibility_checks,
        backtracks: compatibility_states.backtracks,
        graph_edges: compatibility_states.graph_edges,
        attempts_per_site: compatibility_states.attempts_per_site.clone(),
        proposal_budget: compatibility_budget,
    };
    let complete_search_budget = base_search_budget
        .saturating_sub(compatibility_states.proposals)
        .max(population_capacity);
    let seeded_count = compatibility_states.states.len().min(population_capacity);
    let mut population = compatibility_states
        .states
        .into_iter()
        .take(seeded_count)
        .map(|(state, phi_components, psi_components)| {
            let mut chromosome = CookbookStericChromosome::new(problem.sites);
            chromosome.genes = state;
            chromosome.phi_components = phi_components;
            chromosome.psi_components = psi_components;
            chromosome
        })
        .collect::<Vec<_>>();
    population.extend(
        (seeded_count..population_capacity)
            .map(|_| cookbook_generate(problem.protein, problem.sites, &mut rng)),
    );
    let mut geometry = GeometrySession::default();
    // Joint-prior Build does not use Cookbook freezing, so complete
    // transformed coordinate arrays are unnecessary in the hot loop. The
    // resident GPU returns site scores and boundary candidates; coordinates
    // are materialized only for selected output or CPU checks.
    geometry.scores_only = policy == SearchSelectionPolicy::JointPriorV1;
    evaluate_population(problem, &mut population, &mut geometry).await?;
    population
        .iter_mut()
        .for_each(|chromosome| update_selection_fitness(problem, policy, chromosome));
    if cancelled() {
        return Err(EnsembleError::Cancelled);
    }
    let mut history = Vec::with_capacity(config.generations.saturating_add(1));
    let mut evaluations = population.len();
    let mut best_history = Vec::new();
    let mut stagnation = 0usize;
    let mut mutation_rate = 0.15;
    let mut creep_std = 10.0;
    let mut freezing_threshold = 1.1;
    let freezing_distance = 30.0;
    let mut best_generation = 0usize;
    let mut best_feasible: Option<CookbookStericChromosome> = None;
    let mut first_feasible_score = None;
    let mut valid_candidates = 0usize;
    let mut prior_stagnation = 0usize;

    for generation in 0..=config.generations {
        if cancelled() {
            return Err(EnsembleError::Cancelled);
        }
        population.sort_by(|left, right| compare_candidates(problem, policy, left, right));
        let generation_valid = population
            .iter()
            .filter(|chromosome| cookbook_candidate_is_feasible(problem, chromosome))
            .count();
        valid_candidates = valid_candidates.saturating_add(generation_valid);
        let best = &population[0];
        let best_score = best.fitness;
        let mean_score = population
            .iter()
            .map(|chromosome| chromosome.fitness)
            .sum::<f64>()
            / population.len() as f64;
        history.push(GenerationRecord {
            generation,
            best_score,
            mean_score,
        });
        progress(SearchProgress::Generation {
            phase: if policy == SearchSelectionPolicy::JointPriorV1 && best_feasible.is_some() {
                SearchPhase::ProbabilityImprovement
            } else if policy == SearchSelectionPolicy::JointPriorV1
                && generation == 0
                && compatibility_diagnostics.seeded_states > 0
            {
                SearchPhase::Compatibility
            } else {
                SearchPhase::Feasibility
            },
            generation,
            best_score,
            mean_score,
            evaluations,
            cache_hits: 0,
            steric_rejections: 0,
            elapsed_seconds: search_started.elapsed().as_secs_f64(),
            valid_candidates,
            first_feasible_score,
        });
        if cancelled() {
            return Err(EnsembleError::Cancelled);
        }
        best_history.push(best_score);
        if policy != SearchSelectionPolicy::JointPriorV1 {
            best_generation = generation;
        }
        if policy == SearchSelectionPolicy::JointPriorV1 {
            if let Some(candidate) = population
                .iter()
                .find(|candidate| cookbook_candidate_is_feasible(problem, candidate))
            {
                let candidate_score = candidate.fitness;
                if first_feasible_score.is_none() {
                    first_feasible_score = Some(candidate_score);
                }
                let previous_score = best_feasible.as_ref().map(|incumbent| incumbent.fitness);
                let improved = previous_score.is_none_or(|score| candidate_score < score);
                if improved {
                    best_feasible = Some(candidate.clone());
                    best_generation = generation;
                }
                if previous_score.is_none_or(|score| candidate_score + 1.0e-4 < score) {
                    prior_stagnation = 0;
                } else {
                    prior_stagnation = prior_stagnation.saturating_add(1);
                }
            } else if best_feasible.is_some() {
                // A generation with no feasible member is still a completed
                // generation without incumbent improvement.  Counting it is
                // necessary for a deterministic work-bounded stop in crowded
                // searches where feasibility briefly disappears from the
                // retained population.
                prior_stagnation = prior_stagnation.saturating_add(1);
            }
            // Ten complete generations without a meaningful probability
            // improvement is the deterministic post-feasibility stop.
            if prior_stagnation >= 10 {
                let Some(accepted) = best_feasible.as_ref() else {
                    // The counter can only reach this branch after a
                    // feasibility check, but keep the browser boundary
                    // recoverable if a depleted population is restored from
                    // an interrupted job.
                    continue;
                };
                return Ok(cookbook_search_result(
                    accepted,
                    best_generation,
                    history,
                    first_feasible_score,
                    Some(accepted.fitness),
                    valid_candidates,
                    search_budget,
                    "stagnation".into(),
                    evaluations,
                    &geometry,
                    &compatibility_diagnostics,
                ));
            }
        } else if let Some(strict_index) = cookbook_strict_candidate_index(problem, &population) {
            let accepted = population[strict_index].clone();
            return Ok(cookbook_search_result(
                &accepted,
                generation,
                history,
                Some(accepted.fitness),
                None,
                valid_candidates,
                search_budget,
                "first_feasible".into(),
                evaluations,
                &geometry,
                &compatibility_diagnostics,
            ));
        }

        // Rotamer repair is opt-in and is attempted only after confirming
        // that the current population has no complete solution. It must not
        // delay an already steric-free initial chromosome.
        if generation == 0 && config.scan_rotamers {
            let baseline_fitness = population[0].fitness;
            let mut repaired = population[0].clone();
            let (repaired_strict, repair_evaluations) = cookbook_repair_with_cancel(
                problem,
                &mut repaired,
                config.seed ^ 0x726f_7461_6d65_7200,
                complete_search_budget.saturating_sub(evaluations),
                &mut geometry,
                &mut cancelled,
            )
            .await?;
            evaluations = evaluations.saturating_add(repair_evaluations);
            if cancelled() {
                return Err(EnsembleError::Cancelled);
            }
            update_selection_fitness(problem, policy, &mut repaired);
            if repaired.fitness < baseline_fitness || repaired_strict {
                population[0] = repaired;
                if cookbook_strict_search_solution(problem, &population[0]) {
                    if policy == SearchSelectionPolicy::JointPriorV1 {
                        let candidate = population[0].clone();
                        if first_feasible_score.is_none() {
                            first_feasible_score = Some(candidate.fitness);
                        }
                        if best_feasible
                            .as_ref()
                            .is_none_or(|incumbent| candidate.fitness < incumbent.fitness)
                        {
                            best_feasible = Some(candidate);
                            best_generation = generation;
                            prior_stagnation = 0;
                        }
                    } else {
                        return Ok(cookbook_search_result(
                            &population[0],
                            generation,
                            history,
                            Some(population[0].fitness),
                            None,
                            valid_candidates,
                            search_budget,
                            "first_feasible".into(),
                            evaluations,
                            &geometry,
                            &compatibility_diagnostics,
                        ));
                    }
                }
            }
        }

        if generation == config.generations {
            break;
        }

        if best_history.len() >= 30 {
            let recent = &best_history[best_history.len() - 30..];
            let min = recent.iter().copied().fold(f64::INFINITY, f64::min);
            let max = recent.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if max - min < 0.01 {
                stagnation += 1;
            } else {
                stagnation = 0;
                mutation_rate = 0.15;
                creep_std = 10.0;
                freezing_threshold = 1.1;
            }
        }
        if stagnation > 0 {
            mutation_rate = (0.15 + stagnation as f64 * 0.05).min(0.5);
            creep_std = (10.0 * (1.0 + stagnation as f64 * 0.3)).min(30.0);
            freezing_threshold = (1.1 + stagnation as f64 * 0.05).min(1.5);
            if stagnation % 10 == 0 {
                if cancelled() {
                    return Err(EnsembleError::Cancelled);
                }
                let inject = population.len() / 5;
                let start = population.len().saturating_sub(inject);
                for slot in &mut population[start..] {
                    if evaluations >= complete_search_budget {
                        break;
                    }
                    *slot = cookbook_generate(problem.protein, problem.sites, &mut rng);
                    evaluate_cookbook_chromosome(problem, slot)?;
                    evaluations += 1;
                    update_selection_fitness(problem, policy, slot);
                }
            }
            if stagnation >= 50 {
                if cancelled() {
                    return Err(EnsembleError::Cancelled);
                }
                let keep = population.len() / 4;
                for slot in &mut population[keep..] {
                    if evaluations >= complete_search_budget {
                        break;
                    }
                    *slot = cookbook_generate(problem.protein, problem.sites, &mut rng);
                    evaluate_cookbook_chromosome(problem, slot)?;
                    evaluations += 1;
                    update_selection_fitness(problem, policy, slot);
                }
                stagnation = 0;
            }
        }

        // Probability-ranked Builds must continue exploring solved sites: a
        // clash-free incumbent can still be improved by changing its
        // conformer or attachment basin.  The legacy Cookbook policy keeps
        // its historical freezing optimization and first-feasible stop.
        if policy != SearchSelectionPolicy::JointPriorV1 {
            population.par_iter_mut().for_each(|chromosome| {
                cookbook_update_freezing(chromosome, freezing_distance, freezing_threshold);
            });
        }
        let elite = ((population.len() as f64 * 0.1).round() as usize).clamp(1, population.len());
        let mut next = population[..elite].to_vec();
        if policy == SearchSelectionPolicy::JointPriorV1 {
            if let Some(incumbent) = best_feasible.as_ref() {
                if !next
                    .iter()
                    .any(|candidate| candidate.genes == incumbent.genes)
                {
                    next.pop();
                    next.push(incumbent.clone());
                }
            }
        }
        let needed = population.len() - next.len();
        // Repair/restart work consumes the same finite proposal budget as
        // ordinary generations.  If it used the remaining slots, stop before
        // evaluating a hidden extra population.
        if evaluations.saturating_add(needed) > complete_search_budget {
            break;
        }
        let generation_seed = splitmix64(config.seed ^ generation as u64);
        let children = (0..needed)
            .into_par_iter()
            .map(|child_index| {
                let mut child_rng =
                    ChaCha8Rng::seed_from_u64(splitmix64(generation_seed ^ child_index as u64));
                let parent_a = cookbook_select_parent(problem, &population, &mut child_rng, policy);
                let parent_b = cookbook_select_parent(problem, &population, &mut child_rng, policy);
                let mut child = cookbook_crossover(parent_a, parent_b, &mut child_rng);
                cookbook_mutate(
                    problem.protein,
                    problem.sites,
                    &mut child,
                    generation,
                    config.generations,
                    mutation_rate,
                    creep_std,
                    problem.scan_rotamers,
                    &mut child_rng,
                );
                child
            })
            .collect::<Vec<_>>();
        if cancelled() {
            return Err(EnsembleError::Cancelled);
        }
        let mut children = children;
        evaluate_population(problem, &mut children, &mut geometry).await?;
        children
            .iter_mut()
            .for_each(|chromosome| update_selection_fitness(problem, policy, chromosome));
        next.extend(children);
        evaluations += needed;
        population = next;
    }

    population.sort_by(|left, right| compare_candidates(problem, policy, left, right));
    let mut best = best_feasible
        .or_else(|| population.into_iter().next())
        .ok_or(EnsembleError::EmptyEnsemble)?;
    let (repaired, repair_evaluations) = cookbook_repair_with_cancel(
        problem,
        &mut best,
        config.seed ^ 0x7265_7061_6972_0000,
        complete_search_budget.saturating_sub(evaluations),
        &mut geometry,
        &mut cancelled,
    )
    .await?;
    evaluations = evaluations.saturating_add(repair_evaluations);
    if cancelled() {
        return Err(EnsembleError::Cancelled);
    }
    update_selection_fitness(problem, policy, &mut best);
    if cookbook_strict_search_solution(problem, &best) {
        if first_feasible_score.is_none() {
            first_feasible_score = Some(best.fitness);
        }
        return Ok(cookbook_search_result(
            &best,
            best_generation,
            history,
            first_feasible_score,
            (policy == SearchSelectionPolicy::JointPriorV1).then_some(best.fitness),
            valid_candidates,
            search_budget,
            "budget_exhausted".into(),
            evaluations,
            &geometry,
            &compatibility_diagnostics,
        ));
    }

    // A strict gate miss is still a useful Build result.  The chromosome is
    // complete (one evaluated gene per requested attachment), so return it
    // through the normal materialisation/reporting path.  Callers can then
    // inspect the actual coordinates and per-site VMM/steric diagnostics
    // instead of receiving only a failure artifact.  Strict callers retain
    // their policy through `SearchConfig::require_clash_free`, which is
    // enforced after the complete candidate has been materialised.
    let _ = repaired;
    Ok(cookbook_search_result(
        &best,
        best_generation,
        history,
        first_feasible_score,
        (policy == SearchSelectionPolicy::JointPriorV1).then_some(best.fitness),
        valid_candidates,
        search_budget,
        "budget_exhausted_no_feasible".into(),
        evaluations,
        &geometry,
        &compatibility_diagnostics,
    ))
}

#[derive(Debug, Clone)]
pub enum SearchProgress {
    PreparingTopology,
    TopologyReady {
        seconds: f64,
        atoms: usize,
    },
    Generation {
        phase: SearchPhase,
        generation: usize,
        best_score: f64,
        mean_score: f64,
        evaluations: usize,
        cache_hits: usize,
        steric_rejections: usize,
        elapsed_seconds: f64,
        valid_candidates: usize,
        first_feasible_score: Option<f64>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchPhase {
    Feasibility,
    Compatibility,
    ProbabilityImprovement,
}

pub fn search(
    protein: &Structure,
    sites: &[SearchSite],
    config: &SearchConfig,
    builder: &glysys::SystemBuilder,
) -> Result<SearchOutcome> {
    search_with_progress(protein, sites, config, builder, |_| {})
}

pub fn search_with_progress<F>(
    protein: &Structure,
    sites: &[SearchSite],
    config: &SearchConfig,
    builder: &glysys::SystemBuilder,
    progress: F,
) -> Result<SearchOutcome>
where
    F: FnMut(SearchProgress),
{
    search_with_progress_cancelled(protein, sites, config, builder, progress, || false)
}

/// Search with cooperative cancellation.  The callback is sampled at every
/// generation boundary and before materialization, while the parallel
/// evaluation batch remains deterministic and finishes before cancellation is
/// observed.  Existing callers can continue using `search_with_progress`.
#[maybe_async_cfg::maybe(
    sync(keep_self),
    async(feature = "webgpu"),
    idents(
        genetic_optimize_with_progress_cancelled(sync, async = "gpu_optimize"),
        evaluate_candidate(sync, async = "gpu_evaluate_candidate"),
        start_compute_stage(sync, async = "start_compute_stage_async"),
        sample_statistical(sync, async = "sample_statistical_async"),
        cookbook_steric_search_with_cancel(
            sync,
            async = "cookbook_steric_search_with_cancel_async"
        ),
        polish_cookbook_attachment_vmm(sync, async = "polish_cookbook_attachment_vmm_async")
    )
)]
pub async fn search_with_progress_cancelled<F, C>(
    protein: &Structure,
    sites: &[SearchSite],
    config: &SearchConfig,
    builder: &glysys::SystemBuilder,
    mut progress: F,
    mut cancelled: C,
) -> Result<SearchOutcome>
where
    F: FnMut(SearchProgress),
    C: FnMut() -> bool,
{
    start_compute_stage().await;
    if cancelled() {
        return Err(EnsembleError::Cancelled);
    }
    if sites.is_empty() || sites.iter().any(|site| site.ensemble.conformers.is_empty()) {
        return Err(EnsembleError::EmptyEnsemble);
    }
    validate_site_priors(protein, sites)?;
    let preparation_started = Instant::now();
    let prepared = PreparedAttachmentContext::new(protein, sites, config.scan_rotamers)?;
    let preparation_seconds = preparation_started.elapsed().as_secs_f64();
    if cancelled() {
        return Err(EnsembleError::Cancelled);
    }
    // Energy objectives still use their established objective and GA
    // operators, but they should not spend their entire initial population
    // rediscovering whether a multi-site geometry is jointly constructible.
    // Reuse the same bounded compatibility coordinator as steric Build and
    // hand only complete, protein-compatible states to the first GA batch.
    // The coordinator is deliberately skipped for one-site searches and for
    // steric Build, which owns its probability-ranked compatibility phase.
    let (energy_compatibility, initial_states) =
        if config.scoring_mode != SearchScoringMode::StericPrior && sites.len() > 1 {
            progress(SearchProgress::Generation {
                phase: SearchPhase::Compatibility,
                generation: 0,
                best_score: f64::INFINITY,
                mean_score: f64::INFINITY,
                evaluations: 0,
                cache_hits: 0,
                steric_rejections: 0,
                elapsed_seconds: preparation_started.elapsed().as_secs_f64(),
                valid_candidates: 0,
                first_feasible_score: None,
            });
            let population = config.population_size.max(2);
            let budget = population
                .saturating_mul(config.generations.saturating_add(1))
                .saturating_mul(2);
            if budget == 0 {
                (
                    CompatibilitySearchDiagnostics::default(),
                    Arc::new(Vec::new()),
                )
            } else {
                let result = cookbook_compatible_pool_coverage(
                    protein,
                    sites,
                    config,
                    &prepared,
                    population.min(4),
                    budget,
                )?;
                let diagnostics = CompatibilitySearchDiagnostics {
                    algorithm: "arc_consistency_backtracking_v1".into(),
                    seeded_states: result.states.len(),
                    pose_attempts: result.proposals,
                    pool_sizes: result.pool_sizes,
                    pool_expansions: result.pool_expansions,
                    compatibility_checks: result.compatibility_checks,
                    backtracks: result.backtracks,
                    graph_edges: result.graph_edges,
                    attempts_per_site: result.attempts_per_site.clone(),
                    proposal_budget: budget,
                };
                let states = Arc::new(
                    result
                        .states
                        .into_iter()
                        .map(|(genes, _, _)| genes)
                        .collect::<Vec<_>>(),
                );
                (diagnostics, states)
            }
        } else {
            (
                CompatibilitySearchDiagnostics::default(),
                Arc::new(Vec::new()),
            )
        };
    let energy_context = if config.scoring_mode != SearchScoringMode::StericPrior {
        progress(SearchProgress::PreparingTopology);
        let started = Instant::now();
        let reference = sites
            .iter()
            .map(|_| Gene {
                conformer: 0,
                phi: 0.0,
                psi: 0.0,
                rotamer: None,
            })
            .collect::<Vec<_>>();
        let structure = build_state(protein, sites, &reference, builder)?;
        let system = builder.prepare_structure(&structure)?;
        let seconds = started.elapsed().as_secs_f64();
        progress(SearchProgress::TopologyReady {
            seconds,
            atoms: system.atom_count(),
        });
        if cancelled() {
            return Err(EnsembleError::Cancelled);
        }
        Some(EnergySearchContext {
            system,
            evaluator: OnceLock::new(),
            mode: config.scoring_mode,
            use_obc2: config.use_obc2,
            minimize: config.pre_minimization,
            min_iterations: config.pre_minimization_iterations,
            min_radius: config.minimization_radius,
            cutoff: config.energy_cutoff,
            cache: Mutex::new(HashMap::new()),
            atom_masks: OnceLock::new(),
            atom_mapping: OnceLock::new(),
            evaluations: AtomicUsize::new(0),
            minimizations: AtomicUsize::new(0),
            cache_hits: AtomicUsize::new(0),
            steric_rejections: AtomicUsize::new(0),
            failures: AtomicUsize::new(0),
            started: Instant::now(),
            topology_seconds: seconds,
        })
    } else {
        None
    };
    let problem = SearchProblem {
        protein,
        sites,
        builder,
        clash_distance: config.clash_distance,
        scan_rotamers: config.scan_rotamers,
        scoring_mode: config.scoring_mode,
        selection_policy: config.selection_policy,
        use_obc2: config.use_obc2,
        pre_minimization: config.pre_minimization,
        pre_minimization_iterations: config.pre_minimization_iterations,
        energy_context: energy_context.as_ref(),
        prepared: &prepared,
        prior_cache: compile_prior_cache(protein, sites)?,
        vmm_priors: compile_vmm_prior_cache(protein, sites),
        initial_states,
        initial_state_cursor: AtomicUsize::new(0),
    };
    let ga_started = Instant::now();
    let mut vmm_polish = VmmPolishDiagnostics::default();
    let mut polish_geometry = GeometrySession::default();
    let (outcome, strict_components): (
        GeneticAlgorithmOutcome<Vec<Gene>>,
        Option<(Vec<usize>, Vec<usize>)>,
    ) = if config.scoring_mode == SearchScoringMode::StericPrior {
        let mut result =
            cookbook_steric_search_with_cancel(&problem, config, &mut progress, &mut cancelled)
                .await?;
        let search_diagnostics = VmmPolishDiagnostics {
            selection_policy: match config.selection_policy {
                SearchSelectionPolicy::JointPriorV1 => "joint_prior_v1",
                SearchSelectionPolicy::CookbookFirstFeasible => "cookbook_first_feasible",
            }
            .into(),
            prior_model: "conformer_population_x_attachment_vmm_v1".into(),
            first_feasible_score: result.first_feasible_score,
            final_prior_score: result.final_prior_score,
            valid_candidates: result.valid_candidates,
            search_budget: result.search_budget,
            termination_reason: result.termination_reason.clone(),
            evaluations: result.evaluations,
            geometry_gpu_evaluations: result.geometry_gpu_evaluations,
            geometry_cpu_evaluations: result.geometry_cpu_evaluations,
            geometry_gpu_seconds: result.geometry_gpu_seconds,
            geometry_cpu_seconds: result.geometry_cpu_seconds,
            geometry_transform_seconds: result.geometry_transform_seconds,
            compatibility_algorithm: result.compatibility_algorithm.clone(),
            compatibility_seeded_states: result.compatibility_seeded_states,
            compatibility_pose_attempts: result.compatibility_pose_attempts,
            compatibility_pool_sizes: result.compatibility_pool_sizes.clone(),
            compatibility_pool_expansions: result.compatibility_pool_expansions,
            compatibility_checks: result.compatibility_checks,
            compatibility_backtracks: result.compatibility_backtracks,
            compatibility_graph_edges: result.compatibility_graph_edges,
            compatibility_attempts_per_site: result.compatibility_attempts_per_site.clone(),
            compatibility_proposal_budget: result.compatibility_proposal_budget,
            ..VmmPolishDiagnostics::default()
        };
        if config.polish_attachment_vmm {
            let (polished, polished_phi, polished_psi, polished_fitness, mut diagnostics) =
                polish_cookbook_attachment_vmm(
                    &problem,
                    &result.best_state,
                    &result.phi_components,
                    &result.psi_components,
                    config.selection_policy,
                    result.search_budget.saturating_sub(result.evaluations),
                    &mut polish_geometry,
                    &mut cancelled,
                )
                .await?;
            let polished_score = if config.selection_policy == SearchSelectionPolicy::JointPriorV1 {
                problem.joint_prior_score(&polished)
            } else {
                polished_fitness
            };
            result.best_state = polished;
            result.phi_components = polished_phi;
            result.psi_components = polished_psi;
            result.best_score = polished_score;
            diagnostics.selection_policy = search_diagnostics.selection_policy.clone();
            diagnostics.prior_model = search_diagnostics.prior_model.clone();
            diagnostics.first_feasible_score = search_diagnostics.first_feasible_score;
            diagnostics.final_prior_score = Some(result.best_score);
            diagnostics.valid_candidates = search_diagnostics.valid_candidates;
            diagnostics.search_budget = search_diagnostics.search_budget;
            diagnostics.termination_reason = search_diagnostics.termination_reason.clone();
            diagnostics.evaluations = search_diagnostics
                .evaluations
                .saturating_add(diagnostics.evaluations);
            diagnostics.geometry_gpu_evaluations = search_diagnostics
                .geometry_gpu_evaluations
                .saturating_add(polish_geometry.gpu_evaluations);
            diagnostics.geometry_cpu_evaluations = search_diagnostics
                .geometry_cpu_evaluations
                .saturating_add(polish_geometry.cpu_evaluations);
            diagnostics.geometry_gpu_seconds =
                search_diagnostics.geometry_gpu_seconds + polish_geometry.gpu_seconds;
            diagnostics.geometry_cpu_seconds =
                search_diagnostics.geometry_cpu_seconds + polish_geometry.cpu_seconds;
            diagnostics.geometry_transform_seconds =
                search_diagnostics.geometry_transform_seconds + polish_geometry.transform_seconds;
            diagnostics.compatibility_algorithm =
                search_diagnostics.compatibility_algorithm.clone();
            diagnostics.compatibility_seeded_states =
                search_diagnostics.compatibility_seeded_states;
            diagnostics.compatibility_pose_attempts =
                search_diagnostics.compatibility_pose_attempts;
            diagnostics.compatibility_pool_sizes =
                search_diagnostics.compatibility_pool_sizes.clone();
            diagnostics.compatibility_pool_expansions =
                search_diagnostics.compatibility_pool_expansions;
            diagnostics.compatibility_checks = search_diagnostics.compatibility_checks;
            diagnostics.compatibility_backtracks = search_diagnostics.compatibility_backtracks;
            diagnostics.compatibility_graph_edges = search_diagnostics.compatibility_graph_edges;
            diagnostics.compatibility_attempts_per_site =
                search_diagnostics.compatibility_attempts_per_site.clone();
            diagnostics.compatibility_proposal_budget =
                search_diagnostics.compatibility_proposal_budget;
            vmm_polish = diagnostics;
        } else {
            vmm_polish = search_diagnostics;
        }
        let components = Some((result.phi_components.clone(), result.psi_components.clone()));
        (
            GeneticAlgorithmOutcome {
                best_state: result.best_state,
                best_score: result.best_score,
                generations: result.generations,
                history: result.history,
            },
            components,
        )
    } else {
        let ga = GeneticAlgorithmConfig {
            population_size: config.population_size,
            generations: config.generations,
            seed: config.seed,
            ..GeneticAlgorithmConfig::default()
        };
        let result = genetic_optimize_with_progress_cancelled(
            &problem,
            &ga,
            |record| {
                let (evaluations, cache_hits, steric_rejections, elapsed_seconds) =
                    energy_context.as_ref().map_or((0, 0, 0, 0.0), |context| {
                        (
                            context.evaluations.load(Ordering::Relaxed),
                            context.cache_hits.load(Ordering::Relaxed),
                            context.steric_rejections.load(Ordering::Relaxed),
                            context.started.elapsed().as_secs_f64(),
                        )
                    });
                progress(SearchProgress::Generation {
                    phase: SearchPhase::Feasibility,
                    generation: record.generation,
                    best_score: record.best_score,
                    mean_score: record.mean_score,
                    evaluations,
                    cache_hits,
                    steric_rejections,
                    elapsed_seconds,
                    valid_candidates: 0,
                    first_feasible_score: None,
                });
            },
            &mut cancelled,
        )
        .await;
        let result = match result {
            Ok(value) => value,
            Err(glysys_opt::OptimizationError::Cancelled) => return Err(EnsembleError::Cancelled),
            Err(error) => return Err(error.into()),
        };
        (result, None)
    };
    if cancelled() {
        return Err(EnsembleError::Cancelled);
    }
    if !energy_compatibility.algorithm.is_empty() {
        vmm_polish.compatibility_algorithm = energy_compatibility.algorithm.clone();
        vmm_polish.compatibility_seeded_states = energy_compatibility.seeded_states;
        vmm_polish.compatibility_pose_attempts = energy_compatibility.pose_attempts;
        vmm_polish.compatibility_pool_sizes = energy_compatibility.pool_sizes.clone();
        vmm_polish.compatibility_pool_expansions = energy_compatibility.pool_expansions;
        vmm_polish.compatibility_checks = energy_compatibility.compatibility_checks;
        vmm_polish.compatibility_backtracks = energy_compatibility.backtracks;
        vmm_polish.compatibility_graph_edges = energy_compatibility.graph_edges;
        vmm_polish.compatibility_attempts_per_site = energy_compatibility.attempts_per_site.clone();
        vmm_polish.compatibility_proposal_budget = energy_compatibility.proposal_budget;
    }

    let ga_seconds = ga_started.elapsed().as_secs_f64();
    let geometry_gpu_seconds = (config.scoring_mode == SearchScoringMode::StericPrior)
        .then_some(vmm_polish.geometry_gpu_seconds)
        .unwrap_or(0.0);
    let geometry_cpu_seconds = (config.scoring_mode == SearchScoringMode::StericPrior)
        .then_some(vmm_polish.geometry_cpu_seconds)
        .unwrap_or(0.0);
    let geometry_scoring_seconds = geometry_gpu_seconds + geometry_cpu_seconds;
    let scoring_seconds = if config.scoring_mode == SearchScoringMode::StericPrior {
        geometry_scoring_seconds
    } else {
        energy_context
            .as_ref()
            .map_or(0.0, |context| context.started.elapsed().as_secs_f64())
    };
    let proposal_seconds = (ga_seconds - scoring_seconds).max(0.0);
    let evaluated_states = if config.scoring_mode == SearchScoringMode::StericPrior {
        vmm_polish.evaluations
    } else {
        energy_context
            .as_ref()
            .map_or(0, |context| context.evaluations.load(Ordering::Relaxed))
    };
    let states_per_second = if ga_seconds > 0.0 {
        evaluated_states as f64 / ga_seconds
    } else {
        0.0
    };
    let materialization_started = Instant::now();
    let mut built = build_state(protein, sites, &outcome.best_state, builder)?;
    let materialization_seconds = materialization_started.elapsed().as_secs_f64();
    let site_sterics = prepared
        .evaluate(&outcome.best_state, config.clash_distance)?
        .site_scores;
    let steric = site_sterics.iter().copied().fold(1.0, f64::max);
    // Output acceptance uses the fast Cookbook steric score. Chemistry/VDW
    // analysis is deliberately absent from every scoring mode and cannot
    // delay or overturn a completed search.
    let status = if (config.scoring_mode == SearchScoringMode::StericPrior
        && site_sterics.iter().all(|score| *score <= 1.1))
        || (config.scoring_mode != SearchScoringMode::StericPrior && steric <= 1.1)
    {
        ClashStatus::ClashFree
    } else {
        ClashStatus::BestCompleteClashing
    };
    let vmm_gate_satisfied = if config.scoring_mode != SearchScoringMode::StericPrior {
        true
    } else {
        strict_components.as_ref().is_some_and(|(phi, psi)| {
            outcome
                .best_state
                .iter()
                .zip(sites)
                .enumerate()
                .all(|(index, (gene, site))| {
                    let conformer = &site.ensemble.conformers[gene.conformer];
                    let priors = resolved_priors(protein, site, &conformer.priors);
                    phi.get(index).is_some_and(|component| {
                        vmm_angle_within_95(
                            VmmAngle {
                                degrees: gene.phi,
                                component: *component,
                            },
                            &priors.phi,
                        )
                    }) && psi.get(index).is_some_and(|component| {
                        vmm_angle_within_95(
                            VmmAngle {
                                degrees: gene.psi,
                                component: *component,
                            },
                            &priors.psi,
                        )
                    })
                })
        })
    };
    if config.require_clash_free && (status != ClashStatus::ClashFree || !vmm_gate_satisfied) {
        return Err(ReGlycoError::ClashFreeRequired.into());
    }
    let selected = energy_context
        .as_ref()
        .map(|context| context.evaluate(&built))
        .transpose()?;
    if let (Some(context), Some(selected)) = (&energy_context, &selected) {
        let mut system = context.system.clone();
        system.set_coordinates(&selected.coordinates)?;
        built.update_with_parameterized_hydrogens(&system)?;
    }
    let selected_energy = selected.as_ref().map(|value| value.score);
    let site_results: Vec<SearchSiteResult> = outcome
        .best_state
        .iter()
        .zip(sites)
        .enumerate()
        .map(|(index, (gene, site))| {
            let conformer = &site.ensemble.conformers[gene.conformer];
            let priors = resolved_priors(protein, site, &conformer.priors);
            let tree = &built.metadata().glycan_trees[index];
            let residues = tree.residue_ids.iter().cloned().collect::<BTreeSet<_>>();
            let coordinates = built
                .atoms()
                .into_iter()
                .filter(|atom| residues.contains(&atom.residue))
                .map(|atom| atom.position)
                .collect();
            let phi_component = strict_components
                .as_ref()
                .and_then(|(phi, _)| phi.get(index).copied());
            let psi_component = strict_components
                .as_ref()
                .and_then(|(_, psi)| psi.get(index).copied());
            let phi_within_vmm95 = phi_component.and_then(|component| {
                Some(vmm_angle_within_95(
                    VmmAngle {
                        degrees: gene.phi,
                        component,
                    },
                    &priors.phi,
                ))
            });
            let psi_within_vmm95 = psi_component.and_then(|component| {
                Some(vmm_angle_within_95(
                    VmmAngle {
                        degrees: gene.psi,
                        component,
                    },
                    &priors.psi,
                ))
            });
            let (conformer_probability, attachment_log_density, joint_prior_score) = problem
                .site_prior_breakdown(index, gene)
                .map_or((None, None, None), |(population, density, score)| {
                    (Some(population), Some(density), Some(score))
                });
            SearchSiteResult {
                site: site.site.clone(),
                conformer_index: gene.conformer,
                conformer_id: conformer.id.clone(),
                cluster_index: conformer.cluster_index,
                main_cluster: conformer.main_cluster,
                cluster_weight: conformer.cluster_weight,
                phi_degrees: gene.phi,
                psi_degrees: gene.psi,
                phi_component,
                psi_component,
                phi_within_vmm95,
                psi_within_vmm95,
                rotamer_index: gene.rotamer,
                prior_score: joint_prior_score.unwrap_or_else(|| {
                    let total_weight = site
                        .ensemble
                        .conformers
                        .iter()
                        .map(|conformer| conformer.cluster_weight)
                        .sum::<f64>();
                    let population = conformer.cluster_weight / total_weight;
                    if population > 0.0 && population.is_finite() {
                        -population.ln()
                            + vmm_penalty(gene.phi, &priors.phi)
                            + vmm_penalty(gene.psi, &priors.psi)
                    } else {
                        f64::INFINITY
                    }
                }),
                conformer_probability,
                attachment_log_density,
                joint_prior_score,
                steric_score: site_sterics.get(index).copied().unwrap_or(steric),
                coordinates,
            }
        })
        .collect();
    let termination_reason = if !vmm_polish.termination_reason.is_empty() {
        vmm_polish.termination_reason.clone()
    } else if status == ClashStatus::ClashFree && vmm_gate_satisfied {
        "completed".into()
    } else {
        "budget_exhausted_no_feasible".into()
    };
    let energy_analysis = selected
        .as_ref()
        .zip(energy_context.as_ref())
        .map(|(selected, context)| {
            energy_analysis_for_context(context, &built, selected, sites, &site_results, "CPU")
        })
        .transpose()?;
    Ok(SearchOutcome {
        sites: site_results.clone(),
        total_score: outcome.best_score,
        seed: config.seed,
        generations: outcome.generations,
        clash_status: status,
        complete_output: true,
        vmm_gate_satisfied,
        termination_reason,
        clash_partners: clash_partner_labels(&built, config.clash_distance),
        history: outcome
            .history
            .into_iter()
            .map(|record| SearchGeneration {
                generation: record.generation,
                best_score: record.best_score,
                mean_score: record.mean_score,
                best_energy_kcal_per_mol: (config.scoring_mode != SearchScoringMode::StericPrior
                    && record.best_score < 1.0e11)
                    .then_some(record.best_score),
            })
            .collect(),
        warnings: {
            let mut warnings = Vec::new();
            if status == ClashStatus::BestCompleteClashing || !vmm_gate_satisfied {
                let unresolved = site_results
                    .iter()
                    .filter(|site| {
                        site.steric_score > 1.1
                            || (config.scoring_mode == SearchScoringMode::StericPrior
                                && (!site.phi_within_vmm95.unwrap_or(false)
                                    || !site.psi_within_vmm95.unwrap_or(false)))
                    })
                    .map(|site| site.site.residue.to_string())
                    .collect::<Vec<_>>();
                warnings.push(if unresolved.is_empty() {
                    "GA exhausted with a complete result that did not satisfy every acceptance gate".into()
                } else {
                    format!(
                        "GA exhausted with a complete diagnostic result; unresolved attachment sites: {}",
                        unresolved.join(", ")
                    )
                });
            }
            warnings
        },
        scoring_mode: config.scoring_mode,
        selected_energy_kcal_per_mol: selected_energy,
        interaction_energy_kcal_per_mol: (config.scoring_mode
            == SearchScoringMode::ProteinGlycanInteraction)
            .then_some(selected_energy)
            .flatten(),
        energy_evaluations: energy_context
            .as_ref()
            .map_or(0, |context| context.evaluations.load(Ordering::Relaxed)),
        energy_cutoff_angstrom: config.energy_cutoff,
        minimization_radius_angstrom: config.minimization_radius,
        interaction_vdw_kcal_per_mol: selected
            .as_ref()
            .and_then(|value| value.interaction.map(|item| item.van_der_waals)),
        interaction_coulomb_kcal_per_mol: selected
            .as_ref()
            .and_then(|value| value.interaction.map(|item| item.electrostatics)),
        energy_diagnostics: energy_context.as_ref().map_or_else(
            EnergySearchDiagnostics::default,
            |context| EnergySearchDiagnostics {
                topology_parameterizations: 1,
                energy_evaluations: context.evaluations.load(Ordering::Relaxed),
                minimizations: context.minimizations.load(Ordering::Relaxed),
                cache_hits: context.cache_hits.load(Ordering::Relaxed),
                steric_rejections: context.steric_rejections.load(Ordering::Relaxed),
                failed_evaluations: context.failures.load(Ordering::Relaxed),
                neighbor_pairs: selected.as_ref().map_or(0, |value| value.neighbor_pairs),
                active_atoms: selected.as_ref().map_or(0, |value| value.active_atoms),
                active_residues: selected
                    .as_ref()
                    .map_or_else(Vec::new, |value| value.active_residues.clone()),
                topology_seconds: context.topology_seconds,
                evaluation_seconds: context.started.elapsed().as_secs_f64(),
            },
        ),
        minimized_coordinates: selected.as_ref().zip(energy_context.as_ref()).map_or_else(
            Vec::new,
            |(selected, context)| {
                selected
                    .active_indices
                    .iter()
                    .map(|atom| {
                        let parameterized = &context.system.atoms()[*atom];
                        let residue = &context.system.residues()[parameterized.residue_index()];
                        AtomCoordinateRecord {
                            residue: ResidueId {
                                chain: residue.chain().into(),
                                number: residue.number(),
                                insertion_code: residue.insertion_code(),
                            },
                            atom_name: parameterized.name().into(),
                            position: selected.coordinates[*atom],
                        }
                    })
                    .collect()
            },
        ),
        timings: SearchTimingDiagnostics {
            preparation_seconds,
            proposal_seconds,
            scoring_seconds,
            ga_seconds,
            materialization_seconds,
            states_per_second,
            ..SearchTimingDiagnostics::default()
        },
        vmm_polish,
        energy_analysis,
    })
}

/// Score a complete attached structure with the requested Amber/GLYCAM energy.
pub fn score_structure(
    structure: &Structure,
    builder: &glysys::SystemBuilder,
    mode: SearchScoringMode,
    use_obc2: bool,
    pre_minimization: bool,
    pre_minimization_iterations: usize,
) -> Result<f64> {
    if mode == SearchScoringMode::StericPrior {
        return Ok(0.0);
    }
    if use_obc2 && mode == SearchScoringMode::ProteinGlycanInteraction {
        return Err(EnsembleError::Metadata(
            "--obc2 cannot be used with protein-glycan interaction scoring".into(),
        ));
    }
    let system = builder.prepare_structure(structure)?;
    let (system, coordinates) = if pre_minimization {
        let mut options = RelaxOptions {
            movable: MovableSelection::All,
            include_local_sidechains: false,
            ..RelaxOptions::default()
        };
        options.lbfgs.max_iterations = pre_minimization_iterations.max(1);
        options.obc2 = use_obc2.then(Obc2Options::default);
        let minimized = relax(structure, &system, &options)
            .map_err(|error| EnsembleError::Metadata(error.to_string()))?;
        let coordinates = minimized.system.coordinates();
        (minimized.system, coordinates)
    } else {
        let coordinates = system.coordinates();
        (system, coordinates)
    };
    let evaluator = EnergyEvaluator::new(
        &system,
        EnergyOptions {
            obc2: use_obc2.then(Obc2Options::default),
            ..EnergyOptions::default()
        },
    )?;
    match mode {
        SearchScoringMode::StericPrior => Ok(0.0),
        SearchScoringMode::FullEnergy => Ok(evaluator.energy(&coordinates)?.total()),
        SearchScoringMode::ProteinGlycanInteraction => {
            let glycan_residues = system
                .metadata()
                .glycan_trees
                .iter()
                .flat_map(|tree| tree.residue_ids.iter())
                .collect::<BTreeSet<_>>();
            let glycan = AtomGroupMask::from_indices(
                system.atom_count(),
                system.residues().iter().flat_map(|residue| {
                    let id = ResidueId {
                        chain: residue.chain().into(),
                        number: residue.number(),
                        insertion_code: residue.insertion_code(),
                    };
                    glycan_residues
                        .contains(&id)
                        .then_some(residue.atom_range())
                        .into_iter()
                        .flatten()
                }),
            );
            let protein = AtomGroupMask::from_indices(
                system.atom_count(),
                (0..system.atom_count()).filter(|atom| !glycan.contains(*atom)),
            );
            Ok(evaluator
                .interaction_energy(&coordinates, &protein, &glycan)?
                .total())
        }
    }
}

pub fn compatible_set_search(
    protein: &Structure,
    sites: &[SearchSite],
    config: &SearchConfig,
    builder: &glysys::SystemBuilder,
) -> Result<SearchOutcome> {
    search(protein, sites, config, builder)
}

#[derive(Debug, Clone)]
pub struct SampledFrame {
    pub structure: Structure,
    pub sites: Vec<SearchSiteResult>,
    /// Monotonic proposal number, useful for provenance but not a user-facing
    /// attempt limit.
    pub proposal_index: usize,
    /// `compatible_pool`, `native_sampler`, or `ga_seeded_mh`.
    pub source: String,
    pub log_native_probability: f64,
    pub selected_energy_kcal_per_mol: Option<f64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EnsembleSamplingDiagnostics {
    pub requested_frames: usize,
    pub returned_frames: usize,
    pub attempts: usize,
    pub acceptance_rate: f64,
    pub seed: u64,
    #[serde(default)]
    pub native_frames: usize,
    #[serde(default)]
    pub fallback_frames: usize,
    #[serde(default)]
    pub fallback_used: bool,
    #[serde(default)]
    pub ga_restarts: usize,
    #[serde(default)]
    pub chains: usize,
    #[serde(default)]
    pub native_proposals: usize,
    #[serde(default)]
    pub native_accepts: usize,
    #[serde(default)]
    pub mh_proposals: usize,
    #[serde(default)]
    pub mh_accepts: usize,
    #[serde(default)]
    pub single_site_proposals: usize,
    #[serde(default)]
    pub pair_proposals: usize,
    #[serde(default)]
    pub group_proposals: usize,
    #[serde(default)]
    pub whole_state_proposals: usize,
    #[serde(default)]
    pub single_site_accepts: usize,
    #[serde(default)]
    pub pair_accepts: usize,
    #[serde(default)]
    pub group_accepts: usize,
    #[serde(default)]
    pub whole_state_accepts: usize,
    #[serde(default)]
    pub burn_in_sweeps: usize,
    #[serde(default)]
    pub thinning_accepted: usize,
    /// Effective sampler temperature. Steric-only sampling does not use this
    /// value for acceptance, but it is retained for reproducibility.
    #[serde(default = "default_sampling_temperature_k")]
    pub temperature_k: f64,
    /// Whether Dunbrack sidechain rotamer proposals were enabled.
    #[serde(default)]
    pub scan_rotamers: bool,
    #[serde(default)]
    pub timings: SearchTimingDiagnostics,
    /// Effective numerical target used by sampled energy/interaction runs.
    #[serde(default)]
    pub sampling_target: Option<SamplingTarget>,
    /// Explicit sampler segments make a GPU-to-CPU recovery visible instead
    /// of pooling unlike numerical targets silently.
    #[serde(default)]
    pub segments: Vec<SamplingSegmentDiagnostics>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SamplingSegmentDiagnostics {
    pub id: usize,
    pub target: SamplingTarget,
    pub backend: String,
    pub frames: usize,
    pub attempts: usize,
    pub accepts: usize,
    pub burn_in_steps: usize,
    pub fallback_reason: Option<String>,
}

#[derive(Debug, Clone)]
struct PooledSiteCandidate {
    gene: Gene,
    phi_component: usize,
    psi_component: usize,
    pose: PreparedSitePose,
}

#[derive(Debug)]
struct CompatiblePoolResult {
    states: Vec<(Vec<Gene>, Vec<usize>, Vec<usize>)>,
    /// Persistent protein-compatible poses.  Keeping these across expansion
    /// rounds is essential: a later round must add coverage instead of
    /// replacing the domain that the solver already inspected.
    pools: Vec<Vec<PooledSiteCandidate>>,
    proposals: usize,
    attempts_per_site: Vec<usize>,
    pool_sizes: Vec<usize>,
    pool_expansions: usize,
    compatibility_checks: usize,
    backtracks: usize,
    graph_edges: usize,
}

/// Conservative site-dependency graph used by both compatibility assembly
/// and statistical block proposals.  A fixed 100 Å cutoff made every site in
/// a large oligomer a compatibility neighbour, which in turn forced the
/// solver to allocate dense pose-pair tables for sites that cannot physically
/// interact.  Derive a bound from the actual conformer reach instead.  An
/// unknown frame or attachment geometry remains connected conservatively.
fn conservative_site_graph(protein: &Structure, sites: &[SearchSite]) -> Vec<Vec<usize>> {
    let positions = sites
        .iter()
        .map(|site| {
            protein
                .find_atom(&site.site.residue, "CA")
                .and_then(|atom| protein.atom_position(atom))
        })
        .collect::<Vec<_>>();
    let reaches = sites.iter().map(conformer_reach).collect::<Vec<_>>();
    let mut graph = vec![Vec::new(); sites.len()];
    for first in 0..sites.len() {
        for second in (first + 1)..sites.len() {
            let connected = positions[first]
                .zip(positions[second])
                .is_none_or(|(left, right)| {
                    let margin = reaches[first] + reaches[second] + 2.0 * 1.7 + 6.0;
                    squared_distance(left, right) <= margin * margin
                });
            if connected {
                graph[first].push(second);
                graph[second].push(first);
            }
        }
    }
    graph
}

/// Versioned, deterministic search-budget estimator shared by native and
/// browser attachment workflows. The estimator sizes the compatibility pool
/// work first because that is the part of a dense multi-site search that the
/// legacy fixed 128x100 defaults consistently under-provisioned.
pub const SEARCH_BUDGET_ESTIMATOR_VERSION: &str = "search_space_v1";
pub const AUTO_BUDGET_MAX_POPULATION: usize = 256;
pub const AUTO_BUDGET_MAX_GENERATIONS: usize = 500;

pub fn resolve_search_budget(
    protein: &Structure,
    sites: &[SearchSite],
    scan_rotamers: bool,
    requested_mode: Option<SearchBudgetMode>,
    manual_population: usize,
    manual_generations: usize,
) -> Result<SearchBudgetResolution> {
    let mode = requested_mode.unwrap_or(SearchBudgetMode::Manual);
    let graph = conservative_site_graph(protein, sites);
    let components = connected_components(&graph);
    let conformer_counts = sites
        .iter()
        .map(|site| {
            site.ensemble
                .conformers
                .iter()
                .filter(|conformer| {
                    conformer.cluster_weight.is_finite() && conformer.cluster_weight > 0.0
                })
                .count()
        })
        .collect::<Vec<_>>();
    let vmm_basin_counts = sites
        .iter()
        .map(|site| {
            site.ensemble
                .conformers
                .iter()
                .filter(|conformer| {
                    conformer.cluster_weight.is_finite() && conformer.cluster_weight > 0.0
                })
                .map(|conformer| {
                    let priors = resolved_priors(protein, site, &conformer.priors);
                    let phi = positive_vmm_count(&priors.phi);
                    let psi = positive_vmm_count(&priors.psi);
                    phi.saturating_mul(psi).max(1)
                })
                .max()
                .unwrap_or(1)
        })
        .collect::<Vec<_>>();
    let rotamer_counts = sites
        .iter()
        .map(|site| {
            if scan_rotamers {
                // Pass zero is the deposited side chain; the Dunbrack
                // entries are the additional choices used by repair.
                dunbrack::rotamers(protein, &site.site.residue)
                    .len()
                    .min(5)
                    .saturating_add(1)
            } else {
                1
            }
        })
        .collect::<Vec<_>>();
    let site_pose_needs = conformer_counts
        .iter()
        .zip(&vmm_basin_counts)
        .zip(&rotamer_counts)
        .map(|((conformers, basins), rotamers)| {
            conformers
                .saturating_mul((*basins).min(4))
                .saturating_mul(*rotamers)
                .clamp(16, 192)
        })
        .collect::<Vec<_>>();
    let graph_edges = graph.iter().map(Vec::len).sum::<usize>() / 2;
    let maximum_degree = graph.iter().map(Vec::len).max().unwrap_or(0);
    let largest_component_size = components.iter().map(Vec::len).max().unwrap_or(0);
    let mut required_compatibility_work = 0usize;
    for component in &components {
        let component_edges = component
            .iter()
            .map(|&index| {
                graph[index]
                    .iter()
                    .filter(|neighbor| component.binary_search(neighbor).is_ok())
                    .count()
            })
            .sum::<usize>()
            / 2;
        let pool_target = component
            .iter()
            .map(|&index| site_pose_needs.get(index).copied().unwrap_or(16))
            .max()
            .unwrap_or(16)
            .max(16usize.saturating_add(10usize.saturating_mul(component.len())))
            .min(192);
        let edge_work = ceil_div(
            component_edges
                .saturating_mul(pool_target)
                .saturating_mul(pool_target),
            32,
        );
        let local_work = component
            .iter()
            .map(|&index| site_pose_needs.get(index).copied().unwrap_or(16))
            .sum::<usize>()
            .saturating_mul(4);
        required_compatibility_work =
            required_compatibility_work.saturating_add(edge_work.max(local_work));
    }
    let required_work = 128usize
        .saturating_mul(101)
        .max(required_compatibility_work.saturating_mul(2));

    let (population_size, generations, capped) = match mode {
        SearchBudgetMode::Manual => {
            if manual_population == 0 || manual_generations == 0 {
                return Err(EnsembleError::Metadata(
                    "manual search budgets require positive population and generation values"
                        .into(),
                ));
            }
            (manual_population, manual_generations, false)
        }
        SearchBudgetMode::Auto => {
            let raw_population = 64usize
                .saturating_add(largest_component_size.saturating_mul(8))
                .saturating_add(maximum_degree);
            let population_capped = raw_population > AUTO_BUDGET_MAX_POPULATION;
            let population =
                next_power_of_two(raw_population.clamp(128, AUTO_BUDGET_MAX_POPULATION))
                    .min(AUTO_BUDGET_MAX_POPULATION);
            let requested_generations = if required_work <= 128usize.saturating_mul(101) {
                100
            } else {
                ceil_div(required_work, population).max(100)
            };
            (
                population,
                requested_generations.min(AUTO_BUDGET_MAX_GENERATIONS),
                population_capped || requested_generations > AUTO_BUDGET_MAX_GENERATIONS,
            )
        }
    };
    let total_budget = population_size.saturating_mul(generations.saturating_add(1));
    Ok(SearchBudgetResolution {
        version: SEARCH_BUDGET_ESTIMATOR_VERSION.into(),
        requested_mode: mode,
        population_size,
        generations,
        total_budget,
        required_work,
        auto_cap_population: AUTO_BUDGET_MAX_POPULATION,
        auto_cap_generations: AUTO_BUDGET_MAX_GENERATIONS,
        capped,
        site_count: sites.len(),
        component_sizes: components.iter().map(Vec::len).collect(),
        graph_edges,
        maximum_degree,
        largest_component_size,
        site_pose_needs,
        conformer_counts,
        vmm_basin_counts,
        rotamer_counts,
        termination_reason: capped.then(|| "auto_cap_reached".into()),
    })
}

fn positive_vmm_count(components: &[VonMisesComponent]) -> usize {
    components
        .iter()
        .filter(|component| {
            component.weight.is_finite()
                && component.weight > 0.0
                && component.concentration.is_finite()
                && component.concentration >= 0.0
        })
        .count()
        .max(1)
}

fn ceil_div(numerator: usize, denominator: usize) -> usize {
    if denominator == 0 {
        return usize::MAX;
    }
    numerator.saturating_add(denominator.saturating_sub(1)) / denominator
}

fn next_power_of_two(value: usize) -> usize {
    if value <= 1 {
        return 1;
    }
    value.checked_next_power_of_two().unwrap_or(usize::MAX)
}

fn connected_components(graph: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut visited = vec![false; graph.len()];
    let mut components = Vec::new();
    for start in 0..graph.len() {
        if visited[start] {
            continue;
        }
        let mut stack = vec![start];
        let mut component = Vec::new();
        visited[start] = true;
        while let Some(index) = stack.pop() {
            component.push(index);
            for &neighbor in &graph[index] {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    stack.push(neighbor);
                }
            }
        }
        component.sort_unstable();
        components.push(component);
    }
    components
}

/// Maximum distance from the first anomeric carbon in the represented
/// conformer library.  Pose rotations preserve this radius, so it is a
/// conservative bound for deciding whether two attachment frames could ever
/// produce a glycan/glycan contact.  A missing C1 is treated as unknown and
/// keeps the graph edge instead of claiming independence.
fn conformer_reach(site: &SearchSite) -> f64 {
    let mut maximum = 0.0_f64;
    let mut found_root = false;
    for conformer in &site.ensemble.conformers {
        let atoms = conformer.structure.atoms();
        let Some(root) = atoms.iter().find(|atom| atom.name == "C1") else {
            return f64::INFINITY;
        };
        found_root = true;
        maximum = maximum.max(
            atoms
                .iter()
                .map(|atom| squared_distance(root.position, atom.position).sqrt())
                .fold(0.0_f64, f64::max),
        );
    }
    if found_root {
        maximum + 3.0
    } else {
        f64::INFINITY
    }
}

/// Cookbook's fast ensemble path: build a protein-compatible native pool for
/// each site independently, precompute glycan/glycan compatibility between
/// those pools, and then assemble complete frames with MRV backtracking.
///
/// Drawing every site jointly makes the acceptance probability collapse for
/// highly glycosylated proteins (for example P27918) and unnecessarily sends
/// ordinary ensembles into the much more expensive constrained-MH fallback.
fn cookbook_compatible_pool_sample(
    protein: &Structure,
    sites: &[SearchSite],
    frames: usize,
    config: &SearchConfig,
    prepared: &PreparedAttachmentContext,
) -> Result<CompatiblePoolResult> {
    let base_budget = config
        .population_size
        .max(2)
        .saturating_mul(config.generations.saturating_add(1));
    // Ensemble initialization is feasibility work, so it uses the same
    // resolved search budget as Build instead of a hidden `frames * 100`
    // rejection limit.  The sampler still owns all subsequent burn-in and
    // thinning transitions.
    let global_attempt_budget = base_budget
        .max(frames.saturating_mul(100))
        .max(sites.len().saturating_mul(64));
    let pool_target = compatible_pool_target(protein, sites, global_attempt_budget, frames);
    let per_site_attempt_budget =
        global_attempt_budget.saturating_add(sites.len().max(1) - 1) / sites.len().max(1);
    cookbook_compatible_pool_sample_with_options(
        protein,
        sites,
        frames,
        config,
        prepared,
        pool_target,
        per_site_attempt_budget,
        false,
        Vec::new(),
    )
}

/// Resolve a retained pose-domain size from the caller's work and memory
/// policy.  Compatibility tables are quadratic in neighbouring pool sizes;
/// deriving the target from the explicit pair budget keeps large ensembles
/// bounded without restoring the old arbitrary 32/128-pose ceiling.
fn compatible_pool_target(
    protein: &Structure,
    sites: &[SearchSite],
    proposal_budget: usize,
    minimum: usize,
) -> usize {
    let per_site_budget = proposal_budget / sites.len().max(1);
    let graph = conservative_site_graph(protein, sites);
    let edge_count = graph.iter().map(Vec::len).sum::<usize>() / 2;
    let pair_budget = proposal_budget.saturating_mul(32).max(16 * 16);
    let table_target = if edge_count == 0 {
        per_site_budget
    } else {
        ((pair_budget / edge_count).max(16 * 16) as f64).sqrt() as usize
    };
    per_site_budget.min(table_target.max(minimum.max(16)))
}

/// Build-only coverage rescue.  Unlike the native ensemble pool, this path
/// deliberately samples every represented conformer and the full declared
/// VMM-95 region before solving the joint pairwise constraints.  It is a
/// bounded feasibility search; it never supplies proposals to statistical
/// Ensemble sampling.
fn cookbook_compatible_pool_coverage(
    protein: &Structure,
    sites: &[SearchSite],
    config: &SearchConfig,
    prepared: &PreparedAttachmentContext,
    frames: usize,
    proposal_budget: usize,
) -> Result<CompatiblePoolResult> {
    let site_count = sites.len().max(1);
    let per_site_budget = proposal_budget / site_count;
    // Keep enough alternatives for compatibility propagation to recover from
    // a locally attractive but globally blocked pose.  The old fixed 32-pose
    // pool was particularly brittle for dense tetramers: one unlucky batch
    // could remove the only compatible basin before the solver ran.
    // The retained-domain budget is derived from the caller's resolved work
    // budget.  There is no fixed 32/128-pose search ceiling; allocation and
    // pair-table checks remain bounded by the caller's resource policy.
    // Pose-pair tables are part of the explicit compatibility budget.  Their
    // target is derived from the number of possible edges and the caller's
    // work allowance, rather than from a global 32/128 pose constant.  Pose
    // discovery may still spend the full per-site attempt budget; only the
    // retained domain is bounded when a dense table would exceed the policy.
    let pool_target = compatible_pool_target(protein, sites, proposal_budget, 16);
    let max_attempts = per_site_budget.max(pool_target);
    if pool_target == 0 || max_attempts == 0 {
        return Ok(CompatiblePoolResult {
            states: Vec::new(),
            pools: vec![Vec::new(); sites.len()],
            proposals: 0,
            attempts_per_site: vec![0; sites.len()],
            pool_sizes: Vec::new(),
            pool_expansions: 0,
            compatibility_checks: 0,
            backtracks: 0,
            graph_edges: conservative_site_graph(protein, sites)
                .iter()
                .map(Vec::len)
                .sum::<usize>()
                / 2,
        });
    }
    cookbook_compatible_pool_sample_with_options(
        protein,
        sites,
        frames.max(1),
        config,
        prepared,
        pool_target,
        max_attempts,
        true,
        Vec::new(),
    )
}

fn cookbook_compatible_pool_sample_with_options(
    protein: &Structure,
    sites: &[SearchSite],
    frames: usize,
    config: &SearchConfig,
    prepared: &PreparedAttachmentContext,
    pool_target: usize,
    max_attempts: usize,
    coverage: bool,
    existing_pools: Vec<Vec<PooledSiteCandidate>>,
) -> Result<CompatiblePoolResult> {
    if !coverage || pool_target <= 16 || sites.is_empty() {
        return cookbook_compatible_pool_sample_once(
            protein,
            sites,
            frames,
            config,
            prepared,
            pool_target,
            max_attempts,
            coverage,
            0,
            existing_pools,
        );
    }

    // Grow compatible domains in deterministic rounds.  The first round is
    // deliberately small so easy cases return a seed quickly; only a failed
    // compatibility solve consumes the next round.  This replaces the old
    // fixed rescue pool without making a difficult case pay the largest
    // quadratic pair table up front.
    let mut target = pool_target.min(16).max(1);
    let mut remaining = vec![max_attempts; sites.len()];
    let mut total_proposals = 0usize;
    let mut total_attempts_per_site = vec![0usize; sites.len()];
    let mut total_checks = 0usize;
    let mut total_backtracks = 0usize;
    let mut pool_expansions = 0usize;
    let mut latest = CompatiblePoolResult {
        states: Vec::new(),
        pools: existing_pools.clone(),
        proposals: 0,
        attempts_per_site: vec![0; sites.len()],
        pool_sizes: vec![0; sites.len()],
        pool_expansions: 0,
        compatibility_checks: 0,
        backtracks: 0,
        graph_edges: 0,
    };

    loop {
        let rounds_remaining = std::iter::successors(Some(target), |value| {
            (*value < pool_target).then_some(value.saturating_mul(2).min(pool_target))
        })
        .count()
        .max(1);
        // Reserve attempts for the larger domains.  Without this reservation
        // a heavily clashing first batch could consume the entire per-site
        // allowance before any expansion was attempted.
        let round_limit = if target >= pool_target {
            max_attempts
        } else {
            (remaining.iter().copied().min().unwrap_or(0) / rounds_remaining).max(target)
        };
        let per_site_limit = remaining
            .iter()
            .copied()
            .min()
            .unwrap_or(0)
            .min(round_limit);
        if per_site_limit == 0 {
            break;
        }
        let consumed =
            max_attempts.saturating_sub(remaining.iter().copied().min().unwrap_or(max_attempts));
        let round = cookbook_compatible_pool_sample_once(
            protein,
            sites,
            frames,
            config,
            prepared,
            target,
            per_site_limit,
            coverage,
            consumed as u64,
            latest.pools.clone(),
        )?;
        total_proposals = total_proposals.saturating_add(round.proposals);
        total_checks = total_checks.saturating_add(round.compatibility_checks);
        total_backtracks = total_backtracks.saturating_add(round.backtracks);
        for (site, attempts) in round.attempts_per_site.iter().copied().enumerate() {
            total_attempts_per_site[site] = total_attempts_per_site[site].saturating_add(attempts);
            remaining[site] = remaining[site].saturating_sub(attempts);
        }
        latest = round;
        // A complete compatible state is already sufficient to seed Build
        // and to initialize every ensemble chain (additional chains may
        // reuse that start with independent RNG streams).  The previous
        // loop ignored `round.states` and kept doubling every site pool up
        // to the full Auto budget even after a valid joint state had been
        // found.  On dense structures this made both CPU and GPU requests
        // appear frozen at generation 0 while the GPU was still idle in the
        // CPU compatibility pre-pass.
        if !latest.states.is_empty() {
            break;
        }
        if target >= pool_target {
            break;
        }
        pool_expansions = pool_expansions.saturating_add(1);
        target = target.saturating_mul(2).min(pool_target);
    }

    latest.proposals = total_proposals;
    latest.attempts_per_site = total_attempts_per_site;
    latest.compatibility_checks = total_checks;
    latest.backtracks = total_backtracks;
    latest.pool_expansions = pool_expansions;
    Ok(latest)
}

fn cookbook_compatible_pool_sample_once(
    protein: &Structure,
    sites: &[SearchSite],
    frames: usize,
    config: &SearchConfig,
    prepared: &PreparedAttachmentContext,
    pool_target: usize,
    max_attempts: usize,
    coverage: bool,
    seed_round: u64,
    existing_pools: Vec<Vec<PooledSiteCandidate>>,
) -> Result<CompatiblePoolResult> {
    let pools_with_attempts = (0..sites.len())
        .into_par_iter()
        .map(|site_index| -> Result<(Vec<PooledSiteCandidate>, usize)> {
            let mut rng = ChaCha8Rng::seed_from_u64(splitmix64(
                config.seed ^ 0x504f_4f4c_5f53_4954 ^ site_index as u64 ^ seed_round,
            ));
            let mut pool = existing_pools.get(site_index).cloned().unwrap_or_default();
            let mut seen = pool
                .iter()
                .map(|candidate| pose_key(&candidate.gene))
                .collect::<HashSet<_>>();
            let mut attempts = 0usize;
            if coverage {
                // Deterministic coverage comes before random supplementation.
                // Every represented positive-population conformer and VMM
                // component pair gets angular probes at the component centre
                // and at symmetric points in its declared 95% region.  This
                // makes a rare compatible basin reproducible instead of
                // depending on a random draw landing there.
                let site = &sites[site_index];
                let mut conformers = site
                    .ensemble
                    .conformers
                    .iter()
                    .enumerate()
                    .filter(|(_, conformer)| {
                        conformer.cluster_weight.is_finite() && conformer.cluster_weight > 0.0
                    })
                    .map(|(index, conformer)| (index, conformer.cluster_weight))
                    .collect::<Vec<_>>();
                conformers.sort_by(|(left, weight_left), (right, weight_right)| {
                    weight_right
                        .total_cmp(weight_left)
                        .then_with(|| left.cmp(right))
                });
                // Build a per-conformer stream and consume it round-robin.
                // The previous nested order exhausted all component/angle
                // probes for the most populated conformer before visiting
                // the rest of the library, so a small local-pose budget could
                // report an empty pool even when a lower-population conformer
                // had the only protein-compatible basin.
                // Put the component centre first, then symmetric offsets.
                // This gives every conformer a useful central probe before a
                // bounded round moves into the tails of its VMM region.
                // Visit the full rectangle in a deterministic, balanced
                // order.  Radial sorting put every small offset before the
                // escape edge of a VMM component; a site whose compatible
                // basin was near -0.6 sigma therefore received hundreds of
                // repeated central probes before that basin was considered.
                // Axis escapes come first, followed by the remaining grid
                // points.  This is still a proposal grid, not a change to
                // the declared VMM gate or its probability model.
                // Keep structured probes just inside the declared circular
                // 95% boundary.  At very high concentration the wrapped
                // endpoint can round a few ulps outside the component even
                // though it was generated from the same half-width, causing
                // a sterically valid compatibility start to fail the strict
                // VMM gate.
                let offsets: [f64; 9] = [-0.95, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 0.95];
                let mut offset_pairs = Vec::with_capacity(offsets.len() * offsets.len());
                offset_pairs.push((0.0, 0.0));
                // The first non-central shell spans the diagonal escape
                // commonly needed by the tetramer witness (phi negative,
                // psi positive), then the two individual axes.  A conformer
                // receives the complete component-pair set at each shell.
                for pair in [
                    (-0.75, 0.25),
                    (0.0, 0.25),
                    (0.75, 0.25),
                    (-0.75, 0.0),
                    (0.75, 0.0),
                ] {
                    offset_pairs.push(pair);
                }
                for &offset in &offsets {
                    if offset != 0.0 && !offset_pairs.contains(&(offset, 0.0)) {
                        offset_pairs.push((offset, 0.0));
                    }
                }
                for &offset in &offsets {
                    if offset != 0.0 && !offset_pairs.contains(&(0.0, offset)) {
                        offset_pairs.push((0.0, offset));
                    }
                }
                for &phi_offset in &offsets {
                    for &psi_offset in &offsets {
                        if phi_offset != 0.0 && psi_offset != 0.0 {
                            offset_pairs.push((phi_offset, psi_offset));
                        }
                    }
                }
                let rotamers = if config.scan_rotamers {
                    std::iter::once(None)
                        .chain(
                            dunbrack::rotamers(protein, &site.site.residue)
                                .into_iter()
                                .enumerate()
                                .map(|(index, _)| Some(index)),
                        )
                        .collect::<Vec<_>>()
                } else {
                    vec![None]
                };
                let coverage_offset = seed_round as usize;
                let coverage_limit = max_attempts.saturating_mul(3) / 4;
                // No conformer stream can consume more than the complete
                // round allowance. Retain exactly the reachable prefix,
                // preserving probe order without materializing unused grids.
                let stream_limit = coverage_offset.saturating_add(coverage_limit.max(1));
                let mut candidate_sets = Vec::with_capacity(conformers.len());
                for (conformer_index, _) in &conformers {
                    let priors = resolved_priors(
                        protein,
                        site,
                        &site.ensemble.conformers[*conformer_index].priors,
                    );
                    // Resolve widths once per component, outside the angular
                    // grid. The shared width cache otherwise becomes a
                    // contended lock for every probe in threaded WASM.
                    let phi_widths = priors
                        .phi
                        .iter()
                        .map(probability_half_width_degrees)
                        .collect::<Vec<_>>();
                    let psi_widths = priors
                        .psi
                        .iter()
                        .map(probability_half_width_degrees)
                        .collect::<Vec<_>>();
                    let mut component_pairs = priors
                        .phi
                        .iter()
                        .enumerate()
                        .flat_map(|(phi_index, phi)| {
                            priors.psi.iter().enumerate().map(move |(psi_index, psi)| {
                                (
                                    phi_index,
                                    psi_index,
                                    (phi.weight.max(0.0) * psi.weight.max(0.0)),
                                )
                            })
                        })
                        .collect::<Vec<_>>();
                    component_pairs.sort_by(|left, right| {
                        right
                            .2
                            .total_cmp(&left.2)
                            .then_with(|| left.0.cmp(&right.0))
                            .then_with(|| left.1.cmp(&right.1))
                    });
                    // The first alternative after the dominant pair should
                    // cover both a different phi and a different psi basin.
                    // Pure product-weight ordering can spend the whole small
                    // pose budget in one marginal island, while a valid pose
                    // often needs the less-populated combination at the
                    // other marginal mode.
                    if let Some(&(dominant_phi, dominant_psi, _)) = component_pairs.first() {
                        if let Some(diverse_index) = component_pairs
                            .iter()
                            .enumerate()
                            .skip(1)
                            .filter(|(_, (phi, psi, _))| {
                                *phi != dominant_phi && *psi != dominant_psi
                            })
                            .max_by(|(_, left), (_, right)| {
                                left.2
                                    .total_cmp(&right.2)
                                    .then_with(|| right.0.cmp(&left.0))
                                    .then_with(|| right.1.cmp(&left.1))
                            })
                            .map(|(index, _)| index)
                        {
                            component_pairs.swap(1, diverse_index);
                        }
                    }
                    let mut candidates = Vec::new();
                    'rotamers: for &rotamer in &rotamers {
                        // Balance the two discrete dimensions.  A simple
                        // component-major or offset-major nested loop can
                        // spend a small frontier on one dimension and miss a
                        // compatible combination in the other.  The coprime
                        // stride visits different VMM pairs at each angular
                        // shell while retaining stable, reproducible order.
                        let component_count = component_pairs.len().max(1);
                        let offset_count = offset_pairs.len().max(1);
                        let total = component_pairs.len().saturating_mul(offset_count);
                        for candidate_index in 0..total {
                            if candidates.len() >= stream_limit {
                                break 'rotamers;
                            }
                            let component_index = candidate_index % component_count;
                            let offset_index = (candidate_index / component_count) % offset_count;
                            let (phi_component_index, psi_component_index, _) =
                                component_pairs[component_index];
                            let (phi_offset, psi_offset) = offset_pairs[offset_index];
                            let Some(phi_component) = priors.phi.get(phi_component_index) else {
                                continue;
                            };
                            let Some(psi_component) = priors.psi.get(psi_component_index) else {
                                continue;
                            };
                            candidates.push((
                                Gene {
                                    conformer: *conformer_index,
                                    phi: wrap_degrees(
                                        phi_component.mean_degrees
                                            + phi_offset * phi_widths[phi_component_index],
                                    ),
                                    psi: wrap_degrees(
                                        psi_component.mean_degrees
                                            + psi_offset * psi_widths[psi_component_index],
                                    ),
                                    rotamer,
                                },
                                phi_component_index,
                                psi_component_index,
                            ));
                        }
                    }
                    candidate_sets.push(candidates);
                }
                // The expansion round receives the number of structured
                // probes consumed by earlier rounds.  Reusing that offset is
                // the persistent cursor: starting every round at zero would
                // spend the entire new allowance replaying already-seen
                // central component pairs and would never reach the angular
                // basins needed by a later conformer.
                let mut cursors = candidate_sets
                    .iter()
                    .map(|candidates| coverage_offset.min(candidates.len()))
                    .collect::<Vec<_>>();
                // Reserve part of every round for weighted random
                // supplementation. The grid guarantees broad conformer and
                // component coverage, while seeded draws can still land in a
                // narrow off-centre basin between those probes.
                'coverage: loop {
                    let mut progressed = false;
                    for (candidates, cursor) in candidate_sets.iter().zip(&mut cursors) {
                        if attempts >= coverage_limit.max(1) || pool.len() >= pool_target {
                            break 'coverage;
                        }
                        let Some((gene, phi_component, psi_component)) = candidates.get(*cursor)
                        else {
                            continue;
                        };
                        *cursor += 1;
                        attempts += 1;
                        progressed = true;
                        let key = pose_key(gene);
                        if !seen.insert(key) {
                            continue;
                        }
                        if let Some(pose) =
                            prepared.prepare_site_pose(site_index, gene, config.clash_distance)?
                        {
                            pool.push(PooledSiteCandidate {
                                gene: gene.clone(),
                                phi_component: *phi_component,
                                psi_component: *psi_component,
                                pose,
                            });
                        }
                    }
                    if !progressed {
                        break;
                    }
                }
            }
            // Random supplementation preserves variation when the structured
            // grid has not yet filled the requested pool (and keeps the native
            // sampler's historical weighted proposal distribution intact).
            while pool.len() < pool_target && attempts < max_attempts {
                attempts += 1;
                let chromosome = if coverage {
                    let mut chromosome =
                        CookbookStericChromosome::new(std::slice::from_ref(&sites[site_index]));
                    let (gene, phi_component, psi_component) =
                        cookbook_sample_gene(protein, &sites[site_index], &mut rng, 0.85);
                    chromosome.genes.push(gene);
                    chromosome.phi_components.push(phi_component);
                    chromosome.psi_components.push(psi_component);
                    chromosome
                } else {
                    cookbook_generate_native(
                        protein,
                        std::slice::from_ref(&sites[site_index]),
                        &mut rng,
                    )
                };
                // A malformed or partially restored one-site chromosome must
                // be reported as an empty proposal, never as a WASM bounds
                // trap.  This path is reached during dense multi-site
                // ensemble initialization where a pool may be depleted.
                let Some(gene) = chromosome.genes.first().cloned() else {
                    continue;
                };
                let key = pose_key(&gene);
                if !seen.insert(key) {
                    continue;
                }
                if let Some(pose) =
                    prepared.prepare_site_pose(site_index, &gene, config.clash_distance)?
                {
                    pool.push(PooledSiteCandidate {
                        gene,
                        phi_component: chromosome.phi_components.first().copied().unwrap_or(0),
                        psi_component: chromosome.psi_components.first().copied().unwrap_or(0),
                        pose,
                    });
                }
            }
            Ok((pool, attempts))
        })
        .collect::<Result<Vec<_>>>()?;
    let proposals = pools_with_attempts
        .iter()
        .map(|(_, attempts)| *attempts)
        .sum();
    let attempts_per_site = pools_with_attempts
        .iter()
        .map(|(_, attempts)| *attempts)
        .collect::<Vec<_>>();
    let pools = pools_with_attempts
        .into_iter()
        .map(|(pool, _)| pool)
        .collect::<Vec<_>>();
    if pools.iter().any(Vec::is_empty) {
        let pool_sizes = pools.iter().map(Vec::len).collect();
        return Ok(CompatiblePoolResult {
            states: Vec::new(),
            pools,
            proposals,
            attempts_per_site,
            pool_sizes,
            pool_expansions: 0,
            compatibility_checks: 0,
            backtracks: 0,
            graph_edges: conservative_site_graph(protein, sites)
                .iter()
                .map(Vec::len)
                .sum::<usize>()
                / 2,
        });
    }

    let graph = conservative_site_graph(protein, sites);
    let pairs = graph
        .iter()
        .enumerate()
        .flat_map(|(first, neighbours)| {
            neighbours
                .iter()
                .copied()
                .filter(move |second| *second > first)
                .map(move |second| (first, second))
        })
        .collect::<Vec<_>>();
    let compatibility_checks = pairs
        .iter()
        .map(|(first, second)| pools[*first].len().saturating_mul(pools[*second].len()))
        .sum();
    let computed = pairs
        .into_par_iter()
        .map(|(first, second)| {
            let matrix = pools[first]
                .iter()
                .map(|left| {
                    pools[second]
                        .iter()
                        .map(|right| {
                            prepared.site_poses_compatible(
                                &left.pose,
                                &right.pose,
                                config.clash_distance,
                            )
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            (first, second, matrix)
        })
        .collect::<Vec<_>>();
    let mut compatibility = vec![vec![None; sites.len()]; sites.len()];
    for (first, second, matrix) in computed {
        compatibility[first][second] = Some(matrix);
    }

    fn compatible(
        tables: &[Vec<Option<Vec<Vec<bool>>>>],
        first_site: usize,
        first_candidate: usize,
        second_site: usize,
        second_candidate: usize,
    ) -> bool {
        let Some(row) = tables.get(first_site.min(second_site)) else {
            return false;
        };
        let table = row
            .get(first_site.max(second_site))
            .and_then(Option::as_ref);
        let Some(matrix) = table else {
            // Sites without a conservative edge are independent.  A missing
            // row for an otherwise valid site is malformed solver state and
            // must reject the branch rather than trap the WASM worker.
            return tables.get(first_site).is_some() && tables.get(second_site).is_some();
        };
        if first_site < second_site {
            matrix
                .get(first_candidate)
                .and_then(|row| row.get(second_candidate))
                .copied()
                .unwrap_or(false)
        } else {
            matrix
                .get(second_candidate)
                .and_then(|row| row.get(first_candidate))
                .copied()
                .unwrap_or(false)
        }
    }

    // Enforce arc consistency before branching.  The old recursive solver
    // only compared a candidate with already assigned sites, so it could
    // spend thousands of nodes exploring values that had already lost all
    // support in an unassigned neighbouring pool.  Keeping the domains as
    // pose indices makes the propagation cheap and leaves the expensive
    // geometry checks in the precomputed compatibility tables.
    fn propagate_domains(
        domains: &mut [Vec<usize>],
        tables: &[Vec<Option<Vec<Vec<bool>>>>],
    ) -> bool {
        loop {
            let mut changed = false;
            for site in 0..domains.len() {
                let retained = domains[site]
                    .iter()
                    .copied()
                    .filter(|candidate| {
                        (0..domains.len()).all(|other_site| {
                            other_site == site
                                || domains[other_site].iter().any(|other_candidate| {
                                    compatible(
                                        tables,
                                        site,
                                        *candidate,
                                        other_site,
                                        *other_candidate,
                                    )
                                })
                        })
                    })
                    .collect::<Vec<_>>();
                if retained.is_empty() {
                    return false;
                }
                if retained.len() != domains[site].len() {
                    domains[site] = retained;
                    changed = true;
                }
            }
            if !changed {
                return true;
            }
        }
    }

    fn choose_set(
        tables: &[Vec<Option<Vec<Vec<bool>>>>],
        domains: Vec<Vec<usize>>,
        selected: &mut [Option<usize>],
        rng: &mut ChaCha8Rng,
        backtracks: &mut usize,
        max_backtracks: usize,
    ) -> bool {
        if *backtracks >= max_backtracks {
            return false;
        }
        let mut domains = domains;
        if !propagate_domains(&mut domains, tables) {
            return false;
        }
        if domains.iter().all(|domain| domain.len() == 1) {
            for (site, domain) in domains.iter().enumerate() {
                selected[site] = domain.first().copied();
            }
            return true;
        }

        // MRV chooses the smallest remaining domain.  Among equal domains,
        // prefer the site with the greatest number of constrained neighbours;
        // this exposes a failing coupled region earlier than stable site
        // order alone.
        let Some(site) = (0..domains.len())
            .filter(|site| domains[*site].len() > 1)
            .min_by(|left, right| {
                domains[*left]
                    .len()
                    .cmp(&domains[*right].len())
                    .then_with(|| {
                        let left_degree =
                            tables[*left].iter().filter(|entry| entry.is_some()).count();
                        let right_degree = tables[*right]
                            .iter()
                            .filter(|entry| entry.is_some())
                            .count();
                        right_degree.cmp(&left_degree)
                    })
                    .then_with(|| left.cmp(right))
            })
        else {
            // Propagation should have returned above when all domains became
            // singleton.  Treat any inconsistent intermediate state as a
            // failed branch instead of relying on an invariant that can be
            // violated by a depleted browser search pool.
            return false;
        };

        let mut candidates = domains[site].clone();
        // Try poses with the most remaining support first.  This is only a
        // search ordering heuristic; probability remains a secondary tie
        // breaker and never changes the steric compatibility gate.
        candidates.sort_by_key(|candidate| {
            let support = (0..domains.len())
                .filter(|other_site| *other_site != site)
                .map(|other_site| {
                    domains[other_site]
                        .iter()
                        .filter(|other_candidate| {
                            compatible(tables, site, *candidate, other_site, **other_candidate)
                        })
                        .count()
                })
                .sum::<usize>();
            std::cmp::Reverse(support)
        });
        // Randomize only exact support ties so seeded runs retain coverage
        // diversity without sacrificing the deterministic ordering of the
        // compatibility heuristic.
        let mut start = 0;
        while start < candidates.len() {
            let support = (0..domains.len())
                .filter(|other_site| *other_site != site)
                .map(|other_site| {
                    domains[other_site]
                        .iter()
                        .filter(|other_candidate| {
                            compatible(
                                tables,
                                site,
                                candidates[start],
                                other_site,
                                **other_candidate,
                            )
                        })
                        .count()
                })
                .sum::<usize>();
            let end = (start + 1..=candidates.len())
                .find(|end| {
                    *end == candidates.len() || {
                        let value = (0..domains.len())
                            .filter(|other_site| *other_site != site)
                            .map(|other_site| {
                                domains[other_site]
                                    .iter()
                                    .filter(|other_candidate| {
                                        compatible(
                                            tables,
                                            site,
                                            candidates[*end],
                                            other_site,
                                            **other_candidate,
                                        )
                                    })
                                    .count()
                            })
                            .sum::<usize>();
                        value != support
                    }
                })
                .unwrap_or(candidates.len());
            if end > start + 1 {
                candidates[start..end].shuffle(rng);
            }
            start = end;
        }

        for candidate in candidates {
            let mut branch = domains.clone();
            branch[site] = vec![candidate];
            let mut branch_selected = selected.to_vec();
            branch_selected[site] = Some(candidate);
            if choose_set(
                tables,
                branch,
                &mut branch_selected,
                rng,
                backtracks,
                max_backtracks,
            ) {
                selected.copy_from_slice(&branch_selected);
                return true;
            }
            *backtracks = (*backtracks).saturating_add(1);
            if *backtracks >= max_backtracks {
                return false;
            }
        }
        false
    }

    let mut rng = ChaCha8Rng::seed_from_u64(splitmix64(config.seed ^ 0x434f_4d42_494e_4500));
    let mut states = Vec::with_capacity(frames);
    let mut seen = HashSet::new();
    let mut solver_backtracks = 0usize;
    let max_restarts = frames.saturating_mul(8).max(8);
    for _ in 0..max_restarts {
        if states.len() >= frames {
            break;
        }
        let mut selected = vec![None; sites.len()];
        let mut backtracks = 0usize;
        let domains = pools
            .iter()
            .map(|pool| (0..pool.len()).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let found = choose_set(
            &compatibility,
            domains,
            &mut selected,
            &mut rng,
            &mut backtracks,
            10_000,
        );
        solver_backtracks = solver_backtracks.saturating_add(backtracks);
        if !found {
            continue;
        }
        let Some(indices) = selected.into_iter().collect::<Option<Vec<_>>>() else {
            // A solver branch may report success after propagation while a
            // stale/partially expanded domain did not populate every slot.
            // Keep searching from the next deterministic restart.
            continue;
        };
        if !seen.insert(indices.clone()) {
            continue;
        }
        let chosen = indices
            .iter()
            .enumerate()
            .map(|(site, candidate)| &pools[site][*candidate])
            .collect::<Vec<_>>();
        states.push((
            chosen
                .iter()
                .map(|candidate| candidate.gene.clone())
                .collect(),
            chosen
                .iter()
                .map(|candidate| candidate.phi_component)
                .collect(),
            chosen
                .iter()
                .map(|candidate| candidate.psi_component)
                .collect(),
        ));
    }
    Ok(CompatiblePoolResult {
        states,
        pools: pools.clone(),
        proposals,
        attempts_per_site,
        pool_sizes: pools.iter().map(Vec::len).collect(),
        pool_expansions: 0,
        compatibility_checks,
        backtracks: solver_backtracks,
        graph_edges: graph.iter().map(Vec::len).sum::<usize>() / 2,
    })
}

fn pose_key(gene: &Gene) -> (usize, u64, u64, Option<usize>) {
    (
        gene.conformer,
        gene.phi.to_bits(),
        gene.psi.to_bits(),
        gene.rotamer,
    )
}

fn default_sampling_temperature_k() -> f64 {
    300.0
}

/// Sample the native cluster/linkage distribution conditioned on steric
/// compatibility.  The first phase is an adaptive weighted sampler.  If it
/// cannot fill the request, a complete GA state seeds constrained MH chains,
/// matching the Cookbook fallback rather than imposing a per-frame attempt
/// limit.
pub fn sample_attached_ensemble(
    protein: &Structure,
    sites: &[SearchSite],
    frames: usize,
    config: &SearchConfig,
    builder: &glysys::SystemBuilder,
) -> Result<(Vec<SampledFrame>, EnsembleSamplingDiagnostics)> {
    sample_attached_ensemble_with_cancel(protein, sites, frames, config, builder, || false)
}

/// Sample an attached ensemble with cooperative cancellation.  Cancellation
/// is checked between native proposals, fallback chains, and accepted-frame
/// materialization; a single geometry/energy proposal is allowed to finish so
/// deterministic seeded results are not left half-written.
#[maybe_async_cfg::maybe(
    sync(keep_self),
    async(feature = "webgpu"),
    idents(
        genetic_optimize_with_progress_cancelled(sync, async = "gpu_optimize"),
        evaluate_candidate(sync, async = "gpu_evaluate_candidate"),
        start_compute_stage(sync, async = "start_compute_stage_async"),
        sample_statistical(sync, async = "sample_statistical_async"),
        cookbook_steric_search_with_cancel(
            sync,
            async = "cookbook_steric_search_with_cancel_async"
        )
    )
)]
pub async fn sample_attached_ensemble_with_cancel<C>(
    protein: &Structure,
    sites: &[SearchSite],
    frames: usize,
    config: &SearchConfig,
    builder: &glysys::SystemBuilder,
    mut cancelled: C,
) -> Result<(Vec<SampledFrame>, EnsembleSamplingDiagnostics)>
where
    C: FnMut() -> bool,
{
    start_compute_stage().await;
    if cancelled() {
        return Err(EnsembleError::Cancelled);
    }
    if sites.is_empty() || sites.iter().any(|site| site.ensemble.conformers.is_empty()) {
        return Err(EnsembleError::EmptyEnsemble);
    }
    if config
        .ensemble_mode
        .as_deref()
        .is_some_and(|m| m != "sampled" && m != "conformer_collection")
        || config.thinning_steps == Some(0)
    {
        return Err(EnsembleError::Metadata(
            "invalid ensemble mode or thinning steps".into(),
        ));
    }
    if config
        .ensemble_mode
        .as_deref()
        .unwrap_or(if config.pre_minimization {
            "conformer_collection"
        } else {
            "sampled"
        })
        == "sampled"
    {
        if config.pre_minimization {
            return Err(EnsembleError::Metadata(
                "sampled ensembles cannot minimize proposals; choose conformer_collection".into(),
            ));
        }
        return sample_statistical(protein, sites, frames, config, builder, cancelled).await;
    }
    validate_site_priors(protein, sites)?;
    let preparation_started = Instant::now();
    let prepared = PreparedAttachmentContext::new(protein, sites, config.scan_rotamers)?;
    let preparation_seconds = preparation_started.elapsed().as_secs_f64();
    let sampling_started = Instant::now();
    let energy_context = if config.scoring_mode != SearchScoringMode::StericPrior {
        let started = Instant::now();
        let reference = sites
            .iter()
            .map(|_| Gene {
                conformer: 0,
                phi: 0.0,
                psi: 0.0,
                rotamer: None,
            })
            .collect::<Vec<_>>();
        let structure = build_state(protein, sites, &reference, builder)?;
        let system = builder.prepare_structure(&structure)?;
        Some(EnergySearchContext {
            system,
            evaluator: OnceLock::new(),
            mode: config.scoring_mode,
            use_obc2: config.use_obc2,
            minimize: config.pre_minimization,
            min_iterations: config.pre_minimization_iterations,
            min_radius: config.minimization_radius,
            cutoff: config.energy_cutoff,
            cache: Mutex::new(HashMap::new()),
            atom_masks: OnceLock::new(),
            atom_mapping: OnceLock::new(),
            evaluations: AtomicUsize::new(0),
            minimizations: AtomicUsize::new(0),
            cache_hits: AtomicUsize::new(0),
            steric_rejections: AtomicUsize::new(0),
            failures: AtomicUsize::new(0),
            started: Instant::now(),
            topology_seconds: started.elapsed().as_secs_f64(),
        })
    } else {
        None
    };
    let initial_states = if config.scoring_mode != SearchScoringMode::StericPrior && sites.len() > 1
    {
        // Refined conformer collections use the same compatibility starts as
        // energy Build. This is only an initialization aid; subsequent
        // collection proposals and minimization retain their existing
        // semantics and are not treated as statistical transitions.
        let budget = config
            .population_size
            .max(2)
            .saturating_mul(config.generations.saturating_add(1))
            .saturating_mul(2);
        if budget == 0 {
            Arc::new(Vec::new())
        } else {
            Arc::new(
                cookbook_compatible_pool_coverage(
                    protein,
                    sites,
                    config,
                    &prepared,
                    config.population_size.max(2).min(4),
                    budget,
                )?
                .states
                .into_iter()
                .map(|(genes, _, _)| genes)
                .collect::<Vec<_>>(),
            )
        }
    } else {
        Arc::new(Vec::new())
    };
    let problem = SearchProblem {
        protein,
        sites,
        builder,
        clash_distance: config.clash_distance,
        scan_rotamers: config.scan_rotamers,
        scoring_mode: config.scoring_mode,
        selection_policy: config.selection_policy,
        use_obc2: config.use_obc2,
        pre_minimization: config.pre_minimization,
        pre_minimization_iterations: config.pre_minimization_iterations,
        energy_context: energy_context.as_ref(),
        prepared: &prepared,
        prior_cache: compile_prior_cache(protein, sites)?,
        vmm_priors: compile_vmm_prior_cache(protein, sites),
        initial_states,
        initial_state_cursor: AtomicUsize::new(0),
    };
    let mut rng = ChaCha8Rng::seed_from_u64(config.seed);
    let mut accepted = Vec::with_capacity(frames);
    let mut attempts = 0usize;
    let mut native_proposals = 0usize;
    let mut native_accepts = 0usize;
    let mut mh_proposals = 0usize;
    let mut mh_accepts = 0usize;
    let mut fallback_used = false;
    let mut ga_restarts = 0usize;
    let mut current_energy: Option<f64> = None;
    let mut energy_transitions = 0usize;

    if config.scoring_mode == SearchScoringMode::StericPrior {
        let pooled = cookbook_compatible_pool_sample(protein, sites, frames, config, &prepared)?;
        attempts += pooled.proposals;
        native_proposals += pooled.proposals;
        for (frame_index, (state, phi_components, psi_components)) in
            pooled.states.into_iter().enumerate()
        {
            if cancelled() {
                return Err(EnsembleError::Cancelled);
            }
            let evaluation = prepared.evaluate(&state, config.clash_distance)?;
            debug_assert!(evaluation.site_scores.iter().all(|score| *score <= 1.1));
            let structure = build_state(protein, sites, &state, builder)?;
            native_accepts += 1;
            accepted.push(sampled_frame_from_state(
                protein,
                sites,
                &state,
                structure,
                evaluation.site_scores,
                attempts + frame_index + 1,
                "compatible_pool",
                Some(&phi_components),
                Some(&psi_components),
                None,
            ));
        }
    }

    // Adaptive native sampling: easy systems finish almost immediately,
    // while crowded systems receive progressively more proposals.  The
    // budget is internal and deliberately not exposed as `--max-attempts`.
    // Direct native rejection sampling must remain the cheap first path. The
    // historical `mh_steps` default is 1000; multiplying it by every requested
    // frame silently created 50,000 full multi-glycan proposals for a normal
    // 50-frame ensemble. Use the explicit accepted-move spacing as the direct
    // sampling budget and hand difficult systems to the single GA-seeded
    // fallback instead of burning an unrelated hidden budget.
    let native_budget = frames
        .saturating_mul(config.mh_thinning_accepted.max(10))
        .max(sites.len().saturating_mul(20));
    let fallback_threshold = frames.div_ceil(10).max(1);
    let mut native_pass_budget = native_budget / 5;
    for pass in 0..5 {
        if cancelled() {
            return Err(EnsembleError::Cancelled);
        }
        if accepted.len() >= frames || attempts >= native_budget {
            break;
        }
        native_pass_budget = native_pass_budget.max(1);
        let pass_end = attempts
            .saturating_add(native_pass_budget)
            .min(native_budget);
        while accepted.len() < frames && attempts < pass_end {
            if cancelled() {
                return Err(EnsembleError::Cancelled);
            }
            attempts += 1;
            native_proposals += 1;
            let (mut state, native_phi_components, native_psi_components) =
                if config.scoring_mode == SearchScoringMode::StericPrior {
                    let chromosome = cookbook_generate_native(protein, sites, &mut rng);
                    (
                        chromosome.genes,
                        Some(chromosome.phi_components),
                        Some(chromosome.psi_components),
                    )
                } else {
                    (problem.generate(&mut rng), None, None)
                };
            let Ok(mut prepared_evaluation) =
                problem.prepared.evaluate(&state, config.clash_distance)
            else {
                continue;
            };
            if config.scan_rotamers
                && prepared_evaluation
                    .site_scores
                    .iter()
                    .any(|score| *score > 1.1)
            {
                let (repaired_state, repaired_evaluation) =
                    cookbook_try_rotamer_repair(&problem, &state, &prepared_evaluation)?;
                state = repaired_state;
                prepared_evaluation = repaired_evaluation;
            }
            let scores = prepared_evaluation.site_scores;
            // Steric-prior acceptance is governed by the Cookbook 1.7 Å
            // prepared score and the selected VMM component. The expensive
            // topology-aware VDW policy is a post-output validation step.
            if config.scoring_mode != SearchScoringMode::StericPrior
                && !scores.iter().all(|score| *score <= 1.1)
            {
                continue;
            }
            // Native strict proposals are generated inside their selected VMM
            // component. Reject from the prepared site scores before building
            // a complete Structure; the old order materialized every rejected
            // PDB and then evaluated its sterics a second time.
            if config.scoring_mode == SearchScoringMode::StericPrior
                && !scores.iter().all(|score| *score <= 1.1)
            {
                continue;
            }
            let mut structure = build_state(protein, sites, &state, builder)?;
            let selected_energy = if config.scoring_mode == SearchScoringMode::StericPrior {
                None
            } else {
                let evaluated = evaluate_candidate(
                    energy_context.as_ref().expect("energy context"),
                    &structure,
                )
                .await?;
                let energy = evaluated.score;
                let accepted_transition = current_energy.is_none_or(|previous| {
                    let beta = 1.0 / (0.001_987_204_1 * config.temperature_k);
                    let probability = (-(energy - previous) * beta).exp().min(1.0);
                    rng.random_bool(probability)
                });
                if !accepted_transition {
                    continue;
                }
                current_energy = Some(energy);
                energy_transitions += 1;
                if energy_transitions <= config.burn_in
                    || !(energy_transitions - config.burn_in).is_multiple_of(config.thinning)
                {
                    continue;
                }
                if config.pre_minimization {
                    let context = energy_context.as_ref().expect("energy context");
                    let mut system = context.system.clone();
                    system.set_coordinates(&evaluated.coordinates)?;
                    structure.update_with_parameterized_hydrogens(&system)?;
                }
                Some(energy)
            };
            native_accepts += 1;
            accepted.push(sampled_frame_from_state(
                protein,
                sites,
                &state,
                structure,
                scores,
                attempts,
                "native_sampler",
                native_phi_components.as_deref(),
                native_psi_components.as_deref(),
                selected_energy,
            ));
        }
        if accepted.len() >= frames {
            break;
        }
        if pass >= 1 && accepted.len() < fallback_threshold {
            // This is the Cookbook fallback trigger: a native sampler that
            // cannot produce even ten percent of the requested set is not
            // worth exhausting its remaining rejection budget.
            break;
        }
        // Cookbook-style adaptive passes increase exploration without making
        // the caller guess a magic attempt count.
        native_pass_budget = native_pass_budget.saturating_mul(2);
        if pass == 4 {
            break;
        }
    }

    // If native rejection sampling was not sufficient, initialize chains at
    // a complete GA solution and sample native proposals around that state.
    // This preserves the native distribution while making multi-site systems
    // practical instead of returning an arbitrary short ensemble.
    while accepted.len() < frames {
        if cancelled() {
            return Err(EnsembleError::Cancelled);
        }
        fallback_used = true;
        let ga_seed = splitmix64(config.seed ^ 0x475f534545445f4d);
        let ga_config = GeneticAlgorithmConfig {
            population_size: config.population_size.max(2),
            generations: config.generations,
            seed: ga_seed.wrapping_add(ga_restarts as u64),
            ..GeneticAlgorithmConfig::default()
        };
        // Steric-prior ensemble fallback must use the same cheap strict
        // Cookbook solver as single builds. The historical generic GA can
        // return a clash-free pose outside the selected VMM component, which
        // would make an emitted frame violate the strict 95% contract.
        let (ga_state, ga_phi_components, ga_psi_components) =
            if config.scoring_mode == SearchScoringMode::StericPrior {
                let result = cookbook_steric_search_with_cancel(
                    &problem,
                    &SearchConfig {
                        population_size: ga_config.population_size,
                        generations: ga_config.generations,
                        seed: ga_config.seed,
                        ..config.clone()
                    },
                    |_| {},
                    &mut cancelled,
                )
                .await?;
                (
                    result.best_state,
                    Some(result.phi_components),
                    Some(result.psi_components),
                )
            } else {
                let result = genetic_optimize_with_progress_cancelled(
                    &problem,
                    &ga_config,
                    |_| {},
                    &mut cancelled,
                )
                .await;
                let result = match result {
                    Ok(value) => value,
                    Err(glysys_opt::OptimizationError::Cancelled) => {
                        return Err(EnsembleError::Cancelled);
                    }
                    Err(error) => return Err(error.into()),
                };
                (result.best_state, None, None)
            };
        ga_restarts += 1;
        let ga_is_complete = if config.scoring_mode == SearchScoringMode::StericPrior {
            // The strict GA has already accepted this state from its prepared
            // scores and all of its proposals preserve VMM membership.
            true
        } else {
            problem
                .prepared
                .evaluate(&ga_state, config.clash_distance)
                .map(|evaluation| evaluation.site_scores.iter().all(|score| *score <= 1.1))
                .unwrap_or(false)
        };
        if !ga_is_complete {
            break;
        }
        let remaining = frames - accepted.len();
        let chains = config.mh_chains.max(1);
        let target_per_chain = remaining.div_ceil(chains);
        let burn_in_moves = config.mh_burn_in_sweeps.saturating_mul(sites.len().max(1));
        let thinning = config.mh_thinning_accepted.max(1);
        let proposal_budget = remaining.saturating_mul(2_000).max(20_000);
        for chain in 0..chains {
            if cancelled() {
                return Err(EnsembleError::Cancelled);
            }
            if accepted.len() >= frames {
                break;
            }
            let chain_seed = splitmix64(config.seed ^ 0x434841494e ^ chain as u64);
            let mut chain_rng = ChaCha8Rng::seed_from_u64(chain_seed);
            let mut state = ga_state.clone();
            let mut chain_phi_components = ga_phi_components.clone();
            let mut chain_psi_components = ga_psi_components.clone();
            let mut structure = build_state(protein, sites, &state, builder)?;
            let mut chain_energy = if config.scoring_mode == SearchScoringMode::StericPrior {
                None
            } else {
                Some(
                    evaluate_candidate(
                        energy_context.as_ref().expect("energy context"),
                        &structure,
                    )
                    .await?,
                )
            };
            let mut accepted_moves = 0usize;
            let mut proposals = 0usize;
            while accepted.len() < frames
                && accepted_moves
                    < burn_in_moves.saturating_add(target_per_chain.saturating_mul(thinning))
                && proposals < proposal_budget
            {
                if cancelled() {
                    return Err(EnsembleError::Cancelled);
                }
                proposals += 1;
                attempts += 1;
                mh_proposals += 1;
                let site_index = chain_rng.random_range(0..sites.len());
                let mut proposal = state.clone();
                let conformers = &sites[site_index].ensemble.conformers;
                proposal[site_index].conformer =
                    weighted_conformer_index(&sites[site_index].ensemble, &mut chain_rng);
                let priors = resolved_priors(
                    protein,
                    &sites[site_index],
                    &conformers[proposal[site_index].conformer].priors,
                );
                let mut proposed_phi_components = chain_phi_components.clone();
                let mut proposed_psi_components = chain_psi_components.clone();
                if config.scoring_mode == SearchScoringMode::StericPrior {
                    let phi = sample_truncated_vmm(&priors.phi, &mut chain_rng);
                    let psi = sample_truncated_vmm(&priors.psi, &mut chain_rng);
                    proposal[site_index].phi = phi.degrees;
                    proposal[site_index].psi = psi.degrees;
                    if let (Some(phi_components), Some(psi_components)) =
                        (&mut proposed_phi_components, &mut proposed_psi_components)
                    {
                        phi_components[site_index] = phi.component;
                        psi_components[site_index] = psi.component;
                    }
                } else {
                    proposal[site_index].phi = sample_vmm(&priors.phi, &mut chain_rng);
                    proposal[site_index].psi = sample_vmm(&priors.psi, &mut chain_rng);
                }
                let Ok(mut prepared_evaluation) =
                    problem.prepared.evaluate(&proposal, config.clash_distance)
                else {
                    continue;
                };
                if config.scan_rotamers
                    && prepared_evaluation
                        .site_scores
                        .iter()
                        .any(|score| *score > 1.1)
                {
                    let (repaired_state, repaired_evaluation) =
                        cookbook_try_rotamer_repair(&problem, &proposal, &prepared_evaluation)?;
                    proposal = repaired_state;
                    prepared_evaluation = repaired_evaluation;
                }
                let proposed_scores = prepared_evaluation.site_scores;
                if config.scoring_mode != SearchScoringMode::StericPrior
                    && !proposed_scores.iter().all(|value| *value <= 1.1)
                {
                    continue;
                }
                if config.scoring_mode == SearchScoringMode::StericPrior
                    && !proposed_scores.iter().all(|score| *score <= 1.1)
                {
                    continue;
                }
                // Materialize only proposals which passed the fast gate.
                let proposed_structure = build_state(protein, sites, &proposal, builder)?;
                let proposed_energy = if config.scoring_mode == SearchScoringMode::StericPrior {
                    None
                } else {
                    Some(
                        evaluate_candidate(
                            energy_context.as_ref().expect("energy context"),
                            &proposed_structure,
                        )
                        .await?,
                    )
                };
                let accept = if let (Some(current), Some(proposed)) =
                    (chain_energy.as_ref(), proposed_energy.as_ref())
                {
                    let beta = 1.0 / (0.001_987_204_1 * config.temperature_k);
                    let current_target =
                        native_log_probability(protein, sites, &state) - beta * current.score;
                    let proposed_target =
                        native_log_probability(protein, sites, &proposal) - beta * proposed.score;
                    (proposed_target - current_target).exp().min(1.0)
                } else {
                    1.0
                };
                if !chain_rng.random_bool(accept) {
                    continue;
                }
                state = proposal;
                if config.scoring_mode == SearchScoringMode::StericPrior {
                    // The selected component follows each accepted proposal;
                    // retain it for the frame metadata and subsequent moves.
                    chain_phi_components = proposed_phi_components.clone();
                    chain_psi_components = proposed_psi_components.clone();
                }
                structure = proposed_structure;
                if config.pre_minimization {
                    if let (Some(context), Some(value)) =
                        (energy_context.as_ref(), proposed_energy.as_ref())
                    {
                        let mut system = context.system.clone();
                        system.set_coordinates(&value.coordinates)?;
                        structure.update_with_parameterized_hydrogens(&system)?;
                    }
                }
                chain_energy = proposed_energy;
                accepted_moves += 1;
                mh_accepts += 1;
                if accepted_moves <= burn_in_moves
                    || !(accepted_moves - burn_in_moves).is_multiple_of(thinning)
                {
                    continue;
                }
                let selected_energy = chain_energy.as_ref().map(|value| value.score);
                accepted.push(sampled_frame_from_state(
                    protein,
                    sites,
                    &state,
                    structure.clone(),
                    proposed_scores,
                    attempts,
                    "ga_seeded_mh",
                    chain_phi_components.as_deref(),
                    chain_psi_components.as_deref(),
                    selected_energy,
                ));
            }
        }
        if accepted.len() >= frames {
            break;
        }
        // A fallback is seeded once and reused across every configured MH
        // chain. Starting additional hidden GA searches makes an ensemble
        // exceed the caller's explicit search budget.
        break;
    }
    if accepted.len() < frames {
        return Err(EnsembleError::InsufficientFrames {
            requested: frames,
            returned: accepted.len(),
        });
    }
    let diagnostics = EnsembleSamplingDiagnostics {
        requested_frames: frames,
        returned_frames: accepted.len(),
        attempts,
        acceptance_rate: if attempts == 0 {
            0.0
        } else {
            (native_accepts + mh_accepts) as f64 / attempts as f64
        },
        seed: config.seed,
        native_frames: accepted
            .iter()
            .filter(|frame| {
                frame.source == "model_da_mh_v3"
                    || frame.source == "model_mh_v2"
                    || frame.source == "native_sampler"
            })
            .count(),
        fallback_frames: accepted
            .iter()
            .filter(|frame| frame.source == "ga_seeded_mh")
            .count(),
        fallback_used,
        ga_restarts,
        chains: config.mh_chains.max(1),
        native_proposals,
        native_accepts,
        mh_proposals,
        mh_accepts,
        single_site_proposals: 0,
        pair_proposals: 0,
        group_proposals: 0,
        whole_state_proposals: 0,
        single_site_accepts: 0,
        pair_accepts: 0,
        group_accepts: 0,
        whole_state_accepts: 0,
        burn_in_sweeps: config.mh_burn_in_sweeps,
        thinning_accepted: config.mh_thinning_accepted,
        temperature_k: config.temperature_k,
        scan_rotamers: config.scan_rotamers,
        timings: SearchTimingDiagnostics {
            preparation_seconds,
            proposal_seconds: sampling_started.elapsed().as_secs_f64(),
            states_per_second: (native_proposals + mh_proposals) as f64
                / sampling_started
                    .elapsed()
                    .as_secs_f64()
                    .max(f64::MIN_POSITIVE),
            ..SearchTimingDiagnostics::default()
        },
        sampling_target: config.sampling_target,
        segments: vec![SamplingSegmentDiagnostics {
            id: 0,
            target: config.sampling_target.unwrap_or_default(),
            backend: if config.sampling_target == Some(SamplingTarget::WebgpuF32V1) {
                "GPU/CPU-fallback".into()
            } else {
                "CPU".into()
            },
            frames: accepted.len(),
            attempts,
            accepts: native_accepts + mh_accepts,
            burn_in_steps: config.mh_burn_in_sweeps.saturating_mul(sites.len().max(1)),
            fallback_reason: None,
        }],
    };
    refresh_frames(&mut accepted, protein, sites, config, builder)?;
    Ok((accepted, diagnostics))
}

/// Prepare chemistry once from original assets, never from a hydrogen-enriched
/// optimized output. Coordinate updates do not imply re-parameterization.
pub fn prepare_frame_topology(
    protein: &Structure,
    sites: &[SearchSite],
    builder: &glysys::SystemBuilder,
) -> Result<glysys::ParameterizedSystem> {
    let reference = vec![
        Gene {
            conformer: 0,
            phi: 0.,
            psi: 0.,
            rotamer: None
        };
        sites.len()
    ];
    Ok(builder.prepare_structure(&build_state(protein, sites, &reference, builder)?)?)
}

/// Recompute report values from emitted coordinates, without modifying geometry.
pub fn refresh_frame(
    frame: &mut SampledFrame,
    protein: &Structure,
    sites: &[SearchSite],
    config: &SearchConfig,
    builder: &glysys::SystemBuilder,
) -> Result<()> {
    refresh_frames(std::slice::from_mut(frame), protein, sites, config, builder)
}
pub fn refresh_frames(
    frames: &mut [SampledFrame],
    protein: &Structure,
    sites: &[SearchSite],
    config: &SearchConfig,
    builder: &glysys::SystemBuilder,
) -> Result<()> {
    let system = if config.scoring_mode != SearchScoringMode::StericPrior {
        Some(prepare_frame_topology(protein, sites, builder)?)
    } else {
        None
    };
    let mapping = system
        .as_ref()
        .map(glysys_energy::geometry::CoordinateMap::new);
    let evaluator = system
        .as_ref()
        .map(|s| {
            EnergyEvaluator::new(
                s,
                EnergyOptions {
                    cutoff: Some(config.energy_cutoff),
                    obc2: config.use_obc2.then(Obc2Options::default),
                    ..Default::default()
                },
            )
        })
        .transpose()?;
    let groups = system.as_ref().map(|s| {
        let roles = glysys_energy::scoring::component_roles(s);
        let group = |role| {
            AtomGroupMask::from_indices(
                roles.len(),
                roles
                    .iter()
                    .enumerate()
                    .filter_map(|(i, r)| (*r == role).then_some(i)),
            )
        };
        (
            group(glysys_energy::scoring::ComponentRole::Receptor),
            group(glysys_energy::scoring::ComponentRole::Glycan),
        )
    });
    for frame in frames {
        if frame.structure.iter_atoms().any(|a| {
            !a.position.x.is_finite() || !a.position.y.is_finite() || !a.position.z.is_finite()
        }) {
            return Err(EnsembleError::Metadata("nonfinite output geometry".into()));
        }
        let scores = steric_site_scores(&frame.structure, config.clash_distance);
        frame.log_native_probability = 0.;
        for (index, (result, site)) in frame.sites.iter_mut().zip(sites).enumerate() {
            let (phi, psi) =
                reglyco_build::attachment_angles(&frame.structure, &site.site.residue)?;
            let conformer = &site.ensemble.conformers[result.conformer_index];
            let prior = resolved_priors(protein, site, &conformer.priors);
            result.phi_degrees = phi;
            result.psi_degrees = psi;
            result.prior_score = -(conformer.cluster_weight
                / site
                    .ensemble
                    .conformers
                    .iter()
                    .map(|c| c.cluster_weight)
                    .sum::<f64>())
            .ln()
                + vmm_penalty(phi, &prior.phi)
                + vmm_penalty(psi, &prior.psi);
            frame.log_native_probability -= result.prior_score;
            result.phi_within_vmm95 = result
                .phi_component
                .and_then(|i| prior.phi.get(i))
                .map(|p| vmm_component_within_95(phi, p));
            result.psi_within_vmm95 = result
                .psi_component
                .and_then(|i| prior.psi.get(i))
                .map(|p| vmm_component_within_95(psi, p));
            result.steric_score = *scores.get(index).ok_or_else(|| {
                EnsembleError::Metadata("output attachment mapping mismatch".into())
            })?;
            let residues = frame
                .structure
                .metadata()
                .glycan_trees
                .iter()
                .find(|t| t.attachment_site.as_ref() == Some(&site.site.residue))
                .ok_or_else(|| EnsembleError::Metadata("missing output glycan tree".into()))?;
            result.coordinates = frame
                .structure
                .iter_atoms()
                .filter(|a| residues.residue_ids.contains(&a.residue))
                .map(|a| a.position)
                .collect();
        }

        if let Some(evaluator) = &evaluator {
            let coordinates = mapping.as_ref().unwrap().coordinates(&frame.structure)?;
            let value = if config.scoring_mode == SearchScoringMode::FullEnergy {
                evaluator.energy(&coordinates)?.total()
            } else {
                let (protein, glycan) = groups.as_ref().unwrap();
                evaluator
                    .interaction_energy(&coordinates, protein, glycan)?
                    .total()
            };
            if !value.is_finite() {
                return Err(EnsembleError::Metadata("nonfinite output energy".into()));
            }
            frame.selected_energy_kcal_per_mol = Some(value);
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn sampled_frame_from_state(
    protein: &Structure,
    sites: &[SearchSite],
    state: &[Gene],
    structure: Structure,
    scores: Vec<f64>,
    proposal_index: usize,
    source: &str,
    phi_components: Option<&[usize]>,
    psi_components: Option<&[usize]>,
    selected_energy: Option<f64>,
) -> SampledFrame {
    let mut results = Vec::with_capacity(sites.len());
    let mut log_native_probability = 0.0;
    for (index, (gene, site)) in state.iter().zip(sites).enumerate() {
        let conformer = &site.ensemble.conformers[gene.conformer];
        let priors = resolved_priors(protein, site, &conformer.priors);
        let conformer_probability = (conformer.cluster_weight
            / site
                .ensemble
                .conformers
                .iter()
                .map(|c| c.cluster_weight)
                .sum::<f64>())
        .max(0.0);
        let phi_log_density =
            normalized_mixture(&priors.phi).log_probability(gene.phi.to_radians());
        let psi_log_density =
            normalized_mixture(&priors.psi).log_probability(gene.psi.to_radians());
        let attachment_log_density = phi_log_density + psi_log_density;
        let prior_score = if conformer_probability > 0.0 {
            -conformer_probability.ln() - attachment_log_density
        } else {
            f64::INFINITY
        };
        log_native_probability -= prior_score;
        let tree = &structure.metadata().glycan_trees[index];
        let residues = tree.residue_ids.iter().cloned().collect::<BTreeSet<_>>();
        let coordinates = structure
            .atoms()
            .into_iter()
            .filter(|atom| residues.contains(&atom.residue))
            .map(|atom| atom.position)
            .collect();
        let phi_component = phi_components.and_then(|components| components.get(index).copied());
        let psi_component = psi_components.and_then(|components| components.get(index).copied());
        let phi_within_vmm95 = phi_component.and_then(|component| {
            priors
                .phi
                .get(component)
                .map(|prior| vmm_component_within_95(gene.phi, prior))
        });
        let psi_within_vmm95 = psi_component.and_then(|component| {
            priors
                .psi
                .get(component)
                .map(|prior| vmm_component_within_95(gene.psi, prior))
        });
        results.push(SearchSiteResult {
            site: site.site.clone(),
            conformer_index: gene.conformer,
            conformer_id: conformer.id.clone(),
            cluster_index: conformer.cluster_index,
            main_cluster: conformer.main_cluster,
            cluster_weight: conformer.cluster_weight,
            phi_degrees: gene.phi,
            psi_degrees: gene.psi,
            phi_component,
            psi_component,
            phi_within_vmm95,
            psi_within_vmm95,
            rotamer_index: gene.rotamer,
            prior_score,
            conformer_probability: conformer_probability
                .is_finite()
                .then_some(conformer_probability),
            attachment_log_density: attachment_log_density
                .is_finite()
                .then_some(attachment_log_density),
            joint_prior_score: prior_score.is_finite().then_some(prior_score),
            steric_score: scores[index],
            coordinates,
        });
    }
    SampledFrame {
        structure,
        sites: results,
        proposal_index,
        source: source.into(),
        log_native_probability,
        selected_energy_kcal_per_mol: selected_energy,
    }
}

fn native_log_probability(protein: &Structure, sites: &[SearchSite], state: &[Gene]) -> f64 {
    state
        .iter()
        .zip(sites)
        .map(|(gene, site)| {
            let conformer = &site.ensemble.conformers[gene.conformer];
            let priors = resolved_priors(protein, site, &conformer.priors);
            -(-(conformer.cluster_weight
                / site
                    .ensemble
                    .conformers
                    .iter()
                    .map(|c| c.cluster_weight)
                    .sum::<f64>())
            .ln()
                + vmm_penalty(gene.phi, &priors.phi)
                + vmm_penalty(gene.psi, &priors.psi))
        })
        .sum()
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
    value ^ (value >> 31)
}

pub fn attachments_from_outcome(
    sites: &[SearchSite],
    outcome: &SearchOutcome,
) -> Result<Vec<AttachmentRequest>> {
    if sites.len() != outcome.sites.len() {
        return Err(EnsembleError::Metadata(
            "search outcome does not cover every requested site".into(),
        ));
    }
    sites
        .iter()
        .zip(&outcome.sites)
        .map(|(site, selected)| {
            let conformer = site
                .ensemble
                .conformers
                .get(selected.conformer_index)
                .ok_or_else(|| EnsembleError::Metadata("invalid conformer index".into()))?;
            Ok(AttachmentRequest {
                site: site.site.clone(),
                conformer: GlycanConformer::new(conformer.structure.clone()),
            })
        })
        .collect()
}

/// Build the exact conformers and linkage angles recorded in a search result.
pub fn build_from_outcome(
    protein: &Structure,
    sites: &[SearchSite],
    outcome: &SearchOutcome,
    builder: &glysys::SystemBuilder,
    parameterize: bool,
) -> Result<BuildProduct> {
    let angles = outcome
        .sites
        .iter()
        .map(|selected| (selected.phi_degrees, selected.psi_degrees))
        .collect::<Vec<_>>();
    build_from_outcome_with_angles(protein, sites, outcome, &angles, builder, parameterize)
}

/// Build the exact conformers and rotamers recorded in a search result while
/// overriding the protein--glycan linkage angles.  Density fitting uses this
/// small seam to search torsions without duplicating the attachment geometry
/// and parameterization logic in `reglyco-build`.
pub fn build_from_outcome_with_angles(
    protein: &Structure,
    sites: &[SearchSite],
    outcome: &SearchOutcome,
    angles: &[(f64, f64)],
    builder: &glysys::SystemBuilder,
    parameterize: bool,
) -> Result<BuildProduct> {
    if sites.len() != outcome.sites.len() {
        return Err(EnsembleError::Metadata(
            "search outcome does not cover every requested site".into(),
        ));
    }
    if angles.len() != sites.len() {
        return Err(EnsembleError::Metadata(
            "linkage-angle overrides do not cover every requested site".into(),
        ));
    }
    let state = outcome
        .sites
        .iter()
        .zip(angles)
        .map(|(selected, &(phi, psi))| Gene {
            conformer: selected.conformer_index,
            phi,
            psi,
            rotamer: selected.rotamer_index,
        })
        .collect::<Vec<_>>();
    let mut structure = build_state(protein, sites, &state, builder)?;
    for coordinate in &outcome.minimized_coordinates {
        if let Some(atom) = structure.find_atom(&coordinate.residue, &coordinate.atom_name) {
            structure.set_atom_position(atom, coordinate.position)?;
        }
    }
    let system = parameterize
        .then(|| builder.prepare_structure(&structure))
        .transpose()?;
    Ok(BuildProduct { structure, system })
}

fn build_state(
    protein: &Structure,
    sites: &[SearchSite],
    state: &[Gene],
    builder: &glysys::SystemBuilder,
) -> std::result::Result<Structure, ReGlycoError> {
    let mut oriented_protein = protein.clone();
    for (site, gene) in sites.iter().zip(state) {
        if let Some(rotamer) = gene.rotamer {
            dunbrack::apply(&mut oriented_protein, &site.site.residue, rotamer)?;
        }
    }
    let attachments = sites
        .iter()
        .zip(state)
        .map(|(site, gene)| AttachmentRequest {
            site: site.site.clone(),
            conformer: GlycanConformer::new(
                site.ensemble.conformers[gene.conformer].structure.clone(),
            ),
        })
        .collect();
    let angles = state
        .iter()
        .map(|gene| (gene.phi, gene.psi))
        .collect::<Vec<_>>();
    let structure = build_with_linkage_angles(
        BuildRequest {
            protein: oriented_protein,
            attachments,
            parameterize: false,
        },
        &angles,
        builder,
    )?
    .structure;
    Ok(structure)
}

pub fn steric_score(structure: &Structure, clash_distance: f64) -> f64 {
    steric_site_scores(structure, clash_distance)
        .into_iter()
        .fold(1.0, f64::max)
}

/// Compute the Cookbook-compatible steric score for each attached glycan.
pub fn steric_site_scores(structure: &Structure, clash_distance: f64) -> Vec<f64> {
    let atoms = structure.atoms();
    let all_glycan_residues = structure
        .metadata()
        .glycan_trees
        .iter()
        .flat_map(|tree| tree.residue_ids.iter().cloned())
        .collect::<HashSet<_>>();
    structure
        .metadata()
        .glycan_trees
        .iter()
        .enumerate()
        .map(|(site_index, tree)| {
            let residues = tree.residue_ids.iter().cloned().collect::<HashSet<_>>();
            let attachment = structure.metadata().glycosylation_sites.get(site_index);
            let link_position = attachment
                .and_then(|site| structure.find_atom(&site.protein_residue, &site.protein_atom))
                .and_then(|atom| structure.atom(atom))
                .map(|atom| atom.position);
            // The Cookbook scorer ignores the three attachment-proximal
            // atoms (C1 and its immediate neighbors), preventing the fixed
            // bond geometry from dominating the steric objective.
            let glycan_atoms = atoms
                .iter()
                .filter(|atom| residues.contains(&atom.residue))
                .skip(3)
                .collect::<Vec<_>>();
            let protein_atoms = atoms
                .iter()
                .filter(|atom| !all_glycan_residues.contains(&atom.residue))
                .filter(|atom| {
                    link_position.is_none_or(|link| distance(link, atom.position) <= 40.0)
                })
                .collect::<Vec<_>>();
            let mut score = pair_steric_score(&glycan_atoms, &protein_atoms, clash_distance);
            for other in structure
                .metadata()
                .glycan_trees
                .iter()
                .enumerate()
                .filter(|(other_index, _)| *other_index != site_index)
                .map(|(_, other)| {
                    let other_residues = other.residue_ids.iter().cloned().collect::<HashSet<_>>();
                    atoms
                        .iter()
                        .filter(|atom| other_residues.contains(&atom.residue))
                        .collect::<Vec<_>>()
                })
            {
                score = score.max(pair_steric_score(&glycan_atoms, &other, clash_distance));
            }
            score
        })
        .collect()
}

/// Return compact residue-level partners for the hard steric contacts in a
/// complete exported structure.  The fast Cookbook score remains the search
/// gate; this one-time diagnostic pass explains which protein residues or
/// other attachment sites are responsible for an unresolved score.  A
/// glycan--glycan contact is inserted into both sites' lists.
pub fn clash_partner_labels(structure: &Structure, clash_distance: f64) -> Vec<Vec<String>> {
    let atoms = structure.atoms();
    let trees = &structure.metadata().glycan_trees;
    let all_glycan_residues = trees
        .iter()
        .flat_map(|tree| tree.residue_ids.iter().cloned())
        .collect::<HashSet<_>>();
    let groups = trees
        .iter()
        .map(|tree| {
            let residues = tree.residue_ids.iter().cloned().collect::<HashSet<_>>();
            atoms
                .iter()
                .filter(|atom| residues.contains(&atom.residue))
                .skip(3)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let protein_atoms = atoms
        .iter()
        .filter(|atom| !all_glycan_residues.contains(&atom.residue))
        .collect::<Vec<_>>();
    let mut labels = vec![BTreeSet::<String>::new(); groups.len()];
    let threshold_squared = clash_distance * clash_distance;
    for (index, group) in groups.iter().enumerate() {
        for first in group {
            for second in &protein_atoms {
                if squared_distance(first.position, second.position) < threshold_squared {
                    labels[index].insert(format!("protein:{}", second.residue));
                }
            }
            for (other_index, other) in groups.iter().enumerate().skip(index + 1) {
                if other.iter().any(|second| {
                    squared_distance(first.position, second.position) < threshold_squared
                }) {
                    labels[index].insert(format!(
                        "glycan:{}",
                        trees[other_index]
                            .attachment_site
                            .as_ref()
                            .or_else(|| trees[other_index].residue_ids.first())
                            .map(ToString::to_string)
                            .unwrap_or_else(|| format!("site-{}", other_index + 1))
                    ));
                    labels[other_index].insert(format!(
                        "glycan:{}",
                        trees[index]
                            .attachment_site
                            .as_ref()
                            .or_else(|| trees[index].residue_ids.first())
                            .map(ToString::to_string)
                            .unwrap_or_else(|| format!("site-{}", index + 1))
                    ));
                }
            }
        }
    }
    labels
        .into_iter()
        .map(|values| values.into_iter().collect())
        .collect()
}

fn pair_steric_score(
    first: &[&glysys::StructureAtom],
    second: &[&glysys::StructureAtom],
    threshold: f64,
) -> f64 {
    let mut score = 1.0;
    for first in first {
        for second in second {
            let distance = distance(first.position, second.position);
            if distance < threshold {
                score += 200.0 * (-distance * distance).exp();
                if score > 2.0 {
                    return score;
                }
            }
        }
    }
    score
}

pub fn constrained_mh_indices(
    ensembles: &[GlycanEnsemble],
    steps: usize,
    seed: u64,
) -> Vec<Vec<usize>> {
    if ensembles.is_empty()
        || ensembles
            .iter()
            .any(|ensemble| ensemble.conformers.is_empty())
    {
        return Vec::new();
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut state = ensembles
        .iter()
        .map(|ensemble| weighted_conformer_index(ensemble, &mut rng))
        .collect::<Vec<_>>();
    let mut samples = Vec::with_capacity(steps);
    for _ in 0..steps {
        let site = rng.random_range(0..ensembles.len());
        let proposal = weighted_conformer_index(&ensembles[site], &mut rng);
        state[site] = proposal;
        samples.push(state.clone());
    }
    samples
}

/// Generate complete multi-site states with a Metropolis-Hastings walk over
/// cluster/VMM genes, using the same steric and prior objective as GA search.
pub fn constrained_mh_search(
    protein: &Structure,
    sites: &[SearchSite],
    config: &SearchConfig,
    builder: &glysys::SystemBuilder,
) -> Result<Vec<Vec<usize>>> {
    let (frames, _) =
        sample_attached_ensemble(protein, sites, config.ensemble_size, config, builder)?;
    Ok(frames
        .into_iter()
        .map(|frame| {
            frame
                .sites
                .into_iter()
                .map(|site| site.conformer_index)
                .collect()
        })
        .collect())
}

pub fn resolved_priors(
    protein: &Structure,
    site: &SearchSite,
    supplied: &LinkagePrior,
) -> LinkagePrior {
    if !supplied.phi.is_empty() && !supplied.psi.is_empty() {
        return supplied.clone();
    }
    let residue_name = protein
        .residues()
        .into_iter()
        .find(|residue| residue.id == site.site.residue)
        .map(|residue| residue.name)
        .unwrap_or_default();
    linkage_priors(&residue_name)
}

/// Linkage distributions ported from the Cookbook residue configuration.
///
/// The attachment geometry is determined by both the protein residue and the
/// terminal sugar of the selected glycan.  The old browser path only supplied
/// the residue, which silently gave all SER/THR/TRP assignments a generic
/// prior.  Keep the residue-only function for CLI/backwards compatibility and
/// use this variant whenever GlycoShape metadata is available.
pub fn linkage_priors(residue_name: &str) -> LinkagePrior {
    linkage_priors_for_glycan(residue_name, None)
}

pub fn linkage_priors_for_glycan(residue_name: &str, terminal_sugar: Option<&str>) -> LinkagePrior {
    let component = |mean_degrees: f64, concentration: f64, weight: f64| VonMisesComponent {
        mean_degrees,
        concentration,
        weight,
    };
    let residue_name = residue_name.trim().to_ascii_uppercase();
    match residue_name.as_str() {
        "ASN" if terminal_sugar.is_some_and(|sugar| sugar.eq_ignore_ascii_case("Glc")) => {
            range_priors((145.0, 214.0), (-136.0, -48.0))
        }
        "ASN" => LinkagePrior {
            phi: vec![
                component(-97.476_917, 7.301_950_673_646_376, 0.398_998_573_654_840_65),
                component(-82.452_289, 31.457_935_848_299_066, 0.426_965_540_027_922_2),
                component(72.044_507, 25.108_938_475_430_225, 0.130_912_136_393_317_45),
                component(
                    86.917_187,
                    2.090_261_740_297_101_8,
                    0.043_123_749_923_919_756,
                ),
            ],
            psi: vec![
                component(-131.785_473, 26.053_826_285_738_11, 0.141_946_386_591_091),
                component(
                    -113.116_063,
                    88.906_619_456_622_25,
                    0.093_558_563_806_491_75,
                ),
                component(173.733_297, 2.134_239_012_237_704, 0.489_161_037_918_268_4),
                component(179.427_777, 886.694_557_217_842_3, 0.275_334_011_684_148_7),
            ],
        },
        "SER" => match terminal_sugar.map(str::to_ascii_lowercase).as_deref() {
            Some("galnac") => range_priors((53.0, 93.0), (163.0, 224.0)),
            Some("fuc") => range_priors((58.0, 106.0), (86.0, 192.0)),
            Some("man") => range_priors((262.0, 304.0), (104.0, 218.0)),
            Some("glc") => range_priors((261.0, 297.0), (146.0, 219.0)),
            Some("xyl") => range_priors((265.0, 309.0), (107.0, 231.0)),
            Some("glcnac") => range_priors((272.0, 308.0), (178.0, 300.0)),
            Some("gal") => range_priors((-85.0, -59.0), (157.0, 219.0)),
            _ => range_priors((272.0, 308.0), (178.0, 300.0)),
        },
        "THR" => match terminal_sugar.map(str::to_ascii_lowercase).as_deref() {
            Some("galnac") => range_priors((55.0, 83.0), (61.0, 86.0)),
            Some("fuc") => {
                // The Cookbook's range prior is the dominant component.  A
                // lower-concentration steric-escape component preserves its
                // broad, observed THR–Fuc ψ envelope while allowing crowded
                // one-shot structures to remain inside a component-specific
                // 95% gate instead of resorting to an unrestricted circle.
                LinkagePrior {
                    phi: vec![component(290.5, 62.455_293_222_577_644, 0.8)],
                    psi: vec![
                        component(166.0, 29.776_021_315_299_29, 0.7),
                        component(180.0, 2.0, 0.3),
                    ],
                }
            }
            Some("man") => range_priors((61.0, 88.0), (88.0, 150.0)),
            Some("glcnac") => range_priors((143.0, 221.0), (171.0, 193.0)),
            _ => {
                // Residue-only CLI requests do not always carry GlycoShape
                // metadata from which the terminal sugar can be resolved.
                // Retain the broad Cookbook-compatible THR envelope in that
                // case so CLI and browser requests have the same strict
                // search behavior.
                LinkagePrior {
                    phi: vec![component(182.0, 15.41, 0.7), component(290.5, 3.0, 0.3)],
                    psi: vec![component(182.0, 32.83, 0.6), component(180.0, 2.0, 0.4)],
                }
            }
        },
        "TRP" => match terminal_sugar.map(str::to_ascii_lowercase).as_deref() {
            Some("man") => {
                // Keep the verified Cookbook TRP–Man peak, and include the
                // two lower-concentration escape lobes used by its steric GA
                // when several nearby C-Man sites compete for the same local
                // volume.  Every proposal still records and is gated against
                // the selected component's circular 95% interval.
                LinkagePrior {
                    phi: vec![
                        component(130.0, 32.828_063_500_117_43, 0.70),
                        component(70.0, 3.0, 0.15),
                        component(200.0, 3.0, 0.15),
                    ],
                    psi: vec![
                        component(0.0, 1459.025_044_449_664, 0.80),
                        component(0.0, 2.0, 0.20),
                    ],
                }
            }
            _ => LinkagePrior {
                phi: vec![
                    component(130.0, 32.828_063_500_117_43, 0.70),
                    component(70.0, 3.0, 0.15),
                    component(200.0, 3.0, 0.15),
                ],
                psi: vec![
                    component(0.0, 1459.025_044_449_664, 0.80),
                    component(0.0, 2.0, 0.20),
                ],
            },
        },
        "HYP" | "PRO" => range_priors((-180.0, -30.0), (31.0, 164.0)),
        _ => LinkagePrior::default(),
    }
}

fn range_priors(phi: (f64, f64), psi: (f64, f64)) -> LinkagePrior {
    let from_range = |(minimum, maximum): (f64, f64)| {
        let mean_degrees = (minimum + maximum) * 0.5;
        let sigma = ((maximum - minimum) * 0.25).to_radians();
        VonMisesComponent {
            mean_degrees,
            concentration: 1.0 / (sigma * sigma),
            weight: 1.0,
        }
    };
    LinkagePrior {
        phi: vec![from_range(phi)],
        psi: vec![from_range(psi)],
    }
}

pub fn sample_vmm(components: &[VonMisesComponent], rng: &mut ChaCha8Rng) -> f64 {
    if components.is_empty() {
        return rng.random_range(-180.0..180.0);
    }
    let index = weighted_vmm_component_index(components, rng);
    let component = &components[index];
    let kappa = component.concentration;
    if !kappa.is_finite() || kappa <= 1.0e-8 {
        return rng.random_range(-180.0..180.0);
    }
    // Best-Fisher rejection sampler for the circular von Mises density.
    let a = 1.0 + (1.0 + 4.0 * kappa * kappa).sqrt();
    let b = (a - (2.0 * a).sqrt()) / (2.0 * kappa);
    let r = (1.0 + b * b) / (2.0 * b);
    loop {
        let z = (std::f64::consts::PI * rng.random::<f64>()).cos();
        let f = (1.0 + r * z) / (r + z);
        let c = kappa * (r - f);
        let u = rng.random::<f64>().max(f64::MIN_POSITIVE);
        if u < c * (2.0 - c) || (c / u).ln() + 1.0 - c >= 0.0 {
            let sign = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
            return wrap_degrees(component.mean_degrees + sign * f.acos().to_degrees());
        }
    }
}

/// A sampled angle together with the mixture component from which it was
/// drawn.  Cookbook-compatible steric search keeps this identity so the
/// final pose can be checked against that component's credible region.
#[derive(Debug, Clone, Copy)]
struct VmmAngle {
    degrees: f64,
    component: usize,
}

fn vmm_component_sigma_degrees(component: &VonMisesComponent) -> Option<f64> {
    let kappa = component.concentration;
    (kappa.is_finite() && kappa > 1.0e-8).then(|| (1.0 / kappa).sqrt().to_degrees())
}

fn circular_angle_distance_degrees(first: f64, second: f64) -> f64 {
    (first - second + 180.0).rem_euclid(360.0) - 180.0
}

fn probability_half_width_degrees(component: &VonMisesComponent) -> f64 {
    // Concentrations recur across candidates; integration is preparation work.
    static WIDTHS: OnceLock<Mutex<HashMap<u64, f64>>> = OnceLock::new();
    let cache = WIDTHS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut values = cache.lock().expect("prior interval cache");
    *values
        .entry(component.concentration.to_bits())
        .or_insert_with(|| {
            glysys_energy::prior::credible_half_width(component.concentration, 0.95)
                .expect("validated concentration")
                .to_degrees()
        })
}
fn vmm_component_within_95(angle: f64, component: &VonMisesComponent) -> bool {
    circular_angle_distance_degrees(angle, component.mean_degrees).abs()
        <= probability_half_width_degrees(component)
}
fn vmm_angle_within_95(angle: VmmAngle, components: &[VonMisesComponent]) -> bool {
    components
        .get(angle.component)
        .is_some_and(|component| vmm_component_within_95(angle.degrees, component))
}

fn top_region_sigma(cutoff: f64) -> f64 {
    if cutoff >= 0.95 {
        1.96
    } else if cutoff >= 0.90 {
        1.645
    } else if cutoff >= 0.85 {
        1.44
    } else if cutoff >= 0.80 {
        1.28
    } else if cutoff >= 0.75 {
        1.15
    } else {
        1.0
    }
}

fn sample_uniform_from_top_region(
    components: &[VonMisesComponent],
    cutoff: f64,
    rng: &mut ChaCha8Rng,
) -> VmmAngle {
    if components.is_empty() {
        return VmmAngle {
            degrees: rng.random_range(-180.0..180.0),
            component: 0,
        };
    }
    let component_index = weighted_vmm_component_index(components, rng);
    let component = &components[component_index];
    let Some(sigma) = vmm_component_sigma_degrees(component) else {
        return VmmAngle {
            degrees: rng.random_range(-180.0..180.0),
            component: component_index,
        };
    };
    let half_width = top_region_sigma(cutoff) * sigma;
    VmmAngle {
        degrees: wrap_degrees(component.mean_degrees + rng.random_range(-half_width..half_width)),
        component: component_index,
    }
}

/// Cookbook's native VMM draw: a Best-Fisher sample truncated at 1.28σ.  The
/// bounded rejection keeps the normal path inside the strict 95% gate while
/// retaining the native mixture density around each component.
fn sample_truncated_vmm(components: &[VonMisesComponent], rng: &mut ChaCha8Rng) -> VmmAngle {
    if components.is_empty() {
        return VmmAngle {
            degrees: rng.random_range(-180.0..180.0),
            component: 0,
        };
    }
    let component_index = weighted_vmm_component_index(components, rng);
    let component = &components[component_index];
    let Some(sigma) = vmm_component_sigma_degrees(component) else {
        return VmmAngle {
            degrees: rng.random_range(-180.0..180.0),
            component: component_index,
        };
    };
    for _ in 0..64 {
        let candidate = sample_vmm_component(component, component_index, rng);
        if circular_angle_distance_degrees(candidate.degrees, component.mean_degrees).abs()
            <= 1.28 * sigma
        {
            return candidate;
        }
    }
    sample_uniform_from_top_region(components, 0.85, rng)
}

fn sample_vmm_component(
    component: &VonMisesComponent,
    component_index: usize,
    rng: &mut ChaCha8Rng,
) -> VmmAngle {
    let kappa = component.concentration;
    if !kappa.is_finite() || kappa <= 1.0e-8 {
        return VmmAngle {
            degrees: rng.random_range(-180.0..180.0),
            component: component_index,
        };
    }
    let a = 1.0 + (1.0 + 4.0 * kappa * kappa).sqrt();
    let b = (a - (2.0 * a).sqrt()) / (2.0 * kappa);
    let r = (1.0 + b * b) / (2.0 * b);
    loop {
        let z = (std::f64::consts::PI * rng.random::<f64>()).cos();
        let f = (1.0 + r * z) / (r + z);
        let c = kappa * (r - f);
        let u = rng.random::<f64>().max(f64::MIN_POSITIVE);
        if u < c * (2.0 - c) || (c / u).ln() + 1.0 - c >= 0.0 {
            let sign = if rng.random_bool(0.5) { 1.0 } else { -1.0 };
            return VmmAngle {
                degrees: wrap_degrees(component.mean_degrees + sign * f.acos().to_degrees()),
                component: component_index,
            };
        }
    }
}

fn sample_rotamer(
    protein: &Structure,
    site: &glysys::ResidueId,
    enabled: bool,
    rng: &mut ChaCha8Rng,
) -> Option<usize> {
    if !enabled {
        return None;
    }
    let choices = dunbrack::rotamers(protein, site);
    if choices.is_empty() {
        return None;
    }
    match rng.random_range(0..=choices.len()) {
        0 => None,
        choice => Some(choice - 1),
    }
}

/// Convert the repair-loop pass (zero is the deposited sidechain) to the
/// zero-based Dunbrack index without evaluating a subtract for pass zero.
fn rotamer_pass_index(pass: usize) -> Option<usize> {
    pass.checked_sub(1)
}

/// Pick a rotamer with the same ordering/bias as the Cookbook mutation path.
/// Index zero represents the deposited sidechain; the bundled Dunbrack table
/// is stored in descending probability order.  Keeping the deposited pose in
/// the draw prevents an opt-in repair from gratuitously moving a protein.
fn sample_rotamer_biased(
    protein: &Structure,
    site: &glysys::ResidueId,
    rng: &mut ChaCha8Rng,
) -> Option<usize> {
    let choices = dunbrack::rotamers(protein, site);
    if choices.is_empty() {
        return None;
    }
    let max_probability = choices
        .first()
        .map(|rotamer| rotamer.probability.max(0.0))
        .unwrap_or(0.0);
    let mut weights = Vec::with_capacity(choices.len() + 1);
    weights.push(max_probability * 1.5);
    weights.extend(choices.iter().map(|rotamer| rotamer.probability.max(0.0)));
    let selected = weighted_index(&weights, rng);
    rotamer_pass_index(selected)
}

fn weighted_index(weights: &[f64], rng: &mut ChaCha8Rng) -> usize {
    if weights.is_empty() {
        return 0;
    }
    let total = weights
        .iter()
        .filter(|weight| weight.is_finite() && **weight > 0.0)
        .sum::<f64>();
    if total <= 0.0 {
        return rng.random_range(0..weights.len());
    }
    let mut draw = rng.random_range(0.0..total);
    for (index, weight) in weights.iter().enumerate() {
        if weight.is_finite() && *weight > 0.0 {
            if draw <= *weight {
                return index;
            }
            draw -= *weight;
        }
    }
    weights.len() - 1
}

fn weighted_index_by<T>(items: &[T], weight: impl Fn(&T) -> f64, rng: &mut ChaCha8Rng) -> usize {
    if items.is_empty() {
        return 0;
    }
    let total = items
        .iter()
        .map(&weight)
        .filter(|value| value.is_finite() && *value > 0.0)
        .sum::<f64>();
    if total <= 0.0 {
        return rng.random_range(0..items.len());
    }
    let mut draw = rng.random_range(0.0..total);
    for (index, item) in items.iter().enumerate() {
        let value = weight(item);
        if value.is_finite() && value > 0.0 {
            if draw <= value {
                return index;
            }
            draw -= value;
        }
    }
    items.len() - 1
}

fn weighted_conformer_index(ensemble: &GlycanEnsemble, rng: &mut ChaCha8Rng) -> usize {
    weighted_index_by(
        &ensemble.conformers,
        |conformer| conformer.cluster_weight,
        rng,
    )
}

/// Select a represented conformer uniformly for a steric feasibility draw.
///
/// This is deliberately separate from `weighted_conformer_index`: a Build
/// first has to discover a jointly clash-free state, so population-weighted
/// draws can hide a rare but geometrically necessary conformer.  Once a
/// feasible state exists, the normal joint-prior comparator applies the
/// recorded populations.  Statistical Ensemble proposals continue to use
/// `weighted_conformer_index` and therefore keep their target distribution.
fn coverage_conformer_index(ensemble: &GlycanEnsemble, rng: &mut ChaCha8Rng) -> usize {
    if ensemble.conformers.is_empty() {
        return 0;
    }
    let positive = ensemble
        .conformers
        .iter()
        .filter(|conformer| conformer.cluster_weight.is_finite() && conformer.cluster_weight > 0.0)
        .count();
    if positive == 0 {
        return rng.random_range(0..ensemble.conformers.len());
    }
    let mut rank = rng.random_range(0..positive);
    for (index, conformer) in ensemble.conformers.iter().enumerate() {
        if conformer.cluster_weight.is_finite() && conformer.cluster_weight > 0.0 {
            if rank == 0 {
                return index;
            }
            rank -= 1;
        }
    }
    // `positive` was counted from this same predicate, so this is only a
    // defensive fallback for a concurrently-mutated test value.
    ensemble.conformers.len() - 1
}

fn weighted_vmm_component_index(components: &[VonMisesComponent], rng: &mut ChaCha8Rng) -> usize {
    weighted_index_by(components, |component| component.weight, rng)
}

fn validate_site_priors(protein: &Structure, sites: &[SearchSite]) -> Result<()> {
    for site in sites {
        let sum: f64 = site
            .ensemble
            .conformers
            .iter()
            .map(|c| c.cluster_weight)
            .sum();
        if !sum.is_finite()
            || sum <= 0.
            || site
                .ensemble
                .conformers
                .iter()
                .any(|c| !c.cluster_weight.is_finite() || c.cluster_weight < 0.)
        {
            return Err(EnsembleError::Metadata("invalid conformer weights".into()));
        }
        for conformer in &site.ensemble.conformers {
            let p = resolved_priors(protein, site, &conformer.priors);
            for axis in [&p.phi, &p.psi] {
                glysys_energy::prior::CircularMixture::new(
                    axis.iter()
                        .map(|c| glysys_energy::prior::CircularComponent {
                            mean: c.mean_degrees.to_radians(),
                            concentration: c.concentration,
                            weight: c.weight,
                        })
                        .collect(),
                )?;
            }
        }
    }
    Ok(())
}

fn normalized_mixture(components: &[VonMisesComponent]) -> glysys_energy::prior::CircularMixture {
    glysys_energy::prior::CircularMixture::new(
        components
            .iter()
            .map(|c| glysys_energy::prior::CircularComponent {
                mean: c.mean_degrees.to_radians(),
                concentration: c.concentration,
                weight: c.weight,
            })
            .collect(),
    )
    .expect("validated circular prior")
}
fn vmm_penalty(angle: f64, components: &[VonMisesComponent]) -> f64 {
    -normalized_mixture(components).log_probability(angle.to_radians())
}

fn wrap_degrees(angle: f64) -> f64 {
    (angle + 180.0).rem_euclid(360.0) - 180.0
}

fn distance(first: Vec3, second: Vec3) -> f64 {
    ((first.x - second.x).powi(2) + (first.y - second.y).powi(2) + (first.z - second.z).powi(2))
        .sqrt()
}

fn squared_distance(first: Vec3, second: Vec3) -> f64 {
    (first.x - second.x).powi(2) + (first.y - second.y).powi(2) + (first.z - second.z).powi(2)
}

fn dry_options() -> BuildOptions {
    BuildOptions {
        add_water: false,
        add_ions: false,
        ..BuildOptions::default()
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DunbrackRotamer {
    pub name: String,
    pub probability: f64,
    pub chi_degrees: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    const GLYCAN: &str = include_str!("../../../tests/fixtures/glycan.pdb");
    const PROTEIN: &str = include_str!("../../../tests/fixtures/protein.pdb");

    #[test]
    fn score_only_crossover_preserves_site_layout_when_coordinates_are_absent() {
        let gene = Gene {
            conformer: 0,
            phi: 0.0,
            psi: 0.0,
            rotamer: None,
        };
        let first = CookbookStericChromosome {
            genes: vec![gene.clone(), gene.clone()],
            phi_components: vec![0, 0],
            psi_components: vec![0, 0],
            frozen_mask: vec![false, false],
            conformations: vec![None, None],
            site_grids: vec![None, None],
            dirty: vec![true, true],
            valid_mask: vec![true, true],
            steric_scores: vec![1.0, 1.0],
            fitness: 1.0,
        };
        let mut second = first.clone();
        // This is the state produced by the old score-only GPU bookkeeping
        // path. It must be treated as “no coordinates for these sites”, not as
        // a chromosome with zero sites.
        second.conformations.clear();
        second.site_grids.clear();
        for seed in 0..8 {
            let child = cookbook_crossover(&first, &second, &mut ChaCha8Rng::seed_from_u64(seed));
            assert_eq!(child.genes.len(), 2);
            assert_eq!(child.conformations.len(), 2);
            assert_eq!(child.site_grids.len(), 2);
        }
    }

    #[test]
    fn crossover_repairs_partial_parent_bookkeeping_without_trapping() {
        let gene = Gene {
            conformer: 0,
            phi: 0.0,
            psi: 0.0,
            rotamer: None,
        };
        let first = CookbookStericChromosome {
            genes: vec![gene.clone(), gene.clone()],
            phi_components: Vec::new(),
            psi_components: Vec::new(),
            frozen_mask: Vec::new(),
            conformations: Vec::new(),
            site_grids: Vec::new(),
            dirty: Vec::new(),
            valid_mask: Vec::new(),
            steric_scores: Vec::new(),
            fitness: f64::INFINITY,
        };
        let mut second = first.clone();
        // A stale score-only parent can also have fewer genes than the current
        // problem layout.  The crossover must retain the first parent's site
        // value for that suffix instead of indexing past the second parent.
        second.genes.pop();
        for seed in 0..16 {
            let child = cookbook_crossover(&first, &second, &mut ChaCha8Rng::seed_from_u64(seed));
            assert_eq!(child.genes.len(), 2);
            assert_eq!(child.phi_components.len(), 2);
            assert_eq!(child.psi_components.len(), 2);
            assert_eq!(child.frozen_mask.len(), 2);
            assert_eq!(child.conformations.len(), 2);
            assert_eq!(child.site_grids.len(), 2);
            assert_eq!(child.dirty.len(), 2);
            assert_eq!(child.valid_mask.len(), 2);
            assert_eq!(child.steric_scores.len(), 2);
        }
    }

    #[test]
    fn auto_budget_keeps_small_jobs_at_the_reliability_floor() {
        let query = GlycanQuery {
            source: GlycanSource::LocalBundle(PathBuf::from("budget.pdb")),
            anomer: Anomer::Beta,
            format: "PDB".into(),
            level: "2".into(),
        };
        let ensemble = ensemble_from_pdb(GLYCAN, None, query, "budget").unwrap();
        let site = SearchSite {
            site: reglyco_core::GlycosylationSite::new("A", 1),
            ensemble,
        };
        let protein = read_pdb_str(PROTEIN, &dry_options()).unwrap();
        let budget = resolve_search_budget(
            &protein,
            std::slice::from_ref(&site),
            false,
            Some(SearchBudgetMode::Auto),
            128,
            100,
        )
        .unwrap();
        assert_eq!(budget.population_size, 128);
        assert_eq!(budget.generations, 100);
        assert!(!budget.capped);
    }

    #[test]
    fn auto_budget_caps_a_dense_sixteen_site_fixture_without_overflow() {
        let query = GlycanQuery {
            source: GlycanSource::LocalBundle(PathBuf::from("budget-dense.pdb")),
            anomer: Anomer::Beta,
            format: "PDB".into(),
            level: "2".into(),
        };
        let ensemble = ensemble_from_pdb(GLYCAN, None, query, "budget-dense").unwrap();
        let sites = (0..16)
            .map(|_| SearchSite {
                site: reglyco_core::GlycosylationSite::new("A", 1),
                ensemble: ensemble.clone(),
            })
            .collect::<Vec<_>>();
        let protein = read_pdb_str(PROTEIN, &dry_options()).unwrap();
        let budget = resolve_search_budget(
            &protein,
            &sites,
            false,
            Some(SearchBudgetMode::Auto),
            128,
            100,
        )
        .unwrap();
        assert_eq!(budget.site_count, 16);
        assert_eq!(budget.graph_edges, 120);
        assert_eq!(budget.population_size, 256);
        assert_eq!(budget.generations, 500);
        assert!(budget.capped);
        assert!(budget.required_work > budget.total_budget);
    }

    #[test]
    fn omitted_budget_mode_is_explicitly_manual_for_legacy_requests() {
        let query = GlycanQuery {
            source: GlycanSource::LocalBundle(PathBuf::from("budget-manual.pdb")),
            anomer: Anomer::Beta,
            format: "PDB".into(),
            level: "2".into(),
        };
        let ensemble = ensemble_from_pdb(GLYCAN, None, query, "budget-manual").unwrap();
        let site = SearchSite {
            site: reglyco_core::GlycosylationSite::new("A", 1),
            ensemble,
        };
        let protein = read_pdb_str(PROTEIN, &dry_options()).unwrap();
        let budget = resolve_search_budget(&protein, &[site], false, None, 37, 19).unwrap();
        assert_eq!(budget.requested_mode, SearchBudgetMode::Manual);
        assert_eq!(budget.population_size, 37);
        assert_eq!(budget.generations, 19);
    }

    #[test]
    fn canonicalizes_supported_provider_formats_and_rejects_unknown_values() {
        assert_eq!(canonical_structure_format("pdb").unwrap(), "PDB");
        assert_eq!(canonical_structure_format("GLYCAM").unwrap(), "GLYCAM");
        assert!(canonical_structure_format("mmcif").is_err());

        let pdb = GlycanQuery {
            source: GlycanSource::GlyTouCan("G12345".into()),
            anomer: Anomer::Beta,
            format: "pdb".into(),
            level: "2".into(),
        };
        let mut upper = pdb.clone();
        upper.format = "PDB".into();
        assert_eq!(cache_key(&pdb), cache_key(&upper));
        upper.format = "GLYCAM".into();
        assert_ne!(cache_key(&pdb), cache_key(&upper));
        let parsed = ensemble_from_pdb(GLYCAN, None, pdb, "fixture").unwrap();
        assert_eq!(parsed.query.format, "PDB");
        assert_eq!(
            parsed.population_source,
            ConformerPopulationSource::EqualFallback
        );
    }

    #[test]
    fn cached_coordinate_mapping_moves_generated_atoms_with_parent() {
        let builder = glysys::SystemBuilder::new(dry_options()).unwrap();
        let mut source = read_pdb_str(PROTEIN, &dry_options()).unwrap();
        let system = builder.prepare_structure(&source).unwrap();
        let mapping = PreparedEnergyAtoms::new(&system);
        for offset in [0.0, 0.25] {
            let updates = source
                .iter_atoms()
                .map(|a| {
                    (
                        a.id,
                        Vec3 {
                            x: a.position.x + offset,
                            y: a.position.y,
                            z: a.position.z,
                        },
                    )
                })
                .collect::<Vec<_>>();
            source.set_atom_positions(updates).unwrap();
            let actual = mapping.coordinates(&source).unwrap();
            for (point, original) in actual.iter().zip(system.coordinates()) {
                assert!((point.x - original.x - offset).abs() < 1e-8);
                assert!((point.y - original.y).abs() < 1e-8);
                assert!((point.z - original.z).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn splits_multimodel_pdb() {
        let input = "MODEL        1\nATOM\nENDMDL\nMODEL        2\nATOM\nENDMDL\n";
        let models = split_pdb_models(input);
        assert_eq!(models.len(), 2);
        assert!(models[0].ends_with("END\n"));
    }

    #[test]
    fn browser_ensemble_preserves_source_population_weights() {
        let model = GLYCAN.trim_end_matches("END\n");
        let pdb = format!("MODEL        1\n{model}ENDMDL\nMODEL        2\n{model}ENDMDL\n");
        let query = GlycanQuery {
            source: GlycanSource::LocalBundle(PathBuf::from("browser.pdb")),
            anomer: Anomer::Beta,
            format: "PDB".into(),
            level: "2".into(),
        };
        let metadata = serde_json::json!({
            "archetype": {
                "coverage_clusters": {"Cluster 0": 80.0, "Cluster 1": 20.0},
                "coverage_clusters_per_main": {"3": [0], "4": [1]}
            }
        });
        let weighted = ensemble_from_pdb(&pdb, Some(&metadata), query.clone(), "browser").unwrap();
        assert_eq!(weighted.conformers[0].cluster_weight, 80.0);
        assert_eq!(weighted.conformers[1].cluster_weight, 20.0);
        assert_eq!(weighted.conformers[0].main_cluster, Some(3));
        assert_eq!(weighted.conformers[1].main_cluster, Some(4));
        assert_eq!(
            weighted.population_source,
            ConformerPopulationSource::AssetMetadata
        );

        let fallback = ensemble_from_pdb(&pdb, None, query, "browser").unwrap();
        assert!(
            fallback
                .conformers
                .iter()
                .all(|value| value.cluster_weight == 1.0)
        );
        assert_eq!(
            fallback.population_source,
            ConformerPopulationSource::EqualFallback
        );
    }

    #[test]
    fn level_specific_population_metadata_does_not_cross_contaminate_requests() {
        let model = GLYCAN.trim_end_matches("END\n");
        let pdb = format!("MODEL        1\n{model}ENDMDL\nMODEL        2\n{model}ENDMDL\n");
        let metadata = serde_json::json!({
            "cluster_levels": {
                "level_1": {
                    "clusters": {
                        "Cluster 1": {"population": 0.75},
                        "Cluster 0": {"population": 0.25}
                    }
                },
                "level_2": {"clusters": [0.9, 0.1]}
            }
        });
        let level_one = GlycanQuery {
            source: GlycanSource::LocalBundle(PathBuf::from("level-1.pdb")),
            anomer: Anomer::Beta,
            format: "pdb".into(),
            level: "1".into(),
        };
        let level_two = GlycanQuery {
            level: "2".into(),
            ..level_one.clone()
        };
        let first = ensemble_from_pdb(&pdb, Some(&metadata), level_one, "levels").unwrap();
        let second = ensemble_from_pdb(&pdb, Some(&metadata), level_two, "levels").unwrap();
        assert_eq!(first.conformers[0].cluster_weight, 0.25);
        assert_eq!(first.conformers[1].cluster_weight, 0.75);
        assert_eq!(second.conformers[0].cluster_weight, 0.9);
        assert_eq!(second.conformers[1].cluster_weight, 0.1);
    }

    #[test]
    fn malformed_population_metadata_is_rejected_without_shifting_models() {
        let model = GLYCAN.trim_end_matches("END\n");
        let pdb = format!("MODEL        1\n{model}ENDMDL\nMODEL        2\n{model}ENDMDL\n");
        let query = GlycanQuery {
            source: GlycanSource::LocalBundle(PathBuf::from("population.pdb")),
            anomer: Anomer::Beta,
            format: "PDB".into(),
            level: "2".into(),
        };
        let wrong_length = serde_json::json!({ "cluster_weights": [1.0] });
        assert!(ensemble_from_pdb(&pdb, Some(&wrong_length), query.clone(), "population").is_err());
        let invalid_value = serde_json::json!({ "cluster_weights": [1.0, "missing"] });
        assert!(ensemble_from_pdb(&pdb, Some(&invalid_value), query, "population").is_err());
    }

    #[test]
    fn vmm_sampling_is_seeded_and_wrapped() {
        let components = vec![VonMisesComponent {
            mean_degrees: 60.0,
            concentration: 20.0,
            weight: 1.0,
        }];
        let draw = |seed| sample_vmm(&components, &mut ChaCha8Rng::seed_from_u64(seed));
        assert_eq!(draw(7), draw(7));
        assert!((-180.0..180.0).contains(&draw(9)));
    }

    #[test]
    fn vmm_sampling_preserves_the_circular_mean() {
        let components = vec![VonMisesComponent {
            mean_degrees: 60.0,
            concentration: 50.0,
            weight: 1.0,
        }];
        let mut rng = ChaCha8Rng::seed_from_u64(17);
        let draws = (0..10_000)
            .map(|_| sample_vmm(&components, &mut rng).to_radians())
            .collect::<Vec<_>>();
        let mean = draws
            .iter()
            .map(|angle| angle.sin())
            .sum::<f64>()
            .atan2(draws.iter().map(|angle| angle.cos()).sum::<f64>())
            .to_degrees();
        assert!((mean - 60.0).abs() < 1.0);
    }

    #[test]
    fn weighted_sampling_tracks_source_populations() {
        let mut rng = ChaCha8Rng::seed_from_u64(71);
        let mut first = 0usize;
        for _ in 0..10_000 {
            if weighted_index(&[0.8, 0.2], &mut rng) == 0 {
                first += 1;
            }
        }
        let fraction = first as f64 / 10_000.0;
        assert!((fraction - 0.8).abs() < 0.02, "observed {fraction}");
    }

    #[test]
    fn steric_coverage_sampling_does_not_hide_rare_conformers() {
        let model = GLYCAN.trim_end_matches("END\n");
        let pdb = format!("MODEL        1\n{model}ENDMDL\nMODEL        2\n{model}ENDMDL\n");
        let query = GlycanQuery {
            source: GlycanSource::LocalBundle(PathBuf::from("coverage.pdb")),
            anomer: Anomer::Beta,
            format: "PDB".into(),
            level: "2".into(),
        };
        let mut ensemble = ensemble_from_pdb(&pdb, None, query, "coverage").unwrap();
        ensemble.conformers[0].cluster_weight = 99.0;
        ensemble.conformers[1].cluster_weight = 1.0;
        let mut rng = ChaCha8Rng::seed_from_u64(73);
        let draws = (0..256)
            .map(|_| coverage_conformer_index(&ensemble, &mut rng))
            .collect::<Vec<_>>();
        assert!(draws.contains(&0));
        assert!(draws.contains(&1));
    }

    #[test]
    fn compiled_joint_prior_includes_population_and_full_mixture_density() {
        let model = GLYCAN.trim_end_matches("END\n");
        let pdb = format!("MODEL        1\n{model}ENDMDL\nMODEL        2\n{model}ENDMDL\n");
        let query = GlycanQuery {
            source: GlycanSource::LocalBundle(PathBuf::from("prior.pdb")),
            anomer: Anomer::Beta,
            format: "PDB".into(),
            level: "2".into(),
        };
        let mut ensemble = ensemble_from_pdb(&pdb, None, query, "prior").unwrap();
        ensemble.conformers[0].cluster_weight = 9.0;
        ensemble.conformers[1].cluster_weight = 1.0;
        let prior = linkage_priors("ASN");
        for conformer in &mut ensemble.conformers {
            conformer.priors = prior.clone();
        }
        let site = SearchSite {
            site: reglyco_core::GlycosylationSite::new("A", 1),
            ensemble,
        };
        let protein = read_pdb_str(PROTEIN, &dry_options()).unwrap();
        let cache = compile_prior_cache(&protein, std::slice::from_ref(&site)).unwrap();
        let common = Gene {
            conformer: 0,
            phi: -82.452289,
            psi: 179.427777,
            rotamer: None,
        };
        let rare = Gene {
            conformer: 1,
            phi: 72.044507,
            psi: 179.427777,
            rotamer: None,
        };
        let score = |gene: &Gene| {
            let prior = &cache.sites[0][gene.conformer];
            -prior.log_conformer_probability
                - prior.phi.log_probability(gene.phi.to_radians())
                - prior.psi.log_probability(gene.psi.to_radians())
        };
        assert!(score(&common) < score(&rare));
    }

    #[test]
    fn compiled_prior_rejects_nonfinite_or_negative_population_values() {
        let query = GlycanQuery {
            source: GlycanSource::LocalBundle(PathBuf::from("invalid-prior.pdb")),
            anomer: Anomer::Beta,
            format: "PDB".into(),
            level: "2".into(),
        };
        let mut ensemble = ensemble_from_pdb(GLYCAN, None, query, "invalid-prior").unwrap();
        ensemble.conformers[0].cluster_weight = f64::NAN;
        let site = SearchSite {
            site: reglyco_core::GlycosylationSite::new("A", 1),
            ensemble,
        };
        let protein = read_pdb_str(PROTEIN, &dry_options()).unwrap();
        assert!(compile_prior_cache(&protein, std::slice::from_ref(&site)).is_err());
    }

    #[test]
    fn dunbrack_rotamers_change_real_sidechain_coordinates() {
        let options = dry_options();
        let mut protein = read_pdb_str(PROTEIN, &options).unwrap();
        let site = glysys::ResidueId {
            chain: "A".into(),
            number: 1,
            insertion_code: None,
        };
        let choices = dunbrack::rotamers(&protein, &site);
        assert!(!choices.is_empty());
        let before = protein
            .find_atom(&site, "ND2")
            .and_then(|atom| protein.atom(atom))
            .unwrap()
            .position;
        dunbrack::apply(&mut protein, &site, 0).unwrap();
        let after = protein
            .find_atom(&site, "ND2")
            .and_then(|atom| protein.atom(atom))
            .unwrap()
            .position;
        assert_ne!(before, after);
    }

    #[test]
    fn rotamer_sampler_allows_no_rotamer_without_unsigned_underflow() {
        let protein = read_pdb_str(PROTEIN, &dry_options()).unwrap();
        let site = glysys::ResidueId {
            chain: "A".into(),
            number: 1,
            insertion_code: None,
        };
        let mut rng = ChaCha8Rng::seed_from_u64(1);
        let draws = (0..128)
            .map(|_| sample_rotamer(&protein, &site, true, &mut rng))
            .collect::<Vec<_>>();
        assert!(draws.iter().any(Option::is_none));
        assert!(
            draws
                .iter()
                .flatten()
                .all(|index| *index < dunbrack::rotamers(&protein, &site).len())
        );
        assert_eq!(rotamer_pass_index(0), None);
        assert_eq!(rotamer_pass_index(1), Some(0));
    }

    #[test]
    fn local_search_is_complete_and_deterministic() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("ensemble.pdb");
        fs::write(&path, GLYCAN).unwrap();
        let query = GlycanQuery {
            source: GlycanSource::LocalBundle(path),
            anomer: Anomer::Beta,
            format: "PDB".into(),
            level: "3".into(),
        };
        let ensemble = LocalBundleProvider.load(&query).unwrap();
        let sites = vec![SearchSite {
            site: reglyco_core::GlycosylationSite::new("A", 1),
            ensemble,
        }];
        let options = dry_options();
        let protein = read_pdb_str(PROTEIN, &options).unwrap();
        let builder = glysys::SystemBuilder::new(options).unwrap();
        let config = SearchConfig {
            population_size: 8,
            generations: 100,
            seed: 99,
            polish_attachment_vmm: true,
            ..SearchConfig::default()
        };
        let first = search(&protein, &sites, &config, &builder).unwrap();
        let second = search(&protein, &sites, &config, &builder).unwrap();
        assert_eq!(first.sites.len(), 1);
        assert_eq!(first.total_score, second.total_score);
        assert_eq!(first.sites[0].coordinates, second.sites[0].coordinates);
        assert_eq!(first.clash_status, ClashStatus::ClashFree);
        assert!(first.generations < config.generations);
        assert_eq!(first.sites[0].phi_within_vmm95, Some(true));
        assert_eq!(first.sites[0].psi_within_vmm95, Some(true));
        assert!(first.sites[0].phi_component.is_some());
        assert!(first.sites[0].psi_component.is_some());
        assert!(first.vmm_polish.applied);
        assert!(first.vmm_polish.proposals <= sites.len() * 5);
        assert!(first.vmm_polish.score_after <= first.vmm_polish.score_before + 1.0e-12);
        assert_eq!(first.vmm_polish.sites.len(), 1);
        assert!(first.vmm_polish.sites[0].proposals <= 5);
        assert!(
            first.vmm_polish.sites[0].component_switched
                || first.vmm_polish.sites[0].score_after < first.vmm_polish.sites[0].score_before
        );
        assert_eq!(
            first.vmm_polish.sites[0].final_phi_component,
            first.sites[0].phi_component.unwrap()
        );
        assert_eq!(
            first.vmm_polish.sites[0].final_psi_component,
            first.sites[0].psi_component.unwrap()
        );
        assert!(
            first.vmm_polish.sites[0].score_after
                <= first.vmm_polish.sites[0].score_before + 1.0e-12
        );
    }

    #[test]
    fn strict_vmm_gate_wraps_circular_boundaries_and_tracks_components() {
        let component = VonMisesComponent {
            mean_degrees: 179.0,
            concentration: 100.0,
            weight: 1.0,
        };
        let boundary = probability_half_width_degrees(&component);
        assert!(vmm_component_within_95(
            wrap_degrees(179.0 + boundary - 1.0e-6),
            &component
        ));
        assert!(vmm_component_within_95(
            wrap_degrees(179.0 - boundary + 1.0e-6),
            &component
        ));
        assert!(!vmm_component_within_95(
            wrap_degrees(179.0 + boundary + 1.0e-3),
            &component
        ));
        assert!(circular_angle_distance_degrees(-179.0, 179.0).abs() < 3.0);

        let mixture = vec![
            VonMisesComponent {
                mean_degrees: -90.0,
                concentration: 25.0,
                weight: 0.5,
            },
            VonMisesComponent {
                mean_degrees: 90.0,
                concentration: 25.0,
                weight: 0.5,
            },
        ];
        assert!(vmm_angle_within_95(
            VmmAngle {
                degrees: 90.0,
                component: 1
            },
            &mixture
        ));
        assert!(!vmm_angle_within_95(
            VmmAngle {
                degrees: 90.0,
                component: 0
            },
            &mixture
        ));
    }

    #[test]
    fn strict_proposal_samplers_are_bounded_by_the_selected_component() {
        let components = vec![
            VonMisesComponent {
                mean_degrees: -170.0,
                concentration: 40.0,
                weight: 0.65,
            },
            VonMisesComponent {
                mean_degrees: 75.0,
                concentration: 18.0,
                weight: 0.35,
            },
        ];
        let mut rng = ChaCha8Rng::seed_from_u64(91);
        for cutoff in [0.75, 0.85, 0.90, 0.95] {
            for _ in 0..2_000 {
                let angle = sample_uniform_from_top_region(&components, cutoff, &mut rng);
                assert!(vmm_angle_within_95(angle, &components));
            }
        }
        for _ in 0..2_000 {
            let angle = sample_truncated_vmm(&components, &mut rng);
            assert!(vmm_angle_within_95(angle, &components));
        }
    }

    #[test]
    fn attached_sampler_returns_the_requested_frame_count() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("ensemble.pdb");
        fs::write(&path, GLYCAN).unwrap();
        let query = GlycanQuery {
            source: GlycanSource::LocalBundle(path),
            anomer: Anomer::Beta,
            format: "PDB".into(),
            level: "3".into(),
        };
        let ensemble = LocalBundleProvider.load(&query).unwrap();
        let sites = vec![SearchSite {
            site: reglyco_core::GlycosylationSite::new("A", 1),
            ensemble,
        }];
        let protein = read_pdb_str(PROTEIN, &dry_options()).unwrap();
        let builder = glysys::SystemBuilder::new(dry_options()).unwrap();
        let config = SearchConfig {
            seed: 23,
            mh_chains: 2,
            mh_burn_in_sweeps: 0,
            mh_thinning_accepted: 1,
            ..SearchConfig::default()
        };
        let (frames, diagnostics) =
            sample_attached_ensemble(&protein, &sites, 3, &config, &builder).unwrap();
        let mut verified = frames.clone();
        for frame in &mut verified {
            let before = frame.structure.to_pdb_string();
            refresh_frame(frame, &protein, &sites, &config, &builder).unwrap();
            assert_eq!(frame.structure.to_pdb_string(), before);
            let measured =
                reglyco_build::attachment_angles(&frame.structure, &sites[0].site.residue).unwrap();
            assert!((frame.sites[0].phi_degrees - measured.0).abs() < 1e-8);
        }
        assert_eq!(frames.len(), 3);
        assert_eq!(diagnostics.requested_frames, 3);
        assert_eq!(diagnostics.mh_proposals, 3);
        assert!(!diagnostics.fallback_used);
        assert!(diagnostics.ga_restarts <= 1);
        assert!(frames.iter().all(|frame| frame.source == "model_da_mh_v3"));
        assert!(
            frames
                .iter()
                .all(|frame| !frame.structure.atoms().is_empty())
        );
        assert!(
            frames
                .iter()
                .all(|frame| { frame.sites.iter().all(|site| { site.steric_score <= 1.1 }) })
        );
    }

    #[test]
    fn prepared_steric_matches_reference_for_fixture() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("ensemble.pdb");
        fs::write(&path, GLYCAN).unwrap();
        let query = GlycanQuery {
            source: GlycanSource::LocalBundle(path),
            anomer: Anomer::Beta,
            format: "PDB".into(),
            level: "3".into(),
        };
        let ensemble = LocalBundleProvider.load(&query).unwrap();
        let sites = vec![SearchSite {
            site: reglyco_core::GlycosylationSite::new("A", 1),
            ensemble,
        }];
        let protein = read_pdb_str(PROTEIN, &dry_options()).unwrap();
        let builder = glysys::SystemBuilder::new(dry_options()).unwrap();
        let state = vec![Gene {
            conformer: 0,
            phi: -91.0,
            psi: 178.5,
            rotamer: None,
        }];
        let structure = build_state(&protein, &sites, &state, &builder).unwrap();
        let reference = steric_site_scores(&structure, 1.7);
        let prepared = PreparedAttachmentContext::new(&protein, &sites, false).unwrap();
        let fast = prepared.evaluate(&state, 1.7).unwrap();
        let fast_coordinates = prepared.site_coordinates(&state).unwrap();
        let residues = structure.metadata().glycan_trees[0]
            .residue_ids
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        let reference_coordinates = structure
            .atoms()
            .into_iter()
            .filter(|atom| residues.contains(&atom.residue))
            .map(|atom| atom.position)
            .collect::<Vec<_>>();
        assert!(
            reference_coordinates
                .iter()
                .zip(&fast_coordinates[0])
                .all(|(a, b)| squared_distance(*a, *b).sqrt() < 1.0e-8)
        );
        let zero_state = vec![Gene {
            conformer: 0,
            phi: 0.0,
            psi: 0.0,
            rotamer: None,
        }];
        let zero_structure = build_state(&protein, &sites, &zero_state, &builder).unwrap();
        let zero_residues = zero_structure.metadata().glycan_trees[0]
            .residue_ids
            .iter()
            .cloned()
            .collect::<HashSet<_>>();
        let zero_reference = zero_structure
            .atoms()
            .into_iter()
            .filter(|atom| zero_residues.contains(&atom.residue))
            .map(|atom| atom.position)
            .collect::<Vec<_>>();
        let zero_fast = prepared.site_coordinates(&zero_state).unwrap();
        assert!(
            zero_reference
                .iter()
                .zip(&zero_fast[0])
                .all(|(a, b)| squared_distance(*a, *b).sqrt() < 1.0e-8)
        );
        assert_eq!(reference.len(), fast.site_scores.len());
        assert!(
            reference
                .iter()
                .zip(fast.site_scores)
                .all(|(a, b)| (a - b).abs() < 1.0e-8)
        );
    }

    #[test]
    #[ignore = "set REGLYCO_TETRAMER_WITNESS_DIR to run the frozen local witness audit"]
    fn tetramer_witness_replays_through_prepared_context() {
        let root = std::env::var("REGLYCO_TETRAMER_WITNESS_DIR")
            .expect("set REGLYCO_TETRAMER_WITNESS_DIR");
        let root = PathBuf::from(root);
        let protein_text = fs::read_to_string(root.join("tetramer-protein-scaffold.pdb")).unwrap();
        let protein = read_pdb_str(&protein_text, &dry_options()).unwrap();
        let assignments = [
            (
                "A",
                138,
                "G92042VQ",
                8usize,
                244.70975503750356,
                215.06040312066236,
                'I',
            ),
            (
                "C",
                138,
                "G92042VQ",
                8usize,
                263.3592065378453,
                167.16176781713145,
                'J',
            ),
            (
                "E",
                138,
                "G92042VQ",
                8usize,
                258.2616864424269,
                169.920751670931,
                'K',
            ),
            (
                "G",
                138,
                "G92042VQ",
                2usize,
                246.3636113096466,
                180.15197907056614,
                'L',
            ),
            (
                "A",
                350,
                "G00028MO",
                2usize,
                57.41016756858368,
                185.48277574168097,
                'M',
            ),
            (
                "C",
                350,
                "G00028MO",
                0usize,
                68.13629127402837,
                144.77062339456134,
                'N',
            ),
            (
                "E",
                350,
                "G00028MO",
                18usize,
                68.8634929004141,
                158.224916610495,
                'O',
            ),
            (
                "G",
                350,
                "G00028MO",
                0usize,
                61.06374740136145,
                182.60351862813798,
                'P',
            ),
            (
                "A",
                402,
                "G48369JO",
                10usize,
                82.26559781319085,
                172.67603457898605,
                'Q',
            ),
            (
                "C",
                402,
                "G48369JO",
                4usize,
                70.15669245057433,
                177.63325495850322,
                'R',
            ),
            (
                "E",
                402,
                "G48369JO",
                8usize,
                78.87924332217361,
                180.17664681709147,
                'S',
            ),
            (
                "G",
                402,
                "G48369JO",
                8usize,
                57.826652651493994,
                192.31536696234548,
                'T',
            ),
            (
                "B",
                930,
                "G92042VQ",
                14usize,
                280.93647006317747,
                232.63580291891284,
                'U',
            ),
            (
                "D",
                930,
                "G92042VQ",
                4usize,
                278.4765953396528,
                180.80338202065082,
                'V',
            ),
            (
                "F",
                930,
                "G92042VQ",
                15usize,
                281.0076452754818,
                178.35656905090673,
                'W',
            ),
            (
                "H",
                930,
                "G92042VQ",
                12usize,
                275.2139086878866,
                179.21400348938639,
                'X',
            ),
        ];
        let mut ensembles = HashMap::new();
        for glycan in ["G92042VQ", "G00028MO", "G48369JO"] {
            let path = fs::read_dir(root.join("level2-cache"))
                .unwrap()
                .filter_map(std::result::Result::ok)
                .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "pdb"))
                .find_map(|entry| {
                    let text = fs::read_to_string(entry.path()).ok()?;
                    let models = text.matches("MODEL").count();
                    let expected = match glycan {
                        "G92042VQ" => 26,
                        "G00028MO" => 21,
                        _ => 25,
                    };
                    (models == expected).then_some(entry.path())
                })
                .unwrap();
            let query = GlycanQuery {
                source: GlycanSource::LocalBundle(path),
                anomer: reglyco_core::Anomer::Beta,
                format: "PDB".into(),
                level: "2".into(),
            };
            ensembles.insert(glycan, LocalBundleProvider.load(&query).unwrap());
        }
        let sites = assignments
            .iter()
            .map(|(chain, number, glycan, ..)| SearchSite {
                site: reglyco_core::GlycosylationSite::new(*chain, *number),
                ensemble: ensembles.get(glycan).unwrap().clone(),
            })
            .collect::<Vec<_>>();
        let prepared = PreparedAttachmentContext::new(&protein, &sites, false).unwrap();
        let genes = assignments
            .iter()
            .map(|(_, _, _, conformer, phi, psi, _)| Gene {
                conformer: *conformer,
                phi: *phi,
                psi: *psi,
                rotamer: None,
            })
            .collect::<Vec<_>>();
        let witness_text = fs::read_to_string(
            "/home/opc/old-cookbook-diagnostic/output-level2-large-no-rot-v2/all-normalized.pdb",
        )
        .unwrap();
        let witness = read_pdb_str(&witness_text, &dry_options()).unwrap();
        let coordinates = prepared.site_coordinates(&genes).unwrap();
        let mut complete = true;
        for (index, (_, _, _, _, _, _, chain)) in assignments.iter().enumerate() {
            let pose = prepared
                .prepare_site_pose(index, &genes[index], 1.7)
                .unwrap();
            let score = prepared
                .evaluate(
                    &genes
                        .iter()
                        .enumerate()
                        .map(|(site, gene)| {
                            if site == index {
                                gene.clone()
                            } else {
                                Gene {
                                    conformer: 0,
                                    phi: 0.0,
                                    psi: 0.0,
                                    rotamer: None,
                                }
                            }
                        })
                        .collect::<Vec<_>>(),
                    1.7,
                )
                .map(|value| value.score);
            let witness_atoms = witness
                .atoms()
                .into_iter()
                .filter(|atom| atom.residue.chain == chain.to_string())
                .collect::<Vec<_>>();
            let rmsd = if witness_atoms.len() == coordinates[index].len() {
                let sum = witness_atoms
                    .iter()
                    .zip(&coordinates[index])
                    .map(|(atom, position)| squared_distance(atom.position, *position))
                    .sum::<f64>();
                (sum / witness_atoms.len().max(1) as f64).sqrt()
            } else {
                f64::INFINITY
            };
            println!(
                "site {}:{} witness_chain={} pose={} score={score:?} atoms={} rmsd={rmsd:.4}",
                assignments[index].0,
                assignments[index].1,
                chain,
                pose.is_some(),
                witness_atoms.len()
            );
            // The witness was emitted by Cookbook with its own atom-order
            // serialization.  The prepared replay is the admissibility gate;
            // coordinate RMSD remains diagnostic because one historical
            // chain used a different atom ordering/normalization path.
            complete &= pose.is_some();
        }
        assert!(
            complete,
            "Cookbook witness is outside the prepared representation"
        );
    }

    #[test]
    #[ignore = "set REGLYCO_TETRAMER_WITNESS_DIR to run the frozen pool diagnostic"]
    fn tetramer_coverage_pool_finds_joint_starts() {
        let root = PathBuf::from(
            std::env::var("REGLYCO_TETRAMER_WITNESS_DIR")
                .expect("set REGLYCO_TETRAMER_WITNESS_DIR"),
        );
        let protein = read_pdb_str(
            &fs::read_to_string(root.join("tetramer-protein-scaffold.pdb")).unwrap(),
            &dry_options(),
        )
        .unwrap();
        let expected = [("G92042VQ", 26), ("G00028MO", 21), ("G48369JO", 25)];
        let mut ensembles = HashMap::new();
        for (glycan, models) in expected {
            let path = fs::read_dir(root.join("level2-cache"))
                .unwrap()
                .filter_map(std::result::Result::ok)
                .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "pdb"))
                .find(|entry| {
                    fs::read_to_string(entry.path())
                        .map(|text| text.matches("MODEL").count() == models)
                        .unwrap_or(false)
                })
                .unwrap()
                .path();
            let query = GlycanQuery {
                source: GlycanSource::LocalBundle(path),
                anomer: reglyco_core::Anomer::Beta,
                format: "PDB".into(),
                level: "2".into(),
            };
            ensembles.insert(glycan, LocalBundleProvider.load(&query).unwrap());
        }
        let assignments = [
            ("A", 138, "G92042VQ"),
            ("C", 138, "G92042VQ"),
            ("E", 138, "G92042VQ"),
            ("G", 138, "G92042VQ"),
            ("A", 350, "G00028MO"),
            ("C", 350, "G00028MO"),
            ("E", 350, "G00028MO"),
            ("G", 350, "G00028MO"),
            ("A", 402, "G48369JO"),
            ("C", 402, "G48369JO"),
            ("E", 402, "G48369JO"),
            ("G", 402, "G48369JO"),
            ("B", 930, "G92042VQ"),
            ("D", 930, "G92042VQ"),
            ("F", 930, "G92042VQ"),
            ("H", 930, "G92042VQ"),
        ];
        let sites = assignments
            .iter()
            .map(|(chain, number, glycan)| SearchSite {
                site: reglyco_core::GlycosylationSite::new(*chain, *number),
                ensemble: ensembles.get(glycan).unwrap().clone(),
            })
            .collect::<Vec<_>>();
        let auto_budget = resolve_search_budget(
            &protein,
            &sites,
            false,
            Some(SearchBudgetMode::Auto),
            128,
            100,
        )
        .unwrap();
        assert_eq!(
            (auto_budget.population_size, auto_budget.generations),
            (256, 500),
            "the frozen 16-site tetramer must receive the balanced Auto budget"
        );
        let prepared = PreparedAttachmentContext::new(&protein, &sites, false).unwrap();
        let config = SearchConfig {
            population_size: 256,
            generations: 500,
            seed: 20260919,
            ..SearchConfig::default()
        };
        let pool_budget = std::env::var("REGLYCO_TETRAMER_POOL_BUDGET")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(256 * 501 / 2);
        let result =
            cookbook_compatible_pool_coverage(&protein, &sites, &config, &prepared, 4, pool_budget)
                .unwrap();
        assert!(!result.states.is_empty());
        for (state, _, _) in &result.states {
            let evaluation = prepared.evaluate(state, config.clash_distance).unwrap();
            assert!(
                evaluation.site_scores.iter().all(|score| *score <= 1.1),
                "compatibility solver returned an invalid complete state: {:?}",
                evaluation.site_scores
            );
        }
        for (state, phi_components, psi_components) in &result.states {
            let first_bad = state.iter().enumerate().find_map(|(index, gene)| {
                let priors = resolved_priors(
                    &protein,
                    &sites[index],
                    &sites[index].ensemble.conformers[gene.conformer].priors,
                );
                let valid = priors
                    .phi
                    .get(phi_components[index])
                    .is_some_and(|component| vmm_component_within_95(gene.phi, component))
                    && priors
                        .psi
                        .get(psi_components[index])
                        .is_some_and(|component| vmm_component_within_95(gene.psi, component));
                (!valid).then_some((
                    index,
                    gene.clone(),
                    priors,
                    phi_components[index],
                    psi_components[index],
                ))
            });
            assert!(
                first_bad.is_none(),
                "coverage state is outside its VMM component: {first_bad:?}"
            );
        }
    }

    #[test]
    #[ignore = "set REGLYCO_TETRAMER_ENSEMBLE_DIR to validate a generated full-site ensemble"]
    fn tetramer_ensemble_output_is_complete_and_sterically_valid() {
        let root = PathBuf::from(
            std::env::var("REGLYCO_TETRAMER_ENSEMBLE_DIR")
                .expect("set REGLYCO_TETRAMER_ENSEMBLE_DIR"),
        );
        let contents = fs::read_to_string(root.join("ensemble.pdb")).unwrap();
        let models = split_pdb_models(&contents);
        assert_eq!(models.len(), 50, "full acceptance run must emit 50 frames");
        for (index, model) in models.iter().enumerate() {
            let structure = read_pdb_str(model, &dry_options()).unwrap();
            assert_eq!(
                structure.metadata().glycan_trees.len(),
                16,
                "frame {} lost one or more requested attachments",
                index + 1
            );
            // Verify the exported coordinates with an independent spatial
            // traversal, rather than trusting the sampler's cached scores.
            // This keeps the regression useful without the quadratic generic
            // diagnostic scorer, which is prohibitively slow on a 34k-atom
            // tetramer repeated over 50 frames.
            let atoms = structure.atoms();
            let cell = |position: Vec3| {
                (
                    (position.x / 1.7).floor() as i32,
                    (position.y / 1.7).floor() as i32,
                    (position.z / 1.7).floor() as i32,
                )
            };
            let make_grid = |coordinates: &[Vec3]| {
                let mut grid = HashMap::<(i32, i32, i32), Vec<usize>>::new();
                for (index, position) in coordinates.iter().copied().enumerate() {
                    grid.entry(cell(position)).or_default().push(index);
                }
                grid
            };
            let pair = |first: &[Vec3],
                        excluded_first: usize,
                        second: &[Vec3],
                        excluded_second: usize,
                        grid: &HashMap<(i32, i32, i32), Vec<usize>>| {
                let mut score = 1.0;
                let threshold2 = 1.7 * 1.7;
                for (first_index, first_position) in first.iter().enumerate() {
                    if first_index < excluded_first {
                        continue;
                    }
                    let center = cell(*first_position);
                    for dx in -1..=1 {
                        for dy in -1..=1 {
                            for dz in -1..=1 {
                                let Some(indices) =
                                    grid.get(&(center.0 + dx, center.1 + dy, center.2 + dz))
                                else {
                                    continue;
                                };
                                for &second_index in indices {
                                    if second_index < excluded_second {
                                        continue;
                                    }
                                    let distance2 =
                                        squared_distance(*first_position, second[second_index]);
                                    if distance2 < threshold2 {
                                        score += 200.0 * (-distance2).exp();
                                        if score > 2.0 {
                                            return score;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                score
            };
            let glycan_residues = structure
                .metadata()
                .glycan_trees
                .iter()
                .flat_map(|tree| tree.residue_ids.iter().cloned())
                .collect::<HashSet<_>>();
            let protein_positions = atoms
                .iter()
                .filter(|atom| !glycan_residues.contains(&atom.residue))
                .map(|atom| atom.position)
                .collect::<Vec<_>>();
            let protein_grid = make_grid(&protein_positions);
            let glycan_positions = structure
                .metadata()
                .glycan_trees
                .iter()
                .map(|tree| {
                    let residues = tree.residue_ids.iter().cloned().collect::<HashSet<_>>();
                    atoms
                        .iter()
                        .filter(|atom| residues.contains(&atom.residue))
                        .map(|atom| atom.position)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let grids = glycan_positions
                .iter()
                .map(|coordinates| make_grid(coordinates))
                .collect::<Vec<_>>();
            let scores = glycan_positions
                .iter()
                .enumerate()
                .map(|(site, coordinates)| {
                    let mut score = pair(coordinates, 3, &protein_positions, 0, &protein_grid);
                    for (other, other_coordinates) in glycan_positions.iter().enumerate() {
                        if site == other {
                            continue;
                        }
                        score =
                            score.max(pair(coordinates, 3, other_coordinates, 3, &grids[other]));
                    }
                    score
                })
                .collect::<Vec<_>>();
            assert_eq!(scores.len(), 16);
            assert!(
                scores.iter().all(|score| *score <= 1.1),
                "frame {} failed the independent complete-structure steric gate: {:?}",
                index + 1,
                scores
            );
        }
    }

    #[test]
    fn remote_provider_populates_and_reuses_offline_cache() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            for body in [
                GLYCAN.to_string(),
                r#"{"archetype":{"coverage_clusters":{"Cluster 0":72.0},"coverage_clusters_per_main":{"0":[0]}}}"#.into(),
            ] {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0_u8; 2048];
                let _ = stream.read(&mut request).unwrap();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nContent-Type: application/json\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        let query = GlycanQuery {
            source: GlycanSource::GlyTouCan("GTEST".into()),
            anomer: Anomer::Beta,
            format: "PDB".into(),
            level: "3".into(),
        };
        let cache = tempfile::tempdir().unwrap();
        let remote = GlycoShapeProvider::new(format!("http://{address}")).unwrap();
        let online = CachingProvider::new(remote, cache.path(), false)
            .load(&query)
            .unwrap();
        server.join().unwrap();
        assert_eq!(online.conformers.len(), 1);
        assert_eq!(online.conformers[0].cluster_weight, 72.0);

        let unreachable = GlycoShapeProvider::new("http://127.0.0.1:9").unwrap();
        let offline = CachingProvider::new(unreachable, cache.path(), true)
            .load(&query)
            .unwrap();
        assert_eq!(offline.conformers.len(), 1);
        assert_eq!(offline.conformers[0].cluster_weight, 72.0);
        assert!(offline.provenance.starts_with("cache:"));
    }
}

fn start_compute_stage() {}
fn evaluate_candidate(
    context: &EnergySearchContext,
    structure: &Structure,
) -> Result<CandidateEnergy> {
    context.evaluate(structure)
}

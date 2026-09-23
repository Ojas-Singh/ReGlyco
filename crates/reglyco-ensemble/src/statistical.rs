//! Model-distributed ensembles. Every attempted transition contributes residence.
use super::sampling_energy::sampling_surrogate;
#[cfg(feature = "webgpu")]
use super::sampling_energy::sampling_surrogate_async;
use super::*;
#[cfg(test)]
use glysys_energy::prior::log_acceptance;

struct CachedPrior {
    weights: Vec<f64>,
    log_weights: Vec<f64>,
    source: Vec<LinkagePrior>,
    axes: Vec<(
        glysys_energy::prior::CircularMixture,
        glysys_energy::prior::CircularMixture,
    )>,
    rotamers: usize,
}
impl CachedPrior {
    fn new(protein: &Structure, site: &SearchSite, rotamers: bool) -> Result<Self> {
        let weights: Vec<_> = site
            .ensemble
            .conformers
            .iter()
            .map(|c| c.cluster_weight)
            .collect();
        let sum: f64 = weights.iter().sum();
        let source: Vec<_> = site
            .ensemble
            .conformers
            .iter()
            .map(|c| resolved_priors(protein, site, &c.priors))
            .collect();
        let axes = source
            .iter()
            .map(|p| {
                let axis = |c: &[VonMisesComponent]| {
                    glysys_energy::prior::CircularMixture::new(
                        c.iter()
                            .map(|c| glysys_energy::prior::CircularComponent {
                                mean: c.mean_degrees.to_radians(),
                                concentration: c.concentration,
                                weight: c.weight,
                            })
                            .collect(),
                    )
                };
                Ok((axis(&p.phi)?, axis(&p.psi)?))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            log_weights: weights.iter().map(|w| (w / sum).ln()).collect(),
            weights,
            source,
            axes,
            rotamers: if rotamers {
                dunbrack::rotamers(protein, &site.site.residue).len()
            } else {
                0
            },
        })
    }
    fn log_probability(&self, gene: &Gene) -> f64 {
        let Some((phi, psi)) = self.axes.get(gene.conformer) else {
            return f64::NEG_INFINITY;
        };
        let Some(log_weight) = self.log_weights.get(gene.conformer).copied() else {
            return f64::NEG_INFINITY;
        };
        log_weight
            + phi.log_probability(gene.phi.to_radians())
            + psi.log_probability(gene.psi.to_radians())
            - ((self.rotamers + 1) as f64).ln()
    }
    fn generate(&self, rng: &mut ChaCha8Rng) -> Gene {
        let conformer = weighted_index(&self.weights, rng);
        let p = &self.source[conformer];
        let phi = sample_vmm(&p.phi, rng);
        let psi = sample_vmm(&p.psi, rng);
        let rotamer = if self.rotamers == 0 {
            None
        } else {
            match rng.random_range(0..=self.rotamers) {
                0 => None,
                n => Some(n - 1),
            }
        };
        Gene {
            conformer,
            phi,
            psi,
            rotamer,
        }
    }
}
fn proposal_log(priors: &[CachedPrior], state: &[Gene]) -> f64 {
    priors
        .iter()
        .zip(state)
        .map(|(p, g)| p.log_probability(g))
        .sum()
}

#[derive(Debug, Clone, Default)]
struct ProposalBlocks {
    singles: Vec<Vec<usize>>,
    pairs: Vec<Vec<usize>>,
    groups: Vec<Vec<usize>>,
    whole: Vec<usize>,
}

#[derive(Debug, Clone, Copy)]
enum ProposalKind {
    Single,
    Pair,
    Group,
    Whole,
}

fn connected_subset(subset: &[usize], graph: &[Vec<usize>]) -> bool {
    if subset.len() < 2 {
        return true;
    }
    let allowed = subset.iter().copied().collect::<HashSet<_>>();
    let mut visited = HashSet::from([subset[0]]);
    let mut frontier = vec![subset[0]];
    while let Some(site) = frontier.pop() {
        for neighbour in &graph[site] {
            if allowed.contains(neighbour) && visited.insert(*neighbour) {
                frontier.push(*neighbour);
            }
        }
    }
    visited.len() == subset.len()
}

fn enumerate_connected_groups(
    graph: &[Vec<usize>],
    size: usize,
    start: usize,
    current: &mut Vec<usize>,
    groups: &mut Vec<Vec<usize>>,
) {
    if groups.len() >= 20_000 {
        return;
    }
    if current.len() == size {
        if connected_subset(current, graph) {
            groups.push(current.clone());
        }
        return;
    }
    for site in start..graph.len() {
        current.push(site);
        enumerate_connected_groups(graph, size, site + 1, current, groups);
        current.pop();
        if groups.len() >= 20_000 {
            return;
        }
    }
}

fn build_proposal_blocks(protein: &Structure, sites: &[SearchSite]) -> ProposalBlocks {
    let graph = super::conservative_site_graph(protein, sites);
    let singles = (0..sites.len()).map(|site| vec![site]).collect::<Vec<_>>();
    let pairs = (0..sites.len())
        .flat_map(|first| {
            graph[first]
                .iter()
                .copied()
                .filter(move |second| *second > first)
                .map(move |second| vec![first, second])
        })
        .collect::<Vec<_>>();
    let mut groups = Vec::new();
    // Keep the whole-state category distinct from a connected group when a
    // chain has only three or four sites; otherwise the same block would have
    // two selection paths and its reverse proposal probability would be
    // ambiguous.
    let largest_group = 4.min(sites.len().saturating_sub(1));
    for size in 3..=largest_group {
        enumerate_connected_groups(&graph, size, 0, &mut Vec::new(), &mut groups);
    }
    let whole = (sites.len() > 1)
        .then(|| (0..sites.len()).collect::<Vec<_>>())
        .unwrap_or_default();
    ProposalBlocks {
        singles,
        pairs,
        groups,
        whole,
    }
}

fn choose_proposal_block(
    blocks: &ProposalBlocks,
    rng: &mut ChaCha8Rng,
) -> Option<(Vec<usize>, ProposalKind)> {
    let mut categories = Vec::new();
    if !blocks.singles.is_empty() {
        categories.push((0_u8, 50_u32));
    }
    if !blocks.pairs.is_empty() {
        categories.push((1_u8, 25_u32));
    }
    if !blocks.groups.is_empty() {
        categories.push((2_u8, 20_u32));
    }
    if !blocks.whole.is_empty() {
        categories.push((3_u8, 5_u32));
    }
    let total = categories.iter().map(|(_, weight)| *weight).sum::<u32>();
    if total == 0 {
        return None;
    }
    let mut draw = rng.random_range(0..total);
    let category = categories.iter().find_map(|(category, weight)| {
        if draw < *weight {
            Some(*category)
        } else {
            draw -= *weight;
            None
        }
    })?;
    let (choices, kind): (&[Vec<usize>], ProposalKind) = match category {
        0 => (&blocks.singles, ProposalKind::Single),
        1 => (&blocks.pairs, ProposalKind::Pair),
        2 => (&blocks.groups, ProposalKind::Group),
        _ => (std::slice::from_ref(&blocks.whole), ProposalKind::Whole),
    };
    let choice = choices.get(rng.random_range(0..choices.len()))?.clone();
    Some((choice, kind))
}

fn propose_block(
    priors: &[CachedPrior],
    state: &[Gene],
    block: &[usize],
    rng: &mut ChaCha8Rng,
) -> (Vec<Gene>, f64) {
    let mut proposal = state.to_vec();
    // Half of each block's proposals are independent prior resamples.  The
    // other half are symmetric wrapped angular perturbations, which improves
    // local mixing without paying an asymmetric proposal correction.
    let independent = rng.random_bool(0.5);
    if independent {
        for &site in block {
            proposal[site] = priors[site].generate(rng);
        }
        let old_q = block
            .iter()
            .map(|&site| priors[site].log_probability(&state[site]))
            .sum::<f64>();
        let new_q = block
            .iter()
            .map(|&site| priors[site].log_probability(&proposal[site]))
            .sum::<f64>();
        (proposal, old_q - new_q)
    } else {
        for &site in block {
            proposal[site].phi = wrap_degrees(proposal[site].phi + normal_delta_degrees(rng, 8.0));
            proposal[site].psi = wrap_degrees(proposal[site].psi + normal_delta_degrees(rng, 8.0));
        }
        (proposal, 0.0)
    }
}

/// Evaluate a steric proposal by rebuilding only the sites in its proposal
/// block.  Unchanged site poses and their pair compatibility are already part
/// of the committed chain state, so the complete reference gate reduces to
/// the changed-site protein screen plus changed-to-all-site compatibility.
/// This is mathematically the same predicate as `PreparedAttachmentContext::evaluate`
/// for the no-rotamer path, but avoids reconstructing every glycan on every
/// attempted MH transition.
fn incremental_steric_poses(
    prepared: &PreparedAttachmentContext,
    current: &[PreparedSitePose],
    proposal: &[Gene],
    block: &[usize],
    clash_distance: f64,
) -> Result<Option<Vec<PreparedSitePose>>> {
    if current.len() != prepared.site_count() || proposal.len() != prepared.site_count() {
        return Err(EnsembleError::Metadata(
            "incremental steric state has an inconsistent site count".into(),
        ));
    }
    let mut poses = current.to_vec();
    for &site in block {
        let Some(gene) = proposal.get(site) else {
            return Err(EnsembleError::ReGlyco(ReGlycoError::InvalidGeometry));
        };
        let Some(pose) = prepared.prepare_site_pose(site, gene, clash_distance)? else {
            return Ok(None);
        };
        let Some(slot) = poses.get_mut(site) else {
            return Err(EnsembleError::ReGlyco(ReGlycoError::InvalidGeometry));
        };
        *slot = pose;
    }
    for &site in block {
        for other in 0..poses.len() {
            if site == other {
                continue;
            }
            if !prepared.site_poses_compatible(&poses[site], &poses[other], clash_distance) {
                return Ok(None);
            }
        }
    }
    Ok(Some(poses))
}

fn one_sampling_score(values: Vec<f64>, stage: &str) -> Result<f64> {
    let Some(value) = values.into_iter().next() else {
        return Err(EnsembleError::Metadata(format!(
            "{stage} returned no energy for one candidate"
        )));
    };
    if !value.is_finite() {
        return Err(EnsembleError::Metadata(format!(
            "{stage} returned a nonfinite energy"
        )));
    }
    Ok(value)
}

/// Statistical steric sampling is conditioned on the declared 95% VMM
/// regions as well as the complete steric gate.  MH state genes retain the
/// angle and conformer (component identities are proposal metadata), so a
/// state is admissible when each angle belongs to at least one component of
/// its selected conformer's normalized prior.
fn state_vmm_allowed(vmm_priors: &[Vec<LinkagePrior>], state: &[Gene]) -> bool {
    state.iter().enumerate().all(|(index, gene)| {
        let Some(priors) = vmm_priors
            .get(index)
            .and_then(|conformers| conformers.get(gene.conformer))
        else {
            return false;
        };
        priors
            .phi
            .iter()
            .any(|component| vmm_component_within_95(gene.phi, component))
            && priors
                .psi
                .iter()
                .any(|component| vmm_component_within_95(gene.psi, component))
    })
}

fn target(mode: SearchScoringMode, prior: f64, energy: f64, beta: f64) -> f64 {
    match mode {
        SearchScoringMode::StericPrior => prior,
        SearchScoringMode::ProteinGlycanInteraction => prior - beta * energy,
        SearchScoringMode::FullEnergy => -beta * energy,
    }
}
#[maybe_async_cfg::maybe(
    sync(keep_self),
    async(feature = "webgpu"),
    idents(
        evaluate_population(sync, async = "evaluate_population_async"),
        sampling_surrogate(sync, async = "sampling_surrogate_async"),
        genetic_optimize_with_progress_cancelled(sync, async = "gpu_optimize")
    )
)]
pub(super) async fn sample_statistical<C>(
    protein: &Structure,
    sites: &[SearchSite],
    frames: usize,
    config: &SearchConfig,
    builder: &glysys::SystemBuilder,
    mut cancelled: C,
    mut progress: impl FnMut(usize, usize, usize, usize),
) -> Result<(Vec<SampledFrame>, EnsembleSamplingDiagnostics)>
where
    C: FnMut() -> bool,
{
    if frames == 0 || !config.temperature_k.is_finite() || config.temperature_k <= 0. {
        return Err(EnsembleError::Metadata(
            "invalid sampling size or temperature".into(),
        ));
    }
    if sites.is_empty() {
        return Err(EnsembleError::EmptyEnsemble);
    }
    for site in sites {
        let total: f64 = site
            .ensemble
            .conformers
            .iter()
            .map(|c| c.cluster_weight)
            .sum();
        if !total.is_finite()
            || total <= 0.
            || site
                .ensemble
                .conformers
                .iter()
                .any(|c| !c.cluster_weight.is_finite() || c.cluster_weight < 0.)
        {
            return Err(EnsembleError::Metadata("invalid conformer weights".into()));
        }
        for c in &site.ensemble.conformers {
            let p = resolved_priors(protein, site, &c.priors);
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
    let priors = sites
        .iter()
        .map(|site| CachedPrior::new(protein, site, config.scan_rotamers))
        .collect::<Result<Vec<_>>>()?;
    let chains = config.mh_chains.max(1).min(frames);
    let prepared = PreparedAttachmentContext::new(protein, sites, config.scan_rotamers)?;
    let vmm_priors = super::compile_vmm_prior_cache(protein, sites);
    // Build one shared set of compatible starts before creating chains.  The
    // old initializer redrew every full state independently, so a crowded
    // multi-site system paid the same failed geometry work once per chain.
    // Starts are reused only as initial states; every chain keeps an
    // independent RNG stream and performs its own burn-in.
    let shared_starts = if sites.len() > 1 {
        // Use the same deterministic coverage/compatibility coordinator as
        // Build.  The previous weighted random pools could contain no jointly
        // compatible state even when the witness search had one, causing each
        // chain to repeat the failed whole-state initialization independently.
        let initialization_budget = config
            .population_size
            .max(2)
            .saturating_mul(config.generations.saturating_add(1))
            / 2;
        cookbook_compatible_pool_coverage(
            protein,
            sites,
            config,
            &prepared,
            chains.max(1),
            initialization_budget,
        )?
        .states
        .into_iter()
        .filter(|(genes, _, _)| {
            config.scoring_mode != SearchScoringMode::StericPrior
                || state_vmm_allowed(&vmm_priors, genes)
        })
        .map(|(genes, _, _)| genes)
        .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let context = if config.scoring_mode == SearchScoringMode::StericPrior {
        None
    } else {
        let reference = vec![
            Gene {
                conformer: 0,
                phi: 0.,
                psi: 0.,
                rotamer: None
            };
            sites.len()
        ];
        let system =
            builder.prepare_structure(&build_state(protein, sites, &reference, builder)?)?;
        Some(EnergySearchContext {
            system,
            evaluator: OnceLock::new(),
            mode: config.scoring_mode,
            use_obc2: config.use_obc2,
            minimize: false,
            min_iterations: 0,
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
            topology_seconds: 0.,
        })
    };
    let mut scorer = context
        .as_ref()
        .map(super::sampling_energy::SamplingEnergy::new)
        .transpose()?;
    #[cfg(feature = "webgpu")]
    let mut fast_gpu_sampling = config.sampling_target == Some(SamplingTarget::WebgpuF32V1)
        && config.scoring_mode != SearchScoringMode::StericPrior;
    #[cfg(not(feature = "webgpu"))]
    let mut fast_gpu_sampling = false;
    let mut gpu_fallback_reason: Option<String> = None;
    let problem = SearchProblem {
        protein,
        sites,
        builder,
        clash_distance: config.clash_distance,
        scan_rotamers: config.scan_rotamers,
        scoring_mode: config.scoring_mode,
        selection_policy: config.selection_policy,
        use_obc2: config.use_obc2,
        pre_minimization: false,
        pre_minimization_iterations: 0,
        energy_context: context.as_ref(),
        prepared: &prepared,
        prior_cache: super::compile_prior_cache(protein, sites)?,
        vmm_priors: vmm_priors.clone(),
        initial_states: std::sync::Arc::new(Vec::new()),
        initial_state_cursor: std::sync::atomic::AtomicUsize::new(0),
    };
    let mut result = Vec::with_capacity(frames);
    let mut proposals = 0;
    let mut accepts = 0;
    let mut gpu_proposals = 0;
    let mut gpu_accepts = 0;
    let mut cpu_proposals = 0;
    let mut cpu_accepts = 0;
    let mut single_site_proposals = 0;
    let mut pair_proposals = 0;
    let mut group_proposals = 0;
    let mut whole_state_proposals = 0;
    let mut single_site_accepts = 0;
    let mut pair_accepts = 0;
    let mut group_accepts = 0;
    let mut whole_state_accepts = 0;
    let burn = config
        .burn_in_steps
        .unwrap_or(config.mh_burn_in_sweeps.saturating_mul(sites.len()));
    let thinning = config
        .thinning_steps
        .unwrap_or(config.mh_thinning_accepted)
        .max(1);
    let beta = 1. / (0.001_987_204_1 * config.temperature_k);
    let mut restarts = 0;
    let mut states = Vec::new();
    let mut geometry = GeometrySession::default();
    geometry.scores_only = true;
    let proposal_blocks = build_proposal_blocks(protein, sites);
    // Steric-only, no-rotamer sampling can keep the validated per-site poses
    // resident for each chain.  A proposal changes only the selected block,
    // so rebuilding the complete glycoprotein for every rejected transition
    // is unnecessary.  Energy objectives and rotamer moves retain the full
    // evaluator path because their dependencies are global/dynamic.
    // The incremental CPU path is useful when the caller explicitly selected
    // CPU, but it would otherwise make an ordinary steric Ensemble silently
    // bypass the resident GPU geometry evaluator.  GPU requests use the same
    // complete batched steric kernel as Build; this preserves the exact gate
    // while allowing normal easy cases to stay on the device.
    #[cfg(feature = "webgpu")]
    let gpu_steric_requested = super::gpu::requested_backend() != "cpu";
    #[cfg(not(feature = "webgpu"))]
    let gpu_steric_requested = false;
    let incremental_steric = config.scoring_mode == SearchScoringMode::StericPrior
        && !config.scan_rotamers
        && !gpu_steric_requested;
    // A one-site ensemble has no compatibility pool to amortize its start-up
    // work.  Generate and score one resident batch once, then reuse distinct
    // valid starts across chains.  The previous implementation repeated a
    // full f64 geometry evaluation for every chain and could walk the entire
    // population×generation budget before the first frame.
    let mut single_site_starts: Vec<(Vec<Gene>, Vec<f64>)> = Vec::new();
    if sites.len() == 1 {
        let total_attempts = config
            .population_size
            .max(2)
            .saturating_mul(config.generations.max(1));
        let batch_size = config.population_size.clamp(2, 256);
        let mut attempts = 0usize;
        let mut init_rng = ChaCha8Rng::seed_from_u64(splitmix64(config.seed ^ 0x53494e4954535441));
        while attempts < total_attempts && single_site_starts.len() < chains {
            if cancelled() {
                return Err(EnsembleError::Cancelled);
            }
            let count = batch_size.min(total_attempts - attempts);
            let mut candidates = Vec::with_capacity(count);
            for _ in 0..count {
                let genes = priors
                    .iter()
                    .map(|p| p.generate(&mut init_rng))
                    .collect::<Vec<_>>();
                let mut candidate = CookbookStericChromosome::new(sites);
                candidate.genes = genes;
                candidate.phi_components = vec![0; sites.len()];
                candidate.psi_components = vec![0; sites.len()];
                candidates.push(candidate);
            }
            evaluate_population(&problem, &mut candidates, &mut geometry).await?;
            for candidate in candidates {
                if candidate.steric_scores.iter().all(|v| *v <= 1.1)
                    && (config.scoring_mode != SearchScoringMode::StericPrior
                        || state_vmm_allowed(&vmm_priors, &candidate.genes))
                {
                    single_site_starts.push((candidate.genes, candidate.steric_scores));
                    if single_site_starts.len() >= chains {
                        break;
                    }
                }
            }
            attempts = attempts.saturating_add(count);
        }
    }
    let mut chain_poses: Vec<Vec<PreparedSitePose>> = Vec::with_capacity(chains);
    let mut chain_coordinates = Vec::new();
    for chain in 0..chains {
        let count = frames / chains + usize::from(chain < frames % chains);
        let mut rng =
            ChaCha8Rng::seed_from_u64(splitmix64(config.seed ^ 0x535441545632 ^ chain as u64));
        let mut initial = if sites.len() == 1 {
            single_site_starts
                .get(chain % single_site_starts.len().max(1))
                .cloned()
        } else {
            shared_starts
                .get(chain % shared_starts.len().max(1))
                .cloned()
                .and_then(|state| {
                    prepared
                        .evaluate(&state, config.clash_distance)
                        .ok()
                        .filter(|evaluation| {
                            evaluation.site_scores.iter().all(|v| *v <= 1.1)
                                && (config.scoring_mode != SearchScoringMode::StericPrior
                                    || state_vmm_allowed(&vmm_priors, &state))
                        })
                        .map(|evaluation| (state, evaluation.site_scores))
                })
        };
        if initial.is_none() {
            for _ in 0..config.population_size.max(2) * config.generations.max(1) {
                if cancelled() {
                    return Err(EnsembleError::Cancelled);
                }
                let s = priors
                    .iter()
                    .map(|p| p.generate(&mut rng))
                    .collect::<Vec<_>>();
                let e = prepared.evaluate(&s, config.clash_distance)?;
                if e.site_scores.iter().all(|v| *v <= 1.1)
                    && (config.scoring_mode != SearchScoringMode::StericPrior
                        || state_vmm_allowed(&vmm_priors, &s))
                {
                    initial = Some((s, e.site_scores));
                    break;
                }
            }
        }
        let (state, scores) = if let Some(s) = initial {
            s
        } else {
            restarts += 1;
            let ga = GeneticAlgorithmConfig {
                population_size: config.population_size.max(2),
                generations: config.generations.max(1),
                seed: rng.random(),
                ..Default::default()
            };
            let outcome =
                genetic_optimize_with_progress_cancelled(&problem, &ga, |_| {}, &mut cancelled)
                    .await?;
            let e = prepared.evaluate(&outcome.best_state, config.clash_distance)?;
            if e.site_scores.iter().any(|v| *v > 1.1)
                || (config.scoring_mode == SearchScoringMode::StericPrior
                    && !state_vmm_allowed(&vmm_priors, &outcome.best_state))
            {
                return Err(EnsembleError::InsufficientFrames {
                    requested: frames,
                    returned: result.len(),
                });
            }
            (outcome.best_state, e.site_scores)
        };
        let committed_poses = if incremental_steric {
            state
                .iter()
                .enumerate()
                .map(|(site, gene)| {
                    prepared
                        .prepare_site_pose(site, gene, config.clash_distance)?
                        .ok_or(ReGlycoError::InvalidGeometry)
                        .map_err(EnsembleError::from)
                })
                .collect::<Result<Vec<_>>>()?
        } else {
            Vec::new()
        };
        let structure = None::<Structure>;
        let coordinates = if let Some(c) = &context {
            c.coordinates_for(&build_state(protein, sites, &state, builder)?)?
        } else {
            Vec::new()
        };
        let energy = if let Some(e) = &mut scorer {
            if fast_gpu_sampling {
                match sampling_surrogate(e, std::slice::from_ref(&coordinates)).await {
                    Ok(Some(values)) => {
                        match one_sampling_score(values, "GPU sampling initialization") {
                            Ok(value) => value,
                            Err(error) => {
                                fast_gpu_sampling = false;
                                gpu_fallback_reason = Some(error.to_string());
                                one_sampling_score(
                                    e.exact(std::slice::from_ref(&coordinates))?,
                                    "CPU sampling initialization",
                                )?
                            }
                        }
                    }
                    Ok(None) => {
                        fast_gpu_sampling = false;
                        gpu_fallback_reason =
                            Some("GPU sampling evaluator unavailable during initialization".into());
                        one_sampling_score(
                            e.exact(std::slice::from_ref(&coordinates))?,
                            "CPU sampling initialization",
                        )?
                    }
                    Err(error) => {
                        fast_gpu_sampling = false;
                        gpu_fallback_reason =
                            Some(format!("GPU sampling initialization failed: {error}"));
                        one_sampling_score(
                            e.exact(std::slice::from_ref(&coordinates))?,
                            "CPU sampling initialization",
                        )?
                    }
                }
            } else {
                one_sampling_score(
                    e.exact(std::slice::from_ref(&coordinates))?,
                    "CPU sampling initialization",
                )?
            }
        } else {
            0.
        };
        chain_coordinates.push(coordinates);
        let log_q = proposal_log(&priors, &state);
        chain_poses.push(committed_poses);
        states.push((state, scores, structure, energy, log_q, rng, count));
    }
    // Any compatibility GA used to seed a chain has finished by this point.
    // Release its global scoring session once before the sampler starts; the
    // sampler owns a separate resident evaluator for the rest of the job.
    // Releasing it inside every proposal would repeatedly tear down unrelated
    // compatibility resources and erase the benefit of persistent sessions.
    #[cfg(feature = "webgpu")]
    if config.sampling_target == Some(SamplingTarget::WebgpuF32V1) {
        super::gpu::release_compatibility_resources();
    }
    let maximum = states.iter().map(|s| s.6).max().unwrap_or(0);
    let mut burn_boundary = burn;
    let mut final_step = burn.saturating_add(maximum.saturating_mul(thinning));
    let mut step = 1usize;
    while step <= final_step && result.len() < frames {
        if cancelled() {
            return Err(EnsembleError::Cancelled);
        }
        // One proposal per independent chain; no dependent transition is batched.
        let mut candidates = Vec::new();
        let mut active = Vec::new();
        let mut proposal_log_ratios = Vec::new();
        let mut proposal_kinds = Vec::new();
        let mut proposal_blocks_for_slots = Vec::new();
        for (chain, (state, _, _, _, _, rng, count)) in states.iter_mut().enumerate() {
            if step > burn_boundary.saturating_add(count.saturating_mul(thinning)) {
                continue;
            }
            let Some((block, proposal_kind)) = choose_proposal_block(&proposal_blocks, rng) else {
                return Err(EnsembleError::Metadata(
                    "cannot construct an ensemble proposal: no attachment sites are available"
                        .into(),
                ));
            };
            let (genes, log_ratio) = propose_block(&priors, state, &block, rng);
            let mut c = CookbookStericChromosome::new(sites);
            c.genes = genes;
            // These indices affect optimization diagnostics, not the MH density.
            c.phi_components = vec![0; sites.len()];
            c.psi_components = vec![0; sites.len()];
            candidates.push(c);
            active.push(chain);
            proposal_log_ratios.push(log_ratio);
            proposal_kinds.push(proposal_kind);
            proposal_blocks_for_slots.push(block);
        }
        let mut incremental_poses = vec![None; candidates.len()];
        if incremental_steric {
            for (slot, candidate) in candidates.iter_mut().enumerate() {
                let chain = active[slot];
                match incremental_steric_poses(
                    &prepared,
                    &chain_poses[chain],
                    &candidate.genes,
                    &proposal_blocks_for_slots[slot],
                    config.clash_distance,
                )? {
                    Some(poses) => {
                        if state_vmm_allowed(&vmm_priors, &candidate.genes) {
                            candidate.steric_scores = vec![1.0; sites.len()];
                            candidate.valid_mask = vec![true; sites.len()];
                            incremental_poses[slot] = Some(poses);
                        } else {
                            candidate.steric_scores = vec![2.0; sites.len()];
                            candidate.valid_mask = vec![true; sites.len()];
                        }
                    }
                    None => {
                        // Keep the candidate in the normal MH stream as a
                        // rejected steric proposal, without constructing a
                        // complete coordinate state.
                        candidate.steric_scores = vec![2.0; sites.len()];
                        candidate.valid_mask = vec![true; sites.len()];
                    }
                }
            }
        } else {
            evaluate_population(&problem, &mut candidates, &mut geometry).await?;
        }
        // Score old/new pairs together: each pair belongs to an independent chain.
        let mut eligible = Vec::new();
        let mut coordinates = Vec::new();
        for (slot, candidate) in candidates.iter().enumerate() {
            if candidate.steric_scores.iter().all(|v| *v <= 1.1)
                && (config.scoring_mode != SearchScoringMode::StericPrior
                    || state_vmm_allowed(&vmm_priors, &candidate.genes))
            {
                eligible.push(slot);
                if let Some(c) = &context {
                    coordinates.push(chain_coordinates[active[slot]].clone());
                    coordinates.push(c.coordinates_for(&build_state(
                        protein,
                        sites,
                        &candidate.genes,
                        builder,
                    )?)?);
                }
            }
        }
        let mut gpu_failed = false;
        let surrogate = if let Some(e) = &mut scorer {
            if coordinates.is_empty() {
                None
            } else if fast_gpu_sampling {
                match sampling_surrogate(e, &coordinates).await {
                    Ok(Some(values))
                        if values.len() == coordinates.len()
                            && values.iter().all(|v| v.is_finite()) =>
                    {
                        Some(values)
                    }
                    Ok(Some(values)) => {
                        fast_gpu_sampling = false;
                        gpu_failed = true;
                        gpu_fallback_reason = Some(format!(
                            "GPU sampling returned {} scores for {} coordinate states",
                            values.len(),
                            coordinates.len()
                        ));
                        None
                    }
                    Ok(None) => {
                        fast_gpu_sampling = false;
                        gpu_failed = true;
                        gpu_fallback_reason = Some("GPU sampling evaluator unavailable".into());
                        None
                    }
                    Err(error) => {
                        fast_gpu_sampling = false;
                        gpu_failed = true;
                        gpu_fallback_reason = Some(format!("GPU sampling failed: {error}"));
                        None
                    }
                }
            } else {
                match sampling_surrogate(e, &coordinates).await? {
                    Some(values)
                        if values.len() == coordinates.len()
                            && values.iter().all(|v| v.is_finite()) =>
                    {
                        Some(values)
                    }
                    Some(values) => {
                        gpu_failed = true;
                        gpu_fallback_reason = Some(format!(
                            "GPU sampling returned {} scores for {} coordinate states",
                            values.len(),
                            coordinates.len()
                        ));
                        None
                    }
                    None => None,
                }
            }
        } else {
            None
        };
        if gpu_failed {
            // A GPU segment may end only at a committed chain state.  Recompute
            // those current states on CPU and give every chain a fresh burn-in
            // window before CPU frames are emitted.
            let exact_current = scorer
                .as_ref()
                .ok_or_else(|| {
                    EnsembleError::Metadata(
                        "sampling scorer disappeared during GPU fallback".into(),
                    )
                })?
                .exact(&chain_coordinates)?;
            if exact_current.len() != states.len() {
                return Err(EnsembleError::Metadata(format!(
                    "CPU sampling fallback returned {} scores for {} chains",
                    exact_current.len(),
                    states.len()
                )));
            }
            for (state, energy) in states.iter_mut().zip(exact_current) {
                state.3 = energy;
            }
            burn_boundary = step.saturating_add(burn);
            final_step = final_step.saturating_add(burn);
        }
        let mut survivors = Vec::new();
        let mut exact_points = Vec::new();
        let mut approximate_ratios = vec![None; candidates.len()];
        for (pair, &slot) in eligible.iter().enumerate() {
            let chain = active[slot];
            let next_q = proposal_log(&priors, &candidates[slot].genes);
            let survives = if context.is_none() {
                // Steric-only sampling has no energy evaluator.  The GPU (or
                // incremental CPU) geometry path has already applied the
                // complete steric/VMM gate, so the proposal now proceeds to
                // the ordinary prior-only MH acceptance calculation below.
                true
            } else if let Some(scores) = &surrogate {
                let Some(new_score) = scores.get(2 * pair + 1).copied() else {
                    return Err(EnsembleError::Metadata(
                        "sampling score batch is missing a proposal score".into(),
                    ));
                };
                let Some(old_score) = scores.get(2 * pair).copied() else {
                    return Err(EnsembleError::Metadata(
                        "sampling score batch is missing a current score".into(),
                    ));
                };
                let ratio = target(config.scoring_mode, next_q, new_score, beta)
                    - target(config.scoring_mode, states[chain].4, old_score, beta)
                    + proposal_log_ratios[slot];
                approximate_ratios[slot] = Some(ratio);
                // Fast GPU sampling uses this score directly in the target;
                // the acceptance draw is made once below. The reference
                // path keeps the first delayed-acceptance screen.
                fast_gpu_sampling
                    || states[chain].5.random::<f64>().max(f64::MIN_POSITIVE).ln() < ratio.min(0.)
            } else {
                true
            };
            if survives {
                survivors.push((slot, pair));
                if context.is_some() {
                    let Some(point) = coordinates.get(2 * pair + 1) else {
                        return Err(EnsembleError::Metadata(
                            "sampling coordinate batch is missing a proposal".into(),
                        ));
                    };
                    exact_points.push(point.clone());
                }
            }
        }
        let exact_values = if fast_gpu_sampling && surrogate.is_some() {
            let scores = surrogate.as_ref().ok_or_else(|| {
                EnsembleError::Metadata("GPU sampling score batch disappeared".into())
            })?;
            survivors
                .iter()
                .map(|(_, pair)| {
                    scores.get(2 * *pair + 1).copied().ok_or_else(|| {
                        EnsembleError::Metadata("GPU sampling score batch is incomplete".into())
                    })
                })
                .collect::<Result<Vec<_>>>()?
        } else if let Some(e) = &scorer {
            e.exact(&exact_points)?
        } else {
            vec![0.; survivors.len()]
        };
        let mut proposed = vec![None; candidates.len()];
        for ((slot, pair), energy) in survivors.into_iter().zip(exact_values) {
            proposed[slot] = Some((energy, pair));
        }
        for (slot, (candidate, chain)) in candidates.into_iter().zip(active).enumerate() {
            let committed_poses = incremental_poses[slot].take();
            proposals += 1;
            if fast_gpu_sampling {
                gpu_proposals += 1;
            } else {
                cpu_proposals += 1;
            }
            match proposal_kinds[slot] {
                ProposalKind::Single => single_site_proposals += 1,
                ProposalKind::Pair => pair_proposals += 1,
                ProposalKind::Group => group_proposals += 1,
                ProposalKind::Whole => whole_state_proposals += 1,
            }
            let (state, scores, structure, energy, log_q, rng, _) = &mut states[chain];
            if let Some((proposed_energy, pair)) = proposed[slot] {
                let proposal = candidate.genes;
                let proposed_q = proposal_log(&priors, &proposal);
                let exact_ratio = target(config.scoring_mode, proposed_q, proposed_energy, beta)
                    - target(config.scoring_mode, *log_q, *energy, beta)
                    + proposal_log_ratios[slot];
                let log_a = if fast_gpu_sampling {
                    approximate_ratios[slot].unwrap_or(exact_ratio).min(0.)
                } else {
                    (exact_ratio - approximate_ratios[slot].unwrap_or(0.)).min(0.)
                };
                if rng.random::<f64>().max(f64::MIN_POSITIVE).ln() < log_a {
                    *state = proposal;
                    *scores = candidate.steric_scores;
                    *structure = None;
                    *energy = proposed_energy;
                    *log_q = proposed_q;
                    if context.is_some() {
                        let Some(point) = coordinates.get(2 * pair + 1) else {
                            return Err(EnsembleError::Metadata(
                                "sampling coordinate batch is missing an accepted proposal".into(),
                            ));
                        };
                        chain_coordinates[chain] = point.clone();
                    } else if incremental_steric {
                        chain_poses[chain] = committed_poses.ok_or_else(|| {
                            EnsembleError::Metadata(
                                "accepted incremental steric proposal has no committed poses"
                                    .into(),
                            )
                        })?;
                    }
                    accepts += 1;
                    if fast_gpu_sampling {
                        gpu_accepts += 1;
                    } else {
                        cpu_accepts += 1;
                    }
                    match proposal_kinds[slot] {
                        ProposalKind::Single => single_site_accepts += 1,
                        ProposalKind::Pair => pair_accepts += 1,
                        ProposalKind::Group => group_accepts += 1,
                        ProposalKind::Whole => whole_state_accepts += 1,
                    }
                }
            }
            if result.len() < frames
                && step > burn_boundary
                && (step - burn_boundary).is_multiple_of(thinning)
            {
                if structure.is_none() {
                    let mut output = build_state(protein, sites, state, builder)?;
                    if let Some(c) = &context {
                        let mut system = c.system.clone();
                        system.set_coordinates(&chain_coordinates[chain])?;
                        output.update_with_parameterized_hydrogens(&system)?;
                    }
                    *structure = Some(output);
                }
                result.push(sampled_frame_from_state(
                    protein,
                    sites,
                    state,
                    structure.as_ref().cloned().ok_or_else(|| {
                        EnsembleError::Metadata("sampled structure was not materialized".into())
                    })?,
                    scores.clone(),
                    proposals,
                    if fast_gpu_sampling {
                        "gpu_f32_mh_v1"
                    } else if gpu_fallback_reason.is_some() {
                        "cpu_reference_mh_v1"
                    } else {
                        "model_da_mh_v3"
                    },
                    None,
                    None,
                    context.as_ref().map(|_| *energy),
                ));
            }
        }
        if step.is_multiple_of(5) || result.len() == frames {
            progress(step.min(final_step), final_step, result.len(), frames);
        }
        step = step.saturating_add(1);
    }
    let diagnostics = EnsembleSamplingDiagnostics {
        requested_frames: frames,
        returned_frames: result.len(),
        attempts: proposals,
        acceptance_rate: accepts as f64 / proposals.max(1) as f64,
        seed: config.seed,
        native_frames: 0,
        fallback_frames: if gpu_fallback_reason.is_some() {
            result
                .iter()
                .filter(|frame| frame.source == "cpu_reference_mh_v1")
                .count()
        } else {
            0
        },
        fallback_used: restarts > 0 || gpu_fallback_reason.is_some(),
        ga_restarts: restarts,
        chains,
        native_proposals: 0,
        native_accepts: 0,
        mh_proposals: proposals,
        mh_accepts: accepts,
        single_site_proposals,
        pair_proposals,
        group_proposals,
        whole_state_proposals,
        single_site_accepts,
        pair_accepts,
        group_accepts,
        whole_state_accepts,
        burn_in_sweeps: config.mh_burn_in_sweeps,
        thinning_accepted: config.mh_thinning_accepted,
        temperature_k: config.temperature_k,
        scan_rotamers: config.scan_rotamers,
        timings: Default::default(),
        sampling_target: Some(config.sampling_target.unwrap_or_default()),
        segments: if let Some(reason) = gpu_fallback_reason.clone() {
            vec![
                SamplingSegmentDiagnostics {
                    id: 0,
                    target: SamplingTarget::WebgpuF32V1,
                    backend: "GPU".into(),
                    frames: result
                        .iter()
                        .filter(|frame| frame.source == "gpu_f32_mh_v1")
                        .count(),
                    attempts: gpu_proposals,
                    accepts: gpu_accepts,
                    burn_in_steps: burn,
                    fallback_reason: Some(reason),
                },
                SamplingSegmentDiagnostics {
                    id: 1,
                    target: SamplingTarget::CpuReferenceV1,
                    backend: "CPU".into(),
                    frames: result
                        .iter()
                        .filter(|frame| frame.source == "cpu_reference_mh_v1")
                        .count(),
                    attempts: cpu_proposals,
                    accepts: cpu_accepts,
                    burn_in_steps: burn,
                    fallback_reason: None,
                },
            ]
        } else {
            vec![SamplingSegmentDiagnostics {
                id: 0,
                target: config.sampling_target.unwrap_or_default(),
                backend: if config.sampling_target == Some(SamplingTarget::WebgpuF32V1) {
                    "GPU".into()
                } else {
                    "CPU".into()
                },
                frames: result.len(),
                attempts: if config.sampling_target == Some(SamplingTarget::WebgpuF32V1) {
                    gpu_proposals
                } else {
                    cpu_proposals
                },
                accepts: if config.sampling_target == Some(SamplingTarget::WebgpuF32V1) {
                    gpu_accepts
                } else {
                    cpu_accepts
                },
                burn_in_steps: burn,
                fallback_reason: None,
            }]
        },
    };
    Ok((result, diagnostics))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn incremental_steric_proposal_matches_complete_reference() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("ensemble.pdb");
        std::fs::write(&path, include_str!("../../../tests/fixtures/glycan.pdb")).unwrap();
        let query = GlycanQuery {
            source: GlycanSource::LocalBundle(path),
            anomer: Anomer::Beta,
            format: "PDB".into(),
            level: "2".into(),
        };
        let ensemble = LocalBundleProvider.load(&query).unwrap();
        let protein = read_pdb_str(
            include_str!("../../../tests/fixtures/protein.pdb"),
            &dry_options(),
        )
        .unwrap();
        let sites = vec![SearchSite {
            site: reglyco_core::GlycosylationSite::new("A", 1),
            ensemble,
        }];
        let prepared = PreparedAttachmentContext::new(&protein, &sites, false).unwrap();
        let committed = Gene {
            conformer: 0,
            phi: -91.0,
            psi: 178.5,
            rotamer: None,
        };
        let current = vec![
            prepared
                .prepare_site_pose(0, &committed, 1.7)
                .unwrap()
                .expect("committed fixture pose should pass the protein gate"),
        ];
        let proposal = vec![Gene {
            conformer: 0,
            phi: -82.0,
            psi: 177.0,
            rotamer: None,
        }];
        let incremental = incremental_steric_poses(&prepared, &current, &proposal, &[0], 1.7)
            .unwrap()
            .is_some();
        let complete = prepared
            .evaluate(&proposal, 1.7)
            .unwrap()
            .site_scores
            .iter()
            .all(|score| *score <= 1.1);
        assert_eq!(incremental, complete);
    }

    #[test]
    fn delayed_acceptance_satisfies_exact_detailed_balance() {
        let p: [f64; 3] = [0.2, 0.5, 0.3];
        let approximate: [f64; 3] = [0.6, 0.1, 0.3];
        let q: [f64; 3] = [0.8, 0.15, 0.05];
        let transition = |i: usize, j: usize| {
            let first = approximate[j].ln() - approximate[i].ln() + q[i].ln() - q[j].ln();
            let exact = p[j].ln() - p[i].ln() + q[i].ln() - q[j].ln();
            q[j] * first.min(0.).exp() * (exact - first).min(0.).exp()
        };
        for i in 0..3 {
            for j in 0..3 {
                assert!((p[i] * transition(i, j) - p[j] * transition(j, i)).abs() < 1e-14);
            }
        }
    }
    #[test]
    fn independent_proposals_and_residence_recover_target() {
        for seed in 0..5 {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let q: [f64; 2] = [0.9, 0.1];
            let p: [f64; 2] = [0.25, 0.75];
            let mut state = 0;
            let mut counts = [0usize; 2];
            for i in 0..100000 {
                let next = usize::from(rng.random::<f64>() > q[0]);
                let a = log_acceptance(p[state].ln(), p[next].ln(), q[next].ln(), q[state].ln());
                if rng.random::<f64>().ln() < a {
                    state = next;
                }
                if i >= 1000 {
                    counts[state] += 1;
                }
            }
            assert!((counts[1] as f64 / 99000. - p[1]).abs() < 0.025);
        }
    }

    #[test]
    fn coordinated_proposal_mixture_reaches_each_available_block_class() {
        let blocks = ProposalBlocks {
            singles: vec![vec![0], vec![1], vec![2], vec![3]],
            pairs: vec![vec![0, 1], vec![1, 2]],
            groups: vec![vec![0, 1, 2], vec![1, 2, 3]],
            whole: vec![0, 1, 2, 3],
        };
        let mut rng = ChaCha8Rng::seed_from_u64(19);
        let mut counts = [0usize; 4];
        for _ in 0..2_000 {
            let (_, kind) = choose_proposal_block(&blocks, &mut rng).expect("test blocks");
            counts[match kind {
                ProposalKind::Single => 0,
                ProposalKind::Pair => 1,
                ProposalKind::Group => 2,
                ProposalKind::Whole => 3,
            }] += 1;
        }
        assert!(counts.iter().all(|count| *count > 0));
        assert!(counts[0] > counts[1]);
        assert!(counts[1] > counts[3]);
    }
}

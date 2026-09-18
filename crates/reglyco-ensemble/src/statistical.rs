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
        let (phi, psi) = &self.axes[gene.conformer];
        self.log_weights[gene.conformer]
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
) -> Result<(Vec<SampledFrame>, EnsembleSamplingDiagnostics)>
where
    C: FnMut() -> bool,
{
    if frames == 0 || !config.temperature_k.is_finite() || config.temperature_k <= 0. {
        return Err(EnsembleError::Metadata(
            "invalid sampling size or temperature".into(),
        ));
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
    let prepared = PreparedAttachmentContext::new(protein, sites)?;
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
        use_obc2: config.use_obc2,
        pre_minimization: false,
        pre_minimization_iterations: 0,
        energy_context: context.as_ref(),
        prepared: &prepared,
        prior_cache: super::compile_prior_cache(protein, sites)?,
    };
    let mut result = Vec::with_capacity(frames);
    let mut proposals = 0;
    let mut accepts = 0;
    let mut gpu_proposals = 0;
    let mut gpu_accepts = 0;
    let mut cpu_proposals = 0;
    let mut cpu_accepts = 0;
    let chains = config.mh_chains.max(1).min(frames);
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
    let mut chain_coordinates = Vec::new();
    for chain in 0..chains {
        let count = frames / chains + usize::from(chain < frames % chains);
        let mut rng =
            ChaCha8Rng::seed_from_u64(splitmix64(config.seed ^ 0x535441545632 ^ chain as u64));
        let mut initial = None;
        for _ in 0..config.population_size.max(2) * config.generations.max(1) {
            if cancelled() {
                return Err(EnsembleError::Cancelled);
            }
            let s = priors
                .iter()
                .map(|p| p.generate(&mut rng))
                .collect::<Vec<_>>();
            let e = prepared.evaluate(&s, config.clash_distance)?;
            if e.site_scores.iter().all(|v| *v <= 1.1) {
                initial = Some((s, e.site_scores));
                break;
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
            if e.site_scores.iter().any(|v| *v > 1.1) {
                return Err(EnsembleError::InsufficientFrames {
                    requested: frames,
                    returned: result.len(),
                });
            }
            (outcome.best_state, e.site_scores)
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
                    Ok(Some(values)) => values[0],
                    Ok(None) => {
                        fast_gpu_sampling = false;
                        gpu_fallback_reason =
                            Some("GPU sampling evaluator unavailable during initialization".into());
                        e.exact(std::slice::from_ref(&coordinates))?[0]
                    }
                    Err(error) => {
                        fast_gpu_sampling = false;
                        gpu_fallback_reason =
                            Some(format!("GPU sampling initialization failed: {error}"));
                        e.exact(std::slice::from_ref(&coordinates))?[0]
                    }
                }
            } else {
                e.exact(std::slice::from_ref(&coordinates))?[0]
            }
        } else {
            0.
        };
        chain_coordinates.push(coordinates);
        let log_q = proposal_log(&priors, &state);
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
        for (chain, (state, _, _, _, _, rng, count)) in states.iter_mut().enumerate() {
            if step > burn_boundary.saturating_add(count.saturating_mul(thinning)) {
                continue;
            }
            let site = rng.random_range(0..sites.len());
            let draw = priors.iter().map(|p| p.generate(rng)).collect::<Vec<_>>();
            let mut c = CookbookStericChromosome::new(sites);
            c.genes = state.clone();
            c.genes[site] = draw[site].clone();
            // These indices affect optimization diagnostics, not the MH density.
            c.phi_components = vec![0; sites.len()];
            c.psi_components = vec![0; sites.len()];
            candidates.push(c);
            active.push(chain);
        }
        evaluate_population(&problem, &mut candidates, &mut geometry).await?;
        // Score old/new pairs together: each pair belongs to an independent chain.
        let mut eligible = Vec::new();
        let mut coordinates = Vec::new();
        for (slot, candidate) in candidates.iter().enumerate() {
            if candidate.steric_scores.iter().all(|v| *v <= 1.1) {
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
                    Ok(Some(values)) => Some(values),
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
                sampling_surrogate(e, &coordinates).await?
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
                .expect("energy scorer exists after GPU sampling failure")
                .exact(&chain_coordinates)?;
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
            let survives = if let Some(scores) = &surrogate {
                let ratio = target(config.scoring_mode, next_q, scores[2 * pair + 1], beta)
                    - target(config.scoring_mode, states[chain].4, scores[2 * pair], beta)
                    + states[chain].4
                    - next_q;
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
                    exact_points.push(coordinates[2 * pair + 1].clone());
                }
            }
        }
        let exact_values = if fast_gpu_sampling && surrogate.is_some() {
            let scores = surrogate
                .as_ref()
                .expect("fast GPU sampling requires a score batch");
            survivors
                .iter()
                .map(|(_, pair)| scores[2 * *pair + 1])
                .collect()
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
            proposals += 1;
            if fast_gpu_sampling {
                gpu_proposals += 1;
            } else {
                cpu_proposals += 1;
            }
            let (state, scores, structure, energy, log_q, rng, _) = &mut states[chain];
            if let Some((proposed_energy, pair)) = proposed[slot] {
                let proposal = candidate.genes;
                let proposed_q = proposal_log(&priors, &proposal);
                let exact_ratio = target(config.scoring_mode, proposed_q, proposed_energy, beta)
                    - target(config.scoring_mode, *log_q, *energy, beta)
                    + *log_q
                    - proposed_q;
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
                        chain_coordinates[chain] = coordinates[2 * pair + 1].clone();
                    }
                    accepts += 1;
                    if fast_gpu_sampling {
                        gpu_accepts += 1;
                    } else {
                        cpu_accepts += 1;
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
                    structure.as_ref().unwrap().clone(),
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
        burn_in_sweeps: 0,
        thinning_accepted: 0,
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
}

//! One search owns its geometry evaluator; failures do not disable energy kernels.
use super::*;
#[derive(Default)]
pub(super) struct GeometrySession {
    #[cfg(feature = "webgpu")]
    resident: Option<glysys_runtime::StericSession>,
    #[cfg(feature = "webgpu")]
    context: Option<glysys_gpu::GpuContext>,
    #[cfg(feature = "webgpu")]
    ids: Vec<Vec<Vec<u32>>>,
    #[cfg(feature = "webgpu")]
    offsets: Vec<u32>,
    #[cfg(feature = "webgpu")]
    disabled: bool,
    pub(super) scores_only: bool,
    pub(super) gpu_evaluations: usize,
    pub(super) cpu_evaluations: usize,
    pub(super) gpu_seconds: f64,
    pub(super) cpu_seconds: f64,
    pub(super) transform_seconds: f64,
}
pub(super) fn evaluate_population(
    problem: &SearchProblem<'_>,
    population: &mut [CookbookStericChromosome],
    session: &mut GeometrySession,
) -> Result<()> {
    let start = Instant::now();
    population
        .par_iter_mut()
        .map(|c| {
            if session.scores_only {
                let e = problem
                    .prepared
                    .evaluate(&c.genes, problem.clash_distance)?;
                c.steric_scores = e.site_scores.clone();
                Ok(e)
            } else {
                evaluate_cookbook_chromosome(problem, c)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let seconds = start.elapsed().as_secs_f64();
    session.cpu_evaluations = session.cpu_evaluations.saturating_add(population.len());
    session.cpu_seconds += seconds;
    #[cfg(feature = "webgpu")]
    super::gpu::record_cpu_geometry(population.len(), seconds);
    Ok(())
}
#[cfg(feature = "webgpu")]
pub(super) async fn evaluate_population_async(
    problem: &SearchProblem<'_>,
    population: &mut [CookbookStericChromosome],
    session: &mut GeometrySession,
) -> Result<()> {
    if population.is_empty() {
        return Ok(());
    }
    let requested = super::gpu::requested_backend();
    if session.disabled || requested == "cpu" {
        return evaluate_population(problem, population, session);
    }
    if session.resident.is_none() {
        match problem.prepared.gpu_library() {
            Ok((library, ids, offsets)) => {
                if session.context.is_none() {
                    match super::gpu::shared_context().await {
                        Ok(context) => session.context = Some(context),
                        Err(e) => {
                            session.disabled = true;
                            super::gpu::geometry_fallback(e.to_string());
                        }
                    }
                }
                if let Some(context) = session.context.as_ref() {
                    // Size the resident batch from the first actual workload,
                    // bounded by the runtime's adaptive allocation policy.
                    // A fixed 64-candidate allocation wasted device memory on
                    // small searches and throttled larger populations into
                    // unnecessarily small dispatches.
                    let capacity = population.len().clamp(1, 256);
                    match glysys_runtime::StericSession::new(context, &library, capacity as u32)
                        .await
                    {
                        Ok(r) => {
                            session.resident = Some(r);
                            session.ids = ids;
                            session.offsets = offsets;
                        }
                        Err(e) => {
                            session.disabled = true;
                            super::gpu::geometry_fallback(e.to_string());
                        }
                    }
                }
            }
            Err(e) => {
                session.disabled = true;
                super::gpu::geometry_fallback(e.to_string());
            }
        }
    }
    if session.disabled {
        return evaluate_population(problem, population, session);
    }
    let gpu_started = Instant::now();
    let capacity = session.resident.as_ref().unwrap().capacity() as usize;
    let sites = problem.sites.len();
    let mut scores = Vec::new();
    for chunk in population.chunks(capacity) {
        let mut genes = Vec::with_capacity(chunk.len() * sites);
        for chromosome in chunk {
            for (site, g) in chromosome.genes.iter().enumerate() {
                genes.push([
                    session.ids[site][g.rotamer.map_or(0, |r| r + 1)][g.conformer],
                    (g.phi.to_radians() as f32).to_bits(),
                    (g.psi.to_radians() as f32).to_bits(),
                    session.offsets[site],
                ]);
            }
        }
        match session
            .resident
            .as_mut()
            .unwrap()
            .evaluate(&genes, problem.clash_distance as f32)
            .await
        {
            Ok(values) if values.len() == chunk.len() * sites => scores.extend(values),
            Ok(_) => {
                session.disabled = true;
                session.resident = None;
                super::gpu::geometry_fallback("GPU attachment/steric result count mismatch".into());
                return evaluate_population(problem, population, session);
            }
            Err(e) => {
                session.disabled = true;
                session.resident = None;
                super::gpu::geometry_fallback(e.to_string());
                return evaluate_population(problem, population, session);
            }
        }
    }
    let gpu_seconds = gpu_started.elapsed().as_secs_f64();
    session.gpu_evaluations = session.gpu_evaluations.saturating_add(population.len());
    session.gpu_seconds += gpu_seconds;

    // Cookbook freezing needs conformer coordinates. The GPU only supplies
    // the traversal-compatible site scores; coordinate materialization stays
    // on the CPU because those structures are needed by the existing policy.
    let transform_started = Instant::now();
    let conformations = if session.scores_only {
        vec![Vec::new(); population.len()]
    } else {
        population
            .iter()
            .map(|c| problem.prepared.transform_state(&c.genes))
            .collect::<std::result::Result<Vec<_>, _>>()?
    };
    session.transform_seconds += transform_started.elapsed().as_secs_f64();
    for ((c, values), transformed) in population
        .iter_mut()
        .zip(scores.chunks_exact(sites))
        .zip(conformations)
    {
        // Boundary decisions are always made by the same f64 reference.
        if values
            .iter()
            .any(|v| *v < 0. || (*v - 1.1).abs() < 0.003 || (*v - 2.).abs() < 0.003)
        {
            let cpu_started = Instant::now();
            if session.scores_only {
                c.steric_scores = problem
                    .prepared
                    .evaluate(&c.genes, problem.clash_distance)?
                    .site_scores;
            } else {
                evaluate_cookbook_chromosome(problem, c)?;
            }
            let cpu_seconds = cpu_started.elapsed().as_secs_f64();
            session.cpu_evaluations = session.cpu_evaluations.saturating_add(1);
            session.cpu_seconds += cpu_seconds;
            super::gpu::record_cpu_geometry(1, cpu_seconds);
            continue;
        }
        c.steric_scores = values.iter().map(|v| *v as f64).collect();
        c.valid_mask.fill(true);
        c.conformations = transformed.into_iter().map(Some).collect();
        c.dirty.fill(false);
        if !session.scores_only {
            c.fitness = problem.cookbook_fitness(c);
        }
    }
    super::gpu::record_geometry(population.len(), gpu_seconds);
    Ok(())
}

#[cfg(all(test, feature = "webgpu", not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires Vulkan; software is correctness evidence only"]
    fn resident_transforms_and_sterics_match_cpu() {
        pollster::block_on(async {
            let options = dry_options();
            let protein = read_pdb_str(
                include_str!("../../../tests/fixtures/protein.pdb"),
                &options,
            )
            .unwrap();
            let query = GlycanQuery {
                source: GlycanSource::LocalBundle(PathBuf::from("fixture")),
                anomer: Anomer::Beta,
                format: "PDB".into(),
                level: "2".into(),
            };
            let ensemble = ensemble_from_pdb(
                include_str!("../../../tests/fixtures/glycan.pdb"),
                None,
                query,
                "fixture",
            )
            .unwrap();
            let sites = vec![SearchSite {
                site: reglyco_core::GlycosylationSite::new("A", 1),
                ensemble,
            }];
            let prepared = PreparedAttachmentContext::new(&protein, &sites).unwrap();
            let (library, ids, offsets) = prepared.gpu_library().unwrap();
            let context = super::gpu::shared_context().await.unwrap();
            let mut resident = glysys_runtime::StericSession::new(&context, &library, 32)
                .await
                .unwrap();
            let states = (0..32)
                .map(|i| {
                    vec![Gene {
                        conformer: 0,
                        phi: -180. + i as f64 * 11.,
                        psi: 178. - i as f64 * 9.,
                        rotamer: None,
                    }]
                })
                .collect::<Vec<_>>();
            let genes = states
                .iter()
                .map(|s| {
                    [
                        ids[0][0][0],
                        (s[0].phi.to_radians() as f32).to_bits(),
                        (s[0].psi.to_radians() as f32).to_bits(),
                        offsets[0],
                    ]
                })
                .collect::<Vec<_>>();
            let actual = resident.evaluate(&genes, 1.7).await.unwrap();
            for (state, actual) in states.iter().zip(actual) {
                let expected = prepared.evaluate(state, 1.7).unwrap().score;
                assert!(
                    (expected - actual as f64).abs() < 1e-3 + 1e-4 * expected.abs(),
                    "{expected} {actual}"
                );
            }
        });
    }
}

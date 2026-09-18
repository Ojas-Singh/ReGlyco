//! A sampler owns its evaluator and GPU resources for the entire job.
use super::*;
pub(super) struct SamplingEnergy<'a> {
    context: &'a EnergySearchContext,
    reference: EnergyEvaluator<'a>,
    #[cfg(feature = "webgpu")]
    gpu: gpu::Runtime,
}
impl<'a> SamplingEnergy<'a> {
    pub(super) fn new(context: &'a EnergySearchContext) -> Result<Self> {
        Ok(Self {
            context,
            reference: EnergyEvaluator::new(
                &context.system,
                EnergyOptions {
                    cutoff: Some(context.cutoff),
                    obc2: context.use_obc2.then(Obc2Options::default),
                    ..Default::default()
                },
            )?,
            #[cfg(feature = "webgpu")]
            gpu: gpu::Runtime::for_job(),
        })
    }
    pub(super) fn exact(&self, points: &[Vec<Vec3>]) -> Result<Vec<f64>> {
        let start = Instant::now();
        let context = self.context;
        let reference = &self.reference;
        let result = points
            .par_iter()
            .map(|p| {
                let e = if context.mode == SearchScoringMode::ProteinGlycanInteraction {
                    let (protein, glycan, _) = context.masks();
                    reference.interaction_energy(p, protein, glycan)?.total()
                } else {
                    reference.energy(p)?.total()
                };
                if !e.is_finite() {
                    return Err(EnsembleError::Metadata("nonfinite sampling energy".into()));
                }
                Ok(e)
            })
            .collect();
        #[cfg(feature = "webgpu")]
        gpu::record_reference(
            points.len(),
            start.elapsed().as_secs_f64(),
            self.gpu
                .scores_enabled(self.context.mode == SearchScoringMode::ProteinGlycanInteraction),
        );
        #[cfg(not(feature = "webgpu"))]
        let _ = start;
        result
    }
    pub(super) fn surrogate(&mut self, points: &[Vec<Vec3>]) -> Result<Option<Vec<f64>>> {
        let _ = points;
        Ok(None)
    }

    #[cfg(feature = "webgpu")]
    pub(super) async fn surrogate_async(
        &mut self,
        points: &[Vec<Vec3>],
    ) -> Result<Option<Vec<f64>>> {
        if !self
            .gpu
            .scores_enabled(self.context.mode == SearchScoringMode::ProteinGlycanInteraction)
            || points.is_empty()
        {
            return Ok(None);
        }
        let masks = vec![Vec::new(); points.len()];
        let values = self
            .gpu
            .evaluate(
                self.context,
                points,
                &masks,
                false,
                self.context.mode == SearchScoringMode::ProteinGlycanInteraction,
            )
            .await?;
        if !self
            .gpu
            .scores_enabled(self.context.mode == SearchScoringMode::ProteinGlycanInteraction)
        {
            return Ok(None);
        }
        Ok(Some(values.into_iter().map(|e| e.total()).collect()))
    }
}
#[cfg(feature = "webgpu")]
impl Drop for SamplingEnergy<'_> {
    fn drop(&mut self) {
        self.gpu.publish();
    }
}

pub(super) fn sampling_surrogate(
    e: &mut SamplingEnergy<'_>,
    p: &[Vec<Vec3>],
) -> Result<Option<Vec<f64>>> {
    e.surrogate(p)
}
#[cfg(feature = "webgpu")]
pub(super) async fn sampling_surrogate_async(
    e: &mut SamplingEnergy<'_>,
    p: &[Vec<Vec3>],
) -> Result<Option<Vec<f64>>> {
    e.surrogate_async(p).await
}

//! Cost accounting (`methods.Counter`): network evaluations, objective evaluations, objective
//! gradient evaluations (all per candidate) and wall time per stage.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use web_time::Instant;

#[derive(Debug, Default)]
pub struct Counter {
    nfe: AtomicU64,
    objective: AtomicU64,
    objective_grad: AtomicU64,
    stages: Mutex<Vec<(String, f64)>>,
}

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct CounterSnapshot {
    /// network forward evaluations (per conformer)
    pub network_evaluations: u64,
    /// objective evaluations without gradient (per conformer)
    pub objective_evaluations: u64,
    /// objective evaluations with gradient (per conformer)
    pub objective_gradient_evaluations: u64,
    /// (stage, seconds) in execution order
    pub stages: Vec<(String, f64)>,
}

impl Counter {
    pub fn add_nfe(&self, n: usize) {
        self.nfe.fetch_add(n as u64, Ordering::Relaxed);
    }
    pub fn add_objective(&self, n: usize) {
        self.objective.fetch_add(n as u64, Ordering::Relaxed);
    }
    pub fn add_objective_grad(&self, n: usize) {
        self.objective_grad.fetch_add(n as u64, Ordering::Relaxed);
    }
    pub fn add_stage(&self, name: &str, started: Instant) {
        let seconds = started.elapsed().as_secs_f64();
        let mut stages = self.stages.lock().expect("counter lock");
        if let Some(entry) = stages.iter_mut().find(|(n, _)| n == name) {
            entry.1 += seconds;
        } else {
            stages.push((name.to_string(), seconds));
        }
    }
    pub fn snapshot(&self) -> CounterSnapshot {
        CounterSnapshot {
            network_evaluations: self.nfe.load(Ordering::Relaxed),
            objective_evaluations: self.objective.load(Ordering::Relaxed),
            objective_gradient_evaluations: self.objective_grad.load(Ordering::Relaxed),
            stages: self.stages.lock().expect("counter lock").clone(),
        }
    }
}

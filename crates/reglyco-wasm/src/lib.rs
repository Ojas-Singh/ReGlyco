use js_sys::Function;
use reglyco_workflow::{
    InputAssets, ProgressEvent, ReGlycoProfile, ReGlycoRunRequestV1, WorkflowControl,
    analyze_torsions as workflow_analyze_torsions, capabilities as workflow_capabilities,
    execute_with_control,
};
use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind};
use wasm_bindgen::prelude::*;

#[inline]
fn install_panic_hook() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        return (*message).to_string();
    }
    if let Some(message) = payload.downcast_ref::<String>() {
        return message.clone();
    }
    "unknown Rust panic".into()
}

#[cfg(feature = "threaded")]
pub use wasm_bindgen_rayon::init_thread_pool;

// Threaded wasm rebuilds `std` with atomics. getrandom's conditional JS
// dependency is not visible to Cargo during that build-std resolution, so
// the release command selects this browser-only custom backend. Scientific
// workflows remain deterministic from the explicit request seed.
#[cfg(all(target_arch = "wasm32", feature = "threaded"))]
#[unsafe(no_mangle)]
pub unsafe extern "Rust" fn __getrandom_v03_custom(
    destination: *mut u8,
    length: usize,
) -> std::result::Result<(), getrandom::Error> {
    let bytes = unsafe { std::slice::from_raw_parts_mut(destination, length) };
    for chunk in bytes.chunks_mut(4) {
        let random = (js_sys::Math::random() * (u32::MAX as f64 + 1.0)) as u32;
        for (target, value) in chunk.iter_mut().zip(random.to_le_bytes()) {
            *target = value;
        }
    }
    Ok(())
}

struct JsControl {
    callback: Function,
}

impl WorkflowControl for JsControl {
    fn progress(&mut self, event: ProgressEvent) {
        if let Ok(value) = serde_wasm_bindgen::to_value(&event) {
            let _ = self.callback.call1(&JsValue::NULL, &value);
        }
    }

    // The browser adapter normally terminates a dedicated worker for
    // cancellation, but exposing the flag through the shared control trait
    // keeps the WASM contract correct for embedders that can deliver a
    // cancellation signal without tearing down the worker.
    fn cancelled(&self) -> bool {
        false
    }
}

#[wasm_bindgen]
pub fn capabilities() -> Result<JsValue, JsValue> {
    install_panic_hook();
    let profile = if cfg!(feature = "full") {
        ReGlycoProfile::Full
    } else {
        ReGlycoProfile::Public
    };
    serde_wasm_bindgen::to_value(&workflow_capabilities(profile, cfg!(feature = "threaded")))
        .map_err(|error| JsValue::from_str(&error.to_string()))
}

#[wasm_bindgen]
pub fn execute(request: JsValue, assets: JsValue, progress: Function) -> Result<JsValue, JsValue> {
    install_panic_hook();
    let request: ReGlycoRunRequestV1 = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid ReGlyco request: {error}")))?;
    let assets: InputAssets = serde_wasm_bindgen::from_value(assets)
        .map_err(|error| JsValue::from_str(&format!("invalid ReGlyco assets: {error}")))?;
    let mut control = JsControl { callback: progress };
    let result = catch_unwind(AssertUnwindSafe(|| {
        execute_with_control(&request, &assets, &mut control)
    }))
    .map_err(|panic| {
        JsValue::from_str(&format!(
            "ReGlyco execution panic: {}",
            panic_message(panic)
        ))
    })?
    .map_err(|error| JsValue::from_str(&error.to_string()))?;
    serde_wasm_bindgen::to_value(&result).map_err(|error| JsValue::from_str(&error.to_string()))
}

#[wasm_bindgen]
pub fn analyze_torsions(
    request: JsValue,
    assets: JsValue,
    output_pdb: String,
    seed_analysis: JsValue,
) -> Result<JsValue, JsValue> {
    install_panic_hook();
    let request: ReGlycoRunRequestV1 = serde_wasm_bindgen::from_value(request)
        .map_err(|error| JsValue::from_str(&format!("invalid ReGlyco request: {error}")))?;
    let assets: InputAssets = serde_wasm_bindgen::from_value(assets)
        .map_err(|error| JsValue::from_str(&format!("invalid ReGlyco assets: {error}")))?;
    let seed_analysis: serde_json::Value = serde_wasm_bindgen::from_value(seed_analysis)
        .map_err(|error| JsValue::from_str(&format!("invalid torsion analysis seed: {error}")))?;
    let result = workflow_analyze_torsions(&request, &assets, &output_pdb, seed_analysis)
        .map_err(|error| JsValue::from_str(&error.to_string()))?;
    // `result` is an open-ended JSON object assembled by the workflow layer.
    // Passing `serde_json::Value` through serde-wasm-bindgen loses its map
    // entries in some wasm-bindgen runtimes and arrives in the worker as `{}`.
    // Return the canonical JSON payload instead; the analysis worker already
    // accepts JSON strings and parses them before merging the result.
    serde_json::to_string(&result)
        .map(|json| JsValue::from_str(&json))
        .map_err(|error| JsValue::from_str(&error.to_string()))
}

/// Development artifact only. CPU bundles retain their synchronous export.
#[cfg(feature = "webgpu")]
#[wasm_bindgen]
pub async fn execute_async(
    request: JsValue,
    assets: JsValue,
    progress: Function,
) -> Result<JsValue, JsValue> {
    install_panic_hook();
    // Keep the entire adapter body inside one unwind boundary.  The previous
    // boundary covered only the workflow future, leaving setup, diagnostics,
    // and report serialization able to escape as an opaque wasm `unreachable`
    // trap.  A trapped GPU wasm instance cannot be called again, so the worker
    // needs the original panic classified as a recoverable GPU failure.
    let outcome = futures_util::FutureExt::catch_unwind(AssertUnwindSafe(async move {
        let request: ReGlycoRunRequestV1 = serde_wasm_bindgen::from_value(request)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let assets: InputAssets = serde_wasm_bindgen::from_value(assets)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let eligible = matches!(
            request.workflow,
            reglyco_workflow::WorkflowId::NScan
                | reglyco_workflow::WorkflowId::Uniprot
                | reglyco_workflow::WorkflowId::SiteBuild
                | reglyco_workflow::WorkflowId::Ensemble
        );
        reglyco_ensemble::gpu::configure(if eligible {
            &request.options.compute_backend
        } else {
            "cpu"
        });
        let callback = progress.clone();
        reglyco_ensemble::gpu::set_progress(move |backend| {
            let event = ProgressEvent {
                stage: format!("compute_{backend}"),
                message: if backend == "webgpu" {
                    "Compute: GPU".into()
                } else {
                    "Compute: CPU fallback".into()
                },
                current: None,
                total: None,
                fraction: None,
            };
            if let Ok(value) = serde_wasm_bindgen::to_value(&event) {
                let _ = callback.call1(&JsValue::NULL, &value);
            }
        });
        let mut control = JsControl { callback: progress };
        let result = reglyco_workflow::execute_with_control_async(&request, &assets, &mut control)
            .await
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        let diagnostics = reglyco_ensemble::gpu::finish();
        let mut bundle = result;
        if let Some(analysis) = bundle.report.analysis.as_object_mut() {
            if let Some(energy) = analysis
                .get_mut("energyAnalysis")
                .and_then(serde_json::Value::as_object_mut)
            {
                energy.insert(
                    "backend".into(),
                    diagnostics
                        .get("actualBackend")
                        .cloned()
                        .unwrap_or_else(|| serde_json::Value::String("CPU".into())),
                );
            }
        }
        // The workflow assembled the CSV before the asynchronous GPU session
        // published its final backend classification. Refresh it so JSON and
        // CSV carry the same actual execution label.
        if let Some(artifact) = bundle
            .artifacts
            .iter_mut()
            .find(|artifact| artifact.name == "energy.csv")
        {
            artifact.data = reglyco_workflow::AssetData::Text(
                reglyco_workflow::energy_analysis_csv(&bundle.report.analysis),
            );
        }
        bundle.report.analysis["compute"] = diagnostics;
        let report = serde_json::to_string_pretty(&bundle.report)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        for artifact in &mut bundle.artifacts {
            if artifact.name == "report.json" {
                artifact.data = reglyco_workflow::AssetData::Text(report.clone());
            }
        }
        serde_json::to_string(&bundle)
            .map(|json| JsValue::from_str(&json))
            .map_err(|e| JsValue::from_str(&e.to_string()))
    }))
    .await;
    match outcome {
        Ok(result) => result,
        Err(panic) => Err(JsValue::from_str(&format!(
            "GPU execution panic: {}",
            panic_message(panic)
        ))),
    }
}

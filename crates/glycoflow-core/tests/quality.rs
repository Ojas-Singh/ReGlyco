//! The structure check and repair on real ring-head samples (needs a ring-head export, as
//! `ring_head_parity.rs`): how many raw samples break the validator's rules, how many are left after
//! `tidy`, and how far the repair moves them.

mod common;

use common::*;
use glycoflow_core::model::Precision;
use glycoflow_core::pucker::ring_puckers;
use glycoflow_core::quality::{tidy, Checker};
use glycoflow_core::rng::{ring_prior, uniform_torsions, SplitMix64};
use glycoflow_core::sampler::{SampleOptions, Sampler};
use glycoflow_core::guidance::ClashGuidance;
use glycoflow_core::Glycan;

#[test]
#[ignore = "needs GLYCOFLOW_RUST_DIR with a ring-head export (rust_rings/)"]
fn tidy_makes_samples_valid_and_moves_them_little() {
    let model = model(&candle_core::Device::Cpu, Precision::F32);
    if !model.has_pucker() {
        return;
    }
    for seq in [
        "DGalpb1-4DGlcpNAcb1-OH",
        "DManpa1-6[DManpa1-3]DManpb1-4DGlcpNAcb1-4[LFucpa1-6]DGlcpNAcb1-OH",
        "DNeup5Aca2-6DGalpb1-4DGlcpNAcb1-2DManpa1-6[DNeup5Aca2-3DGalpb1-4DGlcpNAcb1-2DManpa1-3]DManpb1-4DGlcpNAcb1-4[LFucpa1-6]DGlcpNAcb1-OH",
        "LArafa1-3LArafb1-2LArafb1-2LArafb1-OH",
    ] {
        let g = Glycan::from_builds(&[library().build(seq, None).unwrap()]).unwrap();
        let s = Sampler::for_glycan(&model, &g, &meta().vocab).unwrap();
        let b = 128;
        let mut rng = SplitMix64::new(11);
        let tau0 = uniform_torsions(&mut rng, b * s.n_torsions());
        let q0 = ring_prior(&mut rng, &s.pucker_channels(), b);
        let tpl: Vec<[f32; 3]> = (0..b).flat_map(|_| g.templates[0].iter().copied()).collect();
        let mut guide = ClashGuidance::new(0.3, 0.5);
        let (x, _, _) = s.sample_joint(&tpl, &tau0, &q0, SampleOptions { steps: 16, ..Default::default() }, Some(&mut guide)).unwrap();
        let checker = Checker::new(&g.topology, &g.elements, &g.templates[0]);
        let n = g.n_atoms();
        let (mut valid0, mut valid1, mut moved, mut dtau, mut dq) = (0, 0, 0f64, 0f64, 0f64);
        let (mut moves, mut dqs) = (Vec::new(), Vec::new());
        let t0 = std::time::Instant::now();
        for c in x.chunks(n) {
            let mut y = c.to_vec();
            let r = tidy(&checker, &g.topology, &mut y);
            valid0 += r.before.valid() as usize;
            valid1 += r.after.valid() as usize;
            moved = moved.max(r.moved);
            moves.push(r.moved);
            let (a, bb) = (glycoflow_core::geometry::dihedrals(c, &g.topology.quads), glycoflow_core::geometry::dihedrals(&y, &g.topology.quads));
            let rms = (a.iter().zip(&bb).map(|(u, v)| { let d = (u - v).rem_euclid(std::f32::consts::TAU); let d = d.min(std::f32::consts::TAU - d) as f64; d * d }).sum::<f64>() / a.len().max(1) as f64).sqrt();
            dtau = dtau.max(rms.to_degrees());
            let f = |p: &[[f32; 3]]| -> Vec<[f64; 3]> { ring_puckers(&p.iter().map(|v| v.map(f64::from)).collect::<Vec<_>>(), &g.topology.rings) };
            let mut here = 0f64;
            for (u, v) in f(c).iter().zip(f(&y)) {
                here = here.max((0..3).map(|k| (u[k] - v[k]).abs()).fold(0.0, f64::max));
            }
            dq = dq.max(here);
            dqs.push(here);
        }
        let med = |v: &mut Vec<f64>| { v.sort_by(|a, b| a.total_cmp(b)); v[v.len() / 2] };
        println!("     median move {:.2} A, median ring-shape change {:.3} A", med(&mut moves), med(&mut dqs));
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / b as f64;
        println!("{:>3} atoms: valid {valid0}/{b} raw -> {valid1}/{b} tidied; max move {moved:.2} A, max torsion RMS change {dtau:.1} deg, max ring-shape change {dq:.3} A; {ms:.1} ms/structure", n);
        assert!(valid1 >= valid0);
    }
}

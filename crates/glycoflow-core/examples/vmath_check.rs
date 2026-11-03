//! Exhaustive bit check of `glycoflow_core::vmath` against the scalar calls it replaces, on wasm32
//! simd128 (where the four-lane versions are used). Built as a cdylib (no wasm-bindgen step) and driven by
//! `scripts/perf/vmath_check.mjs`:
//!
//! ```text
//! RUSTFLAGS='-C target-feature=+simd128 --cfg getrandom_backend="wasm_js"' \
//!   cargo build --release -p glycoflow-core --no-default-features --example vmath_check \
//!   --target wasm32-unknown-unknown
//! node scripts/perf/vmath_check.mjs target/wasm32-unknown-unknown/release/examples/vmath_check.wasm
//! ```
//!
//! `check(kind, lo, hi)` runs the slice function of `kind` (0 exp, 1 silu, 2 gelu) over the f32 bit
//! patterns [lo, hi) in slices of 4093 (not a multiple of four, so the scalar tails are exercised)
//! and returns the number of results whose bits differ from the scalar call. `check_mixed(kind,
//! seed, count)` does the same on pseudo-random values of mixed magnitude, so the lanes of one
//! group of four take different branches. Natively the slice functions are the scalar loops and
//! the library is empty.

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod wasm {
    use glycoflow_core::vmath;

    fn pick(kind: u32) -> (fn(&mut [f32]), fn(f32) -> f32) {
        match kind {
            0 => (vmath::exp_inplace, f32::exp),
            1 => (vmath::silu_inplace, vmath::silu),
            _ => (vmath::gelu_inplace, vmath::gelu),
        }
    }

    fn count_bad(buf: &mut [f32], f: fn(&mut [f32]), f1: fn(f32) -> f32) -> u64 {
        let src = buf.to_vec();
        f(buf);
        src.iter()
            .zip(buf.iter())
            .filter(|(a, b)| f1(**a).to_bits() != b.to_bits())
            .count() as u64
    }

    #[no_mangle]
    pub extern "C" fn check(kind: u32, lo: u32, hi: u32) -> u64 {
        let (f, f1) = pick(kind);
        let mut buf = vec![0f32; 4093];
        let (mut bad, mut b) = (0u64, lo as u64);
        let end = if hi == 0 { 1u64 << 32 } else { hi as u64 };
        while b < end {
            let m = (end - b).min(4093) as usize;
            for (i, v) in buf[..m].iter_mut().enumerate() {
                *v = f32::from_bits((b + i as u64) as u32);
            }
            bad += count_bad(&mut buf[..m], f, f1);
            b += m as u64;
        }
        bad
    }

    #[no_mangle]
    pub extern "C" fn check_mixed(kind: u32, seed: u32, count: u32) -> u64 {
        let (f, f1) = pick(kind);
        let mut rng = glycoflow_core::rng::SplitMix64::new(seed as u64);
        let (mut bad, mut left) = (0u64, count as usize);
        while left > 0 {
            let m = left.min(1 + (rng.next_u64() % 3000) as usize);
            let mut buf: Vec<f32> = (0..m)
                .map(|_| {
                    let r = rng.next_u64();
                    if r % 97 == 0 {
                        f32::from_bits((r >> 32) as u32) // any bit pattern now and then
                    } else {
                        let scale = [3.0f32, 12.0, 120.0][(r % 3) as usize];
                        (rng.uniform_f32() * 2.0 - 1.0) * scale * rng.uniform_f32()
                    }
                })
                .collect();
            bad += count_bad(&mut buf, f, f1);
            left -= m;
        }
        bad
    }
}

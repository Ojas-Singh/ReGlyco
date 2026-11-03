//! Slice versions of the network's transcendental activations (`exp`, SiLU, GELU-erf) for the
//! CPU kernels of [`crate::kernels`].
//!
//! On wasm32 with `simd128` (the browser engine) these evaluate four lanes at a time with exactly
//! the scalar algorithms the engine used before -- the FreeBSD/musl `expf` that `f32::exp` lowers to
//! on wasm32 (compiler-builtins' libm) and the `libm::erff` behind `candle_core::cpu::erf::erf_f32`
//! -- operation for operation in f32 (IEEE add, mul, div and bit manipulation only; wasm has no fused
//! multiply-add), so every result is bit-identical to the scalar call. Every branch of the scalar
//! code is evaluated for the lanes that take it and blended in; NaN, infinities and overflowing
//! arguments fall back to the scalar call. On other targets they are the plain scalar loops: native
//! `f32::exp` is the platform libm (glibc), which these algorithms do not reproduce, so native
//! results are untouched.
//!
//! The wasm lane functions were checked against the scalar calls on every one of the 2^32 f32 bit
//! patterns (see `docs/PERF.md`).

/// `x / (1 + exp(-x))`, as the kernels have always computed it.
#[inline]
pub fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

/// candle's CPU `gelu_erf` formula.
#[inline]
pub fn gelu(v: f32) -> f32 {
    (candle_core::cpu::erf::erf_f32(v * std::f32::consts::FRAC_1_SQRT_2) + 1.) * 0.5 * v
}

/// `x = exp(x)` for every element.
pub fn exp_inplace(x: &mut [f32]) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        wasm::map(x, wasm::exp4, f32::exp);
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    for v in x.iter_mut() {
        *v = v.exp();
    }
}

/// `x = silu(x)` for every element.
pub fn silu_inplace(x: &mut [f32]) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        wasm::map(x, wasm::silu4, silu);
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    for v in x.iter_mut() {
        *v = silu(*v);
    }
}

/// `x = gelu(x)` for every element.
pub fn gelu_inplace(x: &mut [f32]) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        wasm::gelu_slice(x);
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    for v in x.iter_mut() {
        *v = gelu(*v);
    }
}

/// Four-lane `exp` and `erf` on wasm32 simd128, bit-identical to the scalar calls. Public so the
/// exhaustive check can call the lane functions directly.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
// the constants are libm's literals, kept as written there (they round to the same f32)
#[allow(clippy::excessive_precision)]
pub mod wasm {
    use core::arch::wasm32::*;

    /// Apply `f4` to every full group of four and `f1` to the tail.
    #[inline(always)]
    pub fn map(x: &mut [f32], f4: impl Fn(v128) -> v128, f1: impl Fn(f32) -> f32) {
        let mut chunks = x.chunks_exact_mut(4);
        for c in &mut chunks {
            // SAFETY: `c` is exactly four f32 (16 bytes); v128 loads and stores need no alignment.
            unsafe {
                let p = c.as_mut_ptr() as *mut v128;
                v128_store(p, f4(v128_load(p)));
            }
        }
        for v in chunks.into_remainder() {
            *v = f1(*v);
        }
    }

    #[inline(always)]
    fn splat(v: f32) -> v128 {
        f32x4_splat(v)
    }

    #[inline(always)]
    fn usplat(v: u32) -> v128 {
        u32x4_splat(v)
    }

    #[inline(always)]
    fn scalar4(x: v128, f: impl Fn(f32) -> f32) -> v128 {
        f32x4(
            f(f32x4_extract_lane::<0>(x)),
            f(f32x4_extract_lane::<1>(x)),
            f(f32x4_extract_lane::<2>(x)),
            f(f32x4_extract_lane::<3>(x)),
        )
    }

    const LN2_HI: f32 = 6.9314575195e-01;
    const LN2_LO: f32 = 1.4286067653e-06;
    const INV_LN2: f32 = 1.4426950216e+00;
    const P1: f32 = 1.6666625440e-1;
    const P2: f32 = -2.7667332906e-3;

    /// libm `expf` (FreeBSD `e_expf.c`) on four lanes: argument reduction, the rational
    /// approximation and `scalbnf`, each case selected per lane. Lanes with NaN or an argument that
    /// overflows (x >= 88.72) take the scalar call.
    #[inline(always)]
    pub fn exp4(x: v128) -> v128 {
        let neg = i32x4_shr(x, 31); // all ones where the sign bit is set
        let hx = v128_and(x, usplat(0x7fff_ffff));
        let nan = u32x4_gt(hx, usplat(0x7f80_0000));
        let over = v128_andnot(u32x4_ge(hx, usplat(0x42b1_7218)), neg);
        if v128_any_true(v128_or(nan, over)) {
            return scalar4(x, f32::exp);
        }
        let zero = v128_and(neg, u32x4_ge(hx, usplat(0x42cf_f1b5))); // x <= -103.97: 0
        v128_andnot(exp4_core::<true>(x, neg, hx), zero)
    }

    /// `exp4` for arguments with |x| < 87.33 (no NaN, overflow, underflow, and 2^k a normal
    /// number), where libm takes none of its special cases.
    #[inline(always)]
    fn exp4_mid(x: v128) -> v128 {
        let neg = i32x4_shr(x, 31);
        let hx = v128_and(x, usplat(0x7fff_ffff));
        exp4_core::<false>(x, neg, hx)
    }

    /// The reduction, approximation and scaling of `expf` (and its |x| <= 2^-14 case); `WIDE`
    /// handles a 2^k that is not a normal number (k = 128 or k < -126, |x| >= 87.33).
    #[inline(always)]
    fn exp4_core<const WIDE: bool>(x: v128, neg: v128, hx: v128) -> v128 {
        let tiny = u32x4_le(hx, usplat(0x3900_0000)); // |x| <= 2^-14: 1 + x

        // argument reduction: k = 0 (|x| <= 0.5 ln2), +-1 (<= 1.5 ln2) or round(x / ln2)
        let far = u32x4_gt(hx, usplat(0x3f85_1592));
        let near = u32x4_gt(hx, usplat(0x3eb1_7218));
        let half = v128_bitselect(splat(-0.5), splat(0.5), neg);
        let k_far = i32x4_trunc_sat_f32x4(f32x4_add(f32x4_mul(splat(INV_LN2), x), half));
        let k_near = i32x4_sub(
            i32x4_splat(1),
            i32x4_add(v128_and(neg, i32x4_splat(1)), v128_and(neg, i32x4_splat(1))),
        );
        let k = v128_and(v128_bitselect(k_far, k_near, far), near);
        let kf = f32x4_convert_i32x4(k);
        // k = 0 gives hi = x - 0 = x, lo = +0 and x - lo = x: the scalar k = 0 branch exactly
        let hi = f32x4_sub(x, f32x4_mul(kf, splat(LN2_HI)));
        let lo = f32x4_mul(kf, splat(LN2_LO));
        let r = f32x4_sub(hi, lo);

        let xx = f32x4_mul(r, r);
        let c = f32x4_sub(
            r,
            f32x4_mul(xx, f32x4_add(splat(P1), f32x4_mul(xx, splat(P2)))),
        );
        let q = f32x4_div(f32x4_mul(r, c), f32x4_sub(splat(2.0), c));
        let y = f32x4_add(splat(1.0), f32x4_add(f32x4_sub(q, lo), hi));

        // scalbnf(y, k): pre-scale when 2^k is not a normal f32 (k = 128, or k < -126)
        let (y, n) = if WIDE {
            let big = i32x4_gt(k, i32x4_splat(127));
            let small = i32x4_lt(k, i32x4_splat(-126));
            let y = v128_bitselect(
                f32x4_mul(y, splat(f32::from_bits(0x7f00_0000))), // 2^127
                // 2^-126 * 2^24, multiplied out at run time as libm's generic scalbn does
                v128_bitselect(
                    f32x4_mul(
                        y,
                        splat(f32::from_bits(0x0080_0000) * f32::from_bits(0x4b80_0000)),
                    ),
                    y,
                    small,
                ),
                big,
            );
            let n = v128_bitselect(
                i32x4_sub(k, i32x4_splat(127)),
                v128_bitselect(i32x4_add(k, i32x4_splat(102)), k, small),
                big,
            );
            (y, n)
        } else {
            (y, k)
        };
        let scale = i32x4_shl(i32x4_add(n, i32x4_splat(127)), 23);
        let e = f32x4_mul(y, scale);
        v128_bitselect(f32x4_add(splat(1.0), x), e, tiny)
    }

    /// `x / (1 + exp(-x))`
    #[inline(always)]
    pub fn silu4(v: v128) -> v128 {
        f32x4_div(v, f32x4_add(splat(1.0), exp4(f32x4_neg(v))))
    }

    /// `(erf(x / sqrt 2) + 1) * 0.5 * x`
    #[inline(always)]
    pub fn gelu4(v: v128) -> v128 {
        let e = erf4(f32x4_mul(v, splat(std::f32::consts::FRAC_1_SQRT_2)));
        gelu_from_erf(e, v)
    }

    const ERX: f32 = 8.4506291151e-01;
    const EFX8: f32 = 1.0270333290e+00;
    const PP: [f32; 5] = [
        1.2837916613e-01,
        -3.2504209876e-01,
        -2.8481749818e-02,
        -5.7702702470e-03,
        -2.3763017452e-05,
    ];
    const QQ: [f32; 5] = [
        3.9791721106e-01,
        6.5022252500e-02,
        5.0813062117e-03,
        1.3249473704e-04,
        -3.9602282413e-06,
    ];
    const PA: [f32; 7] = [
        -2.3621185683e-03,
        4.1485610604e-01,
        -3.7220788002e-01,
        3.1834661961e-01,
        -1.1089469492e-01,
        3.5478305072e-02,
        -2.1663755178e-03,
    ];
    const QA: [f32; 6] = [
        1.0642088205e-01,
        5.4039794207e-01,
        7.1828655899e-02,
        1.2617121637e-01,
        1.3637083583e-02,
        1.1984500103e-02,
    ];
    const RA: [f32; 8] = [
        -9.8649440333e-03,
        -6.9385856390e-01,
        -1.0558626175e+01,
        -6.2375331879e+01,
        -1.6239666748e+02,
        -1.8460508728e+02,
        -8.1287437439e+01,
        -9.8143291473e+00,
    ];
    const SA: [f32; 8] = [
        1.9651271820e+01,
        1.3765776062e+02,
        4.3456588745e+02,
        6.4538726807e+02,
        4.2900814819e+02,
        1.0863500214e+02,
        6.5702495575e+00,
        -6.0424413532e-02,
    ];
    // RB and SB have one coefficient fewer than RA and SA; the extra 0 adds s * 0 = +0 to a nonzero
    // partial sum, which leaves it unchanged, so one Horner form serves both (s = 1/x^2 > 0).
    const RB: [f32; 8] = [
        -9.8649431020e-03,
        -7.9928326607e-01,
        -1.7757955551e+01,
        -1.6063638306e+02,
        -6.3756646729e+02,
        -1.0250950928e+03,
        -4.8351919556e+02,
        0.0,
    ];
    const SB: [f32; 8] = [
        3.0338060379e+01,
        3.2579251099e+02,
        1.5367296143e+03,
        3.1998581543e+03,
        2.5530502930e+03,
        4.7452853394e+02,
        -2.2440952301e+01,
        0.0,
    ];

    /// c0 + s (c1 + s (c2 + ...)), innermost first, as the scalar code nests it
    #[inline(always)]
    fn horner(s: v128, c: &[v128]) -> v128 {
        let mut acc = c[c.len() - 1];
        for &ci in c[..c.len() - 1].iter().rev() {
            acc = f32x4_add(ci, f32x4_mul(s, acc));
        }
        acc
    }

    #[inline(always)]
    fn consts<const N: usize>(c: &[f32; N]) -> [v128; N] {
        c.map(splat)
    }

    /// 1 + s (c0 + s (c1 + ...)), the scalar denominators
    #[inline(always)]
    fn one_plus(s: v128, c: &[v128]) -> v128 {
        f32x4_add(splat(1.0), f32x4_mul(s, horner(s, c)))
    }

    /// erf for |x| < 0.84375 (the scalar code's first two cases)
    #[inline(always)]
    fn erf_small(x: v128) -> v128 {
        let ix = v128_and(x, usplat(0x7fff_ffff));
        let tiny = u32x4_lt(ix, usplat(0x3180_0000)); // |x| < 2^-28
        let z = f32x4_mul(x, x);
        let r = horner(z, &consts(&PP));
        let s = one_plus(z, &consts(&QQ));
        let y = f32x4_div(r, s);
        let a = f32x4_add(x, f32x4_mul(x, y));
        let t = f32x4_mul(
            splat(0.125),
            f32x4_add(f32x4_mul(splat(8.0), x), f32x4_mul(splat(EFX8), x)),
        );
        v128_bitselect(t, a, tiny)
    }

    /// erfc(|x|) for 0.84375 <= |x| < 1.25 (`erfc1`), from ax = |x|
    #[inline(always)]
    fn erfc_near(ax: v128) -> v128 {
        let s = f32x4_sub(ax, splat(1.0));
        let p = horner(s, &consts(&PA));
        let q = one_plus(s, &consts(&QA));
        f32x4_sub(f32x4_sub(splat(1.0), splat(ERX)), f32x4_div(p, q))
    }

    /// erfc(|x|) for 1.25 <= |x| < 6 (`erfc2`), from ax = |x|, the lanes mixing both of its
    /// coefficient sets
    #[inline(always)]
    fn erfc_far(ax: v128) -> v128 {
        let mid = u32x4_lt(ax, usplat(0x4036_db6d)); // |x| < 1/0.35: RA/SA, else RB/SB
        let s = f32x4_div(splat(1.0), f32x4_mul(ax, ax));
        let pick = |a: &[f32; 8], b: &[f32; 8]| -> [v128; 8] {
            core::array::from_fn(|i| v128_bitselect(splat(a[i]), splat(b[i]), mid))
        };
        erfc_far_tail(ax, horner(s, &pick(&RA, &RB)), one_plus(s, &pick(&SA, &SB)))
    }

    /// erfc(|x|) for 1.25 <= |x| < 1/0.35 (every lane)
    #[inline(always)]
    fn erfc_far_a(ax: v128) -> v128 {
        let s = f32x4_div(splat(1.0), f32x4_mul(ax, ax));
        erfc_far_tail(ax, horner(s, &consts(&RA)), one_plus(s, &consts(&SA)))
    }

    /// erfc(|x|) for 1/0.35 <= |x| < 6 (every lane); RB, SB without their zero padding
    #[inline(always)]
    fn erfc_far_b(ax: v128) -> v128 {
        let s = f32x4_div(splat(1.0), f32x4_mul(ax, ax));
        let rb: [f32; 7] = core::array::from_fn(|i| RB[i]);
        let sb: [f32; 7] = core::array::from_fn(|i| SB[i]);
        erfc_far_tail(ax, horner(s, &consts(&rb)), one_plus(s, &consts(&sb)))
    }

    /// exp(-z^2 - 0.5625) exp((z - x)(z + x) + r / S) / x with z = |x| rounded to 12 bits: both
    /// exponents stay within (-37, 1) for 1.25 <= |x| < 6, so `expf` takes no special case
    #[inline(always)]
    fn erfc_far_tail(ax: v128, r: v128, big_s: v128) -> v128 {
        let z = v128_and(ax, usplat(0xffff_e000));
        let e_a = exp4_mid(f32x4_sub(f32x4_mul(f32x4_neg(z), z), splat(0.5625)));
        let e_b = exp4_mid(f32x4_add(
            f32x4_mul(f32x4_sub(z, ax), f32x4_add(z, ax)),
            f32x4_div(r, big_s),
        ));
        f32x4_div(f32x4_mul(e_a, e_b), ax)
    }

    /// erf = +-(1 - erfc(|x|)), the sign of x
    #[inline(always)]
    fn signed_complement(erfc: v128, x: v128) -> v128 {
        let y = f32x4_sub(splat(1.0), erfc);
        v128_bitselect(f32x4_neg(y), y, i32x4_shr(x, 31))
    }

    /// libm `erff` (FreeBSD `s_erff.c`) on four lanes. NaN and infinities take the scalar call.
    #[inline(always)]
    pub fn erf4(x: v128) -> v128 {
        let ix = v128_and(x, usplat(0x7fff_ffff));
        if v128_any_true(u32x4_ge(ix, usplat(0x7f80_0000))) {
            return scalar4(x, candle_core::cpu::erf::erf_f32);
        }
        let small = u32x4_lt(ix, usplat(0x3f58_0000)); // |x| < 0.84375
        let one = u32x4_lt(ix, usplat(0x3fa0_0000)); // |x| < 1.25 (erfc1)
        let six = u32x4_lt(ix, usplat(0x40c0_0000)); // |x| < 6
        let mut out = splat(0.0);
        if v128_any_true(small) {
            out = v128_bitselect(erf_small(x), out, small);
        }
        // 0.84375 <= |x| < 6: y = 1 - erfc(|x|), signed
        let tail = v128_andnot(six, small);
        if v128_any_true(tail) {
            let mut erfc = splat(0.0);
            let near = v128_and(tail, one);
            if v128_any_true(near) {
                erfc = v128_bitselect(erfc_near(ix), erfc, near);
            }
            let far = v128_andnot(tail, one);
            if v128_any_true(far) {
                erfc = v128_bitselect(erfc_far(ix), erfc, far);
            }
            out = v128_bitselect(signed_complement(erfc, x), out, tail);
        }
        // |x| >= 6: +-(1 - 2^-120)
        let huge = v128_not(six);
        if v128_any_true(huge) {
            let y = signed_complement(splat(f32::from_bits(0x0380_0000)), x);
            out = v128_bitselect(y, out, huge);
        }
        out
    }

    #[inline(always)]
    fn gelu_from_erf(e: v128, v: v128) -> v128 {
        f32x4_mul(f32x4_mul(f32x4_add(e, splat(1.0)), splat(0.5)), v)
    }

    /// Byte shuffles that pack the lanes selected by a 4-bit mask to the front of a vector.
    const LEFT_PACK: [[u8; 16]; 16] = {
        let mut t = [[0x80u8; 16]; 16];
        let mut m = 0;
        while m < 16 {
            let (mut lane, mut at) = (0, 0);
            while lane < 4 {
                if m & (1 << lane) != 0 {
                    let mut b = 0;
                    while b < 4 {
                        t[m][at * 4 + b] = (lane * 4 + b) as u8;
                        b += 1;
                    }
                    at += 1;
                }
                lane += 1;
            }
            m += 1;
        }
        t
    };

    /// GELU over a slice with the lanes sorted by the branch of `erff` they take (and, in `erfc2`,
    /// its coefficient set), so every group of four evaluates one branch only (a vector mixing branches would evaluate all of them). The
    /// lanes of each branch are packed into a buffer with their positions (SIMD left-pack), the
    /// branch is evaluated on whole vectors and the results are written back. Groups holding an
    /// argument with |x / sqrt 2| >= 6 (or NaN, infinity) and the slice's last x.len() % 4 values take
    /// the scalar call.
    pub fn gelu_slice(x: &mut [f32]) {
        const BLK: usize = 512;
        // +4: a packed store writes a whole vector past the current count
        let mut vals = [[0f32; BLK + 4]; 4];
        let mut ids = [[0u32; BLK + 4]; 4];
        let mut res = [0f32; BLK + 4];
        let frac = splat(std::f32::consts::FRAC_1_SQRT_2);
        for blk in x.chunks_mut(BLK) {
            let mut cnt = [0usize; 4];
            let full = blk.len() / 4 * 4;
            for g in (0..full).step_by(4) {
                // SAFETY: g + 4 <= blk.len()
                let v = unsafe { v128_load(blk.as_ptr().add(g) as *const v128) };
                let ix = v128_and(f32x4_mul(v, frac), usplat(0x7fff_ffff));
                let c1 = u32x4_ge(ix, usplat(0x3f58_0000)); // |u| >= 0.84375
                let c2 = u32x4_ge(ix, usplat(0x3fa0_0000)); // |u| >= 1.25
                let c3 = u32x4_ge(ix, usplat(0x4036_db6d)); // |u| >= 1/0.35
                if v128_any_true(u32x4_ge(ix, usplat(0x40c0_0000))) {
                    for v in &mut blk[g..g + 4] {
                        *v = super::gelu(*v);
                    }
                    continue;
                }
                let idx = u32x4_add(usplat(g as u32), u32x4(0, 1, 2, 3));
                for (c, m) in [v128_not(c1), v128_andnot(c1, c2), v128_andnot(c2, c3), c3]
                    .into_iter()
                    .enumerate()
                {
                    let bits = i32x4_bitmask(m) as usize;
                    // SAFETY: bits < 16; LEFT_PACK rows are 16 bytes
                    let sh = unsafe { v128_load(LEFT_PACK[bits].as_ptr() as *const v128) };
                    let at = cnt[c];
                    // SAFETY: at <= BLK - 4 + 4 here (at most `full` lanes are packed per block), and
                    // the buffers have BLK + 4 slots
                    unsafe {
                        v128_store(
                            vals[c].as_mut_ptr().add(at) as *mut v128,
                            i8x16_swizzle(v, sh),
                        );
                        v128_store(
                            ids[c].as_mut_ptr().add(at) as *mut v128,
                            i8x16_swizzle(idx, sh),
                        );
                    }
                    cnt[c] = at + bits.count_ones() as usize;
                }
            }
            for v in &mut blk[full..] {
                *v = super::gelu(*v);
            }
            for c in 0..4 {
                let n = cnt[c];
                if n == 0 {
                    continue;
                }
                // pad the last group with a member of the class (its results are not written back)
                let first = vals[c][0];
                for v in &mut vals[c][n..n.div_ceil(4) * 4] {
                    *v = first;
                }
                for k in (0..n).step_by(4) {
                    // SAFETY: k < n <= BLK and the buffers hold BLK + 4
                    unsafe {
                        let v = v128_load(vals[c].as_ptr().add(k) as *const v128);
                        let u = f32x4_mul(v, frac);
                        let au = v128_and(u, usplat(0x7fff_ffff));
                        let e = match c {
                            0 => erf_small(u),
                            1 => signed_complement(erfc_near(au), u),
                            2 => signed_complement(erfc_far_a(au), u),
                            _ => signed_complement(erfc_far_b(au), u),
                        };
                        v128_store(res.as_mut_ptr().add(k) as *mut v128, gelu_from_erf(e, v));
                    }
                }
                for (&i, &r) in ids[c][..n].iter().zip(&res[..n]) {
                    blk[i as usize] = r;
                }
            }
        }
    }
}

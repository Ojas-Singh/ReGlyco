//! Torsion kinematics with the operation order of `glycoflow/geometry.py` (generic over f32 /
//! f64; the sampler uses f32 like the PyTorch reference), plus the analytic torsion Jacobian and
//! Cartesian -> torsion gradient propagation.

use std::ops::{Add, Div, Mul, Neg, Sub};

/// Floating-point type of the kinematics (f32 or f64).
pub trait Real:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;
    const PI: Self;
    const TAU: Self;
    const NORM_EPS: Self;
    fn fma(self, a: Self, b: Self) -> Self;
    fn sqrt(self) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn atan2(self, x: Self) -> Self;
    fn floor(self) -> Self;
    fn to_f64(self) -> f64;
    fn max(self, o: Self) -> Self {
        if self >= o {
            self
        } else {
            o
        }
    }
}

macro_rules! real_impl {
    ($t:ty) => {
        impl Real for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            const PI: Self = std::f64::consts::PI as $t;
            const TAU: Self = (2.0 * std::f64::consts::PI) as $t;
            const NORM_EPS: Self = 1e-8;
            #[inline]
            fn fma(self, a: Self, b: Self) -> Self {
                self.mul_add(a, b)
            }
            #[inline]
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            #[inline]
            fn sin(self) -> Self {
                <$t>::sin(self)
            }
            #[inline]
            fn cos(self) -> Self {
                <$t>::cos(self)
            }
            #[inline]
            fn atan2(self, x: Self) -> Self {
                <$t>::atan2(self, x)
            }
            #[inline]
            fn floor(self) -> Self {
                <$t>::floor(self)
            }
            #[inline]
            fn to_f64(self) -> f64 {
                self as f64
            }
        }
    };
}
real_impl!(f32);
real_impl!(f64);

pub type P3 = [f32; 3];

#[inline]
fn sub<T: Real>(a: [T; 3], b: [T; 3]) -> [T; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
fn dot<T: Real>(a: [T; 3], b: [T; 3]) -> T {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
/// Cross product with the rounding of PyTorch's CPU kernel (`a1*b2 - a2*b1` contracted to
/// `fma(a1, b2, -(a2*b1))`).
#[inline]
fn cross<T: Real>(a: [T; 3], b: [T; 3]) -> [T; 3] {
    [
        a[1].fma(b[2], -(a[2] * b[1])),
        a[2].fma(b[0], -(a[0] * b[2])),
        a[0].fma(b[1], -(a[1] * b[0])),
    ]
}
#[inline]
fn norm<T: Real>(a: [T; 3]) -> T {
    dot(a, a).sqrt()
}
#[inline]
fn unit<T: Real>(a: [T; 3]) -> [T; 3] {
    let n = norm(a).max(T::NORM_EPS);
    [a[0] / n, a[1] / n, a[2] / n]
}

/// Wrap radians to [-pi, pi): `torch.remainder(angle + pi, 2 pi) - pi`, with the rounding of
/// PyTorch's CPU kernel (`fma(-b, floor(a / b), a)`).
#[inline]
pub fn wrap<T: Real>(angle: T) -> T {
    let a = angle + T::PI;
    (-T::TAU).fma((a / T::TAU).floor(), a) - T::PI
}

/// Dihedral angle p0-p1-p2-p3 (radians).
#[inline]
pub fn dihedral<T: Real>(p0: [T; 3], p1: [T; 3], p2: [T; 3], p3: [T; 3]) -> T {
    let b0 = sub(p0, p1);
    let b1n = unit(sub(p2, p1));
    let b2 = sub(p3, p2);
    let d0 = dot(b0, b1n);
    let v = [
        b0[0] - d0 * b1n[0],
        b0[1] - d0 * b1n[1],
        b0[2] - d0 * b1n[2],
    ];
    let d2 = dot(b2, b1n);
    let w = [
        b2[0] - d2 * b1n[0],
        b2[1] - d2 * b1n[1],
        b2[2] - d2 * b1n[2],
    ];
    dot(cross(b1n, v), w).atan2(dot(v, w))
}

/// Dihedrals of one conformer x [N] for quads [T] -> [T].
pub fn dihedrals<T: Real>(x: &[[T; 3]], quads: &[[usize; 4]]) -> Vec<T> {
    quads
        .iter()
        .map(|q| dihedral(x[q[0]], x[q[1]], x[q[2]], x[q[3]]))
        .collect()
}

/// Rotate the distal side of every torsion bond by `delta` (in torsion order, each about the
/// current bond axis), as `geometry.rotate_torsions`.
pub fn rotate_torsions<T: Real>(
    x: &mut [[T; 3]],
    quads: &[[usize; 4]],
    distal: &[Vec<usize>],
    delta: &[T],
) {
    for (t, q) in quads.iter().enumerate() {
        let b = x[q[1]];
        let u = unit(sub(x[q[2]], b));
        let (sin, cos) = (delta[t].sin(), delta[t].cos());
        let omc = T::ONE - cos;
        for &i in &distal[t] {
            let v = sub(x[i], b);
            let cr = cross(u, v);
            let d = dot(u, v);
            let mut out = [T::ZERO; 3];
            for k in 0..3 {
                out[k] = b[k] + ((v[k] * cos + cr[k] * sin) + u[k] * d * omc);
            }
            x[i] = out;
        }
    }
}

/// Set the torsions of a conformer to `target` (`geometry.set_torsions`).
pub fn set_torsions<T: Real>(
    x: &mut [[T; 3]],
    quads: &[[usize; 4]],
    distal: &[Vec<usize>],
    target: &[T],
) {
    let cur = dihedrals(x, quads);
    let delta: Vec<T> = target
        .iter()
        .zip(&cur)
        .map(|(t, c)| wrap(*t - *c))
        .collect();
    rotate_torsions(x, quads, distal, &delta);
}

/// Analytic Jacobian dx_i/dtau_k = u_k x (x_i - b_k) for i on the distal side of torsion k
/// (zero elsewhere), evaluated at conformer `x`. Returns per torsion the (atom, derivative) pairs.
///
/// Valid for the sequential rotation of [`rotate_torsions`] at any rotation angle: torsions are
/// ordered by the depth of b, so a later rotation never moves the axis of an earlier one, and a
/// rotation nested inside a distal side commutes with the outer one.
pub fn torsion_jacobian<T: Real>(
    x: &[[T; 3]],
    quads: &[[usize; 4]],
    distal: &[Vec<usize>],
) -> Vec<Vec<(usize, [T; 3])>> {
    quads
        .iter()
        .zip(distal)
        .map(|(q, idx)| {
            let b = x[q[1]];
            let u = unit(sub(x[q[2]], b));
            idx.iter().map(|&i| (i, cross(u, sub(x[i], b)))).collect()
        })
        .collect()
}

/// Propagate a Cartesian gradient dE/dx [N] to torsion gradients dE/dtau [T] at conformer `x`:
/// dE/dtau_k = sum_{i distal to k} dE/dx_i . (u_k x (x_i - b_k)) = u_k . sum_i (x_i - b_k) x dE/dx_i.
/// Accumulates in f64.
pub fn torsion_gradient<T: Real>(
    x: &[[T; 3]],
    quads: &[[usize; 4]],
    distal: &[Vec<usize>],
    grad_x: &[[f64; 3]],
) -> Vec<f64> {
    let f = |p: [T; 3]| [p[0].to_f64(), p[1].to_f64(), p[2].to_f64()];
    quads
        .iter()
        .zip(distal)
        .map(|(q, idx)| {
            let b = f(x[q[1]]);
            let u = unit(sub(f(x[q[2]]), b));
            let mut s = [0f64; 3];
            for &i in idx {
                let r = sub(f(x[i]), b);
                let g = grad_x[i];
                s[0] += r[1] * g[2] - r[2] * g[1];
                s[1] += r[2] * g[0] - r[0] * g[2];
                s[2] += r[0] * g[1] - r[1] * g[0];
            }
            dot(u, s)
        })
        .collect()
}

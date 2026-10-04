//! f64 3-vector helpers and the Kabsch fit used by the builder and the ensemble alignment.

pub type V3 = [f64; 3];
pub type M3 = [[f64; 3]; 3];

#[inline]
pub fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
pub fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
#[inline]
pub fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
#[inline]
pub fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
pub fn cross(a: V3, b: V3) -> V3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
#[inline]
pub fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
#[inline]
pub fn normalize(a: V3) -> V3 {
    scale(a, 1.0 / norm(a))
}

/// `m @ v` for a row-major 3x3 matrix.
#[inline]
pub fn matvec(m: &M3, v: V3) -> V3 {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

pub fn mean(points: &[V3]) -> V3 {
    let mut s = [0.0; 3];
    for p in points {
        s = add(s, *p);
    }
    scale(s, 1.0 / points.len() as f64)
}

/// Proper rotation `r` and translation `t` with `r @ src_i + t ~ dst_i` (Kabsch; `builder._align`).
pub fn kabsch(src: &[V3], dst: &[V3]) -> (M3, V3) {
    let cs = mean(src);
    let cd = mean(dst);
    let mut h = nalgebra::Matrix3::<f64>::zeros();
    for (s, d) in src.iter().zip(dst) {
        let a = sub(*s, cs);
        let b = sub(*d, cd);
        for i in 0..3 {
            for j in 0..3 {
                h[(i, j)] += a[i] * b[j];
            }
        }
    }
    let svd = h.svd(true, true);
    let u = svd.u.expect("svd u");
    let vt = svd.v_t.expect("svd v_t");
    // det(V U^T) = det(V) det(U)
    let det = (vt.transpose() * u.transpose()).determinant();
    let d = if det > 0.0 {
        1.0
    } else if det < 0.0 {
        -1.0
    } else {
        0.0
    };
    // r = V diag(1, 1, d) U^T with d on the smallest singular value (numpy sorts them descending)
    let sv = svd.singular_values;
    let kmin = (0..3)
        .min_by(|&a, &b| sv[a].partial_cmp(&sv[b]).unwrap())
        .unwrap();
    let mut r = [[0.0; 3]; 3];
    for k in 0..3 {
        let w = if k == kmin { d } else { 1.0 };
        for i in 0..3 {
            for j in 0..3 {
                // V[i,k] = vt[k,i], U[j,k]
                r[i][j] += w * vt[(k, i)] * u[(j, k)];
            }
        }
    }
    let t = sub(cd, matvec(&r, cs));
    (r, t)
}

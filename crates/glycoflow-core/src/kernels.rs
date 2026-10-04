//! Fused inference kernels for the model's memory-bound parts (candle custom ops, no backward).
//!
//! Each op has a CPU implementation (rayon with the `parallel` feature) and, with the `cuda`
//! feature, a CUDA kernel compiled once per process with NVRTC. They compute the same expressions
//! in the same order as the composed tensor ops of [`crate::model`] (rounding may differ in
//! reductions); the composed path remains available through [`crate::model::Ops::composed`].

use candle_core::{CpuStorage, CustomOp2, CustomOp3, Layout, Result, Shape, Tensor};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Activation applied by [`bias_act`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    None = 0,
    Silu = 1,
    Gelu = 2,
}

fn cpu_f32<'a>(s: &'a CpuStorage, l: &Layout, what: &str) -> Result<&'a [f32]> {
    let v = match s {
        CpuStorage::F32(v) => v.as_slice(),
        _ => candle_core::bail!("{what}: expected f32"),
    };
    match l.contiguous_offsets() {
        Some((a, b)) => Ok(&v[a..b]),
        None => candle_core::bail!("{what}: input must be contiguous"),
    }
}

fn cpu_u32<'a>(s: &'a CpuStorage, l: &Layout, what: &str) -> Result<&'a [u32]> {
    let v = match s {
        CpuStorage::U32(v) => v.as_slice(),
        _ => candle_core::bail!("{what}: expected u32"),
    };
    match l.contiguous_offsets() {
        Some((a, b)) => Ok(&v[a..b]),
        None => candle_core::bail!("{what}: input must be contiguous"),
    }
}

/// Run `f(chunk_index, chunk)` over `out.chunks_mut(chunk)`, in parallel with rayon when enabled.
fn for_chunks(out: &mut [f32], chunk: usize, f: impl Fn(usize, &mut [f32]) + Sync + Send) {
    #[cfg(feature = "parallel")]
    out.par_chunks_mut(chunk)
        .enumerate()
        .for_each(|(i, c)| f(i, c));
    #[cfg(not(feature = "parallel"))]
    out.chunks_mut(chunk).enumerate().for_each(|(i, c)| f(i, c));
}

#[inline]
fn silu(v: f32) -> f32 {
    v / (1.0 + (-v).exp())
}

#[inline]
fn gelu_cpu(v: f32) -> f32 {
    // candle's CPU gelu_erf formula
    (candle_core::cpu::erf::erf_f32(v * std::f32::consts::FRAC_1_SQRT_2) + 1.) * 0.5 * v
}

// ------------------------------------------------------------------------------------------------
// CUDA source (NVRTC)

#[cfg(feature = "cuda")]
mod cuda {
    use candle_core::cuda_backend::cudarc::driver::{CudaSlice, LaunchConfig};
    use candle_core::cuda_backend::{CudaDevice, CudaStorage, WrapErr};
    use candle_core::{Layout, Result};
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};

    pub const SRC: &str = r#"
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, o));
    return v;
}

// grid: x over columns, y over rows (grid-stride)
extern "C" __global__ void gf_bias_act(const float* x, const float* b, float* y, int rows, int c, int act) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= c) return;
    float bc = b[col];
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        size_t i = (size_t)row * c + col;
        float v = x[i] + bc;
        if (act == 1) v = v / (1.0f + expf(-v));
        else if (act == 2) v = v * 0.5f * (1.0f + erff(v * 0.70710678118654752440f));
        y[i] = v;
    }
}

// x [rows = B*N, 3*D] + b -> out [3, B, H, N, dh] (q, k, v split into heads)
extern "C" __global__ void gf_bias_split_heads(const float* x, const float* b, float* out, int rows, int n,
                                               int heads, int dh) {
    int d = heads * dh;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= 3 * d) return;
    int which = col / d, rem = col % d, head = rem / dh, e = rem % dh;
    int nb = rows / n;
    float bc = b[col];
    for (int row = blockIdx.y; row < rows; row += gridDim.y) {
        int bi = row / n, i = row % n;
        out[((((size_t)which * nb + bi) * heads + head) * n + i) * dh + e] = x[(size_t)row * 3 * d + col] + bc;
    }
}

// one warp per row; mode bit 0: modulate y*(1+ada[scale])+ada[shift]; bit 1: affine y*w+b
extern "C" __global__ void gf_ln_mod(const float* x, float* y, int rows, int d, int rows_per_batch,
                                     const float* ada, int ada_stride, int scale_off, int shift_off,
                                     const float* w, const float* bb, int mode) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row >= rows) return;
    const float* xr = x + (size_t)row * d;
    float* yr = y + (size_t)row * d;
    float s = 0.0f;
    for (int k = lane; k < d; k += 32) s += xr[k];
    float mean = warp_sum(s) / d;
    float v = 0.0f;
    for (int k = lane; k < d; k += 32) { float t = xr[k] - mean; v += t * t; }
    float var = warp_sum(v) / d;
    float rstd = 1.0f / sqrtf(var + 1e-5f);
    const float* ar = ada + (size_t)(row / rows_per_batch) * ada_stride;
    for (int k = lane; k < d; k += 32) {
        float t = (xr[k] - mean) * rstd;
        if (mode & 1) t = t * (1.0f + ar[scale_off + k]) + ar[shift_off + k];
        if (mode & 2) t = t * w[k] + bb[k];
        yr[k] = t;
    }
}

extern "C" __global__ void gf_gated_add(const float* h, const float* y, const float* ada, float* out,
                                        unsigned long long n, int d, int rows_per_batch, int ada_stride, int off) {
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long row = i / d;
    int k = (int)(i % d);
    out[i] = h[i] + ada[(row / rows_per_batch) * ada_stride + off + k] * y[i];
}

// one warp per row: softmax(x * scale + bias) (div: x / scale + bias); rows of up to
// 32 * SM_REG values stay in registers
#define SM_REG 16
extern "C" __global__ void gf_scale_bias_softmax(const float* x, const float* bias, float* y,
                                                 int rows, int n, float scale, int div) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row >= rows) return;
    const float* xr = x + (size_t)row * n;
    const float* br = bias + (size_t)row * n;
    float* yr = y + (size_t)row * n;
    float m = -__int_as_float(0x7f800000);
    if (n <= 32 * SM_REG) {
        float v[SM_REG];
#pragma unroll
        for (int t = 0; t < SM_REG; ++t) {
            int j = lane + 32 * t;
            v[t] = -__int_as_float(0x7f800000);
            if (j < n) { v[t] = (div ? xr[j] / scale : xr[j] * scale) + br[j]; m = fmaxf(m, v[t]); }
        }
        m = warp_max(m);
        float s = 0.0f;
#pragma unroll
        for (int t = 0; t < SM_REG; ++t) {
            int j = lane + 32 * t;
            if (j < n) { v[t] = expf(v[t] - m); s += v[t]; }
        }
        s = warp_sum(s);
#pragma unroll
        for (int t = 0; t < SM_REG; ++t) {
            int j = lane + 32 * t;
            if (j < n) yr[j] = v[t] / s;
        }
        return;
    }
    for (int j = lane; j < n; j += 32) {
        float v = (div ? xr[j] / scale : xr[j] * scale) + br[j];
        yr[j] = v;
        m = fmaxf(m, v);
    }
    m = warp_max(m);
    float s = 0.0f;
    for (int j = lane; j < n; j += 32) { float e = expf(yr[j] - m); yr[j] = e; s += e; }
    s = warp_sum(s);
    for (int j = lane; j < n; j += 32) yr[j] = yr[j] / s;
}

// thread per pair (b, i, j): rbf -> Linear -> SiLU -> Linear (+ topology embedding) -> Linear,
// written layer-major: out[((c / H) * B + b) * H + c % H][i][j]. GF_R, GF_P, GF_C are defined
// when the source is compiled (rbf bins, pair dim, depth * heads), so every loop unrolls.
#define GF_NP (GF_R + 1 + GF_P * GF_R + GF_P + GF_P * GF_P + GF_P + 14 * GF_P + GF_C * GF_P + GF_C)
extern "C" __global__ void gf_pair_bias(const float* dist, const unsigned int* topo, const float* params,
                                        float* out, int B, int N, int topo_batched, int H) {
    __shared__ float sp[GF_NP];
    for (int k = threadIdx.x; k < GF_NP; k += blockDim.x) sp[k] = params[k];
    __syncthreads();
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t nn = (size_t)N * N;
    if (idx >= (size_t)B * nn) return;
    int b = (int)(idx / nn);
    int ij = (int)(idx % nn);
    const float* centers = sp;
    const float width = sp[GF_R];
    const float* w0 = sp + GF_R + 1;
    const float* b0 = w0 + GF_P * GF_R;
    const float* w2 = b0 + GF_P;
    const float* b2 = w2 + GF_P * GF_P;
    const float* emb = b2 + GF_P;
    const float* wb = emb + 14 * GF_P;
    const float* bb = wb + GF_C * GF_P;
    float d = dist[idx];
    float rbf[GF_R];
#pragma unroll
    for (int k = 0; k < GF_R; ++k) { float t = (d - centers[k]) / width; rbf[k] = expf(-(t * t)); }
    float h1[GF_P];
#pragma unroll
    for (int o = 0; o < GF_P; ++o) {
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < GF_R; ++k) acc = fmaf(rbf[k], w0[o * GF_R + k], acc);
        float v = acc + b0[o];
        h1[o] = v / (1.0f + expf(-v));
    }
    unsigned int tp = topo[topo_batched ? idx : (size_t)ij];
    const float* e = emb + tp * GF_P;
    float h2[GF_P];
#pragma unroll
    for (int o = 0; o < GF_P; ++o) {
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < GF_P; ++k) acc = fmaf(h1[k], w2[o * GF_P + k], acc);
        h2[o] = (acc + b2[o]) + e[o];
    }
#pragma unroll 8
    for (int c = 0; c < GF_C; ++c) {
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < GF_P; ++k) acc = fmaf(h2[k], wb[c * GF_P + k], acc);
        out[((size_t)((c / H) * B + b) * H + (c % H)) * nn + ij] = acc + bb[c];
    }
}
"#;

    type PtxCache =
        Mutex<HashMap<(usize, usize, usize), std::result::Result<&'static str, String>>>;
    static PTX: OnceLock<PtxCache> = OnceLock::new();

    /// Module name and PTX for the kernels specialised to (rbf bins, pair dim, depth * heads).
    pub fn ptx(dims: (usize, usize, usize)) -> Result<(String, &'static str)> {
        let cache = PTX.get_or_init(|| Mutex::new(HashMap::new()));
        let mut cache = cache.lock().unwrap();
        let entry = cache.entry(dims).or_insert_with(|| {
            let src = format!(
                "#define GF_R {}\n#define GF_P {}\n#define GF_C {}\n{SRC}",
                dims.0, dims.1, dims.2
            );
            candle_core::cuda_backend::cudarc::nvrtc::safe::compile_ptx(src)
                .map(|p| &*Box::leak(p.to_src().into_boxed_str()))
                .map_err(|e| format!("{e:?}"))
        });
        match entry {
            Ok(s) => Ok((
                format!("glycoflow_kernels_{}_{}_{}", dims.0, dims.1, dims.2),
                *s,
            )),
            Err(e) => candle_core::bail!("NVRTC compilation of the GlycoFlow kernels failed: {e}"),
        }
    }

    /// Kernels that do not depend on the model dimensions.
    pub fn ptx_any() -> Result<(String, &'static str)> {
        ptx((32, 32, 64))
    }

    pub fn slice<'a, T: candle_core::cuda_backend::CudaDType>(
        s: &'a CudaStorage,
        l: &Layout,
    ) -> Result<candle_core::cuda_backend::cudarc::driver::CudaView<'a, T>> {
        let v = s.as_cuda_slice::<T>()?;
        match l.contiguous_offsets() {
            Some((a, b)) => Ok(v.slice(a..b)),
            None => candle_core::bail!("custom op input must be contiguous"),
        }
    }

    pub fn alloc(dev: &CudaDevice, n: usize) -> Result<CudaSlice<f32>> {
        // SAFETY: every element is written by the kernel.
        unsafe { dev.alloc::<f32>(n) }
    }

    pub fn grid(n: usize, block: u32) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (n.div_ceil(block as usize) as u32, 1, 1),
            block_dim: (block, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    /// x over columns (256 per block), y over rows (grid-stride beyond 65535)
    pub fn grid2(cols: usize, rows: usize) -> LaunchConfig {
        LaunchConfig {
            grid_dim: (cols.div_ceil(256) as u32, rows.clamp(1, 65535) as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    }

    pub fn wrap(dst: CudaSlice<f32>, dev: &CudaDevice) -> CudaStorage {
        CudaStorage::wrap_cuda_slice(dst, dev.clone())
    }

    pub fn launch(
        b: &mut candle_core::cuda_backend::cudarc::driver::LaunchArgs<'_>,
        cfg: LaunchConfig,
    ) -> Result<()> {
        // SAFETY: argument lists are checked against the kernel signatures above.
        unsafe { b.launch(cfg) }.w()?;
        Ok(())
    }
}

#[cfg(feature = "cuda")]
use candle_core::cuda_backend::cudarc::driver::PushKernelArg;

// ------------------------------------------------------------------------------------------------

/// `act(x + b)` with x [..., C] contiguous and b [C].
struct BiasAct(Act);

impl CustomOp2 for BiasAct {
    fn name(&self) -> &'static str {
        "gf-bias-act"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let x = cpu_f32(s1, l1, "bias_act")?;
        let b = cpu_f32(s2, l2, "bias_act")?;
        let c = b.len();
        let mut y = vec![0f32; x.len()];
        let act = self.0;
        let rows = 4096usize.div_ceil(c).max(1) * c;
        for_chunks(&mut y, rows, |ci, out| {
            let xs = &x[ci * rows..ci * rows + out.len()];
            for (k, (o, v)) in out.iter_mut().zip(xs).enumerate() {
                let v = v + b[k % c];
                *o = match act {
                    Act::None => v,
                    Act::Silu => silu(v),
                    Act::Gelu => gelu_cpu(v),
                };
            }
        });
        Ok((CpuStorage::F32(y), l1.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let dev = s1.device.clone();
        let x = cuda::slice::<f32>(s1, l1)?;
        let b = cuda::slice::<f32>(s2, l2)?;
        let n = l1.shape().elem_count();
        let c = l2.shape().elem_count();
        let rows = n / c;
        let dst = cuda::alloc(&dev, n)?;
        let (module, ptx) = cuda::ptx_any()?;
        let f = dev.get_or_load_custom_func("gf_bias_act", &module, ptx)?;
        let mut a = f.builder();
        let (rows_i, c_i, act) = (rows as i32, c as i32, self.0 as i32);
        a.arg(&x).arg(&b).arg(&dst).arg(&rows_i).arg(&c_i).arg(&act);
        cuda::launch(&mut a, cuda::grid2(c, rows))?;
        Ok((cuda::wrap(dst, &dev), l1.shape().clone()))
    }
}

/// `act(x + b)`, x [..., C] (contiguous), b [C].
pub fn bias_act(x: &Tensor, b: &Tensor, act: Act) -> Result<Tensor> {
    x.contiguous()?.apply_op2_no_bwd(b, &BiasAct(act))
}

// ------------------------------------------------------------------------------------------------

/// `x + b` for x [B*N, 3*D], written as [3, B, H, N, dh] (q, k, v split into heads).
struct BiasSplitHeads {
    n: usize,
    heads: usize,
}

impl CustomOp2 for BiasSplitHeads {
    fn name(&self) -> &'static str {
        "gf-bias-split-heads"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let x = cpu_f32(s1, l1, "split_heads")?;
        let b = cpu_f32(s2, l2, "split_heads")?;
        let (rows, c3) = l1.shape().dims2()?;
        let (n, h) = (self.n, self.heads);
        let d = c3 / 3;
        let dh = d / h;
        let nb = rows / n;
        let mut out = vec![0f32; x.len()];
        // block (which, b, head) = [N, dh]
        for_chunks(&mut out, n * dh, |blk, o| {
            let head = blk % h;
            let bi = (blk / h) % nb;
            let which = blk / (h * nb);
            for i in 0..n {
                let src = (bi * n + i) * c3 + which * d + head * dh;
                for e in 0..dh {
                    o[i * dh + e] = x[src + e] + b[which * d + head * dh + e];
                }
            }
        });
        Ok((CpuStorage::F32(out), Shape::from((3, nb, h, n, dh))))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let dev = s1.device.clone();
        let x = cuda::slice::<f32>(s1, l1)?;
        let b = cuda::slice::<f32>(s2, l2)?;
        let (rows, c3) = l1.shape().dims2()?;
        let (n, h) = (self.n, self.heads);
        let dh = c3 / 3 / h;
        let dst = cuda::alloc(&dev, rows * c3)?;
        let (module, ptx) = cuda::ptx_any()?;
        let f = dev.get_or_load_custom_func("gf_bias_split_heads", &module, ptx)?;
        let mut a = f.builder();
        let (rows_i, n_i, h_i, dh_i) = (rows as i32, n as i32, h as i32, dh as i32);
        a.arg(&x)
            .arg(&b)
            .arg(&dst)
            .arg(&rows_i)
            .arg(&n_i)
            .arg(&h_i)
            .arg(&dh_i);
        cuda::launch(&mut a, cuda::grid2(c3, rows))?;
        Ok((cuda::wrap(dst, &dev), Shape::from((3, rows / n, h, n, dh))))
    }
}

/// `x + b` for x [B*N, 3*D] (the qkv projection without bias), as [3, B, H, N, dh].
pub fn bias_split_heads(x: &Tensor, b: &Tensor, n: usize, heads: usize) -> Result<Tensor> {
    x.contiguous()?
        .apply_op2_no_bwd(b, &BiasSplitHeads { n, heads })
}

// ------------------------------------------------------------------------------------------------

/// Layer norm over the last dim (eps 1e-5), optionally modulated by chunks of a per-batch adaLN
/// vector `ada` [B, K*D] (`y * (1 + ada[scale]) + ada[shift]`) and/or an affine (w, b).
struct LnMod {
    rows_per_batch: usize,
    scale_off: usize,
    shift_off: usize,
    mode: i32,
}

impl LnMod {
    #[allow(clippy::too_many_arguments)]
    fn run_cpu(
        &self,
        x: &[f32],
        d: usize,
        ada: &[f32],
        ada_stride: usize,
        w: &[f32],
        bb: &[f32],
    ) -> Vec<f32> {
        let mut y = vec![0f32; x.len()];
        for_chunks(&mut y, d, |row, out| {
            let xr = &x[row * d..(row + 1) * d];
            let mean = xr.iter().sum::<f32>() / d as f32;
            let var = xr.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / d as f32;
            let rstd = 1.0 / (var + 1e-5).sqrt();
            let ar = if self.mode & 1 != 0 {
                &ada[(row / self.rows_per_batch) * ada_stride..]
            } else {
                &[][..]
            };
            for k in 0..d {
                let mut t = (xr[k] - mean) * rstd;
                if self.mode & 1 != 0 {
                    t = t * (1.0 + ar[self.scale_off + k]) + ar[self.shift_off + k];
                }
                if self.mode & 2 != 0 {
                    t = t * w[k] + bb[k];
                }
                out[k] = t;
            }
        });
        y
    }
}

impl CustomOp3 for LnMod {
    fn name(&self) -> &'static str {
        "gf-ln-mod"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let x = cpu_f32(s1, l1, "ln_mod")?;
        let d = *l1.shape().dims().last().unwrap();
        let (ada, w, bb, stride) = if self.mode & 1 != 0 {
            let a = cpu_f32(s2, l2, "ln_mod")?;
            (a, &[][..], &[][..], *l2.shape().dims().last().unwrap())
        } else {
            (
                &[][..],
                cpu_f32(s2, l2, "ln_mod")?,
                cpu_f32(s3, l3, "ln_mod")?,
                0,
            )
        };
        Ok((
            CpuStorage::F32(self.run_cpu(x, d, ada, stride, w, bb)),
            l1.shape().clone(),
        ))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
        s3: &candle_core::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let dev = s1.device.clone();
        let x = cuda::slice::<f32>(s1, l1)?;
        let p2 = cuda::slice::<f32>(s2, l2)?;
        let p3 = cuda::slice::<f32>(s3, l3)?;
        let n = l1.shape().elem_count();
        let d = *l1.shape().dims().last().unwrap();
        let rows = n / d;
        let dst = cuda::alloc(&dev, n)?;
        let (module, ptx) = cuda::ptx_any()?;
        let f = dev.get_or_load_custom_func("gf_ln_mod", &module, ptx)?;
        let mut a = f.builder();
        let (rows_i, d_i, rpb) = (rows as i32, d as i32, self.rows_per_batch as i32);
        let stride = if self.mode & 1 != 0 {
            *l2.shape().dims().last().unwrap() as i32
        } else {
            0
        };
        let (so, sh, mode) = (self.scale_off as i32, self.shift_off as i32, self.mode);
        // modulate: (ada, unused); affine: (w, b)
        a.arg(&x)
            .arg(&dst)
            .arg(&rows_i)
            .arg(&d_i)
            .arg(&rpb)
            .arg(&p2)
            .arg(&stride)
            .arg(&so)
            .arg(&sh)
            .arg(&p2)
            .arg(&p3)
            .arg(&mode);
        cuda::launch(&mut a, cuda::grid(rows * 32, 256))?;
        Ok((cuda::wrap(dst, &dev), l1.shape().clone()))
    }
}

/// `LN(x) * (1 + ada[:, scale]) + ada[:, shift]` for x [B, N, D], ada [B, K*D] (chunk offsets in units of D).
pub fn ln_modulate(
    x: &Tensor,
    ada: &Tensor,
    scale_chunk: usize,
    shift_chunk: usize,
) -> Result<Tensor> {
    let (_, n, d) = x.dims3()?;
    let op = LnMod {
        rows_per_batch: n,
        scale_off: scale_chunk * d,
        shift_off: shift_chunk * d,
        mode: 1,
    };
    x.contiguous()?
        .apply_op3_no_bwd(&ada.contiguous()?, &ada.contiguous()?, &op)
}

/// `LN(x) * w + b` over the last dim.
pub fn ln_affine(x: &Tensor, w: &Tensor, b: &Tensor) -> Result<Tensor> {
    let op = LnMod {
        rows_per_batch: 1,
        scale_off: 0,
        shift_off: 0,
        mode: 2,
    };
    x.contiguous()?.apply_op3_no_bwd(w, b, &op)
}

// ------------------------------------------------------------------------------------------------

/// `h + ada[:, gate] * y` for h, y [B, N, D], ada [B, K*D].
struct GatedAdd {
    rows_per_batch: usize,
    off: usize,
}

impl CustomOp3 for GatedAdd {
    fn name(&self) -> &'static str {
        "gf-gated-add"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let h = cpu_f32(s1, l1, "gated_add")?;
        let y = cpu_f32(s2, l2, "gated_add")?;
        let ada = cpu_f32(s3, l3, "gated_add")?;
        let d = *l1.shape().dims().last().unwrap();
        let stride = *l3.shape().dims().last().unwrap();
        let mut out = vec![0f32; h.len()];
        for_chunks(&mut out, d, |row, o| {
            let g = &ada[(row / self.rows_per_batch) * stride + self.off..];
            for k in 0..d {
                o[k] = h[row * d + k] + g[k] * y[row * d + k];
            }
        });
        Ok((CpuStorage::F32(out), l1.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
        s3: &candle_core::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let dev = s1.device.clone();
        let h = cuda::slice::<f32>(s1, l1)?;
        let y = cuda::slice::<f32>(s2, l2)?;
        let ada = cuda::slice::<f32>(s3, l3)?;
        let n = l1.shape().elem_count();
        let d = *l1.shape().dims().last().unwrap() as i32;
        let stride = *l3.shape().dims().last().unwrap() as i32;
        let dst = cuda::alloc(&dev, n)?;
        let (module, ptx) = cuda::ptx_any()?;
        let f = dev.get_or_load_custom_func("gf_gated_add", &module, ptx)?;
        let mut a = f.builder();
        let (n64, rpb, off) = (n as u64, self.rows_per_batch as i32, self.off as i32);
        a.arg(&h)
            .arg(&y)
            .arg(&ada)
            .arg(&dst)
            .arg(&n64)
            .arg(&d)
            .arg(&rpb)
            .arg(&stride)
            .arg(&off);
        cuda::launch(&mut a, cuda::grid(n, 256))?;
        Ok((cuda::wrap(dst, &dev), l1.shape().clone()))
    }
}

/// `h + ada[:, gate_chunk] * y` (h, y [B, N, D]; ada [B, K*D]).
pub fn gated_add(h: &Tensor, y: &Tensor, ada: &Tensor, gate_chunk: usize) -> Result<Tensor> {
    let (_, n, d) = h.dims3()?;
    let op = GatedAdd {
        rows_per_batch: n,
        off: gate_chunk * d,
    };
    h.contiguous()?
        .apply_op3_no_bwd(&y.contiguous()?, &ada.contiguous()?, &op)
}

// ------------------------------------------------------------------------------------------------

/// `softmax(x * scale + bias)` (or `x / scale + bias`) over the last dim.
struct ScaleBiasSoftmax {
    scale: f32,
    div: bool,
}

impl CustomOp2 for ScaleBiasSoftmax {
    fn name(&self) -> &'static str {
        "gf-scale-bias-softmax"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let x = cpu_f32(s1, l1, "softmax")?;
        let bias = cpu_f32(s2, l2, "softmax")?;
        let n = *l1.shape().dims().last().unwrap();
        let mut y = vec![0f32; x.len()];
        let (scale, div) = (self.scale, self.div);
        for_chunks(&mut y, n, |row, out| {
            let (xr, br) = (&x[row * n..(row + 1) * n], &bias[row * n..(row + 1) * n]);
            let mut m = f32::NEG_INFINITY;
            for j in 0..n {
                let v = (if div { xr[j] / scale } else { xr[j] * scale }) + br[j];
                out[j] = v;
                m = m.max(v);
            }
            let mut s = 0f32;
            for o in out.iter_mut() {
                *o = (*o - m).exp();
                s += *o;
            }
            for o in out.iter_mut() {
                *o /= s;
            }
        });
        Ok((CpuStorage::F32(y), l1.shape().clone()))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let dev = s1.device.clone();
        let x = cuda::slice::<f32>(s1, l1)?;
        let bias = cuda::slice::<f32>(s2, l2)?;
        let total = l1.shape().elem_count();
        let n = *l1.shape().dims().last().unwrap();
        let rows = total / n;
        let dst = cuda::alloc(&dev, total)?;
        let (module, ptx) = cuda::ptx_any()?;
        let f = dev.get_or_load_custom_func("gf_scale_bias_softmax", &module, ptx)?;
        let mut a = f.builder();
        let (rows_i, n_i, scale, div) = (rows as i32, n as i32, self.scale, self.div as i32);
        a.arg(&x)
            .arg(&bias)
            .arg(&dst)
            .arg(&rows_i)
            .arg(&n_i)
            .arg(&scale)
            .arg(&div);
        cuda::launch(&mut a, cuda::grid(rows * 32, 256))?;
        Ok((cuda::wrap(dst, &dev), l1.shape().clone()))
    }
}

/// `softmax(x * scale + bias)` (`div`: `x / scale + bias`) over the last dim; x and bias have the
/// same shape and are contiguous.
pub fn scale_bias_softmax(x: &Tensor, bias: &Tensor, scale: f32, div: bool) -> Result<Tensor> {
    x.contiguous()?
        .apply_op2_no_bwd(&bias.contiguous()?, &ScaleBiasSoftmax { scale, div })
}

// ------------------------------------------------------------------------------------------------

/// Packed parameters of the pair-bias MLP:
/// centers [R], width, W0 [P,R], b0 [P], W2 [P,P], b2 [P], topology embedding [14,P], Wb [C,P], bb [C].
pub struct PairParams {
    pub r: usize,
    pub p: usize,
    pub c: usize,
    pub heads: usize,
    pub packed: Tensor,
}

struct PairBias<'a> {
    prm: &'a PairParams,
    topo_batched: bool,
}

impl PairBias<'_> {
    fn run_cpu(&self, dist: &[f32], topo: &[u32], p: &[f32], b: usize, n: usize) -> Vec<f32> {
        let (r, pd, c, h) = (self.prm.r, self.prm.p, self.prm.c, self.prm.heads);
        let centers = &p[..r];
        let width = p[r];
        let w0 = &p[r + 1..];
        let b0 = &w0[pd * r..];
        let w2 = &b0[pd..];
        let b2 = &w2[pd * pd..];
        let emb = &b2[pd..];
        let wb = &emb[14 * pd..];
        let bb = &wb[c * pd..];
        // transposed weights: the output channel is the inner (vectorised) loop; every output
        // still accumulates over k in ascending order, as the CUDA kernel
        let tr = |w: &[f32], rows: usize, cols: usize| -> Vec<f32> {
            let mut t = vec![0f32; rows * cols];
            for o in 0..rows {
                for k in 0..cols {
                    t[k * rows + o] = w[o * cols + k];
                }
            }
            t
        };
        let (w0t, w2t, wbt) = (
            tr(&w0[..pd * r], pd, r),
            tr(&w2[..pd * pd], pd, pd),
            tr(&wb[..c * pd], c, pd),
        );
        // rows (b, i) computed in parallel into a [B*N, C, N] scratch, then scattered layer-major
        let mut tmp = vec![0f32; b * n * c * n];
        for_chunks(&mut tmp, c * n, |row, out| {
            let bi = row / n;
            let i = row % n;
            let mut rbf = vec![0f32; r];
            let mut h1 = vec![0f32; pd];
            let mut h2 = vec![0f32; pd];
            let mut o3 = vec![0f32; c];
            for j in 0..n {
                let d = dist[(bi * n + i) * n + j];
                for k in 0..r {
                    let t = (d - centers[k]) / width;
                    rbf[k] = (-(t * t)).exp();
                }
                h1.iter_mut().for_each(|v| *v = 0.0);
                for k in 0..r {
                    let (x, w) = (rbf[k], &w0t[k * pd..(k + 1) * pd]);
                    for (acc, wv) in h1.iter_mut().zip(w) {
                        *acc = x.mul_add(*wv, *acc);
                    }
                }
                for o in 0..pd {
                    h1[o] = silu(h1[o] + b0[o]);
                }
                let tp = topo[if self.topo_batched {
                    (bi * n + i) * n + j
                } else {
                    i * n + j
                }] as usize;
                let e = &emb[tp * pd..(tp + 1) * pd];
                h2.iter_mut().for_each(|v| *v = 0.0);
                for k in 0..pd {
                    let (x, w) = (h1[k], &w2t[k * pd..(k + 1) * pd]);
                    for (acc, wv) in h2.iter_mut().zip(w) {
                        *acc = x.mul_add(*wv, *acc);
                    }
                }
                for o in 0..pd {
                    h2[o] = (h2[o] + b2[o]) + e[o];
                }
                o3.iter_mut().for_each(|v| *v = 0.0);
                for k in 0..pd {
                    let (x, w) = (h2[k], &wbt[k * c..(k + 1) * c]);
                    for (acc, wv) in o3.iter_mut().zip(w) {
                        *acc = x.mul_add(*wv, *acc);
                    }
                }
                for cc in 0..c {
                    out[cc * n + j] = o3[cc] + bb[cc];
                }
            }
        });
        let l = c / h;
        let mut out = vec![0f32; b * n * c * n];
        for_chunks(&mut out, n * n, |blk, o| {
            // blk = (layer * B + b) * H + head
            let head = blk % h;
            let bi = (blk / h) % b;
            let layer = blk / (h * b);
            let cc = layer * h + head;
            for i in 0..n {
                let src = &tmp[((bi * n + i) * c + cc) * n..((bi * n + i) * c + cc + 1) * n];
                o[i * n..(i + 1) * n].copy_from_slice(src);
            }
        });
        let _ = l;
        out
    }
}

impl CustomOp3 for PairBias<'_> {
    fn name(&self) -> &'static str {
        "gf-pair-bias"
    }

    fn cpu_fwd(
        &self,
        s1: &CpuStorage,
        l1: &Layout,
        s2: &CpuStorage,
        l2: &Layout,
        s3: &CpuStorage,
        l3: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        let dist = cpu_f32(s1, l1, "pair_bias")?;
        let topo = cpu_u32(s2, l2, "pair_bias")?;
        let p = cpu_f32(s3, l3, "pair_bias")?;
        let (b, n, _) = l1.shape().dims3()?;
        let out = self.run_cpu(dist, topo, p, b, n);
        let (c, h) = (self.prm.c, self.prm.heads);
        Ok((CpuStorage::F32(out), Shape::from((c / h, b, h, n, n))))
    }

    #[cfg(feature = "cuda")]
    fn cuda_fwd(
        &self,
        s1: &candle_core::CudaStorage,
        l1: &Layout,
        s2: &candle_core::CudaStorage,
        l2: &Layout,
        s3: &candle_core::CudaStorage,
        l3: &Layout,
    ) -> Result<(candle_core::CudaStorage, Shape)> {
        let dev = s1.device.clone();
        let dist = cuda::slice::<f32>(s1, l1)?;
        let topo = cuda::slice::<u32>(s2, l2)?;
        let params = cuda::slice::<f32>(s3, l3)?;
        let (b, n, _) = l1.shape().dims3()?;
        let (r, pd, c, h) = (self.prm.r, self.prm.p, self.prm.c, self.prm.heads);
        let total = b * n * n;
        let dst = cuda::alloc(&dev, total * c)?;
        let (module, ptx) = cuda::ptx((r, pd, c))?;
        let f = dev.get_or_load_custom_func("gf_pair_bias", &module, ptx)?;
        let _ = l3.shape().elem_count();
        let mut a = f.builder();
        let (bi, ni, tb, hi) = (b as i32, n as i32, self.topo_batched as i32, h as i32);
        a.arg(&dist)
            .arg(&topo)
            .arg(&params)
            .arg(&dst)
            .arg(&bi)
            .arg(&ni)
            .arg(&tb)
            .arg(&hi);
        cuda::launch(&mut a, cuda::grid(total, 128))?;
        Ok((cuda::wrap(dst, &dev), Shape::from((c / h, b, h, n, n))))
    }
}

/// Attention biases [L, B, H, N, N] from distances [B, N, N] and topological-distance ids
/// [bg, N, N] (u32, bg = 1 or B).
pub fn pair_bias(dist: &Tensor, topo: &Tensor, prm: &PairParams) -> Result<Tensor> {
    let (b, _, _) = dist.dims3()?;
    let bg = topo.dim(0)?;
    let op = PairBias {
        prm,
        topo_batched: bg == b && b > 1,
    };
    let topo = if bg == 1 || bg == b {
        topo.contiguous()?
    } else {
        candle_core::bail!("topology batch {bg} vs {b}")
    };
    dist.contiguous()?.apply_op3_no_bwd(&topo, &prm.packed, &op)
}

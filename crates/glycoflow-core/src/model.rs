//! `TorsionFlowNet` (glycoflow/model.py) on candle.
//!
//! Inference only. Two implementations of the same expressions: [`Ops::Fused`] (default) runs the
//! memory-bound parts through the custom kernels of [`crate::kernels`] (pair-bias MLP, layer norm +
//! adaLN modulation, bias + activation, gated residual, scale + bias + softmax); [`Ops::Composed`]
//! uses plain candle tensor ops only, all of which have a backward pass (for later autograd use).

use std::collections::HashMap;

use candle_core::{DType, Device, Tensor, D};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::kernels::{self, Act, PairParams};
use crate::topology::{Topology, Vocab, ELEMENTS};

/// Topological distance bins: 0..12 bonds, 13 = farther / padded (`model.TOPO_BINS`).
pub const TOPO_BINS: u32 = 14;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    pub n_elements: usize,
    pub n_atom_names: usize,
    pub n_residues: usize,
    pub n_link_codes: usize,
    pub dim: usize,
    pub depth: usize,
    pub heads: usize,
    pub pair_dim: usize,
    pub rbf_bins: usize,
    pub rbf_max: f64,
    pub ff_mult: usize,
    pub torsion_freqs: usize,
}

/// `weights/glycoflow.json` written by `scripts/export_rust_fixtures.py`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelMeta {
    pub config: ModelConfig,
    pub vocab: Vocab,
    #[serde(default)]
    pub elements: Vec<String>,
}

impl ModelMeta {
    pub fn from_json_str(s: &str) -> Result<Self> {
        let m: ModelMeta = serde_json::from_str(s)?;
        if !m.elements.is_empty()
            && m.elements
                .iter()
                .map(String::as_str)
                .ne(ELEMENTS.iter().copied())
        {
            return Err(Error::Invalid(format!(
                "checkpoint element table {:?} differs from {:?}",
                m.elements, ELEMENTS
            )));
        }
        Ok(m)
    }
}

/// Numerical precision of the network.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Precision {
    /// fp32 everywhere (the parity reference).
    F32,
    /// Linear layers and attention matmuls in bf16 (fp32 accumulate on CUDA), everything else fp32
    /// (close to PyTorch bf16 autocast; not bit-compatible with it).
    Bf16,
}

/// Kernel choices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Ops {
    /// custom fused kernels (inference only)
    Fused,
    /// candle tensor ops only (differentiable)
    Composed,
}

struct Linear {
    /// [in, out] (transposed, contiguous)
    wt: Tensor,
    wt_half: Option<Tensor>,
    b: Tensor,
}

impl Linear {
    fn load(w: &HashMap<String, Tensor>, name: &str, precision: Precision) -> Result<Self> {
        let weight = get(w, &format!("{name}.weight"))?;
        let wt = weight.t()?.contiguous()?;
        let wt_half = match precision {
            Precision::Bf16 => Some(wt.to_dtype(DType::BF16)?),
            Precision::F32 => None,
        };
        Ok(Self {
            wt,
            wt_half,
            b: get(w, &format!("{name}.bias"))?,
        })
    }

    fn forward(&self, x: &Tensor, ops: Ops) -> Result<Tensor> {
        self.forward_act(x, Act::None, ops)
    }

    /// x W^T for x [..., K] flattened to [M, K] -> [M, out] (no bias)
    fn matmul2(&self, x: &Tensor) -> Result<Tensor> {
        let dims = x.dims();
        let k = *dims.last().unwrap();
        let m: usize = dims[..dims.len() - 1].iter().product();
        let x2 = x.reshape((m, k))?;
        Ok(match &self.wt_half {
            Some(wh) => x2.to_dtype(DType::BF16)?.matmul(wh)?.to_dtype(DType::F32)?,
            None => x2.matmul(&self.wt)?,
        })
    }

    /// act(x W^T + b)
    fn forward_act(&self, x: &Tensor, act: Act, ops: Ops) -> Result<Tensor> {
        let dims = x.dims().to_vec();
        let y = self.matmul2(x)?;
        let y = match ops {
            Ops::Fused => kernels::bias_act(&y, &self.b, act)?,
            Ops::Composed => {
                let y = y.broadcast_add(&self.b)?;
                match act {
                    Act::None => y,
                    Act::Silu => y.silu()?,
                    Act::Gelu => y.gelu_erf()?,
                }
            }
        };
        let mut out = dims;
        *out.last_mut().unwrap() = self.b.dim(0)?;
        Ok(y.reshape(out)?)
    }
}

fn get(w: &HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    let t = w
        .get(name)
        .ok_or_else(|| Error::Invalid(format!("missing weight {name}")))?;
    Ok(t.to_dtype(DType::F32)?)
}

/// Two-pass layer norm over the last dim (eps 1e-5), optional affine.
fn layer_norm(x: &Tensor, affine: Option<(&Tensor, &Tensor)>) -> Result<Tensor> {
    let mean = x.mean_keepdim(D::Minus1)?;
    let xc = x.broadcast_sub(&mean)?;
    let var = xc.sqr()?.mean_keepdim(D::Minus1)?;
    let rstd = (var + 1e-5)?.sqrt()?.recip()?;
    let y = xc.broadcast_mul(&rstd)?;
    Ok(match affine {
        Some((g, b)) => y.broadcast_mul(g)?.broadcast_add(b)?,
        None => y,
    })
}

fn softmax_last(x: &Tensor) -> Result<Tensor> {
    let max = x.max_keepdim(D::Minus1)?;
    let e = x.broadcast_sub(&max)?.exp()?;
    let s = e.sum_keepdim(D::Minus1)?;
    Ok(e.broadcast_div(&s)?)
}

/// Cross product over the last dim (size 3).
fn cross(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let c = |t: &Tensor, i: usize| t.narrow(D::Minus1, i, 1);
    let (a0, a1, a2) = (c(a, 0)?, c(a, 1)?, c(a, 2)?);
    let (b0, b1, b2) = (c(b, 0)?, c(b, 1)?, c(b, 2)?);
    let x = (a1.broadcast_mul(&b2)? - a2.broadcast_mul(&b1)?)?;
    let y = (a2.broadcast_mul(&b0)? - a0.broadcast_mul(&b2)?)?;
    let z = (a0.broadcast_mul(&b1)? - a1.broadcast_mul(&b0)?)?;
    Ok(Tensor::cat(&[x, y, z], D::Minus1)?)
}

/// Pairwise distances [B,N,N] as `torch.cdist(x, x)`: for N > 25 PyTorch evaluates
/// sqrt(clamp(|x|^2 + |y|^2 - 2 x.y, 1e-30)) as one K=5 matmul `[-2x, |x|^2, 1] @ [y, 1, |y|^2]^T`
/// (the rounding of that cancellation shifts the fp32 velocities by ~1e-3, so it is mirrored);
/// otherwise the direct difference norm.
fn cdist(x: &Tensor) -> Result<Tensor> {
    let (b, n, _) = x.dims3()?;
    if n <= 25 {
        let diff = x.unsqueeze(2)?.broadcast_sub(&x.unsqueeze(1)?)?;
        return Ok(diff.sqr()?.sum(D::Minus1)?.sqrt()?);
    }
    let norm = x.sqr()?.sum_keepdim(D::Minus1)?; // [B,N,1]
    let ones = Tensor::ones((b, n, 1), x.dtype(), x.device())?;
    let lhs = Tensor::cat(&[(x * -2.0)?, norm.clone(), ones.clone()], D::Minus1)?;
    let rhs = Tensor::cat(&[x.clone(), ones, norm], D::Minus1)?;
    let d2 = lhs.matmul(&rhs.t()?)?;
    Ok(d2.clamp(1e-30f32, f32::INFINITY)?.sqrt()?)
}

struct Block {
    heads: usize,
    ada: Linear,
    qkv: Linear,
    out: Linear,
    ff0: Linear,
    ff2: Linear,
}

impl Block {
    /// h [B,N,D], temb [B,D], bias [B,H,N,N] -> (h, attention probabilities if requested).
    /// Attention logits are `q.k * (1/sqrt(dh))` in all but the last block and `q.k / sqrt(dh)` in
    /// the last one, as PyTorch evaluates `scaled_dot_product_attention` and the explicit last block.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        h: &Tensor,
        temb: &Tensor,
        bias: &Tensor,
        last: bool,
        want_attn: bool,
        precision: Precision,
        ops: Ops,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let (b, n, d) = h.dims3()?;
        let hd = d / self.heads;
        let ada = self.ada.forward(temb, ops)?; // [B,6D]: s1, b1, g1, s2, b2, g2
        let x = match ops {
            Ops::Fused => kernels::ln_modulate(h, &ada, 0, 1)?,
            Ops::Composed => {
                let ch = ada.unsqueeze(1)?.chunk(6, D::Minus1)?;
                layer_norm(h, None)?
                    .broadcast_mul(&(&ch[0] + 1.0)?)?
                    .broadcast_add(&ch[1])?
            }
        };
        let qkv = match ops {
            Ops::Fused => {
                kernels::bias_split_heads(&self.qkv.matmul2(&x)?, &self.qkv.b, n, self.heads)?
            }
            Ops::Composed => self
                .qkv
                .forward(&x, ops)?
                .reshape((b, n, 3, self.heads, hd))?
                .permute((2, 0, 3, 1, 4))?,
        };
        let mut q = qkv.get(0)?.contiguous()?;
        let mut k = qkv.get(1)?.contiguous()?;
        let mut v = qkv.get(2)?.contiguous()?;
        if precision == Precision::Bf16 {
            q = q.to_dtype(DType::BF16)?;
            k = k.to_dtype(DType::BF16)?;
            v = v.to_dtype(DType::BF16)?;
        }
        let qk = q.matmul(&k.t()?)?.to_dtype(DType::F32)?;
        let sqrt_dh = (hd as f64).sqrt();
        let attn = match ops {
            Ops::Fused => {
                if last {
                    kernels::scale_bias_softmax(&qk, bias, sqrt_dh as f32, true)?
                } else {
                    kernels::scale_bias_softmax(&qk, bias, (1.0 / sqrt_dh) as f32, false)?
                }
            }
            Ops::Composed => {
                let logits = if last {
                    qk.broadcast_div(&Tensor::new(&[sqrt_dh as f32], h.device())?)?
                } else {
                    qk.broadcast_mul(&Tensor::new(&[(1.0 / sqrt_dh) as f32], h.device())?)?
                };
                softmax_last(&logits.broadcast_add(bias)?)?
            }
        };
        let o = attn.to_dtype(v.dtype())?.matmul(&v)?.to_dtype(DType::F32)?;
        let o = o.transpose(1, 2)?.reshape((b, n, d))?;
        let o = self.out.forward(&o, ops)?;
        let h = match ops {
            Ops::Fused => {
                let h = kernels::gated_add(h, &o, &ada, 2)?;
                let x = kernels::ln_modulate(&h, &ada, 3, 4)?;
                let f = self
                    .ff2
                    .forward(&self.ff0.forward_act(&x, Act::Gelu, ops)?, ops)?;
                kernels::gated_add(&h, &f, &ada, 5)?
            }
            Ops::Composed => {
                let ch = ada.unsqueeze(1)?.chunk(6, D::Minus1)?;
                let h = (h + o.broadcast_mul(&ch[2])?)?;
                let x = layer_norm(&h, None)?
                    .broadcast_mul(&(&ch[3] + 1.0)?)?
                    .broadcast_add(&ch[4])?;
                let f = self
                    .ff2
                    .forward(&self.ff0.forward_act(&x, Act::Gelu, ops)?, ops)?;
                (h + f.broadcast_mul(&ch[5])?)?
            }
        };
        Ok((h, if want_attn { Some(attn) } else { None }))
    }
}

/// Device tensors describing the graph(s) of a batch. The batch dimension `bg` is either 1 (one
/// glycan shared by every conformer of the batch; broadcast) or the batch size (padded
/// multi-glycan batch, as `dataset.Bucket`).
pub struct GraphBatch {
    pub bg: usize,
    /// some graph has fewer atoms than the batch (key padding needed)
    pub has_padding: bool,
    pub n_atoms: usize,
    pub n_torsions: usize,
    tokens: Tensor,     // [bg,N,5] u32
    topo: Tensor,       // [bg,N,N] u32 (clamped to TOPO_BINS-1)
    sel: Vec<Tensor>,   // 4 x [bg,T,N] one-hot of quad column
    sel_t: Vec<Tensor>, // 2 x [bg,N,T] (columns 1, 2) for the scatter onto bond atoms
    tmask: Tensor,      // [bg,T]
    distal: Tensor,     // [bg,T,N]
    amask: Tensor,      // [bg,N]
    key_pad: Tensor,    // [bg,1,1,N] (0 / -inf)
}

/// Host-side description of one glycan for [`GraphBatch::new`].
pub struct GraphInput<'a> {
    pub tokens: &'a [[u32; 5]],
    pub topology: &'a Topology,
}

impl GraphBatch {
    /// One glycan shared by all conformers of a batch.
    pub fn single(tokens: &[[u32; 5]], topology: &Topology, device: &Device) -> Result<Self> {
        Self::new(&[GraphInput { tokens, topology }], device)
    }

    /// Padded batch (one graph per conformer, or a single graph broadcast over the batch).
    pub fn new(graphs: &[GraphInput], device: &Device) -> Result<Self> {
        let bg = graphs.len();
        if bg == 0 {
            return Err(Error::Invalid("empty graph batch".into()));
        }
        let n = graphs.iter().map(|g| g.topology.n_atoms).max().unwrap();
        let t = graphs
            .iter()
            .map(|g| g.topology.n_torsions())
            .max()
            .unwrap()
            .max(1);
        let mut tokens = vec![0u32; bg * n * 5];
        let mut topo = vec![TOPO_BINS - 1; bg * n * n];
        let mut sel = vec![vec![0f32; bg * t * n]; 4];
        let mut tmask = vec![0f32; bg * t];
        let mut distal = vec![0f32; bg * t * n];
        let mut amask = vec![0f32; bg * n];
        let mut key_pad = vec![f32::NEG_INFINITY; bg * n];
        for (gi, g) in graphs.iter().enumerate() {
            let topo_g = g.topology;
            let ng = topo_g.n_atoms;
            if g.tokens.len() != ng {
                return Err(Error::Invalid("token count != atom count".into()));
            }
            for i in 0..ng {
                tokens[(gi * n + i) * 5..(gi * n + i) * 5 + 5].copy_from_slice(&g.tokens[i]);
                amask[gi * n + i] = 1.0;
                key_pad[gi * n + i] = 0.0;
                for j in 0..ng {
                    topo[(gi * n + i) * n + j] =
                        (topo_g.topo_dist[i * ng + j] as u32).min(TOPO_BINS - 1);
                }
            }
            for (k, q) in topo_g.quads.iter().enumerate() {
                tmask[gi * t + k] = 1.0;
                for col in 0..4 {
                    sel[col][(gi * t + k) * n + q[col]] = 1.0;
                }
                for &i in &topo_g.distal[k] {
                    distal[(gi * t + k) * n + i] = 1.0;
                }
            }
        }
        let sel_t: Vec<Tensor> = [1usize, 2]
            .iter()
            .map(|&col| {
                let mut s = vec![0f32; bg * n * t];
                for gi in 0..bg {
                    for k in 0..t {
                        for i in 0..n {
                            s[(gi * n + i) * t + k] = sel[col][(gi * t + k) * n + i];
                        }
                    }
                }
                Tensor::from_vec(s, (bg, n, t), device)
            })
            .collect::<candle_core::Result<_>>()?;
        Ok(Self {
            bg,
            has_padding: graphs.iter().any(|g| g.topology.n_atoms < n),
            n_atoms: n,
            n_torsions: t,
            tokens: Tensor::from_vec(tokens, (bg, n, 5), device)?,
            topo: Tensor::from_vec(topo, (bg, n, n), device)?,
            sel: sel
                .into_iter()
                .map(|s| Tensor::from_vec(s, (bg, t, n), device))
                .collect::<candle_core::Result<_>>()?,
            sel_t,
            tmask: Tensor::from_vec(tmask, (bg, t), device)?,
            distal: Tensor::from_vec(distal, (bg, t, n), device)?,
            amask: Tensor::from_vec(amask, (bg, n), device)?,
            key_pad: Tensor::from_vec(key_pad, (bg, 1, 1, n), device)?,
        })
    }
}

/// Per-graph quantities that do not depend on the conformer or time (token embeddings, topology
/// pair embeddings).
pub struct Prepared {
    pub graph: GraphBatch,
    h_tok: Tensor,    // [bg,N,D]
    topo_emb: Tensor, // [bg,N,N,P]
}

pub struct TorsionFlowNet {
    pub cfg: ModelConfig,
    pub precision: Precision,
    pub ops: Ops,
    device: Device,
    emb_el: Tensor,
    emb_name: Tensor,
    emb_res: Tensor,
    emb_link: Tensor,
    emb_ring: Tensor,
    emb_topo: Tensor,
    pair_in0: Linear,
    pair_in2: Linear,
    pair_to_bias: Linear,
    torsion_in0: Linear,
    torsion_in2: Linear,
    time0: Linear,
    time2: Linear,
    blocks: Vec<Block>,
    norm_out_w: Tensor,
    norm_out_b: Tensor,
    head0: Linear,
    head2: Linear,
    head4: Linear,
    force_value: Linear,
    force_mix: Linear,
    pair_params: PairParams,
    rbf_centers: Tensor, // [R]
    rbf_width: Tensor,   // [1]
    time_freqs: Tensor,  // [1, D/2]
    torsion_k: Tensor,   // [F]
}

fn emb(table: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let mut shape = ids.dims().to_vec();
    shape.push(table.dim(1)?);
    Ok(table.index_select(&ids.flatten_all()?, 0)?.reshape(shape)?)
}

impl TorsionFlowNet {
    /// Load from safetensors bytes (PyTorch parameter names) and the model config.
    pub fn from_safetensors(
        bytes: &[u8],
        cfg: ModelConfig,
        device: &Device,
        precision: Precision,
    ) -> Result<Self> {
        let w = candle_core::safetensors::load_buffer(bytes, device)?;
        Self::from_tensors(&w, cfg, device, precision)
    }

    pub fn from_tensors(
        w: &HashMap<String, Tensor>,
        cfg: ModelConfig,
        device: &Device,
        precision: Precision,
    ) -> Result<Self> {
        let p = precision;
        let lin = |name: &str| Linear::load(w, name, p);
        let blocks = (0..cfg.depth)
            .map(|i| {
                Ok(Block {
                    heads: cfg.heads,
                    ada: lin(&format!("blocks.{i}.ada"))?,
                    qkv: lin(&format!("blocks.{i}.qkv"))?,
                    out: lin(&format!("blocks.{i}.out"))?,
                    ff0: lin(&format!("blocks.{i}.ff.0"))?,
                    ff2: lin(&format!("blocks.{i}.ff.2"))?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        // torch.linspace(0, rbf_max, bins) in f32: start + step*i below halfway, end - step*(n-1-i) above
        let bins = cfg.rbf_bins;
        let end = cfg.rbf_max as f32;
        let step = end / (bins as f32 - 1.0);
        let centers: Vec<f32> = (0..bins)
            .map(|i| {
                if i < bins / 2 {
                    step * i as f32
                } else {
                    end - step * (bins - 1 - i) as f32
                }
            })
            .collect();
        let half = cfg.dim / 2;
        let neg_log = -(1000f64.ln()) as f32;
        let freqs: Vec<f32> = (0..half)
            .map(|k| (neg_log * k as f32 / half as f32).exp())
            .collect();
        let torsion_k: Vec<f32> = (1..=cfg.torsion_freqs).map(|k| k as f32).collect();
        let mut packed: Vec<f32> = centers.clone();
        packed.push((cfg.rbf_max / bins as f64) as f32);
        for name in [
            "pair_in.0.weight",
            "pair_in.0.bias",
            "pair_in.2.weight",
            "pair_in.2.bias",
            "emb_topo.weight",
            "pair_to_bias.weight",
            "pair_to_bias.bias",
        ] {
            packed.extend(get(w, name)?.flatten_all()?.to_vec1::<f32>()?);
        }
        let pair_params = PairParams {
            r: bins,
            p: cfg.pair_dim,
            c: cfg.depth * cfg.heads,
            heads: cfg.heads,
            packed: Tensor::from_vec(packed.clone(), packed.len(), device)?,
        };
        Ok(Self {
            pair_params,
            emb_el: get(w, "emb_el.weight")?,
            emb_name: get(w, "emb_name.weight")?,
            emb_res: get(w, "emb_res.weight")?,
            emb_link: get(w, "emb_link.weight")?,
            emb_ring: get(w, "emb_ring.weight")?,
            emb_topo: get(w, "emb_topo.weight")?,
            pair_in0: lin("pair_in.0")?,
            pair_in2: lin("pair_in.2")?,
            pair_to_bias: lin("pair_to_bias")?,
            torsion_in0: lin("torsion_in.0")?,
            torsion_in2: lin("torsion_in.2")?,
            time0: lin("time.mlp.0")?,
            time2: lin("time.mlp.2")?,
            blocks,
            norm_out_w: get(w, "norm_out.weight")?,
            norm_out_b: get(w, "norm_out.bias")?,
            head0: lin("head_inv.0")?,
            head2: lin("head_inv.2")?,
            head4: lin("head_inv.4")?,
            force_value: lin("force_value")?,
            force_mix: lin("force_mix")?,
            rbf_centers: Tensor::from_vec(centers, bins, device)?,
            rbf_width: Tensor::new(&[(cfg.rbf_max / bins as f64) as f32], device)?,
            time_freqs: Tensor::from_vec(freqs, (1, half), device)?,
            torsion_k: Tensor::from_vec(torsion_k, cfg.torsion_freqs, device)?,
            cfg,
            precision,
            ops: Ops::Fused,
            device: device.clone(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Token and topology embeddings of a graph batch (computed once per glycan).
    pub fn prepare(&self, graph: GraphBatch) -> Result<Prepared> {
        let tok = &graph.tokens;
        let col =
            |c: usize| -> Result<Tensor> { Ok(tok.narrow(2, c, 1)?.squeeze(2)?.contiguous()?) };
        let h_tok = ((((emb(&self.emb_el, &col(0)?)? + emb(&self.emb_name, &col(1)?)?)?
            + emb(&self.emb_res, &col(2)?)?)?
            + emb(&self.emb_link, &col(3)?)?)?
            + emb(&self.emb_ring, &col(4)?)?)?;
        let topo_emb = emb(&self.emb_topo, &graph.topo)?;
        Ok(Prepared {
            graph,
            h_tok,
            topo_emb,
        })
    }

    fn time_embed(&self, t: &Tensor) -> Result<Tensor> {
        let ang = (t.unsqueeze(1)?.broadcast_mul(&self.time_freqs)? * 1000.0)?;
        let x = Tensor::cat(&[ang.sin()?, ang.cos()?], D::Minus1)?;
        self.time2
            .forward(&self.time0.forward_act(&x, Act::Silu, self.ops)?, self.ops)
    }

    /// Velocities [B,T] for conformers `coords` [B,N,3], current torsions `tors` [B,T] and times
    /// `t` [B] (all f32 on the model device). `prep.graph.bg` must be 1 or B.
    pub fn forward(
        &self,
        prep: &Prepared,
        coords: &Tensor,
        tors: &Tensor,
        t: &Tensor,
    ) -> Result<Tensor> {
        let g = &prep.graph;
        let (b, n, _) = coords.dims3()?;
        let tt = tors.dim(1)?;
        if n != g.n_atoms || tt != g.n_torsions || (g.bg != 1 && g.bg != b) {
            return Err(Error::Invalid(format!(
                "forward: coords [{b},{n}], tors [_, {tt}] do not match graph batch [{}, {}, {}]",
                g.bg, g.n_atoms, g.n_torsions
            )));
        }
        let cfg = &self.cfg;
        let hh = cfg.heads;

        // torsion Fourier features scattered onto the two bond atoms
        let ang = tors.unsqueeze(D::Minus1)?.broadcast_mul(&self.torsion_k)?; // [B,T,F]
        let trig = Tensor::stack(&[ang.sin()?, ang.cos()?], D::Minus1)?.flatten_from(2)?; // sin1,cos1,sin2,cos2
        let ops = self.ops;
        let tfeat = self
            .torsion_in2
            .forward(&self.torsion_in0.forward_act(&trig, Act::Silu, ops)?, ops)?
            .broadcast_mul(&g.tmask.unsqueeze(D::Minus1)?)?;
        let h = prep
            .h_tok
            .broadcast_add(&g.sel_t[0].broadcast_matmul(&tfeat)?)?;
        let mut h = (h + g.sel_t[1].broadcast_matmul(&tfeat)?)?;
        let temb = self.time_embed(t)?;

        // pair bias from distances and topology
        let dist = cdist(coords)?;
        // biases [L, B, H, N, N] (layer-major, so each block's slice is contiguous)
        let biases = match ops {
            Ops::Fused => kernels::pair_bias(&dist, &g.topo, &self.pair_params)?,
            Ops::Composed => {
                let rbf = dist
                    .unsqueeze(D::Minus1)?
                    .broadcast_sub(&self.rbf_centers)?
                    .broadcast_div(&self.rbf_width)?
                    .sqr()?
                    .neg()?
                    .exp()?;
                let pair = self
                    .pair_in2
                    .forward(&self.pair_in0.forward_act(&rbf, Act::Silu, ops)?, ops)?
                    .broadcast_add(&prep.topo_emb)?;
                self.pair_to_bias
                    .forward(&pair, ops)? // [B,N,N,L*H]
                    .reshape((b, n, n, cfg.depth, hh))?
                    .permute((3, 0, 4, 1, 2))?
                    .contiguous()?
            }
        };

        let mut attn = None;
        for (li, block) in self.blocks.iter().enumerate() {
            let mut bias = biases.get(li)?;
            if g.has_padding {
                bias = bias.broadcast_add(&g.key_pad)?;
            }
            let last = li + 1 == self.blocks.len();
            let (h2, a) = block.forward(&h, &temb, &bias, last, last, self.precision, ops)?;
            h = h2;
            if last {
                attn = a;
            }
        }
        let h = match ops {
            Ops::Fused => kernels::ln_affine(&h, &self.norm_out_w, &self.norm_out_b)?,
            Ops::Composed => layer_norm(&h, Some((&self.norm_out_w, &self.norm_out_b)))?,
        };

        // invariant head
        let gath = |c: usize| g.sel[c].broadcast_matmul(&h);
        let inv_in = Tensor::cat(&[gath(0)?, gath(1)?, gath(2)?, gath(3)?, trig], D::Minus1)?;
        let out = self.head4.forward(
            &self.head2.forward_act(
                &self.head0.forward_act(&inv_in, Act::Silu, ops)?,
                Act::Silu,
                ops,
            )?,
            ops,
        )?;
        let v_inv = out.narrow(D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
        let gate = out.narrow(D::Minus1, 1, 1)?.squeeze(D::Minus1)?;

        // torque head: f_i = sum_h c_ih sum_j A_hij s_hj (x_j - x_i)
        let x = coords;
        let attn = attn.ok_or_else(|| Error::Invalid("model has no blocks".into()))?;
        let s = self
            .force_value
            .forward(&h, ops)?
            .transpose(1, 2)?
            .unsqueeze(2)?; // [B,H,1,N]
        let w = attn.broadcast_mul(&s)?; // [B,H,N,N]
        let xh = x.unsqueeze(1)?; // [B,1,N,3]
        let f_h = (w.broadcast_matmul(&xh)? - w.sum_keepdim(D::Minus1)?.broadcast_mul(&xh)?)?; // [B,H,N,3]
        let c = self
            .force_mix
            .forward(&h, ops)?
            .transpose(1, 2)?
            .unsqueeze(D::Minus1)?; // [B,H,N,1]
        let force = c
            .broadcast_mul(&f_h)?
            .sum(1)?
            .broadcast_mul(&g.amask.unsqueeze(D::Minus1)?)?; // [B,N,3]

        let m = &g.distal; // [bg,T,N]
        let xb = g.sel[1].broadcast_matmul(x)?;
        let xc = g.sel[2].broadcast_matmul(x)?;
        let bc = (xc - &xb)?;
        let u = bc.broadcast_div(
            &bc.sqr()?
                .sum_keepdim(D::Minus1)?
                .sqrt()?
                .clamp(1e-12f32, f32::INFINITY)?,
        )?;
        let sum_f = m.broadcast_matmul(&force)?;
        let sum_xf = m.broadcast_matmul(&cross(x, &force)?)?;
        let torque = (&u * (sum_xf - cross(&xb, &sum_f)?)?)?.sum(D::Minus1)?; // [B,T]
        let cnt = m.sum_keepdim(D::Minus1)?; // [bg,T,1]
        let mean_x = m
            .broadcast_matmul(x)?
            .broadcast_div(&cnt.clamp(1f32, f32::INFINITY)?)?; // [B,T,3]
        let xx = x
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&x.unsqueeze(2)?)?
            .reshape((b, n, 9))?;
        let second = m.broadcast_matmul(&xx)?.reshape((b, tt, 3, 3))?;
        let r_mean = (&mean_x - &xb)?;
        let outer = |a: &Tensor| -> Result<Tensor> {
            Ok(a.unsqueeze(D::Minus1)?.broadcast_mul(&a.unsqueeze(2)?)?)
        };
        let cnt4 = cnt.unsqueeze(D::Minus1)?;
        let s_mat = (second - outer(&mean_x)?.broadcast_mul(&cnt4)?)?
            .add(&outer(&r_mean)?.broadcast_mul(&cnt4)?)?;
        let trace = s_mat
            .reshape((b, tt, 9))?
            .narrow(D::Minus1, 0, 1)?
            .squeeze(D::Minus1)?;
        let trace = ((trace
            + s_mat
                .reshape((b, tt, 9))?
                .narrow(D::Minus1, 4, 1)?
                .squeeze(D::Minus1)?)?
            + s_mat
                .reshape((b, tt, 9))?
                .narrow(D::Minus1, 8, 1)?
                .squeeze(D::Minus1)?)?;
        let usu = u
            .unsqueeze(D::Minus1)?
            .broadcast_mul(&s_mat)?
            .sum(2)?
            .mul(&u)?
            .sum(D::Minus1)?;
        let jnorm2 = (trace - usu)?;
        let torque = torque.div(&((jnorm2.clamp(0f32, f32::INFINITY)? + 1.0)?.sqrt()?))?;

        let v = (v_inv + gate.mul(&torque)?)?;
        Ok(v.broadcast_mul(&g.tmask)?)
    }
}

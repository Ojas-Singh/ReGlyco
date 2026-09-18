//! Generic CCP4/MRC density loading and masked map agreement scoring.
//!
//! The MRC reader is deliberately kept separate from the fitting workflow:
//! this crate knows how to put a molecular model and a scalar map in the same
//! coordinate system, while `reglyco-refine` decides how poses are searched.

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use glysys::{AtomId, ResidueId, Structure};
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;
use sha2::{Digest, Sha256};

pub mod fast_score;
#[cfg(not(target_arch = "wasm32"))]
pub mod rcsb;

pub type Result<T> = std::result::Result<T, DensityError>;

#[derive(Debug, thiserror::Error)]
pub enum DensityError {
    #[error("density map I/O failed for {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("density map could not be read: {0}")]
    Map(String),
    #[error("density map has unsupported complex voxel mode")]
    ComplexMap,
    #[error("density map has invalid geometry: {0}")]
    Geometry(String),
    #[error("density map contains no finite voxels")]
    EmptyMap,
    #[error("density target site {0} does not identify a glycan")]
    TargetNotFound(ResidueId),
    #[error("density target has no atoms")]
    EmptyTarget,
    #[error("density target falls outside the map")]
    OutOfMap,
    #[error("density correlation is undefined because the sampled values are constant")]
    ConstantSample,
}

/// A target glycan and the protein residue to which it is attached.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DensityTarget {
    pub site: ResidueId,
    pub glycan_residues: Vec<ResidueId>,
}

impl DensityTarget {
    /// Resolve a target from the glycan-tree metadata retained by GlySys.
    pub fn for_site(structure: &Structure, site: &ResidueId) -> Result<Self> {
        let tree = structure
            .metadata()
            .glycan_trees
            .iter()
            .find(|tree| tree.attachment_site.as_ref() == Some(site))
            .ok_or_else(|| DensityError::TargetNotFound(site.clone()))?;
        if tree.residue_ids.is_empty() {
            return Err(DensityError::EmptyTarget);
        }
        Ok(Self {
            site: site.clone(),
            glycan_residues: tree.residue_ids.clone(),
        })
    }

    /// Resolve every attachment represented in structure metadata.
    pub fn all(structure: &Structure) -> Vec<Self> {
        structure
            .metadata()
            .glycan_trees
            .iter()
            .filter_map(|tree| {
                tree.attachment_site.as_ref().map(|site| Self {
                    site: site.clone(),
                    glycan_residues: tree.residue_ids.clone(),
                })
            })
            .collect()
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct DensityScoreOptions {
    /// Gaussian width in Å. If absent, callers must provide a resolution.
    pub sigma_angstrom: Option<f64>,
    /// Fixed fallback B factor for generated glycan atoms whose input record
    /// has no crystallographic B value. `None` means no additional broadening.
    pub glycan_b_factor: Option<f64>,
    /// Radius of the high-weight mask core in Å.
    pub mask_radius_angstrom: f64,
    /// Width of the cosine mask falloff in Å.
    pub mask_falloff_angstrom: f64,
    /// Apply periodic wrapping when a sample crosses a full unit-cell map.
    pub periodic: bool,
    /// Evidence z-score required for a residue to be considered supported.
    pub support_threshold: f64,
}

impl Default for DensityScoreOptions {
    fn default() -> Self {
        Self {
            sigma_angstrom: None,
            glycan_b_factor: None,
            mask_radius_angstrom: 2.0,
            mask_falloff_angstrom: 1.0,
            periodic: false,
            support_threshold: 0.60,
        }
    }
}

impl DensityScoreOptions {
    pub fn with_resolution(mut self, resolution_angstrom: f64) -> Result<Self> {
        if !resolution_angstrom.is_finite() || resolution_angstrom <= 0.0 {
            return Err(DensityError::Geometry(
                "resolution must be a positive finite value".into(),
            ));
        }
        self.sigma_angstrom = Some(resolution_angstrom / (2.0 * (2.0_f64.ln()).sqrt()));
        Ok(self)
    }

    fn sigma(self) -> Result<f64> {
        let sigma = self.sigma_angstrom.ok_or_else(|| {
            DensityError::Geometry(
                "density sigma is required when no map resolution is available".into(),
            )
        })?;
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(DensityError::Geometry(
                "density sigma must be a positive finite value".into(),
            ));
        }
        if self.mask_radius_angstrom <= 0.0
            || self.mask_falloff_angstrom < 0.0
            || !self.mask_radius_angstrom.is_finite()
            || !self.mask_falloff_angstrom.is_finite()
        {
            return Err(DensityError::Geometry(
                "mask radii must be finite and non-negative".into(),
            ));
        }
        if !self.support_threshold.is_finite() || self.support_threshold < 0.0 {
            return Err(DensityError::Geometry(
                "density support threshold must be finite and non-negative".into(),
            ));
        }
        Ok(sigma)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DensityMapMetadata {
    pub path: PathBuf,
    pub sha256: String,
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub mode: i32,
    pub nxstart: i32,
    pub nystart: i32,
    pub nzstart: i32,
    pub sampling: [usize; 3],
    pub cell_lengths_angstrom: [f64; 3],
    pub cell_angles_degrees: [f64; 3],
    /// Raw one-based MRC `MAPC/MAPR/MAPS` values (column, row, section).
    pub map_axes: [usize; 3],
    pub origin_angstrom: [f64; 3],
    pub space_group: i32,
    pub dmin: f64,
    pub dmax: f64,
    pub dmean: f64,
    pub rms: f64,
    pub warnings: Vec<String>,
    /// Acquisition provenance.  These fields are optional so maps produced
    /// by older ReGlyco versions and user supplied CCP4 files remain
    /// backwards compatible.
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub channel: Option<String>,
    #[serde(default)]
    pub source_url: Option<String>,
    #[serde(default)]
    pub raw_sha256: Option<String>,
    #[serde(default)]
    pub raw_cache_path: Option<PathBuf>,
    #[serde(default)]
    pub map_detail: Option<u8>,
}

#[derive(Debug, Clone)]
pub struct DensityMap {
    metadata: DensityMapMetadata,
    data: Vec<f32>,
    cartesian_to_fractional: Matrix3<f64>,
    fractional_to_cartesian: Matrix3<f64>,
}

impl DensityMap {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let (_, warnings) = mrc::Reader::open_permissive(&path)
            .map_err(|error| DensityError::Map(error.to_string()))?;
        let validation = mrc::validate_full(&path, false)
            .map_err(|error| DensityError::Map(error.to_string()))?;
        if !validation.is_valid() {
            let issues = validation
                .issues
                .iter()
                .map(|issue| issue.message.as_str())
                .collect::<Vec<_>>()
                .join("; ");
            return Err(DensityError::Map(format!(
                "map validation failed: {issues}"
            )));
        }
        let reader =
            mrc::Reader::open(&path).map_err(|error| DensityError::Map(error.to_string()))?;
        let mode = reader.mode();
        if mode.is_complex() {
            return Err(DensityError::ComplexMap);
        }
        let header = reader.header().clone();
        let dimensions = [header.nx, header.ny, header.nz];
        if dimensions.iter().any(|value| *value <= 0) {
            return Err(DensityError::Geometry(
                "map dimensions must be positive".into(),
            ));
        }
        let sampling = [header.mx, header.my, header.mz];
        if sampling.iter().any(|value| *value <= 0) {
            return Err(DensityError::Geometry(
                "unit-cell sampling must be positive".into(),
            ));
        }
        let map_axes = [header.mapc, header.mapr, header.maps];
        if map_axes.iter().any(|axis| !(1..=3).contains(axis))
            || map_axes[0] == map_axes[1]
            || map_axes[0] == map_axes[2]
            || map_axes[1] == map_axes[2]
        {
            return Err(DensityError::Geometry(
                "MAPC/MAPR/MAPS must be a permutation of 1,2,3".into(),
            ));
        }
        let fractional_to_cartesian = cell_matrix(
            [header.xlen as f64, header.ylen as f64, header.zlen as f64],
            [header.alpha as f64, header.beta as f64, header.gamma as f64],
        )?;
        let cartesian_to_fractional = fractional_to_cartesian
            .try_inverse()
            .ok_or_else(|| DensityError::Geometry("unit cell matrix is singular".into()))?;
        let block = reader
            .convert::<f32>()
            .read_volume()
            .map_err(|error| DensityError::Map(error.to_string()))?;
        let data = block.data;
        if data.is_empty() || data.iter().any(|value| !value.is_finite()) {
            return Err(DensityError::EmptyMap);
        }
        // A surprising number of otherwise readable CCP4 files carry zero or
        // stale statistics in the header (especially crops written by older
        // map tools).  Keep the deposited values when they are usable, but
        // derive a deterministic fallback from the actual finite voxels so
        // normalization and support diagnostics do not silently collapse.
        let data_stats = voxel_statistics(&data);
        let header_stats = [
            header.dmin as f64,
            header.dmax as f64,
            header.dmean as f64,
            header.rms as f64,
        ];
        let [dmin, dmax, dmean, rms] = if header_stats.iter().all(|value| value.is_finite())
            && header_stats[0] <= header_stats[1]
            && header_stats[3] > 1.0e-12
        {
            header_stats
        } else {
            data_stats
        };
        let sha256 = sha256_file(&path)?;
        Ok(Self {
            metadata: DensityMapMetadata {
                path,
                sha256,
                nx: dimensions[0] as usize,
                ny: dimensions[1] as usize,
                nz: dimensions[2] as usize,
                mode: header.mode,
                nxstart: header.nxstart,
                nystart: header.nystart,
                nzstart: header.nzstart,
                sampling: [
                    sampling[0] as usize,
                    sampling[1] as usize,
                    sampling[2] as usize,
                ],
                cell_lengths_angstrom: [header.xlen as f64, header.ylen as f64, header.zlen as f64],
                cell_angles_degrees: [header.alpha as f64, header.beta as f64, header.gamma as f64],
                map_axes: [
                    map_axes[0] as usize,
                    map_axes[1] as usize,
                    map_axes[2] as usize,
                ],
                origin_angstrom: [
                    header.origin[0] as f64,
                    header.origin[1] as f64,
                    header.origin[2] as f64,
                ],
                space_group: header.ispg,
                dmin,
                dmax,
                dmean,
                rms,
                warnings,
                source: None,
                channel: None,
                source_url: None,
                raw_sha256: None,
                raw_cache_path: None,
                map_detail: None,
            },
            data,
            cartesian_to_fractional,
            fractional_to_cartesian,
        })
    }

    /// Decode a CCP4/MRC map directly from an uploaded or fetched byte
    /// buffer. Browser workflows use this constructor so map fitting never
    /// needs a temporary file or memory mapping.
    pub fn from_bytes(label: impl Into<PathBuf>, bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 1024 {
            return Err(DensityError::Map(
                "CCP4/MRC header is shorter than 1024 bytes".into(),
            ));
        }
        let little = plausible_mrc_header(bytes, true);
        let big = plausible_mrc_header(bytes, false);
        let is_little = match (little, big) {
            (true, false) => true,
            (false, true) => false,
            (true, true) => bytes.get(212).is_none_or(|value| *value != 0x11),
            (false, false) => {
                return Err(DensityError::Map(
                    "invalid CCP4/MRC dimensions or voxel mode".into(),
                ));
            }
        };
        let i32_at = |offset| read_i32(bytes, offset, is_little);
        let f32_at = |offset| f32::from_bits(read_u32(bytes, offset, is_little));
        let dimensions = [i32_at(0), i32_at(4), i32_at(8)];
        let mode = i32_at(12);
        let sampling = [i32_at(28), i32_at(32), i32_at(36)];
        if sampling.iter().any(|value| *value <= 0) {
            return Err(DensityError::Geometry(
                "unit-cell sampling must be positive".into(),
            ));
        }
        let map_axes = [i32_at(64), i32_at(68), i32_at(72)];
        if map_axes.iter().any(|axis| !(1..=3).contains(axis))
            || map_axes[0] == map_axes[1]
            || map_axes[0] == map_axes[2]
            || map_axes[1] == map_axes[2]
        {
            return Err(DensityError::Geometry(
                "MAPC/MAPR/MAPS must be a permutation of 1,2,3".into(),
            ));
        }
        let cell_lengths = [f32_at(40) as f64, f32_at(44) as f64, f32_at(48) as f64];
        let cell_angles = [f32_at(52) as f64, f32_at(56) as f64, f32_at(60) as f64];
        let fractional_to_cartesian = cell_matrix(cell_lengths, cell_angles)?;
        let cartesian_to_fractional = fractional_to_cartesian
            .try_inverse()
            .ok_or_else(|| DensityError::Geometry("unit cell matrix is singular".into()))?;
        let voxel_count = dimensions
            .iter()
            .try_fold(1usize, |count, value| count.checked_mul(*value as usize))
            .ok_or_else(|| {
                DensityError::Geometry("map dimensions overflow addressable memory".into())
            })?;
        let bytes_per_voxel = match mode {
            0 => 1,
            1 | 6 => 2,
            2 => 4,
            3 | 4 => return Err(DensityError::ComplexMap),
            _ => {
                return Err(DensityError::Map(format!(
                    "unsupported CCP4/MRC voxel mode {mode}"
                )));
            }
        };
        let extended = i32_at(92);
        if extended < 0 {
            return Err(DensityError::Geometry(
                "negative extended-header size".into(),
            ));
        }
        let data_offset = 1024usize + extended as usize;
        let data_bytes = voxel_count
            .checked_mul(bytes_per_voxel)
            .ok_or_else(|| DensityError::Geometry("map byte count overflow".into()))?;
        if data_offset
            .checked_add(data_bytes)
            .is_none_or(|end| end > bytes.len())
        {
            return Err(DensityError::Map("CCP4/MRC voxel data is truncated".into()));
        }
        let mut data = Vec::with_capacity(voxel_count);
        for index in 0..voxel_count {
            let offset = data_offset + index * bytes_per_voxel;
            let value = match mode {
                0 => (bytes[offset] as i8) as f32,
                1 => read_i16(bytes, offset, is_little) as f32,
                2 => f32::from_bits(read_u32(bytes, offset, is_little)),
                6 => read_u16(bytes, offset, is_little) as f32,
                _ => unreachable!(),
            };
            if !value.is_finite() {
                return Err(DensityError::EmptyMap);
            }
            data.push(value);
        }
        if data.is_empty() {
            return Err(DensityError::EmptyMap);
        }
        let data_stats = voxel_statistics(&data);
        let header_stats = [
            f32_at(76) as f64,
            f32_at(80) as f64,
            f32_at(84) as f64,
            f32_at(216) as f64,
        ];
        let [dmin, dmax, dmean, rms] = if header_stats.iter().all(|value| value.is_finite())
            && header_stats[0] <= header_stats[1]
            && header_stats[3] > 1.0e-12
        {
            header_stats
        } else {
            data_stats
        };
        let path = label.into();
        Ok(Self {
            metadata: DensityMapMetadata {
                path,
                sha256: format!("{:x}", Sha256::digest(bytes)),
                nx: dimensions[0] as usize,
                ny: dimensions[1] as usize,
                nz: dimensions[2] as usize,
                mode,
                nxstart: i32_at(16),
                nystart: i32_at(20),
                nzstart: i32_at(24),
                sampling: [
                    sampling[0] as usize,
                    sampling[1] as usize,
                    sampling[2] as usize,
                ],
                cell_lengths_angstrom: cell_lengths,
                cell_angles_degrees: cell_angles,
                map_axes: [
                    map_axes[0] as usize,
                    map_axes[1] as usize,
                    map_axes[2] as usize,
                ],
                origin_angstrom: [f32_at(196) as f64, f32_at(200) as f64, f32_at(204) as f64],
                space_group: i32_at(88),
                dmin,
                dmax,
                dmean,
                rms,
                warnings: Vec::new(),
                source: Some("memory".into()),
                channel: None,
                source_url: None,
                raw_sha256: None,
                raw_cache_path: None,
                map_detail: None,
            },
            data,
            cartesian_to_fractional,
            fractional_to_cartesian,
        })
    }

    pub fn metadata(&self) -> &DensityMapMetadata {
        &self.metadata
    }

    /// Attach acquisition provenance after a downloaded map has been
    /// converted to CCP4.  Keeping this on the map object means every
    /// downstream report sees the same source information as the scorer.
    pub fn set_provenance(
        &mut self,
        source: impl Into<String>,
        channel: Option<String>,
        source_url: Option<String>,
        raw_sha256: Option<String>,
        raw_cache_path: Option<PathBuf>,
        map_detail: Option<u8>,
    ) {
        self.metadata.source = Some(source.into());
        self.metadata.channel = channel;
        self.metadata.source_url = source_url;
        self.metadata.raw_sha256 = raw_sha256;
        self.metadata.raw_cache_path = raw_cache_path;
        self.metadata.map_detail = map_detail;
    }

    pub fn values(&self) -> &[f32] {
        &self.data
    }

    pub fn fractional_to_cartesian(&self, fractional: [f64; 3]) -> [f64; 3] {
        let value = self.fractional_to_cartesian * Vector3::from(fractional);
        [value.x, value.y, value.z]
    }

    pub fn cartesian_to_fractional(&self, cartesian: [f64; 3]) -> [f64; 3] {
        let value = self.cartesian_to_fractional * Vector3::from(cartesian);
        [value.x, value.y, value.z]
    }

    /// Trilinearly sample the map at a Cartesian position. `None` means the
    /// position lies outside a non-periodic cropped map.
    pub fn value_at_cartesian(&self, cartesian: [f64; 3], periodic: bool) -> Option<f64> {
        self.sample(cartesian, periodic)
    }

    /// Trilinear map value and its exact Cartesian gradient.  The derivative
    /// is piecewise linear inside each grid cell and respects CCP4 axis order,
    /// non-orthogonal cells, cropped origins, and periodic wrapping.
    pub fn value_gradient_at_cartesian(
        &self,
        cartesian: [f64; 3],
        periodic: bool,
    ) -> Option<(f64, [f64; 3])> {
        let mut grid = self.cartesian_to_grid(cartesian);
        let cartesian_shape = self.cartesian_shape();
        for (coordinate, extent) in grid.iter_mut().zip(cartesian_shape) {
            if periodic {
                *coordinate = coordinate.rem_euclid(extent as f64);
            } else if *coordinate < 0.0 || *coordinate > extent as f64 - 1.0 {
                return None;
            }
        }
        let base = grid.map(|coordinate| coordinate.floor() as usize);
        let frac = grid.map(|coordinate| coordinate - coordinate.floor());
        let map_axes = self.map_axes_zero_based();
        let mut value = 0.0;
        let mut grid_gradient = [0.0; 3];
        for dx in 0..=1 {
            for dy in 0..=1 {
                for dz in 0..=1 {
                    let index = [base[0] + dx, base[1] + dy, base[2] + dz];
                    let index = if periodic {
                        [
                            index[0] % cartesian_shape[0],
                            index[1] % cartesian_shape[1],
                            index[2] % cartesian_shape[2],
                        ]
                    } else if index
                        .iter()
                        .zip(cartesian_shape)
                        .any(|(coordinate, extent)| *coordinate >= extent)
                    {
                        continue;
                    } else {
                        index
                    };
                    let wx = if dx == 0 { 1.0 - frac[0] } else { frac[0] };
                    let wy = if dy == 0 { 1.0 - frac[1] } else { frac[1] };
                    let wz = if dz == 0 { 1.0 - frac[2] } else { frac[2] };
                    let dwx = if dx == 0 { -1.0 } else { 1.0 };
                    let dwy = if dy == 0 { -1.0 } else { 1.0 };
                    let dwz = if dz == 0 { -1.0 } else { 1.0 };
                    let array_index = [index[map_axes[0]], index[map_axes[1]], index[map_axes[2]]];
                    let sample = self.data[array_index[0]
                        + self.metadata.nx * (array_index[1] + self.metadata.ny * array_index[2])]
                        as f64;
                    value += sample * wx * wy * wz;
                    grid_gradient[0] += sample * dwx * wy * wz;
                    grid_gradient[1] += sample * wx * dwy * wz;
                    grid_gradient[2] += sample * wx * wy * dwz;
                }
            }
        }
        let sampling = Vector3::new(
            self.metadata.sampling[0] as f64,
            self.metadata.sampling[1] as f64,
            self.metadata.sampling[2] as f64,
        );
        let gradient_fractional = Vector3::new(
            grid_gradient[0] * sampling.x,
            grid_gradient[1] * sampling.y,
            grid_gradient[2] * sampling.z,
        );
        let gradient = self.cartesian_to_fractional.transpose() * gradient_fractional;
        Some((value, [gradient.x, gradient.y, gradient.z]))
    }

    pub fn grid_shape(&self) -> [usize; 3] {
        self.cartesian_shape()
    }

    /// Whether the stored volume spans the complete sampled unit cell. Such
    /// maps are naturally periodic even when coordinates in a PDB use a
    /// translated crystallographic image.
    pub fn is_full_unit_cell(&self) -> bool {
        self.metadata.nx == self.metadata.sampling[self.metadata.map_axes[0] - 1]
            && self.metadata.ny == self.metadata.sampling[self.metadata.map_axes[1] - 1]
            && self.metadata.nz == self.metadata.sampling[self.metadata.map_axes[2] - 1]
    }

    fn grid_to_cartesian(&self, grid: [f64; 3]) -> [f64; 3] {
        let sampling = self.metadata.sampling.map(|value| value as f64);
        let array_starts = [
            self.metadata.nxstart as f64,
            self.metadata.nystart as f64,
            self.metadata.nzstart as f64,
        ];
        let mut starts = [0.0; 3];
        for (array_axis, cartesian_axis) in self.map_axes_zero_based().into_iter().enumerate() {
            starts[cartesian_axis] = array_starts[array_axis];
        }
        let fractional = [
            (grid[0] + starts[0]) / sampling[0],
            (grid[1] + starts[1]) / sampling[1],
            (grid[2] + starts[2]) / sampling[2],
        ];
        let mut cart = self.fractional_to_cartesian(fractional);
        for (value, origin) in cart.iter_mut().zip(self.metadata.origin_angstrom) {
            *value += origin;
        }
        cart
    }

    fn cartesian_to_grid(&self, cartesian: [f64; 3]) -> [f64; 3] {
        let shifted = [
            cartesian[0] - self.metadata.origin_angstrom[0],
            cartesian[1] - self.metadata.origin_angstrom[1],
            cartesian[2] - self.metadata.origin_angstrom[2],
        ];
        let fractional = self.cartesian_to_fractional(shifted);
        let sampling = self.metadata.sampling.map(|value| value as f64);
        let array_starts = [
            self.metadata.nxstart as f64,
            self.metadata.nystart as f64,
            self.metadata.nzstart as f64,
        ];
        let mut starts = [0.0; 3];
        for (array_axis, cartesian_axis) in self.map_axes_zero_based().into_iter().enumerate() {
            starts[cartesian_axis] = array_starts[array_axis];
        }
        [
            fractional[0] * sampling[0] - starts[0],
            fractional[1] * sampling[1] - starts[1],
            fractional[2] * sampling[2] - starts[2],
        ]
    }

    fn cartesian_shape(&self) -> [usize; 3] {
        let array_shape = [self.metadata.nx, self.metadata.ny, self.metadata.nz];
        let mut shape = [0; 3];
        for (array_axis, cartesian_axis) in self.map_axes_zero_based().into_iter().enumerate() {
            shape[cartesian_axis] = array_shape[array_axis];
        }
        shape
    }

    fn sample(&self, cartesian: [f64; 3], periodic: bool) -> Option<f64> {
        let mut grid = self.cartesian_to_grid(cartesian);
        let cartesian_shape = self.cartesian_shape();
        for (coordinate, extent) in grid.iter_mut().zip(cartesian_shape) {
            if periodic {
                *coordinate = coordinate.rem_euclid(extent as f64);
            } else if *coordinate < 0.0 || *coordinate > extent as f64 - 1.0 {
                return None;
            }
        }
        let base = grid.map(|coordinate| coordinate.floor() as usize);
        let frac = grid.map(|coordinate| coordinate - coordinate.floor());
        let map_axes = self.map_axes_zero_based();
        let mut value = 0.0;
        for dx in 0..=1 {
            for dy in 0..=1 {
                for dz in 0..=1 {
                    let index = [base[0] + dx, base[1] + dy, base[2] + dz];
                    let index = if periodic {
                        [
                            index[0] % cartesian_shape[0],
                            index[1] % cartesian_shape[1],
                            index[2] % cartesian_shape[2],
                        ]
                    } else if index
                        .iter()
                        .zip(cartesian_shape)
                        .any(|(coordinate, extent)| *coordinate >= extent)
                    {
                        continue;
                    } else {
                        index
                    };
                    let wx = if dx == 0 { 1.0 - frac[0] } else { frac[0] };
                    let wy = if dy == 0 { 1.0 - frac[1] } else { frac[1] };
                    let wz = if dz == 0 { 1.0 - frac[2] } else { frac[2] };
                    let array_index = [index[map_axes[0]], index[map_axes[1]], index[map_axes[2]]];
                    value += self.data[array_index[0]
                        + self.metadata.nx * (array_index[1] + self.metadata.ny * array_index[2])]
                        as f64
                        * wx
                        * wy
                        * wz;
                }
            }
        }
        Some(value)
    }

    /// Read a value at an integer Cartesian grid coordinate without the
    /// matrix inversion and eight-voxel interpolation steps needed for an
    /// arbitrary Cartesian query.  Density masks are accumulated on integer
    /// grid points, so this is exactly the sample that `sample(grid_to_cartesian
    /// (g))` intends to obtain (up to floating-point round-off).
    fn sample_grid(&self, grid: [isize; 3], periodic: bool) -> Option<f64> {
        let shape = self.cartesian_shape();
        let mut index = [0usize; 3];
        for axis in 0..3 {
            let coordinate = if periodic {
                grid[axis].rem_euclid(shape[axis] as isize)
            } else if grid[axis] < 0 || grid[axis] >= shape[axis] as isize {
                return None;
            } else {
                grid[axis]
            };
            index[axis] = coordinate as usize;
        }
        let map_axes = self.map_axes_zero_based();
        let array_index = [index[map_axes[0]], index[map_axes[1]], index[map_axes[2]]];
        Some(
            self.data[array_index[0]
                + self.metadata.nx * (array_index[1] + self.metadata.ny * array_index[2])]
                as f64,
        )
    }

    /// Distance in the map's crystallographic coordinate system. A simple
    /// Cartesian subtraction is wrong for an atom or crop that crosses a
    /// periodic boundary, and is particularly misleading for oblique cells.
    /// Wrapping the fractional displacement before applying the cell matrix
    /// keeps the object continuous at the boundary while retaining the true
    /// non-orthogonal metric.
    /// Cartesian distance respecting periodic unit-cell wrapping when
    /// requested.  Search layers use this for topology/component geometry
    /// without duplicating map-axis handling.
    pub fn map_distance(&self, first: [f64; 3], second: [f64; 3], periodic: bool) -> f64 {
        if !periodic {
            return distance(first, second);
        }
        let origin = self.metadata.origin_angstrom;
        let first_fractional = self.cartesian_to_fractional([
            first[0] - origin[0],
            first[1] - origin[1],
            first[2] - origin[2],
        ]);
        let second_fractional = self.cartesian_to_fractional([
            second[0] - origin[0],
            second[1] - origin[1],
            second[2] - origin[2],
        ]);
        let mut delta = [
            first_fractional[0] - second_fractional[0],
            first_fractional[1] - second_fractional[1],
            first_fractional[2] - second_fractional[2],
        ];
        delta = self.minimum_image_fractional(delta);
        let cartesian = self.fractional_to_cartesian(delta);
        distance(cartesian, [0.0, 0.0, 0.0])
    }

    fn map_displacement(&self, first: [f64; 3], second: [f64; 3], periodic: bool) -> [f64; 3] {
        if !periodic {
            return [
                second[0] - first[0],
                second[1] - first[1],
                second[2] - first[2],
            ];
        }
        let origin = self.metadata.origin_angstrom;
        let first_fractional = self.cartesian_to_fractional([
            first[0] - origin[0],
            first[1] - origin[1],
            first[2] - origin[2],
        ]);
        let second_fractional = self.cartesian_to_fractional([
            second[0] - origin[0],
            second[1] - origin[1],
            second[2] - origin[2],
        ]);
        self.fractional_to_cartesian(self.minimum_image_fractional([
            second_fractional[0] - first_fractional[0],
            second_fractional[1] - first_fractional[1],
            second_fractional[2] - first_fractional[2],
        ]))
    }

    /// Return the shortest fractional displacement under a periodic cell.
    /// Independent component rounding is exact for orthogonal cells but can
    /// choose the wrong lattice image for an oblique unit cell, so inspect the
    /// neighbouring images in the Cartesian metric as well.
    fn minimum_image_fractional(&self, mut delta: [f64; 3]) -> [f64; 3] {
        for component in &mut delta {
            *component -= component.round();
        }
        let mut best = delta;
        let mut best_norm = self
            .fractional_to_cartesian(delta)
            .iter()
            .map(|value| value * value)
            .sum::<f64>();
        for i in -1..=1 {
            for j in -1..=1 {
                for k in -1..=1 {
                    let candidate = [
                        delta[0] + i as f64,
                        delta[1] + j as f64,
                        delta[2] + k as f64,
                    ];
                    let cartesian = self.fractional_to_cartesian(candidate);
                    let norm = cartesian.iter().map(|value| value * value).sum::<f64>();
                    if norm < best_norm {
                        best = candidate;
                        best_norm = norm;
                    }
                }
            }
        }
        best
    }

    fn canonical_cartesian(&self, cartesian: [f64; 3]) -> [f64; 3] {
        let mut grid = self.cartesian_to_grid(cartesian);
        let shape = self.cartesian_shape();
        for (coordinate, extent) in grid.iter_mut().zip(shape) {
            *coordinate = coordinate.rem_euclid(extent as f64);
        }
        self.grid_to_cartesian(grid)
    }

    fn map_axes_zero_based(&self) -> [usize; 3] {
        self.metadata.map_axes.map(|axis| axis - 1)
    }
}

fn read_u16(bytes: &[u8], offset: usize, little: bool) -> u16 {
    let value = [bytes[offset], bytes[offset + 1]];
    if little {
        u16::from_le_bytes(value)
    } else {
        u16::from_be_bytes(value)
    }
}

fn read_i16(bytes: &[u8], offset: usize, little: bool) -> i16 {
    read_u16(bytes, offset, little) as i16
}

fn read_u32(bytes: &[u8], offset: usize, little: bool) -> u32 {
    let value = [
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ];
    if little {
        u32::from_le_bytes(value)
    } else {
        u32::from_be_bytes(value)
    }
}

fn read_i32(bytes: &[u8], offset: usize, little: bool) -> i32 {
    read_u32(bytes, offset, little) as i32
}

fn plausible_mrc_header(bytes: &[u8], little: bool) -> bool {
    let dimensions = [
        read_i32(bytes, 0, little),
        read_i32(bytes, 4, little),
        read_i32(bytes, 8, little),
    ];
    let mode = read_i32(bytes, 12, little);
    dimensions
        .iter()
        .all(|value| (1..=1_000_000).contains(value))
        && matches!(mode, 0 | 1 | 2 | 3 | 4 | 6)
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DensityAtomSupport {
    pub residue: ResidueId,
    pub atom: String,
    pub map_value: Option<f64>,
    pub normalized_value: Option<f64>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DensityResidueSupport {
    pub residue: ResidueId,
    pub atom_support: f64,
    pub ring_support: f64,
    pub connection_support: f64,
    pub confidence: f64,
    pub supported: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DensityLinkageSupport {
    pub donor: ResidueId,
    pub acceptor: ResidueId,
    pub support: f64,
    pub boundary: bool,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DensitySiteScore {
    pub site: ResidueId,
    pub correlation: f64,
    /// Masked correlation of the calculated glycan field against the
    /// observed map after subtracting the fitted protein model.
    #[serde(default)]
    pub protein_adjusted_correlation: f64,
    /// Gaussian residual log-likelihood improvement over a constant-density
    /// null model after fitting a non-negative calculated-map scale.
    #[serde(default)]
    pub likelihood_gain: f64,
    /// Likelihood gain penalized for the additional fitted scale parameter.
    #[serde(default)]
    pub bic_gain: f64,
    pub voxel_count: usize,
    pub supported_atom_fraction: f64,
    #[serde(default)]
    pub ring_support: f64,
    #[serde(default)]
    pub connectivity_support: f64,
    pub atoms: Vec<DensityAtomSupport>,
    #[serde(default)]
    pub residue_support: Vec<DensityResidueSupport>,
    #[serde(default)]
    pub linkage_support: Vec<DensityLinkageSupport>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DensityScore {
    pub correlation: f64,
    /// Masked correlation after protein-background subtraction.
    #[serde(default)]
    pub protein_adjusted_correlation: f64,
    #[serde(default)]
    pub likelihood_gain: f64,
    /// Fixed-ROI likelihood on deterministic optimization voxels.
    #[serde(default)]
    pub training_likelihood_gain: f64,
    /// Fixed-ROI likelihood on spatially disjoint validation voxels.
    #[serde(default)]
    pub heldout_likelihood_gain: f64,
    /// Signed Fo-Fc consistency contribution, when a difference map exists.
    #[serde(default)]
    pub difference_score: f64,
    #[serde(default)]
    pub bic_gain: f64,
    pub sites: Vec<DensitySiteScore>,
    pub sigma_angstrom: f64,
    pub mask_radius_angstrom: f64,
    pub mask_falloff_angstrom: f64,
    pub voxel_count: usize,
    #[serde(default)]
    pub periodic: bool,
    #[serde(default)]
    pub supported_atom_fraction: f64,
    #[serde(default)]
    pub ring_support: f64,
    #[serde(default)]
    pub connectivity_support: f64,
    #[serde(default)]
    pub support_threshold: f64,
    #[serde(default)]
    pub residue_support: Vec<DensityResidueSupport>,
    #[serde(default)]
    pub linkage_support: Vec<DensityLinkageSupport>,
}

/// Paths and geometry of the small maps written for visual inspection of a
/// density fit.  All three files share one grid and header transform.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DensityVisualizationMaps {
    pub observed: PathBuf,
    pub calculated: PathBuf,
    pub mask: PathBuf,
    pub crop_start: [isize; 3],
    pub crop_shape: [usize; 3],
    pub margin_angstrom: f64,
}

/// Smooth O(atoms) search surrogate evaluated directly on the experimental
/// map. Exact masked correlation and likelihood are still used to rank final
/// candidates, while this field makes broad conformer/torsion exploration
/// inexpensive and differentiable.
#[derive(Debug, Clone)]
pub struct DensityFastEvidence {
    pub score: f64,
    pub atom_gradients: BTreeMap<glysys::AtomId, [f64; 3]>,
}

/// Minimal atom record used by the residue-frontier scorer. Coordinates are
/// supplied by the caller's cached geometry workspace; no molecular
/// structure or atom-name allocation is needed during proposal scoring.
#[derive(Debug, Clone)]
pub struct DensityIndexedAtom {
    pub id: AtomId,
    pub element: String,
    pub occupancy: f64,
    pub b_factor: f64,
    pub position: [f64; 3],
}

/// One fixed kernel used by the automatic multiscale search.  The weights are
/// calibrated from the protein shell and are shared by every candidate; a
/// pose therefore cannot improve its score by selecting a convenient blur.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DensityKernelScale {
    pub sigma_angstrom: f64,
    pub weight: f64,
    #[serde(default)]
    pub label: String,
}

/// Optional per-residue broadening accepted after a residue has demonstrated
/// connected density support.  These values are nuisance parameters, not
/// free search dimensions: the refinement layer fits them on a fixed pose,
/// applies a BIC/held-out gate, then freezes them for coordinate proposals.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DensityResidueKernel {
    pub residue: ResidueId,
    pub b_factor: f64,
    pub effective_sigma_angstrom: f64,
    #[serde(default)]
    pub heldout_gain: f64,
    #[serde(default)]
    pub bic_gain: f64,
    #[serde(default)]
    pub accepted: bool,
    #[serde(default)]
    pub reason: String,
}

/// Candidate-independent kernel profile used by the automatic density
/// workflow.  `map_sigma_angstrom` is the instrumental width; B broadening is
/// applied consistently to protein and generated glycan atoms.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DensityKernelProfile {
    pub map_sigma_angstrom: f64,
    #[serde(default)]
    pub fallback_b_factor: Option<f64>,
    #[serde(default)]
    pub scales: Vec<DensityKernelScale>,
    #[serde(default)]
    pub residue_kernels: Vec<DensityResidueKernel>,
    #[serde(default)]
    pub element_weighted: bool,
}

/// A density-derived candidate ring centre.  These hypotheses are
/// deliberately residue-agnostic: topology and stereochemistry assign them
/// to monosaccharide nodes later, so a missing GlycoShape conformer cannot
/// prevent a strong map feature from entering the search.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DensityRingHypothesis {
    pub site: ResidueId,
    pub component_id: usize,
    pub center_angstrom: [f64; 3],
    pub normalized_score: f64,
    pub local_support: f64,
    /// Approximate ring-frame orientation.  The current detector derives the
    /// frame from the local residual-density Hessian; consumers must treat it
    /// as a proposal orientation and may refine it during linkage fitting.
    #[serde(default = "default_ring_orientation")]
    pub orientation_quaternion: [f64; 4],
    /// Residue types for which the matched template was compatible.  Keeping
    /// this on the hypothesis lets the topology solver reject chemically
    /// impossible assignments before exact scoring.
    #[serde(default)]
    pub compatible_residues: Vec<String>,
    /// How unique the local component is relative to its suppressed
    /// neighbours, in [0,1].
    #[serde(default)]
    pub uniqueness: f64,
    #[serde(default = "default_ring_provenance")]
    pub provenance: String,
}

fn default_ring_orientation() -> [f64; 4] {
    [1.0, 0.0, 0.0, 0.0]
}

fn default_ring_provenance() -> String {
    "native_residual_peak".into()
}

/// Return a stable quaternion that maps the template +Z axis onto a local
/// density-gradient direction.  This is deliberately a cheap orientation
/// seed; the refinement layer searches the in-plane rotation and reoptimizes
/// all linkage torsions against the exact map objective.
fn quaternion_from_z_axis(vector: [f64; 3]) -> [f64; 4] {
    let norm = (vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2]).sqrt();
    if !norm.is_finite() || norm < 1.0e-10 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    let target = [vector[0] / norm, vector[1] / norm, vector[2] / norm];
    let dot = target[2].clamp(-1.0, 1.0);
    if dot > 1.0 - 1.0e-8 {
        return [1.0, 0.0, 0.0, 0.0];
    }
    if dot < -1.0 + 1.0e-8 {
        return [0.0, 1.0, 0.0, 0.0];
    }
    let axis = [-target[1], target[0], 0.0];
    let axis_norm = (axis[0] * axis[0] + axis[1] * axis[1]).sqrt();
    let half = 0.5 * dot.acos();
    let scale = half.sin() / axis_norm;
    [
        half.cos(),
        axis[0] * scale,
        axis[1] * scale,
        axis[2] * scale,
    ]
}

fn rotate_template_offset(rotation: [[f64; 3]; 3], offset: [f64; 3]) -> [f64; 3] {
    [
        rotation[0][0] * offset[0] + rotation[0][1] * offset[1] + rotation[0][2] * offset[2],
        rotation[1][0] * offset[0] + rotation[1][1] * offset[1] + rotation[1][2] * offset[2],
        rotation[2][0] * offset[0] + rotation[2][1] * offset[1] + rotation[2][2] * offset[2],
    ]
}

/// Return the shortest circular interval covering canonical grid coordinates.
/// The interval may extend past the unit-cell boundary; callers normalize its
/// fractional endpoints when testing periodic membership.
fn circular_grid_interval(values: &[f64], extent: f64) -> (f64, f64) {
    if values.is_empty() || !extent.is_finite() || extent <= 0.0 {
        return (0.0, extent.max(0.0));
    }
    let mut sorted = values
        .iter()
        .copied()
        .map(|value| value.rem_euclid(extent))
        .collect::<Vec<_>>();
    sorted.sort_by(f64::total_cmp);
    let mut largest_gap = -1.0;
    let mut gap_index = 0usize;
    for index in 0..sorted.len() {
        let next = if index + 1 < sorted.len() {
            sorted[index + 1]
        } else {
            sorted[0] + extent
        };
        let gap = next - sorted[index];
        if gap > largest_gap {
            largest_gap = gap;
            gap_index = index;
        }
    }
    let start = if gap_index + 1 < sorted.len() {
        sorted[gap_index + 1]
    } else {
        sorted[0]
    };
    (start, (extent - largest_gap).max(0.0))
}

fn quaternion_from_euler(roll: f64, pitch: f64, yaw: f64) -> [f64; 4] {
    let (sr, cr) = (roll * 0.5).sin_cos();
    let (sp, cp) = (pitch * 0.5).sin_cos();
    let (sy, cy) = (yaw * 0.5).sin_cos();
    [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]
}

/// Candidate-independent density region used during optimization. The voxel
/// set and weights are fixed when the region is constructed; candidate atoms
/// only update their calculated Gaussian field. This prevents a misplaced
/// branch from hiding unexplained density by moving its mask with itself.
#[derive(Debug, Clone)]
pub struct DensityFixedRegion {
    voxels: Vec<DensityFixedVoxel>,
    sigma_angstrom: f64,
    periodic: bool,
}

impl DensityFixedRegion {
    /// Number of voxels in the reusable candidate-independent ROI.
    pub fn voxel_count(&self) -> usize {
        self.voxels.len()
    }

    /// Return deterministic ownership buckets for diagnostics.  The buckets
    /// are computed after the fixed ownership partition and never change with
    /// candidate coordinates.
    pub fn ownership_buckets(&self) -> (usize, usize, usize) {
        self.voxels.iter().fold((0, 0, 0), |mut counts, voxel| {
            if voxel.weight >= 0.66 {
                counts.0 += 1;
            } else if voxel.weight >= 0.20 {
                counts.1 += 1;
            } else {
                counts.2 += 1;
            }
            counts
        })
    }
}

#[derive(Debug, Clone)]
struct DensityFixedVoxel {
    grid: [isize; 3],
    /// Cartesian coordinate is cached when the fixed ROI is built.  Fixed
    /// scoring is called many times during graph refinement; converting the
    /// same grid coordinate through the unit-cell transform for every atom
    /// and every candidate otherwise dominates the supposedly incremental
    /// path.
    cartesian: [f64; 3],
    /// Fractional coordinate cached alongside the Cartesian coordinate. This
    /// avoids repeated atom/voxel matrix inversions in fixed-region scoring.
    fractional: [f64; 3],
    observed: f64,
    background: f64,
    weight: f64,
}

#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct DensityFixedScore {
    pub correlation: f64,
    pub protein_adjusted_correlation: f64,
    pub likelihood_gain: f64,
    pub training_likelihood_gain: f64,
    pub heldout_likelihood_gain: f64,
    pub bic_gain: f64,
    pub voxel_count: usize,
}

#[derive(Debug, Clone)]
pub struct DensityFixedGradientScore {
    pub score: DensityFixedScore,
    pub atom_gradients: BTreeMap<glysys::AtomId, [f64; 3]>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DensitySigmaTrial {
    pub sigma_angstrom: f64,
    pub training_correlation: f64,
    pub heldout_correlation: f64,
    pub atom_count: usize,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DensitySigmaCalibration {
    pub selected_sigma_angstrom: f64,
    #[serde(default)]
    pub estimated_glycan_b_factor: f64,
    pub method: String,
    pub trials: Vec<DensitySigmaTrial>,
    #[serde(default)]
    pub voxel_spacing_angstrom: [f64; 3],
    #[serde(default)]
    pub anti_alias_floor_angstrom: f64,
    #[serde(default)]
    pub effective_sigma_angstrom: f64,
    /// Width used for the inexpensive capture/local-search field.  This is
    /// derived from the fixed calibration bank (never selected per pose) and
    /// is intentionally broader than the nominal ranking kernel.
    #[serde(default)]
    pub capture_sigma_angstrom: f64,
    #[serde(default)]
    pub search_scales: Vec<DensityKernelScale>,
    #[serde(default)]
    pub residue_kernels: Vec<DensityResidueKernel>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DensitySignedEvidence {
    pub score: f64,
    pub positive_reward: f64,
    pub negative_penalty: f64,
    #[serde(skip)]
    pub atom_gradients: BTreeMap<glysys::AtomId, [f64; 3]>,
}

/// Protein-only Gaussian field over the site neighbourhood plus the fitted
/// linear protein model `intercept + scale * field`, used to score the glycan
/// against the residual map.
#[derive(Debug, Clone)]
struct DensityProteinBackground {
    start: [f64; 3],
    shape: [usize; 3],
    extent: [f64; 3],
    values: Vec<f64>,
    intercept: f64,
    scale: f64,
}

#[derive(Debug, Clone)]
pub struct DensityScorer {
    pub map: DensityMap,
    pub options: DensityScoreOptions,
    protein_background: OnceLock<DensityProteinBackground>,
    kernel_profile: Arc<DensityKernelProfile>,
    ownership: Option<Arc<DensityOwnership>>,
}

#[derive(Debug, Clone)]
struct DensityOwnership {
    owner_index: usize,
    anchors: Vec<[f64; 3]>,
    transition_angstrom: f64,
    periodic: bool,
}

impl DensityOwnership {
    fn weight(&self, map: &DensityMap, point: [f64; 3]) -> f64 {
        if self.anchors.len() <= 1 || self.owner_index >= self.anchors.len() {
            return 1.0;
        }
        let transition = self.transition_angstrom.max(0.25);
        let logits = self
            .anchors
            .iter()
            .map(|anchor| {
                -map.map_distance(point, *anchor, self.periodic).powi(2)
                    / (2.0 * transition * transition)
            })
            .collect::<Vec<_>>();
        let maximum = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let denominator = logits
            .iter()
            .map(|logit| (logit - maximum).exp())
            .sum::<f64>()
            .max(1.0e-30);
        ((logits[self.owner_index] - maximum).exp() / denominator).clamp(0.0, 1.0)
    }
}

impl DensityScorer {
    /// Clone map/options/kernel state without carrying a previously fitted
    /// protein background into another independent glycosylation site.
    /// Background fields are site/structure dependent and must be fitted once
    /// per local scorer.
    pub fn without_protein_background(&self) -> Self {
        Self {
            map: self.map.clone(),
            options: self.options,
            protein_background: OnceLock::new(),
            kernel_profile: self.kernel_profile.clone(),
            ownership: self.ownership.clone(),
        }
    }

    /// Attach a fixed, candidate-independent responsibility to this scorer.
    /// `owner_index` and `anchors` are derived from protein attachment atoms,
    /// never from deposited glycan coordinates.  Responsibilities across all
    /// site scorers sum to one for every map point.
    pub fn with_site_ownership(
        mut self,
        owner_index: usize,
        anchors: Vec<[f64; 3]>,
        transition_angstrom: f64,
    ) -> Self {
        self.ownership = Some(Arc::new(DensityOwnership {
            owner_index,
            anchors,
            transition_angstrom: transition_angstrom.max(0.25),
            periodic: self.options.periodic,
        }));
        self
    }

    /// Keep the same map/kernel/background but clear a previous ownership
    /// responsibility. This is used for the aggregate merged score, where
    /// all site contributions must be evaluated together exactly once.
    pub fn without_site_ownership(mut self) -> Self {
        self.ownership = None;
        self
    }

    /// Fixed ROI ownership counts used by site-level reports.
    pub fn ownership_buckets(&self, region: &DensityFixedRegion) -> (usize, usize, usize) {
        region.ownership_buckets()
    }

    fn ownership_weight(&self, point: [f64; 3]) -> f64 {
        self.ownership
            .as_ref()
            .map(|ownership| ownership.weight(&self.map, point))
            .unwrap_or(1.0)
    }

    fn scoring_atom_amplitude(&self, atom: &glysys::StructureAtom) -> f64 {
        if self.kernel_profile.element_weighted || self.options.glycan_b_factor.is_some() {
            atom_amplitude(atom)
        } else {
            atom.occupancy.clamp(0.0, 1.0)
        }
    }

    fn atom_b_factor(&self, atom: &glysys::StructureAtom) -> f64 {
        if atom.b_factor > 1.0e-6 {
            atom.b_factor.max(0.0)
        } else {
            self.kernel_profile
                .residue_kernels
                .iter()
                .find(|kernel| kernel.residue == atom.residue && kernel.accepted)
                .map(|kernel| kernel.b_factor)
                .or(self.kernel_profile.fallback_b_factor)
                .or(self.options.glycan_b_factor)
                .unwrap_or(0.0)
                .max(0.0)
        }
    }

    fn atom_sigma(&self, map_sigma: f64, atom: &glysys::StructureAtom) -> f64 {
        let b = self.atom_b_factor(atom);
        (map_sigma * map_sigma + b / (8.0 * PI * PI)).sqrt()
    }

    fn indexed_atom_sigma(&self, map_sigma: f64, atom: &DensityIndexedAtom) -> f64 {
        effective_atom_sigma_indexed(
            map_sigma,
            atom,
            self.kernel_profile
                .fallback_b_factor
                .or(self.options.glycan_b_factor),
        )
    }

    pub fn kernel_profile(&self) -> &DensityKernelProfile {
        &self.kernel_profile
    }

    /// Clone the scorer with only its map-kernel width changed. The existing
    /// calibrated B/element profile and fixed ownership responsibility are
    /// retained, so local sharp/nominal stages cannot silently discard the
    /// automatic kernel model.
    pub fn with_options(mut self, options: DensityScoreOptions) -> Result<Self> {
        options.sigma()?;
        self.options = options;
        self.protein_background = OnceLock::new();
        Ok(self)
    }

    /// Change the map-kernel options while retaining an already fitted fixed
    /// protein field. This is used by independent-site capture scorers: the
    /// nominal protein model is shared by every site, while coarse search
    /// kernels are allowed to change only the glycan sampling width. Callers
    /// that change the protein-field model should continue to use
    /// [`DensityScorer::with_options`], which invalidates the cache.
    pub fn with_options_preserve_background(
        mut self,
        options: DensityScoreOptions,
    ) -> Result<Self> {
        options.sigma()?;
        let background = self.protein_background.get().cloned();
        self.options = options;
        self.protein_background = OnceLock::new();
        if let Some(background) = background {
            let _ = self.protein_background.set(background);
        }
        self.ownership = self.ownership.map(|ownership| {
            Arc::new(DensityOwnership {
                periodic: options.periodic,
                ..(*ownership).clone()
            })
        });
        Ok(self)
    }

    /// Whether a candidate-independent protein field has already been fitted
    /// on this scorer. Exposed so multi-site orchestration can avoid silently
    /// replacing a shared field with a site-local one.
    pub fn has_protein_background(&self) -> bool {
        self.protein_background.get().is_some()
    }

    /// Attach a candidate-independent automatic kernel profile.  The scorer
    /// remains cheap to clone because profiles are immutable and reference
    /// counted across all basin and arm workers.
    pub fn with_kernel_profile(mut self, profile: DensityKernelProfile) -> Self {
        self.kernel_profile = Arc::new(profile);
        // The protein field is fitted with the same element/B kernel as the
        // glycan field.  A profile swap (for example after an accepted local
        // residue-B decision) must therefore invalidate a field cached under
        // the previous profile rather than silently mixing two models.
        self.protein_background = OnceLock::new();
        self
    }

    pub fn new(map: DensityMap, options: DensityScoreOptions) -> Result<Self> {
        options.sigma()?;
        Ok(Self {
            map,
            options,
            protein_background: OnceLock::new(),
            kernel_profile: Arc::new(DensityKernelProfile::default()),
            ownership: None,
        })
    }

    /// Build the cached atom-local proposal context used by discrete density
    /// search.  The map and kernel profile are shared with this scorer, while
    /// the fixed protein atoms are derived from the supplied site-local
    /// structure.  Final ranking should still use [`DensityScorer::score`] or
    /// [`DensityScorer::score_fixed_region`].
    pub fn fast_context(
        &self,
        protein: &Structure,
        targets: &[DensityTarget],
    ) -> Result<fast_score::FastDensityContext> {
        let sigma = self.options.sigma()?;
        let fallback_b_factor = self
            .kernel_profile
            .fallback_b_factor
            .or(self.options.glycan_b_factor);
        fast_score::FastDensityContext::new(
            self.map.clone(),
            sigma,
            fallback_b_factor,
            protein,
            targets,
            self.options.periodic,
        )
    }

    /// Calibrate a single candidate-independent map kernel from nearby fixed
    /// protein density. Deposited glycan coordinates are neither required nor
    /// inspected. The held-out spatial fold is authoritative so a broad
    /// kernel cannot win merely by fitting its own local mask.
    pub fn calibrate_sigma_from_protein(
        map: &DensityMap,
        structure: &Structure,
        sites: &[ResidueId],
        base_options: DensityScoreOptions,
        candidates_angstrom: &[f64],
    ) -> Result<DensitySigmaCalibration> {
        let site_set = sites.iter().cloned().collect::<BTreeSet<_>>();
        let site_positions = structure
            .atoms()
            .into_iter()
            .filter(|atom| site_set.contains(&atom.residue))
            .map(|atom| atom.position)
            .collect::<Vec<_>>();
        if site_positions.is_empty() {
            return Err(DensityError::EmptyTarget);
        }
        let mut atoms = structure
            .atoms()
            .into_iter()
            .filter(|atom| {
                !site_set.contains(&atom.residue)
                    && !atom.element.eq_ignore_ascii_case("H")
                    && site_positions.iter().any(|site| {
                        let dx = atom.position.x - site.x;
                        let dy = atom.position.y - site.y;
                        let dz = atom.position.z - site.z;
                        dx * dx + dy * dy + dz * dz <= 12.0_f64.powi(2)
                    })
            })
            .collect::<Vec<_>>();
        // Sigma calibration is a local nuisance fit, not a second whole-map
        // reconstruction.  Large biological assemblies can put thousands of
        // protein atoms inside the union of several 12 Å site shells; using
        // every one of them makes each trial repeat a large BTreeMap splat
        // and can dominate the actual density search.  Keep a deterministic
        // nearest-shell sample that still spans every requested site.
        atoms.sort_by(|left, right| {
            let left_distance = site_positions
                .iter()
                .map(|site| {
                    let dx = left.position.x - site.x;
                    let dy = left.position.y - site.y;
                    let dz = left.position.z - site.z;
                    dx * dx + dy * dy + dz * dz
                })
                .min_by(f64::total_cmp)
                .unwrap_or(f64::INFINITY);
            let right_distance = site_positions
                .iter()
                .map(|site| {
                    let dx = right.position.x - site.x;
                    let dy = right.position.y - site.y;
                    let dz = right.position.z - site.z;
                    dx * dx + dy * dy + dz * dz
                })
                .min_by(f64::total_cmp)
                .unwrap_or(f64::INFINITY);
            left_distance
                .total_cmp(&right_distance)
                .then_with(|| left.id.cmp(&right.id))
        });
        // The calibration statistic is a shell-width estimate, not a full
        // protein-field reconstruction.  Keep a deterministic nearest shell
        // sample, but do not make it so small that a large assembly is
        // represented by one local fragment.  The automatic kernel is shared
        // by every candidate, so this modest cap affects only calibration and
        // not candidate ranking.  In particular, retaining the complete
        // local shell (up to 384 atoms) keeps the estimate stable when the
        // assembly/biological-unit representation changes the nearby protein
        // neighbourhood.
        atoms.truncate(384);
        if atoms.len() < 8 {
            return Err(DensityError::EmptyTarget);
        }
        let mut b_values = atoms
            .iter()
            .map(|atom| atom.b_factor.max(0.0))
            .filter(|value| *value > 1.0e-6)
            .collect::<Vec<_>>();
        b_values.sort_by(f64::total_cmp);
        let estimated_glycan_b_factor = b_values.get(b_values.len() / 2).copied().unwrap_or(0.0);
        // A Gaussian narrower than roughly 0.6 map voxels is under-sampled.
        // Compare the *effective* width (including the shell B estimate), so
        // a well-resolved atom is not rejected merely because its base map
        // sigma is small.
        let sampling = map.metadata.sampling.map(|value| value as f64);
        let voxel_spacing_angstrom = [0, 1, 2].map(|axis| {
            let mut fractional = [0.0; 3];
            fractional[axis] = 1.0 / sampling[axis].max(1.0);
            let cartesian = map.fractional_to_cartesian(fractional);
            cartesian
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt()
        });
        let anti_alias_floor_angstrom =
            0.60 * voxel_spacing_angstrom.iter().copied().fold(0.0, f64::max);
        let shell_b_sigma = (estimated_glycan_b_factor.max(0.0) / (8.0 * PI * PI)).sqrt();
        let mut trials = Vec::new();
        for sigma in candidates_angstrom.iter().copied() {
            if !sigma.is_finite() || sigma <= 0.0 {
                continue;
            }
            let mut options = base_options;
            options.sigma_angstrom = Some(sigma);
            // Calibration and subsequent generated-glycan scoring must use
            // the same element weights and fallback B model.  Previously the
            // protein shell used its deposited B values while Adaptive
            // scored generated glycan atoms with occupancy-only, overly sharp
            // kernels.
            options.glycan_b_factor = Some(estimated_glycan_b_factor);
            let scorer = Self::new(map.clone(), options)?;
            // Calibration samples are independent protein-shell observations;
            // they do not need a full glycan-sized 3-D mask.  A reduced mask
            // radius avoids traversing a large periodic box for every atom
            // and sigma while preserving the shell correlation statistic.
            let samples = scorer.samples_for_atoms_with_radius(&atoms, sigma, 1.5, 0.75)?;
            let (training, heldout): (Vec<_>, Vec<_>) = samples
                .into_iter()
                .partition(|(grid, _, _, _, _)| spatial_fold(*grid) != 0);
            let score = |values: Vec<([isize; 3], f64, f64, f64, f64)>| {
                agreement_statistics(
                    &values
                        .into_iter()
                        .map(|(_, calculated, observed, weight, background)| {
                            (calculated, observed, weight, background)
                        })
                        .collect::<Vec<_>>(),
                )
            };
            let training = score(training)?;
            let heldout = score(heldout)?;
            trials.push(DensitySigmaTrial {
                sigma_angstrom: sigma,
                training_correlation: training.correlation,
                heldout_correlation: heldout.correlation,
                atom_count: atoms.len(),
            });
        }
        trials.sort_by(|left, right| left.sigma_angstrom.total_cmp(&right.sigma_angstrom));
        let selected = trials
            .iter()
            .max_by(|left, right| {
                left.heldout_correlation
                    .total_cmp(&right.heldout_correlation)
                    .then_with(|| {
                        left.training_correlation
                            .total_cmp(&right.training_correlation)
                    })
                    .then_with(|| {
                        (right.sigma_angstrom - 1.0)
                            .abs()
                            .total_cmp(&(left.sigma_angstrom - 1.0).abs())
                    })
            })
            .ok_or(DensityError::ConstantSample)?;
        let effective_sigma_angstrom =
            (selected.sigma_angstrom.powi(2) + shell_b_sigma.powi(2)).sqrt();
        let mut search_scales = vec![
            DensityKernelScale {
                sigma_angstrom: (selected.sigma_angstrom * 0.80).max(0.25),
                weight: 0.20,
                label: "sharp".into(),
            },
            DensityKernelScale {
                sigma_angstrom: selected.sigma_angstrom,
                weight: 0.60,
                label: "nominal".into(),
            },
            DensityKernelScale {
                sigma_angstrom: selected.sigma_angstrom * 1.25,
                weight: 0.20,
                label: "coarse".into(),
            },
        ];
        if effective_sigma_angstrom < anti_alias_floor_angstrom {
            let scale = anti_alias_floor_angstrom / effective_sigma_angstrom.max(1.0e-6);
            for entry in &mut search_scales {
                entry.sigma_angstrom *= scale;
            }
        }
        let capture_sigma_angstrom = search_scales
            .iter()
            .find(|scale| scale.label.eq_ignore_ascii_case("coarse"))
            .map(|scale| scale.sigma_angstrom)
            .unwrap_or((selected.sigma_angstrom * 1.5).max(anti_alias_floor_angstrom))
            .max(anti_alias_floor_angstrom);
        Ok(DensitySigmaCalibration {
            selected_sigma_angstrom: selected.sigma_angstrom,
            estimated_glycan_b_factor,
            method: "local_protein_spatial_holdout_element_b_consistent".into(),
            trials,
            voxel_spacing_angstrom,
            anti_alias_floor_angstrom,
            effective_sigma_angstrom,
            capture_sigma_angstrom,
            search_scales,
            residue_kernels: Vec::new(),
        })
    }

    /// Estimate restrained local B factors for strongly supported residues.
    /// The observed residual tile and topology are fixed before the trial
    /// widths are compared, so this routine cannot be used by a candidate to
    /// broaden its own mask.  It is intentionally conservative: unsupported
    /// or ambiguous residues receive an explicit rejected record and retain
    /// the shared fallback B factor.
    pub fn calibrate_residue_kernels(
        &self,
        structure: &Structure,
        target: &DensityTarget,
    ) -> Result<Vec<DensityResidueKernel>> {
        let resolved = if target.glycan_residues.is_empty() {
            DensityTarget::for_site(structure, &target.site)?
        } else {
            target.clone()
        };
        let fallback = self
            .kernel_profile
            .fallback_b_factor
            .or(self.options.glycan_b_factor)
            .unwrap_or(0.0)
            .max(0.0);
        let trial_b = if fallback > 1.0e-6 {
            [0.75, 1.0, 1.25]
                .map(|factor| (fallback * factor).clamp(0.0, 300.0))
                .to_vec()
        } else {
            vec![0.0, 50.0, 120.0]
        };
        let sigma = self.options.sigma()?;
        let fallback_record = |residue: &ResidueId, reason: &str| DensityResidueKernel {
            residue: residue.clone(),
            b_factor: fallback,
            effective_sigma_angstrom: (sigma * sigma + fallback / (8.0 * PI * PI)).sqrt(),
            heldout_gain: 0.0,
            bic_gain: 0.0,
            accepted: false,
            reason: reason.into(),
        };
        let base_score = match self.score(structure, std::slice::from_ref(&resolved)) {
            Ok(score) => score,
            Err(_) => {
                return Ok(resolved
                    .glycan_residues
                    .iter()
                    .map(|residue| fallback_record(residue, "shared_kernel_retained_map_weak"))
                    .collect());
            }
        };
        let mut result = Vec::new();
        for residue in &resolved.glycan_residues {
            let atoms = structure
                .atoms()
                .into_iter()
                .filter(|atom| atom.residue == *residue && !atom.element.eq_ignore_ascii_case("H"))
                .collect::<Vec<_>>();
            if atoms.is_empty() {
                continue;
            }
            let support = base_score
                .residue_support
                .iter()
                .find(|support| support.residue == *residue);
            let strong = support.is_some_and(|support| {
                support.supported
                    && support.ring_support >= 0.65
                    && support.connection_support >= 0.50
            });
            if !strong {
                result.push(fallback_record(residue, "weak_or_disconnected_support"));
                continue;
            }
            let base_samples = match self.samples_for_atoms(&atoms, sigma) {
                Ok(samples) => samples,
                Err(_) => {
                    result.push(fallback_record(residue, "shared_kernel_retained_map_weak"));
                    continue;
                }
            };
            let base_values = base_samples
                .iter()
                .map(|(_, calculated, observed, weight, background)| {
                    (*calculated, *observed, *weight, *background)
                })
                .collect::<Vec<_>>();
            let base = match agreement_statistics(&base_values) {
                Ok(base) => base,
                Err(_) => {
                    result.push(fallback_record(
                        residue,
                        "shared_kernel_retained_constant_tile",
                    ));
                    continue;
                }
            };
            let mut best: Option<(f64, f64, f64)> = None;
            for candidate_b in trial_b.iter().copied() {
                let mut profile = self.kernel_profile.as_ref().clone();
                profile
                    .residue_kernels
                    .retain(|kernel| kernel.residue != *residue);
                profile.residue_kernels.push(DensityResidueKernel {
                    residue: residue.clone(),
                    b_factor: candidate_b,
                    effective_sigma_angstrom: (sigma * sigma + candidate_b / (8.0 * PI * PI))
                        .sqrt(),
                    heldout_gain: 0.0,
                    bic_gain: 0.0,
                    accepted: true,
                    reason: "trial".into(),
                });
                let trial_scorer = self.clone().with_kernel_profile(profile);
                let Ok(samples) = trial_scorer.samples_for_atoms(&atoms, sigma) else {
                    continue;
                };
                let (training, heldout): (Vec<_>, Vec<_>) = samples
                    .into_iter()
                    .partition(|(grid, _, _, _, _)| spatial_fold(*grid) != 0);
                let training = agreement_statistics(
                    &training
                        .into_iter()
                        .map(|(_, calculated, observed, weight, background)| {
                            (calculated, observed, weight, background)
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap_or(base);
                let heldout = agreement_statistics(
                    &heldout
                        .into_iter()
                        .map(|(_, calculated, observed, weight, background)| {
                            (calculated, observed, weight, background)
                        })
                        .collect::<Vec<_>>(),
                )
                .unwrap_or(training);
                let heldout_gain = heldout.likelihood_gain - base.likelihood_gain;
                let bic_gain = heldout.bic_gain - base.bic_gain;
                if best.is_none_or(|(_, previous_heldout, previous_bic)| {
                    heldout_gain > previous_heldout + 1.0e-12
                        || (heldout_gain - previous_heldout).abs() <= 1.0e-12
                            && bic_gain > previous_bic
                }) {
                    best = Some((candidate_b, heldout_gain, bic_gain));
                }
            }
            let (b_factor, heldout_gain, bic_gain) = best.unwrap_or((fallback, 0.0, 0.0));
            let accepted = heldout_gain > 0.0 && bic_gain >= 6.0;
            let selected_b = if accepted { b_factor } else { fallback };
            result.push(DensityResidueKernel {
                residue: residue.clone(),
                b_factor: selected_b,
                effective_sigma_angstrom: (sigma * sigma + selected_b / (8.0 * PI * PI)).sqrt(),
                heldout_gain: if accepted { heldout_gain } else { 0.0 },
                bic_gain: if accepted { bic_gain } else { 0.0 },
                accepted,
                reason: if accepted {
                    "heldout_bic_improvement".into()
                } else {
                    "shared_kernel_retained".into()
                },
            });
        }
        Ok(result)
    }

    /// Precompute the protein-only Gaussian field over the site neighbourhood
    /// once, aligned with the map grid, and fit its linear scale to the
    /// observed map.  Every later score then compares the glycan against the
    /// residual `observed - protein_model`, so placing a sugar ring inside a
    /// protein blob earns no credit.
    pub fn attach_protein_background(
        &self,
        structure: &Structure,
        targets: &[DensityTarget],
    ) -> Result<()> {
        if self.protein_background.get().is_some() {
            return Ok(());
        }
        let resolved = targets
            .iter()
            .map(|target| {
                if target.glycan_residues.is_empty() {
                    DensityTarget::for_site(structure, &target.site)
                } else {
                    Ok(target.clone())
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let glycan_residues = resolved
            .iter()
            .flat_map(|target| target.glycan_residues.iter().cloned())
            .collect::<BTreeSet<_>>();
        self.options.sigma()?;
        // Deep mode opts into the calibrated, element/B-aware fixed field by
        // supplying the estimated glycan B factor.  Fast/adaptive retain the
        // established broad protein background so their legacy objective and
        // recovery behavior remain unchanged; this is intentionally scoped
        // rather than silently changing the default search landscape.
        let calibrated_background =
            self.kernel_profile.element_weighted || self.options.glycan_b_factor.is_some();
        let sigma_background = if calibrated_background {
            self.options.sigma()?
        } else {
            2.5
        };
        let extent = self.map.cartesian_shape().map(|value| value as f64);
        let cell = self.map.metadata().cell_lengths_angstrom;
        let glycan_grid = structure
            .atoms()
            .into_iter()
            .filter(|atom| glycan_residues.contains(&atom.residue))
            .filter(|atom| !atom.element.eq_ignore_ascii_case("H"))
            .map(|atom| {
                let mut grid = self
                    .map
                    .cartesian_to_grid([atom.position.x, atom.position.y, atom.position.z])
                    .to_vec();
                for (coordinate, cell_extent) in grid.iter_mut().zip(extent) {
                    *coordinate = coordinate.rem_euclid(cell_extent);
                }
                [grid[0], grid[1], grid[2]]
            })
            .collect::<Vec<_>>();
        if glycan_grid.is_empty() {
            return Err(DensityError::EmptyTarget);
        }
        let margin_grid = [
            15.0 / (cell[0] / extent[0]).max(1.0e-6),
            15.0 / (cell[1] / extent[1]).max(1.0e-6),
            15.0 / (cell[2] / extent[2]).max(1.0e-6),
        ];
        let mut start = [0.0_f64; 3];
        let mut end = [0.0_f64; 3];
        for axis in 0..3 {
            let values = glycan_grid
                .iter()
                .map(|grid| grid[axis])
                .collect::<Vec<_>>();
            if self.options.periodic {
                // Choose the shortest circular interval containing the
                // glycan atoms. A glycan crossing a unit-cell boundary would
                // otherwise produce min=0/max=extent and make the protein
                // background span the entire biological map.
                let (interval_start, interval_span) = circular_grid_interval(&values, extent[axis]);
                if interval_span + 2.0 * margin_grid[axis] >= extent[axis] {
                    start[axis] = 0.0;
                    end[axis] = extent[axis];
                } else {
                    start[axis] = interval_start - margin_grid[axis];
                    end[axis] = interval_start + interval_span + margin_grid[axis];
                }
            } else {
                let minimum = values.iter().copied().fold(f64::INFINITY, f64::min);
                let maximum = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                start[axis] = (minimum - margin_grid[axis]).max(0.0);
                end[axis] = (maximum + margin_grid[axis]).min(extent[axis]);
            }
        }
        let start_fractional = [
            (start[0] / extent[0]).rem_euclid(1.0),
            (start[1] / extent[1]).rem_euclid(1.0),
            (start[2] / extent[2]).rem_euclid(1.0),
        ];
        let end_fractional = [
            (end[0] / extent[0]).rem_euclid(1.0),
            (end[1] / extent[1]).rem_euclid(1.0),
            (end[2] / extent[2]).rem_euclid(1.0),
        ];
        let shape = [
            (end[0] - start[0]).ceil().max(1.0) as usize,
            (end[1] - start[1]).ceil().max(1.0) as usize,
            (end[2] - start[2]).ceil().max(1.0) as usize,
        ];
        // A union of nearby sites can legitimately span the complete periodic
        // map axis.  In that case `end_fractional.rem_euclid(1.0)` equals
        // `start_fractional` (usually both are zero), and the ordinary wrapped
        // interval predicate would keep only one measure-zero slice of the
        // protein.  Mark full axes explicitly so the shared independent-site
        // background still contains every fixed protein atom.
        let full_periodic_axis = [0usize, 1, 2]
            .map(|axis| self.options.periodic && end[axis] - start[axis] >= extent[axis] - 1.0e-6);
        let protein_fractional = structure
            .atoms()
            .into_iter()
            .filter(|atom| !glycan_residues.contains(&atom.residue))
            .filter(|atom| !atom.element.eq_ignore_ascii_case("H"))
            .map(|atom| {
                let cartesian = [atom.position.x, atom.position.y, atom.position.z];
                let shifted = [
                    cartesian[0] - self.map.metadata().origin_angstrom[0],
                    cartesian[1] - self.map.metadata().origin_angstrom[1],
                    cartesian[2] - self.map.metadata().origin_angstrom[2],
                ];
                let fractional = self.map.cartesian_to_fractional(shifted);
                (
                    [
                        fractional[0].rem_euclid(1.0),
                        fractional[1].rem_euclid(1.0),
                        fractional[2].rem_euclid(1.0),
                    ],
                    if calibrated_background {
                        atom_amplitude(&atom)
                    } else {
                        atom.occupancy.clamp(0.0, 1.0)
                    },
                    if calibrated_background {
                        self.atom_sigma(sigma_background, &atom)
                    } else {
                        sigma_background
                    },
                )
            })
            .filter(|(fractional, _, _)| {
                (0..3).all(|axis| {
                    if full_periodic_axis[axis] {
                        return true;
                    }
                    let wrapped = if start_fractional[axis] <= end_fractional[axis] {
                        fractional[axis] >= start_fractional[axis]
                            && fractional[axis] <= end_fractional[axis]
                    } else {
                        // Box crosses the periodic boundary.
                        fractional[axis] >= start_fractional[axis]
                            || fractional[axis] <= end_fractional[axis]
                    };
                    wrapped
                })
            })
            .collect::<Vec<_>>();
        if std::env::var_os("REGLYCO_DENSITY_DEBUG").is_some() {
            eprintln!(
                "density-background: glycan_atoms={} box_start={start:?} shape={shape:?} protein_atoms={} sigma={sigma_background:.3}",
                glycan_grid.len(),
                protein_fractional.len(),
            );
        }
        let field_started = std::time::Instant::now();
        let mut values = vec![0.0_f64; shape[0] * shape[1] * shape[2]];
        // Atom-local Gaussian splatting is asymptotically much cheaper than
        // the old voxel-by-all-protein-atom traversal.  The latter is
        // especially pathological for periodic EDS maps: a 60x57x55 site
        // box times a few thousand protein atoms can exceed hundreds of
        // millions of exponentials before the first candidate is scored.
        for (fractional, weight, sigma) in &protein_fractional {
            let mut center = [
                fractional[0] * extent[0],
                fractional[1] * extent[1],
                fractional[2] * extent[2],
            ];
            if self.options.periodic {
                let midpoint = [
                    0.5 * (start[0] + end[0]),
                    0.5 * (start[1] + end[1]),
                    0.5 * (start[2] + end[2]),
                ];
                for axis in 0..3 {
                    center[axis] +=
                        ((midpoint[axis] - center[axis]) / extent[axis]).round() * extent[axis];
                }
            }
            let radius = 3.0 * *sigma;
            let delta = [
                radius * extent[0] / cell[0].max(1.0e-6),
                radius * extent[1] / cell[1].max(1.0e-6),
                radius * extent[2] / cell[2].max(1.0e-6),
            ];
            let ranges = (0..3)
                .map(|axis| {
                    let low = (center[axis] - delta[axis] - start[axis]).floor().max(0.0) as usize;
                    let high = (center[axis] - start[axis] + delta[axis])
                        .ceil()
                        .min(shape[axis] as f64 - 1.0)
                        .max(0.0) as usize;
                    low..=high
                })
                .collect::<Vec<_>>();
            for x in ranges[0].clone() {
                for y in ranges[1].clone() {
                    for z in ranges[2].clone() {
                        let mut voxel_fractional = [
                            (start[0] + x as f64 + 0.5) / extent[0],
                            (start[1] + y as f64 + 0.5) / extent[1],
                            (start[2] + z as f64 + 0.5) / extent[2],
                        ];
                        for component in &mut voxel_fractional {
                            *component = component.rem_euclid(1.0);
                        }
                        let mut offset = [
                            voxel_fractional[0] - fractional[0],
                            voxel_fractional[1] - fractional[1],
                            voxel_fractional[2] - fractional[2],
                        ];
                        if self.options.periodic {
                            for component in &mut offset {
                                *component -= component.round();
                            }
                        }
                        let cartesian = self.map.fractional_to_cartesian(offset);
                        let distance_squared = cartesian
                            .iter()
                            .map(|coordinate| coordinate * coordinate)
                            .sum::<f64>();
                        let value = *weight * (-distance_squared / (2.0 * sigma * sigma)).exp();
                        let index = x + shape[0] * (y + shape[1] * z);
                        values[index] += value;
                    }
                }
            }
        }
        if std::env::var_os("REGLYCO_DENSITY_DEBUG").is_some() {
            eprintln!(
                "density-background: field built in {:.3}s",
                field_started.elapsed().as_secs_f64()
            );
        }
        // Fit the linear protein model against the observed map inside the
        // same box so residual subtraction stays in map-density units.
        // Fit the protein scale only where the field is confidently above
        // noise; including the large low-field background dilutes the scale
        // to almost nothing and leaves the subtraction inert.
        let mut sum_background = 0.0;
        let mut sum_observed = 0.0;
        let mut rich_count = 0usize;
        let mut index = 0usize;
        for z in 0..shape[2] {
            for y in 0..shape[1] {
                for x in 0..shape[0] {
                    if values[index] > 1.0 {
                        let cartesian = self.map.grid_to_cartesian([
                            start[0] + x as f64,
                            start[1] + y as f64,
                            start[2] + z as f64,
                        ]);
                        let observed = self
                            .map
                            .sample(cartesian, self.options.periodic)
                            .unwrap_or(0.0);
                        sum_background += values[index];
                        sum_observed += observed;
                        rich_count += 1;
                    }
                    index += 1;
                }
            }
        }
        let mean_background = if rich_count == 0 {
            0.0
        } else {
            sum_background / rich_count as f64
        };
        let mean_observed = if rich_count == 0 {
            0.0
        } else {
            sum_observed / rich_count as f64
        };
        let mut variance = 0.0;
        let mut covariance = 0.0;
        index = 0;
        for z in 0..shape[2] {
            for y in 0..shape[1] {
                for x in 0..shape[0] {
                    if values[index] > 1.0 {
                        let cartesian = self.map.grid_to_cartesian([
                            start[0] + x as f64,
                            start[1] + y as f64,
                            start[2] + z as f64,
                        ]);
                        let observed = self
                            .map
                            .sample(cartesian, self.options.periodic)
                            .unwrap_or(mean_observed);
                        variance +=
                            (values[index] - mean_background) * (values[index] - mean_background);
                        covariance +=
                            (values[index] - mean_background) * (observed - mean_observed);
                    }
                    index += 1;
                }
            }
        }
        let scale = if variance > 1.0e-20 {
            covariance / variance
        } else {
            0.0
        };
        let intercept = mean_observed - scale * mean_background;
        let background = DensityProteinBackground {
            start,
            shape,
            extent,
            values,
            intercept,
            scale,
        };
        let _ = self.protein_background.set(background);
        Ok(())
    }

    fn protein_model_at(&self, cartesian: [f64; 3]) -> f64 {
        let Some(background) = self.protein_background.get() else {
            return 0.0;
        };
        let shifted = [
            cartesian[0] - self.map.metadata().origin_angstrom[0],
            cartesian[1] - self.map.metadata().origin_angstrom[1],
            cartesian[2] - self.map.metadata().origin_angstrom[2],
        ];
        let fractional = self.map.cartesian_to_fractional(shifted);
        let mut grid = [
            fractional[0] * background.extent[0] - background.start[0],
            fractional[1] * background.extent[1] - background.start[1],
            fractional[2] * background.extent[2] - background.start[2],
        ];
        let shape = background.shape;
        for (coordinate, extent) in grid.iter_mut().zip(shape) {
            *coordinate = coordinate.clamp(0.0, extent as f64 - 1.0);
        }
        let base = grid.map(|coordinate| coordinate.floor() as usize);
        let fraction = grid.map(|coordinate| coordinate - coordinate.floor());
        let mut value = 0.0;
        for dx in 0..=1 {
            for dy in 0..=1 {
                for dz in 0..=1 {
                    let x = base[0] + dx;
                    let y = base[1] + dy;
                    let z = base[2] + dz;
                    if x >= shape[0] || y >= shape[1] || z >= shape[2] {
                        continue;
                    }
                    let weight = (if dx == 0 {
                        1.0 - fraction[0]
                    } else {
                        fraction[0]
                    }) * (if dy == 0 {
                        1.0 - fraction[1]
                    } else {
                        fraction[1]
                    }) * (if dz == 0 {
                        1.0 - fraction[2]
                    } else {
                        fraction[2]
                    });
                    value += weight * background.values[x + shape[0] * (y + shape[1] * z)];
                }
            }
        }
        background.intercept + background.scale * value
    }

    /// Build a fixed optimization region from a set of plausible attached
    /// poses. `exploration_margin_angstrom` adds a low-weight halo so nearby
    /// unexplained density remains visible even when no starting conformer
    /// currently occupies it.
    pub fn fixed_region(
        &self,
        structures: &[&Structure],
        targets: &[DensityTarget],
        exploration_margin_angstrom: f64,
    ) -> Result<DensityFixedRegion> {
        if structures.is_empty() || targets.is_empty() {
            return Err(DensityError::EmptyTarget);
        }
        if !exploration_margin_angstrom.is_finite() || exploration_margin_angstrom < 0.0 {
            return Err(DensityError::Geometry(
                "fixed-region exploration margin must be finite and non-negative".into(),
            ));
        }
        let residues = targets
            .iter()
            .flat_map(|target| target.glycan_residues.iter().cloned())
            .collect::<BTreeSet<_>>();
        let centers = structures
            .iter()
            .flat_map(|structure| structure.atoms())
            .filter(|atom| {
                residues.contains(&atom.residue) && !atom.element.eq_ignore_ascii_case("H")
            })
            .map(|atom| [atom.position.x, atom.position.y, atom.position.z])
            .collect::<Vec<_>>();
        if centers.is_empty() {
            return Err(DensityError::EmptyTarget);
        }
        let core_radius = self.options.mask_radius_angstrom + self.options.mask_falloff_angstrom;
        let radius = core_radius + exploration_margin_angstrom;
        let inverse = &self.map.cartesian_to_fractional;
        let delta = [
            self.map.metadata.sampling[0] as f64 * radius * inverse.row(0).norm(),
            self.map.metadata.sampling[1] as f64 * radius * inverse.row(1).norm(),
            self.map.metadata.sampling[2] as f64 * radius * inverse.row(2).norm(),
        ];
        let shape = self.map.cartesian_shape();
        let mut weights = BTreeMap::<[isize; 3], f64>::new();
        for center in centers {
            let center = if self.options.periodic {
                self.map.canonical_cartesian(center)
            } else {
                center
            };
            let grid_center = self.map.cartesian_to_grid(center);
            for x in (grid_center[0] - delta[0]).floor() as isize
                ..=(grid_center[0] + delta[0]).ceil() as isize
            {
                for y in (grid_center[1] - delta[1]).floor() as isize
                    ..=(grid_center[1] + delta[1]).ceil() as isize
                {
                    for z in (grid_center[2] - delta[2]).floor() as isize
                        ..=(grid_center[2] + delta[2]).ceil() as isize
                    {
                        if !self.options.periodic
                            && (x < 0
                                || y < 0
                                || z < 0
                                || x >= shape[0] as isize
                                || y >= shape[1] as isize
                                || z >= shape[2] as isize)
                        {
                            continue;
                        }
                        let cartesian = self.map.grid_to_cartesian([x as f64, y as f64, z as f64]);
                        let distance =
                            self.map
                                .map_distance(cartesian, center, self.options.periodic);
                        if distance > radius {
                            continue;
                        }
                        let core = mask_weight(
                            distance,
                            self.options.mask_radius_angstrom,
                            self.options.mask_falloff_angstrom,
                        );
                        let halo = if exploration_margin_angstrom > 0.0 && distance > core_radius {
                            0.15 * (1.0 - (distance - core_radius) / exploration_margin_angstrom)
                                .max(0.0)
                        } else {
                            0.0
                        };
                        let key = if self.options.periodic {
                            [
                                x.rem_euclid(shape[0] as isize),
                                y.rem_euclid(shape[1] as isize),
                                z.rem_euclid(shape[2] as isize),
                            ]
                        } else {
                            [x, y, z]
                        };
                        let entry = weights.entry(key).or_insert(0.0);
                        *entry = entry.max(core.max(halo));
                    }
                }
            }
        }
        let mut voxels = Vec::with_capacity(weights.len());
        for (grid, weight) in weights {
            let observed = self
                .map
                .sample_grid(grid, self.options.periodic)
                .ok_or(DensityError::OutOfMap)?;
            let cartesian =
                self.map
                    .grid_to_cartesian([grid[0] as f64, grid[1] as f64, grid[2] as f64]);
            let weight = weight * self.ownership_weight(cartesian);
            if weight <= 1.0e-8 {
                continue;
            }
            voxels.push(DensityFixedVoxel {
                grid,
                cartesian,
                fractional: self.map.cartesian_to_fractional([
                    cartesian[0] - self.map.metadata().origin_angstrom[0],
                    cartesian[1] - self.map.metadata().origin_angstrom[1],
                    cartesian[2] - self.map.metadata().origin_angstrom[2],
                ]),
                observed,
                background: self.protein_model_at(cartesian),
                weight,
            });
        }
        Ok(DensityFixedRegion {
            voxels,
            sigma_angstrom: self.options.sigma()?,
            periodic: self.options.periodic,
        })
    }

    /// Detect strong, non-maximum-suppressed residual peaks in the complete
    /// site envelope. These are inexpensive graph seeds, not final atom
    /// placements; the topology solver subsequently fits a chemically valid
    /// ring and linkage pose to each component.
    pub fn detect_ring_hypotheses(
        &self,
        structure: &Structure,
        target: &DensityTarget,
        maximum: usize,
    ) -> Result<Vec<DensityRingHypothesis>> {
        if maximum == 0 {
            return Ok(Vec::new());
        }
        let residues = target.glycan_residues.iter().collect::<BTreeSet<_>>();
        let atoms = structure
            .atoms()
            .into_iter()
            .filter(|atom| {
                residues.contains(&atom.residue) && !atom.element.eq_ignore_ascii_case("H")
            })
            .collect::<Vec<_>>();
        if atoms.is_empty() {
            return Err(DensityError::EmptyTarget);
        }
        let margin = 12.0;
        // Build the envelope in Cartesian space and transform all eight
        // corners back to grid coordinates.  Multiplying an inverse-cell row
        // by the map sampling count (the former implementation) is only
        // correct for a diagonal cell and can admit an unrelated slab of a
        // periodic map, especially along the long unit-cell axis.
        let mut min = [isize::MAX; 3];
        let mut max = [isize::MIN; 3];
        for atom in &atoms {
            let center = [atom.position.x, atom.position.y, atom.position.z];
            for sx in [-1.0, 1.0] {
                for sy in [-1.0, 1.0] {
                    for sz in [-1.0, 1.0] {
                        let grid = self.map.cartesian_to_grid([
                            center[0] + sx * margin,
                            center[1] + sy * margin,
                            center[2] + sz * margin,
                        ]);
                        for axis in 0..3 {
                            min[axis] = min[axis].min(grid[axis].floor() as isize);
                            max[axis] = max[axis].max(grid[axis].ceil() as isize);
                        }
                    }
                }
            }
        }
        let rms = self.map.metadata.rms.abs().max(1.0e-12);
        // Weak distal rings can disappear below the ordinary residual seed
        // gate after protein subtraction.  Keep this relaxation explicitly
        // opt-in: the topology/frontier consumer still has to reject noise,
        // disconnected peaks, clashes, and poor exact likelihood.  Normal
        // fitting retains the calibrated gate and deterministic behaviour.
        let weak_ring_capture = std::env::var_os("REGLYCO_DENSITY_RING_WEAK").is_some();
        let residual_seed_floor = if weak_ring_capture { 0.05 } else { 0.25 };
        // Coarse voxel bins provide deterministic nonmaximum suppression
        // without the previous 27-neighbour map/background rebuild for every
        // voxel.  The complete ring matched filter below performs the final
        // position/orientation suppression.
        let per_z = (min[2]..=max[2])
            .into_par_iter()
            .map(|z| {
                let mut local_bins = BTreeMap::<[isize; 3], (f64, [isize; 3])>::new();
                for x in min[0]..=max[0] {
                    for y in min[1]..=max[1] {
                        let grid = [x, y, z];
                        let Some(value) = self.map.sample_grid(grid, self.options.periodic) else {
                            continue;
                        };
                        let cart = self.map.grid_to_cartesian([x as f64, y as f64, z as f64]);
                        let residual = (value - self.protein_model_at(cart)) / rms;
                        // Distal glycan rings can be only a fraction of an
                        // rms above the locally calibrated protein field.
                        // Keep weaker positive seeds for the topology solver;
                        // connectivity, matched-filter coverage, and exact
                        // fixed-ROI scoring decide whether they are real.
                        // A distal ring can be substantially weaker than the
                        // attached core after protein subtraction.  The
                        // matched filter below, rather than this seed gate,
                        // decides whether a complete ring is plausible; keep
                        // sub-rms positive seeds so weak intermediate
                        // residues cannot disappear before topology connects
                        // them.
                        if residual < residual_seed_floor {
                            continue;
                        }
                        let key = [x.div_euclid(2), y.div_euclid(2), z.div_euclid(2)];
                        local_bins
                            .entry(key)
                            .and_modify(|entry| {
                                if residual > entry.0 {
                                    *entry = (residual, grid);
                                }
                            })
                            .or_insert((residual, grid));
                    }
                }
                local_bins
            })
            .collect::<Vec<_>>();
        let mut peak_bins = BTreeMap::<[isize; 3], (f64, [isize; 3])>::new();
        for local_bins in per_z {
            for (key, entry) in local_bins {
                peak_bins
                    .entry(key)
                    .and_modify(|current| {
                        if entry.0 > current.0 {
                            *current = entry;
                        }
                    })
                    .or_insert(entry);
            }
        }
        let mut peaks = peak_bins.into_values().collect::<Vec<_>>();
        peaks.sort_by(|left, right| {
            right
                .0
                .total_cmp(&left.0)
                .then_with(|| left.1.cmp(&right.1))
        });
        // Seed selection is anchored at the attached root as well as at the
        // strongest residual peaks.  EDS maps of an assembly often contain a
        // second, equally strong copy elsewhere in the periodic cell; taking
        // only the global top peaks lets that copy consume the entire ring
        // hypothesis budget before the requested glycan is considered.
        // Do not assume `tree.residue_ids` is topologically ordered.  The
        // attachment metadata identifies the actual reducing-end residue;
        // using `.first()` here silently anchored some trees on an internal
        // sugar and let a symmetry-related peak win the root slot.
        let attachment_root = structure
            .metadata()
            .glycosylation_sites
            .iter()
            .find(|site| site.protein_residue == target.site)
            .map(|site| site.glycan_residue.clone())
            .or_else(|| {
                target
                    .glycan_residues
                    .iter()
                    .find(|residue| structure.find_atom(residue, "C1").is_some())
                    .cloned()
            });
        let anchor_center = attachment_root
            .as_ref()
            .and_then(|residue| structure.find_atom(residue, "C1"))
            .and_then(|atom| structure.atom(atom).map(|atom| atom.position))
            .map(|position| [position.x, position.y, position.z]);
        // Convert the chemically valid rings already present in the seed
        // structure into translation/orientation-independent templates.  The
        // structure supplies only the ring shape; all candidate centres are
        // found from the experimental residual map below.
        let ring_names = ["C1", "C2", "C3", "C4", "C5", "O5"];
        let mut template_groups = BTreeMap::<String, Vec<Vec<[f64; 3]>>>::new();
        for residue in structure.residues() {
            if !residues.contains(&residue.id) {
                continue;
            }
            let points = ring_names
                .iter()
                .filter_map(|name| structure.find_atom(&residue.id, name))
                .filter_map(|id| structure.atom(id).map(|atom| atom.position))
                .collect::<Vec<_>>();
            if points.len() < 5 {
                continue;
            }
            let center = [
                points.iter().map(|point| point.x).sum::<f64>() / points.len() as f64,
                points.iter().map(|point| point.y).sum::<f64>() / points.len() as f64,
                points.iter().map(|point| point.z).sum::<f64>() / points.len() as f64,
            ];
            let offsets = points
                .into_iter()
                .map(|point| {
                    [
                        point.x - center[0],
                        point.y - center[1],
                        point.z - center[2],
                    ]
                })
                .collect::<Vec<_>>();
            template_groups
                .entry(residue.name.clone())
                .or_default()
                .push(offsets);
        }
        let templates = template_groups
            .into_iter()
            .filter_map(|(name, groups)| groups.into_iter().next().map(|offsets| (name, offsets)))
            .collect::<Vec<_>>();

        // A small deterministic SO(3) bank is sufficient at crystallographic
        // resolution: the local peak provides translation, while a coarse
        // tilt/azimuth/roll product covers the complete ring frame.  The old
        // bank omitted roll, which made an otherwise correctly seeded ring
        // score like unrelated protein density whenever its in-plane
        // orientation differed from the originating GlycoShape pose.
        let orientations = (0..3)
            .flat_map(|tilt_index| {
                let tilt = [-60.0_f64, 0.0, 60.0][tilt_index].to_radians();
                (0..6).flat_map(move |azimuth_index| {
                    let azimuth = (azimuth_index as f64 * 60.0).to_radians();
                    (0..6).map(move |roll_index| {
                        let roll = (roll_index as f64 * 60.0).to_radians();
                        let (sr, cr) = roll.sin_cos();
                        let (st, ct) = tilt.sin_cos();
                        let (sa, ca) = azimuth.sin_cos();
                        // Rz(azimuth) Ry(tilt) Rx(roll), expressed row-major.
                        let rotation = [
                            [ca * ct, -sa * cr + ca * st * sr, sa * sr + ca * st * cr],
                            [sa * ct, ca * cr + sa * st * sr, -ca * sr + sa * st * cr],
                            [-st, ct * sr, ct * cr],
                        ];
                        let quaternion = quaternion_from_euler(roll, tilt, azimuth);
                        (rotation, quaternion)
                    })
                })
            })
            .collect::<Vec<_>>();

        // Keep a wider seed pool than the final hypothesis budget.  A dense
        // protein slab can contain many stronger local maxima than a weak but
        // chemically real distal sugar; truncating at 128 before matched
        // filtering made that distal component disappear permanently.
        let mut peak_seeds = peaks.iter().copied().take(512).collect::<Vec<_>>();
        if let Some(anchor) = anchor_center {
            let mut nearest = peaks.clone();
            nearest.sort_by(|left, right| {
                let left_cart = self.map.grid_to_cartesian([
                    left.1[0] as f64,
                    left.1[1] as f64,
                    left.1[2] as f64,
                ]);
                let right_cart = self.map.grid_to_cartesian([
                    right.1[0] as f64,
                    right.1[1] as f64,
                    right.1[2] as f64,
                ]);
                self.map
                    .map_distance(anchor, left_cart, self.options.periodic)
                    .total_cmp(
                        &self
                            .map
                            .map_distance(anchor, right_cart, self.options.periodic),
                    )
                    .then_with(|| right.0.total_cmp(&left.0))
            });
            peak_seeds.extend(nearest.into_iter().take(1024));
        } else {
            peak_seeds.extend(peaks.iter().copied().skip(512).take(1024));
        }
        peak_seeds.sort_by(|left, right| left.1.cmp(&right.1));
        peak_seeds.dedup_by(|left, right| left.1 == right.1);
        peak_seeds.truncate(1536);
        // The strongest residual voxel is not always inside a weak distal
        // ring (B6 in the cached 5KZC map is a useful example).  Add a small,
        // topology-sized translation lattice around every native ring centre
        // as proposal seeds.  These are not deposited-coordinate restraints:
        // they are only reachable-volume anchors supplied by the requested
        // chemical template, and the matched filter still has to find
        // positive complete-ring evidence.  Keeping the lattice finite makes
        // this a capture stage rather than another continuous optimizer.
        let local_offsets = [-8.0_f64, -4.0, 0.0, 4.0, 8.0];
        let mut forced_ring_seeds = Vec::<(f64, [isize; 3])>::new();
        for residue in structure.residues() {
            if !residues.contains(&residue.id) {
                continue;
            }
            let points = ring_names
                .iter()
                .filter_map(|name| structure.find_atom(&residue.id, name))
                .filter_map(|id| structure.atom(id).map(|atom| atom.position))
                .collect::<Vec<_>>();
            if points.len() < 5 {
                continue;
            }
            let center = [
                points.iter().map(|point| point.x).sum::<f64>() / points.len() as f64,
                points.iter().map(|point| point.y).sum::<f64>() / points.len() as f64,
                points.iter().map(|point| point.z).sum::<f64>() / points.len() as f64,
            ];
            for dx in local_offsets {
                for dy in local_offsets {
                    for dz in local_offsets {
                        let grid = self.map.cartesian_to_grid([
                            center[0] + dx,
                            center[1] + dy,
                            center[2] + dz,
                        ]);
                        forced_ring_seeds.push((
                            0.0,
                            [
                                grid[0].round() as isize,
                                grid[1].round() as isize,
                                grid[2].round() as isize,
                            ],
                        ));
                    }
                }
            }
        }
        peak_seeds.extend(forced_ring_seeds);
        peak_seeds.sort_by(|left, right| left.1.cmp(&right.1));
        peak_seeds.dedup_by(|left, right| left.1 == right.1);
        peak_seeds.truncate(4096);
        let mut candidates = Vec::<DensityRingHypothesis>::new();
        // The first residue is chemically anchored to Asn.  Add its observed
        // ring pose as a protected seed so a strong symmetry-related copy
        // cannot displace the only hypothesis that can start a valid tree.
        if let Some(root_residue) = attachment_root.as_ref()
            && let Some(root_record) = structure
                .residues()
                .into_iter()
                .find(|residue| residue.id == *root_residue)
        {
            let root_points = ["C1", "C2", "C3", "C4", "C5", "O5"]
                .iter()
                .filter_map(|name| structure.find_atom(root_residue, name))
                .filter_map(|id| structure.atom(id).map(|atom| atom.position))
                .collect::<Vec<_>>();
            if root_points.len() >= 5 {
                let center = [
                    root_points.iter().map(|point| point.x).sum::<f64>() / root_points.len() as f64,
                    root_points.iter().map(|point| point.y).sum::<f64>() / root_points.len() as f64,
                    root_points.iter().map(|point| point.z).sum::<f64>() / root_points.len() as f64,
                ];
                let score = root_points
                    .iter()
                    .filter_map(|point| {
                        let point = [point.x, point.y, point.z];
                        self.map
                            .value_at_cartesian(point, self.options.periodic)
                            .map(|value| (value - self.protein_model_at(point)) / rms)
                    })
                    .sum::<f64>()
                    / root_points.len() as f64;
                candidates.push(DensityRingHypothesis {
                    site: target.site.clone(),
                    component_id: 0,
                    center_angstrom: center,
                    normalized_score: score,
                    local_support: score.max(0.0),
                    orientation_quaternion: [1.0, 0.0, 0.0, 0.0],
                    compatible_residues: vec![root_record.name],
                    uniqueness: 1.0,
                    provenance: "attachment_anchored_ring".into(),
                });
            }
        }
        candidates.extend(
            peak_seeds
                .par_iter()
                .flat_map_iter(|(_peak_score, grid)| {
                    let mut local_candidates = Vec::<DensityRingHypothesis>::new();
                    let peak = self.map.grid_to_cartesian([
                        grid[0] as f64,
                        grid[1] as f64,
                        grid[2] as f64,
                    ]);
                    for (residue_name, offsets) in &templates {
                        for (rotation, quaternion) in &orientations {
                            // Align each ring atom to the seed peak, then score the
                            // complete ring.  This is a matched filter rather than
                            // an atom-local peak detector, so adjacent residues are
                            // retained even when their strongest atoms are nearby.
                            let mut best_score = f64::NEG_INFINITY;
                            let mut best_center = [0.0; 3];
                            for offset in offsets {
                                let rotated = rotate_template_offset(*rotation, *offset);
                                let center = [
                                    peak[0] - rotated[0],
                                    peak[1] - rotated[1],
                                    peak[2] - rotated[2],
                                ];
                                let mut score = 0.0;
                                let mut valid = 0usize;
                                for template_offset in offsets {
                                    let shifted =
                                        rotate_template_offset(*rotation, *template_offset);
                                    let point = [
                                        center[0] + shifted[0],
                                        center[1] + shifted[1],
                                        center[2] + shifted[2],
                                    ];
                                    let Some(value) =
                                        self.map.value_at_cartesian(point, self.options.periodic)
                                    else {
                                        continue;
                                    };
                                    let residual = (value - self.protein_model_at(point)) / rms;
                                    score += residual;
                                    valid += 1;
                                }
                                if valid >= 5 {
                                    score /= valid as f64;
                                    if score > best_score {
                                        best_score = score;
                                        best_center = center;
                                    }
                                }
                            }
                            let matched_filter_floor = if weak_ring_capture { 0.0 } else { 0.10 };
                            if !best_score.is_finite() || best_score < matched_filter_floor {
                                continue;
                            }
                            local_candidates.push(DensityRingHypothesis {
                                site: target.site.clone(),
                                component_id: 0,
                                center_angstrom: best_center,
                                normalized_score: best_score,
                                local_support: best_score.max(0.0),
                                orientation_quaternion: *quaternion,
                                compatible_residues: vec![residue_name.clone()],
                                uniqueness: (best_score / (best_score + 1.0)).clamp(0.0, 1.0),
                                provenance: "native_ring_matched_filter".into(),
                            });
                        }
                    }
                    local_candidates.into_iter()
                })
                .collect::<Vec<_>>(),
        );
        // If the matched filter is too weak for a particularly noisy crop,
        // retain the old residual maxima as a deterministic rescue source.
        if candidates.is_empty() {
            for (score, grid) in peak_seeds.into_iter().take(maximum) {
                let center =
                    self.map
                        .grid_to_cartesian([grid[0] as f64, grid[1] as f64, grid[2] as f64]);
                candidates.push(DensityRingHypothesis {
                    site: target.site.clone(),
                    component_id: 0,
                    center_angstrom: center,
                    normalized_score: score,
                    local_support: score.max(0.0),
                    orientation_quaternion: self
                        .map
                        .value_gradient_at_cartesian(center, self.options.periodic)
                        .map(|(_, gradient)| quaternion_from_z_axis(gradient))
                        .unwrap_or([1.0, 0.0, 0.0, 0.0]),
                    compatible_residues: vec!["NAG".into(), "MAN".into()],
                    uniqueness: (score / (score + 1.0)).clamp(0.0, 1.0),
                    provenance: "native_residual_peak_rescue".into(),
                });
            }
        }
        // Reconcile the protected attachment seed with the strongest
        // compatible map-derived ring in its local reachable shell.  The
        // seed structure supplies the linkage chemistry, but its initial
        // torsions can place C1 several Å from the observed ring.  Keeping
        // the seed's stale centre would protect the wrong origin even though
        // the matched filter has already found the correct local component.
        if let (Some(anchor), Some(root_residue)) = (anchor_center, attachment_root.as_ref()) {
            let root_name = structure
                .residues()
                .into_iter()
                .find(|residue| residue.id == *root_residue)
                .map(|residue| residue.name.clone());
            let best_local = candidates
                .iter()
                .filter(|candidate| candidate.provenance == "native_ring_matched_filter")
                .filter(|candidate| {
                    root_name.as_ref().is_some_and(|name| {
                        candidate
                            .compatible_residues
                            .iter()
                            .any(|candidate_name| candidate_name.eq_ignore_ascii_case(name))
                    })
                })
                .filter(|candidate| {
                    self.map
                        .map_distance(anchor, candidate.center_angstrom, self.options.periodic)
                        <= 6.0
                })
                .max_by(|left, right| {
                    left.normalized_score
                        .total_cmp(&right.normalized_score)
                        .then_with(|| {
                            left.center_angstrom
                                .partial_cmp(&right.center_angstrom)
                                .unwrap()
                        })
                })
                .cloned();
            if let (Some(best_local), Some(protected)) = (
                best_local,
                candidates
                    .iter_mut()
                    .find(|candidate| candidate.provenance == "attachment_anchored_ring"),
            ) {
                if best_local.normalized_score > protected.normalized_score + 0.25 {
                    protected.center_angstrom = best_local.center_angstrom;
                    protected.normalized_score = best_local.normalized_score;
                    protected.local_support = best_local.local_support;
                    protected.orientation_quaternion = best_local.orientation_quaternion;
                }
            }
        }
        candidates.sort_by(|left, right| {
            right
                .normalized_score
                .total_cmp(&left.normalized_score)
                .then_with(|| {
                    left.center_angstrom
                        .partial_cmp(&right.center_angstrom)
                        .unwrap()
                })
        });
        let mut result = Vec::new();
        let mut coarse_bin_counts = BTreeMap::<[isize; 3], usize>::new();
        // Keep the detector site-local.  A periodic EDS map can contain a
        // symmetry-related copy whose atom-local peaks are stronger than the
        // requested glycan.  The chemistry-derived seed structure provides a
        // conservative reachable-radius envelope; candidates outside it are
        // not eligible to consume the finite hypothesis budget.  This is
        // independent of deposited coordinates and is deliberately generous
        // enough to retain folded arms and weak distal density.
        let reachable_radius = anchor_center.map(|anchor| {
            let observed_radius = structure
                .residues()
                .into_iter()
                .filter(|residue| residues.contains(&residue.id))
                .filter_map(|residue| {
                    let points = ["C1", "C2", "C3", "C4", "C5", "O5"]
                        .iter()
                        .filter_map(|name| structure.find_atom(&residue.id, name))
                        .filter_map(|id| structure.atom(id).map(|atom| atom.position))
                        .map(|point| [point.x, point.y, point.z])
                        .collect::<Vec<_>>();
                    points
                        .into_iter()
                        .map(|point| self.map.map_distance(anchor, point, self.options.periodic))
                        .max_by(f64::total_cmp)
                })
                .max_by(f64::total_cmp)
                .unwrap_or(12.0);
            // Add a full ring diameter plus a positional/refinement margin.
            // The lower bound keeps short or incomplete seed structures from
            // over-pruning a real branch; the upper bound excludes distant
            // periodic copies for ordinary N-glycans.
            (observed_radius + 8.0).clamp(18.0, 26.0)
        });
        // Insert the protected attachment hypothesis before score-ranked
        // maxima consume the finite result budget.  The root is a graph
        // constraint, not merely another local maximum: it must remain
        // available even when a dense protein/symmetry copy produces more
        // than `maximum` higher-scoring filters.
        if let Some(mut anchor) = candidates
            .iter()
            .find(|candidate| candidate.provenance == "attachment_anchored_ring")
            .cloned()
        {
            anchor.component_id = 0;
            let bin = anchor
                .center_angstrom
                .map(|value| (value / 8.0).floor() as isize);
            coarse_bin_counts.insert(bin, 1);
            result.push(anchor);
        }
        for mut candidate in candidates {
            if candidate.provenance == "attachment_anchored_ring" {
                continue;
            }
            if let (Some(anchor), Some(radius)) = (anchor_center, reachable_radius)
                && self
                    .map
                    .map_distance(anchor, candidate.center_angstrom, self.options.periodic)
                    > radius
            {
                continue;
            }
            let bin = candidate
                .center_angstrom
                .map(|value| (value / 8.0).floor() as isize);
            let bin_count = coarse_bin_counts.entry(bin).or_default();
            // Preserve weak hypotheses from different parts of the topology
            // envelope while allowing several residue-compatible orientations
            // in one local component.
            if *bin_count >= 8 {
                continue;
            }
            if result.iter().any(|existing: &DensityRingHypothesis| {
                self.map.map_distance(
                    candidate.center_angstrom,
                    existing.center_angstrom,
                    self.options.periodic,
                ) < 1.75
                    && candidate
                        .compatible_residues
                        .iter()
                        .any(|name| existing.compatible_residues.contains(name))
            }) {
                continue;
            }
            *bin_count += 1;
            candidate.component_id = result.len();
            result.push(candidate);
            if result.len() >= maximum {
                break;
            }
        }
        Ok(result)
    }

    /// Return the heavy atoms and their calculated field on a fixed ROI.
    ///
    /// The old implementation walked a rectangular map box around every
    /// atom and looked each voxel up in a hash map.  That is attractive for a
    /// sparse full-unit-cell mask, but is needlessly expensive for the small
    /// dense ROI used by optimization.  Walking the already materialized ROI
    /// once per atom is both simpler and substantially cheaper (and makes the
    /// same traversal available to the analytic-gradient scorer).
    fn fixed_region_field(
        &self,
        region: &DensityFixedRegion,
        structure: &Structure,
        targets: &[DensityTarget],
    ) -> Result<(Vec<glysys::StructureAtom>, Vec<f64>)> {
        let residues = targets
            .iter()
            .flat_map(|target| target.glycan_residues.iter().cloned())
            .collect::<BTreeSet<_>>();
        let atoms = structure
            .atoms()
            .into_iter()
            .filter(|atom| {
                residues.contains(&atom.residue) && !atom.element.eq_ignore_ascii_case("H")
            })
            .collect::<Vec<_>>();
        if atoms.is_empty() {
            return Err(DensityError::EmptyTarget);
        }
        let mut calculated = vec![0.0_f64; region.voxels.len()];
        let cutoff = 3.5 * (region.sigma_angstrom + 1.5);
        let cutoff2 = cutoff * cutoff;
        // Periodic minimum-image search: `cartesian(delta + t)` equals
        // `cartesian(delta) + cartesian(t)` by linearity, so the 27 lattice
        // translations can be precomputed once per field instead of running a
        // full 3x3x3 matrix multiply per voxel-atom pair.
        let translations = if region.periodic {
            let mut table = Vec::with_capacity(27);
            for i in -1..=1 {
                for j in -1..=1 {
                    for k in -1..=1 {
                        table.push(
                            self.map
                                .fractional_to_cartesian([i as f64, j as f64, k as f64]),
                        );
                    }
                }
            }
            table
        } else {
            Vec::new()
        };
        // For the usual orthogonal crystallographic cell, wrapping each
        // fractional component to [-0.5, 0.5] is the exact minimum-image
        // displacement.  The previous generic path still tested all 27
        // lattice images for every atom/voxel pair; on a 30k-voxel ROI that
        // made a single root tie-break needlessly expensive.  Keep the full
        // 27-image fallback for skew cells, but use the direct image when
        // the cell metric is orthogonal to numerical precision.
        let periodic_orthogonal = if region.periodic {
            // The MRC angles are the authoritative cell metric.  Using them
            // avoids treating harmless matrix round-off (or a MAP axis
            // permutation) as a skew cell and accidentally re-entering the
            // 27-image fallback.
            self.map
                .metadata()
                .cell_angles_degrees
                .iter()
                .all(|angle| (*angle - 90.0).abs() <= 1.0e-5)
        } else {
            false
        };
        for atom in &atoms {
            let center = [atom.position.x, atom.position.y, atom.position.z];
            let atom_fractional = self.map.cartesian_to_fractional([
                center[0] - self.map.metadata().origin_angstrom[0],
                center[1] - self.map.metadata().origin_angstrom[1],
                center[2] - self.map.metadata().origin_angstrom[2],
            ]);
            let sigma = self.atom_sigma(region.sigma_angstrom, atom);
            let sigma2 = sigma * sigma;
            let amplitude = self.scoring_atom_amplitude(atom);
            for (index, voxel) in region.voxels.iter().enumerate() {
                let displacement = if region.periodic {
                    let mut fractional = [
                        voxel.fractional[0] - atom_fractional[0],
                        voxel.fractional[1] - atom_fractional[1],
                        voxel.fractional[2] - atom_fractional[2],
                    ];
                    for component in &mut fractional {
                        *component -= component.round();
                    }
                    let base = self.map.fractional_to_cartesian(fractional);
                    if periodic_orthogonal {
                        [
                            base[0] * base[0] + base[1] * base[1] + base[2] * base[2],
                            0.0,
                            0.0,
                        ]
                    } else {
                        let mut best = f64::INFINITY;
                        for translation in &translations {
                            let candidate = [
                                base[0] + translation[0],
                                base[1] + translation[1],
                                base[2] + translation[2],
                            ];
                            let norm = candidate.iter().map(|value| value * value).sum::<f64>();
                            if norm < best {
                                best = norm;
                            }
                        }
                        if best.is_infinite() {
                            [f64::INFINITY, 0.0, 0.0]
                        } else {
                            [best.sqrt(), 0.0, 0.0]
                        }
                    }
                } else {
                    [
                        voxel.cartesian[0] - center[0],
                        voxel.cartesian[1] - center[1],
                        voxel.cartesian[2] - center[2],
                    ]
                };
                let distance2 = displacement.iter().map(|value| value * value).sum::<f64>();
                if distance2 <= cutoff2 {
                    calculated[index] += amplitude * (-distance2 / (2.0 * sigma2)).exp();
                }
            }
        }
        Ok((atoms, calculated))
    }

    /// Build the calculated Gaussian field for a fixed set of placed
    /// residues inside an already-constructed region.  Used by Adaptive arm
    /// screens so the conserved prefix splat is computed once per arm rather
    /// than once per candidate mode.
    pub fn fixed_region_prefix_field(
        &self,
        region: &DensityFixedRegion,
        structure: &Structure,
        targets: &[DensityTarget],
    ) -> Result<Vec<f64>> {
        if targets
            .iter()
            .all(|target| target.glycan_residues.is_empty())
        {
            return Ok(vec![0.0; region.voxels.len()]);
        }
        self.fixed_region_field(region, structure, targets)
            .map(|(_, field)| field)
    }

    pub fn score_fixed_region(
        &self,
        region: &DensityFixedRegion,
        structure: &Structure,
        targets: &[DensityTarget],
    ) -> Result<DensityFixedScore> {
        let (_, calculated) = self.fixed_region_field(region, structure, targets)?;
        let samples = region
            .voxels
            .iter()
            .enumerate()
            .map(|(index, voxel)| {
                (
                    calculated[index],
                    voxel.observed,
                    voxel.weight,
                    voxel.background,
                )
            })
            .collect::<Vec<_>>();
        let agreement = agreement_statistics(&samples)?;
        let (training_samples, heldout_samples): (Vec<_>, Vec<_>) = region
            .voxels
            .iter()
            .enumerate()
            .map(|(index, voxel)| {
                (
                    voxel.grid,
                    (
                        calculated[index],
                        voxel.observed,
                        voxel.weight,
                        voxel.background,
                    ),
                )
            })
            .partition(|(grid, _)| spatial_fold(*grid) != 0);
        let training = agreement_statistics(
            &training_samples
                .into_iter()
                .map(|(_, sample)| sample)
                .collect::<Vec<_>>(),
        )
        .unwrap_or(agreement);
        let heldout = agreement_statistics(
            &heldout_samples
                .into_iter()
                .map(|(_, sample)| sample)
                .collect::<Vec<_>>(),
        )
        .unwrap_or(agreement);
        Ok(DensityFixedScore {
            correlation: agreement.correlation,
            protein_adjusted_correlation: agreement.protein_adjusted_correlation,
            likelihood_gain: agreement.likelihood_gain,
            training_likelihood_gain: training.likelihood_gain,
            heldout_likelihood_gain: heldout.likelihood_gain,
            bic_gain: agreement.bic_gain,
            voxel_count: samples.len(),
        })
    }

    /// Score a candidate residue against the density left after subtracting a
    /// fitted glycan prefix.  The ROI and map background are fixed exactly as
    /// in [`score_fixed_region`], but the observed samples are replaced with
    /// `observed - protein - prefix_field`.  This prevents a child from
    /// receiving credit for a peak already explained by its parent or sibling
    /// and is the density equivalent of freezing an accepted frontier.
    pub fn score_fixed_region_residual(
        &self,
        region: &DensityFixedRegion,
        structure: &Structure,
        placed_targets: &[DensityTarget],
        candidate_targets: &[DensityTarget],
    ) -> Result<DensityFixedScore> {
        let prefix = if placed_targets
            .iter()
            .any(|target| !target.glycan_residues.is_empty())
        {
            self.fixed_region_field(region, structure, placed_targets)?
                .1
        } else {
            vec![0.0; region.voxels.len()]
        };
        self.score_fixed_region_residual_with_prefix(region, structure, candidate_targets, &prefix)
    }

    /// Residual score with a caller-provided fixed prefix field.  The placed
    /// prefix is identical for every proposal that shares the same conserved
    /// residues, so Adaptive arm screens compute it once per arm instead of
    /// rebuilding the splat for each native mode.
    pub fn score_fixed_region_residual_with_prefix(
        &self,
        region: &DensityFixedRegion,
        structure: &Structure,
        candidate_targets: &[DensityTarget],
        prefix: &[f64],
    ) -> Result<DensityFixedScore> {
        if prefix.len() != region.voxels.len() {
            return Err(DensityError::Geometry(
                "fixed-region prefix field length must match the region voxels".into(),
            ));
        }
        let (_, calculated) = self.fixed_region_field(region, structure, candidate_targets)?;
        let samples = region
            .voxels
            .iter()
            .enumerate()
            .map(|(index, voxel)| {
                (
                    calculated[index],
                    voxel.observed - voxel.background - prefix[index],
                    voxel.weight,
                    0.0,
                )
            })
            .collect::<Vec<_>>();
        let agreement = agreement_statistics(&samples)?;
        let (training_samples, heldout_samples): (Vec<_>, Vec<_>) = region
            .voxels
            .iter()
            .enumerate()
            .map(|(index, voxel)| {
                (
                    voxel.grid,
                    (
                        calculated[index],
                        voxel.observed - voxel.background - prefix[index],
                        voxel.weight,
                        0.0,
                    ),
                )
            })
            .partition(|(grid, _)| spatial_fold(*grid) != 0);
        let training = agreement_statistics(
            &training_samples
                .into_iter()
                .map(|(_, sample)| sample)
                .collect::<Vec<_>>(),
        )
        .unwrap_or(agreement);
        let heldout = agreement_statistics(
            &heldout_samples
                .into_iter()
                .map(|(_, sample)| sample)
                .collect::<Vec<_>>(),
        )
        .unwrap_or(agreement);
        Ok(DensityFixedScore {
            correlation: agreement.correlation,
            protein_adjusted_correlation: agreement.protein_adjusted_correlation,
            likelihood_gain: agreement.likelihood_gain,
            training_likelihood_gain: training.likelihood_gain,
            heldout_likelihood_gain: heldout.likelihood_gain,
            bic_gain: agreement.bic_gain,
            voxel_count: samples.len(),
        })
    }

    /// Exact coordinate derivative of the profiled fixed-ROI likelihood.
    /// Linear scale and intercept derivatives vanish by the envelope theorem;
    /// only each atom-local Gaussian derivative remains.
    pub fn score_fixed_region_with_gradients(
        &self,
        region: &DensityFixedRegion,
        structure: &Structure,
        targets: &[DensityTarget],
    ) -> Result<DensityFixedGradientScore> {
        // Build the field once and derive both the value and coordinate
        // derivatives from it.  Calling `score_fixed_region` here used to
        // repeat the complete Gaussian splat before starting the gradient
        // traversal, which made every trust-region iteration needlessly
        // expensive.
        let (atoms, calculated) = self.fixed_region_field(region, structure, targets)?;
        let cutoff = 3.5 * (region.sigma_angstrom + 1.5);
        let samples = region
            .voxels
            .iter()
            .enumerate()
            .map(|(index, voxel)| {
                (
                    calculated[index],
                    voxel.observed,
                    voxel.weight,
                    voxel.background,
                )
            })
            .collect::<Vec<_>>();
        let coefficients = profiled_likelihood_derivatives(&samples)?;
        let mut atom_gradients = BTreeMap::new();
        let cutoff2 = cutoff * cutoff;
        let translations = if region.periodic {
            let mut table = Vec::with_capacity(27);
            for i in -1..=1 {
                for j in -1..=1 {
                    for k in -1..=1 {
                        table.push(
                            self.map
                                .fractional_to_cartesian([i as f64, j as f64, k as f64]),
                        );
                    }
                }
            }
            table
        } else {
            Vec::new()
        };
        for atom in atoms {
            let center = [atom.position.x, atom.position.y, atom.position.z];
            let atom_fractional = self.map.cartesian_to_fractional([
                center[0] - self.map.metadata().origin_angstrom[0],
                center[1] - self.map.metadata().origin_angstrom[1],
                center[2] - self.map.metadata().origin_angstrom[2],
            ]);
            let sigma = self.atom_sigma(region.sigma_angstrom, &atom);
            let sigma2 = sigma * sigma;
            let amplitude = self.scoring_atom_amplitude(&atom);
            let mut gradient = [0.0; 3];
            for (index, voxel) in region.voxels.iter().enumerate() {
                let displacement = if region.periodic {
                    let mut fractional = [
                        voxel.fractional[0] - atom_fractional[0],
                        voxel.fractional[1] - atom_fractional[1],
                        voxel.fractional[2] - atom_fractional[2],
                    ];
                    for component in &mut fractional {
                        *component -= component.round();
                    }
                    let base = self.map.fractional_to_cartesian(fractional);
                    let mut best = f64::INFINITY;
                    for translation in &translations {
                        let candidate = [
                            base[0] + translation[0],
                            base[1] + translation[1],
                            base[2] + translation[2],
                        ];
                        let norm = candidate.iter().map(|value| value * value).sum::<f64>();
                        if norm < best {
                            best = norm;
                        }
                    }
                    if best.is_infinite() {
                        [f64::INFINITY, 0.0, 0.0]
                    } else {
                        [best.sqrt(), 0.0, 0.0]
                    }
                } else {
                    [
                        voxel.cartesian[0] - center[0],
                        voxel.cartesian[1] - center[1],
                        voxel.cartesian[2] - center[2],
                    ]
                };
                let distance2 = displacement.iter().map(|value| value * value).sum::<f64>();
                if distance2 > cutoff2 {
                    continue;
                }
                let field = amplitude * (-distance2 / (2.0 * sigma2)).exp();
                for axis in 0..3 {
                    gradient[axis] += coefficients[index] * field * displacement[axis] / sigma2;
                }
            }
            atom_gradients.insert(atom.id, gradient);
        }
        let agreement = agreement_statistics(&samples)?;
        let (training_samples, heldout_samples): (Vec<_>, Vec<_>) = region
            .voxels
            .iter()
            .enumerate()
            .map(|(index, voxel)| {
                (
                    voxel.grid,
                    (
                        calculated[index],
                        voxel.observed,
                        voxel.weight,
                        voxel.background,
                    ),
                )
            })
            .partition(|(grid, _)| spatial_fold(*grid) != 0);
        let training = agreement_statistics(
            &training_samples
                .into_iter()
                .map(|(_, sample)| sample)
                .collect::<Vec<_>>(),
        )
        .unwrap_or(agreement);
        let heldout = agreement_statistics(
            &heldout_samples
                .into_iter()
                .map(|(_, sample)| sample)
                .collect::<Vec<_>>(),
        )
        .unwrap_or(agreement);
        let score = DensityFixedScore {
            correlation: agreement.correlation,
            protein_adjusted_correlation: agreement.protein_adjusted_correlation,
            likelihood_gain: agreement.likelihood_gain,
            training_likelihood_gain: training.likelihood_gain,
            heldout_likelihood_gain: heldout.likelihood_gain,
            bic_gain: agreement.bic_gain,
            voxel_count: samples.len(),
        };
        Ok(DensityFixedGradientScore {
            score,
            atom_gradients,
        })
    }

    pub fn score(&self, structure: &Structure, targets: &[DensityTarget]) -> Result<DensityScore> {
        if targets.is_empty() {
            return Err(DensityError::EmptyTarget);
        }
        let sigma = self.options.sigma()?;
        let atom_groups = targets
            .iter()
            .map(|target| {
                let target = if target.glycan_residues.is_empty() {
                    DensityTarget::for_site(structure, &target.site)?
                } else {
                    target.clone()
                };
                let atoms = structure
                    .atoms()
                    .into_iter()
                    .filter(|atom| target.glycan_residues.contains(&atom.residue))
                    .collect::<Vec<_>>();
                if atoms.is_empty() {
                    Err(DensityError::EmptyTarget)
                } else {
                    Ok((target, atoms))
                }
            })
            .collect::<Result<Vec<_>>>()?;

        let mut union_samples = BTreeMap::<[isize; 3], (f64, f64, f64, f64)>::new();
        let mut site_scores = Vec::with_capacity(atom_groups.len());
        let mut aggregate_residue_support = Vec::new();
        let mut aggregate_linkage_support = Vec::new();
        for (target, atoms) in &atom_groups {
            let samples = self.samples_for_atoms(atoms, sigma)?;
            let site_values = samples
                .iter()
                .map(|(_, calculated, observed, weight, background)| {
                    (*calculated, *observed, *weight, *background)
                })
                .collect::<Vec<_>>();
            let agreement = agreement_statistics(&site_values)?;
            let correlation = agreement.correlation;
            let atom_support = atoms
                .iter()
                .map(|atom| {
                    let map_value = self.map.sample(
                        [atom.position.x, atom.position.y, atom.position.z],
                        self.options.periodic,
                    );
                    let normalized_value = map_value.map(|value| {
                        let scale = self.map.metadata.rms.abs().max(1.0e-12);
                        let residual = value
                            - self.protein_model_at([
                                atom.position.x,
                                atom.position.y,
                                atom.position.z,
                            ]);
                        (residual - self.map.metadata.dmean) / scale
                    });
                    DensityAtomSupport {
                        residue: atom.residue.clone(),
                        atom: atom.name.clone(),
                        map_value,
                        normalized_value,
                    }
                })
                .collect::<Vec<_>>();
            let heavy_atom_count = atoms
                .iter()
                .filter(|atom| !atom.element.eq_ignore_ascii_case("H"))
                .count();
            let heavy_supported_count = atoms
                .iter()
                .zip(&atom_support)
                .filter(|(atom, support)| {
                    !atom.element.eq_ignore_ascii_case("H")
                        && support
                            .normalized_value
                            .is_some_and(|value| value >= self.options.support_threshold)
                })
                .count();
            let supported_atom_fraction = if heavy_atom_count == 0 {
                0.0
            } else {
                heavy_supported_count as f64 / heavy_atom_count as f64
            };
            let ring_support = self.ring_support(atoms);
            let connectivity_support = self.connectivity_support(structure, target);
            let (residue_support, linkage_support) =
                self.residue_support(structure, target, atoms, &atom_support);
            let voxel_count = site_values.len();
            for (key, calculated, observed, weight, background) in samples {
                let entry = union_samples
                    .entry(key)
                    .or_insert((0.0, observed, 0.0, background));
                entry.0 += calculated;
                entry.1 = observed;
                entry.2 = entry.2.max(weight);
                entry.3 = entry.3.max(background);
            }
            site_scores.push(DensitySiteScore {
                site: target.site.clone(),
                correlation,
                protein_adjusted_correlation: agreement.protein_adjusted_correlation,
                likelihood_gain: agreement.likelihood_gain,
                bic_gain: agreement.bic_gain,
                voxel_count,
                supported_atom_fraction,
                ring_support,
                connectivity_support,
                atoms: atom_support,
                residue_support: residue_support.clone(),
                linkage_support: linkage_support.clone(),
            });
            aggregate_residue_support.extend(residue_support);
            aggregate_linkage_support.extend(linkage_support);
        }
        let all_samples = union_samples.values().copied().collect::<Vec<_>>();
        let combined_agreement = agreement_statistics(&all_samples)?;
        let combined = combined_agreement.correlation;
        let combined_adjusted = combined_agreement.protein_adjusted_correlation;
        let supported_atom_fraction = if site_scores.is_empty() {
            0.0
        } else {
            site_scores
                .iter()
                .map(|site| site.supported_atom_fraction)
                .sum::<f64>()
                / site_scores.len() as f64
        };
        let ring_support = if site_scores.is_empty() {
            0.0
        } else {
            site_scores
                .iter()
                .map(|site| site.ring_support)
                .sum::<f64>()
                / site_scores.len() as f64
        };
        let connectivity_support = if site_scores.is_empty() {
            0.0
        } else {
            site_scores
                .iter()
                .map(|site| site.connectivity_support)
                .sum::<f64>()
                / site_scores.len() as f64
        };
        Ok(DensityScore {
            correlation: combined,
            protein_adjusted_correlation: combined_adjusted,
            likelihood_gain: combined_agreement.likelihood_gain,
            training_likelihood_gain: combined_agreement.likelihood_gain,
            heldout_likelihood_gain: combined_agreement.likelihood_gain,
            difference_score: 0.0,
            bic_gain: combined_agreement.bic_gain,
            sites: site_scores,
            sigma_angstrom: sigma,
            mask_radius_angstrom: self.options.mask_radius_angstrom,
            mask_falloff_angstrom: self.options.mask_falloff_angstrom,
            voxel_count: all_samples.len(),
            periodic: self.options.periodic,
            supported_atom_fraction,
            ring_support,
            connectivity_support,
            residue_support: aggregate_residue_support,
            linkage_support: aggregate_linkage_support,
            support_threshold: self.options.support_threshold,
        })
    }

    pub fn fast_evidence(
        &self,
        structure: &Structure,
        targets: &[DensityTarget],
    ) -> Result<DensityFastEvidence> {
        if targets.is_empty() {
            return Err(DensityError::EmptyTarget);
        }
        let residue_ids = targets
            .iter()
            .map(|target| {
                if target.glycan_residues.is_empty() {
                    DensityTarget::for_site(structure, &target.site)
                } else {
                    Ok(target.clone())
                }
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flat_map(|target| target.glycan_residues)
            .collect::<BTreeSet<_>>();
        let atoms = structure
            .atoms()
            .into_iter()
            .filter(|atom| {
                residue_ids.contains(&atom.residue) && !atom.element.eq_ignore_ascii_case("H")
            })
            .collect::<Vec<_>>();
        if atoms.is_empty() {
            return Err(DensityError::EmptyTarget);
        }
        let scale = self.map.metadata.rms.abs().max(1.0e-12);
        let total_weight = atoms
            .iter()
            .map(|atom| f64::from(element_weight(&atom.element)).sqrt())
            .sum::<f64>()
            .max(1.0e-12);
        let mut score = 0.0;
        let mut atom_gradients = BTreeMap::new();
        for atom in atoms {
            let Some((value, gradient)) = self.map.value_gradient_at_cartesian(
                [atom.position.x, atom.position.y, atom.position.z],
                self.options.periodic,
            ) else {
                continue;
            };
            let weight = f64::from(element_weight(&atom.element)).sqrt() / total_weight
                * self.ownership_weight([atom.position.x, atom.position.y, atom.position.z]);
            let residual =
                value - self.protein_model_at([atom.position.x, atom.position.y, atom.position.z]);
            let normalized = (residual - self.map.metadata.dmean) / scale;
            // Saturation prevents one extreme protein-adjacent peak from
            // dominating the full glycan and keeps the gradient well scaled.
            let transformed = (normalized / 3.0).tanh();
            score += weight * transformed;
            let derivative = weight * (1.0 - transformed * transformed) / (3.0 * scale);
            atom_gradients.insert(
                atom.id,
                [
                    derivative * gradient[0],
                    derivative * gradient[1],
                    derivative * gradient[2],
                ],
            );
        }
        Ok(DensityFastEvidence {
            score,
            atom_gradients,
        })
    }

    /// Indexed-coordinate equivalent of [`fast_evidence`].  The caller
    /// supplies immutable atom metadata plus current positions, allowing the
    /// residue/arm proposal loop to preserve the exact robust atom-local
    /// objective without cloning a complete `Structure`.
    pub fn fast_evidence_indexed(
        &self,
        atoms: &[DensityIndexedAtom],
    ) -> Result<DensityFastEvidence> {
        if atoms.is_empty() {
            return Err(DensityError::EmptyTarget);
        }
        let atoms = atoms
            .iter()
            .filter(|atom| !atom.element.eq_ignore_ascii_case("H"))
            .collect::<Vec<_>>();
        if atoms.is_empty() {
            return Err(DensityError::EmptyTarget);
        }
        let scale = self.map.metadata.rms.abs().max(1.0e-12);
        let total_weight = atoms
            .iter()
            .map(|atom| f64::from(element_weight(&atom.element)).sqrt())
            .sum::<f64>()
            .max(1.0e-12);
        let mut score = 0.0;
        let mut atom_gradients = BTreeMap::new();
        for atom in atoms {
            let Some((value, gradient)) = self
                .map
                .value_gradient_at_cartesian(atom.position, self.options.periodic)
            else {
                continue;
            };
            let weight = f64::from(element_weight(&atom.element)).sqrt() / total_weight
                * self.ownership_weight(atom.position);
            let residual = value - self.protein_model_at(atom.position);
            let normalized = (residual - self.map.metadata.dmean) / scale;
            let transformed = (normalized / 3.0).tanh();
            score += weight * transformed;
            let derivative = weight * (1.0 - transformed * transformed) / (3.0 * scale);
            atom_gradients.insert(
                atom.id,
                [
                    derivative * gradient[0],
                    derivative * gradient[1],
                    derivative * gradient[2],
                ],
            );
        }
        Ok(DensityFastEvidence {
            score,
            atom_gradients,
        })
    }

    /// Scalar-only indexed evidence for proposal ranking.  The gradient map
    /// in [`fast_evidence_indexed`] is intentionally retained for local
    /// refinement, but constructing a `BTreeMap` and evaluating a full map
    /// gradient for every atom is unnecessary during the thousands of root
    /// and linkage screens.  This routine uses the identical robust residual
    /// transform and normalization while doing only the scalar map lookup.
    pub fn fast_evidence_score_indexed(&self, atoms: &[DensityIndexedAtom]) -> Result<f64> {
        if atoms.is_empty() {
            return Err(DensityError::EmptyTarget);
        }
        let atoms = atoms
            .iter()
            .filter(|atom| !atom.element.eq_ignore_ascii_case("H"))
            .collect::<Vec<_>>();
        if atoms.is_empty() {
            return Err(DensityError::EmptyTarget);
        }
        let scale = self.map.metadata.rms.abs().max(1.0e-12);
        let total_weight = atoms
            .iter()
            .map(|atom| f64::from(element_weight(&atom.element)).sqrt())
            .sum::<f64>()
            .max(1.0e-12);
        let mut score = 0.0;
        for atom in atoms {
            let Some(value) = self.map.sample(atom.position, self.options.periodic) else {
                continue;
            };
            let weight = f64::from(element_weight(&atom.element)).sqrt() / total_weight
                * self.ownership_weight(atom.position);
            let residual = value - self.protein_model_at(atom.position);
            let normalized = (residual - self.map.metadata.dmean) / scale;
            score += weight * (normalized / 3.0).tanh();
        }
        Ok(score)
    }

    /// Score an explicit heavy-atom subset against the protein-subtracted
    /// map.  Residue-level evidence is deliberately the default, but local
    /// exocyclic torsion searches need to distinguish a C5--C6/O6 signal from
    /// the much larger pyranose ring contribution.  Keeping this operation in
    /// the density crate guarantees it uses the same periodic sampling,
    /// protein background, element weighting, and robust transform as
    /// `fast_evidence`.
    pub fn atom_local_evidence(
        &self,
        structure: &Structure,
        atom_ids: &std::collections::BTreeSet<AtomId>,
    ) -> Result<DensityFastEvidence> {
        let atoms = structure
            .atoms()
            .into_iter()
            .filter(|atom| atom_ids.contains(&atom.id) && !atom.element.eq_ignore_ascii_case("H"))
            .collect::<Vec<_>>();
        if atoms.is_empty() {
            return Err(DensityError::EmptyTarget);
        }
        let scale = self.map.metadata.rms.abs().max(1.0e-12);
        let total_weight = atoms
            .iter()
            .map(|atom| f64::from(element_weight(&atom.element)).sqrt())
            .sum::<f64>()
            .max(1.0e-12);
        let mut score = 0.0;
        let mut atom_gradients = BTreeMap::new();
        for atom in atoms {
            let Some((value, gradient)) = self.map.value_gradient_at_cartesian(
                [atom.position.x, atom.position.y, atom.position.z],
                self.options.periodic,
            ) else {
                continue;
            };
            let weight = f64::from(element_weight(&atom.element)).sqrt() / total_weight
                * self.ownership_weight([atom.position.x, atom.position.y, atom.position.z]);
            let residual =
                value - self.protein_model_at([atom.position.x, atom.position.y, atom.position.z]);
            let normalized = (residual - self.map.metadata.dmean) / scale;
            let transformed = (normalized / 3.0).tanh();
            score += weight * transformed;
            let derivative = weight * (1.0 - transformed * transformed) / (3.0 * scale);
            atom_gradients.insert(
                atom.id,
                [
                    derivative * gradient[0],
                    derivative * gradient[1],
                    derivative * gradient[2],
                ],
            );
        }
        Ok(DensityFastEvidence {
            score,
            atom_gradients,
        })
    }

    /// Score a newly placed residue against the map left after subtracting
    /// the already accepted glycan prefix.  This is intentionally a dense,
    /// atom-local operation: it samples only the candidate atoms and computes
    /// the small prefix field at those points.  The caller can therefore
    /// grow a tree residue-by-residue without rebuilding a full masked map or
    /// allocating a complete voxel buffer for every proposal.
    ///
    /// `placed_residues` and `candidate_residues` are disjoint sets.  The
    /// protein field is fixed by [`attach_protein_background`], while the
    /// Gaussian width and amplitude are identical to the exact scorer.  A
    /// residual score is deliberately independent of the candidate's own
    /// mask: a misplaced sugar cannot move its ROI onto a convenient peak.
    pub fn fast_residual_evidence(
        &self,
        structure: &Structure,
        placed_residues: &BTreeSet<ResidueId>,
        candidate_residues: &BTreeSet<ResidueId>,
    ) -> Result<DensityFastEvidence> {
        if placed_residues.is_disjoint(candidate_residues) == false {
            return Err(DensityError::Geometry(
                "placed and candidate residue sets must be disjoint".into(),
            ));
        }
        let candidate_atoms = structure
            .atoms()
            .into_iter()
            .filter(|atom| {
                candidate_residues.contains(&atom.residue)
                    && !atom.element.eq_ignore_ascii_case("H")
            })
            .collect::<Vec<_>>();
        if candidate_atoms.is_empty() {
            return Err(DensityError::EmptyTarget);
        }
        let placed_atoms = structure
            .atoms()
            .into_iter()
            .filter(|atom| {
                placed_residues.contains(&atom.residue)
                    && !candidate_residues.contains(&atom.residue)
                    && !atom.element.eq_ignore_ascii_case("H")
            })
            .collect::<Vec<_>>();
        let sigma = self.options.sigma()?;
        let map_scale = self.map.metadata.rms.abs().max(1.0e-12);
        let total_weight = candidate_atoms
            .iter()
            .map(|atom| f64::from(element_weight(&atom.element)).sqrt())
            .sum::<f64>()
            .max(1.0e-12);
        let mut score = 0.0;
        let mut gradients = BTreeMap::new();
        for atom in candidate_atoms {
            let point = [atom.position.x, atom.position.y, atom.position.z];
            let observed = self
                .map
                .sample(point, self.options.periodic)
                .unwrap_or(self.map.metadata.dmean);
            let mut prefix_field = self.protein_model_at(point);
            for placed in &placed_atoms {
                let center = [placed.position.x, placed.position.y, placed.position.z];
                let displacement = self
                    .map
                    .map_displacement(center, point, self.options.periodic);
                let distance2 = displacement.iter().map(|value| value * value).sum::<f64>();
                let sigma_atom = self.atom_sigma(sigma, placed);
                let amplitude = self.scoring_atom_amplitude(placed);
                prefix_field += amplitude * (-distance2 / (2.0 * sigma_atom * sigma_atom)).exp();
            }
            let normalized = (observed - prefix_field - self.map.metadata.dmean) / map_scale;
            let transformed = (normalized / 2.5).tanh();
            let weight = f64::from(element_weight(&atom.element)).sqrt() / total_weight
                * self.ownership_weight([atom.position.x, atom.position.y, atom.position.z]);
            score += weight * transformed;
            // The gradient is with respect to the candidate atom coordinate.
            // It is useful for the local trust-region refinement; the
            // conservative residual score remains the acceptance authority.
            if let Some((_, map_gradient)) = self
                .map
                .value_gradient_at_cartesian(point, self.options.periodic)
            {
                let derivative = weight * (1.0 - transformed * transformed) / (2.5 * map_scale);
                gradients.insert(
                    atom.id,
                    [
                        derivative * map_gradient[0],
                        derivative * map_gradient[1],
                        derivative * map_gradient[2],
                    ],
                );
            }
        }
        Ok(DensityFastEvidence {
            score,
            atom_gradients: gradients,
        })
    }

    /// Indexed variant of [`fast_residual_evidence`] for hot proposal loops.
    /// The caller owns the atom metadata and supplies only the current
    /// coordinates, allowing the scorer to avoid cloning the protein model.
    pub fn fast_residual_evidence_indexed(
        &self,
        placed_atoms: &[DensityIndexedAtom],
        candidate_atoms: &[DensityIndexedAtom],
    ) -> Result<DensityFastEvidence> {
        if candidate_atoms.is_empty() {
            return Err(DensityError::EmptyTarget);
        }
        let sigma = self.options.sigma()?;
        let map_scale = self.map.metadata.rms.abs().max(1.0e-12);
        let total_weight = candidate_atoms
            .iter()
            .map(|atom| f64::from(element_weight(&atom.element)).sqrt())
            .sum::<f64>()
            .max(1.0e-12);
        let mut score = 0.0;
        let mut gradients = BTreeMap::new();
        for atom in candidate_atoms {
            let point = atom.position;
            let observed = self
                .map
                .sample(point, self.options.periodic)
                .unwrap_or(self.map.metadata.dmean);
            let mut prefix_field = self.protein_model_at(point);
            for placed in placed_atoms {
                let displacement =
                    self.map
                        .map_displacement(placed.position, point, self.options.periodic);
                let distance2 = displacement.iter().map(|value| value * value).sum::<f64>();
                let sigma_atom = self.indexed_atom_sigma(sigma, placed);
                let amplitude = if self.kernel_profile.element_weighted
                    || self.options.glycan_b_factor.is_some()
                {
                    indexed_atom_amplitude(placed)
                } else {
                    placed.occupancy.clamp(0.0, 1.0)
                };
                prefix_field += amplitude * (-distance2 / (2.0 * sigma_atom * sigma_atom)).exp();
            }
            let normalized = (observed - prefix_field - self.map.metadata.dmean) / map_scale;
            let transformed = (normalized / 2.5).tanh();
            let weight = f64::from(element_weight(&atom.element)).sqrt() / total_weight
                * self.ownership_weight(point);
            score += weight * transformed;
            if let Some((_, map_gradient)) = self
                .map
                .value_gradient_at_cartesian(point, self.options.periodic)
            {
                let derivative = weight * (1.0 - transformed * transformed) / (2.5 * map_scale);
                gradients.insert(
                    atom.id,
                    [
                        derivative * map_gradient[0],
                        derivative * map_gradient[1],
                        derivative * map_gradient[2],
                    ],
                );
            }
        }
        Ok(DensityFastEvidence {
            score,
            atom_gradients: gradients,
        })
    }

    /// Signed support from a static Fo-Fc map. Negative density is a stronger
    /// veto than positive density is a reward; the primary map remains the
    /// fitting authority.
    pub fn signed_difference_evidence(
        &self,
        structure: &Structure,
        targets: &[DensityTarget],
    ) -> Result<DensitySignedEvidence> {
        let residues = targets
            .iter()
            .map(|target| {
                if target.glycan_residues.is_empty() {
                    DensityTarget::for_site(structure, &target.site)
                } else {
                    Ok(target.clone())
                }
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flat_map(|target| target.glycan_residues)
            .collect::<BTreeSet<_>>();
        let atoms = structure
            .atoms()
            .into_iter()
            .filter(|atom| {
                residues.contains(&atom.residue) && !atom.element.eq_ignore_ascii_case("H")
            })
            .collect::<Vec<_>>();
        if atoms.is_empty() {
            return Err(DensityError::EmptyTarget);
        }
        let normalization = atoms
            .iter()
            .map(|atom| f64::from(element_weight(&atom.element)).sqrt())
            .sum::<f64>()
            .max(1.0e-12);
        let rms = self.map.metadata.rms.abs().max(1.0e-12);
        let mut result = DensitySignedEvidence::default();
        for atom in atoms {
            let Some((value, map_gradient)) = self.map.value_gradient_at_cartesian(
                [atom.position.x, atom.position.y, atom.position.z],
                self.options.periodic,
            ) else {
                continue;
            };
            let weight = f64::from(element_weight(&atom.element)).sqrt() / normalization
                * self.ownership_weight([atom.position.x, atom.position.y, atom.position.z]);
            let normalized = value / rms;
            let magnitude = (normalized.abs() / 3.0).tanh();
            let coefficient = if normalized >= 0.0 { 0.35 } else { 1.0 };
            if normalized >= 0.0 {
                result.positive_reward += weight * magnitude;
                result.score += 0.35 * weight * magnitude;
            } else {
                result.negative_penalty += weight * magnitude;
                result.score -= weight * magnitude;
            }
            let transformed = (normalized.abs() / 3.0).tanh();
            let derivative = coefficient * weight * (1.0 - transformed * transformed) / (3.0 * rms);
            result.atom_gradients.insert(
                atom.id,
                [
                    derivative * map_gradient[0],
                    derivative * map_gradient[1],
                    derivative * map_gradient[2],
                ],
            );
        }
        Ok(result)
    }

    fn interpolate_connection(&self, first: [f64; 3], second: [f64; 3], fraction: f64) -> [f64; 3] {
        if !self.options.periodic {
            return [
                first[0] + fraction * (second[0] - first[0]),
                first[1] + fraction * (second[1] - first[1]),
                first[2] + fraction * (second[2] - first[2]),
            ];
        }
        let first_shifted = [
            first[0] - self.map.metadata.origin_angstrom[0],
            first[1] - self.map.metadata.origin_angstrom[1],
            first[2] - self.map.metadata.origin_angstrom[2],
        ];
        let second_shifted = [
            second[0] - self.map.metadata.origin_angstrom[0],
            second[1] - self.map.metadata.origin_angstrom[1],
            second[2] - self.map.metadata.origin_angstrom[2],
        ];
        let first_fractional = self.map.cartesian_to_fractional(first_shifted);
        let second_fractional = self.map.cartesian_to_fractional(second_shifted);
        let delta = self.map.minimum_image_fractional([
            second_fractional[0] - first_fractional[0],
            second_fractional[1] - first_fractional[1],
            second_fractional[2] - first_fractional[2],
        ]);
        let shifted = self.map.fractional_to_cartesian([
            first_fractional[0] + fraction * delta[0],
            first_fractional[1] + fraction * delta[1],
            first_fractional[2] + fraction * delta[2],
        ]);
        [
            shifted[0] + self.map.metadata.origin_angstrom[0],
            shifted[1] + self.map.metadata.origin_angstrom[1],
            shifted[2] + self.map.metadata.origin_angstrom[2],
        ]
    }

    fn ring_support(&self, atoms: &[glysys::StructureAtom]) -> f64 {
        // GlySys keeps carbohydrate atom names in the conventional PDB form.
        // Include both ring-atom samples and a ring-centre sample.  The latter
        // captures a continuous density blob even when one atom is noisy or
        // absent from an experimental model.
        let mut rings = BTreeMap::<ResidueId, Vec<&glysys::StructureAtom>>::new();
        for atom in atoms {
            if matches!(
                atom.name.trim().to_ascii_uppercase().as_str(),
                "C1" | "C2" | "C3" | "C4" | "C5" | "O5"
            ) {
                rings.entry(atom.residue.clone()).or_default().push(atom);
            }
        }
        let mut positive = 0usize;
        let mut total = 0usize;
        for ring in rings.values() {
            for atom in ring {
                total += 1;
                if self
                    .normalized_residual([atom.position.x, atom.position.y, atom.position.z])
                    .is_some_and(|value| value >= self.options.support_threshold)
                {
                    positive += 1;
                }
            }
            if ring.len() >= 3 {
                let center = self.ring_center(ring);
                total += 1;
                if self
                    .normalized_residual(center)
                    .is_some_and(|value| value >= self.options.support_threshold)
                {
                    positive += 1;
                }
            }
        }
        if total == 0 {
            0.0
        } else {
            positive as f64 / total as f64
        }
    }

    fn normalized_residual(&self, cartesian: [f64; 3]) -> Option<f64> {
        let value = self.map.sample(cartesian, self.options.periodic)?;
        let scale = self.map.metadata.rms.abs().max(1.0e-12);
        let ownership = self.ownership_weight(cartesian).max(0.05);
        Some(
            (value - self.protein_model_at(cartesian) - self.map.metadata.dmean) * ownership
                / scale,
        )
    }

    /// Return the protein-subtracted, map-normalized density at a Cartesian
    /// point.  This small read-only primitive is used by the frontier fitter
    /// for linkage-prefix checks: a distal ring is not considered supported
    /// when the parent-to-child corridor is missing, even if the ring itself
    /// happens to sit on an unrelated positive peak.
    pub fn normalized_residual_at_cartesian(&self, cartesian: [f64; 3]) -> Option<f64> {
        self.normalized_residual(cartesian)
    }

    /// Calculate continuous per-residue and per-linkage evidence.  The
    /// topology is reconstructed from the atom-bond graph and the attachment
    /// bond; residue ordering in a PDB/mmCIF record is never used as a proxy
    /// for parent/child relationships.
    fn residue_support(
        &self,
        structure: &Structure,
        target: &DensityTarget,
        atoms: &[glysys::StructureAtom],
        atom_support: &[DensityAtomSupport],
    ) -> (Vec<DensityResidueSupport>, Vec<DensityLinkageSupport>) {
        let threshold = self.options.support_threshold;
        let target_ids = target
            .glycan_residues
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let mut atoms_by_residue = BTreeMap::<ResidueId, Vec<usize>>::new();
        for (index, atom) in atoms.iter().enumerate() {
            atoms_by_residue
                .entry(atom.residue.clone())
                .or_default()
                .push(index);
        }
        let normalized = |value: Option<f64>| value.map_or(0.0, |value| value.max(0.0));
        let mut atom_fraction = BTreeMap::new();
        for residue in &target.glycan_residues {
            let indexes = atoms_by_residue.get(residue).cloned().unwrap_or_default();
            let heavy = indexes
                .iter()
                .filter(|index| !atoms[**index].element.eq_ignore_ascii_case("H"))
                .count();
            let supported = indexes
                .iter()
                .filter(|index| {
                    !atoms[**index].element.eq_ignore_ascii_case("H")
                        && normalized(atom_support[**index].normalized_value) >= threshold
                })
                .count();
            atom_fraction.insert(
                residue.clone(),
                if heavy == 0 {
                    0.0
                } else {
                    supported as f64 / heavy as f64
                },
            );
        }
        let mut ring_fraction = BTreeMap::new();
        for residue in &target.glycan_residues {
            let ring = atoms_by_residue
                .get(residue)
                .into_iter()
                .flat_map(|indexes| indexes.iter())
                .filter_map(|index| {
                    let atom = &atoms[*index];
                    if matches!(
                        atom.name.trim().to_ascii_uppercase().as_str(),
                        "C1" | "C2" | "C3" | "C4" | "C5" | "O5"
                    ) {
                        Some(atom)
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>();
            let mut positive = 0usize;
            let mut total = 0usize;
            for atom in &ring {
                total += 1;
                if self
                    .normalized_residual([atom.position.x, atom.position.y, atom.position.z])
                    .is_some_and(|value| value >= threshold)
                {
                    positive += 1;
                }
            }
            if ring.len() >= 3 {
                let center = self.ring_center(&ring);
                total += 1;
                if self
                    .normalized_residual(center)
                    .is_some_and(|value| value >= threshold)
                {
                    positive += 1;
                }
            }
            ring_fraction.insert(
                residue.clone(),
                if total == 0 {
                    0.0
                } else {
                    positive as f64 / total as f64
                },
            );
        }

        let atom_by_id = structure
            .atoms()
            .into_iter()
            .map(|atom| (atom.id, atom))
            .collect::<BTreeMap<_, _>>();
        let mut graph = BTreeMap::<ResidueId, BTreeSet<ResidueId>>::new();
        let mut edge_samples = BTreeMap::<(ResidueId, ResidueId), Vec<[f64; 3]>>::new();
        for (left, right) in structure.bonds() {
            let (Some(left_atom), Some(right_atom)) =
                (atom_by_id.get(&left), atom_by_id.get(&right))
            else {
                continue;
            };
            if left_atom.residue == right_atom.residue
                || !target_ids.contains(&left_atom.residue)
                || !target_ids.contains(&right_atom.residue)
            {
                continue;
            }
            graph
                .entry(left_atom.residue.clone())
                .or_default()
                .insert(right_atom.residue.clone());
            graph
                .entry(right_atom.residue.clone())
                .or_default()
                .insert(left_atom.residue.clone());
            let pair = if left_atom.residue <= right_atom.residue {
                (left_atom.residue.clone(), right_atom.residue.clone())
            } else {
                (right_atom.residue.clone(), left_atom.residue.clone())
            };
            for fraction in [0.25_f64, 0.5, 0.75] {
                edge_samples
                    .entry(pair.clone())
                    .or_default()
                    .push(self.interpolate_connection(
                        [
                            left_atom.position.x,
                            left_atom.position.y,
                            left_atom.position.z,
                        ],
                        [
                            right_atom.position.x,
                            right_atom.position.y,
                            right_atom.position.z,
                        ],
                        fraction,
                    ));
            }
        }
        let root = structure
            .metadata()
            .glycosylation_sites
            .iter()
            .find(|site| site.protein_residue == target.site)
            .map(|site| site.glycan_residue.clone())
            .filter(|residue| target_ids.contains(residue))
            .or_else(|| target.glycan_residues.iter().min().cloned());
        let mut parent = BTreeMap::<ResidueId, ResidueId>::new();
        let mut depth = BTreeMap::<ResidueId, usize>::new();
        if let Some(root) = root.clone() {
            let mut queue = std::collections::VecDeque::from([root.clone()]);
            depth.insert(root, 0);
            while let Some(current) = queue.pop_front() {
                let current_depth = depth[&current];
                for neighbour in graph.get(&current).into_iter().flat_map(|set| set.iter()) {
                    if !depth.contains_key(neighbour) {
                        parent.insert(neighbour.clone(), current.clone());
                        depth.insert(neighbour.clone(), current_depth + 1);
                        queue.push_back(neighbour.clone());
                    }
                }
            }
        }
        let mut linkage_support = Vec::new();
        let mut incoming_support = BTreeMap::<ResidueId, Vec<f64>>::new();
        let mut outgoing_support = BTreeMap::<ResidueId, Vec<f64>>::new();
        for (child, parent_residue) in &parent {
            let pair = if child <= parent_residue {
                (child.clone(), parent_residue.clone())
            } else {
                (parent_residue.clone(), child.clone())
            };
            let samples = edge_samples.get(&pair).cloned().unwrap_or_default();
            let support = if samples.is_empty() {
                0.0
            } else {
                samples
                    .iter()
                    .map(|point| {
                        self.normalized_residual(*point)
                            .unwrap_or(0.0)
                            .max(0.0)
                            .min(1.0)
                    })
                    .sum::<f64>()
                    / samples.len() as f64
            };
            incoming_support
                .entry(child.clone())
                .or_default()
                .push(support);
            outgoing_support
                .entry(parent_residue.clone())
                .or_default()
                .push(support);
            linkage_support.push(DensityLinkageSupport {
                donor: parent_residue.clone(),
                acceptor: child.clone(),
                support,
                boundary: support < threshold,
            });
        }
        linkage_support.sort_by(|left, right| {
            left.donor
                .cmp(&right.donor)
                .then_with(|| left.acceptor.cmp(&right.acceptor))
        });
        let mut residue_support = target
            .glycan_residues
            .iter()
            .map(|residue| {
                let atom = *atom_fraction.get(residue).unwrap_or(&0.0);
                let ring = *ring_fraction.get(residue).unwrap_or(&0.0);
                let connection_values = incoming_support
                    .get(residue)
                    .into_iter()
                    .flat_map(|values| values.iter())
                    .chain(
                        outgoing_support
                            .get(residue)
                            .into_iter()
                            .flat_map(|values| values.iter()),
                    )
                    .copied()
                    .collect::<Vec<_>>();
                let connection = if connection_values.is_empty() {
                    0.0
                } else {
                    connection_values.iter().sum::<f64>() / connection_values.len() as f64
                };
                let root_residue = root.as_ref() == Some(residue);
                let denominator = if root_residue { 0.8 } else { 1.0 };
                let confidence = (0.5 * atom + 0.3 * ring + 0.2 * connection) / denominator;
                DensityResidueSupport {
                    residue: residue.clone(),
                    atom_support: atom,
                    ring_support: ring,
                    connection_support: connection,
                    confidence,
                    supported: false,
                }
            })
            .collect::<Vec<_>>();
        // The parent-gating test above needs the completed confidence table;
        // apply it in depth order once all raw confidences are available.
        residue_support
            .sort_by_key(|support| depth.get(&support.residue).copied().unwrap_or(usize::MAX));
        let mut supported = BTreeMap::new();
        for support in &mut residue_support {
            let root_residue = root.as_ref() == Some(&support.residue);
            support.supported = support.confidence >= threshold
                && (root_residue
                    || parent.get(&support.residue).is_some_and(|parent_residue| {
                        supported.get(parent_residue).copied().unwrap_or(false)
                    }));
            supported.insert(support.residue.clone(), support.supported);
        }
        (residue_support, linkage_support)
    }

    fn ring_center(&self, ring: &[&glysys::StructureAtom]) -> [f64; 3] {
        if !self.options.periodic {
            let center = ring.iter().fold([0.0; 3], |mut center, atom| {
                center[0] += atom.position.x;
                center[1] += atom.position.y;
                center[2] += atom.position.z;
                center
            });
            let count = ring.len() as f64;
            return [center[0] / count, center[1] / count, center[2] / count];
        }
        let origin = self.map.metadata.origin_angstrom;
        let first = ring[0].position;
        let anchor = self.map.cartesian_to_fractional([
            first.x - origin[0],
            first.y - origin[1],
            first.z - origin[2],
        ]);
        let mut sum = anchor;
        for atom in &ring[1..] {
            let position = atom.position;
            let fractional = self.map.cartesian_to_fractional([
                position.x - origin[0],
                position.y - origin[1],
                position.z - origin[2],
            ]);
            let delta = self.map.minimum_image_fractional([
                fractional[0] - anchor[0],
                fractional[1] - anchor[1],
                fractional[2] - anchor[2],
            ]);
            sum[0] += anchor[0] + delta[0];
            sum[1] += anchor[1] + delta[1];
            sum[2] += anchor[2] + delta[2];
        }
        let count = ring.len() as f64;
        let fractional = [sum[0] / count, sum[1] / count, sum[2] / count];
        let shifted = self.map.fractional_to_cartesian(fractional);
        [
            shifted[0] + origin[0],
            shifted[1] + origin[1],
            shifted[2] + origin[2],
        ]
    }

    fn connectivity_support(&self, structure: &Structure, target: &DensityTarget) -> f64 {
        let target_ids = target
            .glycan_residues
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let atom_by_id = structure
            .atoms()
            .into_iter()
            .map(|atom| (atom.id, atom))
            .collect::<BTreeMap<_, _>>();
        let mut positive = 0usize;
        let mut total = 0usize;
        for (left, right) in structure.bonds() {
            let Some(a) = atom_by_id.get(&left) else {
                continue;
            };
            let Some(b) = atom_by_id.get(&right) else {
                continue;
            };
            if a.residue == b.residue
                || !target_ids.contains(&a.residue)
                || !target_ids.contains(&b.residue)
            {
                continue;
            }
            for fraction in [0.25_f64, 0.5, 0.75] {
                let point = self.interpolate_connection(
                    [a.position.x, a.position.y, a.position.z],
                    [b.position.x, b.position.y, b.position.z],
                    fraction,
                );
                total += 1;
                if self
                    .normalized_residual(point)
                    .is_some_and(|value| value >= self.options.support_threshold)
                {
                    positive += 1;
                }
            }
        }
        if total == 0 {
            0.0
        } else {
            positive as f64 / total as f64
        }
    }

    fn samples_for_atoms(
        &self,
        atoms: &[glysys::StructureAtom],
        sigma: f64,
    ) -> Result<Vec<([isize; 3], f64, f64, f64, f64)>> {
        self.samples_for_atoms_with_radius(
            atoms,
            sigma,
            self.options.mask_radius_angstrom,
            self.options.mask_falloff_angstrom,
        )
    }

    fn samples_for_atoms_with_radius(
        &self,
        atoms: &[glysys::StructureAtom],
        sigma: f64,
        mask_radius_angstrom: f64,
        mask_falloff_angstrom: f64,
    ) -> Result<Vec<([isize; 3], f64, f64, f64, f64)>> {
        // Hydrogen positions in GlycoShape/GLYCAM templates are useful for
        // stereochemistry but are not independently resolved by ordinary
        // macromolecular X-ray maps. Including them also makes the mask and
        // objective depend on protonation conventions, so density agreement
        // is deliberately heavy-atom only.
        let atoms = atoms
            .iter()
            .filter(|atom| !atom.element.eq_ignore_ascii_case("H"))
            .collect::<Vec<_>>();
        if atoms.is_empty() {
            return Err(DensityError::EmptyTarget);
        }
        let radius = mask_radius_angstrom + mask_falloff_angstrom;
        let centers = atoms
            .iter()
            .map(|atom| {
                let position = [atom.position.x, atom.position.y, atom.position.z];
                if self.options.periodic {
                    self.map.canonical_cartesian(position)
                } else {
                    position
                }
            })
            .collect::<Vec<_>>();
        let shape = self.map.cartesian_shape();
        // Scan each atom's local mask instead of the entire glycan bounding
        // box. A branched glycan can make that box hundreds of times larger
        // than the union of its atom masks. Periodic keys are canonicalized
        // before calculating the site correlation as well as the union score;
        // this guarantees that a one-site union is exactly its site score.
        let inverse = &self.map.cartesian_to_fractional;
        let delta = [
            self.map.metadata.sampling[0] as f64 * radius * inverse.row(0).norm(),
            self.map.metadata.sampling[1] as f64 * radius * inverse.row(1).norm(),
            self.map.metadata.sampling[2] as f64 * radius * inverse.row(2).norm(),
        ];
        let mut accumulated = BTreeMap::<[isize; 3], f64>::new();
        for center in &centers {
            let grid_center = self.map.cartesian_to_grid(*center);
            let ranges = (0..3)
                .map(|axis| {
                    let start = (grid_center[axis] - delta[axis]).floor() as isize;
                    let end = (grid_center[axis] + delta[axis]).ceil() as isize;
                    if !self.options.periodic && (end < 0 || start >= shape[axis] as isize) {
                        return Err(DensityError::OutOfMap);
                    }
                    Ok(start..=end)
                })
                .collect::<Result<Vec<_>>>()?;
            for x in ranges[0].clone() {
                for y in ranges[1].clone() {
                    for z in ranges[2].clone() {
                        let cart = self.map.grid_to_cartesian([x as f64, y as f64, z as f64]);
                        let distance = self.map.map_distance(cart, *center, self.options.periodic);
                        let weight =
                            mask_weight(distance, mask_radius_angstrom, mask_falloff_angstrom)
                                * self.ownership_weight(cart);
                        if weight == 0.0 {
                            continue;
                        }
                        let key = if self.options.periodic {
                            [
                                x.rem_euclid(shape[0] as isize),
                                y.rem_euclid(shape[1] as isize),
                                z.rem_euclid(shape[2] as isize),
                            ]
                        } else {
                            [x, y, z]
                        };
                        let entry = accumulated.entry(key).or_insert(0.0);
                        *entry = entry.max(weight);
                    }
                }
            }
        }
        let mut samples = Vec::with_capacity(accumulated.len());
        for ([x, y, z], weight) in accumulated {
            let cart = self.map.grid_to_cartesian([x as f64, y as f64, z as f64]);
            let observed = self
                .map
                .sample_grid([x, y, z], self.options.periodic)
                .ok_or(DensityError::OutOfMap)?;
            let background = self.protein_model_at(cart);
            let calculated = atoms
                .iter()
                .zip(&centers)
                .map(|(atom, center)| {
                    let distance = self.map.map_distance(cart, *center, self.options.periodic);
                    let atom_sigma = self.atom_sigma(sigma, atom);
                    self.scoring_atom_amplitude(atom)
                        * (-distance * distance / (2.0 * atom_sigma * atom_sigma)).exp()
                })
                .sum::<f64>();
            samples.push(([x, y, z], calculated, observed, weight, background));
        }
        if samples.len() < 3 {
            return Err(DensityError::OutOfMap);
        }
        Ok(samples)
    }

    /// Write observed density, the calculated Gaussian atom field, and the
    /// exact soft mask used by this scorer on one small, overlay-ready grid.
    pub fn write_visualization_maps(
        &self,
        structure: &Structure,
        targets: &[DensityTarget],
        output_dir: impl AsRef<Path>,
        prefix: &str,
        margin_angstrom: f64,
    ) -> Result<DensityVisualizationMaps> {
        if !margin_angstrom.is_finite() || margin_angstrom < 0.0 {
            return Err(DensityError::Geometry(
                "density crop margin must be finite and non-negative".into(),
            ));
        }
        let target_specs = targets
            .iter()
            .map(|target| {
                if target.glycan_residues.is_empty() {
                    DensityTarget::for_site(structure, &target.site)
                } else {
                    Ok(target.clone())
                }
            })
            .collect::<Result<Vec<_>>>()?;
        if target_specs.is_empty() {
            return Err(DensityError::EmptyTarget);
        }
        let target_ids = target_specs
            .iter()
            .flat_map(|target| target.glycan_residues.iter())
            .cloned()
            .collect::<BTreeSet<_>>();
        let atoms = structure
            .atoms()
            .into_iter()
            .filter(|atom| target_ids.contains(&atom.residue))
            .collect::<Vec<_>>();
        if atoms.is_empty() {
            return Err(DensityError::EmptyTarget);
        }
        let sigma = self.options.sigma()?;
        let shape = self.map.cartesian_shape();
        let inverse = &self.map.cartesian_to_fractional;
        let radius = self.options.mask_radius_angstrom + self.options.mask_falloff_angstrom;
        let crop_delta = [
            (self.map.metadata.sampling[0] as f64
                * (margin_angstrom + radius)
                * inverse.row(0).norm())
            .ceil() as isize,
            (self.map.metadata.sampling[1] as f64
                * (margin_angstrom + radius)
                * inverse.row(1).norm())
            .ceil() as isize,
            (self.map.metadata.sampling[2] as f64
                * (margin_angstrom + radius)
                * inverse.row(2).norm())
            .ceil() as isize,
        ];
        let grid_positions = atoms
            .iter()
            .map(|atom| {
                self.map
                    .cartesian_to_grid([atom.position.x, atom.position.y, atom.position.z])
            })
            .collect::<Vec<_>>();
        let mut start = [0isize; 3];
        let mut end = [0isize; 3];
        for axis in 0..3 {
            let min = grid_positions
                .iter()
                .map(|grid| grid[axis])
                .fold(f64::INFINITY, f64::min)
                .floor() as isize
                - crop_delta[axis];
            let max = grid_positions
                .iter()
                .map(|grid| grid[axis])
                .fold(f64::NEG_INFINITY, f64::max)
                .ceil() as isize
                + crop_delta[axis];
            if !self.options.periodic && (min < 0 || max >= shape[axis] as isize) {
                return Err(DensityError::OutOfMap);
            }
            start[axis] = min;
            end[axis] = max;
        }
        let crop_shape = [
            (end[0] - start[0] + 1) as usize,
            (end[1] - start[1] + 1) as usize,
            (end[2] - start[2] + 1) as usize,
        ];
        let map_axes = self.map.map_axes_zero_based();
        let array_shape = [
            crop_shape[map_axes[0]],
            crop_shape[map_axes[1]],
            crop_shape[map_axes[2]],
        ];
        let array_start = [
            self.map.metadata().nxstart as isize + start[map_axes[0]],
            self.map.metadata().nystart as isize + start[map_axes[1]],
            self.map.metadata().nzstart as isize + start[map_axes[2]],
        ];
        let mut observed = vec![0.0_f32; array_shape[0] * array_shape[1] * array_shape[2]];
        let mut calculated = observed.clone();
        let mut mask = observed.clone();
        for az in 0..array_shape[2] {
            for ay in 0..array_shape[1] {
                for ax in 0..array_shape[0] {
                    let array = [ax, ay, az];
                    let mut cart_index = [0usize; 3];
                    for (array_axis, cartesian_axis) in map_axes.into_iter().enumerate() {
                        cart_index[cartesian_axis] = array[array_axis];
                    }
                    let grid = [
                        start[0] + cart_index[0] as isize,
                        start[1] + cart_index[1] as isize,
                        start[2] + cart_index[2] as isize,
                    ];
                    let cart = self.map.grid_to_cartesian([
                        grid[0] as f64,
                        grid[1] as f64,
                        grid[2] as f64,
                    ]);
                    let index = ax + array_shape[0] * (ay + array_shape[1] * az);
                    observed[index] =
                        self.map
                            .sample_grid(grid, self.options.periodic)
                            .ok_or(DensityError::OutOfMap)? as f32;
                    let mut model = 0.0;
                    let mut mask_value: f64 = 0.0;
                    for atom in &atoms {
                        let center = [atom.position.x, atom.position.y, atom.position.z];
                        let d = self.map.map_distance(cart, center, self.options.periodic);
                        let atom_sigma = self.atom_sigma(sigma, atom);
                        model += self.scoring_atom_amplitude(atom)
                            * (-d * d / (2.0 * atom_sigma * atom_sigma)).exp();
                        mask_value = mask_value.max(mask_weight(
                            d,
                            self.options.mask_radius_angstrom,
                            self.options.mask_falloff_angstrom,
                        ));
                    }
                    calculated[index] = model as f32;
                    mask[index] = mask_value as f32;
                }
            }
        }
        let output_dir = output_dir.as_ref();
        std::fs::create_dir_all(output_dir).map_err(|source| DensityError::Io {
            path: output_dir.to_path_buf(),
            source,
        })?;
        let observed_path = output_dir.join(format!("{prefix}-observed.ccp4"));
        let calculated_path = output_dir.join(format!("{prefix}-calculated.ccp4"));
        let mask_path = output_dir.join(format!("{prefix}-mask.ccp4"));
        let write = |path: &Path, values: Vec<f32>| -> Result<()> {
            let mut writer = mrc::create(path)
                .shape(array_shape)
                .sampling([
                    self.map.metadata().sampling[0] as i32,
                    self.map.metadata().sampling[1] as i32,
                    self.map.metadata().sampling[2] as i32,
                ])
                .cell_lengths(
                    self.map.metadata().cell_lengths_angstrom[0] as f32,
                    self.map.metadata().cell_lengths_angstrom[1] as f32,
                    self.map.metadata().cell_lengths_angstrom[2] as f32,
                )
                .cell_angles(
                    self.map.metadata().cell_angles_degrees[0] as f32,
                    self.map.metadata().cell_angles_degrees[1] as f32,
                    self.map.metadata().cell_angles_degrees[2] as f32,
                )
                .nstart([
                    array_start[0] as i32,
                    array_start[1] as i32,
                    array_start[2] as i32,
                ])
                .axis_mapping([
                    self.map.metadata().map_axes[0] as i32,
                    self.map.metadata().map_axes[1] as i32,
                    self.map.metadata().map_axes[2] as i32,
                ])
                .origin([
                    self.map.metadata().origin_angstrom[0] as f32,
                    self.map.metadata().origin_angstrom[1] as f32,
                    self.map.metadata().origin_angstrom[2] as f32,
                ])
                .mode::<f32>()
                .add_label("ReGlyco density visualization crop")
                .finish()
                .map_err(|error| DensityError::Map(error.to_string()))?;
            writer
                .write_block_as(
                    &mrc::VoxelBlock::new([0, 0, 0], array_shape, values)
                        .map_err(|error| DensityError::Map(error.to_string()))?,
                )
                .map_err(|error| DensityError::Map(error.to_string()))?;
            writer
                .update_header_stats()
                .map_err(|error| DensityError::Map(error.to_string()))?;
            writer
                .finalize()
                .map_err(|error| DensityError::Map(error.to_string()))?;
            Ok(())
        };
        write(&observed_path, observed)?;
        write(&calculated_path, calculated)?;
        write(&mask_path, mask)?;
        Ok(DensityVisualizationMaps {
            observed: observed_path,
            calculated: calculated_path,
            mask: mask_path,
            crop_start: start,
            crop_shape,
            margin_angstrom,
        })
    }
}

/// Compatibility options retained for the original deferred API.
#[derive(Debug, Clone, Default)]
pub struct DensityRefinementOptions {
    pub map_path: PathBuf,
    pub resolution_angstrom: Option<f64>,
    pub sigma_angstrom: Option<f64>,
}

pub fn refine_density(options: &DensityRefinementOptions) -> Result<DensityMap> {
    let map = DensityMap::open(&options.map_path)?;
    let mut score_options = DensityScoreOptions {
        sigma_angstrom: options.sigma_angstrom,
        ..DensityScoreOptions::default()
    };
    if score_options.sigma_angstrom.is_none()
        && let Some(resolution) = options.resolution_angstrom
    {
        score_options = score_options.with_resolution(resolution)?;
    }
    score_options.sigma()?;
    Ok(map)
}

fn cell_matrix(lengths: [f64; 3], angles: [f64; 3]) -> Result<Matrix3<f64>> {
    if lengths
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || angles
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0 || *value >= 180.0)
    {
        return Err(DensityError::Geometry(
            "unit-cell lengths and angles must be valid".into(),
        ));
    }
    let [alpha, beta, gamma] = angles.map(|value| value * PI / 180.0);
    let sin_gamma = gamma.sin();
    if sin_gamma.abs() < 1.0e-12 {
        return Err(DensityError::Geometry("gamma angle is singular".into()));
    }
    let a = Vector3::new(lengths[0], 0.0, 0.0);
    let b = Vector3::new(lengths[1] * gamma.cos(), lengths[1] * sin_gamma, 0.0);
    let c_x = lengths[2] * beta.cos();
    let c_y = lengths[2] * (alpha.cos() - beta.cos() * gamma.cos()) / sin_gamma;
    let c_z2 = lengths[2] * lengths[2] - c_x * c_x - c_y * c_y;
    if c_z2 <= 0.0 {
        return Err(DensityError::Geometry(
            "unit-cell vectors are invalid".into(),
        ));
    }
    let c = Vector3::new(c_x, c_y, c_z2.sqrt());
    Ok(Matrix3::from_columns(&[a, b, c]))
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path).map_err(|source| DensityError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let digest = Sha256::digest(bytes);
    Ok(format!("{digest:x}"))
}

fn voxel_statistics(values: &[f32]) -> [f64; 4] {
    let mut minimum = f64::INFINITY;
    let mut maximum = f64::NEG_INFINITY;
    let mut sum = 0.0;
    for &value in values {
        let value = value as f64;
        minimum = minimum.min(value);
        maximum = maximum.max(value);
        sum += value;
    }
    let mean = sum / values.len().max(1) as f64;
    let variance = values
        .iter()
        .map(|value| {
            let delta = *value as f64 - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len().max(1) as f64;
    [minimum, maximum, mean, variance.sqrt()]
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
}

fn mask_weight(distance: f64, core: f64, falloff: f64) -> f64 {
    if distance <= core {
        1.0
    } else if falloff == 0.0 || distance >= core + falloff {
        0.0
    } else {
        let fraction = (distance - core) / falloff;
        0.5 * (1.0 + (PI * fraction).cos())
    }
}

fn element_weight(element: &str) -> u8 {
    match element.trim().to_ascii_uppercase().as_str() {
        "H" => 1,
        "C" => 6,
        "N" => 7,
        "O" => 8,
        "F" => 9,
        "P" => 15,
        "S" => 16,
        "CL" => 17,
        "BR" => 35,
        "I" => 53,
        _ => 6,
    }
}

fn atom_amplitude(atom: &glysys::StructureAtom) -> f64 {
    f64::from(element_weight(&atom.element)) * atom.occupancy.clamp(0.0, 1.0)
}

fn effective_atom_sigma_indexed(
    map_sigma: f64,
    atom: &DensityIndexedAtom,
    fallback_b_factor: Option<f64>,
) -> f64 {
    let b = if atom.b_factor > 1.0e-6 {
        atom.b_factor
    } else {
        fallback_b_factor.unwrap_or(0.0)
    }
    .max(0.0);
    (map_sigma * map_sigma + b / (8.0 * PI * PI)).sqrt()
}

fn indexed_atom_amplitude(atom: &DensityIndexedAtom) -> f64 {
    f64::from(element_weight(&atom.element)) * atom.occupancy.clamp(0.0, 1.0)
}

#[derive(Debug, Clone, Copy)]
struct AgreementStatistics {
    correlation: f64,
    protein_adjusted_correlation: f64,
    likelihood_gain: f64,
    bic_gain: f64,
}

fn spatial_fold(grid: [isize; 3]) -> usize {
    // Two-voxel blocks keep immediately adjacent, strongly correlated map
    // samples in the same fold. Euclidean division makes the assignment
    // stable for negative periodic/unwrapped grid coordinates.
    (grid[0].div_euclid(2) + grid[1].div_euclid(2) + grid[2].div_euclid(2)).rem_euclid(4) as usize
}

fn agreement_statistics(samples: &[(f64, f64, f64, f64)]) -> Result<AgreementStatistics> {
    // Samples are (calculated glycan field, observed map, mask weight,
    // fitted protein model).
    if samples.len() < 3 {
        return Err(DensityError::ConstantSample);
    }
    let weight_sum = samples.iter().map(|(_, _, weight, _)| *weight).sum::<f64>();
    if !weight_sum.is_finite() || weight_sum <= 0.0 {
        return Err(DensityError::ConstantSample);
    }
    let mean_x = samples
        .iter()
        .map(|(x, _, weight, _)| x * weight)
        .sum::<f64>()
        / weight_sum;
    let mean_y = samples
        .iter()
        .map(|(_, y, weight, _)| y * weight)
        .sum::<f64>()
        / weight_sum;
    let mean_z = samples
        .iter()
        .map(|(_, _, weight, z)| z * weight)
        .sum::<f64>()
        / weight_sum;
    let (mut covariance, mut variance_x, mut variance_y, mut variance_z, mut covariance_zy) =
        (0.0, 0.0, 0.0, 0.0, 0.0);
    for (x, y, weight, z) in samples {
        covariance += weight * (x - mean_x) * (y - mean_y);
        variance_x += weight * (x - mean_x).powi(2);
        variance_y += weight * (y - mean_y).powi(2);
        variance_z += weight * (z - mean_z).powi(2);
        covariance_zy += weight * (z - mean_z) * (y - mean_y);
    }
    if variance_x <= 1.0e-20 || variance_y <= 1.0e-20 {
        return Err(DensityError::ConstantSample);
    }
    let correlation = covariance / (variance_x * variance_y).sqrt();
    // Remove the fitted protein model from the observed map; the glycan then
    // competes only against density the protein does not explain.
    let protein_scale = if variance_z > 1.0e-20 {
        covariance_zy / variance_z
    } else {
        0.0
    };
    let protein_intercept = mean_y - protein_scale * mean_z;
    let mut residuals = Vec::with_capacity(samples.len());
    let mut null_residual = 0.0;
    for (_, y, weight, z) in samples {
        let residual = y - protein_intercept - protein_scale * z;
        residuals.push(residual);
        null_residual += weight * residual * residual;
    }
    let mean_residual = residuals
        .iter()
        .zip(samples)
        .map(|(residual, (_, _, weight, _))| residual * weight)
        .sum::<f64>()
        / weight_sum;
    let (mut covariance_xr, mut variance_residual) = (0.0, 0.0);
    for (index, (x, _, weight, _)) in samples.iter().enumerate() {
        covariance_xr += weight * (x - mean_x) * (residuals[index] - mean_residual);
        variance_residual +=
            weight * (residuals[index] - mean_residual) * (residuals[index] - mean_residual);
    }
    let protein_adjusted_correlation = if variance_residual > 1.0e-20 {
        covariance_xr / (variance_x * variance_residual).sqrt()
    } else {
        0.0
    };
    // Fit a non-negative glycan scale on top of the protein model; the
    // likelihood gain measures whether the glycan explains the remaining
    // variance better than protein alone.
    let glycan_scale = (covariance_xr / variance_x).max(0.0);
    let glycan_intercept = mean_residual - glycan_scale * mean_x;
    let mut residual = 0.0;
    for (index, (x, _, weight, _)) in samples.iter().enumerate() {
        let predicted = glycan_intercept + glycan_scale * x;
        residual += weight * (residuals[index] - predicted) * (residuals[index] - predicted);
    }
    let residual = residual.max(1.0e-30);
    let null_residual = null_residual.max(residual);
    // Mask weights are fractional voxel memberships.  Their sum is a more
    // faithful effective observation count than the raw bounding-box size.
    let effective_count = weight_sum.max(3.0);
    let likelihood_gain = (0.5 * effective_count * (null_residual / residual).ln()).max(0.0);
    let bic_gain = (2.0 * likelihood_gain - effective_count.ln()).max(0.0);
    Ok(AgreementStatistics {
        correlation,
        protein_adjusted_correlation,
        likelihood_gain,
        bic_gain,
    })
}

fn profiled_likelihood_derivatives(samples: &[(f64, f64, f64, f64)]) -> Result<Vec<f64>> {
    if samples.len() < 3 {
        return Err(DensityError::ConstantSample);
    }
    let weight_sum = samples.iter().map(|(_, _, weight, _)| *weight).sum::<f64>();
    if weight_sum <= 0.0 || !weight_sum.is_finite() {
        return Err(DensityError::ConstantSample);
    }
    let mean_x = samples.iter().map(|(x, _, w, _)| x * w).sum::<f64>() / weight_sum;
    let mean_y = samples.iter().map(|(_, y, w, _)| y * w).sum::<f64>() / weight_sum;
    let mean_z = samples.iter().map(|(_, _, w, z)| z * w).sum::<f64>() / weight_sum;
    let variance_z = samples
        .iter()
        .map(|(_, _, w, z)| w * (z - mean_z).powi(2))
        .sum::<f64>();
    let covariance_zy = samples
        .iter()
        .map(|(_, y, w, z)| w * (z - mean_z) * (y - mean_y))
        .sum::<f64>();
    let protein_scale = if variance_z > 1.0e-20 {
        covariance_zy / variance_z
    } else {
        0.0
    };
    let protein_intercept = mean_y - protein_scale * mean_z;
    let residuals = samples
        .iter()
        .map(|(_, y, _, z)| y - protein_intercept - protein_scale * z)
        .collect::<Vec<_>>();
    let mean_residual = residuals
        .iter()
        .zip(samples)
        .map(|(residual, (_, _, weight, _))| residual * weight)
        .sum::<f64>()
        / weight_sum;
    let variance_x = samples
        .iter()
        .map(|(x, _, weight, _)| weight * (x - mean_x).powi(2))
        .sum::<f64>();
    if variance_x <= 1.0e-20 {
        return Err(DensityError::ConstantSample);
    }
    let covariance_xr = samples
        .iter()
        .enumerate()
        .map(|(index, (x, _, weight, _))| {
            weight * (x - mean_x) * (residuals[index] - mean_residual)
        })
        .sum::<f64>();
    let scale = (covariance_xr / variance_x).max(0.0);
    let intercept = mean_residual - scale * mean_x;
    let errors = samples
        .iter()
        .enumerate()
        .map(|(index, (x, _, _, _))| residuals[index] - intercept - scale * x)
        .collect::<Vec<_>>();
    let sse = samples
        .iter()
        .zip(&errors)
        .map(|((_, _, weight, _), error)| weight * error * error)
        .sum::<f64>()
        .max(1.0e-30);
    let effective_count = weight_sum.max(3.0);
    Ok(samples
        .iter()
        .zip(errors)
        .map(|((_, _, weight, _), error)| effective_count * weight * scale * error / sse)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::{Compression, write::GzEncoder};
    use glysys::{BuildOptions, GlycanTree, GlycosylationSite, ResidueId, read_pdb_str};
    use mrc::{VoxelBlock, create};
    use std::fs;
    use tempfile::tempdir;

    fn write_map(path: &Path, data: Vec<f32>, shape: [usize; 3]) {
        let mut writer = create(path)
            .shape(shape)
            .cell_lengths(shape[0] as f32, shape[1] as f32, shape[2] as f32)
            .mode::<f32>()
            .finish()
            .unwrap();
        writer
            .write_block_as(&VoxelBlock::new([0, 0, 0], shape, data).unwrap())
            .unwrap();
        writer.update_header_stats().unwrap();
        writer.finalize().unwrap();
    }

    #[test]
    fn gaussian_map_prefers_matching_pose() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("map.mrc");
        let shape = [24, 24, 24];
        let atom = [12.0, 12.0, 12.0];
        let mut data = vec![0.0; shape[0] * shape[1] * shape[2]];
        for z in 0..shape[2] {
            for y in 0..shape[1] {
                for x in 0..shape[0] {
                    let d = distance([x as f64, y as f64, z as f64], atom);
                    data[x + shape[0] * (y + shape[1] * z)] = (-d * d / 2.0).exp() as f32;
                }
            }
        }
        write_map(&path, data, shape);
        let map = DensityMap::open(&path).unwrap();
        assert_eq!(map.metadata().nx, 24);
        assert!(map.is_full_unit_cell());
        assert_eq!(map.metadata().sha256.len(), 64);
        let grid = map.cartesian_to_grid(atom);
        assert!(
            grid.iter()
                .zip(atom)
                .all(|(left, right)| (*left - right).abs() < 1.0e-8)
        );
        let wrapped = map.value_at_cartesian([atom[0] + 24.0, atom[1], atom[2]], true);
        let canonical = map.value_at_cartesian(atom, true);
        assert!((wrapped.unwrap() - canonical.unwrap()).abs() < 1.0e-6);
    }

    #[test]
    fn byte_constructor_matches_file_constructor() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("memory.mrc");
        let shape = [4, 3, 2];
        let data = (0..shape.iter().product::<usize>())
            .map(|value| value as f32)
            .collect();
        write_map(&path, data, shape);
        let bytes = fs::read(&path).unwrap();
        let from_file = DensityMap::open(&path).unwrap();
        let from_memory = DensityMap::from_bytes("memory.mrc", &bytes).unwrap();
        assert_eq!(from_memory.grid_shape(), from_file.grid_shape());
        assert_eq!(from_memory.values(), from_file.values());
        assert_eq!(from_memory.metadata().sha256, from_file.metadata().sha256);
    }

    #[test]
    fn trilinear_cartesian_gradient_matches_central_difference() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("gradient.mrc");
        let shape = [20, 18, 16];
        let mut data = vec![0.0; shape[0] * shape[1] * shape[2]];
        for z in 0..shape[2] {
            for y in 0..shape[1] {
                for x in 0..shape[0] {
                    data[x + shape[0] * (y + shape[1] * z)] =
                        (0.7 * x as f64 - 0.2 * y as f64 + 0.4 * z as f64) as f32;
                }
            }
        }
        write_map(&path, data, shape);
        let map = DensityMap::open(&path).unwrap();
        let point = [7.25, 8.4, 6.75];
        let (_, gradient) = map.value_gradient_at_cartesian(point, false).unwrap();
        for axis in 0..3 {
            let mut plus = point;
            let mut minus = point;
            plus[axis] += 1.0e-4;
            minus[axis] -= 1.0e-4;
            let finite = (map.value_at_cartesian(plus, false).unwrap()
                - map.value_at_cartesian(minus, false).unwrap())
                / 2.0e-4;
            assert!((gradient[axis] - finite).abs() < 1.0e-6);
        }
    }

    #[test]
    fn mask_and_correlation_are_finite_for_nonconstant_samples() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("map.mrc");
        let shape = [24, 24, 24];
        let mut data = vec![0.0; shape[0] * shape[1] * shape[2]];
        for z in 0..shape[2] {
            for y in 0..shape[1] {
                for x in 0..shape[0] {
                    data[x + shape[0] * (y + shape[1] * z)] = (x + 2 * y + 3 * z) as f32;
                }
            }
        }
        write_map(&path, data, shape);
        let map = DensityMap::open(&path).unwrap();
        assert!(map.metadata().warnings.is_empty());
        assert_eq!(mask_weight(0.0, 2.0, 1.0), 1.0);
        assert_eq!(mask_weight(3.0, 2.0, 1.0), 0.0);
        let _ = fs::metadata(&path).unwrap();
    }

    #[test]
    fn xray_samples_ignore_hydrogen_atoms() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("hydrogen.mrc");
        let shape = [32, 32, 32];
        write_map(&path, vec![0.0; shape.iter().product()], shape);
        let scorer = DensityScorer::new(
            DensityMap::open(&path).unwrap(),
            DensityScoreOptions {
                sigma_angstrom: Some(1.0),
                periodic: true,
                ..DensityScoreOptions::default()
            },
        )
        .unwrap();
        let structure = read_pdb_str(
            include_str!("../../../tests/fixtures/glycan.pdb"),
            &BuildOptions::default(),
        )
        .unwrap();
        let all_atoms = structure.atoms();
        assert!(
            all_atoms
                .iter()
                .any(|atom| atom.element.eq_ignore_ascii_case("H"))
        );
        let heavy_atoms = all_atoms
            .iter()
            .filter(|atom| !atom.element.eq_ignore_ascii_case("H"))
            .cloned()
            .collect::<Vec<_>>();
        let with_hydrogens = scorer.samples_for_atoms(&all_atoms, 1.0).unwrap();
        let without_hydrogens = scorer.samples_for_atoms(&heavy_atoms, 1.0).unwrap();
        assert_eq!(with_hydrogens, without_hydrogens);
    }

    #[test]
    fn indexed_fast_evidence_matches_structure_path() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("indexed-fast-evidence.mrc");
        let shape = [32, 32, 32];
        let mut data = vec![0.0; shape.iter().product()];
        for z in 0..shape[2] {
            for y in 0..shape[1] {
                for x in 0..shape[0] {
                    data[x + shape[0] * (y + shape[1] * z)] =
                        (0.03 * x as f64 + 0.07 * y as f64 - 0.02 * z as f64) as f32;
                }
            }
        }
        write_map(&path, data, shape);
        let scorer = DensityScorer::new(
            DensityMap::open(&path).unwrap(),
            DensityScoreOptions {
                sigma_angstrom: Some(1.0),
                periodic: true,
                ..DensityScoreOptions::default()
            },
        )
        .unwrap();
        let structure = read_pdb_str(
            include_str!("../../../tests/fixtures/glycan.pdb"),
            &BuildOptions::default(),
        )
        .unwrap();
        let residue = structure
            .residues()
            .into_iter()
            .find(|residue| residue.name == "0YB")
            .unwrap()
            .id;
        let target = DensityTarget {
            site: residue.clone(),
            glycan_residues: vec![residue.clone()],
        };
        let reference = scorer
            .fast_evidence(&structure, std::slice::from_ref(&target))
            .unwrap();
        let indexed = structure
            .atoms()
            .into_iter()
            .filter(|atom| atom.residue == residue && !atom.element.eq_ignore_ascii_case("H"))
            .map(|atom| DensityIndexedAtom {
                id: atom.id,
                element: atom.element,
                occupancy: atom.occupancy,
                b_factor: atom.b_factor,
                position: [atom.position.x, atom.position.y, atom.position.z],
            })
            .collect::<Vec<_>>();
        let indexed_result = scorer.fast_evidence_indexed(&indexed).unwrap();
        let scalar_result = scorer.fast_evidence_score_indexed(&indexed).unwrap();
        assert!((reference.score - indexed_result.score).abs() < 1.0e-12);
        assert!((reference.score - scalar_result).abs() < 1.0e-12);
        assert_eq!(reference.atom_gradients, indexed_result.atom_gradients);
    }

    #[test]
    fn fixed_site_ownership_is_deterministic_and_bounded() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("ownership.mrc");
        let shape = [32, 32, 32];
        write_map(&path, vec![0.0; shape.iter().product()], shape);
        let scorer = DensityScorer::new(
            DensityMap::open(&path).unwrap(),
            DensityScoreOptions {
                sigma_angstrom: Some(1.0),
                periodic: true,
                ..DensityScoreOptions::default()
            },
        )
        .unwrap()
        .with_site_ownership(0, vec![[10.0, 16.0, 16.0], [22.0, 16.0, 16.0]], 3.0);
        let structure = read_pdb_str(
            include_str!("../../../tests/fixtures/glycan.pdb"),
            &BuildOptions::default(),
        )
        .unwrap();
        let tree = structure
            .residues()
            .into_iter()
            .find(|residue| residue.name == "0YB")
            .unwrap();
        let target = DensityTarget {
            site: tree.id.clone(),
            glycan_residues: vec![tree.id],
        };
        let region = scorer.fixed_region(&[&structure], &[target], 2.0).unwrap();
        assert!(region.voxel_count() > 0);
        let (owned, shared, low) = region.ownership_buckets();
        assert_eq!(owned + shared + low, region.voxel_count());
        assert!(owned > 0);
    }

    #[test]
    fn ring_hypothesis_defaults_are_backward_compatible() {
        let json = r#"{
            "site":{"chain":"A","number":1},
            "component_id":3,
            "center_angstrom":[1.0,2.0,3.0],
            "normalized_score":2.0,
            "local_support":1.0
        }"#;
        let hypothesis: DensityRingHypothesis = serde_json::from_str(json).unwrap();
        assert_eq!(hypothesis.orientation_quaternion, [1.0, 0.0, 0.0, 0.0]);
        assert!(hypothesis.compatible_residues.is_empty());
        assert_eq!(hypothesis.provenance, "native_residual_peak");
    }

    #[test]
    fn ring_orientation_maps_z_axis_deterministically() {
        assert_eq!(
            quaternion_from_z_axis([0.0, 0.0, 4.0]),
            [1.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            quaternion_from_z_axis([0.0, 0.0, -4.0]),
            [0.0, 1.0, 0.0, 0.0]
        );
        let quaternion = quaternion_from_z_axis([1.0, 2.0, 3.0]);
        assert!(quaternion.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn pdb_occupancy_and_b_factor_round_trip_into_structure_atoms() {
        let pdb =
            "HETATM    1  C1  NAG B   1      10.000  10.000  10.000  0.37 42.50           C\nEND\n";
        let structure = read_pdb_str(pdb, &BuildOptions::default()).unwrap();
        let atom = structure.atoms().into_iter().next().unwrap();
        assert!((atom.occupancy - 0.37).abs() < 1.0e-6);
        assert!((atom.b_factor - 42.5).abs() < 1.0e-6);
        let round_trip =
            read_pdb_str(&structure.to_pdb_string(), &BuildOptions::default()).unwrap();
        let atom = round_trip.atoms().into_iter().next().unwrap();
        assert!((atom.occupancy - 0.37).abs() < 1.0e-2);
        assert!((atom.b_factor - 42.5).abs() < 1.0e-2);
    }

    #[test]
    fn axis_permutation_and_subvolume_starts_round_trip() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("permuted.map");
        let shape = [8, 9, 10];
        let mut writer = create(&path)
            .shape(shape)
            .sampling([30, 40, 50])
            .cell_lengths(30.0, 40.0, 50.0)
            .nstart([4, 5, 6])
            .axis_mapping([2, 1, 3])
            .mode::<f32>()
            .finish()
            .unwrap();
        let data = (0..shape[0] * shape[1] * shape[2])
            .map(|value| value as f32)
            .collect();
        writer
            .write_block_as(&VoxelBlock::new([0, 0, 0], shape, data).unwrap())
            .unwrap();
        writer.update_header_stats().unwrap();
        writer.finalize().unwrap();
        let map = DensityMap::open(&path).unwrap();
        assert_eq!(map.metadata().map_axes, [2, 1, 3]);
        let grid = [2.0, 3.0, 4.0];
        let cart = map.grid_to_cartesian(grid);
        assert!(
            cart.iter()
                .zip([7.0, 7.0, 10.0])
                .all(|(left, right)| (*left - right).abs() < 1.0e-6)
        );
        let recovered = map.cartesian_to_grid(cart);
        assert!(
            recovered
                .iter()
                .zip(grid)
                .all(|(left, right)| (*left - right).abs() < 1.0e-8)
        );
    }

    #[test]
    fn gzip_maps_are_read_and_hashed_as_downloaded() {
        let directory = tempdir().unwrap();
        let plain = directory.path().join("plain.mrc");
        let compressed = directory.path().join("compressed.mrc.gz");
        write_map(&plain, vec![1.0; 8 * 8 * 8], [8, 8, 8]);
        let bytes = fs::read(&plain).unwrap();
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        std::io::Write::write_all(&mut encoder, &bytes).unwrap();
        fs::write(&compressed, encoder.finish().unwrap()).unwrap();
        let map = DensityMap::open(&compressed).unwrap();
        assert_eq!(map.values().len(), 8 * 8 * 8);
        assert_eq!(map.metadata().sha256, sha256_file(&compressed).unwrap());
    }

    #[test]
    fn scorer_ranks_an_exact_pose_above_a_translation() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("pose.mrc");
        let pdb = "ATOM      1  ND2 ASN A   1      10.000  10.000  10.000  1.00  0.00           N\nHETATM    2  C1  NAG B   1      12.000  10.000  10.000  1.00  0.00           C\nHETATM    3  C2  NAG B   1      13.200  10.000  10.000  1.00  0.00           C\nHETATM    4  C3  NAG B   1      14.000  11.100  10.000  1.00  0.00           C\nHETATM    5  C4  NAG B   1      13.400  12.300  10.000  1.00  0.00           C\nHETATM    6  C5  NAG B   1      12.100  12.100  10.000  1.00  0.00           C\nHETATM    7  O5  NAG B   1      11.500  11.000  10.000  1.00  0.00           O\nEND\n";
        let mut structure = read_pdb_str(pdb, &BuildOptions::default()).unwrap();
        let site = ResidueId {
            chain: "A".into(),
            number: 1,
            insertion_code: None,
        };
        let glycan = ResidueId {
            chain: "B".into(),
            number: 1,
            insertion_code: None,
        };
        structure.metadata_mut().glycan_trees.push(GlycanTree {
            chain: "B".into(),
            residue_ids: vec![glycan.clone()],
            attachment_site: Some(site.clone()),
        });
        structure.add_glycosylation_site(GlycosylationSite {
            protein_residue: site.clone(),
            protein_atom: "ND2".into(),
            glycan_residue: glycan,
            glycan_atom: "C1".into(),
        });
        let shape = [32, 32, 32];
        let mut data = vec![0.0_f32; shape[0] * shape[1] * shape[2]];
        for z in 0..shape[2] {
            for y in 0..shape[1] {
                for x in 0..shape[0] {
                    let value = structure
                        .atoms()
                        .into_iter()
                        .filter(|atom| atom.residue.chain == "B")
                        .map(|atom| {
                            let distance = distance(
                                [x as f64, y as f64, z as f64],
                                [atom.position.x, atom.position.y, atom.position.z],
                            );
                            (-distance * distance / 2.0).exp()
                        })
                        .sum::<f64>();
                    data[x + shape[0] * (y + shape[1] * z)] = value as f32;
                }
            }
        }
        write_map(&path, data, shape);
        let scorer = DensityScorer::new(
            DensityMap::open(&path).unwrap(),
            DensityScoreOptions {
                sigma_angstrom: Some(1.0),
                ..DensityScoreOptions::default()
            },
        )
        .unwrap();
        let target = DensityTarget::for_site(&structure, &site).unwrap();
        let exact = scorer
            .score(&structure, std::slice::from_ref(&target))
            .unwrap();
        assert!((exact.correlation - exact.sites[0].correlation).abs() < 1.0e-12);
        let mut translated = structure.clone();
        for atom in translated
            .atoms()
            .into_iter()
            .filter(|atom| atom.residue.chain == "B")
        {
            translated
                .set_atom_position(
                    atom.id,
                    glysys::Vec3 {
                        x: atom.position.x + 4.0,
                        y: atom.position.y,
                        z: atom.position.z,
                    },
                )
                .unwrap();
        }
        let decoy = scorer
            .score(&translated, std::slice::from_ref(&target))
            .unwrap();
        assert!(exact.correlation > decoy.correlation);

        // The optimization region is defined once from the reachable poses.
        // Both candidates are then compared against identical voxels, so a
        // translated candidate cannot improve by carrying its mask to a
        // nearby unrelated peak.
        let fixed = scorer
            .fixed_region(
                &[&structure, &translated],
                std::slice::from_ref(&target),
                2.0,
            )
            .unwrap();
        let fixed_exact = scorer
            .score_fixed_region(&fixed, &structure, std::slice::from_ref(&target))
            .unwrap();
        let fixed_decoy = scorer
            .score_fixed_region(&fixed, &translated, std::slice::from_ref(&target))
            .unwrap();
        assert_eq!(fixed_exact.voxel_count, fixed_decoy.voxel_count);
        assert!(fixed_exact.likelihood_gain > fixed_decoy.likelihood_gain);
        assert!(fixed_exact.training_likelihood_gain > fixed_decoy.training_likelihood_gain);
        assert!(fixed_exact.heldout_likelihood_gain > fixed_decoy.heldout_likelihood_gain);
        assert!(fixed_exact.correlation > fixed_decoy.correlation);

        let gradient = scorer
            .score_fixed_region_with_gradients(&fixed, &translated, std::slice::from_ref(&target))
            .unwrap();
        let atom = translated
            .find_atom(&target.glycan_residues[0], "C1")
            .unwrap();
        let position = translated.atom(atom).unwrap().position;
        let epsilon = 1.0e-4;
        let mut plus = translated.clone();
        plus.set_atom_position(
            atom,
            glysys::Vec3 {
                x: position.x + epsilon,
                ..position
            },
        )
        .unwrap();
        let mut minus = translated.clone();
        minus
            .set_atom_position(
                atom,
                glysys::Vec3 {
                    x: position.x - epsilon,
                    ..position
                },
            )
            .unwrap();
        let finite = (scorer
            .score_fixed_region(&fixed, &plus, std::slice::from_ref(&target))
            .unwrap()
            .likelihood_gain
            - scorer
                .score_fixed_region(&fixed, &minus, std::slice::from_ref(&target))
                .unwrap()
                .likelihood_gain)
            / (2.0 * epsilon);
        assert!((gradient.atom_gradients[&atom][0] - finite).abs() < 2.0e-2);
    }

    #[test]
    fn protein_shell_calibration_recovers_the_map_kernel() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("calibration.mrc");
        let pdb = "ATOM      1  CA  ASN A   1      12.000  12.000  12.000  1.00 20.00           C\nATOM      2  CA  ALA A   2      14.000  12.000  12.000  1.00 20.00           C\nATOM      3  CA  ALA A   3      10.000  12.000  12.000  1.00 20.00           C\nATOM      4  CA  ALA A   4      12.000  14.000  12.000  1.00 20.00           C\nATOM      5  CA  ALA A   5      12.000  10.000  12.000  1.00 20.00           C\nATOM      6  CA  ALA A   6      12.000  12.000  14.000  1.00 20.00           C\nATOM      7  CA  ALA A   7      12.000  12.000  10.000  1.00 20.00           C\nATOM      8  CA  ALA A   8      15.000  15.000  12.000  1.00 20.00           C\nATOM      9  CA  ALA A   9       9.000   9.000  12.000  1.00 20.00           C\nATOM     10  CA  ALA A  10      15.000   9.000  12.000  1.00 20.00           C\nEND\n";
        let structure = read_pdb_str(pdb, &BuildOptions::default()).unwrap();
        let shape = [28, 28, 28];
        let sigma = 1.2_f64;
        let mut data = vec![0.0_f32; shape[0] * shape[1] * shape[2]];
        for z in 0..shape[2] {
            for y in 0..shape[1] {
                for x in 0..shape[0] {
                    data[x + shape[0] * (y + shape[1] * z)] = structure
                        .atoms()
                        .into_iter()
                        .map(|atom| {
                            let d = distance(
                                [x as f64, y as f64, z as f64],
                                [atom.position.x, atom.position.y, atom.position.z],
                            );
                            (-d * d / (2.0 * sigma * sigma)).exp()
                        })
                        .sum::<f64>()
                        as f32;
                }
            }
        }
        write_map(&path, data, shape);
        let map = DensityMap::open(&path).unwrap();
        let calibration = DensityScorer::calibrate_sigma_from_protein(
            &map,
            &structure,
            &[ResidueId {
                chain: "A".into(),
                number: 1,
                insertion_code: None,
            }],
            DensityScoreOptions::default(),
            &[0.7, 1.2, 1.7],
        )
        .unwrap();
        assert_eq!(calibration.selected_sigma_angstrom, 1.2);
        assert_eq!(calibration.trials.len(), 3);
        assert!(calibration.effective_sigma_angstrom > calibration.selected_sigma_angstrom);
        assert!(calibration.capture_sigma_angstrom >= calibration.anti_alias_floor_angstrom);
        assert_eq!(calibration.search_scales.len(), 3);
        assert!(
            calibration
                .search_scales
                .iter()
                .all(|scale| scale.sigma_angstrom.is_finite() && scale.weight > 0.0)
        );
    }

    #[test]
    fn signed_difference_density_rewards_positive_and_vetoes_negative_peaks() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("difference.mrc");
        let pdb =
            "HETATM    1  C1  NAG B   1      10.000  10.000  10.000  1.00 20.00           C\nEND\n";
        let structure = read_pdb_str(pdb, &BuildOptions::default()).unwrap();
        let shape = [24, 24, 24];
        let mut data = vec![0.0_f32; shape[0] * shape[1] * shape[2]];
        for z in 0..shape[2] {
            for y in 0..shape[1] {
                for x in 0..shape[0] {
                    let point = [x as f64, y as f64, z as f64];
                    let positive = (-distance(point, [10.0, 10.0, 10.0]).powi(2) / 2.0).exp();
                    let negative = (-distance(point, [14.0, 10.0, 10.0]).powi(2) / 2.0).exp();
                    data[x + shape[0] * (y + shape[1] * z)] = (positive - negative) as f32;
                }
            }
        }
        write_map(&path, data, shape);
        let scorer = DensityScorer::new(
            DensityMap::open(&path).unwrap(),
            DensityScoreOptions {
                sigma_angstrom: Some(1.0),
                ..DensityScoreOptions::default()
            },
        )
        .unwrap();
        let target = DensityTarget {
            site: ResidueId {
                chain: "A".into(),
                number: 1,
                insertion_code: None,
            },
            glycan_residues: vec![ResidueId {
                chain: "B".into(),
                number: 1,
                insertion_code: None,
            }],
        };
        let positive = scorer
            .signed_difference_evidence(&structure, std::slice::from_ref(&target))
            .unwrap();
        let mut negative_structure = structure.clone();
        negative_structure
            .set_atom_position(
                glysys::AtomId(1),
                glysys::Vec3 {
                    x: 14.0,
                    y: 10.0,
                    z: 10.0,
                },
            )
            .unwrap();
        let negative = scorer
            .signed_difference_evidence(&negative_structure, &[target])
            .unwrap();
        assert!(positive.score > negative.score);
        assert!(positive.positive_reward > 0.0);
        assert!(negative.negative_penalty > 0.0);
    }

    #[test]
    fn periodic_boundary_translation_preserves_the_score() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("periodic.mrc");
        let pdb = "HETATM    1  C1  NAG B   1       0.300   4.000   4.000  1.00  0.00           C\nHETATM    2  C2  NAG B   1       1.500   4.000   4.000  1.00  0.00           C\nHETATM    3  C3  NAG B   1       2.100   5.000   4.000  1.00  0.00           C\nHETATM    4  C4  NAG B   1       1.500   6.000   4.000  1.00  0.00           C\nHETATM    5  C5  NAG B   1       0.400   5.700   4.000  1.00  0.00           C\nHETATM    6  O5  NAG B   1      -0.200   4.800   4.000  1.00  0.00           O\nEND\n";
        let mut structure = read_pdb_str(pdb, &BuildOptions::default()).unwrap();
        let site = ResidueId {
            chain: "A".into(),
            number: 1,
            insertion_code: None,
        };
        let glycan = ResidueId {
            chain: "B".into(),
            number: 1,
            insertion_code: None,
        };
        structure.metadata_mut().glycan_trees.push(GlycanTree {
            chain: "B".into(),
            residue_ids: vec![glycan.clone()],
            attachment_site: Some(site.clone()),
        });
        let shape = [16, 16, 16];
        let atoms = structure
            .atoms()
            .into_iter()
            .filter(|atom| atom.residue == glycan)
            .collect::<Vec<_>>();
        let mut data = vec![0.0_f32; shape[0] * shape[1] * shape[2]];
        for z in 0..shape[2] {
            for y in 0..shape[1] {
                for x in 0..shape[0] {
                    let point = [x as f64, y as f64, z as f64];
                    data[x + shape[0] * (y + shape[1] * z)] = atoms
                        .iter()
                        .map(|atom| {
                            let mut dx = point[0] - atom.position.x;
                            dx -= (dx / 16.0).round() * 16.0;
                            let d = (dx * dx
                                + (point[1] - atom.position.y).powi(2)
                                + (point[2] - atom.position.z).powi(2))
                            .sqrt();
                            (-d * d / 2.0).exp()
                        })
                        .sum::<f64>()
                        as f32;
                }
            }
        }
        write_map(&path, data, shape);
        let scorer = DensityScorer::new(
            DensityMap::open(&path).unwrap(),
            DensityScoreOptions {
                sigma_angstrom: Some(1.0),
                periodic: true,
                ..DensityScoreOptions::default()
            },
        )
        .unwrap();
        let target = DensityTarget::for_site(&structure, &site).unwrap();
        let exact = scorer
            .score(&structure, std::slice::from_ref(&target))
            .unwrap();
        let mut translated = structure.clone();
        for atom in translated
            .atoms()
            .into_iter()
            .filter(|atom| atom.residue == glycan)
        {
            translated
                .set_atom_position(
                    atom.id,
                    glysys::Vec3 {
                        x: atom.position.x + 16.0,
                        y: atom.position.y,
                        z: atom.position.z,
                    },
                )
                .unwrap();
        }
        let wrapped = scorer
            .score(&translated, std::slice::from_ref(&target))
            .unwrap();
        assert!((exact.correlation - exact.sites[0].correlation).abs() < 1.0e-12);
        assert!((exact.correlation - wrapped.correlation).abs() < 1.0e-8);
    }

    #[test]
    fn visualization_maps_share_dimensions_and_transforms() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("visualization.mrc");
        write_map(&path, vec![1.0; 20 * 20 * 20], [20, 20, 20]);
        let pdb = "HETATM    1  C1  NAG B   1       8.000   8.000   8.000  1.00  0.00           C\nHETATM    2  C2  NAG B   1       9.200   8.000   8.000  1.00  0.00           C\nHETATM    3  C3  NAG B   1      10.000   9.000   8.000  1.00  0.00           C\nHETATM    4  C4  NAG B   1       9.400  10.000   8.000  1.00  0.00           C\nHETATM    5  C5  NAG B   1       8.100   9.700   8.000  1.00  0.00           C\nHETATM    6  O5  NAG B   1       7.500   8.800   8.000  1.00  0.00           O\nEND\n";
        let mut structure = read_pdb_str(pdb, &BuildOptions::default()).unwrap();
        let site = ResidueId {
            chain: "A".into(),
            number: 1,
            insertion_code: None,
        };
        let glycan = ResidueId {
            chain: "B".into(),
            number: 1,
            insertion_code: None,
        };
        structure.metadata_mut().glycan_trees.push(GlycanTree {
            chain: "B".into(),
            residue_ids: vec![glycan],
            attachment_site: Some(site.clone()),
        });
        let scorer = DensityScorer::new(
            DensityMap::open(&path).unwrap(),
            DensityScoreOptions {
                sigma_angstrom: Some(1.0),
                ..DensityScoreOptions::default()
            },
        )
        .unwrap();
        let maps = scorer
            .write_visualization_maps(
                &structure,
                &[DensityTarget {
                    site,
                    glycan_residues: Vec::new(),
                }],
                directory.path(),
                "crop",
                2.0,
            )
            .unwrap();
        let observed = DensityMap::open(&maps.observed).unwrap();
        let calculated = DensityMap::open(&maps.calculated).unwrap();
        let mask = DensityMap::open(&maps.mask).unwrap();
        assert_eq!(observed.grid_shape(), calculated.grid_shape());
        assert_eq!(observed.grid_shape(), mask.grid_shape());
        assert_eq!(observed.metadata().map_axes, calculated.metadata().map_axes);
        assert_eq!(observed.metadata().map_axes, mask.metadata().map_axes);
        assert_eq!(observed.metadata().nxstart, calculated.metadata().nxstart);
        assert_eq!(observed.metadata().nystart, mask.metadata().nystart);
    }
}

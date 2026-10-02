//! Generic CCP4/MRC density loading, map acquisition, and masked map agreement
//! scoring of a model (`reglyco density`, `reglyco validate`).
//!
//! The GlycoFlow fitter (`reglyco-glycoflow`) evaluates its density
//! likelihood with [`site_likelihood`].

use std::collections::{BTreeMap, BTreeSet};
use std::f64::consts::PI;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use glysys::{ResidueId, Structure};
use nalgebra::{Matrix3, Vector3};
use sha2::{Digest, Sha256};

#[cfg(not(target_arch = "wasm32"))]
pub mod rcsb;
pub mod site_likelihood;

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
    /// Equal to `likelihood_gain` (kept for report compatibility).
    #[serde(default)]
    pub training_likelihood_gain: f64,
    /// Equal to `likelihood_gain` (kept for report compatibility).
    #[serde(default)]
    pub heldout_likelihood_gain: f64,
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
}

impl DensityScorer {
    fn scoring_atom_amplitude(&self, atom: &glysys::StructureAtom) -> f64 {
        if self.options.glycan_b_factor.is_some() {
            atom_amplitude(atom)
        } else {
            atom.occupancy.clamp(0.0, 1.0)
        }
    }

    fn atom_b_factor(&self, atom: &glysys::StructureAtom) -> f64 {
        if atom.b_factor > 1.0e-6 {
            atom.b_factor.max(0.0)
        } else {
            self.options.glycan_b_factor.unwrap_or(0.0).max(0.0)
        }
    }

    fn atom_sigma(&self, map_sigma: f64, atom: &glysys::StructureAtom) -> f64 {
        let b = self.atom_b_factor(atom);
        (map_sigma * map_sigma + b / (8.0 * PI * PI)).sqrt()
    }

    pub fn new(map: DensityMap, options: DensityScoreOptions) -> Result<Self> {
        options.sigma()?;
        Ok(Self {
            map,
            options,
            protein_background: OnceLock::new(),
        })
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
        // A glycan B factor selects the element/B-aware protein field at the
        // map kernel width; otherwise a broad 2.5 Å protein background is used.
        let calibrated_background = self.options.glycan_b_factor.is_some();
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
        Some((value - self.protein_model_at(cartesian) - self.map.metadata.dmean) / scale)
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
                            mask_weight(distance, mask_radius_angstrom, mask_falloff_angstrom);
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

#[derive(Debug, Clone, Copy)]
struct AgreementStatistics {
    correlation: f64,
    protein_adjusted_correlation: f64,
    likelihood_gain: f64,
    bic_gain: f64,
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
}

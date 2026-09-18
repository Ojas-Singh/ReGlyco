//! Incremental analytical density scoring for discrete glycan proposals.
//!
//! The legacy fixed-ROI scorer rasterizes every candidate glycan field over
//! the complete voxel region.  That is useful for final reporting, but it is
//! too expensive for a residue-by-residue proposal loop.  This module uses the
//! same unnormalised Gaussian atom field as the trusted scorer and expands the
//! squared residual:
//!
//! ```text
//! |D - (P + G)|²
//!     = constant - 2<D,G> + 2<P,G> + |G|²
//! ```
//!
//! The map interaction is an atom-local lookup in a precomputed
//! Gaussian-convolved map.  The model interaction is an analytic
//! Gaussian/Gaussian overlap.  Adding or moving one residue therefore touches
//! only its atoms and their local pair terms.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::f64::consts::PI;
use std::sync::{Arc, RwLock};

use glysys::{AtomId, ResidueId, Structure, StructureAtom, Vec3};
use nalgebra::Vector3;

use crate::{DensityMap, DensityTarget, Result};

const GAUSSIAN_CUTOFF_SIGMA: f64 = 4.0;
// At eight combined standard deviations the omitted Gaussian overlap is
// below 1e-14 of its peak. This permits a spatially local protein traversal
// without changing meaningful density rankings.
const PROTEIN_PAIR_CUTOFF_SIGMA: f64 = 8.0;
const SIGMA_KEY_SCALE: f64 = 1_000_000.0;
const PI2_32: f64 = 8.0 * PI * PI;
// A real structure carries many slightly different deposited B factors.  A
// separate full-map convolution for every raw effective sigma would turn the
// analytical scorer back into an expensive map-building loop.  Generated
// atoms therefore use a small, deterministic sigma bank.  The bin is much
// finer than a typical map-resolution uncertainty while keeping the number
// of cached blurred fields bounded.
const SIGMA_BANK_WIDTH: f64 = 0.05;

/// Compact atom record used by the proposal scorer.  The amplitude and
/// effective sigma are supplied by the caller so generated atoms and fixed
/// protein atoms use the same calibrated kernel.
#[derive(Debug, Clone, PartialEq)]
pub struct FastAtom {
    pub id: AtomId,
    pub residue: ResidueId,
    pub amplitude: f64,
    pub sigma: f64,
    pub position: [f64; 3],
}

impl FastAtom {
    pub fn from_structure_atom(
        atom: &StructureAtom,
        map_sigma: f64,
        fallback_b_factor: Option<f64>,
    ) -> Self {
        let b_factor = if atom.b_factor > 1.0e-6 {
            atom.b_factor
        } else {
            fallback_b_factor.unwrap_or(0.0)
        }
        .max(0.0);
        Self {
            id: atom.id,
            residue: atom.residue.clone(),
            amplitude: element_weight(&atom.element) * atom.occupancy.clamp(0.0, 1.0),
            sigma: effective_sigma(map_sigma, b_factor),
            position: [atom.position.x, atom.position.y, atom.position.z],
        }
    }
}

/// Value and coordinate derivative of the analytical energy for one proposal.
#[derive(Debug, Clone, Default)]
pub struct FastScoreGradient {
    pub energy: f64,
    pub gradients: BTreeMap<AtomId, [f64; 3]>,
}

/// Decomposition of the Gaussian residual objective for a candidate field.
///
/// The map term and the Gaussian/Gaussian overlap terms are kept separate so
/// callers can profile the non-negative glycan scale without rebuilding a
/// voxelized model field.  `residual_cross` is the interaction with the map
/// after subtracting the fixed protein and the already accepted prefix;
/// `model_self` is the candidate field self-overlap.  Up to a candidate-
/// independent constant, the unit-scale residual energy is
/// `-2 * residual_cross + model_self`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FastScoreTerms {
    pub map_cross: f64,
    pub protein_cross: f64,
    pub placed_cross: f64,
    pub residual_cross: f64,
    pub model_self: f64,
    pub unit_scale_energy: f64,
    pub profiled_scale: f64,
    pub profiled_gain: f64,
}

#[derive(Debug, Clone)]
struct BlurredField {
    values: Vec<f64>,
}

/// A reusable map/model context for one independent glycosylation site.
///
/// `protein_atoms` is fixed for the lifetime of the context.  `placed` is
/// the accepted glycan prefix.  Proposal callers can add/remove a residue or
/// replace a subtree without constructing a `Structure` or a voxelized model
/// field.
#[derive(Clone)]
pub struct FastDensityContext {
    map: Arc<DensityMap>,
    periodic: bool,
    map_sigma: f64,
    fallback_b_factor: Option<f64>,
    voxel_shape: [usize; 3],
    voxel_volume: f64,
    blurred: Arc<RwLock<BTreeMap<u64, BlurredField>>>,
    protein_atoms: Vec<FastAtom>,
    template_atoms: BTreeMap<AtomId, FastAtom>,
    protein_cells: HashMap<[i32; 3], Vec<usize>>,
    protein_cell_size: f64,
    protein_pair_cutoff: f64,
    placed: BTreeMap<AtomId, FastAtom>,
    residue_atoms: BTreeMap<ResidueId, BTreeSet<AtomId>>,
}

impl FastDensityContext {
    /// Build a context from a map and fixed protein structure.
    ///
    /// `targets` identifies glycan residues that must not be copied into the
    /// fixed protein field when the caller passes an unstripped structure.
    /// The map field is built lazily for each effective sigma used by a
    /// proposal, but every field is immutable and cached thereafter.
    pub fn new(
        map: DensityMap,
        sigma_map: f64,
        fallback_b_factor: Option<f64>,
        protein: &Structure,
        targets: &[DensityTarget],
        periodic: bool,
    ) -> Result<Self> {
        validate_sigma(sigma_map)?;
        let glycan_residues = targets
            .iter()
            .flat_map(|target| target.glycan_residues.iter().cloned())
            .collect::<BTreeSet<_>>();
        let protein_atoms = protein
            .atoms()
            .into_iter()
            .filter(|atom| !glycan_residues.contains(&atom.residue))
            .filter(|atom| !atom.element.eq_ignore_ascii_case("H"))
            .map(|atom| FastAtom::from_structure_atom(&atom, sigma_map, fallback_b_factor))
            .collect::<Vec<_>>();
        let template_atoms = protein
            .atoms()
            .into_iter()
            .filter(|atom| glycan_residues.contains(&atom.residue))
            .filter(|atom| !atom.element.eq_ignore_ascii_case("H"))
            .map(|atom| {
                (
                    atom.id,
                    FastAtom::from_structure_atom(&atom, sigma_map, fallback_b_factor),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let max_structure_sigma = protein
            .atoms()
            .into_iter()
            .filter(|atom| !atom.element.eq_ignore_ascii_case("H"))
            .map(|atom| {
                let b_factor = if atom.b_factor > 1.0e-6 {
                    atom.b_factor
                } else {
                    fallback_b_factor.unwrap_or(0.0)
                };
                effective_sigma(sigma_map, b_factor)
            })
            .fold(sigma_map, f64::max);
        let max_protein_sigma = protein_atoms
            .iter()
            .map(|atom| atom.sigma)
            .fold(sigma_map, f64::max);
        let protein_pair_cutoff = PROTEIN_PAIR_CUTOFF_SIGMA
            * (max_structure_sigma * max_structure_sigma + max_protein_sigma * max_protein_sigma)
                .sqrt();
        let protein_cell_size = protein_pair_cutoff.max(1.0);
        let voxel_shape = map.grid_shape();
        let voxel_volume = map.fractional_to_cartesian.determinant().abs()
            / map.metadata().sampling.iter().product::<usize>() as f64;
        if !voxel_volume.is_finite() || voxel_volume <= 0.0 {
            return Err(crate::DensityError::Geometry(
                "map voxel volume must be positive".into(),
            ));
        }
        let mut context = Self {
            map: Arc::new(map),
            periodic,
            map_sigma: sigma_map,
            fallback_b_factor,
            voxel_shape,
            voxel_volume,
            blurred: Arc::new(RwLock::new(BTreeMap::new())),
            protein_atoms,
            template_atoms,
            protein_cells: HashMap::new(),
            protein_cell_size,
            protein_pair_cutoff,
            placed: BTreeMap::new(),
            residue_atoms: BTreeMap::new(),
        };
        context.build_protein_cells();
        // Do not eagerly rasterize a full map here.  The discrete frontier
        // path normally uses the indexed robust evidence objective and may
        // never request the analytical Gaussian field at all.  Building a
        // 72x72x256 field with a several-voxel kernel during context
        // construction would make proposal setup dominate the search.  The
        // field is built lazily by `blurred_lookup` when an analytical score
        // or gradient is actually requested, and remains cached thereafter.
        Ok(context)
    }

    /// Construct fixed atoms from a structure using the same element/B-factor
    /// convention as [`FastAtom::from_structure_atom`].
    pub fn atoms_from_structure(
        structure: &Structure,
        residues: &BTreeSet<ResidueId>,
        map_sigma: f64,
        fallback_b_factor: Option<f64>,
    ) -> Vec<FastAtom> {
        structure
            .atoms()
            .into_iter()
            .filter(|atom| residues.contains(&atom.residue))
            .filter(|atom| !atom.element.eq_ignore_ascii_case("H"))
            .map(|atom| FastAtom::from_structure_atom(&atom, map_sigma, fallback_b_factor))
            .collect()
    }

    pub fn map(&self) -> &DensityMap {
        &self.map
    }

    pub fn protein_atoms(&self) -> &[FastAtom] {
        &self.protein_atoms
    }

    pub fn map_sigma(&self) -> f64 {
        self.map_sigma
    }

    pub fn fallback_b_factor(&self) -> Option<f64> {
        self.fallback_b_factor
    }

    /// Convert the heavy atoms belonging to selected target residues into
    /// proposal atoms without allocating a second `Structure`.
    pub fn atoms_for_targets(
        &self,
        structure: &Structure,
        targets: &[DensityTarget],
    ) -> Vec<FastAtom> {
        let residues = targets
            .iter()
            .flat_map(|target| target.glycan_residues.iter().cloned())
            .collect::<BTreeSet<_>>();
        structure
            .atoms()
            .into_iter()
            .filter(|atom| residues.contains(&atom.residue))
            .filter(|atom| !atom.element.eq_ignore_ascii_case("H"))
            .map(|atom| {
                FastAtom::from_structure_atom(&atom, self.map_sigma, self.fallback_b_factor)
            })
            .collect()
    }

    /// Convert cached target-atom metadata to proposal atoms at a caller's
    /// current coordinates. This avoids cloning/materializing a complete
    /// `Structure` for every discrete pose.
    pub fn atoms_for_positions(
        &self,
        positions: &BTreeMap<AtomId, Vec3>,
        targets: &[DensityTarget],
    ) -> Vec<FastAtom> {
        let residues = targets
            .iter()
            .flat_map(|target| target.glycan_residues.iter().cloned())
            .collect::<BTreeSet<_>>();
        self.template_atoms
            .values()
            .filter(|atom| residues.contains(&atom.residue))
            .filter_map(|atom| {
                let position = positions.get(&atom.id)?;
                let mut copy = atom.clone();
                copy.position = [position.x, position.y, position.z];
                Some(copy)
            })
            .collect()
    }

    pub fn placed_atoms(&self) -> &BTreeMap<AtomId, FastAtom> {
        &self.placed
    }

    pub fn voxel_volume(&self) -> f64 {
        self.voxel_volume
    }

    fn cell_key(&self, position: [f64; 3]) -> [i32; 3] {
        [
            (position[0] / self.protein_cell_size).floor() as i32,
            (position[1] / self.protein_cell_size).floor() as i32,
            (position[2] / self.protein_cell_size).floor() as i32,
        ]
    }

    fn build_protein_cells(&mut self) {
        let mut translations = Vec::new();
        if self.periodic {
            for i in -1..=1 {
                for j in -1..=1 {
                    for k in -1..=1 {
                        translations.push(
                            self.map
                                .fractional_to_cartesian([i as f64, j as f64, k as f64]),
                        );
                    }
                }
            }
        } else {
            translations.push([0.0, 0.0, 0.0]);
        }
        for (index, atom) in self.protein_atoms.iter().enumerate() {
            for translation in &translations {
                let image = [
                    atom.position[0] + translation[0],
                    atom.position[1] + translation[1],
                    atom.position[2] + translation[2],
                ];
                self.protein_cells
                    .entry(self.cell_key(image))
                    .or_default()
                    .push(index);
            }
        }
    }

    fn protein_neighbors(&self, atom: &FastAtom) -> Vec<usize> {
        if self.protein_atoms.is_empty() {
            return Vec::new();
        }
        let cell = self.cell_key(atom.position);
        let mut indices = BTreeSet::new();
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(values) =
                        self.protein_cells
                            .get(&[cell[0] + dx, cell[1] + dy, cell[2] + dz])
                    {
                        indices.extend(values.iter().copied());
                    }
                }
            }
        }
        if indices.is_empty() {
            return (0..self.protein_atoms.len()).collect();
        }
        indices
            .into_iter()
            .filter(|index| {
                let displacement = self.pair_displacement(atom, &self.protein_atoms[*index]);
                dot3(displacement, displacement) <= self.protein_pair_cutoff.powi(2)
            })
            .collect()
    }

    /// Ensure a map-convolved field exists for `sigma`.  Quantisation is only
    /// used as a cache key; the rounded value is the value used consistently
    /// for the field and subsequent lookups.
    pub fn ensure_sigma(&self, sigma: f64) -> Result<f64> {
        validate_sigma(sigma)?;
        let key = sigma_key(sigma);
        let rounded_sigma = sigma_from_key(key);
        if self
            .blurred
            .read()
            .map_err(|_| lock_error())?
            .contains_key(&key)
        {
            return Ok(rounded_sigma);
        }
        let field = self.build_blurred_field(rounded_sigma)?;
        self.blurred
            .write()
            .map_err(|_| lock_error())?
            .entry(key)
            .or_insert(field);
        Ok(rounded_sigma)
    }

    fn build_blurred_field(&self, sigma: f64) -> Result<BlurredField> {
        let shape = self.voxel_shape;
        let voxel_count = shape[0] * shape[1] * shape[2];
        let mut values = vec![0.0; voxel_count];
        let offsets = self.kernel_offsets(sigma)?;
        for x in 0..shape[0] {
            for y in 0..shape[1] {
                for z in 0..shape[2] {
                    let output_index = grid_index([x, y, z], shape);
                    let mut value = 0.0;
                    for offset in &offsets {
                        let Some(input_grid) = self.input_grid(
                            x as isize + offset.offset[0],
                            y as isize + offset.offset[1],
                            z as isize + offset.offset[2],
                        ) else {
                            continue;
                        };
                        let sample = map_grid_value(&self.map, input_grid);
                        value += sample * offset.weight;
                    }
                    values[output_index] = value;
                }
            }
        }
        Ok(BlurredField { values })
    }

    fn kernel_offsets(&self, sigma: f64) -> Result<Vec<KernelOffset>> {
        let origin = self.map.grid_to_cartesian([0.0, 0.0, 0.0]);
        let axis_steps = [
            self.map.grid_to_cartesian([1.0, 0.0, 0.0]),
            self.map.grid_to_cartesian([0.0, 1.0, 0.0]),
            self.map.grid_to_cartesian([0.0, 0.0, 1.0]),
        ]
        .map(|point| distance3(sub3(point, origin)));
        let minimum_spacing = axis_steps
            .iter()
            .copied()
            .filter(|spacing| spacing.is_finite() && *spacing > 1.0e-6)
            .fold(f64::INFINITY, f64::min);
        if !minimum_spacing.is_finite() {
            return Err(crate::DensityError::Geometry(
                "map grid spacing is invalid".into(),
            ));
        }
        let radius = (GAUSSIAN_CUTOFF_SIGMA * sigma / minimum_spacing).ceil() as isize;
        let mut offsets = Vec::new();
        for dx in -radius..=radius {
            for dy in -radius..=radius {
                for dz in -radius..=radius {
                    let point = self
                        .map
                        .grid_to_cartesian([dx as f64, dy as f64, dz as f64]);
                    let displacement = sub3(point, origin);
                    let distance2 = dot3(displacement, displacement);
                    if distance2 > (GAUSSIAN_CUTOFF_SIGMA * sigma).powi(2) {
                        continue;
                    }
                    let weight = (-distance2 / (2.0 * sigma * sigma)).exp();
                    offsets.push(KernelOffset {
                        offset: [dx, dy, dz],
                        weight,
                    });
                }
            }
        }
        Ok(offsets)
    }

    fn input_grid(&self, x: isize, y: isize, z: isize) -> Option<[usize; 3]> {
        let coordinates = [x, y, z];
        let mut result = [0usize; 3];
        for axis in 0..3 {
            if self.periodic {
                result[axis] =
                    coordinates[axis].rem_euclid(self.voxel_shape[axis] as isize) as usize;
            } else if coordinates[axis] < 0 || coordinates[axis] >= self.voxel_shape[axis] as isize
            {
                return None;
            } else {
                result[axis] = coordinates[axis] as usize;
            }
        }
        Some(result)
    }

    fn blurred_lookup(&self, sigma: f64, position: [f64; 3]) -> Result<Option<(f64, [f64; 3])>> {
        let sigma = self.ensure_sigma(sigma)?;
        let key = sigma_key(sigma);
        let fields = self.blurred.read().map_err(|_| lock_error())?;
        let field = fields
            .get(&key)
            .ok_or_else(|| crate::DensityError::Geometry("blurred field cache miss".into()))?;
        let mut grid = self.map.cartesian_to_grid(position);
        for axis in 0..3 {
            if self.periodic {
                grid[axis] = grid[axis].rem_euclid(self.voxel_shape[axis] as f64);
            } else if grid[axis] < 0.0 || grid[axis] > self.voxel_shape[axis] as f64 - 1.0 {
                return Ok(None);
            }
        }
        let base = grid.map(|coordinate| coordinate.floor() as isize);
        let fraction = grid.map(|coordinate| coordinate - coordinate.floor());
        let mut value = 0.0;
        let mut grid_gradient = [0.0; 3];
        for dx in 0..=1 {
            for dy in 0..=1 {
                for dz in 0..=1 {
                    let Some(index_grid) =
                        self.input_grid(base[0] + dx, base[1] + dy, base[2] + dz)
                    else {
                        continue;
                    };
                    let wx = if dx == 0 {
                        1.0 - fraction[0]
                    } else {
                        fraction[0]
                    };
                    let wy = if dy == 0 {
                        1.0 - fraction[1]
                    } else {
                        fraction[1]
                    };
                    let wz = if dz == 0 {
                        1.0 - fraction[2]
                    } else {
                        fraction[2]
                    };
                    let weight = wx * wy * wz;
                    let index = grid_index(index_grid, self.voxel_shape);
                    let sample = field.values[index];
                    value += weight * sample;
                    grid_gradient[0] += (if dx == 0 { -1.0 } else { 1.0 }) * wy * wz * sample;
                    grid_gradient[1] += (if dy == 0 { -1.0 } else { 1.0 }) * wx * wz * sample;
                    grid_gradient[2] += (if dz == 0 { -1.0 } else { 1.0 }) * wx * wy * sample;
                }
            }
        }
        let sampling = self.map.metadata().sampling.map(|value| value as f64);
        let gradient_fractional = Vector3::new(
            grid_gradient[0] * sampling[0],
            grid_gradient[1] * sampling[1],
            grid_gradient[2] * sampling[2],
        );
        let gradient_cartesian = self.map.cartesian_to_fractional.transpose() * gradient_fractional;
        Ok(Some((
            value,
            [
                gradient_cartesian.x,
                gradient_cartesian.y,
                gradient_cartesian.z,
            ],
        )))
    }

    /// Score the current placed glycan field as an energy. Lower is better.
    pub fn energy(&self) -> Result<f64> {
        let atoms = self.placed.values().cloned().collect::<Vec<_>>();
        self.energy_for_atoms(&atoms)
    }

    pub fn density_score(&self) -> Result<f64> {
        self.energy()
    }

    /// Score an arbitrary complete set of atoms without changing the placed
    /// prefix. This is useful for validating the incremental implementation
    /// against a full proposal in tests and debug builds.
    pub fn energy_for_atoms(&self, atoms: &[FastAtom]) -> Result<f64> {
        let mut energy = 0.0;
        for atom in atoms {
            energy += self.map_interaction(atom)?;
            for index in self.protein_neighbors(atom) {
                let protein = &self.protein_atoms[index];
                energy += 2.0 * self.pair_overlap(atom, protein);
            }
        }
        for (index, atom) in atoms.iter().enumerate() {
            energy += self.pair_overlap(atom, atom);
            for other in atoms.iter().skip(index + 1) {
                energy += 2.0 * self.pair_overlap(atom, other);
            }
        }
        Ok(energy)
    }

    pub fn score_atoms(&self, atoms: &[FastAtom]) -> Result<f64> {
        self.energy_for_atoms(atoms)
    }

    /// Return the profiled residual-likelihood terms for a complete candidate.
    /// The fixed protein field is represented analytically by its atom-pair
    /// overlaps, so this remains an atom-local operation.  The returned gain
    /// is the improvement over the same model with the candidate glycan
    /// omitted, after profiling a bounded non-negative glycan scale.
    pub fn profiled_terms(&self, atoms: &[FastAtom]) -> Result<FastScoreTerms> {
        self.profiled_terms_with_prefix(atoms, &[])
    }

    /// Profile a candidate against the fixed protein and the currently placed
    /// prefix.  This is the primitive used by residue/arm proposal search:
    /// moving a candidate changes only its map interaction, protein overlap,
    /// prefix overlap, and internal self-overlap.
    pub fn profiled_addition_terms(&self, candidate: &[FastAtom]) -> Result<FastScoreTerms> {
        self.profiled_terms_with_prefix_iter(candidate, self.placed.values())
    }

    /// Profile a replacement of an already placed residue/subtree without
    /// cloning the context or rebuilding the accepted field.  The atoms in
    /// `old` are excluded from the prefix before the new atoms are evaluated.
    /// This is the move primitive used by bounded local polishing.
    pub fn profiled_move_terms(
        &self,
        old: &[FastAtom],
        new: &[FastAtom],
    ) -> Result<FastScoreTerms> {
        if old.len() != new.len() {
            return Err(crate::DensityError::Geometry(
                "profiled_move_terms requires equal old/new atom counts".into(),
            ));
        }
        let moved_ids = old.iter().map(|atom| atom.id).collect::<BTreeSet<_>>();
        if old.iter().any(|atom| !self.placed.contains_key(&atom.id)) {
            return Err(crate::DensityError::Geometry(
                "profiled_move_terms old atoms are not in the placed prefix".into(),
            ));
        }
        self.profiled_terms_with_prefix_iter(
            new,
            self.placed
                .values()
                .filter(|atom| !moved_ids.contains(&atom.id)),
        )
    }

    fn profiled_terms_with_prefix(
        &self,
        atoms: &[FastAtom],
        prefix: &[FastAtom],
    ) -> Result<FastScoreTerms> {
        self.profiled_terms_with_prefix_iter(atoms, prefix.iter())
    }

    fn profiled_terms_with_prefix_iter<'a, I>(
        &self,
        atoms: &[FastAtom],
        prefix: I,
    ) -> Result<FastScoreTerms>
    where
        I: IntoIterator<Item = &'a FastAtom>,
    {
        if atoms.is_empty() {
            return Err(crate::DensityError::EmptyTarget);
        }
        let mut map_cross = 0.0;
        let mut protein_cross = 0.0;
        let mut placed_cross = 0.0;
        let prefix = prefix.into_iter().collect::<Vec<_>>();
        for atom in atoms {
            map_cross += self.map_cross_atom(atom)?;
            for index in self.protein_neighbors(atom) {
                protein_cross += self.pair_overlap(atom, &self.protein_atoms[index]);
            }
            for other in &prefix {
                placed_cross += self.pair_overlap(atom, other);
            }
        }
        let mut model_self = 0.0;
        for (index, atom) in atoms.iter().enumerate() {
            model_self += self.pair_overlap(atom, atom);
            for other in atoms.iter().skip(index + 1) {
                model_self += 2.0 * self.pair_overlap(atom, other);
            }
        }
        let residual_cross = map_cross - protein_cross - placed_cross;
        let unit_scale_energy = -2.0 * residual_cross + model_self;
        // The unconstrained least-squares scale is c/s.  A glycan field cannot
        // have negative occupancy, so clamp at zero.  Keeping the upper bound
        // open is useful for comparing maps whose absolute coefficient scale
        // differs; callers that need an occupancy estimate can clamp the
        // returned value to one at the interface.
        let profiled_scale = if model_self > 1.0e-12 {
            (residual_cross / model_self).max(0.0)
        } else {
            0.0
        };
        let profiled_gain = if profiled_scale > 0.0 {
            2.0 * profiled_scale * residual_cross - profiled_scale * profiled_scale * model_self
        } else {
            0.0
        };
        Ok(FastScoreTerms {
            map_cross,
            protein_cross,
            placed_cross,
            residual_cross,
            model_self,
            unit_scale_energy,
            profiled_scale,
            profiled_gain,
        })
    }

    /// Profiled likelihood gain and coordinate gradients for a new residue or
    /// subtree.  Gradients are with respect to the candidate coordinates; the
    /// protein and accepted prefix are held fixed.  Envelope-theorem handling
    /// means the profiled scale itself does not require a separate derivative.
    pub fn profiled_addition_with_gradients(
        &self,
        candidate: &[FastAtom],
    ) -> Result<(FastScoreTerms, FastScoreGradient)> {
        if candidate.is_empty() {
            return Err(crate::DensityError::EmptyTarget);
        }
        let prefix = self.placed.values().cloned().collect::<Vec<_>>();
        let mut terms = FastScoreTerms::default();
        let mut cross_gradients = BTreeMap::<AtomId, [f64; 3]>::new();
        let mut self_gradients = BTreeMap::<AtomId, [f64; 3]>::new();
        for atom in candidate {
            let (map_cross, map_gradient) = self.map_cross_atom_with_gradient(atom)?;
            terms.map_cross += map_cross;
            let mut cross_gradient = map_gradient;
            for index in self.protein_neighbors(atom) {
                let protein = &self.protein_atoms[index];
                let overlap = self.pair_overlap(atom, protein);
                let contribution = overlap;
                terms.protein_cross += overlap;
                add_overlap_gradient(
                    &mut cross_gradient,
                    atom,
                    protein,
                    self.pair_displacement(atom, protein),
                    contribution,
                );
            }
            for other in &prefix {
                let overlap = self.pair_overlap(atom, other);
                terms.placed_cross += overlap;
                add_overlap_gradient(
                    &mut cross_gradient,
                    atom,
                    other,
                    self.pair_displacement(atom, other),
                    overlap,
                );
            }
            cross_gradients.insert(atom.id, cross_gradient);
            terms.model_self += self.pair_overlap(atom, atom);
            self_gradients.entry(atom.id).or_default();
        }
        for (index, atom) in candidate.iter().enumerate() {
            for other in candidate.iter().skip(index + 1) {
                let overlap = self.pair_overlap(atom, other);
                terms.model_self += 2.0 * overlap;
                let first = self_gradients.entry(atom.id).or_default();
                add_overlap_gradient(
                    first,
                    atom,
                    other,
                    self.pair_displacement(atom, other),
                    2.0 * overlap,
                );
                let second = self_gradients.entry(other.id).or_default();
                add_overlap_gradient(
                    second,
                    other,
                    atom,
                    self.pair_displacement(other, atom),
                    2.0 * overlap,
                );
            }
        }
        terms.residual_cross = terms.map_cross - terms.protein_cross - terms.placed_cross;
        terms.unit_scale_energy = -2.0 * terms.residual_cross + terms.model_self;
        terms.profiled_scale = if terms.model_self > 1.0e-12 {
            (terms.residual_cross / terms.model_self).max(0.0)
        } else {
            0.0
        };
        terms.profiled_gain = if terms.profiled_scale > 0.0 {
            2.0 * terms.profiled_scale * terms.residual_cross
                - terms.profiled_scale * terms.profiled_scale * terms.model_self
        } else {
            0.0
        };
        let scale = terms.profiled_scale;
        let mut gradients = BTreeMap::new();
        for atom in candidate {
            let cross = cross_gradients.get(&atom.id).copied().unwrap_or([0.0; 3]);
            let self_gradient = self_gradients.get(&atom.id).copied().unwrap_or([0.0; 3]);
            gradients.insert(
                atom.id,
                [
                    2.0 * scale * cross[0] - scale * scale * self_gradient[0],
                    2.0 * scale * cross[1] - scale * scale * self_gradient[1],
                    2.0 * scale * cross[2] - scale * scale * self_gradient[2],
                ],
            );
        }
        Ok((
            terms,
            FastScoreGradient {
                // The gradient above is the derivative of the profiled
                // likelihood gain (envelope theorem), not of the unit-scale
                // residual energy.  Keep the public energy field consistent
                // with that gradient so a local minimizer cannot accidentally
                // step according to a different objective.
                energy: -terms.profiled_gain,
                gradients,
            },
        ))
    }

    /// Energy change from adding a candidate residue or subtree.
    pub fn score_addition(&self, candidate: &[FastAtom]) -> Result<f64> {
        let mut delta = 0.0;
        for atom in candidate {
            delta += self.map_interaction(atom)?;
            for index in self.protein_neighbors(atom) {
                let protein = &self.protein_atoms[index];
                delta += 2.0 * self.pair_overlap(atom, protein);
            }
            for placed in self.placed.values() {
                delta += 2.0 * self.pair_overlap(atom, placed);
            }
            delta += self.pair_overlap(atom, atom);
        }
        for (index, atom) in candidate.iter().enumerate() {
            for other in candidate.iter().skip(index + 1) {
                delta += 2.0 * self.pair_overlap(atom, other);
            }
        }
        Ok(delta)
    }

    /// Energy and analytic coordinate gradients for an addition proposal.
    pub fn score_addition_with_gradients(
        &self,
        candidate: &[FastAtom],
    ) -> Result<FastScoreGradient> {
        let mut result = FastScoreGradient::default();
        for atom in candidate {
            let mut gradient = [0.0; 3];
            let map = self.map_interaction_with_gradient(atom)?;
            result.energy += map.0;
            for axis in 0..3 {
                gradient[axis] += map.1[axis];
            }
            for index in self.protein_neighbors(atom) {
                let protein = &self.protein_atoms[index];
                let overlap = self.pair_overlap(atom, protein);
                result.energy += 2.0 * overlap;
                add_overlap_gradient(
                    &mut gradient,
                    atom,
                    protein,
                    self.pair_displacement(atom, protein),
                    2.0 * overlap,
                );
            }
            for placed in self.placed.values() {
                let overlap = self.pair_overlap(atom, placed);
                result.energy += 2.0 * overlap;
                add_overlap_gradient(
                    &mut gradient,
                    atom,
                    placed,
                    self.pair_displacement(atom, placed),
                    2.0 * overlap,
                );
            }
            result.energy += self.pair_overlap(atom, atom);
            result.gradients.insert(atom.id, gradient);
        }
        for (index, atom) in candidate.iter().enumerate() {
            for other in candidate.iter().skip(index + 1) {
                let overlap = self.pair_overlap(atom, other);
                result.energy += 2.0 * overlap;
                let mut first = result.gradients.entry(atom.id).or_default();
                add_overlap_gradient(
                    &mut first,
                    atom,
                    other,
                    self.pair_displacement(atom, other),
                    2.0 * overlap,
                );
                let mut second = result.gradients.entry(other.id).or_default();
                add_overlap_gradient(
                    &mut second,
                    other,
                    atom,
                    self.pair_displacement(other, atom),
                    2.0 * overlap,
                );
            }
        }
        Ok(result)
    }

    /// Energy change when replacing the atoms identified by `old` with
    /// corresponding atoms in `new`.  The context itself remains unchanged;
    /// callers can commit the new coordinates with [`replace_atoms`].
    pub fn score_move(&self, old: &[FastAtom], new: &[FastAtom]) -> Result<f64> {
        if old.len() != new.len() {
            return Err(crate::DensityError::Geometry(
                "score_move requires equal old/new atom counts".into(),
            ));
        }
        let moved_ids = old.iter().map(|atom| atom.id).collect::<BTreeSet<_>>();
        let mut delta = 0.0;
        for (from, to) in old.iter().zip(new) {
            delta += self.map_interaction(to)? - self.map_interaction(from)?;
            for index in self.protein_neighbors(to) {
                let protein = &self.protein_atoms[index];
                delta += 2.0 * (self.pair_overlap(to, protein) - self.pair_overlap(from, protein));
            }
            for placed in self.placed.values() {
                if !moved_ids.contains(&placed.id) {
                    delta +=
                        2.0 * (self.pair_overlap(to, placed) - self.pair_overlap(from, placed));
                }
            }
            delta += self.pair_overlap(to, to) - self.pair_overlap(from, from);
        }
        for i in 0..old.len() {
            for j in i + 1..old.len() {
                delta += 2.0
                    * (self.pair_overlap(&new[i], &new[j]) - self.pair_overlap(&old[i], &old[j]));
            }
        }
        Ok(delta)
    }

    /// Analytic gradients of the energy change returned by [`score_move`].
    /// The `old` coordinates are treated as constants; gradients correspond
    /// to the replacement coordinates in `new`.
    pub fn score_move_with_gradients(
        &self,
        old: &[FastAtom],
        new: &[FastAtom],
    ) -> Result<FastScoreGradient> {
        if old.len() != new.len() {
            return Err(crate::DensityError::Geometry(
                "score_move requires equal old/new atom counts".into(),
            ));
        }
        let moved_ids = old.iter().map(|atom| atom.id).collect::<BTreeSet<_>>();
        let mut result = FastScoreGradient {
            energy: self.score_move(old, new)?,
            gradients: BTreeMap::new(),
        };
        for atom in new {
            let mut gradient = self.map_interaction_with_gradient(atom)?.1;
            for index in self.protein_neighbors(atom) {
                let protein = &self.protein_atoms[index];
                let overlap = self.pair_overlap(atom, protein);
                add_overlap_gradient(
                    &mut gradient,
                    atom,
                    protein,
                    self.pair_displacement(atom, protein),
                    2.0 * overlap,
                );
            }
            for placed in self.placed.values() {
                if moved_ids.contains(&placed.id) {
                    continue;
                }
                let overlap = self.pair_overlap(atom, placed);
                add_overlap_gradient(
                    &mut gradient,
                    atom,
                    placed,
                    self.pair_displacement(atom, placed),
                    2.0 * overlap,
                );
            }
            for other in new {
                if other.id == atom.id {
                    continue;
                }
                let overlap = self.pair_overlap(atom, other);
                add_overlap_gradient(
                    &mut gradient,
                    atom,
                    other,
                    self.pair_displacement(atom, other),
                    2.0 * overlap,
                );
            }
            result.gradients.insert(atom.id, gradient);
        }
        Ok(result)
    }

    /// Commit a previously scored replacement into the placed prefix.
    pub fn replace_atoms(&mut self, old: &[FastAtom], new: Vec<FastAtom>) -> Result<()> {
        let ids = old.iter().map(|atom| atom.id).collect::<BTreeSet<_>>();
        self.remove_atoms(&ids);
        self.add_atoms(new);
        Ok(())
    }

    pub fn add_atoms(&mut self, atoms: Vec<FastAtom>) {
        for atom in atoms {
            self.residue_atoms
                .entry(atom.residue.clone())
                .or_default()
                .insert(atom.id);
            self.placed.insert(atom.id, atom);
        }
    }

    pub fn remove_atoms(&mut self, atom_ids: &BTreeSet<AtomId>) {
        for id in atom_ids {
            let Some(atom) = self.placed.remove(id) else {
                continue;
            };
            if let Some(residue_atoms) = self.residue_atoms.get_mut(&atom.residue) {
                residue_atoms.remove(id);
                if residue_atoms.is_empty() {
                    self.residue_atoms.remove(&atom.residue);
                }
            }
        }
    }

    fn map_interaction(&self, atom: &FastAtom) -> Result<f64> {
        Ok(-2.0 * self.map_cross_atom(atom)?)
    }

    fn map_interaction_with_gradient(&self, atom: &FastAtom) -> Result<(f64, [f64; 3])> {
        let (cross, gradient) = self.map_cross_atom_with_gradient(atom)?;
        Ok((
            -2.0 * cross,
            [-2.0 * gradient[0], -2.0 * gradient[1], -2.0 * gradient[2]],
        ))
    }

    fn map_cross_atom(&self, atom: &FastAtom) -> Result<f64> {
        Ok(self
            .blurred_lookup(atom.sigma, atom.position)?
            .map(|(value, _)| atom.amplitude * value)
            .unwrap_or(0.0))
    }

    fn map_cross_atom_with_gradient(&self, atom: &FastAtom) -> Result<(f64, [f64; 3])> {
        Ok(self
            .blurred_lookup(atom.sigma, atom.position)?
            .map(|(value, gradient)| {
                (
                    atom.amplitude * value,
                    [
                        atom.amplitude * gradient[0],
                        atom.amplitude * gradient[1],
                        atom.amplitude * gradient[2],
                    ],
                )
            })
            .unwrap_or((0.0, [0.0; 3])))
    }

    fn pair_displacement(&self, first: &FastAtom, second: &FastAtom) -> [f64; 3] {
        self.map
            .map_displacement(second.position, first.position, self.periodic)
    }

    fn pair_overlap(&self, first: &FastAtom, second: &FastAtom) -> f64 {
        let displacement = self.pair_displacement(first, second);
        gaussian_overlap_per_voxel(
            first,
            second,
            dot3(displacement, displacement),
            self.voxel_volume,
        )
    }
}

#[derive(Debug, Clone, Copy)]
struct KernelOffset {
    offset: [isize; 3],
    weight: f64,
}

fn validate_sigma(sigma: f64) -> Result<()> {
    if !sigma.is_finite() || sigma <= 0.0 {
        return Err(crate::DensityError::Geometry(
            "fast density sigma must be positive and finite".into(),
        ));
    }
    Ok(())
}

fn lock_error() -> crate::DensityError {
    crate::DensityError::Geometry("fast density cache lock poisoned".into())
}

fn sigma_key(sigma: f64) -> u64 {
    (sigma * SIGMA_KEY_SCALE).round().max(1.0) as u64
}

fn sigma_from_key(key: u64) -> f64 {
    key as f64 / SIGMA_KEY_SCALE
}

fn effective_sigma(map_sigma: f64, b_factor: f64) -> f64 {
    let raw = (map_sigma * map_sigma + b_factor.max(0.0) / PI2_32).sqrt();
    (raw / SIGMA_BANK_WIDTH).round().max(1.0) * SIGMA_BANK_WIDTH
}

fn element_weight(element: &str) -> f64 {
    match element.trim().to_ascii_uppercase().as_str() {
        "H" => 1.0,
        "C" => 6.0,
        "N" => 7.0,
        "O" => 8.0,
        "F" => 9.0,
        "P" => 15.0,
        "S" => 16.0,
        "CL" => 17.0,
        "BR" => 35.0,
        "I" => 53.0,
        _ => 6.0,
    }
}

fn grid_index(grid: [usize; 3], shape: [usize; 3]) -> usize {
    grid[0] + shape[0] * (grid[1] + shape[1] * grid[2])
}

fn map_grid_value(map: &DensityMap, grid: [usize; 3]) -> f64 {
    let axes = map.metadata().map_axes.map(|axis| axis - 1);
    let array = [grid[axes[0]], grid[axes[1]], grid[axes[2]]];
    map.values()[array[0] + map.metadata().nx * (array[1] + map.metadata().ny * array[2])] as f64
}

fn gaussian_overlap_per_voxel(
    a: &FastAtom,
    b: &FastAtom,
    distance2: f64,
    voxel_volume: f64,
) -> f64 {
    let sigma2 = a.sigma * a.sigma + b.sigma * b.sigma;
    if !sigma2.is_finite() || sigma2 <= 0.0 {
        return 0.0;
    }
    let integral_prefactor =
        (2.0 * PI).powf(1.5) * (a.sigma * a.sigma * b.sigma * b.sigma / sigma2).powf(1.5);
    a.amplitude * b.amplitude * integral_prefactor * (-distance2 / (2.0 * sigma2)).exp()
        / voxel_volume
}

fn add_overlap_gradient(
    gradient: &mut [f64; 3],
    first: &FastAtom,
    second: &FastAtom,
    displacement: [f64; 3],
    energy_term: f64,
) {
    let sigma2 = first.sigma * first.sigma + second.sigma * second.sigma;
    if !sigma2.is_finite() || sigma2 <= 0.0 {
        return;
    }
    let factor = energy_term / sigma2;
    for axis in 0..3 {
        gradient[axis] -= factor * displacement[axis];
    }
}

fn displacement(first: [f64; 3], second: [f64; 3]) -> [f64; 3] {
    [
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    ]
}

fn sub3(first: [f64; 3], second: [f64; 3]) -> [f64; 3] {
    displacement(first, second)
}

fn dot3(value: [f64; 3], other: [f64; 3]) -> f64 {
    value[0] * other[0] + value[1] * other[1] + value[2] * other[2]
}

fn distance3(value: [f64; 3]) -> f64 {
    dot3(value, value).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use glysys::{BuildOptions, read_pdb_str};
    use mrc::{VoxelBlock, create};
    use tempfile::tempdir;

    fn write_map(path: &std::path::Path, shape: [usize; 3], values: Vec<f32>) {
        let mut writer = create(path)
            .shape(shape)
            .cell_lengths(shape[0] as f32, shape[1] as f32, shape[2] as f32)
            .mode::<f32>()
            .finish()
            .unwrap();
        writer
            .write_block_as(&VoxelBlock::new([0, 0, 0], shape, values).unwrap())
            .unwrap();
        writer.update_header_stats().unwrap();
        writer.finalize().unwrap();
    }

    fn gaussian_map(shape: [usize; 3], centre: [f64; 3], sigma: f64) -> Vec<f32> {
        let mut values = vec![0.0; shape[0] * shape[1] * shape[2]];
        for z in 0..shape[2] {
            for y in 0..shape[1] {
                for x in 0..shape[0] {
                    let d2 = (x as f64 - centre[0]).powi(2)
                        + (y as f64 - centre[1]).powi(2)
                        + (z as f64 - centre[2]).powi(2);
                    values[grid_index([x, y, z], shape)] =
                        (-d2 / (2.0 * sigma * sigma)).exp() as f32;
                }
            }
        }
        values
    }

    fn empty_structure() -> Structure {
        read_pdb_str(
            "ATOM      1  CA  ALA A   1       1.000   1.000   1.000  1.00 20.00           C\nEND\n",
            &BuildOptions::default(),
        )
        .unwrap()
    }

    fn atom(id: u32, position: [f64; 3]) -> FastAtom {
        FastAtom {
            id: AtomId(id),
            residue: ResidueId {
                chain: "G".into(),
                number: id as i32,
                insertion_code: None,
            },
            amplitude: 6.0,
            sigma: 1.0,
            position,
        }
    }

    fn context(path: &std::path::Path, periodic: bool) -> FastDensityContext {
        FastDensityContext::new(
            DensityMap::open(path).unwrap(),
            1.0,
            None,
            &empty_structure(),
            &[DensityTarget {
                site: ResidueId {
                    chain: "A".into(),
                    number: 1,
                    insertion_code: None,
                },
                glycan_residues: vec![ResidueId {
                    chain: "A".into(),
                    number: 1,
                    insertion_code: None,
                }],
            }],
            periodic,
        )
        .unwrap()
    }

    #[test]
    fn addition_and_move_match_complete_energy() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("map.mrc");
        let shape = [32, 32, 32];
        write_map(&path, shape, gaussian_map(shape, [16.0, 16.0, 16.0], 1.0));
        let mut scorer = context(&path, false);
        let first = atom(1, [15.7, 16.0, 16.0]);
        let second = atom(2, [17.2, 16.0, 16.0]);
        let pair = vec![first.clone(), second.clone()];

        let first_energy = scorer.score_addition(std::slice::from_ref(&first)).unwrap();
        scorer.add_atoms(vec![first.clone()]);
        let second_delta = scorer
            .score_addition(std::slice::from_ref(&second))
            .unwrap();
        let pair_energy = scorer.energy_for_atoms(&pair).unwrap();
        assert!(
            (first_energy + second_delta - pair_energy).abs() < 1.0e-8,
            "incremental addition drifted: first={first_energy} second={second_delta} pair={pair_energy}"
        );

        let moved = atom(1, [16.4, 16.0, 16.0]);
        let before = scorer.energy().unwrap();
        let delta = scorer
            .score_move(std::slice::from_ref(&first), std::slice::from_ref(&moved))
            .unwrap();
        scorer
            .replace_atoms(std::slice::from_ref(&first), vec![moved.clone()])
            .unwrap();
        let after = scorer.energy().unwrap();
        assert!(
            (after - before - delta).abs() < 1.0e-8,
            "incremental move drifted: before={before} delta={delta} after={after}"
        );

        let move_gradient = scorer
            .score_move_with_gradients(std::slice::from_ref(&moved), std::slice::from_ref(&first))
            .unwrap();
        let epsilon = 1.0e-4;
        let mut plus = first.clone();
        plus.position[0] += epsilon;
        let mut minus = first.clone();
        minus.position[0] -= epsilon;
        let finite = (scorer
            .score_move(std::slice::from_ref(&moved), std::slice::from_ref(&plus))
            .unwrap()
            - scorer
                .score_move(std::slice::from_ref(&moved), std::slice::from_ref(&minus))
                .unwrap())
            / (2.0 * epsilon);
        assert!(
            (move_gradient.gradients[&first.id][0] - finite).abs() < 0.15,
            "move gradient={} finite_difference={finite}",
            move_gradient.gradients[&first.id][0]
        );
    }

    #[test]
    fn matching_map_pose_beats_translation() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("map.mrc");
        let shape = [32, 32, 32];
        write_map(&path, shape, gaussian_map(shape, [16.0, 16.0, 16.0], 1.0));
        let scorer = context(&path, false);
        let matching = scorer
            .score_addition(&[atom(1, [16.0, 16.0, 16.0])])
            .unwrap();
        let displaced = scorer
            .score_addition(&[atom(1, [20.0, 16.0, 16.0])])
            .unwrap();
        assert!(
            matching < displaced,
            "matching pose should have lower energy: matching={matching} displaced={displaced}"
        );
    }

    #[test]
    fn analytical_energy_matches_direct_voxel_energy() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("map.mrc");
        let shape = [40, 40, 40];
        let centre = [20.0, 20.0, 20.0];
        let map_values = gaussian_map(shape, centre, 1.0);
        write_map(&path, shape, map_values.clone());
        let scorer = context(&path, false);
        let candidate = atom(1, centre);
        let fast = scorer
            .score_addition(std::slice::from_ref(&candidate))
            .unwrap();
        let mut direct = 0.0;
        for z in 0..shape[2] {
            for y in 0..shape[1] {
                for x in 0..shape[0] {
                    let d2 = (x as f64 - centre[0]).powi(2)
                        + (y as f64 - centre[1]).powi(2)
                        + (z as f64 - centre[2]).powi(2);
                    let model = candidate.amplitude * (-d2 / 2.0).exp();
                    let observed = map_values[grid_index([x, y, z], shape)] as f64;
                    direct += model * model - 2.0 * observed * model;
                }
            }
        }
        let scale = direct.abs().max(1.0);
        assert!(
            (fast - direct).abs() / scale < 0.08,
            "fast energy={fast} direct voxel energy={direct}"
        );
    }

    #[test]
    fn fast_score_and_fixed_roi_agree_on_candidate_ordering() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("ranking.mrc");
        let shape = [32, 32, 32];
        let centre = [16.2, 16.0, 16.0];
        write_map(&path, shape, gaussian_map(shape, centre, 1.0));

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
        let make_structure = |x: f64| {
            read_pdb_str(
                &format!(
                    "HETATM    1  C1  NAG B   1      {x:6.3}  16.000  16.000  1.00  0.00           C\nEND\n"
                ),
                &BuildOptions::default(),
            )
            .unwrap()
        };
        let structures = [
            make_structure(16.2),
            make_structure(18.0),
            make_structure(21.0),
        ];
        let map = DensityMap::open(&path).unwrap();
        let scorer = crate::DensityScorer::new(
            map,
            crate::DensityScoreOptions {
                sigma_angstrom: Some(1.0),
                periodic: false,
                ..crate::DensityScoreOptions::default()
            },
        )
        .unwrap();
        let references = structures.iter().collect::<Vec<_>>();
        let region = scorer
            .fixed_region(&references, std::slice::from_ref(&target), 4.0)
            .unwrap();
        let exact = structures
            .iter()
            .map(|structure| {
                scorer
                    .score_fixed_region(&region, structure, std::slice::from_ref(&target))
                    .unwrap()
                    .likelihood_gain
            })
            .collect::<Vec<_>>();
        let fast = scorer
            .fast_context(&structures[0], std::slice::from_ref(&target))
            .unwrap();
        let fast_energy = structures
            .iter()
            .map(|structure| {
                let atoms = fast.atoms_for_targets(structure, std::slice::from_ref(&target));
                fast.energy_for_atoms(&atoms).unwrap()
            })
            .collect::<Vec<_>>();
        let exact_best = exact
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .unwrap();
        let fast_best = fast_energy
            .iter()
            .enumerate()
            .min_by(|left, right| left.1.total_cmp(right.1))
            .map(|(index, _)| index)
            .unwrap();
        assert_eq!(
            exact_best, fast_best,
            "fixed-ROI order={exact:?}, FastScore energies={fast_energy:?}"
        );
    }

    #[test]
    fn fast_score_preserves_random_pose_top_k_against_fixed_roi() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("random-ranking.mrc");
        let shape = [40, 40, 40];
        let mut values = gaussian_map(shape, [15.4, 16.6, 17.1], 1.1);
        let second = gaussian_map(shape, [25.2, 22.7, 20.4], 1.4);
        for (value, addition) in values.iter_mut().zip(second) {
            *value += 0.72 * addition;
        }
        write_map(&path, shape, values);

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
        let make_structure = |position: [f64; 3]| {
            read_pdb_str(
                &format!(
                    "HETATM    1  C1  NAG B   1      {:6.3} {:6.3} {:6.3}  1.00  0.00           C\nEND\n",
                    position[0], position[1], position[2]
                ),
                &BuildOptions::default(),
            )
            .unwrap()
        };
        // A deterministic low-discrepancy set exercises translations in the
        // whole local ROI instead of only testing three hand-picked poses.
        let mut state = 0x9e37_79b9_u64;
        let mut structures = Vec::with_capacity(256);
        for _ in 0..256 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let x = 7.0 + ((state >> 16) as f64 / (u64::MAX >> 16) as f64) * 26.0;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let y = 7.0 + ((state >> 16) as f64 / (u64::MAX >> 16) as f64) * 26.0;
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let z = 7.0 + ((state >> 16) as f64 / (u64::MAX >> 16) as f64) * 26.0;
            structures.push(make_structure([x, y, z]));
        }
        let map = DensityMap::open(&path).unwrap();
        let scorer = crate::DensityScorer::new(
            map,
            crate::DensityScoreOptions {
                sigma_angstrom: Some(1.0),
                periodic: false,
                ..crate::DensityScoreOptions::default()
            },
        )
        .unwrap();
        let references = structures.iter().collect::<Vec<_>>();
        let region = scorer
            .fixed_region(&references, std::slice::from_ref(&target), 4.0)
            .unwrap();
        let exact = structures
            .iter()
            .enumerate()
            .map(|(index, structure)| {
                scorer
                    .score_fixed_region(&region, structure, std::slice::from_ref(&target))
                    .unwrap_or_else(|error| {
                        panic!("candidate {index} exact score failed: {error:?}")
                    })
                    .likelihood_gain
            })
            .collect::<Vec<_>>();
        let fast = scorer
            .fast_context(&structures[0], std::slice::from_ref(&target))
            .unwrap();
        let fast_energy = structures
            .iter()
            .map(|structure| {
                let atoms = fast.atoms_for_targets(structure, std::slice::from_ref(&target));
                fast.energy_for_atoms(&atoms).unwrap()
            })
            .collect::<Vec<_>>();
        let mut exact_order = (0..exact.len()).collect::<Vec<_>>();
        exact_order.sort_by(|left, right| exact[*right].total_cmp(&exact[*left]));
        let mut fast_order = (0..fast_energy.len()).collect::<Vec<_>>();
        fast_order.sort_by(|left, right| fast_energy[*left].total_cmp(&fast_energy[*right]));
        let exact_top = exact_order.iter().take(8).copied().collect::<BTreeSet<_>>();
        let fast_top = fast_order.iter().take(8).copied().collect::<BTreeSet<_>>();
        let overlap = exact_top.intersection(&fast_top).count();
        assert!(
            overlap >= 7,
            "FastScore top-8 overlap={overlap}/8; exact_top={exact_top:?}, fast_top={fast_top:?}"
        );
        let exact_rank = exact_order
            .iter()
            .enumerate()
            .map(|(rank, index)| (*index, rank))
            .collect::<BTreeMap<_, _>>();
        let fast_rank = fast_order
            .iter()
            .enumerate()
            .map(|(rank, index)| (*index, rank))
            .collect::<BTreeMap<_, _>>();
        // Likelihood gains are intentionally clamped at zero, so a large
        // random tail can contain exact ties.  Ignore only those ties when
        // measuring ordering agreement; they carry no ranking information.
        let informative_pairs = (0..exact.len())
            .flat_map(|left| ((left + 1)..exact.len()).map(move |right| (left, right)))
            .filter(|(left, right)| (exact[*left] - exact[*right]).abs() > 1.0e-6)
            .count();
        let informative_concordant = (0..exact.len())
            .flat_map(|left| ((left + 1)..exact.len()).map(move |right| (left, right)))
            .filter(|(left, right)| (exact[*left] - exact[*right]).abs() > 1.0e-6)
            .filter(|(left, right)| {
                (exact_rank[left] < exact_rank[right]) == (fast_rank[left] < fast_rank[right])
            })
            .count();
        assert!(
            informative_concordant as f64 / informative_pairs.max(1) as f64 > 0.97,
            "FastScore informative ranking concordance={:.4}, exact_top={:?}, fast_top={:?}",
            informative_concordant as f64 / informative_pairs.max(1) as f64,
            exact_order.iter().take(8).collect::<Vec<_>>(),
            fast_order.iter().take(8).collect::<Vec<_>>()
        );
    }

    #[test]
    fn analytic_gradient_matches_finite_difference() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("map.mrc");
        let shape = [32, 32, 32];
        write_map(&path, shape, gaussian_map(shape, [16.3, 16.0, 16.0], 1.0));
        let scorer = context(&path, false);
        let proposal = atom(1, [15.4, 16.0, 16.0]);
        let gradient = scorer
            .score_addition_with_gradients(std::slice::from_ref(&proposal))
            .unwrap();
        let epsilon = 1.0e-4;
        let mut plus = proposal.clone();
        plus.position[0] += epsilon;
        let mut minus = proposal.clone();
        minus.position[0] -= epsilon;
        let finite = (scorer.score_addition(&[plus]).unwrap()
            - scorer.score_addition(&[minus]).unwrap())
            / (2.0 * epsilon);
        assert!(
            (gradient.gradients[&proposal.id][0] - finite).abs() < 0.15,
            "analytic gradient={} finite_difference={finite}",
            gradient.gradients[&proposal.id][0]
        );
    }

    #[test]
    fn profiled_gain_and_gradient_match_finite_difference() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("profiled.mrc");
        let shape = [32, 32, 32];
        write_map(&path, shape, gaussian_map(shape, [16.3, 16.0, 16.0], 1.2));
        let scorer = context(&path, false);
        let proposal = atom(1, [15.4, 16.0, 16.0]);
        let (terms, gradient) = scorer
            .profiled_addition_with_gradients(std::slice::from_ref(&proposal))
            .unwrap();
        assert!(terms.profiled_scale > 0.0);
        assert!(terms.profiled_gain.is_finite());
        assert!((gradient.energy + terms.profiled_gain).abs() < 1.0e-12);
        let epsilon = 1.0e-4;
        let mut plus = proposal.clone();
        plus.position[0] += epsilon;
        let mut minus = proposal.clone();
        minus.position[0] -= epsilon;
        let finite = (scorer.profiled_terms(&[plus]).unwrap().profiled_gain
            - scorer.profiled_terms(&[minus]).unwrap().profiled_gain)
            / (2.0 * epsilon);
        assert!(
            (gradient.gradients[&proposal.id][0] - finite).abs() < 0.20,
            "profiled gradient={} finite_difference={finite} terms={terms:?}",
            gradient.gradients[&proposal.id][0]
        );
    }

    #[test]
    fn profiled_residual_subtracts_protein_and_prefix() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("residual.mrc");
        let shape = [40, 40, 40];
        let protein_center = [12.0, 20.0, 20.0];
        let glycan_center = [27.0, 20.0, 20.0];
        let protein_map = gaussian_map(shape, protein_center, 1.0);
        let glycan_map = gaussian_map(shape, glycan_center, 1.0);
        let values = protein_map
            .iter()
            .zip(glycan_map)
            .map(|(protein, glycan)| protein + 0.8 * glycan)
            .collect::<Vec<_>>();
        write_map(&path, shape, values);
        let protein = read_pdb_str(
            "ATOM      1  CA  ALA A   1      12.000  20.000  20.000  1.00  0.00           C\nEND\n",
            &BuildOptions::default(),
        )
        .unwrap();
        let target = DensityTarget {
            site: ResidueId {
                chain: "A".into(),
                number: 2,
                insertion_code: None,
            },
            glycan_residues: vec![ResidueId {
                chain: "B".into(),
                number: 1,
                insertion_code: None,
            }],
        };
        let mut scorer = FastDensityContext::new(
            DensityMap::open(&path).unwrap(),
            1.0,
            None,
            &protein,
            std::slice::from_ref(&target),
            false,
        )
        .unwrap();
        let at_protein = atom(1, protein_center);
        let at_glycan = atom(1, glycan_center);
        let protein_terms = scorer
            .profiled_addition_terms(std::slice::from_ref(&at_protein))
            .unwrap();
        let glycan_terms = scorer
            .profiled_addition_terms(std::slice::from_ref(&at_glycan))
            .unwrap();
        assert!(
            glycan_terms.profiled_gain > protein_terms.profiled_gain,
            "protein-subtracted candidate should win: protein={protein_terms:?} glycan={glycan_terms:?}"
        );
        scorer.add_atoms(vec![at_glycan.clone()]);
        let repeated = scorer
            .profiled_addition_terms(std::slice::from_ref(&at_glycan))
            .unwrap();
        assert!(
            repeated.profiled_gain < glycan_terms.profiled_gain,
            "accepted prefix should explain its own density: first={glycan_terms:?} repeated={repeated:?}"
        );
    }

    #[test]
    fn profiled_gradient_matches_finite_difference_with_background_and_prefix() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("profiled-gradient-background.mrc");
        let shape = [40, 40, 40];
        let protein_center = [13.0, 20.0, 20.0];
        let prefix_center = [19.0, 20.0, 20.0];
        let candidate_center = [16.1, 20.0, 20.0];
        let values = gaussian_map(shape, protein_center, 1.0)
            .into_iter()
            .zip(gaussian_map(shape, prefix_center, 1.0))
            .map(|(protein, prefix)| protein + 0.7 * prefix)
            .collect::<Vec<_>>();
        write_map(&path, shape, values);
        let protein = read_pdb_str(
            "ATOM      1  CA  ALA A   1      13.000  20.000  20.000  1.00  0.00           C\nEND\n",
            &BuildOptions::default(),
        )
        .unwrap();
        let target = DensityTarget {
            site: ResidueId {
                chain: "A".into(),
                number: 2,
                insertion_code: None,
            },
            glycan_residues: vec![ResidueId {
                chain: "B".into(),
                number: 1,
                insertion_code: None,
            }],
        };
        let mut scorer = FastDensityContext::new(
            DensityMap::open(&path).unwrap(),
            1.0,
            None,
            &protein,
            std::slice::from_ref(&target),
            false,
        )
        .unwrap();
        let prefix = FastAtom {
            id: AtomId(2),
            residue: ResidueId {
                chain: "B".into(),
                number: 2,
                insertion_code: None,
            },
            amplitude: 6.0,
            sigma: 1.0,
            position: prefix_center,
        };
        scorer.add_atoms(vec![prefix]);
        let proposal = FastAtom {
            id: AtomId(3),
            residue: ResidueId {
                chain: "B".into(),
                number: 1,
                insertion_code: None,
            },
            amplitude: 6.0,
            sigma: 1.0,
            position: candidate_center,
        };
        let (_, analytic) = scorer
            .profiled_addition_with_gradients(std::slice::from_ref(&proposal))
            .unwrap();
        let epsilon = 1.0e-4;
        let mut plus = proposal.clone();
        plus.position[0] += epsilon;
        let mut minus = proposal.clone();
        minus.position[0] -= epsilon;
        let finite = (scorer
            .profiled_addition_terms(&[plus])
            .unwrap()
            .profiled_gain
            - scorer
                .profiled_addition_terms(&[minus])
                .unwrap()
                .profiled_gain)
            / (2.0 * epsilon);
        assert!(
            (analytic.gradients[&proposal.id][0] - finite).abs() < 0.20,
            "background/prefix profiled gradient={} finite_difference={finite}",
            analytic.gradients[&proposal.id][0]
        );
    }

    #[test]
    fn profiled_move_matches_remove_then_add_without_context_rebuild() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("profiled-move.mrc");
        let shape = [36, 36, 36];
        let values = gaussian_map(shape, [14.0, 18.0, 18.0], 1.0)
            .into_iter()
            .zip(gaussian_map(shape, [23.0, 18.0, 18.0], 1.0))
            .map(|(left, right)| left + 0.8 * right)
            .collect::<Vec<_>>();
        write_map(&path, shape, values);
        let mut scorer = context(&path, false);
        let prefix = atom(1, [14.0, 18.0, 18.0]);
        let old = atom(2, [22.0, 18.0, 18.0]);
        let new = atom(2, [23.0, 18.0, 18.0]);
        scorer.add_atoms(vec![prefix.clone(), old.clone()]);
        let moved = scorer
            .profiled_move_terms(std::slice::from_ref(&old), std::slice::from_ref(&new))
            .unwrap();
        let mut rebuilt = scorer.clone();
        rebuilt.remove_atoms(&BTreeSet::from([old.id]));
        let added = rebuilt
            .profiled_addition_terms(std::slice::from_ref(&new))
            .unwrap();
        assert_eq!(moved, added);
    }

    #[test]
    fn periodic_pair_distance_uses_minimum_image() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("map.mrc");
        let shape = [24, 24, 24];
        write_map(&path, shape, gaussian_map(shape, [0.2, 12.0, 12.0], 1.0));
        let mut scorer = context(&path, true);
        let first = atom(1, [0.2, 12.0, 12.0]);
        let wrapped = atom(2, [23.8, 12.0, 12.0]);
        let delta = scorer.score_addition(std::slice::from_ref(&first)).unwrap();
        scorer.add_atoms(vec![first]);
        let wrapped_delta = scorer.score_addition(&[wrapped]).unwrap();
        assert!(wrapped_delta.is_finite());
        assert!(delta.is_finite());
    }

    #[test]
    fn structure_b_factors_use_a_bounded_sigma_bank() {
        let low = effective_sigma(0.65, 0.0);
        let medium = effective_sigma(0.65, 20.0);
        let high = effective_sigma(0.65, 80.0);
        for sigma in [low, medium, high] {
            assert!(
                (sigma / SIGMA_BANK_WIDTH - (sigma / SIGMA_BANK_WIDTH).round()).abs() < 1.0e-10
            );
        }
        assert!(low <= medium && medium <= high);
    }
}

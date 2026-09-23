//! Prepared, allocation-free geometry for repeated conformer evaluation.
//!
//! The public build APIs still materialize `glysys::Structure` values for
//! output.  Search and sampling use this module instead: immutable protein
//! data and per-conformer attachment poses are compiled once, then proposals
//! are scored from flat coordinate buffers.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use glysys::{AtomId, ResidueId, Structure, Vec3};
use rayon::prelude::*;
use reglyco_build::{LinkageDefinition, orient_glycan_coordinates_for_frame};
use reglyco_core::{ReGlycoError, SearchSite};

use super::Gene;

type Result<T, E = ReGlycoError> = std::result::Result<T, E>;

const GRID_CELL: f64 = 1.7;

#[derive(Debug, Clone)]
struct Pose {
    coordinates: Vec<Vec3>,
    /// Number of attachment-proximal atoms ignored by the Cookbook steric
    /// scorer (C1 plus its immediate linkage neighbors).
    excluded_prefix: usize,
    target_b: Vec3,
    target_link: Vec3,
    psi_base: f64,
    c1_atom: usize,
    o5_atom: usize,
}

#[derive(Debug, Clone)]
struct PreparedSite {
    poses: Vec<Vec<Pose>>,
    rotamer_updates: Vec<Vec<(usize, Vec3)>>,
    rotamer_probabilities: Vec<f64>,
}

#[derive(Debug, Clone)]
struct FlatAtom {
    id: AtomId,
    position: Vec3,
}

type Cell = (i32, i32, i32);

#[derive(Debug, Clone)]
pub(crate) struct SpatialGrid {
    cells: HashMap<Cell, Vec<usize>>,
}

impl SpatialGrid {
    fn build(coordinates: &[Vec3], indices: impl IntoIterator<Item = usize>) -> Self {
        let mut cells = HashMap::<Cell, Vec<usize>>::new();
        for index in indices {
            cells
                .entry(Self::cell(coordinates[index]))
                .or_default()
                .push(index);
        }
        Self { cells }
    }

    fn cell(position: Vec3) -> Cell {
        (
            (position.x / GRID_CELL).floor() as i32,
            (position.y / GRID_CELL).floor() as i32,
            (position.z / GRID_CELL).floor() as i32,
        )
    }

    fn append_candidates(&self, position: Vec3, radius: i32, result: &mut Vec<usize>) {
        let center = Self::cell(position);
        for dx in -radius..=radius {
            for dy in -radius..=radius {
                for dz in -radius..=radius {
                    if let Some(indices) =
                        self.cells
                            .get(&(center.0 + dx, center.1 + dy, center.2 + dz))
                    {
                        result.extend(indices.iter().copied());
                    }
                }
            }
        }
        // Preserve the reference scorer's atom iteration order while reusing
        // one scratch allocation for the complete proposal evaluation.
        result.sort_unstable();
        result.dedup();
    }
}

#[derive(Debug, Clone)]
pub struct PreparedEvaluation {
    pub site_scores: Vec<f64>,
    pub score: f64,
}

/// A single attachment pose which has already passed the protein-contact
/// screen. Ensemble generation keeps these small prepared poses in per-site
/// pools and combines them without rebuilding a complete glycoprotein for
/// every rejected joint draw (the fast Cookbook ensemble path).
#[derive(Debug, Clone)]
pub(crate) struct PreparedSitePose {
    coordinates: Arc<Vec<Vec3>>,
    excluded_prefix: usize,
    grid: Arc<SpatialGrid>,
    bounds: Option<([f64; 3], [f64; 3])>,
}

fn pose_bounds(coordinates: &[Vec3], excluded_prefix: usize) -> Option<([f64; 3], [f64; 3])> {
    let mut atoms = coordinates.iter().skip(excluded_prefix);
    let first = atoms.next()?;
    let mut low = [first.x, first.y, first.z];
    let mut high = low;
    for position in atoms {
        for (axis, value) in [position.x, position.y, position.z].into_iter().enumerate() {
            low[axis] = low[axis].min(value);
            high[axis] = high[axis].max(value);
        }
    }
    Some((low, high))
}

#[derive(Debug, Clone)]
pub struct PreparedAttachmentContext {
    protein_coordinates: Vec<Vec3>,
    protein_indices: Vec<usize>,
    static_grid: SpatialGrid,
    sites: Vec<PreparedSite>,
}

impl PreparedAttachmentContext {
    pub(crate) fn site_count(&self) -> usize {
        self.sites.len()
    }

    pub fn new(
        protein: &Structure,
        sites: &[SearchSite],
        include_rotamers: bool,
    ) -> Result<Self, ReGlycoError> {
        if sites.is_empty() || sites.iter().any(|site| site.ensemble.conformers.is_empty()) {
            return Err(ReGlycoError::MissingGlycanAtom("empty ensemble".into()));
        }

        let all_glycan_residues = protein
            .metadata()
            .glycan_trees
            .iter()
            .flat_map(|tree| tree.residue_ids.iter().cloned())
            .collect::<HashSet<_>>();
        let atoms = protein
            .atoms()
            .into_iter()
            .filter(|atom| !all_glycan_residues.contains(&atom.residue))
            .map(|atom| FlatAtom {
                id: atom.id,
                position: atom.position,
            })
            .collect::<Vec<_>>();
        let protein_coordinates = atoms.iter().map(|atom| atom.position).collect::<Vec<_>>();
        let protein_indices = (0..atoms.len()).collect::<Vec<_>>();
        let static_grid = SpatialGrid::build(&protein_coordinates, protein_indices.iter().copied());

        let mut prepared_sites = Vec::with_capacity(sites.len());
        for site in sites {
            let residue = protein
                .residues()
                .into_iter()
                .find(|residue| residue.id == site.site.residue)
                .ok_or_else(|| ReGlycoError::SiteNotFound(site.site.residue.clone()))?;
            let rule = LinkageDefinition::for_residue(&site.site.residue, &residue.name)?;
            let target_ids = [
                find_atom_id(protein, &site.site.residue, rule.frame_a)?,
                find_atom_id(protein, &site.site.residue, rule.frame_b)?,
                find_atom_id(protein, &site.site.residue, rule.link_atom)?,
            ];
            let target_indices = [
                atoms
                    .iter()
                    .position(|atom| atom.id == target_ids[0])
                    .ok_or_else(|| ReGlycoError::MissingSiteAtom {
                        site: site.site.residue.clone(),
                        atom: target_ids[0].0.to_string(),
                    })?,
                atoms
                    .iter()
                    .position(|atom| atom.id == target_ids[1])
                    .ok_or_else(|| ReGlycoError::MissingSiteAtom {
                        site: site.site.residue.clone(),
                        atom: target_ids[1].0.to_string(),
                    })?,
                atoms
                    .iter()
                    .position(|atom| atom.id == target_ids[2])
                    .ok_or_else(|| ReGlycoError::MissingSiteAtom {
                        site: site.site.residue.clone(),
                        atom: target_ids[2].0.to_string(),
                    })?,
            ];

            let (rotamer_updates, rotamer_probabilities) = if include_rotamers {
                let mut updates_by_rotamer = vec![Vec::new()];
                let mut probabilities = Vec::new();
                let rotamers = super::dunbrack::rotamers(protein, &site.site.residue);
                for (rotamer_index, rotamer) in rotamers.iter().enumerate() {
                    let mut rotated = protein.clone();
                    super::dunbrack::apply(&mut rotated, &site.site.residue, rotamer_index)?;
                    let updates = atoms
                        .iter()
                        .enumerate()
                        .filter_map(|(index, atom)| {
                            let position = rotated.atom_position(atom.id)?;
                            (position != atom.position).then_some((index, position))
                        })
                        .collect::<Vec<_>>();
                    updates_by_rotamer.push(updates);
                    probabilities.push(rotamer.probability);
                }
                (updates_by_rotamer, probabilities)
            } else {
                (vec![Vec::new()], Vec::new())
            };

            let mut poses = Vec::with_capacity(rotamer_updates.len());
            for (_rotamer_slot, updates) in rotamer_updates.iter().enumerate() {
                let mut coordinates = protein_coordinates.clone();
                for &(index, position) in updates {
                    coordinates[index] = position;
                }
                let target_a = coordinates[target_indices[0]];
                let target_b = coordinates[target_indices[1]];
                let target_link = coordinates[target_indices[2]];
                let conformer_poses = site
                    .ensemble
                    .conformers
                    .par_iter()
                    .map(|conformer| {
                        // Build each site's pose against its own protein frame.
                        // A previous cross-site canonical-pose shortcut mapped
                        // frames approximately and introduced small coordinate
                        // errors for repeated glycans (notably dense UniProt
                        // one-shot requests).  Those errors could create false
                        // glycan-glycan clashes compared with the materialized
                        // Cookbook path, so only the immutable source ensemble
                        // is shared; attachment orientation is site-local.
                        make_conformer_pose_from_reference(
                            site,
                            &residue.name,
                            conformer,
                            target_a,
                            target_b,
                            target_link,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                poses.push(conformer_poses);
            }
            prepared_sites.push(PreparedSite {
                poses,
                rotamer_updates,
                rotamer_probabilities,
            });
        }

        Ok(Self {
            protein_coordinates,
            protein_indices,
            static_grid,
            sites: prepared_sites,
        })
    }

    pub fn evaluate(
        &self,
        state: &[Gene],
        clash_distance: f64,
    ) -> Result<PreparedEvaluation, ReGlycoError> {
        if state.len() != self.sites.len() {
            return Err(ReGlycoError::InvalidGeometry);
        }
        let coordinates = self.transform_state(state)?;
        let references = coordinates.iter().map(Vec::as_slice).collect::<Vec<_>>();
        if state.iter().all(|gene| gene.rotamer.is_none()) {
            return self.evaluate_coordinates(
                state,
                &references,
                &self.protein_coordinates,
                &self.static_grid,
                None,
                clash_distance,
            );
        }
        let (protein_coordinates, grid) = self.protein_frame(state)?;
        self.evaluate_coordinates(
            state,
            &references,
            &protein_coordinates,
            &grid,
            None,
            clash_distance,
        )
    }

    /// Prepare one site's pose and apply only the protein/glycan steric test.
    /// Glycan/glycan compatibility is evaluated later when the per-site pools
    /// are combined.
    pub(crate) fn prepare_site_pose(
        &self,
        site_index: usize,
        gene: &Gene,
        clash_distance: f64,
    ) -> Result<Option<PreparedSitePose>, ReGlycoError> {
        let site = self
            .sites
            .get(site_index)
            .ok_or(ReGlycoError::InvalidGeometry)?;
        let rotamer = gene.rotamer.map_or(0, |value| value + 1);
        let pose = site
            .poses
            .get(rotamer)
            .and_then(|conformers| conformers.get(gene.conformer))
            .ok_or(ReGlycoError::InvalidGeometry)?;
        let coordinates = transform_pose(pose, gene.phi, gene.psi)?;

        if rotamer != 0 {
            let mut protein_coordinates = self.protein_coordinates.clone();
            for &(index, position) in site
                .rotamer_updates
                .get(rotamer)
                .ok_or(ReGlycoError::InvalidGeometry)?
            {
                protein_coordinates[index] = position;
            }
            // This local grid lives only for the score below.
            // Keeping the branch here avoids rebuilding it for the common
            // no-rotamer path.
            let dynamic_grid =
                SpatialGrid::build(&protein_coordinates, self.protein_indices.iter().copied());
            let score = protein_score(
                &coordinates,
                pose.excluded_prefix,
                &protein_coordinates,
                &dynamic_grid,
                clash_distance,
            );
            return Ok((score <= 1.1).then(|| PreparedSitePose {
                grid: Arc::new(SpatialGrid::build(&coordinates, 0..coordinates.len())),
                bounds: pose_bounds(&coordinates, pose.excluded_prefix),
                coordinates: Arc::new(coordinates),
                excluded_prefix: pose.excluded_prefix,
            }));
        }

        let score = protein_score(
            &coordinates,
            pose.excluded_prefix,
            &self.protein_coordinates,
            &self.static_grid,
            clash_distance,
        );
        Ok((score <= 1.1).then(|| PreparedSitePose {
            grid: Arc::new(SpatialGrid::build(&coordinates, 0..coordinates.len())),
            bounds: pose_bounds(&coordinates, pose.excluded_prefix),
            coordinates: Arc::new(coordinates),
            excluded_prefix: pose.excluded_prefix,
        }))
    }

    pub(crate) fn site_poses_compatible(
        &self,
        first: &PreparedSitePose,
        second: &PreparedSitePose,
        clash_distance: f64,
    ) -> bool {
        // The full atom/grid test is expensive for the dense all-site pose
        // tables. Disjoint bounding boxes cannot contain a clashing atom pair.
        if let (Some((first_low, first_high)), Some((second_low, second_high))) =
            (first.bounds, second.bounds)
        {
            if (0..3).any(|axis| {
                first_high[axis] + clash_distance <= second_low[axis]
                    || second_high[axis] + clash_distance <= first_low[axis]
            }) {
                return true;
            }
        } else {
            return true;
        }
        let cell_radius = (clash_distance / GRID_CELL).ceil().max(1.0) as i32;
        pair_score(
            &first.coordinates,
            first.excluded_prefix,
            &second.coordinates,
            Some(second.excluded_prefix),
            clash_distance,
            cell_radius,
            &second.grid,
            second.bounds,
        ) <= 1.1
    }

    /// Evaluate a chromosome while reusing transformed coordinates for sites
    /// that were frozen by the Cookbook-style search.  The caller marks a
    /// site dirty whenever its conformer, torsions, or rotamer changes.
    pub(crate) fn evaluate_cached(
        &self,
        state: &[Gene],
        caches: &mut [Option<Vec<Vec3>>],
        grids: &mut [Option<Arc<SpatialGrid>>],
        dirty: &[bool],
        clash_distance: f64,
    ) -> Result<PreparedEvaluation, ReGlycoError> {
        if caches.len() != self.sites.len()
            || grids.len() != self.sites.len()
            || dirty.len() != self.sites.len()
        {
            return Err(ReGlycoError::InvalidGeometry);
        }
        for (index, (gene, site)) in state.iter().zip(&self.sites).enumerate() {
            if !dirty[index] && caches[index].is_some() {
                continue;
            }
            let rotamer = gene.rotamer.map_or(0, |value| value + 1);
            let pose = site
                .poses
                .get(rotamer)
                .and_then(|conformers| conformers.get(gene.conformer))
                .ok_or(ReGlycoError::InvalidGeometry)?;
            caches[index] = Some(transform_pose(pose, gene.phi, gene.psi)?);
        }
        let references = caches
            .iter()
            .map(|coordinates| coordinates.as_deref().ok_or(ReGlycoError::InvalidGeometry))
            .collect::<Result<Vec<_>, _>>()?;
        // The overwhelmingly common path keeps the deposited receptor frame
        // and its spatial index immutable.  Avoid cloning all receptor
        // coordinates and rebuilding the index for every chromosome when no
        // rotamer is active.
        if state.iter().all(|gene| gene.rotamer.is_none()) {
            for (grid, coordinates) in grids.iter_mut().zip(caches.iter()) {
                if grid.is_none() {
                    let coordinates = coordinates
                        .as_deref()
                        .ok_or(ReGlycoError::InvalidGeometry)?;
                    *grid = Some(Arc::new(SpatialGrid::build(
                        coordinates,
                        0..coordinates.len(),
                    )));
                }
            }
            let grid_refs = grids
                .iter()
                .map(|grid| grid.as_ref().cloned().ok_or(ReGlycoError::InvalidGeometry))
                .collect::<Result<Vec<_>, _>>()?;
            return self.evaluate_coordinates(
                state,
                &references,
                &self.protein_coordinates,
                &self.static_grid,
                Some(&grid_refs),
                clash_distance,
            );
        }
        for (grid, coordinates) in grids.iter_mut().zip(caches.iter()) {
            if grid.is_none() {
                let coordinates = coordinates
                    .as_deref()
                    .ok_or(ReGlycoError::InvalidGeometry)?;
                *grid = Some(Arc::new(SpatialGrid::build(
                    coordinates,
                    0..coordinates.len(),
                )));
            }
        }
        let grid_refs = grids
            .iter()
            .map(|grid| grid.as_ref().cloned().ok_or(ReGlycoError::InvalidGeometry))
            .collect::<Result<Vec<_>, _>>()?;
        let (protein_coordinates, grid) = self.protein_frame(state)?;
        self.evaluate_coordinates(
            state,
            &references,
            &protein_coordinates,
            &grid,
            Some(&grid_refs),
            clash_distance,
        )
    }

    fn protein_frame(&self, state: &[Gene]) -> Result<(Vec<Vec3>, SpatialGrid), ReGlycoError> {
        let mut protein_coordinates = self.protein_coordinates.clone();
        let mut dynamic = false;
        for (gene, site) in state.iter().zip(&self.sites) {
            if let Some(rotamer) = gene.rotamer {
                let updates = site.rotamer_updates.get(rotamer + 1).ok_or_else(|| {
                    ReGlycoError::InvalidChemistry {
                        site: ResidueId {
                            chain: String::new(),
                            number: 0,
                            insertion_code: None,
                        },
                        message: format!("rotamer {rotamer} is unavailable"),
                    }
                })?;
                dynamic = true;
                for &(index, position) in updates {
                    protein_coordinates[index] = position;
                }
            }
        }
        let grid = if dynamic {
            SpatialGrid::build(&protein_coordinates, self.protein_indices.iter().copied())
        } else {
            self.static_grid.clone()
        };
        Ok((protein_coordinates, grid))
    }

    pub(crate) fn transform_state(&self, state: &[Gene]) -> Result<Vec<Vec<Vec3>>, ReGlycoError> {
        if state.len() != self.sites.len() {
            return Err(ReGlycoError::InvalidGeometry);
        }
        state
            .iter()
            .zip(&self.sites)
            .map(|(gene, site)| {
                let rotamer = gene.rotamer.map_or(0, |value| value + 1);
                let pose = site
                    .poses
                    .get(rotamer)
                    .and_then(|conformers| conformers.get(gene.conformer))
                    .ok_or(ReGlycoError::InvalidGeometry)?;
                transform_pose(pose, gene.phi, gene.psi)
            })
            .collect()
    }

    fn evaluate_coordinates(
        &self,
        state: &[Gene],
        coordinates: &[&[Vec3]],
        protein_coordinates: &[Vec3],
        grid: &SpatialGrid,
        cached_glycan_grids: Option<&[Arc<SpatialGrid>]>,
        clash_distance: f64,
    ) -> Result<PreparedEvaluation, ReGlycoError> {
        if state.len() != self.sites.len() || coordinates.len() != self.sites.len() {
            return Err(ReGlycoError::InvalidGeometry);
        }
        let poses = state
            .iter()
            .zip(&self.sites)
            .map(|(gene, site)| {
                let rotamer = gene.rotamer.map_or(0, |index| index + 1);
                site.poses
                    .get(rotamer)
                    .and_then(|conformers| conformers.get(gene.conformer))
                    .ok_or(ReGlycoError::InvalidGeometry)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let threshold2 = clash_distance * clash_distance;
        let cell_radius = (clash_distance / GRID_CELL).ceil().max(1.0) as i32;
        let owned_glycan_grids;
        let glycan_grids: &[Arc<SpatialGrid>] = if let Some(grids) = cached_glycan_grids {
            grids
        } else {
            owned_glycan_grids = coordinates
                .iter()
                .map(|coordinates| Arc::new(SpatialGrid::build(coordinates, 0..coordinates.len())))
                .collect::<Vec<_>>();
            &owned_glycan_grids
        };
        let mut candidates = Vec::with_capacity(64);
        let site_scores = poses
            .iter()
            .zip(coordinates.iter())
            .enumerate()
            .map(|(index, (pose, current_coordinates))| {
                let mut score = 1.0;
                for (atom_index, position) in current_coordinates.iter().enumerate() {
                    if atom_index < pose.excluded_prefix {
                        continue;
                    }
                    candidates.clear();
                    grid.append_candidates(*position, cell_radius, &mut candidates);
                    for &protein_index in &candidates {
                        let distance2 =
                            squared_distance(*position, protein_coordinates[protein_index]);
                        if distance2 < threshold2 {
                            let distance = distance2.sqrt();
                            score += 200.0 * (-distance * distance).exp();
                            if score > 2.0 {
                                break;
                            }
                        }
                    }
                    if score > 2.0 {
                        break;
                    }
                }
                for (other_index, other_coordinates) in coordinates.iter().enumerate() {
                    if other_index == index || score > 2.0 {
                        continue;
                    }
                    score = score.max(pair_score(
                        current_coordinates,
                        pose.excluded_prefix,
                        other_coordinates,
                        Some(poses[other_index].excluded_prefix),
                        clash_distance,
                        cell_radius,
                        &glycan_grids[other_index],
                        None,
                    ));
                }
                score
            })
            .collect::<Vec<_>>();
        let score = site_scores.iter().copied().fold(1.0, f64::max);
        Ok(PreparedEvaluation { site_scores, score })
    }

    pub fn rotamer_probability(&self, site: usize, rotamer: Option<usize>) -> Option<f64> {
        rotamer.and_then(|index| {
            self.sites
                .get(site)?
                .rotamer_probabilities
                .get(index)
                .copied()
        })
    }

    #[cfg(test)]
    pub(crate) fn site_coordinates(&self, state: &[Gene]) -> Result<Vec<Vec<Vec3>>, ReGlycoError> {
        state
            .iter()
            .zip(&self.sites)
            .map(|(gene, site)| {
                let rotamer = gene.rotamer.map_or(0, |index| index + 1);
                let pose = site
                    .poses
                    .get(rotamer)
                    .and_then(|conformers| conformers.get(gene.conformer))
                    .ok_or(ReGlycoError::InvalidGeometry)?;
                transform_pose(pose, gene.phi, gene.psi)
            })
            .collect()
    }
}

#[cfg(test)]
mod bounds_tests {
    use super::*;

    fn candidate(points: &[[f64; 3]], excluded_prefix: usize) -> PreparedSitePose {
        let coordinates = points
            .iter()
            .map(|point| Vec3 {
                x: point[0],
                y: point[1],
                z: point[2],
            })
            .collect::<Vec<_>>();
        PreparedSitePose {
            bounds: pose_bounds(&coordinates, excluded_prefix),
            grid: Arc::new(SpatialGrid::build(&coordinates, 0..coordinates.len())),
            coordinates: Arc::new(coordinates),
            excluded_prefix,
        }
    }

    #[test]
    fn pair_bounds_preserve_clash_decisions_and_ignore_attachment_prefix() {
        let context = PreparedAttachmentContext {
            protein_coordinates: Vec::new(),
            protein_indices: Vec::new(),
            static_grid: SpatialGrid::build(&[], std::iter::empty()),
            sites: Vec::new(),
        };
        let first = candidate(&[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], 1);
        let far = candidate(&[[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], 1);
        let near = candidate(&[[0.0, 0.0, 0.0], [10.5, 0.0, 0.0]], 1);
        assert!(context.site_poses_compatible(&first, &far, 1.7));
        assert!(!context.site_poses_compatible(&first, &near, 1.7));
        assert!(context.site_poses_compatible(&first, &candidate(&[[10.0, 0.0, 0.0]], 1), 1.7));
    }
}

fn protein_score(
    coordinates: &[Vec3],
    excluded_prefix: usize,
    protein_coordinates: &[Vec3],
    grid: &SpatialGrid,
    clash_distance: f64,
) -> f64 {
    let threshold2 = clash_distance * clash_distance;
    let cell_radius = (clash_distance / GRID_CELL).ceil().max(1.0) as i32;
    let mut candidates = Vec::with_capacity(64);
    let mut score = 1.0;
    for (atom_index, position) in coordinates.iter().enumerate() {
        if atom_index < excluded_prefix {
            continue;
        }
        candidates.clear();
        grid.append_candidates(*position, cell_radius, &mut candidates);
        for &protein_index in &candidates {
            let distance2 = squared_distance(*position, protein_coordinates[protein_index]);
            if distance2 < threshold2 {
                score += 200.0 * (-distance2).exp();
                if score > 2.0 {
                    return score;
                }
            }
        }
    }
    score
}

fn make_conformer_pose_from_reference(
    site: &SearchSite,
    residue_name: &str,
    conformer: &reglyco_core::EnsembleConformer,
    target_a: Vec3,
    target_b: Vec3,
    target_link: Vec3,
) -> Result<Pose, ReGlycoError> {
    let oriented = orient_glycan_coordinates_for_frame(
        &conformer.structure,
        &site.site.residue,
        residue_name,
        target_a,
        target_b,
        target_link,
        0.0,
        0.0,
    )?;
    let c1_atom = oriented.c1_index;
    let o5_atom = oriented.o5_index;
    let mut coordinates = oriented
        .atoms
        .iter()
        .map(|(_, position)| *position)
        .collect::<Vec<_>>();
    let base_phi = dihedral_degrees(
        target_b,
        target_link,
        coordinates[c1_atom],
        coordinates[o5_atom],
    );
    let c1 = coordinates[c1_atom];
    rotate_all(
        &mut coordinates,
        target_link,
        subtract(c1, target_link),
        (-base_phi).to_radians(),
    )?;
    let psi_base = dihedral_degrees(target_a, target_b, target_link, coordinates[c1_atom]);
    let excluded_prefix = 3.min(coordinates.len());
    Ok(Pose {
        coordinates,
        excluded_prefix,
        target_b,
        target_link,
        c1_atom,
        o5_atom,
        psi_base,
    })
}

fn transform_pose(pose: &Pose, phi: f64, psi: f64) -> Result<Vec<Vec3>, ReGlycoError> {
    let mut coordinates = pose.coordinates.clone();
    let psi_now = pose.psi_base;
    rotate_all(
        &mut coordinates,
        pose.target_link,
        subtract(pose.target_link, pose.target_b),
        (psi - psi_now).to_radians(),
    )?;
    let c1 = coordinates[pose.c1_atom];
    let o5 = coordinates[pose.o5_atom];
    let phi_now = dihedral_degrees(pose.target_b, pose.target_link, c1, o5);
    rotate_all(
        &mut coordinates,
        pose.target_link,
        subtract(c1, pose.target_link),
        (phi - phi_now).to_radians(),
    )?;
    Ok(coordinates)
}

fn rotate_all(
    coordinates: &mut [Vec3],
    origin: Vec3,
    axis: Vec3,
    radians: f64,
) -> Result<(), ReGlycoError> {
    let axis = normalize(axis)?;
    let cosine = radians.cos();
    let sine = radians.sin();
    for position in coordinates {
        let relative = subtract(*position, origin);
        let rotated = add(
            add(scale(relative, cosine), scale(cross(axis, relative), sine)),
            scale(axis, dot(axis, relative) * (1.0 - cosine)),
        );
        *position = add(origin, rotated);
    }
    Ok(())
}

fn pair_score(
    first: &[Vec3],
    excluded_first: usize,
    second: &[Vec3],
    excluded_second: Option<usize>,
    threshold: f64,
    cell_radius: i32,
    grid: &SpatialGrid,
    second_bounds: Option<([f64; 3], [f64; 3])>,
) -> f64 {
    let mut candidates = Vec::with_capacity(64);
    let threshold2 = threshold * threshold;
    let mut score = 1.0;
    for (first_index, first_position) in first.iter().enumerate() {
        if first_index < excluded_first {
            continue;
        }
        if let Some((low, high)) = second_bounds {
            if first_position.x + threshold <= low[0]
                || first_position.x - threshold >= high[0]
                || first_position.y + threshold <= low[1]
                || first_position.y - threshold >= high[1]
                || first_position.z + threshold <= low[2]
                || first_position.z - threshold >= high[2]
            {
                continue;
            }
        }
        candidates.clear();
        grid.append_candidates(*first_position, cell_radius, &mut candidates);
        for &second_index in &candidates {
            if second_index < excluded_second.unwrap_or(0) {
                continue;
            }
            let second_position = &second[second_index];
            let distance2 = squared_distance(*first_position, *second_position);
            if distance2 < threshold2 {
                let distance = distance2.sqrt();
                score += 200.0 * (-distance * distance).exp();
                if score > 2.0 {
                    return score;
                }
            }
        }
    }
    score
}

fn find_atom_id(
    structure: &Structure,
    residue: &ResidueId,
    name: &str,
) -> Result<AtomId, ReGlycoError> {
    structure.find_atom(residue, name).ok_or_else(|| {
        if residue.number == 0 {
            ReGlycoError::MissingGlycanAtom(name.into())
        } else {
            ReGlycoError::MissingSiteAtom {
                site: residue.clone(),
                atom: name.into(),
            }
        }
    })
}

fn squared_distance(first: Vec3, second: Vec3) -> f64 {
    (first.x - second.x).powi(2) + (first.y - second.y).powi(2) + (first.z - second.z).powi(2)
}

fn add(first: Vec3, second: Vec3) -> Vec3 {
    Vec3 {
        x: first.x + second.x,
        y: first.y + second.y,
        z: first.z + second.z,
    }
}

fn subtract(first: Vec3, second: Vec3) -> Vec3 {
    Vec3 {
        x: first.x - second.x,
        y: first.y - second.y,
        z: first.z - second.z,
    }
}

fn scale(vector: Vec3, factor: f64) -> Vec3 {
    Vec3 {
        x: vector.x * factor,
        y: vector.y * factor,
        z: vector.z * factor,
    }
}

fn dot(first: Vec3, second: Vec3) -> f64 {
    first.x * second.x + first.y * second.y + first.z * second.z
}

fn cross(first: Vec3, second: Vec3) -> Vec3 {
    Vec3 {
        x: first.y * second.z - first.z * second.y,
        y: first.z * second.x - first.x * second.z,
        z: first.x * second.y - first.y * second.x,
    }
}

fn normalize(vector: Vec3) -> Result<Vec3, ReGlycoError> {
    let length = dot(vector, vector).sqrt();
    if length <= 1.0e-8 {
        Err(ReGlycoError::InvalidGeometry)
    } else {
        Ok(scale(vector, 1.0 / length))
    }
}

fn dihedral_degrees(first: Vec3, second: Vec3, third: Vec3, fourth: Vec3) -> f64 {
    let b1 = normalize(subtract(third, second)).unwrap_or(Vec3 {
        x: 1.0,
        y: 0.0,
        z: 0.0,
    });
    let b0 = subtract(first, second);
    let b2 = subtract(fourth, third);
    let v = subtract(b0, scale(b1, dot(b0, b1)));
    let w = subtract(b2, scale(b1, dot(b2, b1)));
    dot(cross(b1, v), w).atan2(dot(v, w)).to_degrees()
}

#[cfg(feature = "webgpu")]
impl PreparedAttachmentContext {
    pub(crate) fn gpu_library(
        &self,
    ) -> Result<
        (
            glysys_gpu::steric::AttachmentLibrary,
            Vec<Vec<Vec<u32>>>,
            Vec<u32>,
        ),
        ReGlycoError,
    > {
        use glysys_gpu::steric::{AttachmentLibrary, AttachmentPose, ReceptorUpdate};
        let origin = self.protein_coordinates.first().copied().unwrap_or(Vec3 {
            x: 0.,
            y: 0.,
            z: 0.,
        });
        let point = |p: Vec3| {
            [
                (p.x - origin.x) as f32,
                (p.y - origin.y) as f32,
                (p.z - origin.z) as f32,
                0.,
            ]
        };
        let mut library = AttachmentLibrary {
            protein: self
                .protein_coordinates
                .iter()
                .copied()
                .map(point)
                .collect(),
            poses: Vec::new(),
            coordinates: Vec::new(),
            updates: Vec::new(),
            sites: self.sites.len() as u32,
            candidate_atoms: 0,
        };
        let mut ids = Vec::new();
        let mut offsets = Vec::new();
        for site in &self.sites {
            offsets.push(library.candidate_atoms);
            let Some(default_rotamer) = site.poses.first() else {
                return Err(ReGlycoError::InvalidGeometry);
            };
            let Some(first_pose) = default_rotamer.first() else {
                return Err(ReGlycoError::InvalidGeometry);
            };
            let count = first_pose.coordinates.len();
            library.candidate_atoms += count as u32;
            let mut rotations = Vec::new();
            for (rotamer, poses) in site.poses.iter().enumerate() {
                let mut conformers = Vec::new();
                let start = library.updates.len() as u32;
                let Some(updates) = site.rotamer_updates.get(rotamer) else {
                    return Err(ReGlycoError::InvalidGeometry);
                };
                for &(index, p) in updates {
                    library.updates.push(ReceptorUpdate {
                        value: point(p),
                        indices: [index as u32, 0, 0, 0],
                    });
                }
                let end = library.updates.len() as u32;
                for pose in poses {
                    if pose.coordinates.len() != count {
                        return Err(ReGlycoError::InvalidGeometry);
                    }
                    conformers.push(library.poses.len() as u32);
                    let mut link = point(pose.target_link);
                    link[3] = pose.psi_base.to_radians() as f32;
                    library.poses.push(AttachmentPose {
                        bounds: [
                            library.coordinates.len() as u32,
                            count as u32,
                            pose.excluded_prefix as u32,
                            pose.c1_atom as u32,
                        ],
                        indices: [pose.o5_atom as u32, start, end, 0],
                        b: point(pose.target_b),
                        link,
                    });
                    library
                        .coordinates
                        .extend(pose.coordinates.iter().copied().map(point));
                }
                rotations.push(conformers);
            }
            ids.push(rotations);
        }
        Ok((library, ids, offsets))
    }
}

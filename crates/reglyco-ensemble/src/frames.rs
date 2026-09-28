//! Memory-compact storage for attached-ensemble output frames.
//!
//! Every sampled frame is a complete glycoprotein `Structure`, and each atom
//! record owns several strings.  Keeping hundreds of such frames alive at once
//! is harmless natively but exhausts the 4 GiB wasm32 address space in the
//! browser (a 16-site tetramer holds ~6 MiB per frame).  Frames drawn from one
//! attached ensemble share their topology, so the store keeps one template
//! structure and, per frame, only the atom coordinates that differ from it.
//! Structures are rebuilt exactly (full `f64` coordinates) on demand.

use glysys::{AtomId, Structure, Vec3};
use reglyco_core::SearchSiteResult;

use crate::{Result, SampledFrame};

#[derive(Debug, Clone)]
enum FrameGeometry {
    /// Coordinates that differ from the template, keyed by stable atom id.
    Delta(Vec<(AtomId, Vec3)>),
    /// A frame whose topology differs from the template, kept verbatim.
    Full(Box<Structure>),
}

/// An ordered set of structures that share one topology template.
#[derive(Debug, Clone, Default)]
pub struct StructureStore {
    template: Option<Structure>,
    template_bonds: Vec<(AtomId, AtomId)>,
    entries: Vec<FrameGeometry>,
}

impl StructureStore {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            template: None,
            template_bonds: Vec::new(),
            entries: Vec::with_capacity(capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn push(&mut self, structure: Structure) {
        let Some(template) = &self.template else {
            self.template_bonds = structure.bonds();
            self.template = Some(structure);
            self.entries.push(FrameGeometry::Delta(Vec::new()));
            return;
        };
        let geometry = match coordinate_delta(template, &self.template_bonds, &structure) {
            Some(delta) => FrameGeometry::Delta(delta),
            None => FrameGeometry::Full(Box::new(structure)),
        };
        self.entries.push(geometry);
    }

    /// Rebuild the structure at `index`.
    pub fn get(&self, index: usize) -> Result<Structure> {
        let entry = self.entries.get(index).ok_or_else(|| {
            crate::EnsembleError::Metadata(format!("ensemble frame {index} does not exist"))
        })?;
        match entry {
            FrameGeometry::Full(structure) => Ok(structure.as_ref().clone()),
            FrameGeometry::Delta(delta) => {
                let mut structure = self
                    .template
                    .clone()
                    .ok_or_else(|| crate::EnsembleError::Metadata("empty frame store".into()))?;
                structure.set_atom_positions(delta.iter().copied())?;
                Ok(structure)
            }
        }
    }

    /// Rebuild structures one at a time; only one is alive per iteration.
    pub fn iter(&self) -> impl Iterator<Item = Result<Structure>> + '_ {
        (0..self.len()).map(|index| self.get(index))
    }

    /// Write all structures as one multi-model PDB without materializing
    /// every frame at once.  The output buffer is reserved from the first
    /// model's size because fixed-width PDB models are the same length; this
    /// avoids transient doubling while a large ensemble string grows.
    pub fn write_multi_model_pdb(&self, output: &mut String) -> Result<()> {
        for (index, structure) in self.iter().enumerate() {
            let model = structure?.to_pdb_string();
            if index == 0 {
                let estimate = model.len().saturating_add(32).saturating_mul(self.len());
                output.reserve(estimate.saturating_add(4));
            }
            output.push_str(&format!("MODEL     {:>4}\n", index + 1));
            for line in model.lines().filter(|line| !line.starts_with("END")) {
                output.push_str(line);
                output.push('\n');
            }
            output.push_str("ENDMDL\n");
        }
        output.push_str("END\n");
        Ok(())
    }

    pub fn multi_model_pdb(&self) -> Result<String> {
        let mut output = String::new();
        self.write_multi_model_pdb(&mut output)?;
        Ok(output)
    }
}

/// Coordinates that turn `template` into `structure`, or `None` when the two
/// structures do not share atom identities, bonds, and metadata (in which case
/// the caller must keep the structure verbatim).
fn coordinate_delta(
    template: &Structure,
    template_bonds: &[(AtomId, AtomId)],
    structure: &Structure,
) -> Option<Vec<(AtomId, Vec3)>> {
    let mut delta = Vec::new();
    let mut template_atoms = template.iter_atoms();
    for atom in structure.iter_atoms() {
        let reference = template_atoms.next()?;
        if reference.id != atom.id
            || reference.name != atom.name
            || reference.residue != atom.residue
            || reference.residue_name != atom.residue_name
            || reference.element != atom.element
            || reference.occupancy.to_bits() != atom.occupancy.to_bits()
            || reference.b_factor.to_bits() != atom.b_factor.to_bits()
        {
            return None;
        }
        let (a, b) = (reference.position, atom.position);
        if a.x.to_bits() != b.x.to_bits()
            || a.y.to_bits() != b.y.to_bits()
            || a.z.to_bits() != b.z.to_bits()
        {
            delta.push((atom.id, b));
        }
    }
    if template_atoms.next().is_some()
        || template.metadata() != structure.metadata()
        || template_bonds != structure.bonds().as_slice()
    {
        return None;
    }
    Some(delta)
}

/// Per-frame sampler annotations, without the frame's coordinates.
#[derive(Debug, Clone)]
pub struct FrameRecord {
    pub sites: Vec<SearchSiteResult>,
    pub proposal_index: usize,
    pub source: String,
    pub log_native_probability: f64,
    pub selected_energy_kcal_per_mol: Option<f64>,
}

/// Sampled frames whose coordinates are held in a [`StructureStore`].
#[derive(Debug, Clone, Default)]
pub struct CompactFrames {
    structures: StructureStore,
    records: Vec<FrameRecord>,
}

impl CompactFrames {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            structures: StructureStore::with_capacity(capacity),
            records: Vec::with_capacity(capacity),
        }
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn push(&mut self, frame: SampledFrame) {
        let SampledFrame {
            structure,
            sites,
            proposal_index,
            source,
            log_native_probability,
            selected_energy_kcal_per_mol,
        } = frame;
        self.structures.push(structure);
        self.records.push(FrameRecord {
            sites,
            proposal_index,
            source,
            log_native_probability,
            selected_energy_kcal_per_mol,
        });
    }

    pub fn iter(&self) -> std::slice::Iter<'_, FrameRecord> {
        self.records.iter()
    }

    pub fn iter_mut(&mut self) -> std::slice::IterMut<'_, FrameRecord> {
        self.records.iter_mut()
    }

    pub fn first(&self) -> Option<&FrameRecord> {
        self.records.first()
    }

    pub fn record_mut(&mut self, index: usize) -> Option<&mut FrameRecord> {
        self.records.get_mut(index)
    }

    pub fn structure(&self, index: usize) -> Result<Structure> {
        self.structures.get(index)
    }

    pub fn structures(&self) -> &StructureStore {
        &self.structures
    }

    /// Replace every frame's coordinates (for example after relaxation)
    /// while keeping the sampler annotations.
    pub fn replace_structures(&mut self, structures: StructureStore) -> Result<()> {
        if structures.len() != self.records.len() {
            return Err(crate::EnsembleError::Metadata(format!(
                "replacement ensemble has {} frame(s), expected {}",
                structures.len(),
                self.records.len()
            )));
        }
        self.structures = structures;
        Ok(())
    }

    /// Materialize every frame.  Intended for native callers with ample
    /// memory; browser workflows should stream through [`Self::structure`].
    pub fn into_frames(self) -> Result<Vec<SampledFrame>> {
        let Self {
            structures,
            records,
        } = self;
        records
            .into_iter()
            .enumerate()
            .map(|(index, record)| {
                Ok(SampledFrame {
                    structure: structures.get(index)?,
                    sites: record.sites,
                    proposal_index: record.proposal_index,
                    source: record.source,
                    log_native_probability: record.log_native_probability,
                    selected_energy_kcal_per_mol: record.selected_energy_kcal_per_mol,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glysys::{BuildOptions, read_pdb_str};
    use std::collections::BTreeSet;

    const PROTEIN: &str = include_str!("../../../tests/fixtures/protein.pdb");

    fn protein() -> Structure {
        let options = BuildOptions {
            add_water: false,
            add_ions: false,
            ..BuildOptions::default()
        };
        read_pdb_str(PROTEIN, &options).unwrap()
    }

    #[test]
    fn delta_frames_rebuild_identical_structures() {
        let template = protein();
        let mut moved = template.clone();
        let ids = moved
            .iter_atoms()
            .map(|atom| atom.id)
            .take(3)
            .collect::<Vec<_>>();
        for (offset, id) in ids.iter().enumerate() {
            let mut position = moved.atom_position(*id).unwrap();
            position.x += 0.125 * (offset + 1) as f64;
            moved.set_atom_position(*id, position).unwrap();
        }
        let mut store = StructureStore::with_capacity(2);
        store.push(template.clone());
        store.push(moved.clone());
        assert!(matches!(&store.entries[1], FrameGeometry::Delta(delta) if delta.len() == 3));
        assert_eq!(
            store.get(0).unwrap().to_pdb_string(),
            template.to_pdb_string()
        );
        assert_eq!(store.get(1).unwrap().to_pdb_string(), moved.to_pdb_string());
    }

    #[test]
    fn different_topology_is_kept_verbatim() {
        let template = protein();
        let mut trimmed = template.clone();
        let first = trimmed.residues().into_iter().next().unwrap().id;
        trimmed.remove_residues(&BTreeSet::from([first]));
        let mut store = StructureStore::with_capacity(2);
        store.push(template);
        store.push(trimmed.clone());
        assert!(matches!(store.entries[1], FrameGeometry::Full(_)));
        assert_eq!(
            store.get(1).unwrap().to_pdb_string(),
            trimmed.to_pdb_string()
        );
    }

    #[test]
    fn multi_model_pdb_matches_per_frame_output() {
        let template = protein();
        let mut store = StructureStore::with_capacity(2);
        store.push(template.clone());
        store.push(template.clone());
        let pdb = store.multi_model_pdb().unwrap();
        let model = template
            .to_pdb_string()
            .lines()
            .filter(|line| !line.starts_with("END"))
            .map(|line| format!("{line}\n"))
            .collect::<String>();
        assert_eq!(
            pdb,
            format!("MODEL        1\n{model}ENDMDL\nMODEL        2\n{model}ENDMDL\nEND\n")
        );
    }
}

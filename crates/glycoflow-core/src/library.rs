//! Residue library (`glycoflow/resources/residue_library.json`, `builder.ResidueLibrary`).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// One heavy-atom residue template. For non-root residues `anchor` is the atom bonded to the
/// parent oxygen, and `virtual_ox` / `virtual_cx` are where the parent's Ox and Cx sat.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResidueTemplate {
    pub atoms: Vec<String>,
    pub elements: Vec<String>,
    pub coords: Vec<[f64; 3]>,
    pub bonds: Vec<[usize; 2]>,
    #[serde(default)]
    pub anchor: Option<String>,
    #[serde(default)]
    pub virtual_ox: Option<[f64; 3]>,
    #[serde(default)]
    pub virtual_cx: Option<[f64; 3]>,
}

impl ResidueTemplate {
    /// Index of the first atom called `name` (Python `list.index`).
    pub fn atom_index(&self, name: &str) -> Option<usize> {
        self.atoms.iter().position(|a| a == name)
    }
}

/// A ring-pucker state of a residue code: endocyclic sign pattern, MD population, template.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PuckerState {
    pub sig: i64,
    pub pop: f64,
    pub template: ResidueTemplate,
}

/// Sequence-token -> residue-code maps and one 3D template per residue code.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResidueLibrary {
    /// "DGlcpNAc|b" -> "YB"
    pub token_codes: BTreeMap<String, String>,
    /// "3S" -> "SO3"
    pub mod_codes: BTreeMap<String, String>,
    /// residue code -> template
    pub templates: BTreeMap<String, ResidueTemplate>,
    /// "O3|O6" -> "V"
    #[serde(default)]
    pub prefix_codes: BTreeMap<String, String>,
    /// residue code -> pucker states (most populated first)
    #[serde(default)]
    pub pucker_states: BTreeMap<String, Vec<PuckerState>>,
}

impl ResidueLibrary {
    pub fn from_json_str(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }

    pub fn from_json_slice(b: &[u8]) -> Result<Self> {
        Ok(serde_json::from_slice(b)?)
    }

    /// Pucker states of a residue code (empty when the code has a single state).
    pub fn states(&self, code: &str) -> &[PuckerState] {
        self.pucker_states
            .get(code)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }
}

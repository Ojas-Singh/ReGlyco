//! GLYCAM sequence -> heavy-atom 3D template (`builder.ResidueLibrary.build`).
//!
//! Coordinates are assembled in f64 exactly as the numpy reference and returned in f64; the
//! Python builder casts them to f32 at the end ([`BuiltGlycan::coords_f32`]).

use std::collections::HashMap;
use std::sync::OnceLock;

use regex::Regex;

use crate::error::{Error, Result};
use crate::library::{PuckerState, ResidueLibrary, ResidueTemplate};
use crate::linalg::{cross, dot, kabsch, matvec, norm, sub, V3};
use crate::rng::SplitMix64;
use crate::sequence::{mod_pos, parse_glycam, Sequence};

/// GLYCAM residue prefix: which oxygens carry substituents (`builder.PREFIX`).
fn standard_prefix(key: &[String]) -> Option<&'static str> {
    let k: Vec<&str> = key.iter().map(String::as_str).collect();
    Some(match k.as_slice() {
        [] => "0",
        ["O1"] => "1",
        ["O2"] => "2",
        ["O3"] => "3",
        ["O4"] => "4",
        ["O5"] => "5",
        ["O6"] => "6",
        ["O7"] => "7",
        ["O8"] => "8",
        ["O9"] => "9",
        ["O2", "O3"] => "Z",
        ["O2", "O4"] => "Y",
        ["O2", "O6"] => "X",
        ["O3", "O4"] => "W",
        ["O3", "O6"] => "V",
        ["O4", "O6"] => "U",
        ["O2", "O3", "O4"] => "T",
        ["O2", "O3", "O6"] => "S",
        ["O2", "O4", "O6"] => "R",
        ["O3", "O4", "O6"] => "Q",
        ["O2", "O3", "O4", "O6"] => "P",
        _ => return None,
    })
}

/// Heavy-atom template of a glycan, in the atom order of the Python builder. Atom 0 is the
/// aglycone oxygen.
#[derive(Clone, Debug)]
pub struct BuiltGlycan {
    pub atom_names: Vec<String>,
    pub res_names: Vec<String>,
    /// 1-based residue numbers
    pub res_ids: Vec<i64>,
    pub elements: Vec<String>,
    pub coords: Vec<V3>,
    pub bonds: Vec<[usize; 2]>,
    /// residue path per atom ("agl", "r", "r/4", "r/4/m3" ...)
    pub res_paths: Vec<String>,
    /// per residue (index = res_id - 1): template code, and the pucker state used (None = majority template)
    pub residue_codes: Vec<String>,
    pub residue_states: Vec<Option<usize>>,
}

impl BuiltGlycan {
    pub fn n_atoms(&self) -> usize {
        self.atom_names.len()
    }
    pub fn n_residues(&self) -> usize {
        self.residue_codes.len()
    }
    /// Coordinates as f32 (`np.array(coords, dtype=np.float32)`).
    pub fn coords_f32(&self) -> Vec<[f32; 3]> {
        self.coords
            .iter()
            .map(|p| [p[0] as f32, p[1] as f32, p[2] as f32])
            .collect()
    }
}

/// Chooses a pucker state for a residue: (output residue index, residue code, its states) ->
/// state index, or None for the majority template. Only called for codes with pucker states.
pub type StateChooser<'a> = dyn FnMut(usize, &str, &[PuckerState]) -> Option<usize> + 'a;

fn skip_token_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^G[0-9]{5}[A-Z]{2}\|.$").unwrap())
}

struct Assembly<'l, 'c, 'd> {
    lib: &'l ResidueLibrary,
    seq: Sequence,
    chooser: &'c mut StateChooser<'d>,
    out: BuiltGlycan,
}

fn div(a: V3, s: f64) -> V3 {
    [a[0] / s, a[1] / s, a[2] / s]
}

/// `builder.place`: template coordinates with the anchor bonded to `ox` at the template's own bond
/// length and Cx-O-anchor angle, dihedral before-cx-ox-anchor = 180 deg.
fn place(tpl: &ResidueTemplate, cx: V3, ox: V3, before: V3) -> Result<Vec<V3>> {
    let missing = || Error::Library("template has no anchor / virtual parent atoms".to_string());
    let anchor_name = tpl.anchor.as_deref().ok_or_else(missing)?;
    let anchor = tpl.coords[tpl.atom_index(anchor_name).ok_or_else(missing)?];
    let v_ox = tpl.virtual_ox.ok_or_else(missing)?;
    let v_cx = tpl.virtual_cx.ok_or_else(missing)?;
    let bond = norm(sub(anchor, v_ox));
    let u = sub(v_cx, v_ox);
    let w = sub(anchor, v_ox);
    let angle = (dot(u, w) / norm(u) / norm(w)).clamp(-1.0, 1.0).acos();
    let bc = div(sub(ox, cx), norm(sub(ox, cx)));
    let n = cross(sub(cx, before), bc);
    let n = div(n, norm(n));
    let nb = cross(n, bc);
    let d = [-bond * angle.cos(), bond * angle.sin(), 0.0];
    let frame = [
        [bc[0], nb[0], n[0]],
        [bc[1], nb[1], n[1]],
        [bc[2], nb[2], n[2]],
    ];
    let off = matvec(&frame, d);
    let target = [ox[0] + off[0], ox[1] + off[1], ox[2] + off[2]];
    let (r, t) = kabsch(&[v_cx, v_ox, anchor], &[cx, ox, target]);
    Ok(tpl
        .coords
        .iter()
        .map(|p| {
            let q = matvec(&r, *p);
            [q[0] + t[0], q[1] + t[1], q[2] + t[2]]
        })
        .collect())
}

impl<'l, 'c, 'd> Assembly<'l, 'c, 'd> {
    fn resolve(&mut self, k: usize) -> Result<()> {
        let node = &self.seq.nodes[k];
        let key = format!("{}|{}", node.name, node.anomer);
        let Some(code) = self.lib.token_codes.get(&key) else {
            let mut known: Vec<&str> = self
                .lib
                .token_codes
                .keys()
                .filter(|t| !skip_token_re().is_match(t))
                .map(|t| t.split('|').next().unwrap())
                .collect();
            known.sort();
            known.dedup();
            return Err(Error::Library(format!(
                "residue {} ({}) is not in the training library; known residues: {}",
                node.name,
                if node.anomer == 'a' { "alpha" } else { "beta" },
                known.join(", ")
            )));
        };
        for m in &node.mods {
            if !self.lib.mod_codes.contains_key(m) {
                let known: Vec<&str> = self.lib.mod_codes.keys().map(String::as_str).collect();
                return Err(Error::Library(format!(
                    "modification [{m}] on {} is not in the training library (known: {})",
                    node.name,
                    known.join(", ")
                )));
            }
        }
        self.seq.nodes[k].code = code.clone();
        for c in self.seq.nodes[k].children.clone() {
            self.resolve(c)?;
        }
        Ok(())
    }

    fn prefix(&self, k: usize) -> String {
        let node = &self.seq.nodes[k];
        let mut pos: Vec<u32> = node
            .children
            .iter()
            .map(|&c| self.seq.nodes[c].ppos.unwrap())
            .collect();
        pos.extend(node.mods.iter().map(|m| mod_pos(m)));
        pos.sort_unstable();
        pos.dedup();
        let key: Vec<String> = pos.iter().map(|p| format!("O{p}")).collect();
        if let Some(p) = standard_prefix(&key) {
            return p.to_string();
        }
        self.lib
            .prefix_codes
            .get(&key.join("|"))
            .cloned()
            .unwrap_or_else(|| "?".to_string())
    }

    /// Template for `code`, residue `res_index` of the output (`build.template`).
    fn template(
        &mut self,
        code: &str,
        res_index: usize,
    ) -> Result<(&'l ResidueTemplate, Option<usize>)> {
        let lib: &'l ResidueLibrary = self.lib;
        let tpl = lib
            .templates
            .get(code)
            .ok_or_else(|| Error::Library(format!("no 3D template for residue code {code}")))?;
        let states = lib.states(code);
        if !states.is_empty() {
            if let Some(k) = (self.chooser)(res_index, code, states) {
                let st = states.get(k).ok_or_else(|| {
                    Error::Invalid(format!(
                        "pucker state {k} out of range for {code} ({} states)",
                        states.len()
                    ))
                })?;
                return Ok((&st.template, Some(k)));
            }
        }
        Ok((tpl, None))
    }

    fn add_residue(
        &mut self,
        resname: &str,
        code: &str,
        state: Option<usize>,
        tpl: &ResidueTemplate,
        xyz: &[V3],
        path: &str,
    ) -> HashMap<String, usize> {
        let o = &mut self.out;
        let start = o.atom_names.len();
        let rnum = o.res_ids.last().map(|r| r + 1).unwrap_or(1);
        for ((n, el), x) in tpl.atoms.iter().zip(&tpl.elements).zip(xyz) {
            o.res_paths.push(path.to_string());
            o.atom_names.push(n.clone());
            o.res_names.push(resname.to_string());
            o.res_ids.push(rnum);
            o.elements.push(el.clone());
            o.coords.push(*x);
        }
        o.bonds
            .extend(tpl.bonds.iter().map(|[a, b]| [start + a, start + b]));
        o.residue_codes.push(code.to_string());
        o.residue_states.push(state);
        let mut local = HashMap::new();
        for (k, n) in tpl.atoms.iter().enumerate() {
            local.insert(n.clone(), start + k);
        }
        local
    }

    fn grow(&mut self, k: usize, local: &HashMap<String, usize>, path: &str) -> Result<()> {
        enum Item {
            Child(usize),
            Mod(String),
        }
        let node = &self.seq.nodes[k];
        let mut attach: Vec<(u32, Item)> = node
            .children
            .iter()
            .map(|&c| (self.seq.nodes[c].ppos.unwrap(), Item::Child(c)))
            .collect();
        attach.extend(node.mods.iter().map(|m| (mod_pos(m), Item::Mod(m.clone()))));
        let node_name = node.name.clone();
        for (ppos, item) in attach {
            let o_name = format!("O{ppos}");
            let ox_i = *local.get(&o_name).ok_or_else(|| {
                Error::Library(format!(
                    "{node_name} has no {o_name} to attach a substituent"
                ))
            })?;
            let (cx_i, before) = {
                let o = &self.out;
                let same = |a: usize, b: usize| o.res_ids[a] == o.res_ids[b];
                let cx_i = o
                    .bonds
                    .iter()
                    .find(|[a, b]| (*a == ox_i || *b == ox_i) && same(*a, *b))
                    .map(|[a, b]| if *a == ox_i { *b } else { *a })
                    .ok_or_else(|| {
                        Error::Library(format!("{o_name} of {node_name} has no carbon"))
                    })?;
                let before = o
                    .bonds
                    .iter()
                    .find(|[a, b]| {
                        (*a == cx_i || *b == cx_i) && *a != ox_i && *b != ox_i && same(*a, *b)
                    })
                    .map(|[a, b]| if *a == cx_i { *b } else { *a })
                    .ok_or_else(|| {
                        Error::Library(format!(
                            "cannot orient a substituent on {o_name} of {node_name}"
                        ))
                    })?;
                (cx_i, before)
            };
            let (code, resname, sub_path, child) = match &item {
                Item::Child(c) => {
                    let code = self.seq.nodes[*c].code.clone();
                    (
                        code.clone(),
                        format!("{}{}", self.prefix(*c), code),
                        format!("{path}/{ppos}"),
                        Some(*c),
                    )
                }
                Item::Mod(m) => {
                    let resname = self.lib.mod_codes[m].clone();
                    (
                        resname[1..].to_string(),
                        resname,
                        format!("{path}/m{ppos}"),
                        None,
                    )
                }
            };
            let res_index = self.out.residue_codes.len();
            let (tpl, state) = self.template(&code, res_index)?;
            let c = &self.out.coords;
            let xyz = place(tpl, c[cx_i], c[ox_i], c[before])?;
            let ids = self.add_residue(&resname, &code, state, tpl, &xyz, &sub_path);
            let anchor = tpl
                .anchor
                .as_deref()
                .and_then(|a| ids.get(a).copied())
                .ok_or_else(|| Error::Library(format!("template {code} has no anchor")))?;
            self.out.bonds.push([ox_i, anchor]);
            if let Some(c) = child {
                self.grow(c, &ids, &sub_path)?;
            }
        }
        Ok(())
    }
}

impl ResidueLibrary {
    /// Template with the majority pucker for every ring (`build(seq, anomer)` without rng).
    pub fn build(&self, seq: &str, anomer: Option<&str>) -> Result<BuiltGlycan> {
        self.build_with(seq, anomer, &mut |_, _, _| None)
    }

    /// Template with explicit pucker states: `states[res_index]` (output residue order; index 0 is
    /// the aglycone) is a state index into the residue code's pucker states, or None for the
    /// majority template. Residues beyond `states.len()` use the majority template.
    pub fn build_with_states(
        &self,
        seq: &str,
        anomer: Option<&str>,
        states: &[Option<usize>],
    ) -> Result<BuiltGlycan> {
        self.build_with(seq, anomer, &mut |i, _, _| states.get(i).copied().flatten())
    }

    /// Template with ring puckers drawn from the MD populations (as `build(rng=...)`, with a Rust RNG).
    pub fn build_sampled(
        &self,
        seq: &str,
        anomer: Option<&str>,
        rng: &mut SplitMix64,
    ) -> Result<BuiltGlycan> {
        self.build_with(seq, anomer, &mut |_, _, states| {
            let pops: Vec<f64> = states.iter().map(|s| s.pop).collect();
            Some(rng.categorical(&pops))
        })
    }

    /// General build: `chooser` picks the pucker state of every residue whose code has states.
    /// It is called in the order of the Python builder's `template` calls (reducing residue, then
    /// residues in output order), i.e. the order in which Python draws from its rng.
    pub fn build_with(
        &self,
        seq: &str,
        anomer: Option<&str>,
        chooser: &mut StateChooser<'_>,
    ) -> Result<BuiltGlycan> {
        let mut seq = parse_glycam(seq)?;
        if let Some(a) = anomer.and_then(|a| a.chars().next()) {
            let root = seq.root;
            seq.nodes[root].anomer = a;
        }
        let empty = BuiltGlycan {
            atom_names: vec![],
            res_names: vec![],
            res_ids: vec![],
            elements: vec![],
            coords: vec![],
            bonds: vec![],
            res_paths: vec![],
            residue_codes: vec![],
            residue_states: vec![],
        };
        let mut asm = Assembly {
            lib: self,
            seq,
            chooser,
            out: empty,
        };
        let root = asm.seq.root;
        asm.resolve(root)?;
        let aglycone = asm.seq.aglycone.clone();
        if !self.templates.contains_key(&aglycone) {
            return Err(Error::Library(format!(
                "no template for aglycone {aglycone}"
            )));
        }
        let root_code = asm.seq.nodes[root].code.clone();
        let (red_tpl, red_state) = asm.template(&root_code, 1)?;
        let (agl_tpl, agl_state) = asm.template(&aglycone, 0)?;
        let o_k = agl_tpl
            .elements
            .iter()
            .position(|e| e == "O")
            .ok_or_else(|| Error::Library(format!("aglycone {aglycone} has no oxygen")))?;
        let v_ox = red_tpl.virtual_ox.ok_or_else(|| {
            Error::Library(format!("template {root_code} has no virtual parent oxygen"))
        })?;
        let o0 = agl_tpl.coords[o_k];
        let agl_xyz: Vec<V3> = agl_tpl
            .coords
            .iter()
            .map(|p| {
                [
                    (p[0] - o0[0]) + v_ox[0],
                    (p[1] - o0[1]) + v_ox[1],
                    (p[2] - o0[2]) + v_ox[2],
                ]
            })
            .collect();
        let agl = asm.add_residue(&aglycone, &aglycone, agl_state, agl_tpl, &agl_xyz, "agl");
        let red_name = format!("{}{}", asm.prefix(root), root_code);
        let red = asm.add_residue(
            &red_name,
            &root_code,
            red_state,
            red_tpl,
            &red_tpl.coords,
            "r",
        );
        let anchor = red_tpl
            .anchor
            .as_deref()
            .and_then(|a| red.get(a).copied())
            .ok_or_else(|| Error::Library(format!("template {root_code} has no anchor")))?;
        asm.out.bonds.push([agl[&agl_tpl.atoms[o_k]], anchor]);
        asm.grow(root, &red, "r")?;
        Ok(asm.out)
    }
}

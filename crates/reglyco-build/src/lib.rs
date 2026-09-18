//! Native, deterministic glycoprotein construction on top of GlySys.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use glysys::{
    AtomId, BuildOptions, GlycanTree, GlycosylationSite as SystemGlycosylationSite, ResidueId,
    Structure, SystemBuilder, Vec3, read_pdb, read_pdb_str,
};
pub use reglyco_core::{
    AttachmentRequest, BuildProduct, BuildRequest, BuildResult, GlycanConformer, GlycosylationSite,
    ReGlycoError, Result,
};

/// Remove the exact glycan tree attached at `site`, retaining the returned
/// residue identifiers for a hold-out comparison or audit trail.
pub fn remove_glycan_at_site(
    structure: &Structure,
    site: &ResidueId,
) -> std::result::Result<(Structure, Vec<ResidueId>), ReGlycoError> {
    let tree = structure
        .metadata()
        .glycan_trees
        .iter()
        .find(|tree| tree.attachment_site.as_ref() == Some(site))
        .ok_or_else(|| ReGlycoError::MissingInput(format!("no glycan attached at {site}")))?;
    let residues = tree.residue_ids.clone();
    if residues.is_empty() {
        return Err(ReGlycoError::MissingInput(format!(
            "glycan tree at {site} is empty"
        )));
    }
    let mut stripped = structure.clone();
    stripped.remove_residues(&residues.iter().cloned().collect());
    Ok((stripped, residues))
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProteinSource {
    Local(PathBuf),
    AlphaFold(String),
    Pdb(String),
    PdbAssembly { identifier: String, assembly: u32 },
}

#[derive(Debug, Clone)]
pub struct ProteinFetchOptions {
    pub cache_dir: PathBuf,
    pub offline: bool,
    pub alphafold_api_base: String,
    pub rcsb_files_base: String,
}

impl Default for ProteinFetchOptions {
    fn default() -> Self {
        Self {
            cache_dir: PathBuf::from(".reglyco-cache/proteins"),
            offline: false,
            alphafold_api_base: "https://alphafold.ebi.ac.uk".into(),
            rcsb_files_base: "https://files.rcsb.org".into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct FetchedProtein {
    pub structure: Structure,
    pub provenance: String,
    pub cache_path: Option<PathBuf>,
}

#[derive(Debug, thiserror::Error)]
pub enum ProteinFetchError {
    #[error(transparent)]
    GlySys(#[from] glysys::BuildError),
    #[error("invalid protein identifier {0:?}")]
    InvalidIdentifier(String),
    #[error("protein cache miss in offline mode: {0}")]
    OfflineCacheMiss(PathBuf),
    #[error("protein request failed for {url}: {message}")]
    Request { url: String, message: String },
    #[error("AlphaFold returned no prediction for UniProt accession {0}")]
    AlphaFoldNotFound(String),
    #[error("protein cache I/O failed for {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
}

#[cfg(not(target_arch = "wasm32"))]
pub struct ProteinProvider {
    options: ProteinFetchOptions,
    client: reqwest::blocking::Client,
}

#[cfg(not(target_arch = "wasm32"))]
impl ProteinProvider {
    pub fn new(options: ProteinFetchOptions) -> std::result::Result<Self, ProteinFetchError> {
        let client = reqwest::blocking::Client::builder()
            .user_agent("reglyco-rs/0.1")
            .build()
            .map_err(|error| ProteinFetchError::Request {
                url: "client initialization".into(),
                message: error.to_string(),
            })?;
        Ok(Self { options, client })
    }

    pub fn fetch(
        &self,
        source: &ProteinSource,
        build_options: &BuildOptions,
    ) -> std::result::Result<FetchedProtein, ProteinFetchError> {
        match source {
            ProteinSource::Local(path) => Ok(FetchedProtein {
                structure: read_pdb(path, build_options)?,
                provenance: path.display().to_string(),
                cache_path: None,
            }),
            ProteinSource::AlphaFold(accession) => self.fetch_alphafold(accession, build_options),
            ProteinSource::Pdb(identifier) => self.fetch_pdb(identifier, build_options),
            ProteinSource::PdbAssembly {
                identifier,
                assembly,
            } => self.fetch_pdb_assembly(identifier, *assembly, build_options),
        }
    }

    /// Fetch the PDBe EDS composite (2Fo-Fc) map associated with an entry and
    /// place it in the same provenance cache as the coordinate source.  The
    /// returned path is intentionally format-agnostic; `reglyco-density`
    /// performs the CCP4/MRC validation and coordinate interpretation.
    pub fn fetch_eds_map(
        &self,
        identifier: &str,
    ) -> std::result::Result<PathBuf, ProteinFetchError> {
        validate_identifier(identifier)?;
        let normalized = identifier.to_ascii_lowercase();
        let cache_path = self
            .options
            .cache_dir
            .join(format!("eds-{normalized}.ccp4"));
        if cache_path.is_file() {
            return Ok(cache_path);
        }
        if self.options.offline {
            return Err(ProteinFetchError::OfflineCacheMiss(cache_path));
        }
        let url = format!("https://www.ebi.ac.uk/pdbe/coordinates/files/{normalized}.ccp4");
        let bytes = self
            .client
            .get(&url)
            .send()
            .and_then(reqwest::blocking::Response::error_for_status)
            .and_then(|response| response.bytes())
            .map_err(|error| ProteinFetchError::Request {
                url: url.clone(),
                message: error.to_string(),
            })?;
        std::fs::create_dir_all(&self.options.cache_dir).map_err(|source| {
            ProteinFetchError::Io {
                path: self.options.cache_dir.clone(),
                source,
            }
        })?;
        std::fs::write(&cache_path, &bytes).map_err(|source| ProteinFetchError::Io {
            path: cache_path.clone(),
            source,
        })?;
        Ok(cache_path)
    }

    /// Fetch the PDBe EDS Fo-Fc difference map when available. Difference
    /// density is optional in refinement; callers should warn and continue
    /// with the composite map when this endpoint is unavailable.
    pub fn fetch_eds_difference_map(
        &self,
        identifier: &str,
    ) -> std::result::Result<PathBuf, ProteinFetchError> {
        validate_identifier(identifier)?;
        let normalized = identifier.to_ascii_lowercase();
        let cache_path = self
            .options
            .cache_dir
            .join(format!("eds-{normalized}-diff.ccp4"));
        if cache_path.is_file() {
            return Ok(cache_path);
        }
        if self.options.offline {
            return Err(ProteinFetchError::OfflineCacheMiss(cache_path));
        }
        let url = format!("https://www.ebi.ac.uk/pdbe/coordinates/files/{normalized}_diff.ccp4");
        let bytes = self
            .client
            .get(&url)
            .send()
            .and_then(reqwest::blocking::Response::error_for_status)
            .and_then(|response| response.bytes())
            .map_err(|error| ProteinFetchError::Request {
                url: url.clone(),
                message: error.to_string(),
            })?;
        std::fs::create_dir_all(&self.options.cache_dir).map_err(|source| {
            ProteinFetchError::Io {
                path: self.options.cache_dir.clone(),
                source,
            }
        })?;
        std::fs::write(&cache_path, &bytes).map_err(|source| ProteinFetchError::Io {
            path: cache_path.clone(),
            source,
        })?;
        Ok(cache_path)
    }

    fn fetch_alphafold(
        &self,
        accession: &str,
        build_options: &BuildOptions,
    ) -> std::result::Result<FetchedProtein, ProteinFetchError> {
        validate_identifier(accession)?;
        let normalized = accession.to_ascii_uppercase();
        let cache_path = self
            .options
            .cache_dir
            .join(format!("alphafold-{normalized}.pdb"));
        if cache_path.is_file() {
            return self.read_cache(&cache_path, build_options);
        }
        if self.options.offline {
            return Err(ProteinFetchError::OfflineCacheMiss(cache_path));
        }
        let metadata_url = format!(
            "{}/api/prediction/{}",
            self.options.alphafold_api_base.trim_end_matches('/'),
            normalized
        );
        let metadata = self.get_json(&metadata_url)?;
        let pdb_url = metadata
            .as_array()
            .and_then(|entries| entries.first())
            .and_then(|entry| entry.get("pdbUrl"))
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| ProteinFetchError::AlphaFoldNotFound(normalized.clone()))?;
        let pdb = self.get_text(pdb_url)?;
        self.cache_and_parse(&cache_path, &pdb, pdb_url, build_options)
    }

    fn fetch_pdb(
        &self,
        identifier: &str,
        build_options: &BuildOptions,
    ) -> std::result::Result<FetchedProtein, ProteinFetchError> {
        validate_identifier(identifier)?;
        let normalized = identifier.to_ascii_uppercase();
        let cache_path = self.options.cache_dir.join(format!("pdb-{normalized}.pdb"));
        if cache_path.is_file() {
            return self.read_cache(&cache_path, build_options);
        }
        if self.options.offline {
            return Err(ProteinFetchError::OfflineCacheMiss(cache_path));
        }
        let url = format!(
            "{}/download/{}.pdb",
            self.options.rcsb_files_base.trim_end_matches('/'),
            normalized
        );
        let pdb = self.get_text(&url)?;
        self.cache_and_parse(&cache_path, &pdb, &url, build_options)
    }

    fn fetch_pdb_assembly(
        &self,
        identifier: &str,
        assembly: u32,
        build_options: &BuildOptions,
    ) -> std::result::Result<FetchedProtein, ProteinFetchError> {
        validate_identifier(identifier)?;
        if assembly == 0 {
            return Err(ProteinFetchError::InvalidIdentifier(format!(
                "assembly {assembly}"
            )));
        }
        let normalized = identifier.to_ascii_uppercase();
        let cache_path = self
            .options
            .cache_dir
            .join(format!("pdb-{normalized}-assembly-{assembly}.pdb"));
        if cache_path.is_file() {
            return self.read_cache(&cache_path, build_options);
        }
        if self.options.offline {
            return Err(ProteinFetchError::OfflineCacheMiss(cache_path));
        }
        let url = format!(
            "{}/download/{}.pdb{}.gz",
            self.options.rcsb_files_base.trim_end_matches('/'),
            normalized,
            assembly
        );
        let bytes = self
            .client
            .get(&url)
            .send()
            .and_then(reqwest::blocking::Response::error_for_status)
            .and_then(|response| response.bytes())
            .map_err(|error| ProteinFetchError::Request {
                url: url.clone(),
                message: error.to_string(),
            })?;
        let mut decoder = flate2::read::GzDecoder::new(bytes.as_ref());
        let mut pdb = String::new();
        use std::io::Read;
        decoder
            .read_to_string(&mut pdb)
            .map_err(|error| ProteinFetchError::Request {
                url: url.clone(),
                message: error.to_string(),
            })?;
        let pdb = sanitize_connectivity(&pdb);
        self.cache_and_parse(&cache_path, &pdb, &url, build_options)
    }

    fn get_json(&self, url: &str) -> std::result::Result<serde_json::Value, ProteinFetchError> {
        self.client
            .get(url)
            .send()
            .and_then(reqwest::blocking::Response::error_for_status)
            .and_then(reqwest::blocking::Response::json)
            .map_err(|error| ProteinFetchError::Request {
                url: url.into(),
                message: error.to_string(),
            })
    }

    fn get_text(&self, url: &str) -> std::result::Result<String, ProteinFetchError> {
        self.client
            .get(url)
            .send()
            .and_then(reqwest::blocking::Response::error_for_status)
            .and_then(reqwest::blocking::Response::text)
            .map_err(|error| ProteinFetchError::Request {
                url: url.into(),
                message: error.to_string(),
            })
    }

    fn cache_and_parse(
        &self,
        cache_path: &Path,
        pdb: &str,
        provenance: &str,
        build_options: &BuildOptions,
    ) -> std::result::Result<FetchedProtein, ProteinFetchError> {
        let structure = read_pdb_str(pdb, build_options)?;
        std::fs::create_dir_all(&self.options.cache_dir).map_err(|source| {
            ProteinFetchError::Io {
                path: self.options.cache_dir.clone(),
                source,
            }
        })?;
        std::fs::write(cache_path, pdb).map_err(|source| ProteinFetchError::Io {
            path: cache_path.into(),
            source,
        })?;
        Ok(FetchedProtein {
            structure,
            provenance: provenance.into(),
            cache_path: Some(cache_path.into()),
        })
    }

    fn read_cache(
        &self,
        cache_path: &Path,
        build_options: &BuildOptions,
    ) -> std::result::Result<FetchedProtein, ProteinFetchError> {
        Ok(FetchedProtein {
            structure: read_pdb(cache_path, build_options)?,
            provenance: format!("cache:{}", cache_path.display()),
            cache_path: Some(cache_path.into()),
        })
    }
}

fn validate_identifier(identifier: &str) -> std::result::Result<(), ProteinFetchError> {
    if identifier.is_empty()
        || !identifier
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || character == '_')
    {
        return Err(ProteinFetchError::InvalidIdentifier(identifier.into()));
    }
    Ok(())
}

fn sanitize_connectivity(contents: &str) -> String {
    use std::collections::BTreeSet;

    let mut serials = BTreeSet::new();
    let mut residues = BTreeSet::new();
    for line in contents.lines() {
        if !(line.starts_with("ATOM  ") || line.starts_with("HETATM")) || line.len() < 27 {
            continue;
        }
        if let Ok(serial) = pdb_field(line, 6, 11).parse::<u32>() {
            serials.insert(serial);
        }
        if let Ok(number) = pdb_field(line, 22, 26).parse::<i32>() {
            residues.insert((
                pdb_field(line, 21, 22).to_string(),
                number,
                pdb_char(line, 26),
            ));
        }
    }
    let mut output = String::new();
    for line in contents.lines() {
        let keep = if line.starts_with("CONECT") {
            let values = line
                .as_bytes()
                .get(6..)
                .into_iter()
                .flat_map(|rest| rest.chunks(5))
                .filter_map(|chunk| std::str::from_utf8(chunk).ok()?.trim().parse::<u32>().ok())
                .collect::<Vec<_>>();
            !values.is_empty() && values.iter().all(|value| serials.contains(value))
        } else if line.starts_with("LINK  ") {
            link_residues(line)
                .is_some_and(|(left, right)| residues.contains(&left) && residues.contains(&right))
        } else if line.starts_with("SSBOND") {
            ssbond_residues(line)
                .is_some_and(|(left, right)| residues.contains(&left) && residues.contains(&right))
        } else {
            true
        };
        if keep {
            output.push_str(line);
            output.push('\n');
        }
    }
    output
}

fn pdb_field(line: &str, start: usize, end: usize) -> &str {
    line.get(start..end).unwrap_or("").trim()
}

fn pdb_char(line: &str, index: usize) -> Option<char> {
    line.as_bytes()
        .get(index)
        .copied()
        .map(char::from)
        .filter(|c| !c.is_ascii_whitespace())
}

type PdbResidueKey = (String, i32, Option<char>);

fn link_residues(line: &str) -> Option<(PdbResidueKey, PdbResidueKey)> {
    Some((
        (
            pdb_field(line, 21, 22).to_string(),
            pdb_field(line, 22, 26).parse().ok()?,
            pdb_char(line, 26),
        ),
        (
            pdb_field(line, 51, 52).to_string(),
            pdb_field(line, 52, 56).parse().ok()?,
            pdb_char(line, 56),
        ),
    ))
}

fn ssbond_residues(line: &str) -> Option<(PdbResidueKey, PdbResidueKey)> {
    Some((
        (
            pdb_field(line, 15, 16).to_string(),
            pdb_field(line, 17, 21).parse().ok()?,
            pdb_char(line, 21),
        ),
        (
            pdb_field(line, 29, 30).to_string(),
            pdb_field(line, 31, 35).parse().ok()?,
            pdb_char(line, 35),
        ),
    ))
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct NLinkedSequon {
    pub asparagine: ResidueId,
    pub middle: ResidueId,
    pub serine_or_threonine: ResidueId,
    pub motif: String,
    /// Four-residue sequence context when the preceding residue is adjacent
    /// in the same chain, for example `LNTT`.
    #[serde(default)]
    pub context: String,
}

pub fn scan_n_linked_sequons(structure: &Structure) -> Vec<NLinkedSequon> {
    let mut occupied = structure
        .metadata()
        .glycosylation_sites
        .iter()
        .map(|site| site.protein_residue.clone())
        .collect::<BTreeSet<_>>();
    for residue in structure
        .residues()
        .iter()
        .filter(|residue| residue.name == "ASN")
    {
        if structure
            .find_atom(&residue.id, "ND2")
            .is_some_and(|atom| has_cross_residue_bond(structure, atom, &residue.id))
        {
            occupied.insert(residue.id.clone());
        }
    }
    let residues = structure.residues();
    residues
        .windows(3)
        .enumerate()
        .filter(|(_, window)| {
            window[0].id.chain == window[1].id.chain
                && window[1].id.chain == window[2].id.chain
                && window[0].name == "ASN"
                && !window[1].name.eq_ignore_ascii_case("PRO")
                && matches!(window[2].name.as_str(), "SER" | "THR")
                && !occupied.contains(&window[0].id)
        })
        .map(|(index, window)| {
            let motif = format!(
                "N{}{}",
                residue_letter(&window[1].name),
                residue_letter(&window[2].name)
            );
            let context = index
                .checked_sub(1)
                .and_then(|previous| residues.get(previous))
                .filter(|previous| {
                    previous.id.chain == window[0].id.chain
                        && previous.id.number + 1 == window[0].id.number
                })
                .map(|previous| format!("{}{}", residue_letter(&previous.name), motif))
                .unwrap_or_else(|| motif.clone());
            NLinkedSequon {
                asparagine: window[0].id.clone(),
                middle: window[1].id.clone(),
                serine_or_threonine: window[2].id.clone(),
                motif,
                context,
            }
        })
        .collect()
}

fn residue_letter(name: &str) -> char {
    match name {
        "ALA" => 'A',
        "ARG" => 'R',
        "ASN" => 'N',
        "ASP" => 'D',
        "CYS" => 'C',
        "GLN" => 'Q',
        "GLU" => 'E',
        "GLY" => 'G',
        "HIS" | "HID" | "HIE" | "HIP" => 'H',
        "ILE" => 'I',
        "LEU" => 'L',
        "LYS" => 'K',
        "MET" => 'M',
        "PHE" => 'F',
        "PRO" => 'P',
        "SER" => 'S',
        "THR" => 'T',
        "TRP" => 'W',
        "TYR" => 'Y',
        "VAL" => 'V',
        _ => 'X',
    }
}

/// Attach every requested conformer in order and optionally parameterize the result.
pub fn build(request: BuildRequest, system_builder: &SystemBuilder) -> Result<BuildResult> {
    let angles = request
        .attachments
        .iter()
        .map(|attachment| {
            let residue = request
                .protein
                .residues()
                .into_iter()
                .find(|residue| residue.id == attachment.site.residue)
                .ok_or_else(|| ReGlycoError::SiteNotFound(attachment.site.residue.clone()))?;
            let definition =
                LinkageDefinition::for_residue(&attachment.site.residue, &residue.name)?;
            Ok((
                definition.default_phi_degrees,
                definition.default_psi_degrees,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    build_with_linkage_angles(request, &angles, system_builder)
}

/// Attach conformers using absolute linkage φ/ψ values in input order.
pub fn build_with_linkage_angles(
    request: BuildRequest,
    angles: &[(f64, f64)],
    system_builder: &SystemBuilder,
) -> Result<BuildResult> {
    if angles.len() != request.attachments.len() {
        return Err(ReGlycoError::InvalidGeometry);
    }
    let mut structure = request.protein;
    for attachment in &request.attachments {
        if structure.residues().into_iter().any(|residue| {
            residue.id == attachment.site.residue && residue.name.eq_ignore_ascii_case("PRO")
        }) {
            structure = hydroxylate_proline(
                &structure,
                &attachment.site.residue,
                system_builder.options(),
            )?;
        }
    }
    let mut requested_sites = BTreeSet::new();
    for (attachment, &(phi, psi)) in request.attachments.into_iter().zip(angles) {
        if !requested_sites.insert(attachment.site.residue.clone()) {
            return Err(ReGlycoError::OccupiedSite(attachment.site.residue));
        }
        attach(&mut structure, attachment, phi, psi)?;
    }
    let system = request
        .parameterize
        .then(|| system_builder.prepare_structure(&structure))
        .transpose()?;
    Ok(BuildProduct { structure, system })
}

/// Orient one glycan against a protein linkage frame without appending it.
///
/// This is the geometry-only seam used by high-throughput ensemble scoring.
/// It intentionally shares the same linkage rules and rotation sequence as
/// [`build_with_linkage_angles`], but avoids chain allocation, connectivity
/// updates, and metadata construction for rejected proposals.
pub fn orient_glycan_for_site(
    protein: &Structure,
    source_glycan: &Structure,
    site: &ResidueId,
    phi_degrees: f64,
    psi_degrees: f64,
) -> Result<Structure> {
    let residue = protein
        .residues()
        .into_iter()
        .find(|residue| residue.id == *site)
        .ok_or_else(|| ReGlycoError::SiteNotFound(site.clone()))?;
    let rule = LinkageDefinition::for_residue(site, &residue.name)?;
    let target_a = atom_position(protein, required_atom(protein, site, rule.frame_a)?)?;
    let target_b = atom_position(protein, required_atom(protein, site, rule.frame_b)?)?;
    let target_link = atom_position(protein, required_atom(protein, site, rule.link_atom)?)?;

    orient_glycan_for_rule(
        source_glycan,
        rule,
        target_a,
        target_b,
        target_link,
        phi_degrees,
        psi_degrees,
    )
}

/// Deterministically convert one PRO attachment site to HYP by adding OD1.
/// The placement matches the historical Cookbook construction: a 1.43 Å
/// CG–OD1 bond combining the CB/CG/CD in-plane bisector with the ring normal.
pub fn hydroxylate_proline(
    structure: &Structure,
    site: &ResidueId,
    options: &BuildOptions,
) -> Result<Structure> {
    let residue = structure
        .residues()
        .into_iter()
        .find(|residue| residue.id == *site)
        .ok_or_else(|| ReGlycoError::SiteNotFound(site.clone()))?;
    if residue.name.eq_ignore_ascii_case("HYP") {
        return Ok(structure.clone());
    }
    if !residue.name.eq_ignore_ascii_case("PRO") {
        return Err(ReGlycoError::UnsupportedSite {
            site: site.clone(),
            residue_name: residue.name,
        });
    }
    let position = |name: &str| -> Result<Vec3> {
        atom_position(structure, required_atom(structure, site, name)?)
    };
    let cb = position("CB")?;
    let cg = position("CG")?;
    let cd = position("CD")?;
    let cb_direction = normalize(subtract(cb, cg))?;
    let cd_direction = normalize(subtract(cd, cg))?;
    let normal = normalize(cross(cb_direction, cd_direction))?;
    let bisector = normalize(add(cb_direction, cd_direction))?;
    let radians = 120_f64.to_radians();
    let od1 = add(
        cg,
        scale(
            add(scale(bisector, radians.cos()), scale(normal, radians.sin())),
            1.43,
        ),
    );

    let pdb = structure.to_pdb_string();
    let next_serial = structure
        .atoms()
        .into_iter()
        .map(|atom| atom.id.0)
        .max()
        .unwrap_or(0)
        + 1;
    let insertion = site.insertion_code.unwrap_or(' ');
    let mut output = String::new();
    let mut inserted = false;
    let mut saw_target = false;
    for line in pdb.lines() {
        let is_atom = line.starts_with("ATOM  ") || line.starts_with("HETATM");
        let matching = is_atom
            && pdb_field(line, 21, 22) == site.chain
            && pdb_field(line, 22, 26).parse::<i32>().ok() == Some(site.number)
            && pdb_char(line, 26) == site.insertion_code;
        if matching {
            saw_target = true;
            let mut renamed = line.to_string();
            if renamed.len() >= 20 {
                renamed.replace_range(17..20, "HYP");
            }
            output.push_str(&renamed);
            output.push('\n');
            continue;
        }
        if saw_target && !inserted {
            output.push_str(&format!(
                "ATOM  {next_serial:>5}  OD1 HYP {:1}{:>4}{insertion:1}   {:>8.3}{:>8.3}{:>8.3}  1.00  0.00           O\n",
                site.chain, site.number, od1.x, od1.y, od1.z,
            ));
            inserted = true;
        }
        output.push_str(line);
        output.push('\n');
    }
    if !inserted {
        output.push_str(&format!(
            "ATOM  {next_serial:>5}  OD1 HYP {:1}{:>4}{insertion:1}   {:>8.3}{:>8.3}{:>8.3}  1.00  0.00           O\nEND\n",
            site.chain, site.number, od1.x, od1.y, od1.z,
        ));
    }
    Ok(read_pdb_str(&output, options)?)
}

/// Orient a glycan using a pre-resolved protein linkage frame.
///
/// This is intended for prepared search contexts. The caller supplies the
/// residue name and three frame coordinates, so repeated pose preparation does
/// not need to clone or scan the protein structure.
pub fn orient_glycan_for_frame(
    source_glycan: &Structure,
    site: &ResidueId,
    residue_name: &str,
    target_a: Vec3,
    target_b: Vec3,
    target_link: Vec3,
    phi_degrees: f64,
    psi_degrees: f64,
) -> Result<Structure> {
    let rule = LinkageDefinition::for_residue(site, residue_name)?;
    orient_glycan_for_rule(
        source_glycan,
        rule,
        target_a,
        target_b,
        target_link,
        phi_degrees,
        psi_degrees,
    )
}

/// Coordinate-only equivalent of [`orient_glycan_for_frame`].
///
/// The returned atom order is the source order with `ROH` atoms omitted, just
/// as the materialized attachment path does after removing the leaving group.
/// This is the hot-path API used while preparing many candidate conformers.
#[derive(Debug, Clone)]
pub struct OrientedGlycanCoordinates {
    pub atoms: Vec<(AtomId, Vec3)>,
    pub c1_index: usize,
    pub o5_index: usize,
}

pub fn orient_glycan_coordinates_for_frame(
    source_glycan: &Structure,
    site: &ResidueId,
    residue_name: &str,
    target_a: Vec3,
    target_b: Vec3,
    target_link: Vec3,
    phi_degrees: f64,
    psi_degrees: f64,
) -> Result<OrientedGlycanCoordinates> {
    let rule = LinkageDefinition::for_residue(site, residue_name)?;
    let root = glycan_root(source_glycan)?;
    // Resolve the actual component-specific anomeric atom.  Complete
    // sialic-acid records commonly contain an exocyclic C1 carboxyl carbon
    // as well as the glycosidic C2; using the residue-generic rule's C1 here
    // would orient the search pose against the wrong atom while the material
    // attachment path correctly records C2.
    let c1_name = glycan_attachment_atom_name(source_glycan, &root);
    let c1_id = required_glycan_atom(source_glycan, &root, c1_name)?;
    let o5_id = required_glycan_atom(source_glycan, &root, rule.glycan_o5)?;
    let source_atoms = source_glycan.atoms().into_iter().collect::<Vec<_>>();
    let mut coordinates = source_atoms
        .iter()
        .map(|atom| atom.position)
        .collect::<Vec<_>>();
    let index_of = |id: AtomId| {
        source_atoms
            .iter()
            .position(|atom| atom.id == id)
            .ok_or_else(|| ReGlycoError::MissingGlycanAtom(format!("atom {}", id.0)))
    };
    let c1_source_index = index_of(c1_id)?;
    let o5_source_index = index_of(o5_id)?;
    let leaving_index = source_atoms.iter().position(|atom| {
        atom.residue_name == rule.leaving_residue && atom.name == rule.leaving_atom
    });
    let source_anchor =
        leaving_index.map_or(coordinates[c1_source_index], |index| coordinates[index]);
    let target_anchor = if leaving_index.is_some() {
        target_link
    } else {
        add(
            target_link,
            scale(
                normalize(subtract(target_link, target_b))?,
                rule.bond_length,
            ),
        )
    };
    let translation = subtract(target_anchor, source_anchor);
    for position in &mut coordinates {
        *position = add(*position, translation);
    }
    let translated_c1 = coordinates[c1_source_index];
    let bond_direction = normalize(subtract(translated_c1, target_link))?;
    let bond_target = add(target_link, scale(bond_direction, rule.bond_length));
    let bond_correction = subtract(bond_target, translated_c1);
    for position in &mut coordinates {
        *position = add(*position, bond_correction);
    }

    let c1 = coordinates[c1_source_index];
    let angle_axis = cross(subtract(target_b, target_link), subtract(c1, target_link));
    let angle_delta =
        (rule.bond_angle_degrees - angle_degrees(target_b, target_link, c1)).to_radians();
    rotate_coordinates(&mut coordinates, target_link, angle_axis, angle_delta)?;
    let c1 = coordinates[c1_source_index];
    let psi_now = dihedral_degrees(target_a, target_b, target_link, c1);
    rotate_coordinates(
        &mut coordinates,
        target_link,
        subtract(target_link, target_b),
        (psi_degrees - psi_now).to_radians(),
    )?;
    let c1 = coordinates[c1_source_index];
    let o5 = coordinates[o5_source_index];
    let phi_now = dihedral_degrees(target_b, target_link, c1, o5);
    rotate_coordinates(
        &mut coordinates,
        target_link,
        subtract(c1, target_link),
        (phi_degrees - phi_now).to_radians(),
    )?;

    let mut atoms = Vec::with_capacity(source_atoms.len());
    let mut c1_index = None;
    let mut o5_index = None;
    for (index, atom) in source_atoms.iter().enumerate() {
        if atom.residue_name == "ROH" {
            continue;
        }
        let output_index = atoms.len();
        if index == c1_source_index {
            c1_index = Some(output_index);
        }
        if index == o5_source_index {
            o5_index = Some(output_index);
        }
        atoms.push((atom.id, coordinates[index]));
    }
    Ok(OrientedGlycanCoordinates {
        atoms,
        c1_index: c1_index.ok_or_else(|| ReGlycoError::MissingGlycanAtom("C1".into()))?,
        o5_index: o5_index.ok_or_else(|| ReGlycoError::MissingGlycanAtom("O5".into()))?,
    })
}

fn orient_glycan_for_rule(
    source_glycan: &Structure,
    rule: LinkageDefinition,
    target_a: Vec3,
    target_b: Vec3,
    target_link: Vec3,
    phi_degrees: f64,
    psi_degrees: f64,
) -> Result<Structure> {
    let mut glycan = source_glycan.clone();
    let root = glycan_root(&glycan)?;
    let c1_name = glycan_attachment_atom_name(&glycan, &root);
    let c1_id = required_glycan_atom(&glycan, &root, c1_name)?;
    let o5_id = required_glycan_atom(&glycan, &root, rule.glycan_o5)?;
    let leaving_group = glycan
        .atoms()
        .into_iter()
        .find(|atom| atom.residue_name == rule.leaving_residue && atom.name == rule.leaving_atom);
    let source_anchor = leaving_group
        .as_ref()
        .map_or(atom_position(&glycan, c1_id)?, |atom| atom.position);
    let target_anchor = if leaving_group.is_some() {
        target_link
    } else {
        add(
            target_link,
            scale(
                normalize(subtract(target_link, target_b))?,
                rule.bond_length,
            ),
        )
    };
    let translation = subtract(target_anchor, source_anchor);
    let translated = glycan
        .atoms()
        .into_iter()
        .map(|atom| (atom.id, add(atom.position, translation)))
        .collect::<Vec<_>>();
    glycan.set_atom_positions(translated)?;
    let translated_c1 = atom_position(&glycan, c1_id)?;
    let bond_direction = normalize(subtract(translated_c1, target_link))?;
    let bond_target = add(target_link, scale(bond_direction, rule.bond_length));
    let bond_correction = subtract(bond_target, translated_c1);
    let corrected = glycan
        .atoms()
        .into_iter()
        .map(|atom| (atom.id, add(atom.position, bond_correction)))
        .collect::<Vec<_>>();
    glycan.set_atom_positions(corrected)?;

    let c1 = atom_position(&glycan, c1_id)?;
    let angle_axis = cross(subtract(target_b, target_link), subtract(c1, target_link));
    let angle_delta =
        (rule.bond_angle_degrees - angle_degrees(target_b, target_link, c1)).to_radians();
    rotate_all(&mut glycan, target_link, angle_axis, angle_delta)?;
    let c1 = atom_position(&glycan, c1_id)?;
    let psi_now = dihedral_degrees(target_a, target_b, target_link, c1);
    rotate_all(
        &mut glycan,
        target_link,
        subtract(target_link, target_b),
        (psi_degrees - psi_now).to_radians(),
    )?;
    let c1 = atom_position(&glycan, c1_id)?;
    let o5 = atom_position(&glycan, o5_id)?;
    let phi_now = dihedral_degrees(target_b, target_link, c1, o5);
    rotate_all(
        &mut glycan,
        target_link,
        subtract(c1, target_link),
        (phi_degrees - phi_now).to_radians(),
    )?;
    glycan.remove_residues_named(&["ROH"]);
    Ok(glycan)
}

fn attach(
    protein: &mut Structure,
    attachment: AttachmentRequest,
    phi_degrees: f64,
    psi_degrees: f64,
) -> Result<()> {
    let site = attachment.site.residue;
    let residue = protein
        .residues()
        .into_iter()
        .find(|residue| residue.id == site)
        .ok_or_else(|| ReGlycoError::SiteNotFound(site.clone()))?;
    let rule = LinkageDefinition::for_residue(&site, &residue.name)?;
    let target_atom = required_atom(protein, &site, rule.link_atom)?;
    if protein
        .metadata()
        .glycosylation_sites
        .iter()
        .any(|existing| existing.protein_residue == site && existing.protein_atom == rule.link_atom)
        || has_cross_residue_bond(protein, target_atom, &site)
    {
        return Err(ReGlycoError::OccupiedSite(site));
    }
    let target_a = atom_position(protein, required_atom(protein, &site, rule.frame_a)?)?;
    let target_b = atom_position(protein, required_atom(protein, &site, rule.frame_b)?)?;
    let target_link = atom_position(protein, target_atom)?;

    let mut glycan = attachment.conformer.structure;
    let root = glycan_root(&glycan)?;
    let glycan_atom_name = glycan_attachment_atom_name(&glycan, &root);
    let c1_id = required_glycan_atom(&glycan, &root, glycan_atom_name)?;
    let o5_id = required_glycan_atom(&glycan, &root, rule.glycan_o5)?;
    let leaving_group = glycan
        .atoms()
        .into_iter()
        .find(|atom| atom.residue_name == rule.leaving_residue && atom.name == rule.leaving_atom);
    let source_anchor = leaving_group
        .as_ref()
        .map_or(atom_position(&glycan, c1_id)?, |atom| atom.position);
    let target_anchor = if leaving_group.is_some() {
        target_link
    } else {
        add(
            target_link,
            scale(
                normalize(subtract(target_link, target_b))?,
                rule.bond_length,
            ),
        )
    };
    let translation = subtract(target_anchor, source_anchor);
    let translated = glycan
        .atoms()
        .into_iter()
        .map(|atom| (atom.id, add(atom.position, translation)))
        .collect::<Vec<_>>();
    glycan.set_atom_positions(translated)?;
    // GLYCAM bundles normally encode the leaving-group C1-O1 distance at the
    // desired protein-glycan bond length. Enforce the named chemistry rule so
    // malformed or rounded local bundles cannot produce a stretched linkage.
    let translated_c1 = atom_position(&glycan, c1_id)?;
    let bond_direction = normalize(subtract(translated_c1, target_link))?;
    let bond_target = add(target_link, scale(bond_direction, rule.bond_length));
    let bond_correction = subtract(bond_target, translated_c1);
    let corrected = glycan
        .atoms()
        .into_iter()
        .map(|atom| (atom.id, add(atom.position, bond_correction)))
        .collect::<Vec<_>>();
    glycan.set_atom_positions(corrected)?;

    // Match compute: B-C-C1 = 123°, then set absolute ψ=A-B-C-C1
    // and φ=B-C-C1-O5 using whole-glycan rigid rotations.
    let c1 = atom_position(&glycan, c1_id)?;
    let angle_axis = cross(subtract(target_b, target_link), subtract(c1, target_link));
    let angle_delta =
        (rule.bond_angle_degrees - angle_degrees(target_b, target_link, c1)).to_radians();
    rotate_all(&mut glycan, target_link, angle_axis, angle_delta)?;

    let c1 = atom_position(&glycan, c1_id)?;
    let psi_now = dihedral_degrees(target_a, target_b, target_link, c1);
    rotate_all(
        &mut glycan,
        target_link,
        subtract(target_link, target_b),
        (psi_degrees - psi_now).to_radians(),
    )?;

    let c1 = atom_position(&glycan, c1_id)?;
    let o5 = atom_position(&glycan, o5_id)?;
    let phi_now = dihedral_degrees(target_b, target_link, c1, o5);
    rotate_all(
        &mut glycan,
        target_link,
        subtract(c1, target_link),
        (phi_degrees - phi_now).to_radians(),
    )?;
    glycan.remove_residues_named(&["ROH"]);

    let chain = next_chain(protein)?;
    let appended = protein.append(&glycan, &chain)?;
    let appended_root = appended
        .residue(&root)
        .cloned()
        .ok_or(ReGlycoError::AmbiguousGlycanRoot)?;
    // Most glycans use C1 as the anomeric atom. Keto-acids such as sialic
    // acid are the deliberate exception and use C2; keep the actual atom
    // name in metadata so validation/reporting can distinguish the two
    // chemistries instead of manufacturing a C1 attachment that is not in
    // the source component.
    // `glycan_atom_name` was resolved before orientation so the same
    // component-specific root is used for the transform and metadata.
    // Resolve the mapped source atom again after append. GlySys may renumber
    // atoms while merging chains; the append map preserves that mapping for
    // ordinary C1 roots as well as keto-acid C2 roots.  Use the actual
    // component-specific anomeric atom here: a complete sialic-acid asset
    // can contain an exocyclic C1 carboxyl carbon, but its glycosidic atom is
    // C2.  Falling back to C1 unconditionally silently created an impossible
    // C1 attachment and made the validator report a spurious chemistry error.
    let source_glycan_atom = required_glycan_atom(&glycan, &root, glycan_atom_name)?;
    let glycan_anomeric = appended
        .atom(source_glycan_atom)
        .ok_or_else(|| ReGlycoError::MissingGlycanAtom(glycan_atom_name.into()))?;
    protein.add_bond(target_atom, glycan_anomeric)?;

    let glycan_residues = glycan
        .residues()
        .iter()
        .filter_map(|residue| appended.residue(&residue.id).cloned())
        .collect::<Vec<_>>();
    protein.metadata_mut().glycan_trees.push(GlycanTree {
        chain,
        residue_ids: glycan_residues,
        attachment_site: Some(site.clone()),
    });
    protein.add_glycosylation_site(SystemGlycosylationSite {
        protein_residue: site,
        protein_atom: rule.link_atom.into(),
        glycan_residue: appended_root,
        glycan_atom: glycan_atom_name.into(),
    });
    Ok(())
}

fn has_cross_residue_bond(structure: &Structure, target: AtomId, residue: &ResidueId) -> bool {
    structure.bonds().into_iter().any(|(first, second)| {
        let other = if first == target {
            Some(second)
        } else if second == target {
            Some(first)
        } else {
            None
        };
        other
            .and_then(|atom| structure.atom_residue(atom))
            .is_some_and(|atom_residue| atom_residue != *residue)
    })
}

/// Measure attachment torsions from the actual output geometry.
pub fn attachment_angles(structure: &Structure, site: &ResidueId) -> Result<(f64, f64)> {
    let attachment = structure
        .metadata()
        .glycosylation_sites
        .iter()
        .find(|a| &a.protein_residue == site)
        .ok_or_else(|| ReGlycoError::MissingInput(format!("no attachment at {site}")))?;
    let residue = structure
        .residues()
        .into_iter()
        .find(|r| &r.id == site)
        .ok_or_else(|| ReGlycoError::MissingInput(format!("no residue at {site}")))?;
    let rule = LinkageDefinition::for_residue(site, &residue.name)?;
    let a = atom_position(structure, required_atom(structure, site, rule.frame_a)?)?;
    let b = atom_position(structure, required_atom(structure, site, rule.frame_b)?)?;
    let link = atom_position(
        structure,
        required_atom(structure, site, &attachment.protein_atom)?,
    )?;
    let root = atom_position(
        structure,
        required_glycan_atom(
            structure,
            &attachment.glycan_residue,
            &attachment.glycan_atom,
        )?,
    )?;
    let ring = atom_position(
        structure,
        required_glycan_atom(structure, &attachment.glycan_residue, rule.glycan_o5)?,
    )?;
    let angles = (
        dihedral_degrees(b, link, root, ring),
        dihedral_degrees(a, b, link, root),
    );
    if !angles.0.is_finite() || !angles.1.is_finite() {
        return Err(ReGlycoError::InvalidGeometry);
    }
    Ok(angles)
}

fn required_atom(structure: &Structure, site: &ResidueId, name: &str) -> Result<AtomId> {
    structure
        .find_atom(site, name)
        .ok_or_else(|| ReGlycoError::MissingSiteAtom {
            site: site.clone(),
            atom: name.into(),
        })
}

fn required_glycan_atom(structure: &Structure, residue: &ResidueId, name: &str) -> Result<AtomId> {
    if let Some(atom) = structure.find_atom(residue, name) {
        return Ok(atom);
    }
    // Sialic acids/KDO are keto sugars whose anomeric carbon is C2 and whose
    // ring oxygen is commonly O6.  LinkageDefinition intentionally remains
    // residue-centric for CLI compatibility; resolve this chemically
    // specific fallback at the component boundary instead of making every
    // caller know about the exception.
    let residue_name = structure
        .residues()
        .into_iter()
        .find(|candidate| candidate.id == *residue)
        .map(|candidate| candidate.name.to_ascii_uppercase())
        .unwrap_or_default();
    if name.eq_ignore_ascii_case("C1")
        && matches!(
            residue_name.as_str(),
            "SIA" | "NEU" | "NAN" | "NGC" | "KDN" | "KDO"
        )
    {
        if let Some(atom) = structure.find_atom(residue, "C2") {
            return Ok(atom);
        }
    }
    if name.eq_ignore_ascii_case("O5") {
        let fallback = if matches!(
            residue_name.as_str(),
            "SIA" | "NEU" | "NAN" | "NGC" | "KDN" | "KDO"
        ) {
            // Keto-sugar exports use both conventions: some retain the
            // pyranose ring oxygen as O5, while others expose the ring
            // anchor as O6.  Prefer the chemically explicit O6 form but
            // accept O5 when that is the only ring oxygen present.  The
            // caller still records the actual atom in the attachment and
            // report metadata, so no artificial O5/O6 atom is introduced.
            if structure.find_atom(residue, "O6").is_some() {
                "O6"
            } else {
                "O5"
            }
        } else if matches!(residue_name.as_str(), "ARA" | "ARB" | "AFL" | "RIB") {
            "O4"
        } else {
            ""
        };
        if !fallback.is_empty() {
            if let Some(atom) = structure.find_atom(residue, fallback) {
                return Ok(atom);
            }
        }
    }
    Err(ReGlycoError::MissingGlycanAtom(name.into()))
}

fn glycan_attachment_atom_name(structure: &Structure, residue: &ResidueId) -> &'static str {
    let residue_name = structure
        .residues()
        .into_iter()
        .find(|candidate| candidate.id == *residue)
        .map(|candidate| candidate.name.trim().to_ascii_uppercase())
        .unwrap_or_default();
    // Sialic acids/KDO are keto sugars: C2 is the anomeric carbon even when
    // the exocyclic carboxyl carbon C1 is present.  Prioritising C1 here
    // silently produced an impossible C1 attachment for complete SIA assets.
    if matches!(
        residue_name.as_str(),
        "SIA" | "NEU" | "NAN" | "NGC" | "KDN" | "KDO"
    ) && structure.find_atom(residue, "C2").is_some()
    {
        // The only supported C2 anomeric root in the component dictionary is
        // the keto-acid/sialic family.  `required_glycan_atom` still performs
        // the final component-specific existence check.
        "C2"
    } else if structure.find_atom(residue, "C1").is_some() {
        "C1"
    } else {
        "C1"
    }
}

fn atom_position(structure: &Structure, atom: AtomId) -> Result<Vec3> {
    structure
        .atom_position(atom)
        .ok_or_else(|| ReGlycoError::MissingGlycanAtom(format!("atom {}", atom.0)))
}

fn glycan_root(glycan: &Structure) -> Result<ResidueId> {
    // Keto sugars such as sialic acid may retain an exocyclic C1 carboxyl
    // group as well as their actual anomeric C2.  Record those candidates for
    // the standalone-keto fallback below, while allowing branched glycans
    // with several sialic residues to resolve their reducing root normally.
    let sialic_candidates = glycan
        .residues()
        .into_iter()
        .filter(|residue| {
            matches!(
                residue.name.trim().to_ascii_uppercase().as_str(),
                "SIA" | "NEU" | "NAN" | "NGC" | "KDN" | "KDO"
            ) && glycan.find_atom(&residue.id, "C2").is_some()
                && ["O6", "O5", "O4"]
                    .into_iter()
                    .any(|name| glycan.find_atom(&residue.id, name).is_some())
        })
        .map(|residue| residue.id)
        .collect::<Vec<_>>();
    // A branched N-glycan commonly contains two or more sialic-acid residues.
    // They are valid keto-sugar components, but they are not competing
    // reducing roots.  Only use the sialic candidate directly when the asset
    // has no ordinary C1-root candidates at all; otherwise continue below and
    // resolve the reducing end from its ROH cap.  The previous unconditional
    // `len > 1` error rejected otherwise valid multi-sialylated GlycoShape
    // ensembles before the search could even start.
    let c1_roots = glycan
        .residues()
        .into_iter()
        .filter(|residue| glycan.find_atom(&residue.id, "C1").is_some())
        .map(|residue| residue.id)
        .collect::<Vec<_>>();
    let candidates = glycan
        .residues()
        .into_iter()
        .filter(|residue| {
            glycan.find_atom(&residue.id, "C1").is_some()
                // Pyranoses use O5 as the ring oxygen; furanoses such as
                // arabinose use O4. Both are valid root anchors, with the
                // linkage rule selecting the appropriate one later.
                && (glycan.find_atom(&residue.id, "O5").is_some()
                    || glycan.find_atom(&residue.id, "O4").is_some())
        })
        .map(|residue| residue.id)
        .collect::<Vec<_>>();
    if candidates.len() == 1 {
        return Ok(candidates[0].clone());
    }
    // Keep a unique C1 root resolvable even when its ring anchor is missing;
    // the linkage-specific lookup can then report the precise missing O4/O5
    // atom instead of the less actionable ambiguous-root error.
    if candidates.is_empty() && c1_roots.len() == 1 {
        return Ok(c1_roots[0].clone());
    }
    if candidates.is_empty() && sialic_candidates.len() == 1 {
        return Ok(sialic_candidates[0].clone());
    }
    let cap_oxygens = glycan
        .atoms()
        .into_iter()
        .filter(|atom| atom.residue_name == "ROH" && atom.name == "O1")
        .collect::<Vec<_>>();
    let mut ranked = candidates
        .into_iter()
        .filter_map(|residue| {
            let c1 = glycan
                .find_atom(&residue, "C1")
                .and_then(|atom| glycan.atom(atom))?;
            let distance = cap_oxygens
                .iter()
                .map(|oxygen| squared_distance(c1.position, oxygen.position))
                .reduce(f64::min)?;
            Some((distance, residue))
        })
        .filter(|(distance, _)| *distance <= 2.0f64.powi(2))
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| left.0.total_cmp(&right.0));
    match ranked.as_slice() {
        [(distance, residue), ..]
            if ranked
                .get(1)
                .is_none_or(|second| (second.0 - distance).abs() > 1.0e-6) =>
        {
            Ok(residue.clone())
        }
        _ => Err(ReGlycoError::AmbiguousGlycanRoot),
    }
}

fn next_chain(structure: &Structure) -> Result<String> {
    let used = structure
        .residues()
        .into_iter()
        .map(|residue| residue.id.chain)
        .collect::<BTreeSet<_>>();
    ('A'..='Z')
        .map(|chain| chain.to_string())
        .find(|chain| !used.contains(chain))
        .ok_or(ReGlycoError::NoAvailableChain)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinkageDefinition {
    pub frame_a: &'static str,
    pub frame_b: &'static str,
    pub link_atom: &'static str,
    pub glycan_c1: &'static str,
    pub glycan_o5: &'static str,
    pub leaving_residue: &'static str,
    pub leaving_atom: &'static str,
    pub bond_length: f64,
    pub bond_angle_degrees: f64,
    pub default_phi_degrees: f64,
    pub default_psi_degrees: f64,
}

impl LinkageDefinition {
    pub fn for_residue(site: &ResidueId, name: &str) -> Result<Self> {
        // Coordinate importers do not guarantee residue-name casing or
        // surrounding whitespace.  Normalize once at this boundary so the
        // same linkage rules are used by native, WASM, validation, and
        // prepared-search paths instead of rejecting an otherwise valid
        // lower-case PRO/SER/ASN site.
        let name = name.trim().to_ascii_uppercase();
        match name.as_str() {
            "ASN" => Ok(Self {
                frame_a: "CB",
                frame_b: "CG",
                link_atom: "ND2",
                glycan_c1: "C1",
                glycan_o5: "O5",
                leaving_residue: "ROH",
                leaving_atom: "O1",
                bond_length: 1.45,
                bond_angle_degrees: 123.0,
                default_phi_degrees: -91.0,
                default_psi_degrees: 178.5,
            }),
            "SER" => Ok(Self {
                frame_a: "CA",
                frame_b: "CB",
                link_atom: "OG",
                glycan_c1: "C1",
                glycan_o5: "O5",
                leaving_residue: "ROH",
                leaving_atom: "O1",
                bond_length: 1.43,
                bond_angle_degrees: 123.0,
                default_phi_degrees: -70.0,
                default_psi_degrees: -170.0,
            }),
            "THR" => Ok(Self {
                frame_a: "CA",
                frame_b: "CB",
                link_atom: "OG1",
                glycan_c1: "C1",
                glycan_o5: "O5",
                leaving_residue: "ROH",
                leaving_atom: "O1",
                bond_length: 1.43,
                bond_angle_degrees: 123.0,
                default_phi_degrees: 182.0,
                default_psi_degrees: 182.0,
            }),
            "TRP" => Ok(Self {
                frame_a: "CB",
                frame_b: "CG",
                link_atom: "CD1",
                glycan_c1: "C1",
                glycan_o5: "O5",
                leaving_residue: "ROH",
                leaving_atom: "O1",
                bond_length: 1.54,
                bond_angle_degrees: 123.0,
                default_phi_degrees: 120.0,
                default_psi_degrees: 0.0,
            }),
            "HYP" | "PRO" => Ok(Self {
                frame_a: "CB",
                frame_b: "CG",
                link_atom: "OD1",
                glycan_c1: "C1",
                glycan_o5: "O4",
                leaving_residue: "ROH",
                leaving_atom: "O1",
                bond_length: 1.43,
                bond_angle_degrees: 123.0,
                default_phi_degrees: -105.0,
                default_psi_degrees: 97.5,
            }),
            other => Err(ReGlycoError::UnsupportedSite {
                site: site.clone(),
                residue_name: other.into(),
            }),
        }
    }
}

fn normalize(vector: Vec3) -> Result<Vec3> {
    let length = dot(vector, vector).sqrt();
    if length <= 1.0e-8 {
        Err(ReGlycoError::InvalidGeometry)
    } else {
        Ok(scale(vector, 1.0 / length))
    }
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

fn squared_distance(first: Vec3, second: Vec3) -> f64 {
    (first.x - second.x).powi(2) + (first.y - second.y).powi(2) + (first.z - second.z).powi(2)
}

fn cross(first: Vec3, second: Vec3) -> Vec3 {
    Vec3 {
        x: first.y * second.z - first.z * second.y,
        y: first.z * second.x - first.x * second.z,
        z: first.x * second.y - first.y * second.x,
    }
}

fn angle_degrees(first: Vec3, center: Vec3, third: Vec3) -> f64 {
    let left = subtract(first, center);
    let right = subtract(third, center);
    (dot(left, right) / (dot(left, left) * dot(right, right)).sqrt().max(1.0e-30))
        .clamp(-1.0, 1.0)
        .acos()
        .to_degrees()
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

fn rotate_all(structure: &mut Structure, origin: Vec3, axis: Vec3, radians: f64) -> Result<()> {
    let axis = normalize(axis)?;
    let cosine = radians.cos();
    let sine = radians.sin();
    let updates = structure
        .atoms()
        .into_iter()
        .map(|atom| {
            let relative = subtract(atom.position, origin);
            let rotated = add(
                add(scale(relative, cosine), scale(cross(axis, relative), sine)),
                scale(axis, dot(axis, relative) * (1.0 - cosine)),
            );
            (atom.id, add(origin, rotated))
        })
        .collect::<Vec<_>>();
    Ok(structure.set_atom_positions(updates)?)
}

fn rotate_coordinates(
    coordinates: &mut [Vec3],
    origin: Vec3,
    axis: Vec3,
    radians: f64,
) -> Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::TcpListener;

    const PROTEIN: &str = include_str!("../../../tests/fixtures/protein.pdb");
    const GLYCAN: &str = include_str!("../../../tests/fixtures/glycan.pdb");

    fn dry_options() -> BuildOptions {
        BuildOptions {
            add_water: false,
            add_ions: false,
            ..BuildOptions::default()
        }
    }

    #[test]
    fn supports_trp_c_mannosylation_linkage() {
        let site = ResidueId {
            chain: "B".into(),
            number: 42,
            insertion_code: Some('A'),
        };
        let rule = LinkageDefinition::for_residue(&site, "TRP").unwrap();
        assert_eq!(rule.link_atom, "CD1");
        assert_eq!(rule.glycan_c1, "C1");
        assert!((rule.bond_length - 1.54).abs() < 1.0e-12);
    }

    #[test]
    fn explicitly_converts_proline_to_hydroxyproline_in_memory() {
        let pdb = "\
ATOM      1  N   PRO A   7       0.000   0.000   0.000  1.00 20.00           N
ATOM      2  CA  PRO A   7       1.450   0.000   0.000  1.00 20.00           C
ATOM      3  CB  PRO A   7       1.900   1.400   0.000  1.00 20.00           C
ATOM      4  CG  PRO A   7       0.800   2.200   0.500  1.00 20.00           C
ATOM      5  CD  PRO A   7      -0.200   1.100   0.200  1.00 20.00           C
END
";
        let options = dry_options();
        let structure = read_pdb_str(pdb, &options).unwrap();
        let site = ResidueId {
            chain: "A".into(),
            number: 7,
            insertion_code: None,
        };
        let converted = hydroxylate_proline(&structure, &site, &options).unwrap();
        let residue = converted
            .residues()
            .into_iter()
            .find(|residue| residue.id == site)
            .unwrap();
        assert_eq!(residue.name, "HYP");
        assert!(converted.find_atom(&site, "OD1").is_some());
    }

    #[test]
    fn resolves_sialic_acid_c2_and_o6_attachment_anchors() {
        // Sialic acids are keto sugars: the anomeric carbon is C2 and the
        // conserved ring oxygen is commonly deposited as O6.  Keep this
        // component-specific resolution at the glycan boundary so the
        // residue-centric linkage API remains backwards compatible.
        let pdb = "\\
HETATM    1  C2  SIA B   1       0.000   0.000   0.000  1.00 20.00           C
HETATM    2  C3  SIA B   1       1.400   0.000   0.000  1.00 20.00           C
HETATM    3  C4  SIA B   1       2.100   1.200   0.000  1.00 20.00           C
HETATM    4  C5  SIA B   1       1.300   2.300   0.000  1.00 20.00           C
HETATM    5  C6  SIA B   1       0.000   2.100   0.000  1.00 20.00           C
HETATM    6  O6  SIA B   1      -0.600   1.100   0.000  1.00 20.00           O
END
";
        let glycan = read_pdb_str(pdb, &dry_options()).unwrap();
        let root = glycan_root(&glycan).unwrap();
        assert_eq!(root.chain, "B");
        let c2 = glycan.find_atom(&root, "C2").unwrap();
        let o6 = glycan.find_atom(&root, "O6").unwrap();
        assert_eq!(required_glycan_atom(&glycan, &root, "C1").unwrap(), c2);
        assert_eq!(required_glycan_atom(&glycan, &root, "O5").unwrap(), o6);
        assert_eq!(glycan_attachment_atom_name(&glycan, &root), "C2");
    }

    #[test]
    fn complete_sialic_acid_prefers_c2_even_when_carboxyl_c1_and_o5_are_present() {
        let glycan = read_pdb_str(
            "HETATM    1  C1  SIA B   1      -1.200   0.000   0.000  1.00 20.00           C\nHETATM    2  C2  SIA B   1       0.000   0.000   0.000  1.00 20.00           C\nHETATM    3  C3  SIA B   1       1.300   0.000   0.000  1.00 20.00           C\nHETATM    4  C4  SIA B   1       1.900   1.100   0.000  1.00 20.00           C\nHETATM    5  C5  SIA B   1       1.100   2.000   0.000  1.00 20.00           C\nHETATM    6  C6  SIA B   1      -0.100   1.600   0.000  1.00 20.00           C\nHETATM    7  O5  SIA B   1      -0.700   0.600   0.000  1.00 20.00           O\nEND\n",
            &BuildOptions::default(),
        )
        .unwrap();
        let root = glycan_root(&glycan).unwrap();
        assert_eq!(root.number, 1);
        assert_eq!(glycan_attachment_atom_name(&glycan, &root), "C2");
        assert!(required_glycan_atom(&glycan, &root, "C1").is_ok());
        assert!(required_glycan_atom(&glycan, &root, "O5").is_ok());
    }

    #[test]
    fn finds_only_strict_unoccupied_n_linked_sequons() {
        let pdb = "\
ATOM      1  CA  LEU A   9       0.000   0.000   0.000  1.00 20.00           C
ATOM      2  CA  ASN A  10       1.000   0.000   0.000  1.00 20.00           C
ATOM      3  CA  ALA A  11       2.000   0.000   0.000  1.00 20.00           C
ATOM      4  CA  THR A  12       3.000   0.000   0.000  1.00 20.00           C
ATOM      5  CA  ASN A  13       4.000   0.000   0.000  1.00 20.00           C
ATOM      6  CA  PRO A  14       5.000   0.000   0.000  1.00 20.00           C
ATOM      7  CA  SER A  15       6.000   0.000   0.000  1.00 20.00           C
END
";
        let structure = read_pdb_str(pdb, &dry_options()).unwrap();
        let sites = scan_n_linked_sequons(&structure);
        assert_eq!(sites.len(), 1);
        assert_eq!(sites[0].asparagine.number, 10);
        assert_eq!(sites[0].motif, "NAT");
        assert_eq!(sites[0].context, "LNAT");
    }

    #[test]
    fn attachment_matches_compute_linkage_geometry() {
        let options = dry_options();
        let protein = read_pdb_str(PROTEIN, &options).unwrap();
        let glycan = read_pdb_str(GLYCAN, &options).unwrap();
        let builder = SystemBuilder::new(options).unwrap();
        let product = build_with_linkage_angles(
            BuildRequest {
                protein,
                attachments: vec![AttachmentRequest {
                    site: GlycosylationSite::new("A", 1),
                    conformer: GlycanConformer::new(glycan),
                }],
                parameterize: false,
            },
            &[(-91.0, 178.5)],
            &builder,
        )
        .unwrap();
        let attachment = &product.structure.metadata().glycosylation_sites[0];
        let position = |residue: &ResidueId, atom: &str| {
            product
                .structure
                .find_atom(residue, atom)
                .and_then(|atom| product.structure.atom(atom))
                .unwrap()
                .position
        };
        let a = position(&attachment.protein_residue, "CB");
        let b = position(&attachment.protein_residue, "CG");
        let c = position(&attachment.protein_residue, "ND2");
        let d = position(&attachment.glycan_residue, "C1");
        let e = position(&attachment.glycan_residue, "O5");
        assert!((squared_distance(c, d).sqrt() - 1.45).abs() < 1.0e-8);
        assert!((angle_degrees(b, c, d) - 123.0).abs() < 1.0e-8);
        assert!((dihedral_degrees(a, b, c, d) - 178.5).abs() < 1.0e-8);
        assert!((dihedral_degrees(b, c, d, e) - -91.0).abs() < 1.0e-8);
        assert!(
            product
                .structure
                .residues()
                .iter()
                .all(|residue| residue.name != "ROH")
        );
    }

    #[test]
    fn alphafold_provider_resolves_versioned_url_and_reuses_cache() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            for body in [
                format!(r#"[{{"pdbUrl":"http://{address}/model.pdb"}}]"#),
                PROTEIN.into(),
            ] {
                let (mut stream, _) = listener.accept().unwrap();
                let mut request = [0_u8; 2048];
                let _ = stream.read(&mut request).unwrap();
                let response = format!(
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                stream.write_all(response.as_bytes()).unwrap();
            }
        });
        let cache = tempfile::tempdir().unwrap();
        let provider = ProteinProvider::new(ProteinFetchOptions {
            cache_dir: cache.path().into(),
            alphafold_api_base: format!("http://{address}"),
            ..ProteinFetchOptions::default()
        })
        .unwrap();
        let first = provider
            .fetch(&ProteinSource::AlphaFold("O15552".into()), &dry_options())
            .unwrap();
        server.join().unwrap();
        assert!(first.cache_path.unwrap().is_file());

        let offline = ProteinProvider::new(ProteinFetchOptions {
            cache_dir: cache.path().into(),
            offline: true,
            alphafold_api_base: "http://127.0.0.1:9".into(),
            ..ProteinFetchOptions::default()
        })
        .unwrap()
        .fetch(&ProteinSource::AlphaFold("O15552".into()), &dry_options())
        .unwrap();
        assert!(offline.provenance.starts_with("cache:"));
    }

    #[test]
    fn pdb_provider_downloads_legacy_pdb_coordinates() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut request = [0_u8; 2048];
            let length = stream.read(&mut request).unwrap();
            assert!(String::from_utf8_lossy(&request[..length]).contains("/download/4HHB.pdb"));
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                PROTEIN.len(),
                PROTEIN
            );
            stream.write_all(response.as_bytes()).unwrap();
        });
        let cache = tempfile::tempdir().unwrap();
        let fetched = ProteinProvider::new(ProteinFetchOptions {
            cache_dir: cache.path().into(),
            rcsb_files_base: format!("http://{address}"),
            ..ProteinFetchOptions::default()
        })
        .unwrap()
        .fetch(&ProteinSource::Pdb("4hhb".into()), &dry_options())
        .unwrap();
        server.join().unwrap();
        assert_eq!(fetched.structure.residues().len(), 2);
        assert!(fetched.cache_path.unwrap().ends_with("pdb-4HHB.pdb"));
    }

    #[test]
    fn assembly_connectivity_sanitizer_drops_stale_endpoints() {
        let contents = "ATOM      1  CA  ASN A   1       0.000   0.000   0.000  1.00 20.00           C\nLINK         CA  ASN A   1                 CA  ASN A 999  1.50  1.50\nCONECT    1  999\nEND\n";
        let cleaned = sanitize_connectivity(contents);
        assert!(cleaned.contains("ATOM      1"));
        assert!(!cleaned.contains("LINK"));
        assert!(!cleaned.contains("CONECT"));
    }

    #[test]
    fn removes_only_the_exact_attached_tree() {
        let options = dry_options();
        let protein = read_pdb_str(PROTEIN, &options).unwrap();
        let glycan = read_pdb_str(GLYCAN, &options).unwrap();
        let builder = SystemBuilder::new(options).unwrap();
        let built = build(
            BuildRequest {
                protein,
                attachments: vec![AttachmentRequest {
                    site: GlycosylationSite::new("A", 1),
                    conformer: GlycanConformer::new(glycan),
                }],
                parameterize: false,
            },
            &builder,
        )
        .unwrap();
        let (stripped, removed) = remove_glycan_at_site(
            &built.structure,
            &ResidueId {
                chain: "A".into(),
                number: 1,
                insertion_code: None,
            },
        )
        .unwrap();
        assert!(!removed.is_empty());
        assert!(stripped.metadata().glycan_trees.is_empty());
        assert!(stripped.metadata().glycosylation_sites.is_empty());
        assert_eq!(stripped.residues().len(), 2);
    }
}

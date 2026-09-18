//! RCSB VolumeServer acquisition and BinaryCIF-to-CCP4 conversion.
//!
//! VolumeServer deliberately returns a BinaryCIF document rather than an
//! MRC file.  The small adapter here validates the crystallographic metadata,
//! decodes the selected 2Fo-Fc channel, and writes a normal floating-point
//! CCP4 file which is then read through the same `DensityMap` path as user
//! supplied maps.  No crystallographic interpretation is delegated to the
//! web service or to an external executable.

use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use open_bcif::encoding::{EncodedData, Encoding, decoders};
use open_bcif::streaming::parser::StreamingParser;
use reqwest::blocking::Client;
use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{DensityError, Result};

const USER_AGENT: &str = "ReGlyco/0.1 (RCSB VolumeServer density adapter)";

#[derive(Debug, Clone, Serialize)]
pub struct RcsbMapAcquisition {
    pub pdb_id: String,
    pub channel: String,
    pub detail: u8,
    pub source_url: String,
    pub raw_cache_path: PathBuf,
    pub converted_map_path: PathBuf,
    pub raw_sha256: String,
    pub map_sha256: String,
    pub sample_count: [usize; 3],
    pub axis_order: [usize; 3],
    pub origin_fractional: [f64; 3],
    pub dimensions_fractional: [f64; 3],
    pub cell_lengths_angstrom: [f64; 3],
    pub cell_angles_degrees: [f64; 3],
    pub space_group: i32,
}

#[derive(Debug, Clone)]
struct RcsbVolume {
    channel: String,
    axis_order: [usize; 3],
    origin_fractional: [f64; 3],
    dimensions_fractional: [f64; 3],
    sample_rate: usize,
    sample_count: [usize; 3],
    space_group: i32,
    cell_lengths_angstrom: [f64; 3],
    cell_angles_degrees: [f64; 3],
    values: Vec<f32>,
}

/// Fetch the RCSB 2Fo-Fc volume for a PDB entry.  The entire periodic cell is
/// requested so an atom crossing a cell boundary can be scored with the same
/// map semantics as the viewer.  Existing cache entries are verified and
/// reused without a network request.
pub fn fetch_2fo_fc_map(
    pdb_id: &str,
    cache_dir: impl AsRef<Path>,
    detail: u8,
) -> Result<RcsbMapAcquisition> {
    let pdb_id = pdb_id.trim().to_ascii_lowercase();
    if pdb_id.len() != 4 || !pdb_id.as_bytes().iter().all(u8::is_ascii_alphanumeric) {
        return Err(DensityError::Map(format!(
            "invalid RCSB PDB identifier `{pdb_id}`"
        )));
    }
    if detail > 6 {
        return Err(DensityError::Map(
            "RCSB map detail must be between 0 and 6".into(),
        ));
    }
    let cache_dir = cache_dir.as_ref();
    fs::create_dir_all(cache_dir).map_err(|source| DensityError::Io {
        path: cache_dir.to_path_buf(),
        source,
    })?;
    let stem = format!("rcsb-{pdb_id}-2fo-fc-detail{detail}");
    let raw_cache_path = cache_dir.join(format!("{stem}.bcif"));
    let converted_map_path = cache_dir.join(format!("{stem}.ccp4"));
    let source_url =
        format!("https://maps.rcsb.org/x-ray/{pdb_id}/cell/?encoding=bcif&detail={detail}");

    let raw = if raw_cache_path.is_file() {
        fs::read(&raw_cache_path).map_err(|source| DensityError::Io {
            path: raw_cache_path.clone(),
            source,
        })?
    } else {
        let client = Client::builder()
            .user_agent(USER_AGENT)
            .build()
            .map_err(|error| DensityError::Map(format!("RCSB client setup failed: {error}")))?;
        let response = client
            .get(&source_url)
            .send()
            .map_err(|error| DensityError::Map(format!("RCSB map download failed: {error}")))?;
        if !response.status().is_success() {
            return Err(DensityError::Map(format!(
                "RCSB VolumeServer returned HTTP {} for {}",
                response.status(),
                source_url
            )));
        }
        let bytes = response
            .bytes()
            .map_err(|error| DensityError::Map(format!("RCSB map response failed: {error}")))?
            .to_vec();
        if bytes.is_empty() {
            return Err(DensityError::Map(
                "RCSB returned an empty BinaryCIF map".into(),
            ));
        }
        fs::write(&raw_cache_path, &bytes).map_err(|source| DensityError::Io {
            path: raw_cache_path.clone(),
            source,
        })?;
        bytes
    };
    let raw_sha256 = sha256(&raw);
    let volume = decode_volume(&raw)?;
    if volume.channel != "2fo-fc" {
        return Err(DensityError::Map(format!(
            "RCSB response did not contain the requested 2Fo-Fc channel (got {})",
            volume.channel
        )));
    }
    if !converted_map_path.is_file() {
        write_ccp4(&converted_map_path, &volume)?;
    }
    let map_sha256 = sha256_file(&converted_map_path)?;
    Ok(RcsbMapAcquisition {
        pdb_id,
        channel: "2Fo-Fc".into(),
        detail,
        source_url,
        raw_cache_path,
        converted_map_path,
        raw_sha256,
        map_sha256,
        sample_count: volume.sample_count,
        axis_order: volume.axis_order,
        origin_fractional: volume.origin_fractional,
        dimensions_fractional: volume.dimensions_fractional,
        cell_lengths_angstrom: volume.cell_lengths_angstrom,
        cell_angles_degrees: volume.cell_angles_degrees,
        space_group: volume.space_group,
    })
}

fn decode_volume(bytes: &[u8]) -> Result<RcsbVolume> {
    let mut parser = StreamingParser::new(BufReader::new(bytes));
    let (_, _, block_count) = parser
        .parse_file_metadata()
        .map_err(|error| DensityError::Map(format!("invalid BinaryCIF metadata: {error}")))?;
    if block_count == 0 {
        return Err(DensityError::Map(
            "BinaryCIF contains no data blocks".into(),
        ));
    }
    let mut selected = None;
    for _ in 0..block_count {
        let block = parser
            .next_data_block()
            .map_err(|error| DensityError::Map(format!("invalid BinaryCIF data block: {error}")))?;
        if block.header.eq_ignore_ascii_case("2FO-FC")
            || block.header.eq_ignore_ascii_case("2FO_FC")
            || block.header.eq_ignore_ascii_case("2FOFC")
        {
            selected = Some(block);
            break;
        }
    }
    let block = selected
        .ok_or_else(|| DensityError::Map("RCSB BinaryCIF has no 2Fo-Fc data block".into()))?;
    let info = block
        .categories
        .iter()
        .find(|category| {
            category
                .name
                .trim_start_matches('_')
                .eq_ignore_ascii_case("volume_data_3d_info")
        })
        .ok_or_else(|| DensityError::Map("RCSB BinaryCIF has no volume metadata".into()))?;
    let values_category = block
        .categories
        .iter()
        .find(|category| {
            category
                .name
                .trim_start_matches('_')
                .eq_ignore_ascii_case("volume_data_3d")
        })
        .ok_or_else(|| DensityError::Map("RCSB BinaryCIF has no volume values".into()))?;
    let get = |name: &str| -> Result<Vec<f64>> {
        let column = info
            .columns
            .iter()
            .find(|column| column.name.trim_matches('.').eq_ignore_ascii_case(name))
            .ok_or_else(|| {
                DensityError::Map(format!("RCSB volume metadata is missing `{name}`"))
            })?;
        decode_numeric(&column.data)
    };
    let get_triplet = |name: &str| -> Result<Vec<f64>> {
        let mut values = Vec::with_capacity(3);
        for index in 0..3 {
            let column_name = format!("{name}[{index}]");
            let column = info
                .columns
                .iter()
                .find(|column| {
                    column
                        .name
                        .trim_matches('.')
                        .eq_ignore_ascii_case(&column_name)
                })
                .ok_or_else(|| {
                    DensityError::Map(format!("RCSB volume metadata is missing `{column_name}`"))
                })?;
            values.push(
                decode_numeric(&column.data)?
                    .into_iter()
                    .next()
                    .ok_or_else(|| {
                        DensityError::Map(format!("RCSB metadata `{column_name}` is empty"))
                    })?,
            );
        }
        Ok(values)
    };
    let axis_order = get_triplet("axis_order")?;
    let origin = get_triplet("origin")?;
    let dimensions = get_triplet("dimensions")?;
    let sample_count = get_triplet("sample_count")?;
    let cell_lengths = get_triplet("spacegroup_cell_size")?;
    let cell_angles = get_triplet("spacegroup_cell_angles")?;
    let scalar = |name: &str| -> Result<f64> {
        Ok(get(name)?
            .into_iter()
            .next()
            .ok_or_else(|| DensityError::Map(format!("RCSB metadata `{name}` is empty")))?)
    };
    let axis_order = to_triplet_usize(&axis_order, "axis_order")?;
    if axis_order.iter().any(|axis| *axis > 2) || unique_count(axis_order) != 3 {
        return Err(DensityError::Geometry(
            "RCSB axis_order must be a permutation of 0,1,2".into(),
        ));
    }
    let origin_fractional = to_triplet(&origin, "origin")?;
    let dimensions_fractional = to_triplet(&dimensions, "dimensions")?;
    if dimensions_fractional
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(DensityError::Geometry(
            "RCSB volume dimensions must be positive and finite".into(),
        ));
    }
    let sample_count = to_triplet_usize(&sample_count, "sample_count")?;
    if sample_count.iter().any(|value| *value == 0) {
        return Err(DensityError::Geometry(
            "RCSB sample_count must be positive".into(),
        ));
    }
    let expected = sample_count
        .iter()
        .try_fold(1usize, |acc, value| acc.checked_mul(*value))
        .ok_or_else(|| DensityError::Geometry("RCSB voxel count overflows usize".into()))?;
    if values_category.row_count as usize != expected {
        return Err(DensityError::Map(format!(
            "RCSB voxel count mismatch: metadata {expected}, values {}",
            values_category.row_count
        )));
    }
    let values_column = values_category
        .columns
        .iter()
        .find(|column| column.name.trim_matches('.').eq_ignore_ascii_case("values"))
        .ok_or_else(|| DensityError::Map("RCSB volume values column is missing".into()))?;
    let values = decode_numeric(&values_column.data)?
        .into_iter()
        .map(|value| value as f32)
        .collect::<Vec<_>>();
    if values.len() != expected || values.iter().any(|value| !value.is_finite()) {
        return Err(DensityError::Map(
            "RCSB volume contains a non-finite or incomplete voxel array".into(),
        ));
    }
    let cell_lengths_angstrom = to_triplet(&cell_lengths, "spacegroup_cell_size")?;
    let cell_angles_degrees = to_triplet(&cell_angles, "spacegroup_cell_angles")?;
    let space_group = scalar("spacegroup_number")? as i32;
    let sample_rate = scalar("sample_rate")? as usize;
    if sample_rate == 0 {
        return Err(DensityError::Geometry(
            "RCSB sample_rate must be positive".into(),
        ));
    }
    Ok(RcsbVolume {
        channel: block.header.to_ascii_lowercase(),
        axis_order,
        origin_fractional,
        dimensions_fractional,
        sample_rate,
        sample_count,
        space_group,
        cell_lengths_angstrom,
        cell_angles_degrees,
        values,
    })
}

fn decode_numeric(data: &EncodedData) -> Result<Vec<f64>> {
    if data.encoding.is_empty() {
        return Err(DensityError::Map("BinaryCIF column has no encoding".into()));
    }
    let raw = data.data.as_ref();
    let mut decoded: Option<Vec<f64>> = None;
    for encoding in data.encoding.iter().rev() {
        match encoding {
            Encoding::ByteArray { data_type } => {
                decoded = Some(
                    decoders::decode_byte_array(raw, *data_type).map_err(|error| {
                        DensityError::Map(format!("BinaryCIF byte array: {error}"))
                    })?,
                );
            }
            Encoding::IntegerPacking {
                byte_count,
                is_unsigned,
                src_size,
            } => {
                decoded = Some(
                    decoders::decode_integer_packing(raw, *byte_count, *is_unsigned, *src_size)
                        .map_err(|error| {
                            DensityError::Map(format!("BinaryCIF integer packing: {error}"))
                        })?,
                );
            }
            Encoding::Delta { origin, .. } => {
                decoded = Some(decoders::decode_delta(
                    decoded.take().ok_or_else(|| missing_encoding("Delta"))?,
                    *origin,
                ));
            }
            Encoding::RunLength { src_size, .. } => {
                decoded = Some(decoders::decode_run_length(
                    decoded
                        .take()
                        .ok_or_else(|| missing_encoding("RunLength"))?,
                    *src_size,
                ));
            }
            Encoding::FixedPoint { factor, .. } => {
                decoded = Some(decoders::decode_fixed_point(
                    decoded
                        .take()
                        .ok_or_else(|| missing_encoding("FixedPoint"))?,
                    *factor,
                ));
            }
            Encoding::IntervalQuantization {
                min,
                max,
                num_steps,
                ..
            } => {
                decoded = Some(decoders::decode_interval_quantization(
                    decoded
                        .take()
                        .ok_or_else(|| missing_encoding("IntervalQuantization"))?,
                    *min,
                    *max,
                    *num_steps,
                ));
            }
            Encoding::StringArray { .. } => {
                return Err(DensityError::Map(
                    "BinaryCIF string encoding is not valid for a numeric volume column".into(),
                ));
            }
        }
    }
    decoded.ok_or_else(|| DensityError::Map("BinaryCIF column did not decode".into()))
}

fn missing_encoding(name: &str) -> DensityError {
    DensityError::Map(format!("BinaryCIF {name} encoding has no source values"))
}

fn to_triplet(values: &[f64], name: &str) -> Result<[f64; 3]> {
    if values.len() < 3 || values[..3].iter().any(|value| !value.is_finite()) {
        return Err(DensityError::Map(format!(
            "RCSB `{name}` is not a finite triplet"
        )));
    }
    Ok([values[0], values[1], values[2]])
}

fn to_triplet_usize(values: &[f64], name: &str) -> Result<[usize; 3]> {
    let triplet = to_triplet(values, name)?;
    if triplet
        .iter()
        .any(|value| *value < 0.0 || value.fract() != 0.0)
    {
        return Err(DensityError::Map(format!(
            "RCSB `{name}` is not an integer triplet"
        )));
    }
    Ok([
        triplet[0] as usize,
        triplet[1] as usize,
        triplet[2] as usize,
    ])
}

/// Small deterministic uniqueness helper that avoids pulling a set into the
/// hot BinaryCIF decode path.
fn unique_count(values: [usize; 3]) -> usize {
    let mut values = values.to_vec();
    values.sort_unstable();
    values.dedup();
    values.len()
}

fn write_ccp4(path: &Path, volume: &RcsbVolume) -> Result<()> {
    let mut sampling = [0usize; 3];
    for array_axis in 0..3 {
        let cart_axis = volume.axis_order[array_axis];
        let value = volume.sample_count[array_axis] as f64 * volume.sample_rate as f64
            / volume.dimensions_fractional[cart_axis];
        let rounded = value.round();
        if !rounded.is_finite() || rounded < 1.0 || (value - rounded).abs() > 1.0e-3 {
            return Err(DensityError::Geometry(format!(
                "RCSB grid does not map to integral cell sampling ({value})"
            )));
        }
        sampling[cart_axis] = rounded as usize;
    }
    let mut nstart_cart = [0isize; 3];
    for axis in 0..3 {
        let value = volume.origin_fractional[axis] * sampling[axis] as f64;
        let rounded = value.round();
        if !rounded.is_finite() || (value - rounded).abs() > 1.0e-3 {
            return Err(DensityError::Geometry(format!(
                "RCSB origin does not map to an integral grid start ({value})"
            )));
        }
        nstart_cart[axis] = rounded as isize;
    }
    let mut nstart_array = [0i32; 3];
    for array_axis in 0..3 {
        nstart_array[array_axis] = nstart_cart[volume.axis_order[array_axis]] as i32;
    }
    let shape = volume.sample_count;
    let mut writer = mrc::create(path)
        .shape(shape)
        .sampling([sampling[0] as i32, sampling[1] as i32, sampling[2] as i32])
        .cell_lengths(
            volume.cell_lengths_angstrom[0] as f32,
            volume.cell_lengths_angstrom[1] as f32,
            volume.cell_lengths_angstrom[2] as f32,
        )
        .cell_angles(
            volume.cell_angles_degrees[0] as f32,
            volume.cell_angles_degrees[1] as f32,
            volume.cell_angles_degrees[2] as f32,
        )
        .nstart(nstart_array)
        .axis_mapping([
            volume.axis_order[0] as i32 + 1,
            volume.axis_order[1] as i32 + 1,
            volume.axis_order[2] as i32 + 1,
        ])
        .origin([0.0, 0.0, 0.0])
        .ispg(volume.space_group)
        .mode::<f32>()
        .add_label("ReGlyco RCSB VolumeServer 2Fo-Fc conversion")
        .finish()
        .map_err(|error| DensityError::Map(error.to_string()))?;
    writer
        .write_block_as(
            &mrc::VoxelBlock::new([0, 0, 0], shape, volume.values.clone())
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
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String> {
    let bytes = fs::read(path).map_err(|source| DensityError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(sha256(&bytes))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_volume_server_fixture_when_available() {
        let path = Path::new("/tmp/rcsb-box.v2BzkC.bcif");
        if !path.is_file() {
            return;
        }
        let bytes = fs::read(path).unwrap();
        let volume = decode_volume(&bytes).unwrap();
        assert_eq!(volume.channel, "2fo-fc");
        assert_eq!(
            volume.values.len(),
            volume.sample_count.iter().copied().product::<usize>()
        );
        assert!(volume.values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn converts_volume_server_fixture_to_valid_ccp4_when_available() {
        let path = Path::new("/tmp/rcsb-box.v2BzkC.bcif");
        if !path.is_file() {
            return;
        }
        let bytes = fs::read(path).unwrap();
        let volume = decode_volume(&bytes).unwrap();
        let directory = tempfile::tempdir().unwrap();
        let output = directory.path().join("volume.ccp4");
        write_ccp4(&output, &volume).unwrap();
        let map = crate::DensityMap::open(&output).unwrap();
        assert_eq!(map.metadata().nx, volume.sample_count[0]);
        assert_eq!(map.metadata().map_axes, [2, 1, 3]);
    }
}

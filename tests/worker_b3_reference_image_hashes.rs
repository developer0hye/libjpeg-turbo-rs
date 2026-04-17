//! B3-3: Hash-stability regression for every `*.jpg` under
//! `references/libjpeg-turbo/testimages/`.
//!
//! For each JPEG in that directory we:
//!   1. Decode with our Rust decoder (`decompress_to` for 8-bit precision,
//!      `precision::decompress_12bit` for 12-bit precision).
//!   2. Hash the decoded pixel buffer with `DefaultHasher` (SipHash-1-3) —
//!      the same algorithm the pre-existing `reference_hashes.json` uses.
//!   3. Compare against the expected hash in
//!      `tests/reference_hashes_conformance.json`.
//!
//! The JSON file is committed with `null` entries on first landing. Run
//! `UPDATE_HASHES=1 cargo test --test worker_b3_reference_image_hashes --
//! --nocapture` to record the measured hashes, then commit the updated
//! JSON. Subsequent runs assert exact equality so that any decoder change
//! that perturbs these fixtures is caught as a regression.
//!
//! Namespaced per coordinator guardrails: worker-b3 owns
//! `tests/reference_hashes*.json` and the `worker_b3_` test prefix. The
//! existing reference hash files (bitstream_regression / bitstream_stability)
//! are left untouched.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use libjpeg_turbo_rs::{decompress_to, PixelFormat};

const CONFORMANCE_JSON: &str = include_str!("reference_hashes_conformance.json");
const CONFORMANCE_JSON_REL_PATH: &str = "tests/reference_hashes_conformance.json";
const TESTIMAGES_DIR: &str = "references/libjpeg-turbo/testimages";

fn hash_bytes(data: &[u8]) -> String {
    let mut hasher: DefaultHasher = DefaultHasher::new();
    data.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Parse the reference_hashes_conformance.json file.
///
/// The JSON is flat `{ "key": "value" | null, ... }` plus an arbitrary
/// number of underscore-prefixed metadata keys. We use a minimal line-
/// oriented parser (same pattern as `bitstream_regression.rs`) so that no
/// JSON dependency is introduced.
fn load_expected_hashes() -> HashMap<String, Option<String>> {
    let mut out: HashMap<String, Option<String>> = HashMap::new();
    for line in CONFORMANCE_JSON.lines() {
        let trimmed: &str = line.trim();
        if !trimmed.starts_with('"') || trimmed.starts_with("\"_") {
            continue;
        }
        let parts: Vec<&str> = trimmed.splitn(2, ':').collect();
        if parts.len() != 2 {
            continue;
        }
        let key: String = parts[0].trim().trim_matches('"').to_string();
        let value_str: &str = parts[1].trim().trim_end_matches(',');
        let value: Option<String> = if value_str == "null" {
            None
        } else {
            Some(value_str.trim_matches('"').to_string())
        };
        out.insert(key, value);
    }
    out
}

/// Convert an `Image12` sample buffer to a byte slice for hashing. We
/// serialize as little-endian i16 so the hash is platform-independent.
fn bytes_from_i16_samples(samples: &[i16]) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// List every `*.jpg`/`*.jpeg` file in the testimages directory. Sorted for
/// deterministic iteration order (filesystems do not guarantee order).
fn list_jpegs(dir: &Path) -> Vec<PathBuf> {
    let read: std::fs::ReadDir = match std::fs::read_dir(dir) {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    let mut out: Vec<PathBuf> = Vec::new();
    for entry in read.flatten() {
        let p: PathBuf = entry.path();
        if !p.is_file() {
            continue;
        }
        let ext: String = p
            .extension()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        if ext == "jpg" || ext == "jpeg" {
            out.push(p);
        }
    }
    out.sort();
    out
}

/// Pick the hash key and compute the decoded-pixel hash for a single JPEG.
///
/// Returns `Ok(Some((key, hash)))` on success, `Ok(None)` when the file is
/// a format we deliberately skip (currently: 16-bit, lossless-floating),
/// or `Err` on a genuine decoder failure — which we propagate so the test
/// fails loudly.
fn hash_one(jpeg_path: &Path) -> Result<Option<(String, String)>, String> {
    let name: String = jpeg_path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| format!("non-utf8 filename: {:?}", jpeg_path))?
        .to_string();
    let data: Vec<u8> =
        std::fs::read(jpeg_path).map_err(|e| format!("failed to read {:?}: {:?}", jpeg_path, e))?;

    // Detect precision cheaply via SOFn marker scan. A JPEG file starts
    // with FF D8 and the SOF marker is "FF C0..CF" (excluding C4 / C8 / CC
    // which are DHT/JPG/DAC). Precision is the first byte after the 2-byte
    // length field.
    let precision: u8 = detect_precision(&data).unwrap_or(8);

    match precision {
        8 => {
            let img = decompress_to(&data, PixelFormat::Rgb)
                .map_err(|e| format!("{}: Rust decompress_to Rgb failed: {}", name, e))?;
            assert_eq!(
                img.data.len(),
                img.width * img.height * 3,
                "{}: Rgb buffer size mismatch",
                name
            );
            Ok(Some((format!("{}_rgb8", name), hash_bytes(&img.data))))
        }
        12 => {
            use libjpeg_turbo_rs::precision::decompress_12bit;
            let img = decompress_12bit(&data)
                .map_err(|e| format!("{}: Rust decompress_12bit failed: {}", name, e))?;
            let key: String = if img.num_components == 1 {
                format!("{}_gray12", name)
            } else {
                format!("{}_rgb12", name)
            };
            Ok(Some((key, hash_bytes(&bytes_from_i16_samples(&img.data)))))
        }
        other => {
            eprintln!(
                "{}: precision={} not covered by this hash suite, skipping",
                name, other
            );
            Ok(None)
        }
    }
}

/// Best-effort sample-precision scan. Walks the marker segments looking
/// for the first SOF (FFC0..FFCF except FFC4/FFC8/FFCC). Returns the
/// precision byte, or `None` if the scan bails out.
fn detect_precision(data: &[u8]) -> Option<u8> {
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return None;
    }
    let mut i: usize = 2;
    while i + 3 < data.len() {
        if data[i] != 0xFF {
            return None;
        }
        // Skip padding FF bytes.
        while i < data.len() && data[i] == 0xFF {
            i += 1;
        }
        if i >= data.len() {
            return None;
        }
        let marker: u8 = data[i];
        i += 1;
        // Stand-alone markers have no length.
        match marker {
            0x00 | 0x01 | 0xD0..=0xD9 => continue,
            _ => {}
        }
        if i + 1 >= data.len() {
            return None;
        }
        let len: usize = ((data[i] as usize) << 8) | data[i + 1] as usize;
        if len < 2 || i + len > data.len() {
            return None;
        }
        // SOF markers: C0..CF, excluding C4 (DHT), C8 (JPG reserved), CC (DAC).
        let is_sof: bool =
            (0xC0..=0xCF).contains(&marker) && marker != 0xC4 && marker != 0xC8 && marker != 0xCC;
        if is_sof {
            // Segment body starts at i + 2. Precision is the first byte.
            return Some(data[i + 2]);
        }
        i += len;
    }
    None
}

// ---------------------------------------------------------------------------
// Test + updater
// ---------------------------------------------------------------------------

#[test]
fn conformance_reference_image_hashes_stable() {
    let dir: PathBuf = PathBuf::from(TESTIMAGES_DIR);
    if !dir.exists() {
        eprintln!(
            "SKIP: {} not present; run `git submodule update --init references/libjpeg-turbo`",
            TESTIMAGES_DIR
        );
        return;
    }

    let jpegs: Vec<PathBuf> = list_jpegs(&dir);
    assert!(
        !jpegs.is_empty(),
        "{} contains no JPEG fixtures — submodule broken?",
        TESTIMAGES_DIR
    );

    let expected: HashMap<String, Option<String>> = load_expected_hashes();

    let update_mode: bool = std::env::var("UPDATE_HASHES")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    let mut measured: Vec<(String, String)> = Vec::new();
    let mut mismatches: Vec<String> = Vec::new();
    let mut null_entries: Vec<(String, String)> = Vec::new();

    for path in &jpegs {
        match hash_one(path) {
            Ok(Some((key, hash))) => {
                measured.push((key.clone(), hash.clone()));
                match expected.get(&key) {
                    Some(Some(want)) => {
                        if want != &hash {
                            mismatches
                                .push(format!("{}: expected={} measured={}", key, want, hash));
                        }
                    }
                    Some(None) => {
                        null_entries.push((key, hash));
                    }
                    None => {
                        // New fixture without an entry in the JSON. Treat
                        // it like a null entry so the developer can record
                        // it via UPDATE_HASHES=1.
                        null_entries.push((key, hash));
                    }
                }
            }
            Ok(None) => continue,
            Err(e) => panic!("Hashing failed: {}", e),
        }
    }

    // In update mode we rewrite the JSON to disk and pass. We never modify
    // files silently during a regular test run — that would be a surprise
    // side effect and would violate TDD's red/green discipline.
    if update_mode {
        let new_json: String = build_json(&measured);
        let target: PathBuf = PathBuf::from(CONFORMANCE_JSON_REL_PATH);
        std::fs::write(&target, new_json.as_bytes())
            .unwrap_or_else(|e| panic!("failed to write {:?}: {:?}", target, e));
        eprintln!(
            "UPDATE_HASHES=1: wrote {} measured hashes to {:?}",
            measured.len(),
            target
        );
        return;
    }

    if !null_entries.is_empty() {
        for (key, hash) in &null_entries {
            eprintln!("[record] {} -> {}", key, hash);
        }
        eprintln!(
            "\n{} hash entries are null. Run:\n  UPDATE_HASHES=1 cargo test --test \
             worker_b3_reference_image_hashes -- --nocapture\nThen commit \
             {}.",
            null_entries.len(),
            CONFORMANCE_JSON_REL_PATH
        );
    }

    if !mismatches.is_empty() {
        for m in &mismatches {
            eprintln!("MISMATCH: {}", m);
        }
        panic!(
            "{} conformance fixture hashes changed. Investigate whether this is an \
             intentional decoder change; if so, re-record with UPDATE_HASHES=1.",
            mismatches.len()
        );
    }

    assert!(
        !measured.is_empty(),
        "Conformance hash suite measured 0 hashes — nothing was tested."
    );
}

/// Serialize measured hashes back out as sorted, pretty-printed JSON that
/// matches the existing `reference_hashes*.json` style (4-space indent,
/// double-quoted string values).
fn build_json(entries: &[(String, String)]) -> String {
    let mut sorted: Vec<(String, String)> = entries.to_vec();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out: String = String::new();
    out.push_str("{\n");
    out.push_str(
        "    \"_comment\": \"Known-good hashes for every JPEG under \
         references/libjpeg-turbo/testimages/. Written by \
         tests/worker_b3_reference_image_hashes.rs. Hash algorithm: \
         std::hash::DefaultHasher (SipHash-1-3) over the decoded pixel \
         buffer (Rgb for 8-bit fixtures, raw i16 little-endian samples for \
         12-bit fixtures). To regenerate after an intentional decoder \
         change: UPDATE_HASHES=1 cargo test --test \
         worker_b3_reference_image_hashes -- --nocapture.\",\n",
    );
    out.push_str("    \"_format_version\": 1");
    for (k, v) in &sorted {
        out.push_str(",\n    \"");
        out.push_str(k);
        out.push_str("\": \"");
        out.push_str(v);
        out.push('"');
    }
    out.push_str("\n}\n");
    out
}

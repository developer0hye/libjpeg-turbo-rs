use std::collections::BTreeSet;
/// Corpus test harness: validates libjpeg-turbo-rs against C libjpeg-turbo
/// across a directory of JPEG files.
///
/// Usage:
///   cargo run --example corpus_test -- --corpus-dir tests/corpus/ [--decode-only] [--encode-only] [--transform-only]
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{
    compress, decompress, decompress_to, transform, Image, PixelFormat, Subsampling, TransformOp,
};

// ===========================================================================
// C tool discovery
// ===========================================================================

fn c_tool_path(name: &str) -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from(format!("/opt/homebrew/bin/{}", name));
    if homebrew.exists() {
        return Some(homebrew);
    }
    Command::new("which")
        .arg(name)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

// ===========================================================================
// Temp file management
// ===========================================================================

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn new(suffix: &str) -> Self {
        let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid: u32 = std::process::id();
        Self {
            path: std::env::temp_dir()
                .join(format!("corpus_test_{}_{:06}_{}", pid, counter, suffix)),
        }
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn write_bytes(&self, data: &[u8]) -> std::io::Result<()> {
        let mut f = std::fs::File::create(&self.path)?;
        f.write_all(data)?;
        Ok(())
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
    }
}

// ===========================================================================
// PPM / PGM parsing (inlined from tests/helpers/mod.rs)
// ===========================================================================

fn skip_ws_comments(data: &[u8], mut idx: usize) -> usize {
    loop {
        while idx < data.len() && data[idx].is_ascii_whitespace() {
            idx += 1;
        }
        if idx < data.len() && data[idx] == b'#' {
            while idx < data.len() && data[idx] != b'\n' {
                idx += 1;
            }
        } else {
            break;
        }
    }
    idx
}

fn read_number(data: &[u8], idx: usize) -> Option<(usize, usize)> {
    let mut end: usize = idx;
    while end < data.len() && data[end].is_ascii_digit() {
        end += 1;
    }
    if end == idx {
        return None;
    }
    let val: usize = std::str::from_utf8(&data[idx..end]).ok()?.parse().ok()?;
    Some((val, end))
}

fn parse_ppm(data: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    if data.len() < 3 || &data[0..2] != b"P6" {
        return None;
    }
    let mut pos: usize = 2;
    pos = skip_ws_comments(data, pos);
    let (width, next) = read_number(data, pos)?;
    pos = skip_ws_comments(data, next);
    let (height, next) = read_number(data, pos)?;
    pos = skip_ws_comments(data, next);
    let (maxval, next) = read_number(data, pos)?;
    pos = next;
    if pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }
    let pixels = parse_pnm_samples(
        data.get(pos..)?,
        width.checked_mul(height)?.checked_mul(3)?,
        maxval,
    )?;
    Some((width, height, pixels))
}

fn parse_pgm(data: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    if data.len() < 3 || &data[0..2] != b"P5" {
        return None;
    }
    let mut pos: usize = 2;
    pos = skip_ws_comments(data, pos);
    let (width, next) = read_number(data, pos)?;
    pos = skip_ws_comments(data, next);
    let (height, next) = read_number(data, pos)?;
    pos = skip_ws_comments(data, next);
    let (maxval, next) = read_number(data, pos)?;
    pos = next;
    if pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }
    let pixels = parse_pnm_samples(data.get(pos..)?, width.checked_mul(height)?, maxval)?;
    Some((width, height, pixels))
}

fn parse_pnm_samples(data: &[u8], sample_count: usize, maxval: usize) -> Option<Vec<u8>> {
    if maxval == 0 || maxval > u16::MAX as usize {
        return None;
    }
    if maxval <= u8::MAX as usize {
        let samples = data.get(..sample_count)?;
        if maxval == u8::MAX as usize {
            return Some(samples.to_vec());
        }
        return Some(
            samples
                .iter()
                .map(|sample| (*sample as usize * 255 / maxval) as u8)
                .collect(),
        );
    }

    let byte_count = sample_count.checked_mul(2)?;
    let samples = data.get(..byte_count)?;
    Some(
        samples
            .chunks_exact(2)
            .map(|sample| {
                let value = u16::from_be_bytes([sample[0], sample[1]]) as usize;
                (value * 255 / maxval) as u8
            })
            .collect(),
    )
}

// ===========================================================================
// TSV output row
// ===========================================================================

#[derive(Clone)]
enum TestResult {
    Pass { max_diff: u32 },
    ExpectedReject { notes: String },
    KnownMismatch { max_diff: u32, notes: String },
    Fail { max_diff: u32, notes: String },
    Crash { notes: String },
    Skip { notes: String },
}

impl TestResult {
    fn label(&self) -> &'static str {
        match self {
            TestResult::Pass { .. } => "pass",
            TestResult::ExpectedReject { .. } => "expected-reject",
            TestResult::KnownMismatch { .. } => "known-mismatch",
            TestResult::Fail { .. } => "fail",
            TestResult::Crash { .. } => "crash",
            TestResult::Skip { .. } => "skip",
        }
    }

    fn max_diff_str(&self) -> String {
        match self {
            TestResult::Pass { max_diff } => max_diff.to_string(),
            TestResult::Fail { max_diff, .. } => max_diff.to_string(),
            TestResult::KnownMismatch { max_diff, .. } => max_diff.to_string(),
            TestResult::ExpectedReject { .. }
            | TestResult::Crash { .. }
            | TestResult::Skip { .. } => "-".to_string(),
        }
    }

    fn notes(&self) -> &str {
        match self {
            TestResult::Pass { .. } => "",
            TestResult::ExpectedReject { notes } => notes,
            TestResult::KnownMismatch { notes, .. } => notes,
            TestResult::Fail { notes, .. } => notes,
            TestResult::Crash { notes } => notes,
            TestResult::Skip { notes } => notes,
        }
    }
}

fn print_row(file: &str, operation: &str, result: &TestResult) {
    println!(
        "{}\t{}\t{}\t{}\t{}",
        file,
        operation,
        result.label(),
        result.max_diff_str(),
        result.notes()
    );
}

// ===========================================================================
// Counters
// ===========================================================================

#[derive(Default)]
struct Counts {
    pass: u32,
    expected_reject: u32,
    known_mismatch: u32,
    fail: u32,
    crash: u32,
    skip: u32,
}

impl Counts {
    fn record(&mut self, r: &TestResult) {
        match r {
            TestResult::Pass { .. } => self.pass += 1,
            TestResult::ExpectedReject { .. } => self.expected_reject += 1,
            TestResult::KnownMismatch { .. } => self.known_mismatch += 1,
            TestResult::Fail { .. } => self.fail += 1,
            TestResult::Crash { .. } => self.crash += 1,
            TestResult::Skip { .. } => self.skip += 1,
        }
    }

    fn has_unexpected_outcomes(&self) -> bool {
        self.fail != 0 || self.crash != 0 || self.skip != 0
    }

    fn total(&self) -> u32 {
        self.pass + self.expected_reject + self.known_mismatch + self.fail + self.crash + self.skip
    }
}

#[derive(Default)]
struct ExpectedCoverage {
    observed: BTreeSet<String>,
}

impl ExpectedCoverage {
    fn record(&mut self, relative_path: &Path, operation: &str, result: &TestResult) {
        let path = relative_path.to_string_lossy().replace('\\', "/");
        let is_p4_20_decode =
            path == "fuzz_seeds/24fd23785278a9577686f501e17ee8164f8b977b" && operation == "decode";
        let label = match result {
            TestResult::Pass { .. } if is_p4_20_decode => "pass",
            TestResult::ExpectedReject { .. } => "expected-reject",
            TestResult::KnownMismatch { .. } => "known-mismatch",
            _ => return,
        };
        self.observed
            .insert(format!("{path}\t{operation}\t{label}"));
    }

    fn verify_full_corpus(&self, is_full_corpus_run: bool) -> Result<(), String> {
        if !is_full_corpus_run {
            return Ok(());
        }
        let required = required_expected_outcomes();
        if self.observed == required {
            return Ok(());
        }
        let missing = required
            .difference(&self.observed)
            .cloned()
            .collect::<Vec<_>>();
        let unexpected = self
            .observed
            .difference(&required)
            .cloned()
            .collect::<Vec<_>>();
        Err(format!(
            "expected-outcome coverage mismatch; missing={missing:?}, unexpected={unexpected:?}"
        ))
    }
}

fn required_expected_outcomes() -> BTreeSet<String> {
    const TRANSFORMS: &[&str] = &[
        "transform_rotate90",
        "transform_rotate180",
        "transform_rotate270",
        "transform_fliph",
        "transform_flipv",
        "transform_transpose",
        "transform_transverse",
    ];
    let mut required = BTreeSet::new();
    let mut add = |path: &str, operation: &str, label: &str| {
        required.insert(format!("{path}\t{operation}\t{label}"));
    };

    add(
        "fuzz_seeds/24fd23785278a9577686f501e17ee8164f8b977b",
        "decode",
        p4_20_expected_label(),
    );
    add(
        "fuzz_seeds/crash-cf56f76b13a5eaa5a65b46a3503c0951f034d735",
        "decode",
        "expected-reject",
    );
    let corrupt = "fixtures/fuzz_repro/corrupt_huffman_65x65_422.jpg";
    add(corrupt, "decode", "expected-reject");
    add(corrupt, "encode", "expected-reject");

    for path in [
        "fixtures/fuzz_repro/arith_noninterleaved_16x16_444.jpg",
        "fixtures/fuzz_repro/arith_partial_interleaved_16x16_444.jpg",
    ] {
        for operation in TRANSFORMS {
            add(path, operation, "known-mismatch");
        }
    }
    for path in [
        "fixtures/fuzz_repro/corrupt_huffman_65x65_422.jpg",
        "fixtures/fuzz_repro/multiscan_noninterleaved_64x64_444.jpg",
        "fixtures/real_world/zune_non_interleaved_420_64x64.jpg",
        "fixtures/real_world/zune_non_interleaved_422_65x65.jpg",
        "fixtures/real_world/zune_non_interleaved_440_64x64.jpg",
        "fixtures/real_world/zune_non_interleaved_444_64x64.jpg",
        "fixtures/real_world/zune_tiny_non_interleaved_444_16x16.jpg",
        "fixtures/real_world/zune_mjpeg_huffman_1280x720.jpg",
        "fixtures/real_world/zune_grayscale_progressive_900x675.jpg",
    ] {
        for operation in TRANSFORMS {
            add(path, operation, "expected-reject");
        }
    }
    for path in [
        "fixtures/real_world/pil_cmyk.jpg",
        "fixtures/real_world/zune_ycck_1318x611_4comp.jpg",
        "fixtures/real_world/zune_ycck_progressive_383x740_4comp.jpg",
    ] {
        for operation in TRANSFORMS {
            add(path, operation, "known-mismatch");
        }
    }
    required
}

fn p4_20_expected_label() -> &'static str {
    if std::env::var("JSIMD_FORCENONE").ok().as_deref() == Some("1") {
        return "known-mismatch";
    }
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") {
        return "pass";
    }
    "known-mismatch"
}

// ===========================================================================
// File discovery
// ===========================================================================

fn collect_jpeg_files(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files: Vec<PathBuf> = Vec::new();
    collect_recursive(dir, &mut files)?;
    files.sort();
    Ok(files)
}

fn collect_recursive(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = std::fs::read_dir(dir)
        .map_err(|error| format!("read corpus directory {}: {error}", dir.display()))?;
    for entry in entries {
        let entry =
            entry.map_err(|error| format!("read corpus entry under {}: {error}", dir.display()))?;
        let path: PathBuf = entry.path();
        let file_type = entry
            .file_type()
            .map_err(|error| format!("inspect corpus path {}: {error}", path.display()))?;
        if file_type.is_symlink() {
            return Err(format!(
                "symlink is not allowed in corpus: {}",
                path.display()
            ));
        }
        if file_type.is_dir() {
            collect_recursive(&path, out)?;
            continue;
        }
        if !file_type.is_file() {
            continue;
        }

        let extension_is_jpeg = path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| {
                extension.eq_ignore_ascii_case("jpg") || extension.eq_ignore_ascii_case("jpeg")
            });
        let extensionless_soi = if path.extension().is_none() {
            let mut file = std::fs::File::open(&path)
                .map_err(|error| format!("open corpus input {}: {error}", path.display()))?;
            let mut signature = [0_u8; 2];
            let bytes_read = file
                .read(&mut signature)
                .map_err(|error| format!("read corpus input {}: {error}", path.display()))?;
            bytes_read == signature.len() && signature == [0xff, 0xd8]
        } else {
            false
        };
        if extension_is_jpeg || extensionless_soi {
            out.push(path);
        }
    }
    Ok(())
}

fn operation_applies(path: &Path, operation: &str) -> bool {
    let is_decompress_fuzz_seed = path
        .components()
        .any(|component| component.as_os_str() == "fuzz_seeds");
    !is_decompress_fuzz_seed || operation == "decode"
}

// ===========================================================================
// decode_with_c_djpeg: run djpeg and return (w, h, pixels, is_gray)
// ===========================================================================

fn decode_with_c_djpeg(
    djpeg: &Path,
    jpeg_data: &[u8],
    grayscale: bool,
) -> Result<(usize, usize, Vec<u8>, bool), String> {
    let input_tmp: TempFile = TempFile::new("input.jpg");
    input_tmp
        .write_bytes(jpeg_data)
        .map_err(|e| format!("write temp: {}", e))?;

    if grayscale {
        let out_tmp: TempFile = TempFile::new("out.pgm");
        let status = Command::new(djpeg)
            .args([
                "-strict",
                "-grayscale",
                "-outfile",
                out_tmp.path().to_str().unwrap(),
                input_tmp.path().to_str().unwrap(),
            ])
            .output()
            .map_err(|e| format!("djpeg exec: {}", e))?;
        if !status.status.success() {
            return Err(format!(
                "djpeg failed: {}",
                String::from_utf8_lossy(&status.stderr)
            ));
        }
        let pgm_data: Vec<u8> =
            std::fs::read(out_tmp.path()).map_err(|e| format!("read pgm: {}", e))?;
        let (w, h, pixels) = parse_pgm(&pgm_data).ok_or("failed to parse PGM")?;
        Ok((w, h, pixels, true))
    } else {
        let out_tmp: TempFile = TempFile::new("out.ppm");
        let status = Command::new(djpeg)
            .args([
                "-strict",
                "-ppm",
                "-outfile",
                out_tmp.path().to_str().unwrap(),
                input_tmp.path().to_str().unwrap(),
            ])
            .output()
            .map_err(|e| format!("djpeg exec: {}", e))?;
        if !status.status.success() {
            return Err(format!(
                "djpeg failed: {}",
                String::from_utf8_lossy(&status.stderr)
            ));
        }
        let ppm_data: Vec<u8> =
            std::fs::read(out_tmp.path()).map_err(|e| format!("read ppm: {}", e))?;
        let (w, h, pixels) = parse_ppm(&ppm_data).ok_or("failed to parse PPM")?;
        Ok((w, h, pixels, false))
    }
}

// ===========================================================================
// Tests
// ===========================================================================

fn run_decode_test(djpeg: &Path, jpeg_data: &[u8]) -> TestResult {
    // Step 1: decode with Rust
    let native_image: Image = match decompress(jpeg_data) {
        Ok(img) => img,
        Err(e) => {
            return TestResult::Crash {
                notes: format!("Rust decompress error: {e}"),
            }
        }
    };

    let is_gray: bool = native_image.pixel_format == PixelFormat::Grayscale;
    let rust_img: Image = if is_gray {
        native_image
    } else {
        match decompress_to(jpeg_data, PixelFormat::Rgb) {
            Ok(image) => image,
            Err(error) => {
                return TestResult::Crash {
                    notes: format!("Rust RGB decompress error: {error}"),
                }
            }
        }
    };

    // Step 2: decode with C djpeg
    let (c_w, c_h, c_pixels, _) = match decode_with_c_djpeg(djpeg, jpeg_data, is_gray) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::Crash {
                notes: format!("strict C output parse failed after acceptance: {e}"),
            }
        }
    };

    // Step 3: compare dimensions
    if rust_img.width != c_w || rust_img.height != c_h {
        return TestResult::Fail {
            max_diff: u32::MAX,
            notes: format!(
                "dimension mismatch: Rust {}x{} vs C {}x{}",
                rust_img.width, rust_img.height, c_w, c_h
            ),
        };
    }

    if rust_img.data.len() != c_pixels.len() {
        return TestResult::Fail {
            max_diff: u32::MAX,
            notes: format!(
                "pixel buffer length mismatch: Rust {} vs C {}",
                rust_img.data.len(),
                c_pixels.len()
            ),
        };
    }

    // Step 4: compute max diff
    let max_diff: u32 = rust_img
        .data
        .iter()
        .zip(c_pixels.iter())
        .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs())
        .max()
        .unwrap_or(0);

    if max_diff == 0 {
        TestResult::Pass { max_diff: 0 }
    } else {
        // Find location of first differing pixel
        let bpp: usize = rust_img.pixel_format.bytes_per_pixel();
        let first_diff_byte: usize = rust_img
            .data
            .iter()
            .zip(c_pixels.iter())
            .position(|(&a, &b)| a != b)
            .unwrap_or(0);
        let pixel_idx: usize = first_diff_byte / bpp;
        let px: usize = pixel_idx % rust_img.width;
        let py: usize = pixel_idx / rust_img.width;
        TestResult::Fail {
            max_diff,
            notes: format!("pixel diff at ({},{})", px, py),
        }
    }
}

fn apply_expected_reject(path: &Path, operation: &str, result: TestResult) -> TestResult {
    const CORRUPT_HUFFMAN_SUFFIX: &str = "fixtures/fuzz_repro/corrupt_huffman_65x65_422.jpg";
    const CORRUPT_HUFFMAN_ERROR: &str = "Rust decompress error: corrupt data: invalid Huffman code";
    const P4_20_IDCT_SUFFIX: &str = "fuzz_seeds/24fd23785278a9577686f501e17ee8164f8b977b";
    const P4_21_SAMPLING_SUFFIX: &str = "fuzz_seeds/crash-cf56f76b13a5eaa5a65b46a3503c0951f034d735";

    let normalized_path = path.to_string_lossy().replace('\\', "/");
    let known_transform_gap = |suffixes: &[&str], result: &TestResult, note_fragment: &str| {
        operation.starts_with("transform_")
            && suffixes
                .iter()
                .any(|suffix| normalized_path.ends_with(suffix))
            && result.notes().contains(note_fragment)
    };

    let multi_scan_transform_gaps = [
        "fixtures/fuzz_repro/corrupt_huffman_65x65_422.jpg",
        "fixtures/fuzz_repro/multiscan_noninterleaved_64x64_444.jpg",
        "fixtures/real_world/zune_non_interleaved_420_64x64.jpg",
        "fixtures/real_world/zune_non_interleaved_422_65x65.jpg",
        "fixtures/real_world/zune_non_interleaved_440_64x64.jpg",
        "fixtures/real_world/zune_non_interleaved_444_64x64.jpg",
        "fixtures/real_world/zune_tiny_non_interleaved_444_16x16.jpg",
    ];
    let pinned_transform_pixel_mismatch = |max_diff: u32| -> Option<&'static str> {
        const ALL_OPS: &[&str] = &[
            "transform_rotate90",
            "transform_rotate180",
            "transform_rotate270",
            "transform_fliph",
            "transform_flipv",
            "transform_transpose",
            "transform_transverse",
        ];
        if !ALL_OPS.contains(&operation) {
            return None;
        }

        let arithmetic = normalized_path
            .ends_with("fixtures/fuzz_repro/arith_noninterleaved_16x16_444.jpg")
            || normalized_path
                .ends_with("fixtures/fuzz_repro/arith_partial_interleaved_16x16_444.jpg");
        if arithmetic && max_diff == 157 {
            return Some("known arithmetic multi-scan transform parity gap");
        }

        if normalized_path.ends_with("fixtures/real_world/pil_cmyk.jpg") {
            let expected = match operation {
                "transform_rotate180" | "transform_fliph" | "transform_flipv" => 153,
                _ => 152,
            };
            return (max_diff == expected).then_some("known CMYK transform parity gap");
        }
        if normalized_path.ends_with("fixtures/real_world/zune_ycck_1318x611_4comp.jpg")
            && max_diff == 250
        {
            return Some("known YCCK transform parity gap");
        }
        if normalized_path.ends_with("fixtures/real_world/zune_ycck_progressive_383x740_4comp.jpg")
            && max_diff == 255
        {
            return Some("known progressive YCCK transform parity gap");
        }
        None
    };

    match result {
        TestResult::Fail { max_diff, notes }
            if operation == "decode"
                && normalized_path.ends_with(P4_20_IDCT_SUFFIX)
                && matches!(max_diff, 34 | 255)
                && notes == "pixel diff at (0,0)" =>
        {
            TestResult::KnownMismatch {
                max_diff,
                notes: "P4-20 i16 IDCT full-path fidelity gap".to_string(),
            }
        }
        TestResult::Crash { notes }
            if operation == "decode"
                && normalized_path.ends_with(P4_21_SAMPLING_SUFFIX)
                && notes
                    == "Rust decompress error: corrupt data: chroma upsample factor zero (a chroma component out-samples luma): cb=4x0 cr=4x1" =>
        {
            TestResult::ExpectedReject {
                notes: "P4-21 non-standard chroma-outsamples-luma limitation".to_string(),
            }
        }
        TestResult::Crash { notes }
            if operation == "decode"
                && normalized_path.ends_with(CORRUPT_HUFFMAN_SUFFIX)
                && notes == CORRUPT_HUFFMAN_ERROR =>
        {
            TestResult::ExpectedReject {
                notes: "known corrupt Huffman stream rejected exactly".to_string(),
            }
        }
        TestResult::Crash { notes }
            if operation == "encode"
                && normalized_path.ends_with(CORRUPT_HUFFMAN_SUFFIX)
                && notes == CORRUPT_HUFFMAN_ERROR =>
        {
            TestResult::ExpectedReject {
                notes: "known corrupt Huffman stream cannot seed encode comparison".to_string(),
            }
        }
        TestResult::Fail { max_diff, notes }
            if notes.starts_with("pixel diff (Rust")
                && pinned_transform_pixel_mismatch(max_diff).is_some() =>
        {
            TestResult::KnownMismatch {
                max_diff,
                notes: pinned_transform_pixel_mismatch(max_diff).unwrap().to_string(),
            }
        }
        result
            if known_transform_gap(
                &multi_scan_transform_gaps,
                &result,
                "baseline SOS covers 1 components but frame has 3",
            ) =>
        {
            TestResult::ExpectedReject {
                notes: "known non-interleaved transform limitation".to_string(),
            }
        }
        result
            if known_transform_gap(
                &["fixtures/real_world/zune_mjpeg_huffman_1280x720.jpg"],
                &result,
                "missing DC Huffman table 0",
            ) =>
        {
            TestResult::ExpectedReject {
                notes: "known MJPEG implicit-Huffman transform limitation".to_string(),
            }
        }
        result
            if known_transform_gap(
                &["fixtures/real_world/zune_grayscale_progressive_900x675.jpg"],
                &result,
                "extraneous bytes before marker 0xd9",
            ) =>
        {
            TestResult::ExpectedReject {
                notes: "known progressive grayscale transform output gap".to_string(),
            }
        }
        other => other,
    }
}

fn run_encode_test(djpeg: &Path, cjpeg: &Path, jpeg_data: &[u8]) -> TestResult {
    // Step 1: quick Rust decompress to detect grayscale
    let is_gray: bool = match decompress(jpeg_data) {
        Ok(img) => img.pixel_format == PixelFormat::Grayscale,
        Err(e) => {
            return TestResult::Crash {
                notes: format!("Rust decompress error: {}", e),
            }
        }
    };

    // Step 2: decode with C djpeg to get reference pixels
    let (c_w, c_h, c_pixels, _) = match decode_with_c_djpeg(djpeg, jpeg_data, is_gray) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::Skip {
                notes: format!("C djpeg error: {}", e),
            }
        }
    };

    // Step 3: write PGM or PPM for C cjpeg input
    let (pixel_format, subsampling, cjpeg_sample_args): (PixelFormat, Subsampling, &[&str]) =
        if is_gray {
            (PixelFormat::Grayscale, Subsampling::S444, &["-grayscale"])
        } else {
            (PixelFormat::Rgb, Subsampling::S420, &["-sample", "2x2"])
        };

    let input_tmp: TempFile = if is_gray {
        let pgm_tmp: TempFile = TempFile::new("input.pgm");
        let mut pgm: Vec<u8> = Vec::new();
        write!(pgm, "P5\n{} {}\n255\n", c_w, c_h).unwrap();
        pgm.extend_from_slice(&c_pixels);
        if pgm_tmp.write_bytes(&pgm).is_err() {
            return TestResult::Crash {
                notes: "failed to write PGM".to_string(),
            };
        }
        pgm_tmp
    } else {
        let ppm_tmp: TempFile = TempFile::new("input.ppm");
        let mut ppm: Vec<u8> = Vec::new();
        write!(ppm, "P6\n{} {}\n255\n", c_w, c_h).unwrap();
        ppm.extend_from_slice(&c_pixels);
        if ppm_tmp.write_bytes(&ppm).is_err() {
            return TestResult::Crash {
                notes: "failed to write PPM".to_string(),
            };
        }
        ppm_tmp
    };

    // Step 4: encode with Rust
    let rust_jpeg: Vec<u8> = match compress(&c_pixels, c_w, c_h, pixel_format, 75, subsampling) {
        Ok(j) => j,
        Err(e) => {
            return TestResult::Crash {
                notes: format!("Rust compress error: {}", e),
            }
        }
    };

    // Step 5: encode with C cjpeg
    let c_jpeg_tmp: TempFile = TempFile::new("c_out.jpg");
    let mut cjpeg_args: Vec<&str> = vec!["-quality", "75"];
    cjpeg_args.extend_from_slice(cjpeg_sample_args);
    cjpeg_args.extend_from_slice(&[
        "-outfile",
        c_jpeg_tmp.path().to_str().unwrap(),
        input_tmp.path().to_str().unwrap(),
    ]);
    let status = Command::new(cjpeg).args(&cjpeg_args).output();

    let c_jpeg_data: Vec<u8> = match status {
        Ok(out) if out.status.success() => match std::fs::read(c_jpeg_tmp.path()) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::Skip {
                    notes: format!("read C jpeg: {}", e),
                }
            }
        },
        Ok(out) => {
            return TestResult::Skip {
                notes: format!("cjpeg failed: {}", String::from_utf8_lossy(&out.stderr)),
            }
        }
        Err(e) => {
            return TestResult::Skip {
                notes: format!("cjpeg exec: {}", e),
            }
        }
    };

    // Step 6: decode both Rust and C jpegs with djpeg and compare pixels
    let (rw, rh, r_pixels, _) = match decode_with_c_djpeg(djpeg, &rust_jpeg, is_gray) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::Crash {
                notes: format!("djpeg on Rust output: {}", e),
            }
        }
    };

    let (cw, ch, cp_pixels, _) = match decode_with_c_djpeg(djpeg, &c_jpeg_data, is_gray) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::Skip {
                notes: format!("djpeg on C output: {}", e),
            }
        }
    };

    if rw != cw || rh != ch {
        return TestResult::Fail {
            max_diff: u32::MAX,
            notes: format!(
                "encode dimension mismatch: Rust {}x{} vs C {}x{}",
                rw, rh, cw, ch
            ),
        };
    }

    if r_pixels.len() != cp_pixels.len() {
        return TestResult::Fail {
            max_diff: u32::MAX,
            notes: format!(
                "encode pixel buffer mismatch: Rust {} vs C {}",
                r_pixels.len(),
                cp_pixels.len()
            ),
        };
    }

    let bpp: usize = if is_gray { 1 } else { 3 };
    let max_diff: u32 = r_pixels
        .iter()
        .zip(cp_pixels.iter())
        .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs())
        .max()
        .unwrap_or(0);

    if max_diff == 0 {
        TestResult::Pass { max_diff: 0 }
    } else {
        let first_diff_byte: usize = r_pixels
            .iter()
            .zip(cp_pixels.iter())
            .position(|(&a, &b)| a != b)
            .unwrap_or(0);
        let pixel_idx: usize = first_diff_byte / bpp;
        let px: usize = pixel_idx % rw;
        let py: usize = pixel_idx / rw;
        TestResult::Fail {
            max_diff,
            notes: format!("pixel diff at ({},{})", px, py),
        }
    }
}

fn transform_op_to_jpegtran_args(op: TransformOp) -> Vec<&'static str> {
    match op {
        TransformOp::Rot90 => vec!["-rotate", "90"],
        TransformOp::Rot180 => vec!["-rotate", "180"],
        TransformOp::Rot270 => vec!["-rotate", "270"],
        TransformOp::HFlip => vec!["-flip", "horizontal"],
        TransformOp::VFlip => vec!["-flip", "vertical"],
        TransformOp::Transpose => vec!["-transpose"],
        TransformOp::Transverse => vec!["-transverse"],
        TransformOp::None => vec![],
    }
}

fn transform_op_name(op: TransformOp) -> &'static str {
    match op {
        TransformOp::Rot90 => "transform_rotate90",
        TransformOp::Rot180 => "transform_rotate180",
        TransformOp::Rot270 => "transform_rotate270",
        TransformOp::HFlip => "transform_fliph",
        TransformOp::VFlip => "transform_flipv",
        TransformOp::Transpose => "transform_transpose",
        TransformOp::Transverse => "transform_transverse",
        TransformOp::None => "transform_none",
    }
}

fn run_transform_test(
    jpegtran: &Path,
    djpeg: &Path,
    jpeg_data: &[u8],
    op: TransformOp,
) -> TestResult {
    // Step 1: transform with Rust
    let rust_out: Vec<u8> = match transform(jpeg_data, op) {
        Ok(d) => d,
        Err(e) => {
            // Lossless (SOF3/SOF11) streams don't use AC Huffman tables and
            // can't be transformed via the DCT-based path. Classify as skip.
            let msg = e.to_string();
            if msg.contains("missing AC Huffman table") || msg.contains("lossless") {
                return TestResult::Skip {
                    notes: format!("lossless JPEG not transformable: {}", e),
                };
            }
            return TestResult::Crash {
                notes: format!("Rust transform error: {}", e),
            };
        }
    };

    // Step 2: transform with C jpegtran
    let jt_args: Vec<&str> = transform_op_to_jpegtran_args(op);
    if jt_args.is_empty() {
        // TransformOp::None — both outputs must be byte-identical to input
        if rust_out == jpeg_data {
            return TestResult::Pass { max_diff: 0 };
        } else {
            return TestResult::Fail {
                max_diff: 1,
                notes: "None transform changed bytes".to_string(),
            };
        }
    }

    let input_tmp: TempFile = TempFile::new("input.jpg");
    if input_tmp.write_bytes(jpeg_data).is_err() {
        return TestResult::Crash {
            notes: "failed to write temp input".to_string(),
        };
    }

    let out_tmp: TempFile = TempFile::new("jt_out.jpg");
    let mut cmd = Command::new(jpegtran);
    cmd.arg("-copy").arg("none");
    cmd.args(&jt_args);
    cmd.args([
        "-outfile",
        out_tmp.path().to_str().unwrap(),
        input_tmp.path().to_str().unwrap(),
    ]);

    let status = cmd.output();
    let c_jpeg: Vec<u8> = match status {
        Ok(out) if out.status.success() => match std::fs::read(out_tmp.path()) {
            Ok(d) => d,
            Err(e) => {
                return TestResult::Skip {
                    notes: format!("read jpegtran output: {}", e),
                }
            }
        },
        Ok(out) => {
            return TestResult::Skip {
                notes: format!("jpegtran failed: {}", String::from_utf8_lossy(&out.stderr)),
            }
        }
        Err(e) => {
            return TestResult::Skip {
                notes: format!("jpegtran exec: {}", e),
            }
        }
    };

    // Step 3: decode both outputs with djpeg and compare pixels.
    // Raw bytes may differ (Rust and C generate different Huffman tables),
    // but the decoded pixels must be identical for a lossless transform.
    // Detect grayscale from the source JPEG so djpeg uses the right output format.
    let is_gray: bool = match decompress(jpeg_data) {
        Ok(img) => img.pixel_format == PixelFormat::Grayscale,
        Err(_) => false,
    };
    let rust_decoded = match decode_with_c_djpeg(djpeg, &rust_out, is_gray) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::Crash {
                notes: format!("djpeg on Rust transform output: {}", e),
            }
        }
    };
    let c_decoded = match decode_with_c_djpeg(djpeg, &c_jpeg, is_gray) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::Skip {
                notes: format!("djpeg on C transform output: {}", e),
            }
        }
    };

    let (rw, rh, rust_pixels, _) = rust_decoded;
    let (cw, ch, c_pixels, _) = c_decoded;

    if rw != cw || rh != ch {
        return TestResult::Fail {
            max_diff: u32::MAX,
            notes: format!(
                "transform dimension mismatch: Rust {}x{} vs C {}x{}",
                rw, rh, cw, ch
            ),
        };
    }

    if rust_pixels.len() != c_pixels.len() {
        return TestResult::Fail {
            max_diff: u32::MAX,
            notes: format!(
                "transform pixel buffer mismatch: Rust {} vs C {}",
                rust_pixels.len(),
                c_pixels.len()
            ),
        };
    }

    let max_diff: u32 = rust_pixels
        .iter()
        .zip(c_pixels.iter())
        .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs())
        .max()
        .unwrap_or(0);

    if max_diff == 0 {
        TestResult::Pass { max_diff: 0 }
    } else {
        TestResult::Fail {
            max_diff,
            notes: format!(
                "pixel diff (Rust {} bytes, C {} bytes)",
                rust_out.len(),
                c_jpeg.len()
            ),
        }
    }
}

// ===========================================================================
// Panic-safe wrappers
// ===========================================================================

fn catch_decode(djpeg: &Path, jpeg_data: Vec<u8>) -> TestResult {
    let djpeg_owned: PathBuf = djpeg.to_path_buf();
    match std::panic::catch_unwind(move || run_decode_test(&djpeg_owned, &jpeg_data)) {
        Ok(r) => r,
        Err(e) => TestResult::Crash {
            notes: format!("panicked: {}", panic_msg(e)),
        },
    }
}

fn catch_encode(djpeg: &Path, cjpeg: &Path, jpeg_data: Vec<u8>) -> TestResult {
    let djpeg_owned: PathBuf = djpeg.to_path_buf();
    let cjpeg_owned: PathBuf = cjpeg.to_path_buf();
    match std::panic::catch_unwind(move || run_encode_test(&djpeg_owned, &cjpeg_owned, &jpeg_data))
    {
        Ok(r) => r,
        Err(e) => TestResult::Crash {
            notes: format!("panicked: {}", panic_msg(e)),
        },
    }
}

fn catch_transform(
    jpegtran: &Path,
    djpeg: &Path,
    jpeg_data: Vec<u8>,
    op: TransformOp,
) -> TestResult {
    let jt_owned: PathBuf = jpegtran.to_path_buf();
    let dj_owned: PathBuf = djpeg.to_path_buf();
    match std::panic::catch_unwind(move || run_transform_test(&jt_owned, &dj_owned, &jpeg_data, op))
    {
        Ok(r) => r,
        Err(e) => TestResult::Crash {
            notes: format!("panicked: {}", panic_msg(e)),
        },
    }
}

fn panic_msg(e: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<&str>() {
        s.to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

// ===========================================================================
// CLI parsing
// ===========================================================================

struct Config {
    corpus_dir: PathBuf,
    run_decode: bool,
    run_encode: bool,
    run_transform: bool,
}

fn parse_args() -> Config {
    let args: Vec<String> = std::env::args().collect();
    let mut corpus_dir: Option<PathBuf> = None;
    let mut decode_only: bool = false;
    let mut encode_only: bool = false;
    let mut transform_only: bool = false;

    let mut i: usize = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--corpus-dir" => {
                i += 1;
                if i < args.len() {
                    corpus_dir = Some(PathBuf::from(&args[i]));
                }
            }
            "--decode-only" => decode_only = true,
            "--encode-only" => encode_only = true,
            "--transform-only" => transform_only = true,
            _ => {}
        }
        i += 1;
    }

    let corpus_dir: PathBuf = corpus_dir.unwrap_or_else(|| {
        eprintln!("Usage: corpus_test --corpus-dir <dir> [--decode-only] [--encode-only] [--transform-only]");
        std::process::exit(1);
    });

    // If none of the filter flags is set, run all three
    let run_all: bool = !decode_only && !encode_only && !transform_only;

    Config {
        corpus_dir,
        run_decode: run_all || decode_only,
        run_encode: run_all || encode_only,
        run_transform: run_all || transform_only,
    }
}

// ===========================================================================
// Main
// ===========================================================================

fn main() {
    let cfg: Config = parse_args();

    // Discover C tools once
    let djpeg: Option<PathBuf> = c_tool_path("djpeg");
    let cjpeg: Option<PathBuf> = c_tool_path("cjpeg");
    let jpegtran: Option<PathBuf> = c_tool_path("jpegtran");

    if djpeg.is_none() {
        eprintln!("WARNING: djpeg not found — decode and encode tests will be skipped");
    }
    if cjpeg.is_none() {
        eprintln!("WARNING: cjpeg not found — encode tests will be skipped");
    }
    if jpegtran.is_none() {
        eprintln!("WARNING: jpegtran not found — transform tests will be skipped");
    }

    // Collect JPEG files
    let files: Vec<PathBuf> = collect_jpeg_files(&cfg.corpus_dir).unwrap_or_else(|error| {
        eprintln!("Failed to discover JPEG corpus: {error}");
        std::process::exit(1);
    });
    if files.is_empty() {
        eprintln!("No JPEG files found in {:?}", cfg.corpus_dir);
        std::process::exit(1);
    }

    // TSV header
    println!("file\toperation\tresult\tmax_diff\tnotes");

    let mut decode_counts: Counts = Counts::default();
    let mut encode_counts: Counts = Counts::default();
    let mut transform_counts: Counts = Counts::default();
    let mut expected_coverage = ExpectedCoverage::default();

    let transform_ops: &[TransformOp] = &[
        TransformOp::Rot90,
        TransformOp::Rot180,
        TransformOp::Rot270,
        TransformOp::HFlip,
        TransformOp::VFlip,
        TransformOp::Transpose,
        TransformOp::Transverse,
    ];

    for file_path in &files {
        let file_str: &str = file_path.to_str().unwrap_or("<invalid>");

        let jpeg_data: Vec<u8> = match std::fs::read(file_path) {
            Ok(d) => d,
            Err(e) => {
                let crash: TestResult = TestResult::Crash {
                    notes: format!("read file: {}", e),
                };
                if cfg.run_decode {
                    print_row(file_str, "decode", &crash);
                    decode_counts.record(&crash);
                }
                if cfg.run_encode {
                    print_row(file_str, "encode", &crash);
                    encode_counts.record(&crash);
                }
                if cfg.run_transform {
                    for &op in transform_ops {
                        print_row(file_str, transform_op_name(op), &crash);
                        transform_counts.record(&crash);
                    }
                }
                continue;
            }
        };
        // Decode test
        if cfg.run_decode {
            let result: TestResult = match &djpeg {
                Some(djpeg_path) => catch_decode(djpeg_path, jpeg_data.clone()),
                None => TestResult::Skip {
                    notes: "djpeg not found".to_string(),
                },
            };
            let result = apply_expected_reject(file_path, "decode", result);
            let relative_path = file_path.strip_prefix(&cfg.corpus_dir).unwrap_or(file_path);
            expected_coverage.record(relative_path, "decode", &result);
            print_row(file_str, "decode", &result);
            decode_counts.record(&result);
        }

        // Encode test
        if cfg.run_encode && operation_applies(file_path, "encode") {
            let result: TestResult = match (&djpeg, &cjpeg) {
                (Some(dj), Some(cj)) => catch_encode(dj, cj, jpeg_data.clone()),
                _ => TestResult::Skip {
                    notes: "djpeg or cjpeg not found".to_string(),
                },
            };
            let result = apply_expected_reject(file_path, "encode", result);
            let relative_path = file_path.strip_prefix(&cfg.corpus_dir).unwrap_or(file_path);
            expected_coverage.record(relative_path, "encode", &result);
            print_row(file_str, "encode", &result);
            encode_counts.record(&result);
        }

        // Transform tests
        if cfg.run_transform && operation_applies(file_path, "transform") {
            for &op in transform_ops {
                let result: TestResult = match (&jpegtran, &djpeg) {
                    (Some(jt), Some(dj)) => catch_transform(jt, dj, jpeg_data.clone(), op),
                    _ => TestResult::Skip {
                        notes: "jpegtran or djpeg not found".to_string(),
                    },
                };
                let result = apply_expected_reject(file_path, transform_op_name(op), result);
                let relative_path = file_path.strip_prefix(&cfg.corpus_dir).unwrap_or(file_path);
                expected_coverage.record(relative_path, transform_op_name(op), &result);
                print_row(file_str, transform_op_name(op), &result);
                transform_counts.record(&result);
            }
        }
    }

    // Summary
    println!();
    println!("=== CORPUS TEST SUMMARY ===");
    println!("Total files: {}", files.len());
    if cfg.run_decode {
        println!(
            "Decode:    {} pass, {} expected-reject, {} known-mismatch, {} fail, {} crash, {} skip",
            decode_counts.pass,
            decode_counts.expected_reject,
            decode_counts.known_mismatch,
            decode_counts.fail,
            decode_counts.crash,
            decode_counts.skip
        );
    }
    if cfg.run_encode {
        println!(
            "Encode:    {} pass, {} expected-reject, {} known-mismatch, {} fail, {} crash, {} skip",
            encode_counts.pass,
            encode_counts.expected_reject,
            encode_counts.known_mismatch,
            encode_counts.fail,
            encode_counts.crash,
            encode_counts.skip
        );
    }
    if cfg.run_transform {
        println!(
            "Transform: {} pass, {} expected-reject, {} known-mismatch, {} fail, {} crash, {} skip",
            transform_counts.pass,
            transform_counts.expected_reject,
            transform_counts.known_mismatch,
            transform_counts.fail,
            transform_counts.crash,
            transform_counts.skip
        );
    }

    let is_full_corpus_run = cfg.run_decode
        && cfg.run_encode
        && cfg.run_transform
        && ["generated", "fuzz_seeds", "fixtures"]
            .iter()
            .all(|bucket| cfg.corpus_dir.join(bucket).is_dir());
    let expected_coverage_error = expected_coverage
        .verify_full_corpus(is_full_corpus_run)
        .err();
    if let Some(error) = &expected_coverage_error {
        eprintln!("{error}");
    }

    let failed = expected_coverage_error.is_some()
        || (cfg.run_decode
            && (decode_counts.total() == 0 || decode_counts.has_unexpected_outcomes()))
        || (cfg.run_encode
            && (encode_counts.total() == 0 || encode_counts.has_unexpected_outcomes()))
        || (cfg.run_transform
            && (transform_counts.total() == 0 || transform_counts.has_unexpected_outcomes()));
    if failed {
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        apply_expected_reject, c_tool_path, collect_jpeg_files, operation_applies,
        required_expected_outcomes, run_decode_test, Counts, ExpectedCoverage, TestResult,
    };
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    struct TempTree(PathBuf);

    impl TempTree {
        fn new() -> Self {
            let counter = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "libjpeg_corpus_discovery_{}_{}",
                std::process::id(),
                counter
            ));
            std::fs::create_dir_all(&path).expect("create temp corpus");
            Self(path)
        }
    }

    impl Drop for TempTree {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).ok();
        }
    }

    #[test]
    fn discovery_includes_extensionless_soi_inputs() {
        let corpus = TempTree::new();
        std::fs::write(corpus.0.join("named.jpg"), b"fixture").expect("write named input");
        std::fs::write(corpus.0.join("seed"), b"\xff\xd8\xff\xd9")
            .expect("write extensionless input");
        std::fs::write(corpus.0.join("not-jpeg"), b"plain bytes").expect("write non-JPEG");

        let files = collect_jpeg_files(&corpus.0).expect("discover corpus");

        assert_eq!(
            files,
            vec![corpus.0.join("named.jpg"), corpus.0.join("seed")]
        );
    }

    #[test]
    fn discovery_fails_closed_for_missing_directory() {
        let missing = std::env::temp_dir().join(format!(
            "libjpeg_missing_corpus_{}_{}",
            std::process::id(),
            TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));

        let error = collect_jpeg_files(&missing).expect_err("missing corpus must fail");

        assert!(error.contains("read corpus directory"), "{error}");
        assert!(error.contains(&missing.display().to_string()), "{error}");
    }

    #[test]
    fn color_and_high_precision_fixtures_compare_in_rgb() {
        let Some(djpeg) = c_tool_path("djpeg") else {
            return;
        };
        let fixtures = [
            "tests/fixtures/cmyk_scanner/scanner_64x64.jpg",
            "tests/fixtures/real_world/libjpeg_testorig12_227x149_12bit.jpg",
            "tests/fixtures/real_world/pil_cmyk.jpg",
            "tests/fixtures/real_world/zune_cmyk_600x397_4comp.jpg",
            "tests/fixtures/real_world/zune_ycck_1318x611_4comp.jpg",
            "tests/fixtures/real_world/zune_ycck_progressive_383x740_4comp.jpg",
        ];

        for fixture in fixtures {
            let data = std::fs::read(fixture).expect("read fixture");
            assert!(
                matches!(run_decode_test(&djpeg, &data), TestResult::Pass { .. }),
                "fixture must compare in a common RGB output space: {fixture}"
            );
        }
    }

    #[test]
    fn corrupt_huffman_fixture_is_an_exact_expected_reject() {
        let raw = TestResult::Crash {
            notes: "Rust decompress error: corrupt data: invalid Huffman code".to_string(),
        };

        let classified = apply_expected_reject(
            PathBuf::from("tests/corpus/fixtures/fuzz_repro/corrupt_huffman_65x65_422.jpg")
                .as_path(),
            "decode",
            raw,
        );

        assert!(matches!(classified, TestResult::ExpectedReject { .. }));
    }

    #[test]
    fn expected_reject_does_not_hide_other_paths_or_reasons() {
        let wrong_path = apply_expected_reject(
            PathBuf::from("tests/corpus/fixtures/other.jpg").as_path(),
            "decode",
            TestResult::Crash {
                notes: "Rust decompress error: corrupt data: invalid Huffman code".to_string(),
            },
        );
        let wrong_reason = apply_expected_reject(
            PathBuf::from("tests/corpus/fixtures/fuzz_repro/corrupt_huffman_65x65_422.jpg")
                .as_path(),
            "decode",
            TestResult::Crash {
                notes: "different decoder failure".to_string(),
            },
        );

        assert!(matches!(wrong_path, TestResult::Crash { .. }));
        assert!(matches!(wrong_reason, TestResult::Crash { .. }));
    }

    #[test]
    fn corpus_counts_fail_closed_on_failures_crashes_and_skips() {
        let mut counts = Counts::default();
        counts.record(&TestResult::Pass { max_diff: 0 });
        counts.record(&TestResult::ExpectedReject {
            notes: "intentional".to_string(),
        });
        assert!(!counts.has_unexpected_outcomes());
        assert_eq!(counts.total(), 2);

        for unexpected in [
            TestResult::Fail {
                max_diff: 1,
                notes: "mismatch".to_string(),
            },
            TestResult::Crash {
                notes: "decoder error".to_string(),
            },
            TestResult::Skip {
                notes: "tool missing".to_string(),
            },
        ] {
            let mut counts = Counts::default();
            counts.record(&unexpected);
            assert!(counts.has_unexpected_outcomes());
        }
    }

    #[test]
    fn full_corpus_requires_every_exact_expected_outcome() {
        let required = required_expected_outcomes();
        let complete = ExpectedCoverage {
            observed: required.clone(),
        };
        assert!(complete.verify_full_corpus(true).is_ok());

        let mut missing = required;
        let removed = missing.iter().next().cloned().expect("required outcome");
        missing.remove(&removed);
        let incomplete = ExpectedCoverage { observed: missing };
        let error = incomplete
            .verify_full_corpus(true)
            .expect_err("missing exact outcome must fail full corpus gate");
        let removed_path = removed.split('\t').next().expect("outcome path");
        assert!(error.contains(removed_path), "{error}");
        assert!(incomplete.verify_full_corpus(false).is_ok());
    }

    #[test]
    fn transform_expected_rejects_require_exact_fixture_and_reason_class() {
        let known = apply_expected_reject(
            PathBuf::from("tests/corpus/fixtures/real_world/zune_mjpeg_huffman_1280x720.jpg")
                .as_path(),
            "transform_rotate90",
            TestResult::Crash {
                notes: "Rust transform error: corrupt data: missing DC Huffman table 0".to_string(),
            },
        );
        let wrong_operation = apply_expected_reject(
            PathBuf::from("tests/corpus/fixtures/real_world/zune_mjpeg_huffman_1280x720.jpg")
                .as_path(),
            "decode",
            TestResult::Crash {
                notes: "Rust transform error: corrupt data: missing DC Huffman table 0".to_string(),
            },
        );

        assert!(matches!(known, TestResult::ExpectedReject { .. }));
        assert!(matches!(wrong_operation, TestResult::Crash { .. }));
    }

    #[test]
    fn transform_pixel_mismatches_require_exact_path_operation_and_max_diff() {
        let path =
            PathBuf::from("tests/corpus/fixtures/fuzz_repro/arith_noninterleaved_16x16_444.jpg");
        let exact = apply_expected_reject(
            &path,
            "transform_rotate90",
            TestResult::Fail {
                max_diff: 157,
                notes: "pixel diff (Rust 658 bytes, C 684 bytes)".to_string(),
            },
        );
        let worsened = apply_expected_reject(
            &path,
            "transform_rotate90",
            TestResult::Fail {
                max_diff: 158,
                notes: "pixel diff (Rust 658 bytes, C 684 bytes)".to_string(),
            },
        );
        let wrong_operation = apply_expected_reject(
            &path,
            "decode",
            TestResult::Fail {
                max_diff: 157,
                notes: "pixel diff (Rust 658 bytes, C 684 bytes)".to_string(),
            },
        );

        assert!(matches!(
            exact,
            TestResult::KnownMismatch { max_diff: 157, .. }
        ));
        assert!(matches!(worsened, TestResult::Fail { max_diff: 158, .. }));
        assert!(matches!(
            wrong_operation,
            TestResult::Fail { max_diff: 157, .. }
        ));
    }

    #[test]
    fn tracked_p4_20_and_p4_21_outcomes_are_exactly_classified() {
        let p4_20 = apply_expected_reject(
            PathBuf::from("tests/corpus/fuzz_seeds/24fd23785278a9577686f501e17ee8164f8b977b")
                .as_path(),
            "decode",
            TestResult::Fail {
                max_diff: 34,
                notes: "pixel diff at (0,0)".to_string(),
            },
        );
        let p4_21 = apply_expected_reject(
            PathBuf::from(
                "tests/corpus/fuzz_seeds/crash-cf56f76b13a5eaa5a65b46a3503c0951f034d735",
            )
            .as_path(),
            "decode",
            TestResult::Crash {
                notes: "Rust decompress error: corrupt data: chroma upsample factor zero (a chroma component out-samples luma): cb=4x0 cr=4x1".to_string(),
            },
        );

        assert!(matches!(p4_20, TestResult::KnownMismatch { .. }));
        assert!(matches!(p4_21, TestResult::ExpectedReject { .. }));
    }

    #[test]
    fn decompress_fuzz_seeds_are_not_misused_as_encode_or_transform_corpora() {
        let fuzz_seed = PathBuf::from("tests/corpus/fuzz_seeds/seed");
        let real_fixture = PathBuf::from("tests/corpus/fixtures/real_world/camera.jpg");

        assert!(operation_applies(&fuzz_seed, "decode"));
        assert!(!operation_applies(&fuzz_seed, "encode"));
        assert!(!operation_applies(&fuzz_seed, "transform"));
        assert!(operation_applies(&real_fixture, "decode"));
        assert!(operation_applies(&real_fixture, "encode"));
        assert!(operation_applies(&real_fixture, "transform"));
    }
}

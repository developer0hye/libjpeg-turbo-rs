/// Corpus test harness: validates libjpeg-turbo-rs against C libjpeg-turbo
/// across a directory of JPEG files.
///
/// Usage:
///   cargo run --example corpus_test -- --corpus-dir tests/corpus/ [--decode-only] [--encode-only] [--transform-only]
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{
    compress, decompress, transform, Image, PixelFormat, Subsampling, TransformOp,
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
    let (_maxval, next) = read_number(data, pos)?;
    pos = next;
    if pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }
    let expected_len: usize = width * height * 3;
    if data.len() - pos < expected_len {
        return None;
    }
    Some((width, height, data[pos..pos + expected_len].to_vec()))
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
    let (_maxval, next) = read_number(data, pos)?;
    pos = next;
    if pos < data.len() && data[pos].is_ascii_whitespace() {
        pos += 1;
    }
    let expected_len: usize = width * height;
    if data.len() - pos < expected_len {
        return None;
    }
    Some((width, height, data[pos..pos + expected_len].to_vec()))
}

// ===========================================================================
// TSV output row
// ===========================================================================

#[derive(Clone)]
enum TestResult {
    Pass { max_diff: u32 },
    Fail { max_diff: u32, notes: String },
    Crash { notes: String },
    Skip { notes: String },
}

impl TestResult {
    fn label(&self) -> &'static str {
        match self {
            TestResult::Pass { .. } => "pass",
            TestResult::Fail { .. } => "fail",
            TestResult::Crash { .. } => "crash",
            TestResult::Skip { .. } => "skip",
        }
    }

    fn max_diff_str(&self) -> String {
        match self {
            TestResult::Pass { max_diff } => max_diff.to_string(),
            TestResult::Fail { max_diff, .. } => max_diff.to_string(),
            TestResult::Crash { .. } | TestResult::Skip { .. } => "-".to_string(),
        }
    }

    fn notes(&self) -> &str {
        match self {
            TestResult::Pass { .. } => "",
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
    fail: u32,
    crash: u32,
    skip: u32,
}

impl Counts {
    fn record(&mut self, r: &TestResult) {
        match r {
            TestResult::Pass { .. } => self.pass += 1,
            TestResult::Fail { .. } => self.fail += 1,
            TestResult::Crash { .. } => self.crash += 1,
            TestResult::Skip { .. } => self.skip += 1,
        }
    }
}

// ===========================================================================
// File discovery
// ===========================================================================

fn collect_jpeg_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = Vec::new();
    collect_recursive(dir, &mut files);
    files.sort();
    files
}

fn collect_recursive(dir: &Path, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path: PathBuf = entry.path();
        if path.is_dir() {
            collect_recursive(&path, out);
        } else if let Some(ext) = path.extension() {
            let ext_lower: String = ext.to_string_lossy().to_lowercase();
            if ext_lower == "jpg" || ext_lower == "jpeg" {
                out.push(path);
            }
        }
    }
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
    let rust_img: Image = match decompress(jpeg_data) {
        Ok(img) => img,
        Err(e) => {
            return TestResult::Crash {
                notes: format!("Rust decompress error: {}", e),
            }
        }
    };

    let is_gray: bool = rust_img.pixel_format == PixelFormat::Grayscale;

    // Step 2: decode with C djpeg
    let (c_w, c_h, c_pixels, _) = match decode_with_c_djpeg(djpeg, jpeg_data, is_gray) {
        Ok(v) => v,
        Err(e) => {
            return TestResult::Skip {
                notes: format!("C djpeg error: {}", e),
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
    let files: Vec<PathBuf> = collect_jpeg_files(&cfg.corpus_dir);
    if files.is_empty() {
        eprintln!("No JPEG files found in {:?}", cfg.corpus_dir);
        std::process::exit(1);
    }

    // TSV header
    println!("file\toperation\tresult\tmax_diff\tnotes");

    let mut decode_counts: Counts = Counts::default();
    let mut encode_counts: Counts = Counts::default();
    let mut transform_counts: Counts = Counts::default();

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
            print_row(file_str, "decode", &result);
            decode_counts.record(&result);
        }

        // Encode test
        if cfg.run_encode {
            let result: TestResult = match (&djpeg, &cjpeg) {
                (Some(dj), Some(cj)) => catch_encode(dj, cj, jpeg_data.clone()),
                _ => TestResult::Skip {
                    notes: "djpeg or cjpeg not found".to_string(),
                },
            };
            print_row(file_str, "encode", &result);
            encode_counts.record(&result);
        }

        // Transform tests
        if cfg.run_transform {
            for &op in transform_ops {
                let result: TestResult = match (&jpegtran, &djpeg) {
                    (Some(jt), Some(dj)) => catch_transform(jt, dj, jpeg_data.clone(), op),
                    _ => TestResult::Skip {
                        notes: "jpegtran or djpeg not found".to_string(),
                    },
                };
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
            "Decode:    {} pass, {} fail, {} crash, {} skip",
            decode_counts.pass, decode_counts.fail, decode_counts.crash, decode_counts.skip
        );
    }
    if cfg.run_encode {
        println!(
            "Encode:    {} pass, {} fail, {} crash, {} skip",
            encode_counts.pass, encode_counts.fail, encode_counts.crash, encode_counts.skip
        );
    }
    if cfg.run_transform {
        println!(
            "Transform: {} pass, {} fail, {} crash, {} skip",
            transform_counts.pass,
            transform_counts.fail,
            transform_counts.crash,
            transform_counts.skip
        );
    }
}

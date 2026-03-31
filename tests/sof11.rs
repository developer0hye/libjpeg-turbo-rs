use libjpeg_turbo_rs::{compress_lossless_arithmetic, decompress, Encoder, PixelFormat};

#[test]
fn sof11_grayscale_roundtrip() {
    let pixels: Vec<u8> = (0..=255).collect();
    let jpeg = Encoder::new(&pixels, 16, 16, PixelFormat::Grayscale)
        .lossless(true)
        .arithmetic(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels); // Lossless = exact
}

#[test]
fn sof11_contains_marker() {
    let pixels = vec![128u8; 64];
    let jpeg = Encoder::new(&pixels, 8, 8, PixelFormat::Grayscale)
        .lossless(true)
        .arithmetic(true)
        .encode()
        .unwrap();
    let has_sof11 = jpeg.windows(2).any(|w| w[0] == 0xFF && w[1] == 0xCB);
    assert!(has_sof11);
}

#[test]
fn sof11_gradient_roundtrip() {
    let (w, h) = (32, 32);
    let mut pixels = vec![0u8; w * h];
    for y in 0..h {
        for x in 0..w {
            pixels[y * w + x] = ((x * 7 + y * 3) % 256) as u8;
        }
    }
    let jpeg = Encoder::new(&pixels, w, h, PixelFormat::Grayscale)
        .lossless(true)
        .arithmetic(true)
        .encode()
        .unwrap();
    let img = decompress(&jpeg).unwrap();
    assert_eq!(img.data, pixels);
}

// ===========================================================================
// C cross-validation helpers
// ===========================================================================

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

fn djpeg_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/djpeg");
    if homebrew.exists() {
        return Some(homebrew);
    }
    Command::new("which")
        .arg("djpeg")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

fn cjpeg_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/cjpeg");
    if homebrew.exists() {
        return Some(homebrew);
    }
    Command::new("which")
        .arg("cjpeg")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

/// Check if cjpeg supports the `-arithmetic` flag.
fn cjpeg_supports_arithmetic(cjpeg: &Path) -> bool {
    let output = Command::new(cjpeg).arg("-help").output();
    match output {
        Ok(o) => {
            let text: String = String::from_utf8_lossy(&o.stderr).to_string()
                + &String::from_utf8_lossy(&o.stdout);
            text.contains("arithmetic")
        }
        Err(_) => false,
    }
}

/// Check if cjpeg supports the `-lossless` flag.
fn cjpeg_supports_lossless(cjpeg: &Path) -> bool {
    let output = Command::new(cjpeg).arg("-help").output();
    match output {
        Ok(o) => {
            let text: String = String::from_utf8_lossy(&o.stderr).to_string()
                + &String::from_utf8_lossy(&o.stdout);
            text.contains("lossless")
        }
        Err(_) => false,
    }
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_sof11_{}_{:04}_{}", pid, counter, name))
}

struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn new(name: &str) -> Self {
        Self {
            path: temp_path(name),
        }
    }
    fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        std::fs::remove_file(&self.path).ok();
    }
}

/// Write a PGM (P5 binary) file.
fn write_pgm(path: &Path, width: usize, height: usize, data: &[u8]) {
    use std::io::Write;
    assert_eq!(data.len(), width * height, "PGM data length mismatch");
    let mut file = std::fs::File::create(path).expect("failed to create PGM file");
    write!(file, "P5\n{} {}\n255\n", width, height).expect("failed to write PGM header");
    file.write_all(data).expect("failed to write PGM data");
}

/// Parse a binary PGM (P5) file and return `(width, height, data)`.
fn parse_pgm(path: &Path) -> (usize, usize, Vec<u8>) {
    let raw: Vec<u8> = std::fs::read(path).expect("failed to read PGM file");
    assert!(raw.len() > 3, "PGM too short");
    assert_eq!(&raw[0..2], b"P5", "not a P5 PGM");
    let mut idx: usize = 2;
    idx = skip_whitespace_and_comments(&raw, idx);
    let (width, next) = read_ascii_number(&raw, idx);
    idx = skip_whitespace_and_comments(&raw, next);
    let (height, next) = read_ascii_number(&raw, idx);
    idx = skip_whitespace_and_comments(&raw, next);
    let (_maxval, next) = read_ascii_number(&raw, idx);
    // Single whitespace byte after maxval before binary data
    idx = next + 1;
    let data: Vec<u8> = raw[idx..].to_vec();
    assert_eq!(data.len(), width * height, "PGM pixel data length mismatch");
    (width, height, data)
}

fn skip_whitespace_and_comments(data: &[u8], mut idx: usize) -> usize {
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

fn read_ascii_number(data: &[u8], idx: usize) -> (usize, usize) {
    let mut end: usize = idx;
    while end < data.len() && data[end].is_ascii_digit() {
        end += 1;
    }
    let val: usize = std::str::from_utf8(&data[idx..end])
        .unwrap()
        .parse()
        .unwrap();
    (val, end)
}

/// Generate a deterministic grayscale gradient pattern.
fn generate_grayscale_gradient(w: usize, h: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            let v: u8 = (((x * 7 + y * 13) * 255) / (w * 7 + h * 13).max(1)) as u8;
            pixels.push(v);
        }
    }
    pixels
}

// ===========================================================================
// Test 1: Rust SOF11 encode -> C djpeg decode (exact match)
// ===========================================================================

#[test]
fn c_cross_validation_sof11_rust_encode_c_decode() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (16, 16);
    let pixels: Vec<u8> = generate_grayscale_gradient(w, h);

    // Encode with Rust using lossless arithmetic (SOF11)
    let jpeg: Vec<u8> = compress_lossless_arithmetic(&pixels, w, h, PixelFormat::Grayscale, 1, 0)
        .expect("Rust SOF11 encode failed");

    // Verify SOF11 marker is present
    let has_sof11: bool = jpeg
        .windows(2)
        .any(|pair| pair[0] == 0xFF && pair[1] == 0xCB);
    assert!(
        has_sof11,
        "encoded JPEG should contain SOF11 marker (0xFFCB)"
    );

    let tmp_jpg: TempFile = TempFile::new("sof11_rust_enc.jpg");
    let tmp_out: TempFile = TempFile::new("sof11_rust_enc.pgm");
    std::fs::write(tmp_jpg.path(), &jpeg).expect("write temp JPEG");

    // Decode with C djpeg
    let output = Command::new(&djpeg)
        .arg("-pnm")
        .arg("-outfile")
        .arg(tmp_out.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    if !output.status.success() {
        let stderr: String = String::from_utf8_lossy(&output.stderr).to_string();
        if stderr.contains("arithmetic")
            || stderr.contains("Unsupported JPEG process")
            || stderr.contains("SOF type")
            || stderr.contains("omitted at compile time")
            || stderr.contains("Requested feature")
        {
            eprintln!("SKIP: djpeg does not support arithmetic lossless (SOF11)");
            return;
        }
        panic!("djpeg failed on Rust SOF11 JPEG: {}", stderr);
    }

    let (dw, dh, c_pixels) = parse_pgm(tmp_out.path());
    assert_eq!(dw, w, "width mismatch");
    assert_eq!(dh, h, "height mismatch");
    // SOF11 = lossless arithmetic, must be bit-exact
    assert_eq!(
        c_pixels, pixels,
        "SOF11 lossless arithmetic: Rust encode -> C decode must be pixel-exact"
    );
}

// ===========================================================================
// Test 2: C cjpeg SOF11 encode -> Rust decode (exact match)
// ===========================================================================

#[test]
fn c_cross_validation_sof11_c_encode_rust_decode() {
    let cjpeg: PathBuf = match cjpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpeg not found");
            return;
        }
    };

    if !cjpeg_supports_arithmetic(&cjpeg) {
        eprintln!("SKIP: cjpeg does not support -arithmetic");
        return;
    }
    if !cjpeg_supports_lossless(&cjpeg) {
        eprintln!("SKIP: cjpeg does not support -lossless");
        return;
    }

    let (w, h): (usize, usize) = (16, 16);
    let pixels: Vec<u8> = generate_grayscale_gradient(w, h);

    // Write test data as PGM for cjpeg input
    let tmp_pgm: TempFile = TempFile::new("sof11_c_enc_input.pgm");
    write_pgm(tmp_pgm.path(), w, h, &pixels);

    // Encode with C cjpeg using arithmetic lossless (SOF11)
    let tmp_jpg: TempFile = TempFile::new("sof11_c_enc.jpg");
    let output = Command::new(&cjpeg)
        .arg("-arithmetic")
        .arg("-lossless")
        .arg("1,0")
        .arg("-outfile")
        .arg(tmp_jpg.path())
        .arg(tmp_pgm.path())
        .output()
        .expect("failed to run cjpeg");

    if !output.status.success() {
        let stderr: String = String::from_utf8_lossy(&output.stderr).to_string();
        if stderr.contains("arithmetic")
            || stderr.contains("not compiled")
            || stderr.contains("Unsupported")
            || stderr.contains("omitted at compile time")
            || stderr.contains("Requested feature")
        {
            eprintln!("SKIP: cjpeg does not support arithmetic lossless encoding");
            return;
        }
        panic!("cjpeg -arithmetic -lossless 1,0 failed: {}", stderr);
    }

    // Verify the C-encoded file contains SOF11 marker
    let jpeg_data: Vec<u8> = std::fs::read(tmp_jpg.path()).expect("read cjpeg output");
    let has_sof11: bool = jpeg_data
        .windows(2)
        .any(|pair| pair[0] == 0xFF && pair[1] == 0xCB);
    assert!(
        has_sof11,
        "C cjpeg -arithmetic -lossless should produce SOF11 marker"
    );

    // Decode with Rust
    let img = decompress(&jpeg_data).expect("Rust decompress of C SOF11 JPEG failed");

    assert_eq!(img.width, w, "width mismatch");
    assert_eq!(img.height, h, "height mismatch");
    // SOF11 = lossless arithmetic, must be bit-exact
    assert_eq!(
        img.data, pixels,
        "SOF11 lossless arithmetic: C encode -> Rust decode must be pixel-exact"
    );
}

// ===========================================================================
// Test 3: Full roundtrip interoperability (Rust->C->C->Rust)
// ===========================================================================

#[test]
fn c_cross_validation_sof11_roundtrip_exact() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    let cjpeg: PathBuf = match cjpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpeg not found");
            return;
        }
    };

    if !cjpeg_supports_arithmetic(&cjpeg) {
        eprintln!("SKIP: cjpeg does not support -arithmetic");
        return;
    }
    if !cjpeg_supports_lossless(&cjpeg) {
        eprintln!("SKIP: cjpeg does not support -lossless");
        return;
    }

    let (w, h): (usize, usize) = (16, 16);
    let pixels: Vec<u8> = generate_grayscale_gradient(w, h);

    // Step 1: Rust SOF11 encode
    let jpeg_rust: Vec<u8> =
        compress_lossless_arithmetic(&pixels, w, h, PixelFormat::Grayscale, 1, 0)
            .expect("Rust SOF11 encode failed");

    // Step 2: C djpeg decode of Rust-encoded SOF11
    let tmp_jpg1: TempFile = TempFile::new("sof11_rt_rust.jpg");
    let tmp_pgm1: TempFile = TempFile::new("sof11_rt_step1.pgm");
    std::fs::write(tmp_jpg1.path(), &jpeg_rust).expect("write temp JPEG");

    let output = Command::new(&djpeg)
        .arg("-pnm")
        .arg("-outfile")
        .arg(tmp_pgm1.path())
        .arg(tmp_jpg1.path())
        .output()
        .expect("failed to run djpeg");

    if !output.status.success() {
        let stderr: String = String::from_utf8_lossy(&output.stderr).to_string();
        if stderr.contains("arithmetic")
            || stderr.contains("Unsupported JPEG process")
            || stderr.contains("SOF type")
            || stderr.contains("omitted at compile time")
            || stderr.contains("Requested feature")
        {
            eprintln!("SKIP: djpeg does not support arithmetic lossless (SOF11)");
            return;
        }
        panic!("djpeg failed on Rust SOF11 JPEG: {}", stderr);
    }

    let (dw1, dh1, c_decoded) = parse_pgm(tmp_pgm1.path());
    assert_eq!(dw1, w, "step1 width mismatch");
    assert_eq!(dh1, h, "step1 height mismatch");
    assert_eq!(
        c_decoded, pixels,
        "step1: Rust SOF11 encode -> C djpeg decode must be pixel-exact"
    );

    // Step 3: C cjpeg re-encode the decoded data as SOF11
    let tmp_pgm2: TempFile = TempFile::new("sof11_rt_c_input.pgm");
    write_pgm(tmp_pgm2.path(), w, h, &c_decoded);

    let tmp_jpg2: TempFile = TempFile::new("sof11_rt_c_enc.jpg");
    let output = Command::new(&cjpeg)
        .arg("-arithmetic")
        .arg("-lossless")
        .arg("1,0")
        .arg("-outfile")
        .arg(tmp_jpg2.path())
        .arg(tmp_pgm2.path())
        .output()
        .expect("failed to run cjpeg");

    if !output.status.success() {
        let stderr: String = String::from_utf8_lossy(&output.stderr).to_string();
        if stderr.contains("arithmetic")
            || stderr.contains("not compiled")
            || stderr.contains("Unsupported")
            || stderr.contains("omitted at compile time")
            || stderr.contains("Requested feature")
        {
            eprintln!("SKIP: cjpeg does not support arithmetic lossless encoding");
            return;
        }
        panic!("cjpeg -arithmetic -lossless 1,0 failed: {}", stderr);
    }

    // Step 4: Rust decode of C-encoded SOF11
    let jpeg_c: Vec<u8> = std::fs::read(tmp_jpg2.path()).expect("read cjpeg output");
    let img = decompress(&jpeg_c).expect("Rust decompress of C SOF11 JPEG failed");

    assert_eq!(img.width, w, "step2 width mismatch");
    assert_eq!(img.height, h, "step2 height mismatch");
    // Full roundtrip: original -> Rust enc -> C dec -> C enc -> Rust dec must be exact
    assert_eq!(
        img.data, pixels,
        "SOF11 full roundtrip (Rust->C->C->Rust) must be pixel-exact"
    );
}

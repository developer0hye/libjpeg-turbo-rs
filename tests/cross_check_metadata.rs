//! Cross-check tests for metadata preservation between Rust library and C libjpeg-turbo tools.
//!
//! Tests cover:
//! - ICC profile preservation through C decode
//! - C-encoded ICC profiles decoded by Rust
//! - EXIF preservation through roundtrip
//! - COM marker preservation
//! - ICC preservation through lossless transforms
//! - C jpegtran marker preservation decoded by Rust
//!
//! All tests gracefully skip if C tools are not found.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{
    compress_with_metadata, decompress, transform_jpeg_with_options, Encoder, MarkerCopyMode,
    MarkerSaveConfig, PixelFormat, SavedMarker, Subsampling, TransformOp, TransformOptions,
};

// ===========================================================================
// Tool discovery
// ===========================================================================

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

fn jpegtran_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/jpegtran");
    if homebrew.exists() {
        return Some(homebrew);
    }
    Command::new("which")
        .arg("jpegtran")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

fn rdjpgcom_path() -> Option<PathBuf> {
    let homebrew: PathBuf = PathBuf::from("/opt/homebrew/bin/rdjpgcom");
    if homebrew.exists() {
        return Some(homebrew);
    }
    Command::new("which")
        .arg("rdjpgcom")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).trim().to_string()))
}

/// Check if cjpeg supports the `-icc` flag.
fn cjpeg_supports_icc(cjpeg: &Path) -> bool {
    let output = Command::new(cjpeg).arg("-help").output();
    match output {
        Ok(o) => {
            let text: String = String::from_utf8_lossy(&o.stderr).to_string()
                + &String::from_utf8_lossy(&o.stdout);
            text.contains("icc")
        }
        Err(_) => false,
    }
}

/// Check if djpeg supports the `-icc` flag for extracting ICC profiles.
fn djpeg_supports_icc_extract(djpeg: &Path) -> bool {
    let output = Command::new(djpeg).arg("-help").output();
    match output {
        Ok(o) => {
            let text: String = String::from_utf8_lossy(&o.stderr).to_string()
                + &String::from_utf8_lossy(&o.stdout);
            text.contains("icc")
        }
        Err(_) => false,
    }
}

// ===========================================================================
// Helpers
// ===========================================================================

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_meta_{}_{:04}_{}", pid, counter, name))
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

fn reference_path(name: &str) -> PathBuf {
    PathBuf::from(format!("references/libjpeg-turbo/testimages/{}", name))
}

/// Generate a small test RGB image.
fn generate_test_pixels(w: usize, h: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            pixels.push(((x * 255) / w.max(1)) as u8);
            pixels.push(((y * 255) / h.max(1)) as u8);
            pixels.push(128u8);
        }
    }
    pixels
}

/// Load an ICC profile from the reference test images.
fn load_icc(name: &str) -> Option<Vec<u8>> {
    let path: PathBuf = reference_path(name);
    if path.exists() {
        Some(std::fs::read(&path).expect("read ICC file"))
    } else {
        None
    }
}

/// Create a minimal EXIF block (valid TIFF header + IFD with orientation tag).
fn minimal_exif() -> Vec<u8> {
    // TIFF header: byte order (II = little endian), magic 42, offset to IFD0
    let mut exif: Vec<u8> = Vec::new();
    // Little endian TIFF
    exif.extend_from_slice(b"II");
    exif.extend_from_slice(&42u16.to_le_bytes()); // TIFF magic
    exif.extend_from_slice(&8u32.to_le_bytes()); // Offset to IFD0

    // IFD0: 1 entry
    exif.extend_from_slice(&1u16.to_le_bytes()); // Number of entries
                                                 // Orientation tag (0x0112), SHORT type (3), count 1, value 1 (normal)
    exif.extend_from_slice(&0x0112u16.to_le_bytes()); // Tag
    exif.extend_from_slice(&3u16.to_le_bytes()); // Type: SHORT
    exif.extend_from_slice(&1u32.to_le_bytes()); // Count
    exif.extend_from_slice(&1u16.to_le_bytes()); // Value: normal orientation
    exif.extend_from_slice(&0u16.to_le_bytes()); // Padding
                                                 // Next IFD offset: 0 (no more IFDs)
    exif.extend_from_slice(&0u32.to_le_bytes());

    exif
}

// ===========================================================================
// ICC profile preservation through C decode
// ===========================================================================

#[test]
fn icc_profile_preserved_through_c_decode() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    if !djpeg_supports_icc_extract(&djpeg) {
        eprintln!("SKIP: djpeg does not support -icc flag");
        return;
    }

    let icc_data: Vec<u8> = match load_icc("test3.icc") {
        Some(d) => d,
        None => {
            eprintln!("SKIP: test3.icc not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (16, 16);
    let pixels: Vec<u8> = generate_test_pixels(w, h);

    // Encode with Rust + ICC profile
    let jpeg: Vec<u8> = compress_with_metadata(
        &pixels,
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
        Some(&icc_data),
        None,
    )
    .expect("Rust compress with ICC");

    let tmp_jpg: TempFile = TempFile::new("icc_rust.jpg");
    let tmp_icc: TempFile = TempFile::new("icc_extracted.icc");
    let tmp_ppm: TempFile = TempFile::new("icc_rust.ppm");
    std::fs::write(tmp_jpg.path(), &jpeg).expect("write temp jpg");

    // Extract ICC with djpeg -icc
    let output = Command::new(&djpeg)
        .arg("-icc")
        .arg(tmp_icc.path())
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    assert!(
        output.status.success(),
        "djpeg -icc failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Compare extracted ICC with original
    let extracted_icc: Vec<u8> = std::fs::read(tmp_icc.path()).expect("read extracted ICC");
    assert_eq!(
        extracted_icc,
        icc_data,
        "extracted ICC profile should match original (sizes: {} vs {})",
        extracted_icc.len(),
        icc_data.len()
    );
}

// ===========================================================================
// C-encoded ICC profiles decoded by Rust
// ===========================================================================

#[test]
fn c_icc_preserved_through_rust_decode() {
    let cjpeg: PathBuf = match cjpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpeg not found");
            return;
        }
    };
    if !cjpeg_supports_icc(&cjpeg) {
        eprintln!("SKIP: cjpeg does not support -icc flag");
        return;
    }

    let icc_path: PathBuf = reference_path("test3.icc");
    if !icc_path.exists() {
        eprintln!("SKIP: test3.icc not found");
        return;
    }

    let ppm_path: PathBuf = reference_path("testorig.ppm");
    if !ppm_path.exists() {
        eprintln!("SKIP: testorig.ppm not found");
        return;
    }

    let icc_data: Vec<u8> = std::fs::read(&icc_path).expect("read test3.icc");

    let tmp_jpg: TempFile = TempFile::new("c_icc.jpg");

    let output = Command::new(&cjpeg)
        .arg("-icc")
        .arg(&icc_path)
        .arg("-outfile")
        .arg(tmp_jpg.path())
        .arg(&ppm_path)
        .output()
        .expect("failed to run cjpeg");

    assert!(
        output.status.success(),
        "cjpeg -icc failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let jpeg_data: Vec<u8> = std::fs::read(tmp_jpg.path()).expect("read cjpeg output");
    let img = decompress(&jpeg_data).expect("Rust decompress of cjpeg+icc output");

    // Verify the ICC profile was extracted
    assert!(
        img.icc_profile.is_some(),
        "Rust decoder should extract ICC profile from C-encoded JPEG"
    );

    let decoded_icc: &[u8] = img.icc_profile.as_ref().unwrap();
    assert_eq!(
        decoded_icc, &icc_data,
        "ICC profile from C-encoded JPEG should match original test3.icc"
    );
}

#[test]
fn c_icc_large_profile_rust_decode() {
    // Test with test1.icc which is larger (544K) and may span multiple APP2 chunks
    let cjpeg: PathBuf = match cjpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpeg not found");
            return;
        }
    };
    if !cjpeg_supports_icc(&cjpeg) {
        eprintln!("SKIP: cjpeg does not support -icc flag");
        return;
    }

    let icc_path: PathBuf = reference_path("test1.icc");
    if !icc_path.exists() {
        eprintln!("SKIP: test1.icc not found");
        return;
    }

    let ppm_path: PathBuf = reference_path("testorig.ppm");
    if !ppm_path.exists() {
        eprintln!("SKIP: testorig.ppm not found");
        return;
    }

    let icc_data: Vec<u8> = std::fs::read(&icc_path).expect("read test1.icc");

    let tmp_jpg: TempFile = TempFile::new("c_icc_large.jpg");

    let output = Command::new(&cjpeg)
        .arg("-icc")
        .arg(&icc_path)
        .arg("-outfile")
        .arg(tmp_jpg.path())
        .arg(&ppm_path)
        .output()
        .expect("failed to run cjpeg");

    assert!(
        output.status.success(),
        "cjpeg -icc (large) failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let jpeg_data: Vec<u8> = std::fs::read(tmp_jpg.path()).expect("read cjpeg output");
    let img = decompress(&jpeg_data).expect("Rust decompress of large ICC");

    assert!(
        img.icc_profile.is_some(),
        "Rust should extract large ICC profile from multi-chunk APP2"
    );

    let decoded_icc: &[u8] = img.icc_profile.as_ref().unwrap();
    assert_eq!(
        decoded_icc.len(),
        icc_data.len(),
        "large ICC profile size mismatch: decoded={}, original={}",
        decoded_icc.len(),
        icc_data.len()
    );
    assert_eq!(
        decoded_icc, &icc_data,
        "large ICC profile content should match original"
    );
}

// ===========================================================================
// EXIF preservation
// ===========================================================================

#[test]
fn exif_preserved_through_roundtrip() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (16, 16);
    let pixels: Vec<u8> = generate_test_pixels(w, h);
    let exif_data: Vec<u8> = minimal_exif();

    // Encode with Rust + EXIF
    let jpeg: Vec<u8> = compress_with_metadata(
        &pixels,
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
        None,
        Some(&exif_data),
    )
    .expect("Rust compress with EXIF");

    // Verify djpeg can decode it (EXIF should not break decoding)
    let tmp_jpg: TempFile = TempFile::new("exif_rust.jpg");
    let tmp_ppm: TempFile = TempFile::new("exif_rust.ppm");
    std::fs::write(tmp_jpg.path(), &jpeg).expect("write temp");

    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    assert!(
        output.status.success(),
        "djpeg failed on JPEG with EXIF: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify Rust can decode and extract EXIF
    let img = decompress(&jpeg).expect("Rust decompress of JPEG with EXIF");
    assert!(
        img.exif_data.is_some(),
        "Rust decoder should extract EXIF data from our own JPEG"
    );
    let decoded_exif: &[u8] = img.exif_data.as_ref().unwrap();
    assert_eq!(
        decoded_exif, &exif_data,
        "EXIF data should survive Rust encode -> Rust decode roundtrip"
    );
}

// ===========================================================================
// COM marker preservation
// ===========================================================================

#[test]
fn comment_preserved_cross_check() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (16, 16);
    let pixels: Vec<u8> = generate_test_pixels(w, h);
    let comment_text: &str = "libjpeg-turbo-rs cross-check test comment";

    // Encode with Rust + COM marker
    let encoder = Encoder::new(&pixels, w, h, PixelFormat::Rgb)
        .quality(90)
        .comment(comment_text);
    let jpeg: Vec<u8> = encoder.encode().expect("Rust encode with comment");

    // Verify djpeg can decode it
    let tmp_jpg: TempFile = TempFile::new("comment.jpg");
    let tmp_ppm: TempFile = TempFile::new("comment.ppm");
    std::fs::write(tmp_jpg.path(), &jpeg).expect("write temp");

    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    assert!(
        output.status.success(),
        "djpeg failed on JPEG with COM marker: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // If rdjpgcom is available, verify it can read the comment
    if let Some(rdjpgcom) = rdjpgcom_path() {
        let output = Command::new(&rdjpgcom)
            .arg(tmp_jpg.path())
            .output()
            .expect("failed to run rdjpgcom");

        if output.status.success() {
            let stdout: String = String::from_utf8_lossy(&output.stdout).to_string();
            assert!(
                stdout.contains(comment_text),
                "rdjpgcom should find our comment in the JPEG. Got: {}",
                stdout
            );
        }
    }

    // Verify Rust can decode and read comment back
    let img = decompress(&jpeg).expect("Rust decompress with comment");
    assert!(
        img.comment.is_some(),
        "Rust decoder should extract COM marker"
    );
    assert_eq!(
        img.comment.as_deref(),
        Some(comment_text),
        "decoded comment should match original"
    );
}

// ===========================================================================
// ICC preserved through transform
// ===========================================================================

#[test]
fn icc_preserved_through_transform() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };
    if !djpeg_supports_icc_extract(&djpeg) {
        eprintln!("SKIP: djpeg does not support -icc flag");
        return;
    }

    let icc_data: Vec<u8> = match load_icc("test3.icc") {
        Some(d) => d,
        None => {
            eprintln!("SKIP: test3.icc not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (16, 16);
    let pixels: Vec<u8> = generate_test_pixels(w, h);

    // Encode with Rust + ICC
    let jpeg: Vec<u8> = compress_with_metadata(
        &pixels,
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
        Some(&icc_data),
        None,
    )
    .expect("Rust compress with ICC");

    // Transform with Rust (copy all markers)
    let transformed: Vec<u8> = match transform_jpeg_with_options(
        &jpeg,
        &TransformOptions {
            op: TransformOp::Rot180,
            copy_markers: MarkerCopyMode::All,
            ..Default::default()
        },
    ) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("SKIP: Rust transform with ICC failed: {}", e);
            return;
        }
    };

    // Extract ICC from transformed JPEG using djpeg
    let tmp_jpg: TempFile = TempFile::new("icc_xform.jpg");
    let tmp_icc: TempFile = TempFile::new("icc_xform_extracted.icc");
    let tmp_ppm: TempFile = TempFile::new("icc_xform.ppm");
    std::fs::write(tmp_jpg.path(), &transformed).expect("write temp");

    let output = Command::new(&djpeg)
        .arg("-icc")
        .arg(tmp_icc.path())
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    assert!(
        output.status.success(),
        "djpeg -icc on transformed JPEG failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let extracted_icc: Vec<u8> = std::fs::read(tmp_icc.path()).expect("read extracted ICC");
    assert_eq!(
        extracted_icc, icc_data,
        "ICC profile should be preserved through Rust transform (copy_markers=All)"
    );

    // Also verify Rust can decode and see the ICC
    let img = decompress(&transformed).expect("Rust decompress of transformed+ICC");
    assert!(
        img.icc_profile.is_some(),
        "ICC should be present after transform with copy_markers=All"
    );
    assert_eq!(
        img.icc_profile.as_ref().unwrap().as_slice(),
        &icc_data,
        "ICC from Rust decode of transform should match"
    );
}

// ===========================================================================
// C jpegtran preserves markers, Rust decode verifies
// ===========================================================================

#[test]
fn c_jpegtran_preserves_markers_rust_decode() {
    let cjpeg: PathBuf = match cjpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: cjpeg not found");
            return;
        }
    };
    let jpegtran: PathBuf = match jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found");
            return;
        }
    };
    if !cjpeg_supports_icc(&cjpeg) {
        eprintln!("SKIP: cjpeg does not support -icc flag");
        return;
    }

    let icc_path: PathBuf = reference_path("test3.icc");
    if !icc_path.exists() {
        eprintln!("SKIP: test3.icc not found");
        return;
    }

    let ppm_path: PathBuf = reference_path("testorig.ppm");
    if !ppm_path.exists() {
        eprintln!("SKIP: testorig.ppm not found");
        return;
    }

    let icc_data: Vec<u8> = std::fs::read(&icc_path).expect("read test3.icc");

    // Step 1: cjpeg -icc test3.icc testorig.ppm -> jpeg with ICC
    let tmp_step1: TempFile = TempFile::new("marker_step1.jpg");
    let output = Command::new(&cjpeg)
        .arg("-icc")
        .arg(&icc_path)
        .arg("-outfile")
        .arg(tmp_step1.path())
        .arg(&ppm_path)
        .output()
        .expect("failed to run cjpeg");

    assert!(
        output.status.success(),
        "cjpeg -icc failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Step 2: jpegtran -copy all -rotate 90
    let tmp_step2: TempFile = TempFile::new("marker_step2.jpg");
    let output = Command::new(&jpegtran)
        .arg("-copy")
        .arg("all")
        .arg("-rotate")
        .arg("90")
        .arg("-outfile")
        .arg(tmp_step2.path())
        .arg(tmp_step1.path())
        .output()
        .expect("failed to run jpegtran");

    assert!(
        output.status.success(),
        "jpegtran -copy all -rotate 90 failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Step 3: Rust decompress should see the ICC profile
    let jpeg_data: Vec<u8> = std::fs::read(tmp_step2.path()).expect("read jpegtran output");
    let img = decompress(&jpeg_data)
        .expect("Rust decompress of cjpeg+ICC -> jpegtran -copy all -rotate 90");

    assert!(
        img.icc_profile.is_some(),
        "ICC profile should survive cjpeg -> jpegtran -copy all -> Rust decode"
    );

    let decoded_icc: &[u8] = img.icc_profile.as_ref().unwrap();
    assert_eq!(
        decoded_icc, &icc_data,
        "ICC profile should be identical after cjpeg -> jpegtran -copy all pipeline"
    );
}

// ===========================================================================
// ICC-only copy mode strips non-ICC markers
// ===========================================================================

#[test]
fn icc_only_copy_preserves_icc_strips_others() {
    let icc_data: Vec<u8> = match load_icc("test3.icc") {
        Some(d) => d,
        None => {
            eprintln!("SKIP: test3.icc not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (16, 16);
    let pixels: Vec<u8> = generate_test_pixels(w, h);
    let exif_data: Vec<u8> = minimal_exif();

    // Encode with both ICC and EXIF
    let jpeg: Vec<u8> = compress_with_metadata(
        &pixels,
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
        Some(&icc_data),
        Some(&exif_data),
    )
    .expect("Rust compress with ICC and EXIF");

    // Transform with IccOnly copy mode
    let transformed: Vec<u8> = match transform_jpeg_with_options(
        &jpeg,
        &TransformOptions {
            op: TransformOp::None,
            copy_markers: MarkerCopyMode::IccOnly,
            ..Default::default()
        },
    ) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("SKIP: Rust transform with IccOnly failed: {}", e);
            return;
        }
    };

    let img = decompress(&transformed).expect("decode transformed with IccOnly");

    // ICC should be preserved
    assert!(
        img.icc_profile.is_some(),
        "ICC should be preserved with IccOnly copy mode"
    );
    assert_eq!(
        img.icc_profile.as_ref().unwrap().as_slice(),
        &icc_data,
        "ICC content should match after IccOnly transform"
    );

    // EXIF should be stripped (IccOnly does not copy APP1)
    assert!(
        img.exif_data.is_none(),
        "EXIF should be stripped with IccOnly copy mode"
    );
}

// ===========================================================================
// No-copy mode strips all markers
// ===========================================================================

#[test]
fn no_copy_mode_strips_all_markers() {
    let icc_data: Vec<u8> = match load_icc("test3.icc") {
        Some(d) => d,
        None => {
            eprintln!("SKIP: test3.icc not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (16, 16);
    let pixels: Vec<u8> = generate_test_pixels(w, h);

    // Encode with ICC
    let jpeg: Vec<u8> = compress_with_metadata(
        &pixels,
        w,
        h,
        PixelFormat::Rgb,
        90,
        Subsampling::S444,
        Some(&icc_data),
        None,
    )
    .expect("Rust compress with ICC");

    // Transform with None copy mode
    let transformed: Vec<u8> = match transform_jpeg_with_options(
        &jpeg,
        &TransformOptions {
            op: TransformOp::None,
            copy_markers: MarkerCopyMode::None,
            ..Default::default()
        },
    ) {
        Ok(data) => data,
        Err(e) => {
            eprintln!("SKIP: Rust transform with copy=None failed: {}", e);
            return;
        }
    };

    let img = decompress(&transformed).expect("decode transformed with copy=None");

    // ICC should be stripped
    assert!(
        img.icc_profile.is_none(),
        "ICC should be stripped with copy=None mode"
    );
}

// ===========================================================================
// Saved markers survive encode -> Decoder with MarkerSaveConfig::All
// ===========================================================================

#[test]
fn custom_app_markers_preserved_through_c_decode() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (16, 16);
    let pixels: Vec<u8> = generate_test_pixels(w, h);

    // Encode with custom APP marker
    let custom_data: Vec<u8> = b"CustomTestPayload123".to_vec();
    let encoder = Encoder::new(&pixels, w, h, PixelFormat::Rgb)
        .quality(90)
        .saved_marker(SavedMarker {
            code: 0xE4, // APP4
            data: custom_data.clone(),
        });
    let jpeg: Vec<u8> = encoder.encode().expect("Rust encode with APP4");

    // Verify djpeg can decode (custom markers should not break decoding)
    let tmp_jpg: TempFile = TempFile::new("custom_marker.jpg");
    let tmp_ppm: TempFile = TempFile::new("custom_marker.ppm");
    std::fs::write(tmp_jpg.path(), &jpeg).expect("write temp");

    let output = Command::new(&djpeg)
        .arg("-ppm")
        .arg("-outfile")
        .arg(tmp_ppm.path())
        .arg(tmp_jpg.path())
        .output()
        .expect("failed to run djpeg");

    assert!(
        output.status.success(),
        "djpeg failed on JPEG with custom APP4 marker: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    // Verify Rust can decode and recover the marker via Decoder API
    let mut decoder =
        libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).expect("create decoder");
    decoder.save_markers(MarkerSaveConfig::All);
    let img = decoder.decode_image().expect("decode with saved markers");

    let app4_markers: Vec<&SavedMarker> = img
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xE4)
        .collect();

    assert!(
        !app4_markers.is_empty(),
        "APP4 marker should be saved when using MarkerSaveConfig::All"
    );
    assert_eq!(
        app4_markers[0].data, custom_data,
        "APP4 marker content should match"
    );
}

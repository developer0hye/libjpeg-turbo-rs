use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{
    compress, decompress, Encoder, MarkerCopyMode, PixelFormat, SavedMarker, Subsampling,
    TransformOp, TransformOptions,
};

/// Helper: create a small test JPEG with custom markers embedded.
fn make_jpeg_with_markers() -> Vec<u8> {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let mut encoder = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb);
    encoder = encoder.quality(75);
    encoder = encoder.saved_marker(SavedMarker {
        code: 0xE3,
        data: b"CustomAPP3Data".to_vec(),
    });
    encoder = encoder.saved_marker(SavedMarker {
        code: 0xE5,
        data: b"APP5-payload".to_vec(),
    });
    encoder = encoder.saved_marker(SavedMarker {
        code: 0xFE,
        data: b"saved-comment".to_vec(),
    });
    encoder.encode().unwrap()
}

#[allow(dead_code)]
fn make_basic_jpeg() -> Vec<u8> {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    compress(&pixels, 16, 16, PixelFormat::Rgb, 75, Subsampling::S444).unwrap()
}

#[test]
fn roundtrip_saved_markers_through_encode_decode() {
    let jpeg: Vec<u8> = make_jpeg_with_markers();
    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder.save_markers(libjpeg_turbo_rs::MarkerSaveConfig::All);
    let image = decoder.decode_image().unwrap();

    let app3_markers: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xE3)
        .collect();
    assert!(!app3_markers.is_empty(), "expected APP3 marker to be saved");
    assert_eq!(app3_markers[0].data, b"CustomAPP3Data");

    let app5_markers: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xE5)
        .collect();
    assert!(!app5_markers.is_empty(), "expected APP5 marker to be saved");
    assert_eq!(app5_markers[0].data, b"APP5-payload");
}

#[test]
fn roundtrip_com_marker_via_saved_markers() {
    let jpeg: Vec<u8> = make_jpeg_with_markers();
    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder.save_markers(libjpeg_turbo_rs::MarkerSaveConfig::All);
    let image = decoder.decode_image().unwrap();

    let com_markers: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xFE)
        .collect();
    assert!(
        !com_markers.is_empty(),
        "expected COM marker in saved_markers"
    );
    assert_eq!(com_markers[0].data, b"saved-comment");
}

#[test]
fn default_decoder_does_not_save_unknown_app_markers() {
    let jpeg: Vec<u8> = make_jpeg_with_markers();
    let image = decompress(&jpeg).unwrap();

    let app3_markers: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xE3)
        .collect();
    assert!(
        app3_markers.is_empty(),
        "default decoder should not save APP3 markers"
    );
}

#[test]
fn save_specific_marker_type_only() {
    let jpeg: Vec<u8> = make_jpeg_with_markers();
    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder.save_markers(libjpeg_turbo_rs::MarkerSaveConfig::Specific(vec![0xE3]));
    let image = decoder.decode_image().unwrap();

    let app3: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xE3)
        .collect();
    assert!(!app3.is_empty(), "APP3 should be saved");

    let app5: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xE5)
        .collect();
    assert!(app5.is_empty(), "APP5 should not be saved");
}

#[test]
fn image_markers_accessor_returns_saved_markers() {
    let jpeg: Vec<u8> = make_jpeg_with_markers();
    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder.save_markers(libjpeg_turbo_rs::MarkerSaveConfig::All);
    let image = decoder.decode_image().unwrap();

    let markers: &[SavedMarker] = image.markers();
    assert!(
        markers.iter().any(|m| m.code == 0xE3),
        "markers() should contain APP3"
    );
    assert!(
        markers.iter().any(|m| m.code == 0xE5),
        "markers() should contain APP5"
    );
}

#[test]
fn transform_with_copy_markers_preserves_markers() {
    let jpeg: Vec<u8> = make_jpeg_with_markers();
    let options = TransformOptions {
        op: TransformOp::HFlip,
        copy_markers: libjpeg_turbo_rs::MarkerCopyMode::All,
        ..TransformOptions::default()
    };

    let result: Vec<u8> = libjpeg_turbo_rs::transform_jpeg_with_options(&jpeg, &options).unwrap();

    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&result).unwrap();
    decoder.save_markers(libjpeg_turbo_rs::MarkerSaveConfig::All);
    let image = decoder.decode_image().unwrap();

    let app3: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xE3)
        .collect();
    assert!(!app3.is_empty(), "copy_markers=All should preserve APP3");
    assert_eq!(app3[0].data, b"CustomAPP3Data");

    let app5: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xE5)
        .collect();
    assert!(!app5.is_empty(), "copy_markers=All should preserve APP5");
    assert_eq!(app5[0].data, b"APP5-payload");
}

#[test]
fn transform_with_copy_markers_false_strips_markers() {
    let jpeg: Vec<u8> = make_jpeg_with_markers();
    let options = TransformOptions {
        op: TransformOp::HFlip,
        copy_markers: libjpeg_turbo_rs::MarkerCopyMode::None,
        ..TransformOptions::default()
    };

    let result: Vec<u8> = libjpeg_turbo_rs::transform_jpeg_with_options(&jpeg, &options).unwrap();

    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&result).unwrap();
    decoder.save_markers(libjpeg_turbo_rs::MarkerSaveConfig::All);
    let image = decoder.decode_image().unwrap();

    let custom_markers: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xE3 || m.code == 0xE5 || m.code == 0xFE)
        .collect();
    assert!(
        custom_markers.is_empty(),
        "copy_markers=false should strip all APP/COM markers"
    );
}

#[test]
fn save_app_markers_only_excludes_com() {
    let jpeg: Vec<u8> = make_jpeg_with_markers();
    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder.save_markers(libjpeg_turbo_rs::MarkerSaveConfig::AppOnly);
    let image = decoder.decode_image().unwrap();

    let app3: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xE3)
        .collect();
    assert!(!app3.is_empty(), "APP3 should be saved with AppOnly config");

    let com: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xFE)
        .collect();
    assert!(
        com.is_empty(),
        "COM should not be saved with AppOnly config"
    );
}

#[test]
fn multiple_markers_same_type_all_preserved() {
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let mut encoder = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb);
    encoder = encoder.quality(75);
    encoder = encoder.saved_marker(SavedMarker {
        code: 0xE4,
        data: b"first".to_vec(),
    });
    encoder = encoder.saved_marker(SavedMarker {
        code: 0xE4,
        data: b"second".to_vec(),
    });
    encoder = encoder.saved_marker(SavedMarker {
        code: 0xE4,
        data: b"third".to_vec(),
    });
    let jpeg = encoder.encode().unwrap();

    let mut decoder = libjpeg_turbo_rs::decode::pipeline::Decoder::new(&jpeg).unwrap();
    decoder.save_markers(libjpeg_turbo_rs::MarkerSaveConfig::All);
    let image = decoder.decode_image().unwrap();

    let app4: Vec<&SavedMarker> = image
        .saved_markers
        .iter()
        .filter(|m| m.code == 0xE4)
        .collect();
    assert_eq!(app4.len(), 3, "all three APP4 markers should be preserved");
    assert_eq!(app4[0].data, b"first");
    assert_eq!(app4[1].data, b"second");
    assert_eq!(app4[2].data, b"third");
}

// ===========================================================================
// C cross-validation helpers
// ===========================================================================

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

/// Check if jpegtran supports the `-copy icc` option.
fn jpegtran_supports_copy_icc(jpegtran: &Path) -> bool {
    let output = Command::new(jpegtran).arg("-help").output();
    match output {
        Ok(o) => {
            let text: String = String::from_utf8_lossy(&o.stderr).to_string()
                + &String::from_utf8_lossy(&o.stdout);
            // libjpeg-turbo's jpegtran help mentions "icc" in the -copy option description
            text.contains("-copy icc")
        }
        Err(_) => false,
    }
}

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_mkr_{}_{:04}_{}", pid, counter, name))
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

/// A JPEG marker found by scanning the raw byte stream.
#[derive(Debug)]
struct RawMarker {
    /// Marker code (e.g., 0xE3 for APP3, 0xFE for COM)
    code: u8,
    /// Marker payload (excluding the 2-byte length field itself)
    data: Vec<u8>,
}

/// Scan a JPEG byte stream and extract all APP (0xE0..0xEF) and COM (0xFE) markers.
fn scan_jpeg_markers(jpeg: &[u8]) -> Vec<RawMarker> {
    let mut markers: Vec<RawMarker> = Vec::new();
    let mut pos: usize = 0;
    let len: usize = jpeg.len();

    while pos + 1 < len {
        if jpeg[pos] != 0xFF {
            pos += 1;
            continue;
        }

        let code: u8 = jpeg[pos + 1];
        pos += 2;

        // Skip padding 0xFF bytes
        if code == 0xFF || code == 0x00 {
            continue;
        }

        // SOI (0xD8), EOI (0xD9), RST markers (0xD0..0xD7) have no length
        if code == 0xD8 || code == 0xD9 || (0xD0..=0xD7).contains(&code) {
            continue;
        }

        // SOS (0xDA) — after this comes entropy-coded data; stop scanning
        if code == 0xDA {
            break;
        }

        // Read 2-byte length (big-endian, includes the 2 bytes of the length field)
        if pos + 1 >= len {
            break;
        }
        let marker_len: usize = ((jpeg[pos] as usize) << 8) | (jpeg[pos + 1] as usize);
        if marker_len < 2 || pos + marker_len > len {
            break;
        }

        let payload: Vec<u8> = jpeg[pos + 2..pos + marker_len].to_vec();

        // Collect APP markers (0xE0..0xEF) and COM (0xFE)
        if (0xE0..=0xEF).contains(&code) || code == 0xFE {
            markers.push(RawMarker {
                code,
                data: payload,
            });
        }

        pos += marker_len;
    }

    markers
}

/// Load an ICC profile from the reference test images directory.
fn load_reference_icc(name: &str) -> Option<Vec<u8>> {
    let path: PathBuf = PathBuf::from(format!("references/libjpeg-turbo/testimages/{}", name));
    if path.exists() {
        Some(std::fs::read(&path).expect("read ICC file"))
    } else {
        None
    }
}

// ===========================================================================
// C cross-validation: jpegtran -copy all
// ===========================================================================

#[test]
fn c_jpegtran_copy_all_preserves_markers() {
    let jpegtran: PathBuf = match jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found");
            return;
        }
    };

    // Create input JPEG with APP3 + APP5 + COM markers
    let input_jpeg: Vec<u8> = make_jpeg_with_markers();

    // --- Rust transform: HFlip with copy_markers=All ---
    let rust_options = TransformOptions {
        op: TransformOp::HFlip,
        copy_markers: MarkerCopyMode::All,
        ..TransformOptions::default()
    };
    let rust_output: Vec<u8> =
        libjpeg_turbo_rs::transform_jpeg_with_options(&input_jpeg, &rust_options)
            .expect("Rust transform_jpeg_with_options (HFlip, copy_markers=All)");

    // --- C jpegtran: -copy all -flip horizontal ---
    let tmp_input: TempFile = TempFile::new("c_copy_all_input.jpg");
    let tmp_c_output: TempFile = TempFile::new("c_copy_all_output.jpg");
    std::fs::write(tmp_input.path(), &input_jpeg).expect("write input JPEG");

    let output = Command::new(&jpegtran)
        .arg("-copy")
        .arg("all")
        .arg("-flip")
        .arg("horizontal")
        .arg("-outfile")
        .arg(tmp_c_output.path())
        .arg(tmp_input.path())
        .output()
        .expect("failed to run jpegtran");

    assert!(
        output.status.success(),
        "jpegtran -copy all -flip horizontal failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let c_output: Vec<u8> = std::fs::read(tmp_c_output.path()).expect("read jpegtran output");

    // --- Parse markers from both outputs ---
    let rust_markers: Vec<RawMarker> = scan_jpeg_markers(&rust_output);
    let c_markers: Vec<RawMarker> = scan_jpeg_markers(&c_output);

    // Verify APP3 (0xE3) present in both
    let rust_app3: Vec<&RawMarker> = rust_markers.iter().filter(|m| m.code == 0xE3).collect();
    let c_app3: Vec<&RawMarker> = c_markers.iter().filter(|m| m.code == 0xE3).collect();
    assert!(
        !rust_app3.is_empty(),
        "Rust -copy all should preserve APP3 marker"
    );
    assert!(
        !c_app3.is_empty(),
        "C jpegtran -copy all should preserve APP3 marker"
    );
    assert_eq!(
        rust_app3[0].data, c_app3[0].data,
        "APP3 marker data should match between Rust and C"
    );

    // Verify APP5 (0xE5) present in both
    let rust_app5: Vec<&RawMarker> = rust_markers.iter().filter(|m| m.code == 0xE5).collect();
    let c_app5: Vec<&RawMarker> = c_markers.iter().filter(|m| m.code == 0xE5).collect();
    assert!(
        !rust_app5.is_empty(),
        "Rust -copy all should preserve APP5 marker"
    );
    assert!(
        !c_app5.is_empty(),
        "C jpegtran -copy all should preserve APP5 marker"
    );
    assert_eq!(
        rust_app5[0].data, c_app5[0].data,
        "APP5 marker data should match between Rust and C"
    );

    // Verify COM (0xFE) present in both
    let rust_com: Vec<&RawMarker> = rust_markers.iter().filter(|m| m.code == 0xFE).collect();
    let c_com: Vec<&RawMarker> = c_markers.iter().filter(|m| m.code == 0xFE).collect();
    assert!(
        !rust_com.is_empty(),
        "Rust -copy all should preserve COM marker"
    );
    assert!(
        !c_com.is_empty(),
        "C jpegtran -copy all should preserve COM marker"
    );
    assert_eq!(
        rust_com[0].data, c_com[0].data,
        "COM marker data should match between Rust and C"
    );
}

// ===========================================================================
// C cross-validation: jpegtran -copy none
// ===========================================================================

#[test]
fn c_jpegtran_copy_none_strips_markers() {
    let jpegtran: PathBuf = match jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found");
            return;
        }
    };

    // Create input JPEG with APP3 + APP5 + COM markers
    let input_jpeg: Vec<u8> = make_jpeg_with_markers();

    // --- Rust transform: HFlip with copy_markers=None ---
    let rust_options = TransformOptions {
        op: TransformOp::HFlip,
        copy_markers: MarkerCopyMode::None,
        ..TransformOptions::default()
    };
    let rust_output: Vec<u8> =
        libjpeg_turbo_rs::transform_jpeg_with_options(&input_jpeg, &rust_options)
            .expect("Rust transform_jpeg_with_options (HFlip, copy_markers=None)");

    // --- C jpegtran: -copy none -flip horizontal ---
    let tmp_input: TempFile = TempFile::new("c_copy_none_input.jpg");
    let tmp_c_output: TempFile = TempFile::new("c_copy_none_output.jpg");
    std::fs::write(tmp_input.path(), &input_jpeg).expect("write input JPEG");

    let output = Command::new(&jpegtran)
        .arg("-copy")
        .arg("none")
        .arg("-flip")
        .arg("horizontal")
        .arg("-outfile")
        .arg(tmp_c_output.path())
        .arg(tmp_input.path())
        .output()
        .expect("failed to run jpegtran");

    assert!(
        output.status.success(),
        "jpegtran -copy none -flip horizontal failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let c_output: Vec<u8> = std::fs::read(tmp_c_output.path()).expect("read jpegtran output");

    // --- Parse markers from both outputs ---
    let rust_markers: Vec<RawMarker> = scan_jpeg_markers(&rust_output);
    let c_markers: Vec<RawMarker> = scan_jpeg_markers(&c_output);

    // Filter to only custom APP markers (APP3=0xE3, APP5=0xE5) and COM (0xFE)
    // Note: JFIF APP0 (0xE0) is a standard marker and may be retained even with -copy none
    let rust_custom: Vec<&RawMarker> = rust_markers
        .iter()
        .filter(|m| m.code == 0xE3 || m.code == 0xE5 || m.code == 0xFE)
        .collect();
    let c_custom: Vec<&RawMarker> = c_markers
        .iter()
        .filter(|m| m.code == 0xE3 || m.code == 0xE5 || m.code == 0xFE)
        .collect();

    assert!(
        rust_custom.is_empty(),
        "Rust copy_markers=None should strip APP3/APP5/COM markers, found: {:?}",
        rust_custom.iter().map(|m| m.code).collect::<Vec<_>>()
    );
    assert!(
        c_custom.is_empty(),
        "C jpegtran -copy none should strip APP3/APP5/COM markers, found: {:?}",
        c_custom.iter().map(|m| m.code).collect::<Vec<_>>()
    );
}

// ===========================================================================
// C cross-validation: jpegtran -copy icc
// ===========================================================================

#[test]
fn c_jpegtran_copy_icc_preserves_icc_only() {
    let jpegtran: PathBuf = match jpegtran_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: jpegtran not found");
            return;
        }
    };

    if !jpegtran_supports_copy_icc(&jpegtran) {
        eprintln!("SKIP: jpegtran does not support -copy icc");
        return;
    }

    let icc_data: Vec<u8> = match load_reference_icc("test3.icc") {
        Some(d) => d,
        None => {
            eprintln!("SKIP: test3.icc not found");
            return;
        }
    };

    // Create input JPEG with ICC profile + COM marker
    let pixels: Vec<u8> = vec![128u8; 16 * 16 * 3];
    let comment_text: &str = "cross-check-icc-only-comment";
    let encoder = Encoder::new(&pixels, 16, 16, PixelFormat::Rgb)
        .quality(75)
        .icc_profile(&icc_data)
        .comment(comment_text);
    let input_jpeg: Vec<u8> = encoder.encode().expect("Rust encode with ICC + COM");

    // --- Rust transform: HFlip with copy_markers=IccOnly ---
    let rust_options = TransformOptions {
        op: TransformOp::HFlip,
        copy_markers: MarkerCopyMode::IccOnly,
        ..TransformOptions::default()
    };
    let rust_output: Vec<u8> =
        libjpeg_turbo_rs::transform_jpeg_with_options(&input_jpeg, &rust_options)
            .expect("Rust transform_jpeg_with_options (HFlip, copy_markers=IccOnly)");

    // --- C jpegtran: -copy icc -flip horizontal ---
    let tmp_input: TempFile = TempFile::new("c_copy_icc_input.jpg");
    let tmp_c_output: TempFile = TempFile::new("c_copy_icc_output.jpg");
    std::fs::write(tmp_input.path(), &input_jpeg).expect("write input JPEG");

    let output = Command::new(&jpegtran)
        .arg("-copy")
        .arg("icc")
        .arg("-flip")
        .arg("horizontal")
        .arg("-outfile")
        .arg(tmp_c_output.path())
        .arg(tmp_input.path())
        .output()
        .expect("failed to run jpegtran");

    assert!(
        output.status.success(),
        "jpegtran -copy icc -flip horizontal failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let c_output: Vec<u8> = std::fs::read(tmp_c_output.path()).expect("read jpegtran output");

    // --- Parse markers from both outputs ---
    let rust_markers: Vec<RawMarker> = scan_jpeg_markers(&rust_output);
    let c_markers: Vec<RawMarker> = scan_jpeg_markers(&c_output);

    // ICC is stored as APP2 (0xE2) markers with "ICC_PROFILE\0" prefix
    let icc_prefix: &[u8] = b"ICC_PROFILE\0";

    let rust_icc: Vec<&RawMarker> = rust_markers
        .iter()
        .filter(|m| m.code == 0xE2 && m.data.starts_with(icc_prefix))
        .collect();
    let c_icc: Vec<&RawMarker> = c_markers
        .iter()
        .filter(|m| m.code == 0xE2 && m.data.starts_with(icc_prefix))
        .collect();

    // ICC should be preserved in both
    assert!(
        !rust_icc.is_empty(),
        "Rust IccOnly should preserve ICC (APP2) markers"
    );
    assert!(
        !c_icc.is_empty(),
        "C jpegtran -copy icc should preserve ICC (APP2) markers"
    );

    // Verify ICC marker count matches
    assert_eq!(
        rust_icc.len(),
        c_icc.len(),
        "ICC marker count should match between Rust ({}) and C ({})",
        rust_icc.len(),
        c_icc.len()
    );

    // Verify ICC marker data matches
    for (i, (r, c)) in rust_icc.iter().zip(c_icc.iter()).enumerate() {
        assert_eq!(
            r.data, c.data,
            "ICC marker chunk {} data should match between Rust and C",
            i
        );
    }

    // COM (0xFE) should be stripped in both
    let rust_com: Vec<&RawMarker> = rust_markers.iter().filter(|m| m.code == 0xFE).collect();
    let c_com: Vec<&RawMarker> = c_markers.iter().filter(|m| m.code == 0xFE).collect();

    assert!(rust_com.is_empty(), "Rust IccOnly should strip COM markers");
    assert!(
        c_com.is_empty(),
        "C jpegtran -copy icc should strip COM markers"
    );
}

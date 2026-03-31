/// YUV conversion validation tests.
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::api::yuv;
use libjpeg_turbo_rs::{
    compress, decompress_to, yuv_buf_size, yuv_plane_height, yuv_plane_size, yuv_plane_width,
    PixelFormat, Subsampling,
};

fn gradient_rgb(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height * 3);
    for y in 0..height {
        for x in 0..width {
            pixels.push(((x * 255) / width.max(1)) as u8);
            pixels.push(((y * 255) / height.max(1)) as u8);
            pixels.push((((x + y) * 127) / (width + height).max(1)) as u8);
        }
    }
    pixels
}

const COLOR_SUBSAMPLING: [Subsampling; 6] = [
    Subsampling::S444,
    Subsampling::S422,
    Subsampling::S420,
    Subsampling::S440,
    Subsampling::S411,
    Subsampling::S441,
];

fn yuv_roundtrip_helper(subsamp: Subsampling) {
    let (w, h): (usize, usize) = (48, 48);
    let original: Vec<u8> = gradient_rgb(w, h);
    let yuv_packed: Vec<u8> = yuv::encode_yuv(&original, w, h, PixelFormat::Rgb, subsamp).unwrap();
    assert_eq!(
        yuv_packed.len(),
        yuv_buf_size(w, h, subsamp),
        "size {:?}",
        subsamp
    );
    let decoded: Vec<u8> = yuv::decode_yuv(&yuv_packed, w, h, subsamp, PixelFormat::Rgb).unwrap();
    assert_eq!(decoded.len(), original.len());
    // RGB→YUV→RGB has inherent integer rounding losses.
    // Measured actuals: S444=1, S422=4, S420=5, S440=3, S411=7, S441=7.
    let max_diff: i16 = original
        .iter()
        .zip(decoded.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).abs())
        .max()
        .unwrap_or(0);
    let tolerance: i16 = match subsamp {
        Subsampling::S444 => 1,
        Subsampling::S440 => 3,
        Subsampling::S422 => 4,
        Subsampling::S420 => 5,
        _ => 7, // S411, S441
    };
    assert!(
        max_diff <= tolerance,
        "YUV {:?} max_diff={} (expected <= {})",
        subsamp,
        max_diff,
        tolerance
    );
}

#[test]
fn tjunittest_yuv_roundtrip_444() {
    yuv_roundtrip_helper(Subsampling::S444);
}
#[test]
fn tjunittest_yuv_roundtrip_422() {
    yuv_roundtrip_helper(Subsampling::S422);
}
#[test]
fn tjunittest_yuv_roundtrip_420() {
    yuv_roundtrip_helper(Subsampling::S420);
}
#[test]
fn tjunittest_yuv_roundtrip_440() {
    yuv_roundtrip_helper(Subsampling::S440);
}
#[test]
fn tjunittest_yuv_roundtrip_411() {
    yuv_roundtrip_helper(Subsampling::S411);
}
#[test]
fn tjunittest_yuv_roundtrip_441() {
    yuv_roundtrip_helper(Subsampling::S441);
}

#[test]
fn tjunittest_yuv_roundtrip_various_sizes() {
    for &(w, h) in &[(16usize, 16usize), (35, 27), (48, 48), (100, 1)] {
        for &ss in &[Subsampling::S444, Subsampling::S420] {
            let orig: Vec<u8> = gradient_rgb(w, h);
            let yuv: Vec<u8> = yuv::encode_yuv(&orig, w, h, PixelFormat::Rgb, ss).unwrap();
            let dec: Vec<u8> = yuv::decode_yuv(&yuv, w, h, ss, PixelFormat::Rgb).unwrap();
            assert_eq!(dec.len(), orig.len(), "{}x{} {:?}", w, h, ss);
            // RGB→YUV→RGB roundtrip loss is inherent to chroma subsampling.
            // Loss varies by content and image size (small images have larger
            // relative error from subsampling boundary effects).
            // Measured: S444 max=1, S420 max=12 (at 16x16 gradient).
            let max_diff: i16 = orig
                .iter()
                .zip(dec.iter())
                .map(|(&a, &b)| (a as i16 - b as i16).abs())
                .max()
                .unwrap_or(0);
            let tolerance: i16 = match ss {
                Subsampling::S444 => 1,
                _ => 12, // Chroma subsampling inherent loss
            };
            assert!(
                max_diff <= tolerance,
                "{}x{} {:?} max_diff={} (expected <= {})",
                w,
                h,
                ss,
                max_diff,
                tolerance
            );
        }
    }
}

#[test]
fn tjunittest_yuv_grayscale_roundtrip() {
    let (w, h): (usize, usize) = (48, 48);
    let mut orig: Vec<u8> = Vec::with_capacity(w * h);
    for y in 0..h {
        for x in 0..w {
            orig.push(((x + y) * 4) as u8);
        }
    }
    let yuv: Vec<u8> =
        yuv::encode_yuv(&orig, w, h, PixelFormat::Grayscale, Subsampling::S444).unwrap();
    assert_eq!(yuv.len(), w * h);
    for i in 0..orig.len() {
        assert_eq!(yuv[i], orig[i], "gray Y byte {}", i);
    }
}

#[test]
fn tjunittest_yuv_plane_sizes_aligned() {
    let (w, h): (usize, usize) = (48, 48);
    let cases: [(Subsampling, usize, usize); 6] = [
        (Subsampling::S444, 48, 48),
        (Subsampling::S422, 24, 48),
        (Subsampling::S420, 24, 24),
        (Subsampling::S440, 48, 24),
        (Subsampling::S411, 12, 48),
        (Subsampling::S441, 48, 12),
    ];
    for &(ss, cw, ch) in &cases {
        assert_eq!(yuv_plane_width(0, w, ss), w, "Y width {:?}", ss);
        assert_eq!(yuv_plane_height(0, h, ss), h, "Y height {:?}", ss);
        assert_eq!(yuv_plane_width(1, w, ss), cw, "Cb width {:?}", ss);
        assert_eq!(yuv_plane_height(1, h, ss), ch, "Cb height {:?}", ss);
        let total: usize =
            yuv_plane_size(0, w, h, ss) + yuv_plane_size(1, w, h, ss) + yuv_plane_size(2, w, h, ss);
        assert_eq!(total, yuv_buf_size(w, h, ss), "total {:?}", ss);
    }
}

#[test]
fn tjunittest_yuv_plane_sizes_nonaligned() {
    let (w, h): (usize, usize) = (35, 27);
    for &ss in &COLOR_SUBSAMPLING {
        assert!(yuv_plane_width(0, w, ss) >= w, "Y width {:?}", ss);
        assert!(yuv_plane_height(0, h, ss) >= h, "Y height {:?}", ss);
        assert!(yuv_plane_width(1, w, ss) > 0, "Cb width {:?}", ss);
        assert!(yuv_buf_size(w, h, ss) > 0, "total {:?}", ss);
    }
}

fn compress_from_yuv_helper(subsamp: Subsampling) {
    let (w, h): (usize, usize) = (48, 48);
    let orig: Vec<u8> = gradient_rgb(w, h);
    let yuv: Vec<u8> = yuv::encode_yuv(&orig, w, h, PixelFormat::Rgb, subsamp).unwrap();
    let jpeg: Vec<u8> = yuv::compress_from_yuv(&yuv, w, h, subsamp, 90).unwrap();
    assert!(!jpeg.is_empty());
    let img = decompress_to(&jpeg, PixelFormat::Rgb).unwrap();
    assert_eq!((img.width, img.height), (w, h));
    let direct: Vec<u8> = compress(&orig, w, h, PixelFormat::Rgb, 90, subsamp).unwrap();
    let dimg = decompress_to(&direct, PixelFormat::Rgb).unwrap();
    // YUV-path and direct-path encode the same source at same quality.
    // Decoded pixels should be very close. Differences come from
    // integer rounding in RGB→YUV conversion.
    let max_diff: i16 = img
        .data
        .iter()
        .zip(dimg.data.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).abs())
        .max()
        .unwrap_or(0);
    let tolerance: i16 = match subsamp {
        Subsampling::S444 => 1,
        _ => 5,
    };
    assert!(
        max_diff <= tolerance,
        "yuv vs direct {:?} max_diff={} (expected <= {})",
        subsamp,
        max_diff,
        tolerance
    );
}

#[test]
fn tjunittest_compress_from_yuv_444() {
    compress_from_yuv_helper(Subsampling::S444);
}
#[test]
fn tjunittest_compress_from_yuv_422() {
    compress_from_yuv_helper(Subsampling::S422);
}
#[test]
fn tjunittest_compress_from_yuv_420() {
    compress_from_yuv_helper(Subsampling::S420);
}

fn decompress_to_yuv_helper(subsamp: Subsampling) {
    let (w, h): (usize, usize) = (48, 48);
    let orig: Vec<u8> = gradient_rgb(w, h);
    let jpeg: Vec<u8> = compress(&orig, w, h, PixelFormat::Rgb, 90, subsamp).unwrap();
    let direct: Vec<u8> = decompress_to(&jpeg, PixelFormat::Rgb).unwrap().data;
    let (yuv, yw, yh, ys) = yuv::decompress_to_yuv(&jpeg).unwrap();
    let via: Vec<u8> = yuv::decode_yuv(&yuv, yw, yh, ys, PixelFormat::Rgb).unwrap();
    assert_eq!(via.len(), direct.len());
    // JPEG→YUV→RGB vs JPEG→RGB directly: differences come from
    // merged upsample+color vs separate YUV decode+color paths.
    let max_diff: i16 = via
        .iter()
        .zip(direct.iter())
        .map(|(&a, &b)| (a as i16 - b as i16).abs())
        .max()
        .unwrap_or(0);
    let tolerance: i16 = match subsamp {
        Subsampling::S444 => 1,
        _ => 5,
    };
    assert!(
        max_diff <= tolerance,
        "yuv decode {:?} max_diff={} (expected <= {})",
        subsamp,
        max_diff,
        tolerance
    );
}

#[test]
fn tjunittest_decompress_to_yuv_444() {
    decompress_to_yuv_helper(Subsampling::S444);
}
#[test]
fn tjunittest_decompress_to_yuv_422() {
    decompress_to_yuv_helper(Subsampling::S422);
}
#[test]
fn tjunittest_decompress_to_yuv_420() {
    decompress_to_yuv_helper(Subsampling::S420);
}

#[test]
fn tjunittest_yuv_encode_multiple_pixel_formats() {
    let (w, h): (usize, usize) = (48, 48);
    for &pf in &[
        PixelFormat::Rgb,
        PixelFormat::Bgr,
        PixelFormat::Rgba,
        PixelFormat::Bgra,
    ] {
        let bpp: usize = pf.bytes_per_pixel();
        let mut px: Vec<u8> = vec![0u8; w * h * bpp];
        for y in 0..h {
            for x in 0..w {
                let idx: usize = (y * w + x) * bpp;
                px[idx + pf.red_offset().unwrap()] = ((x * 255) / w) as u8;
                px[idx + pf.green_offset().unwrap()] = ((y * 255) / h) as u8;
                px[idx + pf.blue_offset().unwrap()] = 128;
                if bpp == 4 {
                    let ao = 6
                        - pf.red_offset().unwrap()
                        - pf.green_offset().unwrap()
                        - pf.blue_offset().unwrap();
                    if ao < bpp {
                        px[idx + ao] = 255;
                    }
                }
            }
        }
        let yuv: Vec<u8> = yuv::encode_yuv(&px, w, h, pf, Subsampling::S444).unwrap();
        assert_eq!(yuv.len(), yuv_buf_size(w, h, Subsampling::S444), "{:?}", pf);
        assert!(yuv.iter().any(|&v| v > 0), "{:?}", pf);
    }
}

// ---------------------------------------------------------------------------
// C djpeg cross-validation helpers
// ---------------------------------------------------------------------------

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(name: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("ljt_rs_yuv_{}_{:04}_{}", pid, counter, name))
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

/// Locate the djpeg binary. Checks /opt/homebrew/bin/djpeg first, then falls
/// back to whatever `which djpeg` returns. Returns `None` when not found.
fn djpeg_path() -> Option<PathBuf> {
    let homebrew_path: PathBuf = PathBuf::from("/opt/homebrew/bin/djpeg");
    if homebrew_path.exists() {
        return Some(homebrew_path);
    }

    let output = Command::new("which").arg("djpeg").output().ok()?;
    if output.status.success() {
        let path_str: String = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path_str.is_empty() {
            let path: PathBuf = PathBuf::from(&path_str);
            if path.exists() {
                return Some(path);
            }
        }
    }

    None
}

/// Parse a binary PPM (P6) file into (width, height, rgb_pixels).
/// Returns `None` if the file is not a valid P6 PPM.
fn parse_ppm(data: &[u8]) -> Option<(usize, usize, Vec<u8>)> {
    if data.len() < 3 || &data[0..2] != b"P6" {
        return None;
    }
    let mut pos: usize = 2;

    // Skip whitespace
    while pos < data.len()
        && (data[pos] == b' ' || data[pos] == b'\n' || data[pos] == b'\r' || data[pos] == b'\t')
    {
        pos += 1;
    }

    // Skip comments
    while pos < data.len() && data[pos] == b'#' {
        while pos < data.len() && data[pos] != b'\n' {
            pos += 1;
        }
        if pos < data.len() {
            pos += 1;
        }
    }

    // Parse width
    let width_start: usize = pos;
    while pos < data.len() && data[pos].is_ascii_digit() {
        pos += 1;
    }
    let width: usize = std::str::from_utf8(&data[width_start..pos])
        .ok()?
        .parse()
        .ok()?;

    // Skip whitespace
    while pos < data.len()
        && (data[pos] == b' ' || data[pos] == b'\n' || data[pos] == b'\r' || data[pos] == b'\t')
    {
        pos += 1;
    }

    // Skip comments
    while pos < data.len() && data[pos] == b'#' {
        while pos < data.len() && data[pos] != b'\n' {
            pos += 1;
        }
        if pos < data.len() {
            pos += 1;
        }
    }

    // Parse height
    let height_start: usize = pos;
    while pos < data.len() && data[pos].is_ascii_digit() {
        pos += 1;
    }
    let height: usize = std::str::from_utf8(&data[height_start..pos])
        .ok()?
        .parse()
        .ok()?;

    // Skip whitespace
    while pos < data.len()
        && (data[pos] == b' ' || data[pos] == b'\n' || data[pos] == b'\r' || data[pos] == b'\t')
    {
        pos += 1;
    }

    // Skip comments
    while pos < data.len() && data[pos] == b'#' {
        while pos < data.len() && data[pos] != b'\n' {
            pos += 1;
        }
        if pos < data.len() {
            pos += 1;
        }
    }

    // Parse maxval
    let maxval_start: usize = pos;
    while pos < data.len() && data[pos].is_ascii_digit() {
        pos += 1;
    }
    let _maxval: usize = std::str::from_utf8(&data[maxval_start..pos])
        .ok()?
        .parse()
        .ok()?;

    // Exactly one whitespace character after maxval
    if pos < data.len()
        && (data[pos] == b' ' || data[pos] == b'\n' || data[pos] == b'\r' || data[pos] == b'\t')
    {
        pos += 1;
    }

    // Remaining data is the pixel buffer
    let expected_len: usize = width * height * 3;
    if data.len() - pos < expected_len {
        return None;
    }

    Some((width, height, data[pos..pos + expected_len].to_vec()))
}

// ---------------------------------------------------------------------------
// Test: C djpeg cross-validation for YUV roundtrip
// ---------------------------------------------------------------------------

/// Cross-validate Rust YUV→JPEG compression against C djpeg.
///
/// For S444, S422, S420:
/// 1. Encode a 48x48 RGB pattern to YUV with Rust
/// 2. Compress YUV to JPEG with Rust
/// 3. Decode JPEG with both C djpeg and Rust decompress()
/// 4. Assert pixel-identical output (diff=0)
///
/// This validates that the YUV→JPEG compression path produces standard-
/// compliant JPEG files that C libjpeg-turbo decodes identically to Rust.
#[test]
fn c_djpeg_cross_validation_yuv_roundtrip() {
    let djpeg: PathBuf = match djpeg_path() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: djpeg not found");
            return;
        }
    };

    let (w, h): (usize, usize) = (48, 48);
    let rgb: Vec<u8> = gradient_rgb(w, h);

    let subsamplings: [Subsampling; 3] = [Subsampling::S444, Subsampling::S422, Subsampling::S420];
    let mut tested: usize = 0;
    let mut passed: usize = 0;

    for &ss in &subsamplings {
        tested += 1;

        // Step 1: Encode RGB to YUV with Rust
        let yuv: Vec<u8> =
            yuv::encode_yuv(&rgb, w, h, PixelFormat::Rgb, ss).expect("encode_yuv failed");

        // Step 2: Compress YUV to JPEG with Rust
        let jpeg: Vec<u8> =
            yuv::compress_from_yuv(&yuv, w, h, ss, 90).expect("compress_from_yuv failed");
        assert!(!jpeg.is_empty(), "JPEG output is empty for {:?}", ss);

        // Step 3: Write JPEG to temp file
        let tmp: TempFile = TempFile::new(&format!("yuv_xval_{:?}.jpg", ss));
        std::fs::write(tmp.path(), &jpeg)
            .unwrap_or_else(|e| panic!("failed to write temp JPEG: {}", e));

        // Step 4: Decode with C djpeg -ppm
        let c_output = Command::new(&djpeg)
            .args(["-ppm"])
            .arg(tmp.path())
            .output()
            .unwrap_or_else(|e| panic!("djpeg execution failed: {}", e));
        assert!(
            c_output.status.success(),
            "djpeg failed for {:?}: {}",
            ss,
            String::from_utf8_lossy(&c_output.stderr)
        );

        let (c_w, c_h, c_pixels): (usize, usize, Vec<u8>) = parse_ppm(&c_output.stdout)
            .unwrap_or_else(|| panic!("failed to parse djpeg PPM output for {:?}", ss));
        assert_eq!(
            (c_w, c_h),
            (w, h),
            "C djpeg dimensions mismatch for {:?}",
            ss
        );

        // Step 5: Decode with Rust decompress()
        let rust_img = decompress_to(&jpeg, PixelFormat::Rgb)
            .unwrap_or_else(|e| panic!("Rust decompress failed for {:?}: {}", ss, e));
        assert_eq!(
            (rust_img.width, rust_img.height),
            (w, h),
            "Rust decode dimensions mismatch for {:?}",
            ss
        );

        // Step 6: Compare Rust decode vs C djpeg decode (diff=0)
        assert_eq!(
            rust_img.data.len(),
            c_pixels.len(),
            "pixel buffer length mismatch for {:?}: rust={} c={}",
            ss,
            rust_img.data.len(),
            c_pixels.len()
        );

        let max_diff: i16 = rust_img
            .data
            .iter()
            .zip(c_pixels.iter())
            .map(|(&a, &b)| (a as i16 - b as i16).abs())
            .max()
            .unwrap_or(0);

        assert_eq!(
            max_diff, 0,
            "Rust vs C djpeg pixel diff must be 0 for {:?}, got max_diff={}",
            ss, max_diff
        );

        passed += 1;
    }

    assert_eq!(
        passed, tested,
        "not all subsampling modes passed: {}/{}",
        passed, tested
    );
}

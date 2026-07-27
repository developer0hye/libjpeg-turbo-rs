//! Issue #369: grayscale decode to Argb/Abgr put alpha and blue in the wrong
//! slots (P4-57).
//!
//! The grayscale→colour expansion branches grouped `Argb`/`Abgr` with
//! `Rgba`/`Bgra`/`Rgbx`/`Bgrx` and wrote `[v, v, v, 255]`, so the gray value
//! landed in the alpha slot and 0xFF in the last colour slot (blue for
//! `Argb`, red for `Abgr`). The 3-component colour path (and C
//! libjpeg-turbo's `JCS_EXT_ARGB`/`JCS_EXT_ABGR`) write alpha first:
//! `[255, v, v, v]`.
//!
//! These tests pin the channel placement for every 4bpp format on both the
//! baseline grayscale expansion path and the lossless grayscale expansion
//! path. Baseline gray values come from C `djpeg` (or our own grayscale
//! decode when djpeg is absent); the lossless path uses the encoder input,
//! whose round-trip is exact.

mod helpers;

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

use libjpeg_turbo_rs::{compress_lossless, decompress_into, decompress_to, Encoder, PixelFormat};

static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

fn temp_path(suffix: &str) -> PathBuf {
    let counter: u64 = TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let pid: u32 = std::process::id();
    std::env::temp_dir().join(format!("issue369_{}_{:04}_{}", pid, counter, suffix))
}

struct TempFile {
    path: PathBuf,
}

impl TempFile {
    fn new(suffix: &str) -> Self {
        Self {
            path: temp_path(suffix),
        }
    }

    fn path(&self) -> &PathBuf {
        &self.path
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

fn find_c_tool(name: &str) -> Option<PathBuf> {
    let brew: PathBuf = PathBuf::from(format!("/opt/homebrew/bin/{name}"));
    if brew.exists() {
        return Some(brew);
    }
    let output = Command::new("which").arg(name).output().ok()?;
    if output.status.success() {
        let path_str: String = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path_str.is_empty() {
            return Some(PathBuf::from(path_str));
        }
    }
    None
}

fn generate_gray_gradient(width: usize, height: usize) -> Vec<u8> {
    let mut pixels: Vec<u8> = Vec::with_capacity(width * height);
    for y in 0..height {
        for x in 0..width {
            pixels.push((((x * 7) + (y * 5)) % 256) as u8);
        }
    }
    pixels
}

fn parse_pgm(data: &[u8]) -> (usize, usize, Vec<u8>) {
    assert!(data.len() > 3, "PGM data too short");
    assert_eq!(&data[0..2], b"P5", "not a P5 PGM");
    let mut pos: usize = 2;
    pos = skip_ws_comments(data, pos);
    let (width, next) = read_number(data, pos);
    pos = skip_ws_comments(data, next);
    let (height, next) = read_number(data, pos);
    pos = skip_ws_comments(data, next);
    let (_maxval, next) = read_number(data, pos);
    pos = next + 1;
    let expected_len: usize = width * height;
    assert!(
        data.len() - pos >= expected_len,
        "PGM pixel data too short: need {} bytes, have {}",
        expected_len,
        data.len() - pos,
    );
    (width, height, data[pos..pos + expected_len].to_vec())
}

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

fn read_number(data: &[u8], idx: usize) -> (usize, usize) {
    let mut end: usize = idx;
    while end < data.len() && data[end].is_ascii_digit() {
        end += 1;
    }
    let val: usize = std::str::from_utf8(&data[idx..end])
        .expect("non-UTF8 in header")
        .parse()
        .expect("invalid number in header");
    (val, end)
}

/// Decode `jpeg_data` (a grayscale JPEG) to `format` and assert every pixel
/// places the reference gray value at the R/G/B offsets and 255 at the
/// remaining pad/alpha slot.
fn assert_gray_expansion_placement(
    jpeg_data: &[u8],
    reference_gray: &[u8],
    format: PixelFormat,
    width: usize,
    height: usize,
    label: &str,
) {
    let rust_img = decompress_to(jpeg_data, format)
        .unwrap_or_else(|e| panic!("[{label}] decompress_to {:?} failed: {e}", format));
    assert_eq!(rust_img.width, width, "[{label}] width mismatch");
    assert_eq!(rust_img.height, height, "[{label}] height mismatch");

    let bpp: usize = format.bytes_per_pixel();
    assert_eq!(bpp, 4, "[{label}] test expects a 4bpp format");
    let r_off: usize = format.red_offset().expect("red offset");
    let g_off: usize = format.green_offset().expect("green offset");
    let b_off: usize = format.blue_offset().expect("blue offset");
    // The remaining slot (offsets sum to 0+1+2+3=6) is pad/alpha.
    let pad_off: usize = 6 - r_off - g_off - b_off;

    assert_eq!(rust_img.data.len(), width * height * bpp, "[{label}] size");
    let mut mismatches: usize = 0;
    for (i, &v) in reference_gray.iter().enumerate() {
        let base: usize = i * bpp;
        let px: &[u8] = &rust_img.data[base..base + bpp];
        let ok: bool = px[r_off] == v && px[g_off] == v && px[b_off] == v && px[pad_off] == 255;
        if !ok {
            mismatches += 1;
            if mismatches <= 5 {
                eprintln!(
                    "  [{label}] pixel {i}: expected gray={v} alpha=255 \
                     (r_off={r_off} g_off={g_off} b_off={b_off} pad_off={pad_off}), got {px:?}"
                );
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "[{label}] {mismatches} pixels with wrong channel placement (must be 0)"
    );
}

const ALL_4BPP_FORMATS: &[(PixelFormat, &str)] = &[
    (PixelFormat::Rgba, "RGBA"),
    (PixelFormat::Bgra, "BGRA"),
    (PixelFormat::Rgbx, "RGBX"),
    (PixelFormat::Bgrx, "BGRX"),
    (PixelFormat::Xrgb, "XRGB"),
    (PixelFormat::Xbgr, "XBGR"),
    (PixelFormat::Argb, "ARGB"),
    (PixelFormat::Abgr, "ABGR"),
];

/// Issue #369: baseline grayscale JPEG → every 4bpp format, gray values
/// cross-validated against C `djpeg` and channel placement asserted per
/// format offsets. Before the fix, ARGB/ABGR wrote `[v, v, v, 255]`
/// (alpha slot got the gray value; the 255 landed in blue for ARGB, red
/// for ABGR).
#[test]
fn issue_369_baseline_gray_to_4bpp_channel_placement() {
    let width: usize = 61; // deliberately non-multiple-of-8
    let height: usize = 37;
    let gray: Vec<u8> = generate_gray_gradient(width, height);
    let jpeg_data: Vec<u8> = Encoder::new(&gray, width, height, PixelFormat::Grayscale)
        .quality(95)
        .encode()
        .expect("grayscale encode must succeed");

    // Reference gray values: C djpeg where available (cross-validation),
    // otherwise our own grayscale decode (still pins channel placement, which
    // is the defect under test).
    let reference: Vec<u8> = match find_c_tool("djpeg") {
        Some(djpeg) => {
            let tmp_jpg = TempFile::new("gray.jpg");
            let tmp_pgm = TempFile::new("gray.pgm");
            std::fs::write(tmp_jpg.path(), &jpeg_data).expect("write tmp jpg");
            let output = Command::new(&djpeg)
                .arg("-pnm")
                .arg("-outfile")
                .arg(tmp_pgm.path())
                .arg(tmp_jpg.path())
                .output()
                .expect("failed to run djpeg");
            assert!(
                output.status.success(),
                "djpeg failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let pgm: Vec<u8> = std::fs::read(tmp_pgm.path()).expect("read PGM");
            let (c_w, c_h, c_gray) = parse_pgm(&pgm);
            assert_eq!(c_w, width);
            assert_eq!(c_h, height);
            c_gray
        }
        None => {
            eprintln!("SKIP(C xval): djpeg not found — using Rust grayscale decode as reference");
            decompress_to(&jpeg_data, PixelFormat::Grayscale)
                .expect("Rust grayscale decode failed")
                .data
        }
    };

    for &(fmt, name) in ALL_4BPP_FORMATS {
        assert_gray_expansion_placement(
            &jpeg_data,
            &reference,
            fmt,
            width,
            height,
            &format!("baseline_{name}"),
        );
    }
}

/// Issue #369: the lossless (SOF3) grayscale expansion branch had the same
/// wrong `Argb | Abgr` grouping. Lossless decode reproduces the input
/// exactly, so the input gradient is the reference.
#[test]
fn issue_369_lossless_gray_to_4bpp_channel_placement() {
    let width: usize = 33;
    let height: usize = 21;
    let gray: Vec<u8> = generate_gray_gradient(width, height);
    let jpeg_data: Vec<u8> = compress_lossless(&gray, width, height, PixelFormat::Grayscale)
        .expect("lossless grayscale encode must succeed");

    // Sanity: lossless grayscale round-trip is exact.
    let round_trip = decompress_to(&jpeg_data, PixelFormat::Grayscale)
        .expect("lossless grayscale decode failed");
    assert_eq!(round_trip.data, gray, "lossless round-trip must be exact");

    for &(fmt, name) in ALL_4BPP_FORMATS {
        assert_gray_expansion_placement(
            &jpeg_data,
            &gray,
            fmt,
            width,
            height,
            &format!("lossless_{name}"),
        );
    }
}

/// Issue #369: for gray content, ARGB output must be byte-identical to XRGB
/// (and ABGR to XBGR) — both are `[255, v, v, v]` / `[255, v, v, v]` with
/// identical channel geometry, differing only in the pad byte's meaning.
/// This pins agreement with the 3-component colour path's channel order
/// without needing C tools.
#[test]
fn issue_369_argb_matches_xrgb_for_gray_content() {
    let width: usize = 24;
    let height: usize = 16;
    let gray: Vec<u8> = generate_gray_gradient(width, height);
    let jpeg_data: Vec<u8> = Encoder::new(&gray, width, height, PixelFormat::Grayscale)
        .quality(95)
        .encode()
        .expect("grayscale encode must succeed");

    let argb = decompress_to(&jpeg_data, PixelFormat::Argb).expect("ARGB decode");
    let xrgb = decompress_to(&jpeg_data, PixelFormat::Xrgb).expect("XRGB decode");
    assert_eq!(
        argb.data, xrgb.data,
        "gray→ARGB must equal gray→XRGB ([255, v, v, v])"
    );

    let abgr = decompress_to(&jpeg_data, PixelFormat::Abgr).expect("ABGR decode");
    let xbgr = decompress_to(&jpeg_data, PixelFormat::Xbgr).expect("XBGR decode");
    assert_eq!(
        abgr.data, xbgr.data,
        "gray→ABGR must equal gray→XBGR ([255, v, v, v])"
    );
}

/// Issue #369 surfaced from the #354 caller-buffer review — pin the
/// `decompress_into` entry point explicitly (it shares the expansion loop
/// with `decompress_to` via `take_out_buf`, but the entry point that
/// surfaced the bug deserves its own gate).
#[test]
fn issue_369_decompress_into_matches_decompress_to() {
    let width: usize = 24;
    let height: usize = 16;
    let gray: Vec<u8> = generate_gray_gradient(width, height);
    let jpeg_data: Vec<u8> = Encoder::new(&gray, width, height, PixelFormat::Grayscale)
        .quality(95)
        .encode()
        .expect("grayscale encode must succeed");

    for &(fmt, name) in ALL_4BPP_FORMATS {
        let img = decompress_to(&jpeg_data, fmt)
            .unwrap_or_else(|e| panic!("[{name}] decompress_to failed: {e}"));
        let mut buf: Vec<u8> = vec![0u8; width * height * 4];
        decompress_into(&jpeg_data, fmt, &mut buf)
            .unwrap_or_else(|e| panic!("[{name}] decompress_into failed: {e}"));
        assert_eq!(
            buf, img.data,
            "[{name}] decompress_into must produce the same bytes as decompress_to"
        );
    }
}

/// Issue #369: full-buffer byte equality against the real C libjpeg-turbo
/// TurboJPEG API (`tj3Decompress8`) for a grayscale fixture. `djpeg` has no
/// ARGB/ABGR writer, so this oracle is what pins the byte-level channel
/// order against C. Restricted to the alpha formats (RGBA/BGRA/ARGB/ABGR):
/// TurboJPEG documents the alpha channel as 255 on decompression but leaves
/// the pad byte of the X formats undefined, so only the alpha formats carry
/// a byte-exact C contract.
#[test]
fn issue_369_gray_alpha_formats_byte_exact_vs_c_tj3() {
    let Some(oracle) = helpers::c_oracle::gray_argb_c_oracle() else {
        eprintln!("SKIP: no TurboJPEG dev install found to build the gray-ARGB C oracle");
        return;
    };

    let width: usize = 61;
    let height: usize = 37;
    let gray: Vec<u8> = generate_gray_gradient(width, height);
    let jpeg_data: Vec<u8> = Encoder::new(&gray, width, height, PixelFormat::Grayscale)
        .quality(95)
        .encode()
        .expect("grayscale encode must succeed");
    let tmp_jpg = TempFile::new("tj3.jpg");
    std::fs::write(tmp_jpg.path(), &jpeg_data).expect("write tmp jpg");

    let alpha_formats: &[(PixelFormat, &str)] = &[
        (PixelFormat::Rgba, "RGBA"),
        (PixelFormat::Bgra, "BGRA"),
        (PixelFormat::Argb, "ARGB"),
        (PixelFormat::Abgr, "ABGR"),
    ];
    for &(fmt, name) in alpha_formats {
        let theirs: Vec<u8> =
            helpers::c_oracle::decode_with_gray_argb_c_oracle(&oracle, tmp_jpg.path(), name);
        let ours = decompress_to(&jpeg_data, fmt)
            .unwrap_or_else(|e| panic!("[{name}] decompress_to failed: {e}"));
        assert_eq!(
            ours.data.len(),
            theirs.len(),
            "[{name}] buffer size mismatch vs C tj3"
        );
        let mismatches: usize = ours
            .data
            .iter()
            .zip(theirs.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            mismatches, 0,
            "[{name}] {mismatches} bytes differ from C tj3Decompress8 (must be 0)"
        );
    }
}

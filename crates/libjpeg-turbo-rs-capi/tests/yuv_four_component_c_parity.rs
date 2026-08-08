//! C cross-validation for the P4-120 component-count guard.
//!
//! `yuv_four_component_guard.rs` and `yuv_decompress_planes_component_guard.rs`
//! pin *our* rejection of 4-component (CMYK/YCCK) frames on the TurboJPEG YUV
//! decompress entry points. This file proves the rejection is parity with stock
//! libjpeg-turbo rather than a local invention: it drives
//! `examples/yuv_component_count_c_oracle.c`, linked against a real
//! `libturbojpeg`, over the same fixtures and requires the two implementations
//! to agree on accept-vs-reject for both entry points.
//!
//! Upstream's guard lives in `tj3DecompressToYUVPlanes8`
//! (`references/libjpeg-turbo/src/turbojpeg.c:2229-2230`); its
//! `tj3DecompressToYUV8` inherits it by delegating. This port does not
//! delegate, so both of our entry points carry the check.

use std::ffi::{c_int, c_void};
use std::path::{Path, PathBuf};
use std::process::Command;

use libjpeg_turbo_rs_capi::inner::{compress, PixelFormat, Subsampling};
use libjpeg_turbo_rs_capi::{
    tj3DecompressToYUV8, tj3DecompressToYUVPlanes8, tj3Destroy, tj3Init, tj3YUVBufSize,
};

const TJINIT_DECOMPRESS: c_int = 2;
const TJSAMP_444: c_int = 0;
const ALIGN: c_int = 1;
const WIDTH: usize = 16;
const HEIGHT: usize = 16;

/// A TurboJPEG development install: `turbojpeg.h` plus a linkable
/// `libturbojpeg`. Mirrors the prefix search `tests/helpers/c_oracle.rs` uses in
/// the root crate; duplicated rather than shared because that helper is not
/// reachable from this workspace member.
struct TurboJpegDev {
    include_dir: PathBuf,
    lib_dir: PathBuf,
}

fn find_turbojpeg_dev() -> Option<TurboJpegDev> {
    let mut prefixes: Vec<PathBuf> = Vec::new();
    for var in ["LIBJPEG_TURBO_PREFIX", "CONDA_PREFIX"] {
        if let Ok(prefix) = std::env::var(var) {
            prefixes.push(PathBuf::from(prefix));
        }
    }
    prefixes.extend(
        [
            "/opt/libjpeg-turbo",
            "/opt/homebrew/opt/jpeg-turbo",
            "/opt/homebrew",
            "/usr/local",
            "/usr",
        ]
        .iter()
        .map(PathBuf::from),
    );

    // Debian/Ubuntu put the header in `<prefix>/include` but the linker stub in
    // `<prefix>/lib/<triplet>`, so a plain lib64/lib scan reports a perfectly
    // good install as absent — and the mandatory oracle would silently skip.
    let mut lib_subdirs: Vec<PathBuf> = vec![PathBuf::from("lib64"), PathBuf::from("lib")];
    if let Some(triplet) = host_multiarch_triplet() {
        lib_subdirs.push(PathBuf::from("lib").join(&triplet));
        lib_subdirs.push(PathBuf::from("lib64").join(&triplet));
    }

    for prefix in prefixes {
        let include_dir: PathBuf = prefix.join("include");
        // The oracle is TJ3-only. Ubuntu's stock `libturbojpeg0-dev` is still
        // TurboJPEG 2.1.x, whose header declares no `tj3*` entry point, so
        // accepting it here would turn a skip into a compile failure on a
        // perfectly ordinary developer machine.
        let header: PathBuf = include_dir.join("turbojpeg.h");
        let Ok(header_text) = std::fs::read_to_string(&header) else {
            continue;
        };
        if !header_text.contains("tj3DecompressToYUVPlanes8") {
            continue;
        }
        for lib_subdir in &lib_subdirs {
            let lib_dir: PathBuf = prefix.join(lib_subdir);
            let has_library: bool = ["libturbojpeg.so", "libturbojpeg.dylib", "libturbojpeg.a"]
                .iter()
                .any(|file| lib_dir.join(file).exists());
            if has_library {
                return Some(TurboJpegDev {
                    include_dir,
                    lib_dir,
                });
            }
        }
    }
    None
}

/// The compiler's multiarch triplet (e.g. `x86_64-linux-gnu`), or `None` where
/// the toolchain does not use one (macOS clang prints an empty line).
fn host_multiarch_triplet() -> Option<String> {
    let compiler: String = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let output = Command::new(compiler)
        .arg("-print-multiarch")
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let triplet: String = String::from_utf8_lossy(&output.stdout).trim().to_string();
    (!triplet.is_empty()).then_some(triplet)
}

/// True when a missing TurboJPEG 3 install must fail rather than skip: an
/// explicit `LIBJPEG_TURBO_PREFIX` is a statement that one is provisioned, and
/// skipping would leave the parity gate green while checking nothing. The CI
/// step that runs this test sets it. Deliberately *not* keyed on a bare `CI`
/// variable — a CI service running `cargo test --workspace` without TurboJPEG
/// development files is the missing-tool case CLAUDE.md lets us skip.
fn oracle_is_required() -> bool {
    std::env::var_os("LIBJPEG_TURBO_PREFIX").is_some()
}

/// Build the oracle into a temp dir. `None` means no TurboJPEG development
/// install was found on a developer machine — the one legitimate reason to
/// skip, and not one [`oracle_is_required`] allows. A compile failure once an
/// install *is* present is fatal: it would otherwise hide the parity check
/// behind a green run.
fn build_oracle() -> Option<PathBuf> {
    let install: TurboJpegDev = match find_turbojpeg_dev() {
        Some(install) => install,
        None => {
            assert!(
                !oracle_is_required(),
                "no TurboJPEG 3 development install (a turbojpeg.h declaring tj3* plus a linkable \
                 libturbojpeg) found under LIBJPEG_TURBO_PREFIX={:?} — that variable says one is \
                 provisioned, and skipping here would pass the C parity gate without checking \
                 anything",
                std::env::var_os("LIBJPEG_TURBO_PREFIX")
            );
            return None;
        }
    };

    let source: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("yuv_component_count_c_oracle.c");
    assert!(
        source.exists(),
        "oracle source missing: {}",
        source.display()
    );

    let out_dir: PathBuf = std::env::temp_dir().join(format!(
        "libjpeg_turbo_rs_yuv_oracle_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&out_dir).expect("create oracle build dir");
    let oracle: PathBuf = out_dir.join("yuv_component_count_c_oracle");

    let compiler: String = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let output = Command::new(&compiler)
        .arg("-O2")
        .arg("-o")
        .arg(&oracle)
        .arg(&source)
        .arg(format!("-I{}", install.include_dir.display()))
        .arg(format!("-L{}", install.lib_dir.display()))
        .arg("-lturbojpeg")
        .arg(format!("-Wl,-rpath,{}", install.lib_dir.display()))
        .output()
        .unwrap_or_else(|e| panic!("failed to run {compiler}: {e}"));
    assert!(
        output.status.success(),
        "failed to build the YUV component-count C oracle with {compiler} against {}:\n{}",
        install.include_dir.display(),
        String::from_utf8_lossy(&output.stderr)
    );
    Some(oracle)
}

/// `(yuv8_rc, planes8_rc)` as reported by stock libjpeg-turbo.
fn c_return_codes(oracle: &Path, jpeg: &[u8], label: &str) -> (i32, i32) {
    let jpeg_path: PathBuf = std::env::temp_dir().join(format!(
        "libjpeg_turbo_rs_yuv_oracle_{}_{label}.jpg",
        std::process::id()
    ));
    std::fs::write(&jpeg_path, jpeg).expect("write oracle fixture");

    let output = Command::new(oracle)
        .arg(&jpeg_path)
        .output()
        .unwrap_or_else(|e| panic!("failed to run the C oracle: {e}"));
    let _ = std::fs::remove_file(&jpeg_path);
    assert!(
        output.status.success(),
        "C oracle harness failed for {label}:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout: String = String::from_utf8_lossy(&output.stdout).into_owned();
    let parse = |prefix: &str| -> i32 {
        stdout
            .lines()
            .find_map(|line| line.strip_prefix(prefix)?.split_whitespace().next())
            .unwrap_or_else(|| panic!("no `{prefix}` line in oracle output:\n{stdout}"))
            .strip_prefix("rc=")
            .unwrap_or_else(|| panic!("malformed rc field in oracle output:\n{stdout}"))
            .parse()
            .unwrap_or_else(|e| panic!("unparseable rc in oracle output ({e}):\n{stdout}"))
    };
    (parse("yuv8 "), parse("planes8 "))
}

/// `(yuv8_rc, planes8_rc)` from this crate, with destinations sized the way the
/// TurboJPEG contract documents: `tj3YUVBufSize` bytes packed, 3 plane pointers
/// planar.
fn rust_return_codes(jpeg: &[u8]) -> (c_int, c_int) {
    let mut packed: Vec<u8> =
        vec![0; tj3YUVBufSize(WIDTH as c_int, ALIGN, HEIGHT as c_int, TJSAMP_444)];
    let handle: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
    assert!(!handle.is_null(), "tj3Init");
    let rc_yuv8: c_int = tj3DecompressToYUV8(
        handle,
        jpeg.as_ptr(),
        jpeg.len(),
        packed.as_mut_ptr(),
        ALIGN,
    );
    tj3Destroy(handle);

    let mut planes: Vec<Vec<u8>> = (0..3).map(|_| vec![0u8; WIDTH * HEIGHT]).collect();
    let mut plane_ptrs: Vec<*mut u8> = planes.iter_mut().map(|p| p.as_mut_ptr()).collect();
    let handle: *mut c_void = tj3Init(TJINIT_DECOMPRESS);
    assert!(!handle.is_null(), "tj3Init");
    let rc_planes8: c_int = tj3DecompressToYUVPlanes8(
        handle,
        jpeg.as_ptr(),
        jpeg.len(),
        plane_ptrs.as_mut_ptr(),
        std::ptr::null(),
    );
    tj3Destroy(handle);

    (rc_yuv8, rc_planes8)
}

fn jpeg_fixture(format: PixelFormat, channels: usize) -> Vec<u8> {
    let pixels: Vec<u8> = (0..WIDTH * HEIGHT * channels)
        .map(|i| (i % 251) as u8)
        .collect();
    compress(&pixels, WIDTH, HEIGHT, format, 90, Subsampling::S444)
        .unwrap_or_else(|e| panic!("compress {channels}-channel fixture: {e}"))
}

/// P4-120: our accept/reject decision on both YUV decompress entry points must
/// match stock libjpeg-turbo — rejecting the 4-component frame that used to
/// overrun the caller's buffers, and still accepting a 3-component one.
#[test]
fn yuv_decompress_component_count_matches_c() {
    let Some(oracle) = build_oracle() else {
        eprintln!(
            "SKIP: no TurboJPEG 3 development install found (need a turbojpeg.h declaring tj3* \
             plus a linkable libturbojpeg); set LIBJPEG_TURBO_PREFIX to require it"
        );
        return;
    };

    let cmyk: Vec<u8> = jpeg_fixture(PixelFormat::Cmyk, 4);
    let (c_yuv8, c_planes8) = c_return_codes(&oracle, &cmyk, "cmyk");
    let (rust_yuv8, rust_planes8) = rust_return_codes(&cmyk);
    assert_eq!(
        c_yuv8, -1,
        "upstream tj3DecompressToYUV8 is expected to reject a 4-component frame"
    );
    assert_eq!(
        c_planes8, -1,
        "upstream tj3DecompressToYUVPlanes8 is expected to reject a 4-component frame"
    );
    assert_eq!(
        (rust_yuv8, rust_planes8),
        (c_yuv8, c_planes8),
        "4-component frame: our (yuv8, planes8) return codes must match C"
    );

    let rgb: Vec<u8> = jpeg_fixture(PixelFormat::Rgb, 3);
    let (c_yuv8, c_planes8) = c_return_codes(&oracle, &rgb, "rgb");
    let (rust_yuv8, rust_planes8) = rust_return_codes(&rgb);
    assert_eq!(
        (c_yuv8, c_planes8),
        (0, 0),
        "upstream is expected to accept a 3-component frame"
    );
    assert_eq!(
        (rust_yuv8, rust_planes8),
        (c_yuv8, c_planes8),
        "3-component frame: the guard must not narrow what we accept"
    );
}

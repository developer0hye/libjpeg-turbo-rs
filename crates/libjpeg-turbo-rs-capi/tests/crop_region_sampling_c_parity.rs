//! P4-197 (#618): `tj3SetCroppingRegion` + `tj3Decompress8` on frames whose
//! sampling factors TurboJPEG classifies loosely, against stock TurboJPEG.
//!
//! The iMCU width `tj3SetCroppingRegion` checks a left boundary against comes
//! from `getSubsamp` (`references/libjpeg-turbo/src/turbojpeg.c:431-510`) and
//! `tjMCUWidth[]`; the width the decoder actually crops on comes from the
//! frame's largest horizontal sampling factor. On the four committed
//! `tests/fixtures/crop_sampling_*.jpg` frames (`cjpeg -sample` from 3.2.0 on
//! `testorig.ppm`) the two disagree or the classification is UNKNOWN, which is
//! where a port can accept a region upstream refuses, or decode one wider than
//! the caller's buffer. Both libraries are `dlopen`ed and driven through the
//! same calls; every return code, every error string and every accepted
//! decode's pixels must agree.
//!
//! The one normalisation: our `tj3Decompress8` prefixes its errors
//! `"tj3Decompress8: "` where upstream's THROW writes `"tj3Decompress8(): "`
//! (`turbojpeg.c:279-291`). That prefix difference is pre-existing for every
//! decompress error and is not what this suite measures, so the comparison
//! strips the function-name prefix from both; `tj3SetCroppingRegion`'s strings
//! are compared verbatim.

use std::ffi::{c_char, c_int, c_void, CStr};
use std::path::{Path, PathBuf};

mod helpers;

#[path = "support/cdylib.rs"]
mod cdylib;

const TJINIT_DECOMPRESS: c_int = 1;
const TJPF_RGB: c_int = 0;

#[repr(C)]
#[derive(Clone, Copy)]
struct TjRegion {
    x: c_int,
    y: c_int,
    w: c_int,
    h: c_int,
}

fn stock_turbojpeg() -> Option<PathBuf> {
    let Some(install) = helpers::find_turbojpeg_dev() else {
        assert!(
            !helpers::oracle_is_required(),
            "LIBJPEG_TURBO_PREFIX names no TurboJPEG 3 install"
        );
        return None;
    };
    let found: Option<PathBuf> = ["libturbojpeg.dylib", "libturbojpeg.so", "libturbojpeg.so.0"]
        .iter()
        .map(|name| install.lib_dir.join(name))
        .find(|path| path.exists());
    assert!(
        found.is_some() || !helpers::oracle_is_required(),
        "LIBJPEG_TURBO_PREFIX names an install with no loadable libturbojpeg in {:?}",
        install.lib_dir
    );
    found
}

fn fixture(name: &str) -> Vec<u8> {
    let path: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"))
}

/// `"fn(): msg"` or `"fn: msg"` → `"msg"`, for `tj3Decompress8` only.
fn without_decompress_prefix(message: &str) -> String {
    for prefix in ["tj3Decompress8(): ", "tj3Decompress8: "] {
        if let Some(rest) = message.strip_prefix(prefix) {
            return rest.to_string();
        }
    }
    message.to_string()
}

/// The observable outcome of header → set region → decode on one library.
fn transcript(library: &Path, jpeg: &[u8], region: TjRegion) -> Vec<String> {
    let lib: libloading::Library = unsafe { libloading::Library::new(library) }
        .unwrap_or_else(|e| panic!("dlopen {library:?}: {e}"));
    let mut lines: Vec<String> = Vec::new();
    unsafe {
        let init: libloading::Symbol<unsafe extern "C" fn(c_int) -> *mut c_void> =
            lib.get(b"tj3Init").expect("tj3Init");
        let destroy: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Destroy").expect("tj3Destroy");
        let header: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, usize) -> c_int,
        > = lib
            .get(b"tj3DecompressHeader")
            .expect("tj3DecompressHeader");
        let set_crop: libloading::Symbol<unsafe extern "C" fn(*mut c_void, TjRegion) -> c_int> =
            lib.get(b"tj3SetCroppingRegion")
                .expect("tj3SetCroppingRegion");
        let decompress: libloading::Symbol<
            unsafe extern "C" fn(*mut c_void, *const u8, usize, *mut u8, c_int, c_int) -> c_int,
        > = lib.get(b"tj3Decompress8").expect("tj3Decompress8");
        let error_str: libloading::Symbol<unsafe extern "C" fn(*mut c_void) -> *const c_char> =
            lib.get(b"tj3GetErrorStr").expect("tj3GetErrorStr");
        let get: libloading::Symbol<unsafe extern "C" fn(*mut c_void, c_int) -> c_int> =
            lib.get(b"tj3Get").expect("tj3Get");
        let error_of = |handle: *mut c_void| -> String {
            CStr::from_ptr(error_str(handle))
                .to_string_lossy()
                .into_owned()
        };

        let handle: *mut c_void = init(TJINIT_DECOMPRESS);
        assert!(!handle.is_null(), "tj3Init");
        lines.push(format!(
            "header={}",
            header(handle, jpeg.as_ptr(), jpeg.len())
        ));
        let set_rc: c_int = set_crop(handle, region);
        lines.push(format!("set={set_rc}"));
        if set_rc != 0 {
            lines.push(format!("set_err={}", error_of(handle)));
        } else {
            // Sized for the region, as TurboJPEG documents, with a sentinel
            // tail: a decode wider than the region lands in the tail instead
            // of past the allocation, and the comparison reports it.
            // A zero width/height runs to the edge (`turbojpeg.c:2102-2105`);
            // these decodes are unscaled, so the edge is the SOF's.
            const TJPARAM_JPEGWIDTH: c_int = 5;
            const TJPARAM_JPEGHEIGHT: c_int = 6;
            let width: usize = if region.w == 0 {
                (get(handle, TJPARAM_JPEGWIDTH) - region.x) as usize
            } else {
                region.w as usize
            };
            let height: usize = if region.h == 0 {
                (get(handle, TJPARAM_JPEGHEIGHT) - region.y) as usize
            } else {
                region.h as usize
            };
            let region_bytes: usize = width * height * 3;
            let mut dst: Vec<u8> = vec![0xA5; region_bytes * 2];
            let rc: c_int = decompress(
                handle,
                jpeg.as_ptr(),
                jpeg.len(),
                dst.as_mut_ptr(),
                0,
                TJPF_RGB,
            );
            lines.push(format!("decode={rc}"));
            let tail_untouched: bool = dst[region_bytes..].iter().all(|&b| b == 0xA5);
            lines.push(format!("tail_untouched={tail_untouched}"));
            if rc != 0 {
                lines.push(format!(
                    "decode_err={}",
                    without_decompress_prefix(&error_of(handle))
                ));
            } else {
                lines.push(format!("pixels={:?}", &dst[..region_bytes]));
            }
        }
        destroy(handle);
    }
    lines
}

const CHILD_ENV: &str = "CROP_SAMPLING_PARITY_CHILD";
const TRANSCRIPT_MARKER: &str = "TRANSCRIPT ";

/// Run one library's transcript in a child process — this test binary
/// re-entered through [`child_transcript`]. Both libraries export the same
/// `tj3*` and `jpeg_*` names, and loading them into one process crashed (the
/// harness in `examples/cabi_misuse_harness.c` runs one library per process
/// for the same reason).
fn transcript_in_child(library: &Path, fixture_name: &str, region: TjRegion) -> Vec<String> {
    let output: std::process::Output =
        std::process::Command::new(std::env::current_exe().expect("current_exe"))
            .args([
                "--exact",
                "child_transcript",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(
                CHILD_ENV,
                format!(
                    "{}|{}|{}|{}|{}|{}",
                    library.display(),
                    fixture_name,
                    region.x,
                    region.y,
                    region.w,
                    region.h
                ),
            )
            .output()
            .expect("spawn the child transcript");
    let stdout: String = String::from_utf8_lossy(&output.stdout).into_owned();
    assert!(
        output.status.success(),
        "child for {library:?} {fixture_name} failed: {:?}\nstdout:\n{stdout}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    let lines: Vec<String> = stdout
        .lines()
        .filter_map(|line| line.strip_prefix(TRANSCRIPT_MARKER))
        .map(str::to_string)
        .collect();
    assert!(
        !lines.is_empty(),
        "child for {library:?} printed no transcript:\n{stdout}"
    );
    lines
}

/// The child half: inert unless [`transcript_in_child`] set the variable.
#[test]
fn child_transcript() {
    let Ok(spec) = std::env::var(CHILD_ENV) else {
        return;
    };
    let fields: Vec<&str> = spec.split('|').collect();
    assert_eq!(fields.len(), 6, "malformed child spec {spec:?}");
    let number = |index: usize| -> c_int { fields[index].parse().expect("region field") };
    let region: TjRegion = TjRegion {
        x: number(2),
        y: number(3),
        w: number(4),
        h: number(5),
    };
    for line in transcript(Path::new(fields[0]), &fixture(fields[1]), region) {
        println!("{TRANSCRIPT_MARKER}{line}");
    }
}

#[test]
fn crop_regions_on_loosely_classified_sampling_match_stock_turbojpeg() {
    let Some(stock) = stock_turbojpeg() else {
        eprintln!("SKIP: no TurboJPEG 3 install with a loadable libturbojpeg");
        return;
    };
    let ours: PathBuf = cdylib::cdylib_path();
    let fixtures: [&str; 4] = [
        "crop_sampling_2x2_1x1_2x2.jpg",
        "crop_sampling_2x2_1x2_1x2.jpg",
        "crop_sampling_2x2_2x1_2x1.jpg",
        "crop_sampling_2x1_2x1_2x1.jpg",
    ];
    let regions: [TjRegion; 3] = [
        // On TurboJPEG's 8-pixel grid for 4:4:4/4:4:0, off the decoder's 16.
        TjRegion {
            x: 8,
            y: 0,
            w: 32,
            h: 16,
        },
        // On both grids.
        TjRegion {
            x: 16,
            y: 8,
            w: 32,
            h: 16,
        },
        // Zero width: to the right edge.
        TjRegion {
            x: 32,
            y: 0,
            w: 0,
            h: 8,
        },
    ];
    let mut refused: usize = 0;
    let mut decoded: usize = 0;
    for name in fixtures {
        for region in regions {
            let label: String = format!(
                "{name} {{{}, {}, {}, {}}}",
                region.x, region.y, region.w, region.h
            );
            let theirs: Vec<String> = transcript_in_child(&stock, name, region);
            let mine: Vec<String> = transcript_in_child(&ours, name, region);
            assert_eq!(mine, theirs, "{label}");
            assert!(
                mine.iter().all(|line| line != "tail_untouched=false"),
                "{label}: a decode wrote past the region"
            );
            if mine.iter().any(|line| line.starts_with("pixels=")) {
                decoded += 1;
            } else {
                refused += 1;
            }
        }
    }
    // Both outcomes must occur, or the comparison is one-sided.
    assert!(
        decoded > 0 && refused > 0,
        "decoded {decoded}, refused {refused}"
    );
}

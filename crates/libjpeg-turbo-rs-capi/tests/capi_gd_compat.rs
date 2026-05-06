//! P2-10: libgd round-trip smoke against our cdylib.
//!
//! Drives `examples/gd_smoke/build.sh` then `examples/gd_smoke/run.sh`:
//!   1. Builds a small C harness against libgd (skip if cc / gd absent).
//!   2. Stages our cdylib as the libjpeg provider.
//!   3. The C harness round-trips a fixture PPM through `gdImageJpegPtr`
//!      / `gdImageCreateFromJpegPtr` and panics with a non-zero exit if
//!      pixels diverge beyond the PSNR floor.
//!
//! libgd is the smallest realistic libjpeg consumer in the matrix: no
//! pipeline orchestration, no metadata handling — just the canonical
//! libjpeg encode/decode entry points wrapped behind two function calls.
//! That makes a libgd round-trip the cleanest way to detect a missing
//! symbol or a calling-convention bug in the encode side.
//!
//! Skip-with-reason exit codes from build.sh / run.sh:
//!   build 1 → libgd headers absent
//!   build 2 → cc not on PATH
//!   build 3 → real compile error (panic)
//!   run   1 → binary missing (panic — build.sh should have caught)
//!   run   8 → macOS SIP carve-out
//!   run  11 → libgd bound to mozjpeg (runtime layout incompat)
//! Anything else from run.sh is a real shim bug.

use std::path::{Path, PathBuf};
use std::process::Command;

fn dlext() -> &'static str {
    if cfg!(target_os = "windows") {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    }
}

fn lib_prefix() -> &'static str {
    if cfg!(target_os = "windows") {
        ""
    } else {
        "lib"
    }
}

fn cdylib_path() -> PathBuf {
    if let Ok(p) = std::env::var("CARGO_CDYLIB_FILE_LIBJPEG_TURBO_RS_CAPI") {
        return PathBuf::from(p);
    }
    let exe: PathBuf = std::env::current_exe().expect("current_exe");
    let mut dir: PathBuf = exe.clone();
    while dir.pop() {
        let candidate: PathBuf =
            dir.join(format!("{}libjpeg_turbo_rs_capi.{}", lib_prefix(), dlext()));
        if candidate.exists() {
            return candidate;
        }
    }
    panic!("could not locate cdylib near {}", exe.display());
}

/// libgd's `gdImageJpegPtr` defaults to 4:2:0 chroma subsampling, which
/// roughly halves achievable PSNR on high-frequency chroma content
/// versus ImageMagick's v7 4:4:4 default. We use the same gradient
/// shape as the other consumer fixtures (R/G as smooth gradients) but
/// drop the 8×8 checker on B in favour of a 32-pixel-wide horizontal
/// band on B. 32 px is well below the 4:2:0 chroma block boundary
/// stride (16 px in chroma space), so the band reproduces cleanly at
/// q=75 and PSNR floors comfortably above 28 dB on a reference codec.
fn write_fixture_ppm(path: &Path) {
    const W: usize = 160;
    const H: usize = 120;
    let mut buf: Vec<u8> = Vec::with_capacity(15 + W * H * 3);
    buf.extend_from_slice(format!("P6\n{} {}\n255\n", W, H).as_bytes());
    for y in 0..H {
        for x in 0..W {
            let r: u8 = ((x * 255) / (W - 1)) as u8;
            let g: u8 = ((y * 255) / (H - 1)) as u8;
            let b: u8 = if (y / 32) % 2 == 0 { 80 } else { 176 };
            buf.extend_from_slice(&[r, g, b]);
        }
    }
    std::fs::write(path, &buf).expect("write fixture PPM");
}

#[test]
#[cfg(unix)]
fn libgd_roundtrips_through_our_cdylib() {
    let manifest_dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root: PathBuf = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf();
    let build_sh: PathBuf = workspace_root.join("examples/gd_smoke/build.sh");
    let run_sh: PathBuf = workspace_root.join("examples/gd_smoke/run.sh");
    assert!(build_sh.is_file(), "missing {}", build_sh.display());
    assert!(run_sh.is_file(), "missing {}", run_sh.display());

    let tmp: tempfile::TempDir = tempfile::tempdir().expect("tempdir");

    // ---- Phase 1: build the C harness --------------------------------
    let build_out = Command::new("bash")
        .arg(&build_sh)
        .arg("--out-dir")
        .arg(tmp.path())
        .output()
        .expect("spawn gd_smoke build.sh");
    let build_stderr: String = String::from_utf8_lossy(&build_out.stderr).into_owned();
    match build_out.status.code() {
        Some(0) => { /* compiled */ }
        Some(1) => {
            eprintln!(
                "SKIP: libgd headers / library not found:\n{}",
                build_stderr.trim()
            );
            return;
        }
        Some(2) => {
            eprintln!("SKIP: cc compiler not on PATH:\n{}", build_stderr.trim());
            return;
        }
        Some(3) => panic!("gd_smoke compilation failed:\n{}", build_stderr.trim()),
        other => panic!(
            "gd_smoke build.sh returned unexpected exit {other:?}:\n{}",
            build_stderr.trim()
        ),
    }

    let binary: PathBuf = tmp.path().join("gd_smoke");
    assert!(binary.is_file(), "build.sh did not produce gd_smoke binary");

    // ---- Phase 2: round-trip via run.sh ------------------------------
    let input_ppm: PathBuf = tmp.path().join("input.ppm");
    write_fixture_ppm(&input_ppm);

    let lib: PathBuf = cdylib_path();
    assert!(lib.exists(), "cdylib not built: {}", lib.display());

    let workdir: PathBuf = tmp.path().join("rundir");
    std::fs::create_dir_all(&workdir).expect("create rundir");

    let out = Command::new("bash")
        .arg(&run_sh)
        .arg("--lib")
        .arg(&lib)
        .arg("--binary")
        .arg(&binary)
        .arg("--input")
        .arg(&input_ppm)
        .arg("--workdir")
        .arg(&workdir)
        .arg("--quality")
        .arg("75")
        .arg("--min-psnr")
        // libgd defaults to 4:2:0; with the smoother fixture above the
        // floor settles around 32 dB on a reference codec. 28.0 leaves
        // headroom for normal codec variation but still catches a
        // catastrophically broken Huffman table (would be < 15 dB).
        .arg("28.0")
        .output()
        .expect("spawn gd_smoke run.sh");

    let stdout: String = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr: String = String::from_utf8_lossy(&out.stderr).into_owned();
    let code: i32 = out.status.code().unwrap_or(-1);

    match code {
        0 => println!("libgd round-trip OK:\n{}", stdout.trim()),
        8 => eprintln!(
            "SKIP (macOS SIP): DYLD_INSERT_LIBRARIES blocked; stderr={}",
            stderr.trim()
        ),
        11 => eprintln!(
            "SKIP: libgd bound to mozjpeg (libjpeg-turbo fork); stderr={}",
            stderr.trim()
        ),
        _ => panic!(
            "libgd smoke failed (exit {code}):\n--- stdout ---\n{}\n--- stderr ---\n{}",
            stdout.trim(),
            stderr.trim()
        ),
    }
}

#[test]
#[cfg(not(unix))]
fn libgd_smoke_is_unix_only() {
    eprintln!("SKIP: libgd smoke is Unix-only (loader-injection mechanism)");
}

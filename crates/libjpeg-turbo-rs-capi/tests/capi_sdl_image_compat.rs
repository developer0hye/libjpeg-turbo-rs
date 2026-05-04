//! P2-10: SDL_image decode round-trip smoke against our cdylib.
//!
//! SDL_image only routes JPEG **decode** through libjpeg
//! (`IMG_LoadJPG_RW` -> `jpeg_create_decompress` + `jpeg_mem_src` +
//! `jpeg_read_scanlines`). The library's saver (`IMG_SaveJPG_RW`) uses
//! STB_image_write internally — there's no encode path through libjpeg
//! to exercise.
//!
//! Drives `examples/sdl_image_smoke/build.sh` then `run.sh`:
//!   1. We encode a fixture PPM via `libjpeg_turbo_rs::Encoder` *in
//!      this test process* (so the encoder side is deterministic and
//!      independent of the loader-injection harness).
//!   2. The harness builds the C binary against SDL2 + SDL2_image.
//!   3. run.sh stages our cdylib, then the binary decodes the JPEG
//!      via SDL_image (which calls libjpeg through our cdylib) and
//!      asserts decoded vs reference PSNR > threshold.
//!
//! Skip-with-reason exit codes:
//!   build 1 → SDL2_image headers absent
//!   build 2 → cc not on PATH
//!   build 3 → real compile error (panic)
//!   run   4 → SDL_image build doesn't link libjpeg (STB-only) — true
//!             skip-with-reason, no symbol surface to override
//!   run   8 → macOS SIP carve-out
//!   run  11 → SDL_image bound to mozjpeg (runtime layout incompat)
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

const W: usize = 160;
const H: usize = 120;

/// Build a smooth fixture suitable for q=75 4:2:0 (no high-frequency
/// chroma so floor PSNR stays well above 28 dB on a reference codec).
fn fixture_pixels() -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::with_capacity(W * H * 3);
    for y in 0..H {
        for x in 0..W {
            let r: u8 = ((x * 255) / (W - 1)) as u8;
            let g: u8 = ((y * 255) / (H - 1)) as u8;
            let b: u8 = if (y / 32) % 2 == 0 { 80 } else { 176 };
            buf.extend_from_slice(&[r, g, b]);
        }
    }
    buf
}

fn write_fixture_ppm(path: &Path, pixels: &[u8]) {
    let mut buf: Vec<u8> = Vec::with_capacity(15 + pixels.len());
    buf.extend_from_slice(format!("P6\n{} {}\n255\n", W, H).as_bytes());
    buf.extend_from_slice(pixels);
    std::fs::write(path, &buf).expect("write fixture PPM");
}

#[test]
#[cfg(unix)]
fn sdl_image_decodes_through_our_cdylib() {
    let manifest_dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root: PathBuf = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf();
    let build_sh: PathBuf = workspace_root.join("examples/sdl_image_smoke/build.sh");
    let run_sh: PathBuf = workspace_root.join("examples/sdl_image_smoke/run.sh");
    assert!(build_sh.is_file(), "missing {}", build_sh.display());
    assert!(run_sh.is_file(), "missing {}", run_sh.display());

    let tmp: tempfile::TempDir = tempfile::tempdir().expect("tempdir");

    // ---- Phase 1: build -----------------------------------------------
    let build_out = Command::new("bash")
        .arg(&build_sh)
        .arg("--out-dir")
        .arg(tmp.path())
        .output()
        .expect("spawn sdl_image_smoke build.sh");
    let build_stderr: String = String::from_utf8_lossy(&build_out.stderr).into_owned();
    match build_out.status.code() {
        Some(0) => { /* compiled */ }
        Some(1) => {
            eprintln!(
                "SKIP: SDL2 / SDL2_image headers not found:\n{}",
                build_stderr.trim()
            );
            return;
        }
        Some(2) => {
            eprintln!("SKIP: cc compiler not on PATH:\n{}", build_stderr.trim());
            return;
        }
        Some(3) => panic!(
            "sdl_image_smoke compilation failed:\n{}",
            build_stderr.trim()
        ),
        other => panic!(
            "sdl_image_smoke build.sh returned unexpected exit {other:?}:\n{}",
            build_stderr.trim()
        ),
    }

    let binary: PathBuf = tmp.path().join("sdl_image_smoke");
    assert!(
        binary.is_file(),
        "build.sh did not produce sdl_image_smoke binary"
    );

    // ---- Phase 2: encode the fixture (out-of-band, not via SDL_image) -
    let pixels: Vec<u8> = fixture_pixels();
    let ref_ppm: PathBuf = tmp.path().join("reference.ppm");
    write_fixture_ppm(&ref_ppm, &pixels);

    let jpeg_bytes: Vec<u8> =
        libjpeg_turbo_rs::Encoder::new(&pixels, W, H, libjpeg_turbo_rs::PixelFormat::Rgb)
            .quality(75)
            .encode()
            .expect("encode fixture jpeg");
    let jpeg_path: PathBuf = tmp.path().join("input.jpg");
    std::fs::write(&jpeg_path, &jpeg_bytes).expect("write jpeg");

    // ---- Phase 3: decode through SDL_image with our cdylib injected --
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
        .arg("--jpeg")
        .arg(&jpeg_path)
        .arg("--ref")
        .arg(&ref_ppm)
        .arg("--workdir")
        .arg(&workdir)
        .arg("--min-psnr")
        .arg("28.0")
        .output()
        .expect("spawn sdl_image_smoke run.sh");

    let stdout: String = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr: String = String::from_utf8_lossy(&out.stderr).into_owned();
    let code: i32 = out.status.code().unwrap_or(-1);

    match code {
        0 => println!("SDL_image decode OK:\n{}", stdout.trim()),
        4 => eprintln!(
            "SKIP: SDL_image build does not link libjpeg (STB-only):\n{}",
            stderr.trim()
        ),
        8 => eprintln!(
            "SKIP (macOS SIP): DYLD_INSERT_LIBRARIES blocked; stderr={}",
            stderr.trim()
        ),
        11 => eprintln!(
            "SKIP: SDL_image bound to mozjpeg (libjpeg-turbo fork); stderr={}",
            stderr.trim()
        ),
        _ => panic!(
            "SDL_image smoke failed (exit {code}):\n--- stdout ---\n{}\n--- stderr ---\n{}",
            stdout.trim(),
            stderr.trim()
        ),
    }
}

#[test]
#[cfg(not(unix))]
fn sdl_image_smoke_is_unix_only() {
    eprintln!("SKIP: SDL_image smoke is Unix-only (loader-injection mechanism)");
}

//! P2-10: FFmpeg round-trip smoke against our cdylib.
//!
//! Drives `examples/ffmpeg_smoke/run.sh`. FFmpeg's `mjpeg` codec ships in
//! two flavours:
//!   - **internal** (default for most distro / Homebrew builds): ffmpeg
//!     uses avcodec's own MJPEG encoder/decoder and never touches
//!     libjpeg. Our cdylib has no symbol surface to override here, so
//!     the test exits with a documented skip-with-reason (script exit 9).
//!   - **libjpeg-backed** (ffmpeg configured with `--enable-libjpeg`):
//!     mjpeg routes through `jpeg_create_decompress` /
//!     `jpeg_read_scanlines` directly. The harness exercises this path.
//!
//! When the build is libjpeg-backed, real failures (encode/decode error,
//! PSNR regression) panic; everything else is a logged skip.

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

fn write_fixture_ppm(path: &Path) {
    const W: usize = 160;
    const H: usize = 120;
    let mut buf: Vec<u8> = Vec::with_capacity(15 + W * H * 3);
    buf.extend_from_slice(format!("P6\n{} {}\n255\n", W, H).as_bytes());
    for y in 0..H {
        for x in 0..W {
            let r: u8 = ((x * 255) / (W - 1)) as u8;
            let g: u8 = ((y * 255) / (H - 1)) as u8;
            let b: u8 = if ((x / 8) + (y / 8)) % 2 == 0 {
                32
            } else {
                224
            };
            buf.extend_from_slice(&[r, g, b]);
        }
    }
    std::fs::write(path, &buf).expect("write fixture PPM");
}

#[test]
#[cfg(unix)]
fn ffmpeg_roundtrips_through_our_cdylib() {
    if Command::new("ffmpeg").arg("-version").output().is_err() {
        eprintln!("SKIP: ffmpeg binary not on PATH");
        return;
    }

    let manifest_dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root: PathBuf = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf();
    let script: PathBuf = workspace_root.join("examples/ffmpeg_smoke/run.sh");
    assert!(script.exists(), "missing {}", script.display());

    let tmp: tempfile::TempDir = tempfile::tempdir().expect("tempdir");
    let input_ppm: PathBuf = tmp.path().join("input.ppm");
    write_fixture_ppm(&input_ppm);

    let lib: PathBuf = cdylib_path();
    assert!(lib.exists(), "cdylib not built: {}", lib.display());

    let mut cmd = Command::new("bash");
    cmd.arg(&script)
        .arg("--lib")
        .arg(&lib)
        .arg("--input")
        .arg(&input_ppm)
        .arg("--workdir")
        .arg(tmp.path())
        .arg("--min-psnr")
        .arg("28.0") // ffmpeg's default qtables are slightly lossier than libjpeg's
        .arg("--quality")
        .arg("5"); // -q:v 5 ≈ libjpeg q=80

    let out = cmd.output().expect("spawn ffmpeg smoke script");
    let stdout: String = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr: String = String::from_utf8_lossy(&out.stderr).into_owned();
    let code: i32 = out.status.code().unwrap_or(-1);

    match code {
        0 => println!("ffmpeg round-trip OK:\n{}", stdout.trim()),
        2 => eprintln!(
            "SKIP: script reports ffmpeg unavailable; stderr={}",
            stderr.trim()
        ),
        8 => eprintln!(
            "SKIP (macOS SIP): DYLD_INSERT_LIBRARIES blocked; stderr={}",
            stderr.trim()
        ),
        9 => eprintln!(
            "SKIP: this ffmpeg build does not use libjpeg (internal MJPEG codec only); \
             stderr={}",
            stderr.trim()
        ),
        11 => eprintln!(
            "SKIP: ffmpeg bound to mozjpeg (libjpeg-turbo fork); stderr={}",
            stderr.trim()
        ),
        _ => panic!(
            "ffmpeg smoke failed (exit {code}):\n--- stdout ---\n{}\n--- stderr ---\n{}",
            stdout.trim(),
            stderr.trim()
        ),
    }
}

#[test]
#[cfg(not(unix))]
fn ffmpeg_smoke_is_unix_only() {
    eprintln!("SKIP: ffmpeg smoke is Unix-only (loader-injection mechanism)");
}

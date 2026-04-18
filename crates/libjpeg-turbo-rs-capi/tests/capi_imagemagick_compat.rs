//! B9-3: ImageMagick round-trip smoke test against our cdylib.
//!
//! Drives `examples/imagemagick_smoke/run.sh` which:
//!   1. Symlinks our cdylib under `libjpeg.so.62` / `libjpeg.62.dylib`.
//!   2. Sets `LD_PRELOAD` (Linux) or `DYLD_INSERT_LIBRARIES` (macOS).
//!   3. Runs `convert input.ppm -quality 75 out.jpg` and the reverse.
//!   4. Asserts decoded-vs-original PSNR > threshold.
//!
//! Exit-code contract from the script (see the script header for the
//! full list):
//!   0 success; 2 ImageMagick absent; 8 macOS SIP blocks injection
//!   (both mapped to `eprintln!("SKIP: ...")` + `return`).
//!
//! Every other non-zero exit is a real failure that MUST panic — the
//! whole point of this test is to catch regressions in the ABI-compat
//! layer when a real third-party program (ImageMagick) uses us as a
//! drop-in libjpeg.

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

fn imagemagick_binary() -> Option<(&'static str, PathBuf)> {
    // ImageMagick v7 ships `magick`, v6 ships `convert`. We deliberately
    // check v7 first because `convert` on Windows (and some older Unixes)
    // is shadowed by a filesystem utility of the same name.
    for (label, bin) in [("magick", "magick"), ("convert", "convert")] {
        let out = Command::new("which").arg(bin).output().ok()?;
        if !out.status.success() {
            continue;
        }
        let path: String = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if path.is_empty() {
            continue;
        }
        // For `convert`, disambiguate from Windows filesystem utility by
        // asking for `--version` and requiring the ImageMagick banner.
        if label == "convert" {
            let v = Command::new(&path).arg("--version").output().ok()?;
            let stdout: String = String::from_utf8_lossy(&v.stdout).to_lowercase();
            if !stdout.contains("imagemagick") {
                continue;
            }
        }
        return Some((label, PathBuf::from(path)));
    }
    None
}

/// Synthesise a deterministic PPM that covers smooth gradients + high
/// frequency checker. Big enough (160x120) that quality-75 DCT loss is
/// measurable but still well above the 30 dB PSNR floor.
fn write_fixture_ppm(path: &Path) {
    const W: usize = 160;
    const H: usize = 120;
    let mut buf: Vec<u8> = Vec::with_capacity(15 + W * H * 3);
    buf.extend_from_slice(format!("P6\n{} {}\n255\n", W, H).as_bytes());
    for y in 0..H {
        for x in 0..W {
            // Smooth horizontal/vertical gradients on R/G so chroma
            // subsampling has real signal to lose and recover.
            let r: u8 = ((x * 255) / (W - 1)) as u8;
            let g: u8 = ((y * 255) / (H - 1)) as u8;
            // Checker on B forces the encoder to keep some mid-frequency
            // energy — if our Huffman tables are wrong we'll see blocks
            // here and PSNR will crater.
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
fn imagemagick_roundtrips_through_our_cdylib() {
    // Fast-path skip when the host has no ImageMagick at all — the
    // script would skip internally too, but short-circuiting here keeps
    // CI logs readable and saves a few hundred ms.
    let Some((mode, im_bin)) = imagemagick_binary() else {
        eprintln!("SKIP: ImageMagick (magick/convert) not found on PATH");
        return;
    };

    let manifest_dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // Script lives at workspace-root/examples/imagemagick_smoke/run.sh;
    // the crate's manifest dir is two levels deep (crates/<crate>).
    let workspace_root: PathBuf = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf();
    let script: PathBuf = workspace_root.join("examples/imagemagick_smoke/run.sh");
    assert!(script.exists(), "missing {}", script.display());

    let tmp: tempfile::TempDir = tempfile::tempdir().expect("tempdir");
    let input_ppm: PathBuf = tmp.path().join("input.ppm");
    write_fixture_ppm(&input_ppm);

    let lib: PathBuf = cdylib_path();
    assert!(lib.exists(), "cdylib not built: {}", lib.display());

    // Always shell out through `sh` so a non-executable bit on the
    // script (common after `cargo vendor`/tarball extraction) still
    // works. `sh` here is any POSIX shell; the script's shebang is
    // `bash` for arrays, but invoking via `sh` re-reads the shebang.
    let mut cmd = Command::new("bash");
    cmd.arg(&script)
        .arg("--lib")
        .arg(&lib)
        .arg("--input")
        .arg(&input_ppm)
        .arg("--workdir")
        .arg(tmp.path())
        .arg("--min-psnr")
        .arg("30.0")
        .arg("--quality")
        .arg("75");

    let out = cmd.output().expect("spawn imagemagick smoke script");
    let stdout: String = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr: String = String::from_utf8_lossy(&out.stderr).into_owned();
    let code: i32 = out.status.code().unwrap_or(-1);

    // Real skip-with-reason: ImageMagick was removed between PATH probe
    // and script launch (exit 2), or macOS SIP forbids dyld injection
    // into the resolved binary (exit 8). Anything else is a bug.
    match code {
        0 => {
            // Smoke passed: show PSNR on stdout for CI observability.
            println!(
                "ImageMagick ({} @ {}) round-trip OK:\n{}",
                mode,
                im_bin.display(),
                stdout.trim()
            );
        }
        2 => {
            eprintln!(
                "SKIP: script reports ImageMagick unavailable despite PATH probe; stderr={}",
                stderr.trim()
            );
        }
        8 => {
            // Documented macOS-SIP carve-out. `DYLD_INSERT_LIBRARIES`
            // is stripped by dyld when launching binaries under
            // /usr/**, /bin/**, /sbin/**, /System/**, /Applications/**.
            // Users who want this test to run on macOS must install
            // ImageMagick via Homebrew (/opt/homebrew/** or
            // /usr/local/**) where SIP does not apply.
            eprintln!(
                "SKIP (macOS SIP): DYLD_INSERT_LIBRARIES blocked for {}; {}",
                im_bin.display(),
                stderr.trim()
            );
        }
        _ => panic!(
            "ImageMagick smoke failed (exit {code}):\n--- stdout ---\n{}\n--- stderr ---\n{}",
            stdout.trim(),
            stderr.trim()
        ),
    }
}

#[test]
#[cfg(not(unix))]
fn imagemagick_smoke_is_unix_only() {
    // The DYLD/LD_PRELOAD loader override mechanism does not exist on
    // Windows in the same form; drop-in libjpeg replacement on Windows
    // is covered by a different test path (DLL search order), not this
    // one. Explicitly skip here so the test appears in the runner.
    eprintln!("SKIP: ImageMagick smoke is Unix-only (loader-injection mechanism)");
}

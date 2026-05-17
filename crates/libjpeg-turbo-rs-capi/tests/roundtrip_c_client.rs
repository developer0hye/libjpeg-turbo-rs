//! B9-1: compile `examples/c_client/roundtrip.c` against our cdylib,
//! run it, parse its binary blob output and verify the round-trip
//! pixel fidelity matches the gradient produced by the C program.
//!
//! Unlike the A1-3 smoke client (which only checks SOI/EOI framing),
//! this test reads back every decoded pixel and asserts the
//! compress → decompress per-channel diff stays within a tight bound
//! measured against the actual quality-95 4:4:4 output.

use std::path::{Path, PathBuf};
use std::process::Command;

// Measured bound: at quality=95 TJSAMP_444 the 96x96 ramp produces
// max per-channel diff <= 3. Keep a small headroom so benign encoder
// tweaks don't flip the test red.
const MAX_PIXEL_DIFF: u8 = 5;

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

fn find_cc() -> Option<PathBuf> {
    for candidate in ["cc", "clang", "gcc"] {
        if let Ok(out) = Command::new("which").arg(candidate).output() {
            if out.status.success() {
                let s: String = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if !s.is_empty() {
                    return Some(PathBuf::from(s));
                }
            }
        }
    }
    None
}

#[cfg(unix)]
fn setup_symlinks(lib: &Path, parent: &Path) -> PathBuf {
    let subdir: PathBuf = parent.join("symlinks");
    std::fs::create_dir_all(&subdir).expect("mkdir symlinks");
    // P4-3 (2026-05-17): cdylib default identity flipped to v8. Stage
    // both v8 and v6b versioned names plus the short link-time name so
    // any `-ljpeg`-linked C test or prebuilt v6b consumer resolves
    // through this dir.
    let names: &[&str] = if cfg!(target_os = "macos") {
        &["libjpeg.8.dylib", "libjpeg.62.dylib", "libjpeg.dylib"]
    } else {
        &["libjpeg.so.8", "libjpeg.so.62", "libjpeg.so"]
    };
    for name in names {
        let link = subdir.join(name);
        if !link.exists() {
            std::os::unix::fs::symlink(lib, &link).expect("symlink");
        }
    }
    subdir
}

#[cfg(not(unix))]
fn setup_symlinks(_lib: &Path, parent: &Path) -> PathBuf {
    parent.to_path_buf()
}

fn parse_be_u32(bytes: &[u8]) -> u32 {
    ((bytes[0] as u32) << 24)
        | ((bytes[1] as u32) << 16)
        | ((bytes[2] as u32) << 8)
        | (bytes[3] as u32)
}

#[test]
fn c_roundtrip_client_matches_source_gradient() {
    let cc: PathBuf = match find_cc() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no C compiler (cc/clang/gcc) found on PATH");
            return;
        }
    };

    let manifest_dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let c_src: PathBuf = manifest_dir.join("examples/c_client/roundtrip.c");
    assert!(c_src.exists(), "missing {}", c_src.display());

    let lib: PathBuf = cdylib_path();
    let lib_dir: &Path = lib.parent().expect("cdylib parent");
    let lib_stem: String = lib
        .file_stem()
        .and_then(|s| s.to_str())
        .map(|s| {
            if cfg!(target_os = "windows") {
                s.to_string()
            } else if let Some(rest) = s.strip_prefix("lib") {
                rest.to_string()
            } else {
                s.to_string()
            }
        })
        .expect("lib stem");

    let tmp: tempfile::TempDir = tempfile::tempdir().expect("tempdir");
    let exe: PathBuf = tmp.path().join("roundtrip_client");
    let out_blob: PathBuf = tmp.path().join("roundtrip.bin");

    let symlink_dir: PathBuf = setup_symlinks(&lib, tmp.path());

    let mut cmd = Command::new(&cc);
    cmd.arg(&c_src).arg("-O2").arg("-o").arg(&exe);
    if cfg!(unix) {
        cmd.arg(format!("-L{}", symlink_dir.display()))
            .arg("-ljpeg")
            .arg(format!("-Wl,-rpath,{}", symlink_dir.display()));
    } else {
        cmd.arg(format!("-L{}", lib_dir.display()))
            .arg(format!("-l{}", lib_stem))
            .arg(format!("-Wl,-rpath,{}", lib_dir.display()));
    }
    let status = cmd.status().expect("cc compile");
    assert!(status.success(), "C roundtrip client failed to compile");

    let run = Command::new(&exe)
        .arg(&out_blob)
        .env("LD_LIBRARY_PATH", &symlink_dir)
        .env("DYLD_LIBRARY_PATH", &symlink_dir)
        .output()
        .expect("run roundtrip_client");
    assert!(
        run.status.success(),
        "C roundtrip client exited with code {}: stderr={}",
        run.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&run.stderr)
    );

    // Parse the blob: [w:4 | h:4 | bpp:1 | pixels].
    let blob: Vec<u8> = std::fs::read(&out_blob).expect("read out blob");
    assert!(blob.len() > 9, "blob too small: {}", blob.len());
    let w: u32 = parse_be_u32(&blob[0..4]);
    let h: u32 = parse_be_u32(&blob[4..8]);
    let bpp: u32 = blob[8] as u32;
    assert_eq!(w, 96);
    assert_eq!(h, 96);
    assert_eq!(bpp, 3);
    let pixel_bytes: usize = (w * h * bpp) as usize;
    assert_eq!(blob.len(), 9 + pixel_bytes);

    // Reproduce the same synthetic gradient from the C source.
    let mut expected: Vec<u8> = Vec::with_capacity(pixel_bytes);
    for y in 0..h {
        for x in 0..w {
            expected.push((x * 255 / (w - 1)) as u8);
            expected.push((y * 255 / (h - 1)) as u8);
            expected.push(((x + y) * 255 / (w + h - 2)) as u8);
        }
    }

    let decoded: &[u8] = &blob[9..];
    let mut max_diff: u8 = 0;
    for (&a, &b) in expected.iter().zip(decoded.iter()) {
        let d: u8 = a.abs_diff(b);
        if d > max_diff {
            max_diff = d;
        }
    }
    assert!(
        max_diff <= MAX_PIXEL_DIFF,
        "roundtrip max per-channel diff {max_diff} exceeded bound {MAX_PIXEL_DIFF}"
    );
}

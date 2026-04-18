//! A1-3: end-to-end test for `tj3Compress8`.
//!
//! Two layers of coverage:
//! 1. Rust dlopen smoke test — faster feedback loop, exercises the FFI
//!    contract directly using a synthetic 64x64 RGB gradient.
//! 2. C smoke client (`examples/c_client/compress8_smoke.c`) — compiled
//!    at test time via the system `cc`, then linked against our cdylib
//!    to verify the shim actually works as a drop-in library for native
//!    C binaries.
//!
//! The C client is skipped with `eprintln!("SKIP: ...")` only when the
//! host has no `cc` compiler available — a legitimate environment gap,
//! not a Rust-side failure.

use std::ffi::{c_char, c_int, c_void};
use std::path::{Path, PathBuf};
use std::process::Command;

type TjHandle = *mut c_void;

const TJPARAM_QUALITY: c_int = 3;
const TJPARAM_SUBSAMP: c_int = 4;
const TJINIT_COMPRESS: c_int = 1;
const TJPF_RGB: c_int = 0;
const TJSAMP_444: c_int = 0;

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
    panic!(
        "could not locate cdylib for libjpeg_turbo_rs_capi near {}",
        exe.display()
    );
}

#[test]
fn tj3_compress8_via_dlopen_round_trips_64x64_rgb() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> TjHandle> =
            lib.get(b"tj3Init").expect("tj3Init");
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(TjHandle)> =
            lib.get(b"tj3Destroy").expect("tj3Destroy");
        let tj3_set: libloading::Symbol<unsafe extern "C" fn(TjHandle, c_int, c_int) -> c_int> =
            lib.get(b"tj3Set").expect("tj3Set");
        let tj3_err: libloading::Symbol<unsafe extern "C" fn(TjHandle) -> *const c_char> =
            lib.get(b"tj3GetErrorStr").expect("tj3GetErrorStr");
        let tj3_compress: libloading::Symbol<
            unsafe extern "C" fn(
                TjHandle,
                *const u8,
                c_int,
                c_int,
                c_int,
                c_int,
                *mut *mut u8,
                *mut usize,
            ) -> c_int,
        > = lib.get(b"tj3Compress8").expect("tj3Compress8");
        let tj3_free: libloading::Symbol<unsafe extern "C" fn(*mut c_void)> =
            lib.get(b"tj3Free").expect("tj3Free");

        let h = tj3_init(TJINIT_COMPRESS);
        assert!(!h.is_null());
        assert_eq!(tj3_set(h, TJPARAM_QUALITY, 80), 0);
        assert_eq!(tj3_set(h, TJPARAM_SUBSAMP, TJSAMP_444), 0);

        // 64x64 synthetic gradient.
        let w: c_int = 64;
        let h_px: c_int = 64;
        let mut src: Vec<u8> = Vec::with_capacity((w * h_px * 3) as usize);
        for y in 0..h_px {
            for x in 0..w {
                src.push(x as u8);
                src.push(y as u8);
                src.push(((x + y) / 2) as u8);
            }
        }

        let mut jpeg_buf: *mut u8 = std::ptr::null_mut();
        let mut jpeg_size: usize = 0;
        let rc: c_int = tj3_compress(
            h,
            src.as_ptr(),
            w,
            0,
            h_px,
            TJPF_RGB,
            &mut jpeg_buf,
            &mut jpeg_size,
        );
        assert_eq!(
            rc,
            0,
            "tj3Compress8 failed: {:?}",
            std::ffi::CStr::from_ptr(tj3_err(h))
        );
        assert!(!jpeg_buf.is_null());
        assert!(jpeg_size > 4);

        let jpeg: &[u8] = std::slice::from_raw_parts(jpeg_buf, jpeg_size);
        assert_eq!(&jpeg[0..2], &[0xFF, 0xD8], "SOI");
        assert_eq!(
            &jpeg[jpeg_size - 2..],
            &[0xFF, 0xD9],
            "EOI at end of buffer"
        );

        tj3_free(jpeg_buf as *mut c_void);
        tj3_destroy(h);
    }
}

#[test]
fn tj3_compress8_null_arguments_return_minus_one() {
    let path: PathBuf = cdylib_path();
    let lib: libloading::Library =
        unsafe { libloading::Library::new(&path) }.expect("dlopen cdylib");

    unsafe {
        let tj3_init: libloading::Symbol<unsafe extern "C" fn(c_int) -> TjHandle> =
            lib.get(b"tj3Init").expect("tj3Init");
        let tj3_destroy: libloading::Symbol<unsafe extern "C" fn(TjHandle)> =
            lib.get(b"tj3Destroy").expect("tj3Destroy");
        let tj3_compress: libloading::Symbol<
            unsafe extern "C" fn(
                TjHandle,
                *const u8,
                c_int,
                c_int,
                c_int,
                c_int,
                *mut *mut u8,
                *mut usize,
            ) -> c_int,
        > = lib.get(b"tj3Compress8").expect("tj3Compress8");

        // NULL handle -> -1.
        let mut out_buf: *mut u8 = std::ptr::null_mut();
        let mut out_size: usize = 0;
        let src: [u8; 12] = [0u8; 12];
        assert_eq!(
            tj3_compress(
                std::ptr::null_mut(),
                src.as_ptr(),
                2,
                0,
                2,
                TJPF_RGB,
                &mut out_buf,
                &mut out_size
            ),
            -1
        );

        // NULL srcBuf -> -1.
        let h = tj3_init(TJINIT_COMPRESS);
        assert_eq!(
            tj3_compress(
                h,
                std::ptr::null(),
                2,
                0,
                2,
                TJPF_RGB,
                &mut out_buf,
                &mut out_size
            ),
            -1
        );

        // NULL out pointers -> -1.
        assert_eq!(
            tj3_compress(
                h,
                src.as_ptr(),
                2,
                0,
                2,
                TJPF_RGB,
                std::ptr::null_mut(),
                &mut out_size
            ),
            -1
        );
        assert_eq!(
            tj3_compress(
                h,
                src.as_ptr(),
                2,
                0,
                2,
                TJPF_RGB,
                &mut out_buf,
                std::ptr::null_mut()
            ),
            -1
        );

        // Negative width -> -1.
        assert_eq!(
            tj3_compress(
                h,
                src.as_ptr(),
                -1,
                0,
                2,
                TJPF_RGB,
                &mut out_buf,
                &mut out_size
            ),
            -1
        );

        tj3_destroy(h);
    }
}

// ---------------------------------------------------------------------------
// A1-3: compile and run the C smoke client.
// ---------------------------------------------------------------------------

/// Create a subdirectory under `parent` that holds symlinks exposing
/// the actual cdylib under both the versioned install_name/SONAME and
/// the short `libjpeg.{dylib,so}` link-time name. Returns the subdir.
#[cfg(unix)]
fn setup_symlinks(lib: &Path, parent: &Path) -> PathBuf {
    let subdir: PathBuf = parent.join("symlinks");
    std::fs::create_dir_all(&subdir).expect("mkdir symlinks");
    let (versioned, short): (&str, &str) = if cfg!(target_os = "macos") {
        ("libjpeg.62.dylib", "libjpeg.dylib")
    } else {
        ("libjpeg.so.62", "libjpeg.so")
    };
    for name in [versioned, short] {
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

#[test]
fn c_client_compress8_smoke() {
    // We have real Rust-level coverage in `tj3_compress8_via_dlopen_*`; the
    // C binary is a belt-and-braces link-test. A missing `cc` is a
    // legitimate host gap (e.g., stripped CI images) so we skip rather
    // than fail.
    let cc: PathBuf = match find_cc() {
        Some(p) => p,
        None => {
            eprintln!("SKIP: no C compiler (cc/clang/gcc) found on PATH");
            return;
        }
    };

    let manifest_dir: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let c_src: PathBuf = manifest_dir.join("examples/c_client/compress8_smoke.c");
    assert!(c_src.exists(), "missing {}", c_src.display());

    let lib: PathBuf = cdylib_path();
    let lib_dir: &Path = lib.parent().expect("cdylib parent");
    // Strip a single leading `lib` (not all of them): the produced file
    // is `liblibjpeg_turbo_rs_capi.{dylib,so}`, so `-l` expects
    // `libjpeg_turbo_rs_capi`.
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
    let exe: PathBuf = tmp.path().join("compress8_smoke");

    // Our cdylib's install_name / SONAME (A1-13) is `libjpeg.62.dylib`
    // (macOS) or `libjpeg.so.62` (Linux). Create a symlink directory
    // that exposes BOTH the versioned name (what dyld/ld.so will look
    // up at runtime) and the short `libjpeg` name (what `-ljpeg`
    // resolves to at link time).
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
    assert!(status.success(), "C smoke client failed to compile");

    let run = Command::new(&exe)
        .env("LD_LIBRARY_PATH", &symlink_dir)
        .env("DYLD_LIBRARY_PATH", &symlink_dir)
        .output()
        .expect("run compress8 smoke");
    assert!(
        run.status.success(),
        "C smoke client failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let stdout: String = String::from_utf8_lossy(&run.stdout).into_owned();
    assert!(
        stdout.contains("tj3Compress8 OK"),
        "unexpected stdout: {stdout}"
    );
}

//! P2-8: end-to-end install-tree layout test.
//!
//! Runs `scripts/install_capi.sh` into a tempdir and asserts:
//!
//! 1. The cdylib lands at the SONAME path (`libjpeg.so.62.X.Y` on
//!    Linux / `libjpeg.62.X.Y.dylib` on macOS).
//! 2. The symlink chain resolves (`libjpeg.so → .62 → real cdylib`).
//! 3. Both the libjpeg and libturbojpeg symlink chains exist.
//! 4. The pkg-config file is well-formed (Name/Version/Libs lines).
//! 5. The CMake config exposes `JPEG::JPEG` imported target wiring.
//! 6. All five public C headers are present.
//!
//! When `pkg-config` is on PATH, the test additionally invokes
//! `pkg-config --libs libjpeg` against `PKG_CONFIG_PATH=<staged>` and
//! asserts the returned `-l` line includes `-ljpeg`.
//!
//! Skip-with-reason cases:
//! - Windows (script is bash; packagers there use their own conventions).
//! - `bash` not on PATH.

use std::path::{Path, PathBuf};
use std::process::Command;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn ensure_cdylib(root: &Path) {
    let candidates = [
        root.join("target/release/liblibjpeg_turbo_rs_capi.dylib"),
        root.join("target/release/liblibjpeg_turbo_rs_capi.so"),
    ];
    if candidates.iter().any(|c| c.exists()) {
        return;
    }
    let status = Command::new(env!("CARGO"))
        .args(["build", "-p", "libjpeg-turbo-rs-capi", "--release"])
        .current_dir(root)
        .status()
        .expect("cargo build");
    assert!(status.success(), "pre-test cargo build failed");
}

#[test]
fn install_capi_sh_produces_complete_layout() {
    if cfg!(windows) {
        eprintln!("SKIP: install_capi.sh is bash; Windows packagers use their own conventions");
        return;
    }
    if Command::new("bash").arg("--version").output().is_err() {
        eprintln!("SKIP: bash not on PATH");
        return;
    }

    let root: PathBuf = workspace_root();
    ensure_cdylib(&root);

    let tmp: tempfile::TempDir = tempfile::tempdir().expect("mkdir tempdir");
    let prefix: &str = "/usr";
    let destdir: &Path = tmp.path();

    let status = Command::new("bash")
        .arg(root.join("scripts/install_capi.sh"))
        .args(["--destdir", &destdir.to_string_lossy()])
        .args(["--prefix", prefix])
        .args(["--root", &root.to_string_lossy()])
        .output()
        .expect("invoke install_capi.sh");
    assert!(
        status.status.success(),
        "install_capi.sh failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&status.stdout),
        String::from_utf8_lossy(&status.stderr)
    );

    let staged: PathBuf = destdir.join(prefix.trim_start_matches('/'));
    let lib: PathBuf = staged.join("lib");
    let inc: PathBuf = staged.join("include");
    let pkgcfg: PathBuf = lib.join("pkgconfig");
    let cmake: PathBuf = lib.join("cmake/JPEG");

    // (1) + (2) + (3): symlink chains for both APIs.
    let dev_libjpeg: PathBuf = lib.join(if cfg!(target_os = "macos") {
        "libjpeg.dylib"
    } else {
        "libjpeg.so"
    });
    let major_libjpeg: PathBuf = lib.join(if cfg!(target_os = "macos") {
        "libjpeg.62.dylib"
    } else {
        "libjpeg.so.62"
    });
    assert!(dev_libjpeg.is_symlink(), "{:?} not a symlink", dev_libjpeg);
    assert!(
        major_libjpeg.is_symlink(),
        "{:?} not a symlink",
        major_libjpeg
    );
    let resolved: PathBuf = std::fs::canonicalize(&dev_libjpeg)
        .unwrap_or_else(|e| panic!("canonicalize {:?}: {}", dev_libjpeg, e));
    assert!(
        resolved.is_file(),
        "{:?} → {:?} does not resolve to a file",
        dev_libjpeg,
        resolved
    );

    let dev_libtj: PathBuf = lib.join(if cfg!(target_os = "macos") {
        "libturbojpeg.dylib"
    } else {
        "libturbojpeg.so"
    });
    let major_libtj: PathBuf = lib.join(if cfg!(target_os = "macos") {
        "libturbojpeg.0.dylib"
    } else {
        "libturbojpeg.so.0"
    });
    assert!(dev_libtj.is_symlink(), "{:?} not a symlink", dev_libtj);
    assert!(major_libtj.is_symlink(), "{:?} not a symlink", major_libtj);

    // (4) pkg-config files are well-formed.
    for pc in ["libjpeg.pc", "libturbojpeg.pc"] {
        let path = pkgcfg.join(pc);
        assert!(path.is_file(), "{:?} missing", path);
        let body: String = std::fs::read_to_string(&path).expect("read pc");
        assert!(body.contains("Name: "), "{} missing Name: line", pc);
        assert!(body.contains("Version: "), "{} missing Version: line", pc);
        assert!(body.contains("Libs: "), "{} missing Libs: line", pc);
        assert!(
            body.contains(&format!("prefix={}", prefix)),
            "{} prefix mismatch:\n{}",
            pc,
            body
        );
    }

    // (5) CMake config exposes the JPEG::JPEG imported target.
    let cmake_config = cmake.join("JPEGConfig.cmake");
    assert!(cmake_config.is_file(), "JPEGConfig.cmake missing");
    let cmake_body: String = std::fs::read_to_string(&cmake_config).expect("read cmake");
    for needle in [
        "JPEG_VERSION",
        "JPEG_INCLUDE_DIR",
        "JPEG_LIBRARY",
        "JPEG::JPEG",
    ] {
        assert!(
            cmake_body.contains(needle),
            "JPEGConfig.cmake missing `{}`:\n{}",
            needle,
            cmake_body
        );
    }

    // (6) All five public C headers staged.
    for header in [
        "jpeglib.h",
        "jerror.h",
        "jmorecfg.h",
        "jconfig.h",
        "turbojpeg.h",
    ] {
        let h = inc.join(header);
        assert!(h.is_file(), "header {} not staged at {:?}", header, h);
    }
    // jconfig.h declares JPEG_LIB_VERSION 80 (matches our struct layout).
    let jconfig: String = std::fs::read_to_string(inc.join("jconfig.h")).expect("read jconfig");
    assert!(
        jconfig.contains("JPEG_LIB_VERSION 80"),
        "staged jconfig.h doesn't declare v8 ABI:\n{}",
        jconfig
    );

    // (Optional) pkg-config end-to-end sanity. If pkg-config is on
    // PATH, asking it for `--libs libjpeg` against our staged tree
    // must return a `-ljpeg` flag.
    if Command::new("pkg-config").arg("--version").output().is_ok() {
        let out = Command::new("pkg-config")
            .env("PKG_CONFIG_PATH", &pkgcfg)
            .args(["--libs", "libjpeg"])
            .output()
            .expect("invoke pkg-config");
        assert!(
            out.status.success(),
            "pkg-config --libs libjpeg failed:\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        let libs = String::from_utf8_lossy(&out.stdout);
        assert!(
            libs.contains("-ljpeg"),
            "pkg-config --libs libjpeg returned `{}` (expected -ljpeg)",
            libs.trim()
        );
    } else {
        eprintln!("NOTE: pkg-config not on PATH; skipping the optional --libs check");
    }
}

/// Codex round-1: passing `--soname libjpeg.8.dylib` (the
/// production-safe SONAME from docs/ABI_COMPATIBILITY.md) must change
/// the symlink chain accordingly. Prior to the fix, `--soname` was
/// declared but silently ignored.
#[test]
fn install_capi_sh_honors_soname_override() {
    if cfg!(windows) {
        eprintln!("SKIP: install_capi.sh is bash; Windows packagers use their own conventions");
        return;
    }
    if Command::new("bash").arg("--version").output().is_err() {
        eprintln!("SKIP: bash not on PATH");
        return;
    }

    let root: PathBuf = workspace_root();
    ensure_cdylib(&root);

    let tmp: tempfile::TempDir = tempfile::tempdir().expect("mkdir tempdir");
    let prefix: &str = "/usr";
    let destdir: &Path = tmp.path();

    // Use the v8 SONAMEs the policy doc names.
    let (override_soname, expected_major, expected_dev) = if cfg!(target_os = "macos") {
        ("libjpeg.8.dylib", "libjpeg.8.dylib", "libjpeg.dylib")
    } else {
        ("libjpeg.so.8", "libjpeg.so.8", "libjpeg.so")
    };

    let status = Command::new("bash")
        .arg(root.join("scripts/install_capi.sh"))
        .args(["--destdir", &destdir.to_string_lossy()])
        .args(["--prefix", prefix])
        .args(["--root", &root.to_string_lossy()])
        .args(["--soname", override_soname])
        .output()
        .expect("invoke install_capi.sh");
    assert!(
        status.status.success(),
        "install_capi.sh --soname failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&status.stdout),
        String::from_utf8_lossy(&status.stderr)
    );

    let lib: PathBuf = destdir
        .join(prefix.trim_start_matches('/'))
        .join("lib");

    let major: PathBuf = lib.join(expected_major);
    let dev: PathBuf = lib.join(expected_dev);
    assert!(
        major.is_symlink(),
        "{:?} not a symlink — `--soname {}` was ignored",
        major,
        override_soname
    );
    assert!(dev.is_symlink(), "{:?} dev symlink missing", dev);

    // The default v6b symlink must NOT be staged when an override is
    // active — that would silently double-install both ABIs.
    let v6b: PathBuf = lib.join(if cfg!(target_os = "macos") {
        "libjpeg.62.dylib"
    } else {
        "libjpeg.so.62"
    });
    assert!(
        !v6b.exists(),
        "{:?} should not be installed when --soname overrides the default",
        v6b
    );
}

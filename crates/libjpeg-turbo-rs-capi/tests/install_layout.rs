//! P2-8: end-to-end install-tree layout test.
//!
//! Runs `scripts/install_capi.sh` into a tempdir and asserts:
//!
//! 1. The cdylib lands at the SONAME path (`libjpeg.so.8.X.Y` on
//!    Linux / `libjpeg.8.X.Y.dylib` on macOS — P4-3 v8 default).
//! 2. The symlink chain resolves (`libjpeg.so → .8 → real cdylib`).
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

/// Returns the cdylib identity advertised by `staged` — `@rpath/...`
/// from `otool -D` on macOS, or the `DT_SONAME` from `readelf -d` on
/// Linux. Returns `None` (with a printed SKIP) when the inspection
/// tool isn't on PATH so the test can soft-skip on minimal CI images.
fn cdylib_identity(staged: &Path) -> Option<String> {
    if cfg!(target_os = "macos") {
        if Command::new("which")
            .arg("otool")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            let out = Command::new("otool").arg("-D").arg(staged).output().ok()?;
            let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
            // otool -D output: `<path>:\n<install_name>\n`
            Some(stdout.lines().nth(1).unwrap_or("").trim().to_string())
        } else {
            eprintln!("SKIP cdylib_identity: otool not on PATH");
            None
        }
    } else if Command::new("which")
        .arg("readelf")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        let out = Command::new("readelf")
            .arg("-d")
            .arg(staged)
            .output()
            .ok()?;
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        for line in stdout.lines() {
            if line.contains("SONAME") {
                // Format: `0x... (SONAME)  Library soname: [libjpeg.so.8]`
                if let Some(start) = line.find('[') {
                    if let Some(end) = line[start..].find(']') {
                        return Some(line[start + 1..start + end].to_string());
                    }
                }
            }
        }
        Some(String::new()) // SONAME stripped — caller will fail
    } else {
        eprintln!("SKIP cdylib_identity: readelf not on PATH");
        None
    }
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
    // P4-3 (2026-05-17): default flipped to the v8 ABI SONAME so the
    // install layout now stages `libjpeg.8.dylib` / `libjpeg.so.8`.
    let major_libjpeg: PathBuf = lib.join(if cfg!(target_os = "macos") {
        "libjpeg.8.dylib"
    } else {
        "libjpeg.so.8"
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

    // P4-3 follow-up (Codex stop-time review): the staged cdylib's
    // identity (macOS install_name / Linux DT_SONAME) must agree with
    // the symlink SONAME — otherwise the dynamic linker resolves to
    // the wrong file at run time.
    if let Some(id) = cdylib_identity(&resolved) {
        let expected: &str = if cfg!(target_os = "macos") {
            "libjpeg.8.dylib"
        } else {
            "libjpeg.so.8"
        };
        assert!(
            id.contains(expected),
            "staged cdylib identity is {:?}, expected to contain {:?} \
             (install_capi.sh did not patch the binary identity)",
            id,
            expected
        );
    }

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

/// P4-3 (2026-05-17): the default flipped from v6b → v8, so this test
/// now drives the v6b *opt-in* path. Passing `--soname libjpeg.so.62`
/// (the legacy distro SONAME, now documented-risk per
/// docs/ABI_COMPATIBILITY.md) must stage the v6b symlink chain and
/// must NOT stage the v8 default in parallel.
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

    // Opt into the v6b SONAME (the legacy distro path).
    let (override_soname, expected_major, expected_dev) = if cfg!(target_os = "macos") {
        ("libjpeg.62.dylib", "libjpeg.62.dylib", "libjpeg.dylib")
    } else {
        ("libjpeg.so.62", "libjpeg.so.62", "libjpeg.so")
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

    let lib: PathBuf = destdir.join(prefix.trim_start_matches('/')).join("lib");

    let major: PathBuf = lib.join(expected_major);
    let dev: PathBuf = lib.join(expected_dev);
    assert!(
        major.is_symlink(),
        "{:?} not a symlink — `--soname {}` was ignored",
        major,
        override_soname
    );
    assert!(dev.is_symlink(), "{:?} dev symlink missing", dev);

    // The v8 default symlink must NOT be staged when --soname overrides
    // to v6b — that would silently double-install both ABIs.
    let v8: PathBuf = lib.join(if cfg!(target_os = "macos") {
        "libjpeg.8.dylib"
    } else {
        "libjpeg.so.8"
    });
    assert!(
        !v8.exists(),
        "{:?} should not be installed when --soname overrides the default",
        v8
    );

    // P4-3 follow-up (Codex stop-time review): the staged cdylib's
    // identity (macOS install_name / Linux DT_SONAME) must follow the
    // override too. Without this, the v6b symlink chain would point
    // at a binary still advertising the v8 build-time identity, and
    // load-time resolution would fail.
    let resolved: PathBuf =
        std::fs::canonicalize(&dev).unwrap_or_else(|e| panic!("canonicalize {:?}: {}", dev, e));
    if let Some(id) = cdylib_identity(&resolved) {
        let install_tool_present: bool = if cfg!(target_os = "macos") {
            Command::new("which")
                .arg("install_name_tool")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
        } else {
            Command::new("which")
                .arg("patchelf")
                .output()
                .map(|o| o.status.success())
                .unwrap_or(false)
        };
        if install_tool_present {
            assert!(
                id.contains(override_soname),
                "staged cdylib identity is {:?}, expected to contain {:?} \
                 (install_capi.sh `--soname {}` did not patch the binary \
                 identity even though install_name_tool/patchelf is available)",
                id,
                override_soname,
                override_soname
            );
        } else {
            eprintln!(
                "SKIP identity assertion: neither install_name_tool nor patchelf \
                 on PATH; install_capi.sh emits a warning in this configuration"
            );
        }
    }
}

//! P2-8: end-to-end install-tree layout test.
//!
//! Runs `scripts/install_capi.sh` into a tempdir and asserts:
//!
//! 1. The library lands under its shipped name (`libjpeg.so.8.X.Y` on
//!    Linux / `libjpeg.8.X.Y.dylib` on macOS — P4-3 v8 default — and
//!    `bin/jpeg8.dll` on Windows, the name upstream's v8 MSVC build gives it).
//! 2. The symlink chain resolves (`libjpeg.so → .8 → real cdylib`), or on
//!    Windows, where there is no chain, the import library `lib/jpeg.lib`
//!    binds to the shipped DLL name rather than to cargo's.
//! 3. Both the libjpeg and libturbojpeg identities exist, and the library
//!    loads and resolves one entry point of each API.
//! 4. The pkg-config file is well-formed (Name/Version/Libs lines).
//! 5. The CMake config exposes `JPEG::JPEG` imported target wiring and names
//!    the library a consumer links against.
//! 6. All five public C headers are present.
//!
//! When `pkg-config` is on PATH, the test additionally invokes
//! `pkg-config --libs libjpeg` against `PKG_CONFIG_PATH=<staged>` and
//! asserts the returned `-l` line includes `-ljpeg`.
//!
//! Skip-with-reason case: no bash (on Windows, no Git for Windows). The
//! Windows leg of `capi-abi-checks` runs this suite for real since P4-131's
//! Windows milestone; before it, the platform skipped here.

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "support/cdylib.rs"]
mod cdylib_support;
#[path = "support/shell.rs"]
mod shell;

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

/// The bash that runs the script, or `None` after printing the skip reason.
fn bash_or_skip() -> Option<PathBuf> {
    match shell::bash() {
        Ok(bash) => Some(bash),
        Err(reason) => {
            eprintln!("SKIP: {reason}");
            None
        }
    }
}

/// The name a consumer's dynamic linker records for the libjpeg API: the
/// major SONAME on Linux/macOS, the DLL file name on Windows.
fn libjpeg_major() -> &'static str {
    if cfg!(windows) {
        "jpeg8.dll"
    } else if cfg!(target_os = "macos") {
        "libjpeg.8.dylib"
    } else {
        "libjpeg.so.8"
    }
}

/// What `-ljpeg` resolves to: the dev symlink, or the MSVC import library.
fn libjpeg_dev() -> &'static str {
    if cfg!(windows) {
        "jpeg.lib"
    } else if cfg!(target_os = "macos") {
        "libjpeg.dylib"
    } else {
        "libjpeg.so"
    }
}

fn libturbojpeg_major() -> &'static str {
    if cfg!(windows) {
        "turbojpeg.dll"
    } else if cfg!(target_os = "macos") {
        "libturbojpeg.0.dylib"
    } else {
        "libturbojpeg.so.0"
    }
}

fn libturbojpeg_dev() -> &'static str {
    if cfg!(windows) {
        "turbojpeg.lib"
    } else if cfg!(target_os = "macos") {
        "libturbojpeg.dylib"
    } else {
        "libturbojpeg.so"
    }
}

/// Where the loadable library lives below the prefix. Upstream's MSVC layout
/// puts DLLs in `bin/` beside the executables that load them and keeps `lib/`
/// for the import libraries; everything else is `lib/`.
fn loadable_dir() -> &'static str {
    if cfg!(windows) {
        "bin"
    } else {
        "lib"
    }
}

/// Returns the cdylib identity advertised by `staged` — `@rpath/...`
/// from `otool -D` on macOS, or the `DT_SONAME` from `readelf -d` on
/// Linux. Returns `None` (with a printed SKIP) when the inspection
/// tool isn't on PATH so the test can soft-skip on minimal CI images.
#[cfg(not(windows))]
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

/// P4-131 (Windows): the import library is where the identity lives.
///
/// An MSVC import library is an archive of short import records, each naming
/// the symbol and the DLL it comes from, as NUL-terminated strings. A consumer
/// linked through it records that DLL name in its import table, so if the
/// library still named cargo's `libjpeg_turbo_rs_capi.dll` the consumer would
/// load that — and fail on any host that has only the bundle. Assert the
/// binding by the string, not by trusting the file name.
fn assert_import_library_binds_to(import_library: &Path, dll: &str) {
    let bytes: Vec<u8> = std::fs::read(import_library)
        .unwrap_or_else(|e| panic!("read import library {import_library:?}: {e}"));
    assert!(
        bytes.starts_with(b"!<arch>\n"),
        "{import_library:?} is not a COFF archive (import library); it starts {:02x?}",
        &bytes[..bytes.len().min(8)]
    );
    let mut needle: Vec<u8> = dll.as_bytes().to_vec();
    needle.push(0);
    assert!(
        shell::contains_bytes(&bytes, &needle),
        "{import_library:?} does not bind its imports to {dll:?}"
    );
    assert!(
        !shell::contains_bytes(&bytes, b"libjpeg_turbo_rs_capi.dll"),
        "{import_library:?} still binds to cargo's libjpeg_turbo_rs_capi.dll, so a \
         consumer linked through it would not load the shipped {dll}"
    );
    // And it must carry the exports. A `.def` that lost most rows to a
    // dumpbin parse mismatch still yields a valid archive bound to the right
    // DLL — one that links nothing.
    for symbol in cdylib_support::REQUIRED_EXPORTS
        .iter()
        .chain(["jpeg_finish_decompress", "tj3Compress8"].iter())
    {
        let mut name: Vec<u8> = symbol.as_bytes().to_vec();
        name.push(0);
        assert!(
            shell::contains_bytes(&bytes, &name),
            "{import_library:?} does not export {symbol}"
        );
    }
}

/// The staged loadable library for `major`, after asserting its platform
/// shape: a resolving symlink chain whose binary identity matches the name
/// (Linux/macOS), or a DLL whose import library binds to the name (Windows).
fn assert_library_staged(
    staged: &Path,
    major: &str,
    dev: &str,
    expected_identity: &str,
) -> PathBuf {
    if cfg!(windows) {
        let dll: PathBuf = staged.join("bin").join(major);
        assert!(dll.is_file(), "{dll:?} is not a file");
        let head: Vec<u8> = std::fs::read(&dll).expect("read the staged DLL");
        assert!(
            head.starts_with(b"MZ") && head.len() > 4096,
            "{dll:?} is not a PE image ({} bytes, starts {:02x?})",
            head.len(),
            &head[..head.len().min(2)]
        );
        let import_library: PathBuf = staged.join("lib").join(dev);
        assert!(import_library.is_file(), "{import_library:?} is not a file");
        assert_import_library_binds_to(&import_library, expected_identity);
        return dll;
    }

    let lib: PathBuf = staged.join("lib");
    let dev_path: PathBuf = lib.join(dev);
    let major_path: PathBuf = lib.join(major);
    assert!(dev_path.is_symlink(), "{dev_path:?} not a symlink");
    assert!(major_path.is_symlink(), "{major_path:?} not a symlink");
    let resolved: PathBuf = std::fs::canonicalize(&dev_path)
        .unwrap_or_else(|e| panic!("canonicalize {dev_path:?}: {e}"));
    assert!(
        resolved.is_file(),
        "{dev_path:?} → {resolved:?} does not resolve to a file"
    );

    // P4-3 follow-up (Codex stop-time review): the staged cdylib's
    // identity (macOS install_name / Linux DT_SONAME) must agree with
    // the symlink SONAME — otherwise the dynamic linker resolves to
    // the wrong file at run time.
    #[cfg(not(windows))]
    if let Some(id) = cdylib_identity(&resolved) {
        assert!(
            id.contains(expected_identity),
            "staged cdylib identity is {id:?}, expected to contain \
             {expected_identity:?} (install_capi.sh did not patch the binary identity)"
        );
    }
    resolved
}

#[test]
fn install_capi_sh_produces_complete_layout() {
    let Some(bash) = bash_or_skip() else {
        return;
    };

    let root: PathBuf = workspace_root();
    let cdylib: PathBuf = cdylib_support::cdylib_path();
    let cdylib_dir: &Path = cdylib.parent().expect("Cargo artifact directory");

    let tmp: tempfile::TempDir = tempfile::tempdir().expect("mkdir tempdir");
    let prefix: &str = "/usr";
    let destdir: &Path = tmp.path();

    let status = Command::new(&bash)
        .arg(shell::script_arg(&root.join("scripts/install_capi.sh")))
        .args(["--destdir", &shell::script_arg(destdir)])
        .args(["--prefix", prefix])
        .args(["--root", &shell::script_arg(&root)])
        .env("CAPI_TARGET_DIR", shell::script_arg(cdylib_dir))
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

    // (1) + (2) + (3): both identities, and the library behind them loads.
    // P4-3 (2026-05-17): default flipped to the v8 ABI SONAME so the
    // install layout stages `libjpeg.8.dylib` / `libjpeg.so.8`; on Windows
    // (P4-131) the equivalent is upstream's `jpeg8.dll`.
    let libjpeg: PathBuf =
        assert_library_staged(&staged, libjpeg_major(), libjpeg_dev(), libjpeg_major());
    cdylib_support::assert_library_exports(&libjpeg, cdylib_support::REQUIRED_EXPORTS);
    let libturbojpeg: PathBuf = assert_library_staged(
        &staged,
        libturbojpeg_major(),
        libturbojpeg_dev(),
        libturbojpeg_major(),
    );
    cdylib_support::assert_library_exports(&libturbojpeg, cdylib_support::REQUIRED_EXPORTS);

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

    // (5) CMake config exposes the JPEG::JPEG imported target, and points it
    // at the file a consumer links: the dev symlink, or on Windows the import
    // library — `find_package(JPEG)` against a prefix whose `JPEG_LIBRARY`
    // names a DLL fails at link time, which is exactly what FindJPEG.cmake's
    // own search would never do.
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
    let linked: String = format!("set(JPEG_LIBRARY \"{prefix}/lib/{}\")", libjpeg_dev());
    assert!(
        cmake_body.contains(&linked),
        "JPEGConfig.cmake does not point JPEG_LIBRARY at `{linked}`:\n{cmake_body}"
    );

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
/// must NOT stage the v8 default in parallel. On Windows the override
/// names the DLL (`jpeg62.dll`, as upstream's non-WITH_JPEG8 build), and
/// the import library must follow it.
#[test]
fn install_capi_sh_honors_soname_override() {
    let Some(bash) = bash_or_skip() else {
        return;
    };

    let root: PathBuf = workspace_root();
    let cdylib: PathBuf = cdylib_support::cdylib_path();
    let cdylib_dir: &Path = cdylib.parent().expect("Cargo artifact directory");

    let tmp: tempfile::TempDir = tempfile::tempdir().expect("mkdir tempdir");
    let prefix: &str = "/usr";
    let destdir: &Path = tmp.path();

    // Opt into the v6b SONAME (the legacy distro path).
    let override_soname: &str = if cfg!(windows) {
        "jpeg62.dll"
    } else if cfg!(target_os = "macos") {
        "libjpeg.62.dylib"
    } else {
        "libjpeg.so.62"
    };

    let status = Command::new(&bash)
        .arg(shell::script_arg(&root.join("scripts/install_capi.sh")))
        .args(["--destdir", &shell::script_arg(destdir)])
        .args(["--prefix", prefix])
        .args(["--root", &shell::script_arg(&root)])
        .args(["--soname", override_soname])
        .env("CAPI_TARGET_DIR", shell::script_arg(cdylib_dir))
        .output()
        .expect("invoke install_capi.sh");
    assert!(
        status.status.success(),
        "install_capi.sh --soname failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&status.stdout),
        String::from_utf8_lossy(&status.stderr)
    );

    let staged: PathBuf = destdir.join(prefix.trim_start_matches('/'));

    // The v8 default must NOT be staged when --soname overrides to v6b —
    // that would silently double-install both ABIs.
    let v8: PathBuf = staged.join(loadable_dir()).join(libjpeg_major());
    assert!(
        !v8.exists(),
        "{v8:?} should not be installed when --soname overrides the default"
    );

    if cfg!(windows) {
        // No chain to check; the identity is the import library's binding,
        // and the DLL must exist under the override name.
        let dll: PathBuf =
            assert_library_staged(&staged, override_soname, libjpeg_dev(), override_soname);
        cdylib_support::assert_library_exports(&dll, cdylib_support::REQUIRED_EXPORTS);
        return;
    }

    let lib: PathBuf = staged.join("lib");
    let major: PathBuf = lib.join(override_soname);
    let dev: PathBuf = lib.join(libjpeg_dev());
    assert!(
        major.is_symlink(),
        "{major:?} not a symlink — `--soname {override_soname}` was ignored"
    );
    assert!(dev.is_symlink(), "{dev:?} dev symlink missing");

    // P4-3 follow-up (Codex stop-time review): the staged cdylib's
    // identity (macOS install_name / Linux DT_SONAME) must follow the
    // override too. Without this, the v6b symlink chain would point
    // at a binary still advertising the v8 build-time identity, and
    // load-time resolution would fail.
    #[cfg(not(windows))]
    let resolved: PathBuf =
        std::fs::canonicalize(&dev).unwrap_or_else(|e| panic!("canonicalize {:?}: {}", dev, e));
    #[cfg(not(windows))]
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

/// The release runs the scripts with no `--prefix`, so the platform default
/// — and, on Windows, the drive-stripping DESTDIR arithmetic that goes with
/// `C:/libjpeg-turbo-rs64` — is the path a published bundle takes. Every
/// other test here passes an explicit POSIX prefix, which would leave that
/// branch to run first on a tag.
#[test]
fn install_capi_sh_stages_the_platform_default_prefix_below_destdir() {
    let Some(bash) = bash_or_skip() else {
        return;
    };

    let root: PathBuf = workspace_root();
    let cdylib: PathBuf = cdylib_support::cdylib_path();
    let cdylib_dir: &Path = cdylib.parent().expect("Cargo artifact directory");
    let tmp: tempfile::TempDir = tempfile::tempdir().expect("mkdir tempdir");
    let destdir: &Path = tmp.path();

    let status = Command::new(&bash)
        .arg(shell::script_arg(&root.join("scripts/install_capi.sh")))
        .args(["--destdir", &shell::script_arg(destdir)])
        .args(["--root", &shell::script_arg(&root)])
        .env("CAPI_TARGET_DIR", shell::script_arg(cdylib_dir))
        .output()
        .expect("invoke install_capi.sh");
    assert!(
        status.status.success(),
        "install_capi.sh failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&status.stdout),
        String::from_utf8_lossy(&status.stderr)
    );

    // The prefix baked into the files, and where it lands below DESTDIR:
    // upstream's `c:/libjpeg-turbo64` convention loses its drive, `/usr/local`
    // is appended as-is.
    let (expected_prefix, staged_below): (&str, &str) = if cfg!(windows) {
        ("C:/libjpeg-turbo-rs64", "libjpeg-turbo-rs64")
    } else {
        ("/usr/local", "usr/local")
    };
    let staged: PathBuf = destdir.join(staged_below);
    let pc: PathBuf = staged.join("lib/pkgconfig/libjpeg.pc");
    let body: String =
        std::fs::read_to_string(&pc).unwrap_or_else(|e| panic!("{pc:?} not staged: {e}"));
    assert!(
        body.contains(&format!("prefix={expected_prefix}")),
        "the default prefix is not {expected_prefix}:\n{body}"
    );
    let libjpeg: PathBuf =
        assert_library_staged(&staged, libjpeg_major(), libjpeg_dev(), libjpeg_major());
    cdylib_support::assert_library_exports(&libjpeg, cdylib_support::REQUIRED_EXPORTS);
}

#[test]
fn install_capi_sh_builds_into_capi_target_dir() {
    let Some(bash) = bash_or_skip() else {
        return;
    };

    let root: PathBuf = workspace_root();
    let temp: tempfile::TempDir = tempfile::tempdir().expect("mkdir tempdir");
    let cargo_target_dir: PathBuf = temp.path().join("cargo-target");
    // A stand-in target: the fake cargo below builds nothing, so it need not
    // be one this host can compile for. On Windows the install script
    // accepts only the MSVC target it stages a layout for.
    let build_target: &str = if cfg!(windows) {
        "x86_64-pc-windows-msvc"
    } else {
        "x86_64-unknown-linux-gnu"
    };
    let release_dir: PathBuf = cargo_target_dir.join(build_target).join("release");
    let destdir: PathBuf = temp.path().join("stage");
    let source_cdylib: PathBuf = cdylib_support::cdylib_path();
    let fake_bin_dir: PathBuf = temp.path().join("bin");
    std::fs::create_dir_all(&fake_bin_dir).expect("create fake cargo directory");
    let fake_cargo: PathBuf = fake_bin_dir.join("cargo");
    // `norm` because under Git for Windows' bash the script hands cargo a
    // `C:/...` CARGO_TARGET_DIR (what a native cargo needs) while the test
    // set the expectation from a `C:\...` path; `cygpath` folds both to one
    // spelling. Elsewhere it is the identity.
    std::fs::write(
        &fake_cargo,
        r#"#!/usr/bin/env bash
set -eu
norm() {
    if command -v cygpath >/dev/null 2>&1; then cygpath -u "$1"; else printf '%s\n' "$1"; fi
}
test "$(norm "$CARGO_TARGET_DIR")" = "$(norm "$EXPECTED_CARGO_TARGET_DIR")"
previous=
target=
for argument in "$@"; do
    if [ "$previous" = "--target" ]; then
        target="$argument"
    fi
    previous="$argument"
done
test "$target" = "$EXPECTED_BUILD_TARGET"
mkdir -p "$(norm "$CARGO_TARGET_DIR")/$target/release"
cp "$(norm "$SOURCE_CDYLIB")" "$(norm "$CARGO_TARGET_DIR")/$target/release/$(basename "$(norm "$SOURCE_CDYLIB")")"
"#,
    )
    .expect("write fake cargo");
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&fake_cargo, std::fs::Permissions::from_mode(0o755))
            .expect("chmod fake cargo");
    }
    let inherited_path: std::ffi::OsString = std::env::var_os("PATH").unwrap_or_default();
    let command_path: std::ffi::OsString = std::env::join_paths(
        std::iter::once(fake_bin_dir.clone()).chain(std::env::split_paths(&inherited_path)),
    )
    .expect("construct PATH with fake cargo");

    let output: std::process::Output = Command::new(&bash)
        .arg(shell::script_arg(&root.join("scripts/install_capi.sh")))
        .args(["--build", "--destdir", &shell::script_arg(&destdir)])
        .args(["--prefix", "/usr"])
        .args(["--root", &shell::script_arg(&root)])
        .env("CAPI_TARGET_DIR", shell::script_arg(&release_dir))
        .env("CAPI_BUILD_TARGET", build_target)
        .env("EXPECTED_CARGO_TARGET_DIR", &cargo_target_dir)
        .env("EXPECTED_BUILD_TARGET", build_target)
        .env("SOURCE_CDYLIB", &source_cdylib)
        .env("CARGO", shell::script_arg(&fake_cargo))
        .env("PATH", command_path)
        .env_remove("CARGO_TARGET_DIR")
        .output()
        .expect("invoke install_capi.sh --build");

    assert!(
        output.status.success(),
        "install_capi.sh --build failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        cdylib_support::release_cdylib_path_for_target_in(
            &cargo_target_dir,
            std::ffi::OsStr::new(build_target),
        )
        .is_file(),
        "--build must emit the cdylib below CAPI_TARGET_DIR"
    );
}

#[test]
fn install_capi_sh_rejects_non_release_capi_target_dir_when_building() {
    let Some(bash) = bash_or_skip() else {
        return;
    };

    let root: PathBuf = workspace_root();
    let temp: tempfile::TempDir = tempfile::tempdir().expect("mkdir tempdir");
    let output: std::process::Output = Command::new(&bash)
        .arg(shell::script_arg(&root.join("scripts/install_capi.sh")))
        .args(["--build", "--root", &shell::script_arg(&root)])
        .env(
            "CAPI_TARGET_DIR",
            shell::script_arg(&temp.path().join("custom-output")),
        )
        .env_remove("CARGO_TARGET_DIR")
        .output()
        .expect("invoke install_capi.sh --build with invalid target");

    assert!(!output.status.success(), "invalid build target must fail");
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("must end in /")
            && String::from_utf8_lossy(&output.stderr).contains("/release when building"),
        "failure must explain the CAPI_TARGET_DIR contract: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

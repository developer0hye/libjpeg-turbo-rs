//! A1-14: verify the `libjpeg.pc` and `libturbojpeg.pc` files produced
//! by `build.rs` are well-formed pkg-config manifests pointing at the
//! shim's two supported SONAMEs.
//!
//! The build script exports `CAPI_PKGCONFIG_DIR` via `cargo:rustc-env`,
//! so `env!()` below resolves to the directory holding both .pc files.

use std::path::{Path, PathBuf};
use std::process::Command;

fn pkgconfig_dir() -> PathBuf {
    PathBuf::from(env!("CAPI_PKGCONFIG_DIR"))
}

fn read_pc(dir: &Path, name: &str) -> String {
    std::fs::read_to_string(dir.join(name))
        .unwrap_or_else(|e| panic!("read {}/{}: {e}", dir.display(), name))
}

#[test]
fn libjpeg_pc_advertises_the_libjpeg_abi() {
    let dir: PathBuf = pkgconfig_dir();
    let pc: String = read_pc(&dir, "libjpeg.pc");

    // Mandatory pkg-config fields.
    for key in [
        "prefix=",
        "libdir=",
        "includedir=",
        "Name:",
        "Description:",
        "Version:",
        "Libs:",
        "Cflags:",
    ] {
        assert!(pc.contains(key), "libjpeg.pc missing `{key}`:\n{pc}");
    }

    // `Name: libjpeg` is what `pkg-config --list-all` and downstream
    // Makefiles match on.
    assert!(
        pc.lines().any(|l| l.trim_end() == "Name: libjpeg"),
        "libjpeg.pc must have `Name: libjpeg`:\n{pc}"
    );
    // Libs line must reference the `-ljpeg` we're impersonating.
    assert!(
        pc.contains("-ljpeg"),
        "libjpeg.pc Libs must include -ljpeg:\n{pc}"
    );
    // Version must match the crate's Cargo.toml version.
    assert!(
        pc.contains(&format!("Version: {}", env!("CARGO_PKG_VERSION"))),
        "libjpeg.pc Version mismatch:\n{pc}"
    );
}

#[test]
fn libturbojpeg_pc_advertises_the_turbojpeg_abi() {
    let dir: PathBuf = pkgconfig_dir();
    let pc: String = read_pc(&dir, "libturbojpeg.pc");

    assert!(
        pc.lines().any(|l| l.trim_end() == "Name: libturbojpeg"),
        "libturbojpeg.pc must have `Name: libturbojpeg`:\n{pc}"
    );
    assert!(
        pc.contains("-lturbojpeg"),
        "libturbojpeg.pc Libs must include -lturbojpeg:\n{pc}"
    );
    assert!(
        pc.contains(&format!("Version: {}", env!("CARGO_PKG_VERSION"))),
        "libturbojpeg.pc Version mismatch:\n{pc}"
    );
}

#[test]
fn pkg_config_cli_parses_generated_files_when_available() {
    // `pkg-config` is optional on dev hosts (CLAUDE.md notes it isn't
    // installed on macOS with Homebrew by default). Skip gracefully
    // when missing — the structural checks above already guarantee the
    // files are well-formed. This is only a belt-and-braces check.
    let which = Command::new("which").arg("pkg-config").output();
    let have = matches!(&which, Ok(o) if o.status.success());
    if !have {
        eprintln!("SKIP: pkg-config not on PATH");
        return;
    }

    let dir: PathBuf = pkgconfig_dir();
    for name in ["libjpeg", "libturbojpeg"] {
        let out = Command::new("pkg-config")
            .env("PKG_CONFIG_PATH", &dir)
            .arg("--libs")
            .arg("--cflags")
            .arg(name)
            .output()
            .expect("spawn pkg-config");
        assert!(
            out.status.success(),
            "pkg-config --libs --cflags {name} failed: stderr={}",
            String::from_utf8_lossy(&out.stderr)
        );
        let stdout: String = String::from_utf8_lossy(&out.stdout).into_owned();
        let expected: &str = if name == "libjpeg" {
            "-ljpeg"
        } else {
            "-lturbojpeg"
        };
        assert!(
            stdout.contains(expected),
            "pkg-config output for {name} missing {expected}: {stdout}"
        );
    }
}

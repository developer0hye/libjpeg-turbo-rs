//! P4-131: the release bundle a tag publishes.
//!
//! `scripts/package_capi_release.sh` is the only thing that turns this
//! repository into a downloadable native library. It has one job — take the
//! prefix `scripts/install_capi.sh` stages and hand it back as a checksummed
//! archive — and one prohibition, which is the reason this suite exists at
//! all: it must not stage anything itself. P4-131's third acceptance
//! criterion and P4-124 together say there is **one** staging path, so the
//! artifact a packager downloads is the artifact the downstream harnesses
//! test. A second, quietly divergent path inside the packaging script would
//! satisfy every "the tarball has a `libjpeg.so.8` in it" check ever written,
//! which is why `release_bundle_is_exactly_what_install_capi_sh_stages`
//! compares the bundle against a direct `install_capi.sh` run file by file
//! rather than against a list of expected names.
//!
//! Covered:
//!
//! 1. The archive carries the complete staged prefix — both symlink chains,
//!    both `.pc` files, the CMake config, all five headers — with symlinks
//!    still symlinks rather than dereferenced copies.
//! 2. The bundle describes itself: `BUNDLE.txt` names the version, target and
//!    the prefix baked into the `.pc` / CMake files, because those are
//!    absolute and a packager who unpacks elsewhere has to know.
//! 3. `sha256sum -c` / `shasum -a 256 -c` verifies the archive from the
//!    directory it was downloaded into — the manifest names the bare archive,
//!    not a build-machine path.
//! 4. The bundle contents equal a direct `install_capi.sh` staging run.
//! 5. `--sbom` writes a checksummed CycloneDX SBOM of the capi crate beside
//!    the archive, and `release.yml` attests both archive and SBOM in the job
//!    that built them and attaches them (P4-131 criterion 4 — see the section
//!    banner further down).
//!
//! Skip-with-reason cases mirror `install_layout.rs`: hosts without `bash`
//! (on Windows, without Git for Windows) or `tar`; the SBOM test also skips
//! without `cargo-cyclonedx`. The workflow-shape tests read YAML only and run
//! everywhere. The Windows leg runs the bundle tests for real since P4-131's
//! Windows milestone, against the MSVC layout (`bin/jpeg8.dll` + `lib/jpeg.lib`).

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "support/cdylib.rs"]
mod cdylib_support;
#[path = "support/shell.rs"]
mod shell;

/// The prefix the bundles are staged with in these tests. Deliberately not
/// the script default, so a script that ignored `--prefix` and always baked
/// its own would fail the `.pc` assertions below.
const TEST_PREFIX: &str = "/opt/libjpeg-turbo-rs";

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn have(tool: &str) -> bool {
    Command::new(tool)
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// `None` with a printed reason when this host cannot exercise the packaging
/// script at all.
fn unsupported_host() -> Option<&'static str> {
    if let Err(reason) = shell::bash() {
        return Some(reason);
    }
    if let Err(reason) = shell::tar() {
        return Some(reason);
    }
    None
}

/// A command running one of the staging scripts through the host's bash,
/// with the repository root and the cdylib under test already supplied.
fn script_command(script: &str) -> Command {
    let root: PathBuf = workspace_root();
    let cdylib: PathBuf = cdylib_support::cargo_built_cdylib_path()
        .unwrap_or_else(|e| panic!("could not locate the cdylib under test: {e}"));
    let cdylib_dir: &Path = cdylib.parent().expect("Cargo artifact directory");
    let mut command = Command::new(shell::bash().expect("checked by unsupported_host"));
    command
        .arg(shell::script_arg(&root.join("scripts").join(script)))
        .args(["--root", &shell::script_arg(&root)])
        .env("CAPI_TARGET_DIR", shell::script_arg(cdylib_dir));
    command
}

/// Runs `scripts/package_capi_release.sh` into `outdir` and returns the
/// archive it produced.
///
/// Failure is fatal, never a skip: reaching here means the host *can* run the
/// script, so a non-zero exit is a defect in the thing under test.
fn package_into(outdir: &Path) -> PathBuf {
    let run = script_command("package_capi_release.sh")
        .args(["--outdir", &shell::script_arg(outdir)])
        .args(["--prefix", TEST_PREFIX])
        .output()
        .expect("invoke package_capi_release.sh");
    assert!(
        run.status.success(),
        "package_capi_release.sh failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );

    let archives: Vec<PathBuf> = std::fs::read_dir(outdir)
        .expect("read outdir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.to_string_lossy().ends_with(".tar.gz"))
        .collect();
    assert_eq!(
        archives.len(),
        1,
        "expected exactly one archive in {:?}, found {:?}",
        outdir,
        archives
    );
    archives.into_iter().next().unwrap()
}

/// Unpacks `archive` into `into` and returns the single top-level directory it
/// contains — upstream's convention for a binary tarball, and the reason an
/// unpack cannot scatter files over the user's working directory.
fn extract(archive: &Path, into: &Path) -> PathBuf {
    let untar = Command::new(shell::tar().expect("checked by unsupported_host"))
        .args(["-xzf", &archive.to_string_lossy()])
        .args(["-C", &into.to_string_lossy()])
        .output()
        .expect("invoke tar");
    assert!(
        untar.status.success(),
        "tar -xzf {:?} failed:\n{}",
        archive,
        String::from_utf8_lossy(&untar.stderr)
    );

    let entries: Vec<PathBuf> = std::fs::read_dir(into)
        .expect("read extraction dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    assert_eq!(
        entries.len(),
        1,
        "a binary tarball must unpack into exactly one top-level directory; \
         {:?} produced {:?}",
        archive,
        entries
    );
    let bundle: PathBuf = entries.into_iter().next().unwrap();
    assert!(bundle.is_dir(), "{:?} is not a directory", bundle);
    assert_eq!(
        bundle.file_name().map(|n| n.to_string_lossy().into_owned()),
        archive
            .file_name()
            .map(|n| n.to_string_lossy().trim_end_matches(".tar.gz").to_string()),
        "the top-level directory must match the archive stem so two unpacked \
         bundles never collide"
    );
    bundle
}

fn capi_version() -> String {
    let manifest: String =
        std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("Cargo.toml"))
            .expect("read capi Cargo.toml");
    manifest
        .lines()
        .find_map(|line| line.strip_prefix("version = "))
        .map(|v| v.trim().trim_matches('"').to_string())
        .expect("capi Cargo.toml declares a version")
}

/// The name a consumer binds to: the major SONAME on Linux/macOS, and on
/// Windows the DLL file name, which is what an import table records.
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

/// A file that is a shared library on this platform: ELF, Mach-O (including
/// the universal wrapper) or a PE image — and long enough to be one.
///
/// Existence is not enough. Staging races and truncated links both leave a
/// *present* file, and every other check — the script's own `-e` probe
/// included — passes on a zero-length one.
fn assert_is_shared_library(path: &Path) {
    let head: Vec<u8> = std::fs::read(path).expect("read the staged library");
    assert!(
        head.len() > 4096,
        "{path:?} is {} bytes — a truncated or empty library, not a shared object",
        head.len()
    );
    let is_elf: bool = head.starts_with(b"\x7fELF");
    // Mach-O 64-bit, little-endian (`MH_MAGIC_64` on disk) and the
    // universal-binary wrapper `cafebabe`.
    let is_macho: bool =
        head.starts_with(&[0xcf, 0xfa, 0xed, 0xfe]) || head.starts_with(&[0xca, 0xfe, 0xba, 0xbe]);
    let is_pe: bool = head.starts_with(b"MZ");
    assert!(
        is_elf || is_macho || is_pe,
        "{path:?} does not start with an ELF, Mach-O or PE magic; first four bytes are {:02x?}",
        &head[..4]
    );
}

/// The library for one API, as the unpacked bundle ships it.
///
/// Linux/macOS: `install_capi.sh` stages `libjpeg.so → libjpeg.so.8 →
/// libjpeg.so.8.X.Y`; an archive that dereferenced the links would install
/// three unrelated copies and leave `ldconfig` with nothing to chain, so both
/// links must still be links and resolve inside the bundle.
///
/// Windows (P4-131): there is no chain. The DLL sits in `bin/` under the name
/// consumers load, and `lib/` holds an import library that must bind to that
/// name — cargo's own import library binds to `libjpeg_turbo_rs_capi.dll`,
/// and a bundle that shipped it would link consumers to a file the bundle
/// does not contain.
fn assert_bundled_library(bundle: &Path, bundle_root: &Path, major: &str, dev: &str) {
    if cfg!(windows) {
        let dll: PathBuf = bundle.join("bin").join(major);
        assert!(
            dll.is_file(),
            "{dll:?} is not a file in the unpacked bundle"
        );
        assert_is_shared_library(&dll);
        let import_library: PathBuf = bundle.join("lib").join(dev);
        let bytes: Vec<u8> = std::fs::read(&import_library)
            .unwrap_or_else(|e| panic!("{import_library:?} missing from the bundle: {e}"));
        let mut needle: Vec<u8> = major.as_bytes().to_vec();
        needle.push(0);
        assert!(
            shell::contains_bytes(&bytes, &needle),
            "{import_library:?} does not bind its imports to {major}"
        );
        assert!(
            !shell::contains_bytes(&bytes, b"libjpeg_turbo_rs_capi.dll"),
            "{import_library:?} binds to cargo's libjpeg_turbo_rs_capi.dll, which the \
             bundle does not ship"
        );
        for symbol in cdylib_support::REQUIRED_EXPORTS {
            let mut name: Vec<u8> = symbol.as_bytes().to_vec();
            name.push(0);
            assert!(
                shell::contains_bytes(&bytes, &name),
                "{import_library:?} does not export {symbol}"
            );
        }
        // The DLL survived the GNU-tar-writes / bsdtar-reads round trip only
        // if it still loads. (No containment check against `bundle_root`
        // here: with no symlinks, a regular file cannot point outside.)
        cdylib_support::assert_library_exports(&dll, cdylib_support::REQUIRED_EXPORTS);
        return;
    }

    let lib: PathBuf = bundle.join("lib");
    let dev_path: PathBuf = lib.join(dev);
    let major_path: PathBuf = lib.join(major);
    assert!(
        dev_path.is_symlink(),
        "{dev_path:?} is not a symlink in the unpacked bundle"
    );
    assert!(
        major_path.is_symlink(),
        "{major_path:?} is not a symlink in the unpacked bundle"
    );
    let resolved: PathBuf = std::fs::canonicalize(&dev_path)
        .unwrap_or_else(|e| panic!("{dev_path:?} does not resolve inside the bundle: {e}"));
    assert!(
        resolved.starts_with(bundle_root),
        "{dev_path:?} resolves to {resolved:?}, outside the bundle — the \
         archive is not self-contained"
    );
    assert!(
        resolved.is_file(),
        "{dev_path:?} → {resolved:?} is not a file"
    );
    assert_is_shared_library(&resolved);
    cdylib_support::assert_library_exports(&resolved, cdylib_support::REQUIRED_EXPORTS);
}

/// One entry of a staged tree: a relative path and what is at it.
#[derive(Debug, PartialEq, Eq)]
enum Entry {
    /// A symlink, recorded by its literal target so a dereferenced copy of the
    /// library — three times the bytes and no SONAME chain — cannot pass.
    Symlink(String),
    /// A regular file with its permission bits. The mode is part of the
    /// comparison because it is part of what is installed: a library re-staged
    /// at `0644` is not executable-mapped the way `install -m 0755` leaves it,
    /// and content-only equality would call that identical.
    File { mode: u32, bytes: Vec<u8> },
}

/// Permission bits, or 0 on a platform that has none.
///
/// On Windows both trees compare `0 == 0`, so the mode half of the comparison
/// is inert there and the byte comparison carries it; `std::os::unix` does
/// not exist for the MSVC target.
#[cfg(unix)]
fn permission_bits(meta: &std::fs::Metadata) -> u32 {
    use std::os::unix::fs::PermissionsExt;
    meta.permissions().mode() & 0o777
}

#[cfg(not(unix))]
fn permission_bits(_meta: &std::fs::Metadata) -> u32 {
    0
}

/// Every path under `root`, relative to it, with its content or link target.
fn tree(root: &Path) -> BTreeMap<PathBuf, Entry> {
    let mut entries: BTreeMap<PathBuf, Entry> = BTreeMap::new();
    walk(root, root, &mut entries);
    entries
}

fn walk(root: &Path, dir: &Path, out: &mut BTreeMap<PathBuf, Entry>) {
    for entry in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("read_dir {dir:?}: {e}")) {
        let entry = entry.expect("dir entry");
        let path: PathBuf = entry.path();
        let relative: PathBuf = path
            .strip_prefix(root)
            .expect("child of root")
            .to_path_buf();
        // `symlink_metadata` first: a symlink to a directory would otherwise
        // be walked into, and the two trees would compare equal through the
        // link even if one of them lost the link itself.
        let meta = std::fs::symlink_metadata(&path).expect("symlink_metadata");
        if meta.file_type().is_symlink() {
            let target: PathBuf = std::fs::read_link(&path).expect("read_link");
            out.insert(
                relative,
                Entry::Symlink(target.to_string_lossy().into_owned()),
            );
        } else if meta.is_dir() {
            walk(root, &path, out);
        } else {
            let bytes: Vec<u8> = std::fs::read(&path).expect("read file");
            out.insert(
                relative,
                Entry::File {
                    mode: permission_bits(&meta),
                    bytes: comparable_bytes(&path, bytes),
                },
            );
        }
    }
}

/// The bytes of a staged file as they take part in the tree comparison.
///
/// Everything is compared verbatim except an MSVC import library, which is
/// compared by what it binds. `lib.exe` writes the current time into every
/// archive member it emits, so two stagings of the same DLL a second apart
/// differ in bytes while binding the same symbols to the same DLL name — and
/// that binding is the whole content of an import library. The comparable
/// form is the sorted set of its NUL-terminated strings: every symbol name and
/// the DLL name, with the timestamps left out. (`install_capi.sh` passes
/// `-Brepro`, which removes the stamps on toolchains that honour it; the
/// comparison does not depend on that.)
fn comparable_bytes(path: &Path, bytes: Vec<u8>) -> Vec<u8> {
    let is_import_library: bool = cfg!(windows) && path.extension().is_some_and(|ext| ext == "lib");
    if !is_import_library {
        return bytes;
    }
    let mut strings: Vec<&[u8]> = bytes
        .split(|&b| b == 0)
        .filter(|run| run.len() >= 2 && run.iter().all(|b| b.is_ascii_graphic()))
        .collect();
    strings.sort_unstable();
    strings.dedup();
    strings.join(&b'\n')
}

#[test]
fn release_bundle_carries_the_complete_installed_prefix() {
    if let Some(reason) = unsupported_host() {
        eprintln!("SKIP: {reason}");
        return;
    }
    let out: tempfile::TempDir = tempfile::tempdir().expect("mkdir outdir");
    let archive: PathBuf = package_into(out.path());

    let name: String = archive.file_name().unwrap().to_string_lossy().into_owned();
    let version: String = capi_version();
    assert!(
        name.starts_with("libjpeg-turbo-rs-capi-") && name.contains(&version),
        "archive {name:?} does not name the crate and version {version:?}; a \
         packager downloading two releases into one directory must be able to \
         tell them apart"
    );

    let unpacked: tempfile::TempDir = tempfile::tempdir().expect("mkdir unpack dir");
    let bundle: PathBuf = extract(&archive, unpacked.path());
    let lib: PathBuf = bundle.join("lib");
    let include: PathBuf = bundle.join("include");
    // Compare resolved paths against a resolved root: on macOS the temp
    // directory lives under `/var`, itself a symlink to `/private/var`, so an
    // unresolved root never prefixes anything `canonicalize` returns.
    let bundle_root: PathBuf = std::fs::canonicalize(&bundle).expect("canonicalize bundle root");

    // Both libraries, in the shape the platform's consumers bind to.
    assert_bundled_library(&bundle, &bundle_root, libjpeg_major(), libjpeg_dev());
    assert_bundled_library(
        &bundle,
        &bundle_root,
        libturbojpeg_major(),
        libturbojpeg_dev(),
    );

    for pc in ["libjpeg.pc", "libturbojpeg.pc"] {
        let path: PathBuf = lib.join("pkgconfig").join(pc);
        let body: String =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
        assert!(
            body.contains(&format!("prefix={TEST_PREFIX}")),
            "{pc} was staged with a prefix other than the requested \
             {TEST_PREFIX}:\n{body}"
        );
    }

    let cmake: PathBuf = lib.join("cmake/JPEG/JPEGConfig.cmake");
    let cmake_body: String =
        std::fs::read_to_string(&cmake).unwrap_or_else(|e| panic!("read {cmake:?}: {e}"));
    assert!(
        cmake_body.contains("JPEG::JPEG"),
        "the bundled CMake config does not define the JPEG::JPEG imported \
         target, so `find_package(JPEG)` against this prefix finds nothing \
         usable:\n{cmake_body}"
    );

    for header in [
        "jpeglib.h",
        "jerror.h",
        "jmorecfg.h",
        "jconfig.h",
        "turbojpeg.h",
    ] {
        assert!(
            include.join(header).is_file(),
            "header {header} missing from the bundle — a binary distribution \
             without headers cannot be compiled against"
        );
    }

    // The prefix baked into the `.pc` and CMake files is absolute, so a
    // packager who unpacks somewhere else has to be told what it was.
    let info: String = std::fs::read_to_string(bundle.join("BUNDLE.txt"))
        .expect("BUNDLE.txt describes the bundle");
    for needle in [
        &format!("version: {version}"),
        &format!("prefix: {TEST_PREFIX}"),
        &format!("soname: {}", libjpeg_major()),
    ] {
        assert!(
            info.contains(needle.as_str()),
            "BUNDLE.txt does not record `{needle}`:\n{info}"
        );
    }
    assert!(
        info.contains("target: "),
        "BUNDLE.txt does not record the target triple, so two bundles for \
         different architectures are indistinguishable once unpacked:\n{info}"
    );
}

#[test]
fn release_bundle_checksum_verifies_from_the_download_directory() {
    if let Some(reason) = unsupported_host() {
        eprintln!("SKIP: {reason}");
        return;
    }
    // Both probes assert the tool *ran*, not merely that it spawned: an
    // `is_ok()` probe is true for any binary that starts, so a broken one
    // would turn this into a silent skip.
    //
    // On Windows the verifier is the `sha256sum` inside Git for Windows, which
    // is what BUNDLE.txt tells a Windows user to run and is not on the PATH
    // the test process inherited; it is reached through the same bash that
    // ran the packaging script.
    let checker: &str = if cfg!(windows) {
        let probe = Command::new(shell::bash().expect("checked by unsupported_host"))
            .args(["-c", "sha256sum --version"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if !probe {
            eprintln!("SKIP: Git for Windows' bash has no sha256sum");
            return;
        }
        "sha256sum"
    } else if have("sha256sum") {
        "sha256sum"
    } else if Command::new("shasum")
        .arg("-v")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
    {
        "shasum"
    } else {
        eprintln!("SKIP: neither sha256sum nor shasum is on PATH");
        return;
    };
    // `<checker> -c <manifest>` run from `dir`, the way a user does it.
    let verify_in = |dir: &Path, manifest: &str| -> std::process::Output {
        let mut verify: Command = if cfg!(windows) {
            let mut through_bash =
                Command::new(shell::bash().expect("checked by unsupported_host"));
            through_bash.args(["-c", "sha256sum -c \"$1\"", "sha256sum"]);
            through_bash
        } else {
            let mut direct = Command::new(checker);
            if checker == "shasum" {
                direct.args(["-a", "256", "-c"]);
            } else {
                direct.arg("-c");
            }
            direct
        };
        verify
            .arg(manifest)
            .current_dir(dir)
            .output()
            .expect("invoke the checksum verifier")
    };

    let out: tempfile::TempDir = tempfile::tempdir().expect("mkdir outdir");
    let archive: PathBuf = package_into(out.path());
    let manifest: PathBuf = archive.with_extension("gz.sha256");
    let body: String = std::fs::read_to_string(&manifest)
        .unwrap_or_else(|e| panic!("the archive has no checksum manifest at {manifest:?}: {e}"));

    let archive_name: String = archive.file_name().unwrap().to_string_lossy().into_owned();
    assert!(
        body.trim_end().ends_with(&format!("  {archive_name}")),
        "the manifest must name the bare archive so `{checker} -c` works in \
         the directory it was downloaded into; it says:\n{body}"
    );
    let manifest_name: String = manifest.file_name().unwrap().to_string_lossy().into_owned();

    // Verify the way a user does: from the download directory, by the manifest.
    let verified = verify_in(out.path(), &manifest_name);
    assert!(
        verified.status.success(),
        "`{checker} -c {manifest_name}` rejected the archive it was generated for:\n{}\n{}",
        String::from_utf8_lossy(&verified.stdout),
        String::from_utf8_lossy(&verified.stderr)
    );

    // And that it is a real check, not a tautology: flip one byte and the
    // same command must reject it. Without this the test passes against a
    // manifest generated from whatever the archive happens to be.
    let mut bytes: Vec<u8> = std::fs::read(&archive).expect("read archive");
    let last: usize = bytes.len() - 1;
    bytes[last] ^= 0xff;
    std::fs::write(&archive, &bytes).expect("rewrite archive");
    let rejected = verify_in(out.path(), &manifest_name);
    assert!(
        !rejected.status.success(),
        "a corrupted archive passed `{checker} -c`, so the manifest is not \
         bound to the bytes it ships with"
    );
}

/// The release runs the packaging script with no `--prefix`, so the platform
/// default — `C:/libjpeg-turbo-rs64` on Windows, with the drive-stripping
/// DESTDIR arithmetic both scripts must agree on — is what a published
/// bundle records. Every other test here passes an explicit POSIX prefix.
#[test]
fn release_bundle_records_the_platform_default_prefix() {
    if let Some(reason) = unsupported_host() {
        eprintln!("SKIP: {reason}");
        return;
    }
    let out: tempfile::TempDir = tempfile::tempdir().expect("mkdir outdir");
    let run = script_command("package_capi_release.sh")
        .args(["--outdir", &shell::script_arg(out.path())])
        .output()
        .expect("invoke package_capi_release.sh");
    assert!(
        run.status.success(),
        "package_capi_release.sh failed without --prefix:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let archive: PathBuf = std::fs::read_dir(out.path())
        .expect("read outdir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .find(|p| p.to_string_lossy().ends_with(".tar.gz"))
        .expect("the archive is present");
    let unpacked: tempfile::TempDir = tempfile::tempdir().expect("mkdir unpack dir");
    let bundle: PathBuf = extract(&archive, unpacked.path());
    let bundle_root: PathBuf = std::fs::canonicalize(&bundle).expect("canonicalize bundle root");

    let expected_prefix: &str = if cfg!(windows) {
        "C:/libjpeg-turbo-rs64"
    } else {
        "/usr/local"
    };
    let info: String = std::fs::read_to_string(bundle.join("BUNDLE.txt")).expect("BUNDLE.txt");
    assert!(
        info.contains(&format!("prefix: {expected_prefix}")),
        "BUNDLE.txt does not record the default prefix {expected_prefix}:\n{info}"
    );
    let pc: String = std::fs::read_to_string(bundle.join("lib/pkgconfig/libjpeg.pc"))
        .expect("libjpeg.pc in the bundle");
    assert!(
        pc.contains(&format!("prefix={expected_prefix}")),
        "the bundled libjpeg.pc does not carry the default prefix {expected_prefix}:\n{pc}"
    );
    // And the prefix was found below DESTDIR at all: a disagreement between
    // the two scripts' drive-stripping would have packaged an empty tree.
    assert_bundled_library(&bundle, &bundle_root, libjpeg_major(), libjpeg_dev());
}

/// The release job passes `--target <triple>`, which the other tests here do
/// not: they pin `CAPI_TARGET_DIR` at the host's `deps/` directory, so
/// `install_capi.sh` never derives a target-qualified path and never builds.
///
/// Driving the full `--target … --build` path would mean a fresh release
/// build per platform in every CI leg. What this pins instead is the part
/// that can silently rot — that `--target` reaches the nested build at all.
/// A script that dropped the flag would build and package the *host* library
/// under a cross target's name, and no assertion about the bundle's shape
/// would notice. Naming a triple no toolchain has makes the failure the
/// evidence.
#[test]
fn package_capi_release_sh_threads_target_through_to_the_build() {
    if let Some(reason) = unsupported_host() {
        eprintln!("SKIP: {reason}");
        return;
    }
    let out: tempfile::TempDir = tempfile::tempdir().expect("mkdir outdir");
    // On Windows the scripts refuse any target but the MSVC one they stage a
    // layout for, so the nonexistent triple has to keep that shape to get
    // as far as the build — which is the step this test is about.
    const FAKE_TARGET: &str = if cfg!(windows) {
        "x86_64-nonesuch-windows-msvc"
    } else {
        "x86_64-unknown-nonesuch-elf"
    };

    let run = script_command("package_capi_release.sh")
        .args(["--outdir", &shell::script_arg(out.path())])
        .args(["--prefix", TEST_PREFIX])
        .args(["--target", FAKE_TARGET])
        // Deliberately no CAPI_TARGET_DIR: that is what forces install_capi.sh
        // to derive the release directory from the target, exactly as the
        // release job does.
        .env_remove("CAPI_TARGET_DIR")
        .output()
        .expect("invoke package_capi_release.sh");

    assert!(
        !run.status.success(),
        "packaging succeeded for target {FAKE_TARGET:?}, which no toolchain \
         can build — so the target was not used to select what gets packaged"
    );
    let output: String = format!(
        "{}{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(
        output.contains(FAKE_TARGET),
        "the failure never mentions {FAKE_TARGET:?}, so `--target` did not \
         reach the staging path and the release's cross-built legs would \
         package the host library:\n{output}"
    );
    // cargo's own diagnostic, not one of the scripts' target guards: a guard
    // that rejected the triple up front would also fail and also name it,
    // and the test would hold with `--target` never reaching the build.
    assert!(
        output.contains("error: "),
        "the failure came from a script guard rather than from cargo, so the \
         nested build never saw {FAKE_TARGET:?}:\n{output}"
    );
    let leftovers: Vec<PathBuf> = std::fs::read_dir(out.path())
        .expect("read outdir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    assert!(
        leftovers.is_empty(),
        "a failed packaging run left {leftovers:?} in the output directory; a \
         release would attach it"
    );
}

/// P4-131 criterion 3 / P4-124: one staging path, not two.
///
/// The packaging script must obtain its tree from `scripts/install_capi.sh`
/// and add nothing but its own description. Anything it stages itself — a
/// header copied from a different place, a `.pc` written with different
/// substitutions, a library taken straight from `target/` instead of the
/// relinked one — is a divergence between what we ship and what the
/// downstream harnesses test, and shows up here as a differing entry.
#[test]
fn release_bundle_is_exactly_what_install_capi_sh_stages() {
    if let Some(reason) = unsupported_host() {
        eprintln!("SKIP: {reason}");
        return;
    }
    let out: tempfile::TempDir = tempfile::tempdir().expect("mkdir outdir");
    let archive: PathBuf = package_into(out.path());
    let unpacked: tempfile::TempDir = tempfile::tempdir().expect("mkdir unpack dir");
    let bundle: PathBuf = extract(&archive, unpacked.path());

    // The reference: the staging path itself, same inputs, same prefix.
    let staged_root: tempfile::TempDir = tempfile::tempdir().expect("mkdir staging dir");
    let install = script_command("install_capi.sh")
        .args(["--destdir", &shell::script_arg(staged_root.path())])
        .args(["--prefix", TEST_PREFIX])
        .output()
        .expect("invoke install_capi.sh");
    assert!(
        install.status.success(),
        "install_capi.sh failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&install.stdout),
        String::from_utf8_lossy(&install.stderr)
    );
    let staged: PathBuf = staged_root.path().join(TEST_PREFIX.trim_start_matches('/'));

    let mut bundled: BTreeMap<PathBuf, Entry> = tree(&bundle);
    let reference: BTreeMap<PathBuf, Entry> = tree(&staged);

    // BUNDLE.txt is the packaging script's own contribution and has no
    // counterpart in the staged prefix. It is the *only* one allowed.
    assert!(
        bundled.remove(Path::new("BUNDLE.txt")).is_some(),
        "BUNDLE.txt missing from the bundle"
    );

    let only_in_bundle: Vec<&PathBuf> = bundled
        .keys()
        .filter(|k| !reference.contains_key(*k))
        .collect();
    let only_in_staging: Vec<&PathBuf> = reference
        .keys()
        .filter(|k| !bundled.contains_key(*k))
        .collect();
    assert!(
        only_in_bundle.is_empty() && only_in_staging.is_empty(),
        "the bundle and `install_capi.sh` disagree about what is shipped, so \
         there are two staging paths (P4-131 criterion 3 / P4-124).\n\
         only in the bundle: {only_in_bundle:?}\n\
         only in the staged prefix: {only_in_staging:?}"
    );

    let differing: Vec<String> = bundled
        .iter()
        .filter(|(path, entry)| reference.get(*path) != Some(*entry))
        .map(|(path, entry)| match (entry, &reference[path]) {
            (Entry::Symlink(a), Entry::Symlink(b)) => {
                format!(
                    "  {}: bundle links to {a}, staging links to {b}",
                    path.display()
                )
            }
            (Entry::Symlink(_), Entry::File { .. }) => {
                format!(
                    "  {}: a symlink in the bundle, a file in staging",
                    path.display()
                )
            }
            (Entry::File { .. }, Entry::Symlink(_)) => {
                format!(
                    "  {}: a file in the bundle, a symlink in staging",
                    path.display()
                )
            }
            (
                Entry::File {
                    mode: bundle_mode,
                    bytes: bundle_bytes,
                },
                Entry::File {
                    mode: staged_mode,
                    bytes: staged_bytes,
                },
            ) => format!(
                "  {}: {} bytes mode {bundle_mode:o} in the bundle, {} bytes mode {staged_mode:o} in staging",
                path.display(),
                bundle_bytes.len(),
                staged_bytes.len()
            ),
        })
        .collect();
    assert!(
        differing.is_empty(),
        "the bundle's contents differ from what `install_capi.sh` stages — \
         the packaging script is staging something itself instead of \
         re-using the one path (P4-131 criterion 3 / P4-124):\n{}",
        differing.join("\n")
    );
}

// ---------------------------------------------------------------------------
// P4-131 criterion 4: signing and SBOM.
//
// A checksum published beside the file it covers proves integrity, not
// origin. The release job therefore attests each bundle with Sigstore build
// provenance and a CycloneDX SBOM, both signed by the job's OIDC identity and
// stored under this repository, so `gh attestation verify` can check where a
// download came from. The attestation itself is only observable on a real run
// of `release.yml` — a `workflow_dispatch` exercises it without publishing —
// so what this file pins is the two halves that *can* be checked from a pull
// request: the SBOM the packaging script produces, and the workflow shape
// that signs it. The dispatch run that proved the whole path is cited in
// `docs/last_mile/phase4.md` § P4-131.
// ---------------------------------------------------------------------------

fn have_cargo_cyclonedx() -> bool {
    Command::new("cargo")
        .args(["cyclonedx", "--version"])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn root_manifest() -> String {
    std::fs::read_to_string(workspace_root().join("Cargo.toml")).expect("read root Cargo.toml")
}

/// The root crate's version, read from its `[package]` table specifically: a
/// `version =` in some other table above it must not retarget the assertion.
fn root_crate_version() -> String {
    root_manifest()
        .lines()
        .skip_while(|line| line.trim() != "[package]")
        .skip(1)
        .take_while(|line| !line.starts_with('['))
        .find_map(|line| line.strip_prefix("version = "))
        .map(|v| v.trim().trim_matches('"').to_string())
        .expect("root Cargo.toml declares a version under [package]")
}

/// Every workspace member's directory, from the root manifest's `members`
/// list — so a crate added later is checked for stray SBOMs without anyone
/// remembering to extend a hard-coded list here.
fn workspace_member_dirs() -> Vec<PathBuf> {
    let manifest: String = root_manifest();
    let members: &str = manifest
        .split_once("members = [")
        .map(|(_, rest)| rest)
        .and_then(|rest| rest.split_once(']'))
        .map(|(list, _)| list)
        .expect("root Cargo.toml declares workspace members");
    let dirs: Vec<PathBuf> = members
        .split(',')
        .map(|m| m.trim().trim_matches('"'))
        .filter(|m| !m.is_empty())
        .map(|m| workspace_root().join(m))
        .collect();
    assert!(
        dirs.len() >= 2,
        "expected several workspace members, parsed {dirs:?}"
    );
    dirs
}

/// The Sigstore bundle, once verified, only proves what the SBOM *said* at
/// signing time. This pins what it says: a CycloneDX document whose subject is
/// the capi crate at the version the archive is named after, listing the root
/// crate it compiles against — a document naming some other crate, or an
/// empty component list, would attest and verify just as cleanly.
#[test]
fn release_bundle_ships_a_cyclonedx_sbom_beside_the_archive() {
    if let Some(reason) = unsupported_host() {
        eprintln!("SKIP: {reason}");
        return;
    }
    if !have_cargo_cyclonedx() {
        eprintln!("SKIP: cargo-cyclonedx is not installed (cargo install cargo-cyclonedx)");
        return;
    }
    let out: tempfile::TempDir = tempfile::tempdir().expect("mkdir outdir");

    let run = script_command("package_capi_release.sh")
        .args(["--outdir", &shell::script_arg(out.path())])
        .args(["--prefix", TEST_PREFIX])
        .arg("--sbom")
        .output()
        .expect("invoke package_capi_release.sh");
    assert!(
        run.status.success(),
        "package_capi_release.sh --sbom failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );

    let mut names: Vec<String> = std::fs::read_dir(out.path())
        .expect("read outdir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    let archive: &String = names
        .iter()
        .find(|n| n.ends_with(".tar.gz"))
        .expect("the archive is present");
    let stem: &str = archive.trim_end_matches(".tar.gz");
    let sbom_name: String = format!("{stem}.cdx.json");
    assert!(
        names.contains(&sbom_name),
        "no {sbom_name} beside the archive; outdir holds {names:?}"
    );
    // The SBOM is an attached artifact, so criterion 2 applies to it too.
    assert!(
        names.contains(&format!("{sbom_name}.sha256")),
        "the SBOM has no checksum manifest; outdir holds {names:?}"
    );
    // cargo-cyclonedx writes one document per workspace member. Only the capi
    // crate's belongs in the bundle, and none may be left in the source tree
    // where the next `cargo publish --allow-dirty` would ship it.
    let strays: Vec<PathBuf> = workspace_member_dirs()
        .into_iter()
        .map(|dir| dir.join(&sbom_name))
        .filter(|p| p.exists())
        .collect();
    assert!(
        strays.is_empty(),
        "the SBOM generator left documents in the source tree: {strays:?}"
    );

    let body: String = std::fs::read_to_string(out.path().join(&sbom_name)).expect("read the SBOM");
    // Enough of the document to know it is *our* SBOM, without a JSON
    // dependency: CycloneDX's JSON encoding is stable for these keys.
    assert!(
        body.contains("\"bomFormat\": \"CycloneDX\"")
            || body.contains("\"bomFormat\":\"CycloneDX\""),
        "{sbom_name} is not a CycloneDX document:\n{}",
        &body[..body.len().min(400)]
    );
    let capi_version: String = capi_version();
    let subject_purl: String = format!("pkg:cargo/libjpeg-turbo-rs-capi@{capi_version}");
    assert!(
        body.contains(&subject_purl),
        "{sbom_name} does not describe {subject_purl} — the crate the archive is \
         named after:\n{}",
        &body[..body.len().min(1200)]
    );
    let root_purl: String = format!("pkg:cargo/libjpeg-turbo-rs@{}", root_crate_version());
    assert!(
        body.contains(&root_purl),
        "{sbom_name} does not list {root_purl}, the crate the shim compiles \
         against, so it is not the dependency graph of the shipped library"
    );
}

/// A workflow file's text with line endings normalised. The Windows leg of
/// `capi-abi-checks` checks out with `core.autocrlf`, so the file on disk is
/// CRLF there; every anchor below is written against `\n`.
fn workflow_text(name: &str) -> String {
    std::fs::read_to_string(workspace_root().join(".github/workflows").join(name))
        .unwrap_or_else(|e| panic!("read {name}: {e}"))
        .replace("\r\n", "\n")
}

/// The text of one job of a workflow file: from its key to the next
/// top-level job key.
fn workflow_job(file: &str, job: &str) -> String {
    let workflow: String = workflow_text(file);
    // Anchored to a line start, so a deeper-indented `job:` (a step id, a
    // matrix key) cannot match first.
    let header: String = format!("\n  {job}:\n");
    let start: usize = workflow
        .find(&header)
        .unwrap_or_else(|| panic!("{file} has no `{job}` job"));
    let rest: &str = &workflow[start + header.len()..];
    // The next job: a line indented exactly two spaces that names a key and
    // is not a comment.
    let end: usize = rest
        .lines()
        .scan(0usize, |offset, line| {
            let at: usize = *offset;
            *offset += line.len() + 1;
            Some((at, line))
        })
        .find(|(_, line)| {
            line.starts_with("  ")
                && !line.starts_with("   ")
                && !line.trim_start().starts_with('#')
                && line.trim_end().ends_with(':')
        })
        .map(|(at, _)| at)
        .unwrap_or(rest.len());
    rest[..end].to_string()
}

fn release_workflow_job(job: &str) -> String {
    workflow_job("release.yml", job)
}

/// Whether a step block carries its own `if:` key — at the step's key
/// indentation, so a comment mentioning `if:` does not count.
fn step_is_conditional(step: &str) -> bool {
    step.lines().any(|line| line.starts_with("        if:"))
}

/// The steps of a job's text, each as its own block.
fn workflow_steps(job_text: &str) -> Vec<String> {
    let steps_at: usize = match job_text.find("    steps:\n") {
        Some(at) => at + "    steps:\n".len(),
        None => return Vec::new(),
    };
    let mut steps: Vec<String> = Vec::new();
    for line in job_text[steps_at..].lines() {
        if line.starts_with("      - ") {
            steps.push(String::new());
        }
        if let Some(current) = steps.last_mut() {
            current.push_str(line);
            current.push('\n');
        }
    }
    steps
}

/// The bundle job signs what it built, in the job that built it, on every
/// event that runs it — so a `workflow_dispatch` rehearsal exercises the
/// signing path and a tag cannot be the first time it runs. A step gated on
/// `push` would pass every rehearsal and fail on the release.
#[test]
fn release_workflow_attests_provenance_and_sbom_in_the_bundle_job() {
    let job: String = release_workflow_job("native-artifacts");
    for grant in ["id-token: write", "attestations: write"] {
        assert!(
            job.contains(grant),
            "native-artifacts does not grant `{grant}`, so the attest steps \
             cannot obtain a Sigstore identity or store the result:\n{job}"
        );
    }

    let steps: Vec<String> = workflow_steps(&job);
    let provenance: &String = steps
        .iter()
        .find(|s| s.contains("uses: actions/attest-build-provenance@"))
        .expect("native-artifacts has an actions/attest-build-provenance step");
    let sbom: &String = steps
        .iter()
        .find(|s| s.contains("uses: actions/attest@"))
        .expect("native-artifacts has an actions/attest step for the SBOM");
    for (what, step) in [("provenance", provenance), ("SBOM", sbom)] {
        assert!(
            step.contains("subject-path: dist/*.tar.gz"),
            "the {what} attestation's subject is not the archive the job \
             packaged into dist/:\n{step}"
        );
        assert!(
            !step_is_conditional(step),
            "the {what} attestation is conditional, so a dispatch rehearsal \
             would not exercise it:\n{step}"
        );
    }
    assert!(
        sbom.contains("sbom-path: dist/") && sbom.contains(".cdx.json"),
        "the SBOM attestation does not attest the CycloneDX document the \
         packaging script wrote beside the archive:\n{sbom}"
    );

    // The packaging step must ask for the SBOM, or the attest step has
    // nothing to sign and the failure surfaces one step late.
    let package: &String = steps
        .iter()
        .find(|s| s.contains("scripts/package_capi_release.sh"))
        .expect("native-artifacts runs the packaging script");
    assert!(
        package.contains("--sbom"),
        "the packaging step does not pass --sbom:\n{package}"
    );
}

/// What the release attaches is what a downloader can verify offline: the
/// Sigstore bundles beside the archive, the SBOM and its checksum, and a
/// `SHA256SUMS` that covers the SBOM as well as the archives (criterion 2
/// applies to every attached artifact).
#[test]
fn release_workflow_attaches_the_sbom_and_sigstore_bundles() {
    let job: String = release_workflow_job("github-release");
    // Two paths attach files: `gh release create` for a new release and `gh
    // release upload` for one somebody drafted by hand. Each is one shell
    // command continued over several lines; join the continuations so the
    // globs are whole tokens, and match tokens rather than substrings —
    // `dist/*.cdx.json` is a prefix of `dist/*.cdx.json.sha256`, and a
    // substring count would stay satisfied with the SBOM itself removed.
    let joined: String = job.replace("\\\n", " ");
    for command in ["gh release create", "gh release upload"] {
        let line: &str = joined
            .lines()
            .find(|l| l.contains(command))
            .unwrap_or_else(|| panic!("github-release has no `{command}` command:\n{job}"));
        let tokens: Vec<&str> = line.split_whitespace().collect();
        for glob in [
            "dist/*.tar.gz",
            "dist/*.tar.gz.sha256",
            "dist/*.cdx.json",
            "dist/*.cdx.json.sha256",
            "dist/*.sigstore.json",
            "dist/SHA256SUMS",
        ] {
            assert!(
                tokens.contains(&glob),
                "`{command}` does not attach `{glob}`:\n{line}"
            );
        }
    }
    assert!(
        job.contains("./*.cdx.json.sha256"),
        "SHA256SUMS is folded from the archive checksums only; the SBOM is an \
         attached artifact and must be covered too:\n{job}"
    );
}

/// The matrix rows of a job's `strategy`, each as its own text block.
fn workflow_matrix_rows(job_text: &str) -> Vec<String> {
    let include_at: usize = match job_text.find("        include:\n") {
        Some(at) => at + "        include:\n".len(),
        None => return Vec::new(),
    };
    let mut rows: Vec<String> = Vec::new();
    for line in job_text[include_at..].lines() {
        // A comment at row indentation introduces the *next* row; grouping
        // it with the previous one would let a triple named in a comment
        // satisfy a lookup for that row.
        if line.starts_with("          - ") || line.starts_with("          #") {
            rows.push(String::new());
        } else if !line.starts_with("            ") && !line.trim_start().starts_with('#') {
            break;
        }
        if let Some(current) = rows.last_mut() {
            current.push_str(line);
            current.push('\n');
        }
    }
    rows
}

/// P4-131 criterion 1, Windows: the release builds the MSVC bundle on a
/// Windows runner, and runs its steps under bash there — the job's steps are
/// bash (`set -o pipefail`, `shopt`, `[ ]`), and a Windows runner's default
/// shell is PowerShell, which would fail the first of them.
#[test]
fn release_workflow_builds_the_windows_bundle_under_bash() {
    let job: String = release_workflow_job("native-artifacts");
    let rows: Vec<String> = workflow_matrix_rows(&job);
    let windows: &String = rows
        .iter()
        .find(|row| row.contains("target: x86_64-pc-windows-msvc"))
        .unwrap_or_else(|| {
            panic!("native-artifacts has no x86_64-pc-windows-msvc matrix row:\n{job}")
        });
    assert!(
        windows.contains("os: windows-latest"),
        "the x86_64-pc-windows-msvc bundle is not built on a Windows runner:\n{windows}"
    );
    let defaults_at: usize = job
        .find("    defaults:\n")
        .unwrap_or_else(|| panic!("native-artifacts sets no job-level defaults:\n{job}"));
    let defaults: &str = &job[defaults_at..];
    assert!(
        defaults.contains("      run:\n        shell: bash\n"),
        "native-artifacts does not default its run steps to bash, so the \
         Windows leg would run them under PowerShell:\n{job}"
    );
}

/// The SBOM test can only run where the generator is installed. `capi-abi-checks`
/// installs it on every leg — a Windows-gated step would leave the Windows
/// bundle's SBOM path exercised by nothing but a dispatch rehearsal.
#[test]
fn sbom_generator_is_installed_on_every_capi_abi_checks_leg() {
    let job: String = workflow_job("ci.yml", "capi-abi-checks");
    let steps: Vec<String> = workflow_steps(&job);
    let install: &String = steps
        .iter()
        .find(|step| step.contains("cargo install cargo-cyclonedx"))
        .expect("capi-abi-checks installs cargo-cyclonedx");
    assert!(
        !step_is_conditional(install),
        "capi-abi-checks gates the cargo-cyclonedx install, so the SBOM test skips \
         on that leg:\n{install}"
    );
}

/// `cargo-cyclonedx` is pinned wherever a workflow installs it, and to one
/// version: the release job that ships the SBOM and the CI job that tests the
/// script producing it must run the same generator, or the test proves
/// nothing about the release.
#[test]
fn sbom_generator_is_pinned_to_one_version_across_workflows() {
    let mut pins: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for name in ["ci.yml", "release.yml"] {
        let text: String = workflow_text(name);
        for line in text
            .lines()
            .filter(|l| l.contains("cargo install cargo-cyclonedx"))
        {
            let version: &str = line
                .split_whitespace()
                .skip_while(|w| *w != "--version")
                .nth(1)
                .unwrap_or_else(|| {
                    panic!("{name} installs cargo-cyclonedx without --version: {line}")
                });
            assert!(
                line.contains("--locked"),
                "{name} installs cargo-cyclonedx without --locked: {line}"
            );
            pins.entry(version.to_string())
                .or_default()
                .push(name.to_string());
        }
    }
    assert!(
        pins.values().flatten().any(|n| n == "release.yml")
            && pins.values().flatten().any(|n| n == "ci.yml"),
        "cargo-cyclonedx must be installed by both release.yml and ci.yml; found {pins:?}"
    );
    assert_eq!(
        pins.len(),
        1,
        "cargo-cyclonedx is pinned to more than one version: {pins:?}"
    );
}

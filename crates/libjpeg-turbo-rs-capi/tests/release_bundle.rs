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
//! which is why the last test here compares the bundle against a direct
//! `install_capi.sh` run file by file rather than against a list of expected
//! names.
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
//!
//! Skip-with-reason cases mirror `install_layout.rs`: Windows (the scripts are
//! bash, and a Windows bundle is still open under P4-131), and hosts without
//! `bash` or `tar`.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "support/cdylib.rs"]
mod cdylib_support;

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
    if cfg!(windows) {
        return Some("the packaging scripts are bash; a Windows bundle is still open under P4-131");
    }
    if !have("bash") {
        return Some("bash not on PATH");
    }
    if !have("tar") {
        return Some("tar not on PATH");
    }
    None
}

/// Runs `scripts/package_capi_release.sh` into `outdir` and returns the
/// archive it produced.
///
/// Failure is fatal, never a skip: reaching here means the host *can* run the
/// script, so a non-zero exit is a defect in the thing under test.
fn package_into(outdir: &Path) -> PathBuf {
    let root: PathBuf = workspace_root();
    let cdylib: PathBuf = cdylib_support::cargo_built_cdylib_path()
        .unwrap_or_else(|e| panic!("could not locate the cdylib under test: {e}"));
    let cdylib_dir: &Path = cdylib.parent().expect("Cargo artifact directory");

    let run = Command::new("bash")
        .arg(root.join("scripts/package_capi_release.sh"))
        .args(["--outdir", &outdir.to_string_lossy()])
        .args(["--prefix", TEST_PREFIX])
        .args(["--root", &root.to_string_lossy()])
        .env("CAPI_TARGET_DIR", cdylib_dir)
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
    let untar = Command::new("tar")
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

fn libjpeg_major() -> &'static str {
    if cfg!(target_os = "macos") {
        "libjpeg.8.dylib"
    } else {
        "libjpeg.so.8"
    }
}

fn libjpeg_dev() -> &'static str {
    if cfg!(target_os = "macos") {
        "libjpeg.dylib"
    } else {
        "libjpeg.so"
    }
}

fn libturbojpeg_major() -> &'static str {
    if cfg!(target_os = "macos") {
        "libturbojpeg.0.dylib"
    } else {
        "libturbojpeg.so.0"
    }
}

fn libturbojpeg_dev() -> &'static str {
    if cfg!(target_os = "macos") {
        "libturbojpeg.dylib"
    } else {
        "libturbojpeg.so"
    }
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
/// Every test here skips on Windows — the bundle is a Unix artifact — but the
/// file still has to *compile* there: `cargo test --workspace --no-run` and
/// the `capi-abi-checks` matrix both build it on `windows-latest`, ahead of any
/// runtime skip. `std::os::unix` does not exist for the MSVC target.
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
            out.insert(
                relative,
                Entry::File {
                    mode: permission_bits(&meta),
                    bytes: std::fs::read(&path).expect("read file"),
                },
            );
        }
    }
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

    // Both SONAME chains, with the links intact. `install_capi.sh` stages
    // `libjpeg.so → libjpeg.so.8 → libjpeg.so.8.X.Y`; an archive that
    // dereferenced them would install three unrelated copies and leave
    // `ldconfig` with nothing to chain.
    for (dev, major) in [
        (libjpeg_dev(), libjpeg_major()),
        (libturbojpeg_dev(), libturbojpeg_major()),
    ] {
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
            resolved.starts_with(&bundle_root),
            "{dev_path:?} resolves to {resolved:?}, outside the bundle — the \
             archive is not self-contained"
        );
        assert!(
            resolved.is_file(),
            "{dev_path:?} → {resolved:?} is not a file"
        );

        // Existence is not enough. Staging races and truncated links both
        // leave a *present* file at the end of a resolving chain, and every
        // other check here — the script's own `-e` probe included — passes on
        // a zero-length one. Assert it is actually a shared library.
        let head: Vec<u8> = std::fs::read(&resolved).expect("read the staged library");
        assert!(
            head.len() > 4096,
            "{resolved:?} is {} bytes — a truncated or empty library, not a \
             shared object",
            head.len()
        );
        let is_elf: bool = head.starts_with(b"\x7fELF");
        // Mach-O 64-bit, little-endian (`MH_MAGIC_64` on disk) and the
        // universal-binary wrapper `cafebabe`.
        let is_macho: bool = head.starts_with(&[0xcf, 0xfa, 0xed, 0xfe])
            || head.starts_with(&[0xca, 0xfe, 0xba, 0xbe]);
        assert!(
            is_elf || is_macho,
            "{resolved:?} does not start with an ELF or Mach-O magic; first \
             four bytes are {:02x?}",
            &head[..4]
        );
    }

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
    let checker: &str = if have("sha256sum") {
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

    // Verify the way a user does: from the download directory, by the manifest.
    let mut verify = Command::new(checker);
    if checker == "shasum" {
        verify.args(["-a", "256"]);
    }
    let verified = verify
        .arg("-c")
        .arg(manifest.file_name().unwrap())
        .current_dir(out.path())
        .output()
        .expect("invoke the checksum verifier");
    assert!(
        verified.status.success(),
        "`{checker} -c {}` rejected the archive it was generated for:\n{}\n{}",
        manifest.file_name().unwrap().to_string_lossy(),
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
    let mut reverify = Command::new(checker);
    if checker == "shasum" {
        reverify.args(["-a", "256"]);
    }
    let rejected = reverify
        .arg("-c")
        .arg(manifest.file_name().unwrap())
        .current_dir(out.path())
        .output()
        .expect("invoke the checksum verifier");
    assert!(
        !rejected.status.success(),
        "a corrupted archive passed `{checker} -c`, so the manifest is not \
         bound to the bytes it ships with"
    );
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
    let root: PathBuf = workspace_root();
    let out: tempfile::TempDir = tempfile::tempdir().expect("mkdir outdir");
    const FAKE_TARGET: &str = "x86_64-unknown-nonesuch-elf";

    let run = Command::new("bash")
        .arg(root.join("scripts/package_capi_release.sh"))
        .args(["--outdir", &out.path().to_string_lossy()])
        .args(["--prefix", TEST_PREFIX])
        .args(["--root", &root.to_string_lossy()])
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
    let root: PathBuf = workspace_root();
    let cdylib: PathBuf = cdylib_support::cargo_built_cdylib_path()
        .unwrap_or_else(|e| panic!("could not locate the cdylib under test: {e}"));
    let cdylib_dir: &Path = cdylib.parent().expect("Cargo artifact directory");

    let out: tempfile::TempDir = tempfile::tempdir().expect("mkdir outdir");
    let archive: PathBuf = package_into(out.path());
    let unpacked: tempfile::TempDir = tempfile::tempdir().expect("mkdir unpack dir");
    let bundle: PathBuf = extract(&archive, unpacked.path());

    // The reference: the staging path itself, same inputs, same prefix.
    let staged_root: tempfile::TempDir = tempfile::tempdir().expect("mkdir staging dir");
    let install = Command::new("bash")
        .arg(root.join("scripts/install_capi.sh"))
        .args(["--destdir", &staged_root.path().to_string_lossy()])
        .args(["--prefix", TEST_PREFIX])
        .args(["--root", &root.to_string_lossy()])
        .env("CAPI_TARGET_DIR", cdylib_dir)
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

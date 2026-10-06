#!/usr/bin/env bash
# package_capi_release.sh — turn the prefix `scripts/install_capi.sh` stages
# into the checksummed archive a release attaches.
#
# This is the P4-131 distribution path. Before it, `release.yml` published to
# crates.io and npm only, so replacing a system `libjpeg.so.8` meant cloning
# the repository, installing a Rust toolchain and running the install script
# by hand.
#
# **It stages nothing itself.** Every file in the archive comes from
# `install_capi.sh`; the only file this script adds is `BUNDLE.txt`, which
# describes the bundle. That invariant is enforced by
# `crates/libjpeg-turbo-rs-capi/tests/release_bundle.rs`, which compares the
# unpacked archive against a direct `install_capi.sh` run entry by entry.
#
# It matters because `install_capi.sh` is the path P4-124 will point the
# downstream harnesses at. Today they still stage the raw cargo cdylib, so
# "what you download is what the harnesses test" is the goal this makes
# reachable, not a property the repository has yet — P4-124 is open.
#
# Usage:
#   scripts/package_capi_release.sh --outdir dist
#   scripts/package_capi_release.sh --outdir dist --prefix /usr --build
#
# Flags:
#   --outdir DIR      Where the archive and its checksum are written
#                     (required; created if absent).
#   --prefix DIR      Prefix baked into the staged `.pc` and CMake files
#                     (default "/usr/local"; "C:/libjpeg-turbo-rs64" on
#                     Windows — the same defaults as install_capi.sh).
#                     Absolute by nature — the bundle records it in
#                     BUNDLE.txt so a packager who unpacks somewhere else
#                     knows what to relocate.
#   --target TRIPLE   Cargo target triple, used both for the nested build and
#                     as the bundle's platform label (default: Cargo's host).
#                     On Windows only x86_64-pc-windows-msvc is accepted:
#                     the staged layout is upstream's MSVC one (P4-131).
#   --root DIR        Repository root (default: this script's parent).
#   --build           Force a cdylib build. Passed through; without it, a
#                     missing cdylib is still built by the install script.
#   --sbom            Also write a CycloneDX SBOM of the capi crate for the
#                     target, plus its checksum. Needs `cargo cyclonedx`
#                     (`cargo install cargo-cyclonedx --locked`); refuses to
#                     run without it rather than ship a bundle without the
#                     SBOM the release advertises (P4-131 criterion 4).
#
# Env:
#   CAPI_TARGET_DIR   Passed through to install_capi.sh — the exact
#                     target-qualified Cargo release directory holding the
#                     cdylib and staticlib.
#
# Produces, for version X.Y.Z and target T:
#   ${OUTDIR}/libjpeg-turbo-rs-capi-X.Y.Z-T.tar.gz
#   ${OUTDIR}/libjpeg-turbo-rs-capi-X.Y.Z-T.tar.gz.sha256
# and with --sbom:
#   ${OUTDIR}/libjpeg-turbo-rs-capi-X.Y.Z-T.cdx.json
#   ${OUTDIR}/libjpeg-turbo-rs-capi-X.Y.Z-T.cdx.json.sha256
#
# The archive unpacks into a single directory named after the archive stem,
# holding the staged prefix (`lib/`, `include/`, and on Windows `bin/`) plus
# `BUNDLE.txt`.

set -euo pipefail

# Same helper as install_capi.sh: under Git for Windows' bash the script may
# receive `C:\...` paths and must hand POSIX ones to coreutils; `cygpath`
# exists exactly there and the function is the identity elsewhere.
posix_path() {
    if command -v cygpath >/dev/null 2>&1; then cygpath -u "$1"; else printf '%s\n' "$1"; fi
}

OUTDIR=""
PREFIX=""
TARGET=""
ROOT="$(cd "$(dirname "$(posix_path "$0")")/.." && pwd)"
DO_BUILD=0
DO_SBOM=0
CARGO_BIN="${CARGO:-cargo}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --outdir)  OUTDIR="$2"; shift 2 ;;
        --prefix)  PREFIX="$2"; shift 2 ;;
        --target)  TARGET="$2"; shift 2 ;;
        --root)    ROOT="$2"; shift 2 ;;
        --build)   DO_BUILD=1; shift ;;
        --sbom)    DO_SBOM=1; shift ;;
        -h|--help)
            sed -n '2,/^set -euo/p' "$0" | sed 's/^# *//'
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -n "$OUTDIR" ]] || { echo "--outdir is required" >&2; exit 1; }

# Probe the SBOM generator before the build, for the same reason the output
# directory is resolved before it: a missing tool should cost a shell error,
# not a full release build. Refusing rather than degrading is deliberate —
# the release attests the SBOM, and a bundle that ships without one would
# verify cleanly and simply have nothing to say about its dependencies.
if [[ "$DO_SBOM" -eq 1 ]] && ! "$CARGO_BIN" cyclonedx --version >/dev/null 2>&1; then
    echo "ERROR: --sbom needs cargo-cyclonedx (cargo install cargo-cyclonedx --locked)" >&2
    exit 1
fi

if [[ -z "$TARGET" ]]; then
    TARGET="$("$CARGO_BIN" -vV | tr -d '\r' | sed -n 's/^host: //p')"
    [[ -n "$TARGET" ]] || { echo "could not resolve Cargo host target" >&2; exit 1; }
fi

# `-m1` rather than `| head -1`: under `pipefail`, head closing the pipe can
# leave grep at 141 and abort the script.
VERSION="$(grep -m1 '^version' "$ROOT/crates/libjpeg-turbo-rs-capi/Cargo.toml" | sed 's/.*"\(.*\)".*/\1/')"
[[ -n "$VERSION" ]] || { echo "could not read the capi crate version" >&2; exit 1; }

BUNDLE="libjpeg-turbo-rs-capi-${VERSION}-${TARGET}"
ARCHIVE="${BUNDLE}.tar.gz"

# Resolve the output directory *before* the build, so a misspelled or
# unwritable `--outdir` costs a shell error rather than a full release build.
OUTDIR="$(posix_path "$OUTDIR")"
mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"
# And the root, so paths derived from it compare against absolute ones
# (the SBOM cleanup below excludes the output directory by identity).
ROOT="$(cd "$(posix_path "$ROOT")" && pwd)"

# The names the bundle carries. Recorded in BUNDLE.txt rather than assumed by
# the reader: `install_capi.sh` takes a `--soname` override, and a bundle whose
# chain is not the v8 default must say so where a packager will look.
case "$(uname -s)" in
    Linux*)
        PLATFORM=linux
        SONAME="libjpeg.so.8"
        SONAME_DEV="libjpeg.so"
        SONAME_TJ="libturbojpeg.so.0"
        SONAME_TJ_DEV="libturbojpeg.so"
        ;;
    Darwin*)
        PLATFORM=macos
        SONAME="libjpeg.8.dylib"
        SONAME_DEV="libjpeg.dylib"
        SONAME_TJ="libturbojpeg.0.dylib"
        SONAME_TJ_DEV="libturbojpeg.dylib"
        ;;
    MINGW*|MSYS*|CYGWIN*)
        # P4-131: upstream's MSVC layout. The DLL name is the identity a
        # consumer's import table records, so it stands in for the SONAME;
        # the "dev" entry is the import library. install_capi.sh refuses any
        # other Windows target, but say so here, before a build.
        PLATFORM=windows
        if [[ "$TARGET" != x86_64-*-windows-msvc ]]; then
            echo "unsupported Windows target for packaging: ${TARGET} (P4-131 ships the x86_64-pc-windows-msvc layout only)" >&2
            exit 1
        fi
        SONAME="jpeg8.dll"
        SONAME_DEV="jpeg.lib"
        SONAME_TJ="turbojpeg.dll"
        SONAME_TJ_DEV="turbojpeg.lib"
        ;;
    *) echo "unsupported platform for packaging: $(uname -s)" >&2; exit 1 ;;
esac

# Same defaults as install_capi.sh, which also records the prefix in the
# staged `.pc` and CMake files; keep the two in step.
if [[ -z "$PREFIX" ]]; then
    case "$PLATFORM" in
        windows) PREFIX="C:/libjpeg-turbo-rs64" ;;
        *)       PREFIX="/usr/local" ;;
    esac
fi
PREFIX="${PREFIX//\\//}"

# TAR_IDENTITY forces uid/gid 0 into the archive. Without it the tarball
# records the *build runner's* account, and GNU tar extracting as root honours
# it — so the documented `sudo cp -a` install would leave `/usr/local/lib`
# owned by whatever local user happens to hold uid 1001 on the target host,
# who could then replace a library that root-run programs load.
#
# Chosen by the tar on PATH rather than by platform: Linux and Git for
# Windows carry GNU tar, macOS carries bsdtar, which spells it differently
# and needs the *names* blanked too — it writes uname/gname and a GNU tar
# extraction resolves those first.
TAR_BANNER="$(tar --version 2>/dev/null | head -1 || true)"
case "$TAR_BANNER" in
    *"GNU tar"*) TAR_IDENTITY=(--owner=0 --group=0 --numeric-owner) ;;
    *)           TAR_IDENTITY=(--uid 0 --gid 0 --uname "" --gname "") ;;
esac

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT INT TERM

INSTALL_ARGS=(--destdir "$STAGE/destdir" --prefix "$PREFIX" --root "$ROOT")
[[ "$DO_BUILD" -eq 1 ]] && INSTALL_ARGS+=(--build)
CAPI_BUILD_TARGET="$TARGET" bash "$ROOT/scripts/install_capi.sh" "${INSTALL_ARGS[@]}"

# Where install_capi.sh put the prefix below its DESTDIR: a drive-lettered
# prefix is staged without its drive (see the script).
if [[ "$PREFIX" =~ ^[A-Za-z]: ]]; then
    STAGED="$STAGE/destdir${PREFIX:2}"
else
    STAGED="$STAGE/destdir${PREFIX}"
fi
[[ -d "$STAGED" ]] || { echo "install_capi.sh staged nothing at ${STAGED}" >&2; exit 1; }
mv "$STAGED" "$STAGE/$BUNDLE"

# Self-check before the archive exists, so a release can never attach a
# tarball that is missing the pieces a packager needs. The Rust suite asserts
# far more than this; these are the cases that would otherwise ship silently
# when a staging step degraded to a warning on the release runner.
#
# Both chains and both dev links, not just the libjpeg major: the dev link is
# what `JPEGConfig.cmake` names as `JPEG_LIBRARY`, so a bundle missing it
# resolves `find_package(JPEG)` to a path that does not exist, and the
# libturbojpeg chain is half of what BUNDLE.txt advertises. On Windows the
# loadable file is the DLL in `bin/` and the dev entry the import library.
if [[ "$PLATFORM" == "windows" ]]; then
    LOADABLE_DIR="bin"
else
    LOADABLE_DIR="lib"
fi
REQUIRED=(
    "${LOADABLE_DIR}/${SONAME}"
    "lib/${SONAME_DEV}"
    "${LOADABLE_DIR}/${SONAME_TJ}"
    "lib/${SONAME_TJ_DEV}"
    "lib/pkgconfig/libjpeg.pc"
    "lib/pkgconfig/libturbojpeg.pc"
    "lib/cmake/JPEG/JPEGConfig.cmake"
    "share/doc/libjpeg-turbo-rs-capi/LICENSE-MIT"
    "share/doc/libjpeg-turbo-rs-capi/LICENSE-APACHE"
    "include/jpeglib.h"
    "include/jerror.h"
    "include/jmorecfg.h"
    "include/jconfig.h"
    "include/turbojpeg.h"
)
MISSING=()
for required in "${REQUIRED[@]}"; do
    # -e follows symlinks, which is the point for the SONAME entry: a dangling
    # chain is as broken as an absent one.
    [[ -e "$STAGE/$BUNDLE/$required" ]] || MISSING+=("$required")
done
if [[ ${#MISSING[@]} -gt 0 ]]; then
    echo "ERROR: the staged prefix is incomplete; refusing to package it." >&2
    printf '       missing: %s\n' "${MISSING[@]}" >&2
    exit 1
fi

COMMIT="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"

# The library lines of the contents list, and the platform's checksum
# command. `soname:` is kept as the key on every platform — on Windows it is
# the DLL name, the closest thing a PE consumer has to one.
if [[ "$PLATFORM" == "windows" ]]; then
    LIBRARY_CONTENTS="  bin/                 ${SONAME}, ${SONAME_TJ} (one DLL under both names)
  lib/                 ${SONAME_DEV}, ${SONAME_TJ_DEV} — MSVC import libraries bound to those DLLs"
    VERIFY_COMMAND="  sha256sum -c ${ARCHIVE}.sha256      # from Git for Windows' bash; or
  certutil -hashfile ${ARCHIVE} SHA256    # and compare by eye"
else
    LIBRARY_CONTENTS="  lib/                 both SONAME chains (libjpeg and libturbojpeg)"
    VERIFY_COMMAND="  sha256sum -c ${ARCHIVE}.sha256      # macOS: shasum -a 256 -c"
fi

cat >"$STAGE/$BUNDLE/BUNDLE.txt" <<EOF
libjpeg-turbo-rs C ABI shim — prebuilt native library

name: libjpeg-turbo-rs-capi
version: ${VERSION}
target: ${TARGET}
prefix: ${PREFIX}
soname: ${SONAME}
commit: ${COMMIT}

Staged by scripts/install_capi.sh and packaged by
scripts/package_capi_release.sh. Contents:

${LIBRARY_CONTENTS}
  lib/pkgconfig/       libjpeg.pc, libturbojpeg.pc
  lib/cmake/JPEG/      JPEGConfig.cmake for find_package(JPEG)
  share/doc/           LICENSE-MIT, LICENSE-APACHE
  include/             jpeglib.h, jerror.h, jmorecfg.h, jconfig.h, turbojpeg.h

The prefix above is baked into the .pc and CMake files, which record absolute
paths. Unpacking into ${PREFIX} needs no further work. To install elsewhere,
either rewrite those paths or let pkg-config do it:

  PKG_CONFIG_PATH=<where>/lib/pkgconfig pkg-config --define-prefix --cflags libjpeg

Verify the download before installing. Both files are attached to a GitHub
release: SHA256SUMS covers every bundle in it, the .sha256 covers this one.

${VERIFY_COMMAND}

A checksum proves the bytes arrived intact, not where they came from. An
archive attached to a GitHub release is also attested — Sigstore build
provenance, and a CycloneDX SBOM attached beside it as ${BUNDLE}.cdx.json —
signed by the release workflow's own identity. Verify with the GitHub CLI,
pinning the workflow and the tag you downloaded from (a rehearsal build of
the same workflow from a branch is attested too, and names that branch):

  gh attestation verify ${ARCHIVE} --repo developer0hye/libjpeg-turbo-rs \\
      --signer-workflow developer0hye/libjpeg-turbo-rs/.github/workflows/release.yml \\
      --source-ref refs/tags/<release tag>
  gh attestation verify ${ARCHIVE} --repo developer0hye/libjpeg-turbo-rs \\
      --predicate-type https://cyclonedx.org/bom

Extract with --no-same-owner if you unpack as root; the archive is written
0:0, but a tarball from elsewhere may not be.

This library is not yet a general drop-in replacement for C libjpeg-turbo.
Read docs/RELEASE_ARTIFACTS.md and docs/ABI_COMPATIBILITY.md — in particular
the replacement tiers — before replacing a system libjpeg.
EOF

# Symlinks stay symlinks in both GNU tar and bsdtar, so the SONAME chain
# survives the round trip. Do not add --dereference.
tar "${TAR_IDENTITY[@]}" -czf "$OUTDIR/$ARCHIVE" -C "$STAGE" "$BUNDLE"

# `<hash>  <bare name>` so `sha256sum -c` works in the directory the file
# was downloaded into, not only where it was built.
write_checksum() {
    local file="$1"
    (
        cd "$OUTDIR"
        if command -v sha256sum >/dev/null 2>&1; then
            sha256sum "$file" >"${file}.sha256"
        elif command -v shasum >/dev/null 2>&1; then
            shasum -a 256 "$file" >"${file}.sha256"
        else
            echo "ERROR: neither sha256sum nor shasum is available; every attached artifact must be checksummed (P4-131 criterion 2)" >&2
            exit 1
        fi
    )
}
write_checksum "$ARCHIVE"

PRODUCED=("${OUTDIR}/${ARCHIVE}" "${OUTDIR}/${ARCHIVE}.sha256")

if [[ "$DO_SBOM" -eq 1 ]]; then
    SBOM="${BUNDLE}.cdx.json"
    # `--target` resolves the dependency graph for the bundle's platform, not
    # the packaging host's — the cross-built legs would otherwise describe
    # the wrong library. No toolchain for the target is needed for that.
    #
    # cargo-cyclonedx has no package selector: given a member's manifest it
    # still writes `<name>.json` beside the Cargo.toml of *every* workspace
    # member. The capi crate's is the one the bundle describes; the others are
    # removed so they cannot be left in the source tree for the next
    # `cargo publish --allow-dirty` to ship.
    (cd "$ROOT" && "$CARGO_BIN" cyclonedx \
        --manifest-path crates/libjpeg-turbo-rs-capi/Cargo.toml \
        --format json --spec-version 1.5 \
        --target "$TARGET" \
        --override-filename "${BUNDLE}.cdx" -q)
    mv "$ROOT/crates/libjpeg-turbo-rs-capi/${SBOM}" "$OUTDIR/$SBOM"
    # `-samefile` rather than a path comparison: the one just moved into
    # `--outdir` is excluded by identity, so a relative `--root` or an
    # `--outdir` inside the tree cannot delete it or spare the strays.
    find "$ROOT" -maxdepth 3 -name "$SBOM" ! -samefile "$OUTDIR/$SBOM" -delete
    write_checksum "$SBOM"
    PRODUCED+=("${OUTDIR}/${SBOM}" "${OUTDIR}/${SBOM}.sha256")
fi

echo "Packaged ${BUNDLE}:"
printf '  %s\n' "${PRODUCED[@]}"

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
# `install_capi.sh`, which is the same path P4-124 requires the downstream
# harnesses to test — one staging path, so the artifact a packager downloads
# is the artifact the harnesses measure. The only file this script adds is
# `BUNDLE.txt`, which describes the bundle. That invariant is enforced by
# `crates/libjpeg-turbo-rs-capi/tests/release_bundle.rs`, which compares the
# unpacked archive against a direct `install_capi.sh` run entry by entry.
#
# Usage:
#   scripts/package_capi_release.sh --outdir dist
#   scripts/package_capi_release.sh --outdir dist --prefix /usr --build
#
# Flags:
#   --outdir DIR      Where the archive and its checksum are written
#                     (required; created if absent).
#   --prefix DIR      Prefix baked into the staged `.pc` and CMake files
#                     (default "/usr/local"). Absolute by nature — the bundle
#                     records it in BUNDLE.txt so a packager who unpacks
#                     somewhere else knows what to relocate.
#   --target TRIPLE   Cargo target triple, used both for the nested build and
#                     as the bundle's platform label (default: Cargo's host).
#   --root DIR        Repository root (default: this script's parent).
#   --build           Force a cdylib build. Passed through; without it, a
#                     missing cdylib is still built by the install script.
#
# Env:
#   CAPI_TARGET_DIR   Passed through to install_capi.sh — the exact
#                     target-qualified Cargo release directory holding the
#                     cdylib and staticlib.
#
# Produces, for version X.Y.Z and target T:
#   ${OUTDIR}/libjpeg-turbo-rs-capi-X.Y.Z-T.tar.gz
#   ${OUTDIR}/libjpeg-turbo-rs-capi-X.Y.Z-T.tar.gz.sha256
#
# The archive unpacks into a single directory named after the archive stem,
# holding the staged prefix (`lib/`, `include/`) plus `BUNDLE.txt`.

set -euo pipefail

OUTDIR=""
PREFIX="/usr/local"
TARGET=""
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DO_BUILD=0
CARGO_BIN="${CARGO:-cargo}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --outdir)  OUTDIR="$2"; shift 2 ;;
        --prefix)  PREFIX="$2"; shift 2 ;;
        --target)  TARGET="$2"; shift 2 ;;
        --root)    ROOT="$2"; shift 2 ;;
        --build)   DO_BUILD=1; shift ;;
        -h|--help)
            sed -n '2,/^set -euo/p' "$0" | sed 's/^# *//'
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

[[ -n "$OUTDIR" ]] || { echo "--outdir is required" >&2; exit 1; }

if [[ -z "$TARGET" ]]; then
    TARGET="$("$CARGO_BIN" -vV | sed -n 's/^host: //p')"
    [[ -n "$TARGET" ]] || { echo "could not resolve Cargo host target" >&2; exit 1; }
fi

VERSION="$(grep '^version' "$ROOT/crates/libjpeg-turbo-rs-capi/Cargo.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/')"
[[ -n "$VERSION" ]] || { echo "could not read the capi crate version" >&2; exit 1; }

BUNDLE="libjpeg-turbo-rs-capi-${VERSION}-${TARGET}"
ARCHIVE="${BUNDLE}.tar.gz"

# The SONAME the bundle carries. Recorded in BUNDLE.txt rather than assumed by
# the reader: `install_capi.sh` takes a `--soname` override, and a bundle whose
# chain is not the v8 default must say so where a packager will look.
case "$(uname -s)" in
    Linux*)  SONAME="libjpeg.so.8" ;;
    Darwin*) SONAME="libjpeg.8.dylib" ;;
    # Windows (DLL + import library) is still open under P4-131. Fail here
    # rather than emit a bundle whose shape nothing has verified.
    *) echo "unsupported platform for packaging: $(uname -s) (P4-131 tracks Windows)" >&2; exit 1 ;;
esac

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

INSTALL_ARGS=(--destdir "$STAGE/destdir" --prefix "$PREFIX" --root "$ROOT")
[[ "$DO_BUILD" -eq 1 ]] && INSTALL_ARGS+=(--build)
CAPI_BUILD_TARGET="$TARGET" bash "$ROOT/scripts/install_capi.sh" "${INSTALL_ARGS[@]}"

STAGED="$STAGE/destdir${PREFIX}"
[[ -d "$STAGED" ]] || { echo "install_capi.sh staged nothing at ${STAGED}" >&2; exit 1; }
mv "$STAGED" "$STAGE/$BUNDLE"

# Self-check before the archive exists, so a release can never attach a
# tarball that is missing the pieces a packager needs. The Rust suite asserts
# far more than this; these are the cases that would otherwise ship silently
# when a staging step degraded to a warning on the release runner.
REQUIRED=(
    "lib/${SONAME}"
    "lib/pkgconfig/libjpeg.pc"
    "lib/pkgconfig/libturbojpeg.pc"
    "lib/cmake/JPEG/JPEGConfig.cmake"
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

  lib/                 both SONAME chains (libjpeg and libturbojpeg)
  lib/pkgconfig/       libjpeg.pc, libturbojpeg.pc
  lib/cmake/JPEG/      JPEGConfig.cmake for find_package(JPEG)
  include/             jpeglib.h, jerror.h, jmorecfg.h, jconfig.h, turbojpeg.h

The prefix above is baked into the .pc and CMake files, which record absolute
paths. Unpacking into ${PREFIX} needs no further work. To install elsewhere,
either rewrite those paths or let pkg-config do it:

  PKG_CONFIG_PATH=<where>/lib/pkgconfig pkg-config --define-prefix --cflags libjpeg

Verify the download before installing:

  sha256sum -c ${ARCHIVE}.sha256      # macOS: shasum -a 256 -c

This library is not yet a general drop-in replacement for C libjpeg-turbo.
Read docs/RELEASE_ARTIFACTS.md and docs/ABI_COMPATIBILITY.md — in particular
the replacement tiers — before replacing a system libjpeg.
EOF

mkdir -p "$OUTDIR"
OUTDIR="$(cd "$OUTDIR" && pwd)"

# `tar` stores symlinks as symlinks by default in both GNU tar and bsdtar, so
# the SONAME chain survives the round trip. Do not add --dereference.
tar -czf "$OUTDIR/$ARCHIVE" -C "$STAGE" "$BUNDLE"

# `<hash>  <bare name>` so `sha256sum -c` works in the directory the archive
# was downloaded into, not only where it was built.
(
    cd "$OUTDIR"
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$ARCHIVE" >"${ARCHIVE}.sha256"
    elif command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$ARCHIVE" >"${ARCHIVE}.sha256"
    else
        echo "ERROR: neither sha256sum nor shasum is available; every attached artifact must be checksummed (P4-131 criterion 2)" >&2
        exit 1
    fi
)

cat <<EOF
Packaged ${BUNDLE}:
  ${OUTDIR}/${ARCHIVE}
  ${OUTDIR}/${ARCHIVE}.sha256
EOF

#!/usr/bin/env bash
# fetch_conformance.sh — best-effort fetcher for ITU-T T.83 JPEG conformance
# test vectors used by B3-4's opt-in conformance suite.
#
# Background:
# ITU-T Recommendation T.83 ("Information technology — Digital compression
# and coding of continuous-tone still images: Compliance testing") ships a
# CD-ROM of reference bitstreams (A1.JPG, A2.JPG, F-*.jpg, etc.) for
# DCT-based, sequential, and progressive modes. The Rec text is freely
# available from ITU, but the binary reference vectors are NOT redistributable
# without explicit license. ISO/IEC 10918-2 is the twin document.
#
# This script does not ship the bitstreams. Instead it:
#   1. Checks whether the developer has already placed them under
#      `tests/conformance/t83/` (the path our Rust test looks at).
#   2. If not, prints step-by-step manual-acquisition instructions.
#   3. Exits 0 when the target directory is ready, or 2 when the developer
#      must act. Never overwrites existing files.
#
# The script intentionally does not attempt an unattended download: all
# canonical mirrors we know of are licensed and require per-user acceptance.
# Silent pulling of licensed data would violate ITU's redistribution terms
# and CLAUDE.md's "never commit binary conformance vectors" rule.
#
# Usage:
#   scripts/fetch_conformance.sh           # check / print instructions
#   scripts/fetch_conformance.sh --check   # same as default, exit 2 if missing
#   scripts/fetch_conformance.sh --help    # show usage

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/tests/conformance/t83"

print_usage() {
    cat <<'USAGE'
Usage: scripts/fetch_conformance.sh [--check|--help]

  --check   Exit 2 if T.83 vectors are missing (for CI gating). Default.
  --help    Show this help.

The script never downloads licensed bitstreams. It either reports that the
vectors are already present, or prints instructions for obtaining them.
USAGE
}

print_instructions() {
    cat <<INSTRUCTIONS
[fetch_conformance] ITU-T T.83 reference vectors not found at:
    ${TARGET_DIR}

These reference vectors (A1.JPG, A2.JPG, F-*.jpg, etc.) are licensed by
the ITU and the ISO/IEC copyright office. We CANNOT redistribute them in
this repository. To run the opt-in T.83 conformance suite you must obtain
them yourself:

  1. Buy or request the CD-ROM accompanying:
       - ITU-T Recommendation T.83 (11/1994) "JPEG compliance testing", or
       - ISO/IEC 10918-2:1995 "Compliance testing for JPEG".
     The Rec itself is downloadable free from https://www.itu.int/rec/T-REC-T.83
     but the CD-ROM archive is sold separately.

  2. Some university mirrors host the vectors under research-use licenses.
     Examples (verify the license terms yourself before downloading):
       - ftp://ftp.nal.usda.gov/pub/jpeg-compliance/
       - https://www.ijg.org/files/ (look for compliance tarball)
     Do NOT redistribute what you pull from these mirrors into this repo.

  3. Extract the archive so that the JPEG files sit directly under:
         ${TARGET_DIR}
     For example:
         ${TARGET_DIR}/A1.JPG
         ${TARGET_DIR}/A2.JPG
         ${TARGET_DIR}/F-1.JPG
         ...

  4. Re-run: cargo test --test conformance_t83 -- --include-ignored
     Tests for T.83 vectors will activate automatically once the files exist.
     If still absent, tests skip gracefully with the same message.

The immediate-proxy suite (tests/conformance_t83.rs) runs unconditionally
against libjpeg-turbo's bundled testimages/*.jpg, which IS redistributable
under libjpeg-turbo's license.
INSTRUCTIONS
}

mode="check"
for arg in "$@"; do
    case "${arg}" in
        --help|-h) print_usage; exit 0 ;;
        --check)   mode="check" ;;
        *) echo "[fetch_conformance] unknown argument: ${arg}" >&2
           print_usage
           exit 64
           ;;
    esac
done

# Verify target path is inside the repo we expect. Defensive check in case
# the script is executed from an unexpected location.
case "${TARGET_DIR}" in
    "${REPO_ROOT}"/*) ;;
    *)
        echo "[fetch_conformance] refusing: target path outside repo root" >&2
        exit 70
        ;;
esac

if [[ -d "${TARGET_DIR}" ]]; then
    jpeg_count=$(find "${TARGET_DIR}" -maxdepth 1 -type f \
        \( -iname '*.jpg' -o -iname '*.jpeg' \) 2>/dev/null | wc -l | tr -d ' ')
    if [[ "${jpeg_count}" -gt 0 ]]; then
        echo "[fetch_conformance] Found ${jpeg_count} JPEG file(s) under ${TARGET_DIR}"
        echo "[fetch_conformance] T.83 conformance suite can run."
        exit 0
    fi
fi

print_instructions

case "${mode}" in
    check) exit 2 ;;
    *)     exit 0 ;;
esac

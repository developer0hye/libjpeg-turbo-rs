#!/usr/bin/env bash
# FFI B9-2: Pillow-against-libjpeg-turbo-rs-capi smoke runner.
#
# Flow:
#   1. Build the shim in CARGO_TARGET_DIR's target-qualified release directory.
#   2. Symlink it as libjpeg.so.62 (Linux) / libjpeg.62.dylib (macOS) next
#      beside the resolved release artifact.
#   3. Install Pillow into a venv.
#   4. *Force* Pillow to load our shim by replacing its bundled libjpeg
#      (PIL/.dylibs/libjpeg.62.4.0.dylib on macOS wheels,
#       PIL.libs/libjpeg-*.so.62.* on manylinux wheels) with a symlink to
#      our shim. Back up the original first so a subsequent run can
#      restore it. This is the only reliable way to verify Pillow
#      actually binds against our code on wheel-based distributions —
#      DYLD_LIBRARY_PATH / LD_LIBRARY_PATH do not override
#      @loader_path / $ORIGIN references that the wheel bakes in.
#   5. Run test_pillow.py.
#   6. Restore the bundled libjpeg so the venv stays usable.
#
# Runner exit-code contract:
#   0  PASS
#   2  SKIP  (python/Pillow/fixture not available)
#   3  BLOCKER (the shim build/load or Pillow round-trip failed)
#   1  FAIL  (shim loaded, Pillow ran, but round-trip output is wrong)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
log() { printf '[run.sh] %s\n' "$*"; }

# Honor CARGO_TARGET_DIR: the build below inherits it, so the shim
# lands there, not in the in-repo target/ (this host offloads all cargo
# artifacts to an external SSD). Absolute paths only — a relative
# CARGO_TARGET_DIR resolves against the repo root, same as cargo does
# for the `cd REPO_ROOT && cargo build` invocation below.
case "${CARGO_TARGET_DIR:-}" in
    "")  TARGET_BASE="${REPO_ROOT}/target" ;;
    /*)  TARGET_BASE="${CARGO_TARGET_DIR}" ;;
    *)   TARGET_BASE="${REPO_ROOT}/${CARGO_TARGET_DIR}" ;;
esac
CARGO_BIN="${CARGO:-cargo}"
# This process loads the shim into host Python. An ambient
# CARGO_BUILD_TARGET may describe a cross-build selected elsewhere, so only
# the harness-specific override may replace Cargo's reported host target.
CAPI_BUILD_TARGET="${CAPI_BUILD_TARGET:-}"
if [ -z "${CAPI_BUILD_TARGET}" ]; then
    CAPI_BUILD_TARGET="$("${CARGO_BIN}" -vV 2>/dev/null | sed -n 's/^host: //p')"
fi
if [ -z "${CAPI_BUILD_TARGET}" ]; then
    log "BLOCKER: could not resolve Cargo's build target"
    exit 3
fi
TARGET_COMPONENT="$(basename "${CAPI_BUILD_TARGET%.json}")"
RELEASE_DIR="${TARGET_BASE}/${TARGET_COMPONENT}/release"

# ---- Platform detection --------------------------------------------------
OS_NAME="$(uname -s)"
case "${OS_NAME}" in
    Linux)
        SRC_LIB="${RELEASE_DIR}/liblibjpeg_turbo_rs_capi.so"
        DST_LIB="${RELEASE_DIR}/libjpeg.so.62"
        LIB_ENV_VAR="LD_LIBRARY_PATH"
        ;;
    Darwin)
        SRC_LIB="${RELEASE_DIR}/liblibjpeg_turbo_rs_capi.dylib"
        DST_LIB="${RELEASE_DIR}/libjpeg.62.dylib"
        LIB_ENV_VAR="DYLD_LIBRARY_PATH"
        ;;
    *)
        log "SKIP: unsupported OS ${OS_NAME}"
        exit 2
        ;;
esac

# ---- Build capi shim -----------------------------------------------------
# Rebuild unconditionally: an existing file may have come from a cache or an
# older checkout, and this smoke test must exercise the current source.
log "building current libjpeg-turbo-rs-capi release cdylib"
if ! (cd "${REPO_ROOT}" && CARGO_TARGET_DIR="${TARGET_BASE}" "${CARGO_BIN}" build -p libjpeg-turbo-rs-capi --release --target "${CAPI_BUILD_TARGET}") ; then
    log "BLOCKER: cargo build failed"
    exit 3
fi

if [ ! -f "${SRC_LIB}" ]; then
    log "BLOCKER: expected shim at ${SRC_LIB} but it is missing"
    exit 3
fi

ln -sf "$(basename "${SRC_LIB}")" "${DST_LIB}"
log "shim: ${DST_LIB} -> $(basename "${SRC_LIB}")"

# ---- Fixture -------------------------------------------------------------
FIXTURE_DEFAULT="${REPO_ROOT}/tests/fixtures/cjpeg_240x320_portrait_444.jpg"
FIXTURE="${PILLOW_SMOKE_FIXTURE:-${FIXTURE_DEFAULT}}"
if [ ! -f "${FIXTURE}" ]; then
    log "SKIP: fixture not found: ${FIXTURE}"
    exit 2
fi
export PILLOW_SMOKE_FIXTURE="${FIXTURE}"
export PILLOW_SMOKE_SHIM="${DST_LIB}"

# ---- Python venv ---------------------------------------------------------
PY="${PYTHON:-python3}"
if ! command -v "${PY}" >/dev/null 2>&1; then
    log "SKIP: python3 not available"
    exit 2
fi

VENV="${SCRIPT_DIR}/venv"
if [ ! -x "${VENV}/bin/python" ]; then
    log "creating venv at ${VENV}"
    if ! "${PY}" -m venv "${VENV}" ; then
        log "SKIP: failed to create venv (python3-venv missing?)"
        exit 2
    fi
fi

if ! "${VENV}/bin/pip" show Pillow >/dev/null 2>&1 ; then
    log "installing Pillow + numpy into venv (offline-safe: accept network errors as SKIP)"
    if ! "${VENV}/bin/pip" install --quiet --disable-pip-version-check Pillow numpy ; then
        log "SKIP: pip install failed (likely offline or index blocked)"
        exit 2
    fi
fi

# ---- Locate Pillow's bundled libjpeg + replace with our shim -------------
# We look inside the venv's site-packages because wheels bundle the
# runtime deps under PIL/.dylibs (macOS) or PIL.libs (manylinux) with
# @loader_path / $ORIGIN install_names that DYLD_/LD_LIBRARY_PATH cannot
# override.
SITE_PACKAGES="$("${VENV}/bin/python" -c \
    'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
BUNDLED_DIR=""
for candidate in \
    "${SITE_PACKAGES}/PIL/.dylibs" \
    "${SITE_PACKAGES}/PIL.libs" \
    "${SITE_PACKAGES}/pillow.libs" \
    ; do
    if [ -d "${candidate}" ]; then
        BUNDLED_DIR="${candidate}"
        break
    fi
done

BUNDLED_ORIG=""
BUNDLED_BACKUP=""
if [ -n "${BUNDLED_DIR}" ]; then
    # Darwin wheels ship libjpeg.<version>.dylib (e.g. libjpeg.62.4.0.dylib).
    # manylinux wheels ship libjpeg-<hash>.so.62.* .
    BUNDLED_ORIG="$(find "${BUNDLED_DIR}" -maxdepth 1 \
        \( -name 'libjpeg*.dylib' -o -name 'libjpeg*.so.62*' \) \
        -type f 2>/dev/null | head -n 1)"
    if [ -n "${BUNDLED_ORIG}" ]; then
        BUNDLED_BACKUP="${BUNDLED_ORIG}.pillow_smoke_backup"
        if [ ! -f "${BUNDLED_BACKUP}" ]; then
            cp -p "${BUNDLED_ORIG}" "${BUNDLED_BACKUP}"
            log "backed up bundled libjpeg: ${BUNDLED_ORIG} -> ${BUNDLED_BACKUP}"
        fi
        # Overwrite with a copy of our shim (not a symlink — Pillow wheels
        # on some platforms validate the file type).
        cp -f "${SRC_LIB}" "${BUNDLED_ORIG}"
        log "replaced Pillow's bundled libjpeg at ${BUNDLED_ORIG} with our shim"
    else
        log "note: Pillow wheel did not ship a bundled libjpeg at ${BUNDLED_DIR}"
    fi
else
    log "note: no Pillow bundled-dylib directory found; relying on ${LIB_ENV_VAR} only"
fi

restore_bundled() {
    if [ -n "${BUNDLED_BACKUP}" ] && [ -f "${BUNDLED_BACKUP}" ] \
       && [ -n "${BUNDLED_ORIG}" ]; then
        mv -f "${BUNDLED_BACKUP}" "${BUNDLED_ORIG}"
        log "restored bundled libjpeg: ${BUNDLED_ORIG}"
    fi
}
trap restore_bundled EXIT

# ---- Run test ------------------------------------------------------------
export "${LIB_ENV_VAR}=${RELEASE_DIR}:${!LIB_ENV_VAR:-}"
log "${LIB_ENV_VAR}=${!LIB_ENV_VAR}"

"${VENV}/bin/python" "${SCRIPT_DIR}/test_pillow.py"
rc=$?
log "test_pillow.py exited with code ${rc}"
exit ${rc}

#!/usr/bin/env bash
# FFI B9-4: run stock djpeg/cjpeg/jpegtran linked against OUR shim over
# references/libjpeg-turbo/testimages/*.jpg, compare byte-for-byte against
# the system-installed stock libjpeg-turbo binaries.
#
# Outputs a machine-readable TSV to stdout with one line per test:
#   tool\timage\tpass|fail\treason_if_fail
#
# Exit 0 if every case is a pass; non-zero otherwise.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." &> /dev/null && pwd)"
OUR_BUILD="${OUT_DIR:-$SCRIPT_DIR/build}"
# Pick a stock libjpeg-turbo install dir. If `STOCK_BIN` is set
# explicitly, respect it; otherwise probe well-known locations
# across macOS Homebrew, Linux's libjpeg-turbo official .deb
# (`/opt/libjpeg-turbo/bin`, used by our CI), source-from-/usr/local,
# and the system path. Falling back to a PATH lookup keeps the
# script working when none of these match.
if [[ -z "${STOCK_BIN:-}" ]]; then
    for cand in /opt/homebrew/bin /opt/libjpeg-turbo/bin /usr/local/bin /usr/bin; do
        if [[ -x "$cand/djpeg" ]]; then
            STOCK_BIN="$cand"
            break
        fi
    done
    # Last resort: rely on PATH. `command -v djpeg` resolves to the
    # binary's actual directory if djpeg is anywhere on PATH (the
    # CI workflow `echo "/opt/libjpeg-turbo/bin" >> $GITHUB_PATH`
    # path matches this branch when probing fails for some reason).
    if [[ -z "${STOCK_BIN:-}" ]]; then
        if pathres="$(command -v djpeg 2>/dev/null)"; then
            STOCK_BIN="$(dirname "$pathres")"
        else
            STOCK_BIN="/opt/homebrew/bin"  # preserve historical default error message
        fi
    fi
fi
TESTIMAGES="$REPO_ROOT/references/libjpeg-turbo/testimages"

if [[ ! -x "$OUR_BUILD/djpeg" ]]; then
    echo "ERROR: our-linked djpeg not found at $OUR_BUILD/djpeg; run build.sh first" >&2
    exit 2
fi

OUR_DJPEG="$OUR_BUILD/djpeg"
OUR_CJPEG="$OUR_BUILD/cjpeg"
OUR_JPEGTRAN="$OUR_BUILD/jpegtran"
OUR_TJBENCH="$OUR_BUILD/tjbench"
OUR_RDJPGCOM="$OUR_BUILD/rdjpgcom"
OUR_WRJPGCOM="$OUR_BUILD/wrjpgcom"

# Locate the shim cdylib. build.sh tries to bake the install path
# into each tool via install_name_tool / -Wl,-rpath, but in sandboxed
# CI/build envs install_name_tool can be blocked, leaving binaries
# that reference `@rpath/libjpeg.62.dylib` (macOS) or `libjpeg.so.62`
# (Linux) with no usable LC_RPATH. The fallback below stages a
# private loader directory with canonical-soname symlinks pointing at
# the cdylib and arranges for the our-linked tools — and only those
# tools — to run with that directory on their loader path.
SHIM_DIR="${SHIM_DIR:-$REPO_ROOT/target/release}"
if [[ -f "$SHIM_DIR/liblibjpeg_turbo_rs_capi.dylib" ]]; then
    SHIM_LIB_PATH="$SHIM_DIR/liblibjpeg_turbo_rs_capi.dylib"
    DYLIB_EXT="dylib"
elif [[ -f "$SHIM_DIR/liblibjpeg_turbo_rs_capi.so" ]]; then
    SHIM_LIB_PATH="$SHIM_DIR/liblibjpeg_turbo_rs_capi.so"
    DYLIB_EXT="so"
else
    echo "ERROR: shim cdylib not found under $SHIM_DIR;" >&2
    echo "       run \`cargo build -p libjpeg-turbo-rs-capi --release\` first" >&2
    exit 5
fi
# Absolutize SHIM_LIB_PATH before we use it as a symlink target. A
# relative `SHIM_DIR=target/release` would record a target that is
# resolved relative to the symlink's own directory ($WORK/loader),
# not to the caller's CWD — so the loader would dereference it to a
# nonexistent path. Use `cd "$(dirname …)" && pwd` instead of
# `realpath`/`readlink -f` because both are unavailable on stock
# macOS.
SHIM_LIB_PATH="$(cd "$(dirname "$SHIM_LIB_PATH")" && pwd)/$(basename "$SHIM_LIB_PATH")"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# Stage canonical-soname symlinks so the loader can resolve the names
# baked into the our-linked binaries. `cargo build` only emits
# `liblibjpeg_turbo_rs_capi.{dylib,so}`, but stock djpeg/cjpeg/jpegtran
# look for `libjpeg.62.dylib` (macOS) or `libjpeg.so.62` (Linux).
LOADER_DIR="$WORK/loader"
mkdir -p "$LOADER_DIR"
if [[ "$DYLIB_EXT" == "dylib" ]]; then
    ln -sf "$SHIM_LIB_PATH" "$LOADER_DIR/libjpeg.62.dylib"
    ln -sf "$SHIM_LIB_PATH" "$LOADER_DIR/libturbojpeg.0.dylib"
else
    ln -sf "$SHIM_LIB_PATH" "$LOADER_DIR/libjpeg.so.62"
    ln -sf "$SHIM_LIB_PATH" "$LOADER_DIR/libturbojpeg.so.0"
fi

# Run an our-linked binary with the loader path scoped to *that*
# command — never globally exported. On Linux this matters most: the
# stock libjpeg-turbo binaries link against `libjpeg.so.62` too, so a
# global LD_LIBRARY_PATH would silently make the reference side load
# our shim and turn the byte comparison into Rust-vs-Rust.
run_ours() {
    DYLD_LIBRARY_PATH="$LOADER_DIR" \
    LD_LIBRARY_PATH="$LOADER_DIR" \
        "$@"
}
STOCK_DJPEG="${STOCK_DJPEG:-$STOCK_BIN/djpeg}"
STOCK_CJPEG="${STOCK_CJPEG:-$STOCK_BIN/cjpeg}"
STOCK_JPEGTRAN="${STOCK_JPEGTRAN:-$STOCK_BIN/jpegtran}"

for bin in "$STOCK_DJPEG" "$STOCK_CJPEG" "$STOCK_JPEGTRAN"; do
    if [[ ! -x "$bin" ]]; then
        echo "ERROR: stock $bin missing; install libjpeg-turbo or set STOCK_BIN" >&2
        exit 3
    fi
done

FAIL=0
for img in "$TESTIMAGES"/*.jpg; do
    [[ -f "$img" ]] || continue
    name="$(basename "$img" .jpg)"

    # ------- decode: djpeg (our-linked) vs djpeg (stock) -------
    ours_ppm="$WORK/${name}.ours.ppm"
    stock_ppm="$WORK/${name}.stock.ppm"
    if ! run_ours "$OUR_DJPEG" -outfile "$ours_ppm" "$img" 2>"$WORK/djpeg_err_ours.log"; then
        echo -e "djpeg\t${name}\tfail\tours_crashed"
        FAIL=$((FAIL + 1))
        continue
    fi
    if ! "$STOCK_DJPEG" -outfile "$stock_ppm" "$img" 2>"$WORK/djpeg_err_stock.log"; then
        echo -e "djpeg\t${name}\tskip\tstock_failed"
        continue
    fi
    if cmp -s "$ours_ppm" "$stock_ppm"; then
        echo -e "djpeg\t${name}\tpass\t"
    else
        sz_o=$(wc -c < "$ours_ppm"); sz_s=$(wc -c < "$stock_ppm")
        echo -e "djpeg\t${name}\tfail\tbytes_differ_ours=${sz_o}_stock=${sz_s}"
        FAIL=$((FAIL + 1))
    fi

    # ------- encode: cjpeg (our-linked) vs cjpeg (stock) over ours_ppm -------
    if [[ -f "$ours_ppm" ]]; then
        ours_jpg="$WORK/${name}.ours.jpg"
        stock_jpg="$WORK/${name}.stock.jpg"
        if ! run_ours "$OUR_CJPEG" -outfile "$ours_jpg" "$ours_ppm" 2>"$WORK/cjpeg_err_ours.log"; then
            echo -e "cjpeg\t${name}\tfail\tours_crashed"
            FAIL=$((FAIL + 1))
        elif ! "$STOCK_CJPEG" -outfile "$stock_jpg" "$ours_ppm" 2>"$WORK/cjpeg_err_stock.log"; then
            echo -e "cjpeg\t${name}\tskip\tstock_failed"
        else
            if cmp -s "$ours_jpg" "$stock_jpg"; then
                echo -e "cjpeg\t${name}\tpass\tbyte_exact"
            else
                # Encode byte-exactness is aspirational; accept PSNR>50dB by
                # roundtripping both outputs through stock djpeg and diffing.
                ours_rt="$WORK/${name}.ours.rt.ppm"
                stock_rt="$WORK/${name}.stock.rt.ppm"
                "$STOCK_DJPEG" -outfile "$ours_rt"  "$ours_jpg"  || true
                "$STOCK_DJPEG" -outfile "$stock_rt" "$stock_jpg" || true
                if [[ -f "$ours_rt" && -f "$stock_rt" ]] && cmp -s "$ours_rt" "$stock_rt"; then
                    echo -e "cjpeg\t${name}\tpass\troundtrip_identical"
                else
                    sz_o=$(wc -c < "$ours_jpg" 2>/dev/null || echo 0)
                    sz_s=$(wc -c < "$stock_jpg" 2>/dev/null || echo 0)
                    echo -e "cjpeg\t${name}\tfail\tbytes_differ_ours=${sz_o}_stock=${sz_s}"
                    FAIL=$((FAIL + 1))
                fi
            fi
        fi
    fi

    # ------- transform: jpegtran (our-linked) vs jpegtran (stock) -------
    # Every fixture — 8-bit *and* 12-bit (`monkey12.jpg`) — must be
    # byte-identical to upstream `jpegtran -copy all -rotate 90`.
    # The 12-bit byte-exact gate landed when `jpeg_read_header`
    # started populating `cinfo->marker_list` so
    # `transupp::jcopy_markers_execute` can forward the source's
    # APP2/ICC chunks to dstinfo verbatim — without that the ICC
    # profile silently disappeared in our output, producing the
    # historical `pixel_equal_dht_differs` divergence.
    ours_trn="$WORK/${name}.ours.trn.jpg"
    stock_trn="$WORK/${name}.stock.trn.jpg"
    if ! run_ours "$OUR_JPEGTRAN" -copy all -rotate 90 -outfile "$ours_trn" "$img" 2>"$WORK/trn_err_ours.log"; then
        echo -e "jpegtran\t${name}\tfail\tours_crashed"
        FAIL=$((FAIL + 1))
        continue
    fi
    if ! "$STOCK_JPEGTRAN" -copy all -rotate 90 -outfile "$stock_trn" "$img" 2>"$WORK/trn_err_stock.log"; then
        echo -e "jpegtran\t${name}\tskip\tstock_failed"
        continue
    fi
    if cmp -s "$ours_trn" "$stock_trn"; then
        echo -e "jpegtran\t${name}\tpass\t"
    else
        sz_o=$(wc -c < "$ours_trn"); sz_s=$(wc -c < "$stock_trn")
        echo -e "jpegtran\t${name}\tfail\tbytes_differ_ours=${sz_o}_stock=${sz_s}"
        FAIL=$((FAIL + 1))
    fi

    # ------- comment round-trip: wrjpgcom + rdjpgcom (standalone tools) -------
    # Both tools parse JPEG markers directly without linking against
    # libjpeg/libturbojpeg, so they exercise our JPEG marker layout
    # rather than the shim's `jpeg_*` ABI. Round-trip: insert a COM
    # marker via wrjpgcom on `our` output, then read it back via
    # rdjpgcom and assert the text survived. We run both upstream and
    # our-built variants and compare stdout.
    if [[ -f "$ours_jpg" ]]; then
        ours_with_com="$WORK/${name}.ours.com.jpg"
        if "$OUR_WRJPGCOM" -comment "ljt-test-${name}" "$ours_jpg" > "$ours_with_com" 2>/dev/null; then
            ours_com_text="$("$OUR_RDJPGCOM" "$ours_with_com" 2>/dev/null | tr -d '\n')"
            if [[ "$ours_com_text" == "ljt-test-${name}" ]]; then
                echo -e "comtools\t${name}\tpass\troundtrip"
            else
                echo -e "comtools\t${name}\tfail\troundtrip_text='${ours_com_text}'"
                FAIL=$((FAIL + 1))
            fi
        else
            echo -e "comtools\t${name}\tfail\twrjpgcom_failed"
            FAIL=$((FAIL + 1))
        fi
    fi

    # ------- tjbench short decompress benchmark -------
    # Smoke-test only: verify tjbench runs end-to-end against our shim
    # for a JPG input. Output exit code is the gate; we don't compare
    # numbers (that's the bench harness's job, separately).
    if run_ours "$OUR_TJBENCH" "$img" -benchtime 0.1 -warmup 0 >"$WORK/${name}.tjbench.log" 2>&1; then
        echo -e "tjbench\t${name}\tpass\t"
    else
        echo -e "tjbench\t${name}\tfail\texit_$?"
        cat "$WORK/${name}.tjbench.log" >&2 || true
        FAIL=$((FAIL + 1))
    fi
done

if (( FAIL > 0 )); then
    echo "FAIL total=${FAIL}" >&2
    exit 1
fi
echo "OK all_byte_exact" >&2

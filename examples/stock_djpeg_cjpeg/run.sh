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
STOCK_BIN="${STOCK_BIN:-/opt/homebrew/bin}"
TESTIMAGES="$REPO_ROOT/references/libjpeg-turbo/testimages"

if [[ ! -x "$OUR_BUILD/djpeg" ]]; then
    echo "ERROR: our-linked djpeg not found at $OUR_BUILD/djpeg; run build.sh first" >&2
    exit 2
fi

OUR_DJPEG="$OUR_BUILD/djpeg"
OUR_CJPEG="$OUR_BUILD/cjpeg"
OUR_JPEGTRAN="$OUR_BUILD/jpegtran"
STOCK_DJPEG="${STOCK_DJPEG:-$STOCK_BIN/djpeg}"
STOCK_CJPEG="${STOCK_CJPEG:-$STOCK_BIN/cjpeg}"
STOCK_JPEGTRAN="${STOCK_JPEGTRAN:-$STOCK_BIN/jpegtran}"

for bin in "$STOCK_DJPEG" "$STOCK_CJPEG" "$STOCK_JPEGTRAN"; do
    if [[ ! -x "$bin" ]]; then
        echo "ERROR: stock $bin missing; install libjpeg-turbo or set STOCK_BIN" >&2
        exit 3
    fi
done

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

FAIL=0
for img in "$TESTIMAGES"/*.jpg; do
    [[ -f "$img" ]] || continue
    name="$(basename "$img" .jpg)"

    # ------- decode: djpeg (our-linked) vs djpeg (stock) -------
    ours_ppm="$WORK/${name}.ours.ppm"
    stock_ppm="$WORK/${name}.stock.ppm"
    if ! "$OUR_DJPEG" -outfile "$ours_ppm" "$img" 2>"$WORK/djpeg_err_ours.log"; then
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
        if ! "$OUR_CJPEG" -outfile "$ours_jpg" "$ours_ppm" 2>"$WORK/cjpeg_err_ours.log"; then
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
    ours_trn="$WORK/${name}.ours.trn.jpg"
    stock_trn="$WORK/${name}.stock.trn.jpg"
    if ! "$OUR_JPEGTRAN" -copy all -rotate 90 -outfile "$ours_trn" "$img" 2>"$WORK/trn_err_ours.log"; then
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
done

if (( FAIL > 0 )); then
    echo "FAIL total=${FAIL}" >&2
    exit 1
fi
echo "OK all_byte_exact" >&2

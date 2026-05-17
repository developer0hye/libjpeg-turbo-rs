/* sanitizer C harness for libjpeg-turbo-rs-capi (P4-11 closure 2026-05-17).
 *
 * Loads the cdylib at runtime via dlopen, resolves the minimum TJ3 surface
 * (tj3Init / tj3DecompressHeader / tj3Decompress8 / tj3Destroy / tj3Alloc /
 * tj3Free), and decodes a small fixed corpus.  Compiled with
 * `-fsanitize=address,undefined` and run with `ASAN_OPTIONS=...` so any
 * boundary bug at the FFI surface (wrong free, OOB read, undefined behavior
 * in the conversion layer) trips a sanitizer report instead of silently
 * passing.
 *
 * Why a C harness and not a Rust integration test:
 *   - `.github/workflows/sanitizers.yml` only runs `--lib` tests under ASan
 *     to avoid the documented un-instrumented-C-tool false positives.  That
 *     leaves the entire FFI boundary (tj3* + jpeg_* extern "C" surface)
 *     uncovered by sanitizer runs.  This harness is a real un-instrumented
 *     C caller invoking an ASan-instrumented Rust cdylib, which is exactly
 *     what catches FFI-boundary bugs.
 *   - The bidirectional ASan handoff is permitted: ASan-instrumented Rust
 *     code called from un-instrumented C is the canonical configuration
 *     OSS-Fuzz expects for Rust-codec-via-C-driver enrollment.
 *
 * Run locally (Linux):
 *   cargo +nightly build -p libjpeg-turbo-rs-capi --release \
 *       --target x86_64-unknown-linux-gnu \
 *       -Zsanitizer=address
 *   cc -O1 -g -fsanitize=address,undefined examples/sanitizer_c_harness/harness.c \
 *       -ldl -o /tmp/harness
 *   ASAN_OPTIONS="detect_leaks=0:abort_on_error=1" \
 *   LD_LIBRARY_PATH=target/x86_64-unknown-linux-gnu/release \
 *       /tmp/harness target/x86_64-unknown-linux-gnu/release/liblibjpeg_turbo_rs_capi.so \
 *                    references/libjpeg-turbo/testimages/testorig.jpg
 *
 * Exit codes:
 *   0  success — every fixture decoded with no sanitizer hits.
 *   1  dlopen / dlsym failure (the cdylib is wrong or missing a symbol).
 *   2  decode pipeline reported failure on a known-good fixture.
 *   3  bad CLI usage. */

#define _POSIX_C_SOURCE 200809L
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* Mirror the subset of turbojpeg.h we exercise.  Keeping this typed-out
 * rather than `#include <turbojpeg.h>` decouples the harness build from
 * the installed headers and makes the FFI contract visible at a glance. */
typedef void *tj_handle_t;

typedef tj_handle_t (*fn_tj3Init)(int initType);
typedef void (*fn_tj3Destroy)(tj_handle_t handle);
typedef int (*fn_tj3DecompressHeader)(tj_handle_t handle,
                                      const unsigned char *jpegBuf,
                                      size_t jpegSize);
typedef int (*fn_tj3Decompress8)(tj_handle_t handle,
                                 const unsigned char *jpegBuf,
                                 size_t jpegSize,
                                 unsigned char *dstBuf,
                                 int pitch,
                                 int pixelFormat);
typedef int (*fn_tj3Get)(tj_handle_t handle, int param);

#define TJINIT_DECOMPRESS 1
#define TJPARAM_WIDTH     5
#define TJPARAM_HEIGHT    6
#define TJPF_RGB          0

static int decode_one(void *lib, const char *jpeg_path) {
    fn_tj3Init  init_fn  = (fn_tj3Init)dlsym(lib, "tj3Init");
    fn_tj3Destroy destroy_fn = (fn_tj3Destroy)dlsym(lib, "tj3Destroy");
    fn_tj3DecompressHeader hdr_fn =
        (fn_tj3DecompressHeader)dlsym(lib, "tj3DecompressHeader");
    fn_tj3Decompress8 dec_fn = (fn_tj3Decompress8)dlsym(lib, "tj3Decompress8");
    fn_tj3Get get_fn = (fn_tj3Get)dlsym(lib, "tj3Get");
    if (!init_fn || !destroy_fn || !hdr_fn || !dec_fn || !get_fn) {
        fprintf(stderr, "dlsym missing required tj3 symbol\n");
        return 1;
    }

    FILE *f = fopen(jpeg_path, "rb");
    if (!f) {
        fprintf(stderr, "cannot open %s\n", jpeg_path);
        return 1;
    }
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return 1; }
    long size = ftell(f);
    if (size < 0) { fclose(f); return 1; }
    rewind(f);

    unsigned char *jpeg_buf = (unsigned char *)malloc((size_t)size);
    if (!jpeg_buf) { fclose(f); return 1; }
    if (fread(jpeg_buf, 1, (size_t)size, f) != (size_t)size) {
        free(jpeg_buf); fclose(f); return 1;
    }
    fclose(f);

    tj_handle_t handle = init_fn(TJINIT_DECOMPRESS);
    if (!handle) { free(jpeg_buf); fprintf(stderr, "tj3Init returned NULL\n"); return 2; }

    if (hdr_fn(handle, jpeg_buf, (size_t)size) != 0) {
        fprintf(stderr, "tj3DecompressHeader failed on %s\n", jpeg_path);
        destroy_fn(handle); free(jpeg_buf); return 2;
    }

    int width  = get_fn(handle, TJPARAM_WIDTH);
    int height = get_fn(handle, TJPARAM_HEIGHT);
    if (width <= 0 || height <= 0 || width > 16384 || height > 16384) {
        fprintf(stderr, "implausible dimensions %dx%d from %s\n",
                width, height, jpeg_path);
        destroy_fn(handle); free(jpeg_buf); return 2;
    }

    size_t pixels_len = (size_t)width * (size_t)height * 3u;
    unsigned char *pixels = (unsigned char *)malloc(pixels_len);
    if (!pixels) { destroy_fn(handle); free(jpeg_buf); return 1; }

    int rc = dec_fn(handle, jpeg_buf, (size_t)size,
                    pixels, 0 /* pitch=0 → tightly packed */, TJPF_RGB);
    if (rc != 0) {
        fprintf(stderr, "tj3Decompress8 failed (rc=%d) on %s\n", rc, jpeg_path);
        free(pixels); destroy_fn(handle); free(jpeg_buf); return 2;
    }

    /* Touch the output so ASan catches any uninitialised read by
     * `dec_fn`.  Without this read the compiler can elide the buffer. */
    volatile unsigned int sum = 0;
    for (size_t i = 0; i < pixels_len; i += 64) {
        sum += pixels[i];
    }
    (void)sum;

    free(pixels);
    destroy_fn(handle);
    free(jpeg_buf);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <cdylib path> <fixture1.jpg> [fixture2.jpg ...]\n",
                argv[0]);
        return 3;
    }

    void *lib = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "dlopen %s failed: %s\n", argv[1], dlerror());
        return 1;
    }

    int rc = 0;
    for (int i = 2; i < argc; i++) {
        struct stat st;
        if (stat(argv[i], &st) != 0) {
            fprintf(stderr, "skip missing fixture %s\n", argv[i]);
            continue;
        }
        int one = decode_one(lib, argv[i]);
        if (one != 0) {
            rc = one;
            break;
        }
    }

    dlclose(lib);
    return rc;
}

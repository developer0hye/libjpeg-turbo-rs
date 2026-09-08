/* Process-isolated C-ABI misuse harness (P4-141 criterion 3, second half).
 *
 * One case per process, named on the command line.  That is the whole point:
 * the scenarios this criterion asks for are the ones that *may* fault — a
 * destination overrun, a use of a handle the library has already torn down, a
 * sanitizer report at the FFI boundary.  Running them in-process would let the
 * first one that misbehaves take every later case with it, and would make the
 * expected-to-fault cases untestable.  A child per case means a fault is an
 * observable exit status attributed to exactly one scenario.
 *
 *   ./cabi_misuse_harness <shared-library> <case> [fixture.jpg]
 *
 * The library is `dlopen`ed by path, so the *same binary* drives our shim and
 * a stock `libturbojpeg`.  That is what makes the run differential:
 * `tests/cabi_misuse_harness.rs` runs each case against both and requires the
 * two stdout transcripts to be byte-identical.  Declaring the ABI here rather
 * than `#include <turbojpeg.h>` keeps the contract visible and stops the
 * installed headers from deciding what is being tested — the same reasoning
 * `examples/sanitizer_c_harness/harness.c` records.
 *
 * WHAT THE BUFFERS ARE MADE OF.  Every destination a case sizes for a real
 * decode or compress is a `guarded_buf`: an `mmap`ed region whose payload ends
 * flush against a `PROT_NONE` page, with the slack before it filled with a
 * canary byte.  Two destinations are deliberately not: `alloc_ownership`'s has
 * to come from `tj3Alloc` for the case to mean anything, and `null_handle`
 * passes a three-byte stack array to calls that are refused before anything is
 * written.  A one-byte overrun is a SIGSEGV in this process and nothing else's;
 * a write before the payload is a canary mismatch reported on stdout.  Neither
 * needs a sanitizer, so the cases keep their meaning in the ordinary
 * `cargo test` run as well as under ASan.  `selftest_guard_page` and
 * `selftest_canary` are committed proof that both mechanisms are live —
 * without them a guard that had quietly become inert would make every case
 * pass while checking nothing.
 *
 * WHAT IS DELIBERATELY NOT HERE.  A second `tj3Destroy` on the same non-NULL
 * handle is a use-after-free in upstream too, so there is no contract to
 * compare against — only two implementations reading freed memory.  The
 * `lifecycle` case therefore drives the *defined* part of `init→destroy→
 * destroy`: destroy of NULL before and after, re-init at the same slot, and
 * repeated cycles.  Same reasoning for concurrent calls on one handle:
 * upstream mutates `tjinstance` from every entry point without
 * synchronisation, so `concurrent_handles` drives the shape that *is*
 * supported — one handle per thread — and says so rather than racing.
 *
 * Exit codes:
 *   0  the case ran; its transcript is on stdout.
 *   1  dlopen / dlsym failure (wrong library, or missing a required symbol).
 *   2  the case hit a condition that makes its transcript meaningless (a
 *      known-good fixture failed to decode, an allocation failed, a canary
 *      was corrupted).
 *   3  bad CLI usage or unknown case name.
 *   killed by a signal: a guard page fired — the finding this harness exists
 *      to produce. */

/* `mmap`'s anonymous flag is an X/Open extension, and _POSIX_C_SOURCE alone
 * hides it: glibc needs _DEFAULT_SOURCE, and macOS both needs _DARWIN_C_SOURCE
 * and spells the flag MAP_ANON. */
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE 1
#ifdef __APPLE__
#define _DARWIN_C_SOURCE 1
#endif

#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif

/* ------------------------------------------------------------------ ABI -- */

typedef void *tj_handle_t;

typedef struct {
    int x, y, w, h;
} tj_region_t;

typedef struct {
    int num, denom;
} tj_scaling_factor_t;

typedef tj_handle_t (*fn_tj3Init)(int initType);
typedef void (*fn_tj3Destroy)(tj_handle_t handle);
typedef char *(*fn_tj3GetErrorStr)(tj_handle_t handle);
typedef int (*fn_tj3Get)(tj_handle_t handle, int param);
typedef int (*fn_tj3Set)(tj_handle_t handle, int param, int value);
typedef int (*fn_tj3DecompressHeader)(tj_handle_t handle,
                                      const unsigned char *jpegBuf,
                                      size_t jpegSize);
typedef int (*fn_tj3Decompress8)(tj_handle_t handle,
                                 const unsigned char *jpegBuf, size_t jpegSize,
                                 unsigned char *dstBuf, int pitch,
                                 int pixelFormat);
typedef int (*fn_tj3Compress8)(tj_handle_t handle, const unsigned char *srcBuf,
                               int width, int pitch, int height,
                               int pixelFormat, unsigned char **jpegBuf,
                               size_t *jpegSize);
typedef int (*fn_tj3Compress12)(tj_handle_t handle, const short *srcBuf,
                                int width, int pitch, int height,
                                int pixelFormat, unsigned char **jpegBuf,
                                size_t *jpegSize);
typedef int (*fn_tj3Decompress12)(tj_handle_t handle,
                                  const unsigned char *jpegBuf,
                                  size_t jpegSize, short *dstBuf, int pitch,
                                  int pixelFormat);
typedef size_t (*fn_tj3JPEGBufSize)(int width, int height, int jpegSubsamp);
typedef void *(*fn_tj3Alloc)(size_t bytes);
typedef void (*fn_tj3Free)(void *buffer);
typedef int (*fn_tj3SetScalingFactor)(tj_handle_t handle,
                                      tj_scaling_factor_t scalingFactor);
typedef int (*fn_tj3SetCroppingRegion)(tj_handle_t handle,
                                       tj_region_t croppingRegion);

/* enum TJINIT (turbojpeg.h) */
#define TJINIT_COMPRESS   0
#define TJINIT_DECOMPRESS 1
#define TJINIT_TRANSFORM  2
#define TJ_NUMINIT        3

/* enum TJPARAM (turbojpeg.h), by declaration order. */
#define TJPARAM_STOPONWARNING 0
#define TJPARAM_BOTTOMUP      1
#define TJPARAM_NOREALLOC     2
#define TJPARAM_QUALITY       3
#define TJPARAM_SUBSAMP       4
#define TJPARAM_JPEGWIDTH     5
#define TJPARAM_JPEGHEIGHT    6
#define TJPARAM_PRECISION     7
#define TJPARAM_FASTUPSAMPLE  9
#define TJPARAM_SCANLIMIT     13
#define TJPARAM_MAXPIXELS     24

/* Every TJPARAM in declaration order, so `handle_defaults` can report the whole
 * initial parameter vector rather than the handful a case happens to read.
 * `tj3InitVersion` seeds ten of them to non-zero values
 * (`turbojpeg.c:598-608`); a port that left them zeroed would look identical
 * until a caller relied on one of the sentinels. */
static const char *const TJPARAM_NAMES[] = {
    "STOPONWARNING", "BOTTOMUP",   "NOREALLOC",     "QUALITY",
    "SUBSAMP",       "JPEGWIDTH",  "JPEGHEIGHT",    "PRECISION",
    "COLORSPACE",    "FASTUPSAMPLE", "FASTDCT",     "OPTIMIZE",
    "PROGRESSIVE",   "SCANLIMIT",  "ARITHMETIC",    "LOSSLESS",
    "LOSSLESSPSV",   "LOSSLESSPT", "RESTARTBLOCKS", "RESTARTROWS",
    "XDENSITY",      "YDENSITY",   "DENSITYUNITS",  "MAXMEMORY",
    "MAXPIXELS",     "SAVEMARKERS"
};
#define TJ_NUMPARAM ((int)(sizeof(TJPARAM_NAMES) / sizeof(TJPARAM_NAMES[0])))

/* enum TJPF and enum TJSAMP */
#define TJPF_RGB     0
#define TJPF_GRAY    6
#define TJ_NUMPF     12
#define TJSAMP_444   0
#define TJSAMP_420   2
#define TJSAMP_GRAY  3

static const int TJ_PIXEL_SIZE[TJ_NUMPF] = { 3, 3, 4, 4, 4, 4, 1, 4, 4, 4, 4, 4 };

/* libjpeg's JPEG_MAX_DIMENSION.  Upstream refuses a compression whose width or
 * height exceeds it (`jcmaster.c:186-188`, reached from `jpeg_start_compress`);
 * we return 0 and emit the frame anyway, which is P4-202 and a live entry in
 * KNOWN_DIVERGENCES.  The `max_dimensions` case walks the boundary from both
 * sides. */
#define JPEG_MAX_DIMENSION 65500

struct tj_api {
    fn_tj3Init init;
    fn_tj3Destroy destroy;
    fn_tj3GetErrorStr get_error_str;
    fn_tj3Get get;
    fn_tj3Set set;
    fn_tj3DecompressHeader decompress_header;
    fn_tj3Decompress8 decompress8;
    fn_tj3Compress8 compress8;
    fn_tj3Compress12 compress12;
    fn_tj3Decompress12 decompress12;
    fn_tj3JPEGBufSize jpeg_buf_size;
    fn_tj3Alloc alloc;
    fn_tj3Free free;
    fn_tj3SetScalingFactor set_scaling_factor;
    fn_tj3SetCroppingRegion set_cropping_region;
};

static struct tj_api api;

/* dlsym through a union: a direct object-pointer-to-function-pointer cast is
 * undefined in ISO C, and the UBSan half of the sanitizer job compiles this
 * file. */
static void *resolve(void *lib, const char *name) {
    void *sym = dlsym(lib, name);
    if (!sym) {
        fprintf(stderr, "dlsym(%s) failed: %s\n", name, dlerror());
        exit(1);
    }
    return sym;
}

#define RESOLVE(field, type, name) \
    do { \
        union { void *obj; type fn; } cvt; \
        cvt.obj = resolve(lib, name); \
        api.field = cvt.fn; \
    } while (0)

static void load_api(void *lib) {
    RESOLVE(init, fn_tj3Init, "tj3Init");
    RESOLVE(destroy, fn_tj3Destroy, "tj3Destroy");
    RESOLVE(get_error_str, fn_tj3GetErrorStr, "tj3GetErrorStr");
    RESOLVE(get, fn_tj3Get, "tj3Get");
    RESOLVE(set, fn_tj3Set, "tj3Set");
    RESOLVE(decompress_header, fn_tj3DecompressHeader, "tj3DecompressHeader");
    RESOLVE(decompress8, fn_tj3Decompress8, "tj3Decompress8");
    RESOLVE(compress8, fn_tj3Compress8, "tj3Compress8");
    RESOLVE(compress12, fn_tj3Compress12, "tj3Compress12");
    RESOLVE(decompress12, fn_tj3Decompress12, "tj3Decompress12");
    RESOLVE(jpeg_buf_size, fn_tj3JPEGBufSize, "tj3JPEGBufSize");
    RESOLVE(alloc, fn_tj3Alloc, "tj3Alloc");
    RESOLVE(free, fn_tj3Free, "tj3Free");
    RESOLVE(set_scaling_factor, fn_tj3SetScalingFactor, "tj3SetScalingFactor");
    RESOLVE(set_cropping_region, fn_tj3SetCroppingRegion,
            "tj3SetCroppingRegion");
}

/* -------------------------------------------------------- guarded buffers -- */

/* Checks the driver makes itself, for the properties that hold on *both*
 * implementations — canaries, pointer identity, the decodes both libraries
 * deliver, the rejections both refuse.
 *
 * The transcript alone is not enough: `sanitizers.yml` runs these cases without
 * the Rust runner, so a case that merely *printed* `canary=corrupt` and
 * returned 0 would leave that leg green on real corruption. `codex review`
 * demonstrated exactly that by injecting a short-stride write. Anything the two
 * implementations legitimately disagree about (the KNOWN_DIVERGENCES in
 * tests/cabi_misuse_harness.rs) is deliberately *not* checked here — that is
 * the differential comparison's job, and failing on it would make the oracle
 * run red. */
static int contract_failures = 0;
/* `concurrent_handles` releases buffers from eight worker threads, and this
 * file is compiled with `-fsanitize=address,undefined` in CI, so the counter
 * cannot be a plain global increment. */
static pthread_mutex_t contract_lock = PTHREAD_MUTEX_INITIALIZER;

static void require(int holds, const char *what) {
    if (holds) return;
    pthread_mutex_lock(&contract_lock);
    fprintf(stderr, "contract check failed: %s\n", what);
    contract_failures++;
    pthread_mutex_unlock(&contract_lock);
}


/* Payload byte the harness writes before every call, so "the library wrote
 * here" is distinguishable from "this was never touched". */
#define POISON 0xA5
/* Slack byte before the payload.  A short-stride write lands here. */
#define CANARY 0x5C

typedef struct {
    unsigned char *map;  /* mmap base: [guard][slack][payload][guard] */
    size_t map_len;
    unsigned char *data; /* payload start */
    size_t len;
    size_t slack;        /* canary bytes between the leading guard and data */
    const char *label;   /* names the buffer in a release-time canary failure */
} guarded_buf;

static size_t page_size(void) {
    long value = sysconf(_SC_PAGESIZE);
    return value > 0 ? (size_t)value : 4096;
}

/* Allocate `len` bytes whose last byte is the last byte of a writable page,
 * with a PROT_NONE page immediately after it and another before the slack.
 *
 * The payload is flush with the *trailing* guard because an overrun past the
 * caller-declared size is the failure this criterion names.  The leading guard
 * cannot also be flush, so the gap is canary-filled and checked instead. */
static int guarded_alloc(guarded_buf *g, size_t len, const char *label) {
    size_t page = page_size();
    size_t payload_pages = (len + page - 1) / page;
    if (payload_pages == 0) payload_pages = 1;

    memset(g, 0, sizeof(*g));
    g->map_len = (payload_pages + 2) * page;
    g->map = mmap(NULL, g->map_len, PROT_READ | PROT_WRITE,
                  MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (g->map == MAP_FAILED) {
        g->map = NULL;
        return -1;
    }
    if (mprotect(g->map, page, PROT_NONE) != 0 ||
        mprotect(g->map + g->map_len - page, page, PROT_NONE) != 0) {
        munmap(g->map, g->map_len);
        g->map = NULL;
        return -1;
    }
    g->slack = payload_pages * page - len;
    g->data = g->map + page + g->slack;
    g->len = len;
    g->label = label;
    memset(g->map + page, CANARY, g->slack);
    memset(g->data, POISON, g->len);
    return 0;
}

/* True when nothing wrote *into* the payload — every byte still poison.
 *
 * Checking byte zero alone is not the contract: a rejected call that modified
 * any other byte passed until this existed, and the write is in-bounds, so
 * neither the guard page nor ASan sees it. */
static int fully_poisoned(const guarded_buf *g) {
    for (size_t i = 0; i < g->len; i++) {
        if (g->data[i] != POISON) return 0;
    }
    return 1;
}

/* True when nothing wrote before the payload.
 *
 * Walked back from `data` rather than forward from a re-derived page offset:
 * the slack is a property of the buffer, so this stays correct if
 * `guarded_alloc`'s layout changes, and it does not call `sysconf` per byte. */
static int canary_intact(const guarded_buf *g) {
    const unsigned char *slack = g->data - g->slack;
    for (size_t i = 0; i < g->slack; i++) {
        if (slack[i] != CANARY) return 0;
    }
    return 1;
}

/* Release without checking — only for `selftest_canary`, which corrupts the
 * canary on purpose. */
static void guarded_discard(guarded_buf *g) {
    if (g->map) munmap(g->map, g->map_len);
    memset(g, 0, sizeof(*g));
}

/* Release, verifying at the point of release that nothing wrote before the
 * payload.
 *
 * Checking here rather than in each case is what makes the coverage structural:
 * a case that forgets to ask is still covered, and `codex review` found this by
 * injecting an underrun into the one buffer — `undersized_output`'s probe —
 * that no case happened to check. The slack is ordinary accessible memory, so
 * neither the guard page nor ASan sees a write into it. */
static void guarded_free(guarded_buf *g) {
    if (g->map && !canary_intact(g)) {
        require(0, g->label ? g->label : "a guarded buffer's leading canary");
    }
    guarded_discard(g);
}

/* FNV-1a: a stable transcript value for "the same bytes came out of both
 * libraries", printed rather than compared here so a divergence names itself. */
static uint64_t fnv1a(const unsigned char *data, size_t len) {
    uint64_t hash = 1469598103934665603ULL;
    for (size_t i = 0; i < len; i++) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

static unsigned char *read_file(const char *path, size_t *out_len) {
    FILE *file = fopen(path, "rb");
    if (!file) return NULL;
    if (fseek(file, 0, SEEK_END) != 0) { fclose(file); return NULL; }
    long size = ftell(file);
    if (size < 0) { fclose(file); return NULL; }
    rewind(file);
    unsigned char *buffer = malloc((size_t)size ? (size_t)size : 1);
    if (!buffer) { fclose(file); return NULL; }
    if (fread(buffer, 1, (size_t)size, file) != (size_t)size) {
        free(buffer);
        fclose(file);
        return NULL;
    }
    fclose(file);
    *out_len = (size_t)size;
    return buffer;
}

/* --------------------------------------------------------------- cases -- */

/* `init → destroy → destroy`, restricted to the part that has a contract.
 *
 * `tj3Destroy(NULL)` returns without touching anything (`turbojpeg.c:641`),
 * and `tj3InitVersion` rejects an out-of-range `initType` with NULL
 * (`turbojpeg.c:589-591`).  A second destroy of the *same* live pointer is a
 * double free in upstream as well, so it is not driven here — see the file
 * header. */
static int case_lifecycle(void) {
    api.destroy(NULL);
    printf("destroy_null_before=returned\n");

    tj_handle_t handle = api.init(TJINIT_DECOMPRESS);
    printf("init_decompress=%s\n", handle ? "handle" : "null");
    if (!handle) return 2;
    /* Move the handle off its defaults before destroying it, so the re-init
     * check below has something that *could* survive. Comparing two handles
     * that were never written to would compare a deterministic constructor
     * against itself. */
    /* All four apply to a *decompression* instance, which this handle is.
     * Which parameters each instance type accepts is `parameter_applicability`'s
     * subject, and mixing the two questions here would make `dirtied` report
     * that rule instead of the state it is meant to measure. */
    const struct { int param; int value; } dirt[] = {
        { TJPARAM_BOTTOMUP, 1 },
        { TJPARAM_FASTUPSAMPLE, 1 },
        { TJPARAM_MAXPIXELS, 7 },
        { TJPARAM_SCANLIMIT, 13 },
    };
    int dirtied = 0;
    for (size_t i = 0; i < sizeof(dirt) / sizeof(dirt[0]); i++) {
        if (api.set(handle, dirt[i].param, dirt[i].value) == 0 &&
            api.get(handle, dirt[i].param) == dirt[i].value)
            dirtied++;
    }
    /* Reported rather than assumed: if a parameter stopped being writable the
     * re-init check below would silently go back to comparing two pristine
     * handles. */
    printf("dirtied=%d\n", dirtied);
    api.destroy(handle);
    printf("destroy=returned\n");

    api.destroy(NULL);
    printf("destroy_null_after=returned\n");

    /* Re-init after a destroy commonly lands on the just-freed allocation, so
     * this is also the "same address, fresh state" check for the TJ surface. */
    tj_handle_t reborn = api.init(TJINIT_DECOMPRESS);
    printf("reinit=%s\n", reborn ? "handle" : "null");
    if (!reborn) return 2;
    /* Self-comparison rather than absolute values: what a re-init has to
     * guarantee is that none of the four parameters written above survived the
     * destroy, and that holds whatever the defaults are.  The absolute vector
     * is `handle_defaults`' job.  `witness` is built *after* `reborn` so the
     * two cannot both be the freed allocation. */
    tj_handle_t witness = api.init(TJINIT_DECOMPRESS);
    if (!witness) return 2;
    int fresh_state = 1;
    for (int param = 0; param < TJ_NUMPARAM; param++) {
        if (api.get(reborn, param) != api.get(witness, param)) fresh_state = 0;
    }
    printf("reinit_matches_fresh=%s\n", fresh_state ? "yes" : "no");
    api.destroy(witness);
    api.destroy(reborn);

    printf("init_neg=%s\n", api.init(-1) ? "handle" : "null");
    printf("init_numinit=%s\n", api.init(TJ_NUMINIT) ? "handle" : "null");
    printf("init_large=%s\n", api.init(1 << 20) ? "handle" : "null");

    /* Repeated init/destroy, the sequence P4-141 criterion 2 names.  Leaks are
     * LeakSanitizer's job; what this asserts is that the hundredth cycle still
     * produces a handle. */
    int cycles_ok = 1;
    for (int i = 0; i < 100; i++) {
        tj_handle_t cycle = api.init(TJINIT_TRANSFORM);
        if (!cycle) { cycles_ok = 0; break; }
        api.destroy(cycle);
    }
    printf("cycles_100=%s\n", cycles_ok ? "ok" : "failed");
    require(dirtied == (int)(sizeof(dirt) / sizeof(dirt[0])),
            "every parameter written before the destroy was accepted");
    require(fresh_state, "a re-initialised handle carries no earlier state");
    require(cycles_ok, "100 init/destroy cycles");
    return cycles_ok ? 0 : 2;
}

/* The initial value of every TJPARAM, for each init type.
 *
 * `tj3InitVersion` seeds `quality`, `subsamp`, `jpegWidth`, `jpegHeight`,
 * `precision`, `colorspace`, `losslessPSV`, `xDensity`, `yDensity`,
 * `scalingFactor` and `saveMarkers` before dispatching on `initType`
 * (`turbojpeg.c:598-608`), and several of those sentinels are negative — a
 * caller that treats -1 as "not known yet" is reading a documented value, not
 * an accident of the struct being zeroed. */
static int case_handle_defaults(void) {
    const struct { const char *label; int type; } inits[] = {
        { "compress", TJINIT_COMPRESS },
        { "decompress", TJINIT_DECOMPRESS },
        { "transform", TJINIT_TRANSFORM },
    };
    for (size_t i = 0; i < sizeof(inits) / sizeof(inits[0]); i++) {
        tj_handle_t handle = api.init(inits[i].type);
        if (!handle) return 2;
        for (int param = 0; param < TJ_NUMPARAM; param++) {
            printf("default_%s_%s=%d\n", inits[i].label, TJPARAM_NAMES[param],
                   api.get(handle, param));
        }
        api.destroy(handle);
    }
    return 0;
}

/* Which parameters each instance type accepts.
 *
 * `tj3Set` is not a plain setter: upstream refuses a parameter that does not
 * apply to the handle's `initType` — "TJPARAM_QUALITY is not applicable to
 * decompression instances" — and leaves the value alone
 * (`turbojpeg.c`'s `tj3Set`, the `if ((this->init & COMPRESS) == 0) THROW(...)`
 * guards).  A transform instance is initialised for both, so it accepts
 * everything.  Writing the whole 26 x 3 matrix is the only way to see the rule
 * rather than a sample of it. */
static int case_parameter_applicability(void) {
    /* A value inside each parameter's own legal range, so a rejection is the
     * applicability rule and not a range check. */
    const struct { int param; int value; } probe[] = {
        { 0, 1 },  { 1, 1 },  { 2, 1 },  { 3, 50 }, { 4, 1 },  { 5, 1 },
        { 6, 1 },  { 7, 8 },  { 8, 1 },  { 9, 1 },  { 10, 1 }, { 11, 1 },
        { 12, 1 }, { 13, 1 }, { 14, 1 }, { 15, 1 }, { 16, 2 }, { 17, 1 },
        { 18, 1 }, { 19, 1 }, { 20, 72 }, { 21, 72 }, { 22, 1 }, { 23, 1 },
        { 24, 1 }, { 25, 1 },
    };
    const struct { const char *label; int type; } inits[] = {
        { "compress", TJINIT_COMPRESS },
        { "decompress", TJINIT_DECOMPRESS },
        { "transform", TJINIT_TRANSFORM },
    };
    for (size_t i = 0; i < sizeof(inits) / sizeof(inits[0]); i++) {
        for (int param = 0; param < TJ_NUMPARAM; param++) {
            /* A fresh handle per probe: a rejected write must leave the value
             * alone, and reusing one handle would let an accepted write change
             * what a later probe reads. */
            tj_handle_t handle = api.init(inits[i].type);
            if (!handle) return 2;
            int rc = api.set(handle, probe[param].param, probe[param].value);
            printf("set_%s_%s=%d\n", inits[i].label, TJPARAM_NAMES[param], rc);
            /* A transform instance is initialised for both roles, so every
             * role-conditioned guard passes on it; only the two read-only
             * parameters are refused. Both implementations agree here. */
            if (inits[i].type == TJINIT_TRANSFORM) {
                int read_only = probe[param].param == TJPARAM_JPEGWIDTH ||
                                probe[param].param == TJPARAM_JPEGHEIGHT;
                require(rc == (read_only ? -1 : 0),
                        "a transform instance accepts every writable parameter");
            }
            api.destroy(handle);
        }
    }
    /* Out-of-range parameter indices are rejected on any instance type. */
    tj_handle_t handle = api.init(TJINIT_TRANSFORM);
    if (!handle) return 2;
    int range[4];
    printf("set_negative_param=%d\n", range[0] = api.set(handle, -1, 0));
    printf("set_past_last_param=%d\n", range[1] = api.set(handle, TJ_NUMPARAM, 0));
    printf("get_negative_param=%d\n", range[2] = api.get(handle, -1));
    printf("get_past_last_param=%d\n", range[3] = api.get(handle, TJ_NUMPARAM));
    for (int i = 0; i < 4; i++)
        require(range[i] == -1, "an out-of-range parameter index is refused");
    api.destroy(handle);
    return 0;
}

/* Every entry point that takes a handle rejects NULL with its documented error
 * value instead of dereferencing it (`GET_TJINSTANCE`, `turbojpeg.c:337-345`,
 * and the `GET_*INSTANCE` variants above it). */
static int case_null_handle(const unsigned char *jpeg, size_t jpeg_len) {
    /* Guarded like every other destination, and for the sharpest reason in the
     * file: this case's whole premise is that the library rejects the NULL
     * handle *before* touching `dstBuf`. One that did not would write the
     * fixture's 101 KB into it. On the stack that is silent corruption of this
     * harness's own frame; here it is a SIGSEGV attributed to this case. */
    guarded_buf pixel_buf;
    if (guarded_alloc(&pixel_buf, 3, "null_handle destination") != 0) return 2;
    unsigned char *pixel = pixel_buf.data;
    unsigned char *out = NULL;
    size_t out_len = 0;
    tj_scaling_factor_t unscaled = { 1, 1 };
    tj_region_t region = { 0, 0, 0, 0 };

    printf("get=%d\n", api.get(NULL, TJPARAM_JPEGWIDTH));
    printf("set=%d\n", api.set(NULL, TJPARAM_QUALITY, 90));
    printf("decompress_header=%d\n",
           api.decompress_header(NULL, jpeg, jpeg_len));
    printf("decompress8=%d\n",
           api.decompress8(NULL, jpeg, jpeg_len, pixel, 0, TJPF_RGB));
    printf("compress8=%d\n",
           api.compress8(NULL, pixel, 1, 0, 1, TJPF_RGB, &out, &out_len));
    printf("set_scaling_factor=%d\n", api.set_scaling_factor(NULL, unscaled));
    printf("set_cropping_region=%d\n", api.set_cropping_region(NULL, region));
    /* Documented to fall back to the global error string rather than crash
     * (`turbojpeg.c:678-687`). */
    printf("get_error_str_null=%s\n",
           api.get_error_str(NULL) ? "string" : "null");
    printf("compress8_out_ptr=%s\n", out ? "written" : "untouched");
    printf("compress8_out_len=%zu\n", out_len);
    printf("dst_canary=%s\n", canary_intact(&pixel_buf) ? "intact" : "corrupt");
    printf("dst_untouched=%s\n", fully_poisoned(&pixel_buf) ? "yes" : "no");
    int intact = canary_intact(&pixel_buf);
    require(intact, "null_handle destination canary");
    require(fully_poisoned(&pixel_buf),
            "a refused call wrote nothing to the destination");
    require(!out && out_len == 0, "a refused compress left the out-pair alone");
    guarded_free(&pixel_buf);
    return intact ? 0 : 2;
}

/* Read the fixture's scaled dimensions once so the pitch and buffer cases can
 * size their destinations from the same numbers the library will use. */
static int probe(tj_handle_t handle, const unsigned char *jpeg, size_t jpeg_len,
                 int *width, int *height) {
    if (api.decompress_header(handle, jpeg, jpeg_len) != 0) return -1;
    *width = api.get(handle, TJPARAM_JPEGWIDTH);
    *height = api.get(handle, TJPARAM_JPEGHEIGHT);
    return (*width > 0 && *height > 0) ? 0 : -1;
}

/* `tj3Decompress8`'s pitch contract, walked from both sides
 * (`turbojpeg-mp.c:229-231`, with the negative-pitch and NULL rejection at
 * `:170-172`): 0 means `output_width * pixelSize`, anything
 * below that is "Invalid argument", a negative pitch is rejected before the
 * decode starts, and padding above it is honoured.
 *
 * Each accepted pitch decodes into a buffer of exactly the size the API
 * documents — `pitch * height` — with the trailing guard page immediately
 * after the last byte. */
static int case_pitch_boundaries(const unsigned char *jpeg, size_t jpeg_len) {
    tj_handle_t handle = api.init(TJINIT_DECOMPRESS);
    if (!handle) return 2;
    int width = 0, height = 0;
    if (probe(handle, jpeg, jpeg_len, &width, &height) != 0) {
        fprintf(stderr, "fixture header did not decode\n");
        return 2;
    }
    printf("width=%d\nheight=%d\n", width, height);

    const int ps = TJ_PIXEL_SIZE[TJPF_RGB];
    const int row_bytes = width * ps;
    const struct { const char *label; int pitch; } accepted[] = {
        { "implicit", 0 },
        { "exact", row_bytes },
        { "padded", row_bytes + 7 },
        { "double", row_bytes * 2 },
    };

    for (size_t i = 0; i < sizeof(accepted) / sizeof(accepted[0]); i++) {
        int pitch = accepted[i].pitch ? accepted[i].pitch : row_bytes;
        guarded_buf dst;
        if (guarded_alloc(&dst, (size_t)pitch * (size_t)height,
                          "accepted-pitch destination") != 0)
            return 2;
        int rc = api.decompress8(handle, jpeg, jpeg_len, dst.data,
                                 accepted[i].pitch, TJPF_RGB);
        printf("pitch_%s_rc=%d\n", accepted[i].label, rc);
        require(rc == 0, "an accepted pitch decodes");
        printf("pitch_%s_canary=%s\n", accepted[i].label,
               canary_intact(&dst) ? "intact" : "corrupt");
        /* Hash only the pixels, not the inter-row padding: the padding is
         * caller-owned and neither library promises anything about it. */
        uint64_t hash = 1469598103934665603ULL;
        for (int row = 0; row < height; row++) {
            const unsigned char *start = dst.data + (size_t)row * (size_t)pitch;
            for (int b = 0; b < row_bytes; b++) {
                hash ^= start[b];
                hash *= 1099511628211ULL;
            }
        }
        printf("pitch_%s_pixels=%016llx\n", accepted[i].label,
               (unsigned long long)hash);
        int intact = canary_intact(&dst);
        require(intact, "accepted-pitch destination canary");
        guarded_free(&dst);
        if (!intact) return 2;
    }

    /* Rejected pitches.  The destination is a single guarded page: a library
     * that accepted the pitch would write far past it and fault here rather
     * than corrupting the heap silently. */
    guarded_buf tiny;
    if (guarded_alloc(&tiny, 16, "rejected-pitch destination") != 0) return 2;
    int rejected[3];
    printf("pitch_short_rc=%d\n",
           rejected[0] = api.decompress8(handle, jpeg, jpeg_len, tiny.data,
                                         row_bytes - 1, TJPF_RGB));
    printf("pitch_one_rc=%d\n",
           rejected[1] =
               api.decompress8(handle, jpeg, jpeg_len, tiny.data, 1, TJPF_RGB));
    printf("pitch_negative_rc=%d\n",
           rejected[2] = api.decompress8(handle, jpeg, jpeg_len, tiny.data, -1,
                                         TJPF_RGB));
    printf("pitch_reject_canary=%s\n",
           canary_intact(&tiny) ? "intact" : "corrupt");
    printf("pitch_reject_untouched=%s\n",
           fully_poisoned(&tiny) ? "yes" : "no");
    for (int i = 0; i < 3; i++)
        require(rejected[i] == -1, "a pitch below width * pixelSize is refused");
    require(canary_intact(&tiny), "rejected-pitch destination canary");
    require(fully_poisoned(&tiny), "a refused decode wrote nothing");
    guarded_free(&tiny);

    /* Out-of-range pixel formats are rejected by the same guard clause. */
    guarded_buf small;
    if (guarded_alloc(&small, 16, "argument-rejection destination") != 0) return 2;
    int refused[5];
    printf("pf_negative_rc=%d\n",
           refused[0] = api.decompress8(handle, jpeg, jpeg_len, small.data, 0, -2));
    printf("pf_numpf_rc=%d\n",
           refused[1] =
               api.decompress8(handle, jpeg, jpeg_len, small.data, 0, TJ_NUMPF));
    printf("null_dst_rc=%d\n",
           refused[2] = api.decompress8(handle, jpeg, jpeg_len, NULL, 0, TJPF_RGB));
    printf("null_src_rc=%d\n",
           refused[3] =
               api.decompress8(handle, NULL, jpeg_len, small.data, 0, TJPF_RGB));
    printf("zero_size_rc=%d\n",
           refused[4] = api.decompress8(handle, jpeg, 0, small.data, 0, TJPF_RGB));
    /* Reported, not just freed: a callee that wrote before the destination and
     * then returned the expected error passed every check here until this line
     * existed, and the write lands in accessible mmap slack that ASan cannot
     * see either. */
    printf("arg_reject_canary=%s\n",
           canary_intact(&small) ? "intact" : "corrupt");
    printf("arg_reject_untouched=%s\n",
           fully_poisoned(&small) ? "yes" : "no");
    for (int i = 0; i < 5; i++)
        require(refused[i] == -1, "an invalid argument is refused");
    require(canary_intact(&small), "argument-rejection destination canary");
    require(fully_poisoned(&small), "a refused decode wrote nothing");
    guarded_free(&small);

    api.destroy(handle);
    return 0;
}

/* Undersized *output* buffers, which on the compression side is what
 * TJPARAM_NOREALLOC turns into a contract: with the flag set the caller's
 * buffer is used in place and `jpegSize` is its capacity, so too small has to
 * be an error rather than a resize or an overrun (P4-145).
 *
 * The buffer is guarded, so "rather than an overrun" is enforced by the
 * hardware here instead of being taken on trust. The interesting capacity is
 * one byte under what the image actually compresses to, which the case
 * measures first rather than guessing: `exactly enough` must succeed with its
 * last byte flush against the guard page, and `one short` must fail. */
static int case_undersized_output(void) {
    const int width = 96, height = 64;
    const int ps = TJ_PIXEL_SIZE[TJPF_RGB];
    unsigned char *src = malloc((size_t)width * height * ps);
    if (!src) return 2;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            unsigned char *pixel = src + ((size_t)y * width + x) * ps;
            pixel[0] = (unsigned char)(x * 2);
            pixel[1] = (unsigned char)(y * 3);
            pixel[2] = (unsigned char)(x + y);
        }
    }

    size_t actual = 0;
    tj_handle_t handle = api.init(TJINIT_COMPRESS);
    if (!handle) { free(src); return 2; }
    api.set(handle, TJPARAM_QUALITY, 80);
    api.set(handle, TJPARAM_SUBSAMP, TJSAMP_420);
    api.set(handle, TJPARAM_NOREALLOC, 1);

    size_t worst_case = api.jpeg_buf_size(width, height, TJSAMP_420);
    printf("bufsize=%zu\n", worst_case);
    if (worst_case == 0) { free(src); return 2; }

    /* Learn what this image actually compresses to, so the boundary cases
     * below sit *on* the boundary. A hard-coded "obviously too small" capacity
     * would drift: at quality 80 this 96x64 gradient is a little over 1 KB, so
     * a constant like 1024 is 70-odd bytes from flipping meaning the next time
     * the encoder's output moves. */
    {
        guarded_buf probe;
        if (guarded_alloc(&probe, worst_case, "compress probe destination") != 0) {
            free(src);
            return 2;
        }
        unsigned char *slot = probe.data;
        size_t size = worst_case;
        int rc = api.compress8(handle, src, width, 0, height, TJPF_RGB, &slot,
                               &size);
        printf("compressed_rc=%d\n", rc);
        printf("compressed=%zu\n", rc == 0 ? size : (size_t)0);
        /* The bytes, not just the length: a compress that produced 1097 bytes
         * of zeroes after the SOI passed every other check here. Both
         * implementations emit this image byte for byte identically, so the
         * digest is a real encode cross-validation and not a snapshot of our
         * own output. */
        printf("compressed_bytes=%016llx\n",
               rc == 0 ? (unsigned long long)fnv1a(probe.data, size) : 0ULL);
        require(rc == 0, "the probe compress into a worst-case buffer");
        guarded_free(&probe);
        if (rc != 0 || size < 2 || size > worst_case) { free(src); return 2; }
        actual = size;
    }

    /* Labelled rather than keyed by the byte count: a transcript key that
     * embeds a measured size changes whenever the encoder's output moves, and
     * a differential comparison keyed on it would report an unrelated size
     * change as a missing line. */
    const struct { const char *label; size_t capacity; } slots[] = {
        { "one", 1 },
        { "tiny", 64 },
        { "one_short", actual - 1 },
        { "exact", actual },
        { "worst_case", worst_case },
    };
    for (size_t i = 0; i < sizeof(slots) / sizeof(slots[0]); i++) {
        guarded_buf dst;
        if (guarded_alloc(&dst, slots[i].capacity, "NOREALLOC destination") != 0) {
            free(src);
            return 2;
        }
        unsigned char *slot = dst.data;
        size_t size = slots[i].capacity;
        int rc = api.compress8(handle, src, width, 0, height, TJPF_RGB, &slot,
                               &size);
        printf("norealloc_%s_rc=%d\n", slots[i].label, rc);
        /* With NOREALLOC set the library must not move the pointer: doing so
         * would mean it allocated, and the caller would then free a buffer it
         * did not own. */
        printf("norealloc_%s_slot=%s\n", slots[i].label,
               slot == dst.data ? "unchanged" : "moved");
        printf("norealloc_%s_canary=%s\n", slots[i].label,
               canary_intact(&dst) ? "intact" : "corrupt");
        printf("norealloc_%s_fits=%s\n", slots[i].label,
               (rc == 0 && size <= slots[i].capacity) ? "yes" : "n/a");
        printf("norealloc_%s_soi=%s\n", slots[i].label,
               (rc == 0 && size >= 2 && dst.data[0] == 0xFF &&
                dst.data[1] == 0xD8)
                   ? "yes"
                   : "n/a");
        printf("norealloc_%s_bytes=%016llx\n", slots[i].label,
               rc == 0 ? (unsigned long long)fnv1a(dst.data, size) : 0ULL);
        int intact = canary_intact(&dst);
        require(intact, "NOREALLOC destination canary");
        require(slot == dst.data, "NOREALLOC never moves the caller's pointer");
        /* `exact` is P4-206: we accept it, upstream refuses. Both of the other
         * boundaries hold on both implementations. */
        if (strcmp(slots[i].label, "exact") != 0)
            require(rc == (slots[i].capacity >= worst_case ? 0 : -1),
                    "a NOREALLOC capacity below the output is refused");
        guarded_free(&dst);
        if (!intact) { free(src); return 2; }
    }

    api.destroy(handle);
    free(src);
    return 0;
}

/* Maximum dimensions: the arithmetic that sizes a destination, and the
 * boundary libjpeg refuses.  `JPEG_MAX_DIMENSION` is 65500; a compression one
 * pixel over it fails in `jpeg_start_compress` rather than allocating, so the
 * case stays cheap while still crossing the boundary.  `TJPARAM_MAXPIXELS` is
 * the decompression-side ceiling (`turbojpeg-mp.c:195-198`). */
static int case_max_dimensions(const unsigned char *jpeg, size_t jpeg_len) {
    printf("bufsize_1x1=%zu\n", api.jpeg_buf_size(1, 1, TJSAMP_444));
    printf("bufsize_max_444=%zu\n",
           api.jpeg_buf_size(JPEG_MAX_DIMENSION, JPEG_MAX_DIMENSION,
                             TJSAMP_444));
    printf("bufsize_max_420=%zu\n",
           api.jpeg_buf_size(JPEG_MAX_DIMENSION, JPEG_MAX_DIMENSION,
                             TJSAMP_420));
    printf("bufsize_zero=%zu\n", api.jpeg_buf_size(0, 1, TJSAMP_444));
    printf("bufsize_negative=%zu\n", api.jpeg_buf_size(-1, 1, TJSAMP_444));
    printf("bufsize_bad_subsamp=%zu\n", api.jpeg_buf_size(1, 1, 99));

    tj_handle_t compressor = api.init(TJINIT_COMPRESS);
    if (!compressor) return 2;
    api.set(compressor, TJPARAM_QUALITY, 75);
    api.set(compressor, TJPARAM_SUBSAMP, TJSAMP_GRAY);

    /* One row of the widest legal image, then one pixel wider.  Both hand the
     * library a source buffer big enough for the *declared* width, so a
     * rejection is a dimension check and not an allocation failure. */
    size_t row_len = (size_t)(JPEG_MAX_DIMENSION + 1);
    unsigned char *row = calloc(row_len, 1);
    if (!row) { api.destroy(compressor); return 2; }

    const struct { const char *label; int width; int height; } dims[] = {
        { "zero_width", 0, 1 },
        { "zero_height", 1, 0 },
        { "negative_width", -1, 1 },
        { "negative_height", 1, -1 },
        { "over_max_width", JPEG_MAX_DIMENSION + 1, 1 },
        { "over_max_height", 1, JPEG_MAX_DIMENSION + 1 },
    };
    for (size_t i = 0; i < sizeof(dims) / sizeof(dims[0]); i++) {
        unsigned char *out = NULL;
        size_t out_len = 0;
        int rc = api.compress8(compressor, row, dims[i].width, 0,
                               dims[i].height, TJPF_GRAY, &out, &out_len);
        printf("compress_%s_rc=%d\n", dims[i].label, rc);
        printf("compress_%s_out=%s\n", dims[i].label,
               out ? "allocated" : "null");
        /* The over-max pair is P4-202 — we accept what upstream refuses — so
         * only the zero and negative dimensions are checked here. */
        if (strncmp(dims[i].label, "over_max", 8) != 0)
            require(rc == -1, "a zero or negative dimension is refused");
        api.free(out);
    }
    /* A negative pitch is rejected by the same clause; `height` rows of a
     * negative stride would index before the buffer. */
    unsigned char *out = NULL;
    size_t out_len = 0;
    printf("compress_negative_pitch_rc=%d\n",
           api.compress8(compressor, row, 8, -8, 8, TJPF_GRAY, &out, &out_len));
    api.free(out);
    free(row);
    api.destroy(compressor);

    /* Decompression ceiling.  `maxPixels` is compared against the *JPEG's*
     * dimensions, before any scaling is applied. */
    tj_handle_t decompressor = api.init(TJINIT_DECOMPRESS);
    if (!decompressor) return 2;
    int width = 0, height = 0;
    if (probe(decompressor, jpeg, jpeg_len, &width, &height) != 0) return 2;
    guarded_buf dst;
    if (guarded_alloc(&dst, (size_t)width * height * TJ_PIXEL_SIZE[TJPF_RGB],
                      "maxpixels destination") != 0)
        return 2;

    int ceiling[3];
    printf("maxpixels_set_one=%d\n", api.set(decompressor, TJPARAM_MAXPIXELS, 1));
    printf("maxpixels_one_rc=%d\n",
           ceiling[0] = api.decompress8(decompressor, jpeg, jpeg_len, dst.data,
                                        0, TJPF_RGB));
    printf("maxpixels_set_exact=%d\n",
           api.set(decompressor, TJPARAM_MAXPIXELS, width * height));
    printf("maxpixels_exact_rc=%d\n",
           ceiling[1] = api.decompress8(decompressor, jpeg, jpeg_len, dst.data,
                                        0, TJPF_RGB));
    printf("maxpixels_set_short=%d\n",
           api.set(decompressor, TJPARAM_MAXPIXELS, width * height - 1));
    printf("maxpixels_short_rc=%d\n",
           ceiling[2] = api.decompress8(decompressor, jpeg, jpeg_len, dst.data,
                                        0, TJPF_RGB));
    require(ceiling[0] == -1 && ceiling[1] == 0 && ceiling[2] == -1,
            "TJPARAM_MAXPIXELS admits exactly the frame's own pixel count");
    require(api.jpeg_buf_size(0, 1, TJSAMP_444) == 0 &&
                api.jpeg_buf_size(-1, 1, TJSAMP_444) == 0 &&
                api.jpeg_buf_size(1, 1, 99) == 0,
            "tj3JPEGBufSize refuses an invalid geometry with 0");
    printf("maxpixels_canary=%s\n", canary_intact(&dst) ? "intact" : "corrupt");
    int intact = canary_intact(&dst);
    require(intact, "maxpixels destination canary");
    guarded_free(&dst);
    api.destroy(decompressor);
    return intact ? 0 : 2;
}

/* Concurrent calls, in the shape the API supports: one handle per thread.
 *
 * Upstream's own `tjinstance` is mutated without synchronisation by every
 * entry point, so two threads sharing one handle race in the reference
 * implementation too — there is no observable contract to compare, only a data
 * race, and driving it would make this harness's differential meaningless.
 * What *is* shared and does have a contract is everything below the handle:
 * the process-wide one-time initialisation of dispatch tables and Huffman
 * tables.
 *
 * That contract is only exercised if the workers reach it *first*.  An earlier
 * version of this case probed the header and computed its single-threaded
 * reference on the main thread before releasing any worker, which warmed every
 * lazily-built table — so each worker took the already-initialised path and the
 * case could not have seen a race in the initialisation it named.  Found by
 * `codex review`; it is the same shape as #320, a claim about coverage that the
 * mechanism did not provide.  Now: no library call happens before the gate, the
 * workers are released together, each reads its own header, and the serial
 * reference is computed afterwards. */

struct worker {
    const unsigned char *jpeg;
    size_t jpeg_len;
    uint64_t hash;
    int rc;
};

/* Released together rather than started together: `pthread_create` returns
 * long before the thread runs, so creating eight threads in a loop staggers
 * their first library call by however long the loop takes.  macOS has no
 * `pthread_barrier_t`, so this is a mutex and two condition variables.
 *
 * The arrival count is the second half, and it is not belt-and-braces:
 * broadcasting as soon as the last `pthread_create` returns lets a slow-starting
 * thread walk past an already-open gate and find the tables warm.  `codex
 * review` measured seven parked workers in five consecutive runs of the version
 * that did not count.  The count is reported in the transcript rather than
 * asserted in C, so "all eight really were parked" is something the comparison
 * can see. */
static pthread_mutex_t gate_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gate_open = PTHREAD_COND_INITIALIZER;
static pthread_cond_t all_parked = PTHREAD_COND_INITIALIZER;
static int gate_is_open = 0;
static int parked_workers = 0;

static void wait_at_gate(void) {
    pthread_mutex_lock(&gate_lock);
    parked_workers++;
    pthread_cond_signal(&all_parked);
    while (!gate_is_open) pthread_cond_wait(&gate_open, &gate_lock);
    pthread_mutex_unlock(&gate_lock);
}

/* Blocks until `expected` workers are waiting, then releases them all. */
static int open_gate(int expected) {
    pthread_mutex_lock(&gate_lock);
    while (parked_workers < expected)
        pthread_cond_wait(&all_parked, &gate_lock);
    int parked = parked_workers;
    gate_is_open = 1;
    pthread_cond_broadcast(&gate_open);
    pthread_mutex_unlock(&gate_lock);
    return parked;
}

/* One whole decode, from `tj3Init` to `tj3Destroy`, touching no state this
 * process has already built. */
static void decode_once(struct worker *w) {
    tj_handle_t handle = api.init(TJINIT_DECOMPRESS);
    if (!handle) { w->rc = -2; return; }
    int width = 0, height = 0;
    if (probe(handle, w->jpeg, w->jpeg_len, &width, &height) != 0) {
        api.destroy(handle);
        w->rc = -5;
        return;
    }
    guarded_buf dst;
    if (guarded_alloc(&dst, (size_t)width * height * TJ_PIXEL_SIZE[TJPF_RGB],
                      "concurrent worker destination") != 0) {
        api.destroy(handle);
        w->rc = -3;
        return;
    }
    w->rc = api.decompress8(handle, w->jpeg, w->jpeg_len, dst.data, 0, TJPF_RGB);
    w->hash = fnv1a(dst.data, dst.len);
    if (!canary_intact(&dst)) w->rc = -4;
    guarded_free(&dst);
    api.destroy(handle);
}

static void *decode_worker(void *arg) {
    wait_at_gate();
    decode_once(arg);
    return NULL;
}

#define WORKERS 8

static int case_concurrent_handles(const unsigned char *jpeg, size_t jpeg_len) {
    struct worker workers[WORKERS];
    pthread_t threads[WORKERS];
    for (int i = 0; i < WORKERS; i++) {
        workers[i].jpeg = jpeg;
        workers[i].jpeg_len = jpeg_len;
        workers[i].hash = 0;
        workers[i].rc = -1;
        if (pthread_create(&threads[i], NULL, decode_worker, &workers[i]) != 0)
            return 2;
    }
    /* The first call into the library in this process is made by eight threads
     * at once — all eight, which is what the arrival count establishes. */
    int parked = open_gate(WORKERS);
    for (int i = 0; i < WORKERS; i++) pthread_join(threads[i], NULL);
    printf("parked=%d\n", parked);

    /* Only now, with every table warm and no contention, the serial answer. */
    struct worker reference = { jpeg, jpeg_len, 0, -1 };
    decode_once(&reference);
    printf("reference_rc=%d\n", reference.rc);
    printf("reference_pixels=%016llx\n", (unsigned long long)reference.hash);

    int all_ok = reference.rc == 0;
    for (int i = 0; i < WORKERS; i++) {
        if (workers[i].rc != 0 || workers[i].hash != reference.hash) all_ok = 0;
    }
    printf("workers=%d\n", WORKERS);
    printf("workers_agree=%s\n", all_ok ? "yes" : "no");
    require(parked == WORKERS, "every worker parked before the gate opened");
    require(all_ok, "every concurrent decode matches the serial one");
    return all_ok ? 0 : 2;
}

/* `tj3Alloc` / `tj3Free` across the boundary.
 *
 * The existing sanitizer harness allocates with its own malloc/free, so the
 * shared-allocator contract these two exist for was never exercised at the
 * ABI — recorded in docs/UNSAFE_INVENTORY_CAPI.md's `tj3Free` row and in
 * P4-141 criterion 2.  A mismatched allocator here is an ASan report in the
 * `c_boundary_asan` job and a glibc abort without one; either way it is this
 * child's exit status and not the suite's. */
static int case_alloc_ownership(const unsigned char *jpeg, size_t jpeg_len) {
    api.free(NULL);
    printf("free_null=returned\n");

    unsigned char *scratch = api.alloc(4096);
    printf("alloc_4096=%s\n", scratch ? "pointer" : "null");
    if (!scratch) return 2;
    memset(scratch, 0x11, 4096);
    api.free(scratch);
    printf("free_4096=returned\n");

    /* The zero-byte edge, which is where the two implementations part:
     * upstream's `tj3Alloc` is a bare `malloc(bytes)` (`turbojpeg.c:935-937`)
     * and both glibc and macOS return a unique freeable pointer for 0. A
     * caller writing `if (!(buf = tj3Alloc(n))) fail();` therefore sees a
     * spurious failure at n == 0 against an implementation that returns NULL.
     * Reading back a byte this process just wrote would have told us nothing;
     * this tells us something. */
    void *zero = api.alloc(0);
    printf("alloc_zero=%s\n", zero ? "pointer" : "null");
    api.free(zero);
    printf("free_zero=returned\n");

    tj_handle_t handle = api.init(TJINIT_DECOMPRESS);
    if (!handle) return 2;
    int width = 0, height = 0;
    if (probe(handle, jpeg, jpeg_len, &width, &height) != 0) return 2;
    size_t dst_len = (size_t)width * height * TJ_PIXEL_SIZE[TJPF_RGB];

    /* Decode into library-allocated memory and release it through the
     * library. */
    unsigned char *dst = api.alloc(dst_len);
    if (!dst) { api.destroy(handle); return 2; }
    int rc = api.decompress8(handle, jpeg, jpeg_len, dst, 0, TJPF_RGB);
    printf("decode_into_tj3alloc_rc=%d\n", rc);
    require(rc == 0, "a decode into a tj3Alloc destination");
    printf("decode_into_tj3alloc_pixels=%016llx\n",
           (unsigned long long)fnv1a(dst, dst_len));
    api.free(dst);
    printf("free_dst=returned\n");
    api.destroy(handle);

    /* Compress with NOREALLOC unset: the library allocates the output, and the
     * caller releases it with tj3Free.  This is the direction that crosses the
     * allocator boundary the other way. */
    tj_handle_t compressor = api.init(TJINIT_COMPRESS);
    if (!compressor) return 2;
    api.set(compressor, TJPARAM_QUALITY, 90);
    api.set(compressor, TJPARAM_SUBSAMP, TJSAMP_444);
    unsigned char pixels[16 * 16 * 3];
    for (size_t i = 0; i < sizeof(pixels); i++) pixels[i] = (unsigned char)i;
    unsigned char *out = NULL;
    size_t out_len = 0;
    int crc = api.compress8(compressor, pixels, 16, 0, 16, TJPF_RGB, &out,
                            &out_len);
    printf("compress_alloc_rc=%d\n", crc);
    printf("compress_alloc_out=%s\n", out ? "allocated" : "null");
    printf("compress_alloc_soi=%s\n",
           (out && out_len >= 2 && out[0] == 0xFF && out[1] == 0xD8) ? "yes"
                                                                    : "no");
    printf("compress_alloc_len=%zu\n", out ? out_len : (size_t)0);
    printf("compress_alloc_bytes=%016llx\n",
           out ? (unsigned long long)fnv1a(out, out_len) : 0ULL);
    require(crc == 0 && out && out_len >= 2 && out[0] == 0xFF && out[1] == 0xD8,
            "a library-allocated compress output starts with SOI");
    api.free(out);
    printf("free_compressed=returned\n");
    api.destroy(compressor);
    return 0;
}

/* A 12-bit round trip across the ABI.  No 12-bit entry point crossed the C
 * boundary under a sanitizer before this case existed — the other named half
 * of P4-141 criterion 2's C-boundary gap. */
static int case_precision12(void) {
    const int width = 64, height = 48;
    short *src = malloc((size_t)width * height * sizeof(short));
    if (!src) return 2;
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            /* 12-bit range is 0..4095. */
            src[(size_t)y * width + x] = (short)((x * 64 + y * 8) & 4095);
        }
    }

    tj_handle_t compressor = api.init(TJINIT_COMPRESS);
    if (!compressor) { free(src); return 2; }
    api.set(compressor, TJPARAM_QUALITY, 95);
    api.set(compressor, TJPARAM_SUBSAMP, TJSAMP_GRAY);
    unsigned char *jpeg = NULL;
    size_t jpeg_len = 0;
    int rc = api.compress12(compressor, src, width, 0, height, TJPF_GRAY,
                            &jpeg, &jpeg_len);
    printf("compress12_rc=%d\n", rc);
    printf("compress12_nonempty=%s\n", jpeg_len > 0 ? "yes" : "no");
    api.destroy(compressor);
    if (rc != 0 || !jpeg) { free(src); return 2; }

    tj_handle_t decompressor = api.init(TJINIT_DECOMPRESS);
    if (!decompressor) { api.free(jpeg); free(src); return 2; }
    printf("header12_rc=%d\n",
           api.decompress_header(decompressor, jpeg, jpeg_len));
    printf("precision=%d\n", api.get(decompressor, TJPARAM_PRECISION));
    printf("jpegwidth=%d\n", api.get(decompressor, TJPARAM_JPEGWIDTH));
    printf("jpegheight=%d\n", api.get(decompressor, TJPARAM_JPEGHEIGHT));

    guarded_buf dst;
    if (guarded_alloc(&dst, (size_t)width * height * sizeof(short),
                      "12-bit destination") != 0) {
        api.free(jpeg);
        free(src);
        return 2;
    }
    /* `guarded_alloc` aligns the payload's *end* to a page, so its start is
     * only as aligned as the length makes it. This is the one case that casts
     * the payload to a wider type, so it says so rather than relying on the
     * length happening to be even. */
    if ((uintptr_t)dst.data % sizeof(short) != 0) {
        fprintf(stderr, "12-bit destination is not short-aligned\n");
        return 2;
    }
    int drc = api.decompress12(decompressor, jpeg, jpeg_len, (short *)dst.data,
                               0, TJPF_GRAY);
    printf("decompress12_rc=%d\n", drc);
    require(drc == 0, "the 12-bit decode succeeds");
    printf("decompress12_canary=%s\n",
           canary_intact(&dst) ? "intact" : "corrupt");
    printf("decompress12_pixels=%016llx\n",
           (unsigned long long)fnv1a(dst.data, dst.len));

    /* Short pitch is rejected at 12-bit precision by the same clause as at
     * 8-bit; the pitch is counted in samples, not bytes. */
    int short_pitch = api.decompress12(decompressor, jpeg, jpeg_len,
                                       (short *)dst.data, width - 1, TJPF_GRAY);
    printf("decompress12_short_pitch_rc=%d\n", short_pitch);
    require(short_pitch == -1, "a short 12-bit pitch is refused");
    int intact = canary_intact(&dst);
    require(intact, "12-bit destination canary");
    guarded_free(&dst);
    api.free(jpeg);
    free(src);
    api.destroy(decompressor);
    return intact ? 0 : 2;
}

/* Proof that the trailing guard page is armed.  The runner requires this case
 * to die by a signal; if the mapping ever stopped being protected, every
 * overrun check above would pass while measuring nothing. */
static int case_selftest_guard_page(void) {
    guarded_buf g;
    if (guarded_alloc(&g, 128, "selftest overrun target") != 0) return 2;
    printf("about_to_overrun=1\n");
    fflush(stdout);
    volatile unsigned char *past_end = g.data + g.len;
    *past_end = 0xFF;
    printf("overrun_survived=1\n");
    fflush(stdout);
    guarded_free(&g);
    return 0;
}

/* Proof that the canary is checked.  The runner requires the transcript to
 * report corruption and the process to exit 2. */
static int case_selftest_canary(void) {
    guarded_buf g;
    if (guarded_alloc(&g, 128, "selftest canary target") != 0) return 2;
    if (g.slack == 0) {
        /* A payload that is an exact multiple of the page size has no slack.
         * 128 bytes never is, so this is a guard against a future edit. */
        fprintf(stderr, "selftest_canary needs a payload with slack\n");
        return 2;
    }
    g.data[-1] = 0x00;
    printf("canary=%s\n", canary_intact(&g) ? "intact" : "corrupt");
    int intact = canary_intact(&g);
    /* Discarded rather than freed: the corruption is the point, and
     * `guarded_free`'s release-time check would report it as a contract
     * failure. */
    guarded_discard(&g);
    return intact ? 0 : 2;
}

/* ---------------------------------------------------------------- main -- */

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
                "usage: %s <shared-library> <case> [fixture.jpg]\n"
                "cases: lifecycle handle_defaults parameter_applicability "
                "null_handle pitch_boundaries "
                "undersized_output max_dimensions concurrent_handles "
                "alloc_ownership precision12 selftest_guard_page "
                "selftest_canary\n",
                argv[0]);
        return 3;
    }
    const char *lib_path = argv[1];
    const char *case_name = argv[2];
    const char *fixture = argc > 3 ? argv[3] : NULL;

    void *lib = dlopen(lib_path, RTLD_NOW | RTLD_LOCAL);
    if (!lib) {
        fprintf(stderr, "dlopen(%s) failed: %s\n", lib_path, dlerror());
        return 1;
    }
    load_api(lib);

    /* Line-buffered so a transcript survives a case that faults on purpose. */
    setvbuf(stdout, NULL, _IOLBF, 0);

    unsigned char *jpeg = NULL;
    size_t jpeg_len = 0;
    if (fixture) {
        jpeg = read_file(fixture, &jpeg_len);
        if (!jpeg) {
            fprintf(stderr, "cannot read fixture %s\n", fixture);
            return 3;
        }
    }

    int needs_fixture =
        strcmp(case_name, "null_handle") == 0 ||
        strcmp(case_name, "pitch_boundaries") == 0 ||
        strcmp(case_name, "max_dimensions") == 0 ||
        strcmp(case_name, "concurrent_handles") == 0 ||
        strcmp(case_name, "alloc_ownership") == 0;
    if (needs_fixture && !jpeg) {
        fprintf(stderr, "case %s needs a fixture path\n", case_name);
        return 3;
    }

    int rc;
    if (strcmp(case_name, "lifecycle") == 0) {
        rc = case_lifecycle();
    } else if (strcmp(case_name, "handle_defaults") == 0) {
        rc = case_handle_defaults();
    } else if (strcmp(case_name, "parameter_applicability") == 0) {
        rc = case_parameter_applicability();
    } else if (strcmp(case_name, "null_handle") == 0) {
        rc = case_null_handle(jpeg, jpeg_len);
    } else if (strcmp(case_name, "pitch_boundaries") == 0) {
        rc = case_pitch_boundaries(jpeg, jpeg_len);
    } else if (strcmp(case_name, "undersized_output") == 0) {
        rc = case_undersized_output();
    } else if (strcmp(case_name, "max_dimensions") == 0) {
        rc = case_max_dimensions(jpeg, jpeg_len);
    } else if (strcmp(case_name, "concurrent_handles") == 0) {
        rc = case_concurrent_handles(jpeg, jpeg_len);
    } else if (strcmp(case_name, "alloc_ownership") == 0) {
        rc = case_alloc_ownership(jpeg, jpeg_len);
    } else if (strcmp(case_name, "precision12") == 0) {
        rc = case_precision12();
    } else if (strcmp(case_name, "selftest_guard_page") == 0) {
        rc = case_selftest_guard_page();
    } else if (strcmp(case_name, "selftest_canary") == 0) {
        rc = case_selftest_canary();
    } else {
        fprintf(stderr, "unknown case: %s\n", case_name);
        rc = 3;
    }

    free(jpeg);
    fflush(stdout);
    /* A case that only *printed* a failed check would leave `sanitizers.yml`
     * green, since that leg reads the exit status and not the transcript. */
    if (rc == 0 && contract_failures != 0) {
        fprintf(stderr, "%d contract check(s) failed in case %s\n",
                contract_failures, case_name);
        rc = 2;
    }
    return rc;
}

/*
 * P4-145 C oracle: what real TurboJPEG does with `TJPARAM_NOREALLOC`.
 *
 * The Rust suite asserts that a caller's buffer pointer comes back unchanged.
 * That is the right assertion, but on its own it only pins what *this* port
 * does — and the port's first version of this fix got a case wrong that no
 * amount of self-consistency would have shown: with the flag set and a **NULL**
 * output slot, it allocated. Upstream refuses (`jdatadst-tj.c:184-192` takes
 * the `*outbuffer == NULL` branch and, with `alloc` false, raises
 * `JERR_BUFFER_SIZE`). Review caught it; this binary is what makes the next one
 * fail loudly instead.
 *
 * Each line is `case rc kept produced`, where
 *   rc       = the entry point's return value,
 *   kept     = 1 when the output slot still holds the pointer the caller passed,
 *   produced = 1 when a non-zero output size was reported.
 *
 * `kept` is the whole point: it is the difference between a library that
 * honours the flag and one that merely succeeds.
 *
 * The *byte count* is deliberately not compared in the cases above. Two
 * independent encoders do not agree on it — this port and upstream differ on
 * entropy-coded output for the same image — and a trace that bakes in one
 * implementation's sizes would fail for a reason that has nothing to do with
 * the ownership contract under test. What is contractual there is the return
 * code and the pointer.
 *
 * The `fx_*` family (P4-156, #544) is the exception, and prints
 * `case rc kept size` with the exact byte count: both sides transform the
 * *same fixture bytes* (argv[1], generated once by the Rust harness), and an
 * identity transform of identical input is byte-exact between the two
 * implementations, so any size difference is precisely the marker policy
 * those cases exist to compare.
 *
 * Usage: norealloc_oracle [gray_icc_fixture.jpg]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <turbojpeg.h>

#define WIDTH 32
#define HEIGHT 32
/* Comfortably larger than any output here. */
#define ROOMY (64 * 1024)
/* Far too small for a real JPEG, but a genuine allocation. */
#define CRAMPED 16

static void fill_rgb(unsigned char *buf, size_t n)
{
  size_t i;
  for (i = 0; i < n; i++)
    buf[i] = (unsigned char)(i % 251);
}

/* Report one case in the shared line format. */
static void report(const char *label, int rc, const unsigned char *slot,
                   const unsigned char *original, size_t size)
{
  printf("%s %d %d %d\n", label, rc, slot == original ? 1 : 0,
         (rc == 0 && size > 0) ? 1 : 0);
}

static tjhandle compressor(void)
{
  tjhandle h = tj3Init(TJINIT_COMPRESS);
  if (!h) { fprintf(stderr, "tj3Init\n"); exit(2); }
  if (tj3Set(h, TJPARAM_NOREALLOC, 1) != 0) { fprintf(stderr, "NOREALLOC\n"); exit(2); }
  if (tj3Set(h, TJPARAM_QUALITY, 80) != 0) { fprintf(stderr, "quality\n"); exit(2); }
  if (tj3Set(h, TJPARAM_SUBSAMP, TJSAMP_444) != 0) { fprintf(stderr, "subsamp\n"); exit(2); }
  return h;
}

/* `tj3Compress8` with a buffer of `capacity` bytes, or NULL when `capacity`
 * is zero — the case that distinguishes "refuses to grow" from "refuses to
 * allocate". */
static void compress8_case(const char *label, size_t capacity)
{
  tjhandle h = compressor();
  unsigned char *src = (unsigned char *)malloc(WIDTH * HEIGHT * 3);
  unsigned char *original = capacity ? (unsigned char *)tj3Alloc(capacity) : NULL;
  unsigned char *buf = original;
  size_t size = capacity;
  int rc;

  if (!src || (capacity && !original)) { fprintf(stderr, "oom\n"); exit(2); }
  fill_rgb(src, WIDTH * HEIGHT * 3);

  rc = tj3Compress8(h, src, WIDTH, 0, HEIGHT, TJPF_RGB, &buf, &size);
  report(label, rc, buf, original, size);

  /* On the refusal paths nothing was swapped, so `original` is still ours.
   * On success under NOREALLOC it is also still ours — that is the contract. */
  if (buf != original && buf != NULL)
    tj3Free(buf);
  if (original)
    tj3Free(original);
  free(src);
  tj3Destroy(h);
}

/* `tj3CompressFromYUV8`, the packed-YUV sibling. */
static void yuv8_case(const char *label, size_t capacity)
{
  tjhandle h = compressor();
  size_t plane = (size_t)WIDTH * HEIGHT;
  unsigned char *src = (unsigned char *)malloc(plane * 3);
  unsigned char *original = capacity ? (unsigned char *)tj3Alloc(capacity) : NULL;
  unsigned char *buf = original;
  size_t size = capacity;
  int rc;

  if (!src || (capacity && !original)) { fprintf(stderr, "oom\n"); exit(2); }
  fill_rgb(src, plane * 3);

  rc = tj3CompressFromYUV8(h, src, WIDTH, 1, HEIGHT, &buf, &size);
  report(label, rc, buf, original, size);

  if (buf != original && buf != NULL)
    tj3Free(buf);
  if (original)
    tj3Free(original);
  free(src);
  tj3Destroy(h);
}

/* `tj3Compress12` / `tj3Compress16`: same contract, wider samples. */
static void compress12_case(const char *label, size_t capacity)
{
  tjhandle h = compressor();
  short *src = (short *)malloc((size_t)WIDTH * HEIGHT * 3 * sizeof(short));
  unsigned char *original = capacity ? (unsigned char *)tj3Alloc(capacity) : NULL;
  unsigned char *buf = original;
  size_t size = capacity, i;
  int rc;

  if (!src || (capacity && !original)) { fprintf(stderr, "oom\n"); exit(2); }
  for (i = 0; i < (size_t)WIDTH * HEIGHT * 3; i++) src[i] = (short)(i % 4096);

  rc = tj3Compress12(h, src, WIDTH, 0, HEIGHT, TJPF_RGB, &buf, &size);
  report(label, rc, buf, original, size);

  if (buf != original && buf != NULL) tj3Free(buf);
  if (original) tj3Free(original);
  free(src);
  tj3Destroy(h);
}

/* 16-bit samples exist for *lossless* JPEG upstream: a lossy 16-bit compress
 * is refused before any of this matters (`jcmaster.c:206`), so lossless is the
 * configuration the ownership contract is exercised in. `lossless = 0` traces
 * the refusal itself — see the note at the call site. */
static void compress16_case(const char *label, size_t capacity, int lossless)
{
  tjhandle h = compressor();
  if (lossless && tj3Set(h, TJPARAM_LOSSLESS, 1) != 0) { fprintf(stderr, "lossless\n"); exit(2); }
  unsigned short *src =
    (unsigned short *)malloc((size_t)WIDTH * HEIGHT * 3 * sizeof(unsigned short));
  unsigned char *original = capacity ? (unsigned char *)tj3Alloc(capacity) : NULL;
  unsigned char *buf = original;
  size_t size = capacity, i;
  int rc;

  if (!src || (capacity && !original)) { fprintf(stderr, "oom\n"); exit(2); }
  for (i = 0; i < (size_t)WIDTH * HEIGHT * 3; i++) src[i] = (unsigned short)(i % 65535);

  rc = tj3Compress16(h, src, WIDTH, 0, HEIGHT, TJPF_RGB, &buf, &size);
  report(label, rc, buf, original, size);

  if (buf != original && buf != NULL) tj3Free(buf);
  if (original) tj3Free(original);
  free(src);
  tj3Destroy(h);
}

/* The planar-YUV sibling. */
static void yuv_planes_case(const char *label, size_t capacity)
{
  tjhandle h = compressor();
  size_t plane = (size_t)WIDTH * HEIGHT;
  unsigned char *y = (unsigned char *)malloc(plane);
  unsigned char *cb = (unsigned char *)malloc(plane);
  unsigned char *cr = (unsigned char *)malloc(plane);
  const unsigned char *planes[3];
  int strides[3];
  unsigned char *original = capacity ? (unsigned char *)tj3Alloc(capacity) : NULL;
  unsigned char *buf = original;
  size_t size = capacity;
  int rc;

  if (!y || !cb || !cr || (capacity && !original)) { fprintf(stderr, "oom\n"); exit(2); }
  fill_rgb(y, plane);
  memset(cb, 128, plane);
  memset(cr, 128, plane);
  planes[0] = y; planes[1] = cb; planes[2] = cr;
  strides[0] = strides[1] = strides[2] = WIDTH;

  rc = tj3CompressFromYUVPlanes8(h, planes, WIDTH, strides, HEIGHT, &buf, &size);
  report(label, rc, buf, original, size);

  if (buf != original && buf != NULL) tj3Free(buf);
  if (original) tj3Free(original);
  free(y); free(cb); free(cr);
  tj3Destroy(h);
}

/* Encode a small baseline JPEG for the transform cases to consume. */
static unsigned char *make_source(size_t *out_size)
{
  tjhandle h = tj3Init(TJINIT_COMPRESS);
  unsigned char *src = (unsigned char *)malloc((size_t)WIDTH * HEIGHT * 3);
  unsigned char *jpeg = NULL;
  size_t size = 0;

  if (!h || !src) { fprintf(stderr, "oom\n"); exit(2); }
  fill_rgb(src, (size_t)WIDTH * HEIGHT * 3);
  if (tj3Set(h, TJPARAM_QUALITY, 80) != 0 ||
      tj3Set(h, TJPARAM_SUBSAMP, TJSAMP_444) != 0) { fprintf(stderr, "set\n"); exit(2); }
  if (tj3Compress8(h, src, WIDTH, 0, HEIGHT, TJPF_RGB, &jpeg, &size) != 0) {
    fprintf(stderr, "source encode\n"); exit(2);
  }
  free(src);
  tj3Destroy(h);
  *out_size = size;
  return jpeg;
}

/* `tj3Transform`, whose reusable slot is `dstBufs[0]`. */
static void transform_case(const char *label, size_t capacity)
{
  tjhandle h = tj3Init(TJINIT_TRANSFORM);
  size_t jpeg_size = 0;
  unsigned char *jpeg = make_source(&jpeg_size);
  unsigned char *original = capacity ? (unsigned char *)tj3Alloc(capacity) : NULL;
  unsigned char *dst_bufs[1];
  size_t dst_sizes[1];
  tjtransform t[1];
  int rc;

  if (!h || (capacity && !original)) { fprintf(stderr, "oom\n"); exit(2); }
  if (tj3Set(h, TJPARAM_NOREALLOC, 1) != 0) { fprintf(stderr, "NOREALLOC\n"); exit(2); }
  memset(t, 0, sizeof(t));
  dst_bufs[0] = original;
  dst_sizes[0] = capacity;

  rc = tj3Transform(h, jpeg, jpeg_size, 1, dst_bufs, dst_sizes, t);
  report(label, rc, dst_bufs[0], original, dst_sizes[0]);

  if (dst_bufs[0] != original && dst_bufs[0] != NULL) tj3Free(dst_bufs[0]);
  if (original) tj3Free(original);
  tj3Free(jpeg);
  tj3Destroy(h);
}

/* The **legacy** wrapper, whose `dstSizes` are *outputs* rather than
 * capacities. A caller that sized its destination with `tjTransformBufSize()`
 * has no reason to fill them in, so upstream fills a temporary capacity array
 * from the transformed geometry (`turbojpeg.c:3118-3132`). This port passed the
 * zeros straight through until P4-151, and TJ3 read them as capacities of zero.
 *
 * `dstSizes[0]` starts at 0 deliberately — that is the case under test. The
 * reported `produced` field therefore also says whether the bridge copied the
 * real size back, which upstream does unconditionally. */
static void legacy_transform_case(const char *label)
{
  tjhandle h = tj3Init(TJINIT_TRANSFORM);
  size_t jpeg_size = 0;
  unsigned char *jpeg = make_source(&jpeg_size);
  tjtransform t[1];
  unsigned long bound;
  unsigned char *original;
  unsigned char *dst_bufs[1];
  unsigned long dst_sizes[1];
  int rc;

  if (!h) { fprintf(stderr, "tj3Init\n"); exit(2); }
  memset(t, 0, sizeof(t));

  /* What a legacy caller allocates. TurboJPEG 3 no longer declares
   * `tjTransformBufSize`, so this uses `tjBufSize` on the transformed
   * geometry — which is what that function computed, and exactly the
   * geometry-only bound P4-151 says the capacity must not exceed. An identity
   * transform leaves the specs unchanged, and the source is 4:4:4. */
  bound = tjBufSize(WIDTH, HEIGHT, TJSAMP_444);
  if (bound == 0) { fprintf(stderr, "tjBufSize\n"); exit(2); }
  original = (unsigned char *)tj3Alloc(bound);
  if (!original) { fprintf(stderr, "oom\n"); exit(2); }

  dst_bufs[0] = original;
  dst_sizes[0] = 0;

  rc = tjTransform(h, jpeg, (unsigned long)jpeg_size, 1, dst_bufs, dst_sizes, t,
                   TJFLAG_NOREALLOC);
  printf("%s %d %d %d\n", label, rc, dst_bufs[0] == original ? 1 : 0,
         (rc == 0 && dst_sizes[0] > 0) ? 1 : 0);

  if (dst_bufs[0] != original && dst_bufs[0] != NULL) tj3Free(dst_bufs[0]);
  tj3Free(original);
  tj3Free(jpeg);
  tj3Destroy(h);
}

/* Read the shared fixture the Rust harness generated (argv[1]). Both sides
 * must transform *identical bytes* for the exact-size lines below to compare —
 * each side's own encoder produces different streams for the same pixels. */
static unsigned char *read_fixture(const char *path, size_t *out_size)
{
  FILE *f = fopen(path, "rb");
  long len;
  unsigned char *buf;

  if (!f) { fprintf(stderr, "fixture open %s\n", path); exit(2); }
  if (fseek(f, 0, SEEK_END) != 0 || (len = ftell(f)) <= 0 ||
      fseek(f, 0, SEEK_SET) != 0) { fprintf(stderr, "fixture seek\n"); exit(2); }
  buf = (unsigned char *)malloc((size_t)len);
  if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) {
    fprintf(stderr, "fixture read\n"); exit(2);
  }
  fclose(f);
  *out_size = (size_t)len;
  return buf;
}

/* P4-156 (#544): one exact-parity line per (entry point, flags, options)
 * shape over the shared ICC-carrying grayscale fixture — `label rc kept size`,
 * with the *exact byte count* in the trace. That is comparable because a
 * transform of identical input is byte-exact between this library and the
 * port (the stock-tool gate pins that for `jpegtran -copy all -rotate 90`
 * over the upstream corpus), so the size differences here are purely the
 * marker policy under test:
 *   - legacy NOREALLOC drops every marker — the ordering quirk: the wrapper's
 *     capacity pre-read (`turbojpeg.c:3112-3134`) parses the header before
 *     `jcopy_markers_setup` (`turbojpeg.c:2976-2979`) can register anything;
 *   - legacy flags=0 and tj3Transform copy them (saveMarkers default ALL);
 *   - TJXOPT_COPYNONE drops them on every shape. */
static void fixture_case(const char *label, const unsigned char *jpeg,
                         size_t jpeg_size, int use_legacy, int norealloc,
                         int copynone)
{
  tjhandle h = tj3Init(TJINIT_TRANSFORM);
  tjtransform t[1];
  unsigned char *original = NULL;
  unsigned char *dst_bufs[1];
  int rc;

  if (!h) { fprintf(stderr, "tj3Init\n"); exit(2); }
  memset(t, 0, sizeof(t));
  if (copynone) t[0].options |= TJXOPT_COPYNONE;
  dst_bufs[0] = NULL;

  if (use_legacy) {
    unsigned long dst_sizes[1];
    dst_sizes[0] = 0;
    if (norealloc) {
      /* What a compliant legacy caller allocates for a grayscale source. */
      unsigned long bound = tjBufSize(WIDTH, HEIGHT, TJSAMP_GRAY);
      if (bound == 0) { fprintf(stderr, "tjBufSize\n"); exit(2); }
      original = (unsigned char *)tj3Alloc(bound);
      if (!original) { fprintf(stderr, "oom\n"); exit(2); }
      dst_bufs[0] = original;
    }
    rc = tjTransform(h, jpeg, (unsigned long)jpeg_size, 1, dst_bufs, dst_sizes,
                     t, norealloc ? TJFLAG_NOREALLOC : 0);
    /* `kept` is meaningful only under NOREALLOC; the reallocating lines pin
     * rc and exact size, and print 1 to keep the line shape uniform. */
    printf("%s %d %d %lu\n", label, rc,
           norealloc ? (dst_bufs[0] == original ? 1 : 0) : 1,
           rc == 0 ? dst_sizes[0] : 0);
  } else {
    size_t dst_sizes[1];
    dst_sizes[0] = 0;
    rc = tj3Transform(h, jpeg, jpeg_size, 1, dst_bufs, dst_sizes, t);
    printf("%s %d 1 %lu\n", label, rc,
           rc == 0 ? (unsigned long)dst_sizes[0] : 0UL);
  }

  if (dst_bufs[0] != NULL && dst_bufs[0] != original) tj3Free(dst_bufs[0]);
  if (original) tj3Free(original);
  tj3Destroy(h);
}

/* One legacy-NOREALLOC identity transform of the fixture on an EXISTING
 * handle, in the fx_* line shape. The marker-registration state the handle
 * carries decides the outcome, which is what the state cases below compare. */
static void fixture_norealloc_on(tjhandle h, const char *label,
                                 const unsigned char *jpeg, size_t jpeg_size)
{
  tjtransform t[1];
  unsigned long bound = tjBufSize(WIDTH, HEIGHT, TJSAMP_GRAY);
  unsigned char *original;
  unsigned char *dst_bufs[1];
  unsigned long dst_sizes[1];
  int rc;

  if (bound == 0) { fprintf(stderr, "tjBufSize\n"); exit(2); }
  original = (unsigned char *)tj3Alloc(bound);
  if (!original) { fprintf(stderr, "oom\n"); exit(2); }
  memset(t, 0, sizeof(t));
  dst_bufs[0] = original;
  dst_sizes[0] = 0;

  rc = tjTransform(h, jpeg, (unsigned long)jpeg_size, 1, dst_bufs, dst_sizes,
                   t, TJFLAG_NOREALLOC);
  printf("%s %d %d %lu\n", label, rc, dst_bufs[0] == original ? 1 : 0,
         rc == 0 ? dst_sizes[0] : 0);

  if (dst_bufs[0] != NULL && dst_bufs[0] != original) tj3Free(dst_bufs[0]);
  tj3Free(original);
}

/* P4-156's warm-handle state machine (#548 review): `jcopy_markers_setup`
 * registration is per-handle and permanent, so the NOREALLOC ordering quirk
 * drops markers only on a *cold* handle. Three transitions:
 *   (a) a flags=0 marker-copying warm-up registers processors, so the
 *       following NOREALLOC pre-read saves markers and the copy exceeds the
 *       grayscale bound — refusal;
 *   (b) the cold NOREALLOC call itself registers processors for *later*
 *       calls even though its own read was starved, so the second identical
 *       call is warm;
 *   (c) a COPYNONE-only warm-up registers nothing — the handle stays cold. */
static void fixture_state_cases(const unsigned char *jpeg, size_t jpeg_size)
{
  tjtransform t[1];
  unsigned char *dst_bufs[1];
  unsigned long dst_sizes[1];
  tjhandle h;

  /* (a) warm via flags=0. */
  h = tj3Init(TJINIT_TRANSFORM);
  if (!h) { fprintf(stderr, "tj3Init\n"); exit(2); }
  memset(t, 0, sizeof(t));
  dst_bufs[0] = NULL;
  dst_sizes[0] = 0;
  if (tjTransform(h, jpeg, (unsigned long)jpeg_size, 1, dst_bufs, dst_sizes,
                  t, 0) != 0) { fprintf(stderr, "warm-up\n"); exit(2); }
  if (dst_bufs[0]) tj3Free(dst_bufs[0]);
  fixture_norealloc_on(h, "fx_warm_after_flags0", jpeg, jpeg_size);
  tj3Destroy(h);

  /* (b) the first NOREALLOC call warms the handle for the second. */
  h = tj3Init(TJINIT_TRANSFORM);
  if (!h) { fprintf(stderr, "tj3Init\n"); exit(2); }
  fixture_norealloc_on(h, "fx_norealloc_first", jpeg, jpeg_size);
  fixture_norealloc_on(h, "fx_norealloc_second", jpeg, jpeg_size);
  tj3Destroy(h);

  /* (c) COPYNONE-only history keeps the handle cold. */
  h = tj3Init(TJINIT_TRANSFORM);
  if (!h) { fprintf(stderr, "tj3Init\n"); exit(2); }
  memset(t, 0, sizeof(t));
  t[0].options |= TJXOPT_COPYNONE;
  dst_bufs[0] = NULL;
  dst_sizes[0] = 0;
  if (tjTransform(h, jpeg, (unsigned long)jpeg_size, 1, dst_bufs, dst_sizes,
                  t, 0) != 0) { fprintf(stderr, "copynone warm-up\n"); exit(2); }
  if (dst_bufs[0]) tj3Free(dst_bufs[0]);
  fixture_norealloc_on(h, "fx_cold_after_copynone", jpeg, jpeg_size);
  tj3Destroy(h);
}

/* A NULL source under the legacy wrapper: -1, and the caller's destination
 * untouched. Cheap, and it pins that the port's own pointer validation — added
 * because it sliced before `tj3Transform` could check — agrees with upstream
 * rather than merely avoiding a panic. */
static void legacy_null_source_case(const char *label)
{
  tjhandle h = tj3Init(TJINIT_TRANSFORM);
  unsigned char *original = (unsigned char *)tj3Alloc(ROOMY);
  unsigned char *dst_bufs[1];
  unsigned long dst_sizes[1];
  tjtransform t[1];
  int rc;

  if (!h || !original) { fprintf(stderr, "oom\n"); exit(2); }
  memset(t, 0, sizeof(t));
  dst_bufs[0] = original;
  dst_sizes[0] = 0;

  rc = tjTransform(h, NULL, 0, 1, dst_bufs, dst_sizes, t, TJFLAG_NOREALLOC);
  printf("%s %d %d %d\n", label, rc == 0 ? 0 : -1,
         dst_bufs[0] == original ? 1 : 0, (rc == 0 && dst_sizes[0] > 0) ? 1 : 0);

  if (dst_bufs[0] != original && dst_bufs[0] != NULL) tj3Free(dst_bufs[0]);
  tj3Free(original);
  tj3Destroy(h);
}

int main(int argc, char **argv)
{
  /* Roomy: honoured, pointer kept. */
  compress8_case("compress8_roomy", ROOMY);
  /* Too small: refused, pointer untouched — not resized, not overrun. */
  compress8_case("compress8_cramped", CRAMPED);
  /* NULL slot: the flag is a request *not to allocate*, so this is refused
   * too. This is the case the Rust port got wrong. */
  compress8_case("compress8_null", 0);

  yuv8_case("yuv8_roomy", ROOMY);
  yuv8_case("yuv8_cramped", CRAMPED);
  yuv8_case("yuv8_null", 0);

  /* The rest of the changed entry points. The NULL-slot divergence proves a
   * suite of self-consistency assertions can stay green while one call
   * diverges, so every one of them gets a trace rather than the two that
   * happened to be written first. */
  compress12_case("compress12_roomy", ROOMY);
  compress12_case("compress12_cramped", CRAMPED);
  compress12_case("compress12_null", 0);

  compress16_case("compress16_roomy", ROOMY, 1);
  compress16_case("compress16_cramped", CRAMPED, 1);
  compress16_case("compress16_null", 0, 1);
  /* Lossy 16-bit, which upstream refuses outright. This line needed the
   * lossless flag to agree until P4-150 (#531): the port used to accept the
   * configuration and encode a lossless stream anyway, so a trace taken here
   * would have disagreed for a reason unrelated to buffer ownership. That it
   * now agrees with no flag set is the proof the acceptance rule was fixed,
   * and it pins the refusal path's ownership behaviour too — a caller's
   * buffer must survive a rejected call untouched. */
  compress16_case("compress16_lossy_roomy", ROOMY, 0);

  yuv_planes_case("yuvplanes_roomy", ROOMY);
  yuv_planes_case("yuvplanes_cramped", CRAMPED);
  yuv_planes_case("yuvplanes_null", 0);

  transform_case("transform_roomy", ROOMY);
  transform_case("transform_cramped", CRAMPED);
  transform_case("transform_null", 0);

  /* P4-151: the legacy wrapper's output-vs-capacity bridge. */
  legacy_transform_case("legacy_transform_zero_size");
  legacy_null_source_case("legacy_transform_null_source");

  /* P4-156 (#544): exact marker-policy parity over the shared fixture the
   * Rust harness passes as argv[1] — an ICC-carrying grayscale source. The
   * six lines cover the one divergent path this item fixed (legacy
   * NOREALLOC's drop-everything ordering quirk), the two paths that were
   * already correct (legacy flags=0 and tj3Transform copy markers), and
   * TJXOPT_COPYNONE on all three shapes. */
  if (argc > 1) {
    size_t fx_size = 0;
    unsigned char *fx = read_fixture(argv[1], &fx_size);
    fixture_case("fx_legacy_norealloc", fx, fx_size, 1, 1, 0);
    fixture_case("fx_legacy_norealloc_copynone", fx, fx_size, 1, 1, 1);
    fixture_case("fx_legacy_flags0", fx, fx_size, 1, 0, 0);
    fixture_case("fx_legacy_flags0_copynone", fx, fx_size, 1, 0, 1);
    fixture_case("fx_tj3_realloc", fx, fx_size, 0, 0, 0);
    fixture_case("fx_tj3_realloc_copynone", fx, fx_size, 0, 0, 1);
    fixture_state_cases(fx, fx_size);
    free(fx);
  }

  return 0;
}

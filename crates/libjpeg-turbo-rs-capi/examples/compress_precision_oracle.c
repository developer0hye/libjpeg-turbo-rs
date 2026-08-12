/*
 * P4-150 C oracle: which sample precisions upstream will actually compress.
 *
 * The port treated 16-bit as "always lossless" and quietly encoded a lossless
 * stream whatever the caller asked for. Upstream does something different and
 * more useful: `turbojpeg-mp.c:107-115` sets `cinfo->data_precision` to 16 and
 * only overrides it from `TJPARAM_PRECISION` when `TJPARAM_LOSSLESS` is set;
 * `jpeg_start_compress` then reaches `jcmaster.c:199-208`, where a *lossy*
 * compress accepts precision 8 or 12 and nothing else. A lossy 16-bit call is
 * refused before a byte is written.
 *
 * That is worth an oracle rather than a transcription for two reasons. First,
 * the rule lives one layer below TurboJPEG — reading `turbojpeg-mp.c` alone
 * suggests no such check exists. Second, the interesting cases are the ones
 * where a plausible reading of the source gives the wrong answer: setting
 * `TJPARAM_PRECISION` to 12 does *not* rescue a lossy 16-bit call, because the
 * override is gated on the lossless flag; and the rule does not generalise to
 * `tj3Compress12`, whose lossy path is legal. Both are traced below.
 *
 * The trace also covers *where in the refusal chain* the check sits, which a
 * gate placed naively last gets wrong. Upstream validates the lossless
 * parameters, then installs the destination, then starts the compress
 * (`turbojpeg-mp.c:117-121`), and each stage can refuse — so the three
 * `c16_pt_*`, `c16_*_norealloc_*` and `c16_lossy*` groups disagree with each
 * other about which error a caller sees. No single ordering satisfies them all
 * by accident.
 *
 * Output is `label rc kind=<k> err="<message>"`, one line per case. `kind`
 * names the rule that refused; the message is compared only for the precision
 * refusal, which is libjpeg's own `JERR_BAD_PRECISION` text copied verbatim by
 * `CATCH_LIBJPEG` without TurboJPEG's usual `function():` prefix. A caller that
 * matches on that string is matching on something documented, so the port owes
 * it exactly. The other messages are classified instead: they are TurboJPEG's
 * own, carrying per-call detail this port words differently on purpose.
 *
 * Usage: compress_precision_oracle
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <turbojpeg.h>

#define WIDTH 32
#define HEIGHT 32

/* Which rule refused the call, as opposed to how it phrased it.
 *
 * Two of the messages compared here are libjpeg's, reaching `errStr` through
 * `CATCH_LIBJPEG` verbatim, so the port can and does match them byte for byte.
 * The *precedence* between them is a separate contract — which check runs
 * first — and pinning it by raw text would drag in messages the two libraries
 * have never agreed on (TurboJPEG's own `function():` errors carry per-call
 * detail this port words differently, deliberately: see the note in
 * `norealloc_oracle.c` about not comparing byte counts). Classifying keeps the
 * ordering assertion honest without pretending unrelated strings match. */
static const char *error_kind(const char *message)
{
  if (strstr(message, "data precision")) return "precision";
  if (strstr(message, "too small")) return "buffer";
  /* Upstream words this "Invalid progressive/lossless parameters Ss=.. Al=..";
   * the port names the two TJ parameters the caller actually set. Same rule,
   * different phrasing, so it is classified rather than compared. */
  if (strstr(message, "lossless parameters") || strstr(message, "LOSSLESSPT"))
    return "lossless_params";
  /* P4-155 (#539): upstream's "must be specified" gates. Classified — the
   * substring is the documented part; the function-name prefix is not. */
  if (strstr(message, "TJPARAM_QUALITY must be specified")) return "quality_unset";
  if (strstr(message, "TJPARAM_SUBSAMP must be specified")) return "subsamp_unset";
  return "other";
}

/* Report one case. `rc` of 0 carries no message, so both fields are empty then
 * — printing a stale `errStr` would compare state neither library promises.
 *
 * `err` is printed only for the precision refusal, the one message the port
 * owes byte for byte. For every other outcome the kind is the contract. */
static void report(const char *label, tjhandle handle, int rc)
{
  const char *message = rc == 0 ? "" : tj3GetErrorStr(handle);
  const char *kind = rc == 0 ? "none" : error_kind(message);
  int exact = strcmp(kind, "precision") == 0;

  printf("%s %d kind=%s err=\"%s\"\n", label, rc, kind, exact ? message : "");
}

/* A compressor with the parameters every case shares. `precision` of 0 means
 * "leave TJPARAM_PRECISION alone", and `point_transform` of -1 means the same
 * for TJPARAM_LOSSLESSPT; both are the common case. */
static tjhandle compressor(int lossless, int precision, int point_transform)
{
  tjhandle handle = tj3Init(TJINIT_COMPRESS);

  if (!handle) { fprintf(stderr, "tj3Init\n"); exit(2); }
  if (tj3Set(handle, TJPARAM_QUALITY, 80) != 0 ||
      tj3Set(handle, TJPARAM_SUBSAMP, TJSAMP_444) != 0) {
    fprintf(stderr, "quality/subsamp\n"); exit(2);
  }
  if (lossless && tj3Set(handle, TJPARAM_LOSSLESS, 1) != 0) {
    fprintf(stderr, "lossless\n"); exit(2);
  }
  if (precision && tj3Set(handle, TJPARAM_PRECISION, precision) != 0) {
    fprintf(stderr, "precision\n"); exit(2);
  }
  if (point_transform >= 0 && tj3Set(handle, TJPARAM_LOSSLESSPT, point_transform) != 0) {
    fprintf(stderr, "pt\n"); exit(2);
  }
  return handle;
}

/* `tj3Compress16` under one configuration.
 *
 * `capacity` of 0 leaves the library to allocate, which is the plain case.
 * A non-zero value pre-allocates and sets `TJPARAM_NOREALLOC`, which is what
 * makes the *precedence* between the destination's error and the precision
 * rule observable: upstream installs the destination before
 * `jpeg_start_compress` (`turbojpeg-mp.c:118-120`). `capacity` of `NULL_SLOT`
 * sets the flag but leaves the slot empty. */
#define NULL_SLOT ((size_t)-1)

static void compress16_case(const char *label, int lossless, int precision,
                            size_t capacity, int point_transform)
{
  tjhandle handle = compressor(lossless, precision, point_transform);
  unsigned short *src =
    (unsigned short *)malloc((size_t)WIDTH * HEIGHT * 3 * sizeof(unsigned short));
  unsigned char *buf = NULL;
  size_t size = 0, i;
  int rc;

  if (!src) { fprintf(stderr, "oom\n"); exit(2); }
  for (i = 0; i < (size_t)WIDTH * HEIGHT * 3; i++)
    src[i] = (unsigned short)(i % 65535);

  if (capacity != 0) {
    if (tj3Set(handle, TJPARAM_NOREALLOC, 1) != 0) {
      fprintf(stderr, "norealloc\n"); exit(2);
    }
    if (capacity != NULL_SLOT) {
      buf = (unsigned char *)tj3Alloc(capacity);
      if (!buf) { fprintf(stderr, "oom\n"); exit(2); }
      size = capacity;
    }
  }

  rc = tj3Compress16(handle, src, WIDTH, 0, HEIGHT, TJPF_RGB, &buf, &size);
  report(label, handle, rc);

  if (buf) tj3Free(buf);
  free(src);
  tj3Destroy(handle);
}

/* `tj3Compress12`, to answer whether the rule generalises rather than assume
 * it. Twelve-bit lossy is one of the two precisions `jcmaster.c:206` admits,
 * so this is expected to succeed — but "expected" is what the oracle is for. */
static void compress12_case(const char *label, int lossless)
{
  tjhandle handle = compressor(lossless, 0, -1);
  short *src = (short *)malloc((size_t)WIDTH * HEIGHT * 3 * sizeof(short));
  unsigned char *buf = NULL;
  size_t size = 0, i;
  int rc;

  if (!src) { fprintf(stderr, "oom\n"); exit(2); }
  for (i = 0; i < (size_t)WIDTH * HEIGHT * 3; i++)
    src[i] = (short)(i % 4096);

  rc = tj3Compress12(handle, src, WIDTH, 0, HEIGHT, TJPF_RGB, &buf, &size);
  report(label, handle, rc);

  if (buf) tj3Free(buf);
  free(src);
  tj3Destroy(handle);
}

/* `tj3Compress8`, the baseline: 8-bit lossy is the ordinary path, and its
 * presence keeps a bug that refused *everything* from looking like a pass. */
static void compress8_case(const char *label, int lossless)
{
  tjhandle handle = compressor(lossless, 0, -1);
  unsigned char *src = (unsigned char *)malloc((size_t)WIDTH * HEIGHT * 3);
  unsigned char *buf = NULL;
  size_t size = 0, i;
  int rc;

  if (!src) { fprintf(stderr, "oom\n"); exit(2); }
  for (i = 0; i < (size_t)WIDTH * HEIGHT * 3; i++)
    src[i] = (unsigned char)(i % 251);

  rc = tj3Compress8(handle, src, WIDTH, 0, HEIGHT, TJPF_RGB, &buf, &size);
  report(label, handle, rc);

  if (buf) tj3Free(buf);
  free(src);
  tj3Destroy(handle);
}

/* --- P4-155 (#539): the "must be specified" gates over unset parameters. ---
 *
 * A fresh handle carries TJPARAM_QUALITY = -1 and TJPARAM_SUBSAMP =
 * TJSAMP_UNKNOWN, and every *lossy* compress path refuses until the caller
 * supplies them (`turbojpeg-mp.c:95-98`); the YUV encode/decode paths gate on
 * SUBSAMP alone. The port defaulted to 75 / 4:2:0, so the branch was
 * unreachable and a caller who forgot a parameter silently got substitutes. */

static void fresh_get_case(const char *label)
{
  tjhandle handle = tj3Init(TJINIT_COMPRESS);
  if (!handle) { fprintf(stderr, "tj3Init\n"); exit(2); }
  printf("%s 0 kind=get err=\"quality=%d subsamp=%d\"\n", label,
         tj3Get(handle, TJPARAM_QUALITY), tj3Get(handle, TJPARAM_SUBSAMP));
  tj3Destroy(handle);
}

/* A handle with only the named parameters set. */
static tjhandle partial_compressor(int set_quality, int set_subsamp, int lossless)
{
  tjhandle handle = tj3Init(TJINIT_COMPRESS);
  if (!handle) { fprintf(stderr, "tj3Init\n"); exit(2); }
  if (set_quality && tj3Set(handle, TJPARAM_QUALITY, 80) != 0) {
    fprintf(stderr, "quality\n"); exit(2);
  }
  if (set_subsamp && tj3Set(handle, TJPARAM_SUBSAMP, TJSAMP_444) != 0) {
    fprintf(stderr, "subsamp\n"); exit(2);
  }
  if (lossless && tj3Set(handle, TJPARAM_LOSSLESS, 1) != 0) {
    fprintf(stderr, "lossless\n"); exit(2);
  }
  return handle;
}

static void unset8_case(const char *label, int set_quality, int set_subsamp,
                        int null_src)
{
  tjhandle handle = partial_compressor(set_quality, set_subsamp, 0);
  unsigned char *src = (unsigned char *)malloc((size_t)WIDTH * HEIGHT * 3);
  unsigned char *buf = NULL;
  size_t size = 0, i;
  int rc;

  if (!src) { fprintf(stderr, "oom\n"); exit(2); }
  for (i = 0; i < (size_t)WIDTH * HEIGHT * 3; i++)
    src[i] = (unsigned char)(i % 251);

  rc = tj3Compress8(handle, null_src ? NULL : src, WIDTH, 0, HEIGHT, TJPF_RGB,
                    &buf, &size);
  report(label, handle, rc);

  if (buf) tj3Free(buf);
  free(src);
  tj3Destroy(handle);
}

static void unset12_case(const char *label)
{
  tjhandle handle = partial_compressor(0, 0, 0);
  short *src = (short *)malloc((size_t)WIDTH * HEIGHT * 3 * sizeof(short));
  unsigned char *buf = NULL;
  size_t size = 0, i;
  int rc;

  if (!src) { fprintf(stderr, "oom\n"); exit(2); }
  for (i = 0; i < (size_t)WIDTH * HEIGHT * 3; i++)
    src[i] = (short)(i % 4096);
  rc = tj3Compress12(handle, src, WIDTH, 0, HEIGHT, TJPF_RGB, &buf, &size);
  report(label, handle, rc);
  if (buf) tj3Free(buf);
  free(src);
  tj3Destroy(handle);
}

/* A lossless compress consults neither parameter (`turbojpeg-mp.c:95-98`
 * gates on `!this->lossless`), so both may stay unset. */
static void unset16_lossless_case(const char *label)
{
  tjhandle handle = partial_compressor(0, 0, 1);
  unsigned short *src =
    (unsigned short *)malloc((size_t)WIDTH * HEIGHT * 3 * sizeof(unsigned short));
  unsigned char *buf = NULL;
  size_t size = 0, i;
  int rc;

  if (!src) { fprintf(stderr, "oom\n"); exit(2); }
  for (i = 0; i < (size_t)WIDTH * HEIGHT * 3; i++)
    src[i] = (unsigned short)(i % 65535);
  rc = tj3Compress16(handle, src, WIDTH, 0, HEIGHT, TJPF_RGB, &buf, &size);
  report(label, handle, rc);
  if (buf) tj3Free(buf);
  free(src);
  tj3Destroy(handle);
}

static void unset_encodeyuv8_case(const char *label)
{
  tjhandle handle = tj3Init(TJINIT_COMPRESS);
  unsigned char *src = (unsigned char *)malloc((size_t)WIDTH * HEIGHT * 3);
  /* Generous: large enough for any subsampling had the call proceeded. */
  unsigned char *dst = (unsigned char *)malloc((size_t)WIDTH * HEIGHT * 4);
  size_t i;
  int rc;

  if (!handle || !src || !dst) { fprintf(stderr, "oom\n"); exit(2); }
  for (i = 0; i < (size_t)WIDTH * HEIGHT * 3; i++)
    src[i] = (unsigned char)(i % 251);
  rc = tj3EncodeYUV8(handle, src, WIDTH, 0, HEIGHT, TJPF_RGB, dst, 1);
  report(label, handle, rc);
  free(src);
  free(dst);
  tj3Destroy(handle);
}

static void unset_fromyuvplanes8_case(const char *label)
{
  tjhandle handle = tj3Init(TJINIT_COMPRESS);
  unsigned char *y = (unsigned char *)malloc((size_t)WIDTH * HEIGHT);
  unsigned char *cb = (unsigned char *)malloc((size_t)WIDTH * HEIGHT);
  unsigned char *cr = (unsigned char *)malloc((size_t)WIDTH * HEIGHT);
  const unsigned char *planes[3];
  int strides[3] = { WIDTH, WIDTH, WIDTH };
  unsigned char *buf = NULL;
  size_t size = 0, i;
  int rc;

  if (!handle || !y || !cb || !cr) { fprintf(stderr, "oom\n"); exit(2); }
  for (i = 0; i < (size_t)WIDTH * HEIGHT; i++) y[i] = (unsigned char)(i % 251);
  memset(cb, 128, (size_t)WIDTH * HEIGHT);
  memset(cr, 128, (size_t)WIDTH * HEIGHT);
  planes[0] = y; planes[1] = cb; planes[2] = cr;
  rc = tj3CompressFromYUVPlanes8(handle, planes, WIDTH, strides, HEIGHT,
                                 &buf, &size);
  report(label, handle, rc);
  if (buf) tj3Free(buf);
  free(y); free(cb); free(cr);
  tj3Destroy(handle);
}

static void unset_decodeyuvplanes8_case(const char *label)
{
  tjhandle handle = tj3Init(TJINIT_DECOMPRESS);
  unsigned char *y = (unsigned char *)malloc((size_t)WIDTH * HEIGHT);
  unsigned char *cb = (unsigned char *)malloc((size_t)WIDTH * HEIGHT);
  unsigned char *cr = (unsigned char *)malloc((size_t)WIDTH * HEIGHT);
  const unsigned char *planes[3];
  int strides[3] = { WIDTH, WIDTH, WIDTH };
  unsigned char *dst = (unsigned char *)malloc((size_t)WIDTH * HEIGHT * 4);
  size_t i;
  int rc;

  if (!handle || !y || !cb || !cr || !dst) { fprintf(stderr, "oom\n"); exit(2); }
  for (i = 0; i < (size_t)WIDTH * HEIGHT; i++) y[i] = (unsigned char)(i % 251);
  memset(cb, 128, (size_t)WIDTH * HEIGHT);
  memset(cr, 128, (size_t)WIDTH * HEIGHT);
  planes[0] = y; planes[1] = cb; planes[2] = cr;
  rc = tj3DecodeYUVPlanes8(handle, planes, strides, dst, WIDTH, 0, HEIGHT,
                           TJPF_RGB);
  report(label, handle, rc);
  free(y); free(cb); free(cr); free(dst);
  tj3Destroy(handle);
}

/* Packed-YUV shapes for the #548-review round. The packed wrappers gate the
 * subsampling *before* the pixel-format range check (which lives in the
 * ...Planes8 delegates), so these lines deliberately pass an out-of-range
 * pixelFormat: a port that validates the format first reports the wrong
 * error and a valid-format line cannot tell. `tj3CompressFromYUV8` gates the
 * subsampling in the entry itself (turbojpeg.c:1497-1498), before the
 * delegate's quality gate; and a non-power-of-two align is an argument error
 * that beats every gate (turbojpeg.c:1493-1496). */
static void unset_fromyuv8_case(const char *label, int align)
{
  tjhandle handle = tj3Init(TJINIT_COMPRESS);
  unsigned char *src = (unsigned char *)malloc((size_t)WIDTH * HEIGHT * 4);
  unsigned char *buf = NULL;
  size_t size = 0;
  int rc;

  if (!handle || !src) { fprintf(stderr, "oom\n"); exit(2); }
  memset(src, 128, (size_t)WIDTH * HEIGHT * 4);
  rc = tj3CompressFromYUV8(handle, src, WIDTH, align, HEIGHT, &buf, &size);
  report(label, handle, rc);
  if (buf) tj3Free(buf);
  free(src);
  tj3Destroy(handle);
}

static void unset_encodeyuv8_badpf_case(const char *label)
{
  tjhandle handle = tj3Init(TJINIT_COMPRESS);
  unsigned char *src = (unsigned char *)malloc((size_t)WIDTH * HEIGHT * 3);
  unsigned char *dst = (unsigned char *)malloc((size_t)WIDTH * HEIGHT * 4);
  size_t i;
  int rc;

  if (!handle || !src || !dst) { fprintf(stderr, "oom\n"); exit(2); }
  for (i = 0; i < (size_t)WIDTH * HEIGHT * 3; i++)
    src[i] = (unsigned char)(i % 251);
  rc = tj3EncodeYUV8(handle, src, WIDTH, 0, HEIGHT, 99 /* bad TJPF */, dst, 1);
  report(label, handle, rc);
  free(src);
  free(dst);
  tj3Destroy(handle);
}

static void unset_decodeyuv8_badpf_case(const char *label)
{
  tjhandle handle = tj3Init(TJINIT_DECOMPRESS);
  unsigned char *src = (unsigned char *)malloc((size_t)WIDTH * HEIGHT * 4);
  unsigned char *dst = (unsigned char *)malloc((size_t)WIDTH * HEIGHT * 4);
  int rc;

  if (!handle || !src || !dst) { fprintf(stderr, "oom\n"); exit(2); }
  memset(src, 128, (size_t)WIDTH * HEIGHT * 4);
  rc = tj3DecodeYUV8(handle, src, 1, dst, WIDTH, 0, HEIGHT, 99 /* bad TJPF */);
  report(label, handle, rc);
  free(src);
  free(dst);
  tj3Destroy(handle);
}

static void unset_encodeyuvplanes8_case(const char *label)
{
  tjhandle handle = tj3Init(TJINIT_COMPRESS);
  unsigned char *src = (unsigned char *)malloc((size_t)WIDTH * HEIGHT * 3);
  unsigned char *y = (unsigned char *)malloc((size_t)WIDTH * HEIGHT);
  unsigned char *cb = (unsigned char *)malloc((size_t)WIDTH * HEIGHT);
  unsigned char *cr = (unsigned char *)malloc((size_t)WIDTH * HEIGHT);
  unsigned char *planes[3];
  int strides[3] = { WIDTH, WIDTH, WIDTH };
  size_t i;
  int rc;

  if (!handle || !src || !y || !cb || !cr) { fprintf(stderr, "oom\n"); exit(2); }
  for (i = 0; i < (size_t)WIDTH * HEIGHT * 3; i++)
    src[i] = (unsigned char)(i % 251);
  planes[0] = y; planes[1] = cb; planes[2] = cr;
  rc = tj3EncodeYUVPlanes8(handle, src, WIDTH, 0, HEIGHT, TJPF_RGB,
                           planes, strides);
  report(label, handle, rc);
  free(src); free(y); free(cb); free(cr);
  tj3Destroy(handle);
}

/* The legacy wrappers forward `align`/`pad` raw; a zero (or negative) value
 * must reach the TJ3 entry's validation rather than being clamped to 1 —
 * upstream refuses (#539 re-review). Both parameters here are set, so the
 * only thing under test is the align path. */
static void legacy_encodeyuv3_align0_case(const char *label)
{
  tjhandle handle = tjInitCompress();
  unsigned char *src = (unsigned char *)malloc((size_t)WIDTH * HEIGHT * 3);
  unsigned char *dst = (unsigned char *)malloc((size_t)WIDTH * HEIGHT * 4);
  size_t i;
  int rc;

  if (!handle || !src || !dst) { fprintf(stderr, "oom\n"); exit(2); }
  for (i = 0; i < (size_t)WIDTH * HEIGHT * 3; i++)
    src[i] = (unsigned char)(i % 251);
  rc = tjEncodeYUV3(handle, src, WIDTH, 0, HEIGHT, TJPF_RGB, dst,
                    0 /* align */, TJSAMP_420, 0);
  report(label, handle, rc);
  free(src); free(dst);
  tj3Destroy(handle);
}

static void legacy_decodeyuv_align0_case(const char *label)
{
  tjhandle handle = tjInitDecompress();
  unsigned char *src = (unsigned char *)malloc((size_t)WIDTH * HEIGHT * 4);
  unsigned char *dst = (unsigned char *)malloc((size_t)WIDTH * HEIGHT * 4);
  int rc;

  if (!handle || !src || !dst) { fprintf(stderr, "oom\n"); exit(2); }
  memset(src, 128, (size_t)WIDTH * HEIGHT * 4);
  rc = tjDecodeYUV(handle, src, 0 /* align */, TJSAMP_420, dst, WIDTH, 0,
                   HEIGHT, TJPF_RGB, 0);
  report(label, handle, rc);
  free(src); free(dst);
  tj3Destroy(handle);
}

int main(void)
{
  /* The divergence itself: refused upstream, accepted by the port. */
  compress16_case("c16_lossy", 0, 0, 0, -1);
  /* The configuration 16-bit exists for. */
  compress16_case("c16_lossless", 1, 0, 0, -1);
  /* Inside the window `turbojpeg-mp.c:111-115` honours (BITS_IN_JSAMPLE-3 .. 16). */
  compress16_case("c16_lossless_prec13", 1, 13, 0, -1);
  /* Outside it: silently ignored, so this stays a 16-bit lossless encode
   * rather than becoming an error. */
  compress16_case("c16_lossless_prec12", 1, 12, 0, -1);
  /* The trap: `TJPARAM_PRECISION` is read only when lossless is set, so
   * asking for 12 here does not turn a lossy 16-bit call into a legal one. */
  compress16_case("c16_lossy_prec12", 0, 12, 0, -1);

  /* Precedence against the destination, which upstream installs first. A
   * NOREALLOC slot that cannot be used *at all* — empty, or present with zero
   * capacity — is refused by `jdatadst-tj.c:184-192` before the compress
   * starts, so the buffer error wins over the precision rule. */
  compress16_case("c16_lossy_norealloc_null", 0, 0, NULL_SLOT, -1);
  /* ...but a slot that is merely *too small* does not: its capacity is only
   * tested when output overflows it, which never happens once the compress is
   * refused. So the precision error wins here, and a fix that checked the
   * destination unconditionally first would get this line wrong. */
  compress16_case("c16_lossy_norealloc_cramped", 0, 0, 16, -1);
  /* The same two slots with a legal configuration, so the lines above are
   * read as "which error", not "16-bit under NOREALLOC always fails". */
  compress16_case("c16_lossless_norealloc_null", 1, 0, NULL_SLOT, -1);
  compress16_case("c16_lossless_norealloc_roomy", 1, 0, 64 * 1024, -1);

  /* Precedence against the *lossless parameters*, which upstream validates
   * earlier still: `setCompDefaults` calls `jpeg_enable_lossless` before
   * `jpeg_mem_dest_tj` (`turbojpeg-mp.c:117-120`). With a point transform that
   * is not less than the precision, that error wins over the buffer error even
   * though the slot is unusable — so the destination preflight cannot simply be
   * hoisted to the front of the function. */
  compress16_case("c16_pt_ge_prec_norealloc_null", 1, 13, NULL_SLOT, 13);
  /* The same misconfiguration with a usable slot, so the line above reads as
   * "which error" rather than "an empty slot fails". */
  compress16_case("c16_pt_ge_prec_roomy", 1, 13, 0, 13);
  /* A legal point transform, to show the buffer error returns once the
   * lossless parameters stop being the first thing wrong. */
  compress16_case("c16_pt_lt_prec_norealloc_null", 1, 13, NULL_SLOT, 12);

  /* Does the rule generalise? These lines answer it. */
  compress12_case("c12_lossy", 0);
  compress12_case("c12_lossless", 1);
  compress8_case("c8_lossy", 0);
  compress8_case("c8_lossless", 1);

  /* P4-155 (#539): the unset-parameter matrix. */
  fresh_get_case("p4155_fresh_get");
  unset8_case("p4155_c8_unset_both", 0, 0, 0);
  unset8_case("p4155_c8_unset_quality", 0, 1, 0);
  unset8_case("p4155_c8_unset_subsamp", 1, 0, 0);
  /* Argument validation still wins over the musts (step 1 vs step 2). */
  unset8_case("p4155_c8_arg_precedence", 0, 0, 1);
  unset12_case("p4155_c12_unset_both");
  unset16_lossless_case("p4155_c16_lossless_unset");
  unset_encodeyuv8_case("p4155_encodeyuv8_unset");
  unset_fromyuvplanes8_case("p4155_fromyuvplanes8_unset");
  unset_decodeyuvplanes8_case("p4155_decodeyuvplanes8_unset");
  /* #548-review round: the packed wrappers, discriminating lines. */
  unset_fromyuv8_case("p4155_fromyuv8_unset", 1);
  unset_fromyuv8_case("p4155_fromyuv8_badalign", 3);
  unset_encodeyuv8_badpf_case("p4155_encodeyuv8_badpf_unset");
  unset_decodeyuv8_badpf_case("p4155_decodeyuv8_badpf_unset");
  unset_encodeyuvplanes8_case("p4155_encodeyuvplanes8_unset");
  legacy_encodeyuv3_align0_case("p4155_legacy_encodeyuv3_align0");
  legacy_decodeyuv_align0_case("p4155_legacy_decodeyuv_align0");

  return 0;
}

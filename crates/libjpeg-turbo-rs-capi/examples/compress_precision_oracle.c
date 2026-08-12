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
 * Output is `label rc err="<message>"`, one line per case. The message is
 * compared, not just the return code: on this path it is libjpeg's own
 * `JERR_BAD_PRECISION` text, which `CATCH_LIBJPEG` copies verbatim without
 * TurboJPEG's usual `function():` prefix. A caller that matches on it is
 * matching on something documented, so the port owes it exactly.
 *
 * Usage: compress_precision_oracle
 */

#include <stdio.h>
#include <stdlib.h>

#include <turbojpeg.h>

#define WIDTH 32
#define HEIGHT 32

/* Report one case. `rc` of 0 carries no message, so the field is empty then —
 * printing a stale `errStr` would compare state neither library promises. */
static void report(const char *label, tjhandle handle, int rc)
{
  printf("%s %d err=\"%s\"\n", label, rc, rc == 0 ? "" : tj3GetErrorStr(handle));
}

/* A compressor with the parameters every case shares. `precision` of 0 means
 * "leave TJPARAM_PRECISION alone", which is the common case. */
static tjhandle compressor(int lossless, int precision)
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
  return handle;
}

/* `tj3Compress16` under one configuration. The output buffer is allocated by
 * the library, so nothing here depends on the P4-145 ownership contract. */
static void compress16_case(const char *label, int lossless, int precision)
{
  tjhandle handle = compressor(lossless, precision);
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

/* `tj3Compress12`, to answer whether the rule generalises rather than assume
 * it. Twelve-bit lossy is one of the two precisions `jcmaster.c:206` admits,
 * so this is expected to succeed — but "expected" is what the oracle is for. */
static void compress12_case(const char *label, int lossless)
{
  tjhandle handle = compressor(lossless, 0);
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
  tjhandle handle = compressor(lossless, 0);
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

int main(void)
{
  /* The divergence itself: refused upstream, accepted by the port. */
  compress16_case("c16_lossy", 0, 0);
  /* The configuration 16-bit exists for. */
  compress16_case("c16_lossless", 1, 0);
  /* Inside the window `turbojpeg-mp.c:111-115` honours (BITS_IN_JSAMPLE-3 .. 16). */
  compress16_case("c16_lossless_prec13", 1, 13);
  /* Outside it: silently ignored, so this stays a 16-bit lossless encode
   * rather than becoming an error. */
  compress16_case("c16_lossless_prec12", 1, 12);
  /* The trap: `TJPARAM_PRECISION` is read only when lossless is set, so
   * asking for 12 here does not turn a lossy 16-bit call into a legal one. */
  compress16_case("c16_lossy_prec12", 0, 12);

  /* Does the rule generalise? These lines answer it. */
  compress12_case("c12_lossy", 0);
  compress12_case("c12_lossless", 1);
  compress8_case("c8_lossy", 0);
  compress8_case("c8_lossless", 1);

  return 0;
}

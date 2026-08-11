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
 * The *byte count* is deliberately not compared. Two independent encoders do
 * not agree on it — this port and upstream differ on entropy-coded output for
 * the same image — and a trace that bakes in one implementation's sizes would
 * fail for a reason that has nothing to do with the ownership contract under
 * test. What is contractual here is the return code and the pointer.
 *
 * Usage: norealloc_oracle
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
 * is refused before any of this matters, which would make the trace disagree
 * for a reason that has nothing to do with buffer ownership. Lossless is
 * therefore the configuration under test here. */
static void compress16_case(const char *label, size_t capacity)
{
  tjhandle h = compressor();
  if (tj3Set(h, TJPARAM_LOSSLESS, 1) != 0) { fprintf(stderr, "lossless\n"); exit(2); }
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

int main(void)
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

  compress16_case("compress16_roomy", ROOMY);
  compress16_case("compress16_cramped", CRAMPED);
  compress16_case("compress16_null", 0);

  yuv_planes_case("yuvplanes_roomy", ROOMY);
  yuv_planes_case("yuvplanes_cramped", CRAMPED);
  yuv_planes_case("yuvplanes_null", 0);

  transform_case("transform_roomy", ROOMY);
  transform_case("transform_cramped", CRAMPED);
  transform_case("transform_null", 0);

  return 0;
}

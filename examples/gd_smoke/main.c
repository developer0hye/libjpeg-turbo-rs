/* P2-10: libgd round-trip smoke test against our libjpeg-turbo-rs cdylib.
 *
 * libgd's `gdImageJpegPtr` and `gdImageCreateFromJpegPtr` call the
 * libjpeg C ABI directly (`jpeg_create_compress`, `jpeg_mem_src`,
 * `jpeg_read_scanlines`, etc.). Forcing libgd to bind those symbols
 * against our cdylib via LD_PRELOAD / DYLD_INSERT_LIBRARIES exercises
 * the same drop-in path as the libvips / ImageMagick harnesses, but
 * through a much smaller consumer (just libgd, no surrounding
 * pipeline) — useful for isolating libgd-specific gaps from
 * pipeline-orchestration noise.
 *
 * Usage: gd_smoke <input.ppm> <quality> <min-psnr>
 *
 * Exit codes (consumed by run.sh):
 *   0 success (PSNR >= min)
 *   1 PPM parse error
 *   3 input file not readable
 *   4 gdImageJpegPtr returned NULL (encode failure)
 *   5 gdImageCreateFromJpegPtr returned NULL (decode failure)
 *   6 dimension mismatch between input and decoded image
 *   7 PSNR below threshold
 *   99 usage error
 *
 * Build: examples/gd_smoke/build.sh (detects libgd headers via
 *        pkg-config / common include paths). The resulting binary is
 *        run by run.sh after staging our cdylib under the libjpeg
 *        SONAME.
 */

#include <gd.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_int(const char *s) {
  char *end = NULL;
  long v = strtol(s, &end, 10);
  return (end && *end == '\0') ? (int)v : -1;
}

static unsigned char *read_ppm(const char *path, int *w_out, int *h_out) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  /* Header: P6 W H 255 \n then raw RGB bytes. We do not bother with
   * comments; the fixture writers always emit the canonical form. */
  char magic[3] = {0};
  int w = 0, h = 0, maxv = 0;
  if (fscanf(f, "%2s %d %d %d", magic, &w, &h, &maxv) != 4 ||
      strcmp(magic, "P6") != 0 || maxv != 255 || w <= 0 || h <= 0) {
    fclose(f);
    return NULL;
  }
  /* Single whitespace byte per the Netpbm spec. */
  fgetc(f);
  size_t n = (size_t)w * (size_t)h * 3;
  unsigned char *buf = (unsigned char *)malloc(n);
  if (!buf || fread(buf, 1, n, f) != n) {
    free(buf);
    fclose(f);
    return NULL;
  }
  fclose(f);
  *w_out = w;
  *h_out = h;
  return buf;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s <input.ppm> <quality 0-100> <min-psnr>\n", argv[0]);
    return 99;
  }
  const char *in_ppm = argv[1];
  int quality = parse_int(argv[2]);
  double min_psnr = atof(argv[3]);
  if (quality < 0 || quality > 100) {
    fprintf(stderr, "quality out of range: %d\n", quality);
    return 99;
  }

  int w = 0, h = 0;
  unsigned char *pixels = read_ppm(in_ppm, &w, &h);
  if (!pixels) {
    fprintf(stderr, "failed to read PPM: %s\n", in_ppm);
    return 1;
  }

  /* PPM bytes -> gdImage (truecolor). gdImageSetPixel + gdTrueColor pack
   * into the 32-bit ARGB layout libgd uses internally. */
  gdImagePtr img = gdImageCreateTrueColor(w, h);
  if (!img) {
    free(pixels);
    return 3;
  }
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      size_t i = ((size_t)y * (size_t)w + (size_t)x) * 3;
      int color = gdTrueColor(pixels[i + 0], pixels[i + 1], pixels[i + 2]);
      gdImageSetPixel(img, x, y, color);
    }
  }

  /* Encode through our cdylib. gdImageJpegPtr internally calls
   * jpeg_create_compress + jpeg_set_defaults + jpeg_set_quality +
   * jpeg_start_compress + jpeg_write_scanlines + jpeg_finish_compress
   * + jpeg_destroy_compress — the canonical libjpeg encode path. */
  int jpeg_size = 0;
  void *jpeg_data = gdImageJpegPtr(img, &jpeg_size, quality);
  if (!jpeg_data || jpeg_size <= 0) {
    fprintf(stderr, "gdImageJpegPtr returned NULL\n");
    gdImageDestroy(img);
    free(pixels);
    return 4;
  }

  /* SOI sanity. */
  unsigned char *jp = (unsigned char *)jpeg_data;
  if (jpeg_size < 2 || jp[0] != 0xFF || jp[1] != 0xD8) {
    fprintf(stderr, "encoded bytes missing SOI marker\n");
    gdFree(jpeg_data);
    gdImageDestroy(img);
    free(pixels);
    return 4;
  }

  /* Decode back through our cdylib. */
  gdImagePtr dec = gdImageCreateFromJpegPtr(jpeg_size, jpeg_data);
  if (!dec) {
    fprintf(stderr, "gdImageCreateFromJpegPtr returned NULL\n");
    gdFree(jpeg_data);
    gdImageDestroy(img);
    free(pixels);
    return 5;
  }
  gdFree(jpeg_data);

  if (gdImageSX(dec) != w || gdImageSY(dec) != h) {
    fprintf(stderr, "dim mismatch: input=%dx%d decoded=%dx%d\n", w, h,
            gdImageSX(dec), gdImageSY(dec));
    gdImageDestroy(dec);
    gdImageDestroy(img);
    free(pixels);
    return 6;
  }

  /* Per-channel SSE then PSNR vs original. */
  long long sse = 0;
  size_t count = 0;
  for (int y = 0; y < h; y++) {
    for (int x = 0; x < w; x++) {
      size_t i = ((size_t)y * (size_t)w + (size_t)x) * 3;
      int c = gdImageGetTrueColorPixel(dec, x, y);
      int r = gdTrueColorGetRed(c);
      int g = gdTrueColorGetGreen(c);
      int b = gdTrueColorGetBlue(c);
      int dr = (int)pixels[i + 0] - r;
      int dg = (int)pixels[i + 1] - g;
      int db = (int)pixels[i + 2] - b;
      sse += (long long)dr * dr + (long long)dg * dg + (long long)db * db;
      count += 3;
    }
  }

  double mse = (count > 0) ? (double)sse / (double)count : 0.0;
  double psnr = (mse == 0.0) ? INFINITY : 10.0 * log10(255.0 * 255.0 / mse);
  printf("PSNR=%.3f dB (min=%.3f) mse=%.6f bytes=%zu\n", psnr, min_psnr, mse, count);

  gdImageDestroy(dec);
  gdImageDestroy(img);
  free(pixels);
  return (psnr < min_psnr) ? 7 : 0;
}

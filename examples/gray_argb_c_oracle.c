/*
 * Issue #369 oracle: decode a JPEG through the real C libjpeg-turbo TurboJPEG
 * API (tj3Decompress8) to a caller-named 4bpp pixel format and write the raw
 * pixel buffer to stdout.
 *
 * djpeg has no ARGB/ABGR writer, so the byte-level channel order of
 * TJPF_ARGB/TJPF_ABGR output cannot be pinned with the stock tools; this
 * harness supplies that missing oracle. tests/helpers/c_oracle.rs compiles it
 * on demand against a libjpeg-turbo development install (turbojpeg.h +
 * libturbojpeg).
 *
 * Usage: gray_argb_c_oracle <jpeg-path> <ARGB|ABGR|RGBA|BGRA|RGBX|BGRX|XRGB|XBGR>
 * Output: width*height*4 raw bytes on stdout.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <turbojpeg.h>

static int pixel_format_from_name(const char *name) {
  if (strcmp(name, "ARGB") == 0) return TJPF_ARGB;
  if (strcmp(name, "ABGR") == 0) return TJPF_ABGR;
  if (strcmp(name, "RGBA") == 0) return TJPF_RGBA;
  if (strcmp(name, "BGRA") == 0) return TJPF_BGRA;
  if (strcmp(name, "RGBX") == 0) return TJPF_RGBX;
  if (strcmp(name, "BGRX") == 0) return TJPF_BGRX;
  if (strcmp(name, "XRGB") == 0) return TJPF_XRGB;
  if (strcmp(name, "XBGR") == 0) return TJPF_XBGR;
  return -1;
}

int main(int argc, char **argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s <jpeg-path> <pixel-format>\n", argv[0]);
    return 2;
  }
  int pf = pixel_format_from_name(argv[2]);
  if (pf < 0) {
    fprintf(stderr, "unknown pixel format %s\n", argv[2]);
    return 2;
  }

  FILE *f = fopen(argv[1], "rb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", argv[1]);
    return 1;
  }
  fseek(f, 0, SEEK_END);
  long jpeg_size = ftell(f);
  fseek(f, 0, SEEK_SET);
  unsigned char *jpeg_buf = malloc((size_t)jpeg_size);
  if (!jpeg_buf || fread(jpeg_buf, 1, (size_t)jpeg_size, f) != (size_t)jpeg_size) {
    fprintf(stderr, "cannot read %s\n", argv[1]);
    return 1;
  }
  fclose(f);

  tjhandle handle = tj3Init(TJINIT_DECOMPRESS);
  if (!handle) {
    fprintf(stderr, "tj3Init failed\n");
    return 1;
  }
  if (tj3DecompressHeader(handle, jpeg_buf, (size_t)jpeg_size) < 0) {
    fprintf(stderr, "tj3DecompressHeader failed: %s\n", tj3GetErrorStr(handle));
    return 1;
  }
  int width = tj3Get(handle, TJPARAM_JPEGWIDTH);
  int height = tj3Get(handle, TJPARAM_JPEGHEIGHT);
  size_t out_size = (size_t)width * (size_t)height * 4;
  unsigned char *out = malloc(out_size);
  if (!out) {
    fprintf(stderr, "out of memory\n");
    return 1;
  }
  if (tj3Decompress8(handle, jpeg_buf, (size_t)jpeg_size, out, 0, pf) < 0) {
    fprintf(stderr, "tj3Decompress8 failed: %s\n", tj3GetErrorStr(handle));
    return 1;
  }
  if (fwrite(out, 1, out_size, stdout) != out_size) {
    fprintf(stderr, "short write\n");
    return 1;
  }
  free(out);
  free(jpeg_buf);
  tj3Destroy(handle);
  return 0;
}

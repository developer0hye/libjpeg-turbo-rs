/*
 * C oracle for the TurboJPEG YUV decompress component-count contract (P4-125).
 *
 * Links against stock libjpeg-turbo and reports how upstream's
 * tj3DecompressToYUV8 / tj3DecompressToYUVPlanes8 answer for the JPEG named on
 * the command line, so the Rust regressions can assert parity rather than
 * asserting our own behaviour in isolation.
 *
 * Usage: yuv_component_count_c_oracle <jpeg-path>
 * Prints two lines on stdout, then exits 0:
 *   yuv8 rc=<int> err=<message>
 *   planes8 rc=<int> err=<message>
 * A non-zero exit means the harness itself failed (unreadable file, tj3Init
 * failure) and is never a property of the image under test.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <turbojpeg.h>

/* Big enough for every plane layout the fixtures use; the packed and planar
   destinations are deliberately oversized so a rejection is the only reason a
   call can fail here. */
#define DST_CAPACITY (16 * 1024 * 1024)

static unsigned char *read_file(const char *path, size_t *size_out) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return NULL; }
  long size = ftell(f);
  if (size < 0) { fclose(f); return NULL; }
  rewind(f);
  unsigned char *buf = (unsigned char *)malloc((size_t)size);
  if (!buf) { fclose(f); return NULL; }
  if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
    free(buf);
    fclose(f);
    return NULL;
  }
  fclose(f);
  *size_out = (size_t)size;
  return buf;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s <jpeg-path>\n", argv[0]);
    return 2;
  }

  size_t jpeg_size = 0;
  unsigned char *jpeg = read_file(argv[1], &jpeg_size);
  if (!jpeg) {
    fprintf(stderr, "cannot read %s\n", argv[1]);
    return 2;
  }

  unsigned char *packed = (unsigned char *)malloc(DST_CAPACITY);
  unsigned char *planes[3];
  planes[0] = (unsigned char *)malloc(DST_CAPACITY / 4);
  planes[1] = (unsigned char *)malloc(DST_CAPACITY / 4);
  planes[2] = (unsigned char *)malloc(DST_CAPACITY / 4);
  if (!packed || !planes[0] || !planes[1] || !planes[2]) {
    fprintf(stderr, "allocation failure\n");
    return 2;
  }

  /* A fresh handle per call: a rejection leaves the handle's error slot set,
     and reusing it would blur which call produced which message. */
  tjhandle h = tj3Init(TJINIT_DECOMPRESS);
  if (!h) {
    fprintf(stderr, "tj3Init failed\n");
    return 2;
  }
  int rc_yuv8 = tj3DecompressToYUV8(h, jpeg, jpeg_size, packed, 1);
  printf("yuv8 rc=%d err=%s\n", rc_yuv8, tj3GetErrorStr(h));
  tj3Destroy(h);

  h = tj3Init(TJINIT_DECOMPRESS);
  if (!h) {
    fprintf(stderr, "tj3Init failed\n");
    return 2;
  }
  int rc_planes8 = tj3DecompressToYUVPlanes8(h, jpeg, jpeg_size, planes, NULL);
  printf("planes8 rc=%d err=%s\n", rc_planes8, tj3GetErrorStr(h));
  tj3Destroy(h);

  free(jpeg);
  free(packed);
  free(planes[0]);
  free(planes[1]);
  free(planes[2]);
  return 0;
}

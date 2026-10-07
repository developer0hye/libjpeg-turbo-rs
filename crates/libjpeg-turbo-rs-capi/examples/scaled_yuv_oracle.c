/*
 * P4-234 C oracle: what tj3DecompressToYUV8 / tj3DecompressToYUVPlanes8
 * write at every scaling factor.
 *
 * For each `<workdir>/<label>.jpg` and each of tj3GetScalingFactors()'s
 * factors, decode to packed YUV (align 1) into exactly
 * `tj3YUVBufSize(TJSCALED(w), 1, TJSCALED(h), subsamp)` bytes, and to planes
 * into exactly `tj3YUVPlaneSize` bytes each, and print the return codes and
 * an FNV-1a digest of the bytes. The Rust mirror
 * (`tests/capi_scaled_yuv.rs`) runs the same loop through this crate's
 * exports and compares the transcripts verbatim, so every sample — the
 * padding columns and rows of a plane included — must equal stock's.
 *
 * The first line is `version=<LIBJPEG_TURBO_VERSION_NUMBER>`: TurboJPEG
 * before 3.2 classifies 4:1:0 and 2:4 frames as TJSAMP_UNKNOWN and refuses
 * them, so the mirror compares those labels only against a 3.2+ oracle.
 *
 * Usage: scaled_yuv_oracle <workdir> <label>...
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <jconfig.h>
#include <turbojpeg.h>

static unsigned char *read_file(const char *path, size_t *size)
{
  FILE *file = fopen(path, "rb");
  unsigned char *bytes;
  long length;

  if (!file) return NULL;
  if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) <= 0 ||
      fseek(file, 0, SEEK_SET) != 0) {
    fclose(file);
    return NULL;
  }
  bytes = (unsigned char *)malloc((size_t)length);
  if (!bytes || fread(bytes, 1, (size_t)length, file) != (size_t)length) {
    free(bytes);
    fclose(file);
    return NULL;
  }
  fclose(file);
  *size = (size_t)length;
  return bytes;
}

static unsigned long long fnv1a(const unsigned char *bytes, size_t size,
                                unsigned long long hash)
{
  size_t i;

  for (i = 0; i < size; i++) {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

int main(int argc, char **argv)
{
  int arg, i, count = 0;
  tjscalingfactor *factors = tj3GetScalingFactors(&count);

  if (argc < 3 || !factors) {
    fprintf(stderr, "usage: %s <workdir> <label>...\n", argv[0]);
    return 2;
  }
  printf("version=%d\n", LIBJPEG_TURBO_VERSION_NUMBER);
  for (arg = 2; arg < argc; arg++) {
    char path[4096];
    size_t size = 0;
    unsigned char *jpeg;

    snprintf(path, sizeof(path), "%s/%s.jpg", argv[1], argv[arg]);
    jpeg = read_file(path, &size);
    if (!jpeg) {
      fprintf(stderr, "cannot read %s\n", path);
      return 1;
    }
    for (i = 0; i < count; i++) {
      tjhandle handle = tj3Init(TJINIT_DECOMPRESS);
      int width, height, subsamp, planes, rc, prc, p;
      size_t packed_size, plane_size[3] = { 0, 0, 0 };
      unsigned char *packed, *plane[3] = { NULL, NULL, NULL };
      unsigned long long packed_hash = 14695981039346656037ULL,
                         planar_hash = 14695981039346656037ULL;

      tj3DecompressHeader(handle, jpeg, size);
      tj3SetScalingFactor(handle, factors[i]);
      width = TJSCALED(tj3Get(handle, TJPARAM_JPEGWIDTH), factors[i]);
      height = TJSCALED(tj3Get(handle, TJPARAM_JPEGHEIGHT), factors[i]);
      subsamp = tj3Get(handle, TJPARAM_SUBSAMP);
      planes = subsamp == TJSAMP_GRAY ? 1 : 3;
      packed_size = tj3YUVBufSize(width, 1, height, subsamp);
      packed = (unsigned char *)calloc(packed_size, 1);
      for (p = 0; p < planes; p++) {
        plane_size[p] = tj3YUVPlaneSize(p, width, 0, height, subsamp);
        plane[p] = (unsigned char *)calloc(plane_size[p], 1);
      }
      rc = tj3DecompressToYUV8(handle, jpeg, size, packed, 1);
      prc = tj3DecompressToYUVPlanes8(handle, jpeg, size, plane, NULL);
      if (rc == 0) packed_hash = fnv1a(packed, packed_size, packed_hash);
      for (p = 0; prc == 0 && p < planes; p++)
        planar_hash = fnv1a(plane[p], plane_size[p], planar_hash);
      printf("%s %d/%d %dx%d rc=%d %016llx planar_rc=%d %016llx\n", argv[arg],
             factors[i].num, factors[i].denom, width, height, rc, packed_hash,
             prc, planar_hash);
      for (p = 0; p < planes; p++) free(plane[p]);
      free(packed);
      tj3Destroy(handle);
    }
    free(jpeg);
  }
  return 0;
}

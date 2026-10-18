/*
 * P4-227 C oracle: which handle parameters tj3Transform applies.
 *
 * Upstream's tj3Transform reads eight handle parameters a caller may not
 * expect a lossless transform to read (`turbojpeg.c:2920-3086`):
 *
 *   - TJPARAM_MAXPIXELS against the source, right after the header read and
 *     before any per-transform check (`:2995-2998`, "Image is too large");
 *   - TJPARAM_SCANLIMIT and TJPARAM_MAXMEMORY while the source coefficients
 *     are read (`:2943-2951`; "Progressive JPEG image has more than N
 *     scans", "Memory limit exceeded"), the budget covering the source's
 *     whole-image coefficient arrays plus every transform's workspace;
 *   - TJPARAM_PROGRESSIVE, TJPARAM_ARITHMETIC and TJPARAM_OPTIMIZE for the
 *     output, OR-ed with the per-transform TJXOPT_* flags (`:3029-3035`), and
 *     TJPARAM_RESTARTBLOCKS / TJPARAM_RESTARTROWS (`:3036-3037`).
 *
 * It does not call setDecompParameters, so a transform publishes nothing:
 * every line also prints TJPARAM_JPEGWIDTH, which must stay -1.
 *
 * Output: one line per case, `<label> <case> rc=<rc> sof=<marker> size=<n>
 * hash=<fnv1a-64> jw=<JPEGWIDTH>`, so the Rust mirror
 * (`tests/capi_transform_parameters.rs`) compares the output *bytes*, not
 * just the return code. Error messages are not printed.
 *
 * The TJPARAM_MAXMEMORY cases run only on the `big` label (a 1024x1024 4:4:4
 * frame, 6 MiB of coefficients), at budgets on both sides of the boundary
 * stock's arrays set: 6/7 MiB for an in-place transform, 12/13 MiB for one
 * that needs a workspace, 8/9 MiB for a TJXOPT_GRAY rotation (luma
 * workspace only). Stock also counts its small pool allocations, which
 * the port's estimate does not model; at these budgets the difference never
 * decides the outcome.
 *
 * Usage: transform_parameters_oracle <workdir> <label>...
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* The first SOFn marker code, or 0. */
static int sof_marker(const unsigned char *jpeg, size_t size)
{
  size_t i;

  for (i = 2; i + 1 < size; i++)
    if (jpeg[i] == 0xFF && jpeg[i + 1] >= 0xC0 && jpeg[i + 1] <= 0xCF &&
        jpeg[i + 1] != 0xC4 && jpeg[i + 1] != 0xC8 && jpeg[i + 1] != 0xCC)
      return jpeg[i + 1];
  return 0;
}

static unsigned long long fnv1a(const unsigned char *bytes, size_t size)
{
  unsigned long long hash = 14695981039346656037ULL;
  size_t i;

  for (i = 0; i < size; i++) {
    hash ^= bytes[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

/* The crop region the next `run` applies when TJXOPT_CROP is set. */
static tjregion crop_region = { 0, 0, 0, 0 };

/* One tj3Transform on a fresh TJINIT_TRANSFORM handle with up to two
 * parameters set (`param` < 0 means none). */
static void run(const char *label, const char *case_name,
                const unsigned char *jpeg, size_t size, int param1,
                int value1, int param2, int value2, int op, int options)
{
  tjhandle handle = tj3Init(TJINIT_TRANSFORM);
  unsigned char *dst = NULL;
  size_t dst_size = 0;
  tjtransform transform;
  int rc;

  memset(&transform, 0, sizeof(transform));
  transform.op = op;
  transform.options = options;
  if (options & TJXOPT_CROP) transform.r = crop_region;
  if (param1 >= 0) tj3Set(handle, param1, value1);
  if (param2 >= 0) tj3Set(handle, param2, value2);
  rc = tj3Transform(handle, jpeg, size, 1, &dst, &dst_size, &transform);
  printf("%s %s rc=%d sof=%02x size=%zu hash=%016llx jw=%d\n", label,
         case_name, rc, rc == 0 ? sof_marker(dst, dst_size) : 0,
         rc == 0 ? dst_size : (size_t)0, rc == 0 ? fnv1a(dst, dst_size) : 0ULL,
         tj3Get(handle, TJPARAM_JPEGWIDTH));
  tj3Free(dst);
  tj3Destroy(handle);
}

static int run_label(const char *workdir, const char *label)
{
  char path[4096];
  unsigned char *jpeg;
  size_t size = 0;
  tjhandle probe;
  int pixels;

  snprintf(path, sizeof(path), "%s/%s.jpg", workdir, label);
  jpeg = read_file(path, &size);
  if (!jpeg) {
    fprintf(stderr, "cannot read %s\n", path);
    return 1;
  }
  probe = tj3Init(TJINIT_DECOMPRESS);
  if (!probe || tj3DecompressHeader(probe, jpeg, size) != 0) {
    fprintf(stderr, "%s: probe header failed\n", label);
    return 1;
  }
  pixels = tj3Get(probe, TJPARAM_JPEGWIDTH) * tj3Get(probe, TJPARAM_JPEGHEIGHT);
  tj3Destroy(probe);

  run(label, "plain", jpeg, size, -1, 0, -1, 0, TJXOP_NONE, 0);
  run(label, "rot90", jpeg, size, -1, 0, -1, 0, TJXOP_ROT90, 0);
  /* MAXPIXELS: refused one pixel short of the frame, accepted at it; and
   * refused before the per-transform PERFECT check. */
  run(label, "maxpixels_under", jpeg, size, TJPARAM_MAXPIXELS, pixels - 1, -1,
      0, TJXOP_NONE, 0);
  run(label, "maxpixels_at", jpeg, size, TJPARAM_MAXPIXELS, pixels, -1, 0,
      TJXOP_NONE, 0);
  run(label, "maxpixels_perfect", jpeg, size, TJPARAM_MAXPIXELS, 1, -1, 0,
      TJXOP_TRANSVERSE, TJXOPT_PERFECT);
  /* SCANLIMIT: refuses a stream with more scans, not a single-scan one. */
  run(label, "scanlimit", jpeg, size, TJPARAM_SCANLIMIT, 2, -1, 0, TJXOP_NONE,
      0);
  run(label, "maxmemory_small", jpeg, size, TJPARAM_MAXMEMORY, 1, -1, 0,
      TJXOP_NONE, 0);
  /* The output parameters, and their TJXOPT_* twins. */
  run(label, "progressive", jpeg, size, TJPARAM_PROGRESSIVE, 1, -1, 0,
      TJXOP_NONE, 0);
  run(label, "arithmetic", jpeg, size, TJPARAM_ARITHMETIC, 1, -1, 0,
      TJXOP_NONE, 0);
  run(label, "progressive_arithmetic", jpeg, size, TJPARAM_PROGRESSIVE, 1,
      TJPARAM_ARITHMETIC, 1, TJXOP_ROT90, 0);
  run(label, "optimize", jpeg, size, TJPARAM_OPTIMIZE, 1, -1, 0, TJXOP_NONE,
      0);
  run(label, "arithmetic_optimize", jpeg, size, TJPARAM_ARITHMETIC, 1,
      TJPARAM_OPTIMIZE, 1, TJXOP_NONE, 0);
  run(label, "restartblocks", jpeg, size, TJPARAM_RESTARTBLOCKS, 4, -1, 0,
      TJXOP_NONE, 0);
  run(label, "restartrows", jpeg, size, TJPARAM_RESTARTROWS, 1, -1, 0,
      TJXOP_ROT90, 0);
  /* RESTARTBLOCKS set after RESTARTROWS clears it (`:819-830`). */
  run(label, "restartrows_then_blocks", jpeg, size, TJPARAM_RESTARTROWS, 1,
      TJPARAM_RESTARTBLOCKS, 4, TJXOP_NONE, 0);
  /* Crop regions (P4-240): a zero width or height runs to the edge
   * (JCROP_UNSET, `turbojpeg.c:2979-2986`); a region past the transformed
   * image is "Invalid crop request". */
  {
    static const struct { const char *name; int op; tjregion r; } crops[] = {
      { "crop_to_edge", TJXOP_NONE, { 16, 16, 0, 0 } },
      { "crop_width_to_edge", TJXOP_NONE, { 16, 0, 0, 16 } },
      { "crop_to_edge_rot90", TJXOP_ROT90, { 16, 0, 0, 0 } },
      { "crop_past_edge", TJXOP_NONE, { 48, 0, 32, 32 } },
      { "crop_origin_outside", TJXOP_NONE, { 64, 0, 0, 0 } },
    };
    size_t i;

    for (i = 0; i < sizeof(crops) / sizeof(crops[0]); i++) {
      crop_region = crops[i].r;
      run(label, crops[i].name, jpeg, size, -1, 0, -1, 0, crops[i].op,
          TJXOPT_CROP);
    }
  }
  run(label, "opt_progressive", jpeg, size, -1, 0, -1, 0, TJXOP_NONE,
      TJXOPT_PROGRESSIVE);
  run(label, "opt_arithmetic", jpeg, size, -1, 0, -1, 0, TJXOP_NONE,
      TJXOPT_ARITHMETIC);

  if (strcmp(label, "big") == 0) {
    static const struct { const char *name; int op; int megabytes; } cases[] = {
      { "maxmemory_none_6", TJXOP_NONE, 6 },
      { "maxmemory_none_7", TJXOP_NONE, 7 },
      { "maxmemory_hflip_6", TJXOP_HFLIP, 6 },
      { "maxmemory_hflip_7", TJXOP_HFLIP, 7 },
      { "maxmemory_vflip_12", TJXOP_VFLIP, 12 },
      { "maxmemory_vflip_13", TJXOP_VFLIP, 13 },
      { "maxmemory_rot90_12", TJXOP_ROT90, 12 },
      { "maxmemory_rot90_13", TJXOP_ROT90, 13 },
      { "maxmemory_rot180_12", TJXOP_ROT180, 12 },
      { "maxmemory_rot180_13", TJXOP_ROT180, 13 },
      { "maxmemory_transpose_12", TJXOP_TRANSPOSE, 12 },
      { "maxmemory_transpose_13", TJXOP_TRANSPOSE, 13 },
    };
    size_t i;

    for (i = 0; i < sizeof(cases) / sizeof(cases[0]); i++)
      run(label, cases[i].name, jpeg, size, TJPARAM_MAXMEMORY,
          cases[i].megabytes, -1, 0, cases[i].op, 0);
    /* Grayscale output keeps only the luma workspace: 6 MiB + 2 MiB. */
    run(label, "maxmemory_rot90_gray_8", jpeg, size, TJPARAM_MAXMEMORY, 8, -1,
        0, TJXOP_ROT90, TJXOPT_GRAY);
    run(label, "maxmemory_rot90_gray_9", jpeg, size, TJPARAM_MAXMEMORY, 9, -1,
        0, TJXOP_ROT90, TJXOPT_GRAY);
  }
  free(jpeg);
  return 0;
}

int main(int argc, char **argv)
{
  int i, failures = 0;

  if (argc < 3) {
    fprintf(stderr, "usage: %s <workdir> <label>...\n", argv[0]);
    return 2;
  }
  for (i = 2; i < argc; i++)
    failures += run_label(argv[1], argv[i]);
  return failures ? 1 : 0;
}

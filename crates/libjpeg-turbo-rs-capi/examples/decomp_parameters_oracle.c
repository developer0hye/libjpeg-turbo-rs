/*
 * P4-199 / P4-200 / P4-203 C oracle: what a decompress publishes into the
 * handle.
 *
 * `setDecompParameters` (`turbojpeg.c:514-536`) writes thirteen parameters
 * from the frame header -- SUBSAMP, JPEGWIDTH, JPEGHEIGHT, PRECISION,
 * COLORSPACE, PROGRESSIVE, ARITHMETIC, LOSSLESS, LOSSLESSPSV, LOSSLESSPT,
 * XDENSITY, YDENSITY and DENSITYUNITS -- and it is called from the body
 * `turbojpeg-mp.c` compiles once per precision, *before* the
 * `TJPARAM_MAXPIXELS` refusal five lines later (`:190`, `:195-199`). So:
 *
 *   - all three of tj3Decompress8/12/16 publish the same thirteen;
 *   - the values are the SOF's, unaffected by scaling or cropping;
 *   - a decode refused by TJPARAM_MAXPIXELS has already published them.
 *
 * Each of those is a sentence a reading of the source supports, which is
 * exactly why they are measured here instead: LOSSLESSPSV / LOSSLESSPT are
 * `dinfo.Ss` / `dinfo.Al`, i.e. the *first* scan's spectral start and point
 * transform, which on a progressive stream is the DC scan's `Al` -- a number
 * no one would guess from the parameter's name.
 *
 * Output: one line per case, `<label> <case> rc=<rc> <13 params>`, all
 * deterministic so the Rust mirror (`tests/capi_decomp_parameters.rs`) can be
 * compared verbatim. Error *messages* are not printed: the two libraries word
 * them differently on purpose, and what is pinned here is publication.
 *
 * Cases deliberately absent, because the two implementations diverge on them
 * by design or under another item, and printing them would make this gate
 * fail on something it is not about:
 *
 *   - tj3Decompress8 on a 12-bit frame (stock raises JERR_BAD_PRECISION; the
 *     port downscales to 8 bits);
 *   - tj3Decompress12 on an 8-bit frame (stock promotes `data_precision` to
 *     12 at `turbojpeg-mp.c:191-194`; the port refuses, P4-171);
 *   - TJPARAM_MAXMEMORY refusals (stock's budget reaches only its
 *     whole-image arrays, the port's a header-time estimate that includes the
 *     output buffer, so the two refuse different frames by design).
 *
 * Usage: decomp_parameters_oracle <workdir> <label>...
 * reads `<workdir>/<label>.jpg` for every label; a label starting with
 * `headeronly_` traces tj3DecompressHeader alone.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <turbojpeg.h>

static const int PARAMS[13] = {
  TJPARAM_SUBSAMP, TJPARAM_JPEGWIDTH, TJPARAM_JPEGHEIGHT, TJPARAM_PRECISION,
  TJPARAM_COLORSPACE, TJPARAM_PROGRESSIVE, TJPARAM_ARITHMETIC,
  TJPARAM_LOSSLESS, TJPARAM_LOSSLESSPSV, TJPARAM_LOSSLESSPT,
  TJPARAM_XDENSITY, TJPARAM_YDENSITY, TJPARAM_DENSITYUNITS
};
static const char *NAMES[13] = {
  "subsamp", "jw", "jh", "prec", "cs", "prog", "arith", "lossless", "psv",
  "pt", "xd", "yd", "du"
};

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

static void emit(const char *label, const char *case_name, int rc,
                 tjhandle handle)
{
  int i;

  printf("%s %s rc=%d", label, case_name, rc);
  for (i = 0; i < 13; i++)
    printf(" %s=%d", NAMES[i], tj3Get(handle, PARAMS[i]));
  printf("\n");
}

/* The pixel format a caller picks from the published colour space. */
static int pixel_format_for(tjhandle handle)
{
  switch (tj3Get(handle, TJPARAM_COLORSPACE)) {
  case TJCS_GRAY:  return TJPF_GRAY;
  case TJCS_CMYK:
  case TJCS_YCCK:  return TJPF_CMYK;
  default:         return TJPF_RGB;
  }
}

/* The documented routing: read the header, branch on TJPARAM_PRECISION, call
 * the matching entry point. `buffer` holds at least width*height*4 samples
 * of two bytes, which covers every format used here. */
static int routed_decompress(tjhandle handle, const unsigned char *jpeg,
                             size_t size, void *buffer, int precision,
                             int pixel_format)
{
  if (precision <= 8)
    return tj3Decompress8(handle, jpeg, size, (unsigned char *)buffer, 0,
                          pixel_format);
  if (precision <= 12)
    return tj3Decompress12(handle, jpeg, size, (short *)buffer, 0,
                           pixel_format);
  return tj3Decompress16(handle, jpeg, size, (unsigned short *)buffer, 0,
                         pixel_format);
}

static int run_label(const char *workdir, const char *label,
                     tjhandle sequence)
{
  char path[4096];
  unsigned char *jpeg;
  size_t size = 0;
  tjhandle probe, handle;
  int width, height, precision, pixel_format, lossless, rc;
  void *buffer;

  snprintf(path, sizeof(path), "%s/%s.jpg", workdir, label);
  jpeg = read_file(path, &size);
  if (!jpeg) {
    fprintf(stderr, "cannot read %s\n", path);
    return 1;
  }

  /* A `headeronly_` label traces tj3DecompressHeader alone (P4-142): a
   * stream whose header is intact and whose entropy data is not, or whose
   * later scans are missing, must be read without decoding; one whose frame
   * libjpeg's get_sof / initial_setup refuse must publish nothing. The decode
   * is not traced -- stock reports corrupt data as a warning, which is a
   * different contract. */
  if (strncmp(label, "headeronly_", 11) == 0) {
    handle = tj3Init(TJINIT_DECOMPRESS);
    rc = tj3DecompressHeader(handle, jpeg, size);
    emit(label, "header", rc, handle);
    tj3Destroy(handle);
    free(jpeg);
    return 0;
  }

  /* Size the destination and choose the route from a separate handle, so
   * every case below starts from a handle that has read nothing. */
  probe = tj3Init(TJINIT_DECOMPRESS);
  if (!probe || tj3DecompressHeader(probe, jpeg, size) != 0) {
    fprintf(stderr, "%s: probe header failed\n", label);
    return 1;
  }
  width = tj3Get(probe, TJPARAM_JPEGWIDTH);
  height = tj3Get(probe, TJPARAM_JPEGHEIGHT);
  precision = tj3Get(probe, TJPARAM_PRECISION);
  lossless = tj3Get(probe, TJPARAM_LOSSLESS);
  pixel_format = pixel_format_for(probe);
  tj3Destroy(probe);
  buffer = calloc((size_t)width * (size_t)height * 4, 2);
  if (!buffer) return 1;

  /* 1. tj3DecompressHeader alone. */
  handle = tj3Init(TJINIT_DECOMPRESS);
  rc = tj3DecompressHeader(handle, jpeg, size);
  emit(label, "header", rc, handle);

  /* 2. Then the entry point TJPARAM_PRECISION routes to, on the same handle. */
  rc = routed_decompress(handle, jpeg, size, buffer,
                         tj3Get(handle, TJPARAM_PRECISION), pixel_format);
  emit(label, "routed", rc, handle);
  tj3Destroy(handle);

  /* 3. A decompress with no header call first: it publishes by itself. */
  handle = tj3Init(TJINIT_DECOMPRESS);
  rc = routed_decompress(handle, jpeg, size, buffer, precision, pixel_format);
  emit(label, "direct", rc, handle);
  tj3Destroy(handle);

  /* 4. tj3DecompressHeader applies no TJPARAM_MAXPIXELS test
   * (`turbojpeg.c:1872-1927`); the decompress after it does, having
   * published -- at every precision. */
  handle = tj3Init(TJINIT_DECOMPRESS);
  tj3Set(handle, TJPARAM_MAXPIXELS, 1);
  rc = tj3DecompressHeader(handle, jpeg, size);
  emit(label, "maxpixels_header", rc, handle);
  tj3Destroy(handle);
  handle = tj3Init(TJINIT_DECOMPRESS);
  tj3Set(handle, TJPARAM_MAXPIXELS, 1);
  rc = routed_decompress(handle, jpeg, size, buffer, precision, pixel_format);
  emit(label, "maxpixels", rc, handle);
  tj3Destroy(handle);

  /* 4b. Refused by TJPARAM_SCANLIMIT on a stream with more scans -- also
   * after publishing: stock enforces the limit from its progress monitor
   * during the decode, and the port in the scan walk that follows the
   * header read. A single-scan stream decodes. */
  handle = tj3Init(TJINIT_DECOMPRESS);
  tj3Set(handle, TJPARAM_SCANLIMIT, 2);
  rc = routed_decompress(handle, jpeg, size, buffer, precision, pixel_format);
  emit(label, "scanlimit", rc, handle);
  tj3Destroy(handle);

  /* 5. One long-lived handle across every fixture: each decode must replace
   * what the previous one published, not merge with it. */
  rc = routed_decompress(sequence, jpeg, size, buffer, precision,
                         pixel_format);
  emit(label, "sequence", rc, sequence);

  /* 6-7. Scaled and cropped 8-bit lossy decodes publish the SOF's dimensions,
   * not the output's (P4-200). */
  if (precision == 8 && !lossless) {
    tjscalingfactor half = { 1, 2 };
    tjregion corner = { 0, 0, 8, 8 };

    handle = tj3Init(TJINIT_DECOMPRESS);
    tj3SetScalingFactor(handle, half);
    rc = tj3Decompress8(handle, jpeg, size, (unsigned char *)buffer, 0,
                        pixel_format);
    emit(label, "scaled", rc, handle);
    tj3Destroy(handle);

    handle = tj3Init(TJINIT_DECOMPRESS);
    rc = tj3DecompressHeader(handle, jpeg, size);
    if (rc == 0) rc = tj3SetCroppingRegion(handle, corner);
    if (rc == 0)
      rc = tj3Decompress8(handle, jpeg, size, (unsigned char *)buffer, 0,
                          pixel_format);
    emit(label, "cropped", rc, handle);
    tj3Destroy(handle);
  }

  free(buffer);
  free(jpeg);
  return 0;
}

int main(int argc, char **argv)
{
  tjhandle sequence;
  int i, failures = 0;

  if (argc < 3) {
    fprintf(stderr, "usage: %s <workdir> <label>...\n", argv[0]);
    return 2;
  }
  sequence = tj3Init(TJINIT_DECOMPRESS);
  if (!sequence) return 1;
  for (i = 2; i < argc; i++)
    failures += run_label(argv[1], argv[i], sequence);
  tj3Destroy(sequence);
  return failures ? 1 : 0;
}

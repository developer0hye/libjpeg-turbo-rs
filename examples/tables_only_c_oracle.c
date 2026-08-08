/*
 * Reference oracle for abbreviated tables-only datastreams (P4-116).
 *
 * `jpeg_write_tables()` is a library entry point that the `cjpeg` CLI has
 * never exposed — there is no `-tables-only` switch in any libjpeg or
 * libjpeg-turbo release. `tests/abbreviated_datastream.rs` nevertheless
 * shelled out to `cjpeg -tables-only`, probed for support by looking for the
 * strings "unrecognized"/"unknown option" (cjpeg prints a usage dump instead,
 * so the probe always said "supported"), and then `continue`d past every case
 * when the invocation failed. The matrix compared nothing and reported green.
 *
 * This program is the oracle that flag would have been. It writes the
 * abbreviated stream libjpeg produces for a given quality, so the Rust
 * `Encoder::write_tables()` output can be compared byte-for-byte against a
 * real libjpeg.
 *
 * usage: tables_only_c_oracle <outfile> <quality> <h_samp> <v_samp>
 *
 * The sampling factors are accepted because the caller varies them; libjpeg's
 * tables-only stream carries no SOF, so they must not change the output. The
 * caller asserts exactly that.
 */
#include <stdio.h>
#include <stdlib.h>

#include "jpeglib.h"

int main(int argc, char **argv)
{
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  FILE *out;
  int quality, h_samp, v_samp;

  if (argc != 5) {
    fprintf(stderr, "usage: %s <outfile> <quality> <h_samp> <v_samp>\n", argv[0]);
    return 2;
  }
  quality = atoi(argv[2]);
  h_samp = atoi(argv[3]);
  v_samp = atoi(argv[4]);

  out = fopen(argv[1], "wb");
  if (out == NULL) {
    perror("fopen");
    return 1;
  }

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_stdio_dest(&cinfo, out);

  /* jpeg_set_defaults needs a colour space to derive component count from;
   * the dimensions are irrelevant to a tables-only stream. */
  cinfo.image_width = 64;
  cinfo.image_height = 64;
  cinfo.input_components = 3;
  cinfo.in_color_space = JCS_RGB;
  jpeg_set_defaults(&cinfo);
  /* force_baseline=FALSE is cjpeg's default ("by default, allow 16-bit
   * quantizers", cjpeg.c:321); only the explicit -baseline switch sets it
   * TRUE. Passing TRUE here would clamp low-quality tables to 255 and emit
   * 8-bit DQT, which is not what the reference encoder does at q=10. */
  jpeg_set_quality(&cinfo, quality, FALSE);

  cinfo.comp_info[0].h_samp_factor = h_samp;
  cinfo.comp_info[0].v_samp_factor = v_samp;
  cinfo.comp_info[1].h_samp_factor = 1;
  cinfo.comp_info[1].v_samp_factor = 1;
  cinfo.comp_info[2].h_samp_factor = 1;
  cinfo.comp_info[2].v_samp_factor = 1;

  jpeg_write_tables(&cinfo);
  jpeg_destroy_compress(&cinfo);
  /* In a fail-closed harness the oracle must be the strictest link: an
   * unchecked fclose lets a short write or a full disk produce a truncated
   * file and still exit 0, and the caller would then assert against it. */
  if (fclose(out) != 0) {
    perror("fclose");
    return 1;
  }
  return 0;
}

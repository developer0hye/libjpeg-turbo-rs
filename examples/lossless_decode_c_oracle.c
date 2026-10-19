/*
 * Reference oracle for lossless decode (`precision::decompress_lossless_arbitrary`).
 *
 * `decompress_lossless_arbitrary` returns a SOF3 frame's stored samples,
 * unconverted, at any precision from 2 to 16 bits. `djpeg` cannot stand in
 * for it: it converts colour (and libjpeg refuses RGB output from a lossless
 * YCbCr frame, P4-64), and it writes 12/16-bit output as PNM. This program
 * asks libjpeg for `out_color_space = jpeg_color_space` — a null conversion —
 * and reads scanlines through the API that matches `data_precision` (8-bit
 * for 2–8, 12-bit for 9–12, 16-bit for 13–16), for
 * `fuzz/fuzz_targets/fuzz_lossless_diff_c.rs`.
 *
 * usage: lossless_decode_c_oracle <infile>     ("/dev/stdin" reads a pipe)
 *
 * Output on stdout: `LL <width> <height> <components> <precision>\n`, then
 * the interleaved samples row by row as little-endian 16-bit values. A
 * 12-bit sample is libjpeg's `J12SAMPLE` (a short) reinterpreted as unsigned,
 * so a value libjpeg carries above 2^precision - 1 is reported as is
 * (P4-66). libjpeg's own error handler reports a refusal on stderr and exits
 * non-zero; its first warning, if any, also goes to stderr.
 */
#include <stdio.h>
#include <stdlib.h>

#include "jpeglib.h"

static int put_u16(unsigned int value)
{
  unsigned char bytes[2];
  bytes[0] = (unsigned char)(value & 0xFF);
  bytes[1] = (unsigned char)((value >> 8) & 0xFF);
  return fwrite(bytes, 1, 2, stdout) == 2;
}

int main(int argc, char **argv)
{
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  FILE *in;
  JDIMENSION row_samples, x;
  int precision;

  if (argc != 2) {
    fprintf(stderr, "usage: %s <infile>\n", argv[0]);
    return 2;
  }
  in = fopen(argv[1], "rb");
  if (in == NULL) {
    perror("fopen");
    return 1;
  }

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, in);
  jpeg_read_header(&cinfo, TRUE);
  cinfo.out_color_space = cinfo.jpeg_color_space;
  precision = cinfo.data_precision;
  jpeg_start_decompress(&cinfo);

  printf("LL %u %u %d %d\n", cinfo.output_width, cinfo.output_height,
         cinfo.output_components, precision);
  row_samples = cinfo.output_width * (JDIMENSION)cinfo.output_components;

  if (precision <= 8) {
    JSAMPARRAY row = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo,
                                                JPOOL_IMAGE, row_samples, 1);
    while (cinfo.output_scanline < cinfo.output_height) {
      jpeg_read_scanlines(&cinfo, row, 1);
      for (x = 0; x < row_samples; x++)
        if (!put_u16(row[0][x])) goto write_error;
    }
  } else if (precision <= 12) {
    J12SAMPLE *samples = (J12SAMPLE *)malloc(sizeof(J12SAMPLE) * row_samples);
    J12SAMPROW row[1];
    if (samples == NULL) {
      perror("malloc");
      return 1;
    }
    row[0] = samples;
    while (cinfo.output_scanline < cinfo.output_height) {
      jpeg12_read_scanlines(&cinfo, row, 1);
      for (x = 0; x < row_samples; x++)
        if (!put_u16((unsigned short)samples[x])) goto write_error;
    }
    free(samples);
  } else {
    J16SAMPLE *samples = (J16SAMPLE *)malloc(sizeof(J16SAMPLE) * row_samples);
    J16SAMPROW row[1];
    if (samples == NULL) {
      perror("malloc");
      return 1;
    }
    row[0] = samples;
    while (cinfo.output_scanline < cinfo.output_height) {
      jpeg16_read_scanlines(&cinfo, row, 1);
      for (x = 0; x < row_samples; x++)
        if (!put_u16(samples[x])) goto write_error;
    }
    free(samples);
  }

  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(in);
  /* A short write must fail the oracle, not hand the caller a truncated
   * reference that it would then compare against. */
  if (fflush(stdout) != 0) {
    perror("fflush");
    return 1;
  }
  return 0;

write_error:
  perror("fwrite");
  return 1;
}

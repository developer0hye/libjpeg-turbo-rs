/*
 * Reference oracle for 12-bit four-component (CMYK / YCCK) JPEG.
 *
 * Issue #694. `cjpeg` reads no four-component input and `djpeg` writes no
 * four-component output, so a 12-bit CMYK or YCCK stream has no command-line
 * oracle. This harness drives libjpeg's 12-bit API directly.
 *
 * Usage:
 *   cmyk12_c_oracle encode <width> <height> <quality> cmyk|ycck
 *     Reads interleaved C, M, Y, K samples (native-endian 16-bit, 0..4095)
 *     from stdin and writes a 12-bit JPEG to stdout. `ycck` stores the image
 *     as YCCK (Adobe transform 2), with libjpeg's default sampling for it
 *     (`jcparam.c` `jpeg_set_colorspace`: Y and K 2x2, Cb and Cr 1x1).
 *
 *   cmyk12_c_oracle decode <infile>
 *     Decodes with libjpeg's default output colour space (CMYK for both a
 *     CMYK and a YCCK source) and writes a text header
 *     `<width> <height> <components>\n`, then the interleaved samples
 *     (native-endian 16-bit) row by row.
 *
 * libjpeg's own error handler reports a failure on stderr and exits non-zero.
 *
 * Build (the Rust side does this automatically):
 *   cc -O2 -o cmyk12_c_oracle examples/cmyk12_c_oracle.c \
 *      -I<include-dir> -L<lib-dir> -ljpeg
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <jpeglib.h>

static int encode(int width, int height, int quality, const char *colorspace)
{
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  size_t row_samples = (size_t)width * 4;
  size_t total = row_samples * (size_t)height;
  J12SAMPLE *samples = malloc(total * sizeof(J12SAMPLE));
  unsigned char *jpeg = NULL;
  unsigned long jpeg_size = 0;
  J12SAMPROW row;

  if (!samples) {
    fprintf(stderr, "out of memory\n");
    return 1;
  }
  if (fread(samples, sizeof(J12SAMPLE), total, stdin) != total) {
    fprintf(stderr, "short read: expected %zu samples\n", total);
    free(samples);
    return 1;
  }

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_mem_dest(&cinfo, &jpeg, &jpeg_size);
  cinfo.image_width = (JDIMENSION)width;
  cinfo.image_height = (JDIMENSION)height;
  cinfo.input_components = 4;
  cinfo.in_color_space = JCS_CMYK;
  cinfo.data_precision = 12;
  jpeg_set_defaults(&cinfo);
  jpeg_set_colorspace(&cinfo, strcmp(colorspace, "ycck") ? JCS_CMYK : JCS_YCCK);
  jpeg_set_quality(&cinfo, quality, TRUE);
  jpeg_start_compress(&cinfo, TRUE);
  while (cinfo.next_scanline < cinfo.image_height) {
    row = samples + (size_t)cinfo.next_scanline * row_samples;
    jpeg12_write_scanlines(&cinfo, &row, 1);
  }
  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);

  fwrite(jpeg, 1, jpeg_size, stdout);
  free(jpeg);
  free(samples);
  return 0;
}

static int decode(const char *path)
{
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  FILE *in = fopen(path, "rb");
  J12SAMPLE *row_buffer;
  J12SAMPROW row;
  size_t row_samples;

  if (in == NULL) {
    perror("fopen");
    return 1;
  }
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, in);
  jpeg_read_header(&cinfo, TRUE);
  if (cinfo.data_precision != 12) {
    fprintf(stderr, "not a 12-bit JPEG (precision %d)\n", cinfo.data_precision);
    return 1;
  }
  jpeg_start_decompress(&cinfo);

  row_samples = (size_t)cinfo.output_width * cinfo.output_components;
  row_buffer = malloc(row_samples * sizeof(J12SAMPLE));
  if (!row_buffer) {
    fprintf(stderr, "out of memory\n");
    return 1;
  }
  printf("%u %u %d\n", cinfo.output_width, cinfo.output_height,
         cinfo.output_components);
  while (cinfo.output_scanline < cinfo.output_height) {
    row = row_buffer;
    jpeg12_read_scanlines(&cinfo, &row, 1);
    fwrite(row_buffer, sizeof(J12SAMPLE), row_samples, stdout);
  }
  jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
  fclose(in);
  free(row_buffer);
  return 0;
}

int main(int argc, char **argv)
{
  if (argc == 6 && !strcmp(argv[1], "encode")) {
    int width = atoi(argv[2]);
    int height = atoi(argv[3]);
    int quality = atoi(argv[4]);
    if (width <= 0 || height <= 0 ||
        (strcmp(argv[5], "cmyk") && strcmp(argv[5], "ycck"))) {
      fprintf(stderr, "bad encode arguments\n");
      return 2;
    }
    return encode(width, height, quality, argv[5]);
  }
  if (argc == 3 && !strcmp(argv[1], "decode"))
    return decode(argv[2]);
  fprintf(stderr,
          "usage: %s encode <width> <height> <quality> cmyk|ycck\n"
          "       %s decode <infile>\n",
          argv[0], argv[0]);
  return 2;
}

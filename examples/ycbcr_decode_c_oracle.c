/*
 * Reference oracle for YCbCr output (issue #683, P4-244).
 *
 * libjpeg can return a YCbCr image as YCbCr: set `out_color_space =
 * JCS_YCbCr` and the decoder runs its normal upsampler, then `null_convert`
 * (`jdcolor.c`) copies the planes out untouched. Neither `djpeg` nor
 * TurboJPEG exposes that request, so the Rust
 * `Decoder::set_output_colorspace(ColorSpace::YCbCr)` path had no C
 * reference, and a nearest-neighbour chroma copy that differs from libjpeg on
 * most samples of every subsampled image went unnoticed.
 *
 * usage: ycbcr_decode_c_oracle <infile> <scale_num> <scale_denom> <fancy>
 *                              [<crop_x> <crop_y> <crop_w> <crop_h>]
 *
 * <fancy> is `do_fancy_upsampling` (1 = libjpeg's default, 0 = what
 * `Decoder::set_fast_upsample(true)` asks for). The optional crop follows
 * `djpeg -crop`: `jpeg_crop_scanline` aligns X down and widens the region,
 * `jpeg_skip_scanlines` skips the rows above and below.
 *
 * Output on stdout: a text header `YCbCr <width> <height>\n`, then the
 * interleaved Y, Cb, Cr samples row by row. libjpeg's own error handler
 * reports a refusal (e.g. JERR_CONVERSION_NOTIMPL for an RGB source) on
 * stderr and exits non-zero.
 */
#include <stdio.h>
#include <stdlib.h>

#include "jpeglib.h"

int main(int argc, char **argv)
{
  struct jpeg_decompress_struct cinfo;
  struct jpeg_error_mgr jerr;
  FILE *in;
  JSAMPARRAY row;
  JDIMENSION crop_x = 0, crop_y = 0, crop_w = 0, crop_h = 0;
  JDIMENSION first_row, end_row, skipped, row_bytes;
  int crop;

  if (argc != 5 && argc != 9) {
    fprintf(stderr,
            "usage: %s <infile> <scale_num> <scale_denom> <fancy> "
            "[<crop_x> <crop_y> <crop_w> <crop_h>]\n",
            argv[0]);
    return 2;
  }
  crop = argc == 9;

  in = fopen(argv[1], "rb");
  if (in == NULL) {
    perror("fopen");
    return 1;
  }

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_decompress(&cinfo);
  jpeg_stdio_src(&cinfo, in);
  jpeg_read_header(&cinfo, TRUE);

  cinfo.out_color_space = JCS_YCbCr;
  cinfo.scale_num = (unsigned int)atoi(argv[2]);
  cinfo.scale_denom = (unsigned int)atoi(argv[3]);
  cinfo.do_fancy_upsampling = atoi(argv[4]) ? TRUE : FALSE;
  jpeg_start_decompress(&cinfo);

  first_row = 0;
  end_row = cinfo.output_height;
  if (crop) {
    crop_x = (JDIMENSION)atoi(argv[5]);
    crop_y = (JDIMENSION)atoi(argv[6]);
    crop_w = (JDIMENSION)atoi(argv[7]);
    crop_h = (JDIMENSION)atoi(argv[8]);
    if (crop_x + crop_w > cinfo.output_width ||
        crop_y + crop_h > cinfo.output_height) {
      fprintf(stderr, "crop region exceeds the %ux%u output\n",
              cinfo.output_width, cinfo.output_height);
      return 1;
    }
    jpeg_crop_scanline(&cinfo, &crop_x, &crop_w);
    first_row = crop_y;
    end_row = crop_y + crop_h;
    skipped = jpeg_skip_scanlines(&cinfo, first_row);
    if (skipped != first_row) {
      fprintf(stderr, "skipped %u rows, wanted %u\n", skipped, first_row);
      return 1;
    }
  }

  printf("YCbCr %u %u\n", cinfo.output_width, end_row - first_row);
  row_bytes = cinfo.output_width * (JDIMENSION)cinfo.output_components;
  row = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE,
                                   row_bytes, 1);
  while (cinfo.output_scanline < end_row) {
    jpeg_read_scanlines(&cinfo, row, 1);
    if (fwrite(row[0], 1, row_bytes, stdout) != row_bytes) {
      perror("fwrite");
      return 1;
    }
  }
  if (cinfo.output_scanline < cinfo.output_height)
    jpeg_skip_scanlines(&cinfo, cinfo.output_height - cinfo.output_scanline);

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
}

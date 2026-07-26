/*
 * Byte-exact reference oracle for four-component CMYK encoding.
 *
 * Issue #313. Every other encode cross-check in this tree shells out to
 * `cjpeg`, but cjpeg reads only PNM/BMP/GIF/Targa — it cannot ingest CMYK at
 * all, so the CMYK path has never had a C oracle and its option handling was
 * only ever checked against itself. This harness closes that gap by driving
 * libjpeg directly with `JCS_CMYK`, mirroring TurboJPEG's component layout
 * (`turbojpeg.c:418-427`): components 0 and 3 carry the sampling factors,
 * components 1 and 2 stay at 1x1, and all four share quantization and Huffman
 * table slot 0 (`jcparam.c:383-390`).
 *
 * Reads raw interleaved CMYK bytes from stdin, writes a JPEG to stdout.
 *
 * Usage:
 *   cmyk_encode_c_oracle <width> <height> <quality> <h_samp> <v_samp>
 *                        [--optimize] [--smooth N] [--restart N]
 *                        [--dct int|fast|float]
 *
 * Build (paths vary; the Rust side does this automatically):
 *   cc -O2 -o cmyk_encode_c_oracle examples/cmyk_encode_c_oracle.c \
 *      -I<include-dir> -L<lib-dir> -ljpeg
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <jpeglib.h>

int main(int argc, char **argv)
{
  if (argc < 6) {
    fprintf(stderr,
            "usage: %s <width> <height> <quality> <h_samp> <v_samp> "
            "[--optimize] [--smooth N] [--restart N] [--dct int|fast|float]\n",
            argv[0]);
    return 2;
  }

  int width = atoi(argv[1]);
  int height = atoi(argv[2]);
  int quality = atoi(argv[3]);
  int h_samp = atoi(argv[4]);
  int v_samp = atoi(argv[5]);

  int optimize = 0;
  int smoothing = 0;
  int restart = 0;
  J_DCT_METHOD dct_method = JDCT_ISLOW;

  for (int i = 6; i < argc; i++) {
    if (!strcmp(argv[i], "--optimize")) {
      optimize = 1;
    } else if (!strcmp(argv[i], "--smooth") && i + 1 < argc) {
      smoothing = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--restart") && i + 1 < argc) {
      restart = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "--dct") && i + 1 < argc) {
      const char *name = argv[++i];
      if (!strcmp(name, "int")) dct_method = JDCT_ISLOW;
      else if (!strcmp(name, "fast")) dct_method = JDCT_IFAST;
      else if (!strcmp(name, "float")) dct_method = JDCT_FLOAT;
      else { fprintf(stderr, "unknown --dct %s\n", name); return 2; }
    } else {
      fprintf(stderr, "unknown argument %s\n", argv[i]);
      return 2;
    }
  }

  if (width <= 0 || height <= 0) {
    fprintf(stderr, "bad dimensions %dx%d\n", width, height);
    return 2;
  }

  size_t row_stride = (size_t)width * 4;
  size_t total = row_stride * (size_t)height;
  unsigned char *pixels = malloc(total);
  if (!pixels) {
    fprintf(stderr, "out of memory\n");
    return 1;
  }
  if (fread(pixels, 1, total, stdin) != total) {
    fprintf(stderr, "short read: expected %zu bytes of CMYK\n", total);
    free(pixels);
    return 1;
  }

  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);

  unsigned char *outbuf = NULL;
  unsigned long outsize = 0;
  jpeg_mem_dest(&cinfo, &outbuf, &outsize);

  cinfo.image_width = (JDIMENSION)width;
  cinfo.image_height = (JDIMENSION)height;
  cinfo.input_components = 4;
  cinfo.in_color_space = JCS_CMYK;
  jpeg_set_defaults(&cinfo);
  /* jpeg_set_defaults() already picks JCS_CMYK from in_color_space, but call
     it explicitly so the component setup is unambiguous and matches what
     TurboJPEG does. This also clears write_JFIF_header — a CMYK stream
     carries only the Adobe APP14 marker. */
  jpeg_set_colorspace(&cinfo, JCS_CMYK);

  jpeg_set_quality(&cinfo, quality, TRUE /* force_baseline */);

  /* TurboJPEG's layout: the sampling factors ride on components 0 and 3. */
  cinfo.comp_info[0].h_samp_factor = h_samp;
  cinfo.comp_info[0].v_samp_factor = v_samp;
  cinfo.comp_info[1].h_samp_factor = 1;
  cinfo.comp_info[1].v_samp_factor = 1;
  cinfo.comp_info[2].h_samp_factor = 1;
  cinfo.comp_info[2].v_samp_factor = 1;
  cinfo.comp_info[3].h_samp_factor = h_samp;
  cinfo.comp_info[3].v_samp_factor = v_samp;

  cinfo.optimize_coding = optimize ? TRUE : FALSE;
  cinfo.smoothing_factor = smoothing;
  cinfo.restart_interval = restart;
  cinfo.dct_method = dct_method;

  jpeg_start_compress(&cinfo, TRUE);
  while (cinfo.next_scanline < cinfo.image_height) {
    JSAMPROW row = pixels + (size_t)cinfo.next_scanline * row_stride;
    jpeg_write_scanlines(&cinfo, &row, 1);
  }
  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
  free(pixels);

  if (fwrite(outbuf, 1, outsize, stdout) != outsize) {
    fprintf(stderr, "short write\n");
    free(outbuf);
    return 1;
  }
  free(outbuf);
  return 0;
}

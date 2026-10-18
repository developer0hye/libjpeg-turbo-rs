/* Issue #673: stock libjpeg API oracle for CMYK and installed Huffman tables.
 * Raw pixels on stdin; JPEG on stdout. No Rust library is linked.
 * Args: width height cmyk sample quality progressive arithmetic smooth
 *       restart_blocks restart_rows custom_tables component_script optimize
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <jpeglib.h>

int main(int argc, char **argv) {
  if (argc != 14) return 2;
  struct jpeg_compress_struct c;
  struct jpeg_error_mgr e;
  c.err = jpeg_std_error(&e);
  jpeg_create_compress(&c);
  c.image_width = atoi(argv[1]);
  c.image_height = atoi(argv[2]);
  int cmyk = atoi(argv[3]);
  c.input_components = cmyk ? 4 : 3;
  c.in_color_space = cmyk ? JCS_CMYK : JCS_RGB;
  jpeg_set_defaults(&c);
  jpeg_set_quality(&c, atoi(argv[5]), FALSE);
  const char *f = argv[4];
  for (int i = 0; i < c.num_components; ++i) {
    int h, v;
    if (!f || sscanf(f, "%dx%d", &h, &v) != 2) return 2;
    c.comp_info[i].h_samp_factor = h;
    c.comp_info[i].v_samp_factor = v;
    f = strchr(f, ',');
    if (f) ++f;
  }
  if (atoi(argv[6])) jpeg_simple_progression(&c);
  c.arith_code = atoi(argv[7]);
  c.smoothing_factor = atoi(argv[8]);
  c.restart_interval = atoi(argv[9]);
  c.restart_in_rows = atoi(argv[10]);
  if (atoi(argv[11])) {
    for (int i = 0; i < (cmyk ? 1 : 2); ++i) {
      JHUFF_TBL *tables[] = { c.dc_huff_tbl_ptrs[i], c.ac_huff_tbl_ptrs[i] };
      for (int j = 0; j < 2; ++j) {
        unsigned char v = tables[j]->huffval[0];
        tables[j]->huffval[0] = tables[j]->huffval[1];
        tables[j]->huffval[1] = v;
      }
    }
  }
  jpeg_scan_info scans[12];
  if (atoi(argv[12])) {
    memset(scans, 0, sizeof(scans));
    int n = 0;
    for (int i = c.num_components - 1; i >= 0; --i) {
      scans[n].comps_in_scan = 1; scans[n].component_index[0] = i;
      scans[n++].Al = 1;
    }
    for (int i = c.num_components - 1; i >= 0; --i) {
      scans[n].comps_in_scan = 1; scans[n].component_index[0] = i;
      scans[n].Ss = 1; scans[n++].Se = 63;
      scans[n].comps_in_scan = 1; scans[n].component_index[0] = i;
      scans[n++].Ah = 1;
    }
    c.scan_info = scans; c.num_scans = n;
  }
  c.optimize_coding = atoi(argv[13]);
  size_t stride = (size_t)c.image_width * c.input_components;
  unsigned char *row = malloc(stride);
  if (!row) return 2;
  jpeg_stdio_dest(&c, stdout);
  jpeg_start_compress(&c, TRUE);
  while (c.next_scanline < c.image_height) {
    if (fread(row, 1, stride, stdin) != stride) return 2;
    JSAMPROW rows[] = { row };
    jpeg_write_scanlines(&c, rows, 1);
  }
  jpeg_finish_compress(&c);
  jpeg_destroy_compress(&c);
  free(row);
  return 0;
}

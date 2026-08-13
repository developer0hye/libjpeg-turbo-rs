/*
 * P4-14 (#467) C oracle: where the classic decode sequence enforces
 * `cinfo->mem->max_memory_to_use`.
 *
 * Upstream consults the field through `jpeg_mem_available`
 * (`jmemnobs.c:66-78`): with no backing store, a budget the realized virtual
 * arrays cannot fit raises `JERR_NO_BACKING_STORE` (51). Which *sequences*
 * that bounds is the measured part: a baseline single-scan decode needs no
 * whole-image coefficient array, so a tiny budget may pass where a
 * progressive decode — which realizes full-image arrays — fails.
 *
 * The exact byte threshold is upstream's own accounting and is NOT compared
 * (the port's estimation model is documented as coarser — see the P4-14
 * PARTIAL note); the budgets below sit far on either side of both models:
 * 1000 bytes cannot hold any 64x64 decode state, 100 MB holds all of it.
 *
 * Each line is `case stage code`, where stage is which call raised
 * ("header" / "start" / "scan") or "ok", and code is err->msg_code at the
 * first failure (0 when none).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>

#include <jpeglib.h>

#define WIDTH 64
#define HEIGHT 64

struct trap_error_mgr {
  struct jpeg_error_mgr pub;
  jmp_buf escape;
  int fired;
  int msg_code;
};

static void trap_error_exit(j_common_ptr cinfo)
{
  struct trap_error_mgr *e = (struct trap_error_mgr *)cinfo->err;
  e->fired = 1;
  e->msg_code = cinfo->err->msg_code;
  longjmp(e->escape, 1);
}

static void quiet_output_message(j_common_ptr cinfo) { (void)cinfo; }

/* An in-memory grayscale JPEG, optionally progressive. */
static unsigned char *make_source(int progressive, unsigned long *out_size)
{
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  unsigned char *buf = NULL;
  unsigned long size = 0;
  JSAMPLE row[WIDTH];
  JSAMPROW rows[1];
  int r, i;

  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_mem_dest(&cinfo, &buf, &size);
  cinfo.image_width = WIDTH;
  cinfo.image_height = HEIGHT;
  cinfo.input_components = 1;
  cinfo.in_color_space = JCS_GRAYSCALE;
  jpeg_set_defaults(&cinfo);
  if (progressive)
    jpeg_simple_progression(&cinfo);
  jpeg_start_compress(&cinfo, TRUE);
  for (r = 0; r < HEIGHT; r++) {
    for (i = 0; i < WIDTH; i++) row[i] = (JSAMPLE)((r * 7 + i * 13) & 0xFF);
    rows[0] = row;
    (void)jpeg_write_scanlines(&cinfo, rows, 1);
  }
  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
  *out_size = size;
  return buf;
}

static void run_case(const char *label, int progressive, long budget)
{
  struct jpeg_decompress_struct cinfo;
  struct trap_error_mgr jerr;
  unsigned long src_size = 0;
  unsigned char *src = make_source(progressive, &src_size);
  JSAMPLE row[WIDTH * 4];
  JSAMPROW rows[1];

  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;

  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, src, src_size);

  if (setjmp(jerr.escape) != 0) {
    printf("%s header %d\n", label, jerr.msg_code);
    goto done;
  }
  (void)jpeg_read_header(&cinfo, TRUE);

  /* The caller sets the field after create/header, as libjpeg.txt
   * documents; the sequence below is the one #467 names. */
  cinfo.mem->max_memory_to_use = budget;

  if (setjmp(jerr.escape) != 0) {
    printf("%s start %d\n", label, jerr.msg_code);
    goto done;
  }
  (void)jpeg_start_decompress(&cinfo);

  if (setjmp(jerr.escape) != 0) {
    printf("%s scan %d\n", label, jerr.msg_code);
    goto done;
  }
  while (cinfo.output_scanline < cinfo.output_height) {
    rows[0] = row;
    (void)jpeg_read_scanlines(&cinfo, rows, 1);
  }
  (void)jpeg_finish_decompress(&cinfo);
  printf("%s ok 0\n", label);

done:
  jpeg_destroy_decompress(&cinfo);
  free(src);
}

int main(void)
{
  run_case("baseline_unlimited", 0, 0);
  run_case("baseline_tiny", 0, 1000);
  run_case("baseline_generous", 0, 100L * 1024L * 1024L);
  run_case("progressive_unlimited", 1, 0);
  run_case("progressive_tiny", 1, 1000);
  run_case("progressive_generous", 1, 100L * 1024L * 1024L);
  return 0;
}

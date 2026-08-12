/*
 * P4-154 (#538) C oracle: where the classic API refuses a data_precision.
 *
 * Upstream has two gates, in two different calls:
 *   - `jpeg_start_compress` -> jcmaster.c initial_setup: lossy admits 8 or
 *     12, lossless 2..=16 (`jcmaster.c:196-208`, ERREXIT1 with the value);
 *   - the 8-bit `jpeg_write_scanlines` entry (`jcapistd.c:92-105`): lossy
 *     requires precision == BITS_IN_JSAMPLE (8), lossless 2..=8.
 *
 * Which one a caller actually sees, per (precision, lossless) pair, is what
 * this binary prints — the two gates disagree about 12 (start accepts it for
 * lossy, write refuses it), so transcription is exactly the shortcut that
 * goes wrong. The Rust suite compares its own trace line for line.
 *
 * Each line is `case stage code parm0`, where stage is which call raised
 * ("start" / "write") or "ok", code is err->msg_code at the first failure
 * (0 when none), and parm0 is msg_parm.i[0] — ERREXIT1's offending value.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>

#include <jpeglib.h>

#define WIDTH 8
#define HEIGHT 8

struct trap_error_mgr {
  struct jpeg_error_mgr pub;
  jmp_buf escape;
  int fired;
  int msg_code;
  int parm0;
};

static void trap_error_exit(j_common_ptr cinfo)
{
  struct trap_error_mgr *e = (struct trap_error_mgr *)cinfo->err;
  e->fired = 1;
  e->msg_code = cinfo->err->msg_code;
  e->parm0 = cinfo->err->msg_parm.i[0];
  longjmp(e->escape, 1);
}

static void quiet_output_message(j_common_ptr cinfo) { (void)cinfo; }

static void run_case(const char *label, int precision, int lossless)
{
  struct jpeg_compress_struct cinfo;
  struct trap_error_mgr jerr;
  unsigned char *out = NULL;
  unsigned long out_size = 0;
  JSAMPLE row[WIDTH];
  JSAMPROW rows[1];
  int r;

  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;

  jpeg_create_compress(&cinfo);
  jpeg_mem_dest(&cinfo, &out, &out_size);
  cinfo.image_width = WIDTH;
  cinfo.image_height = HEIGHT;
  cinfo.input_components = 1;
  cinfo.in_color_space = JCS_GRAYSCALE;
  jpeg_set_defaults(&cinfo);
  cinfo.data_precision = precision;
  if (lossless)
    jpeg_enable_lossless(&cinfo, 1, 0);

  if (setjmp(jerr.escape) != 0) {
    printf("%s start %d %d\n", label, jerr.msg_code, jerr.parm0);
    goto done;
  }
  jpeg_start_compress(&cinfo, TRUE);

  if (setjmp(jerr.escape) != 0) {
    printf("%s write %d %d\n", label, jerr.msg_code, jerr.parm0);
    goto done;
  }
  for (r = 0; r < HEIGHT; r++) {
    int i;
    for (i = 0; i < WIDTH; i++) row[i] = (JSAMPLE)(r * WIDTH + i);
    rows[0] = row;
    (void)jpeg_write_scanlines(&cinfo, rows, 1);
  }

  if (setjmp(jerr.escape) != 0) {
    printf("%s finish %d %d\n", label, jerr.msg_code, jerr.parm0);
    goto done;
  }
  jpeg_finish_compress(&cinfo);
  /* The accepted cases print the exact output size and a byte checksum, so a
   * port that accepts a precision and then encodes a different one is caught
   * by the trace: SOF3's precision byte and the predictor arithmetic both
   * feed the sum (#538 review - the first version printed only "ok" and an
   * accepted 2-bit lossless request that produced an 8-bit stream passed). */
  {
    unsigned long sum = 0;
    unsigned long i;
    for (i = 0; i < out_size; i++) sum = (sum + out[i]) & 0xFFFFFFFFUL;
    printf("%s ok %lu %lu\n", label, out_size, sum);
  }

done:
  jpeg_destroy_compress(&cinfo);
  if (out) free(out);
}

/* Width representable, row not: 65500 * 65573 samples > 0xFFFFFFFF. The
 * precision is also invalid (9, lossy), so the line pins which gate fires. */
static void run_width_overflow_case(const char *label)
{
  struct jpeg_compress_struct cinfo;
  struct trap_error_mgr jerr;
  unsigned char *out = NULL;
  unsigned long out_size = 0;

  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;

  jpeg_create_compress(&cinfo);
  jpeg_mem_dest(&cinfo, &out, &out_size);
  cinfo.image_width = 65500;
  cinfo.image_height = 1;
  cinfo.input_components = 1;
  cinfo.in_color_space = JCS_GRAYSCALE;
  jpeg_set_defaults(&cinfo);
  cinfo.input_components = 65573; /* after set_defaults, as a caller can */
  cinfo.data_precision = 9;

  if (setjmp(jerr.escape) != 0) {
    printf("%s start %d %d\n", label, jerr.msg_code, jerr.parm0);
    goto done;
  }
  jpeg_start_compress(&cinfo, TRUE);
  printf("%s accepted 0 0\n", label);

done:
  jpeg_destroy_compress(&cinfo);
  if (out) free(out);
}

int main(void)
{
  /* Lossy: jcmaster admits {8, 12}; the 8-bit write entry admits only 8. */
  run_case("lossy_2", 2, 0);
  run_case("lossy_8", 8, 0);
  run_case("lossy_9", 9, 0);
  run_case("lossy_12", 12, 0);
  run_case("lossy_16", 16, 0);
  /* Lossless: jcmaster admits 2..=16; the 8-bit write entry admits 2..=8. */
  run_case("lossless_2", 2, 1);
  run_case("lossless_8", 8, 1);
  run_case("lossless_9", 9, 1);
  run_case("lossless_12", 12, 1);
  run_case("lossless_16", 16, 1);
  /* Error precedence for a doubly-invalid setup: a row too wide for
   * JDIMENSION *and* a bad precision. Upstream's initial_setup raises
   * JERR_WIDTH_OVERFLOW first (jcmaster.c:190-208); a port that checks
   * precision first reports the wrong error. */
  run_width_overflow_case("mixed_width_overflow_precision_9");
  return 0;
}

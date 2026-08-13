/*
 * P4-14 (#467) C oracle: where the classic decode sequence enforces
 * `cinfo->mem->max_memory_to_use`.
 *
 * Upstream consults the field in exactly one place: `realize_virt_arrays`
 * (`jmemmgr.c:696-745`) via `jpeg_mem_available` (`jmemnobs.c:66-78`),
 * during `jpeg_start_decompress` master selection. Whole-image coefficient
 * arrays exist only for multi-scan streams — progressive *or*
 * non-interleaved sequential, `has_multiple_scans` per `jdinput.c:153-156`
 * — and for any stream in buffered-image mode (`jdmaster.c:709`). With no
 * backing store, a budget those arrays cannot fit raises
 * `JERR_NO_BACKING_STORE` (51) at `jpeg_start_decompress`; everything else
 * is unbounded, so a tiny budget passes a baseline single-scan decode
 * untouched.
 *
 * What upstream counts is *only* the virtual coefficient arrays (plus the
 * small amount already allocated) — not output buffers or component
 * planes. The big_progressive_midband case pins that: a budget above the
 * coefficient bytes but below a whole-pipeline estimate must ACCEPT
 * (review of e492da9 measured a 4.7x over-refusal band when the port used
 * its pipeline estimate for this comparison). The tiny/tight budgets sit
 * far below both models; the generous ones far above.
 *
 * Each line is `case stage code`, where stage is which call raised
 * ("header" / "start" / "scan") or "ok", and code is err->msg_code at the
 * first failure (0 when none). The trailing `hms_*` lines print
 * `jpeg_has_multiple_scans()` after `jpeg_read_header` for each source
 * shape — TRUE for progressive AND for non-interleaved sequential.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>

#include <jpeglib.h>

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

/* Source shapes. */
enum src_kind {
  SRC_BASELINE,        /* single-scan grayscale 64x64 */
  SRC_PROGRESSIVE,     /* progressive grayscale 64x64 */
  SRC_MSS,             /* multi-scan sequential RGB 64x64 (one comp per scan) */
  SRC_BIG_PROGRESSIVE, /* progressive grayscale 1024x1024 */
  SRC_SHORT_PROGRESSIVE /* progressive grayscale 64x8: one access window */
};

/* An in-memory JPEG of the requested shape. */
static unsigned char *make_source(enum src_kind kind, unsigned long *out_size)
{
  struct jpeg_compress_struct cinfo;
  struct jpeg_error_mgr jerr;
  unsigned char *buf = NULL;
  unsigned long size = 0;
  int width = (kind == SRC_BIG_PROGRESSIVE) ? 1024 : 64;
  int height = (kind == SRC_SHORT_PROGRESSIVE) ? 8 : width;
  int color = (kind == SRC_MSS);
  JSAMPLE *row;
  JSAMPROW rows[1];
  int r, i;
  /* Non-interleaved sequential: one full-spectral scan per component,
   * `Ss=0 Se=63 Ah=Al=0` — the shape `cjpeg -scans` produces. */
  static const jpeg_scan_info mss_scans[3] = {
    { 1, { 0 }, 0, 63, 0, 0 },
    { 1, { 1 }, 0, 63, 0, 0 },
    { 1, { 2 }, 0, 63, 0, 0 },
  };

  row = malloc((size_t)width * (color ? 3 : 1));
  cinfo.err = jpeg_std_error(&jerr);
  jpeg_create_compress(&cinfo);
  jpeg_mem_dest(&cinfo, &buf, &size);
  cinfo.image_width = width;
  cinfo.image_height = height;
  cinfo.input_components = color ? 3 : 1;
  cinfo.in_color_space = color ? JCS_RGB : JCS_GRAYSCALE;
  jpeg_set_defaults(&cinfo);
  if (kind == SRC_PROGRESSIVE || kind == SRC_BIG_PROGRESSIVE ||
      kind == SRC_SHORT_PROGRESSIVE)
    jpeg_simple_progression(&cinfo);
  if (kind == SRC_MSS) {
    cinfo.scan_info = mss_scans;
    cinfo.num_scans = 3;
  }
  jpeg_start_compress(&cinfo, TRUE);
  for (r = 0; r < height; r++) {
    if (color) {
      for (i = 0; i < width; i++) {
        row[i * 3 + 0] = (JSAMPLE)((r * 7 + i * 13) & 0xFF);
        row[i * 3 + 1] = (JSAMPLE)((r * 3 + i * 5) & 0xFF);
        row[i * 3 + 2] = (JSAMPLE)((r * 11 + i * 2) & 0xFF);
      }
    } else {
      for (i = 0; i < width; i++) row[i] = (JSAMPLE)((r * 7 + i * 13) & 0xFF);
    }
    rows[0] = row;
    (void)jpeg_write_scanlines(&cinfo, rows, 1);
  }
  jpeg_finish_compress(&cinfo);
  jpeg_destroy_compress(&cinfo);
  free(row);
  *out_size = size;
  return buf;
}

static void run_case(const char *label, enum src_kind kind, long budget,
                     int buffered)
{
  struct jpeg_decompress_struct cinfo;
  struct trap_error_mgr jerr;
  unsigned long src_size = 0;
  unsigned char *src = make_source(kind, &src_size);
  JSAMPLE *row = malloc(4096 * 4);
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
  if (buffered)
    cinfo.buffered_image = TRUE;

  if (setjmp(jerr.escape) != 0) {
    printf("%s start %d\n", label, jerr.msg_code);
    goto done;
  }
  (void)jpeg_start_decompress(&cinfo);

  if (setjmp(jerr.escape) != 0) {
    printf("%s scan %d\n", label, jerr.msg_code);
    goto done;
  }
  if (buffered) {
    /* The documented buffered-image loop (libjpeg.txt). */
    do {
      (void)jpeg_start_output(&cinfo, cinfo.input_scan_number);
      while (cinfo.output_scanline < cinfo.output_height) {
        rows[0] = row;
        (void)jpeg_read_scanlines(&cinfo, rows, 1);
      }
      (void)jpeg_finish_output(&cinfo);
    } while (!jpeg_input_complete(&cinfo));
  } else {
    while (cinfo.output_scanline < cinfo.output_height) {
      rows[0] = row;
      (void)jpeg_read_scanlines(&cinfo, rows, 1);
    }
  }
  (void)jpeg_finish_decompress(&cinfo);
  printf("%s ok 0\n", label);

done:
  jpeg_destroy_decompress(&cinfo);
  free(src);
  free(row);
}

/* `jpeg_has_multiple_scans()` after `jpeg_read_header` for one source. */
static void print_hms(const char *label, enum src_kind kind)
{
  struct jpeg_decompress_struct cinfo;
  struct trap_error_mgr jerr;
  unsigned long src_size = 0;
  unsigned char *src = make_source(kind, &src_size);

  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;

  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, src, src_size);
  if (setjmp(jerr.escape) != 0) {
    printf("%s parse_failed %d\n", label, jerr.msg_code);
  } else {
    (void)jpeg_read_header(&cinfo, TRUE);
    printf("%s %d\n", label, jpeg_has_multiple_scans(&cinfo) ? 1 : 0);
  }
  jpeg_destroy_decompress(&cinfo);
  free(src);
}

int main(void)
{
  run_case("baseline_unlimited", SRC_BASELINE, 0, 0);
  run_case("baseline_tiny", SRC_BASELINE, 1000, 0);
  run_case("baseline_generous", SRC_BASELINE, 100L * 1024L * 1024L, 0);
  run_case("progressive_unlimited", SRC_PROGRESSIVE, 0, 0);
  run_case("progressive_tiny", SRC_PROGRESSIVE, 1000, 0);
  run_case("progressive_generous", SRC_PROGRESSIVE, 100L * 1024L * 1024L, 0);
  run_case("mss_tiny", SRC_MSS, 1000, 0);
  run_case("mss_generous", SRC_MSS, 100L * 1024L * 1024L, 0);
  run_case("buffered_baseline_tiny", SRC_BASELINE, 1000, 1);
  run_case("buffered_baseline_generous", SRC_BASELINE, 100L * 1024L * 1024L, 1);
  /* 1024x1024 gray progressive: coefficient arrays are exactly 2 MiB
   * (128x128 blocks x 128 bytes). 1 MiB refuses under both models; 4 MiB
   * accepts upstream but sat inside the port's 4.7x over-refusal band —
   * the P1-3 regression row. Budgets near 2 MiB + already-allocated
   * overhead are deliberately avoided (the exact threshold is upstream's
   * own accounting and is not compared). */
  run_case("big_progressive_tight", SRC_BIG_PROGRESSIVE, 1L * 1024L * 1024L, 0);
  run_case("big_progressive_midband", SRC_BIG_PROGRESSIVE, 4L * 1024L * 1024L, 0);
  /* 64x8 gray progressive: one row of blocks, so every coefficient array
   * fits a single access window (maxaccess = v_samp x 5 for progressive,
   * jdcoefct.c:897-901) and realize_virt_arrays' max_minheights >= 1 clamp
   * (jmemmgr.c:753-762) realizes it in full at ANY positive budget — even
   * one below the 1024-byte coefficient footprint. A pure
   * footprint-vs-budget model wrongly refuses this (P4-14 verification
   * round 2). */
  run_case("short_progressive_tiny", SRC_SHORT_PROGRESSIVE, 1000, 0);
  print_hms("hms_baseline", SRC_BASELINE);
  print_hms("hms_progressive", SRC_PROGRESSIVE);
  print_hms("hms_mss", SRC_MSS);
  return 0;
}

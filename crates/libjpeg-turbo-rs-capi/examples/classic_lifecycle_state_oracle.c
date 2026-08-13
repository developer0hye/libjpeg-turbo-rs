/*
 * P4-104 / P4-106 (#468) C oracle: classic state transitions and the
 * finish/abort lifecycle, both directions.
 *
 * Upstream contracts measured here (jdapimin.c / jcapimin.c):
 *
 * - `jpeg_read_header` requires START/INHEADER (else JERR_BAD_STATE) and a
 *   successful return leaves DSTATE_READY (202) — the consume path sets it,
 *   so the *direct* call must match the consume-driven one.
 * - `jpeg_finish_decompress` in SCANNING/RAW_OK (non-buffered) raises
 *   JERR_TOO_LITTLE_DATA on unread rows, else finishes, drains to EOI,
 *   calls `term_source` exactly once, and `jpeg_abort`s back to
 *   DSTATE_START (200). From any state other than those, BUFIMAGE and
 *   STOPPING it raises JERR_BAD_STATE. A second finish therefore raises
 *   BAD_STATE. (BUFIMAGE is not exercised here — these cases are all
 *   non-buffered.)
 * - `jpeg_finish_compress` in SCANNING/RAW_OK raises JERR_TOO_LITTLE_DATA
 *   when `next_scanline < image_height`; from any state other than those
 *   or WRCOEFS it raises JERR_BAD_STATE(state). After an error trapped by
 *   the caller, `jpeg_abort` returns the object to a reusable state and a
 *   full compress must then produce a valid JPEG.
 *
 * Line format: `case token...` — tokens are stage/msg_code/global_state
 * observations named per case. Trap handler longjmps, so each case
 * observes the FIRST error only.
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
  int msg_parm0;
};

static void trap_error_exit(j_common_ptr cinfo)
{
  struct trap_error_mgr *e = (struct trap_error_mgr *)cinfo->err;
  e->fired = 1;
  e->msg_code = cinfo->err->msg_code;
  e->msg_parm0 = cinfo->err->msg_parm.i[0];
  longjmp(e->escape, 1);
}

static void quiet_output_message(j_common_ptr cinfo) { (void)cinfo; }

/* Counting source manager over a memory buffer: observable term_source. */
struct counting_src_mgr {
  struct jpeg_source_mgr pub;
  const unsigned char *data;
  unsigned long size;
  int term_calls;
};

static void csrc_init(j_decompress_ptr cinfo) { (void)cinfo; }

static boolean csrc_fill(j_decompress_ptr cinfo)
{
  /* Whole buffer handed out at init; a refill request means EOF. Fabricate
   * EOI exactly as stock's mem source does. */
  static const JOCTET eoi[2] = { 0xFF, 0xD9 };
  struct counting_src_mgr *s = (struct counting_src_mgr *)cinfo->src;
  s->pub.next_input_byte = eoi;
  s->pub.bytes_in_buffer = 2;
  return TRUE;
}

static void csrc_skip(j_decompress_ptr cinfo, long num_bytes)
{
  struct counting_src_mgr *s = (struct counting_src_mgr *)cinfo->src;
  if (num_bytes <= 0)
    return;
  while ((unsigned long)num_bytes > s->pub.bytes_in_buffer) {
    num_bytes -= (long)s->pub.bytes_in_buffer;
    (void)csrc_fill(cinfo);
  }
  s->pub.next_input_byte += num_bytes;
  s->pub.bytes_in_buffer -= num_bytes;
}

static void csrc_term(j_decompress_ptr cinfo)
{
  struct counting_src_mgr *s = (struct counting_src_mgr *)cinfo->src;
  s->term_calls++;
}

static void install_counting_src(j_decompress_ptr cinfo,
                                 struct counting_src_mgr *s,
                                 const unsigned char *data,
                                 unsigned long size)
{
  memset(s, 0, sizeof(*s));
  s->pub.init_source = csrc_init;
  s->pub.fill_input_buffer = csrc_fill;
  s->pub.skip_input_data = csrc_skip;
  s->pub.resync_to_restart = jpeg_resync_to_restart;
  s->pub.term_source = csrc_term;
  s->pub.next_input_byte = data;
  s->pub.bytes_in_buffer = size;
  s->data = data;
  s->size = size;
  cinfo->src = &s->pub;
}

/* An in-memory grayscale JPEG. */
static unsigned char *make_source(unsigned long *out_size)
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

/* ---------------- decompress direction ---------------- */

/* d1: state value after a direct jpeg_read_header. */
static void d1_read_header_state(const unsigned char *src, unsigned long n)
{
  struct jpeg_decompress_struct cinfo;
  struct trap_error_mgr jerr;

  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, src, n);
  if (setjmp(jerr.escape) != 0) {
    printf("d1 err %d\n", jerr.msg_code);
  } else {
    (void)jpeg_read_header(&cinfo, TRUE);
    printf("d1 state %d\n", cinfo.global_state);
  }
  jpeg_destroy_decompress(&cinfo);
}

/* d2: finish_decompress with unread rows. */
static void d2_finish_early(const unsigned char *src, unsigned long n)
{
  struct jpeg_decompress_struct cinfo;
  struct trap_error_mgr jerr;
  JSAMPLE row[WIDTH];
  JSAMPROW rows[1];
  int r;

  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, src, n);
  if (setjmp(jerr.escape) != 0) {
    /* Trap record only — cinfo's fields are unspecified after longjmp. */
    printf("d2 err %d\n", jerr.msg_code);
    jpeg_destroy_decompress(&cinfo);
    return;
  }
  (void)jpeg_read_header(&cinfo, TRUE);
  (void)jpeg_start_decompress(&cinfo);
  for (r = 0; r < HEIGHT / 2; r++) {
    rows[0] = row;
    (void)jpeg_read_scanlines(&cinfo, rows, 1);
  }
  (void)jpeg_finish_decompress(&cinfo);
  printf("d2 ok state %d\n", cinfo.global_state);
  jpeg_destroy_decompress(&cinfo);
}

/* d3: finish_decompress right after read_header (never started). */
static void d3_finish_bad_state(const unsigned char *src, unsigned long n)
{
  struct jpeg_decompress_struct cinfo;
  struct trap_error_mgr jerr;

  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, src, n);
  if (setjmp(jerr.escape) != 0) {
    printf("d3 err %d parm %d\n", jerr.msg_code, jerr.msg_parm0);
    jpeg_destroy_decompress(&cinfo);
    return;
  }
  (void)jpeg_read_header(&cinfo, TRUE);
  (void)jpeg_finish_decompress(&cinfo);
  printf("d3 ok\n");
  jpeg_destroy_decompress(&cinfo);
}

/* d4: full decode + finish with a counting source: term calls + states. */
static void d4_finish_ok(const unsigned char *src, unsigned long n)
{
  struct jpeg_decompress_struct cinfo;
  struct trap_error_mgr jerr;
  struct counting_src_mgr csrc;
  JSAMPLE row[WIDTH];
  JSAMPROW rows[1];
  boolean ok;

  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  jpeg_create_decompress(&cinfo);
  install_counting_src(&cinfo, &csrc, src, n);
  if (setjmp(jerr.escape) != 0) {
    printf("d4 err %d\n", jerr.msg_code);
    jpeg_destroy_decompress(&cinfo);
    return;
  }
  (void)jpeg_read_header(&cinfo, TRUE);
  (void)jpeg_start_decompress(&cinfo);
  while (cinfo.output_scanline < cinfo.output_height) {
    rows[0] = row;
    (void)jpeg_read_scanlines(&cinfo, rows, 1);
  }
  ok = jpeg_finish_decompress(&cinfo);
  printf("d4 ret %d state %d term %d\n", (int)ok, cinfo.global_state,
         csrc.term_calls);
  jpeg_destroy_decompress(&cinfo);
}

/* d5: second finish after a completed one. */
static void d5_double_finish(const unsigned char *src, unsigned long n)
{
  struct jpeg_decompress_struct cinfo;
  struct trap_error_mgr jerr;
  static struct counting_src_mgr csrc; /* static: read after longjmp */
  JSAMPLE row[WIDTH];
  JSAMPROW rows[1];

  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  jpeg_create_decompress(&cinfo);
  install_counting_src(&cinfo, &csrc, src, n);
  if (setjmp(jerr.escape) != 0) {
    /* term count: a refused second finish must not call term again. */
    printf("d5 err %d parm %d term %d\n", jerr.msg_code, jerr.msg_parm0,
           csrc.term_calls);
    jpeg_destroy_decompress(&cinfo);
    return;
  }
  (void)jpeg_read_header(&cinfo, TRUE);
  (void)jpeg_start_decompress(&cinfo);
  while (cinfo.output_scanline < cinfo.output_height) {
    rows[0] = row;
    (void)jpeg_read_scanlines(&cinfo, rows, 1);
  }
  (void)jpeg_finish_decompress(&cinfo);
  (void)jpeg_finish_decompress(&cinfo);
  printf("d5 ok term %d\n", csrc.term_calls);
  jpeg_destroy_decompress(&cinfo);
}

/* d6: abort mid-decode, then reuse the same object for a full decode. */
static void d6_abort_reuse(const unsigned char *src, unsigned long n)
{
  struct jpeg_decompress_struct cinfo;
  struct trap_error_mgr jerr;
  JSAMPLE row[WIDTH];
  JSAMPROW rows[1];
  int r;

  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, src, n);
  if (setjmp(jerr.escape) != 0) {
    printf("d6 err %d\n", jerr.msg_code);
    jpeg_destroy_decompress(&cinfo);
    return;
  }
  (void)jpeg_read_header(&cinfo, TRUE);
  (void)jpeg_start_decompress(&cinfo);
  for (r = 0; r < HEIGHT / 2; r++) {
    rows[0] = row;
    (void)jpeg_read_scanlines(&cinfo, rows, 1);
  }
  jpeg_abort_decompress(&cinfo);
  printf("d6 aborted state %d\n", cinfo.global_state);

  /* Reuse. */
  jpeg_mem_src(&cinfo, src, n);
  (void)jpeg_read_header(&cinfo, TRUE);
  (void)jpeg_start_decompress(&cinfo);
  while (cinfo.output_scanline < cinfo.output_height) {
    rows[0] = row;
    (void)jpeg_read_scanlines(&cinfo, rows, 1);
  }
  (void)jpeg_finish_decompress(&cinfo);
  printf("d6 reused state %d rows %u\n", cinfo.global_state,
         (unsigned)cinfo.output_scanline);
  jpeg_destroy_decompress(&cinfo);
}


/* d7: finish_output and start_output from READY are refused (s3/r4). */
static void d7_output_calls_from_ready(const unsigned char *src,
                                       unsigned long n)
{
  struct jpeg_decompress_struct cinfo;
  struct trap_error_mgr jerr;

  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, src, n);
  if (setjmp(jerr.escape) != 0) {
    printf("d7a err %d parm %d\n", jerr.msg_code, jerr.msg_parm0);
  } else {
    (void)jpeg_read_header(&cinfo, TRUE);
    (void)jpeg_finish_output(&cinfo);
    printf("d7a ok\n");
  }
  jpeg_destroy_decompress(&cinfo);

  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, src, n);
  if (setjmp(jerr.escape) != 0) {
    printf("d7b err %d parm %d\n", jerr.msg_code, jerr.msg_parm0);
  } else {
    (void)jpeg_read_header(&cinfo, TRUE);
    (void)jpeg_start_output(&cinfo, 1);
    printf("d7b ok\n");
  }
  jpeg_destroy_decompress(&cinfo);
}

/* d8: a second jpeg_read_header without finish/abort is refused (u2). */
static void d8_double_read_header(const unsigned char *src, unsigned long n)
{
  struct jpeg_decompress_struct cinfo;
  struct trap_error_mgr jerr;

  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, src, n);
  if (setjmp(jerr.escape) != 0) {
    printf("d8 err %d parm %d\n", jerr.msg_code, jerr.msg_parm0);
    jpeg_destroy_decompress(&cinfo);
    return;
  }
  (void)jpeg_read_header(&cinfo, TRUE);
  (void)jpeg_read_header(&cinfo, TRUE);
  printf("d8 ok\n");
  jpeg_destroy_decompress(&cinfo);
}

/* d9: skipping every row reports input complete (p3). */
static void d9_skip_to_bottom(const unsigned char *src, unsigned long n)
{
  struct jpeg_decompress_struct cinfo;
  struct trap_error_mgr jerr;

  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, src, n);
  if (setjmp(jerr.escape) != 0) {
    printf("d9 err %d\n", jerr.msg_code);
    jpeg_destroy_decompress(&cinfo);
    return;
  }
  (void)jpeg_read_header(&cinfo, TRUE);
  (void)jpeg_start_decompress(&cinfo);
  (void)jpeg_skip_scanlines(&cinfo, cinfo.output_height);
  printf("d9 skipped %u complete %d\n", (unsigned)cinfo.output_scanline,
         (int)jpeg_input_complete(&cinfo));
  (void)jpeg_finish_decompress(&cinfo);
  jpeg_destroy_decompress(&cinfo);
}

/* ---------------- compress direction ---------------- */

static void setup_compress(struct jpeg_compress_struct *cinfo,
                           unsigned char **buf, unsigned long *size)
{
  jpeg_create_compress(cinfo);
  jpeg_mem_dest(cinfo, buf, size);
  cinfo->image_width = WIDTH;
  cinfo->image_height = HEIGHT;
  cinfo->input_components = 1;
  cinfo->in_color_space = JCS_GRAYSCALE;
  jpeg_set_defaults(cinfo);
}

/* c1: finish_compress with half the rows written. */
static void c1_finish_missing_rows(void)
{
  struct jpeg_compress_struct cinfo;
  struct trap_error_mgr jerr;
  unsigned char *buf = NULL;
  unsigned long size = 0;
  JSAMPLE row[WIDTH];
  JSAMPROW rows[1];
  int r;

  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  if (setjmp(jerr.escape) != 0) {
    /* Only the code: non-volatile locals are unspecified after longjmp
     * (d2 reads its struct through a pointer taken before setjmp and
     * happens to be stable; keeping c1 to the trap record is portable). */
    printf("c1 err %d\n", jerr.msg_code);
    jpeg_destroy_compress(&cinfo);
    free(buf);
    return;
  }
  setup_compress(&cinfo, &buf, &size);
  jpeg_start_compress(&cinfo, TRUE);
  memset(row, 0x40, sizeof(row));
  for (r = 0; r < HEIGHT / 2; r++) {
    rows[0] = row;
    (void)jpeg_write_scanlines(&cinfo, rows, 1);
  }
  jpeg_finish_compress(&cinfo);
  printf("c1 ok size %lu\n", size);
  jpeg_destroy_compress(&cinfo);
  free(buf);
}

/* c2: finish_compress without start_compress. */
static void c2_finish_bad_state(void)
{
  struct jpeg_compress_struct cinfo;
  struct trap_error_mgr jerr;
  unsigned char *buf = NULL;
  unsigned long size = 0;

  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  if (setjmp(jerr.escape) != 0) {
    printf("c2 err %d parm %d\n", jerr.msg_code, jerr.msg_parm0);
    jpeg_destroy_compress(&cinfo);
    free(buf);
    return;
  }
  setup_compress(&cinfo, &buf, &size);
  jpeg_finish_compress(&cinfo);
  printf("c2 ok\n");
  jpeg_destroy_compress(&cinfo);
  free(buf);
}

/* c3: double finish. */
static void c3_double_finish(void)
{
  struct jpeg_compress_struct cinfo;
  struct trap_error_mgr jerr;
  unsigned char *buf = NULL;
  unsigned long size = 0;
  JSAMPLE row[WIDTH];
  JSAMPROW rows[1];
  int r;

  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  if (setjmp(jerr.escape) != 0) {
    printf("c3 err %d parm %d\n", jerr.msg_code, jerr.msg_parm0);
    jpeg_destroy_compress(&cinfo);
    free(buf);
    return;
  }
  setup_compress(&cinfo, &buf, &size);
  jpeg_start_compress(&cinfo, TRUE);
  memset(row, 0x40, sizeof(row));
  for (r = 0; r < HEIGHT; r++) {
    rows[0] = row;
    (void)jpeg_write_scanlines(&cinfo, rows, 1);
  }
  jpeg_finish_compress(&cinfo);
  jpeg_finish_compress(&cinfo);
  printf("c3 ok\n");
  jpeg_destroy_compress(&cinfo);
  free(buf);
}

/* c5: error (missing rows) trapped, abort, reuse the same object fully. */
static void c5_error_then_reuse(void)
{
  struct jpeg_compress_struct cinfo;
  struct trap_error_mgr jerr;
  unsigned char *buf = NULL;
  unsigned long size = 0;
  JSAMPLE row[WIDTH];
  JSAMPROW rows[1];
  int r;
  int phase = 0;

  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  if (setjmp(jerr.escape) != 0) {
    if (phase == 0) {
      printf("c5 err %d\n", jerr.msg_code);
      jpeg_abort_compress(&cinfo);
      printf("c5 aborted state %d\n", cinfo.global_state);
      phase = 1;
      /* Reuse: the mem destination was already installed; reinstall to
       * reset it, as a caller would. */
      free(buf);
      buf = NULL;
      size = 0;
      jpeg_mem_dest(&cinfo, &buf, &size);
      jpeg_start_compress(&cinfo, TRUE);
      memset(row, 0x40, sizeof(row));
      for (r = 0; r < HEIGHT; r++) {
        rows[0] = row;
        (void)jpeg_write_scanlines(&cinfo, rows, 1);
      }
      jpeg_finish_compress(&cinfo);
      printf("c5 reused ok %d state %d\n", size > 100 ? 1 : 0,
             cinfo.global_state);
      jpeg_destroy_compress(&cinfo);
      free(buf);
      return;
    }
    printf("c5 err2 %d\n", jerr.msg_code);
    jpeg_destroy_compress(&cinfo);
    free(buf);
    return;
  }
  setup_compress(&cinfo, &buf, &size);
  jpeg_start_compress(&cinfo, TRUE);
  memset(row, 0x40, sizeof(row));
  for (r = 0; r < HEIGHT / 2; r++) {
    rows[0] = row;
    (void)jpeg_write_scanlines(&cinfo, rows, 1);
  }
  jpeg_finish_compress(&cinfo);
  printf("c5 unexpected ok\n");
  jpeg_destroy_compress(&cinfo);
  free(buf);
}

int main(void)
{
  unsigned long n = 0;
  unsigned char *src = make_source(&n);

  d1_read_header_state(src, n);
  d2_finish_early(src, n);
  d3_finish_bad_state(src, n);
  d4_finish_ok(src, n);
  d5_double_finish(src, n);
  d6_abort_reuse(src, n);
  d7_output_calls_from_ready(src, n);
  d8_double_read_header(src, n);
  d9_skip_to_bottom(src, n);
  c1_finish_missing_rows();
  c2_finish_bad_state();
  c3_double_finish();
  c5_error_then_reuse();

  free(src);
  return 0;
}

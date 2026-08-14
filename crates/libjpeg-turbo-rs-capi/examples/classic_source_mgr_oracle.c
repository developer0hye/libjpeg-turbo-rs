/*
 * P4-109 (#469) C oracle: classic source-manager setup and stdio semantics.
 *
 * Upstream contracts measured here (jdatasrc.c):
 *
 * - `jpeg_mem_src` rejects NULL/empty input with JERR_INPUT_EMPTY (43)
 *   before touching state, and refuses to overwrite a source manager it did
 *   not create with JERR_BUFFER_SIZE (jdatasrc.c:270-279) — the same
 *   identity check the destination side carries.
 * - `jpeg_stdio_src` reads through the caller's `FILE *` with fread in
 *   INPUT_BUF_SIZE chunks: a pre-positioned stream decodes from the current
 *   position (fd offset AND stdio buffer state both honored), the position
 *   after a completed decode lands where the chunked reads stopped — NOT
 *   necessarily EOF — so trailing bytes after the JPEG (djpeg-style
 *   concatenated streams) remain readable by the caller.
 *
 * Line format: `case token...`, deterministic.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>
#include <unistd.h>

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

/* m1/m2: NULL / empty input. */
static void m_null_empty(const unsigned char *src, unsigned long n)
{
  struct jpeg_decompress_struct cinfo;
  struct trap_error_mgr jerr;

  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  jpeg_create_decompress(&cinfo);
  if (setjmp(jerr.escape) != 0) {
    printf("m1 err %d\n", jerr.msg_code);
  } else {
    jpeg_mem_src(&cinfo, NULL, n);
    printf("m1 ok\n");
  }
  jpeg_destroy_decompress(&cinfo);

  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  jpeg_create_decompress(&cinfo);
  if (setjmp(jerr.escape) != 0) {
    printf("m2 err %d\n", jerr.msg_code);
  } else {
    jpeg_mem_src(&cinfo, src, 0);
    printf("m2 ok\n");
  }
  jpeg_destroy_decompress(&cinfo);
}

/* m3: installing mem_src over a foreign manager. */
static void fake_init(j_decompress_ptr cinfo) { (void)cinfo; }
static boolean fake_fill(j_decompress_ptr cinfo) { (void)cinfo; return FALSE; }
static void fake_skip(j_decompress_ptr cinfo, long n) { (void)cinfo; (void)n; }
static void fake_term(j_decompress_ptr cinfo) { (void)cinfo; }

static void m3_foreign_manager(const unsigned char *src, unsigned long n)
{
  struct jpeg_decompress_struct cinfo;
  struct trap_error_mgr jerr;
  struct jpeg_source_mgr foreign;

  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  jpeg_create_decompress(&cinfo);
  memset(&foreign, 0, sizeof(foreign));
  foreign.init_source = fake_init;
  foreign.fill_input_buffer = fake_fill;
  foreign.skip_input_data = fake_skip;
  foreign.resync_to_restart = jpeg_resync_to_restart;
  foreign.term_source = fake_term;
  cinfo.src = &foreign;
  if (setjmp(jerr.escape) != 0) {
    printf("m3 err %d\n", jerr.msg_code);
  } else {
    jpeg_mem_src(&cinfo, src, n);
    printf("m3 ok\n");
  }
  jpeg_destroy_decompress(&cinfo);
}

/* m4: mem_src twice on the same object (legal reuse). */
static void m4_mem_src_twice(const unsigned char *src, unsigned long n)
{
  struct jpeg_decompress_struct cinfo;
  struct trap_error_mgr jerr;
  JSAMPLE row[WIDTH];
  JSAMPROW rows[1];

  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  jpeg_create_decompress(&cinfo);
  if (setjmp(jerr.escape) != 0) {
    printf("m4 err %d\n", jerr.msg_code);
    jpeg_destroy_decompress(&cinfo);
    return;
  }
  jpeg_mem_src(&cinfo, src, n);
  (void)jpeg_read_header(&cinfo, TRUE);
  (void)jpeg_start_decompress(&cinfo);
  while (cinfo.output_scanline < cinfo.output_height) {
    rows[0] = row;
    (void)jpeg_read_scanlines(&cinfo, rows, 1);
  }
  (void)jpeg_finish_decompress(&cinfo);
  jpeg_mem_src(&cinfo, src, n);
  (void)jpeg_read_header(&cinfo, TRUE);
  (void)jpeg_start_decompress(&cinfo);
  while (cinfo.output_scanline < cinfo.output_height) {
    rows[0] = row;
    (void)jpeg_read_scanlines(&cinfo, rows, 1);
  }
  (void)jpeg_finish_decompress(&cinfo);
  printf("m4 ok rows %u\n", (unsigned)cinfo.output_scanline);
  jpeg_destroy_decompress(&cinfo);
}


/* m5/m6: cross-installing stdio over mem and mem over stdio — upstream's
 * identity checks treat the other family's manager as foreign
 * (jdatasrc.c: each setup function compares init_source against its own). */
static void m5_m6_cross_install(const unsigned char *src, unsigned long n)
{
  char path[128];
  snprintf(path, sizeof(path), "/tmp/p4109_oracle_cross_%ld.bin", (long)getpid());
  FILE *w = fopen(path, "wb");
  FILE *f;
  struct jpeg_decompress_struct cinfo;
  struct trap_error_mgr jerr;

  fwrite(src, 1, n, w);
  fclose(w);

  /* m5: mem_src first, then stdio_src. */
  f = fopen(path, "rb");
  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  jpeg_create_decompress(&cinfo);
  if (setjmp(jerr.escape) != 0) {
    printf("m5 err %d\n", jerr.msg_code);
  } else {
    jpeg_mem_src(&cinfo, src, n);
    jpeg_stdio_src(&cinfo, f);
    printf("m5 ok\n");
  }
  jpeg_destroy_decompress(&cinfo);
  fclose(f);

  /* m6: stdio_src first, then mem_src. */
  f = fopen(path, "rb");
  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  jpeg_create_decompress(&cinfo);
  if (setjmp(jerr.escape) != 0) {
    printf("m6 err %d\n", jerr.msg_code);
  } else {
    jpeg_stdio_src(&cinfo, f);
    jpeg_mem_src(&cinfo, src, n);
    printf("m6 ok\n");
  }
  jpeg_destroy_decompress(&cinfo);
  fclose(f);
  remove(path);
}

/* f1: pre-positioned FILE — a junk prefix skipped with fseek; also
 *     reports the resting position after a completed decode, with an
 *     8 KiB trailer present.                                        */
/* f2: stdio-buffer interaction — the prefix consumed with fgetc.    */
static void f_stdio_cases(const unsigned char *src, unsigned long n)
{
  char path[128];
  snprintf(path, sizeof(path), "/tmp/p4109_oracle_input_%ld.bin", (long)getpid());
  FILE *w = fopen(path, "wb");
  FILE *f;
  struct jpeg_decompress_struct cinfo;
  struct trap_error_mgr jerr;
  JSAMPLE row[WIDTH];
  JSAMPROW rows[1];
  int i;
  long pos, fsize;

  /* 16-byte junk prefix + JPEG + 8 KiB trailer. The trailer is larger
   * than INPUT_BUF_SIZE (4096) so a chunked reader's resting position is
   * strictly before EOF, while a slurp-to-EOF lands AT EOF — the
   * discriminating observable for djpeg-style concatenated streams. */
  for (i = 0; i < 16; i++) fputc(0x55, w);
  fwrite(src, 1, n, w);
  for (i = 0; i < 8192; i++) fputc((char)(i & 0x7F), w);
  fclose(w);
  fsize = 16 + (long)n + 8192;

  /* f1: skip the prefix with fseek, decode, report rows + position. */
  f = fopen(path, "rb");
  fseek(f, 16, SEEK_SET);
  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  jpeg_create_decompress(&cinfo);
  if (setjmp(jerr.escape) != 0) {
    printf("f1 err %d\n", jerr.msg_code);
    jpeg_destroy_decompress(&cinfo);
    fclose(f);
    return;
  }
  jpeg_stdio_src(&cinfo, f);
  (void)jpeg_read_header(&cinfo, TRUE);
  (void)jpeg_start_decompress(&cinfo);
  while (cinfo.output_scanline < cinfo.output_height) {
    rows[0] = row;
    (void)jpeg_read_scanlines(&cinfo, rows, 1);
  }
  (void)jpeg_finish_decompress(&cinfo);
  pos = ftell(f);
  /* The exact resting position is a chunking detail; what a consumer
   * relies on is whether trailing bytes remain readable. Classify. */
  printf("f1 rows %u pos_class %s\n", (unsigned)cinfo.output_scanline,
         pos < 0 ? "err" : (pos >= fsize ? "eof" : "before_eof"));
  jpeg_destroy_decompress(&cinfo);
  fclose(f);

  /* f2: consume the prefix through the stdio buffer with fgetc. */
  f = fopen(path, "rb");
  for (i = 0; i < 16; i++) (void)fgetc(f);
  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  jpeg_create_decompress(&cinfo);
  if (setjmp(jerr.escape) != 0) {
    printf("f2 err %d\n", jerr.msg_code);
    jpeg_destroy_decompress(&cinfo);
    fclose(f);
    return;
  }
  jpeg_stdio_src(&cinfo, f);
  (void)jpeg_read_header(&cinfo, TRUE);
  (void)jpeg_start_decompress(&cinfo);
  while (cinfo.output_scanline < cinfo.output_height) {
    rows[0] = row;
    (void)jpeg_read_scanlines(&cinfo, rows, 1);
  }
  (void)jpeg_finish_decompress(&cinfo);
  printf("f2 rows %u\n", (unsigned)cinfo.output_scanline);
  jpeg_destroy_decompress(&cinfo);
  fclose(f);

  remove(path);
}

/* f3: I/O failure — an empty FILE yields zero bytes on the very first
 * chunked read; stock's fill_input_buffer ERREXITs (jdatasrc.c start_of_file
 * branch) when jpeg_read_header pulls the first marker. */
static void f3_empty_file(void)
{
  char path[128];
  snprintf(path, sizeof(path), "/tmp/p4109_oracle_empty_%ld.bin", (long)getpid());
  FILE *w = fopen(path, "wb");
  FILE *f;
  struct jpeg_decompress_struct cinfo;
  struct trap_error_mgr jerr;

  fclose(w);
  f = fopen(path, "rb");
  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  jpeg_create_decompress(&cinfo);
  if (setjmp(jerr.escape) != 0) {
    printf("f3 err %d\n", jerr.msg_code);
  } else {
    jpeg_stdio_src(&cinfo, f);
    (void)jpeg_read_header(&cinfo, TRUE);
    printf("f3 ok\n");
  }
  jpeg_destroy_decompress(&cinfo);
  fclose(f);
  remove(path);
}

/* f4: the installed fill_input_buffer driven directly through the ABI.
 * One chunked read serves the whole 6-byte file; each dry read after it
 * warns JWRN_JPEG_EOF and fabricates FF D9 (jdatasrc.c:106-118). */
static void record_emit(j_common_ptr cinfo, int msg_level)
{
  if (msg_level < 0)
    printf("f4 warn %d\n", cinfo->err->msg_code);
}

static void f4_fill_sequence(void)
{
  char path[128];
  snprintf(path, sizeof(path), "/tmp/p4109_oracle_fill_%ld.bin", (long)getpid());
  static const unsigned char payload[6] = { 0x10, 0x20, 0x30, 0x40, 0x50, 0x60 };
  FILE *w = fopen(path, "wb");
  FILE *f;
  struct jpeg_decompress_struct cinfo;
  struct trap_error_mgr jerr;
  int i;

  fwrite(payload, 1, sizeof(payload), w);
  fclose(w);
  f = fopen(path, "rb");
  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.emit_message = record_emit;
  jerr.pub.output_message = quiet_output_message;
  jpeg_create_decompress(&cinfo);
  if (setjmp(jerr.escape) != 0) {
    printf("f4 fatal %d\n", jerr.msg_code);
  } else {
    jpeg_stdio_src(&cinfo, f);
    for (i = 1; i <= 3; i++) {
      boolean r = (*cinfo.src->fill_input_buffer)(&cinfo);
      printf("f4 fill %d ret %d bib %lu first 0x%02X\n", i, (int)r,
             (unsigned long)cinfo.src->bytes_in_buffer,
             (unsigned)cinfo.src->next_input_byte[0]);
    }
  }
  jpeg_destroy_decompress(&cinfo);
  fclose(f);
  remove(path);
}

int main(void)
{
  unsigned long n = 0;
  unsigned char *src = make_source(&n);

  m_null_empty(src, n);
  m3_foreign_manager(src, n);
  m4_mem_src_twice(src, n);
  m5_m6_cross_install(src, n);
  f_stdio_cases(src, n);
  f3_empty_file();
  f4_fill_sequence();

  free(src);
  return 0;
}

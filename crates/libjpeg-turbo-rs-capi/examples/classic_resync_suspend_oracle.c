/*
 * P4-97 (#469) C oracle: jpeg_resync_to_restart against a genuinely
 * suspending source manager.
 *
 * Upstream contracts measured here (jdmarker.c resync_to_restart /
 * next_marker, libjpeg.txt "Decompression suspension"):
 *
 * - The decision table over unread_marker: desired RST or >2 away resumes
 *   (action 1), the next two expected RSTs and valid non-RST markers are
 *   left unread (action 3), prior RSTs and invalid bytes scan forward
 *   (action 2).
 * - JWRN_MUST_RESYNC is emitted once per call, ahead of the decision loop;
 *   JTRC_RECOVERY_ACTION traces each chosen action unconditionally, so a
 *   call that scans forward and re-decides emits it more than once
 *   (msg_code/msg_parm publish even at trace_level 0).
 * - Scan-forward pulls bytes through fill_input_buffer. A FALSE fill is a
 *   FALSE resync return with the cursor at the last committed restart
 *   point; the manager's unconsumed tail must be re-presented on refill
 *   (libjpeg.txt: "the data beyond that position must NOT be discarded").
 * - discarded_bytes lives in the private marker state, so the count a
 *   later JWRN_EXTRANEOUS_DATA reports accumulates ACROSS suspensions.
 *
 * Line format: `<scenario> <token>...`, deterministic; nested fill/msg
 * lines print in call order.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>

#include <jpeglib.h>

static const char *g_name = "";

struct trap_error_mgr {
  struct jpeg_error_mgr pub;
  jmp_buf escape;
};

static void trap_error_exit(j_common_ptr cinfo)
{
  struct trap_error_mgr *e = (struct trap_error_mgr *)cinfo->err;
  printf("%s fatal %d\n", g_name, cinfo->err->msg_code);
  longjmp(e->escape, 1);
}

/* Every emit_message publishes msg_code and the first two int parms
 * regardless of trace_level (jerror.h TRACEMS2/WARNMS2 store, then call). */
static void record_emit(j_common_ptr cinfo, int msg_level)
{
  printf("%s msg %d lvl %d p0 %d p1 %d\n", g_name, cinfo->err->msg_code,
         msg_level, cinfo->err->msg_parm.i[0], cinfo->err->msg_parm.i[1]);
}

static void quiet_output_message(j_common_ptr cinfo) { (void)cinfo; }

/* A suspending source manager per libjpeg.txt: the driver authorizes
 * stream bytes in stages; fill presents [unconsumed tail + fresh bytes]
 * or returns FALSE when nothing new is staged. `bib` in the trace is the
 * restart-point tail at fill time — the INPUT_SYNC parity observable. */
struct suspend_src {
  struct jpeg_source_mgr pub;
  const unsigned char *stream;
  unsigned long fed;   /* stream bytes already placed in buf */
  unsigned long stage; /* stream bytes the driver has authorized */
  unsigned char buf[256];
  int fill_calls;
};

static void suspend_init(j_decompress_ptr cinfo) { (void)cinfo; }
static void suspend_skip(j_decompress_ptr cinfo, long n) { (void)cinfo; (void)n; }
static void suspend_term(j_decompress_ptr cinfo) { (void)cinfo; }

static boolean suspend_fill(j_decompress_ptr cinfo)
{
  struct suspend_src *s = (struct suspend_src *)cinfo->src;
  unsigned long tail = (unsigned long)s->pub.bytes_in_buffer;
  unsigned long fresh = s->stage - s->fed;

  s->fill_calls++;
  if (fresh == 0) {
    printf("%s fill %d bib %lu sus\n", g_name, s->fill_calls, tail);
    return FALSE;
  }
  /* Keep the restart-point tail ahead of the new data (overlap-safe:
   * next_input_byte points into buf). */
  memmove(s->buf, s->pub.next_input_byte, tail);
  memcpy(s->buf + tail, s->stream + s->fed, fresh);
  s->fed = s->stage;
  s->pub.next_input_byte = s->buf;
  s->pub.bytes_in_buffer = tail + fresh;
  printf("%s fill %d bib %lu serve %lu\n", g_name, s->fill_calls, tail, fresh);
  return TRUE;
}

static void run_scenario(const char *name, int desired, int start_marker,
                         const unsigned char *stream,
                         const unsigned long *stages, int n_stages)
{
  struct jpeg_decompress_struct cinfo;
  struct trap_error_mgr jerr;
  struct suspend_src src;
  int step = 0;

  g_name = name;
  memset(&jerr, 0, sizeof(jerr));
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.emit_message = record_emit;
  jerr.pub.output_message = quiet_output_message;
  jpeg_create_decompress(&cinfo);

  memset(&src, 0, sizeof(src));
  src.pub.init_source = suspend_init;
  src.pub.fill_input_buffer = suspend_fill;
  src.pub.skip_input_data = suspend_skip;
  src.pub.resync_to_restart = jpeg_resync_to_restart;
  src.pub.term_source = suspend_term;
  src.pub.next_input_byte = src.buf;
  src.pub.bytes_in_buffer = 0;
  src.stream = stream;
  src.stage = stages[0];
  cinfo.src = &src.pub;
  cinfo.unread_marker = start_marker;

  if (setjmp(jerr.escape) == 0) {
    for (;;) {
      boolean r = jpeg_resync_to_restart(&cinfo, desired);
      printf("%s call %d ret %d um 0x%02X\n", name, step + 1, (int)r,
             (unsigned)cinfo.unread_marker);
      if (r) break;
      step++;
      if (step >= n_stages) {
        printf("%s exhausted\n", name);
        break;
      }
      src.stage = stages[step];
    }
  }
  printf("%s fills %d\n", name, src.fill_calls);
  cinfo.src = NULL; /* src is a stack local; detach before destroy */
  jpeg_destroy_decompress(&cinfo);
}

int main(void)
{
  /* Desired restart number 3 throughout; RST3 = 0xD3. */

  /* s1: the desired restart — discard it and resume (action 1). */
  {
    static const unsigned long stages[] = { 0 };
    run_scenario("s1", 3, 0xD3, NULL, stages, 1);
  }
  /* s2/s3: the next two expected restarts — leave unread (action 3). */
  {
    static const unsigned long stages[] = { 0 };
    run_scenario("s2", 3, 0xD4, NULL, stages, 1);
    run_scenario("s3", 3, 0xD5, NULL, stages, 1);
  }
  /* s3b: a valid non-restart marker (EOI) — action 3. */
  {
    static const unsigned long stages[] = { 0 };
    run_scenario("s3b", 3, 0xD9, NULL, stages, 1);
  }
  /* s4: a prior restart (action 2) scanning through garbage that spans a
   * suspension. Stock accumulates discarded_bytes in the marker state, so
   * the JWRN_EXTRANEOUS_DATA on the second call reports all 8 bytes. */
  {
    static const unsigned char stream[] = {
      0x41, 0x42, 0x43, 0x44, 0x45,       /* call 1: discarded, then sus */
      0x46, 0x47, 0x48, 0xFF, 0xD3,       /* call 2: 3 more + RST3 */
    };
    static const unsigned long stages[] = { 5, 10 };
    run_scenario("s4", 3, 0xD2, stream, stages, 2);
  }
  /* s5: an invalid byte (< M_SOF0) scans forward through a stuffed-zero
   * pair and an FF pad run to EOI — action 3 on the found marker. */
  {
    static const unsigned char stream[] = {
      0xFF, 0x00, 0xFF, 0xFF, 0xFF, 0xD9,
    };
    static const unsigned long stages[] = { 6 };
    run_scenario("s5", 3, 0x01, stream, stages, 1);
  }
  /* s6: suspension inside an FF pad run. The two pad FFs precede any sync
   * point, so the refill must re-present them (bib 2 at the third fill);
   * the found RST7 is >2 from desired — action 1. */
  {
    static const unsigned char stream[] = {
      0xFF, 0xFF,             /* call 1: pad run, then sus */
      0xFF, 0xD7,             /* call 2: run continues into RST7 */
    };
    static const unsigned long stages[] = { 2, 4 };
    run_scenario("s6", 3, 0xD1, stream, stages, 2);
  }

  return 0;
}

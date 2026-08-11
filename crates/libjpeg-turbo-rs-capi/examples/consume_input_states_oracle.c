/*
 * P4-104 (#468) C oracle: print the classic decompressor's real
 * (return code, global_state) trace from stock libjpeg-turbo.
 *
 * The Rust state-machine tests used to compare against sequences transcribed
 * by hand from `jdapimin.c` / `jdapistd.c` after a throwaway probe run. That
 * pins the shim against a *belief* about upstream: a wrong reading would be
 * copied into both the implementation and the expectation, and the suite would
 * stay green. This binary makes upstream answer for itself, so the comparison
 * is against C rather than against a comment.
 *
 * `global_state` is a public member: `jpeglib.h` declares it in the
 * `jpeg_common_fields` macro that opens every jpeg object, and the macro
 * expands inside the struct, so `cinfo->global_state` names it directly. An
 * earlier draft read it through a hand-copied mirror struct, which is both
 * unnecessary and undefined at `-O2` — accessing an object through an
 * unrelated struct type violates C's effective-type rules, and the mirror
 * could silently drift from the header besides.
 *
 * Usage:
 *   consume_input_states_oracle poll    <file>   3 polls of jpeg_consume_input
 *   consume_input_states_oracle coefs   <file>   poll, then jpeg_read_coefficients
 *   consume_input_states_oracle preload <file>   suspending source: poll to SOS,
 *                                                start_decompress, resume by
 *                                                polling, then retry startup
 *
 * Each line of stdout is one observation, `NAME rc state`, so a diff against
 * the Rust trace points at the exact step that diverged.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>

#include <jpeglib.h>

static int state_of(j_decompress_ptr cinfo)
{
  return cinfo->global_state;
}

/* Fatal-error handling: longjmp out instead of exit(), so a diverging run
 * reports the JPEG message rather than dying silently. */
struct trap_error_mgr {
  struct jpeg_error_mgr pub;
  jmp_buf escape;
};

static void trap_error_exit(j_common_ptr cinfo)
{
  struct trap_error_mgr *e = (struct trap_error_mgr *)cinfo->err;
  char buffer[JMSG_LENGTH_MAX];
  (*cinfo->err->format_message) (cinfo, buffer);
  fprintf(stderr, "oracle: fatal libjpeg error: %s\n", buffer);
  longjmp(e->escape, 1);
}

static unsigned char *slurp(const char *path, size_t *len)
{
  FILE *f = fopen(path, "rb");
  unsigned char *buf;
  long size;
  if (!f) { perror(path); exit(2); }
  if (fseek(f, 0, SEEK_END) != 0) { perror("fseek"); exit(2); }
  size = ftell(f);
  if (size < 0) { perror("ftell"); exit(2); }
  rewind(f);
  buf = (unsigned char *)malloc((size_t)size);
  if (!buf) { fprintf(stderr, "oom\n"); exit(2); }
  if (fread(buf, 1, (size_t)size, f) != (size_t)size) {
    fprintf(stderr, "short read of %s\n", path);
    exit(2);
  }
  fclose(f);
  *len = (size_t)size;
  return buf;
}

/* ---- a suspending source manager, mirroring the Rust test's ChunkSource ---- */

#define CHUNK 128

struct chunk_source {
  struct jpeg_source_mgr pub;
  const unsigned char *data;
  size_t served;
  size_t limit;
};

static void chunk_init_source(j_decompress_ptr cinfo) { (void)cinfo; }

static boolean chunk_fill_input_buffer(j_decompress_ptr cinfo)
{
  struct chunk_source *src = (struct chunk_source *)cinfo->src;
  size_t end;
  if (src->served >= src->limit)
    return FALSE;                      /* suspend */
  end = src->served + CHUNK;
  if (end > src->limit)
    end = src->limit;
  src->pub.next_input_byte = src->data + src->served;
  src->pub.bytes_in_buffer = end - src->served;
  src->served = end;
  return TRUE;
}

static void chunk_skip_input_data(j_decompress_ptr cinfo, long num_bytes)
{
  struct chunk_source *src = (struct chunk_source *)cinfo->src;
  size_t remaining;
  size_t step;
  if (num_bytes <= 0)
    return;
  remaining = (size_t)num_bytes;
  step = remaining < src->pub.bytes_in_buffer ? remaining
                                              : src->pub.bytes_in_buffer;
  src->pub.next_input_byte += step;
  src->pub.bytes_in_buffer -= step;
  remaining -= step;
  src->served += remaining;
  if (src->served > src->limit)
    src->served = src->limit;
}

static void chunk_term_source(j_decompress_ptr cinfo) { (void)cinfo; }

/* Offset just past the first SOS header — the point at which a source has
 * delivered a complete header and an incomplete body. Mirrors the Rust
 * helper of the same name. */
static size_t first_sos_end(const unsigned char *jpeg, size_t len)
{
  size_t i;
  for (i = 0; i + 3 < len; i++) {
    if (jpeg[i] == 0xFF && jpeg[i + 1] == 0xDA) {
      size_t seglen = ((size_t)jpeg[i + 2] << 8) | (size_t)jpeg[i + 3];
      return i + 2 + seglen;
    }
  }
  fprintf(stderr, "oracle: no SOS in input\n");
  exit(2);
}

int main(int argc, char **argv)
{
  struct jpeg_decompress_struct cinfo;
  struct trap_error_mgr jerr;
  unsigned char *jpeg;
  size_t len;
  const char *mode;

  if (argc != 3) {
    fprintf(stderr, "usage: %s poll|coefs|preload <file>\n", argv[0]);
    return 2;
  }
  mode = argv[1];
  jpeg = slurp(argv[2], &len);

  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  if (setjmp(jerr.escape)) {
    jpeg_destroy_decompress(&cinfo);
    free(jpeg);
    return 3;
  }
  jpeg_create_decompress(&cinfo);
  printf("create -1 %d\n", state_of(&cinfo));

  if (strcmp(mode, "poll") == 0) {
    int i;
    jpeg_mem_src(&cinfo, jpeg, (unsigned long)len);
    for (i = 0; i < 3; i++) {
      int rc = jpeg_consume_input(&cinfo);
      printf("poll %d %d\n", rc, state_of(&cinfo));
    }
  } else if (strcmp(mode, "coefs") == 0) {
    int rc;
    jvirt_barray_ptr *coefs;
    jpeg_mem_src(&cinfo, jpeg, (unsigned long)len);
    rc = jpeg_consume_input(&cinfo);
    printf("poll %d %d\n", rc, state_of(&cinfo));
    coefs = jpeg_read_coefficients(&cinfo);
    printf("coefs %d %d\n", coefs != NULL, state_of(&cinfo));
    printf("complete %d %d\n", jpeg_input_complete(&cinfo), state_of(&cinfo));
    rc = jpeg_consume_input(&cinfo);
    printf("poll %d %d\n", rc, state_of(&cinfo));
  } else if (strcmp(mode, "preload") == 0) {
    struct chunk_source src;
    int i, rc, ok;
    memset(&src, 0, sizeof(src));
    src.pub.init_source = chunk_init_source;
    src.pub.fill_input_buffer = chunk_fill_input_buffer;
    src.pub.skip_input_data = chunk_skip_input_data;
    src.pub.resync_to_restart = jpeg_resync_to_restart;
    src.pub.term_source = chunk_term_source;
    src.data = jpeg;
    src.served = 0;
    src.limit = first_sos_end(jpeg, len);
    cinfo.src = (struct jpeg_source_mgr *)&src;

    for (i = 0; i < 64; i++) {
      rc = jpeg_consume_input(&cinfo);
      if (rc == JPEG_REACHED_SOS)
        break;
    }
    printf("sos %d %d\n", rc, state_of(&cinfo));

    ok = jpeg_start_decompress(&cinfo);
    printf("startup %d %d\n", ok, state_of(&cinfo));

    src.limit = len;                    /* more bytes arrive */
    for (i = 0; i < 512; i++) {
      rc = jpeg_consume_input(&cinfo);
      if (rc == JPEG_REACHED_EOI)
        break;
    }
    printf("drained %d %d\n", rc, state_of(&cinfo));
    printf("complete %d %d\n", jpeg_input_complete(&cinfo), state_of(&cinfo));

    ok = jpeg_start_decompress(&cinfo);
    printf("retry %d %d\n", ok, state_of(&cinfo));
  } else {
    fprintf(stderr, "unknown mode %s\n", mode);
    free(jpeg);
    return 2;
  }

  jpeg_destroy_decompress(&cinfo);
  free(jpeg);
  return 0;
}

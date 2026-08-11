/*
 * P4-110 C oracle: what stock libjpeg actually does when `jpeg_CreateCompress`
 * / `jpeg_CreateDecompress` are handed a wrong `version` or `structsize`.
 *
 * This is the P0 in the item: the shim ignored both parameters and wrote its
 * full Rust mirror regardless, so a caller declaring a smaller struct — an
 * older ABI, or simply a different build — had memory written past the end of
 * its allocation. Getting the *rejection* right matters as much as getting the
 * detection right: which error code, which two parameters, in which order, and
 * exactly how much of the caller's object may be touched before the guard
 * fires.
 *
 * Rather than transcribe that from `jdapimin.c` and hope, this binary asks the
 * real library and prints what it observes. `capi_create_abi_guards.rs`
 * compares the shim's answers against this output line for line.
 *
 * Usage:
 *   create_abi_guards_oracle decompress
 *   create_abi_guards_oracle compress
 *
 * Output is `case key=value ...`, one line per (version, structsize) case.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <setjmp.h>

#include <jpeglib.h>

/* Bytes of canary past the declared object. A guard that writes the library's
 * own idea of the struct size instead of the caller's would land here. */
#define CANARY 64
#define CANARY_BYTE 0x5A
/* Pre-fill: every byte the library does not deliberately set stays 0xAA, so
 * "was this field zeroed?" is answerable rather than assumed. */
#define FILL_BYTE 0xAA

/* Sentinels for the two fields upstream preserves across the zeroing. */
#define CLIENT_DATA_SENTINEL ((void *)0x1234ABCD)

struct trap_error_mgr {
  struct jpeg_error_mgr pub;
  jmp_buf escape;
  int fired;
  int msg_code;
  int parm0;
  int parm1;
};

static void trap_error_exit(j_common_ptr cinfo)
{
  struct trap_error_mgr *e = (struct trap_error_mgr *)cinfo->err;
  e->fired = 1;
  e->msg_code = cinfo->err->msg_code;
  e->parm0 = cinfo->err->msg_parm.i[0];
  e->parm1 = cinfo->err->msg_parm.i[1];
  longjmp(e->escape, 1);
}

/* Warnings must not be counted as failures, but must not go to stderr either. */
static void quiet_output_message(j_common_ptr cinfo) { (void)cinfo; }

static int canary_intact(const unsigned char *base, size_t declared)
{
  size_t i;
  for (i = 0; i < CANARY; i++)
    if (base[declared + i] != CANARY_BYTE)
      return 0;
  return 1;
}

/* One (version, structsize) case against jpeg_CreateDecompress. */
static void run_decompress(const char *label, int version, size_t declared)
{
  size_t real = sizeof(struct jpeg_decompress_struct);
  unsigned char *block = (unsigned char *)malloc(declared + CANARY);
  j_decompress_ptr cinfo = (j_decompress_ptr)block;
  struct trap_error_mgr jerr;

  if (!block) { fprintf(stderr, "oom\n"); exit(2); }
  memset(block, FILL_BYTE, declared);
  memset(block + declared, CANARY_BYTE, CANARY);

  memset(&jerr, 0, sizeof(jerr));
  cinfo->err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  cinfo->client_data = CLIENT_DATA_SENTINEL;

  if (setjmp(jerr.escape) == 0)
    jpeg_CreateDecompress(cinfo, version, declared);

  printf("%s fired=%d code=%d parm0=%d parm1=%d canary=%d",
         label, jerr.fired, jerr.msg_code, jerr.parm0, jerr.parm1,
         canary_intact(block, declared));

  if (!jerr.fired) {
    /* Only meaningful on the accepted case: the two preserved fields, the
     * discriminator, and a field upstream's memset must have cleared. */
    printf(" err_kept=%d client_kept=%d is_decompressor=%d global_state=%d"
           " src_zero=%d width_zero=%d",
           cinfo->err == &jerr.pub,
           cinfo->client_data == CLIENT_DATA_SENTINEL,
           cinfo->is_decompressor,
           cinfo->global_state,
           cinfo->src == NULL,
           cinfo->image_width == 0);
    jpeg_destroy_decompress(cinfo);
  } else {
    /* `mem` is NULLed before the guards precisely so this is safe. */
    printf(" mem_null=%d", cinfo->mem == NULL);
    jpeg_destroy_decompress(cinfo);
  }
  printf(" real=%d\n", (int)real);
  free(block);
}

/* The same for jpeg_CreateCompress. */
static void run_compress(const char *label, int version, size_t declared)
{
  size_t real = sizeof(struct jpeg_compress_struct);
  unsigned char *block = (unsigned char *)malloc(declared + CANARY);
  j_compress_ptr cinfo = (j_compress_ptr)block;
  struct trap_error_mgr jerr;

  if (!block) { fprintf(stderr, "oom\n"); exit(2); }
  memset(block, FILL_BYTE, declared);
  memset(block + declared, CANARY_BYTE, CANARY);

  memset(&jerr, 0, sizeof(jerr));
  cinfo->err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = trap_error_exit;
  jerr.pub.output_message = quiet_output_message;
  cinfo->client_data = CLIENT_DATA_SENTINEL;

  if (setjmp(jerr.escape) == 0)
    jpeg_CreateCompress(cinfo, version, declared);

  printf("%s fired=%d code=%d parm0=%d parm1=%d canary=%d",
         label, jerr.fired, jerr.msg_code, jerr.parm0, jerr.parm1,
         canary_intact(block, declared));

  if (!jerr.fired) {
    printf(" err_kept=%d client_kept=%d is_decompressor=%d global_state=%d"
           " dest_zero=%d width_zero=%d",
           cinfo->err == &jerr.pub,
           cinfo->client_data == CLIENT_DATA_SENTINEL,
           cinfo->is_decompressor,
           cinfo->global_state,
           cinfo->dest == NULL,
           cinfo->image_width == 0);
    jpeg_destroy_compress(cinfo);
  } else {
    printf(" mem_null=%d", cinfo->mem == NULL);
    jpeg_destroy_compress(cinfo);
  }
  printf(" real=%d\n", (int)real);
  free(block);
}

int main(int argc, char **argv)
{
  if (argc != 2) {
    fprintf(stderr, "usage: %s decompress|compress\n", argv[0]);
    return 2;
  }

  if (strcmp(argv[1], "decompress") == 0) {
    size_t real = sizeof(struct jpeg_decompress_struct);
    run_decompress("ok", JPEG_LIB_VERSION, real);
    run_decompress("badversion_low", JPEG_LIB_VERSION - 10, real);
    run_decompress("badversion_high", JPEG_LIB_VERSION + 10, real);
    run_decompress("badsize_small", JPEG_LIB_VERSION, real - 8);
    run_decompress("badsize_large", JPEG_LIB_VERSION, real + 8);
    /* Version is checked first: with both wrong, the version error wins. */
    run_decompress("both_wrong", JPEG_LIB_VERSION - 10, real - 8);
  } else if (strcmp(argv[1], "compress") == 0) {
    size_t real = sizeof(struct jpeg_compress_struct);
    run_compress("ok", JPEG_LIB_VERSION, real);
    run_compress("badversion_low", JPEG_LIB_VERSION - 10, real);
    run_compress("badversion_high", JPEG_LIB_VERSION + 10, real);
    run_compress("badsize_small", JPEG_LIB_VERSION, real - 8);
    run_compress("badsize_large", JPEG_LIB_VERSION, real + 8);
    run_compress("both_wrong", JPEG_LIB_VERSION - 10, real - 8);
  } else {
    fprintf(stderr, "unknown mode %s\n", argv[1]);
    return 2;
  }
  return 0;
}

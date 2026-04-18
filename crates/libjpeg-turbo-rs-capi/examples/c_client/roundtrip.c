/* B9-1: C client that round-trips a PPM-ish pixel buffer through the
 * TurboJPEG 3 API exposed by our cdylib: tj3Compress8 -> tj3Decompress8.
 *
 * The binary is intentionally self-contained — no libjpeg-turbo header
 * dependency — so it compiles on any host that has a C compiler and
 * our cdylib on the link path. The round-trip result is written to
 * `argv[1]` in a simple binary blob: [width:4 | height:4 | bpp:1 | raw
 * pixels]. The Rust integration test parses that blob back and asserts
 * pixel fidelity.
 *
 * Exit codes:
 *   0  success
 *   2  tj3Init failure
 *   3  tj3Compress8 failure
 *   4  tj3Decompress8 failure
 *   5  malloc failure
 *   6  cannot open output file
 *   7  output write failure
 *   10 wrong number of arguments
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void *tjhandle;

#define TJPARAM_QUALITY 3
#define TJPARAM_SUBSAMP 4
#define TJINIT_COMPRESS 1
#define TJINIT_DECOMPRESS 2
#define TJPF_RGB 0
#define TJSAMP_444 0

extern tjhandle tj3Init(int initType);
extern void tj3Destroy(tjhandle handle);
extern int tj3Set(tjhandle handle, int param, int value);
extern const char *tj3GetErrorStr(tjhandle handle);
extern int tj3Compress8(tjhandle handle, const unsigned char *srcBuf, int width,
                        int pitch, int height, int pixelFormat,
                        unsigned char **jpegBuf, size_t *jpegSize);
extern int tj3Decompress8(tjhandle handle, const unsigned char *jpegBuf,
                          size_t jpegSize, unsigned char *dstBuf, int pitch,
                          int pixelFormat);
extern void tj3Free(void *ptr);

static int write_be_u32(FILE *fp, uint32_t v) {
    unsigned char buf[4];
    buf[0] = (unsigned char)((v >> 24) & 0xFFu);
    buf[1] = (unsigned char)((v >> 16) & 0xFFu);
    buf[2] = (unsigned char)((v >> 8) & 0xFFu);
    buf[3] = (unsigned char)(v & 0xFFu);
    return fwrite(buf, 1, 4, fp) == 4 ? 0 : -1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <output-path>\n", argv[0]);
        return 10;
    }

    /* 96x96 synthetic gradient in RGB — large enough to exercise more
     * than one MCU per dimension for any subsampling, small enough to
     * keep the test fast. */
    const int w = 96;
    const int h = 96;
    const int bpp = 3;
    unsigned char *src = (unsigned char *)malloc((size_t)w * h * bpp);
    if (!src) {
        return 5;
    }
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            unsigned char *p = src + ((size_t)y * w + x) * bpp;
            p[0] = (unsigned char)((x * 255) / (w - 1));
            p[1] = (unsigned char)((y * 255) / (h - 1));
            p[2] = (unsigned char)(((x + y) * 255) / (w + h - 2));
        }
    }

    tjhandle enc = tj3Init(TJINIT_COMPRESS);
    if (!enc) {
        free(src);
        return 2;
    }
    tj3Set(enc, TJPARAM_QUALITY, 95);
    tj3Set(enc, TJPARAM_SUBSAMP, TJSAMP_444);

    unsigned char *jpeg = NULL;
    size_t jpeg_size = 0;
    if (tj3Compress8(enc, src, w, 0, h, TJPF_RGB, &jpeg, &jpeg_size) != 0) {
        fprintf(stderr, "tj3Compress8: %s\n", tj3GetErrorStr(enc));
        tj3Destroy(enc);
        free(src);
        return 3;
    }

    tjhandle dec = tj3Init(TJINIT_DECOMPRESS);
    if (!dec) {
        tj3Free(jpeg);
        tj3Destroy(enc);
        free(src);
        return 2;
    }

    unsigned char *dst = (unsigned char *)malloc((size_t)w * h * bpp);
    if (!dst) {
        tj3Free(jpeg);
        tj3Destroy(enc);
        tj3Destroy(dec);
        free(src);
        return 5;
    }

    if (tj3Decompress8(dec, jpeg, jpeg_size, dst, 0, TJPF_RGB) != 0) {
        fprintf(stderr, "tj3Decompress8: %s\n", tj3GetErrorStr(dec));
        tj3Free(jpeg);
        tj3Destroy(enc);
        tj3Destroy(dec);
        free(src);
        free(dst);
        return 4;
    }

    FILE *fp = fopen(argv[1], "wb");
    if (!fp) {
        tj3Free(jpeg);
        tj3Destroy(enc);
        tj3Destroy(dec);
        free(src);
        free(dst);
        return 6;
    }
    int ok = 0;
    ok |= write_be_u32(fp, (uint32_t)w);
    ok |= write_be_u32(fp, (uint32_t)h);
    ok |= fputc(bpp, fp) == EOF ? -1 : 0;
    if (fwrite(dst, 1, (size_t)w * h * bpp, fp) != (size_t)w * h * bpp) {
        ok = -1;
    }
    if (fclose(fp) != 0) {
        ok = -1;
    }

    tj3Free(jpeg);
    tj3Destroy(enc);
    tj3Destroy(dec);
    free(src);
    free(dst);

    return ok == 0 ? 0 : 7;
}

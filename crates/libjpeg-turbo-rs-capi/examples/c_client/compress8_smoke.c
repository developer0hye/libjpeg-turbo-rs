/* A1-3: C smoke client for tj3Compress8.
 *
 * Links against the `libjpeg-turbo-rs-capi` cdylib and round-trips a
 * 64x64 RGB image: tj3Init -> tj3Set(quality) -> tj3Compress8 ->
 * verify JPEG SOI marker -> tj3Free -> tj3Destroy.
 *
 * Intentionally avoids a standard `turbojpeg.h` include so this file
 * stays self-contained and compiles even when libjpeg-turbo headers
 * are not installed on the host.
 */

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef void *tjhandle;

/* TJPARAM identifiers (must match turbojpeg.h) */
#define TJPARAM_QUALITY 3
#define TJPARAM_SUBSAMP 4

/* TJPF (pixel format) identifiers */
#define TJPF_RGB 0

/* TJSAMP (chroma subsampling) identifiers */
#define TJSAMP_444 0

/* TJINIT flags */
#define TJINIT_COMPRESS 1

extern tjhandle tj3Init(int initType);
extern void tj3Destroy(tjhandle handle);
extern int tj3Set(tjhandle handle, int param, int value);
extern int tj3Get(tjhandle handle, int param);
extern const char *tj3GetErrorStr(tjhandle handle);
extern int tj3Compress8(tjhandle handle, const unsigned char *srcBuf, int width,
                        int pitch, int height, int pixelFormat,
                        unsigned char **jpegBuf, size_t *jpegSize);
extern void tj3Free(void *ptr);

int main(void) {
    const int w = 64;
    const int h = 64;
    const int bpp = 3; /* RGB */
    unsigned char *src = (unsigned char *)malloc((size_t)w * h * bpp);
    if (!src) {
        fprintf(stderr, "malloc src failed\n");
        return 1;
    }
    /* Synthetic gradient: r = x, g = y, b = (x + y) / 2 */
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            unsigned char *p = src + (y * w + x) * bpp;
            p[0] = (unsigned char)x;
            p[1] = (unsigned char)y;
            p[2] = (unsigned char)((x + y) / 2);
        }
    }

    tjhandle h1 = tj3Init(TJINIT_COMPRESS);
    if (!h1) {
        fprintf(stderr, "tj3Init failed\n");
        free(src);
        return 2;
    }
    if (tj3Set(h1, TJPARAM_QUALITY, 80) != 0) {
        fprintf(stderr, "tj3Set(QUALITY)=%d failed: %s\n", 80,
                tj3GetErrorStr(h1));
        tj3Destroy(h1);
        free(src);
        return 3;
    }
    if (tj3Set(h1, TJPARAM_SUBSAMP, TJSAMP_444) != 0) {
        fprintf(stderr, "tj3Set(SUBSAMP)=%d failed: %s\n", TJSAMP_444,
                tj3GetErrorStr(h1));
        tj3Destroy(h1);
        free(src);
        return 4;
    }

    unsigned char *jpeg = NULL;
    size_t jpeg_size = 0;
    int rc = tj3Compress8(h1, src, w, 0 /* pitch */, h, TJPF_RGB, &jpeg,
                          &jpeg_size);
    if (rc != 0 || !jpeg || jpeg_size < 4) {
        fprintf(stderr,
                "tj3Compress8 rc=%d size=%zu ptr=%p err=%s\n",
                rc, jpeg_size, (void *)jpeg, tj3GetErrorStr(h1));
        tj3Destroy(h1);
        free(src);
        return 5;
    }
    /* JPEG starts with SOI 0xFFD8 */
    if (jpeg[0] != 0xFF || jpeg[1] != 0xD8) {
        fprintf(stderr,
                "output does not begin with SOI (got %02x %02x)\n",
                jpeg[0], jpeg[1]);
        tj3Free(jpeg);
        tj3Destroy(h1);
        free(src);
        return 6;
    }
    /* JPEG ends with EOI 0xFFD9 */
    if (jpeg[jpeg_size - 2] != 0xFF || jpeg[jpeg_size - 1] != 0xD9) {
        fprintf(stderr,
                "output does not end with EOI (got %02x %02x)\n",
                jpeg[jpeg_size - 2], jpeg[jpeg_size - 1]);
        tj3Free(jpeg);
        tj3Destroy(h1);
        free(src);
        return 7;
    }

    printf("tj3Compress8 OK: %zu bytes\n", jpeg_size);
    tj3Free(jpeg);
    tj3Destroy(h1);
    free(src);
    return 0;
}

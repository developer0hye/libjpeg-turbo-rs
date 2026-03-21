#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <turbojpeg.h>

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

int main(int argc, char **argv) {
    const char *path = "tests/fixtures/gradient_640x480.jpg";
    if (argc > 1) path = argv[1];

    FILE *f = fopen(path, "rb");
    if (!f) { perror("fopen"); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *jpegBuf = malloc(fsize);
    fread(jpegBuf, 1, fsize, f);
    fclose(f);

    tjhandle handle = tj3Init(TJINIT_DECOMPRESS);
    tj3Set(handle, TJPARAM_FASTUPSAMPLE, 0);
    tj3Set(handle, TJPARAM_FASTDCT, 0);

    if (tj3DecompressHeader(handle, jpegBuf, fsize) < 0) {
        fprintf(stderr, "header: %s\n", tj3GetErrorStr(handle));
        return 1;
    }

    int width = tj3Get(handle, TJPARAM_JPEGWIDTH);
    int height = tj3Get(handle, TJPARAM_JPEGHEIGHT);
    printf("Image: %dx%d\n", width, height);

    int pixfmt = TJPF_RGB;
    int pitch = width * tjPixelSize[pixfmt];
    unsigned char *imgBuf = malloc(pitch * height);

    /* Warmup */
    for (int i = 0; i < 100; i++) {
        tj3Decompress8(handle, jpegBuf, fsize, imgBuf, pitch, pixfmt);
    }

    /* Benchmark */
    int iters = 5000;
    double start = now_ns();
    for (int i = 0; i < iters; i++) {
        tj3Decompress8(handle, jpegBuf, fsize, imgBuf, pitch, pixfmt);
    }
    double elapsed = now_ns() - start;
    printf("libjpeg-turbo 3.1.3 (C, NEON): %.3f us/decode (%d iters)\n",
           elapsed / iters / 1000.0, iters);

    /* Scalar-only (disable SIMD) */
    tj3Destroy(handle);
    handle = tj3Init(TJINIT_DECOMPRESS);
    tj3Set(handle, TJPARAM_FASTUPSAMPLE, 0);
    tj3Set(handle, TJPARAM_FASTDCT, 0);
    /* No direct way to disable SIMD in TJ3 API, skip scalar comparison */

    free(imgBuf);
    free(jpegBuf);
    tj3Destroy(handle);
    return 0;
}

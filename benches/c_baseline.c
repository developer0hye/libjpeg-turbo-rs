#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <turbojpeg.h>

static double now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}

static void bench_file(const char *path, int iters) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "skip: %s (not found)\n", path); return; }
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
        fprintf(stderr, "header error: %s (%s)\n", path, tj3GetErrorStr(handle));
        free(jpegBuf); tj3Destroy(handle);
        return;
    }

    int width = tj3Get(handle, TJPARAM_JPEGWIDTH);
    int height = tj3Get(handle, TJPARAM_JPEGHEIGHT);

    int pixfmt = TJPF_RGB;
    int pitch = width * tjPixelSize[pixfmt];
    unsigned char *imgBuf = malloc(pitch * height);

    /* Warmup */
    for (int i = 0; i < 100; i++) {
        tj3Decompress8(handle, jpegBuf, fsize, imgBuf, pitch, pixfmt);
    }

    /* Benchmark */
    double start = now_ns();
    for (int i = 0; i < iters; i++) {
        tj3Decompress8(handle, jpegBuf, fsize, imgBuf, pitch, pixfmt);
    }
    double elapsed = now_ns() - start;
    double us = elapsed / iters / 1000.0;

    printf("%-35s %4dx%-4d  %10.1f µs  (%d iters)\n",
           path, width, height, us, iters);

    free(imgBuf);
    free(jpegBuf);
    tj3Destroy(handle);
}

int main(void) {
    printf("%-35s %-10s %12s\n", "File", "Size", "Time");
    printf("-------------------------------------------------------------------\n");

    /* Resolution scaling (4:2:0, photo-like) */
    bench_file("tests/fixtures/photo_64x64_420.jpg",     20000);
    bench_file("tests/fixtures/photo_320x240_420.jpg",    5000);
    bench_file("tests/fixtures/gradient_640x480.jpg",     5000);
    bench_file("tests/fixtures/photo_1280x720_420.jpg",    2000);
    bench_file("tests/fixtures/photo_1920x1080_420.jpg",    500);
    bench_file("tests/fixtures/photo_2560x1440_420.jpg",    200);
    bench_file("tests/fixtures/photo_3840x2160_420.jpg",    100);

    /* Subsampling modes */
    bench_file("tests/fixtures/photo_320x240_444.jpg",    5000);
    bench_file("tests/fixtures/photo_320x240_422.jpg",    5000);
    bench_file("tests/fixtures/photo_640x480_444.jpg",    5000);
    bench_file("tests/fixtures/photo_640x480_422.jpg",    5000);
    bench_file("tests/fixtures/photo_1920x1080_444.jpg",   500);
    bench_file("tests/fixtures/photo_1920x1080_422.jpg",   500);

    /* Content types */
    bench_file("tests/fixtures/graphic_640x480_420.jpg",  5000);
    bench_file("tests/fixtures/checker_640x480_420.jpg",  5000);
    bench_file("tests/fixtures/graphic_1920x1080_420.jpg", 500);

    /* Restart markers */
    bench_file("tests/fixtures/photo_640x480_420_rst.jpg", 5000);

    return 0;
}

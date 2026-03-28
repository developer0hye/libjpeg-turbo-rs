/**
 * C libjpeg-turbo decoding benchmark matrix (Linux version).
 *
 * Compile:
 *   cc -O2 -o bench_c_decode_linux examples/bench_c_decode_linux.c \
 *      -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -ljpeg \
 *      -Wl,-rpath,$CONDA_PREFIX/lib
 *
 * Run:
 *   ./bench_c_decode_linux
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <jpeglib.h>

typedef struct {
    const char *path;
    const char *name;
    int iters;
} BenchCase;

static const BenchCase CASES[] = {
    {"tests/fixtures/photo_64x64_420.jpg",          "photo_64x64_420",       20000},
    {"tests/fixtures/photo_320x240_420.jpg",         "photo_320x240_420",     5000},
    {"tests/fixtures/gradient_640x480.jpg",           "decode_640x480",        5000},
    {"tests/fixtures/photo_1280x720_420.jpg",        "photo_1280x720_420",    2000},
    {"tests/fixtures/photo_1920x1080_420.jpg",       "photo_1920x1080_420",    500},
    {"tests/fixtures/photo_2560x1440_420.jpg",       "photo_2560x1440_420",    200},
    {"tests/fixtures/photo_3840x2160_420.jpg",       "photo_3840x2160_420",    100},
    {"tests/fixtures/photo_640x480_444.jpg",          "photo_640x480_444",     5000},
    {"tests/fixtures/photo_640x480_422.jpg",          "photo_640x480_422",     5000},
    {"tests/fixtures/photo_1920x1080_444.jpg",       "photo_1920x1080_444",    500},
    {"tests/fixtures/photo_1920x1080_422.jpg",       "photo_1920x1080_422",    500},
    {"tests/fixtures/graphic_640x480_420.jpg",       "graphic_640x480_420",   5000},
    {"tests/fixtures/checker_640x480_420.jpg",       "checker_640x480_420",   5000},
    {"tests/fixtures/graphic_1920x1080_420.jpg",     "graphic_1920x1080_420", 1000},
    {"tests/fixtures/photo_640x480_420_rst.jpg",     "photo_640x480_420_rst", 5000},
    {"tests/fixtures/photo_640x480_444_prog.jpg",    "prog_640x480_444",      5000},
    {"tests/fixtures/photo_640x480_422_prog.jpg",    "prog_640x480_422",      5000},
    {NULL, NULL, 0}
};

static unsigned char *read_file(const char *path, size_t *out_len) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    *out_len = (size_t)ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *buf = (unsigned char *)malloc(*out_len);
    fread(buf, 1, *out_len, f);
    fclose(f);
    return buf;
}

static int decode_jpeg_mem(const unsigned char *data, size_t len,
                           int *out_w, int *out_h) {
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, data, (unsigned long)len);
    jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);

    *out_w = (int)cinfo.output_width;
    *out_h = (int)cinfo.output_height;
    int row_stride = cinfo.output_width * cinfo.output_components;
    unsigned char *row = (unsigned char *)malloc(row_stride);

    while (cinfo.output_scanline < cinfo.output_height) {
        JSAMPROW rows[1] = {row};
        jpeg_read_scanlines(&cinfo, rows, 1);
    }

    free(row);
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return 0;
}

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

int main(void) {
    printf("%-30s %10s %12s\n", "Benchmark", "Size", "Time (us)");
    for (int i = 0; i < 60; i++) putchar('-');
    putchar('\n');

    for (const BenchCase *c = CASES; c->path; c++) {
        size_t len;
        unsigned char *data = read_file(c->path, &len);
        if (!data) {
            fprintf(stderr, "skip: %s (not found)\n", c->path);
            continue;
        }

        int w, h;
        /* Warmup */
        for (int i = 0; i < 200; i++) {
            decode_jpeg_mem(data, len, &w, &h);
        }

        /* Benchmark */
        double t0 = now_us();
        for (int i = 0; i < c->iters; i++) {
            decode_jpeg_mem(data, len, &w, &h);
        }
        double t1 = now_us();
        double us = (t1 - t0) / c->iters;

        printf("%-30s %4dx%-4d %10.1f\n", c->name, w, h, us);

        free(data);
    }
    return 0;
}

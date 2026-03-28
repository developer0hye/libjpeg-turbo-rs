/**
 * C libjpeg-turbo decoding benchmark matrix (baseline + progressive).
 *
 * Compile (macOS with Homebrew libjpeg-turbo):
 *   cc -O2 -o bench_c_decode_matrix examples/bench_c_decode_matrix.c \
 *      -I/opt/homebrew/opt/jpeg-turbo/include \
 *      -L/opt/homebrew/opt/jpeg-turbo/lib -ljpeg \
 *      -Wl,-rpath,/opt/homebrew/opt/jpeg-turbo/lib
 *
 * Run:
 *   ./bench_c_decode_matrix
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mach/mach_time.h>
#include <jpeglib.h>

typedef struct {
    const char *path;
    int iters;
} BenchCase;

static const BenchCase CASES[] = {
    /* Resolution scaling (4:2:0, baseline) */
    {"tests/fixtures/photo_64x64_420.jpg",       20000},
    {"tests/fixtures/photo_320x240_420.jpg",      5000},
    {"tests/fixtures/gradient_640x480.jpg",        5000},
    {"tests/fixtures/photo_1280x720_420.jpg",     2000},
    {"tests/fixtures/photo_1920x1080_420.jpg",     500},
    {"tests/fixtures/photo_2560x1440_420.jpg",     200},
    {"tests/fixtures/photo_3840x2160_420.jpg",     100},
    /* Subsampling modes (baseline) */
    {"tests/fixtures/photo_640x480_444.jpg",       5000},
    {"tests/fixtures/photo_640x480_422.jpg",       5000},
    {"tests/fixtures/photo_1920x1080_444.jpg",      500},
    {"tests/fixtures/photo_1920x1080_422.jpg",      500},
    /* Progressive */
    {"tests/fixtures/photo_640x480_444_prog.jpg",  5000},
    {"tests/fixtures/photo_640x480_422_prog.jpg",  5000},
    {"tests/fixtures/photo_1920x1080_420_prog.jpg", 500},
    {"tests/fixtures/photo_1920x1080_444_prog.jpg", 500},
    {"tests/fixtures/photo_3840x2160_420_prog.jpg", 100},
    {NULL, 0}
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
    static mach_timebase_info_data_t tb = {0, 0};
    if (tb.denom == 0) mach_timebase_info(&tb);
    uint64_t t = mach_absolute_time();
    return (double)t * tb.numer / tb.denom / 1000.0;
}

int main(void) {
    printf("%-50s %10s %12s %8s\n", "File", "Size", "Time", "Iters");
    for (int i = 0; i < 80; i++) putchar('-');
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
        for (int i = 0; i < 100; i++) {
            decode_jpeg_mem(data, len, &w, &h);
        }

        /* Benchmark */
        double t0 = now_us();
        for (int i = 0; i < c->iters; i++) {
            decode_jpeg_mem(data, len, &w, &h);
        }
        double t1 = now_us();
        double us = (t1 - t0) / c->iters;

        printf("%-50s %4dx%-4d %10.1f µs  (%d iters)\n",
               c->path, w, h, us, c->iters);

        free(data);
    }
    return 0;
}

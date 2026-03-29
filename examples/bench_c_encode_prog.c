/**
 * C libjpeg-turbo progressive encoding benchmark (Linux version).
 *
 * Compile:
 *   cc -O2 -o bench_c_encode_prog examples/bench_c_encode_prog.c \
 *      -I$CONDA_PREFIX/include -L$CONDA_PREFIX/lib -ljpeg \
 *      -Wl,-rpath,$CONDA_PREFIX/lib
 *
 * Run:
 *   ./bench_c_encode_prog
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <jpeglib.h>

static unsigned char *read_jpeg(const char *path, int *out_w, int *out_h) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);
    jpeg_stdio_src(&cinfo, f);
    jpeg_read_header(&cinfo, TRUE);
    cinfo.out_color_space = JCS_RGB;
    jpeg_start_decompress(&cinfo);

    int w = (int)cinfo.output_width;
    int h = (int)cinfo.output_height;
    int stride = w * 3;
    unsigned char *pixels = (unsigned char *)malloc((size_t)stride * (size_t)h);

    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char *row = pixels + cinfo.output_scanline * stride;
        jpeg_read_scanlines(&cinfo, &row, 1);
    }

    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    fclose(f);

    *out_w = w;
    *out_h = h;
    return pixels;
}

static unsigned long compress_jpeg_progressive(const unsigned char *pixels, int w, int h,
                                               int quality, int h_samp, int v_samp,
                                               unsigned char **out_buf) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    unsigned long out_size = 0;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    jpeg_mem_dest(&cinfo, out_buf, &out_size);

    cinfo.image_width = (JDIMENSION)w;
    cinfo.image_height = (JDIMENSION)h;
    cinfo.input_components = 3;
    cinfo.in_color_space = JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE);

    cinfo.comp_info[0].h_samp_factor = h_samp;
    cinfo.comp_info[0].v_samp_factor = v_samp;
    cinfo.comp_info[1].h_samp_factor = 1;
    cinfo.comp_info[1].v_samp_factor = 1;
    cinfo.comp_info[2].h_samp_factor = 1;
    cinfo.comp_info[2].v_samp_factor = 1;

    jpeg_simple_progression(&cinfo);

    jpeg_start_compress(&cinfo, TRUE);

    int stride = w * 3;
    while (cinfo.next_scanline < cinfo.image_height) {
        const unsigned char *row = pixels + cinfo.next_scanline * stride;
        jpeg_write_scanlines(&cinfo, (JSAMPARRAY)&row, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);

    return out_size;
}

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

typedef struct {
    const char *fixture;
    int h_samp;
    int v_samp;
    const char *sub_str;
    int iters;
} EncodeCase;

int main(void) {
    EncodeCase cases[] = {
        {"tests/fixtures/photo_64x64_420.jpg",       2, 2, "420", 20000},
        {"tests/fixtures/photo_320x240_420.jpg",      2, 2, "420", 5000},
        {"tests/fixtures/photo_640x480_422.jpg",      2, 2, "420", 5000},
        {"tests/fixtures/photo_1280x720_420.jpg",     2, 2, "420", 2000},
        {"tests/fixtures/photo_1920x1080_420.jpg",    2, 2, "420", 500},
        {"tests/fixtures/photo_640x480_444.jpg",      1, 1, "444", 5000},
        {"tests/fixtures/photo_640x480_422.jpg",      2, 1, "422", 5000},
        {"tests/fixtures/photo_1920x1080_444.jpg",    1, 1, "444", 500},
        {"tests/fixtures/photo_1920x1080_422.jpg",    2, 1, "422", 500},
    };
    int ncases = (int)(sizeof(cases) / sizeof(cases[0]));

    printf("%-50s %10s %12s %8s\n", "Case", "Size", "Time", "Iters");
    for (int i = 0; i < 85; i++) putchar('-');
    putchar('\n');

    for (int c = 0; c < ncases; c++) {
        int w, h;
        unsigned char *pixels = read_jpeg(cases[c].fixture, &w, &h);
        if (!pixels) {
            fprintf(stderr, "skip: %s (not found)\n", cases[c].fixture);
            continue;
        }

        /* Warmup */
        for (int i = 0; i < 100; i++) {
            unsigned char *buf = NULL;
            compress_jpeg_progressive(pixels, w, h, 75, cases[c].h_samp, cases[c].v_samp, &buf);
            free(buf);
        }

        /* Benchmark */
        double t0 = now_us();
        for (int i = 0; i < cases[c].iters; i++) {
            unsigned char *buf = NULL;
            compress_jpeg_progressive(pixels, w, h, 75, cases[c].h_samp, cases[c].v_samp, &buf);
            free(buf);
        }
        double elapsed = now_us() - t0;
        double per_iter = elapsed / (double)cases[c].iters;

        printf("C_prog_encode_%4dx%-4d_%-3s                        %4dx%-4d %10.1f us  (%d iters)\n",
               w, h, cases[c].sub_str, w, h, per_iter, cases[c].iters);

        free(pixels);
    }

    return 0;
}

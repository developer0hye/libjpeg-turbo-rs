/*
 * libtiff end-to-end JPEG round-trip via our libjpeg shim.
 *
 * This program exercises the jpeg_write_raw_data / jpeg_read_raw_data path
 * that libtiff uses internally when encoding/decoding a COMPRESSION_JPEG TIFF
 * strip.  libtiff calls TIFFjpeg_write_raw_data / TIFFjpeg_read_raw_data
 * (which in turn call the libjpeg C-API entry points) for every strip I/O
 * operation.
 *
 * Algorithm:
 *   1. Build a 64x64 RGB checkerboard in memory.
 *   2. TIFFOpen a temp file, set COMPRESSION_JPEG + JPEGQUALITY=75 +
 *      ROWSPERSTRIP=8 (one DCTSIZE worth of rows per strip), write all strips
 *      via TIFFWriteEncodedStrip.
 *   3. TIFFClose, re-open for read, read all strips via TIFFReadEncodedStrip.
 *   4. Compare with tolerance 32 LSB (Q=75 on a checker: measured max-diff
 *      typically 0 for PHOTOMETRIC_RGB; 32 is a generous safe margin).
 *   5. Exit 0 on success, 1 on pixel mismatch, 2 on API failure.
 *
 * The binary must be run with our cdylib staged on DYLD_LIBRARY_PATH /
 * LD_LIBRARY_PATH so libtiff resolves its JPEG calls against our shim rather
 * than the system libjpeg.  See run.sh.
 */

#include <tiffio.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define IMG_WIDTH      64
#define IMG_HEIGHT     64
#define SPP            3       /* samples per pixel (RGB) */
/* ROWSPERSTRIP must be a multiple of DCTSIZE (8) for COMPRESSION_JPEG.
 * Using DCTSIZE itself (8) keeps each strip small and exercises the strip
 * loop maximally. */
#define ROWS_PER_STRIP 8

/* Maximum per-channel absolute difference allowed after a Q=75 JPEG round-trip.
 * For PHOTOMETRIC_RGB the measured max-diff is typically 0 because libtiff
 * applies the JPEG codec in YCbCr space internally but converts back to RGB
 * before returning the samples.  32 is a conservative safe upper bound. */
#define MAX_DIFF_TOLERANCE 32

static void build_checkerboard(uint8_t *buf)
{
    for (int y = 0; y < IMG_HEIGHT; y++) {
        for (int x = 0; x < IMG_WIDTH; x++) {
            /* 8x8 squares so each square is exactly one DCT block. */
            int bright = (((x / 8) + (y / 8)) & 1) == 0;
            buf[(y * IMG_WIDTH + x) * SPP + 0] = bright ? 220 : 35;
            buf[(y * IMG_WIDTH + x) * SPP + 1] = bright ? 180 : 60;
            buf[(y * IMG_WIDTH + x) * SPP + 2] = bright ? 120 : 90;
        }
    }
}

int main(int argc, char **argv)
{
    const char *tif_path = (argc > 1) ? argv[1]
                                      : "/tmp/libtiff_integration_test.tif";

    /* Suppress libtiff's default error/warning output so the test produces
     * clean stdout.  We surface errors only via the return-code checks below. */
    TIFFSetWarningHandler(NULL);
    TIFFSetErrorHandler(NULL);

    /* -----------------------------------------------------------------------
     * Allocate and fill the source image.
     * -------------------------------------------------------------------- */
    size_t img_bytes = (size_t)IMG_WIDTH * IMG_HEIGHT * SPP;
    uint8_t *src = (uint8_t *)malloc(img_bytes);
    if (!src) {
        fprintf(stderr, "FAIL: malloc src\n");
        return 2;
    }
    build_checkerboard(src);

    /* -----------------------------------------------------------------------
     * Write phase: strip-based JPEG encoding.
     * TIFFWriteEncodedStrip is the correct API for COMPRESSION_JPEG; it
     * delivers exactly one strip (ROWSPERSTRIP rows) per call, matching the
     * iMCU-row granularity that libjpeg requires.  libtiff in turn calls
     * jpeg_write_raw_data (or the equivalent scanline path) inside its
     * TIFFjpeg_write_raw_data helper.
     * -------------------------------------------------------------------- */
    TIFF *tif = TIFFOpen(tif_path, "w");
    if (!tif) {
        fprintf(stderr, "FAIL: TIFFOpen(\"%s\", \"w\")\n", tif_path);
        free(src);
        return 2;
    }

    TIFFSetField(tif, TIFFTAG_IMAGEWIDTH,      (uint32_t)IMG_WIDTH);
    TIFFSetField(tif, TIFFTAG_IMAGELENGTH,     (uint32_t)IMG_HEIGHT);
    TIFFSetField(tif, TIFFTAG_SAMPLESPERPIXEL, (uint16_t)SPP);
    TIFFSetField(tif, TIFFTAG_BITSPERSAMPLE,   (uint16_t)8);
    TIFFSetField(tif, TIFFTAG_ORIENTATION,     ORIENTATION_TOPLEFT);
    TIFFSetField(tif, TIFFTAG_PLANARCONFIG,    PLANARCONFIG_CONTIG);
    TIFFSetField(tif, TIFFTAG_PHOTOMETRIC,     PHOTOMETRIC_RGB);
    TIFFSetField(tif, TIFFTAG_COMPRESSION,     COMPRESSION_JPEG);
    TIFFSetField(tif, TIFFTAG_JPEGQUALITY,     75);
    TIFFSetField(tif, TIFFTAG_ROWSPERSTRIP,    (uint32_t)ROWS_PER_STRIP);

    tstrip_t nstrips  = TIFFNumberOfStrips(tif);
    tsize_t  sbytes   = (tsize_t)(IMG_WIDTH * ROWS_PER_STRIP * SPP);

    for (tstrip_t s = 0; s < nstrips; s++) {
        const uint8_t *strip_ptr = src + (size_t)s * (size_t)sbytes;
        tsize_t written = TIFFWriteEncodedStrip(tif, s, (void *)strip_ptr,
                                                sbytes);
        if (written < 0) {
            fprintf(stderr, "FAIL: TIFFWriteEncodedStrip strip=%u\n",
                    (unsigned)s);
            TIFFClose(tif);
            free(src);
            return 2;
        }
    }
    TIFFClose(tif);

    /* -----------------------------------------------------------------------
     * Read phase: strip-based JPEG decoding.
     * -------------------------------------------------------------------- */
    tif = TIFFOpen(tif_path, "r");
    if (!tif) {
        fprintf(stderr, "FAIL: TIFFOpen(\"%s\", \"r\")\n", tif_path);
        free(src);
        return 2;
    }

    /* Verify the header survived the round-trip. */
    uint32_t rw = 0, rh = 0;
    uint16_t rspp = 0;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH,      &rw);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH,     &rh);
    TIFFGetField(tif, TIFFTAG_SAMPLESPERPIXEL, &rspp);

    if ((int)rw != IMG_WIDTH || (int)rh != IMG_HEIGHT || rspp != SPP) {
        fprintf(stderr,
            "FAIL: header mismatch: got %ux%u spp=%u, expected %ux%u spp=%u\n",
            rw, rh, (unsigned)rspp,
            (unsigned)IMG_WIDTH, (unsigned)IMG_HEIGHT, (unsigned)SPP);
        TIFFClose(tif);
        free(src);
        return 2;
    }

    uint8_t *dst = (uint8_t *)malloc(img_bytes);
    if (!dst) {
        fprintf(stderr, "FAIL: malloc dst\n");
        TIFFClose(tif);
        free(src);
        return 2;
    }

    nstrips = TIFFNumberOfStrips(tif);
    for (tstrip_t s = 0; s < nstrips; s++) {
        uint8_t *strip_ptr = dst + (size_t)s * (size_t)sbytes;
        tsize_t got = TIFFReadEncodedStrip(tif, s, strip_ptr, sbytes);
        if (got < 0) {
            fprintf(stderr, "FAIL: TIFFReadEncodedStrip strip=%u\n",
                    (unsigned)s);
            TIFFClose(tif);
            free(src);
            free(dst);
            return 2;
        }
    }
    TIFFClose(tif);

    /* -----------------------------------------------------------------------
     * Pixel comparison with JPEG-loss tolerance.
     * -------------------------------------------------------------------- */
    int max_diff = 0;
    int fail_row = -1, fail_col = -1, fail_ch = -1;
    for (int y = 0; y < IMG_HEIGHT; y++) {
        for (int x = 0; x < IMG_WIDTH; x++) {
            for (int c = 0; c < SPP; c++) {
                int idx = (y * IMG_WIDTH + x) * SPP + c;
                int d   = abs((int)src[idx] - (int)dst[idx]);
                if (d > max_diff) {
                    max_diff = d;
                    fail_row = y;
                    fail_col = x;
                    fail_ch  = c;
                }
            }
        }
    }

    free(src);
    free(dst);

    if (max_diff > MAX_DIFF_TOLERANCE) {
        fprintf(stderr,
            "FAIL: pixel mismatch at row=%d col=%d ch=%d "
            "max_diff=%d (tolerance=%d)\n",
            fail_row, fail_col, fail_ch, max_diff, MAX_DIFF_TOLERANCE);
        return 1;
    }

    printf("OK libtiff_integration max_diff=%d tolerance=%d\n",
           max_diff, MAX_DIFF_TOLERANCE);
    return 0;
}

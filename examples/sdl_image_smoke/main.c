/* P2-10: SDL_image decode round-trip smoke test against our cdylib.
 *
 * SDL_image only routes JPEG *decode* through libjpeg
 * (`IMG_LoadJPG_RW` -> jpeg_create_decompress + jpeg_mem_src +
 * jpeg_read_scanlines). The library's saver (`IMG_SaveJPG_RW`) uses
 * STB_image_write internally — there's no encode path through libjpeg
 * to exercise.
 *
 * The harness reads a pre-encoded JPEG plus its reference PPM, decodes
 * the JPEG via SDL_image (with our cdylib LD_PRELOAD'd in), and
 * compares the decoded surface to the reference PPM channel-by-channel.
 * Encode happens outside this binary (the Rust wrapper drives it via
 * the cjpeg CLI / library encoder) so encode and decode paths stay
 * cleanly separated.
 *
 * Usage: sdl_image_smoke <jpeg> <reference.ppm> <min-psnr>
 *
 * Exit codes:
 *   0  PSNR >= min — pass
 *   1  reference PPM read failure
 *   2  JPEG read failure (file I/O)
 *   3  SDL_Init failed
 *   4  IMG_Init failed (or SDL_image not built with JPEG support)
 *   5  IMG_Load_RW returned NULL
 *   6  surface dimension mismatch vs reference
 *   7  PSNR below threshold
 *  99  usage error
 */

#include <SDL.h>
#include <SDL_image.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *read_ppm(const char *path, int *w_out, int *h_out) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  char magic[3] = {0};
  int w = 0, h = 0, maxv = 0;
  if (fscanf(f, "%2s %d %d %d", magic, &w, &h, &maxv) != 4 ||
      strcmp(magic, "P6") != 0 || maxv != 255 || w <= 0 || h <= 0) {
    fclose(f);
    return NULL;
  }
  fgetc(f);
  size_t n = (size_t)w * (size_t)h * 3;
  unsigned char *buf = (unsigned char *)malloc(n);
  if (!buf || fread(buf, 1, n, f) != n) {
    free(buf);
    fclose(f);
    return NULL;
  }
  fclose(f);
  *w_out = w;
  *h_out = h;
  return buf;
}

static unsigned char *slurp(const char *path, size_t *out_size) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  fseek(f, 0, SEEK_END);
  long sz = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (sz <= 0) {
    fclose(f);
    return NULL;
  }
  unsigned char *buf = (unsigned char *)malloc((size_t)sz);
  if (!buf || fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
    free(buf);
    fclose(f);
    return NULL;
  }
  fclose(f);
  *out_size = (size_t)sz;
  return buf;
}

int main(int argc, char **argv) {
  if (argc != 4) {
    fprintf(stderr, "usage: %s <jpeg> <reference.ppm> <min-psnr>\n", argv[0]);
    return 99;
  }
  const char *jpeg_path = argv[1];
  const char *ref_ppm = argv[2];
  double min_psnr = atof(argv[3]);

  int rw = 0, rh = 0;
  unsigned char *ref = read_ppm(ref_ppm, &rw, &rh);
  if (!ref) {
    fprintf(stderr, "failed to read reference PPM: %s\n", ref_ppm);
    return 1;
  }

  size_t jpeg_size = 0;
  unsigned char *jpeg = slurp(jpeg_path, &jpeg_size);
  if (!jpeg) {
    fprintf(stderr, "failed to slurp jpeg: %s\n", jpeg_path);
    free(ref);
    return 2;
  }

  /* SDL_image's JPEG decode path needs surfaces (SDL_CreateRGBSurface)
   * but does *not* need a display — SDL_INIT_VIDEO would fail on
   * headless CI runners ("video driver did not add any displays"),
   * which is exactly where this test must run. Force the dummy video
   * driver so SDL_Init(SDL_INIT_VIDEO) reports success without an
   * attached display, and allow the env to be overridden by users who
   * actually have a display. */
  if (!getenv("SDL_VIDEODRIVER")) {
    setenv("SDL_VIDEODRIVER", "dummy", 0);
  }
  if (SDL_Init(SDL_INIT_VIDEO) != 0) {
    fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
    free(jpeg);
    free(ref);
    return 3;
  }
  /* IMG_Init returns the bitmask of formats successfully initialised.
   * We require IMG_INIT_JPG specifically — anything else means the
   * SDL_image build does not link libjpeg (e.g. STB-only build) and
   * there is nothing for our cdylib to override. */
  int initted = IMG_Init(IMG_INIT_JPG);
  if ((initted & IMG_INIT_JPG) == 0) {
    fprintf(stderr, "IMG_Init JPG failed: %s\n", IMG_GetError());
    SDL_Quit();
    free(jpeg);
    free(ref);
    return 4;
  }

  SDL_RWops *rwops = SDL_RWFromMem(jpeg, (int)jpeg_size);
  /* freesrc=1 → IMG closes the rwops for us. The jpeg backing buffer
   * must outlive the call (rwops references it directly), so we free
   * `jpeg` only after the surface is materialised. */
  SDL_Surface *surface = IMG_LoadTyped_RW(rwops, 1, "JPG");
  if (!surface) {
    fprintf(stderr, "IMG_LoadTyped_RW failed: %s\n", IMG_GetError());
    IMG_Quit();
    SDL_Quit();
    free(jpeg);
    free(ref);
    return 5;
  }

  if (surface->w != rw || surface->h != rh) {
    fprintf(stderr, "dim mismatch: ref=%dx%d sdl=%dx%d\n", rw, rh, surface->w, surface->h);
    SDL_FreeSurface(surface);
    IMG_Quit();
    SDL_Quit();
    free(jpeg);
    free(ref);
    return 6;
  }

  /* Convert whatever pixel format SDL gave us to RGB24 so we can
   * compare against the PPM directly. SDL_ConvertSurfaceFormat
   * allocates a new surface in the target format. */
  SDL_Surface *rgb = SDL_ConvertSurfaceFormat(surface, SDL_PIXELFORMAT_RGB24, 0);
  SDL_FreeSurface(surface);
  if (!rgb) {
    fprintf(stderr, "SDL_ConvertSurfaceFormat failed: %s\n", SDL_GetError());
    IMG_Quit();
    SDL_Quit();
    free(jpeg);
    free(ref);
    return 5;
  }

  long long sse = 0;
  size_t count = 0;
  /* SDL surfaces use `pitch` (bytes per row including padding) — copy
   * row by row to avoid pitch != w*3. */
  if (SDL_LockSurface(rgb) != 0) {
    fprintf(stderr, "SDL_LockSurface failed: %s\n", SDL_GetError());
    SDL_FreeSurface(rgb);
    IMG_Quit();
    SDL_Quit();
    free(jpeg);
    free(ref);
    return 5;
  }
  for (int y = 0; y < rh; y++) {
    const unsigned char *row = (const unsigned char *)rgb->pixels + (size_t)y * (size_t)rgb->pitch;
    const unsigned char *ref_row = ref + (size_t)y * (size_t)rw * 3;
    for (int x = 0; x < rw * 3; x++) {
      int d = (int)row[x] - (int)ref_row[x];
      sse += (long long)d * d;
    }
    count += (size_t)rw * 3;
  }
  SDL_UnlockSurface(rgb);

  double mse = (count > 0) ? (double)sse / (double)count : 0.0;
  double psnr = (mse == 0.0) ? INFINITY : 10.0 * log10(255.0 * 255.0 / mse);
  printf("PSNR=%.3f dB (min=%.3f) mse=%.6f bytes=%zu\n", psnr, min_psnr, mse, count);

  SDL_FreeSurface(rgb);
  IMG_Quit();
  SDL_Quit();
  free(jpeg);
  free(ref);
  return (psnr < min_psnr) ? 7 : 0;
}

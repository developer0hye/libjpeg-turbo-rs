/*
 * P4-165 C oracle: how many planes the YUV entry points touch for TJSAMP_GRAY.
 *
 * `tj3YUVBufSize(w, align, h, TJSAMP_GRAY)` sizes exactly one plane, and the
 * eight YUV entry points are documented against that number. Upstream keeps
 * the two consistent in one place: the packed wrappers set
 * `planes[1] = planes[2] = NULL; strides[1] = strides[2] = 0` for GRAY
 * (`turbojpeg.c:1504-1507`, `:1756-1759`, `:2416-2419`, `:2732-2735`) and the
 * `…Planes8` workers then loop over `cinfo->num_components`, which
 * `setCompDefaults` / `setDecodeDefaults` have already made 1
 * (`:2502-2504`). A GRAY call therefore reads and writes plane 0 and nothing
 * else.
 *
 * This is worth an oracle rather than a transcription because the divergence
 * it pins is invisible from the outside on a correct implementation: the
 * observable difference between "sizes one plane" and "sizes three" is memory
 * the caller never allocated. The port mapped TJSAMP_GRAY onto the 4:4:4
 * geometry — right for plane 0's dimensions, wrong for the plane count — so
 * every packed path handled ~3x the contract length and every plane-array
 * path walked slots 1 and 2 of an array upstream lets the caller size at one.
 *
 * The trace makes both halves observable without ever relying on an actual
 * out-of-bounds access. Destination buffers are allocated large enough to
 * absorb a three-plane write and prefilled with GUARD_BYTE past the
 * one-plane contract length; source buffers are allocated the same way and
 * filled with SENTINEL_BYTE past it. A correct implementation leaves the
 * guard clean and never sees a sentinel byte; the port under test writes into
 * the guard (encode/decompress) and folds sentinel bytes into its output as
 * chroma (decode/compress). Every byte examined is owned by this process.
 *
 * Digests cover the *contract data* — `pw` bytes per row at the `align`
 * stride — and deliberately not the inter-row alignment padding. Upstream
 * never writes those pad bytes (its row pointers advance by the stride but
 * each row copy is `pw` wide), while this port zero-fills them. That is a
 * difference inside the caller's own buffer, about bytes upstream documents
 * nothing for, and folding it into this trace would report it as a
 * plane-count failure.
 *
 * Output is one line per case, `<geometry> <case> <fields...>`, all fields
 * deterministic so the Rust mirror can be compared verbatim.
 *
 * Usage: tj3_yuv_gray_oracle <workdir>
 *
 * The workdir receives one grayscale JPEG per geometry, written by this
 * program with `tj3Compress8`. The decompress-side cases read it back, and
 * the Rust mirror reads the very same files, so both implementations are
 * measured against identical JPEG bytes rather than against whatever each
 * one's encoder happens to produce.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <turbojpeg.h>

/* Fills the destination bytes past the one-plane contract length. A correct
 * GRAY call never writes them. */
#define GUARD_BYTE 0x5A
/* Fills the source bytes past the one-plane contract length, and the whole of
 * the chroma slots of a plane array. A correct GRAY call never reads them. */
#define SENTINEL_BYTE 0xC3
/* Headroom past the largest three-plane span any case can produce, so a
 * three-plane write lands in memory this program owns. */
#define SLACK 64
#define QUALITY 90

/* Round `value` up to a multiple of `alignment` (a power of two here). */
static int pad_up(int value, int alignment)
{
  return ((value + alignment - 1) / alignment) * alignment;
}

/* FNV-1a, 64-bit. Reproduced byte for byte in the Rust mirror; any stable
 * checksum would do, and this one needs no library. */
static unsigned long long fnv1a(const unsigned char *bytes, size_t count,
                                unsigned long long hash)
{
  size_t i;

  for (i = 0; i < count; i++) {
    hash ^= (unsigned long long)bytes[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

#define FNV_OFFSET 14695981039346656037ULL

/* Digest of `height` rows of `width` bytes read at `stride`. See the header
 * note on why the pad between rows is excluded. */
static unsigned long long row_digest(const unsigned char *plane, int width,
                                     int height, int stride)
{
  unsigned long long hash = FNV_OFFSET;
  int row;

  for (row = 0; row < height; row++)
    hash = fnv1a(&plane[(size_t)row * stride], (size_t)width, hash);
  return hash;
}

/* "clean" when every byte in `[from, to)` still holds GUARD_BYTE. */
static const char *guard_state(const unsigned char *buffer, size_t from,
                               size_t to)
{
  size_t i;

  for (i = from; i < to; i++)
    if (buffer[i] != GUARD_BYTE) return "DIRTY";
  return "clean";
}

/* The luma test pattern. Non-constant in both axes so a plane written at the
 * wrong stride, or transposed, changes the digest. */
static unsigned char y_sample(int x, int y)
{
  return (unsigned char)((x * 7 + y * 13) & 0xFF);
}

/* The packed-pixel test pattern for the encode direction. */
static void rgb_sample(int x, int y, unsigned char *out)
{
  out[0] = (unsigned char)((x * 5 + y * 3) & 0xFF);
  out[1] = (unsigned char)((x * 11 + y * 7) & 0xFF);
  out[2] = (unsigned char)((x * 3 + y * 17) & 0xFF);
}

static unsigned char *make_rgb(int width, int height)
{
  unsigned char *pixels = (unsigned char *)malloc((size_t)width * height * 3);
  int x, y;

  if (!pixels) { fprintf(stderr, "malloc\n"); exit(1); }
  for (y = 0; y < height; y++)
    for (x = 0; x < width; x++)
      rgb_sample(x, y, &pixels[((size_t)y * width + x) * 3]);
  return pixels;
}

/* A `total`-byte buffer whose first `PAD(width, align) * height` bytes are the
 * packed GRAY plane the contract describes, and whose remaining bytes are
 * SENTINEL_BYTE. The alignment padding inside each row is sentinel too:
 * upstream skips it, so a reader that folds it into the image is as wrong as
 * one that runs past the end. */
static unsigned char *make_packed_gray_source(int width, int height, int align,
                                              size_t total)
{
  unsigned char *buffer = (unsigned char *)malloc(total);
  int stride = pad_up(width, align);
  int x, y;

  if (!buffer) { fprintf(stderr, "malloc\n"); exit(1); }
  memset(buffer, SENTINEL_BYTE, total);
  for (y = 0; y < height; y++)
    for (x = 0; x < width; x++)
      buffer[(size_t)y * stride + x] = y_sample(x, y);
  return buffer;
}

/* The chroma test patterns, for the three-plane control cases below. */
static unsigned char cb_sample(int x, int y)
{
  return (unsigned char)((x * 3 + y * 29) & 0xFF);
}

static unsigned char cr_sample(int x, int y)
{
  return (unsigned char)((x * 23 + y * 5) & 0xFF);
}

/* A `total`-byte buffer holding a full three-plane packed YUV image at
 * `subsamp`, with SENTINEL_BYTE past the planes and in each row's alignment
 * padding.
 *
 * These feed the control cases: the port routes its packed entry points
 * through the *planes* functions now, for every subsampling and not only for
 * GRAY, so the three-plane path needs pinning against stock too. Without it
 * the reroute rests on a reading of the two functions rather than on a
 * measurement. */
static unsigned char *make_packed_ycbcr_source(int width, int height, int align,
                                               int subsamp, size_t total)
{
  unsigned char *buffer = (unsigned char *)malloc(total);
  size_t offset = 0;
  int component, x, y;

  if (!buffer) { fprintf(stderr, "malloc\n"); exit(1); }
  memset(buffer, SENTINEL_BYTE, total);
  for (component = 0; component < 3; component++) {
    int pw = tj3YUVPlaneWidth(component, width, subsamp);
    int ph = tj3YUVPlaneHeight(component, height, subsamp);
    int stride = pad_up(pw, align);

    for (y = 0; y < ph; y++) {
      for (x = 0; x < pw; x++) {
        unsigned char sample = component == 0 ? y_sample(x, y) :
                               component == 1 ? cb_sample(x, y) : cr_sample(x, y);

        buffer[offset + (size_t)y * stride + x] = sample;
      }
    }
    offset += (size_t)stride * ph;
  }
  return buffer;
}

/* The same plane, tightly packed, for the `…Planes8` entry points called with
 * a NULL stride array. */
static unsigned char *make_tight_gray_plane(int width, int height)
{
  unsigned char *plane = (unsigned char *)malloc((size_t)width * height);
  int x, y;

  if (!plane) { fprintf(stderr, "malloc\n"); exit(1); }
  for (y = 0; y < height; y++)
    for (x = 0; x < width; x++)
      plane[(size_t)y * width + x] = y_sample(x, y);
  return plane;
}

static unsigned char *make_filled(size_t size, int byte)
{
  unsigned char *buffer = (unsigned char *)malloc(size);

  if (!buffer) { fprintf(stderr, "malloc\n"); exit(1); }
  memset(buffer, byte, size);
  return buffer;
}

/* "yes" when every pixel of a packed RGB buffer has R == G == B.
 *
 * The structural signature of a grayscale decode: YCbCr -> RGB with
 * Cb = Cr = 128 collapses to R = G = B = Y, and nothing else does. A decoder
 * that read sentinel bytes as chroma produces colour here whatever its
 * digest. */
static const char *rgb_is_gray(const unsigned char *pixels, int width,
                               int height)
{
  size_t i, count = (size_t)width * height;

  for (i = 0; i < count; i++)
    if (pixels[i * 3] != pixels[i * 3 + 1] || pixels[i * 3] != pixels[i * 3 + 2])
      return "no";
  return "yes";
}

/* How faithfully a JPEG produced from the GRAY plane decodes back to it.
 *
 * Exact JPEG bytes are not comparable across two independent encoders, so the
 * compress-from-YUV cases assert the round trip instead: "ok" means every
 * sample came back within 16 of the source. Measured at 9-12 across the 10
 * compress cases here (stock, quality 90, a gradient that is deliberately
 * hostile to an 8x8 DCT); 16 is that worst case plus a small margin, so a
 * quantisation difference between two encoders cannot read as a plane-count
 * failure, which is the contract actually being pinned. A number in place of
 * "ok" is a real divergence: chroma folded into the luma plane moves this into
 * the hundreds. */
static const char *roundtrip_state(const unsigned char *decoded,
                                   int width, int height, char *scratch)
{
  int worst = 0, x, y;

  for (y = 0; y < height; y++) {
    for (x = 0; x < width; x++) {
      int diff = (int)decoded[(size_t)y * width + x] - (int)y_sample(x, y);

      if (diff < 0) diff = -diff;
      if (diff > worst) worst = diff;
    }
  }
  if (worst <= 16) return "ok";
  sprintf(scratch, "bad(%d)", worst);
  return scratch;
}

typedef struct {
  const char *label;
  int width;
  int height;
  int align;
} geometry;

static const geometry GEOMETRIES[] = {
  { "g64x64a1", 64, 64, 1 },
  { "g64x64a4", 64, 64, 4 },
  { "g33x17a1", 33, 17, 1 },
  { "g33x17a4", 33, 17, 4 },
  { "g17x33a1", 17, 33, 1 }
};
#define GEOMETRY_COUNT ((int)(sizeof(GEOMETRIES) / sizeof(GEOMETRIES[0])))

/* Write the shared grayscale JPEG fixture for one geometry and return its
 * bytes. Produced with `tj3Compress8` from a TJPF_GRAY source, which is the
 * one path here that does not go through the code under test. */
static unsigned char *write_gray_fixture(const char *workdir,
                                         const geometry *geo, size_t *size)
{
  tjhandle handle = tj3Init(TJINIT_COMPRESS);
  unsigned char *gray = make_tight_gray_plane(geo->width, geo->height);
  unsigned char *jpeg = NULL;
  char path[4096];
  FILE *file;

  *size = 0;
  if (!handle) { fprintf(stderr, "tj3Init\n"); exit(1); }
  if (tj3Set(handle, TJPARAM_QUALITY, QUALITY) != 0 ||
      tj3Set(handle, TJPARAM_SUBSAMP, TJSAMP_GRAY) != 0) {
    fprintf(stderr, "tj3Set: %s\n", tj3GetErrorStr(handle));
    exit(1);
  }
  if (tj3Compress8(handle, gray, geo->width, 0, geo->height, TJPF_GRAY, &jpeg,
                   size) != 0) {
    fprintf(stderr, "tj3Compress8: %s\n", tj3GetErrorStr(handle));
    exit(1);
  }
  snprintf(path, sizeof(path), "%s/%s.jpg", workdir, geo->label);
  file = fopen(path, "wb");
  if (!file || fwrite(jpeg, 1, *size, file) != *size) {
    fprintf(stderr, "write %s\n", path);
    exit(1);
  }
  fclose(file);
  free(gray);
  tj3Destroy(handle);
  return jpeg;
}

/* --- Pixels -> YUV ------------------------------------------------------ */

static void case_encode_yuv8(const geometry *geo, size_t gray_size,
                             size_t capacity)
{
  tjhandle handle = tj3Init(TJINIT_COMPRESS);
  unsigned char *pixels = make_rgb(geo->width, geo->height);
  unsigned char *dst = make_filled(capacity, GUARD_BYTE);
  int stride = pad_up(geo->width, geo->align);
  int rc;

  if (!handle) { fprintf(stderr, "tj3Init\n"); exit(1); }
  tj3Set(handle, TJPARAM_SUBSAMP, TJSAMP_GRAY);
  rc = tj3EncodeYUV8(handle, pixels, geo->width, 0, geo->height, TJPF_RGB, dst,
                     geo->align);
  printf("%s encodeyuv8 rc=%d ydigest=%016llx guard=%s\n", geo->label, rc,
         row_digest(dst, geo->width, geo->height, stride),
         guard_state(dst, gray_size, capacity));
  free(pixels);
  free(dst);
  tj3Destroy(handle);
}

static void case_encode_yuv_planes8(const geometry *geo, size_t chroma_capacity)
{
  tjhandle handle = tj3Init(TJINIT_COMPRESS);
  unsigned char *pixels = make_rgb(geo->width, geo->height);
  unsigned char *luma =
    make_filled((size_t)geo->width * geo->height, GUARD_BYTE);
  /* Slots 1 and 2 hold real, owned buffers rather than NULL so that a worker
   * which walks all three planes writes here instead of running off the end
   * of a one-element array. Upstream accepts either for GRAY: its NULL check
   * is `subsamp != TJSAMP_GRAY && (!dstPlanes[1] || !dstPlanes[2])`
   * (turbojpeg.c:1589-1590). */
  unsigned char *chroma[2];
  unsigned char *planes[3];
  int rc, i;

  if (!handle) { fprintf(stderr, "tj3Init\n"); exit(1); }
  for (i = 0; i < 2; i++) chroma[i] = make_filled(chroma_capacity, GUARD_BYTE);
  planes[0] = luma;  planes[1] = chroma[0];  planes[2] = chroma[1];
  tj3Set(handle, TJPARAM_SUBSAMP, TJSAMP_GRAY);
  rc = tj3EncodeYUVPlanes8(handle, pixels, geo->width, 0, geo->height, TJPF_RGB,
                           planes, NULL);
  printf("%s encodeplanes8 rc=%d ydigest=%016llx guard1=%s guard2=%s\n",
         geo->label, rc,
         row_digest(luma, geo->width, geo->height, geo->width),
         guard_state(chroma[0], 0, chroma_capacity),
         guard_state(chroma[1], 0, chroma_capacity));
  free(pixels);
  free(luma);
  for (i = 0; i < 2; i++) free(chroma[i]);
  tj3Destroy(handle);
}

/* --- YUV -> pixels ------------------------------------------------------ */

static void case_decode_yuv8(const geometry *geo, size_t capacity)
{
  tjhandle handle = tj3Init(TJINIT_DECOMPRESS);
  unsigned char *src =
    make_packed_gray_source(geo->width, geo->height, geo->align, capacity);
  unsigned char *pixels =
    make_filled((size_t)geo->width * geo->height * 3, GUARD_BYTE);
  int rc;

  if (!handle) { fprintf(stderr, "tj3Init\n"); exit(1); }
  tj3Set(handle, TJPARAM_SUBSAMP, TJSAMP_GRAY);
  rc = tj3DecodeYUV8(handle, src, geo->align, pixels, geo->width, 0,
                     geo->height, TJPF_RGB);
  printf("%s decodeyuv8 rc=%d rgbdigest=%016llx rgbgray=%s\n", geo->label, rc,
         row_digest(pixels, geo->width * 3, geo->height, geo->width * 3),
         rgb_is_gray(pixels, geo->width, geo->height));
  free(src);
  free(pixels);
  tj3Destroy(handle);
}

static void case_decode_yuv_planes8(const geometry *geo,
                                    size_t chroma_capacity)
{
  tjhandle handle = tj3Init(TJINIT_DECOMPRESS);
  unsigned char *luma = make_tight_gray_plane(geo->width, geo->height);
  unsigned char *chroma[2];
  const unsigned char *planes[3];
  unsigned char *pixels =
    make_filled((size_t)geo->width * geo->height * 3, GUARD_BYTE);
  int rc, i;

  if (!handle) { fprintf(stderr, "tj3Init\n"); exit(1); }
  for (i = 0; i < 2; i++)
    chroma[i] = make_filled(chroma_capacity, SENTINEL_BYTE);
  planes[0] = luma;  planes[1] = chroma[0];  planes[2] = chroma[1];
  tj3Set(handle, TJPARAM_SUBSAMP, TJSAMP_GRAY);
  rc = tj3DecodeYUVPlanes8(handle, planes, NULL, pixels, geo->width, 0,
                           geo->height, TJPF_RGB);
  printf("%s decodeplanes8 rc=%d rgbdigest=%016llx rgbgray=%s\n", geo->label,
         rc, row_digest(pixels, geo->width * 3, geo->height, geo->width * 3),
         rgb_is_gray(pixels, geo->width, geo->height));
  free(luma);
  for (i = 0; i < 2; i++) free(chroma[i]);
  free(pixels);
  tj3Destroy(handle);
}

/* --- YUV -> JPEG -------------------------------------------------------- */

/* Report a compressed result by what a caller can check about it: the header
 * a decompressor reads back, and whether the image survived the round trip.
 * Not by its bytes — two encoders agreeing on JPEG output is a stronger claim
 * than this item is about, and one this port does not owe on the YUV path. */
static void report_compressed(const geometry *geo, const char *case_name,
                              int rc, const unsigned char *jpeg,
                              size_t jpeg_size)
{
  tjhandle handle;
  unsigned char *decoded;
  char scratch[64];

  if (rc != 0 || jpeg == NULL || jpeg_size == 0) {
    printf("%s %s rc=%d jw=0 jh=0 jsubsamp=-2 roundtrip=none\n", geo->label,
           case_name, rc);
    return;
  }
  handle = tj3Init(TJINIT_DECOMPRESS);
  if (!handle) { fprintf(stderr, "tj3Init\n"); exit(1); }
  if (tj3DecompressHeader(handle, jpeg, jpeg_size) != 0) {
    printf("%s %s rc=%d jw=0 jh=0 jsubsamp=-3 roundtrip=none\n", geo->label,
           case_name, rc);
    tj3Destroy(handle);
    return;
  }
  decoded = make_filled((size_t)geo->width * geo->height, GUARD_BYTE);
  if (tj3Decompress8(handle, jpeg, jpeg_size, decoded, 0, TJPF_GRAY) != 0) {
    printf("%s %s rc=%d jw=%d jh=%d jsubsamp=%d roundtrip=undecodable\n",
           geo->label, case_name, rc, tj3Get(handle, TJPARAM_JPEGWIDTH),
           tj3Get(handle, TJPARAM_JPEGHEIGHT), tj3Get(handle, TJPARAM_SUBSAMP));
    free(decoded);
    tj3Destroy(handle);
    return;
  }
  printf("%s %s rc=%d jw=%d jh=%d jsubsamp=%d roundtrip=%s\n", geo->label,
         case_name, rc, tj3Get(handle, TJPARAM_JPEGWIDTH),
         tj3Get(handle, TJPARAM_JPEGHEIGHT), tj3Get(handle, TJPARAM_SUBSAMP),
         roundtrip_state(decoded, geo->width, geo->height, scratch));
  free(decoded);
  tj3Destroy(handle);
}

static void case_compress_from_yuv8(const geometry *geo, size_t capacity)
{
  tjhandle handle = tj3Init(TJINIT_COMPRESS);
  unsigned char *src =
    make_packed_gray_source(geo->width, geo->height, geo->align, capacity);
  unsigned char *jpeg = NULL;
  size_t jpeg_size = 0;
  int rc;

  if (!handle) { fprintf(stderr, "tj3Init\n"); exit(1); }
  tj3Set(handle, TJPARAM_QUALITY, QUALITY);
  tj3Set(handle, TJPARAM_SUBSAMP, TJSAMP_GRAY);
  rc = tj3CompressFromYUV8(handle, src, geo->width, geo->align, geo->height,
                           &jpeg, &jpeg_size);
  report_compressed(geo, "fromyuv8", rc, jpeg, jpeg_size);
  free(src);
  tj3Free(jpeg);
  tj3Destroy(handle);
}

static void case_compress_from_yuv_planes8(const geometry *geo,
                                           size_t chroma_capacity)
{
  tjhandle handle = tj3Init(TJINIT_COMPRESS);
  unsigned char *luma = make_tight_gray_plane(geo->width, geo->height);
  unsigned char *chroma[2];
  const unsigned char *planes[3];
  unsigned char *jpeg = NULL;
  size_t jpeg_size = 0;
  int rc, i;

  if (!handle) { fprintf(stderr, "tj3Init\n"); exit(1); }
  for (i = 0; i < 2; i++)
    chroma[i] = make_filled(chroma_capacity, SENTINEL_BYTE);
  planes[0] = luma;  planes[1] = chroma[0];  planes[2] = chroma[1];
  tj3Set(handle, TJPARAM_QUALITY, QUALITY);
  tj3Set(handle, TJPARAM_SUBSAMP, TJSAMP_GRAY);
  rc = tj3CompressFromYUVPlanes8(handle, planes, geo->width, NULL, geo->height,
                                 &jpeg, &jpeg_size);
  report_compressed(geo, "fromyuvplanes8", rc, jpeg, jpeg_size);
  free(luma);
  for (i = 0; i < 2; i++) free(chroma[i]);
  tj3Free(jpeg);
  tj3Destroy(handle);
}

/* --- JPEG -> YUV -------------------------------------------------------- */

static void case_decompress_to_yuv8(const geometry *geo,
                                    const unsigned char *jpeg,
                                    size_t jpeg_size, size_t gray_size,
                                    size_t capacity)
{
  tjhandle handle = tj3Init(TJINIT_DECOMPRESS);
  unsigned char *dst = make_filled(capacity, GUARD_BYTE);
  int stride = pad_up(geo->width, geo->align);
  int rc;

  if (!handle) { fprintf(stderr, "tj3Init\n"); exit(1); }
  rc = tj3DecompressToYUV8(handle, jpeg, jpeg_size, dst, geo->align);
  printf("%s toyuv8 rc=%d ydigest=%016llx guard=%s\n", geo->label, rc,
         row_digest(dst, geo->width, geo->height, stride),
         guard_state(dst, gray_size, capacity));
  free(dst);
  tj3Destroy(handle);
}

static void case_decompress_to_yuv_planes8(const geometry *geo,
                                           const unsigned char *jpeg,
                                           size_t jpeg_size,
                                           size_t chroma_capacity)
{
  tjhandle handle = tj3Init(TJINIT_DECOMPRESS);
  unsigned char *luma =
    make_filled((size_t)geo->width * geo->height, GUARD_BYTE);
  unsigned char *chroma[2];
  unsigned char *planes[3];
  int rc, i;

  if (!handle) { fprintf(stderr, "tj3Init\n"); exit(1); }
  for (i = 0; i < 2; i++) chroma[i] = make_filled(chroma_capacity, GUARD_BYTE);
  planes[0] = luma;  planes[1] = chroma[0];  planes[2] = chroma[1];
  rc = tj3DecompressToYUVPlanes8(handle, jpeg, jpeg_size, planes, NULL);
  printf("%s toyuvplanes8 rc=%d ydigest=%016llx guard1=%s guard2=%s\n",
         geo->label, rc,
         row_digest(luma, geo->width, geo->height, geo->width),
         guard_state(chroma[0], 0, chroma_capacity),
         guard_state(chroma[1], 0, chroma_capacity));
  free(luma);
  for (i = 0; i < 2; i++) free(chroma[i]);
  tj3Destroy(handle);
}

/* --- Three-plane controls ----------------------------------------------
 *
 * Not about TJSAMP_GRAY: these pin the *other* subsamplings through the same
 * entry points, because the port stopped inferring its plane count from the
 * packed buffer's length and now splits the buffer itself before calling the
 * planar workers. That reroute touches every subsampling, and the GRAY cases
 * above cannot see it. `TJPF_GRAY` is included because it is the one output
 * format whose internal branch changed — a three-plane source decoded to
 * grayscale used to take the port's own single-plane shortcut. */

static void case_decode_yuv8_three_plane(const geometry *geo, int subsamp,
                                         int pixel_format,
                                         const char *case_name, int bpp)
{
  tjhandle handle = tj3Init(TJINIT_DECOMPRESS);
  size_t source_size =
    tj3YUVBufSize(geo->width, geo->align, geo->height, subsamp);
  unsigned char *src = make_packed_ycbcr_source(geo->width, geo->height,
                                                geo->align, subsamp,
                                                source_size + SLACK);
  unsigned char *pixels =
    make_filled((size_t)geo->width * geo->height * bpp, GUARD_BYTE);
  int rc;

  if (!handle) { fprintf(stderr, "tj3Init\n"); exit(1); }
  tj3Set(handle, TJPARAM_SUBSAMP, subsamp);
  rc = tj3DecodeYUV8(handle, src, geo->align, pixels, geo->width, 0,
                     geo->height, pixel_format);
  printf("%s %s rc=%d digest=%016llx\n", geo->label, case_name, rc,
         row_digest(pixels, geo->width * bpp, geo->height, geo->width * bpp));
  free(src);
  free(pixels);
  tj3Destroy(handle);
}

int main(int argc, char **argv)
{
  int i;

  if (argc != 2) {
    fprintf(stderr, "usage: %s <workdir>\n", argv[0]);
    return 2;
  }

  for (i = 0; i < GEOMETRY_COUNT; i++) {
    const geometry *geo = &GEOMETRIES[i];
    /* The contract length: what `tj3YUVBufSize` promises a GRAY caller. */
    size_t gray_size =
      tj3YUVBufSize(geo->width, geo->align, geo->height, TJSAMP_GRAY);
    /* The three-plane length a 4:4:4 caller would need, which is exactly what
     * a worker that mapped GRAY onto 4:4:4 computes — plus headroom, so that
     * such a worker writes into memory this program owns. */
    size_t capacity =
      tj3YUVBufSize(geo->width, geo->align, geo->height, TJSAMP_444) + SLACK;
    size_t chroma_capacity = (size_t)geo->width * geo->height + SLACK;
    unsigned char *jpeg;
    size_t jpeg_size;

    printf("%s bufsize=%zu capacity=%zu\n", geo->label, gray_size, capacity);
    case_encode_yuv8(geo, gray_size, capacity);
    case_encode_yuv_planes8(geo, chroma_capacity);
    case_decode_yuv8(geo, capacity);
    case_decode_yuv_planes8(geo, chroma_capacity);
    case_compress_from_yuv8(geo, capacity);
    case_compress_from_yuv_planes8(geo, chroma_capacity);

    jpeg = write_gray_fixture(argv[1], geo, &jpeg_size);
    case_decompress_to_yuv8(geo, jpeg, jpeg_size, gray_size, capacity);
    case_decompress_to_yuv_planes8(geo, jpeg, jpeg_size, chroma_capacity);
    tj3Free(jpeg);

    /* Three-plane controls for the packed reroute. */
    case_decode_yuv8_three_plane(geo, TJSAMP_420, TJPF_RGB, "decodeyuv8_420rgb",
                                 3);
    case_decode_yuv8_three_plane(geo, TJSAMP_420, TJPF_GRAY,
                                 "decodeyuv8_420gray", 1);
  }
  return 0;
}

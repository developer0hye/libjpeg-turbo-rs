/*
 * C oracle for the TurboJPEG YUV plane-dimension component bound (P4-126).
 *
 * Upstream rejects an out-of-range component index in tj3YUVPlaneWidth /
 * tj3YUVPlaneHeight (references/libjpeg-turbo/src/turbojpeg.c:1115, lines
 * 1123-1125):
 *
 *     nc = (subsamp == TJSAMP_GRAY ? 1 : 3);
 *     if (componentID < 0 || componentID >= nc)
 *       THROWG("Invalid argument", 0);
 *
 * The `nc` term is the part that cannot be expressed in this port's
 * root-crate helpers: their `Subsampling` type has no grayscale variant, so
 * `TJSAMP_GRAY` arrives indistinguishable from `TJSAMP_444`. The legacy
 * tjPlaneWidth / tjPlaneHeight inherit the bound upstream only because they
 * delegate (turbojpeg.c):
 *
 *     int retval = tj3YUVPlaneWidth(componentID, width, subsamp);
 *     return (retval == 0) ? -1 : retval;
 *
 * Prints one line per (subsamp, componentID) pair so the Rust side can compare
 * the whole matrix rather than the handful of cases someone thought to name:
 *
 *     <subsamp> <componentID> <tjPlaneWidth> <tjPlaneHeight> <tj3W> <tj3H>
 *
 * Usage: yuv_plane_index_c_oracle <width> <height>
 */

#include <stdio.h>
#include <stdlib.h>
#include <turbojpeg.h>

/* Widest documented index probe: one below the first valid index through one
 * past the largest plane count any TJSAMP_* uses. */
#define CID_MIN (-1)
#define CID_MAX 4

int main(int argc, char **argv)
{
  int width, height, subsamp, cid;

  if (argc != 3) {
    fprintf(stderr, "usage: %s <width> <height>\n", argv[0]);
    return 2;
  }
  width = atoi(argv[1]);
  height = atoi(argv[2]);

  for (subsamp = 0; subsamp < TJ_NUMSAMP; subsamp++) {
    for (cid = CID_MIN; cid <= CID_MAX; cid++) {
      printf("%d %d %d %d %d %d\n", subsamp, cid,
             tjPlaneWidth(cid, width, subsamp),
             tjPlaneHeight(cid, height, subsamp),
             tj3YUVPlaneWidth(cid, width, subsamp),
             tj3YUVPlaneHeight(cid, height, subsamp));
    }
  }
  return 0;
}

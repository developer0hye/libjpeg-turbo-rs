/* Minimal jconfig.h for stock_djpeg_cjpeg link-against-our-shim build.
 *
 * This file is generated in-tree so the stock djpeg/cjpeg/jpegtran sources
 * compile without an upstream cmake run. The macros here mirror what a
 * 64-bit aarch64 / x86_64 cmake configuration would emit; any differences
 * would only affect host-side sizeof() / feature detection — NOT the
 * ABI-visible symbols that our shim would have to supply.
 */
#ifndef JCONFIG_H
#define JCONFIG_H

#define JPEG_LIB_VERSION 80
#define LIBJPEG_TURBO_VERSION 3.1.2
#define LIBJPEG_TURBO_VERSION_NUMBER 3001002
#define BITS_IN_JSAMPLE 8
#define HAVE_STDLIB_H 1
#define HAVE_STDDEF_H 1
#define HAVE_UNSIGNED_CHAR 1
#define HAVE_UNSIGNED_SHORT 1
#define MEM_SRCDST_SUPPORTED 1
#define C_ARITH_CODING_SUPPORTED 1
#define D_ARITH_CODING_SUPPORTED 1
#define WITH_SIMD 0

#endif /* JCONFIG_H */

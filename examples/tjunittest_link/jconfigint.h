/*
 * Minimal jconfigint.h stub for building `tjunittest` against the Rust
 * TurboJPEG shim.
 *
 * The real libjpeg-turbo generates this file from `jconfigint.h.in` via
 * CMake. `tjunittest.c` only needs a handful of defines to compile — we
 * set them to sane defaults for a modern 64-bit Unix toolchain. If this
 * file is wrong the compiler will tell us; we are not building any
 * in-tree C source beyond tjunittest.c + tjutil.c + md5 helpers.
 */

#ifndef JCONFIGINT_H
#define JCONFIGINT_H

#ifndef BUILD
#define BUILD "libjpeg-turbo-rs"
#endif

#ifndef HIDDEN
#define HIDDEN __attribute__((visibility("hidden")))
#endif

#ifndef INLINE
#define INLINE inline
#endif

#ifndef THREAD_LOCAL
#if defined(__GNUC__) || defined(__clang__)
#define THREAD_LOCAL __thread
#else
#define THREAD_LOCAL
#endif
#endif

#ifndef PACKAGE_NAME
#define PACKAGE_NAME "libjpeg-turbo-rs"
#endif

#ifndef VERSION
#define VERSION "0.1.0"
#endif

#ifndef SIZEOF_SIZE_T
#if defined(__LP64__) || defined(_WIN64)
#define SIZEOF_SIZE_T 8
#else
#define SIZEOF_SIZE_T 4
#endif
#endif

/* BITS_IN_JSAMPLE is only needed for cjpeg/djpeg-style integration — keep
 * the default 8 for tjunittest. */
#ifndef BITS_IN_JSAMPLE
#define BITS_IN_JSAMPLE 8
#endif

#ifndef FALLTHROUGH
#if defined(__has_attribute)
#if __has_attribute(fallthrough)
#define FALLTHROUGH __attribute__((fallthrough));
#else
#define FALLTHROUGH
#endif
#else
#define FALLTHROUGH
#endif
#endif

#endif /* !JCONFIGINT_H */

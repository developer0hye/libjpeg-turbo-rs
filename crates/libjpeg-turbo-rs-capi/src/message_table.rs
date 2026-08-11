//! Upstream's standard JPEG message table, in `msg_code` order.
//!
//! `jpeg_std_error` left `jpeg_message_table` null until P4-146 (#518), so
//! `default_format_message` always took its "bogus message code" fallback —
//! every classic error rendered as that string for a C consumer calling
//! `format_message` or `output_message`, whatever `msg_code` said.
//!
//! **Generated from the pinned upstream header, not transcribed.** The entries
//! come from a C program that `#include`s `jerror.h` twice — once for the enum,
//! once with `JMESSAGE` redefined to build the table — compiled at
//! `JPEG_LIB_VERSION 80`. That matters more than it looks: `jerror.h` has 134
//! `JMESSAGE` lines but only 129 entries at v8, because several are
//! version-conditional and a few appear twice under opposite `#if
//! JPEG_LIB_VERSION` guards. A line-order parse counts both and misaligns
//! everything after the first divergence — and for a code-indexed table, a
//! misaligned entry is a *wrong* message, not a missing one. That is why
//! `JERR_IMAGE_TOO_BIG` is 42 rather than the 45 a naive parse suggests.
//!
//! `capi_classic_error_codes.rs` re-runs that same C probe over the whole
//! table, so a submodule bump that reorders or re-gates a message fails the
//! build instead of silently shifting every later string.
//!
//! **Two entries are not probe output.** `JMSG_COPYRIGHT` (75) and
//! `JMSG_VERSION` (76) resolve from `jversion.h`, which the submodule ships
//! only as `jversion.h.in` — so the probe defines placeholder macros for them,
//! and the first version of this table shipped `"unused by this probe"` as the
//! live text for both. They now carry the configured values:
//! `JVERSION` from `jversion.h.in`'s `JPEG_LIB_VERSION >= 80` arm, and
//! `JCOPYRIGHT` with `@COPYRIGHT_YEAR@` resolved from `CMakeLists.txt:18`. The
//! whole-table test re-derives both from those same two files and compares
//! them exactly, so a submodule bump that changes either fails here rather
//! than leaving a stale string behind a "not the placeholder" check.

/// Number of entries at `JPEG_LIB_VERSION 80`.
pub(crate) const JPEG_MESSAGE_COUNT: usize = 129;

/// The messages themselves, NUL-terminated so each doubles as a C string.
static JPEG_MESSAGES: [&[u8]; JPEG_MESSAGE_COUNT] = [
    b"Bogus message code %d\0", // 0 JMSG_NOMESSAGE
    b"ALIGN_TYPE is wrong, please fix\0", // 1 JERR_BAD_ALIGN_TYPE
    b"MAX_ALLOC_CHUNK is wrong, please fix\0", // 2 JERR_BAD_ALLOC_CHUNK
    b"Bogus buffer control mode\0", // 3 JERR_BAD_BUFFER_MODE
    b"Invalid component ID %d in SOS\0", // 4 JERR_BAD_COMPONENT_ID
    b"Invalid crop request\0", // 5 JERR_BAD_CROP_SPEC
    b"DCT coefficient (lossy) or spatial difference (lossless) out of range\0", // 6 JERR_BAD_DCT_COEF
    b"IDCT output block size %d not supported\0", // 7 JERR_BAD_DCTSIZE
    b"Component index %d: mismatching sampling ratio %d:%d, %d:%d, %c\0", // 8 JERR_BAD_DROP_SAMPLING
    b"Bogus Huffman table definition\0", // 9 JERR_BAD_HUFF_TABLE
    b"Bogus input colorspace\0", // 10 JERR_BAD_IN_COLORSPACE
    b"Bogus JPEG colorspace\0", // 11 JERR_BAD_J_COLORSPACE
    b"Bogus marker length\0", // 12 JERR_BAD_LENGTH
    b"Wrong JPEG library version: library is %d, caller expects %d\0", // 13 JERR_BAD_LIB_VERSION
    b"Sampling factors too large for interleaved scan\0", // 14 JERR_BAD_MCU_SIZE
    b"Invalid memory pool code %d\0", // 15 JERR_BAD_POOL_ID
    b"Unsupported JPEG data precision %d\0", // 16 JERR_BAD_PRECISION
    b"Invalid progressive/lossless parameters Ss=%d Se=%d Ah=%d Al=%d\0", // 17 JERR_BAD_PROGRESSION
    b"Invalid progressive/lossless parameters at scan script entry %d\0", // 18 JERR_BAD_PROG_SCRIPT
    b"Bogus sampling factors\0", // 19 JERR_BAD_SAMPLING
    b"Invalid scan script at entry %d\0", // 20 JERR_BAD_SCAN_SCRIPT
    b"Improper call to JPEG library in state %d\0", // 21 JERR_BAD_STATE
    b"JPEG parameter struct mismatch: library thinks size is %u, caller expects %u\0", // 22 JERR_BAD_STRUCT_SIZE
    b"Bogus virtual array access\0", // 23 JERR_BAD_VIRTUAL_ACCESS
    b"Buffer passed to JPEG library is too small\0", // 24 JERR_BUFFER_SIZE
    b"Suspension not allowed here\0", // 25 JERR_CANT_SUSPEND
    b"CCIR601 sampling not implemented yet\0", // 26 JERR_CCIR601_NOTIMPL
    b"Too many color components: %d, max %d\0", // 27 JERR_COMPONENT_COUNT
    b"Unsupported color conversion request\0", // 28 JERR_CONVERSION_NOTIMPL
    b"Bogus DAC index %d\0", // 29 JERR_DAC_INDEX
    b"Bogus DAC value 0x%x\0", // 30 JERR_DAC_VALUE
    b"Bogus DHT index %d\0", // 31 JERR_DHT_INDEX
    b"Bogus DQT index %d\0", // 32 JERR_DQT_INDEX
    b"Empty JPEG image (DNL not supported)\0", // 33 JERR_EMPTY_IMAGE
    b"Read from EMS failed\0", // 34 JERR_EMS_READ
    b"Write to EMS failed\0", // 35 JERR_EMS_WRITE
    b"Didn't expect more than one scan\0", // 36 JERR_EOI_EXPECTED
    b"Input file read error\0", // 37 JERR_FILE_READ
    b"Output file write error --- out of disk space?\0", // 38 JERR_FILE_WRITE
    b"Fractional sampling not implemented yet\0", // 39 JERR_FRACT_SAMPLE_NOTIMPL
    b"Huffman code size table overflow\0", // 40 JERR_HUFF_CLEN_OVERFLOW
    b"Missing Huffman code table entry\0", // 41 JERR_HUFF_MISSING_CODE
    b"Maximum supported image dimension is %u pixels\0", // 42 JERR_IMAGE_TOO_BIG
    b"Empty input file\0", // 43 JERR_INPUT_EMPTY
    b"Premature end of input file\0", // 44 JERR_INPUT_EOF
    b"Cannot transcode due to multiple use of quantization table %d\0", // 45 JERR_MISMATCHED_QUANT_TABLE
    b"Scan script does not transmit all data\0", // 46 JERR_MISSING_DATA
    b"Invalid color quantization mode change\0", // 47 JERR_MODE_CHANGE
    b"Requested features are incompatible\0", // 48 JERR_NOTIMPL
    b"Requested feature was omitted at compile time\0", // 49 JERR_NOT_COMPILED
    b"Arithmetic table 0x%02x was not defined\0", // 50 JERR_NO_ARITH_TABLE
    b"Memory limit exceeded\0", // 51 JERR_NO_BACKING_STORE
    b"Huffman table 0x%02x was not defined\0", // 52 JERR_NO_HUFF_TABLE
    b"JPEG datastream contains no image\0", // 53 JERR_NO_IMAGE
    b"Quantization table 0x%02x was not defined\0", // 54 JERR_NO_QUANT_TABLE
    b"Not a JPEG file: starts with 0x%02x 0x%02x\0", // 55 JERR_NO_SOI
    b"Insufficient memory (case %d)\0", // 56 JERR_OUT_OF_MEMORY
    b"Cannot quantize more than %d color components\0", // 57 JERR_QUANT_COMPONENTS
    b"Cannot quantize to fewer than %d colors\0", // 58 JERR_QUANT_FEW_COLORS
    b"Cannot quantize to more than %d colors\0", // 59 JERR_QUANT_MANY_COLORS
    b"Invalid JPEG file structure: two SOF markers\0", // 60 JERR_SOF_DUPLICATE
    b"Invalid JPEG file structure: missing SOS marker\0", // 61 JERR_SOF_NO_SOS
    b"Unsupported JPEG process: SOF type 0x%02x\0", // 62 JERR_SOF_UNSUPPORTED
    b"Invalid JPEG file structure: two SOI markers\0", // 63 JERR_SOI_DUPLICATE
    b"Invalid JPEG file structure: SOS before SOF\0", // 64 JERR_SOS_NO_SOF
    b"Failed to create temporary file %s\0", // 65 JERR_TFILE_CREATE
    b"Read failed on temporary file\0", // 66 JERR_TFILE_READ
    b"Seek failed on temporary file\0", // 67 JERR_TFILE_SEEK
    b"Write failed on temporary file --- out of disk space?\0", // 68 JERR_TFILE_WRITE
    b"Application transferred too few scanlines\0", // 69 JERR_TOO_LITTLE_DATA
    b"Unsupported marker type 0x%02x\0", // 70 JERR_UNKNOWN_MARKER
    b"Virtual array controller messed up\0", // 71 JERR_VIRTUAL_BUG
    b"Image too wide for this implementation\0", // 72 JERR_WIDTH_OVERFLOW
    b"Read from XMS failed\0", // 73 JERR_XMS_READ
    b"Write to XMS failed\0", // 74 JERR_XMS_WRITE
    b"Copyright (C) 1991-2026 The libjpeg-turbo Project and many others\0", // 75 JMSG_COPYRIGHT
    b"8d  15-Jan-2012\0", // 76 JMSG_VERSION
    b"Caution: quantization tables are too coarse for baseline JPEG\0", // 77 JTRC_16BIT_TABLES
    b"Adobe APP14 marker: version %d, flags 0x%04x 0x%04x, transform %d\0", // 78 JTRC_ADOBE
    b"Unknown APP0 marker (not JFIF), length %u\0", // 79 JTRC_APP0
    b"Unknown APP14 marker (not Adobe), length %u\0", // 80 JTRC_APP14
    b"Define Arithmetic Table 0x%02x: 0x%02x\0", // 81 JTRC_DAC
    b"Define Huffman Table 0x%02x\0", // 82 JTRC_DHT
    b"Define Quantization Table %d  precision %d\0", // 83 JTRC_DQT
    b"Define Restart Interval %u\0", // 84 JTRC_DRI
    b"Freed EMS handle %u\0", // 85 JTRC_EMS_CLOSE
    b"Obtained EMS handle %u\0", // 86 JTRC_EMS_OPEN
    b"End Of Image\0", // 87 JTRC_EOI
    b"        %3d %3d %3d %3d %3d %3d %3d %3d\0", // 88 JTRC_HUFFBITS
    b"JFIF APP0 marker: version %d.%02d, density %dx%d  %d\0", // 89 JTRC_JFIF
    b"Warning: thumbnail image size does not match data length %u\0", // 90 JTRC_JFIF_BADTHUMBNAILSIZE
    b"JFIF extension marker: type 0x%02x, length %u\0", // 91 JTRC_JFIF_EXTENSION
    b"    with %d x %d thumbnail image\0", // 92 JTRC_JFIF_THUMBNAIL
    b"Miscellaneous marker 0x%02x, length %u\0", // 93 JTRC_MISC_MARKER
    b"Unexpected marker 0x%02x\0", // 94 JTRC_PARMLESS_MARKER
    b"        %4u %4u %4u %4u %4u %4u %4u %4u\0", // 95 JTRC_QUANTVALS
    b"Quantizing to %d = %d*%d*%d colors\0", // 96 JTRC_QUANT_3_NCOLORS
    b"Quantizing to %d colors\0", // 97 JTRC_QUANT_NCOLORS
    b"Selected %d colors for quantization\0", // 98 JTRC_QUANT_SELECTED
    b"At marker 0x%02x, recovery action %d\0", // 99 JTRC_RECOVERY_ACTION
    b"RST%d\0", // 100 JTRC_RST
    b"Smoothing not supported with nonstandard sampling ratios\0", // 101 JTRC_SMOOTH_NOTIMPL
    b"Start Of Frame 0x%02x: width=%u, height=%u, components=%d\0", // 102 JTRC_SOF
    b"    Component %d: %dhx%dv q=%d\0", // 103 JTRC_SOF_COMPONENT
    b"Start of Image\0", // 104 JTRC_SOI
    b"Start Of Scan: %d components\0", // 105 JTRC_SOS
    b"    Component %d: dc=%d ac=%d\0", // 106 JTRC_SOS_COMPONENT
    b"  Ss=%d, Se=%d, Ah=%d, Al=%d\0", // 107 JTRC_SOS_PARAMS
    b"Closed temporary file %s\0", // 108 JTRC_TFILE_CLOSE
    b"Opened temporary file %s\0", // 109 JTRC_TFILE_OPEN
    b"JFIF extension marker: JPEG-compressed thumbnail image, length %u\0", // 110 JTRC_THUMB_JPEG
    b"JFIF extension marker: palette thumbnail image, length %u\0", // 111 JTRC_THUMB_PALETTE
    b"JFIF extension marker: RGB thumbnail image, length %u\0", // 112 JTRC_THUMB_RGB
    b"Unrecognized component IDs %d %d %d, assuming YCbCr (lossy) or RGB (lossless)\0", // 113 JTRC_UNKNOWN_IDS
    b"Freed XMS handle %u\0", // 114 JTRC_XMS_CLOSE
    b"Obtained XMS handle %u\0", // 115 JTRC_XMS_OPEN
    b"Unknown Adobe color transform code %d\0", // 116 JWRN_ADOBE_XFORM
    b"Corrupt JPEG data: bad arithmetic code\0", // 117 JWRN_ARITH_BAD_CODE
    b"Inconsistent progression sequence for component %d coefficient %d\0", // 118 JWRN_BOGUS_PROGRESSION
    b"Corrupt JPEG data: %u extraneous bytes before marker 0x%02x\0", // 119 JWRN_EXTRANEOUS_DATA
    b"Corrupt JPEG data: premature end of data segment\0", // 120 JWRN_HIT_MARKER
    b"Corrupt JPEG data: bad Huffman code\0", // 121 JWRN_HUFF_BAD_CODE
    b"Warning: unknown JFIF revision number %d.%02d\0", // 122 JWRN_JFIF_MAJOR
    b"Premature end of JPEG file\0", // 123 JWRN_JPEG_EOF
    b"Corrupt JPEG data: found marker 0x%02x instead of RST%d\0", // 124 JWRN_MUST_RESYNC
    b"Invalid SOS parameters for sequential JPEG\0", // 125 JWRN_NOT_SEQUENTIAL
    b"Application transferred too many scanlines\0", // 126 JWRN_TOO_MUCH_DATA
    b"Corrupt JPEG data: bad ICC marker\0", // 127 JWRN_BOGUS_ICC
    b"Invalid restart interval %d; must be an integer multiple of the number of MCUs in an MCU row (%d)\0", // 128 JERR_BAD_RESTART
];

/// `[*const u8; N]` is not `Sync` on its own, so the table is wrapped.
///
/// The pointers address `'static` byte strings in read-only memory and are
/// never written through — which is exactly the guarantee C's
/// `const char * const []` makes, and what `jpeg_error_mgr` expects.
struct MessageTable([*const u8; JPEG_MESSAGE_COUNT]);

// SAFETY: every pointer targets a `'static` NUL-terminated byte string that is
// never mutated, so sharing the table across threads cannot race.
unsafe impl Sync for MessageTable {}

const fn build_table() -> MessageTable {
    let mut table: [*const u8; JPEG_MESSAGE_COUNT] = [core::ptr::null(); JPEG_MESSAGE_COUNT];
    let mut i: usize = 0;
    while i < JPEG_MESSAGE_COUNT {
        table[i] = JPEG_MESSAGES[i].as_ptr();
        i += 1;
    }
    MessageTable(table)
}

static JPEG_MESSAGE_TABLE: MessageTable = build_table();

/// Pointer to install in `jpeg_error_mgr::jpeg_message_table`.
///
/// `*const u8` rather than `*const c_char` because that is what this crate's
/// ABI mirror declares; the two are layout-identical and C sees `char *`.
pub(crate) fn message_table_ptr() -> *const *const u8 {
    JPEG_MESSAGE_TABLE.0.as_ptr()
}

/// `last_jpeg_message`: the highest valid index, as upstream's
/// `jpeg_std_error` sets it (`JMSG_LASTMSGCODE - 1`).
pub(crate) fn last_jpeg_message() -> std::ffi::c_int {
    (JPEG_MESSAGE_COUNT - 1) as std::ffi::c_int
}

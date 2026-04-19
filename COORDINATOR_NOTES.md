# Coordinator Notes — SA1 (Decompress Struct ABI Mirror)

## Scope

Worker SA1 is responsible for `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs`
— specifically the decompress-side sections (`JpegDecompressPublic`,
`jpeg_CreateDecompress`, `jpeg_read_header`, `jpeg_start_decompress`,
and friends). The encode-side (`JpegCompressPublic`, `JpegCompressPublic::priv_ptr`,
`jpeg_CreateCompress`, and all `JpegComponentInfoCompress` definitions)
remain owned by worker SA2.

## Shared-section edits

- **None.** `JpegErrorMgr`, `JpegSourceMgr`, `JpegDestinationMgr`, and
  `JpegCompressPublic` were not modified. The only new decode-side
  sub-structs (`JQuantTblPublic`, `JHuffTblPublic`,
  `JpegComponentInfoPublic`, `JpegMarkerStructPublic`) are decode-only
  types referenced from `JpegDecompressPublic` pointer fields.

## Public ABI change

`JpegDecompressPublic` is now a **byte-exact mirror** of libjpeg's
`struct jpeg_decompress_struct` (`JPEG_LIB_VERSION = 80`). The previous
21-field subset is gone; the new layout is ~80 fields and ~592 bytes on
LP64 targets.

The old layout exposed a trailing `priv_ptr: *mut c_void` for Rust-side
state. That field has been removed because writing to it overflowed
real libjpeg callers' `sizeof(struct jpeg_decompress_struct)`
allocation. Private state now lives in a thread-local side table keyed
by the `cinfo` pointer (`DECOMPRESS_PRIVATE_STATE`); see
`decompress_private_raw` / `decompress_private_insert` /
`decompress_private_remove` in `jpeglib.rs`.

## Follow-up suggestions (non-blocking)

- Surface `JpegMetadata::is_arithmetic`, `saw_adobe_marker`,
  `restart_interval`, and `adobe_transform` through the public Rust
  decoder API so `jpeg_read_header` can populate `cinfo.arith_code`,
  `cinfo.saw_Adobe_marker`, `cinfo.restart_interval`, and
  `cinfo.Adobe_transform` faithfully (currently approximated or left at
  default).
- Quant- and Huffman-table pointer arrays (`cinfo.quant_tbl_ptrs[]`,
  `cinfo.dc_huff_tbl_ptrs[]`, `cinfo.ac_huff_tbl_ptrs[]`) are initialized
  to NULL. Real libjpeg populates them with pool-allocated tables during
  `jpeg_read_header`. Callers that iterate them expecting live pointers
  (e.g., `jpegtran -copy all`) will need a follow-up that allocates
  backing `JQuantTblPublic`/`JHuffTblPublic` and wires them in.

## Dev-deps

No workspace `Cargo.toml` edits required; offset assertions use stable
`std::mem::offset_of!` (MSRV 1.77+, project MSRV 1.87).

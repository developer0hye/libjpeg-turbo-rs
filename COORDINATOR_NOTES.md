# Coordinator notes (SA1/SA2 shared worktree file)

## SA2 activity (agent-a833b9f1 worktree, worktree-agent-a833b9f1 branch)

### Mission summary
Make `JpegCompressPublic` a byte-exact ABI mirror of
`struct jpeg_compress_struct` as declared in
`references/libjpeg-turbo/src/jpeglib.h` for `JPEG_LIB_VERSION >= 80`.
The target ABI is exactly what stock `cjpeg` gets when compiled with the
`jconfig.h` produced by `examples/stock_djpeg_cjpeg/build.sh` (which
defines `JPEG_LIB_VERSION 80`). Total C struct size on 64-bit LP64: 584
bytes.

### Shared-struct coordination
- `JpegErrorMgr`, `JpegSourceMgr`, `JpegDestinationMgr`, `JpegCompressPublic`,
  `JpegDecompressPublic`, `JpegComponentInfoCompress` are declared in
  `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs`. SA2 owns only the
  compress-side types; SA1 owns decompress-side.
- SA2 introduces the following new compress-adjacent types:
  `JQuantTblPublic`, `JHuffTblPublic`, `JpegScanInfoPublic`,
  `JpegSavedMarker`, `JpegMemoryMgrPublic`, `JpegProgressMgrPublic`.
  If SA1 also needs these, SA2 defers to SA1's declaration; we'll keep
  them at the top of `jpeglib.rs` in the shared prelude region. Names
  chosen to avoid conflicting with any existing SA1 names.
- `JpegComponentInfoCompress` is replaced by `JpegComponentInfoPublic`
  (full libjpeg `jpeg_component_info` ABI shape). Old name kept as a
  type alias to avoid churn in internal compress helpers.

### Private-state storage
Because stock cjpeg allocates `struct jpeg_compress_struct cinfo;` on
the stack (exactly 584 bytes), we cannot append a `priv_ptr` tail
field past the real libjpeg layout — writing beyond offset 584 would
corrupt caller memory. Instead, SA2 reuses the `master:
*mut jpeg_comp_master` slot (field near tail of the libjpeg struct)
as the Rust-side private-state pointer. Semantically this is safe:
`master` is documented as opaque libjpeg internal state and stock
callers never dereference it. The old `priv_ptr` field is removed.

### Subtasks
- [ ] SA2-1: full struct ABI + sub-struct declarations
- [ ] SA2-2: compile-time offset assertions for 20+ key fields
- [ ] SA2-3: `jpeg_CreateCompress` / `jpeg_set_defaults` /
      `jpeg_set_quality` / `jpeg_set_colorspace` populate the full
      field set per `jinit_compress_master` semantics
- [ ] SA2-4: `jpeg_start_compress` populates derived fields
      (`num_components`, `comp_info` array, `jpeg_width`/`jpeg_height`,
      `write_JFIF_header`, `progressive_mode`, etc.)
- [ ] SA2-5: `cargo test -p libjpeg-turbo-rs-capi` green
- [ ] SA2-6: end-to-end stock `cjpeg` → stock `djpeg` roundtrip

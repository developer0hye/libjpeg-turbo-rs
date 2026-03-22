# Phase 5: Codec Completion Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete all missing codec features: DRI encode, lossless encode extension, SOF10/11, grayscale-from-color, custom tables, DCT methods.

**Architecture:** Each task is an independent branch merged via PR. Tasks touch mostly isolated encode pipeline paths.

**Tech Stack:** Rust, cargo test, TDD (Red-Green-Refactor)

---

## File Map

| Task | Create | Modify | Test |
|------|--------|--------|------|
| 1. DRI restart encode | — | `src/encode/pipeline.rs`, `src/api/encoder.rs` | `tests/restart_encode.rs` |
| 2. Lossless encode extension | — | `src/encode/pipeline.rs`, `src/api/encoder.rs` | `tests/lossless_encode.rs` (extend) |
| 3. SOF10 arith progressive encode | — | `src/encode/pipeline.rs`, `src/encode/arithmetic.rs`, `src/api/encoder.rs`, `src/encode/marker_writer.rs`, `src/api/high_level.rs`, `src/lib.rs` | `tests/sof10_encode.rs` |
| 4. SOF11 lossless arithmetic | — | `src/encode/pipeline.rs`, `src/encode/arithmetic.rs`, `src/decode/pipeline.rs`, `src/decode/arithmetic.rs`, `src/encode/marker_writer.rs` | `tests/sof11.rs` |
| 5. Grayscale-from-color | — | `src/api/encoder.rs`, `src/encode/pipeline.rs` | `tests/grayscale_encode.rs` |
| 6. Custom quant tables | — | `src/api/encoder.rs`, `src/encode/pipeline.rs` | `tests/custom_quant.rs` |
| 7. Custom Huffman tables | — | `src/api/encoder.rs`, `src/encode/pipeline.rs` | `tests/custom_huffman.rs` |
| 8. Custom scan script | — | `src/api/encoder.rs`, `src/encode/pipeline.rs` | `tests/custom_scan.rs` |
| 9. DctMethod selection | `src/encode/fdct_fast.rs`, `src/encode/fdct_float.rs` | `src/encode/pipeline.rs`, `src/api/encoder.rs`, `src/encode/mod.rs` | `tests/dct_method.rs` |

---

## Execution Order

Recommended sequence (each independent but builds naturally):

1. **DRI restart encode** — simplest, foundational for correctness
2. **Lossless encode extension** — extends existing lossless path
3. **Grayscale-from-color** — simple utility
4. **Custom quant tables** — extends Encoder
5. **Custom Huffman tables** — extends Encoder
6. **Custom scan script** — extends progressive path
7. **SOF10 arith progressive encode** — combines progressive + arithmetic
8. **SOF11 lossless arithmetic** — combines lossless + arithmetic
9. **DctMethod selection** — new FDCT implementations

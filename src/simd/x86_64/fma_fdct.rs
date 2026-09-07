//! FMA-dispatched twin of the scalar float FDCT + quantise kernel
//! (P4-133, #464).
//!
//! `fdct_float_workspace` writes its rotators with `__mul_add_compat` —
//! `f32::mul_add` on `std` — so they round once, the way clang's
//! `-ffp-contract=on` fuses them in C's `jfdctflt.c`. On a baseline x86_64
//! build that is a libm `fmaf` call per rotator — 32 calls per block — and
//! the 2026-09-08 A/B measured it at 18–23 % of the whole `-dct float`
//! encode. Compiling the same body under `target_feature(enable = "fma")`
//! turns each call into one `vfmadd`. The coefficients cannot change:
//! `mul_add` is a single-rounding fused operation by contract whether libm
//! emulates it or the CPU executes it, so the twin is bit-identical to the
//! scalar kernel and the existing `cjpeg -dct float` parity suites prove it.
//! (On `no_std` the shim is an unfused `a * b + c` in both copies and
//! `cpu_has!` answers from `target_feature`, so a `+fma` `no_std` build
//! installs a twin that is still identical to its scalar sibling.)
//!
//! The kernel is installed once, when the encoder's kernel set is built
//! (`x86_64::encoder_routines`), not chosen per block.

use crate::simd::QuantDivisors;

/// Float FDCT + quantise + zigzag with `f32::mul_add` emitted as `vfmadd`.
///
/// The CPU feature is checked *here*, not assumed of the caller (P4-135,
/// #474): `encoder_routines()` installs this only after its own check, but a
/// safe function must hold for every caller. Without FMA the scalar
/// reference kernel runs, so the result is the same on every CPU.
pub(crate) fn fma_fdct_float_quantize(
    input: &mut [i16; 64],
    quant: &QuantDivisors,
    output: &mut [i16; 64],
) {
    if crate::cpu_has!("fma") {
        // SAFETY: FMA confirmed immediately above; every buffer is a
        // fixed-size array covering the 64 elements the kernel touches.
        unsafe { fma_fdct_float_quantize_inner(input, quant, output) }
    } else {
        crate::simd::scalar::scalar_fdct_float_quantize(input, quant, output);
    }
}

/// The scalar body re-emitted under the FMA feature context.
///
/// # Safety
/// The CPU must support FMA.
#[target_feature(enable = "fma")]
pub(crate) unsafe fn fma_fdct_float_quantize_inner(
    input: &mut [i16; 64],
    quant: &QuantDivisors,
    output: &mut [i16; 64],
) {
    crate::simd::scalar::fdct_float_quantize_body(input, quant, output);
}

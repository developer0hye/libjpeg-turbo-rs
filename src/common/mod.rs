pub mod arith_tables;
pub mod bufsize;
pub mod error;
pub mod exif;
pub mod huffman_table;
pub mod icc;
pub mod jfif;
pub mod quant_table;
pub mod sample;
pub mod traits;
pub mod types;

/// Runtime CPU-feature detection that degrades to compile-time gating
/// on `no_std` (issue #356).
///
/// `is_x86_feature_detected!` lives in `std`; a `no_std` build has no
/// CPUID probe available, so it answers from `target_feature` instead —
/// which is what a firmware/embedded build wants anyway (the target is
/// known at compile time). Semantics on `std` builds are unchanged.
#[macro_export]
#[doc(hidden)]
macro_rules! cpu_has {
    // `tt`, not `literal`: is_x86_feature_detected! inspects raw tokens
    // and cannot see through an opaque literal fragment.
    ($feat:tt) => {{
        #[cfg(feature = "std")]
        {
            // NOTE: std-only macro — must NOT be rewritten to core::arch
            // (it does not exist there). Only reachable on x86_64 + std.
            std::arch::is_x86_feature_detected!($feat)
        }
        #[cfg(not(feature = "std"))]
        {
            cfg!(target_feature = $feat)
        }
    }};
}

/// `no_std` float helpers (issue #356).
///
/// `f32::round`, `f64::cbrt` and `mul_add` live in `std` (they lower to
/// libm calls). A `no_std` build has no libm by default, so these
/// provide the same results for the finite, in-range inputs the codec
/// feeds them: quantization-table scaling and FDCT constants.
#[cfg(not(feature = "std"))]
pub(crate) mod float_compat {
    /// Round half away from zero, matching `f32::round`.
    #[inline]
    pub(crate) fn round_f32(x: f32) -> f32 {
        let t = x as i64 as f32;
        let frac = x - t;
        if x >= 0.0 {
            if frac >= 0.5 {
                t + 1.0
            } else {
                t
            }
        } else if frac <= -0.5 {
            t - 1.0
        } else {
            t
        }
    }

    /// Cube root via Newton iteration, matching `f64::cbrt` closely
    /// enough for quantization-table scaling (inputs are small positive
    /// reals).
    #[inline]
    pub(crate) fn cbrt_f64(x: f64) -> f64 {
        if x == 0.0 {
            return 0.0;
        }
        let sign = if x < 0.0 { -1.0 } else { 1.0 };
        let a = if x < 0.0 { -x } else { x };
        // Seed from the exponent halving trick, then refine.
        let mut y = a / 3.0 + 1.0;
        for _ in 0..40 {
            let next = (2.0 * y + a / (y * y)) / 3.0;
            if (next - y).abs() < 1e-15 * next.abs().max(1.0) {
                y = next;
                break;
            }
            y = next;
        }
        sign * y
    }

    /// Fused-multiply-add fallback (unfused; the codec's uses are not
    /// precision-critical to the fused rounding).
    #[inline]
    pub(crate) fn mul_add_f32(a: f32, b: f32, c: f32) -> f32 {
        a * b + c
    }
}

/// Float-method shims so the codec compiles on `no_std` (issue #356).
/// On `std` these delegate to the standard methods, so results are
/// bit-identical to before; on `no_std` they use `float_compat`.
pub(crate) trait FloatCompat {
    fn __round_compat(self) -> Self;
    fn __mul_add_compat(self, b: Self, c: Self) -> Self;
}

impl FloatCompat for f32 {
    #[inline]
    fn __round_compat(self) -> f32 {
        #[cfg(feature = "std")]
        {
            self.round()
        }
        #[cfg(not(feature = "std"))]
        {
            crate::common::float_compat::round_f32(self)
        }
    }
    #[inline]
    fn __mul_add_compat(self, b: f32, c: f32) -> f32 {
        #[cfg(feature = "std")]
        {
            self.mul_add(b, c)
        }
        #[cfg(not(feature = "std"))]
        {
            crate::common::float_compat::mul_add_f32(self, b, c)
        }
    }
}

pub(crate) trait FloatCompat64 {
    fn __cbrt_compat(self) -> Self;
    fn __round_compat(self) -> Self;
    fn __floor_compat(self) -> Self;
}

impl FloatCompat64 for f64 {
    #[inline]
    fn __cbrt_compat(self) -> f64 {
        #[cfg(feature = "std")]
        {
            self.cbrt()
        }
        #[cfg(not(feature = "std"))]
        {
            crate::common::float_compat::cbrt_f64(self)
        }
    }
    #[inline]
    fn __round_compat(self) -> f64 {
        #[cfg(feature = "std")]
        {
            self.round()
        }
        #[cfg(not(feature = "std"))]
        {
            crate::common::float_compat::round_f32(self as f32) as f64
        }
    }
    #[inline]
    fn __floor_compat(self) -> f64 {
        #[cfg(feature = "std")]
        {
            self.floor()
        }
        #[cfg(not(feature = "std"))]
        {
            let t = self as i64 as f64;
            if self < 0.0 && t != self {
                t - 1.0
            } else {
                t
            }
        }
    }
}

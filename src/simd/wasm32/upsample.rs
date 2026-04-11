//! WASM simd128-accelerated fancy horizontal 2x upsampling.
//!
//! Triangle filter with alternating bias (matches libjpeg-turbo):
//!   output[2i]   = (3 * input[i] + input[i-1] + 1) >> 2
//!   output[2i+1] = (3 * input[i] + input[i+1] + 2) >> 2

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

/// WASM simd128 fancy horizontal 2x upsample.
pub fn wasm_fancy_upsample_h2v1(input: &[u8], in_width: usize, output: &mut [u8]) {
    if in_width == 0 {
        return;
    }
    if in_width == 1 {
        output[0] = input[0];
        output[1] = input[0];
        return;
    }

    output[0] = input[0];
    output[1] = ((3 * input[0] as u16 + input[1] as u16 + 2) >> 2) as u8;

    let last: usize = in_width - 1;
    output[last * 2] = ((3 * input[last] as u16 + input[last - 1] as u16 + 1) >> 2) as u8;
    output[last * 2 + 1] = input[last];

    if in_width <= 2 {
        return;
    }

    unsafe {
        wasm_fancy_h2v1_inner(input, in_width, output);
    }
}

/// Load 8 u8 values, zero-extend to i16.
#[inline(always)]
unsafe fn load_u8x8_as_u16(ptr: *const u8) -> v128 {
    u16x8_extend_low_u8x16(v128_load64_zero(ptr as *const u64))
}

#[target_feature(enable = "simd128")]
unsafe fn wasm_fancy_h2v1_inner(input: &[u8], in_width: usize, output: &mut [u8]) {
    let inptr: *const u8 = input.as_ptr();
    let outptr: *mut u8 = output.as_mut_ptr();

    let three: v128 = i16x8_splat(3);
    let one: v128 = i16x8_splat(1);
    let two: v128 = i16x8_splat(2);
    let zero: v128 = i32x4_splat(0);

    let mut i: usize = 1;

    while i + 8 <= in_width - 1 {
        let left: v128 = load_u8x8_as_u16(inptr.add(i - 1));
        let cur: v128 = load_u8x8_as_u16(inptr.add(i));
        let right: v128 = load_u8x8_as_u16(inptr.add(i + 1));

        let cur3: v128 = i16x8_mul(cur, three);
        // even: bias +1
        let even: v128 = u16x8_shr(i16x8_add(i16x8_add(cur3, left), one), 2);
        // odd: bias +2
        let odd: v128 = u16x8_shr(i16x8_add(i16x8_add(cur3, right), two), 2);

        let even_u8: v128 = u8x16_narrow_i16x8(even, zero);
        let odd_u8: v128 = u8x16_narrow_i16x8(odd, zero);

        // Interleave even/odd bytes
        let interleaved: v128 =
            i8x16_shuffle::<0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23>(
                even_u8, odd_u8,
            );
        v128_store(outptr.add(i * 2) as *mut v128, interleaved);

        i += 8;
    }

    // Scalar tail
    while i < in_width - 1 {
        let left: u16 = input[i - 1] as u16;
        let cur: u16 = input[i] as u16;
        let right: u16 = input[i + 1] as u16;
        output[i * 2] = ((3 * cur + left + 1) >> 2) as u8;
        output[i * 2 + 1] = ((3 * cur + right + 2) >> 2) as u8;
        i += 1;
    }
}

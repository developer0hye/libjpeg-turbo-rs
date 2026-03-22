/// Marker trait for JPEG sample types (8-bit, 12-bit, 16-bit).
pub trait Sample: Copy + Default + Into<i32> + Send + Sync + 'static {
    /// Bit depth of this sample type.
    const BITS_PER_SAMPLE: u8;
    /// Maximum representable value.
    const MAX_VAL: i32;
    /// Center value (used as initial predictor in lossless).
    const CENTER: i32;
    /// If true, this precision is only valid for lossless JPEG.
    const IS_LOSSLESS_ONLY: bool;

    /// Clamp an i32 to the valid range and convert.
    fn from_i32_clamped(v: i32) -> Self;
}

impl Sample for u8 {
    const BITS_PER_SAMPLE: u8 = 8;
    const MAX_VAL: i32 = 255;
    const CENTER: i32 = 128;
    const IS_LOSSLESS_ONLY: bool = false;

    #[inline]
    fn from_i32_clamped(v: i32) -> Self {
        v.clamp(0, 255) as u8
    }
}

impl Sample for i16 {
    const BITS_PER_SAMPLE: u8 = 12;
    const MAX_VAL: i32 = 4095;
    const CENTER: i32 = 2048;
    const IS_LOSSLESS_ONLY: bool = false;

    #[inline]
    fn from_i32_clamped(v: i32) -> Self {
        v.clamp(0, 4095) as i16
    }
}

impl Sample for u16 {
    const BITS_PER_SAMPLE: u8 = 16;
    const MAX_VAL: i32 = 65535;
    const CENTER: i32 = 32768;
    const IS_LOSSLESS_ONLY: bool = true;

    #[inline]
    fn from_i32_clamped(v: i32) -> Self {
        v.clamp(0, 65535) as u16
    }
}

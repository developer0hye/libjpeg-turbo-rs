/// Per-component layout shared by progressive and arithmetic scan decoding.
pub(super) struct CompInfo {
    /// Buffer width in blocks (rounded up to MCU alignment: mcus_x * h_samp).
    pub(super) blocks_x: usize,
    /// Buffer height in blocks (rounded up to MCU alignment: mcus_y * v_samp).
    pub(super) blocks_y: usize,
    pub(super) h_samp: usize,
    pub(super) v_samp: usize,
    pub(super) comp_w: usize,
    pub(super) block_size: usize,
    /// Actual encoded block columns for non-interleaved scans.
    pub(super) width_in_blocks: usize,
    /// Actual encoded block rows for non-interleaved scans.
    pub(super) height_in_blocks: usize,
}

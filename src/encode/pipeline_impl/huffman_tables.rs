use super::{build_huff_table, tables, HuffTable, HuffmanTableDef, Vec};

/// Resolved Huffman tables: the encoding tables plus the exact bits/values that
/// must be written into the DHT markers, so the two can never disagree.
pub(super) struct ResolvedHuffman {
    pub(super) dc_luma_bits: [u8; 17],
    pub(super) dc_luma_values: Vec<u8>,
    pub(super) ac_luma_bits: [u8; 17],
    pub(super) ac_luma_values: Vec<u8>,
    pub(super) dc_chroma_bits: [u8; 17],
    pub(super) dc_chroma_values: Vec<u8>,
    pub(super) ac_chroma_bits: [u8; 17],
    pub(super) ac_chroma_values: Vec<u8>,
    pub(super) dc_luma: HuffTable,
    pub(super) ac_luma: HuffTable,
    pub(super) dc_chroma: HuffTable,
    pub(super) ac_chroma: HuffTable,
}

impl ResolvedHuffman {
    /// Custom slot 0 overrides luma, slot 1 chroma; unset slots use Annex K.
    pub(super) fn resolve(
        custom_dc: Option<&[Option<HuffmanTableDef>; 4]>,
        custom_ac: Option<&[Option<HuffmanTableDef>; 4]>,
    ) -> Self {
        fn pick(
            custom: Option<&[Option<HuffmanTableDef>; 4]>,
            slot: usize,
            default_bits: &[u8; 17],
            default_values: &[u8],
        ) -> ([u8; 17], Vec<u8>) {
            match custom.and_then(|tables| tables[slot].as_ref()) {
                Some(table) => (table.bits, table.values.clone()),
                None => (*default_bits, default_values.to_vec()),
            }
        }

        let (dc_luma_bits, dc_luma_values) = pick(
            custom_dc,
            0,
            &tables::DC_LUMINANCE_BITS,
            &tables::DC_LUMINANCE_VALUES,
        );
        let (ac_luma_bits, ac_luma_values) = pick(
            custom_ac,
            0,
            &tables::AC_LUMINANCE_BITS,
            &tables::AC_LUMINANCE_VALUES,
        );
        let (dc_chroma_bits, dc_chroma_values) = pick(
            custom_dc,
            1,
            &tables::DC_CHROMINANCE_BITS,
            &tables::DC_CHROMINANCE_VALUES,
        );
        let (ac_chroma_bits, ac_chroma_values) = pick(
            custom_ac,
            1,
            &tables::AC_CHROMINANCE_BITS,
            &tables::AC_CHROMINANCE_VALUES,
        );

        Self {
            dc_luma: build_huff_table(&dc_luma_bits, &dc_luma_values),
            ac_luma: build_huff_table(&ac_luma_bits, &ac_luma_values),
            dc_chroma: build_huff_table(&dc_chroma_bits, &dc_chroma_values),
            ac_chroma: build_huff_table(&ac_chroma_bits, &ac_chroma_values),
            dc_luma_bits,
            dc_luma_values,
            ac_luma_bits,
            ac_luma_values,
            dc_chroma_bits,
            dc_chroma_values,
            ac_chroma_bits,
            ac_chroma_values,
        }
    }
}

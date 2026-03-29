/// Arithmetic entropy decoder for JPEG (ITU-T T.81).
///
/// Precise port of jdarith.c from libjpeg-turbo.
use crate::common::arith_tables::*;
use crate::common::error::{JpegError, Result};
use crate::common::quant_table::ZIGZAG_ORDER;

#[derive(Clone, Copy)]
enum StatRef {
    Dc(usize, usize),
    Ac(usize, usize),
    Fixed(usize),
}

/// Arithmetic entropy decoder state.
pub struct ArithDecoder<'a> {
    data: &'a [u8],
    pos: usize,
    unread_marker: bool,

    c: i64,  // C register (use i64 to avoid overflow issues, matching JLONG)
    a: i64,  // A register
    ct: i32, // bit shift counter

    pub last_dc_val: [i32; 4],
    dc_context: [usize; 4],

    dc_stats: [[u8; DC_STAT_BINS]; 4],
    ac_stats: [[u8; AC_STAT_BINS]; 4],
    fixed_bin: [u8; 4],

    arith_dc_l: [u8; 4],
    arith_dc_u: [u8; 4],
    #[allow(dead_code)]
    arith_ac_k: [u8; 4],
}

impl<'a> ArithDecoder<'a> {
    pub fn new(data: &'a [u8], pos: usize) -> Self {
        Self {
            data,
            pos,
            unread_marker: false,
            c: 0,
            a: 0,
            ct: -16,
            last_dc_val: [0; 4],
            dc_context: [0; 4],
            dc_stats: [[0; DC_STAT_BINS]; 4],
            ac_stats: [[0; AC_STAT_BINS]; 4],
            fixed_bin: [113, 0, 0, 0],
            arith_dc_l: [0; 4],
            arith_dc_u: [1; 4],
            arith_ac_k: [5; 4],
        }
    }

    pub fn set_dc_conditioning(&mut self, table: usize, l: u8, u: u8) {
        if table < 4 {
            self.arith_dc_l[table] = l;
            self.arith_dc_u[table] = u;
        }
    }

    pub fn set_ac_conditioning(&mut self, table: usize, kx: u8) {
        if table < 4 {
            self.arith_ac_k[table] = kx;
        }
    }

    /// Read next byte, matching jdarith.c get_byte + 0xFF handling.
    fn get_byte(&mut self) -> i32 {
        if self.unread_marker {
            return 0;
        }
        if self.pos >= self.data.len() {
            return 0;
        }
        let mut data = self.data[self.pos] as i32;
        self.pos += 1;

        if data == 0xFF {
            // Handle stuffed bytes and markers
            loop {
                if self.pos >= self.data.len() {
                    self.unread_marker = true;
                    return 0;
                }
                data = self.data[self.pos] as i32;
                self.pos += 1;
                if data != 0xFF {
                    break;
                }
                // Swallow extra 0xFF bytes
            }
            if data == 0 {
                data = 0xFF; // stuffed zero → restore 0xFF data
            } else {
                // Hit a marker — supply zeros until done
                self.unread_marker = true;
                data = 0;
            }
        }
        data
    }

    fn get_stat(&self, r: StatRef) -> u8 {
        match r {
            StatRef::Dc(tbl, idx) => self.dc_stats[tbl][idx.min(DC_STAT_BINS - 1)],
            StatRef::Ac(tbl, idx) => self.ac_stats[tbl][idx.min(AC_STAT_BINS - 1)],
            StatRef::Fixed(idx) => self.fixed_bin[idx],
        }
    }

    fn set_stat(&mut self, r: StatRef, val: u8) {
        match r {
            StatRef::Dc(tbl, idx) => self.dc_stats[tbl][idx.min(DC_STAT_BINS - 1)] = val,
            StatRef::Ac(tbl, idx) => self.ac_stats[tbl][idx.min(AC_STAT_BINS - 1)] = val,
            StatRef::Fixed(idx) => self.fixed_bin[idx] = val,
        }
    }

    /// Core binary arithmetic decode — exact port of jdarith.c arith_decode().
    fn decode(&mut self, r: StatRef) -> Result<u8> {
        // Renormalization & data input per section D.2.6
        while self.a < 0x8000 {
            self.ct -= 1;
            if self.ct < 0 {
                // Need to fetch next data byte
                let data = self.get_byte();
                self.c = (self.c << 8) | data as i64;
                self.ct += 8;
                if self.ct < 0 {
                    // Need more initial bytes
                    self.ct += 1;
                    if self.ct == 0 {
                        // Got 2 initial bytes → re-init A
                        self.a = 0x8000;
                    }
                }
            }
            self.a <<= 1;
        }

        // Fetch values from ARITAB
        let sv = self.get_stat(r) as i32;
        let mut qe = ARITAB[(sv & 0x7F) as usize] as i64;
        let nl = (qe & 0xFF) as u8;
        qe >>= 8;
        let nm = (qe & 0xFF) as u8;
        qe >>= 8;
        // qe now contains Qe_Value

        // Decode per sections D.2.4 & D.2.5
        let mut temp = self.a - qe;
        self.a = temp;
        temp <<= self.ct;

        let mut sv_out = sv;
        if self.c >= temp {
            self.c -= temp;
            // Conditional LPS exchange
            if self.a < qe {
                self.a = qe;
                self.set_stat(r, ((sv & 0x80) ^ nm as i32) as u8);
            } else {
                self.a = qe;
                self.set_stat(r, ((sv & 0x80) ^ nl as i32) as u8);
                sv_out ^= 0x80; // Exchange LPS/MPS
            }
        } else if self.a < 0x8000 {
            // Conditional MPS exchange
            if self.a < qe {
                self.set_stat(r, ((sv & 0x80) ^ nl as i32) as u8);
                sv_out ^= 0x80;
            } else {
                self.set_stat(r, ((sv & 0x80) ^ nm as i32) as u8);
            }
        }

        Ok((sv_out >> 7) as u8)
    }

    /// Decode DC coefficient for one block (sequential arithmetic).
    ///
    /// Ported from jdarith.c decode_mcu (DC section).
    /// Context layout in dc_stats\[tbl\]:
    ///   [0..3]   = zero diff context (S0, SS, SP, SN)
    ///   [4..7]   = small positive diff context
    ///   [8..11]  = small negative diff context
    ///   [12..15] = large positive diff context
    ///   [16..19] = large negative diff context
    ///   [20..]   = magnitude category encoding (X1=20)
    pub fn decode_dc_sequential(
        &mut self,
        block: &mut [i16; 64],
        comp_idx: usize,
        dc_tbl: usize,
    ) -> Result<()> {
        // S0 = dc_stats[tbl][dc_context[ci]]
        let s0 = self.dc_context[comp_idx];

        // Figure F.19: Decode_DC_DIFF — is difference zero?
        if self.decode(StatRef::Dc(dc_tbl, s0))? == 0 {
            self.dc_context[comp_idx] = 0;
            block[0] = self.last_dc_val[comp_idx] as i16;
            return Ok(());
        }

        // Figure F.22: Decoding the sign of v
        let sign = self.decode(StatRef::Dc(dc_tbl, s0 + 1))? as usize;
        let st_base = s0 + 2 + sign; // SP (s0+2) for positive, SN (s0+3) for negative

        // Figure F.23: Decoding the magnitude category of v
        let mut m: i32 = self.decode(StatRef::Dc(dc_tbl, st_base))? as i32;
        // Track the stats bin position (st) for Figure F.24.
        // In C: st starts at st_base; if m!=0, st is set to dc_stats[tbl]+20
        // and advances through the magnitude loop.
        let mut st_pos: usize = st_base;
        if m != 0 {
            // Magnitude > 0, use X1 bins starting at index 20
            st_pos = 20;
            while self.decode(StatRef::Dc(dc_tbl, st_pos))? != 0 {
                m <<= 1;
                if m == 0x8000 {
                    return Err(JpegError::CorruptData("arithmetic DC overflow".into()));
                }
                st_pos += 1;
            }
        }

        // Section F.1.4.4.1.2: Establish dc_context conditioning category
        let l_thresh = (1i32 << self.arith_dc_l[dc_tbl]) >> 1;
        let u_thresh = (1i32 << self.arith_dc_u[dc_tbl]) >> 1;
        if m < l_thresh {
            self.dc_context[comp_idx] = 0; // zero diff category
        } else if m > u_thresh {
            self.dc_context[comp_idx] = 12 + sign * 4; // large diff category
        } else {
            self.dc_context[comp_idx] = 4 + sign * 4; // small diff category
        }

        // Figure F.24: Decoding the magnitude bit pattern of v
        // C reference: st += 14; same bin used for all magnitude bits
        let mag_st: usize = st_pos + 14;
        let mut v = m;
        let mut bit_mask = m >> 1;
        while bit_mask != 0 {
            if self.decode(StatRef::Dc(dc_tbl, mag_st))? != 0 {
                v |= bit_mask;
            }
            bit_mask >>= 1;
        }

        v += 1; // v is 1-based
        let v = if sign != 0 { -v } else { v };

        self.last_dc_val[comp_idx] = (self.last_dc_val[comp_idx] + v) & 0xFFFF;
        block[0] = self.last_dc_val[comp_idx] as i16;
        Ok(())
    }

    /// Decode AC coefficients for one block (sequential arithmetic).
    ///
    /// Ported from jdarith.c decode_mcu (AC section).
    /// Block output is in zigzag order (matching encoder's quantize_block output).
    pub fn decode_ac_sequential(&mut self, block: &mut [i16; 64], ac_tbl: usize) -> Result<()> {
        let mut k = 1usize;
        while k <= 63 {
            // EOB decision
            let mut st = 3 * (k - 1);
            if self.decode(StatRef::Ac(ac_tbl, st))? != 0 {
                break; // EOB
            }

            // Zero-run
            while self.decode(StatRef::Ac(ac_tbl, st + 1))? == 0 {
                st += 3;
                k += 1;
                if k > 63 {
                    return Err(JpegError::CorruptData(
                        "arithmetic AC spectral overflow".into(),
                    ));
                }
            }

            // Figure F.22: Decoding the sign of v
            let sign = self.decode(StatRef::Fixed(0))?;
            // st += 2 in C (advance past EOB and zero-run bins to magnitude bin)
            let mut st_pos: usize = st + 2;

            // Figure F.23: Decoding the magnitude category of v
            let mut m: i32 = self.decode(StatRef::Ac(ac_tbl, st_pos))? as i32;
            if m != 0 && self.decode(StatRef::Ac(ac_tbl, st_pos))? != 0 {
                m <<= 1;
                let kx = self.arith_ac_k[ac_tbl] as usize;
                st_pos = if k <= kx { 189 } else { 217 };
                while self.decode(StatRef::Ac(ac_tbl, st_pos))? != 0 {
                    m <<= 1;
                    if m == 0x8000 {
                        return Err(JpegError::CorruptData("arithmetic AC overflow".into()));
                    }
                    st_pos += 1;
                }
            }

            // Figure F.24: Decoding the magnitude bit pattern of v
            // C reference: st += 14; same bin used for all magnitude bits
            let mag_st: usize = st_pos + 14;
            let mut v = m;
            let mut bit_mask = m >> 1;
            while bit_mask != 0 {
                if self.decode(StatRef::Ac(ac_tbl, mag_st))? != 0 {
                    v |= bit_mask;
                }
                bit_mask >>= 1;
            }

            v += 1; // v is 1-based
            let v = if sign != 0 { -v } else { v };

            // Output in natural (dezigzagged) order, matching C's jpeg_natural_order
            block[ZIGZAG_ORDER[k]] = v as i16;
            k += 1;
        }

        Ok(())
    }

    /// Decode DC coefficient for progressive first scan (arithmetic).
    pub fn decode_dc_first_progressive(
        &mut self,
        block: &mut [i16; 64],
        comp_idx: usize,
        dc_tbl: usize,
        al: u8,
    ) -> Result<()> {
        // Same as decode_dc_sequential but output is LEFT_SHIFT(last_dc_val, al)
        let s0 = self.dc_context[comp_idx];

        if self.decode(StatRef::Dc(dc_tbl, s0))? == 0 {
            self.dc_context[comp_idx] = 0;
            block[0] = (self.last_dc_val[comp_idx] << al) as i16;
            return Ok(());
        }

        let sign = self.decode(StatRef::Dc(dc_tbl, s0 + 1))? as usize;
        let st_base = s0 + 2 + sign;

        let mut m: i32 = self.decode(StatRef::Dc(dc_tbl, st_base))? as i32;
        let mut st_pos: usize = st_base;
        if m != 0 {
            st_pos = 20;
            while self.decode(StatRef::Dc(dc_tbl, st_pos))? != 0 {
                m <<= 1;
                if m == 0x8000 {
                    return Err(JpegError::CorruptData("arithmetic DC overflow".into()));
                }
                st_pos += 1;
            }
        }

        let l_thresh = (1i32 << self.arith_dc_l[dc_tbl]) >> 1;
        let u_thresh = (1i32 << self.arith_dc_u[dc_tbl]) >> 1;
        if m < l_thresh {
            self.dc_context[comp_idx] = 0;
        } else if m > u_thresh {
            self.dc_context[comp_idx] = 12 + sign * 4;
        } else {
            self.dc_context[comp_idx] = 4 + sign * 4;
        }

        // Figure F.24: C reference uses st += 14 from last magnitude bin
        let mag_st: usize = st_pos + 14;
        let mut v = m;
        let mut bit_mask = m >> 1;
        while bit_mask != 0 {
            if self.decode(StatRef::Dc(dc_tbl, mag_st))? != 0 {
                v |= bit_mask;
            }
            bit_mask >>= 1;
        }

        v += 1;
        let v = if sign != 0 { -v } else { v };

        self.last_dc_val[comp_idx] = (self.last_dc_val[comp_idx] + v) & 0xFFFF;
        block[0] = (self.last_dc_val[comp_idx] << al) as i16;
        Ok(())
    }

    /// Decode DC refinement for progressive (arithmetic).
    pub fn decode_dc_refine_progressive(&mut self, block: &mut [i16; 64], al: u8) -> Result<()> {
        let p1 = 1i16 << al;
        if self.decode(StatRef::Fixed(0))? != 0 {
            block[0] |= p1;
        }
        Ok(())
    }

    /// Decode AC first scan for progressive (arithmetic).
    pub fn decode_ac_first_progressive(
        &mut self,
        block: &mut [i16; 64],
        ac_tbl: usize,
        ss: u8,
        se: u8,
        al: u8,
    ) -> Result<()> {
        let mut k = ss as usize;
        while k <= se as usize {
            let mut st = 3 * (k - 1);
            if self.decode(StatRef::Ac(ac_tbl, st))? != 0 {
                break; // EOB
            }
            while self.decode(StatRef::Ac(ac_tbl, st + 1))? == 0 {
                st += 3;
                k += 1;
                if k > se as usize {
                    return Err(JpegError::CorruptData(
                        "arithmetic AC spectral overflow".into(),
                    ));
                }
            }
            let sign = self.decode(StatRef::Fixed(0))?;
            // st += 2 in C (advance to magnitude context bin)
            let mut st_pos: usize = st + 2;
            let mut m: i32 = self.decode(StatRef::Ac(ac_tbl, st_pos))? as i32;
            if m != 0 && self.decode(StatRef::Ac(ac_tbl, st_pos))? != 0 {
                m <<= 1;
                let kx = self.arith_ac_k[ac_tbl] as usize;
                st_pos = if k <= kx { 189 } else { 217 };
                while self.decode(StatRef::Ac(ac_tbl, st_pos))? != 0 {
                    m <<= 1;
                    if m == 0x8000 {
                        return Err(JpegError::CorruptData("arithmetic AC overflow".into()));
                    }
                    st_pos += 1;
                }
            }
            // Figure F.24: C reference uses st += 14 from last magnitude bin
            let mag_st: usize = st_pos + 14;
            let mut v = m;
            let mut bit_mask = m >> 1;
            while bit_mask != 0 {
                if self.decode(StatRef::Ac(ac_tbl, mag_st))? != 0 {
                    v |= bit_mask;
                }
                bit_mask >>= 1;
            }
            v += 1;
            let v = if sign != 0 { -v } else { v };
            // Output in natural (dezigzagged) order, shifted by al
            block[ZIGZAG_ORDER[k]] = (v << al) as i16;
            k += 1;
        }
        Ok(())
    }

    /// Decode AC refinement for progressive (arithmetic).
    pub fn decode_ac_refine_progressive(
        &mut self,
        block: &mut [i16; 64],
        ac_tbl: usize,
        ss: u8,
        se: u8,
        al: u8,
    ) -> Result<()> {
        let p1 = 1i16 << al;
        let m1 = (-1i16) << al;

        // Establish EOBx (previous stage end-of-block) index
        let mut kex = se as usize;
        while kex > 0 {
            if block[ZIGZAG_ORDER[kex]] != 0 {
                break;
            }
            kex -= 1;
        }

        let mut k = ss as usize;
        while k <= se as usize {
            let st = 3 * (k - 1);
            if k > kex && self.decode(StatRef::Ac(ac_tbl, st))? != 0 {
                break; // EOB
            }
            loop {
                let natural = ZIGZAG_ORDER[k];
                if block[natural] != 0 {
                    // Previously nonzero coefficient: read correction bit
                    if self.decode(StatRef::Ac(ac_tbl, st + 2))? != 0 {
                        if block[natural] < 0 {
                            block[natural] = block[natural].wrapping_add(m1);
                        } else {
                            block[natural] = block[natural].wrapping_add(p1);
                        }
                    }
                    break;
                }
                if self.decode(StatRef::Ac(ac_tbl, st + 1))? != 0 {
                    // Newly nonzero coefficient
                    if self.decode(StatRef::Fixed(0))? != 0 {
                        block[natural] = m1;
                    } else {
                        block[natural] = p1;
                    }
                    break;
                }
                k += 1;
                if k > se as usize {
                    return Err(JpegError::CorruptData(
                        "arithmetic AC spectral overflow".into(),
                    ));
                }
            }
            k += 1;
        }
        Ok(())
    }

    pub fn position(&self) -> usize {
        self.pos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoder_initializes_correctly() {
        let data = [0u8; 16];
        let decoder = ArithDecoder::new(&data, 0);
        assert_eq!(decoder.a, 0);
        assert_eq!(decoder.ct, -16);
        assert_eq!(decoder.last_dc_val, [0; 4]);
    }
}

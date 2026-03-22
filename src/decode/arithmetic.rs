/// Arithmetic entropy decoder for JPEG (ITU-T T.81).
///
/// Precise port of jdarith.c from libjpeg-turbo.
use crate::common::arith_tables::*;
use crate::common::error::{JpegError, Result};

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
            fixed_bin: [0; 4],
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
    pub fn decode_dc_sequential(
        &mut self,
        block: &mut [i16; 64],
        comp_idx: usize,
        dc_tbl: usize,
    ) -> Result<()> {
        let ctx = self.dc_context[comp_idx];

        // Is DC difference nonzero?
        let is_nonzero = self.decode(StatRef::Dc(dc_tbl, ctx * 4))?;

        if is_nonzero == 0 {
            self.dc_context[comp_idx] = 0;
            block[0] = self.last_dc_val[comp_idx] as i16;
            return Ok(());
        }

        // Sign (fixed 0.5 probability)
        let sign = self.decode(StatRef::Fixed(0))?;

        // Magnitude (unary)
        let stat_offset = ctx * 4 + 2;
        let mut m: i32 = 1;
        loop {
            let bit = self.decode(StatRef::Dc(dc_tbl, stat_offset))?;
            if bit == 0 {
                break;
            }
            m <<= 1;
            if m > 0x7FFF {
                return Err(JpegError::CorruptData("arithmetic DC overflow".into()));
            }
        }

        // Magnitude bits
        let mut v = m;
        if m > 1 {
            let mut bit_pos = m >> 1;
            while bit_pos > 0 {
                let bit = self.decode(StatRef::Fixed(0))?;
                if bit != 0 {
                    v |= bit_pos;
                }
                bit_pos >>= 1;
            }
        }

        let v = if sign != 0 { -v } else { v };

        // Update DC context
        let l = self.arith_dc_l[dc_tbl] as i32;
        let u = self.arith_dc_u[dc_tbl] as i32;
        self.dc_context[comp_idx] = if v < l {
            4 + (if v <= -(1 << u) { 12 } else { 0 })
        } else if v > u {
            8 + (if v >= (1 << u) { 12 } else { 0 })
        } else {
            0
        };

        self.last_dc_val[comp_idx] += v;
        block[0] = self.last_dc_val[comp_idx] as i16;
        Ok(())
    }

    /// Decode AC coefficients for one block (sequential arithmetic).
    pub fn decode_ac_sequential(&mut self, block: &mut [i16; 64], ac_tbl: usize) -> Result<()> {
        let mut k = 1usize;
        while k < 64 {
            // EOB?
            let eob = self.decode(StatRef::Ac(ac_tbl, 3 * (k - 1)))?;
            if eob != 0 {
                break;
            }

            // Zero-run
            loop {
                let is_nonzero = self.decode(StatRef::Ac(ac_tbl, 3 * (k - 1) + 1))?;
                if is_nonzero != 0 {
                    break;
                }
                k += 1;
                if k >= 64 {
                    return Ok(());
                }
            }

            // Sign
            let sign = self.decode(StatRef::Fixed(0))?;

            // Magnitude (unary)
            let stat_base = 3 * (k - 1) + 2;
            let mut m: i32 = 1;
            loop {
                let bit = self.decode(StatRef::Ac(ac_tbl, stat_base))?;
                if bit == 0 {
                    break;
                }
                m <<= 1;
                if m > 0x7FFF {
                    return Err(JpegError::CorruptData("arithmetic AC overflow".into()));
                }
            }

            // Magnitude bits
            let mut v = m;
            if m > 1 {
                let mut bit_pos = m >> 1;
                while bit_pos > 0 {
                    let bit = self.decode(StatRef::Fixed(0))?;
                    if bit != 0 {
                        v |= bit_pos;
                    }
                    bit_pos >>= 1;
                }
            }

            block[k] = if sign != 0 { -v as i16 } else { v as i16 };
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

/// Arithmetic entropy encoder for JPEG (ITU-T T.81).
///
/// Implements the QM-coder binary arithmetic encoder used for
/// SOF9 (sequential arithmetic) JPEG encoding.
use crate::common::arith_tables::*;

/// Which statistics table to read/write.
#[derive(Clone, Copy)]
enum StatRef {
    Dc(usize, usize),
    Ac(usize, usize),
    Fixed(usize),
}

/// Arithmetic entropy encoder state.
pub struct ArithEncoder {
    c: u32,
    a: u32,
    ct: i32,
    sc: i32,
    buffer: i32,

    output: Vec<u8>,

    pub last_dc_val: [i32; 4],
    dc_context: [usize; 4],

    dc_stats: [[u8; DC_STAT_BINS]; 4],
    ac_stats: [[u8; AC_STAT_BINS]; 4],
    fixed_bin: [u8; 4],

    arith_dc_l: [u8; 4],
    arith_dc_u: [u8; 4],
    arith_ac_k: [u8; 4],
}

impl ArithEncoder {
    pub fn new(capacity: usize) -> Self {
        Self {
            c: 0,
            a: 0x10000,
            ct: 11,
            sc: 0,
            buffer: -1,
            output: Vec::with_capacity(capacity),
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

    fn emit_byte(&mut self, val: u8) {
        self.output.push(val);
        if val == 0xFF {
            self.output.push(0x00);
        }
    }

    fn get_stat(&self, r: StatRef) -> u8 {
        match r {
            StatRef::Dc(tbl, idx) => self.dc_stats[tbl][idx],
            StatRef::Ac(tbl, idx) => self.ac_stats[tbl][idx],
            StatRef::Fixed(idx) => self.fixed_bin[idx],
        }
    }

    fn set_stat(&mut self, r: StatRef, val: u8) {
        match r {
            StatRef::Dc(tbl, idx) => self.dc_stats[tbl][idx] = val,
            StatRef::Ac(tbl, idx) => self.ac_stats[tbl][idx] = val,
            StatRef::Fixed(idx) => self.fixed_bin[idx] = val,
        }
    }

    /// Core binary arithmetic encode using a StatRef.
    fn encode(&mut self, r: StatRef, val: u8) {
        let sv = self.get_stat(r);
        let state_idx = (sv & 0x7F) as usize;
        let entry = ARITAB[state_idx];
        let qe = qe_value(entry);
        let nm = next_mps(entry);
        let nl_byte = next_lps_with_switch(entry);
        let mps_val = sv >> 7;

        self.a -= qe;

        if val != mps_val {
            // LPS
            if self.a >= qe {
                self.c += self.a;
                self.a = qe;
            }
            self.set_stat(r, (sv & 0x80) ^ nl_byte);
        } else {
            // MPS
            if self.a >= 0x8000 {
                return;
            }
            if self.a < qe {
                self.c += self.a;
                self.a = qe;
            }
            self.set_stat(r, (sv & 0x80) | nm);
        }

        // Renormalization
        while self.a < 0x8000 {
            self.a <<= 1;
            self.c <<= 1;
            self.ct -= 1;

            if self.ct == 0 {
                let temp = self.c >> 19;
                if temp > 0xFF {
                    if self.buffer >= 0 {
                        self.emit_byte((self.buffer as u8).wrapping_add(1));
                    }
                    while self.sc > 0 {
                        self.emit_byte(0x00);
                        self.sc -= 1;
                    }
                    self.buffer = (temp & 0xFF) as i32;
                } else if temp == 0xFF {
                    self.sc += 1;
                } else {
                    if self.buffer >= 0 {
                        self.emit_byte(self.buffer as u8);
                    }
                    while self.sc > 0 {
                        self.emit_byte(0xFF);
                        self.sc -= 1;
                    }
                    self.buffer = temp as i32;
                }
                self.c &= 0x7FFFF;
                self.ct = 8;
            }
        }
    }

    /// Encode DC coefficient for one block (sequential arithmetic).
    pub fn encode_dc_sequential(&mut self, block: &[i16; 64], comp_idx: usize, dc_tbl: usize) {
        let dc_val = block[0] as i32;
        let diff = dc_val - self.last_dc_val[comp_idx];
        self.last_dc_val[comp_idx] = dc_val;

        let ctx = self.dc_context[comp_idx];

        // Step 1: Is difference nonzero?
        let is_nonzero = if diff != 0 { 1u8 } else { 0u8 };
        self.encode(StatRef::Dc(dc_tbl, ctx * 4), is_nonzero);

        if diff == 0 {
            self.dc_context[comp_idx] = 0;
            return;
        }

        // Step 2: Sign
        let (sign, abs_diff) = if diff < 0 {
            (1u8, (-diff) as u32)
        } else {
            (0u8, diff as u32)
        };
        self.encode(StatRef::Fixed(0), sign);

        // Step 3: Magnitude (unary)
        let stat_offset = (ctx * 4 + 2).min(DC_STAT_BINS - 1);
        let mut m: u32 = 1;
        while m < abs_diff {
            self.encode(StatRef::Dc(dc_tbl, stat_offset), 1);
            m <<= 1;
        }
        self.encode(StatRef::Dc(dc_tbl, stat_offset), 0);

        // Step 4: Magnitude bits
        if m > 1 {
            let mut bit_pos = m >> 1;
            while bit_pos > 0 {
                let bit = if abs_diff & bit_pos != 0 { 1u8 } else { 0u8 };
                self.encode(StatRef::Fixed(0), bit);
                bit_pos >>= 1;
            }
        }

        // Update DC context
        let l = self.arith_dc_l[dc_tbl] as i32;
        let u = self.arith_dc_u[dc_tbl] as i32;
        self.dc_context[comp_idx] = if diff < l {
            4 + (if diff <= -(1 << u) { 12 } else { 0 })
        } else if diff > u {
            8 + (if diff >= (1 << u) { 12 } else { 0 })
        } else {
            0
        };
    }

    /// Encode AC coefficients for one block (sequential arithmetic).
    pub fn encode_ac_sequential(&mut self, block: &[i16; 64], ac_tbl: usize) {
        // Find last nonzero coefficient
        let mut last_nonzero = 0usize;
        for k in (1..64).rev() {
            if block[k] != 0 {
                last_nonzero = k;
                break;
            }
        }

        let mut k = 1usize;
        while k <= last_nonzero {
            // Not EOB
            let eob_idx = (3 * (k - 1)).min(AC_STAT_BINS - 1);
            self.encode(StatRef::Ac(ac_tbl, eob_idx), 0);

            // Zero-run
            while block[k] == 0 {
                let zr_idx = (3 * (k - 1) + 1).min(AC_STAT_BINS - 1);
                self.encode(StatRef::Ac(ac_tbl, zr_idx), 0);
                k += 1;
            }
            let nz_idx = (3 * (k - 1) + 1).min(AC_STAT_BINS - 1);
            self.encode(StatRef::Ac(ac_tbl, nz_idx), 1);

            // Sign
            let val = block[k];
            let (sign, abs_val) = if val < 0 {
                (1u8, (-val) as u32)
            } else {
                (0u8, val as u32)
            };
            self.encode(StatRef::Fixed(0), sign);

            // Magnitude (unary)
            let stat_base = (3 * (k - 1) + 2).min(AC_STAT_BINS - 1);
            let mut m: u32 = 1;
            while m < abs_val {
                self.encode(StatRef::Ac(ac_tbl, stat_base), 1);
                m <<= 1;
            }
            self.encode(StatRef::Ac(ac_tbl, stat_base), 0);

            // Magnitude bits
            if m > 1 {
                let mut bit_pos = m >> 1;
                while bit_pos > 0 {
                    let bit = if abs_val & bit_pos != 0 { 1u8 } else { 0u8 };
                    self.encode(StatRef::Fixed(0), bit);
                    bit_pos >>= 1;
                }
            }

            k += 1;
        }

        // EOB
        if last_nonzero < 63 && k < 64 {
            let idx = (3 * (k - 1)).min(AC_STAT_BINS - 1);
            self.encode(StatRef::Ac(ac_tbl, idx), 1);
        }
    }

    /// Finish encoding: flush remaining bits.
    pub fn finish(&mut self) {
        let temp = (self.a.wrapping_sub(1).wrapping_add(self.c)) & 0xFFFF0000;
        let temp = if temp < self.c { temp + 0x8000 } else { temp };
        self.c = temp;

        // Flush final bytes
        for _ in 0..2 {
            self.c <<= 1;
            self.ct -= 1;
            if self.ct == 0 {
                let byte = self.c >> 19;
                if byte > 0xFF {
                    if self.buffer >= 0 {
                        self.emit_byte((self.buffer as u8).wrapping_add(1));
                    }
                    while self.sc > 0 {
                        self.emit_byte(0x00);
                        self.sc -= 1;
                    }
                    self.buffer = (byte & 0xFF) as i32;
                } else if byte == 0xFF {
                    self.sc += 1;
                } else {
                    if self.buffer >= 0 {
                        self.emit_byte(self.buffer as u8);
                    }
                    while self.sc > 0 {
                        self.emit_byte(0xFF);
                        self.sc -= 1;
                    }
                    self.buffer = byte as i32;
                }
                self.c &= 0x7FFFF;
                self.ct = 8;
            }
        }

        // Emit remaining buffer
        if self.buffer >= 0 {
            self.emit_byte(self.buffer as u8);
        }
        while self.sc > 0 {
            self.emit_byte(0xFF);
            self.sc -= 1;
        }
    }

    pub fn data(&self) -> &[u8] {
        &self.output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_initializes_correctly() {
        let enc = ArithEncoder::new(256);
        assert_eq!(enc.a, 0x10000);
        assert_eq!(enc.ct, 11);
        assert_eq!(enc.buffer, -1);
    }

    #[test]
    fn encode_enough_bits_produces_output() {
        let mut enc = ArithEncoder::new(256);
        // Encode enough bits to force output
        for _ in 0..20 {
            enc.encode(StatRef::Fixed(0), 0);
            enc.encode(StatRef::Fixed(0), 1);
        }
        enc.finish();
        assert!(!enc.data().is_empty());
    }

    #[test]
    fn encode_multiple_mps() {
        let mut enc = ArithEncoder::new(256);
        for _ in 0..100 {
            enc.encode(StatRef::Fixed(0), 0);
        }
        enc.finish();
        assert!(enc.data().len() < 20);
    }
}

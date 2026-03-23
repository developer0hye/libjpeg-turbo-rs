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
    /// Counter for pending 0x00 output values which might be discarded
    /// at the end ("Pacman" termination per ITU-T T.81 Figure D.15).
    zc: i32,
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
            zc: 0,
            buffer: -1,
            output: Vec::with_capacity(capacity),
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

    /// Raw byte output (matches C emit_byte — no auto-stuffing).
    fn emit_byte(&mut self, val: u8) {
        self.output.push(val);
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

        // Renormalization & data output per section D.1.6
        // Matches jcarith.c arith_encode() renormalization exactly.
        loop {
            self.a <<= 1;
            self.c <<= 1;
            self.ct -= 1;

            if self.ct == 0 {
                let temp = self.c >> 19;
                if temp > 0xFF {
                    // Handle overflow over all stacked 0xFF bytes
                    if self.buffer >= 0 {
                        while self.zc > 0 {
                            self.emit_byte(0x00);
                            self.zc -= 1;
                        }
                        let byte = (self.buffer + 1) as u8;
                        self.emit_byte(byte);
                        if byte == 0xFF {
                            self.emit_byte(0x00);
                        }
                    }
                    // Carry-over converts stacked 0xFF bytes to 0x00
                    self.zc += self.sc;
                    self.sc = 0;
                    self.buffer = (temp & 0xFF) as i32;
                } else if temp == 0xFF {
                    self.sc += 1;
                } else {
                    // Output all stacked 0xFF bytes, they will not overflow
                    if self.buffer == 0 {
                        self.zc += 1;
                    } else if self.buffer >= 0 {
                        while self.zc > 0 {
                            self.emit_byte(0x00);
                            self.zc -= 1;
                        }
                        self.emit_byte(self.buffer as u8);
                    }
                    if self.sc > 0 {
                        while self.zc > 0 {
                            self.emit_byte(0x00);
                            self.zc -= 1;
                        }
                        while self.sc > 0 {
                            self.emit_byte(0xFF);
                            self.emit_byte(0x00);
                            self.sc -= 1;
                        }
                    }
                    self.buffer = (temp & 0xFF) as i32;
                }
                self.c &= 0x7FFFF;
                self.ct += 8;
            }

            if self.a >= 0x8000 {
                break;
            }
        }
    }

    /// Encode DC coefficient for one block (sequential arithmetic).
    ///
    /// Ported from jcarith.c encode_mcu_DC_first / encode_mcu sections.
    /// Context layout in dc_stats[tbl]:
    ///   [0..3]   = zero diff context (S0, SS, SP, SN)
    ///   [4..7]   = small positive diff context
    ///   [8..11]  = small negative diff context
    ///   [12..15] = large positive diff context
    ///   [16..19] = large negative diff context
    ///   [20..]   = magnitude category encoding (X1=20)
    pub fn encode_dc_sequential(&mut self, block: &[i16; 64], comp_idx: usize, dc_tbl: usize) {
        let dc_val = block[0] as i32;
        let mut v: i32 = dc_val - self.last_dc_val[comp_idx];

        // S0 = dc_stats[tbl][dc_context[ci]]
        let s0 = self.dc_context[comp_idx];

        if v == 0 {
            // Zero difference
            self.encode(StatRef::Dc(dc_tbl, s0), 0);
            self.dc_context[comp_idx] = 0;
            return;
        }

        self.last_dc_val[comp_idx] = dc_val;
        self.encode(StatRef::Dc(dc_tbl, s0), 1); // nonzero

        // Sign encoding + stat pointer selection (Table F.4)
        let st: usize;
        if v > 0 {
            self.encode(StatRef::Dc(dc_tbl, s0 + 1), 0); // SS: positive
            st = s0 + 2; // SP
            self.dc_context[comp_idx] = 4; // small positive
        } else {
            v = -v;
            self.encode(StatRef::Dc(dc_tbl, s0 + 1), 1); // SS: negative
            st = s0 + 3; // SN
            self.dc_context[comp_idx] = 8; // small negative
        }

        // Magnitude category encoding (Figure F.8)
        let mut m: i32 = 0;
        v -= 1; // v is now (abs_diff - 1)
        let v_orig = v; // save for magnitude bits
        if v != 0 {
            self.encode(StatRef::Dc(dc_tbl, st), 1);
            m = 1;
            let mut v2: i32 = v;
            let mut x1 = 20usize; // Table F.4: X1 = 20
            v2 >>= 1;
            while v2 != 0 {
                self.encode(StatRef::Dc(dc_tbl, x1), 1);
                m <<= 1;
                x1 += 1;
                v2 >>= 1;
            }
            // Magnitude terminator at the X1 position
            self.encode(StatRef::Dc(dc_tbl, x1), 0);

            // Update context based on magnitude vs conditioning thresholds
            let l_thresh = (1i32 << self.arith_dc_l[dc_tbl]) >> 1;
            let u_thresh = (1i32 << self.arith_dc_u[dc_tbl]) >> 1;
            if m < l_thresh {
                self.dc_context[comp_idx] = 0;
            } else if m > u_thresh {
                self.dc_context[comp_idx] += 8; // promote to large category
            }

            // Magnitude bit pattern (Figure F.9) — uses adaptive stats bin st+14
            let mag_st = x1 + 14;
            let mut bit_mask = m >> 1;
            while bit_mask != 0 {
                let bit = if (bit_mask & v_orig) != 0 { 1u8 } else { 0u8 };
                self.encode(StatRef::Dc(dc_tbl, mag_st), bit);
                bit_mask >>= 1;
            }
        } else {
            // v was 1 (abs_diff == 1), magnitude category 0
            self.encode(StatRef::Dc(dc_tbl, st), 0);
            // Context update: m=0 < any positive L threshold → set to 0
            let l_thresh = (1i32 << self.arith_dc_l[dc_tbl]) >> 1;
            if m < l_thresh {
                self.dc_context[comp_idx] = 0;
            }
        }
    }

    /// Encode AC coefficients for one block (sequential arithmetic).
    ///
    /// Ported from jcarith.c encode_mcu (AC section).
    /// Block is expected in zigzag order (as output by quantize_block).
    pub fn encode_ac_sequential(&mut self, block: &[i16; 64], ac_tbl: usize) {
        // Establish EOB (end-of-block) index
        let mut ke: usize = 63;
        while ke > 0 {
            if block[ke] != 0 {
                break;
            }
            ke -= 1;
        }

        // Encode AC coefficients (Figure F.5)
        let mut k = 1usize;
        while k <= ke {
            let mut st = 3 * (k - 1);
            self.encode(StatRef::Ac(ac_tbl, st), 0); // EOB decision: not EOB

            // Zero-run
            let mut v: i32 = block[k] as i32;
            while v == 0 {
                self.encode(StatRef::Ac(ac_tbl, st + 1), 0);
                st += 3;
                k += 1;
                v = block[k] as i32;
            }
            self.encode(StatRef::Ac(ac_tbl, st + 1), 1); // nonzero

            // Sign
            if v > 0 {
                self.encode(StatRef::Fixed(0), 0);
            } else {
                v = -v;
                self.encode(StatRef::Fixed(0), 1);
            }

            st += 2;

            // Magnitude category encoding (Figure F.8)
            let mut m: i32 = 0;
            v -= 1;
            let v_orig = v;
            if v != 0 {
                self.encode(StatRef::Ac(ac_tbl, st), 1);
                m = 1;
                let mut v2 = v >> 1;
                if v2 != 0 {
                    self.encode(StatRef::Ac(ac_tbl, st), 1);
                    m <<= 1;
                    let kx = self.arith_ac_k[ac_tbl] as usize;
                    st = if k <= kx { 189 } else { 217 };
                    v2 >>= 1;
                    while v2 != 0 {
                        self.encode(StatRef::Ac(ac_tbl, st), 1);
                        m <<= 1;
                        st += 1;
                        v2 >>= 1;
                    }
                }
            }
            self.encode(StatRef::Ac(ac_tbl, st), 0); // magnitude terminator

            // Magnitude bit pattern (Figure F.9) — uses adaptive stats bin st+14
            let mag_st = st + 14;
            let mut bit_mask = m >> 1;
            while bit_mask != 0 {
                let bit = if (bit_mask & v_orig) != 0 { 1u8 } else { 0u8 };
                self.encode(StatRef::Ac(ac_tbl, mag_st), bit);
                bit_mask >>= 1;
            }

            k += 1;
        }

        // Encode EOB decision if k <= 63
        if k <= 63 {
            let st = 3 * (k - 1);
            self.encode(StatRef::Ac(ac_tbl, st), 1);
        }
    }

    /// Finish encoding: flush remaining bits.
    ///
    /// Implements Section D.1.8 of ITU-T T.81 with "Pacman" termination.
    /// Ported from jcarith.c finish_pass().
    pub fn finish(&mut self) {
        // Find the c in the coding interval with the largest
        // number of trailing zero bits
        let temp: u32 = (self.a.wrapping_sub(1).wrapping_add(self.c)) & 0xFFFF0000;
        self.c = if temp < self.c { temp + 0x8000 } else { temp };

        // Send remaining bytes to output — shift by ct bits at once
        self.c <<= self.ct;

        if self.c & 0xF8000000 != 0 {
            // One final overflow has to be handled
            if self.buffer >= 0 {
                while self.zc > 0 {
                    self.emit_byte(0x00);
                    self.zc -= 1;
                }
                let byte = (self.buffer + 1) as u8;
                self.emit_byte(byte);
                if byte == 0xFF {
                    self.emit_byte(0x00);
                }
            }
            // Carry-over converts stacked 0xFF bytes to 0x00
            self.zc += self.sc;
            self.sc = 0;
        } else {
            if self.buffer == 0 {
                self.zc += 1;
            } else if self.buffer >= 0 {
                while self.zc > 0 {
                    self.emit_byte(0x00);
                    self.zc -= 1;
                }
                self.emit_byte(self.buffer as u8);
            }
            if self.sc > 0 {
                while self.zc > 0 {
                    self.emit_byte(0x00);
                    self.zc -= 1;
                }
                while self.sc > 0 {
                    self.emit_byte(0xFF);
                    self.emit_byte(0x00);
                    self.sc -= 1;
                }
            }
        }

        // Output final bytes only if they are not 0x00
        if self.c & 0x7FFF800 != 0 {
            while self.zc > 0 {
                self.emit_byte(0x00);
                self.zc -= 1;
            }
            let byte1 = ((self.c >> 19) & 0xFF) as u8;
            self.emit_byte(byte1);
            if byte1 == 0xFF {
                self.emit_byte(0x00);
            }
            if self.c & 0x7F800 != 0 {
                let byte2 = ((self.c >> 11) & 0xFF) as u8;
                self.emit_byte(byte2);
                if byte2 == 0xFF {
                    self.emit_byte(0x00);
                }
            }
        }
    }

    /// Reset encoder state for a new scan, keeping output buffer.
    ///
    /// Clears arithmetic coding state, DC prediction, and statistics
    /// so the encoder is ready for a fresh progressive scan.
    pub fn reset(&mut self) {
        self.c = 0;
        self.a = 0x10000;
        self.ct = 11;
        self.sc = 0;
        self.zc = 0;
        self.buffer = -1;
        self.output.clear();
        self.last_dc_val = [0; 4];
        self.dc_context = [0; 4];
        self.dc_stats = [[0; DC_STAT_BINS]; 4];
        self.ac_stats = [[0; AC_STAT_BINS]; 4];
        self.fixed_bin = [113, 0, 0, 0];
    }

    /// Encode DC coefficient for first progressive scan (DC first, Ah=0).
    ///
    /// Like encode_dc_sequential but applies point transform shift by `al`.
    /// Ported from jcarith.c encode_mcu_DC_first.
    pub fn encode_dc_first(&mut self, block: &[i16; 64], comp_idx: usize, dc_tbl: usize, al: u8) {
        // Apply point transform (arithmetic right shift by Al)
        let m_val: i32 = (block[0] as i32) >> al;

        let s0: usize = self.dc_context[comp_idx];

        if m_val - self.last_dc_val[comp_idx] == 0 {
            self.encode(StatRef::Dc(dc_tbl, s0), 0);
            self.dc_context[comp_idx] = 0;
            return;
        }

        let mut v: i32 = m_val - self.last_dc_val[comp_idx];
        self.last_dc_val[comp_idx] = m_val;
        self.encode(StatRef::Dc(dc_tbl, s0), 1);

        // Sign encoding + stat pointer selection (Table F.4)
        let st: usize;
        if v > 0 {
            self.encode(StatRef::Dc(dc_tbl, s0 + 1), 0); // positive
            st = s0 + 2;
            self.dc_context[comp_idx] = 4;
        } else {
            v = -v;
            self.encode(StatRef::Dc(dc_tbl, s0 + 1), 1); // negative
            st = s0 + 3;
            self.dc_context[comp_idx] = 8;
        }

        // Magnitude category encoding (Figure F.8)
        let mut m: i32 = 0;
        v -= 1;
        let v_orig: i32 = v;
        if v != 0 {
            self.encode(StatRef::Dc(dc_tbl, st), 1);
            m = 1;
            let mut v2: i32 = v;
            let mut x1: usize = 20;
            v2 >>= 1;
            while v2 != 0 {
                self.encode(StatRef::Dc(dc_tbl, x1), 1);
                m <<= 1;
                x1 += 1;
                v2 >>= 1;
            }
            self.encode(StatRef::Dc(dc_tbl, x1), 0);

            // Context conditioning (Section F.1.4.4.1.2)
            let l_thresh: i32 = (1i32 << self.arith_dc_l[dc_tbl]) >> 1;
            let u_thresh: i32 = (1i32 << self.arith_dc_u[dc_tbl]) >> 1;
            if m < l_thresh {
                self.dc_context[comp_idx] = 0;
            } else if m > u_thresh {
                self.dc_context[comp_idx] += 8;
            }

            // Magnitude bit pattern (Figure F.9) — uses adaptive stats bin st+14
            let mag_st: usize = x1 + 14;
            let mut bit_mask: i32 = m >> 1;
            while bit_mask != 0 {
                let bit: u8 = if (bit_mask & v_orig) != 0 { 1 } else { 0 };
                self.encode(StatRef::Dc(dc_tbl, mag_st), bit);
                bit_mask >>= 1;
            }
        } else {
            self.encode(StatRef::Dc(dc_tbl, st), 0);
            let l_thresh: i32 = (1i32 << self.arith_dc_l[dc_tbl]) >> 1;
            if m < l_thresh {
                self.dc_context[comp_idx] = 0;
            }
        }
    }

    /// Encode DC coefficient for successive approximation refinement scan (Ah!=0).
    ///
    /// Simply emits the Al'th bit of the DC coefficient using fixed probability.
    /// Ported from jcarith.c encode_mcu_DC_refine.
    pub fn encode_dc_refine(&mut self, block: &[i16; 64], al: u8) {
        let bit: u8 = ((block[0] >> al) & 1) as u8;
        self.encode(StatRef::Fixed(0), bit);
    }

    /// Encode AC coefficients for first progressive scan (AC first, Ah=0).
    ///
    /// Encodes AC coefficients in spectral range [ss, se] with point transform
    /// shift by `al`. Ported from jcarith.c encode_mcu_AC_first.
    pub fn encode_ac_first(&mut self, block: &[i16; 64], ac_tbl: usize, ss: u8, se: u8, al: u8) {
        let ss_idx: usize = ss as usize;
        let se_idx: usize = se as usize;

        // Establish EOB index: find highest nonzero coefficient after shift
        let mut ke: usize = se_idx;
        while ke >= ss_idx {
            let v_raw: i16 = block[ke];
            let v_abs: i32 = if v_raw >= 0 {
                v_raw as i32
            } else {
                -(v_raw as i32)
            };
            if (v_abs >> al) != 0 {
                break;
            }
            if ke == ss_idx {
                // All coefficients are zero after shift — encode EOB and return
                let st: usize = 3 * (ss_idx - 1);
                self.encode(StatRef::Ac(ac_tbl, st), 1);
                return;
            }
            ke -= 1;
        }

        // Encode AC coefficients (Figure F.5)
        let mut k: usize = ss_idx;
        while k <= ke {
            let mut st: usize = 3 * (k - 1);
            self.encode(StatRef::Ac(ac_tbl, st), 0); // EOB decision: not EOB

            // Zero-run with point transform
            loop {
                let v_raw: i16 = block[k];
                let mut v: i32;
                if v_raw >= 0 {
                    v = (v_raw as i32) >> al;
                    if v != 0 {
                        self.encode(StatRef::Ac(ac_tbl, st + 1), 1);
                        self.encode(StatRef::Fixed(0), 0); // positive sign
                        break;
                    }
                } else {
                    v = -(v_raw as i32);
                    v >>= al;
                    if v != 0 {
                        self.encode(StatRef::Ac(ac_tbl, st + 1), 1);
                        self.encode(StatRef::Fixed(0), 1); // negative sign
                        break;
                    }
                }
                self.encode(StatRef::Ac(ac_tbl, st + 1), 0);
                st += 3;
                k += 1;
            }

            // v is the absolute value of the shifted coefficient
            let v_raw: i16 = block[k];
            let mut v: i32 = if v_raw >= 0 {
                (v_raw as i32) >> al
            } else {
                (-(v_raw as i32)) >> al
            };

            st += 2;

            // Magnitude category encoding (Figure F.8)
            let mut m: i32 = 0;
            v -= 1;
            let v_orig: i32 = v;
            if v != 0 {
                self.encode(StatRef::Ac(ac_tbl, st), 1);
                m = 1;
                let mut v2: i32 = v >> 1;
                if v2 != 0 {
                    self.encode(StatRef::Ac(ac_tbl, st), 1);
                    m <<= 1;
                    let kx: usize = self.arith_ac_k[ac_tbl] as usize;
                    let mut st2: usize = if k <= kx { 189 } else { 217 };
                    v2 >>= 1;
                    while v2 != 0 {
                        self.encode(StatRef::Ac(ac_tbl, st2), 1);
                        m <<= 1;
                        st2 += 1;
                        v2 >>= 1;
                    }
                    st = st2;
                }
            }
            self.encode(StatRef::Ac(ac_tbl, st), 0); // magnitude terminator

            // Magnitude bit pattern (Figure F.9) — uses adaptive stats bin st+14
            let mag_st: usize = st + 14;
            let mut bit_mask: i32 = m >> 1;
            while bit_mask != 0 {
                let bit: u8 = if (bit_mask & v_orig) != 0 { 1 } else { 0 };
                self.encode(StatRef::Ac(ac_tbl, mag_st), bit);
                bit_mask >>= 1;
            }

            k += 1;
        }

        // Encode EOB if k <= se
        if k <= se_idx {
            let st: usize = 3 * (k - 1);
            self.encode(StatRef::Ac(ac_tbl, st), 1);
        }
    }

    /// Encode AC coefficients for successive approximation refinement scan (Ah!=0).
    ///
    /// Interleaves correction bits for previously-nonzero coefficients with
    /// newly-significant coefficients. Ported from jcarith.c encode_mcu_AC_refine.
    pub fn encode_ac_refine(
        &mut self,
        block: &[i16; 64],
        ac_tbl: usize,
        ss: u8,
        se: u8,
        al: u8,
        ah: u8,
    ) {
        let ss_idx: usize = ss as usize;
        let se_idx: usize = se as usize;

        // Establish EOB (end-of-block) index for current approximation
        let mut ke: usize = ss_idx;
        let mut found_ke: bool = false;
        for i in (ss_idx..=se_idx).rev() {
            let v_raw: i16 = block[i];
            let v_abs: i32 = if v_raw >= 0 {
                v_raw as i32
            } else {
                -(v_raw as i32)
            };
            if (v_abs >> al) != 0 {
                ke = i;
                found_ke = true;
                break;
            }
        }
        if !found_ke {
            // All zero after current shift — encode EOB
            if ss_idx > 0 {
                let st: usize = 3 * (ss_idx - 1);
                self.encode(StatRef::Ac(ac_tbl, st), 1);
            }
            return;
        }

        // Establish EOBx (previous stage end-of-block) index
        let mut kex: usize = 0;
        for i in (ss_idx..=ke).rev() {
            let v_raw: i16 = block[i];
            let v_abs: i32 = if v_raw >= 0 {
                v_raw as i32
            } else {
                -(v_raw as i32)
            };
            if (v_abs >> ah) != 0 {
                kex = i;
                break;
            }
        }

        // Figure G.10: Encode_AC_Coefficients_SA
        let mut k: usize = ss_idx;
        while k <= ke {
            let st: usize = 3 * (k - 1);
            if k > kex {
                self.encode(StatRef::Ac(ac_tbl, st), 0); // EOB decision
            }
            loop {
                let v_raw: i16 = block[k];
                if v_raw >= 0 {
                    let v_shifted: i32 = (v_raw as i32) >> al;
                    if v_shifted != 0 {
                        if v_shifted >> 1 != 0 {
                            // Previously nonzero: emit correction bit
                            let st2: usize = 3 * (k - 1) + 2;
                            self.encode(StatRef::Ac(ac_tbl, st2), (v_shifted & 1) as u8);
                        } else {
                            // Newly nonzero
                            let st2: usize = 3 * (k - 1) + 1;
                            self.encode(StatRef::Ac(ac_tbl, st2), 1);
                            self.encode(StatRef::Fixed(0), 0); // positive sign
                        }
                        break;
                    }
                } else {
                    let v_abs: i32 = (-(v_raw as i32)) >> al;
                    if v_abs != 0 {
                        if v_abs >> 1 != 0 {
                            // Previously nonzero: emit correction bit
                            let st2: usize = 3 * (k - 1) + 2;
                            self.encode(StatRef::Ac(ac_tbl, st2), (v_abs & 1) as u8);
                        } else {
                            // Newly nonzero
                            let st2: usize = 3 * (k - 1) + 1;
                            self.encode(StatRef::Ac(ac_tbl, st2), 1);
                            self.encode(StatRef::Fixed(0), 1); // negative sign
                        }
                        break;
                    }
                }
                // Zero coefficient — encode zero run
                let st2: usize = 3 * (k - 1) + 1;
                self.encode(StatRef::Ac(ac_tbl, st2), 0);
                k += 1;
            }
            k += 1;
        }

        // Encode EOB if k <= se
        if k <= se_idx {
            let st: usize = 3 * (k - 1);
            self.encode(StatRef::Ac(ac_tbl, st), 1);
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

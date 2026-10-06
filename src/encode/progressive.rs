// libjpeg-turbo-rs: alloc prelude (no_std support, issue #356)
#[allow(unused_imports)]
use alloc::vec::Vec;
#[allow(unused_imports)]
use alloc::{format, vec};

use crate::common::error::{JpegError, Result};
use crate::common::types::ScanScript;
/// Progressive JPEG scan script generation and encoding.
///
/// Generates a simple progressive scan order following libjpeg-turbo's
/// default progression (from jcparam.c simple_progression).
/// Description of one progressive scan.
#[derive(Debug, Clone)]
pub struct ProgressiveScan {
    /// Component indices in this scan (0-based).
    pub component_indices: Vec<usize>,
    /// Spectral selection start (0 = DC).
    pub ss: u8,
    /// Spectral selection end.
    pub se: u8,
    /// Successive approximation high bit (0 = first scan for this band).
    pub ah: u8,
    /// Successive approximation low bit.
    pub al: u8,
}

/// The highest `Ah`/`Al` C allows for 8-bit data (`jcmaster.c:349`): an `Al`
/// above 10 drives first-scan DC values out of range.
const MAX_AH_AL_8BIT: u8 = 10;

/// `MAX_COMPS_IN_SCAN` (`jpeglib.h`).
const MAX_COMPS_IN_SCAN: usize = 4;

/// Refuse a caller-supplied progressive scan script that C's `validate_script`
/// (`jcmaster.c:279-439`) would refuse, before any encoding work runs.
///
/// This is the primary guard between `Encoder::scan_script` and the AC kernels
/// (the kernels' safe wrappers also assert the band):
/// the band `ss..=se` becomes `band_len = se - ss + 1`, and the x86_64 SSE2
/// preparation kernels index a `[i16; 64]` block and two `[u16; 64]` outputs
/// with it unchecked (issue #610). The rules are C's progressive-mode rules —
/// the script this API takes always selects progressive coding, so a first
/// entry shaped like C's sequential (`Ss = 0, Se = 63`) or lossless
/// (`Ss != 0, Se = 0`) selector is refused by the same rules rather than
/// reinterpreted.
///
/// `num_components` is the frame's component count. The entry number in the
/// error is 1-based like C's `scanno`; `0` names the whole script.
pub(crate) fn validate_scan_script(script: &[ScanScript], num_components: usize) -> Result<()> {
    let invalid =
        |entry: usize, reason: &'static str| JpegError::InvalidScanScript { entry, reason };
    if script.is_empty() {
        // jcmaster.c:291-292, JERR_BAD_SCAN_SCRIPT with entry 0.
        return Err(invalid(0, "the script has no scans"));
    }
    // Per component and coefficient: -1 until the coefficient is first sent,
    // then the Al it was last sent with (C's `last_bitpos`).
    let mut last_bitpos: Vec<[i16; 64]> = vec![[-1i16; 64]; num_components];
    for (index, entry) in script.iter().enumerate() {
        let scan_number: usize = index + 1;
        let components: &[u8] = &entry.components;
        // jcmaster.c:330-332, JERR_COMPONENT_COUNT.
        if components.is_empty() || components.len() > MAX_COMPS_IN_SCAN {
            return Err(invalid(
                scan_number,
                "a scan carries one to four components",
            ));
        }
        // jcmaster.c:333-340, JERR_BAD_SCAN_SCRIPT.
        for (position, &component) in components.iter().enumerate() {
            if usize::from(component) >= num_components {
                return Err(invalid(scan_number, "component index past the frame"));
            }
            if position > 0 && component <= components[position - 1] {
                return Err(invalid(
                    scan_number,
                    "components must appear in frame order, each once",
                ));
            }
        }
        let (ss, se, ah, al) = (entry.ss, entry.se, entry.ah, entry.al);
        // jcmaster.c:355-357, JERR_BAD_PROG_SCRIPT from here to the loop's end.
        if ss >= 64 || se < ss || se >= 64 || ah > MAX_AH_AL_8BIT || al > MAX_AH_AL_8BIT {
            return Err(invalid(
                scan_number,
                "progression parameters out of range (Ss <= Se <= 63, Ah and Al <= 10)",
            ));
        }
        if ss == 0 {
            if se != 0 {
                return Err(invalid(
                    scan_number,
                    "a DC scan cannot carry AC coefficients",
                ));
            }
        } else if components.len() != 1 {
            return Err(invalid(
                scan_number,
                "an AC scan carries exactly one component",
            ));
        }
        for &component in components {
            let bitpos: &mut [i16; 64] = &mut last_bitpos[usize::from(component)];
            if ss != 0 && bitpos[0] < 0 {
                return Err(invalid(
                    scan_number,
                    "an AC scan precedes its component's DC scan",
                ));
            }
            for last_al in &mut bitpos[usize::from(ss)..=usize::from(se)] {
                if *last_al < 0 {
                    if ah != 0 {
                        return Err(invalid(
                            scan_number,
                            "the first scan of a coefficient must have Ah = 0",
                        ));
                    }
                } else if i16::from(ah) != *last_al || u16::from(al) + 1 != u16::from(ah) {
                    return Err(invalid(
                        scan_number,
                        "a refinement scan needs Ah = the previous Al and Al = Ah - 1",
                    ));
                }
                *last_al = i16::from(al);
            }
        }
    }
    // jcmaster.c:422-430: progressive mode only requires some DC per component.
    if last_bitpos.iter().any(|bitpos: &[i16; 64]| bitpos[0] < 0) {
        return Err(invalid(
            0,
            "the script does not transmit every component's DC",
        ));
    }
    Ok(())
}

/// Generate a simple progressive scan script.
///
/// Follows libjpeg-turbo's default progression (jcparam.c
/// `jpeg_simple_progression`) exactly.
///
/// For 3-component YCbCr images (the common case), uses the optimized
/// YCbCr script (10 scans): chroma gets full band 1-63 in a single scan
/// with Al=1, while luma gets split bands (1-5, 6-63) with Al=2.
///
/// For other component counts, uses the generic all-purpose script.
pub fn simple_progression(num_components: usize) -> Vec<ProgressiveScan> {
    simple_progression_for(num_components, true)
}

/// The scan script C picks, given the component count *and* the colorspace.
///
/// `jpeg_simple_progression` takes the tuned 10-scan script only when
/// `ncomps == 3 && jpeg_color_space == JCS_YCbCr` (`jcparam.c`); every other
/// three-component colorspace gets the 14-scan all-purpose script. The
/// distinction matters because that script's shortcuts are chroma-specific —
/// "chroma data is too small to be worth expending many scans on" is a
/// statement about Cb and Cr, and is simply false of G and B.
pub fn simple_progression_for(num_components: usize, ycbcr: bool) -> Vec<ProgressiveScan> {
    if num_components == 3 && ycbcr {
        ycbcr_progression()
    } else {
        generic_progression(num_components)
    }
}

/// YCbCr-specific progression matching C jcparam.c (10 scans).
///
/// Order follows the C source exactly:
/// 1. DC first (all 3 comps), Al=1
/// 2. Y AC first, band 1-5, Al=2
/// 3. Cr AC first, band 1-63, Al=1
/// 4. Cb AC first, band 1-63, Al=1
/// 5. Y AC first, band 6-63, Al=2
/// 6. Y AC refine, band 1-63, Ah=2, Al=1
/// 7. DC refine (all 3 comps), Ah=1, Al=0
/// 8. Cr AC refine, band 1-63, Ah=1, Al=0
/// 9. Cb AC refine, band 1-63, Ah=1, Al=0
/// 10. Y AC refine, band 1-63, Ah=1, Al=0
fn ycbcr_progression() -> Vec<ProgressiveScan> {
    vec![
        // 1. DC first: all components
        ProgressiveScan {
            component_indices: vec![0, 1, 2],
            ss: 0,
            se: 0,
            ah: 0,
            al: 1,
        },
        // 2. Y AC first: band 1-5
        ProgressiveScan {
            component_indices: vec![0],
            ss: 1,
            se: 5,
            ah: 0,
            al: 2,
        },
        // 3. Cr AC first: full band, Al=1 (chroma is small, one scan suffices)
        ProgressiveScan {
            component_indices: vec![2],
            ss: 1,
            se: 63,
            ah: 0,
            al: 1,
        },
        // 4. Cb AC first: full band, Al=1
        ProgressiveScan {
            component_indices: vec![1],
            ss: 1,
            se: 63,
            ah: 0,
            al: 1,
        },
        // 5. Y AC first: band 6-63
        ProgressiveScan {
            component_indices: vec![0],
            ss: 6,
            se: 63,
            ah: 0,
            al: 2,
        },
        // 6. Y AC refine: Ah=2, Al=1
        ProgressiveScan {
            component_indices: vec![0],
            ss: 1,
            se: 63,
            ah: 2,
            al: 1,
        },
        // 7. DC refine: all components
        ProgressiveScan {
            component_indices: vec![0, 1, 2],
            ss: 0,
            se: 0,
            ah: 1,
            al: 0,
        },
        // 8. Cr AC refine: Ah=1, Al=0
        ProgressiveScan {
            component_indices: vec![2],
            ss: 1,
            se: 63,
            ah: 1,
            al: 0,
        },
        // 9. Cb AC refine: Ah=1, Al=0
        ProgressiveScan {
            component_indices: vec![1],
            ss: 1,
            se: 63,
            ah: 1,
            al: 0,
        },
        // 10. Y AC refine: Ah=1, Al=0 (largest scan, comes last)
        ProgressiveScan {
            component_indices: vec![0],
            ss: 1,
            se: 63,
            ah: 1,
            al: 0,
        },
    ]
}

/// Generic all-purpose progression for non-YCbCr (jcparam.c else branch).
pub(crate) fn generic_progression(num_components: usize) -> Vec<ProgressiveScan> {
    let mut scans = Vec::new();
    let all_comps: Vec<usize> = (0..num_components).collect();

    // DC first scan: all components, Ah=0, Al=1
    scans.push(ProgressiveScan {
        component_indices: all_comps.clone(),
        ss: 0,
        se: 0,
        ah: 0,
        al: 1,
    });

    // AC first scans: per-component, spectral bands, Ah=0, Al=2
    for ci in 0..num_components {
        scans.push(ProgressiveScan {
            component_indices: vec![ci],
            ss: 1,
            se: 5,
            ah: 0,
            al: 2,
        });
    }
    for ci in 0..num_components {
        scans.push(ProgressiveScan {
            component_indices: vec![ci],
            ss: 6,
            se: 63,
            ah: 0,
            al: 2,
        });
    }

    // AC refine: per-component, full band, Ah=2, Al=1
    for ci in 0..num_components {
        scans.push(ProgressiveScan {
            component_indices: vec![ci],
            ss: 1,
            se: 63,
            ah: 2,
            al: 1,
        });
    }

    // DC refine: all components, Ah=1, Al=0
    scans.push(ProgressiveScan {
        component_indices: all_comps,
        ss: 0,
        se: 0,
        ah: 1,
        al: 0,
    });

    // AC refine: per-component, full band, Ah=1, Al=0
    for ci in 0..num_components {
        scans.push(ProgressiveScan {
            component_indices: vec![ci],
            ss: 1,
            se: 63,
            ah: 1,
            al: 0,
        });
    }

    scans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_progression_grayscale() {
        let scans = simple_progression(1);
        // Generic: 1 DC first + 2 AC first + 1 AC refine + 1 DC refine + 1 AC refine = 6
        assert_eq!(scans.len(), 6);
        assert_eq!(scans[0].ss, 0);
        assert_eq!(scans[0].se, 0);
        assert_eq!(scans[0].ah, 0);
        assert_eq!(scans[0].al, 1);
        assert_eq!(scans[1].al, 2);
        assert_eq!(scans[3].ah, 2);
        assert_eq!(scans[3].al, 1);
        assert_eq!(scans[5].ah, 1);
        assert_eq!(scans[5].al, 0);
    }

    #[test]
    fn simple_progression_3_components_ycbcr() {
        let scans = simple_progression(3);
        // YCbCr-specific: 10 scans matching C jcparam.c
        assert_eq!(scans.len(), 10);
        // 1. DC first: all 3 comps
        assert_eq!(scans[0].component_indices, vec![0, 1, 2]);
        assert_eq!(scans[0].ss, 0);
        assert_eq!(scans[0].al, 1);
        // 2. Y AC first: band 1-5
        assert_eq!(scans[1].component_indices, vec![0]);
        assert_eq!(scans[1].se, 5);
        assert_eq!(scans[1].al, 2);
        // 3. Cr AC first: full band, Al=1
        assert_eq!(scans[2].component_indices, vec![2]);
        assert_eq!(scans[2].se, 63);
        assert_eq!(scans[2].al, 1);
        // 4. Cb AC first: full band, Al=1
        assert_eq!(scans[3].component_indices, vec![1]);
        assert_eq!(scans[3].al, 1);
        // 10. Y AC refine (last, largest scan)
        let last = &scans[9];
        assert_eq!(last.component_indices, vec![0]);
        assert_eq!(last.ah, 1);
        assert_eq!(last.al, 0);
    }

    #[test]
    fn generic_progression_3_components_rgb() {
        let scans = generic_progression(3);
        assert_eq!(scans.len(), 14);
        assert_eq!(scans[0].component_indices, vec![0, 1, 2]);
        for (component, scan) in scans[1..4].iter().enumerate() {
            assert_eq!(scan.component_indices, vec![component]);
            assert_eq!((scan.ss, scan.se, scan.ah, scan.al), (1, 5, 0, 2));
        }
        for (component, scan) in scans[4..7].iter().enumerate() {
            assert_eq!(scan.component_indices, vec![component]);
            assert_eq!((scan.ss, scan.se, scan.ah, scan.al), (6, 63, 0, 2));
        }
    }
}

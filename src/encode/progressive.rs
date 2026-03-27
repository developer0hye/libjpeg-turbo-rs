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

/// Generate a simple progressive scan script.
///
/// Follows libjpeg-turbo's default progression (jcparam.c `jpeg_simple_progression`).
/// Uses successive approximation for both DC and AC coefficients:
///
/// 1. DC first (all components interleaved), Al=1
/// 2. AC first scans per-component for bands (1-5, 6-63), Al=2
/// 3. AC refine scans per-component (1-63), Ah=2, Al=1
/// 4. DC refine (all components), Ah=1, Al=0
/// 5. AC refine scans per-component (1-63), Ah=1, Al=0
pub fn simple_progression(num_components: usize) -> Vec<ProgressiveScan> {
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
        // 1 DC first + 2 AC first + 1 AC refine + 1 DC refine + 1 AC refine = 6
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
    fn simple_progression_3_components() {
        let scans = simple_progression(3);
        // 1 DC first + 6 AC first + 3 AC refine + 1 DC refine + 3 AC refine = 14
        assert_eq!(scans.len(), 14);
        assert_eq!(scans[0].ss, 0);
        assert_eq!(scans[0].se, 0);
        assert_eq!(scans[0].component_indices.len(), 3);
        assert_eq!(scans[0].al, 1);
        let last = &scans[13];
        assert_eq!(last.ah, 1);
        assert_eq!(last.al, 0);
        assert_eq!(last.component_indices, vec![2]);
    }
}

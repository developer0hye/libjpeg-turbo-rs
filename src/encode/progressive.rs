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
/// Follows libjpeg-turbo's default progression for 1 or 3 components:
/// 1. DC first (all components interleaved), Al=1
/// 2. AC scans per-component for spectral bands (1-5, 6-63)
/// 3. DC refine (all components), Al=0
/// 4. AC refine scans per-component
pub fn simple_progression(num_components: usize) -> Vec<ProgressiveScan> {
    let mut scans = Vec::new();

    if num_components == 1 {
        // Grayscale: DC successive approximation, AC full precision per band
        let comp = vec![0];

        // DC first, Al=1
        scans.push(ProgressiveScan {
            component_indices: comp.clone(),
            ss: 0,
            se: 0,
            ah: 0,
            al: 1,
        });

        // AC 1-5, Al=0
        scans.push(ProgressiveScan {
            component_indices: comp.clone(),
            ss: 1,
            se: 5,
            ah: 0,
            al: 0,
        });

        // AC 6-63, Al=0
        scans.push(ProgressiveScan {
            component_indices: comp.clone(),
            ss: 6,
            se: 63,
            ah: 0,
            al: 0,
        });

        // DC refine, Al=0
        scans.push(ProgressiveScan {
            component_indices: comp,
            ss: 0,
            se: 0,
            ah: 1,
            al: 0,
        });
    } else {
        // Color: interleaved DC, per-component AC
        // Uses DC successive approximation (first + refine) but no AC successive
        // approximation refine scans. AC refine encoding is complex and the
        // standard Huffman tables lack EOBRUN symbols needed for proper batching.
        let all_comps: Vec<usize> = (0..num_components).collect();

        // DC first scan: all components, Al=1
        scans.push(ProgressiveScan {
            component_indices: all_comps.clone(),
            ss: 0,
            se: 0,
            ah: 0,
            al: 1,
        });

        // AC scans: per-component, spectral bands
        for ci in 0..num_components {
            // AC 1-5, Al=0 (full precision, no refine needed)
            scans.push(ProgressiveScan {
                component_indices: vec![ci],
                ss: 1,
                se: 5,
                ah: 0,
                al: 0,
            });
        }

        for ci in 0..num_components {
            // AC 6-63, Al=0 (full precision, no refine needed)
            scans.push(ProgressiveScan {
                component_indices: vec![ci],
                ss: 6,
                se: 63,
                ah: 0,
                al: 0,
            });
        }

        // DC refine: all components, Al=0
        scans.push(ProgressiveScan {
            component_indices: all_comps,
            ss: 0,
            se: 0,
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
        assert!(scans.len() >= 4);
        // First scan should be DC
        assert_eq!(scans[0].ss, 0);
        assert_eq!(scans[0].se, 0);
    }

    #[test]
    fn simple_progression_3_components() {
        let scans = simple_progression(3);
        assert!(scans.len() >= 8);
        // First scan: DC, all components, Al=1
        assert_eq!(scans[0].ss, 0);
        assert_eq!(scans[0].se, 0);
        assert_eq!(scans[0].component_indices.len(), 3);
        assert_eq!(scans[0].al, 1);
    }
}

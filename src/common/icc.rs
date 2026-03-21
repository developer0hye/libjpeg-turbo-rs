use crate::common::types::IccChunk;

/// Reassemble an ICC profile from APP2 marker chunks.
///
/// Validates that all chunks report the same `num_markers`, sequence numbers
/// are contiguous from 1 to `num_markers`, and there are no duplicates.
/// Returns `None` if the chunks are empty, invalid, or incomplete.
pub fn reassemble_icc_profile(chunks: &[IccChunk]) -> Option<Vec<u8>> {
    if chunks.is_empty() {
        return None;
    }

    let num_markers = chunks[0].num_markers;
    if num_markers == 0 {
        return None;
    }

    // All chunks must agree on the total count
    if chunks.iter().any(|c| c.num_markers != num_markers) {
        return None;
    }

    // Check for valid seq_no range and no duplicates
    let mut seen = vec![false; num_markers as usize];
    for chunk in chunks {
        if chunk.seq_no == 0 || chunk.seq_no > num_markers {
            return None;
        }
        let idx = (chunk.seq_no - 1) as usize;
        if seen[idx] {
            return None; // duplicate
        }
        seen[idx] = true;
    }

    // Check no gaps
    if seen.iter().any(|&s| !s) {
        return None;
    }

    // Reassemble in sequence order
    let mut sorted: Vec<&IccChunk> = chunks.iter().collect();
    sorted.sort_by_key(|c| c.seq_no);

    let total_len: usize = sorted.iter().map(|c| c.data.len()).sum();
    let mut profile = Vec::with_capacity(total_len);
    for chunk in sorted {
        profile.extend_from_slice(&chunk.data);
    }

    Some(profile)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chunk(seq_no: u8, num_markers: u8, data: &[u8]) -> IccChunk {
        IccChunk {
            seq_no,
            num_markers,
            data: data.to_vec(),
        }
    }

    #[test]
    fn single_chunk_profile() {
        let chunks = vec![make_chunk(1, 1, &[0x00, 0x01, 0x02, 0x03])];
        let profile = reassemble_icc_profile(&chunks).unwrap();
        assert_eq!(profile, vec![0x00, 0x01, 0x02, 0x03]);
    }

    #[test]
    fn multi_chunk_reassembles_in_seq_order() {
        // Chunks arrive out of order
        let chunks = vec![
            make_chunk(2, 3, &[0x04, 0x05]),
            make_chunk(1, 3, &[0x01, 0x02, 0x03]),
            make_chunk(3, 3, &[0x06]),
        ];
        let profile = reassemble_icc_profile(&chunks).unwrap();
        assert_eq!(profile, vec![0x01, 0x02, 0x03, 0x04, 0x05, 0x06]);
    }

    #[test]
    fn empty_chunks_returns_none() {
        assert!(reassemble_icc_profile(&[]).is_none());
    }

    #[test]
    fn duplicate_seq_no_returns_none() {
        let chunks = vec![make_chunk(1, 2, &[0x01]), make_chunk(1, 2, &[0x02])];
        assert!(reassemble_icc_profile(&chunks).is_none());
    }

    #[test]
    fn gap_in_seq_returns_none() {
        // seq 1 and 3, missing 2
        let chunks = vec![make_chunk(1, 3, &[0x01]), make_chunk(3, 3, &[0x03])];
        assert!(reassemble_icc_profile(&chunks).is_none());
    }

    #[test]
    fn inconsistent_num_markers_returns_none() {
        let chunks = vec![make_chunk(1, 2, &[0x01]), make_chunk(2, 3, &[0x02])];
        assert!(reassemble_icc_profile(&chunks).is_none());
    }

    #[test]
    fn seq_no_zero_returns_none() {
        let chunks = vec![make_chunk(0, 1, &[0x01])];
        assert!(reassemble_icc_profile(&chunks).is_none());
    }

    #[test]
    fn seq_no_exceeds_num_markers_returns_none() {
        let chunks = vec![make_chunk(2, 1, &[0x01])];
        assert!(reassemble_icc_profile(&chunks).is_none());
    }

    #[test]
    fn num_markers_zero_returns_none() {
        let chunks = vec![make_chunk(1, 0, &[0x01])];
        assert!(reassemble_icc_profile(&chunks).is_none());
    }
}

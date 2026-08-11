// libjpeg-turbo-rs: alloc prelude (no_std support, issue #356)
use crate::common::error::{JpegError, Result};
use crate::common::try_alloc::try_reserved_vec;
use crate::common::types::IccChunk;
#[allow(unused_imports)]
use alloc::vec::Vec;
#[allow(unused_imports)]
use alloc::{format, vec};

/// Reassemble an ICC profile from APP2 marker chunks.
///
/// Validates that all chunks report the same `num_markers`, sequence numbers
/// are contiguous from 1 to `num_markers`, and there are no duplicates.
///
/// Returns `None` if the chunks are empty, invalid or incomplete. See
/// [`try_reassemble_icc_profile`] for the form that reports an allocation
/// refusal instead of folding it into `None`.
pub fn reassemble_icc_profile(chunks: &[IccChunk]) -> Option<Vec<u8>> {
    // Public since before P4-144, so the signature stays: `common` is exported
    // from the crate root and a `Result` here would break every downstream
    // caller for a change they did not ask for. Allocation refusal folds into
    // `None`, which is still an improvement — this function used to abort the
    // process instead.
    try_reassemble_icc_profile(chunks).ok().flatten()
}

/// The fallible form, for callers that can report the refusal.
///
/// `Ok(None)` is a malformed or absent profile — a soft outcome, since a broken
/// ICC profile must not fail an otherwise-valid decode. `Err` is the allocator
/// refusing the reassembly buffer, which is a different thing entirely and used
/// to abort the process: the profile is the sum of up to 255 APP2 segments, so
/// this is an input-sized allocation and was the last one on the decode path
/// that could not be caught (P4-144).
pub fn try_reassemble_icc_profile(chunks: &[IccChunk]) -> Result<Option<Vec<u8>>> {
    if chunks.is_empty() {
        return Ok(None);
    }

    let num_markers = chunks[0].num_markers;
    if num_markers == 0 {
        return Ok(None);
    }

    // All chunks must agree on the total count
    if chunks.iter().any(|c| c.num_markers != num_markers) {
        return Ok(None);
    }

    // Check for valid seq_no range and no duplicates.
    //
    // Fixed storage rather than `vec![false; n]`: `num_markers` is a `u8`, so
    // 256 slots covers every possible value, and an allocation here — however
    // small — would abort before either fallible reservation below is reached,
    // contradicting this function's whole contract.
    let mut seen: [bool; 256] = [false; 256];
    for chunk in chunks {
        if chunk.seq_no == 0 || chunk.seq_no > num_markers {
            return Ok(None);
        }
        let idx = (chunk.seq_no - 1) as usize;
        if seen[idx] {
            return Ok(None); // duplicate
        }
        seen[idx] = true;
    }

    // Check no gaps. Only the first `num_markers` slots are in play — the
    // array is fixed at 256 so it never allocates, not because every slot is
    // meaningful.
    if seen[..num_markers as usize].iter().any(|&s| !s) {
        return Ok(None);
    }

    // Reassemble in sequence order. Both the list and the sort are
    // allocations: `collect` grows infallibly, and `sort_by_key` is stable, so
    // it reserves scratch. Sequence numbers are unique here — duplicates were
    // rejected above — so the unstable sort is equivalent and allocates
    // nothing (P4-144).
    let mut sorted: Vec<&IccChunk> = Vec::new();
    sorted
        .try_reserve_exact(chunks.len())
        .map_err(|_| JpegError::AllocationFailed {
            what: "ICC chunk list",
            bytes: ((chunks.len() as u128) * (core::mem::size_of::<&IccChunk>() as u128))
                .min(u64::MAX as u128) as u64,
        })?;
    sorted.extend(chunks.iter());
    sorted.sort_unstable_by_key(|c| c.seq_no);

    let total_len: usize = sorted.iter().map(|c| c.data.len()).sum();
    let mut profile: Vec<u8> = try_reserved_vec(total_len, "ICC profile")?;
    for chunk in sorted {
        profile.extend_from_slice(&chunk.data);
    }

    Ok(Some(profile))
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
        let profile = reassemble_icc_profile(&chunks).expect("valid");
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
        let profile = reassemble_icc_profile(&chunks).expect("valid");
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

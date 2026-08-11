//! Fallible allocation helpers for input-derived buffers.
//!
//! `vec![0u8; n]`, `Vec::with_capacity(n)` and `slice.to_vec()` **abort the
//! process** when the allocator refuses. For a size that came out of a JPEG
//! stream that turns a hostile or merely large file into an uncatchable denial
//! of service, so every such size goes through `try_reserve_exact` here and
//! surfaces refusal as [`JpegError::AllocationFailed`] (P4-136, P4-144).
//!
//! These lived in `api::progressive_output` until P4-144 needed the same three
//! in `common::icc` and `decode::marker`. Copying them would have been the
//! `compress_*` family P4-40 was filed for, one file later.

// libjpeg-turbo-rs: alloc prelude (no_std support, issue #356)
#[allow(unused_imports)]
use alloc::vec::Vec;

use crate::common::error::{JpegError, Result};

/// Allocate `len` copies of `value`, reporting allocator refusal as an error.
///
/// `vec![value; len]` aborts the process when the allocator refuses, so a
/// header asking for more memory than the machine has becomes an
/// uncatchable denial of service. Every size here is derived from the JPEG
/// stream, so refusal has to be a recoverable [`JpegError`] like the other
/// limits this decoder enforces (P4-136 criterion 4).
///
/// The byte count is checked separately from `try_reserve_exact` so that a
/// `len` too large to *express* in bytes reports the geometry limit that
/// caused it rather than being reported as a failed allocation the machine
/// was never asked to perform. This is the 32-bit hazard: `blocks_x *
/// blocks_y` fits `usize` on a 32-bit target while `* 128` for the
/// coefficient buffers does not (P4-136 criterion 5).
pub(crate) fn try_filled_vec<T: Clone>(len: usize, value: T, what: &'static str) -> Result<Vec<T>> {
    let elem_size: usize = core::mem::size_of::<T>();
    let bytes: usize = len
        .checked_mul(elem_size)
        .filter(|total| *total <= isize::MAX as usize)
        .ok_or(JpegError::LimitExceeded {
            what,
            // Widen rather than saturate. This product is a *diagnostic*,
            // never a span — but P4-139 is removing `saturating_*` from this
            // crate's size arithmetic wholesale, and a saturating call sitting
            // three lines from a real span computation is precisely what a
            // grep-based gate cannot tell apart. `u128` cannot overflow for
            // any `usize` times any element size.
            actual: ((len as u128) * (elem_size as u128)).min(u64::MAX as u128) as u64,
            limit: isize::MAX as u64,
        })?;

    let mut buf: Vec<T> = Vec::new();
    buf.try_reserve_exact(len)
        .map_err(|_| JpegError::AllocationFailed {
            what,
            bytes: bytes as u64,
        })?;
    // Capacity is already reserved, so this fills in place and cannot
    // reallocate — the abort path `vec![]` would have taken is gone.
    buf.resize(len, value);
    Ok(buf)
}

/// Empty `Vec<u8>` with exactly `len` bytes reserved — the fallible
/// counterpart to `Vec::with_capacity` for buffers filled by `extend_*`
/// rather than by indexing. Same rationale as [`try_filled_vec`], without
/// paying for a zero-fill the caller immediately overwrites.
///
/// `len` is bytes, so no element-size product can overflow here; callers
/// still owe `checked_plane_size` on whatever geometry produced `len`.
pub(crate) fn try_reserved_vec(len: usize, what: &'static str) -> Result<Vec<u8>> {
    let mut buf: Vec<u8> = Vec::new();
    buf.try_reserve_exact(len)
        .map_err(|_| JpegError::AllocationFailed {
            what,
            bytes: len as u64,
        })?;
    Ok(buf)
}

/// Fallible copy of an input slice.
pub(crate) fn try_copy_of(data: &[u8], what: &'static str) -> Result<Vec<u8>> {
    let mut buf: Vec<u8> = try_reserved_vec(data.len(), what)?;
    buf.extend_from_slice(data);
    Ok(buf)
}

/// Fallible clone of an optional metadata blob.
///
/// `Option<Vec<u8>>::clone` allocates infallibly, and these blobs are
/// input-sized: EXIF, XMP, IPTC and a reassembled ICC profile are each as
/// large as the file chose to make them. They do not *amplify* the way a
/// geometry-derived buffer does — the caller already holds the bytes — but the
/// clone still doubles the peak for that blob, and an abort there is
/// unrecoverable where an error is not (P4-144).
pub(crate) fn try_clone_opt(data: &Option<Vec<u8>>, what: &'static str) -> Result<Option<Vec<u8>>> {
    match data {
        Some(bytes) => Ok(Some(try_copy_of(bytes, what)?)),
        None => Ok(None),
    }
}

/// Fallible clone of an optional text blob (a COM comment).
///
/// Same reasoning as [`try_clone_opt`]: a marker segment is at most 64 KiB,
/// but "small" is not "zero", and `String::clone` aborts on refusal like every
/// other infallible allocation.
pub(crate) fn try_clone_opt_string(
    data: &Option<alloc::string::String>,
    what: &'static str,
) -> Result<Option<alloc::string::String>> {
    match data {
        Some(text) => {
            let bytes: Vec<u8> = try_copy_of(text.as_bytes(), what)?;
            // The source was valid UTF-8 and this is a byte-for-byte copy.
            Ok(Some(alloc::string::String::from_utf8(bytes).map_err(
                |_| JpegError::CorruptData(alloc::format!("{what}: invalid UTF-8")),
            )?))
        }
        None => Ok(None),
    }
}

/// Fallible deep clone of the saved-marker list.
///
/// Both the outer `Vec` and each marker's payload are input-sized, so both go
/// through `try_reserve_exact` rather than aborting on refusal (P4-144).
pub(crate) fn try_clone_saved_markers(
    markers: &[crate::common::types::SavedMarker],
) -> Result<Vec<crate::common::types::SavedMarker>> {
    use crate::common::types::SavedMarker;

    let mut out: Vec<SavedMarker> = Vec::new();
    out.try_reserve_exact(markers.len())
        .map_err(|_| JpegError::AllocationFailed {
            what: "saved marker list",
            // A diagnostic, not a span: `u128` so the product cannot wrap for
            // any `len` and any element size (P4-139 keeps `saturating_*` out
            // of this crate's size arithmetic).
            bytes: ((markers.len() as u128) * (core::mem::size_of::<SavedMarker>() as u128))
                .min(u64::MAX as u128) as u64,
        })?;
    for m in markers {
        out.push(SavedMarker {
            code: m.code,
            data: try_copy_of(&m.data, "saved marker payload")?,
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::types::IccChunk;

    /// The whole point of the module: refusal is a value, not an abort.
    ///
    /// `#[cfg_attr(miri, ignore)]` because Miri reports an unservable request
    /// as resource exhaustion and kills the interpreter rather than returning
    /// null, so the probe cannot observe the error there. Under ASan the same
    /// request aborts with `allocation-size-too-big` unless
    /// `allocator_may_return_null=1`, which `sanitizers.yml` sets (P4-136).
    ///
    /// 64-bit only: on a 32-bit target `isize::MAX` is ~2 GiB, which the
    /// allocator may legitimately serve — the probe would then either fail
    /// `expect_err` or actually reserve two gigabytes. The analogous P4-136
    /// test is gated for the same reason.
    #[test]
    #[cfg(target_pointer_width = "64")]
    #[cfg_attr(miri, ignore = "Miri aborts on an unservable request; see P4-136")]
    fn reserving_more_than_the_machine_can_serve_is_an_error() {
        let err: JpegError =
            try_reserved_vec(isize::MAX as usize, "test buffer").expect_err("must refuse");
        assert!(
            matches!(err, JpegError::AllocationFailed { what, bytes }
                if what == "test buffer" && bytes == isize::MAX as u64),
            "expected AllocationFailed, got {err:?}"
        );
    }

    #[test]
    fn cloning_none_allocates_nothing() {
        assert_eq!(try_clone_opt(&None, "absent").expect("no allocation"), None);
    }

    #[test]
    fn cloning_some_copies_the_bytes() {
        let src: Option<Vec<u8>> = Some(vec![1u8, 2, 3, 4]);
        let copy: Option<Vec<u8>> = try_clone_opt(&src, "present").expect("small clone");
        assert_eq!(copy.as_deref(), Some(&[1u8, 2, 3, 4][..]));
    }

    fn chunk(seq: u8, total: u8, data: &[u8]) -> IccChunk {
        IccChunk {
            seq_no: seq,
            num_markers: total,
            data: data.to_vec(),
        }
    }

    /// A malformed profile stays a *soft* outcome. Conflating it with an
    /// allocation failure would turn "this file's ICC is broken" into "this
    /// decode failed", which is the opposite of what the metadata path
    /// promises.
    #[test]
    fn invalid_chunks_are_ok_none_not_an_error() {
        use crate::common::icc::try_reassemble_icc_profile;

        assert_eq!(
            try_reassemble_icc_profile(&[]).expect("empty is soft"),
            None
        );
        // A gap: chunk 2 of 2 present, chunk 1 missing.
        let gap: Vec<IccChunk> = vec![chunk(2, 2, b"tail")];
        assert_eq!(try_reassemble_icc_profile(&gap).expect("gap is soft"), None);
        // A duplicate sequence number.
        let dup: Vec<IccChunk> = vec![chunk(1, 2, b"a"), chunk(1, 2, b"b")];
        assert_eq!(try_reassemble_icc_profile(&dup).expect("dup is soft"), None);
    }

    #[test]
    fn a_complete_profile_reassembles_in_sequence_order() {
        use crate::common::icc::try_reassemble_icc_profile;

        let chunks: Vec<IccChunk> = vec![chunk(2, 2, b"world"), chunk(1, 2, b"hello ")];
        let profile: Vec<u8> = try_reassemble_icc_profile(&chunks)
            .expect("allocation")
            .expect("valid");
        assert_eq!(profile, b"hello world");
    }
}

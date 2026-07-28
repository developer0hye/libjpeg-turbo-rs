//! Regression test for fractional chroma upsample ratios.
//!
//! Found by scheduled Fuzz Smoke `fuzz_decompress_lenient` run 29977722126
//! (2026-07-23, crash-4a2b926c): a baseline SOF with Y=4x1, Cb=3x1, Cr=1x1
//! gives a luma/chroma width ratio of 4/3. The upsample-factor computation
//! truncates that to 1 ("no upsampling needed"), so the direct-copy path
//! read luma-width rows out of a 3/4-width chroma plane and panicked with
//! a slice-bounds failure. C libjpeg-turbo rejects every non-integer
//! sampling ratio up front with ERREXIT(JERR_FRACT_SAMPLE_NOTIMPL)
//! ("Fractional sampling not implemented yet", verified: djpeg exits 1) —
//! match it with a clean `Unsupported` error in both strict and lenient
//! modes.
//!
//! Tracked as P4-36 in docs/last_mile/phase4.md.

mod helpers;

use std::path::PathBuf;

use libjpeg_turbo_rs::{Decoder, JpegError};

fn decode_hex(s: &str) -> Vec<u8> {
    let compact: String = s.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    assert!(compact.len().is_multiple_of(2));
    (0..compact.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&compact[i..i + 2], 16).expect("valid hex byte"))
        .collect()
}

/// Fuzz fixture from crash-4a2b926c (run 29977722126): 64x64 baseline,
/// Y=4x1 Cb=3x1 Cr=1x1 → fractional 4/3 luma/chroma width ratio.
fn fixture_fractional_sampling() -> Vec<u8> {
    decode_hex(
        r#"
        ffd8ffe000104a46494600010100000100010000ffdb0043000302020302
        020303030304030304050805050404050a070706080c0a0c0c0b0a0b0b0d
        0e12100d0e110e0b0b1016101113141515150c0f171816141812141514ff
        db0043010b040405040509050509140d0b0d141414141414141414141414
        141414141414143b2500000014141414141414141414141414ffec141414
        14ffd91414141414ffc00011080040004003014100023101031101ffc400
        1f0000010501010101010100000000000000000102030405060708090a0b
        ffc400b5100002010303020403050504040000017d010203000411ffdc21
        3141061351010000000000012fa1082360b1c11552d1f02433627282090a
        161718191a2526272816141812141514ffca000b080010001001011100ff
        cc00040010ffda0008010100000001c1ffcc00041005ffda000c01010001
        05025aa43e6ca613bcffcc00041085ffda0008010100263f0250dedef326
        9667b54a5fb7369c6bdbe5fa6599084a8fbdcf87a5232b732dac7cffcc00
        041005ffda0008010100013f21000000000000000000000000008b000001
        00002cbbe36aab6dc9ff000a1bfea64ffc91ff00ed947fc96eff00a8f23f
        002323232323232323232323232323232323232323232323232323232323
        232323232323232323232323232323232323232323232323232323232323
        232323232323232323232323232323232323002ff647fdbcf9be6ffdf1b7
        1e57be73db1ca7f0a1bfea64ff00c91ffed94007fc96effa82ff00647fdb
        5d03000000000000cf9be6ff00df1b71e57be73db1c9ff000a1bfea64ffc
        91ff00e588181d7233721829c73fca31016874009d553300002cbbe36aab
        6de588181d72377218293fca31016874009d5533c70850e64c8dbd0156ce
        41fbdd46781f31000039ea47a56845a2b8924b88513ed2c0932ac4d9752c
        c5b7e39c7371efd8f3c1ae925d136c2716507fa140ee0795858931f1b71e
        57be73db1ca7f0e0ff0000abaaaaaaaaaaaa0aa1bfea64ff00c91ffe2a00
        0000ffffffffffffffffffffffffffffffffffffffffffffffffffffffff
        ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff
        ffffffffffffffffffffffffffffffffffffd94007fc96effa82ff00647f
        dbcf9be6ff00df1b71e57be73db1fc96effac2ff00647fdbcf9be6ff00df
        1b71e585133db1c9ff000a0100000000000000ed94be6ffdf1b71e570100
        000000000a2fbe73db1ca7f0a1bfea64ff00f21ffed94007fc96effa82ff
        00647fdbcf9be6ff00df1b71e57be700103db1c9ff000a1bfea64ffc91bc
        f9be6f3afdf1b410a002790011c7dcf7ddffffffffffffffff5e5e5e5e5e
        5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e
        5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e
        5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e5e
        5e5e5e5e5e5e5e5e5e5e5e5e5e5e5effffffffffffffffffff7effffffff
        ffd99edf9f5d8c14b0325affed6e88960b92921dcc1fa1055b9c1c719cf7
        3904e2b6a02967b752e592cc1fa1055b9c1c719cf73904e2b6a0d267b752
        e59249924cab630b1f185c0dcc4b15c0c9ee4f1dce90b24819167220f283
        346879
        "#,
    )
}

#[test]
fn fractional_sampling_errors_instead_of_panicking_strict() {
    let source: Vec<u8> = fixture_fractional_sampling();
    let decoder = Decoder::new(&source).expect("header parse should succeed");
    let err = decoder.decode_image().expect_err(
        "fractional 4/3 chroma ratio must be rejected like C JERR_FRACT_SAMPLE_NOTIMPL",
    );
    assert!(
        matches!(&err, JpegError::Unsupported(msg) if msg.contains("fractional")),
        "expected the fractional-sampling rejection, got {err:?}"
    );
}

#[test]
fn fractional_sampling_errors_instead_of_panicking_lenient() {
    let source: Vec<u8> = fixture_fractional_sampling();
    let mut decoder = Decoder::new(&source).expect("header parse should succeed");
    decoder.set_lenient(true);
    decoder.set_scan_limit(100);
    // The fuzz_decompress_lenient contract: no panic. djpeg also fails
    // fatally here ("Fractional sampling not implemented yet"), so an
    // error result is the correct lenient outcome too.
    let err = decoder.decode_image().expect_err(
        "djpeg rejects fractional sampling fatally; lenient decode must error, not panic",
    );
    assert!(
        matches!(&err, JpegError::Unsupported(msg) if msg.contains("fractional")),
        "expected the fractional-sampling rejection, got {err:?}"
    );
}

/// C cross-validation: stock djpeg must reject this stream (exit != 0 with
/// the fractional-sampling error), pinning that an error — not a decode —
/// is the correct contract.
#[test]
fn fractional_sampling_rejected_by_c_djpeg() {
    let djpeg: PathBuf = require_c_tool!("djpeg");
    let source: Vec<u8> = fixture_fractional_sampling();
    let jpeg_file = helpers::TempFile::new("fractional_sampling.jpg");
    jpeg_file.write_bytes(&source);
    let output = std::process::Command::new(&djpeg)
        .arg("-pnm")
        .arg(jpeg_file.path())
        .output()
        .expect("djpeg should spawn");
    assert!(
        !output.status.success(),
        "expected djpeg to reject fractional sampling, got exit {:?} (stderr: {})",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
}

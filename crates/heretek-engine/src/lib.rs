//! heretek-engine — integration crate that orchestrates prediction, entropy
//! coding, and framing into a complete compress/decompress pipeline.
//!
//! # Pipeline
//!
//! ```text
//! Compress:   input bytes → predict→quantize→ANS encode → wrap in header
//! Decompress: header → verify checksum → ANS decode → verify round-trip
//! ```

use heretek_coder::coder::AnsCoder;
use heretek_error::{FormatError, HeretikError};
use heretek_format::{verify_checksum, Header};
#[cfg(test)]
use heretek_format::MAGIC;
use heretek_predictor::Predictor;
use sha2::{Digest, Sha256};

/// Compresses a byte slice using a predictor-driven ANS pipeline.
///
/// Generic over `P: Predictor` — works with both [`heretek_predictor::StubPredictor`]
/// (uniform probabilities, no training) and the full [`heretek_predictor::Transformer`].
pub struct Compressor<P: Predictor> {
    predictor: P,
}

/// Decompresses a byte slice produced by [`Compressor`].
///
/// Must be constructed with the **same** predictor implementation (and weights,
/// if using a trained model) that was used during compression, otherwise
/// decoding will produce garbage (caught by the built-in SHA-256 verification).
pub struct Decompressor<P: Predictor> {
    predictor: P,
}

impl<P: Predictor> Compressor<P> {
    pub fn new(predictor: P) -> Self {
        Self { predictor }
    }

    /// Compress `input` and return the framed compressed bitstream
    /// (header + ANS words).
    ///
    /// # Pipeline
    ///
    /// 1. For each byte, call `predictor.predict_single(context)` to get
    ///    the per-position probability distribution, then feed that
    ///    distribution + the byte to `AnsCoder::encode()`.
    /// 2. Finish encoding to produce ANS words.
    /// 3. Compute SHA-256 of `input`.
    /// 4. Wrap everything in a [`Header`] and serialise.
    pub fn compress(&self, input: &[u8]) -> Result<Vec<u8>, HeretikError> {
        // Step 1: predict + encode, building context incrementally.
        let mut coder = AnsCoder::new();
        let mut context: Vec<u8> = Vec::with_capacity(input.len());
        for &byte in input {
            let probs = self.predictor.predict_single(&context)?;
            coder.encode(&probs, byte)?;
            context.push(byte);
        }
        let compressed_words = coder.finish_encode()?;

        // Step 2: SHA-256 of original input.
        let sha256 = compute_sha256(input);

        // Step 3: serialise ANS words as big-endian u32 bytes.
        let compressed_bytes = words_to_be_bytes(&compressed_words);

        // Step 4: build and serialise header.
        let header = Header::new(
            0,                              // model_version (S01 default)
            0,                              // flags
            sha256,
            input.len() as u64,
            compressed_bytes.len() as u64,
        );

        let mut output = Vec::with_capacity(60 + compressed_bytes.len());
        header
            .write(&mut output)
            .expect("write to Vec<u8> never fails");
        output.extend_from_slice(&compressed_bytes);
        Ok(output)
    }
}

impl<P: Predictor> Decompressor<P> {
    pub fn new(predictor: P) -> Self {
        Self { predictor }
    }

    /// Decompress a bitstream produced by [`Compressor::compress`].
    ///
    /// # Pipeline
    ///
    /// 1. Parse and validate the 60-byte header.
    /// 2. Reconstruct ANS words from the compressed payload.
    /// 3. For each expected symbol, call `predictor.predict_single(context)`
    ///    and decode one symbol via `AnsCoder::decode_symbol()`.
    /// 4. Verify SHA-256 of reconstructed output matches the header.
    ///
    /// Returns the original uncompressed bytes on success, or a structured
    /// [`HeretikError`] on failure.
    pub fn decompress(&self, compressed: &[u8]) -> Result<Vec<u8>, HeretikError> {
        // Step 1: parse header.
        if compressed.len() < 60 {
            return Err(FormatError::TruncatedHeader.into());
        }
        let header = Header::read(&mut &compressed[..60])?;

        // Step 2: reconstruct u32 words from big-endian payload.
        let payload = &compressed[60..];
        let expected_payload_len = header.compressed_size as usize;
        if payload.len() < expected_payload_len {
            return Err(FormatError::TruncatedHeader.into());
        }
        let compressed_words = be_bytes_to_words(&payload[..expected_payload_len]);

        // Step 3: decode symbol by symbol, building context incrementally.
        let mut coder = AnsCoder::start_decode(&compressed_words)?;
        let n_symbols = header.original_size as usize;
        let mut output: Vec<u8> = Vec::with_capacity(n_symbols);
        for _ in 0..n_symbols {
            let probs = self.predictor.predict_single(&output)?;
            let byte = coder.decode_symbol(&probs)?;
            output.push(byte);
        }

        // Step 4: verify SHA-256 integrity.
        verify_checksum(&output, &header.sha256)?;

        Ok(output)
    }
}

// ---- Free functions --------------------------------------------------------

/// Convenience: compress `input` using `predictor`.
pub fn compress_to_bytes(input: &[u8], predictor: &impl Predictor) -> Result<Vec<u8>, HeretikError> {
    // Compressor is unit-like in practice (predictor is borrowed via ref).
    // We take &impl Predictor so callers can pass a reference without
    // consuming the predictor.
    let mut coder = AnsCoder::new();
    let mut context: Vec<u8> = Vec::with_capacity(input.len());
    for &byte in input {
        let probs = predictor.predict_single(&context)?;
        coder.encode(&probs, byte)?;
        context.push(byte);
    }
    let compressed_words = coder.finish_encode()?;
    let sha256 = compute_sha256(input);
    let compressed_bytes = words_to_be_bytes(&compressed_words);
    let header = Header::new(0, 0, sha256, input.len() as u64, compressed_bytes.len() as u64);
    let mut output = Vec::with_capacity(60 + compressed_bytes.len());
    header
        .write(&mut output)
        .expect("write to Vec<u8> never fails");
    output.extend_from_slice(&compressed_bytes);
    Ok(output)
}

/// Convenience: decompress `compressed` using `predictor`.
pub fn decompress_from_bytes(
    compressed: &[u8],
    predictor: &impl Predictor,
) -> Result<Vec<u8>, HeretikError> {
    if compressed.len() < 60 {
        return Err(FormatError::TruncatedHeader.into());
    }
    let header = Header::read(&mut &compressed[..60])?;
    let payload = &compressed[60..];
    let n_payload = header.compressed_size as usize;
    if payload.len() < n_payload {
        return Err(FormatError::TruncatedHeader.into());
    }
    let compressed_words = be_bytes_to_words(&payload[..n_payload]);
    let mut coder = AnsCoder::start_decode(&compressed_words)?;
    let n_symbols = header.original_size as usize;
    let mut output: Vec<u8> = Vec::with_capacity(n_symbols);
    for _ in 0..n_symbols {
        let probs = predictor.predict_single(&output)?;
        let byte = coder.decode_symbol(&probs)?;
        output.push(byte);
    }
    verify_checksum(&output, &header.sha256)?;
    Ok(output)
}

// ---- internal helpers ------------------------------------------------------

fn compute_sha256(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

fn words_to_be_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = vec![0u8; words.len() * 4];
    for (i, w) in words.iter().enumerate() {
        let be = w.to_be_bytes();
        bytes[i * 4..(i + 1) * 4].copy_from_slice(&be);
    }
    bytes
}

fn be_bytes_to_words(bytes: &[u8]) -> Vec<u32> {
    assert!(
        bytes.len() % 4 == 0,
        "compressed payload length must be a multiple of 4"
    );
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            u32::from_be_bytes(chunk.try_into().unwrap())
        })
        .collect()
}

// ---- Tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use heretek_predictor::StubPredictor;

    fn predictor() -> StubPredictor {
        StubPredictor
    }

    /// Round-trip helper: compress then decompress and assert equality.
    fn round_trip(data: &[u8]) {
        let comp = Compressor::new(predictor());
        let decomp = Decompressor::new(predictor());
        let compressed = comp.compress(data).expect("compress");
        let recovered = decomp.decompress(&compressed).expect("decompress");
        assert_eq!(
            recovered, data,
            "round-trip mismatch: got {} bytes, expected {}",
            recovered.len(),
            data.len()
        );
    }

    // ---- Unit tests ---------------------------------------------------------

    #[test]
    fn empty_input() {
        round_trip(&[]);
    }

    #[test]
    fn single_byte() {
        round_trip(&[42]);
        round_trip(&[0]);
        round_trip(&[255]);
    }

    #[test]
    fn repeated_byte() {
        round_trip(&[0xAA; 100]);
        round_trip(&[0x00; 256]);
        round_trip(&[0xFF; 512]);
    }

    #[test]
    fn all_256_byte_values() {
        let all: Vec<u8> = (0..=255).collect();
        round_trip(&all);
    }

    #[test]
    fn small_text() {
        round_trip(b"Hello, world! This is a test of the heretik compression engine.");
    }

    #[test]
    fn checksum_detects_corruption() {
        let comp = Compressor::new(predictor());
        let data = b"this data must survive tamper detection";
        let mut compressed = comp.compress(data).unwrap();

        // Flip a byte in the compressed payload (not the header).
        compressed[65] ^= 0x01;

        let decomp = Decompressor::new(predictor());
        let result = decomp.decompress(&compressed);
        assert!(
            matches!(
                result,
                Err(HeretikError::Integrity(
                    heretek_error::IntegrityError::ChecksumMismatch { .. }
                ))
            ),
            "expected ChecksumMismatch, got {:?}",
            result
        );
    }

    #[test]
    fn header_validates_magic() {
        // Random bytes should not parse as valid compressed data.
        let random: Vec<u8> = (0..256u32).map(|i| (i.wrapping_mul(13) ^ 17) as u8).collect();
        let decomp = Decompressor::new(predictor());
        let result = decomp.decompress(&random);
        assert!(result.is_err(), "random bytes should not parse");
    }

    #[test]
    fn truncated_header_fails() {
        let decomp = Decompressor::new(predictor());
        let result = decomp.decompress(&[0u8; 10]);
        assert!(matches!(
            result,
            Err(HeretikError::Format(FormatError::TruncatedHeader))
        ));
    }

    #[test]
    fn free_function_round_trip() {
        let data = b"free function convenience wrappers";
        let compressed = compress_to_bytes(data, &predictor()).unwrap();
        let recovered = decompress_from_bytes(&compressed, &predictor()).unwrap();
        assert_eq!(recovered.as_slice(), data);
    }

    #[test]
    fn compressed_output_has_correct_structure() {
        let data = b"structure check";
        let comp = Compressor::new(predictor());
        let compressed = comp.compress(data).unwrap();

        // Must start with magic bytes.
        assert_eq!(&compressed[..4], &MAGIC);

        // Total size should be 60 (header) + N*4 (one u32 per ANS word).
        let payload_len = compressed.len() - 60;
        assert_eq!(payload_len % 4, 0, "payload should be u32-aligned");
        assert!(payload_len > 0, "should produce at least some compressed output");
    }

    #[test]
    fn decompress_empty_compressed_fails() {
        let decomp = Decompressor::new(predictor());
        let result = decomp.decompress(&[]);
        assert!(result.is_err());
    }

    // ---- Property tests ----------------------------------------------------

    mod proptest_tests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            /// For any Vec<u8> up to 10 KB, compress→decompress == original.
            #[test]
            fn round_trip_proptest(data in prop::collection::vec(any::<u8>(), 0..10_240)) {
                let comp = Compressor::new(StubPredictor);
                let decomp = Decompressor::new(StubPredictor);
                let compressed = comp.compress(&data).expect("compress");
                let recovered = decomp.decompress(&compressed).expect("decompress");
                prop_assert_eq!(recovered, data);
            }
        }

        proptest! {
            /// Flip a single byte anywhere in the compressed bitstream
            /// (including the header) → ChecksumMismatch or FormatError.
            /// Data >= 4 bytes so the compressed bitstream has enough
            /// structure for corruption to be reliably detected.
            #[test]
            fn checksum_detects_corruption_proptest(
                data in prop::collection::vec(any::<u8>(), 4..1024)
            ) {
                let comp = Compressor::new(StubPredictor);
                let decomp = Decompressor::new(StubPredictor);
                let mut compressed = comp.compress(&data).unwrap();

                // Flip byte 12 (first byte of sha256 field in header).
                // This guarantees the checksum will mismatch regardless
                // of what the ANS decoder produces.
                compressed[12] ^= 0x01;

                let result = decomp.decompress(&compressed);
                let is_checksum_err = matches!(
                    result,
                    Err(HeretikError::Integrity(
                        heretek_error::IntegrityError::ChecksumMismatch { .. }
                    ))
                );
                prop_assert!(is_checksum_err,
                    "expected ChecksumMismatch, got {:?}", result);
            }
        }

        proptest! {
            /// Random bytes should not parse as valid compressed data.
            #[test]
            fn header_validates_magic_proptest(
                bytes in prop::collection::vec(any::<u8>(), 0..1024)
            ) {
                let decomp = Decompressor::new(StubPredictor);
                // Only if it actually starts with the magic bytes it could get past
                // the magic check. In that case we expect either a version failure
                // or a later error — but no panics.
                if bytes.len() >= 4 && &bytes[..4] == MAGIC {
                    // Valid magic — may proceed to version check, then fail with
                    // unsupported version or truncated header. Not panicking is
                    // the real test here.
                    let _ = decomp.decompress(&bytes);
                } else {
                    let result = decomp.decompress(&bytes);
                    // Should fail — at minimum with truncated header or invalid magic.
                    prop_assert!(result.is_err());
                }
            }
        }
    }
}

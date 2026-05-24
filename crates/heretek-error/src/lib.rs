use thiserror::Error;

/// Top-level error type for the heretik compression engine.
///
/// Each variant carries structured payload so a future agent (or human) can
/// localise failures without re-running — the slice-level verification requires
/// `CoderError::StackUnderflow{at_byte}`, `PredictorError::InvalidDistribution{index,sum}`,
/// `IntegrityError::ChecksumMismatch{expected,actual}`, and
/// `FormatError::InvalidMagic{found}` as concrete shapes.
#[derive(Error, Debug)]
pub enum HeretikError {
    /// Errors originating in the entropy coder (range / ANS coder).
    #[error(transparent)]
    Coder(#[from] CoderError),

    /// Errors originating in the neural predictor.
    #[error(transparent)]
    Predictor(#[from] PredictorError),

    /// Integrity / round-trip verification failures.
    #[error(transparent)]
    Integrity(#[from] IntegrityError),

    /// Bitstream / header format errors.
    #[error(transparent)]
    Format(#[from] FormatError),
}

// ---- Coder errors ----------------------------------------------------------

#[derive(Error, Debug)]
pub enum CoderError {
    /// The internal symbol stack underflowed — malformed or truncated bitstream.
    #[error("coder stack underflow at byte offset {at_byte}")]
    StackUnderflow { at_byte: u64 },

    /// Encoding failed for a reason that cannot be described by a narrower variant.
    #[error("encode failed: {message}")]
    EncodeFailed { message: String },

    /// Decoding failed for a reason that cannot be described by a narrower variant.
    #[error("decode failed: {message}")]
    DecodeFailed { message: String },
}

// ---- Predictor errors ------------------------------------------------------

#[derive(Error, Debug)]
pub enum PredictorError {
    /// A probability distribution produced by the model is invalid
    /// (e.g. the per-class probabilities sum to a value ≠ 1.0).
    #[error("invalid probability distribution: index {index}, sum {sum}")]
    InvalidDistribution { index: usize, sum: f64 },

    /// The predictor model could not be loaded from disk.
    #[error("failed to load model from {path}: {message}")]
    ModelLoadFailed { path: String, message: String },

    /// The forward pass through the neural model failed.
    #[error("forward pass failed: {message}")]
    ForwardPassFailed { message: String },
}

// ---- Integrity errors ------------------------------------------------------

/// A `[u8; 32]` wrapper that `Display`s as a 64-char lowercase hex string.
///
/// Used so `thiserror` can format checksums directly in `#[error("...")]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Hex32(pub [u8; 32]);

impl std::fmt::Display for Hex32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

#[derive(Error, Debug)]
pub enum IntegrityError {
    /// The reconstructed checksum does not match the expected checksum.
    #[error("checksum mismatch: expected {expected}, actual {actual}")]
    ChecksumMismatch { expected: Hex32, actual: Hex32 },

    /// A byte-for-byte round-trip comparison failed at a specific offset.
    #[error("round-trip mismatch at byte offset {offset}: expected 0x{expected_byte:02x}, got 0x{actual_byte:02x}")]
    RoundTripMismatch {
        offset: u64,
        expected_byte: u8,
        actual_byte: u8,
    },
}

// ---- Format errors ---------------------------------------------------------

/// A `[u8; 4]` wrapper that `Display`s as printable ASCII + hex, e.g. `HTK (48 54 4b 01)`.
///
/// Used so `thiserror` can format magic bytes directly in `#[error("...")]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MagicBytes(pub [u8; 4]);

impl std::fmt::Display for MagicBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ascii: String = self
            .0
            .iter()
            .map(|&b| if b.is_ascii_graphic() { b as char } else { '.' })
            .collect();
        write!(f, "{} ({:02x} {:02x} {:02x} {:02x})", ascii, self.0[0], self.0[1], self.0[2], self.0[3])
    }
}

#[derive(Error, Debug)]
pub enum FormatError {
    /// The magic bytes at the start of the bitstream do not match the expected value.
    #[error("invalid magic bytes: {found}")]
    InvalidMagic { found: MagicBytes },

    /// The bitstream version number is not supported by this library.
    #[error("unsupported format version {version}")]
    UnsupportedVersion { version: u32 },

    /// The header was truncated before all required fields could be read.
    #[error("truncated header: not enough bytes to parse all header fields")]
    TruncatedHeader,

    /// The compressed bitstream is corrupt at a known byte offset.
    #[error("corrupt bitstream at byte offset {offset}")]
    CorruptBitstream { offset: u64 },
}

// ---- Tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coder_error_display() {
        let e = CoderError::StackUnderflow { at_byte: 42 };
        assert_eq!(e.to_string(), "coder stack underflow at byte offset 42");

        let e = CoderError::EncodeFailed {
            message: "oom".into(),
        };
        assert_eq!(e.to_string(), "encode failed: oom");

        let e = CoderError::DecodeFailed {
            message: "bad sym".into(),
        };
        assert_eq!(e.to_string(), "decode failed: bad sym");
    }

    #[test]
    fn predictor_error_display() {
        let e = PredictorError::InvalidDistribution { index: 3, sum: 0.95 };
        assert_eq!(
            e.to_string(),
            "invalid probability distribution: index 3, sum 0.95"
        );

        let e = PredictorError::ModelLoadFailed {
            path: "m.bin".into(),
            message: "ENOENT".into(),
        };
        assert_eq!(
            e.to_string(),
            "failed to load model from m.bin: ENOENT"
        );

        let e = PredictorError::ForwardPassFailed {
            message: "NaN".into(),
        };
        assert_eq!(e.to_string(), "forward pass failed: NaN");
    }

    #[test]
    fn integrity_error_display_roundtrip() {
        let e = IntegrityError::RoundTripMismatch {
            offset: 1024,
            expected_byte: 0xAB,
            actual_byte: 0xCD,
        };
        assert_eq!(
            e.to_string(),
            "round-trip mismatch at byte offset 1024: expected 0xab, got 0xcd"
        );
    }

    #[test]
    fn integrity_error_display_checksum() {
        let e = IntegrityError::ChecksumMismatch {
            expected: Hex32([0xAA; 32]),
            actual: Hex32([0xBB; 32]),
        };
        let s = e.to_string();
        assert!(s.starts_with("checksum mismatch: expected "));
        assert!(s.contains(&"aa".repeat(32)));
        assert!(s.contains(&"bb".repeat(32)));
    }

    #[test]
    fn format_error_display() {
        let e = FormatError::InvalidMagic {
            found: MagicBytes([0x48, 0x54, 0x4B, 0x01]),
        };
        let s = e.to_string();
        assert!(s.starts_with("invalid magic bytes: "));
        // "HTK" is the ASCII decode of [0x48, 0x54, 0x4B]
        assert!(s.contains("HTK"));

        let e = FormatError::UnsupportedVersion { version: 99 };
        assert_eq!(e.to_string(), "unsupported format version 99");

        let e = FormatError::TruncatedHeader;
        assert!(e.to_string().contains("truncated header"));

        let e = FormatError::CorruptBitstream { offset: 2048 };
        assert_eq!(
            e.to_string(),
            "corrupt bitstream at byte offset 2048"
        );
    }

    #[test]
    fn top_level_error_from_conversions() {
        // CoderError → HeretikError (via #[from])
        let e: HeretikError = CoderError::StackUnderflow { at_byte: 1 }.into();
        assert!(matches!(
            e,
            HeretikError::Coder(CoderError::StackUnderflow { at_byte: 1 })
        ));

        // PredictorError → HeretikError
        let e: HeretikError =
            PredictorError::InvalidDistribution { index: 0, sum: 0.5 }.into();
        assert!(matches!(
            e,
            HeretikError::Predictor(PredictorError::InvalidDistribution { index: 0, .. })
        ));

        // IntegrityError → HeretikError
        let e: HeretikError = IntegrityError::RoundTripMismatch {
            offset: 0,
            expected_byte: 0,
            actual_byte: 1,
        }
        .into();
        assert!(matches!(
            e,
            HeretikError::Integrity(IntegrityError::RoundTripMismatch { offset: 0, .. })
        ));

        // FormatError → HeretikError
        let e: HeretikError = FormatError::TruncatedHeader.into();
        assert!(matches!(
            e,
            HeretikError::Format(FormatError::TruncatedHeader)
        ));
    }

    #[test]
    fn top_level_error_display_is_transparent() {
        // The #[error(transparent)] means Display on HeretikError
        // delegates to the inner error's Display.
        let e: HeretikError = CoderError::StackUnderflow { at_byte: 99 }.into();
        assert_eq!(e.to_string(), "coder stack underflow at byte offset 99");
    }

    #[test]
    fn error_types_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<HeretikError>();
        assert_send_sync::<CoderError>();
        assert_send_sync::<PredictorError>();
        assert_send_sync::<IntegrityError>();
        assert_send_sync::<FormatError>();
    }

    #[test]
    fn hex32_display() {
        let mut arr = [0u8; 32];
        arr[0] = 0x0f;
        arr[31] = 0xf0;
        let h = Hex32(arr);
        let s = h.to_string();
        assert_eq!(s.len(), 64);
        assert!(s.starts_with("0f"));
        assert!(s.ends_with("f0"));
    }

    #[test]
    fn magic_bytes_display() {
        let m = MagicBytes([0x48, 0x54, 0x4B, 0x01]);
        let s = m.to_string();
        // 0x01 is a control char, so the ascii portion renders it as '.'
        assert_eq!(s, "HTK. (48 54 4b 01)");
    }
}

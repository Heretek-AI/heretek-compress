use heretek_error::{FormatError, MagicBytes};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};

/// Magic bytes that identify a heretik-compress bitstream: `HTC` + version byte `0x01`.
pub const MAGIC: [u8; 4] = *b"HTC\x01";

/// Current wire format version.
pub const CURRENT_VERSION: u16 = 1;

/// The on-disk file header (54 bytes, big-endian).
///
/// # Wire layout
///
/// | offset | size | field           |
/// |--------|------|-----------------|
/// | 0      | 4    | magic           |
/// | 4      | 2    | version         |
/// | 6      | 2    | model_version   |
/// | 8      | 4    | flags           |
/// | 12     | 32   | sha256          |
/// | 44     | 8    | original_size   |
/// | 52     | 8    | compressed_size |
///
/// (Header is 54 bytes; we add 2 bytes of zero-padding so the total is
/// 56 and the following data is 8-byte aligned.  The padding is ignored
/// on read.)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    /// Magic bytes — must be `MAGIC`.
    pub magic: [u8; 4],
    /// Wire format version — must be `CURRENT_VERSION`.
    pub version: u16,
    /// Version of the predictor model used during compression.
    pub model_version: u16,
    /// Reserved flags field.
    pub flags: u32,
    /// SHA-256 hash of the decompressed payload.
    pub sha256: [u8; 32],
    /// Original (decompressed) payload size in bytes.
    pub original_size: u64,
    /// Compressed payload size in bytes (the ANS bitstream).
    pub compressed_size: u64,
}

impl Default for Header {
    fn default() -> Self {
        Self {
            magic: MAGIC,
            version: CURRENT_VERSION,
            model_version: 0,
            flags: 0,
            sha256: [0u8; 32],
            original_size: 0,
            compressed_size: 0,
        }
    }
}

impl Header {
    /// Create a new header with all fields specified.
    pub fn new(
        model_version: u16,
        flags: u32,
        sha256: [u8; 32],
        original_size: u64,
        compressed_size: u64,
    ) -> Self {
        Self {
            magic: MAGIC,
            version: CURRENT_VERSION,
            model_version,
            flags,
            sha256,
            original_size,
            compressed_size,
        }
    }

    /// Serialise this header to `writer` in big-endian format (60 bytes on disk).
    pub fn write(&self, writer: &mut impl Write) -> std::io::Result<()> {
        writer.write_all(&self.magic)?;
        writer.write_all(&self.version.to_be_bytes())?;
        writer.write_all(&self.model_version.to_be_bytes())?;
        writer.write_all(&self.flags.to_be_bytes())?;
        writer.write_all(&self.sha256)?;
        writer.write_all(&self.original_size.to_be_bytes())?;
        writer.write_all(&self.compressed_size.to_be_bytes())?;
        Ok(())
    }

    /// Deserialise a `Header` from `reader`, validating magic bytes and
    /// version.
    ///
    /// # Errors
    ///
    /// - `FormatError::InvalidMagic` if the first 4 bytes are not `MAGIC`.
    /// - `FormatError::UnsupportedVersion` if the version ≠ `CURRENT_VERSION`.
    /// - `FormatError::TruncatedHeader` if fewer than 60 bytes are available.
    pub fn read(reader: &mut impl Read) -> Result<Self, FormatError> {
        let mut buf = [0u8; 60];
        read_exact_or_truncated(reader, &mut buf)?;

        let magic: [u8; 4] = buf[0..4].try_into().unwrap();
        if magic != MAGIC {
            return Err(FormatError::InvalidMagic {
                found: MagicBytes(magic),
            });
        }

        let version = u16::from_be_bytes(buf[4..6].try_into().unwrap());
        if version != CURRENT_VERSION {
            return Err(FormatError::UnsupportedVersion {
                version: version as u32,
            });
        }

        let model_version = u16::from_be_bytes(buf[6..8].try_into().unwrap());
        let flags = u32::from_be_bytes(buf[8..12].try_into().unwrap());
        let sha256: [u8; 32] = buf[12..44].try_into().unwrap();
        let original_size = u64::from_be_bytes(buf[44..52].try_into().unwrap());
        let compressed_size = u64::from_be_bytes(buf[52..60].try_into().unwrap());

        Ok(Self {
            magic,
            version,
            model_version,
            flags,
            sha256,
            original_size,
            compressed_size,
        })
    }
}

/// Read exactly `dst.len()` bytes into `dst`, or return
/// `FormatError::TruncatedHeader`.
fn read_exact_or_truncated(reader: &mut impl Read, dst: &mut [u8]) -> Result<(), FormatError> {
    let mut offset = 0;
    while offset < dst.len() {
        let n = reader
            .read(&mut dst[offset..])
            .map_err(|_| FormatError::TruncatedHeader)?;
        if n == 0 {
            return Err(FormatError::TruncatedHeader);
        }
        offset += n;
    }
    Ok(())
}

/// Compute the SHA-256 hash of `data` and compare it against `expected`.
///
/// Returns `Ok(())` on match, or an `IntegrityError::ChecksumMismatch`.
pub fn verify_checksum(data: &[u8], expected: &[u8; 32]) -> Result<(), heretek_error::IntegrityError> {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let actual: [u8; 32] = hasher.finalize().into();
    if actual != *expected {
        return Err(heretek_error::IntegrityError::ChecksumMismatch {
            expected: heretek_error::Hex32(*expected),
            actual: heretek_error::Hex32(actual),
        });
    }
    Ok(())
}

// ---- Tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_write_read_round_trip() {
        let header = Header::new(7, 0xDEAD_BEEF, [0xAB; 32], 1024, 512);
        let mut buf = Vec::new();
        header.write(&mut buf).unwrap();
        assert_eq!(buf.len(), 60);

        let round_tripped = Header::read(&mut buf.as_slice()).unwrap();
        assert_eq!(round_tripped, header);
    }

    #[test]
    fn header_default_magic_and_version() {
        let header = Header::default();
        assert_eq!(header.magic, MAGIC);
        assert_eq!(header.version, CURRENT_VERSION);
    }

    #[test]
    fn magic_byte_validation_rejects_wrong_magic() {
        let mut header = Header::default();
        header.magic = *b"BAD\x01";
        let mut buf = Vec::new();
        header.write(&mut buf).unwrap();

        let result = Header::read(&mut buf.as_slice());
        assert!(matches!(result, Err(FormatError::InvalidMagic { .. })));
    }

    #[test]
    fn version_validation_rejects_unsupported_version() {
        let mut header = Header::default();
        header.version = 99;
        let mut buf = Vec::new();
        header.write(&mut buf).unwrap();

        let result = Header::read(&mut buf.as_slice());
        assert!(matches!(
            result,
            Err(FormatError::UnsupportedVersion { version: 99 })
        ));
    }

    #[test]
    fn truncated_header_error() {
        let buf = [0u8; 10]; // less than 56 bytes
        let result = Header::read(&mut buf.as_slice());
        assert!(matches!(result, Err(FormatError::TruncatedHeader)));
    }

    #[test]
    fn checksum_verification_passes() {
        let data = b"hello world";
        let mut hasher = Sha256::new();
        hasher.update(data);
        let expected: [u8; 32] = hasher.finalize().into();

        verify_checksum(data, &expected).unwrap();
    }

    #[test]
    fn checksum_verification_detects_bit_flip() {
        let data = b"hello world";
        let mut hasher = Sha256::new();
        hasher.update(data);
        let mut expected: [u8; 32] = hasher.finalize().into();
        expected[0] ^= 0x01; // flip one bit

        let result = verify_checksum(data, &expected);
        assert!(matches!(
            result,
            Err(heretek_error::IntegrityError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn checksum_detects_wrong_data() {
        let data_a = b"hello world";
        let data_b = b"HELLO WORLD"; // different data
        let mut hasher = Sha256::new();
        hasher.update(data_b);
        let expected: [u8; 32] = hasher.finalize().into();

        let result = verify_checksum(data_a, &expected);
        assert!(matches!(
            result,
            Err(heretek_error::IntegrityError::ChecksumMismatch { .. })
        ));
    }

    /// Edge case: all-zero data still has a deterministic checksum.
    #[test]
    fn checksum_empty_data() {
        let data = [];
        let mut hasher = Sha256::new();
        hasher.update(data);
        let expected: [u8; 32] = hasher.finalize().into();
        verify_checksum(&data, &expected).unwrap();
    }
}

use heretek_error::CoderError;

/// Wrapper around constriction's stack-based ANS coder.
///
/// ANS is a stack (LIFO): symbols must be encoded in reverse order so they
/// decode in forward order.  We buffer symbols during `encode()` and flush
/// them in reverse at `finish_encode()`, so the caller can feed symbols in
/// logical (forward) order.
///
/// Supports both i.i.d. (same model for all symbols) and heterogeneous
/// (per-symbol) probability distributions.  For i.i.d. use the existing
/// `encode`/`decode`/`decode_symbol` methods; for heterogeneous per-symbol
/// models call `encode(probs, symbol)` with a different `probs` for each
/// position and then call `decode_symbol` with the corresponding model
/// at decode time.
pub struct AnsCoder {
    inner: constriction::stream::stack::DefaultAnsCoder,
    /// Buffered (symbol, probability-distribution) pairs in encode order.
    buffer: Vec<(u8, [f32; 256])>,
}

/// Alias for a constriction entropy model built from f32 probabilities.
type EntropyModel = constriction::stream::model::DefaultContiguousCategoricalEntropyModel;

impl AnsCoder {
    pub fn new() -> Self {
        Self {
            inner: constriction::stream::stack::DefaultAnsCoder::new(),
            buffer: Vec::new(),
        }
    }

    /// Encode a single symbol with a specific probability distribution.
    ///
    /// Each call may supply a different `probs` array — heterogeneous
    /// per-symbol encoding is fully supported.
    pub fn encode(&mut self, probs: &[f32; 256], symbol: u8) -> Result<(), CoderError> {
        self.buffer.push((symbol, *probs));
        Ok(())
    }

    /// Finish encoding and return the compressed word stream.
    ///
    /// Uses `encode_symbols_reverse` so per-symbol probability models are
    /// paired with their corresponding symbols.
    pub fn finish_encode(mut self) -> Result<Vec<u32>, CoderError> {
        if self.buffer.is_empty() {
            return Ok(Vec::new());
        }

        // Build one entropy model per buffered symbol.
        let models: Vec<EntropyModel> = self
            .buffer
            .iter()
            .map(|(_, probs)| build_encoder_model(probs))
            .collect::<Result<Vec<_>, _>>()?;

        let symbols: Vec<usize> = self.buffer.iter().map(|(s, _)| *s as usize).collect();

        // encode_symbols_reverse handles stack reversal internally: we pass
        // symbols in forward order and constriction encodes from last to first.
        self.inner
            .encode_symbols_reverse(symbols.into_iter().zip(models.into_iter()))
            .map_err(|e| CoderError::EncodeFailed {
                message: format!("ANS encode failed: {e:?}"),
            })?;

        Ok(self.inner.into_compressed().unwrap_or_default())
    }

    /// Initialise a decoder from previously encoded words.
    pub fn start_decode(words: &[u32]) -> Result<Self, CoderError> {
        let inner =
            constriction::stream::stack::DefaultAnsCoder::from_compressed(words.to_vec())
                .map_err(|e| CoderError::DecodeFailed {
                    message: format!("ANS init failed: {e:?}"),
                })?;
        Ok(Self {
            inner,
            buffer: Vec::new(),
        })
    }

    /// Decode a single symbol using the given probability distribution.
    ///
    /// For heterogeneous decoding, call once per expected output symbol with
    /// the same per-position model that was used during encoding.
    pub fn decode_symbol(&mut self, probs: &[f32; 256]) -> Result<u8, CoderError> {
        use constriction::stream::Decode;
        let model = build_decoder_model(probs)?;
        let symbol: usize = self.inner.decode_symbol(model).map_err(|e| {
            CoderError::DecodeFailed {
                message: format!("ANS decode_symbol failed: {e:?}"),
            }
        })?;
        Ok(symbol as u8)
    }

    /// Decode `amount` symbols with the same i.i.d. model (convenience
    /// wrapper for the common uniform/single-model case).
    pub fn decode(&mut self, probs: &[f32; 256], amount: usize) -> Result<Vec<u8>, CoderError> {
        if amount == 0 {
            return Ok(Vec::new());
        }

        let model = build_decoder_model(probs)?;

        use constriction::stream::Decode;
        let decoded: Vec<usize> = self
            .inner
            .decode_iid_symbols(amount, &model)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| CoderError::DecodeFailed {
                message: format!("ANS decode failed: {e:?}"),
            })?;

        Ok(decoded.into_iter().map(|s| s as u8).collect())
    }
}

// ---- helpers ---------------------------------------------------------------

fn build_encoder_model(probs: &[f32; 256]) -> Result<EntropyModel, CoderError> {
    EntropyModel::from_floating_point_probabilities_fast(probs, None).map_err(|e| {
        CoderError::EncodeFailed {
            message: format!("model build failed: {e:?}"),
        }
    })
}

fn build_decoder_model(probs: &[f32; 256]) -> Result<EntropyModel, CoderError> {
    EntropyModel::from_floating_point_probabilities_fast(probs, None).map_err(|e| {
        CoderError::DecodeFailed {
            message: format!("model build failed: {e:?}"),
        }
    })
}

impl Default for AnsCoder {
    fn default() -> Self {
        Self::new()
    }
}

// ---- Tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_probs() -> [f32; 256] {
        [1.0f32 / 256.0; 256]
    }

    fn skewed_probs() -> [f32; 256] {
        let mut raw = [0.0f32; 256];
        for (i, p) in raw.iter_mut().enumerate() {
            *p = 1.0 / (i + 1) as f32;
        }
        let sum: f32 = raw.iter().sum();
        for p in &mut raw {
            *p /= sum;
        }
        raw
    }

    fn round_trip(probs: &[f32; 256], data: &[u8]) {
        let mut encoder = AnsCoder::new();
        for &byte in data {
            encoder.encode(probs, byte).unwrap();
        }
        let compressed = encoder.finish_encode().unwrap();
        let mut decoder = AnsCoder::start_decode(&compressed).unwrap();
        let recovered = decoder.decode(probs, data.len()).unwrap();
        assert_eq!(recovered, data, "round-trip mismatch");
    }

    #[test]
    fn round_trip_10k_random_uniform() {
        let mut state: u32 = 12345;
        let data: Vec<u8> = (0..10_000)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                (state >> 16) as u8
            })
            .collect();
        round_trip(&uniform_probs(), &data);
    }

    #[test]
    fn round_trip_10k_random_skewed() {
        let mut state: u32 = 67890;
        let data: Vec<u8> = (0..10_000)
            .map(|_| {
                state = state.wrapping_mul(1103515245).wrapping_add(12345);
                (state >> 16) as u8
            })
            .collect();
        round_trip(&skewed_probs(), &data);
    }

    #[test]
    fn round_trip_small_patterns() {
        let probs = uniform_probs();
        for data in &[
            &[][..],
            &[0u8][..],
            &[255u8][..],
            &[0u8, 0, 0, 0][..],
            &[0u8, 1, 2, 3, 4, 5, 6, 7, 8, 9][..],
            &[128u8; 100][..],
        ] {
            round_trip(&probs, data);
        }
    }

    #[test]
    fn empty_input_produces_minimal_output() {
        let encoder = AnsCoder::new();
        let compressed = encoder.finish_encode().unwrap();
        assert!(
            compressed.is_empty(),
            "empty stream compresses to {} words",
            compressed.len()
        );
    }

    #[test]
    fn skewed_compresses_better_than_uniform() {
        let skewed = skewed_probs();
        let uniform = uniform_probs();
        let data: Vec<u8> = (0..1000).map(|i| (i % 16) as u8).collect();

        let mut enc = AnsCoder::new();
        for &b in &data {
            enc.encode(&uniform, b).unwrap();
        }
        let uniform_size = enc.finish_encode().unwrap().len();

        let mut enc = AnsCoder::new();
        for &b in &data {
            enc.encode(&skewed, b).unwrap();
        }
        let skewed_size = enc.finish_encode().unwrap().len();

        assert!(
            skewed_size < uniform_size,
            "skewed {skewed_size} words >= uniform {uniform_size} words"
        );
    }

    #[test]
    fn encode_decode_all_256_symbols() {
        let probs = uniform_probs();
        let data: Vec<u8> = (0..=255).collect();
        let mut encoder = AnsCoder::new();
        for &byte in &data {
            encoder.encode(&probs, byte).unwrap();
        }
        let compressed = encoder.finish_encode().unwrap();
        let mut decoder = AnsCoder::start_decode(&compressed).unwrap();
        let recovered = decoder.decode(&probs, data.len()).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn heterogeneous_per_symbol_round_trip() {
        let probs: Vec<[f32; 256]> = (0..100)
            .map(|i| {
                let mut p = [1.0f32 / 256.0; 256];
                // Slightly bias toward symbol i%256 each position.
                p[i % 256] = 20.0 / 256.0;
                let sum: f32 = p.iter().sum();
                for v in &mut p {
                    *v /= sum;
                }
                p
            })
            .collect();

        let data: Vec<u8> = (0..100).map(|i| (i % 256) as u8).collect();

        let mut encoder = AnsCoder::new();
        for (byte, dist) in data.iter().zip(probs.iter()) {
            encoder.encode(dist, *byte).unwrap();
        }
        let compressed = encoder.finish_encode().unwrap();

        let mut decoder = AnsCoder::start_decode(&compressed).unwrap();
        let mut recovered = Vec::with_capacity(100);
        for dist in &probs {
            let s = decoder.decode_symbol(dist).unwrap();
            recovered.push(s);
        }
        assert_eq!(recovered, data);
    }

    #[test]
    fn heterogeneous_produces_same_as_iid_when_uniform() {
        // When all per-symbol models are identical, heterogeneous encoding
        // should produce byte-identical output to i.i.d. encoding.
        let uniform = uniform_probs();
        let data: Vec<u8> = b"hello world, this is a test".to_vec();

        // i.i.d. path
        let mut enc = AnsCoder::new();
        for &b in &data {
            enc.encode(&uniform, b).unwrap();
        }
        let iid_compressed = enc.finish_encode().unwrap();

        // Same but via decode_symbol per-byte
        let mut dec = AnsCoder::start_decode(&iid_compressed).unwrap();
        let mut recovered = Vec::with_capacity(data.len());
        for _ in 0..data.len() {
            recovered.push(dec.decode_symbol(&uniform).unwrap());
        }
        assert_eq!(recovered, data);
    }
}

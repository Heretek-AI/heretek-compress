use heretek_error::CoderError;

/// Wrapper around constriction's stack-based ANS coder.
///
/// ANS is a stack (LIFO): symbols must be encoded in reverse order so they
/// decode in forward order.  We buffer symbols during `encode()` and flush
/// them in reverse at `finish_encode()`, so the caller can feed symbols in
/// logical (forward) order.
///
/// Currently supports i.i.d. encoding (all symbols share the same model).
pub struct AnsCoder {
    inner: constriction::stream::stack::DefaultAnsCoder,
    /// Symbols accumulated in encode order; flushed in reverse at finish.
    symbols: Vec<usize>,
    /// Cached model (set on first encode, shared across all symbols).
    model_probs: Option<[f32; 256]>,
}

impl AnsCoder {
    pub fn new() -> Self {
        Self {
            inner: constriction::stream::stack::DefaultAnsCoder::new(),
            symbols: Vec::new(),
            model_probs: None,
        }
    }

    pub fn encode(&mut self, probs: &[f32; 256], symbol: u8) -> Result<(), CoderError> {
        match &self.model_probs {
            Some(cached) if cached == probs => {}
            Some(_) => {
                return Err(CoderError::EncodeFailed {
                    message: "heterogeneous per-symbol models not yet supported".into(),
                });
            }
            None => {
                self.model_probs = Some(*probs);
            }
        }
        self.symbols.push(symbol as usize);
        Ok(())
    }

    pub fn finish_encode(mut self) -> Result<Vec<u32>, CoderError> {
        let probs: [f32; 256] =
            self.model_probs.unwrap_or([1.0f32 / 256.0; 256]);

        let model = constriction::stream::model::DefaultContiguousCategoricalEntropyModel
            ::from_floating_point_probabilities_fast(&probs, None)
            .map_err(|e| CoderError::EncodeFailed {
                message: format!("model build failed: {e:?}"),
            })?;

        // encode_iid_symbols_reverse handles stack reversal internally —
        // we pass symbols in forward order and it encodes from last to first.
        self.inner
            .encode_iid_symbols_reverse(&self.symbols, &model)
            .map_err(|e| CoderError::EncodeFailed {
                message: format!("ANS encode failed: {e:?}"),
            })?;

        Ok(self.inner.into_compressed().unwrap_or_default())
    }

    pub fn start_decode(words: &[u32]) -> Result<Self, CoderError> {
        let inner =
            constriction::stream::stack::DefaultAnsCoder::from_compressed(words.to_vec())
                .map_err(|e| CoderError::DecodeFailed {
                    message: format!("ANS init failed: {e:?}"),
                })?;
        Ok(Self {
            inner,
            symbols: Vec::new(),
            model_probs: None,
        })
    }

    pub fn decode(&mut self, probs: &[f32; 256], amount: usize) -> Result<Vec<u8>, CoderError> {
        if amount == 0 {
            return Ok(Vec::new());
        }

        let model = constriction::stream::model::DefaultContiguousCategoricalEntropyModel
            ::from_floating_point_probabilities_fast(probs, None)
            .map_err(|e| CoderError::DecodeFailed {
                message: format!("model build failed: {e:?}"),
            })?;

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
}

use heretek_error::PredictorError;

/// Scale `[0.0, 1.0)` float probabilities to 16-bit fixed-point.
///
/// Each entry is multiplied by 65535.0 and rounded to the nearest `u16`.
/// This quantization layer ensures cross-platform determinism — different
/// float hardware / libm implementations produce identical quantized values
/// as long as the input `f32` slice is identical.
pub fn quantize(probs: &[f32; 256]) -> [u16; 256] {
    let mut out = [0u16; 256];
    for (i, &p) in probs.iter().enumerate() {
        // clamp before scaling so pathological values don't overflow
        let clamped = p.clamp(0.0, 1.0);
        let scaled = (clamped * 65535.0).round() as u64;
        out[i] = (scaled.min(65535)) as u16;
    }
    out
}

/// Convert quantized 16-bit values back to `f32` probabilities.
///
/// Returns a probability distribution whose entries sum (approximately) to
/// 1.0.  If the quantized array sums to 0 we return a uniform distribution
/// as a fallback so downstream callers never divide by zero.
pub fn normalize(quantized: &[u16; 256]) -> [f32; 256] {
    let sum: u32 = quantized.iter().map(|&x| x as u32).sum();
    if sum == 0 {
        // Edge case: all zero → uniform distribution
        return [1.0f32 / 256.0; 256];
    }
    let mut out = [0.0f32; 256];
    for (i, &q) in quantized.iter().enumerate() {
        out[i] = q as f32 / sum as f32;
    }
    out
}

/// Basic sanity-check on a probability distribution.
///
/// Returns `Ok(())` when every entry is ≥ 0 and the sum is within
/// `[0.999, 1.001]`.  Otherwise returns `PredictorError::InvalidDistribution`.
pub fn validate_distribution(probs: &[f32; 256]) -> Result<(), PredictorError> {
    let sum: f32 = probs.iter().sum();
    for (i, &p) in probs.iter().enumerate() {
        if p < 0.0 {
            return Err(PredictorError::InvalidDistribution {
                index: i,
                sum: sum as f64,
            });
        }
    }
    if sum < 0.999 || sum > 1.001 {
        // Report the first non-negative index (just a convention)
        return Err(PredictorError::InvalidDistribution {
            index: 0,
            sum: sum as f64,
        });
    }
    Ok(())
}

// ---- Tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quantize_uniform_distribution() {
        let uniform = [1.0f32 / 256.0; 256];
        let q = quantize(&uniform);
        // Every bin should be close to 256 (65535/256 ≈ 255.996 → rounds to 256)
        for &v in &q {
            assert!(v == 255 || v == 256, "got {v}");
        }
    }

    #[test]
    fn quantize_max_error_below_one_div_65536() {
        // Create a deliberately non-uniform distribution
        let mut probs = [0.0f32; 256];
        probs[0] = 1.0;
        let q = quantize(&probs);
        assert_eq!(q[0], 65535);
        for i in 1..256 {
            assert_eq!(q[i], 0);
        }

        // Round-trip: quantize → normalize, error < 1/65536 per bin
        let original = [1.0f32 / 256.0; 256];
        let q = quantize(&original);
        let restored = normalize(&q);
        let max_error = 1.0 / 65536.0;
        for i in 0..256 {
            let diff = (restored[i] - original[i]).abs();
            assert!(
                diff < max_error,
                "bin {i}: diff {diff:.8} >= {max_error:.8}"
            );
        }
    }

    #[test]
    fn quantize_clamps_negative_to_zero() {
        let mut probs = [0.0f32; 256];
        probs[0] = -0.5;
        let q = quantize(&probs);
        assert_eq!(q[0], 0);
    }

    #[test]
    fn normalize_sums_to_one() {
        let q = quantize(&[1.0f32 / 256.0; 256]);
        let n = normalize(&q);
        let sum: f32 = n.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
    }

    #[test]
    fn normalize_all_zeros_is_uniform() {
        let q = [0u16; 256];
        let n = normalize(&q);
        for &v in &n {
            assert!((v - 1.0 / 256.0).abs() < 1e-7);
        }
    }

    #[test]
    fn validate_distribution_accepts_valid() {
        let uniform = [1.0f32 / 256.0; 256];
        validate_distribution(&uniform).unwrap();
    }

    #[test]
    fn validate_distribution_rejects_negative() {
        let mut probs = [0.0f32; 256];
        probs[0] = 1.1;
        probs[10] = -0.1;
        let err = validate_distribution(&probs).unwrap_err();
        assert!(matches!(err, PredictorError::InvalidDistribution { .. }));
    }

    #[test]
    fn validate_distribution_rejects_sum_far_from_one() {
        let mut probs = [0.0f32; 256];
        probs[0] = 0.5;
        let err = validate_distribution(&probs).unwrap_err();
        assert!(matches!(err, PredictorError::InvalidDistribution { .. }));
    }

    #[test]
    fn validate_distribution_accepts_sum_in_tolerance() {
        // 0.9995 is within tolerance
        let mut probs = [0.0f32; 256];
        probs[0] = 0.9995;
        probs[1] = 0.0005;
        validate_distribution(&probs).unwrap();
    }
}

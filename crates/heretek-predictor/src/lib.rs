//! Neural predictor for heretik — byte-level autoregressive transformer
//! that produces per-position probability distributions fed into the ANS coder.
//!
//! Two implementations of the [`Predictor`] trait:
//! - [`StubPredictor`]: uniform distribution (testing without trained weights)
//! - [`Transformer`]: decoder-only transformer backed by candle

use std::path::Path;

use candle_core::{DType, Device, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder, VarMap};
use heretek_error::PredictorError;
use serde::{Deserialize, Serialize};

// ---- Predictor trait -------------------------------------------------------

/// A byte-level predictor that maps an input context into per-position
/// probability distributions over the next 256 possible bytes.
///
/// `predict(context)` returns `N` distributions where `distribution[i]` is
/// the predicted distribution for `context[i]` given `context[0..i]`.
/// `predict_single(context)` returns the distribution for the byte immediately
/// after `context`.
pub trait Predictor {
    /// Per-position distributions: for an input of `N` bytes, returns `N`
    /// distributions where entry `i` predicts `context[i]` given `context[0..i]`.
    fn predict(&self, context: &[u8]) -> Result<Vec<[f32; 256]>, PredictorError>;

    /// Single distribution for the byte after `context`.
    fn predict_single(&self, context: &[u8]) -> Result<[f32; 256], PredictorError>;
}

// ---- Config ----------------------------------------------------------------

/// Transformer model hyper-parameters.
///
/// Supports serde round-tripping so configs can be saved alongside weights
/// and re-loaded on the far side.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Config {
    /// Number of transformer decoder layers.
    #[serde(default = "default_num_layers")]
    pub num_layers: usize,

    /// Embedding dimension (must be divisible by `num_heads`).
    #[serde(default = "default_embed_dim")]
    pub embed_dim: usize,

    /// Number of attention heads per layer.
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,

    /// Maximum context window (positional embedding table size).
    #[serde(default = "default_context_window")]
    pub context_window: usize,

    /// Vocabulary size — always 256 for byte-level prediction.
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
}

fn default_num_layers() -> usize {
    8
}
fn default_embed_dim() -> usize {
    384
}
fn default_num_heads() -> usize {
    6
}
fn default_context_window() -> usize {
    512
}
fn default_vocab_size() -> usize {
    256
}

impl Default for Config {
    fn default() -> Self {
        Self {
            num_layers: default_num_layers(),
            embed_dim: default_embed_dim(),
            num_heads: default_num_heads(),
            context_window: default_context_window(),
            vocab_size: default_vocab_size(),
        }
    }
}

// ---- Transformer -----------------------------------------------------------

/// A single decoder layer: causal self-attention + FFN with pre-norm
/// residuals.
struct DecoderLayer {
    /// Combined QKV projection: `embed_dim → 3 * embed_dim`.
    qkv_proj: Linear,
    /// Output projection after attention mixture.
    out_proj: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    /// FFN expansion: `embed_dim → 4 * embed_dim`.
    ff_up: Linear,
    /// FFN contraction: `4 * embed_dim → embed_dim`.
    ff_down: Linear,
    num_heads: usize,
    head_dim: usize,
}

impl DecoderLayer {
    fn forward(&self, x: &Tensor, mask: &Tensor) -> candle_core::Result<Tensor> {
        let (seq_len, embed_dim) = (x.dims()[0], x.dims()[1]);

        // ---- Self-attention --------------------------------------------------
        let qkv = self.qkv_proj.forward(x)?; // (seq_len, 3 * embed_dim)
        let qkv = qkv.reshape((seq_len, 3, self.num_heads, self.head_dim))?;
        let qkv = qkv.permute((1, 2, 0, 3))?; // (3, num_heads, seq_len, head_dim)

        let q = qkv.get(0)?; // (num_heads, seq_len, head_dim)
        let k = qkv.get(1)?;
        let v = qkv.get(2)?;

        let scale = 1.0_f64 / (self.head_dim as f64).sqrt();
        // scores: (num_heads, seq_len, seq_len)
        let scores = q.matmul(&k.transpose(1, 2)?)?;
        let scores = (scores * scale)?;
        let scores = scores.broadcast_add(mask)?;
        let attn_weights = candle_nn::ops::softmax(&scores, 2)?;
        // attn_out: (num_heads, seq_len, head_dim)
        let attn_out = attn_weights.matmul(&v)?;
        // Merge heads: (seq_len, embed_dim)
        let attn_out = attn_out.permute((1, 0, 2))?.reshape((seq_len, embed_dim))?;
        let attn_out = self.out_proj.forward(&attn_out)?;

        // Residual + norm
        let x = (x + attn_out)?;
        let x = self.norm1.forward(&x)?;

        // ---- FFN ---------------------------------------------------------------
        let ff = self.ff_up.forward(&x)?;
        let ff = ff.gelu()?;
        let ff = self.ff_down.forward(&ff)?;

        // Residual + norm
        let x = (x + ff)?;
        self.norm2.forward(&x)
    }
}

/// Byte-level autoregressive transformer (decoder-only).
///
/// Created either with random initialisation (`Transformer::new`) or loaded
/// from safetensors checkpoint files (`Transformer::load`) or from an
/// in-memory buffer (`Transformer::load_from_buffer`).  Implements
/// [`Predictor`] so it slots directly into the heretek-engine pipeline.
pub struct Transformer {
    config: Config,
    token_embed: Embedding,
    pos_embed: Embedding,
    layers: Vec<DecoderLayer>,
    norm: LayerNorm,
    output_proj: Linear,
    device: Device,
    /// Retained for `save()` — only `Some` when the model was created via
    /// `Transformer::new()`.  `None` when loaded from a buffer or mmap file.
    varmap: Option<VarMap>,
}

impl Transformer {
    /// Create a transformer with random (Xavier-like) initialisation.
    ///
    /// The returned model retains a [`VarMap`] that can be persisted via
    /// [`Transformer::save`].
    pub fn new(config: &Config) -> candle_core::Result<Self> {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, dtype, &device);

        let mut model = Self::build(config, vb, device)?;
        model.varmap = Some(varmap);
        Ok(model)
    }

    /// Load a transformer from one or more safetensors files.
    ///
    /// # Safety
    ///
    /// The underlying safetensors mmap is considered unsafe because the file
    /// may change on disk while the mapping is live.
    pub unsafe fn load(path: &Path, config: &Config) -> candle_core::Result<Self> {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let vb = VarBuilder::from_mmaped_safetensors(&[path], dtype, &device)?;

        Self::build(config, vb, device)
    }

    /// Load a transformer from an in-memory safetensors buffer.
    ///
    /// This is the safe loading path — the buffer is fully parsed upfront
    /// and no mmap is used.  Suitable for `include_bytes!` embedding and
    /// general use where mmap is undesirable.
    ///
    /// Models loaded via this path **cannot** be saved back (no `VarMap`
    /// is retained).
    pub fn load_from_buffer(data: &[u8], config: &Config) -> Result<Self, PredictorError> {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let vb = VarBuilder::from_slice_safetensors(data, dtype, &device).map_err(|e| {
            PredictorError::ModelLoadFailed {
                path: "<buffer>".to_string(),
                message: format!("failed to parse safetensors from buffer: {e}"),
            }
        })?;

        Self::build(config, vb, device).map_err(|e| PredictorError::ModelLoadFailed {
            path: "<buffer>".to_string(),
            message: format!("build from buffer: {e}"),
        })
    }

    /// Save the model weights to a safetensors file.
    ///
    /// Only available when the model was created with [`Transformer::new`].
    /// Returns an error if the model was loaded from a buffer or mmap file
    /// (no `VarMap` retained).
    pub fn save(&self, path: &Path) -> Result<(), PredictorError> {
        let varmap = self.varmap.as_ref().ok_or_else(|| {
            PredictorError::ModelLoadFailed {
                path: path.display().to_string(),
                message: "cannot save a model that was loaded from a buffer or file".to_string(),
            }
        })?;
        varmap.save(path).map_err(|e| PredictorError::ModelLoadFailed {
            path: path.display().to_string(),
            message: format!("failed to save safetensors: {e}"),
        })
    }

    /// Shared construction logic used by `new`, `load`, and `load_from_buffer`.
    fn build(config: &Config, vb: VarBuilder, device: Device) -> candle_core::Result<Self> {
        if config.embed_dim % config.num_heads != 0 {
            candle_core::bail!(
                "embed_dim ({}) must be divisible by num_heads ({})",
                config.embed_dim,
                config.num_heads
            );
        }

        let head_dim = config.embed_dim / config.num_heads;

        let vb_tok = vb.pp("token_embed");
        let vb_pos = vb.pp("pos_embed");

        let token_embed = candle_nn::embedding(config.vocab_size, config.embed_dim, vb_tok)?;
        let pos_embed =
            candle_nn::embedding(config.context_window, config.embed_dim, vb_pos)?;

        let mut layers = Vec::with_capacity(config.num_layers);
        for i in 0..config.num_layers {
            let vb_layer = vb.pp(format!("layer_{i}"));
            let layer = DecoderLayer {
                qkv_proj: candle_nn::linear(
                    config.embed_dim,
                    3 * config.embed_dim,
                    vb_layer.pp("qkv"),
                )?,
                out_proj: candle_nn::linear(
                    config.embed_dim,
                    config.embed_dim,
                    vb_layer.pp("out"),
                )?,
                norm1: candle_nn::layer_norm(config.embed_dim, 1e-5, vb_layer.pp("norm1"))?,
                norm2: candle_nn::layer_norm(config.embed_dim, 1e-5, vb_layer.pp("norm2"))?,
                ff_up: candle_nn::linear(
                    config.embed_dim,
                    4 * config.embed_dim,
                    vb_layer.pp("ff_up"),
                )?,
                ff_down: candle_nn::linear(
                    4 * config.embed_dim,
                    config.embed_dim,
                    vb_layer.pp("ff_down"),
                )?,
                num_heads: config.num_heads,
                head_dim,
            };
            layers.push(layer);
        }

        let norm =
            candle_nn::layer_norm(config.embed_dim, 1e-5, vb.pp("final_norm"))?;
        let output_proj = candle_nn::linear(
            config.embed_dim,
            config.vocab_size,
            vb.pp("output"),
        )?;

        Ok(Self {
            config: config.clone(),
            token_embed,
            pos_embed,
            layers,
            norm,
            output_proj,
            device,
            varmap: None,
        })
    }

    // ---- helpers ------------------------------------------------------------

    /// Build a causal (lower-triangular) attention mask of shape
    /// `(1, seq_len, seq_len)` so allowed positions are `0` and masked
    /// positions are `-inf`.
    fn causal_mask(seq_len: usize, device: &Device) -> candle_core::Result<Tensor> {
        // Build mask manually: for i < j (upper triangle excl diag), mask = -inf.
        let mut data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j > i {
                    data[i * seq_len + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask =
            Tensor::from_vec(data, (seq_len, seq_len), device)?;
        // Unsqueeze to (1, seq_len, seq_len) for broadcasting over heads.
        mask.unsqueeze(0)
    }

    /// Convert a 2-D probability tensor `(seq_len, vocab_size)` into
    /// a `Vec<[f32; 256]>`.
    fn probs_tensor_to_vec(
        probs: &Tensor,
        seq_len: usize,
    ) -> Result<Vec<[f32; 256]>, PredictorError> {
        let flat = probs.flatten_all()
            .map_err(|e| PredictorError::ForwardPassFailed {
                message: format!("flatten: {e}"),
            })?
            .to_vec1::<f32>()
            .map_err(|e| PredictorError::ForwardPassFailed {
                message: format!("failed to extract probabilities: {e}"),
            })?;
        let vocab_size = flat.len() / seq_len;
        let mut result = Vec::with_capacity(seq_len);
        for chunk in flat.chunks_exact(vocab_size) {
            let mut arr = [0.0f32; 256];
            arr[..vocab_size].copy_from_slice(chunk);
            result.push(arr);
        }
        Ok(result)
    }

    /// Run the forward pass for a sequence of token indices.
    fn forward(&self, tokens: &[i64]) -> candle_core::Result<Tensor> {
        let seq_len = tokens.len();
        let input = Tensor::new(tokens, &self.device)?;
        let positions: Vec<i64> = (0..seq_len as i64).collect();
        let pos = Tensor::new(positions.as_slice(), &self.device)?;

        let tok_emb = self.token_embed.forward(&input)?; // (seq_len, embed_dim)
        let pos_emb = self.pos_embed.forward(&pos)?; // (seq_len, embed_dim)
        let mut x = (tok_emb + pos_emb)?;

        let mask = Self::causal_mask(seq_len, &self.device)?;
        for layer in &self.layers {
            x = layer.forward(&x, &mask)?;
        }

        let x = self.norm.forward(&x)?;
        let logits = self.output_proj.forward(&x)?; // (seq_len, vocab_size)

        candle_nn::ops::softmax(&logits, 1) // (seq_len, vocab_size)
    }
}

impl Predictor for Transformer {
    fn predict(&self, context: &[u8]) -> Result<Vec<[f32; 256]>, PredictorError> {
        let n = context.len();
        if n == 0 {
            return Ok(Vec::new());
        }

        // Causal input: [BOS=0] + context[0..n-1].  Each output[i] predicts
        // context[i] given context[0..i].
        let mut tokens: Vec<i64> = Vec::with_capacity(n);
        tokens.push(0); // BOS
        tokens.extend(context[..n - 1].iter().map(|&b| b as i64));

        let probs = self.forward(&tokens).map_err(|e| {
            PredictorError::ForwardPassFailed {
                message: format!("transformer forward: {e}"),
            }
        })?;

        Self::probs_tensor_to_vec(&probs, n)
    }

    fn predict_single(&self, context: &[u8]) -> Result<[f32; 256], PredictorError> {
        let n = context.len();
        // [BOS=0] + context — length n+1.  Last output predicts byte after context.
        let mut tokens: Vec<i64> = Vec::with_capacity(n + 1);
        tokens.push(0);
        tokens.extend(context.iter().map(|&b| b as i64));

        let probs = self.forward(&tokens).map_err(|e| {
            PredictorError::ForwardPassFailed {
                message: format!("transformer forward (single): {e}"),
            }
        })?;

        // Take last row.
        let last = probs.get(n).map_err(|e| {
            PredictorError::ForwardPassFailed {
                message: format!("extract last prediction: {e}"),
            }
        })?;
        let flat = last.flatten_all()
            .map_err(|e| PredictorError::ForwardPassFailed {
                message: format!("flatten single: {e}"),
            })?
            .to_vec1::<f32>()
            .map_err(|e| PredictorError::ForwardPassFailed {
                message: format!("failed to extract single probabilities: {e}"),
            })?;

        let mut arr = [0.0f32; 256];
        arr[..flat.len()].copy_from_slice(&flat);
        Ok(arr)
    }
}

// ---- StubPredictor ---------------------------------------------------------

/// A stub predictor that returns a uniform distribution `[1.0/256; 256]`
/// for every position, regardless of context.
///
/// Useful for integration testing the engine pipeline without a trained model.
pub struct StubPredictor;

impl Predictor for StubPredictor {
    fn predict(&self, context: &[u8]) -> Result<Vec<[f32; 256]>, PredictorError> {
        let uniform = [1.0f32 / 256.0; 256];
        Ok(vec![uniform; context.len()])
    }

    fn predict_single(&self, _context: &[u8]) -> Result<[f32; 256], PredictorError> {
        Ok([1.0f32 / 256.0; 256])
    }
}

// ---- Tests ----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ----------------------------------------------------------------- StubPredictor

    #[test]
    fn stub_predict_returns_correct_shape() {
        let stub = StubPredictor;
        let ctx: Vec<u8> = (0..100).map(|i| i as u8).collect();
        let dists = stub.predict(&ctx).unwrap();
        assert_eq!(dists.len(), 100);
        for row in &dists {
            let sum: f32 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.001,
                "row sum {sum} not within 0.001 of 1.0"
            );
        }
    }

    #[test]
    fn stub_predict_empty_input() {
        let stub = StubPredictor;
        let dists = stub.predict(&[]).unwrap();
        assert!(dists.is_empty());
    }

    #[test]
    fn stub_predict_single_is_uniform() {
        let stub = StubPredictor;
        let d = stub.predict_single(&[1, 2, 3]).unwrap();
        let sum: f32 = d.iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
        let expected = 1.0f32 / 256.0;
        assert!((d[0] - expected).abs() < 1e-7);
    }

    #[test]
    fn stub_single_length_input() {
        let stub = StubPredictor;
        let dists = stub.predict(&[42]).unwrap();
        assert_eq!(dists.len(), 1);
        let sum: f32 = dists[0].iter().sum();
        assert!((sum - 1.0).abs() < 0.001);
    }

    // ----------------------------------------------------------------- Config

    #[test]
    fn config_defaults_match_spec() {
        let c = Config::default();
        assert_eq!(c.num_layers, 8, "num_layers");
        assert_eq!(c.embed_dim, 384, "embed_dim");
        assert_eq!(c.num_heads, 6, "num_heads");
        assert_eq!(c.context_window, 512, "context_window");
        assert_eq!(c.vocab_size, 256, "vocab_size");
    }

    #[test]
    fn config_serde_round_trip() {
        let c = Config::default();
        let json = serde_json::to_string(&c).unwrap();
        let c2: Config = serde_json::from_str(&json).unwrap();
        assert_eq!(c, c2, "config should round-trip through JSON");
    }

    #[test]
    fn config_partial_deser_uses_defaults() {
        let json = r#"{"num_layers": 2}"#;
        let c: Config = serde_json::from_str(json).unwrap();
        assert_eq!(c.num_layers, 2);
        assert_eq!(c.embed_dim, 384); // default
        assert_eq!(c.num_heads, 6); // default
        assert_eq!(c.context_window, 512); // default
        assert_eq!(c.vocab_size, 256); // default
    }

    // ----------------------------------------------------------------- Transformer

    /// Check that a freshly-initialised transformer produces valid
    /// probability distributions (rows sum to 1.0).
    #[test]
    fn transformer_random_weights_produces_valid_distributions() {
        let config = Config {
            num_layers: 2,
            embed_dim: 64,
            num_heads: 4,
            context_window: 256,
            vocab_size: 256,
        };
        let t = Transformer::new(&config).unwrap();

        let ctx: Vec<u8> = (0..50).map(|i| (i * 5 + 17) as u8).collect();
        let dists = t.predict(&ctx).unwrap();

        assert_eq!(dists.len(), 50, "should return one dist per input byte");
        for (i, row) in dists.iter().enumerate() {
            let sum: f32 = row.iter().sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "row {i} sum {sum} not within 0.01 of 1.0"
            );
        }
    }

    #[test]
    fn transformer_predict_single_is_valid_distribution() {
        let config = Config {
            num_layers: 2,
            embed_dim: 64,
            num_heads: 4,
            context_window: 256,
            vocab_size: 256,
        };
        let t = Transformer::new(&config).unwrap();
        let d = t.predict_single(&[1, 2, 3, 4]).unwrap();
        let sum: f32 = d.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[test]
    fn transformer_empty_context_returns_empty() {
        let config = Config {
            num_layers: 1,
            embed_dim: 32,
            num_heads: 2,
            context_window: 64,
            vocab_size: 256,
        };
        let t = Transformer::new(&config).unwrap();
        let dists = t.predict(&[]).unwrap();
        assert!(dists.is_empty());
    }

    #[test]
    fn trait_object_compiles() {
        // Verify that both implementations work through the trait.
        fn accept_predictor(_p: &dyn Predictor) {}
        accept_predictor(&StubPredictor);

        let config = Config {
            num_layers: 1,
            embed_dim: 32,
            num_heads: 2,
            context_window: 64,
            vocab_size: 256,
        };
        let t = Transformer::new(&config).unwrap();
        accept_predictor(&t);
    }

    #[test]
    fn stub_predictor_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<StubPredictor>();
        // Transformer is also Send+Sync because candle types are.
        assert_send_sync::<Transformer>();
    }

    // ----------------------------------------------------------------- save / load_from_buffer tests

    /// Create a tiny config so model construction and prediction is fast.
    fn tiny_config() -> Config {
        Config {
            num_layers: 1,
            embed_dim: 32,
            num_heads: 2,
            context_window: 64,
            vocab_size: 256,
        }
    }

    /// Predictions for a fixed context — used as a fingerprint to verify
    /// that different loading paths produce identical outputs.
    fn predict_fingerprint(model: &Transformer) -> Vec<[f32; 256]> {
        let ctx: Vec<u8> = (0..20).map(|i| (i * 13 + 7) as u8).collect();
        model.predict(&ctx).unwrap()
    }

    #[test]
    fn save_and_load_from_buffer_round_trip() {
        let config = tiny_config();
        let model = Transformer::new(&config).unwrap();

        // Capture predictions from the fresh model.
        let fresh_fingerprint = predict_fingerprint(&model);

        // Save to a temp file.
        let tmp = std::env::temp_dir().join(format!(
            "heretek_t01_roundtrip_{}.safetensors",
            std::process::id()
        ));
        model.save(&tmp).unwrap();

        // Read the saved file into a buffer.
        let data = std::fs::read(&tmp).unwrap();

        // Load from buffer.
        let reloaded = Transformer::load_from_buffer(&data, &config).unwrap();
        let reloaded_fingerprint = predict_fingerprint(&reloaded);

        // Predictions must match exactly.
        assert_eq!(
            fresh_fingerprint.len(),
            reloaded_fingerprint.len(),
            "fingerprint lengths must match"
        );
        for (i, (a, b)) in fresh_fingerprint
            .iter()
            .zip(reloaded_fingerprint.iter())
            .enumerate()
        {
            for (j, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (va - vb).abs() < 1e-6,
                    "mismatch at position ({i}, class {j}): fresh={va}, reloaded={vb}"
                );
            }
        }

        // Clean up.
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn corrupted_buffer_is_rejected() {
        let config = tiny_config();
        // Feed invalid safetensors data (just random bytes, not a valid header).
        let garbage: Vec<u8> = (0..256).map(|i| i as u8).collect();
        let result = Transformer::load_from_buffer(&garbage, &config);
        assert!(
            result.is_err(),
            "load_from_buffer must reject corrupted data"
        );
    }

    #[test]
    fn save_on_buffer_loaded_model_returns_error() {
        let config = tiny_config();
        let model = Transformer::new(&config).unwrap();

        // Save to temp, read back, load from buffer.
        let tmp = std::env::temp_dir().join(format!(
            "heretek_t01_save_error_{}.safetensors",
            std::process::id()
        ));
        model.save(&tmp).unwrap();
        let data = std::fs::read(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);

        let buffer_model = Transformer::load_from_buffer(&data, &config).unwrap();

        // Attempting to save should fail.
        let out_path = std::env::temp_dir().join(format!(
            "heretek_t01_should_not_exist_{}.safetensors",
            std::process::id()
        ));
        let result = buffer_model.save(&out_path);
        assert!(
            result.is_err(),
            "save on buffer-loaded model must return an error"
        );
        let _ = std::fs::remove_file(&out_path);
    }

    #[test]
    fn load_from_buffer_rejects_mismatched_config() {
        let config = tiny_config();
        let model = Transformer::new(&config).unwrap();

        let tmp = std::env::temp_dir().join(format!(
            "heretek_t01_cfgchk_{}.safetensors",
            std::process::id()
        ));
        model.save(&tmp).unwrap();
        let data = std::fs::read(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);

        // Use a config with different embed_dim — the tensor shapes won't match.
        let bad_config = Config {
            embed_dim: 64, // original was 32
            ..config
        };
        let result = Transformer::load_from_buffer(&data, &bad_config);
        assert!(
            result.is_err(),
            "load_from_buffer must reject config with incompatible shapes"
        );
    }

    #[test]
    fn new_model_can_save() {
        let config = tiny_config();
        let model = Transformer::new(&config).unwrap();

        let tmp = std::env::temp_dir().join(format!(
            "heretek_t01_new_save_{}.safetensors",
            std::process::id()
        ));
        let result = model.save(&tmp);
        assert!(result.is_ok(), "new model must be savable: {result:?}");

        // Verify the file exists and is non-empty.
        let meta = std::fs::metadata(&tmp).unwrap();
        assert!(meta.len() > 0, "saved file must be non-empty");

        let _ = std::fs::remove_file(&tmp);
    }
}

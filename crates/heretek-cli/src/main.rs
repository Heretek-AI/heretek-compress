//! heretek-cli — command-line interface for heretik-compress.
//!
//! Provides `compress`, `decompress`, and `benchmark` subcommands.
//! Exit codes: 0 = success, 1 = I/O, 2 = format/integrity, 3 = model load failure.

use clap::{Parser, Subcommand};
use std::io::BufWriter;
use std::path::PathBuf;
use std::time::Instant;

use heretek_engine::{compress_to_bytes, decompress_from_bytes};
use heretek_error::HeretikError;
use heretek_predictor::{Config, Predictor, StubPredictor, Transformer};

// ---- CLI definition --------------------------------------------------------

#[derive(Parser)]
#[command(name = "heretek", about = "Neural compression engine", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compress a file.
    Compress {
        /// Input file to compress.
        input: PathBuf,
        /// Output file for compressed data.
        output: PathBuf,
        /// Path to model weights (safetensors) — overrides default.
        #[arg(long)]
        model: Option<PathBuf>,
    },
    /// Decompress a file.
    Decompress {
        /// Input file (compressed .hc).
        input: PathBuf,
        /// Output file for decompressed data.
        output: PathBuf,
        /// Path to model weights (safetensors) — overrides default.
        #[arg(long)]
        model: Option<PathBuf>,
    },
    /// Benchmark compression on a file (report ratio + timing).
    Benchmark {
        /// Input file to benchmark.
        input: PathBuf,
        /// Path to model weights (safetensors) — overrides default.
        #[arg(long)]
        model: Option<PathBuf>,
    },
}

// ---- Default model helpers -------------------------------------------------

/// Load the default model from `models/default.*` on disk relative to the
/// `heretek-cli` crate root, or fall back to `StubPredictor` with a warning.
fn load_default_model() -> Box<dyn Predictor> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".into());
    let model_path = PathBuf::from(&manifest_dir).join("../../models/default.safetensors");
    let config_path = PathBuf::from(&manifest_dir).join("../../models/default_config.json");

    match try_load_model(&model_path, &config_path) {
        Ok(transformer) => {
            eprintln!(
                "[heretek] loaded default model from {}",
                model_path.display()
            );
            Box::new(transformer)
        }
        Err(e) => {
            eprintln!("[heretek] warning: could not load default model: {e}");
            eprintln!("[heretek] falling back to StubPredictor (uniform distribution).");
            eprintln!("[heretek] compression will not be effective without a trained model.");
            Box::new(StubPredictor)
        }
    }
}

/// Try to load a Transformer from safetensors + config.json on disk.
fn try_load_model(model_path: &PathBuf, config_path: &PathBuf) -> Result<Transformer, String> {
    if !model_path.exists() {
        return Err(format!("{} not found", model_path.display()));
    }
    if !config_path.exists() {
        return Err(format!("{} not found", config_path.display()));
    }

    let config_json = std::fs::read_to_string(config_path)
        .map_err(|e| format!("failed to read config: {e}"))?;
    let config: Config = serde_json::from_str(&config_json)
        .map_err(|e| format!("failed to parse config: {e}"))?;

    let data = std::fs::read(model_path)
        .map_err(|e| format!("failed to read model file: {e}"))?;
    Transformer::load_from_buffer(&data, &config)
        .map_err(|e| format!("model load failed: {e}"))
}

/// Load a model from a user-supplied `--model` path.
fn load_explicit_model(path: &PathBuf) -> Result<Box<dyn Predictor>, String> {
    // Derive the config path by replacing .safetensors with _config.json,
    // or looking side-by-side for default_config.json.
    let config_path = if path
        .file_name()
        .map(|n| n == "default.safetensors")
        .unwrap_or(false)
    {
        path.with_file_name("default_config.json")
    } else {
        // Try the same directory for default_config.json.
        if let Some(parent) = path.parent() {
            parent.join("default_config.json")
        } else {
            PathBuf::from("default_config.json")
        }
    };

    if !config_path.exists() {
        return Err(format!("model config not found at {}", config_path.display()));
    }

    let config_json = std::fs::read_to_string(&config_path)
        .map_err(|e| format!("failed to read config: {e}"))?;
    let config: Config = serde_json::from_str(&config_json)
        .map_err(|e| format!("failed to parse config: {e}"))?;

    let data = std::fs::read(path)
        .map_err(|e| format!("failed to read model: {e}"))?;
    let transformer = Transformer::load_from_buffer(&data, &config)
        .map_err(|e| format!("model load failed: {e}"))?;

    Ok(Box::new(transformer))
}

// ---- Exit code mapping -----------------------------------------------------

fn map_error_to_exit_code(e: &anyhow::Error) -> i32 {
    // Walk the chain looking for structured error types.
    for cause in e.chain() {
        // Try to downcast to HeretikError.
        if let Some(he) = cause.downcast_ref::<HeretikError>() {
            match he {
                HeretikError::Format(_) | HeretikError::Integrity(_) => return 2,
                HeretikError::Predictor(_) => return 3,
                HeretikError::Coder(_) => return 2,
            }
        }
    }
    // Default: I/O or unknown → exit 1.
    1
}

// ---- Command implementations -----------------------------------------------

fn cmd_compress(input: &PathBuf, output: &PathBuf, predictor: &dyn Predictor) -> anyhow::Result<()> {
    eprintln!("[heretek] reading input: {}", input.display());
    let data = std::fs::read(input)?;

    eprintln!("[heretek] compressing {} bytes...", data.len());
    let compressed =
        compress_to_bytes(&data, predictor).map_err(|e| anyhow::anyhow!(e))?;

    eprintln!(
        "[heretek] compressed {} → {} bytes (ratio {:.2}%)",
        data.len(),
        compressed.len(),
        100.0 * compressed.len() as f64 / data.len() as f64
    );

    eprintln!("[heretek] writing output: {}", output.display());
    // Use a BufWriter for efficient I/O.
    let file = std::fs::File::create(output)?;
    let mut writer = BufWriter::new(file);
    std::io::Write::write_all(&mut writer, &compressed)?;

    Ok(())
}

fn cmd_decompress(
    input: &PathBuf,
    output: &PathBuf,
    predictor: &dyn Predictor,
) -> anyhow::Result<()> {
    eprintln!("[heretek] reading compressed input: {}", input.display());
    let compressed = std::fs::read(input)?;

    eprintln!("[heretek] decompressing {} bytes...", compressed.len());
    let decompressed =
        decompress_from_bytes(&compressed, predictor).map_err(|e| anyhow::anyhow!(e))?;

    eprintln!(
        "[heretek] decompressed {} → {} bytes",
        compressed.len(),
        decompressed.len()
    );

    eprintln!("[heretek] verifying SHA-256 checksum...");
    // checksum is verified inside decompress_from_bytes — reaching here
    // means it passed.

    eprintln!("[heretek] writing output: {}", output.display());
    // Use BufWriter for efficient I/O.
    let file = std::fs::File::create(output)?;
    let mut writer = BufWriter::new(file);
    std::io::Write::write_all(&mut writer, &decompressed)?;

    Ok(())
}

fn cmd_benchmark(input: &PathBuf, predictor: &dyn Predictor) -> anyhow::Result<()> {
    eprintln!("[heretek] reading input: {}", input.display());
    let data = std::fs::read(input)?;
    let original_size = data.len();

    eprintln!("[heretek] benchmarking compression of {} input bytes...", original_size);
    let start = Instant::now();
    let compressed =
        compress_to_bytes(&data, predictor).map_err(|e| anyhow::anyhow!(e))?;
    let elapsed = start.elapsed();

    let compressed_size = compressed.len();
    let ratio = compressed_size as f64 / original_size as f64;

    println!("   input size:   {original_size:>12} bytes");
    println!("   compressed:   {compressed_size:>12} bytes");
    println!("   ratio:        {ratio:>12.4} ({:.2}%)", ratio * 100.0);
    println!("   wall time:    {elapsed:>12.2?}");

    Ok(())
}

// ---- Tests -----------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use heretek_engine::{compress_to_bytes, decompress_from_bytes};
    use heretek_error::{CoderError, FormatError, HeretikError, IntegrityError, PredictorError};
    use heretek_predictor::{Config, Predictor, StubPredictor, Transformer};
    // ---- Convenience helpers -----------------------------------------------

    fn stub() -> StubPredictor {
        StubPredictor
    }

    /// Wrap a `HeretikError` in an `anyhow::Error` matching the pattern used
    /// by `cmd_compress` / `cmd_decompress`.
    fn wrap_err(e: HeretikError) -> anyhow::Error {
        anyhow::anyhow!(e)
    }

    // ---- Round-trip tests (via engine free functions) ---------------------

    #[test]
    fn round_trip_empty() {
        let data = &[];
        let compressed = compress_to_bytes(data, &stub()).expect("compress empty");
        let recovered = decompress_from_bytes(&compressed, &stub()).expect("decompress empty");
        assert_eq!(recovered, data);
    }

    #[test]
    fn round_trip_single_byte() {
        for byte in [0u8, 42, 128, 255] {
            let data = &[byte];
            let compressed = compress_to_bytes(data, &stub()).expect("compress single");
            let recovered = decompress_from_bytes(&compressed, &stub()).expect("decompress single");
            assert_eq!(recovered, data, "mismatch for byte {byte}");
        }
    }

    #[test]
    fn round_trip_1kb_text() {
        // 1 KB of printable ASCII (deterministic repeating pattern).
        let data: Vec<u8> = (0..1024).map(|i| (i % 95 + 32) as u8).collect();
        let compressed = compress_to_bytes(&data, &stub()).expect("compress 1KB");
        let recovered = decompress_from_bytes(&compressed, &stub()).expect("decompress 1KB");
        assert_eq!(recovered, data);
    }

    #[test]
    fn round_trip_64kb_random() {
        // 64 KB of pseudo-random bytes (deterministic for reproducibility).
        let mut data = vec![0u8; 65536];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i.wrapping_mul(17).wrapping_add(13) ^ (i >> 3)) as u8;
        }
        let compressed = compress_to_bytes(&data, &stub()).expect("compress 64KB");
        let recovered = decompress_from_bytes(&compressed, &stub()).expect("decompress 64KB");
        assert_eq!(recovered, data);
    }

    // ---- Round-trip via CLI command functions (temp files) -----------------

    #[test]
    fn cli_compress_decompress_round_trip() {
        let dir =
            std::env::temp_dir().join(format!("heretek_t04_cli_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let input_path = dir.join("input.bin");
        let compressed_path = dir.join("output.hc");
        let restored_path = dir.join("restored.bin");

        let original: Vec<u8> = (0..2048).map(|i| (i % 256) as u8).collect();
        std::fs::write(&input_path, &original).unwrap();

        let predictor = stub();
        cmd_compress(&input_path, &compressed_path, &predictor).expect("compress via CLI");
        cmd_decompress(&compressed_path, &restored_path, &predictor).expect("decompress via CLI");

        let restored = std::fs::read(&restored_path).unwrap();
        assert_eq!(restored, original, "CLI round-trip mismatch");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- Exit code mapping -------------------------------------------------

    #[test]
    fn exit_code_format_truncated_header_maps_to_2() {
        let e = wrap_err(FormatError::TruncatedHeader.into());
        assert_eq!(map_error_to_exit_code(&e), 2);
    }

    #[test]
    fn exit_code_format_invalid_magic_maps_to_2() {
        let e = wrap_err(
            FormatError::InvalidMagic {
                found: heretek_error::MagicBytes([0x00, 0x00, 0x00, 0x00]),
            }
            .into(),
        );
        assert_eq!(map_error_to_exit_code(&e), 2);
    }

    #[test]
    fn exit_code_format_unsupported_version_maps_to_2() {
        let e = wrap_err(
            FormatError::UnsupportedVersion { version: 99 }.into(),
        );
        assert_eq!(map_error_to_exit_code(&e), 2);
    }

    #[test]
    fn exit_code_model_load_failed_maps_to_3() {
        let e = wrap_err(
            PredictorError::ModelLoadFailed {
                path: "test_model.bin".into(),
                message: "file not found".into(),
            }
            .into(),
        );
        assert_eq!(map_error_to_exit_code(&e), 3);
    }

    #[test]
    fn exit_code_checksum_mismatch_maps_to_2() {
        let e = wrap_err(
            IntegrityError::ChecksumMismatch {
                expected: heretek_error::Hex32([0xAA; 32]),
                actual: heretek_error::Hex32([0xBB; 32]),
            }
            .into(),
        );
        assert_eq!(map_error_to_exit_code(&e), 2);
    }

    #[test]
    fn exit_code_coder_stack_underflow_maps_to_2() {
        let e = wrap_err(CoderError::StackUnderflow { at_byte: 0 }.into());
        assert_eq!(map_error_to_exit_code(&e), 2);
    }

    #[test]
    fn exit_code_plain_io_error_maps_to_1() {
        // A plain std::io::Error that is NOT wrapped in a HeretikError.
        let e = anyhow::anyhow!(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "no such file or directory"
        ));
        assert_eq!(
            map_error_to_exit_code(&e),
            1,
            "bare I/O error must default to exit code 1"
        );
    }

    // ---- Model load from buffer --------------------------------------------

    /// Creates a tiny transformer, saves to disk, reads back, and loads via
    /// `load_from_buffer`.  Uses a dynamically-constructed model rather than
    /// `include_bytes!` because the trained default model is 58 MB — embedding
    /// it at compile time would inflate the test binary and slow every build.
    #[test]
    fn model_load_from_buffer_produces_valid_distributions() {
        let config = Config {
            num_layers: 1,
            embed_dim: 32,
            num_heads: 2,
            context_window: 64,
            vocab_size: 256,
        };
        let model = Transformer::new(&config).expect("create tiny model");

        // Save to temp file, read back as buffer.
        let tmp = std::env::temp_dir().join(format!(
            "heretek_t04_buffer_{}.safetensors",
            std::process::id()
        ));
        model.save(&tmp).expect("save model");
        let data = std::fs::read(&tmp).expect("read saved model");
        let _ = std::fs::remove_file(&tmp);

        let loaded =
            Transformer::load_from_buffer(&data, &config).expect("load from buffer");

        // Forward pass on various context lengths.
        for ctx_len in &[0usize, 1, 16, 64] {
            let ctx: Vec<u8> = (0..*ctx_len).map(|i| (i * 37 + 13) as u8).collect();
            let dists = loaded.predict(&ctx).expect("predict must succeed");
            assert_eq!(dists.len(), *ctx_len);
            for (i, row) in dists.iter().enumerate() {
                let sum: f32 = row.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 0.01,
                    "row {i} sum {sum} not close to 1.0"
                );
            }
        }

        let single = loaded
            .predict_single(&[10, 20, 30])
            .expect("predict_single");
        let sum: f32 = single.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    // ---- Corrupted compressed data → exit code 2 --------------------------

    #[test]
    fn corrupted_compressed_data_exit_code_2() {
        let data = b"data that will be corrupted in transit";
        let mut compressed = compress_to_bytes(data, &stub()).expect("compress");

        // Bit-flip deep in the payload (byte 65 — past the 60-byte header).
        if compressed.len() > 65 {
            compressed[65] ^= 0x01;
        } else {
            let last = compressed.len() - 1;
            compressed[last] ^= 0x01;
        }

        let result = decompress_from_bytes(&compressed, &stub());
        assert!(result.is_err(), "corrupted data must fail decompression");

        // Wrap the same way cmd_decompress does.
        let exit_code = map_error_to_exit_code(&anyhow::anyhow!(result.unwrap_err()));
        assert_eq!(
            exit_code, 2,
            "checksum mismatch from corruption must map to exit code 2"
        );
    }

    // ---- Missing input file → exit code 1 ----------------------------------

    #[test]
    fn missing_input_file_exit_code_1() {
        let missing =
            std::path::PathBuf::from("/tmp/heretek_definitely_missing_xyz123.bin");
        // Ensure it really doesn't exist.
        let _ = std::fs::remove_file(&missing);

        let result = cmd_compress(
            &missing,
            &std::path::PathBuf::from("/tmp/heretek_should_not_exist_out.hc"),
            &stub(),
        );
        assert!(result.is_err(), "missing input file must fail");
        let exit_code = map_error_to_exit_code(&result.unwrap_err());
        assert_eq!(exit_code, 1, "file-not-found I/O error must map to exit code 1");
    }

    // ---- Exit code matrix: table-driven coverage --------------------------

    #[test]
    fn exit_code_matrix_complete() {
        let cases: Vec<(&str, HeretikError, i32)> = vec![
            (
                "FormatError::TruncatedHeader",
                FormatError::TruncatedHeader.into(),
                2,
            ),
            (
                "FormatError::InvalidMagic",
                FormatError::InvalidMagic {
                    found: heretek_error::MagicBytes([0xDE, 0xAD, 0xBE, 0xEF]),
                }
                .into(),
                2,
            ),
            (
                "FormatError::UnsupportedVersion",
                FormatError::UnsupportedVersion { version: 99 }.into(),
                2,
            ),
            (
                "FormatError::CorruptBitstream",
                FormatError::CorruptBitstream { offset: 2048 }.into(),
                2,
            ),
            (
                "PredictorError::ModelLoadFailed",
                PredictorError::ModelLoadFailed {
                    path: "m".into(),
                    message: "e".into(),
                }
                .into(),
                3,
            ),
            (
                "PredictorError::InvalidDistribution",
                PredictorError::InvalidDistribution {
                    index: 0,
                    sum: 0.5,
                }
                .into(),
                3,
            ),
            (
                "PredictorError::ForwardPassFailed",
                PredictorError::ForwardPassFailed {
                    message: "NaN".into(),
                }
                .into(),
                3,
            ),
            (
                "IntegrityError::ChecksumMismatch",
                IntegrityError::ChecksumMismatch {
                    expected: heretek_error::Hex32([0x00; 32]),
                    actual: heretek_error::Hex32([0xFF; 32]),
                }
                .into(),
                2,
            ),
            (
                "IntegrityError::RoundTripMismatch",
                IntegrityError::RoundTripMismatch {
                    offset: 0,
                    expected_byte: 0xAB,
                    actual_byte: 0xCD,
                }
                .into(),
                2,
            ),
            (
                "CoderError::StackUnderflow",
                CoderError::StackUnderflow { at_byte: 42 }.into(),
                2,
            ),
            (
                "CoderError::EncodeFailed",
                CoderError::EncodeFailed {
                    message: "test".into(),
                }
                .into(),
                2,
            ),
            (
                "CoderError::DecodeFailed",
                CoderError::DecodeFailed {
                    message: "test".into(),
                }
                .into(),
                2,
            ),
        ];

        for (name, error, expected_code) in cases {
            let e = wrap_err(error);
            let code = map_error_to_exit_code(&e);
            assert_eq!(
                code, expected_code,
                "{name}: expected exit code {expected_code}, got {code}"
            );
        }
    }
}

// ---- main ------------------------------------------------------------------

fn main() {
    let cli = Cli::parse();

    let result = match &cli.command {
        Command::Compress {
            input,
            output,
            model,
        } => {
            let predictor: Box<dyn Predictor> = match model {
                Some(p) => match load_explicit_model(p) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("[heretek] error: {e}");
                        std::process::exit(3);
                    }
                },
                None => load_default_model(),
            };
            cmd_compress(input, output, predictor.as_ref())
        }
        Command::Decompress {
            input,
            output,
            model,
        } => {
            let predictor: Box<dyn Predictor> = match model {
                Some(p) => match load_explicit_model(p) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("[heretek] error: {e}");
                        std::process::exit(3);
                    }
                },
                None => load_default_model(),
            };
            cmd_decompress(input, output, predictor.as_ref())
        }
        Command::Benchmark { input, model } => {
            let predictor: Box<dyn Predictor> = match model {
                Some(p) => match load_explicit_model(p) {
                    Ok(m) => m,
                    Err(e) => {
                        eprintln!("[heretek] error: {e}");
                        std::process::exit(3);
                    }
                },
                None => load_default_model(),
            };
            cmd_benchmark(input, predictor.as_ref())
        }
    };

    if let Err(e) = result {
        let code = map_error_to_exit_code(&e);
        eprintln!("[heretek] error: {e:?}");
        std::process::exit(code);
    }
}

fn main() {
    // If the default model exists at compile time, tell Cargo to re-check
    // when it changes and expose its directory as an env var.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let model_path = std::path::PathBuf::from(&manifest_dir)
        .join("../../models/default.safetensors");
    if model_path.exists() {
        println!("cargo:rustc-cfg=default_model_available");
        println!("cargo:rerun-if-changed={}", model_path.display());
    }
    // Always rerun if the config changes (tiny file, could be embedded).
    let config_path = std::path::PathBuf::from(&manifest_dir)
        .join("../../models/default_config.json");
    if config_path.exists() {
        println!("cargo:rerun-if-changed={}", config_path.display());
    }
}

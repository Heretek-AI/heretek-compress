#!/usr/bin/env python3
"""Training pipeline for the heretik byte-level autoregressive transformer.

Downloads enwik8, trains a decoder-only transformer on next-byte prediction,
and exports weights in safetensors format with a companion config.json.

Usage:
    python training/train.py --epochs 1 --limit 1000 --output-dir models/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from safetensors.torch import save_file
from tqdm import tqdm

from model import Config, Transformer


# ---- Data ------------------------------------------------------------------

def download_enwik8(output_path: Path) -> Path:
    """Download enwik8 (100 MB of English Wikipedia markup, Hutter Prize).

    Cached locally so repeated runs don't re-download.
    """
    output_path.mkdir(parents=True, exist_ok=True)
    url = "https://mattmahoney.net/dc/enwik8.zip"
    zip_path = output_path / "enwik8.zip"
    raw_path = output_path / "enwik8"

    if raw_path.exists():
        print(f"Using cached enwik8 at {raw_path}", file=sys.stderr)
        return raw_path

    if not zip_path.exists():
        print(f"Downloading enwik8 from {url} ...", file=sys.stderr)
        response = requests.get(url, stream=True)
        total = int(response.headers.get("content-length", 0))
        with open(zip_path, "wb") as f:
            with tqdm(total=total, unit="B", unit_scale=True, desc="enwik8.zip") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))

    print("Extracting enwik8 ...", file=sys.stderr)
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract("enwik8", output_path)

    return raw_path


def build_dataset(raw_bytes: bytes, context_window: int, limit: int = 0) -> torch.Tensor:
    """Chunk byte stream into context_window-byte segments and prepend BOS token (0).

    Each segment is [BOS=0, b0, b1, ..., b_{context_window-1}] — exactly
    context_window+1 tokens.  During training, the causal model predicts
    tokens[1:] from probs[0:-1], so the loss covers all context_window bytes.

    Args:
        raw_bytes: Raw byte sequence.
        context_window: Segment length in bytes.
        limit: If > 0, cap the number of segments (for smoke testing).

    Returns:
        Tensor of shape (num_segments, context_window + 1) where column 0 is BOS
        and columns 1.. are the context bytes.
    """
    seg_len = context_window + 1  # BOS + context_window bytes
    max_chunks = max(0, len(raw_bytes) // context_window)
    if limit > 0:
        max_chunks = min(max_chunks, limit)

    segments = []
    for i in range(max_chunks):
        start = i * context_window
        chunk = raw_bytes[start : start + context_window]
        if len(chunk) < context_window:
            break
        tokens = [0] + list(chunk)  # BOS + context_window bytes
        segments.append(tokens)

    if not segments:
        return torch.empty((0, seg_len), dtype=torch.long)

    return torch.tensor(segments, dtype=torch.long)


# ---- Training --------------------------------------------------------------

def train(
    model: Transformer,
    data: torch.Tensor,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> list[float]:
    """Train the model on byte-level next-byte prediction.

    Returns list of per-batch losses for diagnostics.
    """
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()

    num_samples = data.shape[0]
    losses: list[float] = []

    for epoch in range(epochs):
        # Shuffle segments each epoch
        perm = torch.randperm(num_samples)
        data_shuffled = data[perm]

        num_batches = max(1, num_samples // batch_size)
        pbar = tqdm(
            range(0, num_samples, batch_size),
            total=num_batches,
            desc=f"epoch {epoch + 1}/{epochs}",
            unit="batch",
            file=sys.stderr,
        )

        batch_losses: list[float] = []
        for start in pbar:
            end = min(start + batch_size, num_samples)
            batch = data_shuffled[start:end].to(device)  # (B, seq_len + 1)

            # Input: [BOS, byte_0, ..., byte_{context_window-1}]
            # Target: next byte at each position (shifted left by 1)
            # For position i (0-indexed), model sees tokens[0..i], predicts tokens[i+1]
            # We use causal self-attention so the forward pass handles this internally:
            # forward(tokens) -> probs; probs[i] predicts token[i] given tokens[0..i].
            # So we reshape batch into one long sequence per element and accumulate loss.
            total_loss = torch.tensor(0.0, device=device)
            count = 0
            for b in range(batch.shape[0]):
                # tokens: [BOS=0, byte_0, byte_1, ..., byte_{context_window}]
                tokens = batch[b]  # (context_window + 1,)
                # forward: (seq_len,) -> (seq_len, vocab_size)
                # probs[i] predicts tokens[i] given tokens[0..i]
                probs = model(tokens)  # (context_window+1, vocab_size)
                # Cross-entropy: at position i, predict token[i] given prior context
                # We predict all positions 0..context_window → target is tokens[0..context_window]
                loss = criterion(probs[:-1], tokens[1:])  # predict tokens[1:] from probs[0..-2]
                total_loss = total_loss + loss
                count += 1

            avg_loss = total_loss / count
            optimizer.zero_grad()
            avg_loss.backward()
            # Clip gradients for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            loss_val = avg_loss.item()
            batch_losses.append(loss_val)
            pbar.set_postfix(loss=f"{loss_val:.4f}")

        epoch_avg = sum(batch_losses) / max(len(batch_losses), 1)
        losses.extend(batch_losses)
        print(f"epoch {epoch + 1}/{epochs}  avg_loss={epoch_avg:.4f}", file=sys.stderr)

    return losses


# ---- Export ----------------------------------------------------------------

def export_model(
    model: Transformer,
    config: Config,
    output_dir: Path,
    name: str = "default",
) -> tuple[Path, Path]:
    """Export model weights as safetensors and config as JSON.

    Returns (safetensors_path, config_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save safetensors
    tensor_dict = model._tensor_dict()
    safetensors_path = output_dir / f"{name}.safetensors"
    save_file(tensor_dict, str(safetensors_path))

    # Save config JSON
    config_path = output_dir / f"{name}_config.json"
    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)

    return safetensors_path, config_path


# ---- CLI -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train heretik byte-level transformer on enwik8"
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs (default: 1)"
    )
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max training segments; 0 = use all available (default: 0)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models/",
        help="Output directory for model + config (default: models/)"
    )
    args = parser.parse_args()

    print("=== heretik training pipeline ===", file=sys.stderr)
    print(f"epochs={args.epochs}  limit={args.limit}  batch_size={args.batch_size}  lr={args.lr}", file=sys.stderr)

    config = Config()
    print(f"Model config: {config.to_dict()}", file=sys.stderr)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", file=sys.stderr)

    # ---- Load data ----------------------------------------------------------
    print("--- Loading data ---", file=sys.stderr)
    cache_dir = Path(args.output_dir) / ".cache"
    raw_path = download_enwik8(cache_dir)
    raw_bytes = raw_path.read_bytes()
    print(f"Loaded {len(raw_bytes):,} bytes from {raw_path}", file=sys.stderr)

    data = build_dataset(raw_bytes, config.context_window, limit=args.limit)
    print(f"Training data: {data.shape[0]} segments of length {data.shape[1]}", file=sys.stderr)

    if data.shape[0] == 0:
        print("No training data — nothing to do.", file=sys.stderr)
        return

    # ---- Train --------------------------------------------------------------
    print("--- Training ---", file=sys.stderr)
    model = Transformer(config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}", file=sys.stderr)

    losses = train(model, data, args.epochs, args.batch_size, args.lr, device)
    final_loss = losses[-1] if losses else float("inf")
    print(f"Training complete. Final batch loss: {final_loss:.4f}", file=sys.stderr)

    # ---- Export -------------------------------------------------------------
    print("--- Exporting model ---", file=sys.stderr)
    safetensors_path, config_path = export_model(
        model, config, Path(args.output_dir)
    )
    st_size = safetensors_path.stat().st_size
    print(f"Exported: {safetensors_path} ({st_size:,} bytes)", file=sys.stderr)
    print(f"Config:   {config_path}", file=sys.stderr)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()

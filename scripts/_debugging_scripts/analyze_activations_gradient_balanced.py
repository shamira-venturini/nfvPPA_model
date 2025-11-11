#!/usr/bin/env python3
"""
analyze_activations_gradient_balanced.py
----------------------------------------
Compute per-neuron attribution using Gradient x Activation for a BALANCED priming analysis.
This is a fast proxy for causal effect, suitable for layer sweeps and initial neuron ranking.

Usage:
 python analyze_activations_gradient_balanced.py \
   --model gpt2-large \
   --corpus_file data/PILOT_BALANCED_transitive_500.csv \
   --output_dir results/GRADIENT_ATTRIBUTION_balanced/balanced_transitive_L12 \
   --layer_idx 12
"""

import argparse
import os
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)


def tokenize_row(tokenizer, text):
    """Tokenizes a string of text and returns tensor on the correct device."""
    # The script assumes tokens are space-separated in the CSV.
    toks = text.strip().split()
    ids = tokenizer.convert_tokens_to_ids(toks)
    return torch.tensor([ids], dtype=torch.long).to(DEVICE)


def compute_gradient_attribution(model, input_ids, layer_idx):
    """
    Computes Gradient x Activation for a given input and layer.
    Returns a numpy vector of attribution scores for each neuron.
    """
    model.zero_grad()

    target_tensor = None

    def hook_fn(module, inp, out):
        nonlocal target_tensor
        out.retain_grad()  # Ensure gradient is saved for this tensor
        target_tensor = out

    # Hook to the MLP output (c_proj) of the specified layer
    handle = model.transformer.h[layer_idx].mlp.c_proj.register_forward_hook(hook_fn)

    # 1. Forward pass to get the log probability of the entire sequence
    outputs = model(input_ids)
    logits = outputs.logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_ids = input_ids[0]

    if token_ids.shape[0] <= 1:
        handle.remove()
        return np.zeros(model.config.n_embd, dtype=np.float32)

    # Sum log-probs of all tokens (excluding the first token's prediction)
    target_log_prob = log_probs[0, :-1, :].gather(1, token_ids[1:].unsqueeze(-1)).sum()

    # 2. Backward pass to compute gradients
    target_log_prob.backward()

    handle.remove()  # Clean up the hook immediately

    if target_tensor is None or target_tensor.grad is None:
        return np.zeros(model.config.n_embd, dtype=np.float32)

    # 3. Calculate Gradient x Activation
    # We sum across the sequence dimension to get one score per neuron
    grad_x_activation = (target_tensor.grad * target_tensor).squeeze(0).sum(dim=0)

    return grad_x_activation.detach().cpu().numpy()


def run_balanced_gradient_analysis(model, tokenizer, corpus_df, layer_idx, output_dir, sample_limit=None):
    """
    Runs the full balanced analysis, reading from all four columns (x_px, x_py, y_px, y_py).
    """
    ensure_dir(output_dir)

    attribution_matrix = []
    summary_data = []

    # Limit the number of rows if a sample_limit is provided
    if sample_limit:
        corpus_df = corpus_df.head(sample_limit)

    iterator = tqdm(corpus_df.iterrows(), total=len(corpus_df), desc=f"Layer {layer_idx} Gradient Attribution")

    for idx, row in iterator:
        # Tokenize all four conditions from the row
        ids_px = tokenize_row(tokenizer, row["x_px"])
        ids_py = tokenize_row(tokenizer, row["x_py"])
        ids_y_px = tokenize_row(tokenizer, row["y_px"])
        ids_y_py = tokenize_row(tokenizer, row["y_py"])

        # Calculate Gradient Attribution for all four sentences
        attr_px = compute_gradient_attribution(model, ids_px, layer_idx)
        attr_py = compute_gradient_attribution(model, ids_py, layer_idx)
        attr_y_px = compute_gradient_attribution(model, ids_y_px, layer_idx)
        attr_y_py = compute_gradient_attribution(model, ids_y_py, layer_idx)

        # Calculate the priming effect for each prime type
        priming_effect_active_prime = attr_px - attr_py
        priming_effect_passive_prime = attr_y_py - attr_y_px

        # Average them for a balanced, robust score
        balanced_priming_attribution = (priming_effect_active_prime + priming_effect_passive_prime) / 2.0

        attribution_matrix.append(balanced_priming_attribution)
        summary_data.append({
            "index": idx,
            "attribution_mean": balanced_priming_attribution.mean(),
            "attribution_std": balanced_priming_attribution.std()
        })

    # Save the results
    attribution_matrix = np.stack(attribution_matrix)
    np.save(Path(output_dir) / "gradient_attribution_vectors.npy", attribution_matrix)
    pd.DataFrame(summary_data).to_csv(Path(output_dir) / "gradient_attribution_summary.csv", index=False)

    with open(Path(output_dir) / "metadata.json", "w") as f:
        json.dump({
            "model": model.name_or_path,
            "layer_idx": layer_idx,
            "method": "gradient_x_activation_balanced",
            "samples": len(attribution_matrix)
        }, f, indent=2)

    print(f"\nSaved per-neuron Gradient Attributions: {attribution_matrix.shape} → {output_dir}")
    print("Mean Attribution across dataset:", attribution_matrix.mean())


def main():
    parser = argparse.ArgumentParser(description="Per-neuron Gradient x Activation Attribution (Balanced Analysis)")
    parser.add_argument("--model", type=str, required=True, help="e.g., gpt2-large")
    parser.add_argument("--corpus_file", type=str, required=True, help="Path to the balanced priming CSV file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output files.")
    parser.add_argument("--layer_idx", type=int, required=True, help="The transformer layer to analyze.")
    parser.add_argument("--sample_limit", type=int, default=None,
                        help="Limit the number of sentences to process for testing.")
    args = parser.parse_args()

    print(f"--- Starting BALANCED Gradient-Based Neuron Analysis ---")
    print(f"Using device: {DEVICE}")

    model = GPT2LMHeadModel.from_pretrained(args.model).to(DEVICE).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    corpus_df = pd.read_csv(args.corpus_file)

    run_balanced_gradient_analysis(model, tokenizer, corpus_df, args.layer_idx,
                                   args.output_dir, sample_limit=args.sample_limit)


if __name__ == "__main__":
    main()
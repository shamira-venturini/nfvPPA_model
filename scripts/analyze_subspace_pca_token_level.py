#!/usr/bin/env python3
"""
analyze_activations_gradient_final.py
-------------------------------------
Computes per-neuron attribution using Gradient x Activation, specifically tailored
to the priming data structure where columns represent a 2x2 (Prime x Target) design.

Correctly calculates a balanced priming effect based on this structure.
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
    toks = text.strip().split()
    ids = tokenizer.convert_tokens_to_ids(toks)
    return torch.tensor([ids], dtype=torch.long).to(DEVICE)


def compute_gradient_attribution(model, input_ids, layer_idx):
    model.zero_grad()
    target_tensor = None

    def hook_fn(module, inp, out):
        nonlocal target_tensor
        out.retain_grad()
        target_tensor = out

    handle = model.transformer.h[layer_idx].mlp.c_proj.register_forward_hook(hook_fn)

    outputs = model(input_ids)
    logits = outputs.logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_ids = input_ids[0]

    if token_ids.shape[0] <= 1:
        handle.remove()
        return np.zeros(model.config.n_embd, dtype=np.float32)

    target_log_prob = log_probs[0, :-1, :].gather(1, token_ids[1:].unsqueeze(-1)).sum()
    target_log_prob.backward()
    handle.remove()

    if target_tensor is None or target_tensor.grad is None:
        return np.zeros(model.config.n_embd, dtype=np.float32)

    grad_x_activation = (target_tensor.grad * target_tensor).squeeze(0).sum(dim=0)
    return grad_x_activation.detach().cpu().numpy()


# ==============================================================================
# === SCRIPT LOGIC CORRECTED FOR YOUR SPECIFIC CSV STRUCTURE ===================
# ==============================================================================
def run_balanced_gradient_analysis(model, tokenizer, corpus_df, layer_idx, output_dir, sample_limit=None):
    ensure_dir(output_dir)
    attribution_matrix = []
    summary_data = []

    if sample_limit:
        corpus_df = corpus_df.head(sample_limit)

    iterator = tqdm(corpus_df.iterrows(), total=len(corpus_df), desc=f"Layer {layer_idx} Gradient Attribution")

    for idx, row in iterator:
        # Tokenize all four conditions based on the 2x2 design
        ids_act_prime_act_target = tokenize_row(tokenizer, row["x_px"])  # Congruent
        ids_pas_prime_act_target = tokenize_row(tokenizer, row["x_py"])  # Incongruent
        ids_act_prime_pas_target = tokenize_row(tokenizer, row["y_px"])  # Incongruent
        ids_pas_prime_pas_target = tokenize_row(tokenizer, row["y_py"])  # Congruent

        # Calculate Gradient Attribution for all four sentences
        attr_congruent_1 = compute_gradient_attribution(model, ids_act_prime_act_target, layer_idx)
        attr_congruent_2 = compute_gradient_attribution(model, ids_pas_prime_pas_target, layer_idx)
        attr_incongruent_1 = compute_gradient_attribution(model, ids_pas_prime_act_target, layer_idx)
        attr_incongruent_2 = compute_gradient_attribution(model, ids_act_prime_pas_target, layer_idx)

        # Average the attributions for the two congruent conditions
        avg_attr_congruent = (attr_congruent_1 + attr_congruent_2) / 2.0

        # Average the attributions for the two incongruent conditions
        avg_attr_incongruent = (attr_incongruent_1 + attr_incongruent_2) / 2.0

        # The priming effect is the difference between congruent and incongruent processing
        balanced_priming_attribution = avg_attr_congruent - avg_attr_incongruent

        attribution_matrix.append(balanced_priming_attribution)
        summary_data.append({
            "index": idx,
            "attribution_mean": balanced_priming_attribution.mean(),
            "attribution_std": balanced_priming_attribution.std()
        })

    # --- The rest of the script remains the same ---
    attribution_matrix = np.stack(attribution_matrix)
    np.save(Path(output_dir) / "gradient_attribution_vectors.npy", attribution_matrix)
    pd.DataFrame(summary_data).to_csv(Path(output_dir) / "gradient_attribution_summary.csv", index=False)

    with open(Path(output_dir) / "metadata.json", "w") as f:
        json.dump({
            "model": model.name_or_path,
            "layer_idx": layer_idx,
            "method": "gradient_x_activation_balanced_2x2",
            "samples": len(attribution_matrix)
        }, f, indent=2)

    print(f"\nSaved per-neuron Gradient Attributions: {attribution_matrix.shape} → {output_dir}")
    print("Mean Attribution across dataset:", attribution_matrix.mean())


def main():
    parser = argparse.ArgumentParser(description="Per-neuron Gradient Attribution (2x2 Balanced Design)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--corpus_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--layer_idx", type=int, required=True)
    parser.add_argument("--sample_limit", type=int, default=None)
    args = parser.parse_args()

    print(f"--- Starting BALANCED Gradient-Based Analysis (2x2 Design) ---")
    print(f"Using device: {DEVICE}")

    model = GPT2LMHeadModel.from_pretrained(args.model).to(DEVICE).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    corpus_df = pd.read_csv(args.corpus_file)

    run_balanced_gradient_analysis(model, tokenizer, corpus_df, args.layer_idx,
                                   args.output_dir, sample_limit=args.sample_limit)


if __name__ == "__main__":
    main()
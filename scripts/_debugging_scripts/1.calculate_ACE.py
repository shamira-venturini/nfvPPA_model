#!/usr/bin/env python3
"""
analyze_activations_ACE_fullneurons.py
-------------------------------------
Compute per-neuron Average Causal Effect (ACE) for each sentence in a priming corpus.
Implements batched neuron ablation per Duan et al. (2025), with sentence- or token-level options.

Usage:
 python analyze_activations_ACE_fullneurons.py \
   --model gpt2-large \
   --corpus_file data/core_subset.csv \
   --output_dir outputs/core_layer10 \
   --layer_idx 10 \
   --mode token \
   --batch_size 64
"""

import argparse, os, json
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)


def tokenize_row(tokenizer, text, raw_text=False):
    if raw_text:
        enc = tokenizer(text, return_tensors='pt')
        return enc.input_ids.to(DEVICE)
    toks = text.strip().split()
    ids = tokenizer.convert_tokens_to_ids(toks)
    return torch.tensor([ids], dtype=torch.long).to(DEVICE)


@torch.no_grad()
def compute_layer_means(model, tokenizer, corpus_df, layer_idx,
                        target_col, target_start_col,
                        raw_text=False, sample_limit=None):
    """Compute mean activation vector for the given layer."""
    hidden_size = model.config.n_embd
    sums = np.zeros(hidden_size, dtype=np.float64)
    count = 0
    it = corpus_df.iterrows()
    if sample_limit: it = list(it)[:sample_limit]
    for _, row in tqdm(it, desc=f"Layer{layer_idx} means"):
        ids = tokenize_row(tokenizer, row[target_col], raw_text)
        outs = model(ids, output_hidden_states=True)
        hidden = outs.hidden_states[layer_idx + 1].squeeze(0).cpu().numpy()
        start = int(row[target_start_col])
        if start >= hidden.shape[0]: continue
        seg = hidden[start:]
        sums += seg.sum(0)
        count += seg.shape[0]
    return (sums / max(count, 1)).astype(np.float32)


def compute_logprob(model, input_ids):
    outs = model(input_ids)
    logits = outs.logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_ids = input_ids[0]
    if token_ids.shape[0] <= 1:
        return 0.0
    lp = log_probs[0, :-1, :].gather(1, token_ids[1:].unsqueeze(-1)).sum()
    return lp.item()


def calculate_batched_ACE(model, input_ids, layer_idx, baseline_vec, batch_size=64):
    """Compute per-neuron ACE with batched ablations."""
    hidden_size = model.config.n_embd
    base_logprob = compute_logprob(model, input_ids)
    effects = np.zeros(hidden_size, dtype=np.float32)

    # Prepare hook factory for batched replacement
    def make_hook(neuron_batch):
        def hook(module, inp, out):
            out = out.clone()
            out[..., neuron_batch] = torch.tensor(baseline_vec[neuron_batch],
                                                  device=out.device)
            return out

        return hook

    # Iterate in batches
    for start in range(0, hidden_size, batch_size):
        nbatch = list(range(start, min(start + batch_size, hidden_size)))
        handle = model.transformer.h[layer_idx].mlp.c_proj.register_forward_hook(
            make_hook(nbatch)
        )
        ablated_logprob = compute_logprob(model, input_ids)
        handle.remove()
        effects[nbatch] = ablated_logprob - base_logprob

    return effects


# ==============================================================================
# === SCRIPT LOGIC UPDATED IN THIS FUNCTION ====================================
# ==============================================================================
def run_ACE(model, tokenizer, corpus_df, layer_idx, output_dir,
            mode="sentence", batch_size=64, raw_text=False, sample_limit=None):
    ensure_dir(output_dir)
    baseline_file = Path(output_dir) / f"layer{layer_idx}_means.npy"
    if baseline_file.exists():
        baseline_vec = np.load(baseline_file)
    else:
        # Baseline means are still computed on one condition, which is standard practice.
        baseline_vec = compute_layer_means(
            model, tokenizer, corpus_df, layer_idx,
            target_col="x_px", target_start_col="prime_x_start_idx",
            raw_text=raw_text, sample_limit=sample_limit
        )
        np.save(baseline_file, baseline_vec)

    hidden_size = model.config.n_embd
    ACE_matrix = []
    ACE_summary = []

    iterator = tqdm(list(corpus_df.iterrows())[:sample_limit] if sample_limit else corpus_df.iterrows(),
                    total=(sample_limit or len(corpus_df)), desc=f"Layer{layer_idx} ACE")

    for idx, row in iterator:
        # --- MODIFICATION START ---
        # 1. Tokenize all four conditions from the row
        ids_px = tokenize_row(tokenizer, row["x_px"], raw_text)
        ids_py = tokenize_row(tokenizer, row["x_py"], raw_text)
        ids_y_px = tokenize_row(tokenizer, row["y_px"], raw_text)
        ids_y_py = tokenize_row(tokenizer, row["y_py"], raw_text)

        # 2. Calculate ACE for all four full sentences
        eff_px = calculate_batched_ACE(model, ids_px, layer_idx, baseline_vec, batch_size)
        eff_py = calculate_batched_ACE(model, ids_py, layer_idx, baseline_vec, batch_size)
        eff_y_px = calculate_batched_ACE(model, ids_y_px, layer_idx, baseline_vec, batch_size)
        eff_y_py = calculate_batched_ACE(model, ids_y_py, layer_idx, baseline_vec, batch_size)

        # 3. Calculate the priming effect for each prime type
        # Effect from Active Prime: (Congruent Target - Incongruent Target)
        priming_effect_active_prime = eff_px - eff_py
        # Effect from Passive Prime: (Congruent Target - Incongruent Target)
        priming_effect_passive_prime = eff_y_py - eff_y_px

        # 4. Average them for a balanced, robust score
        balanced_priming_effect = (priming_effect_active_prime + priming_effect_passive_prime) / 2.0

        # 5. Append the balanced effect to the results
        ACE_matrix.append(balanced_priming_effect)
        ACE_summary.append({
            "index": idx,
            "ACE_mean": balanced_priming_effect.mean(),
            "ACE_std": balanced_priming_effect.std()
        })
        # --- MODIFICATION END ---

    ACE_matrix = np.stack(ACE_matrix)
    np.save(Path(output_dir) / "ACE_vectors.npy", ACE_matrix)
    pd.DataFrame(ACE_summary).to_csv(Path(output_dir) / "ACE_summary.csv", index=False)

    with open(Path(output_dir) / "metadata.json", "w") as f:
        json.dump({
            "model": model.name_or_path,
            "layer_idx": layer_idx,
            "mode": mode,
            "batch_size": batch_size,
            "samples": len(ACE_matrix),
            "analysis_type": "balanced_priming"  # Added for clarity
        }, f, indent=2)

    print(f"\nSaved per-neuron ACEs: {ACE_matrix.shape} → {output_dir}")
    print("Mean ACE across dataset:", ACE_matrix.mean())


def main():
    p = argparse.ArgumentParser(description="Per-neuron ACE (Duan et al. 2025, batched version)")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--corpus_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--layer_idx", type=int, required=True)
    p.add_argument("--mode", type=str, default="sentence", choices=["sentence", "token"])
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--raw_text", action="store_true")
    p.add_argument("--sample_limit", type=int, default=None)
    args = p.parse_args()

    model = GPT2LMHeadModel.from_pretrained(args.model).to(DEVICE).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    corpus_df = pd.read_csv(args.corpus_file)

    run_ACE(model, tokenizer, corpus_df, args.layer_idx,
            args.output_dir, mode=args.mode,
            batch_size=args.batch_size,
            raw_text=args.raw_text,
            sample_limit=args.sample_limit)


if __name__ == "__main__":
    main()
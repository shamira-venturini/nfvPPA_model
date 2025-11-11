#!/usr/bin/env python3
"""
analyze_subspace_pca_token_level_final_v2.py
--------------------------------------------
Finds the principal subspace at the main verb of the target sentence.

This version uses the correct logic to find the verb index based on the
start_idx columns and the known structure of the BPE-tokenized target sentences.
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
from sklearn.decomposition import PCA

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)


def tokenize_row(tokenizer, text):
    # The tokenizer must be called directly to get the correct BPE tokens
    # We also add the beginning of text token if the model expects it.
    # For GPT-2, the <|endoftext|> is the BOS token.
    return tokenizer(text, return_tensors='pt').input_ids.to(DEVICE)


@torch.no_grad()
def get_token_activation(model, input_ids, layer_idx, token_idx):
    """Gets the hidden state for a given layer AT A SPECIFIC TOKEN INDEX."""
    outputs = model(input_ids, output_hidden_states=True)
    activations = outputs.hidden_states[layer_idx + 1].squeeze(0)

    if token_idx >= activations.shape[0]:
        print(f"Warning: token_idx {token_idx} is out of bounds for seq len {activations.shape[0]}.")
        return np.zeros(model.config.n_embd, dtype=np.float32)

    token_activation = activations[token_idx]
    return token_activation.cpu().numpy()


def run_token_subspace_analysis(model, tokenizer, corpus_df, layer_idx, output_dir, sample_limit=None):
    ensure_dir(output_dir)

    if sample_limit:
        corpus_df = corpus_df.head(sample_limit)

    difference_vectors = []
    iterator = tqdm(corpus_df.iterrows(), total=len(corpus_df), desc=f"Layer {layer_idx} Token-Level Subspace Analysis")

    for _, row in iterator:
        # --- LOGIC CORRECTED FOR BPE TOKENIZATION ---

        # 1. Define the four conditions
        text_congruent_1 = row["x_px"]  # Active Prime -> Active Target
        text_congruent_2 = row["y_py"]  # Passive Prime -> Passive Target
        text_incongruent_1 = row["x_py"]  # Passive Prime -> Active Target
        text_incongruent_2 = row["y_px"]  # Active Prime -> Passive Target

        # 2. Calculate the correct verb indices based on BPE token structure
        # For Active Targets ("A secretary smelled..."), the verb is the 3rd token of the target.
        verb_idx_active_target = int(row['prime_x_start_idx']) + 2

        # For Passive Targets ("A machine was smelled..."), the verb is the 4th token of the target.
        verb_idx_passive_target = int(row['prime_y_start_idx']) + 3

        # 3. Get activations AT THE CORRECT VERB TOKEN for all four conditions
        act_congruent_1 = get_token_activation(model, tokenize_row(tokenizer, text_congruent_1), layer_idx,
                                               verb_idx_active_target)
        act_congruent_2 = get_token_activation(model, tokenize_row(tokenizer, text_congruent_2), layer_idx,
                                               verb_idx_passive_target)
        act_incongruent_1 = get_token_activation(model, tokenize_row(tokenizer, text_incongruent_1), layer_idx,
                                                 verb_idx_active_target)
        act_incongruent_2 = get_token_activation(model, tokenize_row(tokenizer, text_incongruent_2), layer_idx,
                                                 verb_idx_passive_target)

        # 4. Calculate difference vectors
        diff_active_target = act_congruent_1 - act_incongruent_1
        difference_vectors.append(diff_active_target)

        diff_passive_target = act_congruent_2 - act_incongruent_2
        difference_vectors.append(diff_passive_target)

    # --- Perform PCA ---
    X = np.stack(difference_vectors)
    pca = PCA(n_components=10)
    pca.fit(X)
    pc1 = pca.components_[0]

    # --- Save and Analyze the Results ---
    np.save(Path(output_dir) / "pca_components_token_level.npy", pca.components_)
    np.save(Path(output_dir) / "pca_explained_variance_token_level.npy", pca.explained_variance_ratio_)

    print(f"\n--- Token-Level Subspace Analysis Complete for Layer {layer_idx} ---")
    print(f"Explained variance by PC1 (at the verb): {pca.explained_variance_ratio_[0] * 100:.2f}%")

    top_neuron_indices = np.argsort(np.abs(pc1))[-20:][::-1]

    print("\n--- Top 20 Neurons Contributing to the 'Verb Processing Subspace' (PC1) ---")
    pc1_df = pd.DataFrame({
        'Neuron Index': top_neuron_indices,
        'Contribution to PC1': pc1[top_neuron_indices]
    })
    print(pc1_df.to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Find grammar subspace at the token-level using PCA (Corrected for BPE).")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--corpus_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--layer_idx", type=int, required=True)
    parser.add_argument("--sample_limit", type=int, default=None)
    args = parser.parse_args()

    model = GPT2LMHeadModel.from_pretrained(args.model).to(DEVICE).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    corpus_df = pd.read_csv(args.corpus_file)

    required_cols = ['x_px', 'x_py', 'y_px', 'y_py', 'prime_x_start_idx', 'prime_y_start_idx']
    if not all(col in corpus_df.columns for col in required_cols):
        print(f" ERROR: Your CSV file is missing one of the required columns: {required_cols}")
        return

    run_token_subspace_analysis(model, tokenizer, corpus_df, args.layer_idx,
                                args.output_dir, sample_limit=args.sample_limit)


if __name__ == "__main__":
    main()
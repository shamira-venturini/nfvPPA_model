# scripts/analyze_activations.py (FINAL, INTEGRATED VERSION)
# This script runs the model and performs gradient-based analysis in one go.

import argparse
import os
import torch
import numpy as np
from glob import glob
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from init_corpora import init_corpora
from diagnnose.corpus import Corpus

DEVICE = "cpu"


def get_activations_and_calculate_effect(input_ids, target_start_idx):
    """
    Runs the model with gradients enabled, gets the final hidden state,
    and calculates the direct effect of each neuron.
    """
    # Ensure the model is in eval mode for consistent outputs, but keep gradients on.
    model.eval()

    # Run the model. Gradients are enabled by default outside a no_grad context.
    outputs = model(input_ids, output_hidden_states=True)

    # Get the final layer's hidden state. It's "live" and has a computation graph.
    final_hidden_state = outputs.hidden_states[-1].squeeze(0)  # Remove batch dimension

    # --- This is the same logic as before, but now it will work ---
    logits = outputs.logits.squeeze(0)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    target_token_ids = input_ids.squeeze(0)[target_start_idx:]

    if len(target_token_ids) == 0:
        return np.zeros(model.config.n_embd)

    target_log_probs = log_probs[target_start_idx - 1:-1].gather(1, target_token_ids.unsqueeze(-1)).sum()

    # Zero out any previous gradients before the new backward pass.
    model.zero_grad()
    target_log_probs.backward()

    # The gradient is now stored in the hidden state tensor itself.
    # We access it via an internal attribute because it's not a leaf node.
    hidden_state_grad = final_hidden_state.grad

    direct_effects = final_hidden_state * hidden_state_grad
    neuron_effects = direct_effects.sum(dim=0)

    return neuron_effects.detach().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find influential neurons using the Accumulative Direct Effect method.")
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., gpt2-large).")
    parser.add_argument("--data_dir", type=str, required=True, help="Directory containing the raw .csv corpus files.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the final neuron effect scores.")
    args = parser.parse_args()

    print(f"--- Starting Neuron Analysis for {args.data_dir} ---")
    print("Loading model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained(args.model).to(DEVICE)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)

    primed_corpora, COLUMNS = init_corpora(args.data_dir, tokenizer)

    hidden_size = model.config.n_embd
    total_effect_px = np.zeros(hidden_size)
    total_effect_py = np.zeros(hidden_size)

    # We only need to analyze the Active Target conditions
    corpus_name_px = 'x_px'  # Active Target | Active Prime
    corpus_name_py = 'x_py'  # Active Target | Passive Prime

    for corpus_label, corpus_path in primed_corpora.items():
        print(f"\nProcessing corpus: {corpus_label}")

        # Use the diagnnose Corpus loader to read the processed data
        corpus = Corpus.create(path=corpus_path, header_from_first_line=True, tokenize_columns=list(COLUMNS.keys()),
                               convert_numerical=True, sep=",")

        for item in tqdm(corpus, desc=f"Analyzing {corpus_label}"):
            # --- Congruent Condition ---
            tokens_px = getattr(item, corpus_name_px)
            input_ids_px = torch.tensor([tokenizer.convert_tokens_to_ids(tokens_px)]).to(DEVICE)
            start_idx_px = getattr(item, COLUMNS[corpus_name_px])
            effect_px = get_activations_and_calculate_effect(input_ids_px, start_idx_px)
            total_effect_px += effect_px

            # --- Incongruent Condition ---
            tokens_py = getattr(item, corpus_name_py)
            input_ids_py = torch.tensor([tokenizer.convert_tokens_to_ids(tokens_py)]).to(DEVICE)
            start_idx_py = getattr(item, COLUMNS[corpus_name_py])
            effect_py = get_activations_and_calculate_effect(input_ids_py, start_idx_py)
            total_effect_py += effect_py

    priming_effect_per_neuron = total_effect_px - total_effect_py

    print("\n--- Analysis Complete ---")
    np.save(args.output_file, priming_effect_per_neuron)
    print(f"Neuron priming effect scores saved to: {args.output_file}")

    top_10_indices = np.argsort(priming_effect_per_neuron)[-10:][::-1]
    print(f"\nTop 10 most influential neuron indices for priming:")
    print(top_10_indices)
    print(f"Their effect scores:")
    print(priming_effect_per_neuron[top_10_indices])
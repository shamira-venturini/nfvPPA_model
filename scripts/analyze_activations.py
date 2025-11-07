# scripts/analyze_activations.py (FINAL, GPU-ENABLED VERSION)

import argparse
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# This line correctly and automatically selects the GPU if available
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

captured_grad = None


def save_grad_hook(grad):
    global captured_grad
    captured_grad = grad


def get_activations_and_calculate_effect(input_ids, target_start_idx):
    model.eval()

    # The input_ids are already on the correct device from the main loop
    outputs = model(input_ids, output_hidden_states=True)
    final_hidden_state = outputs.hidden_states[-1]

    hook_handle = final_hidden_state.register_hook(save_grad_hook)

    logits = outputs.logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    target_token_ids = input_ids[0, target_start_idx:]

    if len(target_token_ids) == 0:
        hook_handle.remove()
        return np.zeros(model.config.n_embd)

    target_log_probs = log_probs[0, target_start_idx - 1:-1].gather(1, target_token_ids.unsqueeze(-1)).sum()

    model.zero_grad()
    target_log_probs.backward()

    hook_handle.remove()

    hidden_state_grad = captured_grad

    if hidden_state_grad is None:
        return np.zeros(model.config.n_embd)

    direct_effects = final_hidden_state.squeeze(0) * hidden_state_grad.squeeze(0)
    neuron_effects = direct_effects.sum(dim=0)

    return neuron_effects.detach().cpu().numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find influential neurons using the Accumulative Direct Effect method.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--corpus_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    print(f"--- Starting Neuron Analysis for {os.path.basename(args.corpus_file)} ---")
    print(f"Using device: {DEVICE}")

    if DEVICE == "cpu":
        print("\nWARNING: CUDA not available. This script will be very slow on the CPU.")

    print("Loading model and tokenizer...")
    # --- CHANGE 1: Move the model to the GPU right after loading ---
    model = GPT2LMHeadModel.from_pretrained(args.model).to(DEVICE)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)

    corpus_df = pd.read_csv(args.corpus_file)

    hidden_size = model.config.n_embd
    total_effect_px = np.zeros(hidden_size)
    total_effect_py = np.zeros(hidden_size)

    col_tokens_px = 'x_px'
    col_tokens_py = 'x_py'
    col_start_idx_px = 'prime_x_start_idx'
    col_start_idx_py = 'prime_y_start_idx'

    for index, row in tqdm(corpus_df.iterrows(), total=len(corpus_df),
                           desc=f"Analyzing {os.path.basename(args.corpus_file)}"):
        # --- CHANGE 2: Move the data tensors to the GPU ---
        tokens_px = row[col_tokens_px].split(' ')
        input_ids_px = torch.tensor([tokenizer.convert_tokens_to_ids(tokens_px)]).to(DEVICE)
        start_idx_px = row[col_start_idx_px]
        effect_px = get_activations_and_calculate_effect(input_ids_px, start_idx_px)
        total_effect_px += effect_px

        tokens_py = row[col_tokens_py].split(' ')
        input_ids_py = torch.tensor([tokenizer.convert_tokens_to_ids(tokens_py)]).to(DEVICE)
        start_idx_py = row[col_start_idx_py]
        effect_py = get_activations_and_calculate_effect(input_ids_py, start_idx_py)
        total_effect_py += effect_py

    priming_effect_per_neuron = total_effect_px - total_effect_py

    print("\n--- Analysis Complete ---")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    np.save(args.output_file, priming_effect_per_neuron)
    print(f"Neuron priming effect scores saved to: {args.output_file}")

    top_10_indices = np.argsort(priming_effect_per_neuron)[-10:][::-1]
    print(f"\nTop 10 most influential neuron indices for priming:")
    print(top_10_indices)
    print(f"Their effect scores:")
    print(priming_effect_per_neuron[top_10_indices])
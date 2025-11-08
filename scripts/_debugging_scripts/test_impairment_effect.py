# scripts/test_impairment_effect.py
# This script uses a sensitive metric (Priming Effect) to validate the neuron list.

import argparse
import torch
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def forward_pass_with_impairment_hooks(model, neuron_indices_to_impair, alpha):
    def dampen_activations_hook(module, input, output):
        hidden_state = output[0]
        indices_on_device = torch.tensor(neuron_indices_to_impair).to(DEVICE)
        hidden_state[:, :, indices_on_device] *= alpha
        return (hidden_state,) + output[1:]

    final_layer = model.transformer.h[-1]
    hook_handle = final_layer.register_forward_hook(dampen_activations_hook)
    return hook_handle


def get_target_log_prob(model, input_ids, target_start_idx):
    """Calculates the log probability of the target portion of the input."""
    with torch.no_grad():
        logits = model(input_ids).logits
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    target_token_ids = input_ids[0, target_start_idx:]
    if len(target_token_ids) == 0: return 0
    target_log_probs = log_probs[0, target_start_idx - 1:-1].gather(1, target_token_ids.unsqueeze(-1)).sum()
    return target_log_probs.item()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the effect of neuronal impairment on the structural priming effect.")
    parser.add_argument("--model", type=str, default="gpt2-large")
    parser.add_argument("--neuron_indices_file", type=str, required=True)
    parser.add_argument("--corpus_file", type=str, required=True,
                        help="Path to a PROCESSED .csv file from 'new_corpora'.")
    parser.add_argument("--alpha", type=float, default=0.0, help="Dampening factor (0.0 for full ablation).")
    args = parser.parse_args()

    print(f"--- Starting Impairment Validation Test ---")
    print(f"Using device: {DEVICE}")
    model = GPT2LMHeadModel.from_pretrained(args.model).to(DEVICE)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)

    with open(args.neuron_indices_file, 'r') as f:
        neuron_indices_to_impair = [int(idx.strip()) for idx in f.read().split(',')]
    print(f"Loaded {len(neuron_indices_to_impair)} neuron indices to impair.")

    corpus_df = pd.read_csv(args.corpus_file)

    healthy_pe_scores = []
    impaired_pe_scores = []

    # --- Run the test on the IMPAIRED model first ---
    print(f"\n--- Testing IMPAIRED model (alpha={args.alpha}) ---")
    hook_handle = forward_pass_with_impairment_hooks(model, neuron_indices_to_impair, args.alpha)
    for index, row in tqdm(corpus_df.iterrows(), total=len(corpus_df), desc="Impaired Model"):
        # Congruent
        tokens_px = row['x_px'].split(' ')
        input_ids_px = torch.tensor([tokenizer.convert_tokens_to_ids(tokens_px)]).to(DEVICE)
        logp_x_px = get_target_log_prob(model, input_ids_px, row['prime_x_start_idx'])
        # Incongruent
        tokens_py = row['x_py'].split(' ')
        input_ids_py = torch.tensor([tokenizer.convert_tokens_to_ids(tokens_py)]).to(DEVICE)
        logp_x_py = get_target_log_prob(model, input_ids_py, row['prime_y_start_idx'])
        impaired_pe_scores.append(logp_x_px - logp_x_py)
    hook_handle.remove()  # Restore model to healthy state

    # --- Run the test on the HEALTHY model ---
    print("\n--- Testing HEALTHY model ---")
    for index, row in tqdm(corpus_df.iterrows(), total=len(corpus_df), desc="Healthy Model"):
        # Congruent
        tokens_px = row['x_px'].split(' ')
        input_ids_px = torch.tensor([tokenizer.convert_tokens_to_ids(tokens_px)]).to(DEVICE)
        logp_x_px = get_target_log_prob(model, input_ids_px, row['prime_x_start_idx'])
        # Incongruent
        tokens_py = row['x_py'].split(' ')
        input_ids_py = torch.tensor([tokenizer.convert_tokens_to_ids(tokens_py)]).to(DEVICE)
        logp_x_py = get_target_log_prob(model, input_ids_py, row['prime_y_start_idx'])
        healthy_pe_scores.append(logp_x_px - logp_x_py)

    # --- Compare the data ---
    mean_healthy_pe = np.mean(healthy_pe_scores)
    mean_impaired_pe = np.mean(impaired_pe_scores)
    reduction = (mean_healthy_pe - mean_impaired_pe) / mean_healthy_pe * 100 if mean_healthy_pe != 0 else 0

    print("\n--- RESULTS ---")
    print(f"Mean Priming Effect (Healthy Model):   {mean_healthy_pe:.4f}")
    print(f"Mean Priming Effect (Impaired Model):  {mean_impaired_pe:.4f}")
    print(f"Reduction in Priming Effect: {reduction:.2f}%")
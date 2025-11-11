#!/usr/bin/env python3
"""
probe_syntax_representation_flexible.py
---------------------------------------
Probes the representation of prime sentence syntax (Active vs. Passive)
across all layers of a specified model.

Accepts command-line arguments for model, corpora, output directory,
and an optional sample limit for quick testing.
"""

import argparse
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_prime_sentences(df):
    """Extracts active and passive primes from the corpus dataframe."""
    active_primes = df['x_px'].apply(lambda x: x.split('.')[0].replace('<|endoftext|>', '').strip() + '.').tolist()
    passive_primes = df['x_py'].apply(lambda x: x.split('.')[0].replace('<|endoftext|>', '').strip() + '.').tolist()
    return active_primes, passive_primes


@torch.no_grad()
def get_last_token_activations_all_layers(model, tokenizer, sentences):
    """
    Gets the hidden state of the last token for a batch of sentences
    across all layers of the model.
    """
    inputs = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
    outputs = model(**inputs, output_hidden_states=True)
    hidden_states = outputs.hidden_states

    last_token_indices = inputs['attention_mask'].sum(dim=1) - 1

    all_layer_activations = []
    for layer_idx in range(len(hidden_states)):
        layer_hidden_state = hidden_states[layer_idx]
        last_token_activations = layer_hidden_state[torch.arange(len(sentences)), last_token_indices]
        all_layer_activations.append(last_token_activations.cpu().numpy())

    return np.array(all_layer_activations)


def run_probing_analysis(args):
    """Main function to run the full probing analysis."""
    print(f"Using device: {DEVICE}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model: {args.model_name}...")
    model = GPT2LMHeadModel.from_pretrained(args.model_name).to(DEVICE).eval()
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading and preparing data...")
    core_df = pd.read_csv(args.core_corpus_file)
    anomalous_df = pd.read_csv(args.anomalous_corpus_file)

    if args.sample_limit:
        print(f"Applying sample limit: {args.sample_limit} rows.")
        core_df = core_df.head(args.sample_limit)
        anomalous_df = anomalous_df.head(args.sample_limit)

    core_active, core_passive = extract_prime_sentences(core_df)
    anom_active, anom_passive = extract_prime_sentences(anomalous_df)

    print("Extracting activations for CORE sentences...")
    core_active_acts = get_last_token_activations_all_layers(model, tokenizer, core_active)
    core_passive_acts = get_last_token_activations_all_layers(model, tokenizer, core_passive)

    print("Extracting activations for ANOMALOUS sentences...")
    anom_active_acts = get_last_token_activations_all_layers(model, tokenizer, anom_active)
    anom_passive_acts = get_last_token_activations_all_layers(model, tokenizer, anom_passive)

    X_core = np.concatenate([core_active_acts, core_passive_acts], axis=1)
    y_core = np.array([0] * len(core_active) + [1] * len(core_passive))

    X_anom = np.concatenate([anom_active_acts, anom_passive_acts], axis=1)
    y_anom = np.array([0] * len(anom_active) + [1] * len(anom_passive))

    num_layers = X_core.shape[0]
    results = {'layer': [], 'accuracy_core': [], 'accuracy_anomalous': []}

    print("\nStarting probing analysis across layers...")
    for layer in tqdm(range(num_layers), desc="Probing Layers"):
        X_core_layer = X_core[layer, :, :]
        X_anom_layer = X_anom[layer, :, :]

        X_train, X_test, y_train, y_test = train_test_split(
            X_core_layer, y_core, test_size=0.2, random_state=42, stratify=y_core
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        X_anom_scaled = scaler.transform(X_anom_layer)

        probe = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
        probe.fit(X_train_scaled, y_train)

        preds_core = probe.predict(X_test_scaled)
        acc_core = accuracy_score(y_test, preds_core)

        preds_anom = probe.predict(X_anom_scaled)
        acc_anom = accuracy_score(y_anom, preds_anom)

        results['layer'].append(layer)
        results['accuracy_core'].append(acc_core)
        results['accuracy_anomalous'].append(acc_anom)

    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(args.output_dir, 'probe_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nSaved results table to {results_csv_path}")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(results_df['layer'], results_df['accuracy_core'] * 100, marker='o', linestyle='-',
            label='Probe on CORE (Plausible) Primes')
    ax.plot(results_df['layer'], results_df['accuracy_anomalous'] * 100, marker='o', linestyle='-',
            label='Probe on ANOMALOUS Primes')
    ax.set_title(f'Probe Accuracy for Prime Syntax Across Layers ({args.model_name})', fontsize=16)
    ax.set_xlabel('Model Layer (0=Embeddings)', fontsize=12)
    ax.set_ylabel('Probe Accuracy (%)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(45, 105)
    plt.tight_layout()

    plot_path = os.path.join(args.output_dir, 'probe_accuracy_plot.png')
    plt.savefig(plot_path)
    print(f"Saved plot to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Run a probing analysis for prime sentence syntax.")
    parser.add_argument("--model_name", type=str, default="gpt2-large", help="Name of the Hugging Face model to use.")
    parser.add_argument("--core_corpus_file", type=str, required=True,
                        help="Path to the CORE (plausible) corpus CSV file.")
    parser.add_argument("--anomalous_corpus_file", type=str, required=True,
                        help="Path to the ANOMALOUS corpus CSV file.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results (plot and CSV).")
    parser.add_argument("--sample_limit", type=int, default=None,
                        help="Optional: use only the first N samples for a quick test.")
    args = parser.parse_args()
    run_probing_analysis(args)


if __name__ == "__main__":
    main()
# scripts/2.validate_behaviour.py (Version 3.0 - With POS Tagging)

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import ast
import spacy


def parse_numpy_string(s):
    try:
        cleaned_s = s.strip().replace('[', '').replace(']', '').replace('\n', '')
        return np.fromstring(cleaned_s, sep=' ')
    except:
        return np.array([])


def analyze_and_plot(analysis_filepath):
    print(f"\n--- Analyzing File: {os.path.basename(analysis_filepath)} ---")
    df = pd.read_csv(analysis_filepath)

    # --- VALIDATION (UNCHANGED) ---
    if 's_pe_active' in df.columns:
        print(f"\n[VALIDATION] Mean s-PE for ACTIVE Target: {df['s_pe_active'].mean():.4f}")
    if 's_pe_passive' in df.columns:
        print(f"[VALIDATION] Mean s-PE for PASSIVE Target: {df['s_pe_passive'].mean():.4f}")

    # --- W-PE ANALYSIS WITH POS TAGGING ---
    for target_type in ['active', 'passive']:
        wpe_col = f'w_pe_{target_type}'
        token_col = f'target_tokens_{target_type}'
        if wpe_col in df.columns and not df[wpe_col].isnull().all():
            print(f"\n[ANALYSIS] Plotting w-PE for {target_type.upper()} target...")

            df[f'{wpe_col}_list'] = df[wpe_col].apply(parse_numpy_string)
            df[f'{token_col}_list'] = df[token_col].apply(ast.literal_eval)

            w_pe_scores = df[f'{wpe_col}_list'].tolist()
            token_lists = df[f'{token_col}_list'].tolist()

            # --- NEW: Get POS tags and align them ---
            # Reconstruct the first sentence to get its POS tags
            first_sentence_tokens = [t.replace('Ġ', ' ') for t in token_lists[0]]
            first_sentence_text = "".join(first_sentence_tokens).strip()

            doc = nlp(first_sentence_text)
            pos_labels = [f"{token.text}-{token.pos_}" for token in doc]

            # Simple alignment: Assume one POS tag per subword token for plotting
            # A more advanced alignment would be needed for perfect accuracy
            aligned_labels = []
            token_idx = 0
            for spacy_token in doc:
                word = spacy_token.text
                label = f"{spacy_token.text}_{spacy_token.pos_}"
                # Find how many subword tokens this word corresponds to
                num_subwords = 0
                reconstructed_word = ""
                while token_idx < len(first_sentence_tokens) and len(reconstructed_word) < len(word):
                    reconstructed_word += first_sentence_tokens[token_idx].replace('Ġ', '')
                    aligned_labels.append(label)
                    num_subwords += 1
                    token_idx += 1

            max_len = max(len(s) for s in w_pe_scores if s is not None)
            padded_scores = np.array(
                [np.pad(s, (0, max_len - len(s)), 'constant', constant_values=np.nan) for s in w_pe_scores])
            mean_w_pe_per_token = np.nanmean(padded_scores, axis=0)

            # Use the new POS labels for the plot
            plot_labels = aligned_labels[:len(mean_w_pe_per_token)]

            # --- Plotting ---
            plt.figure(figsize=(14, 7))
            plt.bar(range(len(mean_w_pe_per_token)), mean_w_pe_per_token, color='mediumseagreen')
            plt.axhline(0, color='grey', linestyle='--')
            plt.title(
                f'Token-Level Priming Effect (w-PE) with POS Tags for {target_type.upper()} Target\n({os.path.basename(analysis_filepath)})')
            plt.ylabel('Mean Priming Effect (Log-Probability Difference)')
            plt.xlabel('Token Position with Part-of-Speech Tag')
            plt.xticks(range(len(plot_labels)), plot_labels, rotation=60, ha="right")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()

            plot_filename = os.path.basename(analysis_filepath).replace('.csv', f'_wpe_plot_{target_type}_pos.png')
            plot_save_path = os.path.join(os.path.dirname(analysis_filepath), plot_filename)
            plt.savefig(plot_save_path)
            print(f" -> Plot with POS tags saved to: {plot_save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and plot priming effect data.")
    parser.add_argument("analysis_file", type=str, help="Path to the _analysis.csv file to process.")
    args = parser.parse_args()

    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    analyze_and_plot(args.analysis_file)
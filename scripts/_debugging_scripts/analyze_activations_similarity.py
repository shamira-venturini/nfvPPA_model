# scripts/analyze_activations_similarity.py (FINAL, CORRECTED PATHS)

import argparse
import os
import torch
import numpy as np
from glob import glob
from tqdm import tqdm


def get_mean_activations(corpus_dir, column_name):
    """
    Calculates the average activation vector for a given condition across all sentences.
    """
    # --- DEBUG: Print the path we are searching in ---
    search_path = os.path.join(corpus_dir, column_name, '*.pt')
    print(f"  -> Searching for files with pattern: {search_path}")

    file_list = glob(search_path)
    num_items = len(file_list)

    if num_items == 0:
        print(f"  -> WARNING: Found 0 files for this condition. Returning zeros.")
        return np.zeros(1280)  # Return a zero vector if no files are found

    print(f"  -> Found {num_items} files. Calculating mean activation...")

    hidden_size = 1280
    total_activations = np.zeros(hidden_size)

    for file_path in tqdm(file_list, desc=f"Loading {column_name}"):
        try:
            data = torch.load(file_path)
            mean_sentence_activation = data['activations']['layer_-1_hx'].mean(dim=0).numpy()
            total_activations += mean_sentence_activation
        except Exception as e:
            print(f"Warning: Could not load or process file {file_path}. Error: {e}")
            num_items -= 1  # Adjust count if a file is corrupted

    return total_activations / num_items if num_items > 0 else np.zeros(hidden_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find syntax-sensitive neurons via representational similarity.")
    parser.add_argument("--raw_data_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    print("--- Starting Representational Similarity Analysis ---")

    # --- THIS IS THE FIX: Find the folders by their prefix ---
    core_dir_pattern = os.path.join(args.raw_data_dir, "CORE_transitive_*")
    anom_dir_pattern = os.path.join(args.raw_data_dir, "ANOMALOUS_chomsky_transitive_*")

    try:
        core_dir = glob(core_dir_pattern)[0]
        anom_dir = glob(anom_dir_pattern)[0]
    except IndexError:
        print("ERROR: Could not find the data directories. Please check your --raw_data_dir path.")
        print(f"       Was looking for folders starting with 'CORE_transitive' and 'ANOMALOUS_chomsky_transitive'")
        exit()

    print(f"\nAnalyzing CORE corpus at: {core_dir}")
    avg_active_core = get_mean_activations(core_dir, 'x_px')
    avg_passive_core = get_mean_activations(core_dir, 'y_py')

    print(f"\nAnalyzing ANOMALOUS corpus at: {anom_dir}")
    avg_active_anom = get_mean_activations(anom_dir, 'x_px')
    avg_passive_anom = get_mean_activations(anom_dir, 'y_py')

    if np.all(avg_active_core == 0) or np.all(avg_passive_core == 0):
        print("\nERROR: Failed to load activations for one or more conditions. Cannot proceed.")
    else:
        cross_structure_diff = np.abs(avg_active_core - avg_passive_core)
        within_active_diff = np.abs(avg_active_core - avg_active_anom)
        within_passive_diff = np.abs(avg_passive_core - avg_passive_anom)

        epsilon = 1e-6
        syntax_score = cross_structure_diff / (within_active_diff + within_passive_diff + epsilon)

        print("\n--- Analysis Complete ---")
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        np.save(args.output_file, syntax_score)
        print(f"Neuron syntax scores saved to: {args.output_file}")

        top_50_indices = np.argsort(syntax_score)[-50:][::-1]
        print(f"\nTop 50 most syntax-sensitive neuron indices (Similarity Method):")
        print(top_50_indices.tolist())
        print(f"\nTheir syntax scores:")
        print(syntax_score[top_50_indices])
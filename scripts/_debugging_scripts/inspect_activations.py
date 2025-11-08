# scripts/inspect_activations.py
# This script loads the bahavioral_scores data and a raw data file to show the actual activations.

import argparse
import torch
import numpy as np
import os


def inspect_neuron_activations(neuron_effects_file, raw_data_file, num_to_show=10):
    """
    Loads neuron rankings and a sample data file, then prints the activations
    of the top-ranked neurons for that sample.
    """
    print(f"--- Inspecting Neuron Activations ---")

    # --- Step 1: Load the Neuron Ranking ---
    try:
        neuron_effects = np.load(neuron_effects_file)
    except FileNotFoundError:
        print(f"ERROR: Neuron effects file not found at '{neuron_effects_file}'")
        return

    # Get the indices of the top N neurons based on the (flawed) bahavioral_scores
    top_neuron_indices = np.argsort(neuron_effects)[-num_to_show:][::-1]

    print(f"\nLoaded neuron rankings from: {neuron_effects_file}")
    print(f"Top {num_to_show} most influential neuron indices (according to the file):")
    print(top_neuron_indices)

    # --- Step 2: Load a Sample Raw Data File ---
    try:
        raw_data = torch.load(raw_data_file)
    except FileNotFoundError:
        print(f"\nERROR: Raw data file not found at '{raw_data_file}'")
        print("Please make sure you have a .pt file from your 'data/raw_activations' folder.")
        return

    print(f"\nLoaded sample raw data from: {raw_data_file}")

    # Extract the final layer hidden state (the activations)
    activations = raw_data['activations']['layer_-1_hx']

    # --- Step 3: Print the Activations for the Top Neurons ---
    print("\n--- Activations for the Top-Ranked Neurons (for this single sentence) ---")
    print("Note: These values are the MEAN activation across all tokens in this sentence.\n")

    for neuron_idx in top_neuron_indices:
        # Get the activation values for this specific neuron across all tokens
        neuron_activation_over_time = activations[:, neuron_idx]

        # Calculate the mean activation for this sentence
        mean_activation = neuron_activation_over_time.mean().item()

        # Print the result - IT WILL NOT BE ZERO
        print(f"Neuron #{neuron_idx:<4}: Mean Activation = {mean_activation:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect the raw activations of top-ranked neurons.")
    parser.add_argument("--neuron_effects_file", type=str, required=True,
                        help="Path to the .npy file with neuron scores.")
    parser.add_argument("--sample_data_file", type=str, required=True,
                        help="Path to a single .pt file from the 'raw_activations' folder.")
    args = parser.parse_args()

    inspect_neuron_activations(args.neuron_effects_file, args.sample_data_file)
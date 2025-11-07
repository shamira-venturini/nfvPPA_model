# scripts/test_impaired_generation.py
# This script tests the effect of neuronal impairment on sentence GENERATION.

import argparse
import torch
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEVICE = "cpu"


def forward_pass_with_impairment_hooks(model, top_neuron_indices, alpha):
    """
    This is a more advanced and robust way to apply the impairment.
    It uses "forward hooks" to modify the activations on the fly.
    """

    # The hook function itself
    def dampen_activations_hook(module, input, output):
        # The output of a transformer block is a tuple; the hidden state is the first element
        hidden_state = output[0]
        # Dampen the specific neurons
        hidden_state[:, :, top_neuron_indices] *= alpha
        return (hidden_state,) + output[1:]

    # We will attach this hook to the final layer of the model
    final_layer = model.transformer.h[-1]
    hook_handle = final_layer.register_forward_hook(dampen_activations_hook)

    return hook_handle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the effect of neuronal impairment on language generation.")
    parser.add_argument("--model", type=str, default="gpt2-large", help="Model name.")
    parser.add_argument("--neuron_effects_file", type=str, required=True,
                        help="Path to the .npy file with neuron scores.")
    parser.add_argument("--num_impaired_neurons", type=int, default=50, help="Number of top neurons to impair.")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="Dampening factor (0.0 = full ablation, 1.0 = no effect).")
    parser.add_argument("--num_generations", type=int, default=10, help="Number of sentences to generate.")
    args = parser.parse_args()

    # --- Step 1: Load Everything ---
    print("Loading model, tokenizer, and neuron effects...")
    model = GPT2LMHeadModel.from_pretrained(args.model).to(DEVICE)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    neuron_effects = np.load(args.neuron_effects_file)
    top_neuron_indices = np.argsort(neuron_effects)[-args.num_impaired_neurons:]
    print(f"Identified the top {args.num_impaired_neurons} most influential neurons.")

    # --- Step 2: Define the Generation Prompts ---
    # These are simple, open-ended prompts to encourage sentence production.
    prompts = [
        "The scientist discovered",
        "After the game, the team went",
        "The book on the table was",
        "Because the weather was so nice,",
        "The artist painted a picture of",
    ]
    prompts = prompts[:args.num_generations]

    # --- Step 3: Generate with the HEALTHY Model ---
    print("\n--- Generating with HEALTHY model ---")
    healthy_generations = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        output = model.generate(
            input_ids,
            max_length=30,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        healthy_generations.append(generated_text)
        print(f"  Prompt: '{prompt}'\n  Output: '{generated_text}'")

    # --- Step 4: Generate with the IMPAIRED Model ---
    print(f"\n--- Generating with IMPAIRED model (alpha={args.alpha}) ---")

    # Attach the hook to the model to create the "lesion"
    hook_handle = forward_pass_with_impairment_hooks(model, top_neuron_indices, args.alpha)

    impaired_generations = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        output = model.generate(
            input_ids,
            max_length=30,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        impaired_generations.append(generated_text)
        print(f"  Prompt: '{prompt}'\n  Output: '{generated_text}'")

    # IMPORTANT: Remove the hook to restore the model to its healthy state
    hook_handle.remove()

    # --- Step 5: Save Results for Analysis ---
    results_df = pd.DataFrame({
        'prompt': prompts,
        'healthy_generation': healthy_generations,
        'impaired_generation': impaired_generations
    })
    output_file = f"generation_results_alpha_{args.alpha}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nGeneration results saved to {output_file}")
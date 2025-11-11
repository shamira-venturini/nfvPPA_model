# scripts/3.test_impairment.py (FINAL, CORRECTED LOGIC)

import argparse
import torch
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def forward_pass_with_impairment_hooks(model, neuron_indices_to_impair, alpha):
    """ Applies the impairment using forward hooks. """

    def dampen_activations_hook(module, input, output):
        hidden_state = output[0]
        indices_on_device = torch.tensor(neuron_indices_to_impair).to(DEVICE)
        hidden_state[:, :, indices_on_device] *= alpha
        return (hidden_state,) + output[1:]

    final_layer = model.transformer.h[-1]
    hook_handle = final_layer.register_forward_hook(dampen_activations_hook)
    return hook_handle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the effect of neuronal impairment on language generation.")
    parser.add_argument("--model", type=str, default="gpt2-large", help="Model name.")
    # THIS IS THE NEW, CORRECT ARGUMENT
    parser.add_argument("--neuron_indices_file", type=str, required=True,
                        help="Path to a .txt file containing a comma-separated list of neuron indices to impair.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dampening factor.")
    parser.add_argument("--num_generations", type=int, default=10, help="Number of sentences to generate.")
    args = parser.parse_args()

    print(f"--- Starting Impaired Generation Test ---")
    print(f"Using device: {DEVICE}")

    print("Loading model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained(args.model).to(DEVICE)
    tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    # --- THIS IS THE NEW, CORRECT LOADING LOGIC ---
    with open(args.neuron_indices_file, 'r') as f:
        neuron_indices_to_impair = [int(idx.strip()) for idx in f.read().split(',')]
    print(f"Loaded {len(neuron_indices_to_impair)} neuron indices to impair from {args.neuron_indices_file}.")

    prompts = [
        "The scientist discovered", "After the game, the team went", "The book on the table was",
        "Because the weather was so nice,", "The artist painted a picture of", "To fix the car, the mechanic needed",
        "The old house at the end of the street", "Despite the rain, the children decided",
        "The secret to making a perfect cake is", "In the middle of the night, a strange noise"
    ]
    prompts = prompts[:args.num_generations]

    # --- Generation loops are the same ---
    print("\n--- Generating with HEALTHY model ---")
    healthy_generations = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        output = model.generate(
            input_ids, max_length=30, num_return_sequences=1, do_sample=True,
            top_k=50, top_p=0.95, temperature=1.0, pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        healthy_generations.append(generated_text)
        print(f"  Prompt: '{prompt}'\n  Output: '{generated_text}'")

    print(f"\n--- Generating with IMPAIRED model (alpha={args.alpha}) ---")
    hook_handle = forward_pass_with_impairment_hooks(model, neuron_indices_to_impair, args.alpha)

    impaired_generations = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(DEVICE)
        output = model.generate(
            input_ids, max_length=30, num_return_sequences=1, do_sample=True,
            top_k=50, top_p=0.95, temperature=1.0, pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        impaired_generations.append(generated_text)
        print(f"  Prompt: '{prompt}'\n  Output: '{generated_text}'")

    hook_handle.remove()

    results_df = pd.DataFrame({
        'prompt': prompts,
        'healthy_generation': healthy_generations,
        'impaired_generation': impaired_generations
    })
    output_file = f"generation_results_alpha_{args.alpha}.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\nGeneration results saved to {output_file}")
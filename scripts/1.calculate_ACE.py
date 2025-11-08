#!/usr/bin/env python3
"""
analyze_activations_ACE_updated.py

Updated script to compute per-neuron Average Causal Effect (ACE) for priming corpora
(compatible with the user's Prime-LM-style CSVs). Features:
 - computes and caches per-layer per-neuron baseline means
 - supports sentence-level ACE (fast) and token-level ACE (detailed)
 - uses baseline replacement (mean) rather than zeroing
 - batched neuron ablation (to speed up runs)
 - layer sweep support, group-ablation diagnostics, and output of top-k neurons

Usage (example):
 python analyze_activations_ACE_updated.py \
   --model gpt2-large \
   --corpus_file data/core_subset.csv \
   --output_dir outputs/core_layer10 \
   --layer_idx 10 \
   --mode token  # or 'sentence' for faster runs

Notes:
 - The script assumes your CSV has columns: x_px, prime_x_start_idx, x_py, prime_y_start_idx
   where x_px/x_py contain *tokenized* text separated by spaces (the user's previous format).
 - If your CSV contains raw text (not pre-tokenized), set --raw_text True and the tokenizer
   will be used to encode full strings.

"""

import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_dir(d):
    Path(d).mkdir(parents=True, exist_ok=True)


def tokenize_row(tokenizer, text, raw_text=False):
    """Return input_ids tensor for a row. If raw_text True, we call tokenizer on the full string.
    Otherwise we assume `text` is tokenized and space-separated tokens compatible with
    tokenizer.convert_tokens_to_ids.
    """
    if raw_text:
        enc = tokenizer(text, return_tensors='pt')
        return enc.input_ids.to(DEVICE)
    else:
        # assume space-separated tokens
        toks = text.strip().split()
        ids = tokenizer.convert_tokens_to_ids(toks)
        return torch.tensor([ids], dtype=torch.long).to(DEVICE)


@torch.no_grad()
def compute_layer_means(model, tokenizer, corpus_df, layer_idx, target_col, target_start_col,
                        raw_text=False, sample_limit=None, progress=True):
    """Compute per-neuron mean activation for the given layer across target positions.
    Returns numpy array shape (hidden_size,)
    """
    hidden_size = model.config.n_embd
    sums = np.zeros(hidden_size, dtype=np.float64)
    count = 0

    it = corpus_df.iterrows()
    if sample_limit is not None:
        it = list(it)[:sample_limit]

    iterator = tqdm(it, desc=f"Computing means L{layer_idx}") if progress else it
    for _, row in iterator:
        input_ids = tokenize_row(tokenizer, row[target_col], raw_text=raw_text)
        outputs = model(input_ids, output_hidden_states=True)
        # hidden states indexing: hidden_states[layer_index] where layer_index=0 is input embeddings
        # GPT2 returns tuple length = num_layers+1
        # final residual stream at layer l is outputs.hidden_states[layer_idx + 1]
        # but we'll use outputs.hidden_states[layer_idx] convention: test empirically
        # Use the last residual output for the given transformer layer:
        hidden_states = outputs.hidden_states
        # choose hidden at the requested transformer block index
        # hidden_states[ layer_idx + 1 ] corresponds to output after layer_idx block
        layer_hidden = hidden_states[layer_idx + 1].squeeze(0).cpu().numpy()  # shape (seq_len, hidden)

        # target token positions
        start = int(row[target_start_col])
        seq_len = layer_hidden.shape[0]
        if start >= seq_len:
            continue
        # collect activations from start .. end-1 (exclude last because logits at last come from previous)
        # we keep the same convention as your previous scripts: target region = input_ids[ start: ]
        segment = layer_hidden[start:]
        sums += segment.sum(axis=0)
        count += segment.shape[0]

    if count == 0:
        return np.zeros(hidden_size, dtype=np.float32)
    means = (sums / float(count)).astype(np.float32)
    return means


def compute_token_logprobs(model, input_ids):
    """Compute per-token conditional log-probabilities for the sequence.
    Returns 1D numpy array length = seq_len-1 where entry t corresponds to log P(token_{t} | tokens[:t])
    """
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits  # shape (1, seq_len, vocab)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_ids = input_ids[0]
        # we want log probs for tokens at positions 1..seq_len-1 coming from logits at positions 0..seq_len-2
        if token_ids.shape[0] <= 1:
            return np.array([])
import os
from datasets import load_dataset
import pandas as pd
from evo import Evo
import torch
from torch.utils.data import TensorDataset, DataLoader
from evo.scoring import prepare_batch
import gc
from pathlib import Path

from subset_by_logits import collect_logits, process_logits
from extract_activations import collect_hidden_states, collect_last_hidden_states

def run():
    print("Starting script...")
    evo_model = Evo('evo-1-8k-base')
    print("Model loaded")
    print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    dataset_type='stage1'
    save_dir='../sae-dataset/activation_datasets'

    # Create directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    if dataset_type == 'stage2':
        data = pd.read_csv('./stage2/stage2_validation.csv')
        print(f"Data loaded: {len(data)} rows")

        cleaned_data = data['text'].str.split('|', 2).str[-1]
        print("Data cleaned")

        sequences=cleaned_data.tolist()
    else:
        data = pd.read_csv('./stage1/stage1_train_1000000.csv')
        # Truncate each sequence to first 8000 characters
        sequences = [text[:8000] if isinstance(text, str) else '' for text in data['text'].tolist()]
        # sequences = data['text'].tolist()

    device = 'cuda'
    model, tokenizer = evo_model.model, evo_model.tokenizer
    model.to(device)
    print(f"After model to GPU: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    model.eval()

    # First collect all logits
    logits, input_ids = collect_logits(model, sequences, tokenizer)
    print("Finished collecting logits")

    # Process them together
    scores, high_indices = process_logits(logits, input_ids, sequences)

    # Save the dataset of sequences
    subset_data = data.iloc[high_indices].copy()  # Create a copy to avoid modifying original
    subset_data['text'] = subset_data['text'].str[:8000]  # Truncate to first 8K chars
    subset_data.to_csv(f"{save_dir}/{dataset_type}_filtered_labeled_dataset.csv", index=False)

    # Get high scoring sequences
    high_scoring_sequences = [sequences[i] for i in high_indices]

    activations = collect_last_hidden_states(model, high_scoring_sequences, tokenizer)
    print("Finished collecting activations")

    # Save the last layer activations
    last_layer_idx = len(model.blocks) - 1  # Get the index of the last layer
    save_path = f"{save_dir}/{dataset_type}_filtered_layer_{last_layer_idx}.pt"
    torch.save(activations, save_path)
    print(f"Saved last layer {last_layer_idx} (shape: {activations.shape})")
    
    # Save activations for each layer
#    for layer_idx, layer_activation in activations.items():
#        save_path = f"{save_dir}/stage2_filtered_layer_{layer_idx}.pt"
#        torch.save(layer_activation, save_path)
#        print(f"Saved layer {layer_idx} ({layer_activation.shape})")

if __name__ == "__main__":
    run()

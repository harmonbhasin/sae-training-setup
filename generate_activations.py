import os
from datasets import load_dataset
import pandas as pd
from evo import Evo
import torch
from torch.utils.data import TensorDataset, DataLoader
from evo.scoring import prepare_batch
import gc

from subset_by_logits import collect_logits, process_logits
from extract_activations import collect_hidden_states

def run():
    evo_model = Evo('evo-1-8k-base')
    data = pd.read_csv('./stage2/stage2_validation.csv')
    cleaned_data = data['text'].str.split('|', 2).str[-1]

    device = 'cuda'
    model, tokenizer = evo_model.model, evo_model.tokenizer
    model.to(device)
    model.eval()

    sequences=cleaned_data.tolist()

    # First collect all logits
    logits, input_ids = collect_logits(model, sequences, tokenizer)

    # Process them together
    scores, high_indices = process_logits(logits, input_ids, sequences)

    # Get high scoring sequences
    high_scoring_sequences = [sequences[i] for i in high_indices]

    activations = collect_hidden_states(model, high_scoring_sequences, tokenizer, batch_size=2, device='cuda')

    save_dir='activation_datasets'

    # Create directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Save activations for each layer
    for layer_idx, layer_activation in activations.items():
        save_path = f"{save_dir}/stage2_filtered_layer_{layer_idx}.pt"
        torch.save(layer_activation, save_path)
        print(f"Saved layer {layer_idx} ({layer_activation.shape})")

    subset_data = data.iloc[high_indices]
    subset_data.to_csv(f"{save_dir}/stage2_filtered_labeled_dataset.csv", index=False)

if __name__ == "__main__":
    run()

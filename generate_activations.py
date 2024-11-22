import os
from datasets import load_dataset
import pandas as pd
from evo import Evo
import torch
from torch.utils.data import TensorDataset, DataLoader
from evo.scoring import prepare_batch
import gc

from scoring import collect_logits_and_score_batch

def save_high_scoring_sequences(scores, high_idx, sequences, save_dir='results'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Save high-scoring sequences with their scores
    high_scoring_seqs = [sequences[i] for i in high_idx]
    pd.DataFrame({
        'sequence': high_scoring_seqs,
        'score': [scores[i] for i in high_idx],
        'index': high_idx
    }).to_csv(f'{save_dir}/high_scoring_sequences.csv', index=False)

def run():
    evo_model = Evo('evo-1-8k-base')
    data = pd.read_csv('./stage2/stage2_validation.csv')
    cleaned_data = data['text'].str.split('|', 2).str[-1]
    
    device = 'cuda'
    model = evo_model.model.to(device).eval()
    
    scores, _, high_idx = collect_logits_and_score_batch(
        model, cleaned_data.tolist(), evo_model.tokenizer
    )
    
    save_high_scoring_sequences(scores, high_idx, cleaned_data.tolist())

def collect_hidden_states(model, sequences, tokenizer, target_layer, device='cuda'):
    """
    Collect hidden states from a specific layer for each sequence.
    Processes one sequence at a time to manage memory while preserving full context.

    Args:
        model: The Evo model
        sequences: List of sequences to process
        tokenizer: The tokenizer
        target_layer: Index of layer to collect activations from
        device: Computing device
        
    Returns:
        torch.Tensor: Activations from target layer, shape (total_tokens, hidden_dim)
    """
    all_activations = []

    # Move model to GPU
    model = model.to(device)

    for idx, sequence in enumerate(sequences):
        # Storage for current sequence
        hidden_states = []
        
        # Register hook for target layer only
        def hook_fn(module, input, output):
            hidden_states.append(output[0].detach().cpu())
            
        # Add hook only to target layer
        hook = model.blocks[target_layer].register_forward_hook(hook_fn)
        
        try:
            # Clear GPU memory
            torch.cuda.empty_cache()
            
            # Prepare single sequence
            input_ids, _ = prepare_batch(
                [sequence],
                tokenizer,
                prepend_bos=False,
                device='cpu'
            )
            
            # Move to GPU and process
            with torch.no_grad():
                input_ids = input_ids.to(device)
                logits, *_ = model(input_ids)
                
                # Ensure GPU sync
                torch.cuda.synchronize()
            
            # Process hidden states
            if hidden_states:
                # Reshape to (seq_len, hidden_dim)
                layer_activations = hidden_states[0].reshape(-1, hidden_states[0].shape[-1])
                all_activations.append(layer_activations)
            
            # Progress update
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(sequences)} sequences")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM error processing sequence {idx}, skipping...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
        finally:
            # Clean up
            hook.remove()
            torch.cuda.empty_cache()

    # Concatenate all activations
    return torch.cat(all_activations, dim=0)

if __name__ == "__main__":
    run()

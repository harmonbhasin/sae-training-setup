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


if __name__ == "__main__":
    run()

import os
from datasets import load_dataset
import pandas as pd

def download_genome_splits():
    # Load all splits
    dataset = load_dataset("LongSafari/open-genome", 'stage2')
    
    # Create directory
    os.makedirs('stage2', exist_ok=True)
    
    # Save each split
    splits = {}
    for split in ['train', 'validation', 'test']:
        df = pd.DataFrame(dataset[split])
        output_path = os.path.join('stage2', f'stage2_{split}.csv')
        df.to_csv(output_path, index=False)
        splits[split] = df
        print(f"Saved {split} split ({len(df)} rows) to {output_path}")
    
    return splits

def load_genome_splits():
    splits = {}
    for split in ['train', 'validation', 'test']:
        filepath = os.path.join('stage2', f'stage2_{split}.csv')
        splits[split] = pd.read_csv(filepath)
    return splits

if __name__ == "__main__":
    splits = download_genome_splits()
    for split, df in splits.items():
        print(f"\n{split.capitalize()} split info:")
        print(f"Rows: {len(df)}")
        print("Columns:", df.columns.tolist())

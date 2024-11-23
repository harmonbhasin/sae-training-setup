import os
from datasets import load_dataset
import pandas as pd
from itertools import islice

def download_genome_splits(max_entries_per_split=2_000_000):
    """
    Download and save genome dataset splits, limiting to max_entries_per_split per split.
    
    Args:
        max_entries_per_split (int): Maximum number of entries to save per split
        
    Returns:
        dict: Dictionary containing DataFrames for each split
    """
    # Create directory
    os.makedirs('stage2', exist_ok=True)
    
    # Load dataset in streaming mode
    dataset = load_dataset("LongSafari/open-genome", 'stage2', streaming=True)
    
    splits = {}
    for split_name in ['train', 'validation', 'test']:
        print(f"\nProcessing {split_name} split...")
        
        # Get the iterator for the current split
        split_iter = dataset[split_name]
        
        # Stream and collect the first max_entries_per_split entries
        entries = []
        try:
            for entry in islice(split_iter, max_entries_per_split):
                entries.append(entry)
                
                # Print progress every 100k entries
                if len(entries) % 100_000 == 0:
                    print(f"Processed {len(entries):,} entries...")
                    
        except Exception as e:
            print(f"Error while streaming {split_name} split: {str(e)}")
            continue
            
        # Convert to DataFrame
        df = pd.DataFrame(entries)
        
        # Save to CSV
        output_path = os.path.join('stage2', f'stage2_{split_name}_{max_entries_per_split}.csv')
        df.to_csv(output_path, index=False)
        splits[split_name] = df
        
        print(f"Saved {len(df):,} rows to {output_path}")
    
    return splits

def load_genome_splits(max_entries=2_000_000):
    """
    Load the saved genome splits.
    
    Args:
        max_entries (int): Number of entries that were saved per split
        
    Returns:
        dict: Dictionary containing DataFrames for each split
    """
    splits = {}
    for split in ['train', 'validation', 'test']:
        filepath = os.path.join('stage2', f'stage2_{split}_{max_entries}.csv')
        if os.path.exists(filepath):
            splits[split] = pd.read_csv(filepath)
        else:
            print(f"Warning: {filepath} not found")
    return splits

if __name__ == "__main__":
    # Download the splits
    splits = download_genome_splits()
    
    # Print information about each split
    for split, df in splits.items():
        print(f"\n{split.capitalize()} split info:")
        print(f"Rows: {len(df):,}")
        print("Memory usage: {:.2f} MB".format(df.memory_usage(deep=True).sum() / 1024**2))
        print("Columns:", df.columns.tolist())

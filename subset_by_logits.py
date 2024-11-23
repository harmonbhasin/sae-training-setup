import torch
import numpy as np
from evo.scoring import prepare_batch, logits_to_logprobs

def collect_logits(model, sequences, tokenizer, batch_size=10, device='cuda'):
    """Collect logits and input_ids from sequences in memory-efficient batches."""
    all_logits = []
    all_input_ids = []
    
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        try:
            torch.cuda.empty_cache()
            input_ids, *_ = prepare_batch(
                batch_sequences,
                tokenizer,
                prepend_bos=True,
                device=device
            )
            
            # Print memory usage before forward pass
            gpu_mem_before = torch.cuda.memory_allocated()/1e9
            
            with torch.inference_mode():
                logits, *_ = model(input_ids)
                
                # Print memory at peak (after forward pass, before moving to CPU)
                gpu_mem_peak = torch.cuda.memory_allocated()/1e9
                
                logits = logits.cpu()
                input_ids = input_ids.cpu()
                
                # Print memory after moving to CPU
                gpu_mem_after = torch.cuda.memory_allocated()/1e9
                
                print(f"Batch {i//batch_size + 1} (size {len(batch_sequences)}): "
                      f"GPU Memory - Before: {gpu_mem_before:.2f}GB, "
                      f"Peak: {gpu_mem_peak:.2f}GB, "
                      f"After cleanup: {gpu_mem_after:.2f}GB")
                
                split_logits = torch.split(logits, 1)
                split_input_ids = torch.split(input_ids, 1)
                all_logits.extend([l.detach() for l in split_logits])
                all_input_ids.extend([ids.detach() for ids in split_input_ids])

            del logits, input_ids
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM error at batch {i}, reducing batch size...")
                torch.cuda.empty_cache()
                return collect_logits(
                    model, sequences, tokenizer,
                    batch_size=max(1, batch_size//2),
                    device=device
                )
            raise e
            
    return all_logits, all_input_ids

def process_logits(logits, input_ids, input_sequences):
    """Process collected logits to get scores and high-scoring indices."""
    sequence_scores = []
    
    for logit, ids in zip(logits, input_ids):
        # Calculate logprobs using both logits and input_ids
        logprobs = logits_to_logprobs(logit, ids, trim_bos=True)
        # Get mean score
        score = torch.mean(logprobs, dim=1).item()
        sequence_scores.append(score)
    
    # Convert to numpy array for finding high scores
    scores_array = np.array(sequence_scores)
    median_score = np.median(scores_array)
    high_scoring_indices = np.where(scores_array > median_score)[0]
    
    return sequence_scores, high_scoring_indices

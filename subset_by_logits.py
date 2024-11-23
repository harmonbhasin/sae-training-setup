import torch
import numpy as np
from evo.scoring import prepare_batch, logits_to_logprobs

def collect_logits(model, sequences, tokenizer, batch_size=2, device='cuda'):
    """Collect logits and input_ids from sequences in memory-efficient batches."""
    all_logits = []
    all_input_ids = []
    model = model.cpu()
    
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        try:
            torch.cuda.empty_cache()
            input_ids, *_ = prepare_batch(
                batch_sequences,
                tokenizer,
                prepend_bos=True,
                device='cpu'
            )
            
            model = model.to(device)
            input_ids = input_ids.to(device)
            
            with torch.inference_mode():
                logits, *_ = model(input_ids)
                logits = logits.cpu()
                # Store both logits and input_ids
                split_logits = torch.split(logits, 1)
                split_input_ids = torch.split(input_ids.cpu(), 1)
                
                all_logits.extend([l.detach() for l in split_logits])
                all_input_ids.extend([ids.detach() for ids in split_input_ids])
            
            model = model.cpu()
            del logits, input_ids
            torch.cuda.empty_cache()
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM error at batch {i}, reducing batch size...")
                model = model.cpu()
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

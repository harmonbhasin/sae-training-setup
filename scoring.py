import torch
import numpy as np
from evo.scoring import prepare_batch, logits_to_logprobs

import torch
import numpy as np
from evo.scoring import prepare_batch, logits_to_logprobs

def collect_logits_and_score_batch(model, sequences, tokenizer, batch_size=32, device='cuda'):
    all_logits = []
    sequence_scores = []
    
    # Move model to device
    model = model.to(device)
    
    for i in range(0, len(sequences), batch_size):
        batch_sequences = sequences[i:i + batch_size]
        
        try:
            # Clear cache before processing new batch
            torch.cuda.empty_cache()
            
            input_ids, seq_lengths = prepare_batch(
                batch_sequences,
                tokenizer,
                prepend_bos=True,
                device='cpu'  # First prepare on CPU
            )
            
            # Move to device just before inference
            input_ids = input_ids.to(device)
            
            with torch.inference_mode():
                logits, _ = model(input_ids)
                # Move logits to CPU immediately after computation
                logits = logits.cpu()
                input_ids = input_ids.cpu()
                
                logprobs = logits_to_logprobs(logits, input_ids, trim_bos=True)
                scores = torch.mean(logprobs, dim=1).numpy()
                
                sequence_scores.extend(scores.tolist())
                all_logits.extend([l.detach() for l in torch.split(logits, 1)])
                
            if (i + batch_size) % (batch_size * 10) == 0:
                print(f"Processed {i + batch_size}/{len(sequences)} sequences")
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"OOM error processing batch {i}, reducing batch size...")
                # Move model back to CPU before recursive call
                model = model.cpu()
                torch.cuda.empty_cache()
                return collect_logits_and_score_batch(
                    model, sequences, tokenizer, 
                    batch_size=batch_size//2, device=device
                )
            raise e
            
        torch.cuda.empty_cache()
    
    # Move model back to CPU after processing
    model = model.cpu()
    torch.cuda.empty_cache()
    
    scores_array = np.array(sequence_scores)
    median_score = np.median(scores_array)
    high_scoring_indices = np.where(scores_array > median_score)[0]
    
    return sequence_scores, all_logits, high_scoring_indices

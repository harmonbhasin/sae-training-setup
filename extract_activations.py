import torch
import numpy as np
from evo.scoring import prepare_batch, logits_to_logprobs

def collect_hidden_states(model, sequences, tokenizer, batch_size=2, device='cuda'):
    """
    Collect hidden states from all layers for sequences in batches.
    
    Args:
        model: The Evo model
        sequences: List of sequences to process
        tokenizer: The tokenizer
        batch_size: Number of sequences to process at once
        device: Computing device
    
    Returns:
        dict: Layer index to tensor mapping (layer_idx -> torch.Tensor(total_tokens, hidden_dim))
    """
    all_layer_states = {i: [] for i in range(len(model.blocks))}
    model = model.cpu()
    
    def hook_fn(layer_idx):
        def hook(module, input, output):
            all_layer_states[layer_idx].append(output[0].detach().cpu())
        return hook
    
    hooks = [block.register_forward_hook(hook_fn(i)) for i, block in enumerate(model.blocks)]
    
    try:
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            try:
                torch.cuda.empty_cache()
                input_ids, *_ = prepare_batch(batch, tokenizer, prepend_bos=False, device='cpu')
                
                model = model.to(device)
                with torch.inference_mode():
                    model(input_ids.to(device))
                model = model.cpu()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error at batch {i}, reducing batch size...")
                    model = model.cpu()
                    torch.cuda.empty_cache()
                    for h in hooks:
                        h.remove()
                    return collect_hidden_states(
                        model, sequences, tokenizer,
                        batch_size=max(1, batch_size//2),
                        device=device
                    )
                raise e
    finally:
        for h in hooks:
            h.remove()
    
    return {
        layer: torch.cat([states.reshape(-1, states.shape[-1]) 
                         for states in all_layer_states[layer]], dim=0)
        for layer in all_layer_states
    }

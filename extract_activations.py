import torch
import numpy as np
from evo.scoring import prepare_batch, logits_to_logprobs

def collect_hidden_states(model, sequences, tokenizer, batch_size=2, device='cuda'):
    """
    Optimized version that keeps model on GPU and manages memory better
    """
    all_layer_states = {i: [] for i in range(len(model.blocks))}
    
    def hook_fn(layer_idx):
        def hook(module, input, output):
            # Move only the output to CPU, keep model on GPU
            all_layer_states[layer_idx].append(output[0].detach().cpu())
        return hook
    
    # Keep model on GPU throughout
    model = model.to(device)
    hooks = [block.register_forward_hook(hook_fn(i)) for i, block in enumerate(model.blocks)]
    
    try:
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            try:
                torch.cuda.empty_cache()
                input_ids, *_ = prepare_batch(batch, tokenizer, prepend_bos=False, device=device)
                
                with torch.inference_mode():
                    model(input_ids)
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error at batch {i}, reducing batch size...")
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

def collect_last_hidden_states(model, sequences, tokenizer, batch_size=2, device='cuda'):
    """
    Optimized version that only collects the last layer's activations
    """
    last_layer_idx = len(model.blocks) - 1
    last_layer_states = []
    
    def hook_fn(module, input, output):
        # Move only the output to CPU, keep model on GPU
        last_layer_states.append(output[0].detach().cpu())
    
    # Keep model on GPU throughout
    model = model.to(device)
    # Only register hook for the last layer
    hook = model.blocks[last_layer_idx].register_forward_hook(hook_fn)
    
    try:
        for i in range(0, len(sequences), batch_size):
            batch = sequences[i:i + batch_size]
            try:
                torch.cuda.empty_cache()
                input_ids, *_ = prepare_batch(batch, tokenizer, prepend_bos=False, device=device)
                with torch.inference_mode():
                    model(input_ids)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"OOM error at batch {i}, reducing batch size...")
                    torch.cuda.empty_cache()
                    hook.remove()
                    return collect_last_hidden_states(
                        model, sequences, tokenizer,
                        batch_size=max(1, batch_size//2),
                        device=device
                    )
                raise e
    finally:
        hook.remove()
    
    # Return only the last layer's states
    return torch.cat([states.reshape(-1, states.shape[-1]) 
                     for states in last_layer_states], dim=0)

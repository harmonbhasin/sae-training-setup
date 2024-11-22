def save_model_local(model_name='evo-1-8k-base', save_dir='saved_models'):
    os.makedirs(save_dir, exist_ok=True)
    
    # Download and load model
    evo_model = Evo(model_name)
    
    # Save model and tokenizer
    torch.save(evo_model.model.state_dict(), f'{save_dir}/model.pt')
    torch.save(evo_model.tokenizer, f'{save_dir}/tokenizer.pt')

def load_saved_model(save_dir='saved_models'):
    # Load model architecture first
    evo_model = Evo('evo-1-8k-base', download_weights=False)
    
    # Load saved weights and tokenizer
    evo_model.model.load_state_dict(torch.load(f'{save_dir}/model.pt'))
    evo_model.tokenizer = torch.load(f'{save_dir}/tokenizer.pt')
    
    return evo_model

if __name__ == "__main__":
    load_save_model()

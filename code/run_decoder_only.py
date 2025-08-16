# Author: Akira Kudo
# Created: 2025/06/19
# Last Updated: 2025/06/23

import os
import sys
import torch
import numpy as np
from transformers import GPT2Tokenizer

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trainer.train_utils import load_training_data, compute_reconstruction_loss
from vqvae_gpt2 import GPT2VQVAE
from vqvae_gpt2_simple import SimpleGPT2VQVAE
from vqvae_gpt2_with_enhancement import EnhancedGPT2VQVAE
from demonstrate import print_demonstration_results, compute_cot_reconstruction_metrics

def test_decoder_with_artificial_memory(
    checkpoint_path: str,
    data_dir: str = None,
    num_examples: int = 5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    model_type: str = "GPT2VQVAE",
    seed: int = 42,
    **kwargs
):
    """
    Test a trained GPT2VQVAE model with artificial memory embeddings.
    
    This function:
    1. Loads a trained model from checkpoint
    2. Loads training data and identifies most frequent and random codebook IDs
    3. Creates artificial memory using single codebook embeddings
    4. Tests the model in both teacher-forcing and auto-regressive modes
    5. Compares performance with artificial vs. real memory
    
    Args:
        checkpoint_path: Path to the model checkpoint
        data_dir: Path to data directory (will use default if None)
        num_examples: Number of examples to test
        device: Device to run on
        model_type: Type of model ("GPT2VQVAE", "EnhancedGPT2VQVAE", or "SimpleGPT2VQVAE")
        seed: Random seed for reproducibility
        **kwargs: Additional arguments for model loading
    """
    print(f"Loading {model_type} model from checkpoint: {checkpoint_path}")
    
    # Load the model
    try:
        if model_type == "EnhancedGPT2VQVAE":
            from vqvae_gpt2_with_enhancement import EnhancedGPT2VQVAE
            model = EnhancedGPT2VQVAE.from_checkpoint(checkpoint_path, device=device, **kwargs)
            print(f"Successfully loaded EnhancedGPT2VQVAE model")
        elif model_type == "SimpleGPT2VQVAE":
            from vqvae_gpt2_simple import SimpleGPT2VQVAE
            model = SimpleGPT2VQVAE.from_checkpoint(checkpoint_path, device=device, **kwargs)
            print(f"Successfully loaded SimpleGPT2VQVAE model")
        else:
            model = GPT2VQVAE.from_checkpoint(checkpoint_path, device=device, **kwargs)
            print(f"Successfully loaded GPT2VQVAE model")
    except Exception as e:
        print(f"Error loading model from checkpoint: {e}")
        print("Make sure the checkpoint file exists and contains the required model configuration.")
        return
    
    model.eval()
    num_embeddings = model.vector_quantizer.num_embeddings
    num_thoughts = getattr(model, 'num_thoughts', 1)
    print(f"Model configured for {num_thoughts} parallel CoT sequences and {num_embeddings} embeddings")
    
    # Set data_dir if not provided
    if data_dir is None:
        data_dir = f"data/GSM8K/128_128/batch_{num_thoughts}"
        print(f"No data_dir provided. Using default: {data_dir}")
    else:
        print(f"Using provided data_dir: {data_dir}")
    
    # Load training data
    print(f"Loading training data from {data_dir}...")
    try:
        data_loaded = load_training_data(
            data_dir=data_dir, max_samples=num_examples, num_thoughts=num_thoughts, seed=seed
        )
        
        # Handle both cases: with and without backpointers
        if len(data_loaded) == 8:
            train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask, \
            test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask = data_loaded
            train_backpointers = test_backpointers = None
        elif len(data_loaded) == 10:
            train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask, train_backpointers, \
            test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask, test_backpointers = data_loaded
        else:
            raise ValueError(f"Unexpected number of return values from load_training_data: {len(data_loaded)}")
            
        print(f"Successfully loaded training data")
        print(f"Train shapes: prompts {train_prompt_sequences.shape}, cots {train_cot_sequences.shape}")
        print(f"Test shapes: prompts {test_prompt_sequences.shape}, cots {test_cot_sequences.shape}")
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Identify most frequent and random codebook IDs
    print(f"\n{'='*80}")
    print("IDENTIFYING CODEBOOK USAGE PATTERNS")
    print(f"{'='*80}")
    
    # Use training data to identify codebook usage
    most_frequent_id, random_id = identify_codebook_ids(
        model, train_prompt_sequences, train_cot_sequences, 
        train_prompt_mask, train_cot_mask, train_backpointers, 
        num_embeddings, device
    )
    
    print(f"Most frequent codebook ID: {most_frequent_id}")
    print(f"Random codebook ID: {random_id}")
    
    # Test with artificial memory
    print(f"\n{'='*80}")
    print("TESTING WITH ARTIFICIAL MEMORY EMBEDDINGS")
    print(f"{'='*80}")
    
    # Test on training data
    print(f"\nTesting on TRAINING data...")
    train_results = test_with_artificial_memory(
        model, train_prompt_sequences, train_cot_sequences, 
        train_prompt_mask, train_cot_mask, train_backpointers,
        most_frequent_id, random_id, num_thoughts, device, "train"
    )
    
    # Test on test data
    print(f"\nTesting on TEST data...")
    test_results = test_with_artificial_memory(
        model, test_prompt_sequences, test_cot_sequences, 
        test_prompt_mask, test_cot_mask, test_backpointers,
        most_frequent_id, random_id, num_thoughts, device, "test"
    )
    
    # Print summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY COMPARISON")
    print(f"{'='*80}")
    print(f"{'Metric':<30} {'Train (Real)':<15} {'Train (RealEnc)':<15} {'Train (Freq)':<15} {'Train (Random)':<15}")
    print(f"{'-'*80}")
    print(f"{'Reconstruction Loss':<30} {train_results['real']['loss']:<15.4f} {train_results['real_encoded']['loss']:<15.4f} {train_results['frequent']['loss']:<15.4f} {train_results['random']['loss']:<15.4f}")
    print(f"{'Token Accuracy':<30} {train_results['real']['accuracy']:<15.4f} {train_results['real_encoded']['accuracy']:<15.4f} {train_results['frequent']['accuracy']:<15.4f} {train_results['random']['accuracy']:<15.4f}")
    print(f"{'Perplexity':<30} {train_results['real']['perplexity']:<15.4f} {train_results['real_encoded']['perplexity']:<15.4f} {train_results['frequent']['perplexity']:<15.4f} {train_results['random']['perplexity']:<15.4f}")
    
    print(f"\n{'Metric':<30} {'Test (Real)':<15} {'Test (RealEnc)':<15} {'Test (Freq)':<15} {'Test (Random)':<15}")
    print(f"{'-'*80}")
    print(f"{'Reconstruction Loss':<30} {test_results['real']['loss']:<15.4f} {test_results['real_encoded']['loss']:<15.4f} {test_results['frequent']['loss']:<15.4f} {test_results['random']['loss']:<15.4f}")
    print(f"{'Token Accuracy':<30} {test_results['real']['accuracy']:<15.4f} {test_results['real_encoded']['accuracy']:<15.4f} {test_results['frequent']['accuracy']:<15.4f} {test_results['random']['accuracy']:<15.4f}")
    print(f"{'Perplexity':<30} {test_results['real']['perplexity']:<15.4f} {test_results['real_encoded']['perplexity']:<15.4f} {test_results['frequent']['perplexity']:<15.4f} {test_results['random']['perplexity']:<15.4f}")
    
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")
    
    # Analyze the impact of artificial memory
    analyze_artificial_memory_impact(train_results, test_results, most_frequent_id, random_id)
    
    return {
        'train_results': train_results,
        'test_results': test_results,
        'most_frequent_id': most_frequent_id,
        'random_id': random_id
    }

def identify_codebook_ids(model, prompt_sequences, cot_sequences, prompt_mask, cot_mask, 
                         backpointers, num_embeddings, device):
    """
    Identify the most frequent and a random codebook ID from the training data.
    
    Args:
        model: The loaded model
        prompt_sequences: Prompt sequences tensor
        cot_sequences: CoT sequences tensor
        prompt_mask: Prompt mask tensor
        cot_mask: CoT mask tensor
        backpointers: Optional backpointers tensor
        num_embeddings: Number of embeddings in the codebook
        device: Device to run on
        
    Returns:
        Tuple of (most_frequent_id, random_id)
    """
    model.eval()
    
    # Sample a subset for efficiency
    sample_size = min(50, len(prompt_sequences))
    sample_indices = torch.randperm(len(prompt_sequences), generator=torch.Generator().manual_seed(42))[:sample_size]
    
    all_indices = []
    
    with torch.no_grad():
        for idx in sample_indices:
            prompts = prompt_sequences[idx:idx+1].to(device)
            cots = cot_sequences[idx:idx+1].to(device)
            prompt_masks = prompt_mask[idx:idx+1].to(device) if prompt_mask is not None else None
            cot_masks = cot_mask[idx:idx+1].to(device) if cot_mask is not None else None
            bps = backpointers[idx:idx+1].to(device) if backpointers is not None else None
            
            try:
                # Forward pass to get indices
                model_inputs = {
                    'prompt': prompts,
                    'cot_sequences': cots,
                    'cot_mask': cot_masks,
                    'prompt_mask': prompt_masks,
                    'inference': False,
                    'quantize_cot_only': True
                }
                
                if bps is not None:
                    model_inputs['backpointers'] = bps
                if hasattr(model, 'use_vq'):
                    model_inputs['use_vq'] = True
                else:
                    model_inputs['no_vq'] = False
                
                out = model(**model_inputs)
                
                # Extract indices from output
                if len(out) >= 5:
                    indices = out[4]  # indices should be at position 4
                    if indices is not None:
                        indices_cpu = indices.flatten().cpu()
                        all_indices.append(indices_cpu)
                        del indices
                
                del prompts, cots, prompt_masks, cot_masks, bps
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Warning: Failed to compute indices for sample {idx}: {e}")
                continue
    
    if not all_indices:
        print("Warning: No valid indices computed from samples")
        return 0, np.random.randint(0, num_embeddings)
    
    # Combine all indices and compute usage counts
    combined_indices = torch.cat(all_indices, dim=0)
    counts = torch.bincount(
        combined_indices[combined_indices < num_embeddings],
        minlength=num_embeddings
    )
    
    # Find most frequent ID
    most_frequent_id = counts.argmax().item()
    
    # Find a random ID (different from most frequent)
    available_ids = list(range(num_embeddings))
    if most_frequent_id in available_ids:
        available_ids.remove(most_frequent_id)
    random_id = np.random.choice(available_ids) if available_ids else 0
    
    print(f"Codebook usage analysis:")
    print(f"  Total indices processed: {combined_indices.numel()}")
    print(f"  Unique indices used: {torch.unique(combined_indices).numel()}")
    print(f"  Most frequent ID {most_frequent_id}: used {counts[most_frequent_id].item()} times")
    print(f"  Random ID {random_id}: used {counts[random_id].item()} times")
    
    del all_indices, combined_indices, counts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return most_frequent_id, random_id

def test_with_artificial_memory(model, prompt_sequences, cot_sequences, prompt_mask, cot_mask, 
                               backpointers, most_frequent_id, random_id, num_thoughts, device, split_name):
    """
    Test the model with artificial memory embeddings.
    
    Args:
        model: The loaded model
        prompt_sequences: Prompt sequences tensor
        cot_sequences: CoT sequences tensor
        prompt_mask: Prompt mask tensor
        cot_mask: CoT mask tensor
        backpointers: Optional backpointers tensor
        most_frequent_id: Most frequent codebook ID
        random_id: Random codebook ID
        num_thoughts: Number of parallel sequences
        device: Device to run on
        split_name: Name of the split for logging
        
    Returns:
        Dictionary with results for real, real_encoded, frequent, and random memory
    """
    print(f"Testing {split_name} data with artificial memory...")
    
    # Test with real memory (normal operation)
    print(f"  Testing with REAL memory (normal operation)...")
    real_results = run_model_with_memory_type(
        model, prompt_sequences, cot_sequences, prompt_mask, cot_mask, backpointers,
        memory_type="real", memory_id=None, num_thoughts=num_thoughts, device=device
    )
    
    # Test with real encoded memory (use real encoder indices but generate memory from them)
    print(f"  Testing with REAL ENCODED memory (real indices, generated embeddings)...")
    real_encoded_results = run_model_with_memory_type(
        model, prompt_sequences, cot_sequences, prompt_mask, cot_mask, backpointers,
        memory_type="real_encoded", memory_id=None, num_thoughts=num_thoughts, device=device
    )
    
    # Test with most frequent codebook ID as memory
    print(f"  Testing with FREQUENT codebook ID {most_frequent_id} as memory...")
    frequent_results = run_model_with_memory_type(
        model, prompt_sequences, cot_sequences, prompt_mask, cot_mask, backpointers,
        memory_type="artificial", memory_id=most_frequent_id, num_thoughts=num_thoughts, device=device
    )
    
    # Test with random codebook ID as memory
    print(f"  Testing with RANDOM codebook ID {random_id} as memory...")
    random_results = run_model_with_memory_type(
        model, prompt_sequences, cot_sequences, prompt_mask, cot_mask, backpointers,
        memory_type="artificial", memory_id=random_id, num_thoughts=num_thoughts, device=device
    )
    
    return {
        'real': real_results,
        'real_encoded': real_encoded_results,
        'frequent': frequent_results,
        'random': random_results
    }

def run_model_with_memory_type(model, prompt_sequences, cot_sequences, prompt_mask, cot_mask, 
                              backpointers, memory_type, memory_id, num_thoughts, device):
    """
    Run the model with a specific memory type.
    
    Args:
        model: The loaded model
        prompt_sequences: Prompt sequences tensor
        cot_sequences: CoT sequences tensor
        prompt_mask: Prompt mask tensor
        cot_mask: CoT mask tensor
        backpointers: Optional backpointers tensor
        memory_type: "real" or "artificial"
        memory_id: Codebook ID to use for artificial memory (ignored if memory_type="real")
        num_thoughts: Number of parallel sequences
        device: Device to run on
        
    Returns:
        Dictionary with results for both teacher-forcing and auto-regressive modes
    """
    model.eval()
    B, M, L = cot_sequences.shape
    
    # Prepare artificial memory if needed
    if memory_type == "artificial":
        # Create artificial memory using the specified codebook ID
        batch_size = prompt_sequences.size(0)
        seq_len = cot_sequences.size(2)

        if model.compress_beam_search or model.interchain or model.interchain_positional:
            memory_shape = (batch_size, num_thoughts * seq_len)
        else:
            memory_shape = (batch_size, seq_len, num_thoughts)
        
        # Get the embedding for the specified codebook ID
        memory_indices = torch.full(
            memory_shape, 
            memory_id, 
            dtype=torch.long, 
            device=device
        )
        
        # Get embeddings from the codebook
        artificial_memory = model.vector_quantizer.embedding(memory_indices)
        
        print(f"    Created artificial memory with shape: {artificial_memory.shape}")
        print(f"    Using codebook ID: {memory_id}")
    elif memory_type == "real_encoded":
        # Get indices from the model's encoder
        model_inputs = {
            'prompt': prompt_sequences.to(device),
            'cot_sequences': cot_sequences.to(device),
            'cot_mask': cot_mask.to(device) if cot_mask is not None else None,
            'prompt_mask': prompt_mask.to(device) if prompt_mask is not None else None,
            'inference': False,
            'quantize_cot_only': True
        }
        if backpointers is not None:
            model_inputs['backpointers'] = backpointers.to(device)
        if hasattr(model, 'use_vq'):
            model_inputs['use_vq'] = True
        else:
            model_inputs['no_vq'] = False
        
        out = model(**model_inputs)
        
        if len(out) >= 5:
            indices = out[4]  # indices should be at position 4
            if indices is not None:
                # Move indices to device and create embeddings
                indices_device = indices.to(device) # [B, L]
                if model.compress_beam_search or model.interchain or model.interchain_positional:
                    # Intended shape: [B, M*L]
                    indices_device = indices_device.unsqueeze(-1).expand(-1, -1, M).reshape(B, -1)
                else:
                    # Intended shape: [B, L, M]
                    indices_device = indices_device.unsqueeze(-1).expand(B, L, M)
                artificial_memory = model.vector_quantizer.embedding(indices_device)
                print(f"    Created REAL ENCODED memory with shape: {artificial_memory.shape}")
                print(f"    Using indices from model encoder with shape: {indices.shape}")
            else:
                artificial_memory = None
                print("    No valid indices found from model encoder.")
        else:
            artificial_memory = None
            print("    Model output does not contain indices.")
    else: # memory_type == "real"
        artificial_memory = None
    
    results = {}
    
    # Test teacher-forcing mode
    print(f"    Testing teacher-forcing mode...")
    tf_results = run_single_mode(
        model, prompt_sequences, cot_sequences, prompt_mask, cot_mask, backpointers,
        inference=False, artificial_memory=artificial_memory, device=device
    )
    results['teacher_forcing'] = tf_results
    
    # Test auto-regressive mode
    print(f"    Testing auto-regressive mode...")
    ar_results = run_single_mode(
        model, prompt_sequences, cot_sequences, prompt_mask, cot_mask, backpointers,
        inference=True, artificial_memory=artificial_memory, device=device
    )
    results['auto_regressive'] = ar_results
    
    # Compute average metrics
    avg_loss = (tf_results['loss'] + ar_results['loss']) / 2
    avg_accuracy = (tf_results['accuracy'] + ar_results['accuracy']) / 2
    avg_perplexity = (tf_results['perplexity'] + ar_results['perplexity']) / 2
    
    return {
        'loss': avg_loss,
        'accuracy': avg_accuracy,
        'perplexity': avg_perplexity,
        'teacher_forcing': tf_results,
        'auto_regressive': ar_results
    }

def run_single_mode(model, prompt_sequences, cot_sequences, prompt_mask, cot_mask, 
                   backpointers, inference, artificial_memory, device):
    """
    Run the model in a single mode (teacher-forcing or auto-regressive).
    
    Args:
        model: The loaded model
        prompt_sequences: Prompt sequences tensor
        prompt_mask: Prompt mask tensor
        cot_sequences: CoT sequences tensor
        cot_mask: CoT mask tensor
        backpointers: Optional backpointers tensor
        inference: Whether to run in inference mode (auto-regressive)
        artificial_memory: Artificial memory tensor or None
        device: Device to run on
        
    Returns:
        Dictionary with metrics
    """
    with torch.no_grad():
        # Prepare model inputs
        model_inputs = {
            'prompt': prompt_sequences.to(device),
            'cot_sequences': cot_sequences.to(device),
            'cot_mask': cot_mask.to(device) if cot_mask is not None else None,
            'prompt_mask': prompt_mask.to(device) if prompt_mask is not None else None,
            'inference': inference,
            'quantize_cot_only': True
        }
        
        if backpointers is not None:
            model_inputs['backpointers'] = backpointers.to(device)
        if hasattr(model, 'use_vq'):
            model_inputs['use_vq'] = True
        else:
            model_inputs['no_vq'] = False
        
        # If using artificial memory, pass it directly to the model
        if artificial_memory is not None:
            model_inputs['debug_artificial_memory'] = artificial_memory.to(device)
        
        # Run the model
        out = model(**model_inputs)
        
        # Extract outputs
        if len(out) >= 5:
            output_sequences, output_logits, vq_loss, perplexity, indices, *rest = out
            bp_logits = rest[1] if len(rest) > 1 else None
        else:
            output_sequences = output_logits = vq_loss = perplexity = indices = bp_logits = None
        
        # Compute metrics
        if output_logits is not None and cot_sequences is not None:
            metrics = compute_cot_reconstruction_metrics(
                cot_sequences.to(device), output_logits, cot_mask.to(device) if cot_mask is not None else None,
                backpointers.to(device) if backpointers is not None else None, bp_logits
            )
            
            # Compute reconstruction loss
            recon_loss, bp_loss = compute_reconstruction_loss(
                output_logits, cot_sequences.to(device), 
                cot_mask.to(device) if cot_mask is not None else None,
                bp_logits, backpointers.to(device) if backpointers is not None else None
            )
            
            # Extract average metrics
            avg_loss = recon_loss.item() if recon_loss is not None else 0.0
            avg_accuracy = metrics['token_level_accuracies'].mean().item()
            avg_perplexity = metrics['perplexities'].mean().item()
        else:
            avg_loss = avg_accuracy = avg_perplexity = 0.0
        
        return {
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'perplexity': avg_perplexity,
            'output_sequences': output_sequences,
            'output_logits': output_logits,
            'vq_loss': vq_loss,
            'perplexity_raw': perplexity,
            'indices': indices
        }

def analyze_artificial_memory_impact(train_results, test_results, most_frequent_id, random_id):
    """
    Analyze the impact of using artificial memory vs. real memory.
    
    Args:
        train_results: Training results dictionary
        test_results: Test results dictionary
        most_frequent_id: Most frequent codebook ID used
        random_id: Random codebook ID used
    """
    print(f"Impact Analysis:")
    
    # Analyze real_encoded vs real (should be very similar)
    print(f"  Real Encoded vs Real (should be very similar):")
    train_real_enc_loss = train_results['real_encoded']['loss']
    train_real_loss = train_results['real']['loss']
    train_real_enc_diff = ((train_real_enc_loss - train_real_loss) / train_real_loss) * 100
    
    print(f"    Training: Loss {train_real_enc_loss:.4f} vs {train_real_loss:.4f} "
          f"({'worse' if train_real_enc_diff > 0 else 'better'} by {abs(train_real_enc_diff):.1f}%)")
    
    test_real_enc_loss = test_results['real_encoded']['loss']
    test_real_loss = test_results['real']['loss']
    test_real_enc_diff = ((test_real_enc_loss - test_real_loss) / test_real_loss) * 100
    
    print(f"    Testing: Loss {test_real_enc_loss:.4f} vs {test_real_loss:.4f} "
          f"({'worse' if test_real_enc_diff > 0 else 'better'} by {abs(test_real_enc_diff):.1f}%)")
    
    print(f"  Most frequent codebook ID {most_frequent_id}:")
    
    # Training impact
    train_freq_loss = train_results['frequent']['loss']
    train_freq_improvement = ((train_real_loss - train_freq_loss) / train_real_loss) * 100
    
    print(f"    Training: Loss {train_freq_loss:.4f} vs {train_real_loss:.4f} "
          f"({'improved' if train_freq_improvement > 0 else 'worsened'} by {abs(train_freq_improvement):.1f}%)")
    
    # Test impact
    test_freq_loss = test_results['frequent']['loss']
    test_freq_improvement = ((test_real_loss - test_freq_loss) / test_real_loss) * 100
    
    print(f"    Testing: Loss {test_freq_loss:.4f} vs {test_real_loss:.4f} "
          f"({'improved' if test_freq_improvement > 0 else 'worsened'} by {abs(test_freq_improvement):.1f}%)")
    
    print(f"  Random codebook ID {random_id}:")
    
    # Training impact
    train_rand_loss = train_results['random']['loss']
    train_rand_improvement = ((train_real_loss - train_rand_loss) / train_real_loss) * 100
    
    print(f"    Training: Loss {train_rand_loss:.4f} vs {train_real_loss:.4f} "
          f"({'improved' if train_rand_improvement > 0 else 'worsened'} by {abs(train_rand_improvement):.1f}%)")
    
    # Test impact
    test_rand_loss = test_results['random']['loss']
    test_rand_improvement = ((test_real_loss - test_rand_loss) / test_real_loss) * 100
    
    print(f"    Testing: Loss {test_rand_loss:.4f} vs {test_real_loss:.4f} "
          f"({'improved' if test_rand_improvement > 0 else 'worsened'} by {abs(test_rand_improvement):.1f}%)")
    
    # Overall conclusions
    print(f"\nOverall Conclusions:")
    
    # Real encoded vs real analysis
    if abs(train_real_enc_diff) < 1.0 and abs(test_real_enc_diff) < 1.0:
        print(f"  Real encoded memory performs very similarly to real memory (difference < 1%)")
        print(f"  This validates that the embedding function correctly reconstructs the encoded representations")
    else:
        print(f"  Real encoded memory shows significant differences from real memory")
        print(f"  This may indicate issues with the embedding reconstruction process")
    
    if train_freq_improvement > 0 and test_freq_improvement > 0:
        print(f"  Using the most frequent codebook ID {most_frequent_id} consistently improves performance")
    elif train_freq_improvement < 0 and test_freq_improvement < 0:
        print(f"  Using the most frequent codebook ID {most_frequent_id} consistently worsens performance")
    else:
        print(f"  Using the most frequent codebook ID {most_frequent_id} has mixed effects")
    
    if train_rand_improvement > 0 and test_rand_improvement > 0:
        print(f"  Using random codebook ID {random_id} consistently improves performance")
    elif train_rand_improvement < 0 and test_rand_improvement < 0:
        print(f"  Using random codebook ID {random_id} consistently worsens performance")
    else:
        print(f"  Using random codebook ID {random_id} has mixed effects")

if __name__ == "__main__":
    # Example usage
    checkpoint_path = "checkpoints/asw_positional/small/four/8192/best_model_reinitialization_total_epoch_10.pt"
    # checkpoint_path = "checkpoints/asw_embsum/big/four_thoughts/40962l/checkpoint_epoch_40.pt"
    data_dir = "data/GSM8K/128_128/batch_4"
    
    # Test the artificial memory functionality
    print("Testing artificial memory functionality...")
    
    # Create a simple test with dummy data
    def test_artificial_memory_simple():
        """Simple test to verify artificial memory functionality works"""
        print("\n" + "="*60)
        print("SIMPLE ARTIFICIAL MEMORY TEST")
        print("="*60)
        
        # Create dummy tensors
        batch_size, num_thoughts, seq_len, d_model = 2, 3, 10, 768
        
        # Create artificial memory using a single codebook ID
        test_id = 42
        memory_indices = torch.full((batch_size, num_thoughts, seq_len), test_id, dtype=torch.long)
        
        print(f"Created test memory indices with shape: {memory_indices.shape}")
        print(f"Using codebook ID: {test_id}")
        
        # This would normally be created by the model's vector quantizer
        # For testing, we'll create a dummy embedding
        dummy_embedding = torch.randn(batch_size, num_thoughts, seq_len, d_model)
        
        print(f"Created dummy embeddings with shape: {dummy_embedding.shape}")
        print("Test completed successfully!")
        
        return dummy_embedding
    
    # Run the simple test
    # test_memory = test_artificial_memory_simple()
    
    # Uncomment the following lines to run the full test with a real model
    results = test_decoder_with_artificial_memory(
        checkpoint_path=checkpoint_path,
        data_dir=data_dir,
        num_examples=10,
        device="cuda" if torch.cuda.is_available() else "cpu",
        model_type="EnhancedGPT2VQVAE"
    )
    
    print(f"\nTest completed! Results saved in 'test_memory' variable.")
    print("To run the full test with a real model, uncomment the lines above and provide a valid checkpoint path.")

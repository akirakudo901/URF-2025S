# Author: Akira Kudo
# Created: 2025/06/19
# Last Updated: 2025/06/23

import os
from typing import Any, Dict, Optional, Tuple, List
import json
import yaml

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

import torch
from torch import nn
from torch.utils.data import TensorDataset

from vqvae_gpt2 import compute_perplexity

def _reorganize_sequences(prompt_sequences, cot_sequences, prompt_mask, cot_mask, num_thoughts, split_name):
    """
    Helper method to reorganize sequences when num_thoughts is specified.
    
    Args:
        prompt_sequences: Prompt sequences tensor
        cot_sequences: CoT sequences tensor
        prompt_mask: Prompt mask tensor
        cot_mask: CoT mask tensor
        num_thoughts: Number of parallel sequences to use
        split_name: Name of the split (train/test) for logging
        
    Returns:
        Tuple of reorganized tensors
    """
    current_num_thoughts = cot_sequences.shape[1]
    
    # Calculate how many multiples of num_thoughts can fit within current_num_thoughts
    num_batches = current_num_thoughts // num_thoughts
    remainder = current_num_thoughts % num_thoughts
    
    if remainder > 0:
        print(f"Warning: {split_name} dataset has {current_num_thoughts} sequences, not perfectly divisible by {num_thoughts}")
        print(f"Will use {num_batches * num_thoughts} sequences (dropping {remainder} sequences)")
    
    print(f"Reorganizing {split_name} dataset: {current_num_thoughts} sequences → {num_batches} batches of {num_thoughts} sequences each")
    
    # Calculate new dataset size (each original sample becomes num_batches samples)
    original_batch_size = prompt_sequences.shape[0]
    new_batch_size = original_batch_size * num_batches
    usable_sequences = num_batches * num_thoughts
    
    def reorganize_sequence_pair(sequence_1d, sequence_2d):
        """Helper to reorganize a pair of prompt/cot sequences or masks"""
        # Repeat 1D sequence to [batch_size * num_batches, seq_len] 
        sequence_1d = sequence_1d.unsqueeze(1).repeat(1, num_batches, 1).reshape(-1, sequence_1d.size(-1))
        
        # Reshape 2D sequence to [batch_size * num_batches, num_thoughts, seq_len]
        sequence_2d = sequence_2d[:, :usable_sequences, :]  # Remove remainder
        sequence_2d = sequence_2d.view(original_batch_size, num_batches, num_thoughts, -1)
        sequence_2d = sequence_2d.transpose(1, 2).contiguous().view(new_batch_size, num_thoughts, -1)
        
        return sequence_1d, sequence_2d
    
    # Reorganize sequences and masks
    prompt_sequences, cot_sequences = reorganize_sequence_pair(prompt_sequences, cot_sequences)
    prompt_mask, cot_mask = reorganize_sequence_pair(prompt_mask, cot_mask)
    
    print(f"Reorganized {split_name} data shapes:")
    print(f"  prompt_sequences: {prompt_sequences.shape}")
    print(f"  cot_sequences: {cot_sequences.shape}")
    print(f"  prompt_mask: {prompt_mask.shape}")
    print(f"  cot_mask: {cot_mask.shape}")
    print(f"  {split_name} dataset size increased from {original_batch_size} to {new_batch_size} samples")
    
    return prompt_sequences, cot_sequences, prompt_mask, cot_mask

def validate_model_data_compatibility(model_config: Dict[str, Any], 
                                    prompt_sequences: torch.Tensor,
                                    cot_sequences: torch.Tensor,
                                    prompt_mask: torch.Tensor,
                                    cot_mask: torch.Tensor) -> None:
    """
    Validate that model configuration is compatible with the loaded data.
    
    Args:
        model_config: Model configuration dictionary
        prompt_sequences: Prompt sequences tensor
        cot_sequences: Chain-of-thought sequences tensor
        prompt_mask: Prompt mask tensor
        cot_mask: Chain-of-thought mask tensor
    """
    print("Validating model-data compatibility...")
    
    # Check num_thoughts compatibility
    num_thoughts = model_config.get('num_thoughts', None)
    if num_thoughts is not None:
        data_num_thoughts = cot_sequences.shape[1]
        if data_num_thoughts != num_thoughts:
            print(f"⚠️  Warning: Data has {data_num_thoughts} parallel sequences, "
                  f"but model config specifies {num_thoughts}.")
            if data_num_thoughts > num_thoughts:
                print(f"   → Dataset will be truncated to {num_thoughts} sequences during loading.")
            else:
                print(f"   → Error: Dataset has insufficient parallel sequences!")
                raise ValueError(f"Dataset only has {data_num_thoughts} parallel sequences, "
                               f"but model requires {num_thoughts}.")
        else:
            print(f"✅ num_thoughts compatibility: {num_thoughts} sequences")
    
    # Check vocabulary size compatibility
    vocab_size = model_config.get('vocab_size', None)
    if vocab_size is not None:
        # Check if any token IDs exceed the vocabulary size
        max_prompt_token = prompt_sequences.max().item()
        max_cot_token = cot_sequences.max().item()
        max_token = max(max_prompt_token, max_cot_token)
        
        if max_token >= vocab_size:
            print(f"⚠️  Warning: Data contains token ID {max_token}, "
                  f"but model vocab_size is {vocab_size}.")
        else:
            print(f"✅ Vocabulary compatibility: max token {max_token} < vocab_size {vocab_size}")
    
    # Check sequence length compatibility
    n_positions = model_config.get('n_positions', None)
    if n_positions is not None:
        prompt_len = prompt_sequences.shape[1]
        cot_len = cot_sequences.shape[2]
        total_len = prompt_len + cot_len
        
        if total_len > n_positions:
            print(f"⚠️  Warning: Total sequence length {total_len} exceeds model's n_positions {n_positions}.")
        else:
            print(f"✅ Sequence length compatibility: {total_len} <= n_positions {n_positions}")
    
    # Check tensor shapes consistency
    batch_size = prompt_sequences.shape[0]
    expected_shapes = {
        'prompt_sequences': (batch_size, prompt_sequences.shape[1]),
        'cot_sequences': (batch_size, cot_sequences.shape[1], cot_sequences.shape[2]),
        'prompt_mask': (batch_size, prompt_mask.shape[1]),
        'cot_mask': (batch_size, cot_mask.shape[1], cot_mask.shape[2])
    }
    
    actual_shapes = {
        'prompt_sequences': prompt_sequences.shape,
        'cot_sequences': cot_sequences.shape,
        'prompt_mask': prompt_mask.shape,
        'cot_mask': cot_mask.shape
    }
    
    print("✅ Tensor shape consistency:")
    for name, expected_shape in expected_shapes.items():
        actual_shape = actual_shapes[name]
        if actual_shape == expected_shape:
            print(f"   {name}: {actual_shape}")
        else:
            print(f"   ⚠️  {name}: expected {expected_shape}, got {actual_shape}")
    
    print("✅ Model-data compatibility validation complete!")

def load_config(config_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load configuration from a YAML or JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Tuple of (model_config, training_config)
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.endswith('.yaml') or config_path.endswith('.yml'):
            config = yaml.safe_load(f)
        elif config_path.endswith('.json'):
            config = json.load(f)
        else:
            raise ValueError("Configuration file must be .yaml, .yml, or .json")
    
    # Extract model and training configs
    model_config = config.get('model_config', {})
    training_config = config.get('training_config', {})
    
    # Validate required fields
    required_model_fields = ['vocab_size', 'd_model', 'num_embeddings']
    required_training_fields = ['learning_rate', 'num_epochs', 'batch_size']
    
    missing_model = [field for field in required_model_fields if field not in model_config]
    missing_training = [field for field in required_training_fields if field not in training_config]
    
    if missing_model:
        raise ValueError(f"Missing required model config fields: {missing_model}")
    if missing_training:
        raise ValueError(f"Missing required training config fields: {missing_training}")
    
    return model_config, training_config


def create_default_config(output_path: str, enhanced_vq: bool = False, phased_training: bool = False, memory_optimized_tracking: bool = False, only_latent_decode: bool = False):
    """
    Create a default configuration file with memory optimizations.
    
    Args:
        output_path: Path where to save the default config
        enhanced_vq: Whether to include enhanced VQ-VAE configuration options
        phased_training: Whether to include phased training configuration options (requires enhanced_vq=True)
        memory_optimized_tracking: Whether to use memory-optimized codebook tracking
        only_latent_decode: Whether to enable only_latent_decode mode (default: False)
    """
    # Validate phased training requires enhanced VQ-VAE
    if phased_training and not enhanced_vq:
        raise ValueError("Phased training requires enhanced VQ-VAE (enhanced_vq=True)")
    
    # Base model configuration
    model_config = {
        'vocab_size': 50257,  # GPT2 vocabulary size
        'd_model': 768,       # GPT2 model dimension
        'num_embeddings': 512,  # VQ codebook size
        'commitment_cost': 0.25,  # VQ commitment cost
        'aggregation_hidden_dim': 1024,  # Aggregation MLP hidden dim
        'num_thoughts': 40,   # Number of parallel sequences
        'n_positions': 1024,   # Maximum sequence length
        'only_latent_decode': False, # If True, decoder ignores cross-attention and decodes from prompt embeddings + latents only
        # Pretrained model settings
        'use_pretrained_encoder': True,  # Load pretrained weights for encoder
        'use_pretrained_decoder': True,  # and decoder
        'pretrained_model_name': 'gpt2',  # Use GPT2-Small (124M parameters)
        # Encoder-specific configuration (smaller, more efficient)
        "encoder_n_layer": 6,        # Smaller encoder
        "encoder_n_head": 12,
        "encoder_n_inner": None,     # Will be set to 4*d_model
        "encoder_dropout": 0.1,
        "encoder_activation_function": "gelu",
        # Decoder-specific configuration (larger, more powerful)
        "decoder_n_layer": 12,       # Larger decoder
        "decoder_n_head": 12,
        "decoder_n_inner": None,     # Will be set to 4*d_model
        "decoder_dropout": 0.1,
        "decoder_activation_function": "gelu"
    }
    
    # Add enhanced VQ-VAE specific parameters if requested
    if enhanced_vq:
        model_config.update({
            # Enhanced Vector Quantizer specific parameters
            'ema_decay': 0.99,           # EMA decay rate for codebook updates
            'diversity_gamma': 0.1,      # Weight for diversity-promoting loss
            'reset_threshold': 0.1,      # Threshold for codebook reset (usage ratio)
            'reset_frequency': 1000,     # Frequency of codebook reset checks
            'use_ema': True,             # Whether to use EMA updates
            'reset_stop_fraction': 0.2,  # Fraction of training during which resets are allowed (will be converted to max_reset_steps)
        })
    
    # Base training configuration
    training_config = {
        'learning_rate': 1e-4,
        'weight_decay': 0.01,
        'beta1': 0.9,
        'beta2': 0.999,
        'use_lr_scheduler': True,
        'min_lr': 1e-6,
        'num_epochs': 50,
        'batch_size': 2,  # Reduced for memory efficiency
        'gradient_clip': 1.0,
        'vq_loss_weight': 1.0,
        'quantize_cot_only': True,
        'save_every': 5,
        'checkpoint_dir': 'checkpoints/gpt2vqvae',
        'pad_token_id': 50256, # eos_token for GPT2TokenizerFast
        'val_split': 0.1,
        'num_measurements_per_epoch': 20,  # Number of detailed metrics per epoch
        # Memory optimization settings
        'use_mixed_precision': True,
        'gradient_accumulation_steps': 4,  # Effective batch size = batch_size * gradient_accumulation_steps
        'use_gradient_checkpointing': True,  # Enable gradient checkpointing for memory efficiency
        'use_dynamic_batching': False,  # Enable for variable sequence lengths
        'max_tokens_per_batch': 8192,  # For dynamic batching
        'max_samples': None,  # Limit samples for debugging (set to number for testing)
        # Perplexity threshold settings
        'perplexity_threshold': 1.5,  # Training aborts when 20-step average perplexity goes below this value
        'perplexity_window_size': 20,  # Number of steps to average for perplexity threshold check
        # Checkpoint settings
        'minimum_batches_for_checkpoint': 200,  # Minimum number of batches trained before saving aborted checkpoint
        # Data config
        'data_dir': 'data/GSM8K'
    }
    
    # Add enhanced VQ-VAE specific training parameters if requested
    if enhanced_vq:
        training_config.update({
            # Enhanced codebook tracking
            'enhanced_codebook_tracking': True,  # Enable enhanced codebook monitoring
        })
        
        # Add memory-optimized tracking configuration if requested
        if memory_optimized_tracking:
            training_config.update({
                # Granular codebook tracking control for memory optimization
                'codebook_tracking_config': {
                    'track_usage_stats': True,        # Track basic usage statistics
                    'track_diversity_metrics': False, # Disable diversity metrics (expensive)
                    'track_ema_stats': False,         # Disable EMA statistics (expensive)
                    'track_reset_stats': True,        # Track reset statistics
                    'track_reservoir_stats': False,   # Disable reservoir statistics
                    'save_tracking_history': False,   # Don't save history to save memory
                    'print_tracking_info': True,      # Still print basic info
                }
            })
        else:
            # Default tracking configuration (all enabled)
            training_config.update({
                'codebook_tracking_config': {
                    'track_usage_stats': True,
                    'track_diversity_metrics': True,
                    'track_ema_stats': True,
                    'track_reset_stats': True,
                    'track_reservoir_stats': True,
                    'save_tracking_history': True,
                    'print_tracking_info': True,
                }
            })
        
        # Add phased training parameters if requested
        if phased_training:
            training_config.update({
                # Phased training parameters
                'initialization_steps': 1500,    # Steps to train in no-vq mode
                'r_reestim': 500,                # Frequency of codebook reinitialization
                'quantization_start': 5000,      # Step to start normal VQ training
                'codebook_lr_multiplier': 1.0,   # Codebook learning rate multiplier
            })
    
    default_config = {
        'model_config': model_config,
        'training_config': training_config
    }
    
    # Determine file format based on extension
    if output_path.endswith('.yaml') or output_path.endswith('.yml'):
        with open(output_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
    elif output_path.endswith('.json'):
        with open(output_path, 'w') as f:
            json.dump(default_config, f, indent=2)
    else:
        # Default to YAML
        output_path = output_path + '.yaml'
        with open(output_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False, indent=2)
    
    print(f"Default configuration saved to: {output_path}")
    print("Memory optimization settings included:")
    print("  - Mixed precision training: Enabled")
    print("  - Gradient accumulation: 4 steps")
    print("  - Gradient checkpointing: Enabled")
    print("  - Memory-efficient dataset: Enabled")
    print("  - Reduced batch size: 2 (effective batch size: 8)")
    print("  - Detailed metrics logging: 20 measurements per epoch")
    print("  - Perplexity threshold monitoring: 1.5 (aborts training when 20-step average < 1.5)")
    print("  - Minimum batches for checkpoint: 200 (only save aborted checkpoints if trained enough)")
    print("Pretrained model settings:")
    print("  - Encoder: GPT2-Small pretrained weights (use_pretrained_encoder: True)")
    print("  - Decoder: GPT2-Small pretrained weights (use_pretrained_decoder: True)")
    
    if enhanced_vq:
        print("\nEnhanced VQ-VAE settings included:")
        print("  - EMA updates: Enabled (decay: 0.99)")
        print("  - Diversity loss: Enabled (gamma: 0.1)")
        print("  - Automatic codebook reset: Enabled (threshold: 0.1, frequency: 1000)")
        print("  - Reset stop fraction: 0.2 (resets only allowed for first 20% of training)")
        print("  - Enhanced codebook tracking: Enabled")
        print("\nEnhanced codebook training scheme reduces to normal VQ-VAE when:")
        print("  - ema_decay = 0.0 (no EMA updates)")
        print("  - diversity_gamma = 0.0 (no diversity loss)")
        print("  - reset_threshold = 0.0 (no automatic resets)")
        print("  - use_ema = False (EMA disabled)")
        print("  - reset_stop_fraction = 0.0 (no reset timing limit)")
        
        if phased_training:
            print("\nPhased training settings included:")
            print("  - Initialization phase: 0 to 1500 steps (no-vq mode)")
            print("  - Reinitialization phase: 1500 to 5000 steps (VQ + periodic reinitialization)")
            print("  - Reinitialization frequency: every 500 steps")
            print("  - Normal training phase: after 5000 steps")
            print("  - Codebook learning rate multiplier: 1.0x (no difference)")
            print("\nPhased training recipe:")
            print("  1. Train without VQ to learn good representations")
            print("  2. Periodically reinitialize codebook using KMeans on reservoir samples")
            print("  3. Switch to normal VQ-VAE training with all enhancements")
    
    print("\nYou can modify this file and use it for training.")

def compute_reconstruction_loss(output_logits: torch.Tensor, 
                              target_sequences: torch.Tensor, 
                              target_mask: torch.Tensor,
                              pad_token_id: int = 50256,
                              reduction="mean") -> torch.Tensor:
    """
    Compute reconstruction loss between predicted logits and target sequences.
    
    Args:
        output_logits: Predicted logits [batch_size, M, L, vocab_size]
        target_sequences: Target sequences [batch_size, M, L]
        target_mask: Target mask [batch_size, M, L]
        pad_token_id: Token ID for padding (to ignore in loss computation)
        reduction: optionally specifies if the criterion reduces, in case we wanna compute per-item loss
        
    Returns:
        torch.Tensor: Reconstruction loss only
    """
    # Flatten all dimensions except vocab_size
    logits_flat = output_logits.reshape(-1, output_logits.size(-1))
    targets_flat = target_sequences.view(-1)
    mask_flat = target_mask.view(-1).bool()
    
    # Create loss function
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction=reduction)
    
    # Compute reconstruction loss
    recon_loss = torch.tensor(0.0, device=output_logits.device)
    if mask_flat.sum() > 0:
        recon_loss = criterion(logits_flat[mask_flat], targets_flat[mask_flat])
    
    return recon_loss

def create_codebook_usage_heatmap(counts_np: np.ndarray, 
                                 num_embeddings: int,
                                 title: str = "Codebook Usage Heatmap",
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (12, 8),
                                 cmap: str = 'viridis',
                                 show_counts: bool = True) -> None:
    """
    Create and optionally save a heatmap showing codebook usage from counts.
    
    Args:
        counts_np (np.ndarray): Numpy array of codebook usage counts [num_embeddings]
        num_embeddings (int): Total number of embeddings in the codebook
        title (str): Title for the heatmap
        save_path (Optional[str]): Path to save the heatmap image (if None, only displays)
        figsize (Tuple[int, int]): Figure size (width, height)
        cmap (str): Colormap for the heatmap
        show_counts (bool): Whether to show count values on the heatmap
        
    Example:
        >>> counts = np.array([10, 5, 3, 0, 2])  # Usage counts for 5 embeddings
        >>> create_codebook_usage_heatmap(counts, num_embeddings=5, save_path="heatmap.png")
    """
    # Ensure counts is the right shape
    if counts_np.ndim > 1:
        counts_np = counts_np.flatten()
    
    
    # Create the heatmap
    _, ax = plt.subplots(figsize=figsize)
    
    # Create a 2D array for the heatmap (reshape to make it more visually appealing)
    # Try to make it roughly square-ish
    cols = int(np.ceil(np.sqrt(num_embeddings)))
    rows = int(np.ceil(num_embeddings / cols))
    
    # Pad with zeros if needed
    padded_size = rows * cols
    counts_padded = np.zeros(padded_size)
    counts_padded[:num_embeddings] = counts_np
    
    # Reshape to 2D
    heatmap_data = counts_padded.reshape(rows, cols)
    
    # Create a custom colormap that makes zeros white and uses the original colormap for non-zero values
    
    # Get the original colormap
    original_cmap = plt.get_cmap(cmap)
    
    # Create a custom colormap with white for zeros and the original colormap for non-zeros
    # We'll use a very small value (like 0.001) to represent zeros in the colormap
    # and then replace those values with white
    n_colors = 256
    colors = original_cmap(np.linspace(0, 1, n_colors))
    
    # Replace the first color with white to represent zeros
    colors[0] = [0, 0, 0, 0.9]  # White with full opacity
    
    # Create the custom colormap
    custom_cmap = ListedColormap(colors)
    
    # Create the heatmap with the custom colormap
    im = ax.imshow(heatmap_data, cmap=custom_cmap, aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Usage Count', rotation=270, labelpad=15)
    
    # Update colorbar ticks to show the actual range (excluding the artificial 0.001 value)
    # Get the actual min and max values from the original data
    actual_min = np.min(heatmap_data[heatmap_data > 0]) if np.any(heatmap_data > 0) else 0
    actual_max = np.max(heatmap_data)
    
    # Set colorbar ticks to show meaningful values
    if actual_max > 0:
        # Create ticks that include 0 and some intermediate values
        tick_values = [0] + list(np.linspace(actual_min, actual_max, 5))
        tick_values = [int(v) for v in tick_values if v >= 0]
        cbar.set_ticks(tick_values)
        cbar.set_ticklabels([str(v) for v in tick_values])
    
    # Set title and labels
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Column', fontsize=12)
    ax.set_ylabel('Row', fontsize=12)
    
    # Add count annotations if requested
    if show_counts:
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < num_embeddings:
                    count = int(heatmap_data[i, j])
                    # For zero values, use black text on white background
                    if count == 0:
                        text_color = 'white'
                    else:
                        # Choose text color based on background brightness for non-zero values
                        text_color = 'white' if count < np.max(heatmap_data) / 2 else 'black'
                    ax.text(j, i, str(count), ha='center', va='center', 
                           color=text_color, fontweight='bold')
    
    # Add statistics text
    total_usage = counts_np.sum()
    unique_usage = (counts_np > 0).sum()
    usage_percentage = (unique_usage / num_embeddings) * 100
    
    stats_text = f'Total usage: {total_usage}\nUnique codes: {unique_usage}/{num_embeddings} ({usage_percentage:.1f}%)'
    ax.text(0.02, 1.08, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Codebook usage heatmap saved to: {save_path}")
    
    plt.close()

def create_codebook_usage_timeline_plot(codebook_history: List[np.ndarray], 
                                      num_embeddings: int,
                                      measurement_points: List[int],
                                      title: str = "Codebook Usage Over Time",
                                      save_path: Optional[str] = None,
                                      figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Create a 3D visualization showing codebook usage distribution over time.
    
    Args:
        codebook_history: List of count numpy arrays, one per measurement point
        num_embeddings: Total number of embeddings in the codebook
        measurement_points: List of measurement point indices (e.g., batch numbers)
        title: Title for the plot
        save_path: Optional path to save the plot
        figsize: Figure size (width, height)
    """
    if not codebook_history:
        print("Warning: No codebook history to plot")
        return
    
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax: Any = fig.add_subplot(111, projection='3d')  # Type annotation for 3D axes
    
    # Create meshgrid for 3D surface
    x = np.arange(num_embeddings)  # Codebook indices
    y = np.array(measurement_points)  # Time points
    X, Y = np.meshgrid(x, y)
    
    # Create Z matrix (usage counts over time)
    Z = np.array(codebook_history)
    
    # Create 3D surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Set labels and title
    ax.set_xlabel('Codebook Index')
    ax.set_ylabel('Measurement Point')
    ax.set_zlabel('Usage Count')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add statistics text
    total_measurements = len(codebook_history)
    avg_unique_codes = np.mean([(counts > 0).sum() for counts in codebook_history])
    stats_text = f'Total measurements: {total_measurements}\nAvg unique codes: {avg_unique_codes:.1f}'
    ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Codebook usage timeline plot saved to: {save_path}")
    
    plt.close()

def sample_and_compute_codebook_usage(model: Any,  # Changed from GPT2VQVAE to Any to handle both model types
                                    dataset: TensorDataset,  # More specific type
                                    sample_size: int = 100,
                                    device: str = "cuda",
                                    use_vq: bool = True) -> Tuple[torch.Tensor, float]:
    """
    Randomly sample examples from dataset and compute codebook usage statistics.
    
    Args:
        model: The VQ-VAE model (GPT2VQVAE or SimpleGPT2VQVAE)
        dataset: Dataset to sample from
        sample_size: Number of examples to sample
        device: Device to run computation on
        use_vq: Whether to use vector quantization (for SimpleGPT2VQVAE)
        
    Returns:
        Tuple of (indices_tensor, perplexity)
    """
    model_was_training = model.training
    model.eval()
    
    # Randomly sample indices
    total_samples = len(dataset)
    if sample_size > total_samples:
        sample_size = total_samples
        print(f"Warning: Requested sample_size {sample_size} exceeds dataset size {total_samples}")
    
    sample_indices = torch.randperm(total_samples, generator=torch.Generator().manual_seed(42))[:sample_size]
    
    # Collect all indices from sampled examples
    all_indices = []
    
    with torch.no_grad():
        for idx in sample_indices:
            prompts, cots, prompt_masks, cot_masks = dataset[idx]
            
            # Move to device
            prompts = prompts.unsqueeze(0).to(device)  # Add batch dimension
            cots = cots.unsqueeze(0).to(device)
            prompt_masks = prompt_masks.unsqueeze(0).to(device) if prompt_masks is not None else None
            cot_masks = cot_masks.unsqueeze(0).to(device) if cot_masks is not None else None
            
            # Forward pass to get indices
            try:
                # Check if model is SimpleGPT2VQVAE and pass use_vq parameter
                if model.__class__.__name__ == 'SimpleGPT2VQVAE':
                    _, _, _, _, indices = model(
                        prompt=prompts,
                        cot_sequences=cots,
                        cot_mask=cot_masks,
                        prompt_mask=prompt_masks,
                        inference=False,
                        quantize_cot_only=True,
                        use_vq=use_vq
                    )
                else:
                    # Fallback for other model types (GPT2VQVAE, etc.)
                    _, _, _, _, indices, _ = model(
                        prompt=prompts,
                        cot_sequences=cots,
                        cot_mask=cot_masks,
                        prompt_mask=prompt_masks,
                        inference=False,
                        quantize_cot_only=True
                    )
                
                if indices is not None:
                    # Move indices to CPU immediately to prevent GPU memory accumulation
                    indices_cpu = indices.flatten().cpu()
                    all_indices.append(indices_cpu)
                    # Explicitly delete GPU tensor
                    del indices
                    
            except Exception as e:
                print(f"Warning: Failed to compute indices for sample {idx}: {e}")
                continue
            finally:
                # Clear intermediate tensors
                del prompts, cots, prompt_masks, cot_masks
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    if not all_indices:
        print("Warning: No valid indices computed from samples")
        return torch.zeros(model.vector_quantizer.num_embeddings, dtype=torch.long), 0.0
    
    # Combine all indices (all should be on CPU now)
    combined_indices = torch.cat(all_indices, dim=0)
    
    # Compute usage counts using numpy's bincount
    counts = torch.from_numpy(
        np.bincount(
            combined_indices.numpy(),
            minlength=model.vector_quantizer.num_embeddings
        )
    ).long()
    
    # Compute perplexity from counts
    perplexity = compute_perplexity(counts, "counts")

    if model_was_training:
        model.train()
    
    # Clear intermediate tensors
    del all_indices, combined_indices
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Ensure counts are on CPU to prevent GPU memory accumulation
    return counts, perplexity.item()

def load_training_data(data_dir: str, 
                       max_samples: Optional[int] = None, 
                       num_thoughts: Optional[int] = None,
                       seed: Optional[int] = 42
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load training and test data with memory-efficient loading and truncate based on num_thoughts.
    
    Args:
        data_dir: Directory containing the preprocessed data files (with train/ and test/ subdirectories)
        max_samples: Maximum number of samples to load (for debugging/memory constraints)
        num_thoughts: Number of parallel sequences to use (truncates dataset if needed)
        seed: Seed for sampling samples randomly when max_samples is specified
        
    Returns:
        Tuple of (train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask,
                    test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask)
    """
    # Define train and test directories
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    
    required_files = [
        "prompt_sequences.pt",
        "cot_sequences_tensor.pt", 
        "prompt_mask.pt",
        "cot_mask.pt"
    ]
    
    # Check if train and test directories exist
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    
    # Check if all required files exist in both directories
    missing_files = []
    for split_dir, split_name in [(train_dir, "train"), (test_dir, "test")]:
        for file_name in required_files:
            file_path = os.path.join(split_dir, file_name)
            if not os.path.exists(file_path):
                missing_files.append(f"{split_name}/{file_name}")
    
    if missing_files:
        raise FileNotFoundError(f"Missing data files: {missing_files}")
    
    # Load train tensors with memory mapping if available
    try:
        print(f"Loading train data from {train_dir}...")
        train_prompt_sequences = torch.load(os.path.join(train_dir, "prompt_sequences.pt"), map_location='cpu')
        train_cot_sequences = torch.load(os.path.join(train_dir, "cot_sequences_tensor.pt"), map_location='cpu')
        train_prompt_mask = torch.load(os.path.join(train_dir, "prompt_mask.pt"), map_location='cpu')
        train_cot_mask = torch.load(os.path.join(train_dir, "cot_mask.pt"), map_location='cpu')
    except Exception as e:
        print(f"Warning: Could not use memory mapping for train data: {e}")
        # Fallback to regular loading
        train_prompt_sequences = torch.load(os.path.join(train_dir, "prompt_sequences.pt"))
        train_cot_sequences = torch.load(os.path.join(train_dir, "cot_sequences_tensor.pt"))
        train_prompt_mask = torch.load(os.path.join(train_dir, "prompt_mask.pt"))
        train_cot_mask = torch.load(os.path.join(train_dir, "cot_mask.pt"))
    
    # Load test tensors with memory mapping if available
    try:
        print(f"Loading test data from {test_dir}...")
        test_prompt_sequences = torch.load(os.path.join(test_dir, "prompt_sequences.pt"), map_location='cpu')
        test_cot_sequences = torch.load(os.path.join(test_dir, "cot_sequences_tensor.pt"), map_location='cpu')
        test_prompt_mask = torch.load(os.path.join(test_dir, "prompt_mask.pt"), map_location='cpu')
        test_cot_mask = torch.load(os.path.join(test_dir, "cot_mask.pt"), map_location='cpu')
    except Exception as e:
        print(f"Warning: Could not use memory mapping for test data: {e}")
        # Fallback to regular loading
        test_prompt_sequences = torch.load(os.path.join(test_dir, "prompt_sequences.pt"))
        test_cot_sequences = torch.load(os.path.join(test_dir, "cot_sequences_tensor.pt"))
        test_prompt_mask = torch.load(os.path.join(test_dir, "prompt_mask.pt"))
        test_cot_mask = torch.load(os.path.join(test_dir, "cot_mask.pt"))
    
    print(f"Train data shapes:")
    print(f"  prompt_sequences: {train_prompt_sequences.shape}")
    print(f"  cot_sequences: {train_cot_sequences.shape}")
    print(f"  prompt_mask: {train_prompt_mask.shape}")
    print(f"  cot_mask: {train_cot_mask.shape}")
    
    print(f"Test data shapes:")
    print(f"  prompt_sequences: {test_prompt_sequences.shape}")
    print(f"  cot_sequences: {test_cot_sequences.shape}")
    print(f"  prompt_mask: {test_prompt_mask.shape}")
    print(f"  cot_mask: {test_cot_mask.shape}")
    
    # Validate and reorganize based on num_thoughts for both train and test
    if num_thoughts is not None:
        # Check train data
        train_current_num_thoughts = train_cot_sequences.shape[1]
        test_current_num_thoughts = test_cot_sequences.shape[1]
        
        print(f"Current num_thoughts in train dataset: {train_current_num_thoughts}")
        print(f"Current num_thoughts in test dataset: {test_current_num_thoughts}")
        print(f"Requested num_thoughts: {num_thoughts}")
        
        if train_current_num_thoughts < num_thoughts or test_current_num_thoughts < num_thoughts:
            raise ValueError(f"Dataset only has {min(train_current_num_thoughts, test_current_num_thoughts)} parallel sequences, "
                            f"but model requires {num_thoughts}. Please regenerate dataset with more sequences.")
        
        # Process train data if needed
        if train_current_num_thoughts > num_thoughts:
            train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask = _reorganize_sequences(
                train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask, num_thoughts, "train"
            )
        
        # Process test data if needed
        if test_current_num_thoughts > num_thoughts:
            test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask = _reorganize_sequences(
                test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask, num_thoughts, "test"
            )
    
    # Sample the data using the random indices
    if max_samples is not None:
        # Generate random indices for sampling
        torch.manual_seed(seed)  # For reproducible sampling
            
        def pick_N_random_samples(prompt_sequences, cot_sequences, prompt_mask, cot_mask, N, type):
            # Limit samples if specified
            total_examples = len(prompt_sequences)
        
            # Randomly sample num_examples
            if N > total_examples:
                print(f"Warning: Requested {N} examples but only {total_examples} {type} examples available. Using all examples.")
                N = total_examples

            sample_indices = torch.randperm(total_examples)[:N]

            prompt_sequences = prompt_sequences[sample_indices]
            cot_sequences = cot_sequences[sample_indices]
            prompt_mask = prompt_mask[sample_indices]
            cot_mask = cot_mask[sample_indices]
            return prompt_sequences, cot_sequences, prompt_mask, cot_mask
        
        train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask = pick_N_random_samples(
            train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask, max_samples, "training"
            )
        
        test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask = pick_N_random_samples(
            test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask, max_samples, "testing"
            )

        print(f"Randomly sampled {len(train_prompt_sequences)} training examples and {len(test_prompt_sequences)} testing examples using seed {seed}")
    
    print(f"Final train data shapes:")
    print(f"  prompt_sequences: {train_prompt_sequences.shape}")
    print(f"  cot_sequences: {train_cot_sequences.shape}")
    print(f"  prompt_mask: {train_prompt_mask.shape}")
    print(f"  cot_mask: {train_cot_mask.shape}")
    
    print(f"Final test data shapes:")
    print(f"  prompt_sequences: {test_prompt_sequences.shape}")
    print(f"  cot_sequences: {test_cot_sequences.shape}")
    print(f"  prompt_mask: {test_prompt_mask.shape}")
    print(f"  cot_mask: {test_cot_mask.shape}")
    
    # Calculate memory usage
    train_memory_gb = (
        train_prompt_sequences.element_size() * train_prompt_sequences.numel() +
        train_cot_sequences.element_size() * train_cot_sequences.numel() +
        train_prompt_mask.element_size() * train_prompt_mask.numel() +
        train_cot_mask.element_size() * train_cot_mask.numel()
    ) / 1e9
    
    test_memory_gb = (
        test_prompt_sequences.element_size() * test_prompt_sequences.numel() +
        test_cot_sequences.element_size() * test_cot_sequences.numel() +
        test_prompt_mask.element_size() * test_prompt_mask.numel() +
        test_cot_mask.element_size() * test_cot_mask.numel()
    ) / 1e9
    
    total_memory_gb = train_memory_gb + test_memory_gb
    
    print(f"Train data memory usage: {train_memory_gb:.2f} GB")
    print(f"Test data memory usage: {test_memory_gb:.2f} GB")
    print(f"Total data memory usage: {total_memory_gb:.2f} GB")
    
    return train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask, \
            test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask
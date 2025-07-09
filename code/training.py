# Author: Akira Kudo
# Created: 2025/06/19
# Last Updated: 2025/06/23

"""
Training script for GPT2VQVAE models with comprehensive memory profiling.

This script includes PyTorch profiler integration to track memory usage in different parts of the training process.

To enable profiling, add the following to your training_config:
{
    "profiler_enabled": true,
    "profiler_schedule": {
        "wait": 0,      # Wait 0 steps before starting
        "warmup": 0,    # No warmup
        "active": 6,    # Profile for 6 steps
        "repeat": 1     # Repeat once
    }
}

The profiler will track:
- Forward pass operations (## forward_pass ##)
- Backward pass operations (## backward_pass ##)
- Optimizer steps (## optimizer_step ##)
- Validation forward passes (## validation_forward ##)
- Checkpoint saving (## save_checkpoint ##)
- Codebook plot saving (## save_codebook_plots ##)
- Training history plotting (## plot_training_history ##)
- Memory usage plotting (## plot_memory_usage ##)
- Training visualization saving (## save_training_visualizations ##)

Profiler outputs:
- Chrome trace file (.json.gz) for timeline analysis
- Memory timeline HTML file for memory usage visualization
- Files are saved with hostname and timestamp prefix
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import Tuple, Optional, Dict, Any, List
import os
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
# import wandb  # Optional: for experiment tracking
import argparse
import yaml
import gc
# import psutil
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
# from torch.cuda.amp import autocast, GradScaler
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
from matplotlib.colors import ListedColormap
import traceback
import socket
from datetime import datetime
from torch.autograd.profiler import record_function
import json
import sys
import os

# Add the current directory to the path to import phone_notification
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from phone_notification import send_notification

# GPU memory monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    print("Warning: nvidia-ml-py3 not available. Install with: pip install nvidia-ml-py3")
    NVML_AVAILABLE = False

# Import the GPT2VQVAE model
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from vqvae_gpt2 import GPT2VQVAE, compute_perplexity
from vqvae_gpt2_simple import SimpleGPT2VQVAE
from vqvae_gpt2_with_enhancement import EnhancedGPT2VQVAE

TRACK_MEMORY = False
TRACK_IN_EPOCH_MEMORY = False
SEND_NOTIFICATION = True

# Profiler configuration
TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

def trace_handler(prof: torch.profiler.profile):
    """Handler for profiler traces - saves Chrome trace and memory timeline."""
    # Prefix for file names
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{host_name}_{timestamp}"

    # Construct the trace file
    print("Exporting chrome trace...", end="")
    prof.export_chrome_trace(f"{file_prefix}.json.gz")
    print("DONE!")

    # Construct the memory timeline file
    print("Exporting memory timeline...", end="")
    prof.export_memory_timeline(f"{file_prefix}.html", device="cuda:0")
    print("DONE!")

class TrainingAbortedException(Exception):
    """
    Custom exception raised when training is aborted due to perplexity threshold or other conditions.
    """
    def __init__(self, reason: str, epoch: int, metrics: Dict[str, Any], final_perplexity: Optional[float] = None):
        self.reason = reason
        self.epoch = epoch
        self.metrics = metrics
        self.final_perplexity = final_perplexity
        super().__init__(f"Training aborted at epoch {epoch}: {reason}")

class GPUMemoryMonitor:
    """
    Monitor GPU memory usage using nvidia-ml-py3.
    """
    
    def __init__(self, device_id: int = 0):
        """
        Initialize GPU memory monitor.
        
        Args:
            device_id: GPU device ID to monitor
        """
        self.device_id = device_id
        self.nvml_available = NVML_AVAILABLE
        
        if self.nvml_available:
            try:
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                self.device_name = pynvml.nvmlDeviceGetName(self.handle)
                print(f"GPU Memory Monitor initialized for: {self.device_name}")
            except Exception as e:
                print(f"Warning: Could not initialize NVML for device {device_id}: {e}")
                self.nvml_available = False
            
    def get_memory_info(self) -> Dict[str, float]:
        """
        Get current GPU memory information.
        
        Returns:
            Dictionary with memory usage in GB
        """
        if not self.nvml_available:
            return {
                'total_gb': 0.0,
                'used_gb': 0.0,
                'free_gb': 0.0,
                'utilization_percent': 0.0
            }
        
        try:
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
            total_gb = mem_info.total / (1024**3)
            used_gb = mem_info.used / (1024**3)
            free_gb = mem_info.free / (1024**3)
            
            # Get utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self.handle)
            utilization_percent = util.memory
            
            return {
                'total_gb': total_gb,
                'used_gb': used_gb,
                'free_gb': free_gb,
                'utilization_percent': utilization_percent
            }
        except Exception as e:
            print(f"Warning: Could not get GPU memory info: {e}")
            return {
                'total_gb': 0.0,
                'used_gb': 0.0,
                'free_gb': 0.0,
                'utilization_percent': 0.0
            }
    
    def log_memory_usage(self, stage: str, print_info: bool = True) -> Dict[str, float]:
        """
        Log memory usage at a specific stage.
        
        Args:
            stage: Description of the current stage
            print_info: Whether to print the memory information
            
        Returns:
            Dictionary with memory usage information
        """
        mem_info = self.get_memory_info()
        
        if print_info:
            print(f"GPU Memory Usage ({stage}):")
            print(f"  Total: {mem_info['total_gb']:.2f} GB")
            print(f"  Used: {mem_info['used_gb']:.2f} GB ({mem_info['utilization_percent']:.1f}%)")
            print(f"  Free: {mem_info['free_gb']:.2f} GB")
        
        return mem_info
    
    def get_pytorch_memory_info(self) -> Dict[str, float]:
        """
        Get PyTorch-specific memory information.
        
        Returns:
            Dictionary with PyTorch memory usage in GB
        """
        if not torch.cuda.is_available():
            return {
                'allocated_gb': 0.0,
                'reserved_gb': 0.0,
                'max_allocated_gb': 0.0
            }
        
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        reserved_gb = torch.cuda.memory_reserved() / (1024**3)
        max_allocated_gb = torch.cuda.max_memory_allocated() / (1024**3)
        
        return {
            'allocated_gb': allocated_gb,
            'reserved_gb': reserved_gb,
            'max_allocated_gb': max_allocated_gb
        }
    
    def log_pytorch_memory_usage(self, stage: str, print_info: bool = True) -> Dict[str, float]:
        """
        Log PyTorch-specific memory usage.
        
        Args:
            stage: Description of the current stage
            print_info: Whether to print the memory information
            
        Returns:
            Dictionary with PyTorch memory usage information
        """
        pytorch_mem = self.get_pytorch_memory_info()
        
        if print_info:
            print(f"PyTorch Memory Usage ({stage}):")
            print(f"  Allocated: {pytorch_mem['allocated_gb']:.2f} GB")
            print(f"  Reserved: {pytorch_mem['reserved_gb']:.2f} GB")
            print(f"  Max Allocated: {pytorch_mem['max_allocated_gb']:.2f} GB")
        
        return pytorch_mem

def dynamic_batch_sampler(dataset, max_tokens_per_batch: int = 8192):
    """
    Dynamic batch sampler that groups samples by sequence length to minimize padding.
    
    Args:
        dataset: Dataset to sample from
        max_tokens_per_batch: Maximum tokens per batch
    
    Yields:
        Batch indices
    """
    # Group samples by sequence length
    length_groups = {}
    for idx in range(len(dataset)):
        _, cot_seq, _, _ = dataset[idx]
        seq_len = cot_seq.size(-1)
        if seq_len not in length_groups:
            length_groups[seq_len] = []
        length_groups[seq_len].append(idx)
    
    # Create batches for each length group
    for seq_len, indices in length_groups.items():
        max_samples_per_batch = max_tokens_per_batch // seq_len
        
        for i in range(0, len(indices), max_samples_per_batch):
            batch_indices = indices[i:i + max_samples_per_batch]
            yield batch_indices

class GPT2VQVAETrainer:
    """
    Trainer class for GPT2VQVAE model with configurable hyperparameters and memory optimizations.
    """
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 training_config: Dict[str, Any],
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 tracking_functions: Optional[Dict[str, Any]] = None, 
                 run_name : Optional[str] = "ANONYM_RUN"):
        """
        Initialize the trainer.
        
        Args:
            model_config: Dictionary containing model hyperparameters
            training_config: Dictionary containing training hyperparameters
            device: Device to train on
            tracking_functions: Optional dict containing custom tracking functions:
                - 'track_codebook_usage': Function to track codebook usage
                - 'save_codebook_plots': Function to save codebook plots
                - 'tracking_enabled': Boolean to enable/disable tracking
            run_name: Optional string for the name of the run, as texted via send_notification.
        """
        self.model_config = model_config
        self.training_config = training_config
        self.device = device
        self.run_name = run_name
        
        # Set up tracking functions (default to base class methods)
        if tracking_functions is None:
            tracking_functions = {}
        
        self.track_codebook_usage_func = tracking_functions.get('track_codebook_usage', self._default_track_codebook_usage)
        self.save_codebook_plots_func = tracking_functions.get('save_codebook_plots', self._default_save_codebook_plots)
        self.tracking_enabled = tracking_functions.get('tracking_enabled', True)
        
        # Initialize GPU memory monitor
        if device.startswith("cuda"):
            device_id = int(device.split(":")[-1]) if ":" in device else 0
            self.memory_monitor = GPUMemoryMonitor(device_id)
        else:
            self.memory_monitor = None
        
        # Memory optimization settings
        self.use_mixed_precision = training_config.get('use_mixed_precision', True)
        self.gradient_accumulation_steps = training_config.get('gradient_accumulation_steps', 1)
        self.use_gradient_checkpointing = training_config.get('use_gradient_checkpointing', True)
        
        self.ensure_numeric_types(self.model_config)
        self.ensure_numeric_types(self.training_config)

        # Memory monitoring
        self.memory_stats = []
        
        # Log memory before model initialization
        if self.memory_monitor:
            # TODO SEE IF IT WORKS CORRECTLY
            self.log_memory_usage("before_model_init")
            # self.memory_monitor.log_memory_usage("before_model_init")
            # self.memory_monitor.log_pytorch_memory_usage("before_model_init")
        
        # Initialize model
        self.model = GPT2VQVAE(**model_config).to(device)
        
        # Log memory after model weights loaded to GPU
        if self.memory_monitor:
            # TODO SEE IF IT WORKS CORRECTLY
            self.log_memory_usage("after_model_weights_loaded")
            # self.memory_monitor.log_memory_usage("after_model_weights_loaded")
            # self.memory_monitor.log_pytorch_memory_usage("after_model_weights_loaded")
        
        # Enable gradient checkpointing if specified
        if self.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
        
            # Log gradient checkpointing status
            if hasattr(self.model, 'get_gradient_checkpointing_status'):
                status = self.model.get_gradient_checkpointing_status()
                print(f"Gradient checkpointing status:")
                print(f"  Model enabled: {status['model_enabled']}")
                print(f"  Encoder enabled: {status['encoder_enabled']}")
                print(f"  Decoder enabled: {status['decoder_enabled']}")
            elif hasattr(self.model, 'is_gradient_checkpointing_enabled'):
                print(f"Model gradient checkpointing: {self.model.is_gradient_checkpointing_enabled()}")
            else:
                print("Model gradient checkpointing: Not supported by this model")
        
        # Log memory before optimizer initialization
        if self.memory_monitor:
            # TODO SEE IF IT WORKS CORRECTLY
            self.log_memory_usage("before_optimizer_init")
            # self.memory_monitor.log_memory_usage("before_optimizer_init")
            # self.memory_monitor.log_pytorch_memory_usage("before_optimizer_init")
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 0.01),
            betas=(training_config.get('beta1', 0.9), training_config.get('beta2', 0.999))
        )
        
        # Log memory after optimizer initialization
        if self.memory_monitor:
            # TODO SEE IF IT WORKS CORRECTLY
            self.log_memory_usage("after_optimizer_init")
            # self.memory_monitor.log_memory_usage("after_optimizer_init")
            # self.memory_monitor.log_pytorch_memory_usage("after_optimizer_init")
        
        # Initialize mixed precision training
        if self.use_mixed_precision:
            self.scaler = GradScaler('cuda')
            # self.scaler = GradScaler()
            print("Mixed precision training enabled")
        else:
            self.scaler = None
        
        # Initialize learning rate scheduler
        if training_config.get('use_lr_scheduler', True):
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config['num_epochs'],
                eta_min=training_config.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = None
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=training_config.get('pad_token_id', 50256))
        
        # Training history - epoch-level metrics
        self.train_losses = []
        self.val_losses = []
        self.vq_losses = []
        self.perplexities = []
        
        # Training history - detailed metrics within epochs
        self.detailed_train_losses = []  # List of lists: [epoch_1_metrics, epoch_2_metrics, ...]
        self.detailed_vq_losses = []
        self.detailed_perplexities = []
        self.detailed_batch_indices = []  # List of lists: [epoch_1_indices, epoch_2_indices, ...]
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
        # Initialize gradient checkpointing as disabled by default
        self._gradient_checkpointing_enabled = False
        
        # Codebook usage tracking (legacy - now handled by tracking functions)
        self.codebook_tracking_enabled = self.training_config.get('codebook_tracking_enabled', True)
        self.codebook_sample_size = self.training_config.get('codebook_sample_size', 100)
        self.codebook_history = []  # List of count tensors over time
        self.codebook_perplexities = []  # List of perplexities over time
        self.codebook_measurement_points = []  # List of measurement point indices
        
        if self.tracking_enabled:
            print(f"Codebook tracking enabled with custom functions")
        else:
            print("Codebook tracking disabled")
        
        # Profiler configuration
        self.profiler_enabled = training_config.get('profiler_enabled', False)
        self.profiler_schedule = training_config.get('profiler_schedule', {
            'wait': 0,      
            'warmup': 1,    
            'active': 2,    
            'repeat': 1     
        })
        self.profiler_activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        
        if self.profiler_enabled:
            print("PyTorch profiler enabled")
            print(f"Profiler schedule: {self.profiler_schedule}")
    
    def _default_track_codebook_usage(self, dataset: Any, measurement_point: int) -> None:
        """Default codebook tracking function for base trainer."""
        if not self.codebook_tracking_enabled:
            return
        
        try:
            # Sample and compute codebook usage
            counts, perplexity = sample_and_compute_codebook_usage(
                self.model, 
                dataset, 
                self.codebook_sample_size, 
                self.device,
                use_vq=self.training_config.get('use_vq', True)
            )
            
            # Store results - ensure counts are on CPU to prevent GPU memory accumulation
            self.codebook_history.append(counts.cpu().detach())
            self.codebook_perplexities.append(perplexity)
            self.codebook_measurement_points.append(measurement_point)
            
            # Print current statistics
            unique_codes = (counts > 0).sum().item()
            total_usage = counts.sum().item()
            print(f"\nCodebook tracking (point {measurement_point}): "
                  f"Unique codes: {unique_codes}/{self.model.vector_quantizer.num_embeddings} "
                  f"({unique_codes/self.model.vector_quantizer.num_embeddings*100:.1f}%), "
                  f"Perplexity: {perplexity:.2f}")
            
            # Explicitly delete GPU tensors to prevent memory accumulation
            del counts
            
        except Exception as e:
            print(f"Warning: Failed to track codebook usage: {e}")
        finally:
            # Ensure cleanup even on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    def _default_save_codebook_plots(self, save_dir: str, epoch: int) -> None:
        """Default codebook plotting function for base trainer."""
        if not self.codebook_tracking_enabled or not self.codebook_history:
            return
        
        try:
            os.makedirs(save_dir, exist_ok=True)
            
            # Save current heatmap
            current_counts = self.codebook_history[-1]
            heatmap_path = os.path.join(save_dir, f"codebook_usage_epoch_{epoch}.png")
            create_codebook_usage_heatmap(
                current_counts,
                num_embeddings=self.model.vector_quantizer.num_embeddings,
                title=f"Codebook Usage - Epoch {epoch}",
                save_path=heatmap_path
            )
            
            # Save timeline plot
            timeline_path = os.path.join(save_dir, f"codebook_usage_timeline_epoch_{epoch}.png")
            create_codebook_usage_timeline_plot(
                self.codebook_history,
                num_embeddings=self.model.vector_quantizer.num_embeddings,
                measurement_points=self.codebook_measurement_points,
                title=f"Codebook Usage Over Time - Up to Epoch {epoch}",
                save_path=timeline_path
            )
            
            print(f"Codebook tracking plots saved to {save_dir}")
            
            # Clear cache after plotting
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Failed to save codebook tracking plots: {e}")
    
    # Ensure all numeric values are properly typed
    def ensure_numeric_types(self, config_dict):
        """Ensure all numeric values in config are properly typed."""
        for key, value in config_dict.items():
            if isinstance(value, str):
                try:
                    if 'e' in value.lower() or '.' in value:
                        config_dict[key] = float(value)
                    else:
                        config_dict[key] = int(value)
                except ValueError:
                    pass  # Keep as string if conversion fails
            elif isinstance(value, bool):
                config_dict[key] = bool(value)
    
    def log_memory_usage(self, stage: str):
        """Log current memory usage using the GPU memory monitor."""
        if self.memory_monitor:
            # Log both GPU and PyTorch memory usage
            gpu_mem = self.memory_monitor.log_memory_usage(stage, print_info=False)
            pytorch_mem = self.memory_monitor.log_pytorch_memory_usage(stage, print_info=False)
            
            # Store combined memory stats
            self.memory_stats.append({
                'stage': stage,
                'gpu_total_gb': gpu_mem['total_gb'],
                'gpu_used_gb': gpu_mem['used_gb'],
                'gpu_free_gb': gpu_mem['free_gb'],
                'gpu_utilization_percent': gpu_mem['utilization_percent'],
                'pytorch_allocated_gb': pytorch_mem['allocated_gb'],
                'pytorch_reserved_gb': pytorch_mem['reserved_gb'],
                'pytorch_max_allocated_gb': pytorch_mem['max_allocated_gb']
            })
            
            # Print combined information
            print(f"Memory Usage ({stage}):")
            print(f"  GPU: {gpu_mem['used_gb']:.2f}GB used / {gpu_mem['total_gb']:.2f}GB total ({gpu_mem['utilization_percent']:.1f}%)")
            print(f"  PyTorch: {pytorch_mem['allocated_gb']:.2f}GB allocated, {pytorch_mem['reserved_gb']:.2f}GB reserved")
        else:
            # Fallback to original method if no memory monitor
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                reserved = torch.cuda.memory_reserved() / 1e9
                max_allocated = torch.cuda.max_memory_allocated() / 1e9
                
                self.memory_stats.append({
                    'stage': stage,
                    'allocated_gb': allocated,
                    'reserved_gb': reserved,
                    'max_allocated_gb': max_allocated
                })
                
                print(f"Memory usage ({stage}): {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {max_allocated:.2f}GB max")
    

    
    def create_data_loader(self, 
                          dataset: torch.utils.data.Dataset,
                          batch_size: int,
                          shuffle: bool = True) -> DataLoader:
        """
        Create a DataLoader for training/validation with memory optimizations.
        
        Args:
            dataset: PyTorch dataset (e.g., from random_split)
            batch_size: Batch size for training
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader for the dataset
        """
        # Use dynamic batching if specified
        if self.training_config.get('use_dynamic_batching', False):
            max_tokens = self.training_config.get('max_tokens_per_batch', 8192)
            # Create sampler with proper type handling
            if shuffle:
                sampler = torch.utils.data.RandomSampler(dataset)  # type: ignore
            else:
                sampler = torch.utils.data.SequentialSampler(dataset)  # type: ignore
            batch_sampler = torch.utils.data.BatchSampler(sampler, batch_size=batch_size, drop_last=False)
            return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=0)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
            
    def train_epoch(self, train_loader: DataLoader, num_measurements_per_epoch: int, current_epoch: int = 0) -> Dict[str, Any]:
        """
        Train for one epoch with memory optimizations.
        
        Args:
            train_loader: Training data loader
            num_measurements_per_epoch: Number of equally spaced measurements to log during the epoch
            current_epoch: Current epoch number (for exception handling)
            
        Returns:
            Dictionary containing training metrics with both detailed and average metrics
        """
        self.model.train()
        total_loss = 0.0
        total_vq_loss = 0.0
        total_perplexity = 0.0
        num_batches = 0
        accumulation_steps = 0
        
        # Calculate measurement intervals
        total_batches = len(train_loader)
        measurement_interval = max(1, total_batches // num_measurements_per_epoch)
        
        # Storage for detailed metrics
        detailed_losses = []
        detailed_vq_losses = []
        detailed_perplexities = []
        detailed_batch_indices = []

        # Perplexity threshold monitoring
        perplexity_threshold = self.training_config.get('perplexity_threshold', 1.5)
        perplexity_window_size = self.training_config.get('perplexity_window_size', 20)
        recent_perplexities = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (prompts, cots, prompt_masks, cot_masks) in enumerate(progress_bar):
            # TODO DEBUG PURPOSE
            if TRACK_IN_EPOCH_MEMORY:
                self.log_memory_usage(f"before batch {batch_idx}")
            
            # Profiler step if enabled
            # if hasattr(self, 'profiler_enabled') and self.profiler_enabled and self.profiler_context is not None:
            #     self.profiler_context.step()
            # Move to device
            prompts = prompts.to(self.device, non_blocking=True)
            cots = cots.to(self.device, non_blocking=True)
            prompt_masks = prompt_masks.to(self.device, non_blocking=True)
            cot_masks = cot_masks.to(self.device, non_blocking=True)

            # TODO DEBUG PURPOSE
            if TRACK_IN_EPOCH_MEMORY:
                self.log_memory_usage(f"after data loading, batch {batch_idx}")
            
            # Forward pass and loss calculation
            with record_function("## forward_pass ##"):
                total_loss_batch, vq_loss, perplexity, _ = self._forward_pass(
                    prompts, cots, prompt_masks, cot_masks
                )
            
            # TODO DEBUG PURPOSE
            if TRACK_IN_EPOCH_MEMORY:
                self.log_memory_usage(f"after forward pass, batch {batch_idx}")
            
            # Scale loss and backward pass
            with record_function("## backward_pass ##"):
                scaled_loss = total_loss_batch / self.gradient_accumulation_steps
                if self.use_mixed_precision and self.scaler is not None:
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()
            
            # TODO DEBUG PURPOSE
            if TRACK_IN_EPOCH_MEMORY:
                self.log_memory_usage(f"after backward pass, batch {batch_idx}")

            accumulation_steps += 1
            
            # Update weights every gradient_accumulation_steps
            if accumulation_steps % self.gradient_accumulation_steps == 0:
                with record_function("## optimizer_step ##"):
                    self._update_weights()
                # TODO DEBUG PURPOSE
                if TRACK_IN_EPOCH_MEMORY:
                    self.log_memory_usage(f"after weight update, batch {batch_idx}")

            
            # Extract scalar values and detach tensors to prevent memory accumulation
            total_loss_batch_item = total_loss_batch.item()
            vq_loss_item = vq_loss.item()
            perplexity_item = perplexity.item()
            
            # Update metrics
            total_loss += total_loss_batch_item
            total_vq_loss += vq_loss_item
            total_perplexity += perplexity_item
            num_batches += 1
            
            # Update perplexity monitoring
            recent_perplexities.append(perplexity_item)
            if len(recent_perplexities) > perplexity_window_size:
                recent_perplexities.pop(0)
            
            # Check perplexity threshold
            if len(recent_perplexities) >= perplexity_window_size:
                avg_perplexity = sum(recent_perplexities) / len(recent_perplexities)
                if avg_perplexity < perplexity_threshold:
                    print(f"\n🎯 Perplexity threshold reached! Average perplexity over last {perplexity_window_size} steps: {avg_perplexity:.4f} < {perplexity_threshold}")
                    print(f"Training aborted.")
                    
                    # Calculate final metrics
                    final_metrics = {
                        'detailed_losses': detailed_losses,
                        'detailed_vq_losses': detailed_vq_losses,
                        'detailed_perplexities': detailed_perplexities,
                        'detailed_batch_indices': detailed_batch_indices,
                        'avg_loss': total_loss / num_batches,
                        'avg_vq_loss': total_vq_loss / num_batches,
                        'avg_perplexity': total_perplexity / num_batches,
                        'aborted': True,
                        'abort_reason': f'perplexity_threshold_{perplexity_threshold}',
                        'final_avg_perplexity': avg_perplexity
                    }
                    
                    # Raise exception with all the necessary information
                    raise TrainingAbortedException(
                        reason=f'perplexity_threshold_{perplexity_threshold}',
                        epoch=current_epoch + 1,  # Current epoch number
                        metrics=final_metrics,
                        final_perplexity=avg_perplexity
                    )

            # Log detailed metrics at regular intervals
            if batch_idx % measurement_interval == 0:
                # TODO DEBUG PURPOSE
                if TRACK_IN_EPOCH_MEMORY:
                    self.log_memory_usage(f"before detailed info, batch {batch_idx}")

                detailed_losses.append(total_loss_batch_item)
                detailed_vq_losses.append(vq_loss_item)
                detailed_perplexities.append(perplexity_item)
                detailed_batch_indices.append(batch_idx)
                
                # Track codebook usage at measurement intervals
                if self.tracking_enabled:

                    # TODO DEBUG PURPOSE
                    if TRACK_IN_EPOCH_MEMORY:
                        self.log_memory_usage(f"before getting the dataset, batch {batch_idx}")

                    # Get the dataset from the data loader
                    dataset: Any = train_loader.dataset
                    if hasattr(dataset, 'dataset'):  # Handle SubsetRandomSampler case
                        dataset = dataset.dataset

                    # TODO DEBUG PURPOSE
                    if TRACK_IN_EPOCH_MEMORY:
                        self.log_memory_usage(f"after getting the dataset, batch {batch_idx}")

                    self.track_codebook_usage_func(dataset, batch_idx)

                    # TODO DEBUG PURPOSE
                    if TRACK_IN_EPOCH_MEMORY:
                        self.log_memory_usage(f"after running codebook usage function, batch {batch_idx}")
                    
                
                # TODO DEBUG PURPOSE
                if TRACK_IN_EPOCH_MEMORY:
                    self.log_memory_usage(f"after detailed info, batch {batch_idx}")
            
            # Update progress bar
            acc_step = (accumulation_steps % self.gradient_accumulation_steps) + 1
            current_avg_perplexity = sum(recent_perplexities) / len(recent_perplexities) if recent_perplexities else 0
            progress_bar.set_postfix({
                'loss': f"{total_loss_batch_item:.4f}",
                'vq_loss': f"{vq_loss_item:.4f}",
                'perplexity': f"{perplexity_item:.2f}",
                'avg_perplexity': f"{current_avg_perplexity:.2f}",
                'accum_steps': f"{acc_step}/{self.gradient_accumulation_steps}"
            })

            # TODO DEBUG PURPOSE
            if TRACK_IN_EPOCH_MEMORY:
                self.log_memory_usage(f"after getting the dataset, batch {batch_idx}")
            
            # Explicitly delete intermediate tensors to prevent memory accumulation
            del total_loss_batch, vq_loss, perplexity, scaled_loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # TODO DEBUG PURPOSE
        if TRACK_IN_EPOCH_MEMORY:
            self.log_memory_usage(f"before getting average metrics, batch {batch_idx}")

        # Calculate averages
        avg_metrics = self._get_average_metrics(total_loss, total_vq_loss, total_perplexity, num_batches)
        
        # TODO DEBUG PURPOSE
        if TRACK_IN_EPOCH_MEMORY:
            self.log_memory_usage(f"after getting average metrics, batch {batch_idx}")
        
        # Return both detailed and average metrics
        return {
            'detailed_losses': detailed_losses,
            'detailed_vq_losses': detailed_vq_losses,
            'detailed_perplexities': detailed_perplexities,
            'detailed_batch_indices': detailed_batch_indices,
            'avg_loss': avg_metrics['loss'],
            'avg_vq_loss': avg_metrics['vq_loss'],
            'avg_perplexity': avg_metrics['perplexity'],
            'aborted': False
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate the model with memory optimizations.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_vq_loss = 0.0
        total_perplexity = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_idx, (prompts, cots, prompt_masks, cot_masks) in enumerate(tqdm(val_loader, desc="Validation")):
                # Profiler step if enabled
                # if hasattr(self, 'profiler_enabled') and self.profiler_enabled and self.profiler_context is not None:
                #     self.profiler_context.step()
                # Move to device
                prompts = prompts.to(self.device, non_blocking=True)
                cots = cots.to(self.device, non_blocking=True)
                prompt_masks = prompt_masks.to(self.device, non_blocking=True)
                cot_masks = cot_masks.to(self.device, non_blocking=True)
                
                # Forward pass and loss calculation
                with record_function("## validation_forward ##"):
                    total_loss_batch, vq_loss, perplexity, _ = self._forward_pass(
                        prompts, cots, prompt_masks, cot_masks
                    )
                
                # Update metrics
                total_loss += total_loss_batch.item()
                total_vq_loss += vq_loss.item()
                total_perplexity += perplexity.item()
                num_batches += 1
        
        # Calculate averages
        return self._get_average_metrics(total_loss, total_vq_loss, total_perplexity, num_batches)
    
    def _forward_pass(self, prompts, cots, prompt_masks, cot_masks):
        """Helper function for forward pass and loss calculation"""
        if self.use_mixed_precision:
            with autocast('cuda'):
                _, output_logits, vq_loss, perplexity, indices = self.model(
                    prompt=prompts,
                    cot_sequences=cots,
                    cot_mask=cot_masks,
                    prompt_mask=prompt_masks,
                    inference=False,
                    quantize_cot_only=self.training_config.get('quantize_cot_only', True)
                )
                recon_loss = compute_reconstruction_loss(output_logits, cots, cot_masks)
                total_loss_batch = recon_loss + self.training_config.get('vq_loss_weight', 1.0) * vq_loss
        else:
            _, output_logits, vq_loss, perplexity, indices = self.model(
                prompt=prompts,
                cot_sequences=cots,
                cot_mask=cot_masks,
                prompt_mask=prompt_masks,
                inference=False,
                quantize_cot_only=self.training_config.get('quantize_cot_only', True)
            )
            recon_loss = compute_reconstruction_loss(output_logits, cots, cot_masks)
            total_loss_batch = recon_loss + self.training_config.get('vq_loss_weight', 1.0) * vq_loss
            
        return total_loss_batch, vq_loss, perplexity, indices
    
    def _update_weights(self):
        """Helper function for updating weights"""
        if self.use_mixed_precision and self.scaler is not None:
            if self.training_config.get('gradient_clip', 1.0) > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config['gradient_clip']
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            if self.training_config.get('gradient_clip', 1.0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.training_config['gradient_clip']
                )
            self.optimizer.step()
        
        self.optimizer.zero_grad()
    
    def _get_average_metrics(self, total_loss, total_vq_loss, total_perplexity, num_batches):
        """Helper function for calculating average metrics"""
        return {
            'loss': total_loss / num_batches,
            'vq_loss': total_vq_loss / num_batches,
            'perplexity': total_perplexity / num_batches
        }
    
    def send_training_start_phone_notification(self) -> bool:
        """
        Send a formatted phone notification when training has started.
        
        Args:
            auth_info_path: Path to the auth info YAML file (optional)
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        try:
            # Format the message
            message = f"Training Started for {self.run_name}!\n"
            return send_notification(message)

        except Exception as e:
            print(f"Error sending training completion phone notification: {e}")
            return False

    def send_training_completion_phone_notification(self, final_metrics: Dict[str, float], 
                                                  training_duration: Optional[float] = None,
                                                  aborted: bool = False) -> bool:
        """
        Send a formatted phone notification when training is complete.
        
        Args:
            final_metrics: Dictionary containing final training metrics
            training_duration: Duration of training in seconds (optional)
            aborted: Whether training was aborted (default: False)
            auth_info_path: Path to the auth info YAML file (optional)
            
        Returns:
            bool: True if notification was sent successfully, False otherwise
        """
        try:
            # Format the message
            if aborted:
                message = f"🛑 Training Aborted for {self.run_name}!\n\n"
            else:
                message = f"🎉 Training Complete for {self.run_name}!\n\n"
            
            # Add basic metrics
            if 'loss' in final_metrics:
                message += f"Final Loss: {final_metrics['loss']:.4f}\n"
            if 'vq_loss' in final_metrics:
                message += f"VQ Loss: {final_metrics['vq_loss']:.4f}\n"
            if 'perplexity' in final_metrics:
                message += f"Perplexity: {final_metrics['perplexity']:.2f}\n"
            
            # Add training duration if provided
            if training_duration is not None:
                hours = int(training_duration // 3600)
                minutes = int((training_duration % 3600) // 60)
                seconds = int(training_duration % 60)
                message += f"Duration: {hours:02d}:{minutes:02d}:{seconds:02d}\n"
            
            # Add additional information
            message += f"Model: {self.model_config.get('model_type', 'GPT2VQVAE')}\n"
            message += f"Device: {self.device}\n"
            message += f"Best Val Loss: {self.best_val_loss:.4f}\n"
            message += f"Epochs: {len(self.train_losses)}\n"
            
            # Send the phone notification
            return send_notification(message)
            
        except Exception as e:
            print(f"Error sending training completion phone notification: {e}")
            return False
    

    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False, checkpoint_path: Optional[str] = None, **kwargs):
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            metrics: Current metrics
            is_best: Whether this is the best model so far
            checkpoint_path: Path to save the checkpoint (optional)
            **kwargs: Additional data to append to the checkpoint
        """
        checkpoint_dir = self.training_config.get('checkpoint_dir', 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'vq_losses': self.vq_losses,
            'perplexities': self.perplexities,
            'detailed_train_losses': self.detailed_train_losses,
            'detailed_vq_losses': self.detailed_vq_losses,
            'detailed_perplexities': self.detailed_perplexities,
            'detailed_batch_indices': self.detailed_batch_indices
        }
        
        # Add any additional data passed as kwargs
        checkpoint.update(kwargs)
        
        # Save best model if this is the best so far
        if is_best:
            # Remove any existing best model checkpoints
            for file in os.listdir(checkpoint_dir):
                if 'best_model' in file and file.endswith('.pt'):
                    os.remove(os.path.join(checkpoint_dir, file))
            
            best_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch}.pt')
            torch.save(checkpoint, best_path)
            self.best_model_path = best_path
            print(f"New best model saved (epoch {epoch}) with validation loss: {metrics['loss']:.4f}")
        else:
            # Save regular checkpoint
            if checkpoint_path is None:
                checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved to: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Validate model and training configurations
        def _check_config_mismatch(checkpoint_config, current_config, config_type):
            """Helper function to check and report configuration mismatches."""
            if checkpoint_config != current_config:
                print(f"Warning: {config_type} configuration mismatch detected!")
                print(f"Differences between checkpoint and current {config_type} config:")
                for key in set(checkpoint_config.keys()) | set(current_config.keys()):
                    if key not in checkpoint_config:
                        print(f"  {key}: missing in checkpoint, current: {current_config[key]}")
                    elif key not in current_config:
                        print(f"  {key}: missing in current, checkpoint: {checkpoint_config[key]}")
                    elif checkpoint_config[key] != current_config[key]:
                        print(f"  {key}: checkpoint={checkpoint_config[key]}, current={current_config[key]}")
                print(f"Continuing with current {config_type} configuration...")

        if 'model_config' in checkpoint:
            _check_config_mismatch(checkpoint['model_config'], self.model_config, "model")
        
        if 'training_config' in checkpoint:
            _check_config_mismatch(checkpoint['training_config'], self.training_config, "training")
    
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if checkpoint['scheduler_state_dict'] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # Check and update scheduler parameters if needed
            loaded_T_max = self.scheduler.T_max
            loaded_eta_min = self.scheduler.eta_min
            config_T_max = self.training_config['num_epochs']
            config_eta_min = self.training_config.get('min_lr', 1e-6)
            mismatch = False
            if loaded_T_max != config_T_max:
                print(f"Warning: Scheduler T_max from checkpoint ({loaded_T_max}) does not match current config ({config_T_max}). Overriding to config value.")
                self.scheduler.T_max = config_T_max
                mismatch = True
            if loaded_eta_min != config_eta_min:
                print(f"Warning: Scheduler eta_min from checkpoint ({loaded_eta_min}) does not match current config ({config_eta_min}). Overriding to config value.")
                self.scheduler.eta_min = config_eta_min
                mismatch = True
            if mismatch:
                print("Scheduler parameters have been updated to match the current training configuration.")
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.vq_losses = checkpoint.get('vq_losses', [])
        self.perplexities = checkpoint.get('perplexities', [])
        
        self.detailed_train_losses = checkpoint.get('detailed_train_losses', [])
        self.detailed_vq_losses = checkpoint.get('detailed_vq_losses', [])
        self.detailed_perplexities = checkpoint.get('detailed_perplexities', [])
        self.detailed_batch_indices = checkpoint.get('detailed_batch_indices', [])
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print("Checkpoint loaded successfully. Configuration validation completed.")
        
        return checkpoint
    
    def train(self, 
              train_prompt_sequences: torch.Tensor,
              train_cot_sequences: torch.Tensor,
              train_prompt_mask: torch.Tensor,
              train_cot_mask: torch.Tensor,
              test_prompt_sequences: torch.Tensor,
              test_cot_sequences: torch.Tensor,
              test_prompt_mask: torch.Tensor,
              test_cot_mask: torch.Tensor,
              resume_from: Optional[str] = None,
              num_measurements_per_epoch: Optional[int] = None,
              seed: int = 42):
        """
        Train the model with memory optimizations using pre-split train and test data.
        
        Args:
            train_prompt_sequences: Training prompt sequences
            train_cot_sequences: Training CoT sequences
            train_prompt_mask: Training prompt masks
            train_cot_mask: Training CoT masks
            test_prompt_sequences: Test prompt sequences
            test_cot_sequences: Test CoT sequences
            test_prompt_mask: Test prompt masks
            test_cot_mask: Test CoT masks
            resume_from: Path to checkpoint to resume from
            num_measurements_per_epoch: Number of metrics saved per epoch
            seed: Random seed for reproducibility
        """

        # Start timing
        training_start_time = datetime.now()

        # FOR DEBUG
        if TRACK_MEMORY:
            torch.cuda.memory._record_memory_history(max_entries=10000)

        # Log initial memory usage
        self.log_memory_usage("training_start")
        
        # Create train and test datasets directly from provided tensors
        train_dataset = TensorDataset(train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask)
        test_dataset = TensorDataset(test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask)
        
        # Validate dataset sizes
        train_size = len(train_dataset)
        test_size = len(test_dataset)

        if train_size == 0:
            raise Exception(f"The training dataset has size 0.")
        elif test_size == 0:
            raise Exception(f"The test dataset has size 0.")
        
        if num_measurements_per_epoch is None:
            num_measurements_per_epoch = self.training_config.get("num_measurements_per_epoch", 25)
            print(f"Logging {num_measurements_per_epoch} measurements per epoch as per training config (or if not given, default)")
        else:
            print(f"Logging {num_measurements_per_epoch} measurements per epoch as given as parameter")
        
        # Ensure num_measurements_per_epoch is an integer for type checking
        assert num_measurements_per_epoch is not None
        num_measurements_per_epoch = int(num_measurements_per_epoch)
        
        print(f"Training samples: {train_size}")
        print(f"Test samples: {test_size}")
        
        # Create data loaders with memory optimizations
        train_loader = self.create_data_loader(
            train_dataset,
            batch_size=self.training_config['batch_size'], 
            shuffle=True
        )
        
        test_loader = self.create_data_loader(
            test_dataset,
            batch_size=self.training_config['batch_size'], 
            shuffle=False
        )
        
        # Resume from checkpoint if specified
        start_epoch = 0
        if resume_from:
            self.load_checkpoint(resume_from)
            start_epoch = len(self.train_losses)
        
        # Training loop with profiler
        try:
            # Send message that training has started
            if SEND_NOTIFICATION:
                self.send_training_start_phone_notification()

            # Set up profiler for this epoch if enabled
            self.profiler_context = None
            if self.profiler_enabled:
                print(f"Start profiler tracking.")
                self.profiler_context = torch.profiler.profile(
                    activities=self.profiler_activities,
                    schedule=torch.profiler.schedule(**self.profiler_schedule),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                    on_trace_ready=trace_handler,
                )
                self.profiler_context.__enter__()

            for epoch in range(start_epoch, self.training_config['num_epochs']):
                print(f"\nEpoch {epoch + 1}/{self.training_config['num_epochs']}")
                print("-" * 50)

                # Take a profiler step
                if hasattr(self, 'profiler_enabled') and self.profiler_enabled and self.profiler_context is not None:
                    self.profiler_context.step()
            
                
                # Log memory before epoch
                self.log_memory_usage(f"epoch_{epoch+1}_start")
                
                # Train with resume support
                train_metrics = self.train_epoch(
                    train_loader,
                    num_measurements_per_epoch,  # This is guaranteed to be int from earlier logic
                    epoch
                )
                
                # Log memory after training
                self.log_memory_usage(f"epoch_{epoch+1}_after_train")
                
                # Test
                test_metrics = self.validate(test_loader)
                
                # Log memory after testing
                self.log_memory_usage(f"epoch_{epoch+1}_after_test")
                
                # Update learning rate
                if self.scheduler:
                    self.scheduler.step()
                
                # Store epoch-level metrics
                self.train_losses.append(train_metrics['avg_loss'])
                self.val_losses.append(test_metrics['loss'])
                self.vq_losses.append(train_metrics['avg_vq_loss'])
                self.perplexities.append(train_metrics['avg_perplexity'])
                
                # Store detailed metrics
                self.detailed_train_losses.append(train_metrics['detailed_losses'])
                self.detailed_vq_losses.append(train_metrics['detailed_vq_losses'])
                self.detailed_perplexities.append(train_metrics['detailed_perplexities'])
                self.detailed_batch_indices.append(train_metrics['detailed_batch_indices'])
                
                # Print metrics
                print(f"Train Loss: {train_metrics['avg_loss']:.4f}")
                print(f"Test Loss: {test_metrics['loss']:.4f}")
                print(f"VQ Loss: {train_metrics['avg_vq_loss']:.4f}")
                print(f"Perplexity: {train_metrics['avg_perplexity']:.2f}")
                print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
                
                # Save checkpoint
                is_best = test_metrics['loss'] < self.best_val_loss
                if is_best:
                    self.best_val_loss = test_metrics['loss']
                    with record_function("## save_checkpoint ##"):
                        self.save_checkpoint(epoch + 1, test_metrics, True)
                
                if (epoch + 1) % self.training_config.get('save_every', 5) == 0:
                    with record_function("## save_checkpoint ##"):
                        self.save_checkpoint(epoch + 1, test_metrics, False)
                
                # Save codebook tracking plots
                if self.tracking_enabled:
                    with record_function("## save_codebook_plots ##"):
                        checkpoint_dir = self.training_config.get('checkpoint_dir', 'checkpoints')
                        codebook_dir = os.path.join(checkpoint_dir, 'codebook_tracking')
                        self.save_codebook_plots_func(codebook_dir, epoch + 1)
                
                # Clear cache after each epoch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

                # DEBUG
                if TRACK_MEMORY:
                    # after the third epoch and repeated
                    if (epoch + 1) % 3 == 0:
                        try:
                            torch.cuda.memory._dump_snapshot(f"training_cuda_memory_tracking_epoch{epoch+1}.pickle")
                        except Exception as e:
                            print("Failed to log memory tracking.")
            
            # Clean up profiler if active
            if self.profiler_context is not None:
                self.profiler_context.__exit__(None, None, None)
                self.profiler_context = None
                print(f"Profiler completed for epoch {epoch}")

            if TRACK_MEMORY:
                # Stop recording memory snapshot history.
                torch.cuda.memory._record_memory_history(enabled=None)

        
        except TrainingAbortedException as e:
            # Handle aborted training
            print(f"\n🛑 Training aborted: {e.reason}")
            if e.final_perplexity is not None:
                print(f"Final average perplexity: {e.final_perplexity:.4f}")
            
            # Store the current epoch's metrics even though training was aborted
            self.train_losses.append(e.metrics['avg_loss'])
            self.vq_losses.append(e.metrics['avg_vq_loss'])
            self.perplexities.append(e.metrics['avg_perplexity'])
            
            # Add a dummy validation loss for plotting purposes (use training loss as proxy)
            self.val_losses.append(e.metrics['avg_loss'])
            
            # Store detailed metrics from the aborted epoch
            self.detailed_train_losses.append(e.metrics['detailed_losses'])
            self.detailed_vq_losses.append(e.metrics['detailed_vq_losses'])
            self.detailed_perplexities.append(e.metrics['detailed_perplexities'])
            self.detailed_batch_indices.append(e.metrics['detailed_batch_indices'])
            
            # Create a dummy validation metrics for checkpoint saving
            # Use the training metrics as a proxy since we didn't complete validation
            dummy_val_metrics = {
                'loss': e.metrics['avg_loss'],  # Use training loss as proxy
                'vq_loss': e.metrics['avg_vq_loss'],
                'perplexity': e.metrics['avg_perplexity']
            }
            
            # Save checkpoint for the aborted training
            checkpoint_dir = self.training_config.get('checkpoint_dir', 'checkpoints')
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Get minimum batches threshold for saving checkpoint
            minimum_batches = self.training_config.get('minimum_batches_for_checkpoint', 200)
            
            # Calculate total batches trained (detailed_losses contains one entry per measurement interval)
            total_batches_trained = len(e.metrics['detailed_losses'])
            # Estimate total batches based on measurements and measurement frequency
            total_batches_trained = total_batches_trained * num_measurements_per_epoch
            
            # Save as a special "aborted" checkpoint if we've trained enough batches
            if total_batches_trained >= minimum_batches:
                aborted_checkpoint_path = os.path.join(checkpoint_dir, f'aborted_training_epoch_{e.epoch}.pt')
                with record_function("## save_checkpoint ##"):
                    self.save_checkpoint(e.epoch, dummy_val_metrics, is_best=False, checkpoint_path=aborted_checkpoint_path)
                print(f"Aborted training checkpoint saved (trained {total_batches_trained} batches, threshold: {minimum_batches})")

                # Also save as best model if it's better than previous best
                if e.metrics['avg_loss'] < self.best_val_loss:
                    self.best_val_loss = e.metrics['avg_loss']
                    best_aborted_path = os.path.join(checkpoint_dir, f'best_model_aborted_epoch_{e.epoch}.pt')
                    with record_function("## save_checkpoint ##"):
                        self.save_checkpoint(e.epoch, dummy_val_metrics, is_best=True, checkpoint_path=best_aborted_path)
                    print(f"New best model (from aborted training) saved to: {best_aborted_path}")
            else:
                print(f"Skipping checkpoint save - only trained {total_batches_trained} batches, need at least {minimum_batches}")
            
            
            
            # Note: All visualizations will be saved by save_training_visualizations below
            
            # Log final memory usage for aborted training
            self.log_memory_usage("training_aborted_end")
            
            print(f"\nTraining aborted! Best validation loss so far: {self.best_val_loss:.4f}")
            print(f"Training completed at epoch {e.epoch} due to perplexity threshold.")
            
            # Clear cache after aborted training
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Calculate training duration for aborted training
            training_end_time = datetime.now()
            training_duration = (training_end_time - training_start_time).total_seconds()
            
            # Get final metrics from the aborted epoch
            final_metrics = {
                'loss': e.metrics['avg_loss'],
                'vq_loss': e.metrics['avg_vq_loss'],
                'perplexity': e.metrics['avg_perplexity']
            }
            
            # Send the phone notification with aborted status
            if SEND_NOTIFICATION:
                _ = self.send_training_completion_phone_notification(
                    final_metrics=final_metrics,
                    training_duration=training_duration,
                    aborted=True
                )
            
            # Save comprehensive training visualizations for aborted training
            save_training_visualizations(self, prefix="aborted_training")
            
            # Re-raise the exception to be caught by the main function
            raise
        
        # Log final memory usage
        self.log_memory_usage("training_end")
        
        # Calculate training duration
        training_end_time = datetime.now()
        training_duration = (training_end_time - training_start_time).total_seconds()
        
        print(f"\nTraining completed! Best validation loss: {self.best_val_loss:.4f}")
        print(f"Training duration: {training_duration:.2f} seconds ({training_duration/3600:.2f} hours)")
        
        
        # Get final metrics from the last validation
        final_metrics = {
            'loss': self.val_losses[-1] if self.val_losses else 0.0,
            'vq_loss': self.vq_losses[-1] if self.vq_losses else 0.0,
            'perplexity': self.perplexities[-1] if self.perplexities else 0.0
        }
        
        # Send the phone notification
        if SEND_NOTIFICATION:
            _ = self.send_training_completion_phone_notification(
                final_metrics=final_metrics,
                training_duration=training_duration
            )
        
        # Save comprehensive training visualizations
        save_training_visualizations(self, prefix="training")
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """
        Plot training history including detailed metrics within epochs.
        
        Args:
            save_path: Path to save the plot
        """
        # Create a larger figure to accommodate detailed plots
        fig, axes = plt.subplots(3, 2, figsize=(20, 15))
        
        # Epoch-level metrics (top row)
        # Loss plot
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_title('Training and Validation Loss (Epoch Level)')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # VQ Loss plot
        axes[0, 1].plot(self.vq_losses, label='VQ Loss', color='red')
        axes[0, 1].set_title('Vector Quantization Loss (Epoch Level)')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('VQ Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot epoch-level perplexity
        axes[1, 0].plot(self.perplexities, label='Epoch Perplexity', color='orange')
        axes[1, 0].set_title('Codebook Perplexity (Epoch Level)')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Perplexity')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Detailed metrics within epochs
        if self.detailed_train_losses:
            # Flatten all detailed metrics for plotting
            all_detailed_losses = []
            all_detailed_vq_losses = []
            all_detailed_perplexities = []
            all_detailed_indices = []
            epoch_boundaries = []
            
            # Calculate global batch indices
            global_batch_idx = 0
            for epoch_idx, (epoch_losses, epoch_vq_losses, epoch_perplexities, epoch_indices) in enumerate(
                zip(self.detailed_train_losses, self.detailed_vq_losses, 
                    self.detailed_perplexities, self.detailed_batch_indices)
            ):
                # Mark the start of each epoch
                epoch_boundaries.append((global_batch_idx, epoch_idx + 1))
                
                for batch_idx, (loss, vq_loss, perplexity) in enumerate(
                    zip(epoch_losses, epoch_vq_losses, epoch_perplexities)
                ):
                    all_detailed_losses.append(loss)
                    all_detailed_vq_losses.append(vq_loss)
                    all_detailed_perplexities.append(perplexity)
                    all_detailed_indices.append(global_batch_idx + batch_idx)
                global_batch_idx += len(epoch_losses)
            
            # Plot detailed training loss
            axes[1, 1].plot(all_detailed_indices, all_detailed_losses, label='Detailed Train Loss', alpha=0.7)
            # Add epoch boundary lines and annotations
            for boundary, epoch_num in epoch_boundaries:
                axes[1, 1].axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
                axes[1, 1].text(boundary, axes[1, 1].get_ylim()[1], f'Epoch {epoch_num}', 
                              rotation=90, va='top', ha='right')
            axes[1, 1].set_title('Detailed Training Loss (Within Epochs)')
            axes[1, 1].set_xlabel('Measurement Index')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
            
            # Plot detailed VQ loss
            axes[2, 0].plot(all_detailed_indices, all_detailed_vq_losses, label='Detailed VQ Loss', color='red', alpha=0.7)
            # Add epoch boundary lines and annotations
            for boundary, epoch_num in epoch_boundaries:
                axes[2, 0].axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
                axes[2, 0].text(boundary, axes[2, 0].get_ylim()[1], f'Epoch {epoch_num}',
                              rotation=90, va='top', ha='right')
            axes[2, 0].set_title('Detailed VQ Loss (Within Epochs)')
            axes[2, 0].set_xlabel('Measurement Index')
            axes[2, 0].set_ylabel('VQ Loss')
            axes[2, 0].legend()
            axes[2, 0].grid(True)
            
            # Plot detailed perplexity
            axes[2, 1].plot(all_detailed_indices, all_detailed_perplexities, label='Detailed Perplexity', color='green', alpha=0.7)
            # Add epoch boundary lines and annotations
            for boundary, epoch_num in epoch_boundaries:
                axes[2, 1].axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
                axes[2, 1].text(boundary, axes[2, 1].get_ylim()[1], f'Epoch {epoch_num}',
                              rotation=90, va='top', ha='right')
            axes[2, 1].set_title('Detailed Codebook Perplexity (Within Epochs)')
            axes[2, 1].set_xlabel('Measurement Index')
            axes[2, 1].set_ylabel('Perplexity')
            axes[2, 1].legend()
            axes[2, 1].grid(True)
            
        else:
            # Fallback to original plots if no detailed data
            axes[1, 1].text(0.5, 0.5, 'No detailed metrics available', ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Detailed Training Loss')
            
            axes[2, 0].text(0.5, 0.5, 'No detailed metrics available', ha='center', va='center', transform=axes[2, 0].transAxes)
            axes[2, 0].set_title('Detailed VQ Loss')
            
            axes[2, 1].text(0.5, 0.5, 'No detailed metrics available', ha='center', va='center', transform=axes[2, 1].transAxes)
            axes[2, 1].set_title('Detailed Perplexity')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history saved to {save_path}")
        
        plt.close()
    
    def plot_memory_usage(self, save_path: Optional[str] = None):
        """Plot memory usage throughout training."""
        with record_function("## plot_memory_usage ##"):
            if not self.memory_stats:
                print("No memory statistics available")
                return
        
        # Check if we have GPU memory stats (indicates memory_monitor was used)
        has_gpu_stats = 'gpu_total_gb' in self.memory_stats[0]
        
        if has_gpu_stats:
            # Case 1: Both GPU and PyTorch memory tracking
            fig, axes = plt.subplots(3, 1, figsize=(12, 12))
            
            # Extract data
            stages = [stat['stage'] for stat in self.memory_stats]
            gpu_used = [stat['gpu_used_gb'] for stat in self.memory_stats]
            gpu_total = [stat['gpu_total_gb'] for stat in self.memory_stats]
            gpu_utilization = [stat['gpu_utilization_percent'] for stat in self.memory_stats]
            pytorch_allocated = [stat['pytorch_allocated_gb'] for stat in self.memory_stats]
            pytorch_reserved = [stat['pytorch_reserved_gb'] for stat in self.memory_stats]
            pytorch_max_allocated = [stat['pytorch_max_allocated_gb'] for stat in self.memory_stats]
            
            # Plot GPU memory usage
            axes[0].plot(range(len(stages)), gpu_used, label='GPU Used', marker='o', color='blue')
            axes[0].plot(range(len(stages)), gpu_total, label='GPU Total', marker='s', color='red', linestyle='--')
            axes[0].set_title('GPU Memory Usage Throughout Training')
            axes[0].set_xlabel('Training Stage')
            axes[0].set_ylabel('Memory (GB)')
            axes[0].legend()
            axes[0].grid(True)
            axes[0].set_xticks(range(len(stages)))
            axes[0].set_xticklabels(stages, rotation=45, ha='right')
            
            # Plot GPU utilization
            axes[1].plot(range(len(stages)), gpu_utilization, label='GPU Utilization', color='green', marker='^')
            axes[1].set_title('GPU Memory Utilization')
            axes[1].set_xlabel('Training Stage')
            axes[1].set_ylabel('Utilization (%)')
            axes[1].legend()
            axes[1].grid(True)
            axes[1].set_xticks(range(len(stages)))
            axes[1].set_xticklabels(stages, rotation=45, ha='right')
            
            # Plot PyTorch memory usage
            axes[2].plot(range(len(stages)), pytorch_allocated, label='PyTorch Allocated', marker='o')
            axes[2].plot(range(len(stages)), pytorch_reserved, label='PyTorch Reserved', marker='s')
            axes[2].plot(range(len(stages)), pytorch_max_allocated, label='PyTorch Max Allocated', color='red', marker='^')
            axes[2].set_title('PyTorch Memory Usage')
            axes[2].set_xlabel('Training Stage')
            axes[2].set_ylabel('Memory (GB)')
            axes[2].legend()
            axes[2].grid(True)
            axes[2].set_xticks(range(len(stages)))
            axes[2].set_xticklabels(stages, rotation=45, ha='right')
            
        else:
            # Case 2: Only PyTorch memory tracking (fallback case)
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Extract data
            stages = [stat['stage'] for stat in self.memory_stats]
            allocated = [stat['allocated_gb'] for stat in self.memory_stats]
            reserved = [stat['reserved_gb'] for stat in self.memory_stats]
            max_allocated = [stat['max_allocated_gb'] for stat in self.memory_stats]
            
            # Plot allocated vs reserved memory
            axes[0].plot(range(len(stages)), allocated, label='Allocated', marker='o')
            axes[0].plot(range(len(stages)), reserved, label='Reserved', marker='s')
            axes[0].set_title('PyTorch Memory Usage Throughout Training')
            axes[0].set_xlabel('Training Stage')
            axes[0].set_ylabel('Memory (GB)')
            axes[0].legend()
            axes[0].grid(True)
            axes[0].set_xticks(range(len(stages)))
            axes[0].set_xticklabels(stages, rotation=45, ha='right')
            
            # Plot max allocated memory
            axes[1].plot(range(len(stages)), max_allocated, label='Max Allocated', color='red', marker='^')
            axes[1].set_title('Maximum PyTorch Memory Usage')
            axes[1].set_xlabel('Training Stage')
            axes[1].set_ylabel('Memory (GB)')
            axes[1].legend()
            axes[1].grid(True)
            axes[1].set_xticks(range(len(stages)))
            axes[1].set_xticklabels(stages, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Memory usage plot saved to {save_path}")
        
        plt.close()

    def load_training_data(self, data_dir: str, max_samples: Optional[int] = None, num_thoughts: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Load training and test data with memory-efficient loading and truncate based on num_thoughts.
        
        Args:
            data_dir: Directory containing the preprocessed data files (with train/ and test/ subdirectories)
            max_samples: Maximum number of samples to load (for debugging/memory constraints)
            num_thoughts: Number of parallel sequences to use (truncates dataset if needed)
            
        Returns:
            Tuple of (train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask,
                     test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask)
        """
        # Log memory before loading data
        if self.memory_monitor:
            # TODO SEE IF IT WORKS CORRECTLY
            self.log_memory_usage("before_data_loading")
            # self.memory_monitor.log_memory_usage("before_data_loading")
            # self.memory_monitor.log_pytorch_memory_usage("before_data_loading")
        
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
                train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask = self._reorganize_sequences(
                    train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask, num_thoughts, "train"
                )
            
            # Process test data if needed
            if test_current_num_thoughts > num_thoughts:
                test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask = self._reorganize_sequences(
                    test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask, num_thoughts, "test"
                )
        
        # Limit samples if specified
        if max_samples is not None:
            train_prompt_sequences = train_prompt_sequences[:max_samples]
            train_cot_sequences = train_cot_sequences[:max_samples]
            train_prompt_mask = train_prompt_mask[:max_samples]
            train_cot_mask = train_cot_mask[:max_samples]
            
            test_prompt_sequences = test_prompt_sequences[:max_samples]
            test_cot_sequences = test_cot_sequences[:max_samples]
            test_prompt_mask = test_prompt_mask[:max_samples]
            test_cot_mask = test_cot_mask[:max_samples]
        
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

    def _reorganize_sequences(self, prompt_sequences, cot_sequences, prompt_mask, cot_mask, num_thoughts, split_name):
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



def create_default_config(output_path: str, enhanced_vq: bool = False, phased_training: bool = False, memory_optimized_tracking: bool = False):
    """
    Create a default configuration file with memory optimizations.
    
    Args:
        output_path: Path where to save the default config
        enhanced_vq: Whether to include enhanced VQ-VAE configuration options
        phased_training: Whether to include phased training configuration options (requires enhanced_vq=True)
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
                              pad_token_id: int = 50256) -> torch.Tensor:
    """
    Compute reconstruction loss between predicted logits and target sequences.
    
    Args:
        output_logits: Predicted logits [batch_size, M, L, vocab_size]
        target_sequences: Target sequences [batch_size, M, L]
        target_mask: Target mask [batch_size, M, L]
        pad_token_id: Token ID for padding (to ignore in loss computation)
        
    Returns:
        torch.Tensor: Reconstruction loss only
    """
    # Flatten all dimensions except vocab_size
    logits_flat = output_logits.reshape(-1, output_logits.size(-1))
    targets_flat = target_sequences.view(-1)
    mask_flat = target_mask.view(-1).bool()
    
    # Create loss function
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    # Compute reconstruction loss
    recon_loss = torch.tensor(0.0, device=output_logits.device)
    if mask_flat.sum() > 0:
        recon_loss = criterion(logits_flat[mask_flat], targets_flat[mask_flat])
    
    return recon_loss

def generate_gpt2_baseline_texts(prompt: torch.Tensor, 
                                prompt_mask: torch.Tensor,
                                cot_gt: torch.Tensor,
                                num_thoughts: int,
                                tokenizer,
                                gpt2_model) -> List[str]:
    """
    Generate GPT2 baseline texts for comparison with the model outputs.
    
    Args:
        prompt: Prompt tensor [1, prompt_length]
        prompt_mask: Prompt mask tensor [1, prompt_length]
        cot_gt: Ground truth CoT tensor [1, num_thoughts, cot_length]
        num_thoughts: Number of CoT sequences
        tokenizer: GPT2 tokenizer
        gpt2_model: Pre-trained GPT2 model
        
    Returns:
        List of GPT2 baseline texts, one for each CoT sequence
    """
    gpt2_baseline_texts = []
    if gpt2_model is not None and tokenizer is not None:
        try:
            # Remove prompt padding and add "Let's think step by step."
            prompt_tokens = prompt[0]
            if prompt_mask is not None:
                prompt_tokens = prompt_tokens[prompt_mask[0].bool()]
            
            # Add "Let's think step by step." after the prompt
            step_by_step_text = "Let's think step by step."
            step_by_step_tokens = tokenizer.encode(step_by_step_text, add_special_tokens=False)
            
            # Combine prompt tokens with step-by-step tokens
            combined_tokens = torch.cat([
                prompt_tokens,
                torch.tensor(step_by_step_tokens, dtype=prompt_tokens.dtype, device=prompt_tokens.device)
            ])
            
            input_ids = combined_tokens.unsqueeze(0)
            
            # Generate continuation (length: same as a single CoT sequence)
            max_gen_len = cot_gt.shape[-1]
            gpt2_outputs = gpt2_model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + max_gen_len,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
            # Extract only the generated part (excluding the prompt + step-by-step text)
            generated_tokens = gpt2_outputs[0][input_ids.shape[1]:]
            # Repeat the same baseline for each CoT sequence for fair comparison
            for _ in range(num_thoughts):
                gpt2_baseline_texts.append(tokenizer.decode(generated_tokens, skip_special_tokens=True))
            
        except Exception as e:
            print(f"GPT2 baseline generation failed: {e}")
            gpt2_baseline_texts = ["FAILED"] * num_thoughts
    else:
        gpt2_baseline_texts = ["FAILED"] * num_thoughts
    
    return gpt2_baseline_texts

def demonstrate_model_from_checkpoint(checkpoint_path: str, 
                                    data_dir: str,
                                    num_examples: int = 3,
                                    device: str = "cuda" if torch.cuda.is_available() else "cpu",
                                    use_vq: bool = True,
                                    model_type: str = "GPT2VQVAE",
                                    seed: int = 42,
                                    **kwargs):
    """
    Demonstrate GPT2VQVAE, EnhancedGPT2VQVAE, or SimpleGPT2VQVAE model generation capabilities from a checkpoint.
    
    This function automatically loads the model configuration from the checkpoint file,
    eliminating the need to manually specify model_config.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        data_dir: Directory containing training data
        num_examples: Number of examples to generate
        device: Device to run the model on
        use_vq: Whether to use vector quantization (for SimpleGPT2VQVAE)
        model_type: Type of model to use ("GPT2VQVAE", "EnhancedGPT2VQVAE", or "SimpleGPT2VQVAE")
        seed: Random seed for sampling examples (default: 42)
        **kwargs: Additional arguments to override checkpoint configuration
    """
    print(f"Loading {model_type} model from checkpoint: {checkpoint_path}")
    
    # Load model using the new from_checkpoint method
    try:
        if model_type == "EnhancedGPT2VQVAE":
            model = EnhancedGPT2VQVAE.from_checkpoint(checkpoint_path, device=device, **kwargs)
            print(f"Successfully loaded EnhancedGPT2VQVAE model")
        elif model_type == "SimpleGPT2VQVAE":
            model = SimpleGPT2VQVAE.from_checkpoint(checkpoint_path, device=device, **kwargs)
            print(f"Successfully loaded SimpleGPT2VQVAE model with use_vq={use_vq}")
        else:
            model = GPT2VQVAE.from_checkpoint(checkpoint_path, device=device, **kwargs)
            print(f"Successfully loaded GPT2VQVAE model")
    except Exception as e:
        print(f"Error loading model from checkpoint: {e}")
        print("Make sure the checkpoint file exists and contains the required model configuration.")
        return
    
    model.eval()
    
    # Get num_embeddings and num_thoughts from model configuration
    num_embeddings = model_config.get('num_embeddings', 512)
    num_thoughts = model_config.get('num_thoughts', 32)
    print(f"Model configured for {num_thoughts} parallel CoT sequences and {num_embeddings} embeddings")
    
    # Initialize accumulators for indices counts
    all_indices_tf = []
    all_indices_ar = []
    
    # Load tokenizer and GPT2 model for baseline comparison
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        print("Loaded GPT2 tokenizer")
        
        # Load pre-trained GPT2 model for baseline comparison
        gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        gpt2_model.eval()
        print("Loaded pre-trained GPT2 model for baseline comparison")
    except Exception as e:
        print(f"Warning: Could not load GPT2 model/tokenizer: {e}")
        tokenizer = None
        gpt2_model = None
    
    # Load some example data
    print(f"Loading example data from: {data_dir}")
    try:
        # Load all data first
        all_prompt_sequences = torch.load(os.path.join(data_dir, "prompt_sequences.pt"))
        all_cot_sequences = torch.load(os.path.join(data_dir, "cot_sequences_tensor.pt"))
        all_prompt_mask = torch.load(os.path.join(data_dir, "prompt_mask.pt"))
        all_cot_mask = torch.load(os.path.join(data_dir, "cot_mask.pt"))
        
        total_examples = len(all_prompt_sequences)
        print(f"Loaded {total_examples} total examples")
        
        # Randomly sample num_examples
        if num_examples > total_examples:
            print(f"Warning: Requested {num_examples} examples but only {total_examples} available. Using all examples.")
            num_examples = total_examples
        
        # Generate random indices for sampling
        torch.manual_seed(seed)  # For reproducible sampling
        sample_indices = torch.randperm(total_examples)[:num_examples]
        # Sample the data using the random indices
        prompt_sequences = all_prompt_sequences[sample_indices]
        cot_sequences = all_cot_sequences[sample_indices]
        prompt_mask = all_prompt_mask[sample_indices]
        cot_mask = all_cot_mask[sample_indices]
        
        print(f"Randomly sampled {len(prompt_sequences)} examples using seed {seed}")
        
        # Check if data has enough CoT sequences
        data_num_thoughts = cot_sequences.shape[1]
        if data_num_thoughts < num_thoughts:
            print(f"Warning: Data only has {data_num_thoughts} CoT sequences, but model expects {num_thoughts}")
            print(f"Will use {data_num_thoughts} sequences for demonstration")
            num_thoughts = data_num_thoughts
        elif data_num_thoughts > num_thoughts:
            print(f"Data has {data_num_thoughts} CoT sequences, truncating to {num_thoughts} for model compatibility")
            cot_sequences = cot_sequences[:, :num_thoughts, :]
            cot_mask = cot_mask[:, :num_thoughts, :]
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Function to decode tokens to text
    def decode_tokens(tokens, mask=None):
        if tokenizer is None:
            return f"[Tokens: {tokens.tolist()}]"
        
        if mask is not None:
            # Apply mask to remove padding
            tokens = tokens[mask.bool()]
        
        try:
            return tokenizer.decode(tokens, skip_special_tokens=True)
        except Exception as e:
            return f"[Decode error: {e}, tokens: {tokens.tolist()}]"
    
    print("\n" + "="*80)
    print(f"{model_type} GENERATION DEMONSTRATION")
    print("="*80)
    
    with torch.no_grad():
        for i in range(min(num_examples, len(prompt_sequences))):
            print(f"\n--- Example {i+1} ---")
            
            # Prepare single example
            prompt = prompt_sequences[i:i+1].to(device)  # [1, K]
            cot_gt = cot_sequences[i:i+1].to(device)     # [1, M, L]
            prompt_mask_ex = prompt_mask[i:i+1].to(device) if prompt_mask is not None else None
            cot_mask_ex = cot_mask[i:i+1].to(device) if cot_mask is not None else None
            
            # Decode ground truth
            prompt_text = decode_tokens(prompt[0], prompt_mask_ex[0] if prompt_mask_ex is not None else None)
            print(f"Prompt: {prompt_text}")
            
            # Generate with teacher forcing (inference=False)
            try:
                # Prepare base model inputs
                model_inputs = {
                    'prompt': prompt,
                    'cot_sequences': cot_gt, 
                    'cot_mask': cot_mask_ex,
                    'prompt_mask': prompt_mask_ex,
                    'inference': False,  # Teacher forcing
                    'quantize_cot_only': True
                }

                # Add use_vq parameter for SimpleGPT2VQVAE
                if model_type == "SimpleGPT2VQVAE":
                    model_inputs['use_vq'] = use_vq

                # Run model with prepared inputs
                _, output_logits_tf, vq_loss_tf, perplexity_tf, indices_tf = model(**model_inputs)
                
                # Get predicted tokens from logits
                predicted_tokens_tf = torch.argmax(output_logits_tf, dim=-1)  # [B, M, L]
                
                # Accumulate indices for teacher forcing
                if indices_tf is not None:
                    all_indices_tf.append(indices_tf.flatten())
                
            except Exception as e:
                print(f"Teacher forcing generation failed: {e}")
                predicted_tokens_tf = None
                vq_loss_tf = None
                perplexity_tf = None
                indices_tf = None
                raise # TODO REMOVE
            
            # Generate auto-regressively (inference=True)
            try:
                # Prepare model inputs
                model_inputs = {
                    'prompt': prompt,
                    'cot_sequences': cot_gt,
                    'cot_mask': cot_mask_ex,
                    'prompt_mask': prompt_mask_ex,
                    'inference': True,  # Auto-regressive
                    'quantize_cot_only': True
                }
                
                # Add use_vq parameter for SimpleGPT2VQVAE
                if model_type == "SimpleGPT2VQVAE":
                    model_inputs['use_vq'] = use_vq
                    
                # Run model with prepared inputs
                output_sequences_ar, output_logits_ar, vq_loss_ar, perplexity_ar, indices_ar = model(**model_inputs)
                
                # Accumulate indices for auto-regressive
                if indices_ar is not None:
                    all_indices_ar.append(indices_ar.flatten())
                
            except Exception as e:
                print(f"Auto-regressive generation failed: {e}")
                output_sequences_ar = None
                vq_loss_ar = None
                perplexity_ar = None
                indices_ar = None

            # Generate baseline using pre-trained GPT2
            gpt2_baseline_texts = generate_gpt2_baseline_texts(
                prompt, prompt_mask, cot_gt, num_thoughts, tokenizer, gpt2_model
            )
            
            # Display side-by-side comparison for each CoT sequence
            print(f"\n{'='*120}")
            print(f"SIDE-BY-SIDE COMPARISON FOR EXAMPLE {i+1}")
            print(f"{'='*120}")
            
            # Print metrics
            if vq_loss_tf is not None and perplexity_tf is not None:
                print(f"Teacher Forcing - VQ Loss: {vq_loss_tf.item():.4f}, Perplexity: {perplexity_tf.item():.2f}")
                if indices_tf is not None:
                    unique_indices = torch.unique(indices_tf).numel()
                    print(f"  Codebook usage: {unique_indices} unique indices out of {indices_tf.numel()} total")
                
                # Compute reconstruction loss for teacher forcing
                if output_logits_tf is not None and cot_gt is not None and cot_mask_ex is not None:
                    recon_loss_tf = compute_reconstruction_loss(output_logits_tf, cot_gt, cot_mask_ex)
                    print(f"  Reconstruction Loss: {recon_loss_tf.item():.4f}")
                    
            if vq_loss_ar is not None and perplexity_ar is not None:
                print(f"Auto-regressive - VQ Loss: {vq_loss_ar.item():.4f}, Perplexity: {perplexity_ar.item():.2f}")
                if indices_ar is not None:
                    unique_indices = torch.unique(indices_ar).numel()
                    print(f"  Codebook usage: {unique_indices} unique indices out of {indices_ar.numel()} total")
                
                # Compute reconstruction loss for auto-regressive
                if output_logits_ar is not None and cot_gt is not None and cot_mask_ex is not None:
                    recon_loss_ar = compute_reconstruction_loss(output_logits_ar, cot_gt, cot_mask_ex)
                    print(f"  Reconstruction Loss: {recon_loss_ar.item():.4f}")
                    
            print()
            
            # Function to chunk text into 20-word segments
            def chunk_text(text, chunk_size=20):
                """Split text into chunks of specified word count and on newlines."""
                if not text:
                    return []
                
                # Split on newlines first
                lines = text.split('\n')
                chunks = []
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    words = line.split()
                    for i in range(0, len(words), chunk_size):
                        chunk = ' '.join(words[i:i + chunk_size])
                        if chunk:
                            chunks.append(chunk)
                
                return chunks
            
            # Display each CoT sequence with chunked side-by-side comparison
            for j in range(num_thoughts):
                print(f"\n--- CoT {j+1} ---")
                
                # Get the three versions of the CoT
                cot_gt_text = decode_tokens(cot_gt[0, j], cot_mask_ex[0, j] if cot_mask_ex is not None else None)
                
                if predicted_tokens_tf is not None:
                    cot_tf_text = decode_tokens(predicted_tokens_tf[0, j], cot_mask_ex[0, j] if cot_mask_ex is not None else None)
                else:
                    cot_tf_text = "FAILED"
                
                if output_sequences_ar is not None:
                    cot_ar_text = decode_tokens(output_sequences_ar[0, j], cot_mask_ex[0, j] if cot_mask_ex is not None else None)
                else:
                    cot_ar_text = "FAILED"
                
                # GPT2 baseline text
                gpt2_text = gpt2_baseline_texts[j] if j < len(gpt2_baseline_texts) else "FAILED"
                
                # Chunk all four texts
                gt_chunks = chunk_text(cot_gt_text)
                tf_chunks = chunk_text(cot_tf_text)
                ar_chunks = chunk_text(cot_ar_text)
                gpt2_chunks = chunk_text(gpt2_text)
                
                # Find the maximum number of chunks
                max_chunks = max(len(gt_chunks), len(tf_chunks), len(ar_chunks), len(gpt2_chunks))
                
                # Display chunks side-by-side
                for chunk_idx in range(max_chunks):
                    gt_chunk = gt_chunks[chunk_idx] if chunk_idx < len(gt_chunks) else ""
                    tf_chunk = tf_chunks[chunk_idx] if chunk_idx < len(tf_chunks) else ""
                    ar_chunk = ar_chunks[chunk_idx] if chunk_idx < len(ar_chunks) else ""
                    gpt2_chunk = gpt2_chunks[chunk_idx] if chunk_idx < len(gpt2_chunks) else ""
                    
                    # Pad chunks to same length for alignment
                    max_length = max(len(gt_chunk), len(tf_chunk), len(ar_chunk), len(gpt2_chunk))
                    gt_chunk_padded = gt_chunk.ljust(max_length)
                    tf_chunk_padded = tf_chunk.ljust(max_length)
                    ar_chunk_padded = ar_chunk.ljust(max_length)
                    gpt2_chunk_padded = gpt2_chunk.ljust(max_length)
                    
                    print(f"Original:        {gt_chunk_padded}")
                    print(f"Teacher Forced:  {tf_chunk_padded}")
                    print(f"Auto-regressive: {ar_chunk_padded}")
                    print(f"GPT2 Baseline:   {gpt2_chunk_padded}")
                    
                    # Add separator after each chunk (except the last one)
                    if chunk_idx < max_chunks - 1:
                        print("-" * 60)
            
            print("\n" + "="*120)
    
    if use_vq:
        # Create comprehensive heatmaps from accumulated indices
        print(f"\n{'='*80}")
        print("GENERATING COMPREHENSIVE CODEBOOK USAGE HEATMAPS")
        print(f"{'='*80}")
        
        # Combine all accumulated indices
        if all_indices_tf:
            combined_indices_tf = torch.cat(all_indices_tf, dim=0)
            print(f"Teacher Forcing - Total indices: {combined_indices_tf.numel()}")
            print(f"Teacher Forcing - Unique indices: {torch.unique(combined_indices_tf).numel()}")
            
            # Create comprehensive heatmap for teacher forcing
            heatmap_path_tf = os.path.join(os.path.dirname(checkpoint_path), "codebook_usage_tf_comprehensive.png")
            # Convert indices to counts
            counts_tf = torch.bincount(combined_indices_tf[combined_indices_tf < num_embeddings], 
                                      minlength=num_embeddings)
            create_codebook_usage_heatmap(
                counts_tf, 
                num_embeddings=num_embeddings,
                title=f"Teacher Forcing Codebook Usage - All {num_examples} Examples",
                save_path=heatmap_path_tf
            )
        
        if all_indices_ar:
            combined_indices_ar = torch.cat(all_indices_ar, dim=0)
            print(f"Auto-regressive - Total indices: {combined_indices_ar.numel()}")
            print(f"Auto-regressive - Unique indices: {torch.unique(combined_indices_ar).numel()}")
            
            # Create comprehensive heatmap for auto-regressive
            heatmap_path_ar = os.path.join(os.path.dirname(checkpoint_path), "codebook_usage_ar_comprehensive.png")
            # Convert indices to counts
            counts_ar = torch.bincount(combined_indices_ar[combined_indices_ar < num_embeddings], 
                                      minlength=num_embeddings)
            create_codebook_usage_heatmap(
                counts_ar, 
                num_embeddings=num_embeddings,
                title=f"Auto-regressive Codebook Usage - All {num_examples} Examples",
                save_path=heatmap_path_ar
            )
    
    print("\nDemonstration completed!")

def create_codebook_usage_heatmap(counts: torch.Tensor, 
                                 num_embeddings: int,
                                 title: str = "Codebook Usage Heatmap",
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (12, 8),
                                 cmap: str = 'viridis',
                                 show_counts: bool = True) -> None:
    """
    Create and optionally save a heatmap showing codebook usage from counts.
    
    Args:
        counts (torch.Tensor): Tensor of codebook usage counts [num_embeddings]
        num_embeddings (int): Total number of embeddings in the codebook
        title (str): Title for the heatmap
        save_path (Optional[str]): Path to save the heatmap image (if None, only displays)
        figsize (Tuple[int, int]): Figure size (width, height)
        cmap (str): Colormap for the heatmap
        show_counts (bool): Whether to show count values on the heatmap
        
    Example:
        >>> counts = torch.tensor([10, 5, 3, 0, 2])  # Usage counts for 5 embeddings
        >>> create_codebook_usage_heatmap(counts, num_embeddings=5, save_path="heatmap.png")
    """
    # Ensure counts is the right shape and convert to numpy
    if counts.dim() > 1:
        counts = counts.flatten()
    
    counts_np = counts.cpu().numpy()
    
    # Create the heatmap
    fig, ax = plt.subplots(figsize=figsize)
    
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

def create_codebook_usage_timeline_plot(codebook_history: List[torch.Tensor], 
                                      num_embeddings: int,
                                      measurement_points: List[int],
                                      title: str = "Codebook Usage Over Time",
                                      save_path: Optional[str] = None,
                                      figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Create a 3D visualization showing codebook usage distribution over time.
    
    Args:
        codebook_history: List of count tensors, one per measurement point
        num_embeddings: Total number of embeddings in the codebook
        measurement_points: List of measurement point indices (e.g., batch numbers)
        title: Title for the plot
        save_path: Optional path to save the plot
        figsize: Figure size (width, height)
    """
    if not codebook_history:
        print("Warning: No codebook history to plot")
        return
    
    # Convert to numpy arrays
    history_np = [counts.numpy() for counts in codebook_history]
    
    # Create 3D plot
    fig = plt.figure(figsize=figsize)
    ax: Any = fig.add_subplot(111, projection='3d')  # Type annotation for 3D axes
    
    # Create meshgrid for 3D surface
    x = np.arange(num_embeddings)  # Codebook indices
    y = np.array(measurement_points)  # Time points
    X, Y = np.meshgrid(x, y)
    
    # Create Z matrix (usage counts over time)
    Z = np.array(history_np)
    
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
    avg_unique_codes = np.mean([(counts > 0).sum() for counts in history_np])
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
                    _, _, _, _, indices = model(
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

def main():
    """
    Main function for command-line training with memory optimizations.
    """
    parser = argparse.ArgumentParser(description='Train GPT2VQVAE model with memory optimizations and perplexity threshold monitoring')
    parser.add_argument('--config', '-c', type=str, #not required anymore as it will be skipped when demonstrating
                       help='Path to configuration file (YAML or JSON)')
    parser.add_argument('--data-dir', type=str, default=None,
                       help='Override data directory from config')
    parser.add_argument('--resume-from', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    parser.add_argument('--create-config', type=str, default=None,
                       help='Create a default configuration file at the specified path')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to train on (cuda/cpu)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum number of samples to load (for debugging/memory constraints)')
    parser.add_argument('--monitor-gpu-memory', action='store_true', default=True,
                       help='Monitor GPU memory usage using nvidia-ml-py3 (default: True)')
    parser.add_argument('--num-thoughts', type=int, default=None,
                       help='Override num_thoughts parameter from config (truncates dataset if needed)')
    parser.add_argument('--demonstrate', type=str, default=None,
                       help='Demonstrate model generation from checkpoint (provide checkpoint path)')
    parser.add_argument('--demonstrate-custom', type=str, default=None,
                       help='Demonstrate model generation from checkpoint using custom prompt-CoT files (provide checkpoint path)')
    parser.add_argument('--prompt-file', type=str, default='test/test_prompt.txt',
                       help='Path to file containing custom prompt (used with --demonstrate-custom)')
    parser.add_argument('--cot-file', type=str, default='test/test_cot.txt',
                       help='Path to file containing custom CoT (used with --demonstrate-custom)')
    parser.add_argument('--num-examples', type=int, default=3,
                       help='Number of examples to generate in demonstration mode')
    parser.add_argument('--demo-seed', type=int, default=42,
                       help='Random seed for demonstration sampling (default: 42)')
    parser.add_argument('--perplexity-threshold', type=float, default=None,
                       help='Training will abort when 20-step average perplexity goes below this value (default: 1.5)')
    parser.add_argument('--perplexity-window-size', type=int, default=None,
                       help='Number of steps to average for perplexity threshold check (default: 20)')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                       help='Override checkpoint directory from config')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate from config')
    parser.add_argument('--min-lr', type=float, default=None,
                       help='Override minimum learning rate from config')
    parser.add_argument('--vq-loss-weight', type=float, default=None,
                       help='Override VQ loss weight from config')
    parser.add_argument('--minimum-batches-for-checkpoint', type=int, default=None,
                       help='Override minimum batches required to save aborted checkpoint (default: 200)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size from config')
    parser.add_argument('--simple', action='store_true', default=False,
                       help='Use SimpleGPT2VQVAETrainer and SimpleGPT2VQVAE model (default: False)')
    parser.add_argument('--enhanced', action='store_true', default=False,
                       help='Use EnhancedGPT2VQVAETrainer and EnhancedGPT2VQVAE model (default: False)')
    parser.add_argument('--phased', action='store_true', default=False,
                       help='Use PhasedEnhancedGPT2VQVAETrainer for phased training (requires --enhanced)')
    parser.add_argument('--use-vq', action='store_true', default=None,
                       help='Enable vector quantization (default: True for SimpleGPT2VQVAE, True for GPT2VQVAE)')
    parser.add_argument('--no-vq', action='store_true', default=False,
                       help='Disable vector quantization (passes encoder outputs directly to decoder)')
    
    # Enhanced VQ-VAE specific arguments
    parser.add_argument('--ema-decay', type=float, default=None,
                       help='Override EMA decay rate for enhanced VQ-VAE (default: 0.99)')
    parser.add_argument('--diversity-gamma', type=float, default=None,
                       help='Override diversity loss weight for enhanced VQ-VAE (default: 0.1)')
    parser.add_argument('--reset-threshold', type=float, default=None,
                       help='Override codebook reset threshold for enhanced VQ-VAE (default: 0.1)')
    parser.add_argument('--reset-frequency', type=int, default=None,
                       help='Override codebook reset frequency for enhanced VQ-VAE (default: 1000)')
    parser.add_argument('--disable-ema', action='store_true', default=False,
                       help='Disable EMA updates in enhanced VQ-VAE')
    parser.add_argument('--enhanced-codebook-tracking', action='store_true', default=None,
                       help='Enable enhanced codebook tracking (default: True for enhanced VQ-VAE)')
    parser.add_argument('--no-enhanced-codebook-tracking', action='store_true', default=False,
                       help='Disable enhanced codebook tracking')
    
    # Phased training specific arguments
    parser.add_argument('--initialization-steps', type=int, default=None,
                       help='Override initialization_steps for phased training (default: 1500)')
    parser.add_argument('--r-reestim', type=int, default=None,
                       help='Override r_reestim for phased training (default: 500)')
    parser.add_argument('--quantization-start', type=int, default=None,
                       help='Override quantization_start for phased training (default: 5000)')
    parser.add_argument('--codebook-lr-multiplier', type=float, default=None,
                       help='Override codebook_lr_multiplier for phased training (default: 1.0)')
    
    args = parser.parse_args()
    
    # Validate phased training arguments
    if args.phased and not args.enhanced:
        print("Error: --phased requires --enhanced to be enabled")
        print("Please use both --enhanced and --phased flags together")
        return

    if not args.config and not (args.demonstrate or args.demonstrate_custom):
        print("Error: --config required unless --demonstrate or --demonstrate-custom is specified")
        return
    
    # Check for nvidia-ml-py3 availability
    if args.monitor_gpu_memory and not NVML_AVAILABLE:
        print("Warning: nvidia-ml-py3 not available for GPU memory monitoring.")
        print("Install it with: pip install nvidia-ml-py3")
        print("Continuing without detailed GPU memory monitoring...")
    
    # Handle create-config option
    if args.create_config:
        # Determine if enhanced VQ-VAE config is requested
        create_default_config(args.create_config, enhanced_vq=args.enhanced, phased_training=args.phased)
        return
    
    try:
        # Load configuration
        if args.config:
            print(f"Loading configuration from: {args.config}")
            model_config, training_config = load_config(args.config)
            # Set run name to be used for push notifications
            run_name = os.path.basename(args.config)
        else:
            model_config, training_config = {}, {}
        
        # Override data directory if specified
        if args.data_dir:
            training_config['data_dir'] = args.data_dir
            print(f"Overriding data_dir from config to: {args.data_dir}")
        
        # Override max samples if specified
        if args.max_samples:
            training_config['max_samples'] = args.max_samples
            print(f"Overriding max_samples from config to: {args.max_samples}")
        
        # Override perplexity threshold if specified
        if args.perplexity_threshold:
            training_config['perplexity_threshold'] = args.perplexity_threshold
            print(f"Overriding perplexity_threshold from config to: {args.perplexity_threshold}")
        
        if args.perplexity_window_size:
            training_config['perplexity_window_size'] = args.perplexity_window_size
            print(f"Overriding perplexity_window_size from config to: {args.perplexity_window_size}")
        
        # Override checkpoint directory if specified
        if args.checkpoint_dir:
            training_config['checkpoint_dir'] = args.checkpoint_dir
            print(f"Overriding checkpoint_dir from config to: {args.checkpoint_dir}")
        
        # Override learning rate if specified
        if args.lr:
            training_config['learning_rate'] = args.lr
            print(f"Overriding learning_rate from config to: {args.lr}")
        
        # Override minimum learning rate if specified
        if args.min_lr:
            training_config['min_lr'] = args.min_lr
            print(f"Overriding min_lr from config to: {args.min_lr}")
        
        # Override VQ loss weight if specified
        if args.vq_loss_weight:
            training_config['vq_loss_weight'] = args.vq_loss_weight
            print(f"Overriding vq_loss_weight from config to: {args.vq_loss_weight}")
        
        # Override minimum batches for checkpoint if specified
        if args.minimum_batches_for_checkpoint:
            training_config['minimum_batches_for_checkpoint'] = args.minimum_batches_for_checkpoint
            print(f"Overriding minimum_batches_for_checkpoint from config to: {args.minimum_batches_for_checkpoint}")
        
        # Override batch size if specified
        if args.batch_size:
            training_config['batch_size'] = args.batch_size
            print(f"Overriding batch_size from config to: {args.batch_size}")
        
        # Handle use_vq argument
        if args.no_vq:
            training_config['use_vq'] = False
            print("Disabling vector quantization (--no-vq flag)")
        elif args.use_vq is not None:
            training_config['use_vq'] = args.use_vq
            print(f"Setting use_vq to: {args.use_vq}")
        else:
            # Default behavior: use_vq=True for both models
            training_config['use_vq'] = True
            print("Using default use_vq=True")
        
        # Handle enhanced VQ-VAE specific arguments
        if args.enhanced:
            print("Enhanced VQ-VAE mode enabled")
            
            # Override enhanced VQ-VAE parameters if specified
            if args.ema_decay is not None:
                model_config['ema_decay'] = args.ema_decay
                print(f"Overriding ema_decay to: {args.ema_decay}")
            
            if args.diversity_gamma is not None:
                model_config['diversity_gamma'] = args.diversity_gamma
                print(f"Overriding diversity_gamma to: {args.diversity_gamma}")
            
            if args.reset_threshold is not None:
                model_config['reset_threshold'] = args.reset_threshold
                print(f"Overriding reset_threshold to: {args.reset_threshold}")
            
            if args.reset_frequency is not None:
                model_config['reset_frequency'] = args.reset_frequency
                print(f"Overriding reset_frequency to: {args.reset_frequency}")
            
            if args.disable_ema:
                model_config['use_ema'] = False
                print("Disabling EMA updates (--disable-ema flag)")
            
            # Handle enhanced codebook tracking
            if args.no_enhanced_codebook_tracking:
                training_config['enhanced_codebook_tracking'] = False
                print("Disabling enhanced codebook tracking (--no-enhanced-codebook-tracking flag)")
            elif args.enhanced_codebook_tracking is not None:
                training_config['enhanced_codebook_tracking'] = args.enhanced_codebook_tracking
                print(f"Setting enhanced_codebook_tracking to: {args.enhanced_codebook_tracking}")
            else:
                # Default behavior: enable enhanced codebook tracking for enhanced VQ-VAE
                training_config['enhanced_codebook_tracking'] = True
                print("Using default enhanced_codebook_tracking=True for enhanced VQ-VAE")
            
            # Handle phased training parameters
            if args.phased:
                print("Phased training mode enabled")
                
                # Override phased training parameters if specified
                if args.initialization_steps is not None:
                    training_config['initialization_steps'] = args.initialization_steps
                    print(f"Overriding initialization_steps to: {args.initialization_steps}")
                
                if args.r_reestim is not None:
                    training_config['r_reestim'] = args.r_reestim
                    print(f"Overriding r_reestim to: {args.r_reestim}")
                
                if args.quantization_start is not None:
                    training_config['quantization_start'] = args.quantization_start
                    print(f"Overriding quantization_start to: {args.quantization_start}")
                
                if args.codebook_lr_multiplier is not None:
                    training_config['codebook_lr_multiplier'] = args.codebook_lr_multiplier
                    print(f"Overriding codebook_lr_multiplier to: {args.codebook_lr_multiplier}")
        else:
            # For non-enhanced models, disable enhanced codebook tracking
            training_config['enhanced_codebook_tracking'] = False
        
        # Set device
        if args.device:
            device = args.device
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Using device: {device}")
        if device == "cuda":
            print(f"CUDA device: {torch.cuda.get_device_name()}")
            print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            # Show GPU memory monitoring status
            if args.monitor_gpu_memory and NVML_AVAILABLE:
                print("GPU memory monitoring: Enabled (nvidia-ml-py3)")
            elif args.monitor_gpu_memory and not NVML_AVAILABLE:
                print("GPU memory monitoring: Disabled (nvidia-ml-py3 not available)")
            else:
                print("GPU memory monitoring: Disabled (--monitor-gpu-memory=False)")
        
        data_dir = training_config.get('data_dir', 'data/GSM8K')
        max_samples = training_config.get('max_samples', None)
        num_thoughts = model_config.get('num_thoughts', None)  # Extract from model config

        # Run demonstration if requested
        if args.demonstrate:
            if args.phased or args.enhanced:
                model_type = "EnhancedGPT2VQVAE"  # Phased trainer uses EnhancedGPT2VQVAE model
            elif args.simple:
                model_type = "SimpleGPT2VQVAE"
            else:
                model_type = "GPT2VQVAE"
            print(f"\nRunning demonstration with checkpoint: {args.demonstrate}")
            print(f"Using model type: {model_type}")
            demonstrate_model_from_checkpoint(
                checkpoint_path=args.demonstrate,
                data_dir=data_dir,
                num_examples=args.num_examples,
                device=device,
                use_vq=training_config.get('use_vq', True),
                model_type=model_type,
                seed=args.demo_seed, 
                **model_config
            )
            return  # Exit after demonstration
        
        # Run custom demonstration if requested
        if args.demonstrate_custom:
            if args.phased or args.enhanced:
                model_type = "EnhancedGPT2VQVAE"  # Phased trainer uses EnhancedGPT2VQVAE model
            elif args.simple:
                model_type = "SimpleGPT2VQVAE"
            else:
                model_type = "GPT2VQVAE"
            print(f"\nRunning custom demonstration with checkpoint: {args.demonstrate_custom}")
            print(f"Using model type: {model_type}")
            print(f"Prompt file: {args.prompt_file}")
            print(f"CoT file: {args.cot_file}")
            demonstrate_custom_prompt_cot(
                checkpoint_path=args.demonstrate_custom,
                prompt_file=args.prompt_file,
                cot_file=args.cot_file,
                device=device,
                use_vq=training_config.get('use_vq', True),
                model_type=model_type,
                **model_config
            )
            return  # Exit after demonstration
        
        # Command-line argument takes precedence
        if args.num_thoughts is not None:
            num_thoughts = args.num_thoughts
            print(f"Overriding num_thoughts from config ({model_config.get('num_thoughts', 'not set')}) "
                  f"to command-line value: {num_thoughts}")
            # Update model config for consistency
            model_config['num_thoughts'] = num_thoughts
        
        print(f"Loading data from: {data_dir}")
        if num_thoughts is not None:
            print(f"Using num_thoughts: {num_thoughts}")
        
        # Initialize trainer first to use its memory-efficient loading method
        print("Initializing trainer...")
        
        # If resume_from, initialize weights randomly
        if args.resume_from:
            model_config['use_pretrained_encoder'] = False
            model_config['use_pretrained_decoder'] = False
            if args.phased:
                trainer = PhasedEnhancedGPT2VQVAETrainer(model_config, training_config, device=device, run_name=run_name)
            elif args.enhanced:
                trainer = EnhancedGPT2VQVAETrainer(model_config, training_config, device=device, run_name=run_name)
            elif args.simple:
                trainer = SimpleGPT2VQVAETrainer(model_config, training_config, device=device, run_name=run_name)
            else:
                trainer = GPT2VQVAETrainer(model_config, training_config, device=device, run_name=run_name)
            print(f"Resuming from checkpoint: {args.resume_from}")
            trainer.load_checkpoint(args.resume_from)
        else:
            if args.phased:
                trainer = PhasedEnhancedGPT2VQVAETrainer(model_config, training_config, device=device, run_name=run_name)
            elif args.enhanced:
                trainer = EnhancedGPT2VQVAETrainer(model_config, training_config, device=device, run_name=run_name)
            elif args.simple:
                trainer = SimpleGPT2VQVAETrainer(model_config, training_config, device=device, run_name=run_name)
            else:
                trainer = GPT2VQVAETrainer(model_config, training_config, device=device, run_name=run_name)

        # Load training and test data using memory-efficient method with num_thoughts truncation
        train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask, \
        test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask = trainer.load_training_data(
            data_dir, max_samples=max_samples, num_thoughts=num_thoughts
        )
        
        # Validate model-data compatibility for both train and test sets
        print("Validating train data compatibility...")
        validate_model_data_compatibility(model_config, train_prompt_sequences, train_cot_sequences, train_prompt_mask, train_cot_mask)
        print("Validating test data compatibility...")
        validate_model_data_compatibility(model_config, test_prompt_sequences, test_cot_sequences, test_prompt_mask, test_cot_mask)
        
        # Start training
        print("Starting training...")
        print(f"Perplexity threshold monitoring: {training_config.get('perplexity_threshold', 1.5)} (window size: {training_config.get('perplexity_window_size', 20)})")
        print(f"Minimum batches for aborted checkpoint: {training_config.get('minimum_batches_for_checkpoint', 200)}")
        trainer.train(
            train_prompt_sequences=train_prompt_sequences,
            train_cot_sequences=train_cot_sequences,
            train_prompt_mask=train_prompt_mask,
            train_cot_mask=train_cot_mask,
            test_prompt_sequences=test_prompt_sequences,
            test_cot_sequences=test_cot_sequences,
            test_prompt_mask=test_prompt_mask,
            test_cot_mask=test_cot_mask,
            resume_from=args.resume_from,
            num_measurements_per_epoch=training_config.get('num_measurements_per_epoch', 20)
        )
        
        
        print(f"Best model saved to: {trainer.best_model_path}")
        
        # Print final memory statistics
        if trainer.memory_stats:
            final_memory = trainer.memory_stats[-1]
            if 'pytorch_allocated_gb' in final_memory:
                print(f"Final memory usage: {final_memory['pytorch_allocated_gb']:.2f}GB allocated, {final_memory['pytorch_max_allocated_gb']:.2f}GB max")
            else:
                print(f"Final memory usage: {final_memory['allocated_gb']:.2f}GB allocated, {final_memory['max_allocated_gb']:.2f}GB max")
        
    except TrainingAbortedException as e:
        # Handle training abortion gracefully
        print(f"\n✅ Training completed successfully (aborted due to {e.reason})")
        print(f"Training stopped at epoch {e.epoch} with final perplexity: {e.final_perplexity:.4f}")
        
        # Print final memory statistics for aborted training
        if trainer.memory_stats:
            final_memory = trainer.memory_stats[-1]
            if 'pytorch_allocated_gb' in final_memory:
                print(f"Final memory usage: {final_memory['pytorch_allocated_gb']:.2f}GB allocated, {final_memory['pytorch_max_allocated_gb']:.2f}GB max")
            else:
                print(f"Final memory usage: {final_memory['allocated_gb']:.2f}GB allocated, {final_memory['max_allocated_gb']:.2f}GB max")
    
    except torch.OutOfMemoryError as e:
        # Training failed due to OOM - clear any existing cache
        print("*"*60)
        print("Training halted due to OOM error: clearing cache & collecting garbage.")
        print("*"*60)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Save training visualizations before re-raising the exception
        save_training_visualizations(trainer, prefix="oom_error")
        raise
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check that the configuration file and data files exist.")
    except Exception as e:
        print(f"Training error: {e}")
        traceback.print_exc()
        
        # Save training visualizations before exiting
        if 'trainer' in locals():
            print("Saving training results as figures.")
            save_training_visualizations(trainer, prefix="error")

class SimpleGPT2VQVAETrainer(GPT2VQVAETrainer):
    """
    Trainer for SimpleGPT2VQVAE, inherits from GPT2VQVAETrainer but uses SimpleGPT2VQVAE as the model.
    """
    def __init__(self, 
                 model_config: Dict[str, Any], 
                 training_config: Dict[str, Any], 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu", 
                 run_name : Optional[str] = "ANONYM_RUN"):
        super().__init__(model_config, training_config, device, run_name)

        self.ensure_numeric_types(model_config)
        self.ensure_numeric_types(training_config)

        # Replace the model with SimpleGPT2VQVAE
        self.model = SimpleGPT2VQVAE(**model_config).to(device)
        # Re-initialize optimizer and scheduler for the new model
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 0.01),
            betas=(training_config.get('beta1', 0.9), training_config.get('beta2', 0.999))
        )
        if training_config.get('use_lr_scheduler', True):
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config['num_epochs'],
                eta_min=training_config.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = None
        if self.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled for SimpleGPT2VQVAE")
        print("SimpleGPT2VQVAE trainer initialized successfully.")
    
    def _forward_pass(self, prompts, cots, prompt_masks, cot_masks):
        """Helper function for forward pass and loss calculation with use_vq parameter"""

        # Get use_vq from training config
        use_vq = self.training_config.get('use_vq', True)
        
        if self.use_mixed_precision:
            with autocast('cuda'):
                _, output_logits, vq_loss, perplexity, indices = self.model(
                    prompt=prompts,
                    cot_sequences=cots,
                    cot_mask=cot_masks,
                    prompt_mask=prompt_masks,
                    inference=False,
                    quantize_cot_only=self.training_config.get('quantize_cot_only', True),
                    use_vq=use_vq
                )
                recon_loss = compute_reconstruction_loss(output_logits, cots, cot_masks)
                total_loss_batch = recon_loss + self.training_config.get('vq_loss_weight', 1.0) * vq_loss
        else:
            _, output_logits, vq_loss, perplexity, indices = self.model(
                prompt=prompts,
                cot_sequences=cots,
                cot_mask=cot_masks,
                prompt_mask=prompt_masks,
                inference=False,
                quantize_cot_only=self.training_config.get('quantize_cot_only', True),
                use_vq=use_vq
            )
            recon_loss = compute_reconstruction_loss(output_logits, cots, cot_masks)
            total_loss_batch = recon_loss + self.training_config.get('vq_loss_weight', 1.0) * vq_loss
            
        return total_loss_batch, vq_loss, perplexity, indices


class EnhancedGPT2VQVAETrainer(GPT2VQVAETrainer):
    """
    Trainer for EnhancedGPT2VQVAE, inherits from GPT2VQVAETrainer but uses EnhancedGPT2VQVAE as the model.
    
    This enhanced trainer provides additional functionality for the enhanced vector quantizer:
    - EMA (Exponential Moving Average) updates for codebook learning
    - Diversity-promoting loss to encourage uniform codebook usage
    - Automatic codebook reset mechanisms for unused embeddings
    - Enhanced monitoring and statistics for codebook health
    
    The enhanced codebook training scheme reduces to normal VQ-VAE training when:
    - ema_decay = 0.0 (no EMA updates)
    - diversity_gamma = 0.0 (no diversity loss)
    - reset_threshold = 0.0 (no automatic resets)
    - use_ema = False (EMA disabled)
    """
    def __init__(self, 
                 model_config: Dict[str, Any], 
                 training_config: Dict[str, Any], 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu", 
                 tracking_functions: Optional[Dict[str, Any]] = None, 
                 run_name : Optional[str] = "ANONYM_RUN"):
        # Filter out enhanced VQ-VAE specific parameters for parent constructor
        enhanced_vq_params = {
            'ema_decay', 'diversity_gamma', 'reset_threshold', 
            'reset_frequency', 'use_ema', 'reset_stop_fraction',
            # for further enhancement
            'max_reset_steps', 'reservoir_size', 'reset_strategy', 'use_batch_norm'
        }
        
        # Create filtered configs for parent constructor
        filtered_model_config = {k: v for k, v in model_config.items() if k not in enhanced_vq_params}
        filtered_training_config = {k: v for k, v in training_config.items() if k != 'enhanced_codebook_tracking'}
        
        # Set up enhanced tracking functions, for those not provided
        enhanced_tracking_functions = {
            'track_codebook_usage': tracking_functions.get('track_codebook_usage', self._enhanced_track_codebook_usage),
            'save_codebook_plots': tracking_functions.get('save_codebook_plots', self._enhanced_save_codebook_plots),
            'tracking_enabled': tracking_functions.get('tracking_enabled', training_config.get('enhanced_codebook_tracking', True))
        }
        
        # Call parent constructor with filtered configs and enhanced tracking functions
        super().__init__(filtered_model_config, filtered_training_config, device, enhanced_tracking_functions, run_name)

        # Store original configs for enhanced features
        self.model_config = model_config
        self.training_config = training_config
        
        self.ensure_numeric_types(self.model_config)
        self.ensure_numeric_types(self.training_config)
        
        # Manage some model_config entries
        if 'max_reset_steps' not in model_config:
            model_config['max_reset_steps'] = None
        
        # Replace the model with EnhancedGPT2VQVAE using original config
        self.model = EnhancedGPT2VQVAE(**model_config).to(device)
        
        # Re-initialize optimizer and scheduler for the new model
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=training_config['learning_rate'],
            weight_decay=training_config.get('weight_decay', 0.01),
            betas=(training_config.get('beta1', 0.9), training_config.get('beta2', 0.999))
        )
        if training_config.get('use_lr_scheduler', True):
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=training_config['num_epochs'],
                eta_min=training_config.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = None
        if self.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled for EnhancedGPT2VQVAE")
        
        # Enhanced codebook tracking with granular control
        self.enhanced_codebook_tracking = training_config.get('enhanced_codebook_tracking', True)
        
        # Granular tracking flags for memory optimization
        tracking_config = training_config.get('codebook_tracking_config', {})
        self.track_usage_stats = tracking_config.get('track_usage_stats', True)
        self.track_diversity_metrics = tracking_config.get('track_diversity_metrics', True)
        self.track_ema_stats = tracking_config.get('track_ema_stats', True)
        self.track_reset_stats = tracking_config.get('track_reset_stats', True)
        self.track_reservoir_stats = tracking_config.get('track_reservoir_stats', True)
        self.save_tracking_history = tracking_config.get('save_tracking_history', True)
        self.print_tracking_info = tracking_config.get('print_tracking_info', True)
        
        # Initialize tracking history only if needed
        if self.enhanced_codebook_tracking and self.save_tracking_history:
            self.codebook_stats_history = []
            self.diversity_history = []
        else:
            self.codebook_stats_history = None
            self.diversity_history = None
        
        if self.enhanced_codebook_tracking:
            print("Enhanced codebook tracking enabled with granular control:")
            print(f"  - Usage stats: {self.track_usage_stats}")
            print(f"  - Diversity metrics: {self.track_diversity_metrics}")
            print(f"  - EMA stats: {self.track_ema_stats}")
            print(f"  - Reset stats: {self.track_reset_stats}")
            print(f"  - Reservoir stats: {self.track_reservoir_stats}")
            print(f"  - Save history: {self.save_tracking_history}")
            print(f"  - Print info: {self.print_tracking_info}")
            print(f"EMA decay: {model_config.get('ema_decay', 0.99)}")
            print(f"Diversity gamma: {model_config.get('diversity_gamma', 0.1)}")
            print(f"Reset threshold: {model_config.get('reset_threshold', 0.1)}")
            print(f"Reset frequency: {model_config.get('reset_frequency', 1000)}")
            print(f"Use EMA: {model_config.get('use_ema', True)}")
            print(f"Reset stop fraction: {self.model_config.get('reset_stop_fraction', 0.2)}")
        else:
            print("Enhanced codebook tracking disabled")
        print("EnhancedGPT2VQVAE trainer initialized successfully.")

        # Training phase tracking - no current use, but might be useful later
        self.current_step = 0

    def train_epoch(self, train_loader, num_measurements_per_epoch, current_epoch=0):
        # Set max_reset_steps on the first epoch if it's still None
        if self.model.vector_quantizer.max_reset_steps is None:
            # Calculate total training steps
            num_epochs = self.training_config['num_epochs']
            steps_per_epoch = len(train_loader)
            total_training_steps = num_epochs * steps_per_epoch
            
            # Calculate max_reset_steps from reset_stop_fraction
            reset_stop_fraction = self.model_config.get('reset_stop_fraction', 0.2)
            max_reset_steps = int(reset_stop_fraction * total_training_steps)
            
            # Set max_reset_steps on the vector quantizer
            self.model.vector_quantizer.max_reset_steps = max_reset_steps
            print(f"Set max_reset_steps to {max_reset_steps} (based on {total_training_steps} total steps, {reset_stop_fraction*100:.0f}% of training)")
        
        # Call the parent train_epoch method
        return super().train_epoch(train_loader, num_measurements_per_epoch, current_epoch)

    def _update_weights(self):
        """
        Enhanced weight update that increments step counter.
        """
        # Call parent weight update
        super()._update_weights()
        # Increment step counter
        self.current_step += 1

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False, checkpoint_path: Optional[str] = None):
        """
        Enhanced checkpoint saving that includes phased training state.
        
        Args:
            epoch: Current epoch
            metrics: Training metrics
            is_best: Whether this is the best model so far
            checkpoint_path: Optional custom checkpoint path
        """
        # Call parent save_checkpoint with current_step info as kwargs
        super().save_checkpoint(epoch, metrics, is_best, checkpoint_path, 
                                **{
                                    "current_step" : self.current_step
                                    })

    def load_checkpoint(self, checkpoint_path: str):
        """
        Enhanced checkpoint loading that restores phased training state.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        # Load checkpoint using parent method
        checkpoint = super().load_checkpoint(checkpoint_path)
        
        # Restore phased training state if available
        if 'current_step' in checkpoint:
            self.current_step = checkpoint['current_step']
            print(f"Restored training step: {self.current_step}")
        else:
            raise Exception("Checkpoint does not contain 'current_step' information. Phased training cannot be resumed properly without this data.")
        
        return checkpoint
    
    def _enhanced_track_codebook_usage(self, dataset: Any, measurement_point: int) -> None:
        """
        Enhanced codebook tracking with additional statistics.
        
        Args:
            dataset: Dataset to sample from
            measurement_point: Current measurement point index
        """
        # First, call the parent class's default codebook tracking
        super()._default_track_codebook_usage(dataset, measurement_point)
        
        # Then, call the enhanced codebook tracking
        if not self.enhanced_codebook_tracking:
            return
        
        try:
            # Initialize stats dictionaries
            codebook_stats = {}
            diversity_metrics = {}
            
            # Get codebook statistics based on tracking flags
            if self.track_usage_stats or self.track_ema_stats or self.track_reset_stats or self.track_reservoir_stats:
                codebook_stats = self.model.get_vector_quantizer_stats()
            
            # Get diversity metrics only if needed
            if self.track_diversity_metrics:
                diversity_metrics = self.model.get_embedding_diversity()
            
            # Store results only if history saving is enabled
            if self.save_tracking_history:
                # Convert any tensors in codebook_stats to CPU
                cpu_codebook_stats = {}
                for key, value in codebook_stats.items():
                    if isinstance(value, torch.Tensor):
                        cpu_codebook_stats[key] = value.cpu().detach()
                    else:
                        cpu_codebook_stats[key] = value
                
                # Convert any tensors in diversity_metrics to CPU
                cpu_diversity_metrics = {}
                for key, value in diversity_metrics.items():
                    if isinstance(value, torch.Tensor):
                        cpu_diversity_metrics[key] = value.cpu().detach()
                    else:
                        cpu_diversity_metrics[key] = value
                
                self.codebook_stats_history.append(cpu_codebook_stats)
                self.diversity_history.append(cpu_diversity_metrics)
            
            # Print enhanced statistics only if enabled
            if self.print_tracking_info:
                print(f"\nEnhanced Codebook tracking (point {measurement_point}):")
                
                # Print usage statistics
                if self.track_usage_stats and codebook_stats:
                    print(f"  Total usage: {codebook_stats.get('total_usage', 'N/A')}")
                    print(f"  Unused codes: {codebook_stats.get('unused_codes', 'N/A')}/{self.model.vector_quantizer.num_embeddings} "
                          f"({codebook_stats.get('unused_ratio', 0)*100:.1f}%)")
                    
                    # Print rarely used codes statistics
                    codes_below_05 = codebook_stats.get('codes_below_0.5_percent', 0)
                    codes_below_1 = codebook_stats.get('codes_below_1.0_percent', 0)
                    codes_below_5 = codebook_stats.get('codes_below_5.0_percent', 0)
                    print(f"  Codes below 0.5%: {codes_below_05}")
                    print(f"  Codes below 1.0%: {codes_below_1}")
                    print(f"  Codes below 5.0%: {codes_below_5}")
                
                # Print reset statistics
                if self.track_reset_stats and codebook_stats:
                    print(f"  Reset counter: {codebook_stats.get('reset_counter', 'N/A')}")
                
                # Print diversity metrics
                if self.track_diversity_metrics and diversity_metrics:
                    print(f"  Mean similarity: {diversity_metrics.get('mean_similarity', 0):.4f}")
                    print(f"  Embedding norm mean: {diversity_metrics.get('embedding_norm_mean', 0):.4f}")
                
                # Print EMA statistics
                if self.track_ema_stats and codebook_stats and 'ema_cluster_sizes' in codebook_stats:
                    ema_usage = (codebook_stats['ema_cluster_sizes'] > 0).sum().item()
                    print(f"  EMA active clusters: {ema_usage}/{self.model.vector_quantizer.num_embeddings}")
                
                # Print reservoir statistics
                if self.track_reservoir_stats and codebook_stats:
                    reservoir_size = codebook_stats.get('reservoir_size', 'N/A')
                    reservoir_count = codebook_stats.get('reservoir_count', 'N/A')
                    print(f"  Reservoir size: {reservoir_size}, count: {reservoir_count}")
            
            # Clear cache after enhanced tracking
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Failed to track enhanced codebook usage: {e}")
    
    def _enhanced_save_codebook_plots(self, save_dir: str, epoch: int) -> None:
        """
        Save enhanced codebook tracking visualizations.
        
        Args:
            save_dir: Directory to save plots
            epoch: Current epoch number
        """
        # First, call the parent class's default codebook plotting
        super()._default_save_codebook_plots(save_dir, epoch)
        
        # Then, save the enhanced codebook plots only if tracking is enabled and history exists
        if not self.enhanced_codebook_tracking or not self.save_tracking_history or not self.codebook_stats_history:
            return
        
        try:
            # Save enhanced statistics plots only if we have usage/reset/EMA stats
            if self.track_usage_stats or self.track_reset_stats or self.track_ema_stats:
                stats_path = os.path.join(save_dir, f"enhanced_codebook_stats_epoch_{epoch}.png")
                self._plot_enhanced_codebook_stats(stats_path)
            
            # Save diversity evolution plots only if diversity tracking is enabled
            if self.track_diversity_metrics and self.diversity_history:
                diversity_path = os.path.join(save_dir, f"codebook_diversity_evolution_epoch_{epoch}.png")
                self._plot_diversity_evolution(diversity_path)
            
            print(f"Enhanced codebook plots saved to {save_dir}")
            
            # Clear cache after enhanced plotting
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Failed to save enhanced codebook plots: {e}")
    
    def _plot_enhanced_codebook_stats(self, save_path: str) -> None:
        """Plot enhanced codebook statistics over time."""
        if not self.codebook_stats_history or not self.save_tracking_history:
            return
        
        # Create a larger figure to accommodate more plots
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        
        # Extract data
        epochs = list(range(len(self.codebook_stats_history)))
        total_usage = [stats['total_usage'] for stats in self.codebook_stats_history]
        unused_ratio = [stats['unused_ratio'] for stats in self.codebook_stats_history]
        reset_counter = [stats['reset_counter'] for stats in self.codebook_stats_history]
        
        # Extract rarely used codes data
        codes_below_05 = np.array([stats.get('codes_below_0.5_percent', 0) for stats in self.codebook_stats_history])
        codes_below_1  = np.array([stats.get('codes_below_1.0_percent', 0) for stats in self.codebook_stats_history])
        codes_below_5  = np.array([stats.get('codes_below_5.0_percent', 0) for stats in self.codebook_stats_history])
        unused_codes   = np.array([stats.get('unused_codes',            0) for stats in self.codebook_stats_history])

        # Plot total usage
        axes[0, 0].plot(epochs, total_usage, marker='o')
        axes[0, 0].set_title('Total Codebook Usage')
        axes[0, 0].set_xlabel('Measurement Point')
        axes[0, 0].set_ylabel('Total Usage')
        axes[0, 0].grid(True)
        
        # Plot unused ratio
        axes[0, 1].plot(epochs, unused_ratio, marker='s', color='red')
        axes[0, 1].set_title('Unused Code Ratio')
        axes[0, 1].set_xlabel('Measurement Point')
        axes[0, 1].set_ylabel('Unused Ratio')
        axes[0, 1].grid(True)
        
        # Plot reset counter
        axes[1, 0].plot(epochs, reset_counter, marker='^', color='green')
        axes[1, 0].set_title('Reset Counter')
        axes[1, 0].set_xlabel('Measurement Point')
        axes[1, 0].set_ylabel('Reset Count')
        axes[1, 0].grid(True)
        
        # Plot rarely used codes (stacked area plot) - now including 0% usage
        axes[1, 1].fill_between(epochs, 0, unused_codes, alpha=0.7, label='0% (Unused)', color='darkred')
        axes[1, 1].fill_between(epochs, unused_codes, unused_codes + codes_below_05, alpha=0.7, label='0-0.5%', color='lightcoral')
        axes[1, 1].fill_between(epochs, unused_codes + codes_below_05, unused_codes + codes_below_1, alpha=0.7, label='0.5-1.0%', color='orange')
        axes[1, 1].fill_between(epochs, unused_codes + codes_below_1, unused_codes + codes_below_5, alpha=0.7, label='1.0-5.0%', color='gold')
        axes[1, 1].set_title('Code Usage Distribution')
        axes[1, 1].set_xlabel('Measurement Point')
        axes[1, 1].set_ylabel('Number of Codes')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Plot EMA cluster sizes if available
        if 'ema_cluster_sizes' in self.codebook_stats_history[0]:
            ema_active = [(stats['ema_cluster_sizes'] > 0).sum().item() for stats in self.codebook_stats_history]
            axes[2, 0].plot(epochs, ema_active, marker='d', color='purple')
            axes[2, 0].set_title('EMA Active Clusters')
            axes[2, 0].set_xlabel('Measurement Point')
            axes[2, 0].set_ylabel('Active Clusters')
            axes[2, 0].grid(True)
        else:
            axes[2, 0].text(0.5, 0.5, 'EMA not enabled', ha='center', va='center', transform=axes[2, 0].transAxes)
            axes[2, 0].set_title('EMA Active Clusters')
        
        # Plot individual rarely used codes lines for better visibility - now including 0% usage
        axes[2, 1].plot(epochs, unused_codes, marker='o', label='0% (Unused)', color='darkred', linewidth=2)
        axes[2, 1].plot(epochs, codes_below_05, marker='s', label='Below 0.5%', color='red', linewidth=2)
        axes[2, 1].plot(epochs, codes_below_1, marker='^', label='Below 1.0%', color='orange', linewidth=2)
        axes[2, 1].plot(epochs, codes_below_5, marker='d', label='Below 5.0%', color='gold', linewidth=2)
        axes[2, 1].set_title('Code Usage Thresholds (Individual Lines)')
        axes[2, 1].set_xlabel('Measurement Point')
        axes[2, 1].set_ylabel('Number of Codes')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    

    
    def _plot_diversity_evolution(self, save_path: str) -> None:
        """Plot codebook diversity metrics over time."""
        if not self.diversity_history or not self.save_tracking_history:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract data
        epochs = list(range(len(self.diversity_history)))
        mean_similarity = [metrics['mean_similarity'] for metrics in self.diversity_history]
        max_similarity = [metrics['max_similarity'] for metrics in self.diversity_history]
        embedding_norm_mean = [metrics['embedding_norm_mean'] for metrics in self.diversity_history]
        embedding_norm_std = [metrics['embedding_norm_std'] for metrics in self.diversity_history]
        
        # Plot mean similarity
        axes[0, 0].plot(epochs, mean_similarity, marker='o', color='blue')
        axes[0, 0].set_title('Mean Embedding Similarity')
        axes[0, 0].set_xlabel('Measurement Point')
        axes[0, 0].set_ylabel('Mean Similarity')
        axes[0, 0].grid(True)
        
        # Plot max similarity
        axes[0, 1].plot(epochs, max_similarity, marker='s', color='red')
        axes[0, 1].set_title('Max Embedding Similarity')
        axes[0, 1].set_xlabel('Measurement Point')
        axes[0, 1].set_ylabel('Max Similarity')
        axes[0, 1].grid(True)
        
        # Plot embedding norm mean
        axes[1, 0].plot(epochs, embedding_norm_mean, marker='^', color='green')
        axes[1, 0].set_title('Mean Embedding Norm')
        axes[1, 0].set_xlabel('Measurement Point')
        axes[1, 0].set_ylabel('Mean Norm')
        axes[1, 0].grid(True)
        
        # Plot embedding norm std
        axes[1, 1].plot(epochs, embedding_norm_std, marker='d', color='purple')
        axes[1, 1].set_title('Embedding Norm Std')
        axes[1, 1].set_xlabel('Measurement Point')
        axes[1, 1].set_ylabel('Norm Std')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def manual_codebook_reset(self, reset_strategy='random'):
        """
        Manually reset the codebook using the enhanced vector quantizer.
        
        Args:
            reset_strategy (str): Strategy for reset - 'random', 'uniform', or 'kmeans'
        """
        self.model.manual_codebook_reset(reset_strategy)
    
    def set_ema_decay(self, new_decay):
        """
        Update the EMA decay rate in the enhanced vector quantizer.
        
        Args:
            new_decay (float): New EMA decay rate
        """
        self.model.set_ema_decay(new_decay)
    
    def disable_ema(self):
        """Disable EMA updates in the enhanced vector quantizer."""
        self.model.disable_ema()
    
    def enable_ema(self, decay=None):
        """
        Enable EMA updates in the enhanced vector quantizer.
        
        Args:
            decay (float, optional): New EMA decay rate
        """
        self.model.enable_ema(decay)

def demonstrate_custom_prompt_cot(checkpoint_path: str,
                                 prompt_file: str = "test_prompt.txt",
                                 cot_file: str = "test_cot.txt",
                                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                                 use_vq: bool = True,
                                 model_type: str = "GPT2VQVAE",
                                 **kwargs):
    """
    Demonstrate GPT2VQVAE, EnhancedGPT2VQVAE, or SimpleGPT2VQVAE model generation capabilities from a checkpoint
    using custom prompt and CoT from plain text files.
    
    This function automatically loads the model configuration from the checkpoint file,
    eliminating the need to manually specify model_config.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        prompt_file: Path to file containing the prompt in plain text
        cot_file: Path to file containing the CoT in plain text
        device: Device to run the model on
        use_vq: Whether to use vector quantization (for SimpleGPT2VQVAE)
        model_type: Type of model to use ("GPT2VQVAE", "EnhancedGPT2VQVAE", or "SimpleGPT2VQVAE")
        **kwargs: Additional arguments to override checkpoint configuration
    """
    print(f"Loading {model_type} model from checkpoint: {checkpoint_path}")
    
    # Load model using the new from_checkpoint method
    try:
        if model_type == "EnhancedGPT2VQVAE":
            model = EnhancedGPT2VQVAE.from_checkpoint(checkpoint_path, device=device, **kwargs)
            print(f"Successfully loaded EnhancedGPT2VQVAE model")
        elif model_type == "SimpleGPT2VQVAE":
            model = SimpleGPT2VQVAE.from_checkpoint(checkpoint_path, device=device, **kwargs)
            print(f"Successfully loaded SimpleGPT2VQVAE model with use_vq={use_vq}")
        else:
            model = GPT2VQVAE.from_checkpoint(checkpoint_path, device=device, **kwargs)
            print(f"Successfully loaded GPT2VQVAE model")
    except Exception as e:
        print(f"Error loading model from checkpoint: {e}")
        print("Make sure the checkpoint file exists and contains the required model configuration.")
        return
    
    model.eval()
    
    # Get num_embeddings and num_thoughts from model configuration
    num_embeddings = model.vector_quantizer.num_embeddings
    num_thoughts = 1 if model_type == "SimpleGPT2VQVAE" else model.num_thoughts
    print(f"Model configured for {num_thoughts} parallel CoT sequences and {num_embeddings} embeddings")
    
    # Load tokenizer and GPT2 model for baseline comparison
    try:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        print("Loaded GPT2 tokenizer")
        
        # Load pre-trained GPT2 model for baseline comparison
        gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
        gpt2_model.eval()
        print("Loaded pre-trained GPT2 model for baseline comparison")
    except Exception as e:
        print(f"Warning: Could not load GPT2 model/tokenizer: {e}")
        tokenizer = None
        gpt2_model = None
    
    # Load prompt and CoT from files
    print(f"Loading prompt from: {prompt_file}")
    print(f"Loading CoT from: {cot_file}")
    
    try:
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt_text = f.read().strip()
        
        with open(cot_file, 'r', encoding='utf-8') as f:
            cot_text = f.read().strip()
        
        print(f"Prompt: {prompt_text}")
        print(f"CoT: {cot_text}")
        
    except FileNotFoundError as e:
        print(f"Error: Could not find file {e.filename}")
        return
    except Exception as e:
        print(f"Error reading files: {e}")
        return
    
    # Tokenize prompt and CoT
    if tokenizer is None:
        print("Error: Tokenizer not available")
        return
    
    try:
        print("For now, we will pad both the prompt and cot to length 128.")
        # Tokenize and pad prompt to length 128
        prompt_tokens = tokenizer.encode(prompt_text, return_tensors='pt', padding='max_length', max_length=128)
        prompt_length = prompt_tokens.shape[1]
        
        # Tokenize and pad CoT to length 128 
        cot_tokens = tokenizer.encode(cot_text, return_tensors='pt', padding='max_length', max_length=128)
        cot_length = cot_tokens.shape[1]
        
        print(f"Prompt tokens: {prompt_tokens.shape} (length: {prompt_length})")
        print(f"CoT tokens: {cot_tokens.shape} (length: {cot_length})")
        
        # Check sequence length limits
        n_positions = model.encoder_config.n_positions
        total_length = prompt_length + cot_length
        
        if total_length > n_positions:
            print(f"Warning: Total sequence length {total_length} exceeds model's n_positions {n_positions}")
            print("Truncating sequences...")
            
            # Truncate CoT to fit within position limit
            max_cot_length = n_positions - prompt_length
            if max_cot_length <= 0:
                print("Error: Prompt is too long, cannot fit CoT")
                return
            
            cot_tokens = cot_tokens[:, :max_cot_length]
            cot_length = max_cot_length
            print(f"Truncated CoT to {cot_length} tokens")
        
    except Exception as e:
        print(f"Error tokenizing text: {e}")
        return
    
    # Create masks for prompt and CoT
    prompt_mask = torch.ones_like(prompt_tokens, dtype=torch.bool)
    cot_mask = torch.ones_like(cot_tokens, dtype=torch.bool)
    
    # Expand CoT to match num_thoughts (repeat the same CoT multiple times)
    # This simulates having multiple parallel CoT sequences
    cot_tokens_expanded = cot_tokens.unsqueeze(0).repeat(1, num_thoughts, 1)  # [1, num_thoughts, cot_length]
    cot_mask_expanded = cot_mask.unsqueeze(0).repeat(1, num_thoughts, 1)      # [1, num_thoughts, cot_length]
    
    print(f"Expanded CoT shape: {cot_tokens_expanded.shape}")
    
    # Function to decode tokens to text
    def decode_tokens(tokens, mask=None):
        if tokenizer is None:
            return f"[Tokens: {tokens.tolist()}]"
        
        if mask is not None:
            # Apply mask to remove padding
            tokens = tokens[mask.bool()]
        
        try:
            return tokenizer.decode(tokens, skip_special_tokens=True)
        except Exception as e:
            return f"[Decode error: {e}, tokens: {tokens.tolist()}]"
    
    print("\n" + "="*80)
    print(f"{model_type} CUSTOM PROMPT-COT DEMONSTRATION")
    print("="*80)
    
    with torch.no_grad():
        # Move tensors to device
        prompt = prompt_tokens.to(device)  # [1, prompt_length]
        cot_gt = cot_tokens_expanded.to(device)  # [1, num_thoughts, cot_length]
        prompt_mask_ex = prompt_mask.to(device)
        cot_mask_ex = cot_mask_expanded.to(device)
        
        print(f"Input shapes:")
        print(f"  Prompt: {prompt.shape}")
        print(f"  CoT: {cot_gt.shape}")
        print(f"  Prompt mask: {prompt_mask_ex.shape}")
        print(f"  CoT mask: {cot_mask_ex.shape}")
        
        # Generate with teacher forcing (inference=False)
        try:
            # Prepare base model inputs
            model_inputs = {
                'prompt': prompt,
                'cot_sequences': cot_gt, 
                'cot_mask': cot_mask_ex,
                'prompt_mask': prompt_mask_ex,
                'inference': False,  # Teacher forcing
                'quantize_cot_only': True
            }

            # Add use_vq parameter for SimpleGPT2VQVAE
            if model_type == "SimpleGPT2VQVAE":
                model_inputs['use_vq'] = use_vq

            # Run model with prepared inputs
            _, output_logits_tf, vq_loss_tf, perplexity_tf, indices_tf = model(**model_inputs)
            
            # Get predicted tokens from logits
            predicted_tokens_tf = torch.argmax(output_logits_tf, dim=-1)  # [B, M, L]
            
        except Exception as e:
            print(f"Teacher forcing generation failed: {e}")
            predicted_tokens_tf = None
            vq_loss_tf = None
            perplexity_tf = None
            indices_tf = None
        
        # Generate auto-regressively (inference=True)
        try:
            # Prepare model inputs
            model_inputs = {
                'prompt': prompt,
                'cot_sequences': cot_gt,
                'cot_mask': cot_mask_ex,
                'prompt_mask': prompt_mask_ex,
                'inference': True,  # Auto-regressive
                'quantize_cot_only': True
            }
            
            # Add use_vq parameter for SimpleGPT2VQVAE
            if model_type == "SimpleGPT2VQVAE":
                model_inputs['use_vq'] = use_vq
                
            # Run model with prepared inputs
            output_sequences_ar, output_logits_ar, vq_loss_ar, perplexity_ar, indices_ar = model(**model_inputs)
            
        except Exception as e:
            print(f"Auto-regressive generation failed: {e}")
            output_sequences_ar = None
            vq_loss_ar = None
            perplexity_ar = None
            indices_ar = None

        # Generate baseline using pre-trained GPT2
        gpt2_baseline_texts = generate_gpt2_baseline_texts(
            prompt, prompt_mask_ex, cot_gt, num_thoughts, tokenizer, gpt2_model
        )
        
        # Display results
        print(f"\n{'='*120}")
        print(f"DEMONSTRATION RESULTS")
        print(f"{'='*120}")
        
        # Print metrics
        if vq_loss_tf is not None and perplexity_tf is not None:
            print(f"Teacher Forcing - VQ Loss: {vq_loss_tf.item():.4f}, Perplexity: {perplexity_tf.item():.2f}")
            if indices_tf is not None:
                unique_indices = torch.unique(indices_tf).numel()
                print(f"  Codebook usage: {unique_indices} unique indices out of {indices_tf.numel()} total")
            
            # Compute reconstruction loss for teacher forcing
            if output_logits_tf is not None and cot_gt is not None and cot_mask_ex is not None:
                recon_loss_tf = compute_reconstruction_loss(output_logits_tf, cot_gt, cot_mask_ex)
                print(f"  Reconstruction Loss: {recon_loss_tf.item():.4f}")
                
        if vq_loss_ar is not None and perplexity_ar is not None:
            print(f"Auto-regressive - VQ Loss: {vq_loss_ar.item():.4f}, Perplexity: {perplexity_ar.item():.2f}")
            if indices_ar is not None:
                unique_indices = torch.unique(indices_ar).numel()
                print(f"  Codebook usage: {unique_indices} unique indices out of {indices_ar.numel()} total")
            
            # Compute reconstruction loss for auto-regressive
            if output_logits_ar is not None and cot_gt is not None and cot_mask_ex is not None:
                recon_loss_ar = compute_reconstruction_loss(output_logits_ar, cot_gt, cot_mask_ex)
                print(f"  Reconstruction Loss: {recon_loss_ar.item():.4f}")
                
        print()
        
        # Function to chunk text into 20-word segments
        def chunk_text(text, chunk_size=20):
            """Split text into chunks of specified word count and on newlines."""
            if not text:
                return []
            
            # Split on newlines first
            lines = text.split('\n')
            chunks = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                words = line.split()
                for i in range(0, len(words), chunk_size):
                    chunk = ' '.join(words[i:i + chunk_size])
                    if chunk:
                        chunks.append(chunk)
            
            return chunks
        
        # Display results for each CoT sequence (they should all be the same since we repeated the same CoT)
        print(f"\n{'='*120}")
        print(f"GENERATION RESULTS COMPARISON")
        print(f"{'='*120}")
        
        # Show original prompt and CoT
        print(f"Original Prompt: {prompt_text}")
        print(f"Original CoT: {cot_text}")
        print()
        
        # Show results for the first CoT sequence (they should all be identical)
        j = 0
        print(f"--- CoT Sequence {j+1} (all sequences are identical) ---")
        
        # Get the three versions of the CoT
        cot_gt_text = decode_tokens(cot_gt[0, j], cot_mask_ex[0, j] if cot_mask_ex is not None else None)
        
        if predicted_tokens_tf is not None:
            cot_tf_text = decode_tokens(predicted_tokens_tf[0, j], cot_mask_ex[0, j] if cot_mask_ex is not None else None)
        else:
            cot_tf_text = "FAILED"
        
        if output_sequences_ar is not None:
            cot_ar_text = decode_tokens(output_sequences_ar[0, j], cot_mask_ex[0, j] if cot_mask_ex is not None else None)
        else:
            cot_ar_text = "FAILED"
        
        # GPT2 baseline text
        gpt2_text = gpt2_baseline_texts[j] if j < len(gpt2_baseline_texts) else "FAILED"
        
        # Chunk all four texts
        gt_chunks = chunk_text(cot_gt_text)
        tf_chunks = chunk_text(cot_tf_text)
        ar_chunks = chunk_text(cot_ar_text)
        gpt2_chunks = chunk_text(gpt2_text)
        
        # Find the maximum number of chunks
        max_chunks = max(len(gt_chunks), len(tf_chunks), len(ar_chunks), len(gpt2_chunks))
        
        # Display chunks side-by-side
        for chunk_idx in range(max_chunks):
            gt_chunk = gt_chunks[chunk_idx] if chunk_idx < len(gt_chunks) else ""
            tf_chunk = tf_chunks[chunk_idx] if chunk_idx < len(tf_chunks) else ""
            ar_chunk = ar_chunks[chunk_idx] if chunk_idx < len(ar_chunks) else ""
            gpt2_chunk = gpt2_chunks[chunk_idx] if chunk_idx < len(gpt2_chunks) else ""
            
            # Pad chunks to same length for alignment
            max_length = max(len(gt_chunk), len(tf_chunk), len(ar_chunk), len(gpt2_chunk))
            gt_chunk_padded = gt_chunk.ljust(max_length)
            tf_chunk_padded = tf_chunk.ljust(max_length)
            ar_chunk_padded = ar_chunk.ljust(max_length)
            gpt2_chunk_padded = gpt2_chunk.ljust(max_length)
            
            print(f"Original:        {gt_chunk_padded}")
            print(f"Teacher Forced:  {tf_chunk_padded}")
            print(f"Auto-regressive: {ar_chunk_padded}")
            print(f"GPT2 Baseline:   {gpt2_chunk_padded}")
            
            # Add separator after each chunk (except the last one)
            if chunk_idx < max_chunks - 1:
                print("-" * 60)
        
        print("\n" + "="*120)
    
    # Create codebook usage heatmaps if VQ is enabled
    if use_vq:
        print(f"\n{'='*80}")
        print("GENERATING CODEBOOK USAGE HEATMAPS")
        print(f"{'='*80}")
        
        # Create heatmap for teacher forcing
        if indices_tf is not None:
            print(f"Teacher Forcing - Total indices: {indices_tf.numel()}")
            print(f"Teacher Forcing - Unique indices: {torch.unique(indices_tf).numel()}")
            
            # Create heatmap for teacher forcing
            heatmap_path_tf = os.path.join(os.path.dirname(checkpoint_path), "codebook_usage_tf_custom.png")
            # Convert indices to counts
            counts_tf = torch.bincount(indices_tf[indices_tf < num_embeddings], 
                                      minlength=num_embeddings)
            create_codebook_usage_heatmap(
                counts_tf, 
                num_embeddings=num_embeddings,
                title=f"Teacher Forcing Codebook Usage - Custom Prompt-CoT",
                save_path=heatmap_path_tf
            )
        
        # Create heatmap for auto-regressive
        if indices_ar is not None:
            print(f"Auto-regressive - Total indices: {indices_ar.numel()}")
            print(f"Auto-regressive - Unique indices: {torch.unique(indices_ar).numel()}")
            
            # Create heatmap for auto-regressive
            heatmap_path_ar = os.path.join(os.path.dirname(checkpoint_path), "codebook_usage_ar_custom.png")
            # Convert indices to counts
            counts_ar = torch.bincount(indices_ar[indices_ar < num_embeddings], 
                                      minlength=num_embeddings)
            create_codebook_usage_heatmap(
                counts_ar, 
                num_embeddings=num_embeddings,
                title=f"Auto-regressive Codebook Usage - Custom Prompt-CoT",
                save_path=heatmap_path_ar
            )
    
    print("\nCustom prompt-CoT demonstration completed!")

class PhasedEnhancedGPT2VQVAETrainer(EnhancedGPT2VQVAETrainer):
    """
    Specialized trainer for EnhancedGPT2VQVAE that implements a phased training approach:
    
    1. Initialization Phase (0 to initialization_steps):
       - Train using no-vq mode to learn good representations
       
    2. Reinitialization Phase (initialization_steps to quantization_start):
       - Every r_reestim steps, fully reinitialize codebook using KMeans on reservoir samples
       - Continue training with VQ enabled
       
    3. Normal Training Phase (after quantization_start):
       - Standard VQ-VAE training with all enhancements
    """
    
    def __init__(self, 
                 model_config: Dict[str, Any], 
                 training_config: Dict[str, Any], 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu", 
                 run_name : Optional[str] = "ANONYM_RUN"):
        """
        Initialize the phased trainer.
        
        Args:
            model_config: Model configuration dictionary
            training_config: Training configuration dictionary with phased training parameters:
                - initialization_steps: Number of steps to train in no-vq mode
                - r_reestim: Frequency of codebook reinitialization during reinitialization phase
                - quantization_start: Step at which to start normal VQ training
            device: Device to train on
            run_name: Optional string for the name of the run, as texted via send_notification.
        """
        # Extract phased training parameters
        self.initialization_steps = training_config.get('initialization_steps', 1500)
        self.r_reestim = training_config.get('r_reestim', 500)
        self.quantization_start = training_config.get('quantization_start', 5000)
        
        # Extract codebook learning rate multiplier
        self.codebook_lr_multiplier = training_config.get('codebook_lr_multiplier', 1.0)
        
        # Validate parameters
        if self.initialization_steps >= self.quantization_start:
            raise ValueError("initialization_steps must be less than quantization_start")
        if self.r_reestim <= 0:
            raise ValueError("r_reestim must be positive")
        if self.codebook_lr_multiplier <= 0:
            raise ValueError("codebook_lr_multiplier must be positive")
        
        # Replace the track codebook usage function with the phased version (which ignores the initialization stage)
        phased_tracking_functions = {
            'track_codebook_usage': self._phased_track_codebook_usage
        }

        # Initialize parent trainer
        super().__init__(model_config, training_config, device, phased_tracking_functions, run_name)
        
        # Override optimizer with codebook-specific learning rates
        self._setup_codebook_optimizer()
        
        print(f"PhasedEnhancedGPT2VQVAETrainer initialized with:")
        print(f"  - Initialization phase: 0 to {self.initialization_steps} steps (no-vq mode)")
        print(f"  - Reinitialization phase: {self.initialization_steps} to {self.quantization_start} steps")
        print(f"  - Reinitialization frequency: every {self.r_reestim} steps")
        print(f"  - Normal training phase: after {self.quantization_start} steps")
        print(f"  - Codebook learning rate multiplier: {self.codebook_lr_multiplier}x")
    
    def _setup_codebook_optimizer(self):
        """
        Set up optimizer with different learning rates for codebook parameters.
        Codebook parameters get learning_rate * codebook_lr_multiplier,
        while all other parameters get the normal learning_rate.
        """
        # if the multipler is 1.0, no change
        if self.codebook_lr_multiplier == 1.0:
            return
        
        # Get the base learning rate
        base_lr = self.training_config['learning_rate']
        codebook_lr = base_lr * self.codebook_lr_multiplier
        
        # Separate parameters into codebook and non-codebook groups
        codebook_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            # TODO: CONSIDER IF aggregation_mlp SHOULD GET HIGHER LR OR NOT
            # TBH, HARD TO SAY. WILL START WITHOUT.
            # if 'vector_quantizer' in name or 'aggregation_mlp' in name:
            if 'vector_quantizer' in name:
                codebook_params.append(param)
            else:
                other_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = [
            {
                'params': other_params,
                'lr': base_lr,
                'weight_decay': self.training_config.get('weight_decay', 0.01),
                'betas': (self.training_config.get('beta1', 0.9), self.training_config.get('beta2', 0.999))
            },
            {
                'params': codebook_params,
                'lr': codebook_lr,
                'weight_decay': self.training_config.get('weight_decay', 0.01),
                'betas': (self.training_config.get('beta1', 0.9), self.training_config.get('beta2', 0.999))
            }
        ]
        
        # CRITICAL FIX: Clear old optimizer state before recreating
        if hasattr(self, 'optimizer') and self.optimizer is not None:
            # Clear the old optimizer's state to prevent memory accumulation
            self.optimizer.state.clear()
            # Explicitly delete the old optimizer to free memory
            del self.optimizer
        
        # Recreate optimizer with parameter groups
        self.optimizer = optim.AdamW(param_groups)
        
        # CRITICAL FIX: Clear any existing scheduler state before recreating
        if hasattr(self, 'scheduler') and self.scheduler is not None:
            del self.scheduler
        
        # Recreate scheduler if it exists
        if self.training_config.get('use_lr_scheduler', True):
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.training_config['num_epochs'],
                eta_min=self.training_config.get('min_lr', 1e-6)
            )
        else:
            self.scheduler = None
        
        print(f"Optimizer reconfigured with separate learning rates:")
        print(f"  - Main parameters: {base_lr:.2e}")
        print(f"  - Codebook parameters: {codebook_lr:.2e} ({self.codebook_lr_multiplier}x multiplier)")
        print(f"  - Codebook parameters count: {len(codebook_params)}")
        print(f"  - Other parameters count: {len(other_params)}")
    
    def _determine_training_phase(self, current_step: int) -> str:
        """
        Determine the current training phase based on the step number.
        
        Args:
            current_step: Current training step
            
        Returns:
            str: Current phase name
        """
        if current_step < self.initialization_steps:
            return "initialization"
        elif current_step < self.quantization_start:
            return "reinitialization"
        else:
            return "normal"
    
    def _should_reinitialize_codebook(self, current_step: int) -> bool:
        """
        Check if codebook should be reinitialized at the current step.
        
        Args:
            current_step: Current training step
            
        Returns:
            bool: True if codebook should be reinitialized
        """
        if current_step < self.initialization_steps:
            return False
        
        if current_step >= self.quantization_start:
            return False
        
        # Check if we're in reinitialization phase using modulo
        if current_step >= self.initialization_steps:
            adjusted_step = current_step - self.initialization_steps
            return (adjusted_step + 1) % self.r_reestim == 0
        
        return False
    
    def _reinitialize_codebook(self, current_step: int):
        """
        Reinitialize the codebook using the EnhancedVectorQuantizer's _reset_codebook method.
        
        Args:
            current_step: Current training step for logging
        """
        print(f"\n=== Codebook Reinitialization at step {current_step} ===")
        
        # Get a dummy input tensor for the device reference (required by _reset_codebook)
        device = self.model.vector_quantizer.embedding.weight.device
        
        # Use the EnhancedVectorQuantizer's _reset_codebook method with 'full' strategy
        # This will perform K-means++ clustering on reservoir samples and reset the entire codebook
        self.model.vector_quantizer._reset_codebook(device, reset_strategy='full')
    
    def _forward_pass(self, prompts, cots, prompt_masks, cot_masks):
        """
        Enhanced forward pass that handles different training phases.
        
        Args:
            prompts: Prompt sequences
            cots: Chain-of-thought sequences
            prompt_masks: Prompt attention masks
            cot_masks: COT attention masks
            
        Returns:
            tuple: (total_loss_batch, vq_loss, perplexity, indices)
        """
        # Determine current phase
        current_phase = self._determine_training_phase(self.current_step)
        
        # Check if we need to reinitialize codebook
        if self._should_reinitialize_codebook(self.current_step):
            self._reinitialize_codebook(self.current_step)
        
        # Determine whether to use VQ based on current phase
        use_vq = current_phase != "initialization"
        
        # Update phase tracking for logging
        last_step_phase = self._determine_training_phase(self.current_step-1)
        if current_phase != last_step_phase:
            print(f"\n=== Phase transition: {last_step_phase} -> {current_phase} at step {self.current_step-1} ===")
        
        # Perform forward pass with appropriate VQ setting and handle mixed precision
        if self.use_mixed_precision:
            with autocast('cuda'):
                _, output_logits, vq_loss, perplexity, indices = self.model(
                    prompt=prompts,
                    cot_sequences=cots,
                    cot_mask=cot_masks,
                    prompt_mask=prompt_masks,
                    inference=False,
                    quantize_cot_only=self.training_config.get('quantize_cot_only', True),
                    no_vq=not use_vq
                )
                recon_loss = compute_reconstruction_loss(output_logits, cots, cot_masks)
                total_loss_batch = recon_loss + self.training_config.get('vq_loss_weight', 1.0) * vq_loss
        else:
            _, output_logits, vq_loss, perplexity, indices = self.model(
                prompt=prompts,
                cot_sequences=cots,
                cot_mask=cot_masks,
                prompt_mask=prompt_masks,
                inference=False,
                quantize_cot_only=self.training_config.get('quantize_cot_only', True),
                no_vq=not use_vq
            )
            recon_loss = compute_reconstruction_loss(output_logits, cots, cot_masks)
            total_loss_batch = recon_loss + self.training_config.get('vq_loss_weight', 1.0) * vq_loss
        
        return total_loss_batch, vq_loss, perplexity, indices
    
    def train_epoch(self, train_loader, num_measurements_per_epoch, current_epoch=0):
        """
        Enhanced train_epoch that tracks training steps and handles phase transitions.
        
        Args:
            train_loader: Training data loader
            num_measurements_per_epoch: Number of measurements per epoch
            current_epoch: Current epoch number
            
        Returns:
            dict: Training metrics for the epoch
        """
        # Set max_reset_steps on the first epoch if it's still None
        if current_epoch == 0 and self.model.vector_quantizer.max_reset_steps is None:
            # Set max_reset_steps to quantization_start to disable automatic resets during initialization
            self.model.vector_quantizer.max_reset_steps = self.quantization_start
        
        # Call parent train_epoch
        metrics = super().train_epoch(train_loader, num_measurements_per_epoch, current_epoch)
        
        return metrics

    def _phased_track_codebook_usage(self, dataset: Any, measurement_point: int) -> None:
        """
        Enhanced codebook tracking with additional statistics.
        
        Args:
            dataset: Dataset to sample from
            measurement_point: Current measurement point index
        """
        if self._determine_training_phase(self.current_step) == "initialization": return
        # Only call the parent's _enhanced_track_codebook_usage if we are out of the warm up phase
        super()._enhanced_track_codebook_usage(dataset, measurement_point)
        
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Enhanced checkpoint loading that restores phased training state.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        # Load checkpoint using parent method
        checkpoint = super().load_checkpoint(checkpoint_path)
        
        # Update current phase
        print(f"Current training phase: {self._determine_training_phase(self.current_step)}")
        
        return checkpoint

def save_training_visualizations(trainer, save_dir: str = None, prefix: str = "training"):
    """
    Save comprehensive training visualizations including all available plots and metrics.
    
    Args:
        trainer: The trainer instance with training data
        save_dir: Directory to save visualizations (defaults to trainer's checkpoint directory)
        prefix: Prefix for saved files
    """
    with record_function("## save_training_visualizations ##"):
        DEFAULT_FOLDER = './training_visualizations'
        try:
            if save_dir is None:
                training_config = getattr(trainer, 'training_config', None)
                if training_config is None:
                    save_dir = DEFAULT_FOLDER
                else:
                    save_dir = training_config.get('checkpoint_dir', DEFAULT_FOLDER)
            
            os.makedirs(save_dir, exist_ok=True)
            
            print(f"\n📊 Saving comprehensive training visualizations to {save_dir}")
            
            # 1. Training history plots
            history_path = os.path.join(save_dir, f"{prefix}_history.png")
            with record_function("## plot_training_history ##"):
                trainer.plot_training_history(history_path)
            
            # 2. Memory usage plots
            memory_path = os.path.join(save_dir, f"{prefix}_memory_usage.png")
            trainer.plot_memory_usage(memory_path)
            
            # 3. Codebook tracking plots (if enabled)
            if trainer.tracking_enabled:
                codebook_dir = os.path.join(save_dir, f"{prefix}_codebook_tracking")
                os.makedirs(codebook_dir, exist_ok=True)
                
                # Save current codebook plots
                if hasattr(trainer, 'save_codebook_plots_func'):
                    trainer.save_codebook_plots_func(codebook_dir, len(trainer.train_losses))
            
            print(f"✅ All training visualizations saved successfully!")
            print(f"   📈 Training history: {history_path}")
            print(f"   💾 Memory usage: {memory_path}")
            if trainer.tracking_enabled:
                print(f"   🎯 Codebook tracking: {codebook_dir}")
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to save some training visualizations: {e}")
            # Try to save at least basic plots
            try:
                if save_dir is None:
                    save_dir = getattr(trainer, 'checkpoint_dir', './training_visualizations')
                os.makedirs(save_dir, exist_ok=True)
                
                # Fallback: save basic training history
                history_path = os.path.join(save_dir, f"{prefix}_history_fallback.png")
                with record_function("## plot_training_history ##"):
                    trainer.plot_training_history(history_path)
                print(f"   📈 Basic training history saved: {history_path}")
            except Exception as fallback_error:
                print(f"   ❌ Failed to save even basic plots: {fallback_error}")


if __name__ == "__main__":
    # Run main function for command-line training
    main()